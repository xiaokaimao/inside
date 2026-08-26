"""Append the official basic complementary-contribution baseline to Wine.

This is deliberately a separate augmentation entry point.  The expensive
800,000-pair reference and the four already-computed OFA/Frame-OFA methods in
the source report are treated as immutable.  Only the basic ``cc_shap``
sampling law from the authors' official implementation is rerun against the
same Wine cooperative game.

For a general utility, one complementary-contribution sample evaluates both
``v(S)`` and ``v(N \\ S)``.  Consequently, matching an even total utility-call
budget ``T`` gives ``T / 2`` CC samples.  Unlike OFA, basic CC does not require
the separate ``2n+2`` exact-boundary batch.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from frame_ofa import GameEvaluator

from .iris_data_valuation import summarize_method
from .iris_sklearn_game import SklearnClassificationGame
from .sklearn_data import load_sklearn_train_test_split


METHOD_KEY = "official_cc_basic"
SOURCE_METHOD_KEYS = frozenset(
    {
        "official_ofa_fixed_ratio",
        "iid_linear_ofa",
        "frame_coupled_linear",
        "frame_orbit_ratio",
    }
)
UPSTREAM_REPOSITORY = (
    "https://github.com/ZJU-DIVER/ShapleyValueApproximation"
)
UPSTREAM_FUNCTION = "shapley.gtsv.cc_shap"


def _array_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_checkpoint(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_array_json)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_source_report(report: Mapping[str, Any]) -> None:
    """Check the protocol fields needed for a safe Wine-only augmentation."""
    if report.get("status") != "complete":
        raise ValueError("the source Wine report must have status='complete'")
    configuration = report.get("configuration")
    dataset = report.get("dataset")
    ground_truth = report.get("ground_truth")
    boundary = report.get("boundary")
    results = report.get("results_by_inner_budget")
    if not all(
        isinstance(value, Mapping)
        for value in (
            configuration,
            dataset,
            ground_truth,
            boundary,
            results,
        )
    ):
        raise ValueError("the source report is missing required objects")
    assert isinstance(configuration, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(ground_truth, Mapping)
    assert isinstance(boundary, Mapping)
    assert isinstance(results, Mapping)

    if dataset.get("dataset") != "wine":
        raise ValueError("this augmenter only accepts the Wine report")
    num_players = int(configuration.get("num_players", -1))
    repeats = int(configuration.get("repeats", -1))
    if num_players != 142:
        raise ValueError("the formal Wine report must contain 142 players")
    if repeats < 2:
        raise ValueError("the source report must contain at least two repeats")
    if int(configuration.get("gt_pairs", -1)) != 800_000:
        raise ValueError(
            "the formal Wine report must use 800,000 ground-truth pairs"
        )
    truth = np.asarray(ground_truth.get("values"), dtype=np.float64)
    truth_se = np.asarray(
        ground_truth.get("standard_errors"), dtype=np.float64
    )
    if truth.shape != (num_players,) or truth_se.shape != (num_players,):
        raise ValueError("ground-truth vectors disagree with the player count")
    if not np.isfinite(truth).all() or not np.isfinite(truth_se).all():
        raise ValueError("ground-truth vectors contain non-finite values")
    efficiency_target = float(boundary.get("efficiency_target", np.nan))
    if not np.isfinite(efficiency_target):
        raise ValueError("the boundary efficiency target must be finite")

    budgets = configuration.get("inner_budgets")
    totals = configuration.get("total_call_budgets")
    if not isinstance(budgets, list) or not isinstance(totals, list):
        raise ValueError("the source report is missing its budget lists")
    if len(budgets) != 5 or len(totals) != 5:
        raise ValueError("the formal Wine report must contain five budgets")
    boundary_calls = int(boundary.get("utility_calls", -1))
    expected_totals = [int(value) + boundary_calls for value in budgets]
    if [int(value) for value in totals] != expected_totals:
        raise ValueError("source total-call accounting is inconsistent")
    if any(value % 2 for value in expected_totals):
        raise ValueError("every total call budget must be even for paired CC")

    for budget, expected_total in zip(budgets, expected_totals):
        row = results.get(str(int(budget)))
        if not isinstance(row, Mapping):
            raise ValueError(f"missing source result for budget {budget}")
        methods = row.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError(f"budget {budget} is missing its methods")
        missing = SOURCE_METHOD_KEYS - set(methods)
        if missing:
            raise ValueError(
                f"budget {budget} is missing source methods {sorted(missing)}"
            )
        if int(row.get("total_utility_calls_per_estimate", -1)) != int(
            expected_total
        ):
            raise ValueError(
                f"budget {budget} has inconsistent per-estimate calls"
            )
        for method in SOURCE_METHOD_KEYS:
            estimates = np.asarray(
                methods[method].get("estimates"), dtype=np.float64
            )
            if estimates.shape != (repeats, num_players):
                raise ValueError(
                    f"budget {budget} method {method} has invalid estimates"
                )
            if not np.isfinite(estimates).all():
                raise ValueError(
                    f"budget {budget} method {method} is non-finite"
                )


def cc_call_plan(report: Mapping[str, Any]) -> list[dict[str, int]]:
    """Return same-total-call CC budgets in formal budget order."""
    _validate_source_report(report)
    configuration = report["configuration"]
    results = report["results_by_inner_budget"]
    plan: list[dict[str, int]] = []
    for budget in configuration["inner_budgets"]:
        inner_budget = int(budget)
        total_calls = int(
            results[str(inner_budget)]["total_utility_calls_per_estimate"]
        )
        if total_calls % 2:
            raise ValueError("a CC total-call budget must be even")
        plan.append(
            {
                "inner_budget_key": inner_budget,
                "total_utility_calls": total_calls,
                "complementary_contributions": total_calls // 2,
            }
        )
    return plan


def _estimate_official_basic_cc(
    evaluator: GameEvaluator,
    *,
    num_players: int,
    num_pairs: int,
    seed: int,
    num_tasks: int,
) -> Any:
    """Isolate the core API call so its mathematical contract is explicit."""
    # Imported lazily so call-accounting tests remain independent of the core
    # implementation. ``estimate_basic_cc`` is a clean, deterministic port of
    # the statistic in the official ``gtsv.cc_shap`` implementation.
    from frame_ofa import estimate_basic_cc

    return estimate_basic_cc(
        evaluator,
        num_players=num_players,
        num_pairs=num_pairs,
        seed=seed,
        num_tasks=num_tasks,
    )


def _diagnostic_value(diagnostics: Any, name: str) -> Any:
    if isinstance(diagnostics, Mapping):
        return diagnostics[name]
    return getattr(diagnostics, name)


def _assert_reconstructed_dataset(
    stored: Mapping[str, Any], reconstructed: Mapping[str, Any]
) -> None:
    """Prevent an expensive baseline run on a different train/test game."""
    keys = (
        "dataset",
        "dataset_seed",
        "test_size",
        "train_original_indices",
        "test_original_indices",
        "train_labels",
        "test_labels",
        "standardization_mean",
        "standardization_std",
    )
    for key in keys:
        if key not in stored or key not in reconstructed:
            raise ValueError(f"dataset metadata is missing {key!r}")
        left = stored[key]
        right = reconstructed[key]
        if key in {"standardization_mean", "standardization_std"}:
            if not np.array_equal(
                np.asarray(left, dtype=np.float64),
                np.asarray(right, dtype=np.float64),
            ):
                raise ValueError(f"reconstructed dataset differs at {key}")
        elif left != right:
            raise ValueError(f"reconstructed dataset differs at {key}")


def _initialize_or_resume_report(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    output_path: Path,
    cc_seed: int,
    jobs: int,
    chunksize: int,
    start_method: str,
    cc_tasks: int,
) -> dict[str, Any]:
    """Create an augmentation report or safely resume its completed budgets."""
    source_hash = _source_digest(source_path)
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = report.get("configuration", {}).get("cc_baseline", {})
        if metadata.get("source_sha256") != source_hash:
            raise ValueError("existing CC output was built from another source")
        if int(metadata.get("seed", -1)) != cc_seed:
            raise ValueError("existing CC output uses another CC seed")
        expected_execution = {
            "jobs": jobs,
            "chunksize": chunksize,
            "start_method": start_method,
            "num_tasks": cc_tasks,
        }
        for key, expected in expected_execution.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"existing CC output uses another {key} setting"
                )
        return report

    report = copy.deepcopy(source)
    report["status"] = "cc_augmentation_running"
    configuration = report["configuration"]
    plan = cc_call_plan(source)
    configuration["cc_baseline"] = {
        "method_key": METHOD_KEY,
        "source_repository": UPSTREAM_REPOSITORY,
        "source_function": UPSTREAM_FUNCTION,
        "source_sha256": source_hash,
        "sampling": (
            "j iid uniform on {1,...,n}; coalition iid uniform conditional "
            "on size j"
        ),
        "sample_unit": (
            "one complementary contribution = two utility evaluations"
        ),
        "budget_rule": (
            "m = total_utility_calls_per_estimate / 2; no OFA boundary"
        ),
        "missing_stratum_rule": "zero (matches official cc_shap)",
        "seed": cc_seed,
        "jobs": jobs,
        "chunksize": chunksize,
        "start_method": start_method,
        "num_tasks": cc_tasks,
        "worker_threads": 1,
        "additional_physical_utility_calls": int(
            sum(item["total_utility_calls"] for item in plan)
            * int(configuration["repeats"])
        ),
    }
    _write_checkpoint(output_path, report)
    return report


def run_augmentation(args: argparse.Namespace) -> dict[str, Any]:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    _validate_source_report(source)
    report = _initialize_or_resume_report(
        source=source,
        source_path=args.input,
        output_path=args.output,
        cc_seed=args.cc_seed,
        jobs=args.jobs,
        chunksize=args.chunksize,
        start_method=args.start_method,
        cc_tasks=args.cc_tasks,
    )
    plan = cc_call_plan(source)
    configuration = report["configuration"]
    num_players = int(configuration["num_players"])
    repeats = int(configuration["repeats"])
    truth_values = np.asarray(
        report["ground_truth"]["values"], dtype=np.float64
    )
    truth_se = np.asarray(
        report["ground_truth"]["standard_errors"], dtype=np.float64
    )
    efficiency_target = float(report["boundary"]["efficiency_target"])

    test_size = float(report["dataset"]["test_size"])
    dataset_seed = int(report["dataset"]["dataset_seed"])
    game_args, reconstructed = load_sklearn_train_test_split(
        "wine", test_size=test_size, dataset_seed=dataset_seed
    )
    _assert_reconstructed_dataset(report["dataset"], reconstructed)
    expected_model = (
        "sklearn.svm.SVC(C="
        f"{args.regularization}, kernel='rbf', gamma='scale')"
    )
    if configuration.get("model") != expected_model:
        raise ValueError(
            "the source model does not match the requested regularization"
        )
    game_args = game_args | {
        "model": "rbf_svm",
        "regularization": args.regularization,
    }

    pending = [
        item
        for item in plan
        if METHOD_KEY
        not in report["results_by_inner_budget"][
            str(item["inner_budget_key"])
        ]["methods"]
    ]
    if not pending:
        report["status"] = "complete"
        _write_checkpoint(args.output, report)
        return report

    with GameEvaluator(
        SklearnClassificationGame,
        game_args,
        n_jobs=args.jobs,
        chunksize=args.chunksize,
        start_method=args.start_method,
        worker_threads=1,
    ) as evaluator:
        for budget_index, item in enumerate(plan):
            inner_budget = item["inner_budget_key"]
            result = report["results_by_inner_budget"][str(inner_budget)]
            if METHOD_KEY in result["methods"]:
                print(
                    f"CC budget {inner_budget:,}: already complete, skipping",
                    flush=True,
                )
                continue
            num_samples = item["complementary_contributions"]
            total_calls = item["total_utility_calls"]
            print(
                f"CC budget {budget_index + 1}/{len(plan)}: "
                f"{num_samples:,} pairs = {total_calls:,} calls, "
                f"{repeats} repeats",
                flush=True,
            )
            estimates: list[np.ndarray] = []
            total_seconds: list[float] = []
            missing_fractions: list[float] = []
            repeat_diagnostics: list[dict[str, int]] = []
            for repeat in range(repeats):
                seed = (
                    args.cc_seed + budget_index * 100_000 + repeat
                )
                started = time.perf_counter()
                cc_result = _estimate_official_basic_cc(
                    evaluator,
                    num_players=num_players,
                    num_pairs=num_samples,
                    seed=seed,
                    num_tasks=args.cc_tasks,
                )
                total_seconds.append(time.perf_counter() - started)
                estimate = np.asarray(cc_result.values, dtype=np.float64)
                diagnostics = cc_result.diagnostics
                if estimate.shape != (num_players,):
                    raise RuntimeError(
                        "basic CC returned an invalid Shapley-vector shape"
                    )
                if not np.isfinite(estimate).all():
                    raise RuntimeError("basic CC returned non-finite values")
                observed_pairs = int(
                    _diagnostic_value(diagnostics, "num_pairs")
                )
                observed_calls = int(
                    _diagnostic_value(diagnostics, "utility_evaluations")
                )
                missing_strata = int(
                    _diagnostic_value(diagnostics, "missing_strata")
                )
                minimum_positive_count = int(
                    _diagnostic_value(
                        diagnostics, "minimum_positive_count"
                    )
                )
                observed_tasks = int(
                    _diagnostic_value(diagnostics, "num_tasks")
                )
                if observed_pairs != num_samples:
                    raise RuntimeError("basic CC pair accounting disagrees")
                if observed_calls != total_calls:
                    raise RuntimeError("basic CC utility-call accounting disagrees")
                estimates.append(estimate)
                missing_fractions.append(missing_strata / (num_players**2))
                repeat_diagnostics.append(
                    {
                        "num_pairs": observed_pairs,
                        "utility_evaluations": observed_calls,
                        "missing_strata": missing_strata,
                        "minimum_positive_count": minimum_positive_count,
                        "num_tasks": observed_tasks,
                    }
                )
                print(
                    f"  repeat {repeat + 1}/{repeats}: "
                    f"total {total_seconds[-1]:.2f}s, "
                    f"missing {missing_fractions[-1]:.3%}",
                    flush=True,
                )

            estimate_array = np.asarray(estimates, dtype=np.float64)
            summary = summarize_method(
                estimate_array,
                truth_values,
                truth_se,
                efficiency_target,
                bootstrap_seed=args.cc_seed + budget_index * 100 + 4,
            )
            summary.update(
                {
                    "mean_total_seconds": float(np.mean(total_seconds)),
                    "utility_calls_per_estimate": total_calls,
                    "complementary_contributions_per_estimate": (
                        num_samples
                    ),
                    "missing_stratum_fractions": missing_fractions,
                    "mean_missing_stratum_fraction": float(
                        np.mean(missing_fractions)
                    ),
                    "max_missing_stratum_fraction": float(
                        np.max(missing_fractions)
                    ),
                    "cc_diagnostics_by_repeat": repeat_diagnostics,
                }
            )
            result["methods"][METHOD_KEY] = summary
            result["cc_missing_stratum_fraction_mean"] = summary[
                "mean_missing_stratum_fraction"
            ]
            official_rmse = float(
                result["methods"]["official_ofa_fixed_ratio"][
                    "aggregate_rmse"
                ]
            )
            result["cc_rmse_reduction_vs_official_ratio"] = (
                official_rmse - summary["aggregate_rmse"]
            ) / official_rmse
            _write_checkpoint(args.output, report)
            print(
                f"  aggregate RMSE CC = {summary['aggregate_rmse']:.4e}",
                flush=True,
            )

    report["status"] = "complete"
    _write_checkpoint(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/wine_full_train_rbf_svm_frame_ofa_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json"
        ),
    )
    parser.add_argument("--cc-seed", type=int, default=1_910_001)
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=512)
    parser.add_argument("--cc-tasks", type=int, default=128)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--regularization", type=float, default=1.0)
    args = parser.parse_args()
    if not args.input.exists():
        raise ValueError(f"input report does not exist: {args.input}")
    if args.cc_seed < 0:
        raise ValueError("cc-seed must be nonnegative")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if args.chunksize < 1:
        raise ValueError("chunksize must be positive")
    if args.cc_tasks < 1:
        raise ValueError("cc-tasks must be positive")
    if args.regularization <= 0:
        raise ValueError("regularization must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_augmentation(args)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

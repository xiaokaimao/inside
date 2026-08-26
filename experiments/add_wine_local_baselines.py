"""Run locally ported Shapley baselines on the formal Wine game.

The source report and its 800,000-pair Monte Carlo reference are immutable.
To honor the requested three-repeat comparison, this script takes the first
three stored repeats of the existing OFA/Frame-OFA/basic-CC estimates,
recomputes every summary metric, then evaluates each newly ported baseline
three times at the same five *total physical utility-call caps*.

The external ``integral_shapley/src/core`` tree is never imported or edited.
Compatible formulas are adapted to the local Boolean-coalition
``GameEvaluator`` interface with explicit call accounting.  GELS-Shapley is
instead a clean-room Algorithm-3 port checked against the paper authors'
official ``watml/fastpvalue`` repository.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from frame_ofa import GameEvaluator

from .add_wine_cc_results import _validate_source_report
from .iris_data_valuation import paired_rmse_difference, summarize_method
from .iris_sklearn_game import SklearnClassificationGame
from .sklearn_data import load_sklearn_train_test_split


SOURCE_METHODS = (
    "official_ofa_fixed_ratio",
    "frame_orbit_ratio",
    "iid_linear_ofa",
    "frame_coupled_linear",
    "official_cc_basic",
)

DEFAULT_NEW_METHODS = (
    "gels_shapley",
    "kernel_shap_sampled",
    "group_testing",
    "diff",
    "s_diff",
    "tmc_shapley",
    "stratified_marginal_mc",
)

METHOD_LABELS = {
    "gels_shapley": "GELS-Shapley",
    "kernel_shap_sampled": "KernelSHAP (sampled Gram)",
    "group_testing": "Group Testing",
    "diff": "Diff (differential matrix)",
    "s_diff": "S-Diff (external strict port)",
    "tmc_shapley": "TMC-Shapley",
    "stratified_marginal_mc": "Stratified marginal MC",
}

GELS_SHAPLEY_CONFIGURATION = {
    "paper": (
        "Faster Approximation of Probabilistic and Distributional Values "
        "via Least Squares, ICLR 2024, Algorithm 3"
    ),
    "paper_url": "https://openreview.net/forum?id=lvSMIsztka",
    "official_repository": "https://github.com/watml/fastpvalue",
    "verified_commit": "34392c53f8d609aebb5e0c0e57c165411d291a46",
    "official_source_file": "utils/estimators.py",
    "downloaded_source_sha256": (
        "2eff99580289e3f2374a72b720baa0284a2fff71341be47d5b757c585ff6e2fa"
    ),
    "official_class": "GELS_shapley",
    "implementation": "independent_clean_room_port_from_paper_formula",
    "sampling": "iid_q_s_proportional_to_1_over_s_times_n_minus_s",
    "aggregation": "per_player_self_normalized_inclusion_ratio",
    "zero_count_fallback": "raw_ratio_zero_as_in_official_code",
    "physical_call_rule": "two_boundaries_plus_B_minus_two_inner_calls",
    "paired_sampling": False,
}

TMC_CONFIGURATION = {
    "source": (
        "/home/maoxiaokai/python_project/integral_shapley/"
        "src/core/tmc_shapley_methods.py"
    ),
    "source_sha256": (
        "d64548f00ceded3660764b8cd02d646b2548f4e19d0bee53e"
        "474d53dac3cbac1"
    ),
    "fixed_permutation_budget_rule": "floor((B - 2) / n)",
    "truncation_tolerance": 1e-3,
    "relative_tolerance": True,
    "resolved_tolerance_rule": "1e-3 * abs(v(full))",
    "truncation_patience": 5,
    "required_consecutive_near_full_prefixes": 6,
    "efficiency_projection": False,
    "utility_error_policy": "propagate",
    "full_prefix_re_evaluated_on_complete_paths": True,
}


def _source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compact_json(value: Any) -> Any:
    """Convert diagnostics without dumping multi-megabyte state matrices."""
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _compact_json(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        numeric = np.asarray(value)
        return {
            "omitted_dense_array": True,
            "shape": list(numeric.shape),
            "dtype": str(numeric.dtype),
            "minimum": float(numeric.min()),
            "maximum": float(numeric.max()),
            "sum": float(numeric.sum(dtype=np.float64)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _compact_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_checkpoint(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            default=_compact_json,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_comparison_source(
    source: Mapping[str, Any], *, repeats: int
) -> None:
    _validate_source_report(source)
    if repeats < 2:
        raise ValueError("the comparison requires at least two repeats")
    source_repeats = int(source["configuration"]["repeats"])
    if source_repeats < repeats:
        raise ValueError("the source report has too few stored repeats")
    for budget in source["configuration"]["inner_budgets"]:
        methods = source["results_by_inner_budget"][str(budget)]["methods"]
        missing = sorted(set(SOURCE_METHODS) - set(methods))
        if missing:
            raise ValueError(
                f"source budget {budget} is missing methods {missing}"
            )


def _summary_from_source(
    source_summary: Mapping[str, Any],
    *,
    repeats: int,
    truth: np.ndarray,
    truth_se: np.ndarray,
    efficiency_target: float,
    bootstrap_seed: int,
    method: str,
) -> dict[str, Any]:
    estimates = np.asarray(source_summary["estimates"], dtype=np.float64)
    selected = estimates[:repeats].copy()
    summary = summarize_method(
        selected,
        truth,
        truth_se,
        efficiency_target,
        bootstrap_seed=bootstrap_seed,
    )
    summary["source_repeat_indices"] = list(range(repeats))
    summary["reused_from_source_report"] = True
    if method == "official_cc_basic":
        fractions = [
            float(value)
            for value in source_summary["missing_stratum_fractions"][
                :repeats
            ]
        ]
        summary.update(
            {
                "utility_calls_per_estimate": int(
                    source_summary["utility_calls_per_estimate"]
                ),
                "complementary_contributions_per_estimate": int(
                    source_summary[
                        "complementary_contributions_per_estimate"
                    ]
                ),
                "missing_stratum_fractions": fractions,
                "mean_missing_stratum_fraction": float(
                    np.mean(fractions)
                ),
                "max_missing_stratum_fraction": float(
                    np.max(fractions)
                ),
                "cc_diagnostics_by_repeat": copy.deepcopy(
                    source_summary["cc_diagnostics_by_repeat"][:repeats]
                ),
            }
        )
    return summary


def _refresh_existing_comparisons(
    row: dict[str, Any], truth: np.ndarray, *, seed: int
) -> None:
    methods = row["methods"]
    official = float(methods["official_ofa_fixed_ratio"]["aggregate_rmse"])
    orbit = float(methods["frame_orbit_ratio"]["aggregate_rmse"])
    iid = float(methods["iid_linear_ofa"]["aggregate_rmse"])
    coupled = float(methods["frame_coupled_linear"]["aggregate_rmse"])
    cc = float(methods["official_cc_basic"]["aggregate_rmse"])
    row["orbit_rmse_reduction_vs_official_ratio"] = (
        official - orbit
    ) / official
    row["frame_coupled_rmse_reduction_vs_iid_linear"] = (
        iid - coupled
    ) / iid
    row["cc_rmse_reduction_vs_official_ratio"] = (official - cc) / official
    row["cc_missing_stratum_fraction_mean"] = float(
        methods["official_cc_basic"]["mean_missing_stratum_fraction"]
    )
    row["paired_rmse_differences"] = {
        "official_ratio_minus_orbit_ratio": paired_rmse_difference(
            np.asarray(methods["official_ofa_fixed_ratio"]["estimates"]),
            np.asarray(methods["frame_orbit_ratio"]["estimates"]),
            truth,
            seed=seed,
        ),
        "iid_linear_minus_frame_coupled_linear": paired_rmse_difference(
            np.asarray(methods["iid_linear_ofa"]["estimates"]),
            np.asarray(methods["frame_coupled_linear"]["estimates"]),
            truth,
            seed=seed + 1,
        ),
    }


def _new_comparison_report(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    repeats: int,
    methods: tuple[str, ...],
    seed: int,
    jobs: int,
    chunksize: int,
    num_tasks: int,
    start_method: str,
) -> dict[str, Any]:
    """Create a three-repeat report without mutating the source object."""
    _validate_comparison_source(source, repeats=repeats)
    report = copy.deepcopy(source)
    configuration = report["configuration"]
    configuration["repeats"] = repeats
    configuration["baseline_comparison"] = {
        "source_report": str(source_path),
        "source_sha256": _source_digest(source_path),
        "source_repeats": int(source["configuration"]["repeats"]),
        "source_repeat_indices": list(range(repeats)),
        "new_methods": list(methods),
        "method_labels": {
            method: METHOD_LABELS[method] for method in methods
        },
        "call_budget_semantics": "total physical utility-call cap",
        "seed": seed,
        "jobs": jobs,
        "chunksize": chunksize,
        "num_tasks": num_tasks,
        "start_method": start_method,
        "worker_threads": 1,
        "kernel_shap_ridge": 1e-8,
        "tmc_shapley": copy.deepcopy(TMC_CONFIGURATION),
        "gels_shapley": copy.deepcopy(GELS_SHAPLEY_CONFIGURATION),
    }
    if isinstance(configuration.get("cc_baseline"), Mapping):
        cc_metadata = dict(configuration["cc_baseline"])
        cc_metadata["source_report_additional_physical_utility_calls"] = (
            cc_metadata.get("additional_physical_utility_calls")
        )
        cc_metadata["additional_physical_utility_calls"] = 0
        cc_metadata["comparison_reused_from_source_report"] = True
        cc_metadata["comparison_source_repeat_indices"] = list(
            range(repeats)
        )
        configuration["cc_baseline"] = cc_metadata
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    truth_se = np.asarray(
        report["ground_truth"]["standard_errors"], dtype=np.float64
    )
    efficiency_target = float(report["boundary"]["efficiency_target"])
    for budget_index, budget in enumerate(configuration["inner_budgets"]):
        row = report["results_by_inner_budget"][str(budget)]
        source_methods = row["methods"]
        row["methods"] = {
            method: _summary_from_source(
                source_methods[method],
                repeats=repeats,
                truth=truth,
                truth_se=truth_se,
                efficiency_target=efficiency_target,
                bootstrap_seed=seed + budget_index * 100 + method_index,
                method=method,
            )
            for method_index, method in enumerate(SOURCE_METHODS)
        }
        _refresh_existing_comparisons(
            row, truth, seed=seed + budget_index * 1000
        )
    report["status"] = "baseline_augmentation_running"
    return report


def _initialize_or_resume(
    *,
    source: Mapping[str, Any],
    source_path: Path,
    output_path: Path,
    repeats: int,
    methods: tuple[str, ...],
    seed: int,
    jobs: int,
    chunksize: int,
    num_tasks: int,
    start_method: str,
) -> dict[str, Any]:
    expected_source_hash = _source_digest(source_path)
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        metadata = report.get("configuration", {}).get(
            "baseline_comparison", {}
        )
        expected = {
            "source_sha256": expected_source_hash,
            "source_repeat_indices": list(range(repeats)),
            "new_methods": list(methods),
            "seed": seed,
            "jobs": jobs,
            "chunksize": chunksize,
            "num_tasks": num_tasks,
            "start_method": start_method,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"existing output uses a different {key} setting"
                )
        return report
    report = _new_comparison_report(
        source,
        source_path=source_path,
        repeats=repeats,
        methods=methods,
        seed=seed,
        jobs=jobs,
        chunksize=chunksize,
        num_tasks=num_tasks,
        start_method=start_method,
    )
    _write_checkpoint(output_path, report)
    return report


def _estimate_method(
    method: str,
    evaluator: GameEvaluator,
    *,
    num_players: int,
    total_call_budget: int,
    seed: int,
    num_tasks: int,
) -> Any:
    if method == "gels_shapley":
        from frame_ofa.regression_baselines import estimate_gels_shapley

        return estimate_gels_shapley(
            evaluator, num_players, total_call_budget, seed,
            num_tasks=num_tasks,
        )
    if method == "kernel_shap_sampled":
        from frame_ofa.regression_baselines import estimate_kernel_shap

        return estimate_kernel_shap(
            evaluator,
            num_players,
            total_call_budget,
            seed,
            num_tasks=num_tasks,
            ridge=1e-8,
        )
    if method == "group_testing":
        from frame_ofa.group_testing import estimate_group_testing

        return estimate_group_testing(
            evaluator, num_players, total_call_budget, seed,
            num_tasks=num_tasks,
        )
    if method == "diff":
        from frame_ofa.differential import estimate_diff

        return estimate_diff(
            evaluator, num_players, total_call_budget, seed,
            num_tasks=num_tasks,
        )
    if method == "s_diff":
        from frame_ofa.stratified_differential import estimate_sdiff

        return estimate_sdiff(
            evaluator, num_players, total_call_budget, seed,
            num_tasks=num_tasks,
        )
    if method == "tmc_shapley":
        from frame_ofa.tmc import estimate_tmc_shapley

        return estimate_tmc_shapley(
            evaluator, num_players, total_call_budget, seed,
            num_tasks=num_tasks,
        )
    if method == "stratified_marginal_mc":
        from frame_ofa.traditional_mc import (
            estimate_stratified_marginal_mc,
        )

        return estimate_stratified_marginal_mc(
            evaluator, num_players, total_call_budget, seed,
            num_tasks=num_tasks,
        )
    raise ValueError(f"unsupported baseline method: {method}")


def _diagnostic(diagnostics: Any, name: str) -> Any:
    if isinstance(diagnostics, Mapping):
        return diagnostics[name]
    return getattr(diagnostics, name)


def _assert_dataset_match(
    stored: Mapping[str, Any], reconstructed: Mapping[str, Any]
) -> None:
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
        if key in {"standardization_mean", "standardization_std"}:
            equal = np.array_equal(
                np.asarray(stored[key], dtype=np.float64),
                np.asarray(reconstructed[key], dtype=np.float64),
            )
        else:
            equal = stored[key] == reconstructed[key]
        if not equal:
            raise ValueError(f"reconstructed dataset differs at {key}")


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if args.input.resolve() == args.output.resolve():
        raise ValueError("--output must differ from --input")
    methods = tuple(args.methods)
    if len(set(methods)) != len(methods):
        raise ValueError("--methods must be distinct")
    unknown = sorted(set(methods) - set(DEFAULT_NEW_METHODS))
    if unknown:
        raise ValueError(f"unknown baseline methods: {unknown}")

    source = json.loads(args.input.read_text(encoding="utf-8"))
    report = _initialize_or_resume(
        source=source,
        source_path=args.input,
        output_path=args.output,
        repeats=args.repeats,
        methods=methods,
        seed=args.seed,
        jobs=args.jobs,
        chunksize=args.chunksize,
        num_tasks=args.num_tasks,
        start_method=args.start_method,
    )
    configuration = report["configuration"]
    num_players = int(configuration["num_players"])
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    truth_se = np.asarray(
        report["ground_truth"]["standard_errors"], dtype=np.float64
    )
    efficiency_target = float(report["boundary"]["efficiency_target"])

    game_args, reconstructed = load_sklearn_train_test_split(
        "wine",
        test_size=float(report["dataset"]["test_size"]),
        dataset_seed=int(report["dataset"]["dataset_seed"]),
    )
    _assert_dataset_match(report["dataset"], reconstructed)
    expected_model = (
        "sklearn.svm.SVC(C="
        f"{args.regularization}, kernel='rbf', gamma='scale')"
    )
    if configuration.get("model") != expected_model:
        raise ValueError("source model and requested regularization differ")
    game_args = game_args | {
        "model": "rbf_svm",
        "regularization": args.regularization,
    }

    pending = any(
        method
        not in report["results_by_inner_budget"][str(budget)]["methods"]
        for method in methods
        for budget in configuration["inner_budgets"]
    )
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
        for method_index, method in enumerate(methods):
            for budget_index, inner_budget in enumerate(
                configuration["inner_budgets"]
            ):
                row = report["results_by_inner_budget"][str(inner_budget)]
                if method in row["methods"]:
                    print(
                        f"{method} budget {inner_budget:,}: complete, skip",
                        flush=True,
                    )
                    continue
                call_cap = int(row["total_utility_calls_per_estimate"])
                print(
                    f"{method} budget {budget_index + 1}/5: "
                    f"cap={call_cap:,}, repeats={args.repeats}",
                    flush=True,
                )
                estimates: list[np.ndarray] = []
                seconds: list[float] = []
                calls: list[int] = []
                unused: list[int] = []
                diagnostics_by_repeat: list[dict[str, Any]] = []
                for repeat in range(args.repeats):
                    method_seed = (
                        args.seed
                        + method_index * 10_000_000
                        + budget_index * 100_000
                        + repeat
                    )
                    started = time.perf_counter()
                    result = _estimate_method(
                        method,
                        evaluator,
                        num_players=num_players,
                        total_call_budget=call_cap,
                        seed=method_seed,
                        num_tasks=args.num_tasks,
                    )
                    seconds.append(time.perf_counter() - started)
                    estimate = np.asarray(result.values, dtype=np.float64)
                    if estimate.shape != (num_players,):
                        raise RuntimeError(
                            f"{method} returned invalid estimate shape"
                        )
                    if not np.isfinite(estimate).all():
                        raise RuntimeError(
                            f"{method} returned non-finite estimates"
                        )
                    actual_calls = int(
                        _diagnostic(
                            result.diagnostics, "utility_evaluations"
                        )
                    )
                    target_calls = int(
                        _diagnostic(
                            result.diagnostics, "target_call_budget"
                        )
                    )
                    unused_calls = int(
                        _diagnostic(result.diagnostics, "unused_calls")
                    )
                    if target_calls != call_cap:
                        raise RuntimeError(
                            f"{method} target-call accounting disagrees"
                        )
                    if actual_calls + unused_calls != call_cap:
                        raise RuntimeError(
                            f"{method} physical-call accounting disagrees"
                        )
                    estimates.append(estimate)
                    calls.append(actual_calls)
                    unused.append(unused_calls)
                    diagnostics_by_repeat.append(
                        _compact_json(result.diagnostics)
                    )
                    print(
                        f"  repeat {repeat + 1}/{args.repeats}: "
                        f"calls={actual_calls:,}, unused={unused_calls:,}, "
                        f"{seconds[-1]:.2f}s",
                        flush=True,
                    )

                estimate_array = np.asarray(estimates, dtype=np.float64)
                summary = summarize_method(
                    estimate_array,
                    truth,
                    truth_se,
                    efficiency_target,
                    bootstrap_seed=(
                        args.seed + method_index * 10_000 + budget_index
                    ),
                )
                summary.update(
                    {
                        "method_label": METHOD_LABELS[method],
                        "target_total_call_budget": call_cap,
                        "actual_utility_calls_per_estimate": calls,
                        "unused_calls_per_estimate": unused,
                        "mean_actual_utility_calls": float(np.mean(calls)),
                        "mean_unused_calls": float(np.mean(unused)),
                        "total_seconds_by_repeat": seconds,
                        "mean_total_seconds": float(np.mean(seconds)),
                        "diagnostics_by_repeat": diagnostics_by_repeat,
                    }
                )
                row["methods"][method] = summary
                _write_checkpoint(args.output, report)
                print(
                    f"  aggregate RMSE={summary['aggregate_rmse']:.4e}",
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
            "results/"
            "wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_"
            "3repeats_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_NEW_METHODS,
        default=list(DEFAULT_NEW_METHODS),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2_910_001)
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=512)
    parser.add_argument("--num-tasks", type=int, default=128)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--regularization", type=float, default=1.0)
    args = parser.parse_args()
    if not args.input.exists():
        raise ValueError(f"input report does not exist: {args.input}")
    if args.repeats < 2:
        raise ValueError("repeats must be at least two")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if args.jobs < 1 or args.chunksize < 1 or args.num_tasks < 1:
        raise ValueError("jobs, chunksize, and num-tasks must be positive")
    if args.regularization <= 0:
        raise ValueError("regularization must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_comparison(args)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

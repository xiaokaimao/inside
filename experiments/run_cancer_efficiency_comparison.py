"""Build the estimated-time Efficiency experiment for Cancer valuation.

The completed and independently validated Cancer comparison is reused.  The
only new numerical work is a small scalar-utility timing calibration.  For
every repeat and budget we apply the same conservative proxy as the Airport
and Wine Efficiency experiments::

    estimated_time = actual_calls * median_scalar_utility_seconds
                     + I[method == INSIDE-Greedy] * design_seconds

Actual physical calls preserve TMC-Shapley truncation.  Only the measured
production design time of INSIDE-Greedy is charged; all baseline overheads are
set to zero, which deliberately favours the baselines.  Consistent with the
Cancer main comparison, S-Diff is omitted from presentation under the stated
quadratic-memory resource policy.  OFA is the requested ratio estimator.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np

from experiments.iris_sklearn_game import SklearnClassificationGame
from experiments.run_cancer_inside_comparison import (
    DATASET,
    DATASET_SEED,
    NUM_PLAYERS,
    NUM_TEST,
    TEST_SIZE,
    validate_reconstructed_dataset,
)
from experiments.sklearn_data import load_sklearn_train_test_split
from experiments.validate_inside_comparison import (
    validate_report as validate_source_report,
)
from frame_ofa import iid_ofa_design


EXPERIMENT_ID = "cancer_estimated_time_efficiency_k64_lambda1over16"
METHOD_ORDER = (
    "inside_greedy",
    "inside_orbit",
    "ofa",
    "cc",
    "kernel_shap",
    "tmc_shapley",
)
METHOD_LABELS = {
    "inside_greedy": "INSIDE-Greedy",
    "inside_orbit": "INSIDE-Orbit",
    "ofa": "OFA",
    "cc": "CC",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}
SOURCE_METHODS = {
    "inside_greedy": "inside_greedy",
    "inside_orbit": "inside_orbit",
    "ofa": "ofa_iid_ratio",
    "cc": "cc",
    "kernel_shap": "kernel_shap",
    "tmc_shapley": "tmc_shapley",
}
DEFAULT_INPUT = Path(
    "results/json/cancer_inside_comparison_per_size_ratio_k64_lambda1over16_"
    "3repeats_228k_4p55m.json"
)
DEFAULT_OUTPUT = Path("results/json/cancer_efficiency_k64_lambda1over16.json")
DEFAULT_CSV = Path("results/csv/cancer_efficiency_k64_lambda1over16.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite(value: Any, *, name: str, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _actual_calls_by_repeat(
    summary: Mapping[str, Any], *, repeats: int, target: int
) -> list[float]:
    raw = summary.get("actual_utility_calls_by_repeat")
    if raw is None:
        raw = summary.get("actual_utility_calls_per_estimate")
    if raw is None:
        raw = [summary.get("mean_actual_utility_calls", target)] * repeats
    elif isinstance(raw, (int, float, np.integer, np.floating)):
        raw = [raw] * repeats
    values = [
        _finite(value, name="actual utility calls", positive=True)
        for value in raw
    ]
    if len(values) != repeats:
        raise ValueError("actual utility-call vector has the wrong length")
    if any(value > target + 1e-9 for value in values):
        raise ValueError("actual utility calls exceed the target budget")
    return values


def _validate_source_identity(report: Mapping[str, Any]) -> dict[str, Any]:
    audit = validate_source_report(report)
    if audit.get("status") != "ready_to_share":
        raise ValueError("Cancer source report failed its independent audit")
    if report.get("dataset", {}).get("dataset") != DATASET:
        raise ValueError("source report is not Cancer")
    configuration = report.get("configuration", {})
    if int(configuration.get("num_players", -1)) != NUM_PLAYERS:
        raise ValueError("Cancer source player count is wrong")
    if int(configuration.get("n_test", -1)) != NUM_TEST:
        raise ValueError("Cancer source test count is wrong")
    if int(configuration.get("repeats", -1)) != 3:
        raise ValueError("Cancer Efficiency requires three repeats")
    greedy = configuration.get("inside", {}).get("greedy", {})
    expected = {
        "candidate_pool": 64,
        "mean_balance_mode": "normalized",
        "mean_balance_lambda0": 1.0 / 16.0,
        "second_moment_scope": "per_size",
        "estimator": "OFA conditional-mean ratio",
    }
    for key, value in expected.items():
        observed = greedy.get(key)
        if isinstance(value, float):
            if not math.isclose(
                float(observed), value, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError(f"Cancer Greedy {key} is wrong")
        elif observed != value:
            raise ValueError(f"Cancer Greedy {key} is wrong")
    configured_methods = set(configuration.get("methods", []))
    required = set(SOURCE_METHODS.values()) | {"ofa_iid_linear", "s_diff"}
    if not required.issubset(configured_methods):
        raise ValueError("Cancer source method configuration is incomplete")
    return audit


def _reconstruct_cancer_game(
    source: Mapping[str, Any],
) -> tuple[SklearnClassificationGame, dict[str, Any]]:
    game_args, metadata = load_sklearn_train_test_split(
        DATASET, test_size=TEST_SIZE, dataset_seed=DATASET_SEED
    )
    validate_reconstructed_dataset(source["dataset"], metadata)
    game = SklearnClassificationGame(
        **(game_args | {"model": "rbf_svm", "regularization": 1.0})
    )
    if game.num_players != NUM_PLAYERS:
        raise RuntimeError("reconstructed Cancer game has the wrong size")
    return game, metadata


def _benchmark_scalar_cancer_utility(
    source: Mapping[str, Any],
    *,
    calls_per_block: int,
    blocks: int,
    warmup_calls: int,
    seed: int,
) -> dict[str, Any]:
    """Benchmark scalar RBF-SVM accuracy calls on one fixed Q* batch."""
    if calls_per_block < 1 or blocks < 2 or warmup_calls < 0:
        raise ValueError("utility benchmark sizes are invalid")
    game, _ = _reconstruct_cancer_game(source)
    design = iid_ofa_design(
        NUM_PLAYERS,
        calls_per_block,
        seed=seed,
        compute_diagnostics=False,
    )
    coalitions = np.asarray(design.coalitions, dtype=bool)
    if coalitions.shape != (calls_per_block, NUM_PLAYERS):
        raise RuntimeError("Cancer utility benchmark design has the wrong shape")

    warmup = coalitions[: min(warmup_calls, calls_per_block)]
    warmup_checksum = 0.0
    for coalition in warmup:
        warmup_checksum += float(game.evaluate(coalition))

    block_seconds: list[float] = []
    block_checksums: list[float] = []
    for _ in range(blocks):
        checksum = 0.0
        started = time.perf_counter()
        for coalition in coalitions:
            checksum += float(game.evaluate(coalition))
        block_seconds.append(time.perf_counter() - started)
        block_checksums.append(checksum)
    if not np.allclose(
        block_checksums,
        np.repeat(block_checksums[0], blocks),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Cancer utility benchmark is not deterministic")

    seconds_per_call = np.asarray(block_seconds, dtype=np.float64) / calls_per_block
    sizes = coalitions.sum(axis=1, dtype=np.int64)
    return {
        "protocol": (
            "warmed repeated scalar RBF-SVC accuracy calls on a fixed Q* "
            "coalition batch"
        ),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        "seed": int(seed),
        "calls_per_block": int(calls_per_block),
        "blocks": int(blocks),
        "warmup_calls": int(len(warmup)),
        "warmup_checksum": float(warmup_checksum),
        "scalar_block_checksums": [float(value) for value in block_checksums],
        "scalar_block_seconds": [float(value) for value in block_seconds],
        "scalar_seconds_per_call_by_block": [
            float(value) for value in seconds_per_call
        ],
        "median_scalar_seconds_per_call": float(np.median(seconds_per_call)),
        "mean_scalar_seconds_per_call": float(np.mean(seconds_per_call)),
        "minimum_scalar_seconds_per_call": float(np.min(seconds_per_call)),
        "maximum_scalar_seconds_per_call": float(np.max(seconds_per_call)),
        "coalition_distribution": (
            "Shapley-specific OFA Q*: q_s proportional to "
            "1/sqrt(s(n-s)), uniform within size"
        ),
        "coalition_batch_sha256": hashlib.sha256(
            np.packbits(coalitions, axis=1).tobytes()
        ).hexdigest(),
        "coalition_size_summary": {
            "minimum": int(sizes.min()),
            "median": float(np.median(sizes)),
            "mean": float(np.mean(sizes)),
            "maximum": int(sizes.max()),
        },
        "model": "sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='scale')",
        "utility": "fixed-test classification accuracy",
        "timing_clock": "time.perf_counter",
        "used_for_estimated_time": "median_scalar_seconds_per_call",
    }


def _greedy_design_records(
    source_row: Mapping[str, Any], *, repeats: int
) -> tuple[list[float], list[dict[str, Any]]]:
    summary = source_row.get("methods", {}).get("inside_greedy")
    if not isinstance(summary, Mapping):
        raise ValueError("Cancer source lacks INSIDE-Greedy")
    diagnostics = summary.get("diagnostics_by_repeat")
    if not isinstance(diagnostics, list) or len(diagnostics) != repeats:
        raise ValueError("Cancer Greedy design-time grid is incomplete")
    seconds: list[float] = []
    provenance: list[dict[str, Any]] = []
    for repeat, diagnostic in enumerate(diagnostics):
        if not isinstance(diagnostic, Mapping):
            raise ValueError("Cancer Greedy diagnostics are malformed")
        design_diagnostics = diagnostic.get("design_diagnostics", {})
        elapsed = _finite(
            diagnostic.get("design_seconds"),
            name="Greedy design seconds",
            positive=True,
        )
        if int(diagnostic.get("candidate_pool", -1)) != 64:
            raise ValueError("Cancer Greedy candidate pool is wrong")
        if diagnostic.get("second_moment_scope") != "per_size":
            raise ValueError("Cancer Greedy second-moment scope is wrong")
        seconds.append(elapsed)
        provenance.append(
            {
                "repeat": repeat,
                "seconds": elapsed,
                "candidate_pool": int(diagnostic["candidate_pool"]),
                "second_moment_scope": str(diagnostic["second_moment_scope"]),
                "mean_balance_lambda0": float(
                    diagnostic["mean_balance_lambda0"]
                ),
                "relabel_seed": (
                    None
                    if design_diagnostics["relabel_seed"] is None
                    else int(design_diagnostics["relabel_seed"])
                ),
                "relabel_permutation_sha256": str(
                    design_diagnostics["relabel_permutation_sha256"]
                ),
            }
        )
    return seconds, provenance


def build_efficiency_report(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    utility_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the audited estimated-time proxy to the Cancer report."""
    source_audit = _validate_source_identity(source)
    tau = _finite(
        utility_calibration.get("median_scalar_seconds_per_call"),
        name="median scalar utility time",
        positive=True,
    )
    configuration = source["configuration"]
    inner_budgets = [int(value) for value in configuration["inner_budgets"]]
    total_budgets = [int(value) for value in configuration["total_call_budgets"]]
    if len(inner_budgets) != 5 or len(total_budgets) != 5:
        raise ValueError("Cancer source must contain five budget points")
    repeats = int(configuration["repeats"])

    result_rows: list[dict[str, Any]] = []
    for budget_index, (inner, target) in enumerate(
        zip(inner_budgets, total_budgets, strict=True)
    ):
        source_row = source["results_by_inner_budget"].get(str(inner))
        if not isinstance(source_row, Mapping):
            raise ValueError(f"Cancer source lacks inner budget {inner}")
        if int(source_row["total_utility_calls_per_estimate"]) != target:
            raise ValueError("Cancer source budget accounting is inconsistent")
        design_seconds, design_provenance = _greedy_design_records(
            source_row, repeats=repeats
        )
        source_methods = source_row["methods"]
        methods: dict[str, Any] = {}
        for method in METHOD_ORDER:
            source_method = SOURCE_METHODS[method]
            summary = source_methods.get(source_method)
            if not isinstance(summary, Mapping):
                raise ValueError(f"Cancer source lacks method {source_method}")
            aggregate_rmse = _finite(
                summary.get("aggregate_rmse"),
                name=f"{method} aggregate RMSE",
                positive=True,
            )
            interval = [
                _finite(value, name=f"{method} RMSE interval", positive=True)
                for value in summary.get("aggregate_rmse_bootstrap_95", [])
            ]
            if len(interval) != 2 or interval[0] > interval[1]:
                raise ValueError(f"{method} RMSE interval is invalid")
            actual_calls = _actual_calls_by_repeat(
                summary, repeats=repeats, target=target
            )
            charged_design = (
                design_seconds if method == "inside_greedy" else [0.0] * repeats
            )
            estimated_utility = [calls * tau for calls in actual_calls]
            estimated_total = [
                utility + design
                for utility, design in zip(
                    estimated_utility, charged_design, strict=True
                )
            ]
            methods[method] = {
                "label": METHOD_LABELS[method],
                "aggregate_rmse": aggregate_rmse,
                "aggregate_rmse_bootstrap_95": interval,
                "actual_utility_calls_by_repeat": actual_calls,
                "mean_actual_utility_calls": float(np.mean(actual_calls)),
                "estimated_utility_seconds_by_repeat": estimated_utility,
                "mean_estimated_utility_seconds": float(np.mean(estimated_utility)),
                "charged_design_seconds_by_repeat": charged_design,
                "mean_charged_design_seconds": float(np.mean(charged_design)),
                "charged_design_provenance_by_repeat": (
                    design_provenance if method == "inside_greedy" else []
                ),
                "estimated_total_seconds_by_repeat": estimated_total,
                "mean_estimated_total_seconds": float(np.mean(estimated_total)),
                "estimated_total_seconds_range": [
                    float(np.min(estimated_total)),
                    float(np.max(estimated_total)),
                ],
                "source_method": source_method,
            }
        result_rows.append(
            {
                "budget_index": budget_index,
                "inner_utility_calls": inner,
                "target_total_utility_calls": target,
                "methods": methods,
            }
        )

    report = {
        "status": "complete",
        "experiment": EXPERIMENT_ID,
        "configuration": {
            "dataset": DATASET,
            "players": NUM_PLAYERS,
            "test_observations": NUM_TEST,
            "repeats": repeats,
            "methods": list(METHOD_ORDER),
            "method_labels": METHOD_LABELS,
            "omitted_source_methods": [
                {
                    "method": "ofa_iid_linear",
                    "reason": "OFA is represented by the requested ratio estimator",
                },
                {
                    "method": "s_diff",
                    "reason": (
                        "quadratic-memory baseline omitted from the Cancer "
                        "presentation under the stated resource policy"
                    ),
                    "actual_oom_observed": False,
                },
            ],
            "time_formula": (
                "actual_utility_calls * median_scalar_utility_seconds + "
                "I[method=INSIDE-Greedy] * measured_design_seconds"
            ),
            "time_interpretation": "estimated utility-dominated one-shot time",
            "actual_calls_policy": (
                "use per-repeat physical calls; preserves TMC truncation"
            ),
            "charged_nonutility_overhead": [
                "INSIDE-Greedy production design time"
            ],
            "omitted_nonutility_overhead": [
                "INSIDE-Orbit design and aggregation",
                "OFA sampling and aggregation",
                "CC pairing and aggregation",
                "KernelSHAP sampling and linear solve",
                "TMC permutation and truncation bookkeeping",
            ],
            "baseline_favouring_caveat": (
                "Only INSIDE-Greedy is charged a nonutility overhead, so the "
                "proxy is conservative in favour of every displayed baseline."
            ),
            "utility_cost_caveat": (
                "RBF-SVM utility cost depends on coalition composition and "
                "size; one common Q*-averaged scalar calibration is used as "
                "the requested cross-method proxy, not measured end-to-end time."
            ),
            "greedy_design_protocol": {
                "candidate_pool": 64,
                "mean_balance_lambda0": 1.0 / 16.0,
                "mean_balance_mode": "normalized",
                "second_moment_scope": "per_size",
                "estimator": "OFA conditional-mean ratio",
            },
            "source_report": {
                "path": str(source_path.resolve()),
                "sha256": _sha256(source_path.resolve()),
            },
        },
        "dataset": source["dataset"],
        "game": source["game"],
        "ground_truth": source["ground_truth"],
        "utility_calibration": dict(utility_calibration),
        "source_audit": source_audit,
        "results": result_rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    validate_efficiency_report(report)
    return report


def validate_efficiency_report(report: Mapping[str, Any]) -> None:
    """Recompute every scalar identity consumed by the Efficiency figure."""
    if report.get("status") != "complete":
        raise ValueError("Cancer Efficiency report is incomplete")
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("unexpected Cancer Efficiency experiment")
    configuration = report.get("configuration", {})
    if configuration.get("methods") != list(METHOD_ORDER):
        raise ValueError("Cancer Efficiency method order is wrong")
    if configuration.get("charged_nonutility_overhead") != [
        "INSIDE-Greedy production design time"
    ]:
        raise ValueError("Cancer Efficiency overhead policy is wrong")
    omissions = configuration.get("omitted_source_methods")
    if not isinstance(omissions, list) or {
        item.get("method") for item in omissions if isinstance(item, Mapping)
    } != {"ofa_iid_linear", "s_diff"}:
        raise ValueError("Cancer Efficiency omissions are wrong")
    tau = _finite(
        report.get("utility_calibration", {}).get(
            "median_scalar_seconds_per_call"
        ),
        name="median scalar utility time",
        positive=True,
    )
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Cancer Efficiency report must contain five rows")
    previous_target = 0
    for expected_index, row in enumerate(rows):
        if int(row.get("budget_index", -1)) != expected_index:
            raise ValueError("Cancer Efficiency budget indices are wrong")
        target = int(row.get("target_total_utility_calls", 0))
        if target <= previous_target:
            raise ValueError("Cancer Efficiency call budgets must increase")
        previous_target = target
        methods = row.get("methods")
        if not isinstance(methods, Mapping) or set(methods) != set(METHOD_ORDER):
            raise ValueError("Cancer Efficiency row method set is wrong")
        for method in METHOD_ORDER:
            summary = methods[method]
            if summary.get("label") != METHOD_LABELS[method]:
                raise ValueError(f"{method} has the wrong label")
            calls = np.asarray(
                summary.get("actual_utility_calls_by_repeat"), dtype=np.float64
            )
            designs = np.asarray(
                summary.get("charged_design_seconds_by_repeat"), dtype=np.float64
            )
            utility = np.asarray(
                summary.get("estimated_utility_seconds_by_repeat"), dtype=np.float64
            )
            totals = np.asarray(
                summary.get("estimated_total_seconds_by_repeat"), dtype=np.float64
            )
            if any(
                array.shape != (3,) for array in (calls, designs, utility, totals)
            ):
                raise ValueError(f"{method} repeat vectors have the wrong shape")
            if (
                not all(np.all(np.isfinite(array)) for array in (calls, designs, utility, totals))
                or np.any(calls <= 0.0)
                or np.any(calls > target)
            ):
                raise ValueError(f"{method} repeat vectors are invalid")
            if method == "inside_greedy":
                if np.any(designs <= 0.0):
                    raise ValueError("Greedy design time must be positive")
            elif np.any(designs != 0.0):
                raise ValueError(f"{method} must not be charged design time")
            if not np.allclose(utility, calls * tau, rtol=0.0, atol=1e-10):
                raise ValueError(f"{method} utility times are inconsistent")
            if not np.allclose(
                totals, utility + designs, rtol=0.0, atol=1e-10
            ):
                raise ValueError(f"{method} estimated times are inconsistent")
            expected_means = {
                "mean_actual_utility_calls": float(np.mean(calls)),
                "mean_estimated_utility_seconds": float(np.mean(utility)),
                "mean_charged_design_seconds": float(np.mean(designs)),
                "mean_estimated_total_seconds": float(np.mean(totals)),
            }
            for key, expected_mean in expected_means.items():
                observed = _finite(summary.get(key), name=f"{method} {key}")
                if not math.isclose(
                    observed, expected_mean, rel_tol=0.0, abs_tol=1e-10
                ):
                    raise ValueError(f"{method} {key} is inconsistent")
            time_range = np.asarray(
                summary.get("estimated_total_seconds_range"), dtype=np.float64
            )
            if time_range.shape != (2,) or not np.allclose(
                time_range,
                np.asarray([np.min(totals), np.max(totals)]),
                rtol=0.0,
                atol=1e-10,
            ):
                raise ValueError(f"{method} estimated-time range is inconsistent")
            _finite(
                summary.get("aggregate_rmse"),
                name=f"{method} aggregate RMSE",
                positive=True,
            )
            interval = np.asarray(
                summary.get("aggregate_rmse_bootstrap_95"), dtype=np.float64
            )
            if (
                interval.shape != (2,)
                or not np.all(np.isfinite(interval))
                or np.any(interval <= 0.0)
                or interval[0] > interval[1]
            ):
                raise ValueError(f"{method} RMSE interval is invalid")
            provenance = summary.get("charged_design_provenance_by_repeat")
            if not isinstance(provenance, list):
                raise ValueError(f"{method} design provenance must be a list")
            if method == "inside_greedy":
                if len(provenance) != 3:
                    raise ValueError("Greedy design provenance is incomplete")
                if not np.allclose(
                    np.asarray([item["seconds"] for item in provenance]),
                    designs,
                    rtol=0.0,
                    atol=0.0,
                ):
                    raise ValueError("Greedy design provenance is inconsistent")
            elif provenance:
                raise ValueError(f"{method} must not have design provenance")


def write_efficiency_csv(path: Path, report: Mapping[str, Any]) -> None:
    validate_efficiency_report(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "budget_index",
                "inner_utility_calls",
                "target_total_utility_calls",
                "method",
                "label",
                "aggregate_rmse",
                "rmse_ci_lower",
                "rmse_ci_upper",
                "mean_actual_utility_calls",
                "mean_estimated_utility_seconds",
                "mean_charged_design_seconds",
                "mean_estimated_total_seconds",
            ),
        )
        writer.writeheader()
        for row in report["results"]:
            for method in METHOD_ORDER:
                summary = row["methods"][method]
                interval = summary["aggregate_rmse_bootstrap_95"]
                writer.writerow(
                    {
                        "budget_index": row["budget_index"],
                        "inner_utility_calls": row["inner_utility_calls"],
                        "target_total_utility_calls": row[
                            "target_total_utility_calls"
                        ],
                        "method": method,
                        "label": summary["label"],
                        "aggregate_rmse": summary["aggregate_rmse"],
                        "rmse_ci_lower": interval[0],
                        "rmse_ci_upper": interval[1],
                        "mean_actual_utility_calls": summary[
                            "mean_actual_utility_calls"
                        ],
                        "mean_estimated_utility_seconds": summary[
                            "mean_estimated_utility_seconds"
                        ],
                        "mean_charged_design_seconds": summary[
                            "mean_charged_design_seconds"
                        ],
                        "mean_estimated_total_seconds": summary[
                            "mean_estimated_total_seconds"
                        ],
                    }
                )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--calls-per-block", type=int, default=1_000)
    parser.add_argument("--blocks", type=int, default=7)
    parser.add_argument("--warmup-calls", type=int, default=100)
    parser.add_argument("--benchmark-seed", type=int, default=20260902)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = args.input.resolve()
    if not source_path.exists():
        raise ValueError(f"Cancer source report is missing: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    _validate_source_identity(source)
    calibration = _benchmark_scalar_cancer_utility(
        source,
        calls_per_block=args.calls_per_block,
        blocks=args.blocks,
        warmup_calls=args.warmup_calls,
        seed=args.benchmark_seed,
    )
    report = build_efficiency_report(
        source,
        source_path=source_path,
        utility_calibration=calibration,
    )
    _atomic_write_json(args.output.resolve(), report)
    write_efficiency_csv(args.csv_output.resolve(), report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "csv": str(args.csv_output.resolve()),
                "median_scalar_seconds_per_call": calibration[
                    "median_scalar_seconds_per_call"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

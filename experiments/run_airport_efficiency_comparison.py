"""Build an estimated-time efficiency comparison for the Airport game.

This experiment does not rerun any Shapley estimator.  It combines the latest
Airport INSIDE ablation with the completed all-baseline report, benchmarks one
scalar Airport utility call, and converts each method's *actual* utility-call
count into a common time proxy.  The required production design time is added
for INSIDE-Greedy only, exactly as requested::

    estimated_time = actual_calls * median_utility_seconds
                     + inside_greedy_design_seconds

The result is deliberately called an estimated utility-dominated time, not a
measured end-to-end runtime.  Algebraic overhead for the other methods is set
to zero, which is conservative in favour of the baselines.
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
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.airport_game import AIRPORT_COSTS
from experiments.run_airport_inside_ablations import (
    validate_report as validate_ablation_report,
)
from experiments.validate_airport_all_baselines import (
    validate_report as validate_baseline_report,
)
from frame_ofa import evaluate_airport, iid_ofa_design


EXPERIMENT_ID = "airport_estimated_time_efficiency_k64_lambda1over16"
METHOD_ORDER = (
    "inside_greedy",
    "inside_orbit",
    "ofa",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)
METHOD_LABELS = {
    "inside_greedy": "INSIDE-Greedy",
    "inside_orbit": "INSIDE-Orbit",
    "ofa": "OFA",
    "cc": "CC",
    "s_diff": "S-Diff",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}
ABLATION_METHODS = {
    "inside_greedy": "full_inside",
    "inside_orbit": "inside_orbit",
}
BASELINE_METHODS = {
    "ofa": "official_ofa_fixed_ratio",
    "cc": "official_cc_basic",
    "s_diff": "s_diff",
    "kernel_shap": "kernel_shap_sampled",
    "tmc_shapley": "tmc_shapley",
}
DEFAULT_ABLATION = Path(
    "results/json/airport_inside_ablations_k64_lambda1over16.json"
)
DEFAULT_BASELINES = Path(
    "results/json/airport_100_all_baselines_3repeats_50k_1m.json"
)
DEFAULT_OUTPUT = Path(
    "results/json/airport_efficiency_k64_lambda1over16.json"
)
DEFAULT_CSV = Path(
    "results/csv/airport_efficiency_k64_lambda1over16.csv"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
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


def _ordered_ablation_rows(
    report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    budgets = tuple(
        int(value)
        for value in report["configuration"]["inner_utility_call_budgets"]
    )
    rows = report["results_by_inner_budget"]
    if set(rows) != {str(value) for value in budgets}:
        raise ValueError("ablation budget keys do not match its configuration")
    return [rows[str(value)] for value in budgets]


def _baseline_rows_by_inner(
    report: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    rows = report["results_by_inner_budget"]
    result = {int(key): value for key, value in rows.items()}
    if len(result) != 5:
        raise ValueError("baseline report must contain five budgets")
    return result


def _benchmark_scalar_airport_utility(
    *, calls_per_block: int, blocks: int, warmup_calls: int, seed: int
) -> dict[str, Any]:
    if calls_per_block < 1 or blocks < 2 or warmup_calls < 0:
        raise ValueError("utility benchmark sizes are invalid")
    design = iid_ofa_design(
        len(AIRPORT_COSTS),
        calls_per_block,
        seed=seed,
        compute_diagnostics=False,
    )
    coalitions = np.asarray(design.coalitions, dtype=bool)
    if len(coalitions) != calls_per_block:
        raise RuntimeError("utility benchmark design has the wrong size")

    warmup = coalitions[: min(warmup_calls, calls_per_block)]
    warmup_checksum = 0.0
    for coalition in warmup:
        warmup_checksum += float(evaluate_airport(coalition, AIRPORT_COSTS))

    scalar_seconds: list[float] = []
    scalar_checksums: list[float] = []
    for _ in range(blocks):
        checksum = 0.0
        started = time.perf_counter()
        for coalition in coalitions:
            checksum += float(evaluate_airport(coalition, AIRPORT_COSTS))
        scalar_seconds.append(time.perf_counter() - started)
        scalar_checksums.append(checksum)

    # This sensitivity measurement is retained but is not used on the x-axis.
    # The requested proxy is explicitly based on one scalar utility call.
    batch_seconds: list[float] = []
    batch_checksums: list[float] = []
    for _ in range(blocks):
        started = time.perf_counter()
        utilities = np.asarray(
            evaluate_airport(coalitions, AIRPORT_COSTS), dtype=np.float64
        )
        batch_seconds.append(time.perf_counter() - started)
        batch_checksums.append(float(utilities.sum()))

    if not np.allclose(
        scalar_checksums,
        batch_checksums,
        rtol=0.0,
        atol=1e-10,
    ):
        raise RuntimeError("scalar and vectorized Airport utilities disagree")
    per_call = np.asarray(scalar_seconds, dtype=np.float64) / calls_per_block
    batch_per_call = (
        np.asarray(batch_seconds, dtype=np.float64) / calls_per_block
    )
    coalition_sha256 = hashlib.sha256(
        np.packbits(coalitions, axis=1).tobytes()
    ).hexdigest()
    return {
        "protocol": "warmed repeated scalar calls on a fixed Q* coalition batch",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "seed": int(seed),
        "calls_per_block": int(calls_per_block),
        "blocks": int(blocks),
        "warmup_calls": int(len(warmup)),
        "warmup_checksum": float(warmup_checksum),
        "coalition_distribution": (
            "Shapley-specific OFA Q*: sample size from q_s proportional to "
            "1/sqrt(s(n-s)), then uniformly within size"
        ),
        "coalition_batch_sha256": coalition_sha256,
        "scalar_block_seconds": [float(value) for value in scalar_seconds],
        "scalar_seconds_per_call_by_block": [
            float(value) for value in per_call
        ],
        "median_scalar_seconds_per_call": float(np.median(per_call)),
        "mean_scalar_seconds_per_call": float(np.mean(per_call)),
        "minimum_scalar_seconds_per_call": float(np.min(per_call)),
        "maximum_scalar_seconds_per_call": float(np.max(per_call)),
        "vectorized_block_seconds": [float(value) for value in batch_seconds],
        "median_vectorized_seconds_per_call": float(
            np.median(batch_per_call)
        ),
        "timing_clock": "time.perf_counter",
        "used_for_estimated_time": "median_scalar_seconds_per_call",
    }


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


def _ablation_method_cells(
    report: Mapping[str, Any], *, budget_index: int, repeats: int
) -> list[Mapping[str, Any]]:
    selected = sorted(
        (
            cell
            for cell in report["raw_cells"]
            if int(cell["budget_index"]) == budget_index
        ),
        key=lambda cell: int(cell["repeat"]),
    )
    if len(selected) != repeats:
        raise ValueError("ablation has an incomplete repeat grid")
    return selected


def _verify_ablation_aggregate(
    summary: Mapping[str, Any],
    *,
    method: str,
    cells: Sequence[Mapping[str, Any]],
    truth: np.ndarray,
) -> None:
    estimates = np.asarray(
        [cell["methods"][method]["estimate"] for cell in cells],
        dtype=np.float64,
    )
    if estimates.shape != (len(cells), len(truth)) or not np.all(
        np.isfinite(estimates)
    ):
        raise ValueError(f"{method} raw estimates are invalid")
    expected = float(
        np.sqrt(np.mean(np.square(estimates - truth[None, :])))
    )
    stored = _finite(
        summary.get("aggregate_rmse"),
        name=f"{method} aggregate RMSE",
        positive=True,
    )
    if not math.isclose(stored, expected, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError(f"{method} aggregate RMSE disagrees with raw estimates")


def _method_source_summary(
    method: str,
    *,
    ablation_row: Mapping[str, Any],
    baseline_row: Mapping[str, Any],
) -> Mapping[str, Any]:
    if method in ABLATION_METHODS:
        return ablation_row["methods"][ABLATION_METHODS[method]]
    return baseline_row["methods"][BASELINE_METHODS[method]]


def build_efficiency_report(
    ablation: Mapping[str, Any],
    baselines: Mapping[str, Any],
    *,
    ablation_path: Path,
    baseline_path: Path,
    utility_calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge audited source results and apply the estimated-time formula."""
    validate_ablation_report(ablation)
    baseline_audit = validate_baseline_report(baselines)
    if (
        baseline_audit.get("status") != "ready_to_share"
        or any(
            value != "passed"
            for value in baseline_audit.get("checks", {}).values()
        )
    ):
        raise ValueError("Airport baseline report failed its independent audit")

    tau = _finite(
        utility_calibration.get("median_scalar_seconds_per_call"),
        name="median scalar utility time",
        positive=True,
    )
    ablation_rows = _ordered_ablation_rows(ablation)
    baseline_rows = _baseline_rows_by_inner(baselines)
    repeats = int(ablation["configuration"]["repeats"])
    if repeats != 3:
        raise ValueError("Airport efficiency report requires three repeats")
    if int(baselines["configuration"]["repeats"]) != repeats:
        raise ValueError("Airport source reports use different repeat counts")

    ablation_truth = np.asarray(
        ablation["ground_truth"]["values"], dtype=np.float64
    )
    baseline_truth = np.asarray(
        baselines["ground_truth"]["values"], dtype=np.float64
    )
    if not np.array_equal(ablation_truth, baseline_truth):
        raise ValueError("Airport source reports use different ground truth")

    result_rows: list[dict[str, Any]] = []
    for budget_index, ablation_row in enumerate(ablation_rows):
        inner = int(ablation_row["inner_utility_calls"])
        target = int(ablation_row["total_utility_calls"])
        baseline_row = baseline_rows.get(inner)
        if baseline_row is None:
            raise ValueError(f"baseline report lacks inner budget {inner}")
        baseline_target = int(baseline_row["total_utility_calls_per_estimate"])
        if baseline_target != target:
            raise ValueError("source reports disagree on total call budget")
        ablation_cells = _ablation_method_cells(
            ablation, budget_index=budget_index, repeats=repeats
        )
        greedy_design_records = []
        for cell in ablation_cells:
            raw = cell["methods"]["full_inside"]
            diagnostics = raw["design_diagnostics"]
            greedy_design_records.append(
                {
                    "repeat": int(cell["repeat"]),
                    "seconds": _finite(
                        raw["design_seconds"],
                        name="Greedy design seconds",
                        positive=True,
                    ),
                    "design_jobs": int(diagnostics["design_jobs"]),
                    "design_start_method": str(
                        diagnostics["design_start_method"]
                    ),
                    "frame_diagnostics_computed": bool(
                        diagnostics["frame_diagnostics_computed"]
                    ),
                    "coalition_design_sha256": str(
                        raw["coalition_design_sha256"]
                    ),
                }
            )
        greedy_design = [
            float(record["seconds"]) for record in greedy_design_records
        ]

        method_rows: dict[str, Any] = {}
        for method in METHOD_ORDER:
            summary = _method_source_summary(
                method,
                ablation_row=ablation_row,
                baseline_row=baseline_row,
            )
            if method in ABLATION_METHODS:
                source_method = ABLATION_METHODS[method]
                _verify_ablation_aggregate(
                    summary,
                    method=source_method,
                    cells=ablation_cells,
                    truth=ablation_truth,
                )
            aggregate_rmse = _finite(
                summary["aggregate_rmse"],
                name=f"{method} aggregate RMSE",
                positive=True,
            )
            interval = [
                _finite(value, name=f"{method} RMSE interval", positive=True)
                for value in summary["aggregate_rmse_bootstrap_95"]
            ]
            if len(interval) != 2 or interval[0] > interval[1]:
                raise ValueError(f"{method} RMSE interval is invalid")
            if method in ABLATION_METHODS:
                source_method = ABLATION_METHODS[method]
                actual_calls = [
                    _finite(
                        cell["methods"][source_method][
                            "actual_utility_calls"
                        ],
                        name=f"{method} actual utility calls",
                        positive=True,
                    )
                    for cell in ablation_cells
                ]
            else:
                actual_calls = _actual_calls_by_repeat(
                    summary, repeats=repeats, target=target
                )
            design_seconds = (
                greedy_design if method == "inside_greedy" else [0.0] * repeats
            )
            estimated_utility = [calls * tau for calls in actual_calls]
            estimated_total = [
                utility + design
                for utility, design in zip(
                    estimated_utility, design_seconds, strict=True
                )
            ]
            method_rows[method] = {
                "label": METHOD_LABELS[method],
                "aggregate_rmse": aggregate_rmse,
                "aggregate_rmse_bootstrap_95": interval,
                "actual_utility_calls_by_repeat": actual_calls,
                "mean_actual_utility_calls": float(np.mean(actual_calls)),
                "estimated_utility_seconds_by_repeat": estimated_utility,
                "mean_estimated_utility_seconds": float(
                    np.mean(estimated_utility)
                ),
                "charged_design_seconds_by_repeat": design_seconds,
                "mean_charged_design_seconds": float(np.mean(design_seconds)),
                "charged_design_provenance_by_repeat": (
                    greedy_design_records
                    if method == "inside_greedy"
                    else []
                ),
                "estimated_total_seconds_by_repeat": estimated_total,
                "mean_estimated_total_seconds": float(np.mean(estimated_total)),
                "estimated_total_seconds_range": [
                    float(np.min(estimated_total)),
                    float(np.max(estimated_total)),
                ],
                "source_method": (
                    ABLATION_METHODS.get(method)
                    or BASELINE_METHODS[method]
                ),
                "source_report": (
                    "ablation" if method in ABLATION_METHODS else "baselines"
                ),
            }
        result_rows.append(
            {
                "budget_index": budget_index,
                "inner_utility_calls": inner,
                "target_total_utility_calls": target,
                "methods": method_rows,
            }
        )

    report = {
        "status": "complete",
        "experiment": EXPERIMENT_ID,
        "configuration": {
            "dataset": "airport",
            "players": len(AIRPORT_COSTS),
            "repeats": repeats,
            "methods": list(METHOD_ORDER),
            "method_labels": METHOD_LABELS,
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
                "S-Diff O(n^2) state and reconstruction",
                "KernelSHAP sampling and linear solve",
                "TMC permutation and truncation bookkeeping",
            ],
            "baseline_favouring_caveat": (
                "Only INSIDE-Greedy is charged a nonutility overhead, so the "
                "proxy is conservative in favour of every baseline."
            ),
            "greedy_design_protocol": {
                "candidate_pool": 64,
                "mean_balance_lambda0": 1.0 / 16.0,
                "second_moment_scope": "per_size",
                "topology_policy": (
                    "retain each production run's measured topology and "
                    "record it per budget/repeat"
                ),
                "configured_design_jobs_at_report_completion": int(
                    ablation["configuration"]["design_jobs"]
                ),
                "observed_design_jobs": sorted(
                    {
                        int(
                            cell["methods"]["full_inside"][
                                "design_diagnostics"
                            ]["design_jobs"]
                        )
                        for cell in ablation["raw_cells"]
                    }
                ),
                "compute_frame_diagnostics": False,
                "included": (
                    "systematic size allocation, candidate generation, "
                    "per-size greedy selection, materialization, relabeling"
                ),
                "excluded": "separate geometry diagnostics and plotting",
            },
            "source_reports": {
                "ablation": {
                    "path": str(ablation_path.resolve()),
                    "sha256": _sha256(ablation_path),
                },
                "baselines": {
                    "path": str(baseline_path.resolve()),
                    "sha256": _sha256(baseline_path),
                },
            },
        },
        "utility_calibration": dict(utility_calibration),
        "ground_truth": {
            "algorithm": "exact threshold-game decomposition",
            "values": ablation_truth.tolist(),
        },
        "results": result_rows,
    }
    validate_efficiency_report(report)
    return report


def validate_efficiency_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "complete" or report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("efficiency report identity/status is invalid")
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("efficiency report lacks configuration")
    if tuple(configuration.get("methods", ())) != METHOD_ORDER:
        raise ValueError("efficiency method order is invalid")
    tau = _finite(
        report["utility_calibration"]["median_scalar_seconds_per_call"],
        name="utility calibration",
        positive=True,
    )
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("efficiency report must contain five budgets")
    previous_target = 0
    for row in rows:
        target = int(row["target_total_utility_calls"])
        if target <= previous_target:
            raise ValueError("efficiency call budgets must increase")
        previous_target = target
        methods = row.get("methods")
        if not isinstance(methods, Mapping) or set(methods) != set(METHOD_ORDER):
            raise ValueError("efficiency row has the wrong method set")
        for method in METHOD_ORDER:
            summary = methods[method]
            rmse = _finite(
                summary.get("aggregate_rmse"),
                name=f"{method} aggregate RMSE",
            )
            if rmse < 0.0:
                raise ValueError(f"{method} aggregate RMSE must be nonnegative")
            interval = summary.get("aggregate_rmse_bootstrap_95")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{method} RMSE interval is invalid")
            interval_values = [
                _finite(value, name=f"{method} RMSE interval")
                for value in interval
            ]
            if (
                any(value < 0.0 for value in interval_values)
                or interval_values[0] > interval_values[1]
            ):
                raise ValueError(f"{method} RMSE interval is invalid")
            calls = np.asarray(
                summary["actual_utility_calls_by_repeat"], dtype=np.float64
            )
            designs = np.asarray(
                summary["charged_design_seconds_by_repeat"], dtype=np.float64
            )
            totals = np.asarray(
                summary["estimated_total_seconds_by_repeat"], dtype=np.float64
            )
            if calls.shape != (3,) or designs.shape != (3,) or totals.shape != (3,):
                raise ValueError(f"{method} repeat timing shape is invalid")
            if (
                not np.all(np.isfinite(calls))
                or np.any(calls <= 0.0)
                or np.any(calls > target)
                or not np.all(calls == np.floor(calls))
            ):
                raise ValueError(f"{method} actual calls are invalid")
            if not np.all(np.isfinite(designs)) or np.any(designs < 0.0):
                raise ValueError(f"{method} design times are invalid")
            expected = calls * tau + designs
            if not np.allclose(totals, expected, rtol=0.0, atol=1e-12):
                raise ValueError(f"{method} estimated-time formula is inconsistent")
            if method == "inside_greedy":
                if np.any(designs <= 0.0):
                    raise ValueError("Greedy design time must be positive")
            elif np.any(designs != 0.0):
                raise ValueError(f"{method} must not be charged design time")
            for key, expected_mean in (
                ("mean_actual_utility_calls", float(np.mean(calls))),
                (
                    "mean_estimated_utility_seconds",
                    float(np.mean(calls * tau)),
                ),
                ("mean_charged_design_seconds", float(np.mean(designs))),
            ):
                observed_mean = _finite(
                    summary.get(key), name=f"{method} {key}"
                )
                if not math.isclose(
                    observed_mean,
                    expected_mean,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"{method} {key} is inconsistent")
            utility_seconds = np.asarray(
                summary.get("estimated_utility_seconds_by_repeat"),
                dtype=np.float64,
            )
            if utility_seconds.shape != (3,) or not np.allclose(
                utility_seconds, calls * tau, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"{method} utility times are inconsistent")
            mean_total = _finite(
                summary["mean_estimated_total_seconds"],
                name=f"{method} mean estimated time",
                positive=True,
            )
            if not math.isclose(
                mean_total, float(np.mean(expected)), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{method} mean estimated time is inconsistent")
            total_range = np.asarray(
                summary.get("estimated_total_seconds_range"),
                dtype=np.float64,
            )
            expected_range = np.asarray(
                [float(np.min(expected)), float(np.max(expected))]
            )
            if total_range.shape != (2,) or not np.allclose(
                total_range, expected_range, rtol=0.0, atol=1e-12
            ):
                raise ValueError(f"{method} estimated-time range is inconsistent")
            provenance = summary.get("charged_design_provenance_by_repeat")
            if not isinstance(provenance, list):
                raise ValueError(f"{method} design provenance must be a list")
            if method == "inside_greedy":
                if len(provenance) != 3:
                    raise ValueError("Greedy design provenance is incomplete")
                provenance_seconds = np.asarray(
                    [item["seconds"] for item in provenance], dtype=np.float64
                )
                if not np.array_equal(provenance_seconds, designs):
                    raise ValueError("Greedy design provenance is inconsistent")
                if any(
                    item.get("frame_diagnostics_computed") is not False
                    or int(item.get("design_jobs", 0)) < 1
                    or not item.get("design_start_method")
                    for item in provenance
                ):
                    raise ValueError("Greedy design provenance is invalid")
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
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--calls-per-block", type=int, default=100_000)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--warmup-calls", type=int, default=2_000)
    parser.add_argument("--benchmark-seed", type=int, default=20260901)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_path = args.ablation.resolve()
    baseline_path = args.baselines.resolve()
    if not ablation_path.exists() or not baseline_path.exists():
        raise ValueError("an Airport source report is missing")
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    baselines = json.loads(baseline_path.read_text(encoding="utf-8"))
    calibration = _benchmark_scalar_airport_utility(
        calls_per_block=args.calls_per_block,
        blocks=args.blocks,
        warmup_calls=args.warmup_calls,
        seed=args.benchmark_seed,
    )
    report = build_efficiency_report(
        ablation,
        baselines,
        ablation_path=ablation_path,
        baseline_path=baseline_path,
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

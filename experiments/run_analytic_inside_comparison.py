"""Compare the two fixed INSIDE algorithms with analytic-game baselines.

This runner deliberately keeps only the two algorithms used in the paper:

* ``INSIDE-Greedy`` is the fixed ``K=64, lambda0=1/16`` per-size row-wise
  frame-coupled design with the OFA conditional-mean (ratio) estimator;
* ``INSIDE-Orbit`` is the ``K=4`` size-stratified cyclic-orbit design with the
  OFA conditional-mean (ratio) estimator.

Airport and weighted-voting games have exact Shapley values, so this script
can benchmark both algorithms and all baselines without a Monte Carlo ground
truth.  Parallelism is across method/repeat tasks.  The parent process writes
an atomic checkpoint whenever one such task finishes, allowing an interrupted
run to resume without recomputing completed method/repeat pairs.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from experiments.airport_game import (
    AIRPORT_CLASS_COUNTS,
    AIRPORT_COSTS,
    game_fingerprint as airport_game_fingerprint,
)
from experiments.us_electoral_voting_game import (
    US_ELECTORAL_NAMES,
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
    game_fingerprint as voting_game_fingerprint,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_basic_cc,
    estimate_coupled,
    estimate_kernel_shap,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    estimate_sdiff,
    estimate_tmc_shapley,
    evaluate_airport,
    evaluate_weighted_voting,
    exact_airport_shapley,
    exact_shapley_shubik,
    iid_ofa_design,
    inner_size_distribution,
    per_size_frame_coupled_design,
)


METHOD_ORDER = (
    "inside_greedy",
    "inside_orbit",
    "ofa_iid_linear",
    "ofa_iid_ratio",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)

METHOD_LABELS = {
    "inside_greedy": "INSIDE-Greedy",
    "inside_orbit": "INSIDE-Orbit",
    "ofa_iid_linear": "OFA linear (IID)",
    "ofa_iid_ratio": "OFA ratio (IID)",
    "cc": "CC",
    "s_diff": "S-Diff",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}

LEGACY_GLOBAL_LINEAR_EXPERIMENT_ID = (
    "analytic_inside_baseline_comparison_normalized_mean_balance"
)
LEGACY_PER_SIZE_RATIO_EXPERIMENT_ID = (
    "analytic_inside_baseline_comparison_per_size_ratio"
)
EXPERIMENT_ID = (
    "analytic_inside_baseline_comparison_per_size_ratio_k64_lambda1over16"
)
PROTOCOL_VERSION = "per_size_ratio_k64_lambda1over16"
INSIDE_GREEDY_CANDIDATE_POOL = 64
INSIDE_ORBIT_CANDIDATE_POOL = 4
# Compatibility for the historical ablation runners that import this symbol.
# The canonical runner below always uses the two method-specific constants.
INSIDE_CANDIDATE_POOL = INSIDE_ORBIT_CANDIDATE_POOL
# Dimensionless coefficient.  ``per_size_frame_coupled_design`` converts this
# to the raw first/second-moment score scale using the realized size schedule.
# Keeping the mode explicit prevents these results from being confused with
# the legacy raw ``mean_balance=0.1`` runs.
INSIDE_MEAN_BALANCE_MODE = "normalized"
LEGACY_INSIDE_MEAN_BALANCE_LAMBDA0 = 1.0
INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0 = 1.0 / 16.0
# Compatibility for historical reconstruction/ablation runners importing the
# former shared constant.  The canonical runner uses the method-specific
# coefficient above.
INSIDE_MEAN_BALANCE_LAMBDA0 = LEGACY_INSIDE_MEAN_BALANCE_LAMBDA0
DEFAULT_BUDGET_MULTIPLIERS = (500, 1000, 2000, 5000, 10000)


def default_output_path(dataset: str) -> Path:
    _dataset_arrays(dataset)
    return Path(
        f"results/{dataset}_inside_baseline_comparison_"
        "per_size_ratio_k64_lambda1over16.json"
    )


def _dataset_arrays(dataset: str) -> tuple[np.ndarray, int | None]:
    if dataset == "airport":
        return AIRPORT_COSTS, None
    if dataset == "voting":
        return US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
    raise ValueError("dataset must be 'airport' or 'voting'")


def _num_players(dataset: str) -> int:
    values, _ = _dataset_arrays(dataset)
    return len(values)


def _evaluate(dataset: str, coalitions: np.ndarray) -> np.ndarray:
    if dataset == "airport":
        return np.asarray(
            evaluate_airport(coalitions, AIRPORT_COSTS), dtype=np.float64
        )
    if dataset == "voting":
        return np.asarray(
            evaluate_weighted_voting(
                coalitions, US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
            ),
            dtype=np.float64,
        )
    raise ValueError("dataset must be 'airport' or 'voting'")


def _exact_truth(dataset: str) -> tuple[np.ndarray, str]:
    if dataset == "airport":
        return (
            exact_airport_shapley(AIRPORT_COSTS),
            "exact threshold-game decomposition",
        )
    if dataset == "voting":
        return (
            exact_shapley_shubik(
                US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
            ),
            "exact cardinality-weight dynamic programming",
        )
    raise ValueError("dataset must be 'airport' or 'voting'")


class AnalyticGame:
    """Scalar-or-batch game interface used by copied baseline tasks."""

    def __init__(self, dataset: str) -> None:
        _dataset_arrays(dataset)
        self.dataset = dataset

    def evaluate(self, coalition: np.ndarray) -> float | np.ndarray:
        values = _evaluate(self.dataset, coalition)
        if np.asarray(coalition).ndim == 1:
            return float(values)
        return values


class SerialAnalyticEvaluator:
    """Vectorized evaluation with serial coarse tasks inside an outer worker."""

    def __init__(self, dataset: str) -> None:
        self.game = AnalyticGame(dataset)

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        return np.asarray(self.game.evaluate(coalitions), dtype=np.float64)

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]:
        if chunksize < 1:
            raise ValueError("chunksize must be positive")
        return [task_func(self.game, payload) for payload in payloads]


def _compact_json(value: Any) -> Any:
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


def _seed_for(
    base_seed: int,
    repeat: int,
    budget_index: int,
    method_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, repeat, budget_index, method_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _bootstrap_rmse_interval(
    squared_error_by_repeat: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    repeats = len(squared_error_by_repeat)
    indices = rng.integers(0, repeats, size=(samples, repeats))
    draws = np.sqrt(squared_error_by_repeat[indices].mean(axis=1))
    return tuple(
        float(value) for value in np.quantile(draws, [0.025, 0.975])
    )


def total_call_budgets(
    dataset: str,
    budget_multipliers: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    num_players = _num_players(dataset)
    if not budget_multipliers or any(value < 1 for value in budget_multipliers):
        raise ValueError("budget multipliers must be positive")
    if any(
        left >= right
        for left, right in zip(budget_multipliers, budget_multipliers[1:])
    ):
        raise ValueError("budget multipliers must be strictly increasing")
    inner = tuple(num_players * value for value in budget_multipliers)
    boundary_calls = 2 * num_players + 2
    return inner, tuple(value + boundary_calls for value in inner)


def _diagnostic_value(diagnostics: Any, name: str) -> Any:
    if isinstance(diagnostics, Mapping):
        return diagnostics[name]
    return getattr(diagnostics, name)


def _ratio_coverage_diagnostics(
    coalitions: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
    """Audit every player/size conditional mean used by OFA ratio.

    INSIDE-Greedy deliberately uses ``missing="raise"``.  Retaining these
    compact counts in every raw task makes that coverage contract independently
    auditable without storing the full coalition matrix.
    """
    coalitions = np.asarray(coalitions, dtype=bool)
    sizes = np.asarray(sizes, dtype=np.int64)
    if coalitions.ndim != 2 or sizes.shape != (len(coalitions),):
        raise ValueError("coalitions and sizes have incompatible shapes")
    num_players = coalitions.shape[1]
    expected_sizes, _, _ = inner_size_distribution(num_players)
    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    missing_inclusion = 0
    missing_exclusion = 0
    observed_sizes = 0
    exactly_balanced_sizes = 0
    for size in expected_sizes:
        rows = coalitions[sizes == size]
        if not len(rows):
            inclusion = np.zeros(num_players, dtype=np.int64)
        else:
            observed_sizes += 1
            inclusion = rows.sum(axis=0, dtype=np.int64)
        exclusion = len(rows) - inclusion
        minimum_inclusion = min(minimum_inclusion, int(inclusion.min()))
        minimum_exclusion = min(minimum_exclusion, int(exclusion.min()))
        missing_inclusion += int(np.count_nonzero(inclusion == 0))
        missing_exclusion += int(np.count_nonzero(exclusion == 0))
        exactly_balanced_sizes += int(
            len(rows) > 0 and np.all(inclusion == inclusion[0])
        )
    all_covered = missing_inclusion == 0 and missing_exclusion == 0
    return {
        "all_player_size_strata_covered": all_covered,
        "expected_inner_sizes": int(len(expected_sizes)),
        "observed_inner_sizes": observed_sizes,
        "missing_inner_sizes": int(len(expected_sizes) - observed_sizes),
        "minimum_inclusion_count": int(minimum_inclusion),
        "minimum_exclusion_count": int(minimum_exclusion),
        "missing_inclusion_strata": missing_inclusion,
        "missing_exclusion_strata": missing_exclusion,
        "exactly_1_balanced_sizes": exactly_balanced_sizes,
    }


def _run_ofa_method(
    dataset: str,
    method: str,
    *,
    inner_calls: int,
    total_calls: int,
    seed: int,
) -> tuple[np.ndarray, int, Any]:
    num_players = _num_players(dataset)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        _evaluate(dataset, boundary_rows), num_players
    )
    design_started = time.perf_counter()
    if method == "inside_greedy":
        design = per_size_frame_coupled_design(
            num_players,
            inner_calls,
            seed=seed,
            candidate_pool=INSIDE_GREEDY_CANDIDATE_POOL,
            mean_balance=INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0,
            mean_balance_mode=INSIDE_MEAN_BALANCE_MODE,
        )
    elif method == "inside_orbit":
        if inner_calls % num_players:
            raise ValueError("INSIDE-Orbit requires n-multiple inner calls")
        design = cyclic_orbit_frame_design(
            num_players,
            num_orbits=inner_calls // num_players,
            seed=seed,
            candidate_pool=INSIDE_ORBIT_CANDIDATE_POOL,
        )
    elif method in {"ofa_iid_linear", "ofa_iid_ratio"}:
        design = iid_ofa_design(
            num_players,
            inner_calls,
            seed=seed,
            compute_diagnostics=False,
        )
    else:
        raise ValueError(f"not an OFA method: {method}")
    design_seconds = time.perf_counter() - design_started

    utilities = _evaluate(dataset, design.coalitions)
    coverage: dict[str, Any] | None = None
    if method == "inside_greedy":
        coverage = _ratio_coverage_diagnostics(
            design.coalitions, design.sizes
        )
        if not coverage["all_player_size_strata_covered"]:
            raise ValueError(
                "INSIDE-Greedy lacks an in/out observation for at least one "
                "player/size stratum"
            )
        values = estimate_official_ratio_ofa(
            design, utilities, boundary, missing="raise"
        )
        estimator = "ofa_conditional_mean_ratio_missing_raise"
    elif method == "ofa_iid_linear":
        values = estimate_coupled(
            design, utilities, boundary, baseline="linear"
        )
        estimator = "coupled_linear"
    elif method == "inside_orbit":
        values = estimate_ratio_ofa(design, utilities, boundary)
        estimator = "ofa_conditional_mean_ratio_strict_balanced"
    else:
        values = estimate_official_ratio_ofa(
            design, utilities, boundary, missing="zero"
        )
        estimator = "ofa_conditional_mean_ratio_missing_zero"

    actual_calls = len(boundary_rows) + len(design.coalitions)
    if actual_calls != total_calls:
        raise RuntimeError("OFA physical call accounting mismatch")
    diagnostics = {
        "utility_evaluations": actual_calls,
        "boundary_utility_evaluations": len(boundary_rows),
        "inner_utility_evaluations": len(design.coalitions),
        "design_method": design.method,
        "second_moment_scope": (
            "per_size" if method == "inside_greedy" else None
        ),
        "estimator": estimator,
        "official_ratio_missing_policy": (
            "raise" if method == "inside_greedy" else None
        ),
        "candidate_pool": (
            INSIDE_GREEDY_CANDIDATE_POOL
            if method == "inside_greedy"
            else (
                INSIDE_ORBIT_CANDIDATE_POOL
                if method == "inside_orbit"
                else None
            )
        ),
        "mean_balance_mode": (
            INSIDE_MEAN_BALANCE_MODE if method == "inside_greedy" else None
        ),
        "mean_balance_lambda0": (
            INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0
            if method == "inside_greedy"
            else None
        ),
        "mean_balance_effective_raw": (
            float(design.diagnostics["mean_balance_effective_raw"])
            if method == "inside_greedy"
            else None
        ),
        "coverage": coverage,
        "design_seconds": design_seconds,
        "design_diagnostics": _compact_json(design.diagnostics),
    }
    return np.asarray(values, dtype=np.float64), actual_calls, diagnostics


def _run_external_method(
    dataset: str,
    method: str,
    *,
    total_calls: int,
    seed: int,
    num_tasks: int,
) -> tuple[np.ndarray, int, Any]:
    evaluator = SerialAnalyticEvaluator(dataset)
    num_players = _num_players(dataset)
    if method == "cc":
        result = estimate_basic_cc(
            evaluator,
            num_players,
            total_calls // 2,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "s_diff":
        result = estimate_sdiff(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "kernel_shap":
        result = estimate_kernel_shap(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
            ridge=1e-8,
        )
    elif method == "tmc_shapley":
        result = estimate_tmc_shapley(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    else:
        raise ValueError(f"unsupported baseline: {method}")
    actual_calls = int(
        _diagnostic_value(result.diagnostics, "utility_evaluations")
    )
    if actual_calls > total_calls:
        raise RuntimeError("baseline exceeded its physical-call cap")
    return (
        np.asarray(result.values, dtype=np.float64),
        actual_calls,
        _compact_json(result.diagnostics),
    )


def _run_method_across_budgets(
    dataset: str,
    repeat: int,
    method: str,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    base_seed: int,
    num_tasks: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    method_index = METHOD_ORDER.index(method)
    for budget_index, (inner_calls, total_calls) in enumerate(
        zip(inner_budgets, total_budgets, strict=True)
    ):
        seed = _seed_for(base_seed, repeat, budget_index, method_index)
        started = time.perf_counter()
        if method in {
            "inside_greedy",
            "inside_orbit",
            "ofa_iid_linear",
            "ofa_iid_ratio",
        }:
            values, actual_calls, diagnostics = _run_ofa_method(
                dataset,
                method,
                inner_calls=inner_calls,
                total_calls=total_calls,
                seed=seed,
            )
        else:
            values, actual_calls, diagnostics = _run_external_method(
                dataset,
                method,
                total_calls=total_calls,
                seed=seed,
                num_tasks=num_tasks,
            )
        elapsed = time.perf_counter() - started
        if values.shape != (_num_players(dataset),) or not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(f"{method} returned an invalid estimate")
        rows.append(
            {
                "repeat": repeat,
                "method": method,
                "seed": seed,
                "inner_utility_call_budget": inner_calls,
                "target_total_utility_calls": total_calls,
                "actual_utility_calls": actual_calls,
                "estimate": values.tolist(),
                "elapsed_seconds": elapsed,
                "diagnostics": diagnostics,
            }
        )
    return {"repeat": repeat, "method": method, "rows": rows}


def _aggregate(
    task_reports: list[dict[str, Any]],
    truth: np.ndarray,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    flat_rows = [row for report in task_reports for row in report["rows"]]
    grand_value = float(truth.sum())
    summaries: dict[str, Any] = {}
    for budget_index, (inner_calls, total_calls) in enumerate(
        zip(inner_budgets, total_budgets, strict=True)
    ):
        method_summaries: dict[str, Any] = {}
        for method_index, method in enumerate(METHOD_ORDER):
            selected = sorted(
                (
                    row
                    for row in flat_rows
                    if row["target_total_utility_calls"] == total_calls
                    and row["method"] == method
                ),
                key=lambda row: row["repeat"],
            )
            if len(selected) != repeats:
                raise RuntimeError("method-budget cell has missing repeats")
            estimates = np.asarray(
                [row["estimate"] for row in selected], dtype=np.float64
            )
            squared_error = np.mean(
                np.square(estimates - truth[None, :]), axis=1
            )
            interval = _bootstrap_rmse_interval(
                squared_error,
                bootstrap_samples,
                _seed_for(94127, 0, budget_index, method_index),
            )
            actual_calls = np.asarray(
                [row["actual_utility_calls"] for row in selected],
                dtype=np.float64,
            )
            method_summaries[method] = {
                "label": METHOD_LABELS[method],
                "estimates": estimates.tolist(),
                "aggregate_rmse": float(np.sqrt(squared_error.mean())),
                "aggregate_rmse_bootstrap_95": list(interval),
                "mean_repeat_rmse": float(np.mean(np.sqrt(squared_error))),
                "std_repeat_rmse": float(
                    np.std(np.sqrt(squared_error), ddof=1)
                ),
                "bias_l2": float(
                    np.linalg.norm(estimates.mean(axis=0) - truth)
                ),
                "mean_efficiency_residual": float(
                    np.mean(estimates.sum(axis=1) - grand_value)
                ),
                "max_absolute_efficiency_residual": float(
                    np.max(np.abs(estimates.sum(axis=1) - grand_value))
                ),
                "target_total_utility_calls": total_calls,
                "actual_utility_calls_by_repeat": (
                    actual_calls.astype(int).tolist()
                ),
                "mean_actual_utility_calls": float(actual_calls.mean()),
                "mean_elapsed_seconds": float(
                    np.mean([row["elapsed_seconds"] for row in selected])
                ),
                "diagnostics_by_repeat": [
                    row["diagnostics"] for row in selected
                ],
            }
        summaries[str(inner_calls)] = {
            "inner_utility_calls": inner_calls,
            "total_utility_calls_per_estimate": total_calls,
            "methods": method_summaries,
        }
    return summaries


def _game_metadata(dataset: str) -> dict[str, Any]:
    if dataset == "airport":
        return {
            "dataset": dataset,
            "model": "100-player airport cost game",
            "players": len(AIRPORT_COSTS),
            "cost_class_counts": list(AIRPORT_CLASS_COUNTS),
            "utility": "v(S) = max_{i in S} c_i; v(empty) = 0",
            "fingerprint_sha256": airport_game_fingerprint(),
        }
    return {
        "dataset": dataset,
        "model": "51-jurisdiction U.S. Electoral College bloc game",
        "players": len(US_ELECTORAL_WEIGHTS),
        "player_names": list(US_ELECTORAL_NAMES),
        "weights": US_ELECTORAL_WEIGHTS.tolist(),
        "quota": US_ELECTORAL_QUOTA,
        "utility": "v(S) = 1[sum_{i in S} w_i >= 270]",
        "fingerprint_sha256": voting_game_fingerprint(),
    }


def _configuration(
    dataset: str,
    budget_multipliers: tuple[int, ...],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    processes: int,
    base_seed: int,
    bootstrap_samples: int,
    num_tasks: int,
    reused_non_greedy_report: dict[str, str] | None,
) -> dict[str, Any]:
    num_players = _num_players(dataset)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": dataset,
        "budget_multipliers": list(budget_multipliers),
        "inner_utility_call_budgets": list(inner_budgets),
        "total_call_budgets": list(total_budgets),
        "boundary_utility_calls_for_ofa_methods": 2 * num_players + 2,
        "repeats": repeats,
        "processes": min(processes, repeats * len(METHOD_ORDER)),
        "base_seed": base_seed,
        "bootstrap_samples": bootstrap_samples,
        "num_tasks_per_serial_baseline": num_tasks,
        "methods": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "inside": {
            "greedy": {
                "sections": "4.1--4.3",
                "design": "per_size_frame_coupled_design",
                "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                "mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
                "mean_balance_lambda0": INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0,
                "first_moment_scope": "per_size",
                "second_moment_scope": "per_size",
                "estimator": (
                    "ofa_conditional_mean_ratio_missing_raise"
                ),
                "ratio_missing_policy": "raise",
                "random_relabeling": "uniform batch-wide player permutation",
            },
            "orbit": {
                "section": "4.4",
                "design": "cyclic_orbit_frame_design",
                "candidate_pool": INSIDE_ORBIT_CANDIDATE_POOL,
                "first_moment_scope": (
                    "per_size_exact_by_complete_orbits"
                ),
                "second_moment_scope": "per_size",
                "estimator": (
                    "ofa_conditional_mean_ratio_strict_balanced"
                ),
            },
        },
        "seed_semantics": (
            "SeedSequence([base_seed, repeat, budget_index, method_index]); "
            "independent of worker scheduling and process count"
        ),
        "parallelism": (
            "outer method-by-repeat; serial evaluator per worker; process "
            "count changes scheduling only"
        ),
        "checkpoint_granularity": "completed method-by-repeat task",
        "call_budget_semantics": "total physical utility-call cap",
        "tmc_x_coordinate": "mean observed physical calls after truncation",
        "reused_non_greedy_report": reused_non_greedy_report,
    }


def _write_checkpoint(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_compact_json)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _task_key(report: Mapping[str, Any]) -> tuple[int, str]:
    return int(report["repeat"]), str(report["method"])


def _validate_task_report_structure(
    report: Mapping[str, Any],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> tuple[int, str, list[Any]]:
    repeat, method = _task_key(report)
    if repeat < 0 or repeat >= repeats or method not in METHOD_ORDER:
        raise ValueError("checkpoint contains an unknown task")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(inner_budgets):
        raise ValueError("checkpoint contains an incomplete task")
    if any(
        int(row.get("repeat", -1)) != repeat
        or str(row.get("method", "")) != method
        for row in rows
    ):
        raise ValueError("checkpoint task rows disagree with their task key")
    observed = sorted(
        (
            int(row["inner_utility_call_budget"]),
            int(row["target_total_utility_calls"]),
        )
        for row in rows
    )
    if observed != sorted(zip(inner_budgets, total_budgets, strict=True)):
        raise ValueError("checkpoint task budgets disagree with configuration")
    return repeat, method, rows


def _validate_task_report(
    report: Mapping[str, Any],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> None:
    _, method, rows = _validate_task_report_structure(
        report, inner_budgets, total_budgets, repeats
    )
    if method == "inside_greedy":
        for row in rows:
            diagnostics = row.get("diagnostics")
            if not isinstance(diagnostics, Mapping):
                raise ValueError("INSIDE-Greedy checkpoint has no diagnostics")
            if diagnostics.get("mean_balance_mode") != INSIDE_MEAN_BALANCE_MODE:
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong balance mode"
                )
            if not math.isclose(
                float(diagnostics.get("mean_balance_lambda0", math.nan)),
                INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong lambda0"
                )
            num_players = len(row.get("estimate", ()))
            if num_players < 4:
                raise ValueError(
                    "INSIDE-Greedy checkpoint has an invalid estimate length"
                )
            expected_factor = (num_players - 2.0) / (num_players - 1.0)
            expected_effective = (
                INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0 * expected_factor
            )
            if not math.isclose(
                float(
                    diagnostics.get(
                        "mean_balance_effective_raw", math.nan
                    )
                ),
                expected_effective,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong effective lambda"
                )
            design_diagnostics = diagnostics.get("design_diagnostics")
            if not isinstance(design_diagnostics, Mapping):
                raise ValueError(
                    "INSIDE-Greedy checkpoint has no design diagnostics"
                )
            expected_normalization = {
                "mean_balance_mean_weight_squared": 1.0,
                "mean_balance_dimension_correction": expected_factor,
                "mean_balance_normalization_factor": expected_factor,
                "mean_balance_effective_raw": expected_effective,
            }
            for key, expected in expected_normalization.items():
                if not math.isclose(
                    float(design_diagnostics.get(key, math.nan)),
                    expected,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "INSIDE-Greedy checkpoint has an invalid "
                        f"normalization diagnostic: {key}"
                    )
            if int(diagnostics.get("candidate_pool", -1)) != (
                INSIDE_GREEDY_CANDIDATE_POOL
            ):
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong candidate pool"
                )
            if diagnostics.get("design_method") != "frame_coupled_per_size":
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong design"
                )
            if diagnostics.get("second_moment_scope") != "per_size":
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong moment scope"
                )
            if diagnostics.get("estimator") != (
                "ofa_conditional_mean_ratio_missing_raise"
            ):
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong estimator"
                )
            if diagnostics.get("official_ratio_missing_policy") != "raise":
                raise ValueError(
                    "INSIDE-Greedy checkpoint has the wrong missing policy"
                )
            coverage = diagnostics.get("coverage")
            if not isinstance(coverage, Mapping) or not coverage.get(
                "all_player_size_strata_covered"
            ):
                raise ValueError(
                    "INSIDE-Greedy checkpoint lacks complete ratio coverage"
                )
            if (
                int(coverage.get("minimum_inclusion_count", 0)) <= 0
                or int(coverage.get("minimum_exclusion_count", 0)) <= 0
                or int(coverage.get("missing_inclusion_strata", 1)) != 0
                or int(coverage.get("missing_exclusion_strata", 1)) != 0
                or int(coverage.get("missing_inner_sizes", 1)) != 0
            ):
                raise ValueError(
                    "INSIDE-Greedy checkpoint coverage counts are invalid"
                )


def _validate_legacy_global_linear_task_report(
    report: Mapping[str, Any],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> None:
    """Validate historical global-linear tasks for ablation reproduction."""
    _, method, rows = _validate_task_report_structure(
        report, inner_budgets, total_budgets, repeats
    )
    if method != "inside_greedy":
        return
    for row in rows:
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("legacy INSIDE-Greedy task has no diagnostics")
        if diagnostics.get("mean_balance_mode") != INSIDE_MEAN_BALANCE_MODE:
            raise ValueError("legacy INSIDE-Greedy has the wrong balance mode")
        if not math.isclose(
            float(diagnostics.get("mean_balance_lambda0", math.nan)),
            LEGACY_INSIDE_MEAN_BALANCE_LAMBDA0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("legacy INSIDE-Greedy has the wrong lambda0")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_reused_non_greedy_tasks(
    path: Path,
    *,
    dataset: str,
    budget_multipliers: tuple[int, ...],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    base_seed: int,
    num_tasks: int,
    truth: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load unchanged baselines from a compatible complete legacy run."""
    source_path = path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("reused non-Greedy report must be complete")
    configuration = source.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("reused report has no configuration")
    expected = {
        "dataset": dataset,
        "budget_multipliers": list(budget_multipliers),
        "inner_utility_call_budgets": list(inner_budgets),
        "total_call_budgets": list(total_budgets),
        "repeats": repeats,
        "base_seed": base_seed,
        "num_tasks_per_serial_baseline": num_tasks,
        "methods": list(METHOD_ORDER),
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(
                f"reused report configuration.{key} does not match this run"
            )
    source_truth = np.asarray(
        source.get("ground_truth", {}).get("values", []), dtype=np.float64
    )
    if source_truth.shape != truth.shape or not np.allclose(
        source_truth, truth, rtol=0.0, atol=1e-14
    ):
        raise ValueError("reused report ground truth does not match")
    raw_tasks = source.get("raw_tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("reused report has no raw task audit trail")
    selected: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for raw in raw_tasks:
        if str(raw.get("method")) == "inside_greedy":
            continue
        _validate_task_report(raw, inner_budgets, total_budgets, repeats)
        key = _task_key(raw)
        if key in seen:
            raise ValueError("reused report contains duplicate tasks")
        seen.add(key)
        selected.append(copy.deepcopy(dict(raw)))
    expected_keys = {
        (repeat, method)
        for repeat in range(repeats)
        for method in METHOD_ORDER
        if method != "inside_greedy"
    }
    if seen != expected_keys:
        raise ValueError("reused report lacks a complete non-Greedy task set")
    return selected, {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "reuse_scope": "all methods except INSIDE-Greedy",
    }
def _initial_report(
    dataset: str,
    configuration: dict[str, Any],
    truth: np.ndarray,
    truth_algorithm: str,
    truth_seconds: float,
) -> dict[str, Any]:
    grand_value = float(truth.sum())
    expected_grand = 10.0 if dataset == "airport" else 1.0
    return {
        "status": "running",
        "experiment": EXPERIMENT_ID,
        "game": _game_metadata(dataset),
        "configuration": configuration,
        "ground_truth": {
            "algorithm": truth_algorithm,
            "values": truth.tolist(),
            "sum": grand_value,
            "efficiency_error": float(abs(grand_value - expected_grand)),
            "computation_seconds": truth_seconds,
        },
        "results_by_inner_budget": {},
        "raw_tasks": [],
        "experiment_wall_seconds": 0.0,
    }


def _initialize_or_resume_report(
    *,
    output_path: Path | None,
    resume: bool,
    dataset: str,
    configuration: dict[str, Any],
    truth: np.ndarray,
    truth_algorithm: str,
    truth_seconds: float,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> dict[str, Any]:
    if output_path is None or not resume or not output_path.exists():
        return _initial_report(
            dataset, configuration, truth, truth_algorithm, truth_seconds
        )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("checkpoint belongs to a different experiment")
    saved_configuration = report.get("configuration")
    if not isinstance(saved_configuration, dict):
        raise ValueError("checkpoint configuration must be an object")
    # The outer worker count affects wall time only, not any random draw or
    # estimator.  Permit a resumed job to use a safer/different concurrency.
    saved_comparable = dict(saved_configuration)
    current_comparable = dict(configuration)
    saved_comparable.pop("processes", None)
    current_comparable.pop("processes", None)
    if saved_comparable != current_comparable:
        raise ValueError("checkpoint configuration does not match this run")
    report["configuration"]["processes"] = configuration["processes"]
    saved_truth = np.asarray(
        report.get("ground_truth", {}).get("values", []), dtype=np.float64
    )
    if saved_truth.shape != truth.shape or not np.allclose(
        saved_truth, truth, rtol=0.0, atol=1e-14
    ):
        raise ValueError("checkpoint ground truth does not match this run")
    raw_tasks = report.get("raw_tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("checkpoint raw_tasks must be a list")
    seen: set[tuple[int, str]] = set()
    for task_report in raw_tasks:
        _validate_task_report(
            task_report,
            inner_budgets,
            total_budgets,
            repeats,
        )
        key = _task_key(task_report)
        if key in seen:
            raise ValueError("checkpoint contains a duplicate task")
        seen.add(key)
    return report


def run_experiment(
    *,
    dataset: str,
    budget_multipliers: tuple[int, ...] = DEFAULT_BUDGET_MULTIPLIERS,
    repeats: int = 3,
    processes: int = 8,
    base_seed: int = 20260827,
    bootstrap_samples: int = 5000,
    num_tasks: int = 128,
    output_path: Path | None = None,
    resume: bool = True,
    reuse_non_greedy_from: Path | None = None,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("at least two repeats are required")
    if processes < 1 or bootstrap_samples < 1 or num_tasks < 1:
        raise ValueError("processes, bootstrap samples, and tasks must be positive")
    inner_budgets, total_budgets = total_call_budgets(
        dataset, budget_multipliers
    )
    num_players = _num_players(dataset)
    if min(budget_multipliers) < num_players - 3:
        raise ValueError(
            "the smallest multiplier cannot cover every INSIDE-Orbit size"
        )

    truth_started = time.perf_counter()
    truth, truth_algorithm = _exact_truth(dataset)
    truth_seconds = time.perf_counter() - truth_started
    if truth.shape != (num_players,) or not np.all(np.isfinite(truth)):
        raise RuntimeError("exact ground truth is invalid")

    reused_tasks: list[dict[str, Any]] = []
    reused_metadata: dict[str, str] | None = None
    if reuse_non_greedy_from is not None:
        reused_tasks, reused_metadata = _load_reused_non_greedy_tasks(
            reuse_non_greedy_from,
            dataset=dataset,
            budget_multipliers=budget_multipliers,
            inner_budgets=inner_budgets,
            total_budgets=total_budgets,
            repeats=repeats,
            base_seed=base_seed,
            num_tasks=num_tasks,
            truth=truth,
        )
    configuration = _configuration(
        dataset,
        budget_multipliers,
        inner_budgets,
        total_budgets,
        repeats,
        processes,
        base_seed,
        bootstrap_samples,
        num_tasks,
        reused_metadata,
    )
    report = _initialize_or_resume_report(
        output_path=output_path,
        resume=resume,
        dataset=dataset,
        configuration=configuration,
        truth=truth,
        truth_algorithm=truth_algorithm,
        truth_seconds=truth_seconds,
        inner_budgets=inner_budgets,
        total_budgets=total_budgets,
        repeats=repeats,
    )
    existing = {_task_key(item) for item in report["raw_tasks"]}
    for reused in reused_tasks:
        if _task_key(reused) not in existing:
            report["raw_tasks"].append(reused)
    report["raw_tasks"].sort(
        key=lambda row: (
            int(row["repeat"]), METHOD_ORDER.index(row["method"])
        )
    )
    # Write a fresh running header before the first potentially long task.
    # This also prevents ``--no-resume`` from leaving a stale old report on
    # disk while the replacement run is still computing its first task.
    if output_path is not None:
        _write_checkpoint(output_path, report)
    completed = {_task_key(item) for item in report["raw_tasks"]}
    tasks = [
        (
            dataset,
            repeat,
            method,
            inner_budgets,
            total_budgets,
            base_seed,
            num_tasks,
        )
        for repeat in range(repeats)
        for method in METHOD_ORDER
        if (repeat, method) not in completed
    ]

    wall_start = time.perf_counter()

    def record(task_report: dict[str, Any]) -> None:
        _validate_task_report(
            task_report, inner_budgets, total_budgets, repeats
        )
        key = _task_key(task_report)
        if key in completed:
            raise RuntimeError("worker returned an already completed task")
        report["raw_tasks"].append(task_report)
        report["raw_tasks"].sort(
            key=lambda row: (
                int(row["repeat"]), METHOD_ORDER.index(row["method"])
            )
        )
        completed.add(key)
        report["status"] = "running"
        report["experiment_wall_seconds"] = float(
            report.get("experiment_wall_seconds", 0.0)
            + time.perf_counter()
            - wall_start
        )
        if output_path is not None:
            _write_checkpoint(output_path, report)

    if processes == 1:
        for task in tasks:
            task_report = _run_method_across_budgets(*task)
            record(task_report)
            wall_start = time.perf_counter()
            print(
                f"completed repeat={task_report['repeat'] + 1}, "
                f"method={task_report['method']}",
                flush=True,
            )
    elif tasks:
        context = mp.get_context("spawn")
        first_failure: BaseException | None = None
        with ProcessPoolExecutor(
            max_workers=min(processes, len(tasks)), mp_context=context
        ) as executor:
            futures = [
                executor.submit(_run_method_across_budgets, *task)
                for task in tasks
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    task_report = future.result()
                except BaseException as error:
                    if first_failure is None:
                        first_failure = error
                    print(
                        f"task failure {index}/{len(tasks)}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                record(task_report)
                wall_start = time.perf_counter()
                print(
                    f"completed {index}/{len(tasks)} pending tasks: "
                    f"repeat={task_report['repeat'] + 1}, "
                    f"method={task_report['method']}",
                    flush=True,
                )
        if first_failure is not None:
            raise RuntimeError(
                "one or more analytic comparison tasks failed; completed "
                "tasks were checkpointed and the run can be resumed"
            ) from first_failure

    expected_tasks = repeats * len(METHOD_ORDER)
    if len(report["raw_tasks"]) != expected_tasks:
        raise RuntimeError("experiment ended with incomplete tasks")
    report["results_by_inner_budget"] = _aggregate(
        report["raw_tasks"],
        truth,
        inner_budgets,
        total_budgets,
        repeats,
        bootstrap_samples,
    )
    report["experiment_wall_seconds"] = float(
        report.get("experiment_wall_seconds", 0.0)
        + time.perf_counter()
        - wall_start
    )
    report["status"] = "complete"
    if output_path is not None:
        _write_checkpoint(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("airport", "voting"), required=True)
    parser.add_argument(
        "--budget-multipliers",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGET_MULTIPLIERS,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--processes", type=int, default=8)
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--num-tasks", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--reuse-non-greedy-from",
        type=Path,
        help=(
            "reuse the seven unchanged baseline/orbit task traces from a "
            "compatible complete legacy report and rerun only INSIDE-Greedy"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing checkpoint and start a fresh report",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output or default_output_path(args.dataset)
    report = run_experiment(
        dataset=args.dataset,
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        processes=args.processes,
        base_seed=args.base_seed,
        bootstrap_samples=args.bootstrap_samples,
        num_tasks=args.num_tasks,
        output_path=output,
        resume=not args.no_resume,
        reuse_non_greedy_from=args.reuse_non_greedy_from,
    )
    print(
        f"saved {output} ({report['status']}, "
        f"{len(report['raw_tasks'])} method-repeat tasks)",
        flush=True,
    )


if __name__ == "__main__":
    main()

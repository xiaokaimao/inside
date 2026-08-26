"""Ablate the second-moment scope used by INSIDE-Greedy.

The established analytic comparison calls :func:`frame_coupled_design`, whose
second-moment operator is one global, radially weighted frame.  This runner
recomputes that design and a clean per-size counterpart under identical OFA
size schedules, seeds, budgets, estimators, and first-moment settings.  The
seven unchanged methods are copied from a compatible complete analytic
report, so expensive work is restricted to the two Greedy variants.

The output deliberately keeps an audit trail: copied and recomputed raw
method-repeat-budget cells, the source report hash, size-schedule and relabel
hashes, and an explicit numerical comparison with the historical global
trace.  Exact equality with that trace is not expected because this ablation
uses a separate shared relabel substream to control player labels.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.run_analytic_inside_comparison import (
    DEFAULT_BUDGET_MULTIPLIERS,
    INSIDE_CANDIDATE_POOL,
    INSIDE_MEAN_BALANCE_LAMBDA0,
    INSIDE_MEAN_BALANCE_MODE,
    LEGACY_GLOBAL_LINEAR_EXPERIMENT_ID as SOURCE_EXPERIMENT_ID,
    METHOD_ORDER as SOURCE_METHOD_ORDER,
    _bootstrap_rmse_interval,
    _compact_json,
    _evaluate,
    _exact_truth,
    _game_metadata,
    _num_players,
    _seed_for,
    _sha256,
    _validate_legacy_global_linear_task_report as _validate_source_task_report,
    _write_checkpoint,
    total_call_budgets,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    paired_frame_scope_designs,
)


GREEDY_METHOD_ORDER = (
    "inside_greedy_global",
    "inside_greedy_per_size",
)
COPIED_METHOD_ORDER = tuple(
    method for method in SOURCE_METHOD_ORDER if method != "inside_greedy"
)
METHOD_ORDER = GREEDY_METHOD_ORDER + COPIED_METHOD_ORDER
METHOD_LABELS = {
    "inside_greedy_global": "INSIDE-Greedy (global)",
    "inside_greedy_per_size": "INSIDE-Greedy (per-size)",
    "inside_orbit": "INSIDE-Orbit",
    "ofa_iid_linear": "OFA linear (IID)",
    "ofa_iid_ratio": "OFA ratio (IID)",
    "cc": "CC",
    "s_diff": "S-Diff",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}

EXPERIMENT_ID = "analytic_inside_second_moment_scope_ablation"
GLOBAL_SOURCE_METHOD = "inside_greedy"
DEFAULT_SOURCE_REPORTS = {
    "airport": Path(
        "results/airport_inside_comparison_normalized_mean_balance_"
        "3repeats_50k_1m.json"
    ),
    "voting": Path(
        "results/voting_inside_comparison_normalized_mean_balance_"
        "3repeats_25k_510k.json"
    ),
}


def _cell_key(report: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(report["repeat"]),
        str(report["method"]),
        int(report["budget_index"]),
    )


def _size_schedule_sha256(sizes: np.ndarray) -> str:
    """Hash one size schedule using an explicit, platform-stable encoding."""
    schedule = np.ascontiguousarray(np.asarray(sizes, dtype="<i8"))
    return hashlib.sha256(schedule.tobytes(order="C")).hexdigest()


def _size_counts(sizes: np.ndarray, num_players: int) -> list[int]:
    return np.bincount(
        np.asarray(sizes, dtype=np.int64), minlength=num_players + 1
    )[2 : num_players - 1].astype(int).tolist()


def _shared_greedy_seed(
    base_seed: int, repeat: int, budget_index: int
) -> int:
    """Use the historical global-Greedy stream for *both* variants."""
    return _seed_for(base_seed, repeat, budget_index, 0)


def _run_greedy_pair_cell(
    dataset: str,
    repeat: int,
    budget_index: int,
    inner_calls: int,
    total_calls: int,
    base_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute one strictly paired global/per-size budget cell."""
    seed = _shared_greedy_seed(base_seed, repeat, budget_index)
    num_players = _num_players(dataset)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        _evaluate(dataset, boundary_rows), num_players
    )

    cell_started = time.perf_counter()
    design_started = time.perf_counter()
    global_design, per_size_design = paired_frame_scope_designs(
        num_players,
        inner_calls,
        seed=seed,
        candidate_pool=INSIDE_CANDIDATE_POOL,
        mean_balance=INSIDE_MEAN_BALANCE_LAMBDA0,
        mean_balance_mode=INSIDE_MEAN_BALANCE_MODE,
    )
    paired_design_seconds = time.perf_counter() - design_started
    if not np.array_equal(global_design.sizes, per_size_design.sizes):
        raise RuntimeError("paired designs produced different size schedules")
    global_schedule_hash = _size_schedule_sha256(global_design.sizes)
    per_size_schedule_hash = _size_schedule_sha256(per_size_design.sizes)
    if global_schedule_hash != per_size_schedule_hash:
        raise RuntimeError("paired size-schedule hashes disagree")
    global_relabel_hash = global_design.diagnostics[
        "relabel_permutation_sha256"
    ]
    per_size_relabel_hash = per_size_design.diagnostics[
        "relabel_permutation_sha256"
    ]
    if global_relabel_hash != per_size_relabel_hash:
        raise RuntimeError("paired final relabeling hashes disagree")

    results: list[dict[str, Any]] = []
    for method, design in zip(
        GREEDY_METHOD_ORDER,
        (global_design, per_size_design),
        strict=True,
    ):
        utilities = _evaluate(dataset, design.coalitions)
        values = np.asarray(
            estimate_coupled(
                design, utilities, boundary, baseline="linear"
            ),
            dtype=np.float64,
        )
        actual_calls = len(boundary_rows) + len(design.coalitions)
        if actual_calls != total_calls:
            raise RuntimeError("Greedy physical-call accounting mismatch")
        if values.shape != (num_players,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"{method} returned an invalid estimate")
        diagnostics = {
            "utility_evaluations": actual_calls,
            "boundary_utility_evaluations": len(boundary_rows),
            "inner_utility_evaluations": len(design.coalitions),
            "design_method": design.method,
            "second_moment_scope": design.diagnostics[
                "second_moment_scope"
            ],
            "estimator": "coupled_linear",
            "candidate_pool": INSIDE_CANDIDATE_POOL,
            "mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
            "mean_balance_lambda0": INSIDE_MEAN_BALANCE_LAMBDA0,
            "mean_balance_effective_raw": float(
                design.diagnostics["mean_balance_effective_raw"]
            ),
            "shared_seed_stream": "historical inside_greedy method_index=0",
            "paired_scope_cell": True,
            "size_schedule_sha256": global_schedule_hash,
            "relabel_permutation_sha256": global_relabel_hash,
            "size_counts_for_sizes_2_through_n_minus_2": _size_counts(
                design.sizes, num_players
            ),
            "paired_design_seconds": paired_design_seconds,
            "design_diagnostics": _compact_json(design.diagnostics),
        }
        results.append(
            {
                "repeat": repeat,
                "method": method,
                "budget_index": budget_index,
                "seed": seed,
                "inner_utility_call_budget": inner_calls,
                "target_total_utility_calls": total_calls,
                "actual_utility_calls": actual_calls,
                "estimate": values.tolist(),
                # This is deliberately pair-cell wall time: both designs are
                # constructed jointly to control the random relabeling.
                "elapsed_seconds": time.perf_counter() - cell_started,
                "diagnostics": diagnostics,
            }
        )
    return results[0], results[1]


def _validate_cell(
    row: Mapping[str, Any],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    base_seed: int,
) -> None:
    repeat, method, budget_index = _cell_key(row)
    if repeat < 0 or repeat >= repeats or method not in METHOD_ORDER:
        raise ValueError("checkpoint contains an unknown task")
    if budget_index < 0 or budget_index >= len(inner_budgets):
        raise ValueError("checkpoint contains an unknown budget index")
    observed = (
        int(row["inner_utility_call_budget"]),
        int(row["target_total_utility_calls"]),
    )
    expected_budget = (
        inner_budgets[budget_index],
        total_budgets[budget_index],
    )
    if observed != expected_budget:
        raise ValueError("checkpoint cell budget disagrees with configuration")
    if method in GREEDY_METHOD_ORDER:
        expected_seed = _shared_greedy_seed(
            base_seed, repeat, budget_index
        )
        if int(row.get("seed", -1)) != expected_seed:
            raise ValueError("Greedy checkpoint uses a non-shared seed")
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("Greedy checkpoint lacks diagnostics")
        expected_scope = (
            "global_weighted"
            if method == "inside_greedy_global"
            else "per_size"
        )
        if diagnostics.get("second_moment_scope") != expected_scope:
            raise ValueError("Greedy checkpoint has the wrong scope")
        if not diagnostics.get("size_schedule_sha256"):
            raise ValueError("Greedy checkpoint lacks a schedule hash")
        if not diagnostics.get("relabel_permutation_sha256"):
            raise ValueError("Greedy checkpoint lacks a relabel hash")


def _load_and_validate_source(
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
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Return copied tasks, the historical Greedy tasks, and provenance."""
    source_path = path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source report does not exist: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("source report must be complete")
    if source.get("experiment") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source report is not the legacy global-linear run")
    configuration = source.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("source report has no configuration")
    expected = {
        "dataset": dataset,
        "budget_multipliers": list(budget_multipliers),
        "inner_utility_call_budgets": list(inner_budgets),
        "total_call_budgets": list(total_budgets),
        "repeats": repeats,
        "base_seed": base_seed,
        "num_tasks_per_serial_baseline": num_tasks,
        "methods": list(SOURCE_METHOD_ORDER),
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(
                f"source report configuration.{key} is incompatible"
            )
    inside = configuration.get("inside")
    if not isinstance(inside, Mapping):
        raise ValueError("source report lacks INSIDE configuration")
    required_inside = {
        "candidate_pool": INSIDE_CANDIDATE_POOL,
        "greedy_mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
        "greedy_mean_balance_lambda0": INSIDE_MEAN_BALANCE_LAMBDA0,
        "greedy_estimator": "coupled linear",
    }
    for key, value in required_inside.items():
        if inside.get(key) != value:
            raise ValueError(f"source INSIDE setting {key} is incompatible")

    source_truth = np.asarray(
        source.get("ground_truth", {}).get("values", []), dtype=np.float64
    )
    if source_truth.shape != truth.shape or not np.allclose(
        source_truth, truth, rtol=0.0, atol=1e-14
    ):
        raise ValueError("source report ground truth does not match")
    raw_tasks = source.get("raw_tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("source report has no raw task audit trail")

    copied: list[dict[str, Any]] = []
    historical_global: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for raw in raw_tasks:
        _validate_source_task_report(
            raw, inner_budgets, total_budgets, repeats
        )
        key = (int(raw["repeat"]), str(raw["method"]))
        if key in seen:
            raise ValueError("source report contains duplicate tasks")
        seen.add(key)
        destination = (
            historical_global
            if key[1] == GLOBAL_SOURCE_METHOD
            else copied
        )
        for source_row in raw["rows"]:
            row = copy.deepcopy(dict(source_row))
            budget_index = inner_budgets.index(
                int(row["inner_utility_call_budget"])
            )
            row["budget_index"] = budget_index
            estimate = np.asarray(row.get("estimate", []), dtype=np.float64)
            if estimate.shape != truth.shape or not np.all(
                np.isfinite(estimate)
            ):
                raise ValueError("source report contains an invalid estimate")
            actual_calls = int(row.get("actual_utility_calls", -1))
            target_calls = int(row["target_total_utility_calls"])
            if actual_calls < 1 or actual_calls > target_calls:
                raise ValueError("source report contains invalid call counts")
            destination.append(row)
    expected_keys = {
        (repeat, method)
        for repeat in range(repeats)
        for method in SOURCE_METHOD_ORDER
    }
    if seen != expected_keys:
        raise ValueError("source report lacks a complete task set")

    # Reports created before the explicit scope diagnostic can still be
    # identified unambiguously by the historical global design method and
    # its radial normalization factor.
    for row in historical_global:
        diagnostics = row.get("diagnostics", {})
        design_diagnostics = diagnostics.get("design_diagnostics", {})
        if diagnostics.get("design_method") != "frame_coupled":
            raise ValueError("source Greedy is not frame_coupled")
        scope = design_diagnostics.get("second_moment_scope")
        if scope not in {None, "global_weighted"}:
            raise ValueError("source Greedy is not global-weighted")
        mean_weight_squared = float(
            design_diagnostics.get(
                "mean_balance_mean_weight_squared", math.nan
            )
        )
        if not math.isfinite(mean_weight_squared) or mean_weight_squared <= 1:
            raise ValueError("source Greedy lacks radial normalization")
        expected_seed = _shared_greedy_seed(
            base_seed, int(row["repeat"]), int(row["budget_index"])
        )
        if int(row.get("seed", -1)) != expected_seed:
            raise ValueError("source Greedy does not use method_index=0 seeds")

    provenance = {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "source_experiment": source.get("experiment"),
        "copied_methods": list(COPIED_METHOD_ORDER),
        "recomputed_methods": list(GREEDY_METHOD_ORDER),
        "copy_scope": "seven unchanged methods; raw tasks and diagnostics",
    }
    return copied, historical_global, provenance


def _aggregate(
    task_reports: list[dict[str, Any]],
    truth: np.ndarray,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    flat_rows = task_reports
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
                _seed_for(531991, 0, budget_index, method_index),
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


def _validation_checks(
    task_reports: list[dict[str, Any]],
    historical_global: list[dict[str, Any]],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> dict[str, Any]:
    current_rows = {
        (int(row["repeat"]), int(row["target_total_utility_calls"])): row
        for row in task_reports
        if row["method"] == "inside_greedy_global"
    }
    per_size_rows = {
        (int(row["repeat"]), int(row["target_total_utility_calls"])): row
        for row in task_reports
        if row["method"] == "inside_greedy_per_size"
    }
    historical_rows = {
        (int(row["repeat"]), int(row["target_total_utility_calls"])): row
        for row in historical_global
    }
    expected = {
        (repeat, total_calls)
        for repeat in range(repeats)
        for total_calls in total_budgets
    }
    if set(current_rows) != expected or set(per_size_rows) != expected:
        raise RuntimeError("scope comparison is missing a Greedy cell")
    if set(historical_rows) != expected:
        raise RuntimeError("source report is missing a historical Greedy cell")

    maximum_difference = 0.0
    exactly_equal = True
    schedules_match = True
    relabels_match = True
    for key in sorted(expected):
        current = np.asarray(current_rows[key]["estimate"], dtype=np.float64)
        historical = np.asarray(
            historical_rows[key]["estimate"], dtype=np.float64
        )
        maximum_difference = max(
            maximum_difference,
            float(np.max(np.abs(current - historical))),
        )
        exactly_equal = exactly_equal and bool(
            np.array_equal(current, historical)
        )
        global_hash = current_rows[key]["diagnostics"][
            "size_schedule_sha256"
        ]
        per_size_hash = per_size_rows[key]["diagnostics"][
            "size_schedule_sha256"
        ]
        schedules_match = schedules_match and global_hash == per_size_hash
        global_relabel_hash = current_rows[key]["diagnostics"][
            "relabel_permutation_sha256"
        ]
        per_size_relabel_hash = per_size_rows[key]["diagnostics"][
            "relabel_permutation_sha256"
        ]
        relabels_match = (
            relabels_match and global_relabel_hash == per_size_relabel_hash
        )

    if not schedules_match:
        raise RuntimeError("global and per-size size schedules do not match")
    if not relabels_match:
        raise RuntimeError("global and per-size final relabelings do not match")
    return {
        "passed": True,
        "expected_method_repeat_budget_cells": (
            repeats * len(METHOD_ORDER) * len(inner_budgets)
        ),
        "expected_budget_cells_per_method": len(inner_budgets),
        "global_vs_per_size_size_schedules_match": True,
        "global_vs_per_size_final_relabels_match": True,
        "historical_global_numerical_comparison": {
            "source_method": GLOBAL_SOURCE_METHOD,
            "recomputed_method": "inside_greedy_global",
            "exactly_equal": exactly_equal,
            "maximum_absolute_estimate_difference": maximum_difference,
            "exact_equality_expected": False,
            "distribution_equivalent": True,
            "reason": (
                "the ablation rerun uses an independent batch-wide relabel "
                "substream shared by both scopes; the historical source "
                "drew its relabel after its candidate stream"
            ),
        },
    }


def _configuration(
    *,
    dataset: str,
    budget_multipliers: tuple[int, ...],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    processes: int,
    base_seed: int,
    bootstrap_samples: int,
    num_tasks: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    num_players = _num_players(dataset)
    return {
        "dataset": dataset,
        "budget_multipliers": list(budget_multipliers),
        "inner_utility_call_budgets": list(inner_budgets),
        "total_call_budgets": list(total_budgets),
        "boundary_utility_calls_for_greedy_methods": 2 * num_players + 2,
        "repeats": repeats,
        "processes": min(
            processes,
            repeats * len(inner_budgets),
        ),
        "base_seed": base_seed,
        "bootstrap_samples": bootstrap_samples,
        "num_tasks_per_serial_baseline": num_tasks,
        "methods": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "inside_greedy": {
            "candidate_pool": INSIDE_CANDIDATE_POOL,
            "mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
            "mean_balance_lambda0": INSIDE_MEAN_BALANCE_LAMBDA0,
            "estimator": "coupled linear",
            "global_second_moment": "one radial-weighted operator",
            "per_size_second_moment": "one unweighted operator per size",
            "first_moment": "per-size for both variants",
            "shared_seed_rule": (
                "SeedSequence([base_seed, repeat, budget_index, 0])"
            ),
            "shared_relabel_rule": (
                "SeedSequence([shared_seed, 0x1A51DE]); independent of "
                "candidate draws and shared by both scopes"
            ),
        },
        "inside": {
            "candidate_pool": INSIDE_CANDIDATE_POOL,
            "greedy_mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
            "greedy_mean_balance_lambda0": INSIDE_MEAN_BALANCE_LAMBDA0,
            "greedy_estimator": "coupled linear",
            "orbit_estimator": "OFA conditional-mean ratio",
        },
        "parallelism": (
            "outer paired-repeat-by-budget; up to 15 useful joint tasks"
        ),
        "checkpoint_granularity": (
            "completed paired repeat-budget cell (two raw method cells)"
        ),
        "call_budget_semantics": "total physical utility-call cap",
        "tmc_x_coordinate": "mean observed physical calls after truncation",
        "source_report": dict(provenance),
    }


def _initial_report(
    dataset: str,
    configuration: dict[str, Any],
    truth: np.ndarray,
    truth_algorithm: str,
    truth_seconds: float,
    copied_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    expected_grand = 10.0 if dataset == "airport" else 1.0
    grand_value = float(truth.sum())
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
        "raw_tasks": copy.deepcopy(copied_tasks),
        "validation": {},
        "experiment_wall_seconds": 0.0,
    }


def _initialize_or_resume(
    *,
    output_path: Path | None,
    resume: bool,
    fresh_report: dict[str, Any],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    base_seed: int,
) -> dict[str, Any]:
    if output_path is None or not resume or not output_path.exists():
        return fresh_report
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    if saved.get("experiment") != EXPERIMENT_ID:
        raise ValueError("checkpoint belongs to a different experiment")
    saved_configuration = saved.get("configuration")
    if not isinstance(saved_configuration, dict):
        raise ValueError("checkpoint configuration must be an object")
    saved_comparable = copy.deepcopy(saved_configuration)
    fresh_comparable = copy.deepcopy(fresh_report["configuration"])
    saved_comparable.pop("processes", None)
    fresh_comparable.pop("processes", None)
    if saved_comparable != fresh_comparable:
        raise ValueError("checkpoint configuration does not match this run")
    saved["configuration"]["processes"] = fresh_report["configuration"][
        "processes"
    ]
    if not np.allclose(
        np.asarray(saved.get("ground_truth", {}).get("values", [])),
        np.asarray(fresh_report["ground_truth"]["values"]),
        rtol=0.0,
        atol=1e-14,
    ):
        raise ValueError("checkpoint ground truth does not match this run")
    raw_tasks = saved.get("raw_tasks")
    if not isinstance(raw_tasks, list):
        raise ValueError("checkpoint raw_tasks must be a list")
    seen: set[tuple[int, str, int]] = set()
    for row in raw_tasks:
        _validate_cell(
            row, inner_budgets, total_budgets, repeats, base_seed
        )
        key = _cell_key(row)
        if key in seen:
            raise ValueError("checkpoint contains a duplicate task")
        seen.add(key)
    expected_copied = {
        (repeat, method, budget_index)
        for repeat in range(repeats)
        for method in COPIED_METHOD_ORDER
        for budget_index in range(len(inner_budgets))
    }
    if not expected_copied.issubset(seen):
        raise ValueError("checkpoint lacks the copied source task set")
    return saved


def run_experiment(
    *,
    dataset: str,
    source_report: Path,
    budget_multipliers: tuple[int, ...] = DEFAULT_BUDGET_MULTIPLIERS,
    repeats: int = 3,
    processes: int = 15,
    base_seed: int = 20260827,
    bootstrap_samples: int = 5000,
    num_tasks: int = 128,
    output_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("at least two repeats are required")
    if processes < 1 or bootstrap_samples < 1 or num_tasks < 1:
        raise ValueError("processes, bootstrap samples, and tasks must be positive")
    inner_budgets, total_budgets = total_call_budgets(
        dataset, budget_multipliers
    )
    truth_started = time.perf_counter()
    truth, truth_algorithm = _exact_truth(dataset)
    truth_seconds = time.perf_counter() - truth_started
    if truth.shape != (_num_players(dataset),) or not np.all(
        np.isfinite(truth)
    ):
        raise RuntimeError("exact ground truth is invalid")

    copied_tasks, historical_global, provenance = _load_and_validate_source(
        source_report,
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
        dataset=dataset,
        budget_multipliers=budget_multipliers,
        inner_budgets=inner_budgets,
        total_budgets=total_budgets,
        repeats=repeats,
        processes=processes,
        base_seed=base_seed,
        bootstrap_samples=bootstrap_samples,
        num_tasks=num_tasks,
        provenance=provenance,
    )
    fresh_report = _initial_report(
        dataset,
        configuration,
        truth,
        truth_algorithm,
        truth_seconds,
        copied_tasks,
    )
    report = _initialize_or_resume(
        output_path=output_path,
        resume=resume,
        fresh_report=fresh_report,
        inner_budgets=inner_budgets,
        total_budgets=total_budgets,
        repeats=repeats,
        base_seed=base_seed,
    )
    report["raw_tasks"].sort(
        key=lambda row: (
            int(row["repeat"]),
            METHOD_ORDER.index(row["method"]),
            int(row["budget_index"]),
        )
    )
    if output_path is not None:
        if output_path.resolve() == source_report.resolve():
            raise ValueError("output path must differ from the source report")
        _write_checkpoint(output_path, report)

    completed = {_cell_key(item) for item in report["raw_tasks"]}
    tasks = []
    for repeat in range(repeats):
        for budget_index, (inner_calls, total_calls) in enumerate(
            zip(inner_budgets, total_budgets, strict=True)
        ):
            pair_keys = {
                (repeat, method, budget_index)
                for method in GREEDY_METHOD_ORDER
            }
            present = pair_keys & completed
            if present and present != pair_keys:
                raise ValueError(
                    "checkpoint contains only one half of a paired Greedy cell"
                )
            if not present:
                tasks.append(
                    (
                        dataset,
                        repeat,
                        budget_index,
                        inner_calls,
                        total_calls,
                        base_seed,
                    )
                )
    accumulated_wall = float(report.get("experiment_wall_seconds", 0.0))
    wall_started = time.perf_counter()

    def record(pair: tuple[dict[str, Any], dict[str, Any]]) -> None:
        nonlocal accumulated_wall, wall_started
        if len(pair) != 2:
            raise RuntimeError("paired worker did not return two method cells")
        keys: set[tuple[int, str, int]] = set()
        for cell in pair:
            _validate_cell(
                cell,
                inner_budgets,
                total_budgets,
                repeats,
                base_seed,
            )
            key = _cell_key(cell)
            if key in completed or key in keys:
                raise RuntimeError("worker returned a duplicate task")
            keys.add(key)
        repeat_budget = {
            (key[0], key[2]) for key in keys
        }
        if len(repeat_budget) != 1 or {key[1] for key in keys} != set(
            GREEDY_METHOD_ORDER
        ):
            raise RuntimeError("worker returned a mismatched Greedy pair")
        first, second = pair
        if (
            first["diagnostics"]["size_schedule_sha256"]
            != second["diagnostics"]["size_schedule_sha256"]
        ):
            raise RuntimeError("worker pair has different size schedules")
        if (
            first["diagnostics"]["relabel_permutation_sha256"]
            != second["diagnostics"]["relabel_permutation_sha256"]
        ):
            raise RuntimeError("worker pair has different final relabelings")
        report["raw_tasks"].extend(pair)
        report["raw_tasks"].sort(
            key=lambda row: (
                int(row["repeat"]),
                METHOD_ORDER.index(row["method"]),
                int(row["budget_index"]),
            )
        )
        completed.update(keys)
        accumulated_wall += time.perf_counter() - wall_started
        report["experiment_wall_seconds"] = accumulated_wall
        report["status"] = "running"
        if output_path is not None:
            _write_checkpoint(output_path, report)
        wall_started = time.perf_counter()

    if processes == 1:
        for task in tasks:
            pair = _run_greedy_pair_cell(*task)
            record(pair)
            cell = pair[0]
            print(
                f"completed repeat={cell['repeat'] + 1}, "
                f"paired-scopes, budget={cell['budget_index'] + 1}",
                flush=True,
            )
    elif tasks:
        context = mp.get_context("spawn")
        first_failure: BaseException | None = None
        with ProcessPoolExecutor(
            max_workers=min(processes, len(tasks)), mp_context=context
        ) as executor:
            futures = [
                executor.submit(_run_greedy_pair_cell, *task)
                for task in tasks
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    pair = future.result()
                except BaseException as error:
                    if first_failure is None:
                        first_failure = error
                    print(
                        f"task failure {index}/{len(tasks)}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                record(pair)
                cell = pair[0]
                print(
                    f"completed {index}/{len(tasks)} pending tasks: "
                    f"repeat={cell['repeat'] + 1}, "
                    "paired-scopes, "
                    f"budget={cell['budget_index'] + 1}",
                    flush=True,
                )
        if first_failure is not None:
            raise RuntimeError(
                "one or more scope-ablation tasks failed; completed tasks "
                "were checkpointed and the run can be resumed"
            ) from first_failure

    expected_tasks = repeats * len(METHOD_ORDER) * len(inner_budgets)
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
    report["validation"] = _validation_checks(
        report["raw_tasks"],
        historical_global,
        inner_budgets,
        total_budgets,
        repeats,
    )
    report["experiment_wall_seconds"] = (
        accumulated_wall + time.perf_counter() - wall_started
    )
    report["status"] = "complete"
    if output_path is not None:
        _write_checkpoint(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("airport", "voting"), required=True)
    parser.add_argument(
        "--source-report",
        type=Path,
        help="compatible complete normalized 8-method analytic report",
    )
    parser.add_argument(
        "--budget-multipliers",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGET_MULTIPLIERS,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--processes", type=int, default=15)
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--num-tasks", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore an existing checkpoint and start a fresh report",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source_report or DEFAULT_SOURCE_REPORTS[args.dataset]
    output = args.output or Path(
        f"results/{args.dataset}_inside_global_vs_per_size.json"
    )
    report = run_experiment(
        dataset=args.dataset,
        source_report=source,
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        processes=args.processes,
        base_seed=args.base_seed,
        bootstrap_samples=args.bootstrap_samples,
        num_tasks=args.num_tasks,
        output_path=output,
        resume=not args.no_resume,
    )
    print(
        f"saved {output} ({report['status']}, "
        f"{len(report['raw_tasks'])} method-repeat-budget cells)",
        flush=True,
    )


if __name__ == "__main__":
    main()

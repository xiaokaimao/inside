"""Sweep INSIDE-Greedy's normalized lambda0 and candidate-pool size K.

This is an ablation runner, deliberately separate from the paper-facing
runner so the registered defaults remain ``lambda0=1`` and ``K=4``.  Airport
and U.S. Electoral College voting use their existing exact games, exact
Shapley truths, five budget multipliers, three repeats, and registered seed
schedule.

Every Greedy configuration uses per-size first/second-moment design and the
covered official OFA ratio estimator.  A single fixed INSIDE-Orbit reference
uses complete cyclic orbits and strict balanced OFA ratio.  Physical utility
calls are identical within every dataset/repeat/budget comparison.

Two-stage use is supported:

1. ``--stage screening`` defaults to budget indices 0 and 2 and scans a grid.
2. ``--stage validation --configs lambda:K ...`` evaluates selected settings
   (plus the registered default reference) on all five budgets.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.run_analytic_inside_comparison import (
    DEFAULT_BUDGET_MULTIPLIERS,
    _compact_json,
    _evaluate,
    _exact_truth,
    _game_metadata,
    _num_players,
    _ratio_coverage_diagnostics,
    _seed_for,
    total_call_budgets,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    per_size_frame_coupled_design,
)


EXPERIMENT_ID = "inside_greedy_per_size_ratio_hyperparameter_sweep"
DEFAULT_LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
DEFAULT_CANDIDATE_POOLS = (1, 2, 4, 8)
REGISTERED_LAMBDA0 = 1.0
REGISTERED_CANDIDATE_POOL = 4
REGISTERED_REPEATS = 3
REGISTERED_BASE_SEED = 20260827
ORBIT_CANDIDATE_POOL = 4
SCREENING_BUDGET_INDICES = (0, 2)
PAIR_TOLERANCE = 1e-15


Configuration = tuple[float, int]


def _configuration_id(lambda0: float, candidate_pool: int) -> str:
    return f"lambda0={lambda0:.12g}|K={candidate_pool}"


def _normalize_configurations(
    configurations: Sequence[Configuration],
) -> tuple[Configuration, ...]:
    normalized: list[Configuration] = []
    seen: set[Configuration] = set()
    for raw_lambda, raw_pool in configurations:
        lambda0 = float(raw_lambda)
        candidate_pool = int(raw_pool)
        if not math.isfinite(lambda0) or lambda0 < 0.0:
            raise ValueError("lambda0 values must be finite and nonnegative")
        if candidate_pool < 1:
            raise ValueError("candidate-pool sizes must be positive")
        item = (lambda0, candidate_pool)
        if item not in seen:
            normalized.append(item)
            seen.add(item)
    if not normalized:
        raise ValueError("at least one hyperparameter configuration is required")
    default = (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)
    if default not in seen:
        normalized.append(default)
    return tuple(normalized)


def configuration_grid(
    lambdas: Sequence[float], candidate_pools: Sequence[int]
) -> tuple[Configuration, ...]:
    # With K=1 there is no candidate choice, so lambda0 cannot affect the
    # selected rows.  Keep one registered-lambda cell instead of redundantly
    # evaluating the same design once per lambda.
    items: list[Configuration] = []
    for raw_pool in candidate_pools:
        pool = int(raw_pool)
        if pool == 1:
            items.append((REGISTERED_LAMBDA0, 1))
        else:
            items.extend((float(lambda0), pool) for lambda0 in lambdas)
    return _normalize_configurations(items)


def selected_budget_indices(
    stage: str,
    budget_count: int,
    explicit: Sequence[int] | None = None,
) -> tuple[int, ...]:
    if stage not in {"screening", "validation"}:
        raise ValueError("stage must be 'screening' or 'validation'")
    if budget_count < 1:
        raise ValueError("budget_count must be positive")
    if explicit is None:
        chosen = (
            (
                tuple(index for index in SCREENING_BUDGET_INDICES if index < budget_count)
                if budget_count == len(DEFAULT_BUDGET_MULTIPLIERS)
                else tuple(range(budget_count))
            )
            if stage == "screening"
            else tuple(range(budget_count))
        )
    else:
        chosen = tuple(int(index) for index in explicit)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("budget indices must be nonempty and unique")
    if any(index < 0 or index >= budget_count for index in chosen):
        raise ValueError("budget index is outside the configured budget grid")
    return tuple(sorted(chosen))


def _shared_relabel_seed(seed: int) -> int:
    """Match the independently registered relabel substream used in the ratio run."""
    return int(
        np.random.SeedSequence([seed, 0x1A51DE]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _size_schedule_sha256(sizes: np.ndarray) -> str:
    encoded = np.ascontiguousarray(np.asarray(sizes, dtype="<i8"))
    return hashlib.sha256(encoded.tobytes(order="C")).hexdigest()


def _cell_key(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    if cell["method"] == "inside_orbit":
        return (
            "inside_orbit",
            int(cell["repeat"]),
            int(cell["budget_index"]),
        )
    return (
        "inside_greedy",
        float(cell["lambda0"]),
        int(cell["candidate_pool"]),
        int(cell["repeat"]),
        int(cell["budget_index"]),
    )


def _coverage_failure_cell(
    payload: Mapping[str, Any],
    *,
    seed: int,
    design_seconds: float,
    diagnostics: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    return {
        **dict(payload),
        "status": "coverage_failure",
        "seed": seed,
        "estimate": None,
        "rmse": None,
        "planned_total_utility_calls": int(payload["total_calls"]),
        "actual_utility_calls": 0,
        "coverage_failure": True,
        "failure_reason": reason,
        "design_seconds": design_seconds,
        "utility_seconds": 0.0,
        "diagnostics": _compact_json(diagnostics),
    }


def _run_cell(payload: Mapping[str, Any]) -> dict[str, Any]:
    dataset = str(payload["dataset"])
    method = str(payload["method"])
    repeat = int(payload["repeat"])
    budget_index = int(payload["budget_index"])
    inner_calls = int(payload["inner_calls"])
    total_calls = int(payload["total_calls"])
    base_seed = int(payload["base_seed"])
    truth = np.asarray(payload["truth"], dtype=np.float64)
    num_players = _num_players(dataset)

    if method == "inside_greedy":
        lambda0 = float(payload["lambda0"])
        candidate_pool = int(payload["candidate_pool"])
        seed = _seed_for(base_seed, repeat, budget_index, 0)
        relabel_seed = _shared_relabel_seed(seed)
        started = time.perf_counter()
        design = per_size_frame_coupled_design(
            num_players,
            inner_calls,
            seed=seed,
            candidate_pool=candidate_pool,
            mean_balance=lambda0,
            mean_balance_mode="normalized",
            relabel_seed=relabel_seed,
        )
        design_seconds = time.perf_counter() - started
        coverage = _ratio_coverage_diagnostics(design.coalitions, design.sizes)
        diagnostics = {
            "design_method": design.method,
            "estimator": "ofa_conditional_mean_ratio_missing_raise",
            "second_moment_scope": "per_size",
            "mean_balance_mode": "normalized",
            "mean_balance_lambda0": lambda0,
            "candidate_pool": candidate_pool,
            "relabel_seed": relabel_seed,
            "size_schedule_sha256": _size_schedule_sha256(design.sizes),
            "relabel_permutation_sha256": design.diagnostics[
                "relabel_permutation_sha256"
            ],
            "coverage": coverage,
            "design_diagnostics": _compact_json(design.diagnostics),
        }
        if not coverage["all_player_size_strata_covered"]:
            return _coverage_failure_cell(
                payload,
                seed=seed,
                design_seconds=design_seconds,
                diagnostics=diagnostics,
                reason="missing player/size inclusion or exclusion stratum",
            )
    elif method == "inside_orbit":
        if inner_calls % num_players:
            raise ValueError("Orbit reference requires n-multiple inner calls")
        seed = _seed_for(base_seed, repeat, budget_index, 1)
        started = time.perf_counter()
        design = cyclic_orbit_frame_design(
            num_players,
            num_orbits=inner_calls // num_players,
            seed=seed,
            candidate_pool=ORBIT_CANDIDATE_POOL,
        )
        design_seconds = time.perf_counter() - started
        coverage = _ratio_coverage_diagnostics(design.coalitions, design.sizes)
        diagnostics = {
            "design_method": design.method,
            "estimator": "ofa_conditional_mean_ratio_strict_balanced",
            "candidate_pool": ORBIT_CANDIDATE_POOL,
            "coverage": coverage,
            "design_diagnostics": _compact_json(design.diagnostics),
        }
        if not coverage["all_player_size_strata_covered"]:
            return _coverage_failure_cell(
                payload,
                seed=seed,
                design_seconds=design_seconds,
                diagnostics=diagnostics,
                reason="Orbit unexpectedly lacks ratio coverage",
            )
    else:
        raise ValueError(f"unknown method: {method}")

    boundary_rows = boundary_coalitions(num_players)
    utility_started = time.perf_counter()
    boundary = boundary_from_utilities(
        _evaluate(dataset, boundary_rows), num_players
    )
    utilities = _evaluate(dataset, design.coalitions)
    if method == "inside_greedy":
        estimate = estimate_official_ratio_ofa(
            design, utilities, boundary, missing="raise"
        )
    else:
        estimate = estimate_ratio_ofa(design, utilities, boundary)
    utility_seconds = time.perf_counter() - utility_started
    estimate = np.asarray(estimate, dtype=np.float64)
    actual_calls = len(boundary_rows) + len(design.coalitions)
    if actual_calls != total_calls:
        raise RuntimeError("physical utility-call accounting mismatch")
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        raise RuntimeError("estimator returned an invalid Shapley vector")
    rmse = float(np.sqrt(np.mean(np.square(estimate - truth))))
    diagnostics.update(
        {
            "boundary_utility_evaluations": len(boundary_rows),
            "inner_utility_evaluations": len(design.coalitions),
            "utility_evaluations": actual_calls,
        }
    )
    return {
        **dict(payload),
        "status": "ok",
        "seed": seed,
        "estimate": estimate.tolist(),
        "rmse": rmse,
        "planned_total_utility_calls": total_calls,
        "actual_utility_calls": actual_calls,
        "coverage_failure": False,
        "failure_reason": None,
        "design_seconds": design_seconds,
        "utility_seconds": utility_seconds,
        "diagnostics": _compact_json(diagnostics),
    }


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return float(numerator / denominator)


def _paired_metrics(
    candidate: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    reference_by_cell = {
        (int(cell["repeat"]), int(cell["budget_index"])): cell
        for cell in reference
    }
    differences: list[float] = []
    ratios: list[float] = []
    wins = 0
    losses = 0
    ties = 0
    unavailable = 0
    for cell in candidate:
        key = (int(cell["repeat"]), int(cell["budget_index"]))
        if key not in reference_by_cell:
            raise RuntimeError("paired reference lacks a repeat-budget cell")
        other = reference_by_cell[key]
        if cell["status"] != "ok" or other["status"] != "ok":
            unavailable += 1
            continue
        difference = float(cell["rmse"] - other["rmse"])
        differences.append(difference)
        ratio = _safe_ratio(float(cell["rmse"]), float(other["rmse"]))
        if ratio is not None:
            ratios.append(ratio)
        if difference < -PAIR_TOLERANCE:
            wins += 1
        elif difference > PAIR_TOLERANCE:
            losses += 1
        else:
            ties += 1
    compared = wins + losses + ties
    return {
        "difference_definition": "candidate RMSE minus reference RMSE; negative favors candidate",
        "rmse_difference_by_repeat": differences,
        "rmse_ratio_by_repeat": ratios,
        "mean_rmse_difference": float(np.mean(differences)) if differences else None,
        "mean_rmse_ratio": float(np.mean(ratios)) if ratios else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "unavailable_pairs": unavailable,
        "win_rate_all_registered_repeats": float(wins / len(candidate)),
        "win_rate_available_pairs": float(wins / compared) if compared else None,
        "tie_tolerance": PAIR_TOLERANCE,
    }


def _aggregate_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    successful = [cell for cell in cells if cell["status"] == "ok"]
    rmses = np.asarray([cell["rmse"] for cell in successful], dtype=np.float64)
    complete = len(successful) == len(cells)
    return {
        "registered_repeats": len(cells),
        "successful_repeats": len(successful),
        "coverage_failures": len(cells) - len(successful),
        "coverage_failure_rate": float((len(cells) - len(successful)) / len(cells)),
        "aggregate_rmse": (
            float(np.sqrt(np.mean(np.square(rmses)))) if complete else None
        ),
        "mean_repeat_rmse": float(rmses.mean()) if len(rmses) else None,
        "rmse_by_repeat": [cell["rmse"] for cell in cells],
        "mean_design_seconds": float(
            np.mean([float(cell["design_seconds"]) for cell in cells])
        ),
        "design_seconds_by_repeat": [
            float(cell["design_seconds"]) for cell in cells
        ],
        "actual_utility_calls_by_repeat": [
            int(cell["actual_utility_calls"]) for cell in cells
        ],
    }


def aggregate_report(
    raw_cells: Sequence[Mapping[str, Any]],
    configurations: Sequence[Configuration],
    budget_indices: Sequence[int],
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(
        (copy.deepcopy(dict(cell)) for cell in raw_cells),
        key=lambda cell: (
            int(cell["budget_index"]),
            int(cell["repeat"]),
            0 if cell["method"] == "inside_orbit" else 1,
            (
                -1.0
                if cell.get("lambda0") is None
                else float(cell["lambda0"])
            ),
            int(cell.get("candidate_pool", -1)),
        ),
    )
    orbit_by_budget = {
        budget_index: [
            cell
            for cell in ordered
            if cell["method"] == "inside_orbit"
            and int(cell["budget_index"]) == budget_index
        ]
        for budget_index in budget_indices
    }
    default_by_budget = {
        budget_index: [
            cell
            for cell in ordered
            if cell["method"] == "inside_greedy"
            and float(cell["lambda0"]) == REGISTERED_LAMBDA0
            and int(cell["candidate_pool"]) == REGISTERED_CANDIDATE_POOL
            and int(cell["budget_index"]) == budget_index
        ]
        for budget_index in budget_indices
    }
    for budget_index in budget_indices:
        if len(orbit_by_budget[budget_index]) != repeats:
            raise RuntimeError("Orbit reference is incomplete")
        if len(default_by_budget[budget_index]) != repeats:
            raise RuntimeError("registered default reference is incomplete")

    results: dict[str, Any] = {}
    ranking: list[dict[str, Any]] = []
    for lambda0, candidate_pool in configurations:
        configuration_id = _configuration_id(lambda0, candidate_pool)
        budget_results: dict[str, Any] = {}
        all_candidate: list[Mapping[str, Any]] = []
        all_default: list[Mapping[str, Any]] = []
        all_orbit: list[Mapping[str, Any]] = []
        for budget_index in budget_indices:
            cells = [
                cell
                for cell in ordered
                if cell["method"] == "inside_greedy"
                and float(cell["lambda0"]) == lambda0
                and int(cell["candidate_pool"]) == candidate_pool
                and int(cell["budget_index"]) == budget_index
            ]
            if len(cells) != repeats:
                raise RuntimeError("Greedy configuration is incomplete")
            default_cells = default_by_budget[budget_index]
            orbit_cells = orbit_by_budget[budget_index]
            summary = _aggregate_cells(cells)
            default_summary = _aggregate_cells(default_cells)
            orbit_summary = _aggregate_cells(orbit_cells)
            summary.update(
                {
                    "aggregate_rmse_ratio_to_registered_default": _safe_ratio(
                        summary["aggregate_rmse"], default_summary["aggregate_rmse"]
                    ),
                    "aggregate_rmse_ratio_to_inside_orbit": _safe_ratio(
                        summary["aggregate_rmse"], orbit_summary["aggregate_rmse"]
                    ),
                    "paired_vs_registered_default": _paired_metrics(
                        cells, default_cells
                    ),
                    "paired_vs_inside_orbit": _paired_metrics(cells, orbit_cells),
                }
            )
            budget_results[str(budget_index)] = summary
            all_candidate.extend(cells)
            all_default.extend(default_cells)
            all_orbit.extend(orbit_cells)
        overall_default = _paired_metrics(all_candidate, all_default)
        overall_orbit = _paired_metrics(all_candidate, all_orbit)
        coverage_failures = sum(cell["status"] != "ok" for cell in all_candidate)
        valid_default_ratios = [
            budget_results[str(index)][
                "aggregate_rmse_ratio_to_registered_default"
            ]
            for index in budget_indices
        ]
        valid_orbit_ratios = [
            budget_results[str(index)]["aggregate_rmse_ratio_to_inside_orbit"]
            for index in budget_indices
        ]
        score = (
            float(np.exp(np.mean(np.log(valid_default_ratios))))
            if coverage_failures == 0
            and all(value is not None and value > 0.0 for value in valid_default_ratios)
            else None
        )
        result = {
            "lambda0": lambda0,
            "candidate_pool": candidate_pool,
            "is_registered_default": (
                lambda0 == REGISTERED_LAMBDA0
                and candidate_pool == REGISTERED_CANDIDATE_POOL
            ),
            "budgets": budget_results,
            "overall": {
                "coverage_failures": coverage_failures,
                "total_cells": len(all_candidate),
                "coverage_failure_rate": float(coverage_failures / len(all_candidate)),
                "geometric_mean_rmse_ratio_to_registered_default": score,
                "geometric_mean_rmse_ratio_to_inside_orbit": (
                    float(np.exp(np.mean(np.log(valid_orbit_ratios))))
                    if coverage_failures == 0
                    and all(value is not None and value > 0.0 for value in valid_orbit_ratios)
                    else None
                ),
                "paired_vs_registered_default": overall_default,
                "paired_vs_inside_orbit": overall_orbit,
                "mean_design_seconds": float(
                    np.mean([float(cell["design_seconds"]) for cell in all_candidate])
                ),
            },
        }
        results[configuration_id] = result
        ranking.append(
            {
                "configuration_id": configuration_id,
                "lambda0": lambda0,
                "candidate_pool": candidate_pool,
                "coverage_failures": coverage_failures,
                "score_geometric_mean_rmse_ratio_to_default": score,
                "paired_win_rate_vs_default": overall_default[
                    "win_rate_all_registered_repeats"
                ],
                "paired_win_rate_vs_orbit": overall_orbit[
                    "win_rate_all_registered_repeats"
                ],
                "mean_design_seconds": result["overall"]["mean_design_seconds"],
            }
        )

    ranking.sort(
        key=lambda row: (
            int(row["coverage_failures"]),
            float("inf")
            if row["score_geometric_mean_rmse_ratio_to_default"] is None
            else float(row["score_geometric_mean_rmse_ratio_to_default"]),
            float(row["mean_design_seconds"]),
        )
    )

    default_lookup = {
        (int(cell["repeat"]), int(cell["budget_index"])): cell
        for cells in default_by_budget.values()
        for cell in cells
    }
    orbit_lookup = {
        (int(cell["repeat"]), int(cell["budget_index"])): cell
        for cells in orbit_by_budget.values()
        for cell in cells
    }
    for cell in ordered:
        if cell["method"] != "inside_greedy":
            continue
        key = (int(cell["repeat"]), int(cell["budget_index"]))
        default = default_lookup[key]
        orbit = orbit_lookup[key]
        cell["rmse_ratio_to_registered_default_same_repeat"] = _safe_ratio(
            cell["rmse"], default["rmse"]
        )
        cell["rmse_ratio_to_inside_orbit_same_repeat"] = _safe_ratio(
            cell["rmse"], orbit["rmse"]
        )
    return results, ranking, ordered


def _initial_report(
    *,
    dataset: str,
    stage: str,
    configurations: Sequence[Configuration],
    requested_configurations: Sequence[Configuration],
    budget_multipliers: Sequence[int],
    budget_indices: Sequence[int],
    repeats: int,
    base_seed: int,
    processes: int,
    truth: np.ndarray,
    truth_algorithm: str,
) -> dict[str, Any]:
    inner_budgets, total_budgets = total_call_budgets(
        dataset, tuple(int(value) for value in budget_multipliers)
    )
    num_players = _num_players(dataset)
    efficiency_target = 10.0 if dataset == "airport" else 1.0
    return {
        "status": "running",
        "experiment": EXPERIMENT_ID,
        "game": _game_metadata(dataset),
        "configuration": {
            "dataset": dataset,
            "stage": stage,
            "holdout_role": (
                "hyperparameter_selection"
                if stage == "screening"
                else "confirmation_on_preregistered_holdout_seed"
            ),
            "selection_protocol": (
                "screen configurations on the screening report; pass only selected lambda0:K settings to a validation-stage run with a distinct base_seed"
            ),
            "budget_multipliers": list(budget_multipliers),
            "all_inner_utility_call_budgets": list(inner_budgets),
            "all_total_utility_call_budgets": list(total_budgets),
            "selected_budget_indices": list(budget_indices),
            "selected_inner_utility_call_budgets": [
                inner_budgets[index] for index in budget_indices
            ],
            "selected_total_utility_call_budgets": [
                total_budgets[index] for index in budget_indices
            ],
            "requested_configurations": [
                {"lambda0": lambda0, "candidate_pool": pool}
                for lambda0, pool in requested_configurations
            ],
            "evaluated_configurations": [
                {"lambda0": lambda0, "candidate_pool": pool}
                for lambda0, pool in configurations
            ],
            "registered_default": {
                "lambda0": REGISTERED_LAMBDA0,
                "candidate_pool": REGISTERED_CANDIDATE_POOL,
            },
            "inside_greedy_design": "per_size_frame_coupled_design",
            "inside_greedy_estimator": "ofa_conditional_mean_ratio_missing_raise",
            "inside_orbit_design": "cyclic_orbit_frame_design",
            "inside_orbit_estimator": "ofa_conditional_mean_ratio_strict_balanced",
            "inside_orbit_candidate_pool": ORBIT_CANDIDATE_POOL,
            "mean_balance_mode": "normalized",
            "repeats": repeats,
            "registered_repeats": REGISTERED_REPEATS,
            "base_seed": base_seed,
            "registered_base_seed": REGISTERED_BASE_SEED,
            "uses_distinct_seed_from_registered_screening": (
                stage == "validation" and base_seed != REGISTERED_BASE_SEED
            ),
            "seed_schedule": (
                "SeedSequence([base_seed, repeat, budget_index, method_index]); "
                "Greedy method_index=0, Orbit method_index=1"
            ),
            "greedy_relabel_substream": "SeedSequence([greedy_seed, 0x1A51DE])",
            "boundary_utility_calls": 2 * num_players + 2,
            "processes": processes,
            "checkpoint_granularity": "one method/configuration-repeat-budget cell",
            "call_budget_semantics": "same physical utility-call cap for every successful paired cell",
            "preregistered_protocol": (
                repeats == REGISTERED_REPEATS
                and base_seed == REGISTERED_BASE_SEED
                and tuple(budget_multipliers) == DEFAULT_BUDGET_MULTIPLIERS
            ),
        },
        "ground_truth": {
            "algorithm": truth_algorithm,
            "values": truth.tolist(),
            "sum": float(truth.sum()),
            "efficiency_target": efficiency_target,
            "efficiency_error": float(abs(truth.sum() - efficiency_target)),
        },
        "raw_cells": [],
        "results_by_configuration": {},
        "screening_ranking": [],
        "validation": {},
    }


def _write_json_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _expected_payloads(
    report: Mapping[str, Any], truth: np.ndarray
) -> list[dict[str, Any]]:
    configuration = report["configuration"]
    dataset = configuration["dataset"]
    base = {
        "dataset": dataset,
        "base_seed": int(configuration["base_seed"]),
        "truth": truth.tolist(),
    }
    inner_budgets = configuration["all_inner_utility_call_budgets"]
    total_budgets = configuration["all_total_utility_call_budgets"]
    payloads: list[dict[str, Any]] = []
    for budget_index in configuration["selected_budget_indices"]:
        for repeat in range(int(configuration["repeats"])):
            payloads.append(
                {
                    **base,
                    "method": "inside_orbit",
                    "repeat": repeat,
                    "budget_index": budget_index,
                    "inner_calls": int(inner_budgets[budget_index]),
                    "total_calls": int(total_budgets[budget_index]),
                    "lambda0": None,
                    "candidate_pool": ORBIT_CANDIDATE_POOL,
                }
            )
            for item in configuration["evaluated_configurations"]:
                payloads.append(
                    {
                        **base,
                        "method": "inside_greedy",
                        "repeat": repeat,
                        "budget_index": budget_index,
                        "inner_calls": int(inner_budgets[budget_index]),
                        "total_calls": int(total_budgets[budget_index]),
                        "lambda0": float(item["lambda0"]),
                        "candidate_pool": int(item["candidate_pool"]),
                    }
                )
    return payloads


def _validate_completed_report(
    report: Mapping[str, Any], expected_payloads: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    raw_cells = report["raw_cells"]
    expected_keys = {_cell_key(payload) for payload in expected_payloads}
    observed_keys = {_cell_key(cell) for cell in raw_cells}
    if len(raw_cells) != len(observed_keys) or observed_keys != expected_keys:
        raise RuntimeError("sweep report has missing or duplicate cells")
    configuration = report["configuration"]
    seed_mismatches = 0
    call_mismatches = 0
    invalid_ratio_cells = 0
    coverage_failures = 0
    greedy_schedule_pairing_mismatches = 0
    greedy_relabel_pairing_mismatches = 0
    for cell in raw_cells:
        method_index = 0 if cell["method"] == "inside_greedy" else 1
        expected_seed = _seed_for(
            int(configuration["base_seed"]),
            int(cell["repeat"]),
            int(cell["budget_index"]),
            method_index,
        )
        seed_mismatches += int(int(cell["seed"]) != expected_seed)
        if cell["status"] == "coverage_failure":
            coverage_failures += 1
            continue
        call_mismatches += int(
            int(cell["actual_utility_calls"]) != int(cell["total_calls"])
        )
        estimator = str(cell["diagnostics"]["estimator"])
        invalid_ratio_cells += int("ratio" not in estimator)
    greedy_groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for cell in raw_cells:
        if cell["method"] == "inside_greedy":
            greedy_groups.setdefault(
                (int(cell["repeat"]), int(cell["budget_index"])), []
            ).append(cell)
    for cells in greedy_groups.values():
        schedule_hashes = {
            str(cell["diagnostics"]["size_schedule_sha256"])
            for cell in cells
        }
        relabel_hashes = {
            str(cell["diagnostics"]["relabel_permutation_sha256"])
            for cell in cells
        }
        greedy_schedule_pairing_mismatches += int(len(schedule_hashes) != 1)
        greedy_relabel_pairing_mismatches += int(len(relabel_hashes) != 1)
    if (
        seed_mismatches
        or call_mismatches
        or invalid_ratio_cells
        or greedy_schedule_pairing_mismatches
        or greedy_relabel_pairing_mismatches
    ):
        raise RuntimeError("seed, call, or estimator validation failed")
    return {
        "passed": True,
        "expected_cells": len(expected_keys),
        "observed_cells": len(observed_keys),
        "seed_schedule_mismatches": seed_mismatches,
        "successful_call_budget_mismatches": call_mismatches,
        "non_ratio_estimator_cells": invalid_ratio_cells,
        "greedy_size_schedule_pairing_mismatches": (
            greedy_schedule_pairing_mismatches
        ),
        "greedy_relabel_pairing_mismatches": greedy_relabel_pairing_mismatches,
        "all_greedy_hyperparameters_share_paired_size_schedules": True,
        "all_greedy_hyperparameters_share_paired_relabels": True,
        "coverage_failures": coverage_failures,
        "exact_ground_truth": True,
        "same_successful_physical_call_budget": True,
    }


def run_experiment(
    *,
    dataset: str,
    configurations: Sequence[Configuration],
    stage: str = "screening",
    budget_multipliers: Sequence[int] = DEFAULT_BUDGET_MULTIPLIERS,
    budget_indices: Sequence[int] | None = None,
    repeats: int = REGISTERED_REPEATS,
    base_seed: int = REGISTERED_BASE_SEED,
    processes: int = 1,
    output_path: Path | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    if dataset not in {"airport", "voting"}:
        raise ValueError("dataset must be 'airport' or 'voting'")
    if repeats < 1 or processes < 1:
        raise ValueError("repeats and processes must be positive")
    budget_multipliers = tuple(int(value) for value in budget_multipliers)
    # Reuse the canonical budget validator.
    total_call_budgets(dataset, budget_multipliers)
    selected_indices = selected_budget_indices(
        stage, len(budget_multipliers), budget_indices
    )
    requested = tuple((float(value), int(pool)) for value, pool in configurations)
    normalized = _normalize_configurations(requested)
    truth, truth_algorithm = _exact_truth(dataset)
    report = _initial_report(
        dataset=dataset,
        stage=stage,
        configurations=normalized,
        requested_configurations=requested,
        budget_multipliers=budget_multipliers,
        budget_indices=selected_indices,
        repeats=repeats,
        base_seed=base_seed,
        processes=processes,
        truth=truth,
        truth_algorithm=truth_algorithm,
    )
    if output_path is not None and resume and output_path.exists():
        saved = json.loads(output_path.read_text(encoding="utf-8"))
        comparable_saved = copy.deepcopy(saved.get("configuration"))
        comparable_current = copy.deepcopy(report["configuration"])
        if isinstance(comparable_saved, dict):
            comparable_saved.pop("processes", None)
        comparable_current.pop("processes", None)
        if comparable_saved != comparable_current:
            raise ValueError("resume report configuration does not match")
        report = saved
        report["configuration"]["processes"] = processes
        report["status"] = "running"

    payloads = _expected_payloads(report, truth)
    existing = {_cell_key(cell) for cell in report["raw_cells"]}
    pending = [payload for payload in payloads if _cell_key(payload) not in existing]

    def accept(cell: dict[str, Any]) -> None:
        report["raw_cells"].append(cell)
        report["raw_cells"].sort(key=lambda item: str(_cell_key(item)))
        if output_path is not None:
            _write_json_atomic(output_path, report)

    if not pending:
        pass
    elif processes == 1:
        for payload in pending:
            accept(_run_cell(payload))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(processes, len(pending)), mp_context=context
        ) as executor:
            future_map = {
                executor.submit(_run_cell, payload): payload for payload in pending
            }
            for future in as_completed(future_map):
                accept(future.result())

    results, ranking, enriched_cells = aggregate_report(
        report["raw_cells"], normalized, selected_indices, repeats
    )
    report["raw_cells"] = enriched_cells
    report["results_by_configuration"] = results
    report["screening_ranking"] = ranking
    report["validation"] = _validate_completed_report(report, payloads)
    report["status"] = "complete"
    if output_path is not None:
        _write_json_atomic(output_path, report)
    return report


def _parse_configuration(text: str) -> Configuration:
    try:
        lambda_text, pool_text = text.split(":", 1)
        return float(lambda_text), int(pool_text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "configurations must have the form lambda0:K, e.g. 0.5:8"
        ) from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("airport", "voting"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("screening", "validation"), default="screening")
    parser.add_argument("--lambdas", nargs="+", type=float, default=list(DEFAULT_LAMBDAS))
    parser.add_argument(
        "--candidate-pools", nargs="+", type=int, default=list(DEFAULT_CANDIDATE_POOLS)
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=_parse_configuration,
        help="explicit lambda0:K settings; overrides the Cartesian grid",
    )
    parser.add_argument(
        "--budget-indices",
        nargs="+",
        type=int,
        help="0-based indices into the registered five-budget grid",
    )
    parser.add_argument(
        "--budget-multipliers",
        nargs="+",
        type=int,
        default=list(DEFAULT_BUDGET_MULTIPLIERS),
    )
    parser.add_argument("--repeats", type=int, default=REGISTERED_REPEATS)
    parser.add_argument("--base-seed", type=int, default=REGISTERED_BASE_SEED)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configurations = (
        tuple(args.configs)
        if args.configs
        else configuration_grid(args.lambdas, args.candidate_pools)
    )
    report = run_experiment(
        dataset=args.dataset,
        configurations=configurations,
        stage=args.stage,
        budget_multipliers=args.budget_multipliers,
        budget_indices=args.budget_indices,
        repeats=args.repeats,
        base_seed=args.base_seed,
        processes=args.processes,
        output_path=args.output,
        resume=args.resume,
    )
    best = report["screening_ranking"][0]
    print(
        f"complete: saved {args.output}; best={best['configuration_id']}; "
        f"score={best['score_geometric_mean_rmse_ratio_to_default']}"
    )


if __name__ == "__main__":
    main()

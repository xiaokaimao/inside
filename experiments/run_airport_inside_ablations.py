"""Run the three Airport ablations requested for INSIDE.

The experiment has one auditable six-design protocol:

* component ablation: OFA (IID), First-only, Frame-only, Full INSIDE;
* orbit ablation: paired Random-Orbit and INSIDE-Orbit;
* geometry audit: fixed-slice first and second moments for every design.

All six methods use the same strict OFA conditional-mean ratio estimator.
First/Frame/Full share their randomized-systematic size schedule and final
player relabeling.  The two orbit designs share every K=4 candidate pool,
size allocation, and relabeling; only their base-coalition selection differs.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
from scipy import stats

from experiments.airport_game import (
    AIRPORT_CLASS_COUNTS,
    AIRPORT_COSTS,
    game_fingerprint,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_official_ratio_ofa,
    evaluate_airport,
    exact_airport_shapley,
    fixed_slice_moment_diagnostics,
    iid_ofa_design,
    inner_size_distribution,
    paired_cyclic_orbit_designs,
    per_size_frame_coupled_design,
)


EXPERIMENT_ID = "airport_inside_component_orbit_geometry_ablations"
PROTOCOL_VERSION = "airport_ablation_v1"
METHOD_ORDER = (
    "iid_ofa",
    "first_only",
    "frame_only",
    "full_inside",
    "random_orbit",
    "inside_orbit",
)
METHOD_LABELS = {
    "iid_ofa": "OFA (IID)",
    "first_only": "First-only",
    "frame_only": "Frame-only",
    "full_inside": "Full INSIDE",
    "random_orbit": "Random-Orbit",
    "inside_orbit": "INSIDE-Orbit",
}
COMPONENT_METHODS = METHOD_ORDER[:4]
ORBIT_METHODS = METHOD_ORDER[4:]
GEOMETRY_METHODS = ("iid_ofa", "full_inside", "inside_orbit")
DEFAULT_BUDGET_MULTIPLIERS = (500, 1000, 2000, 5000, 10000)
GREEDY_CANDIDATE_POOL = 64
ORBIT_CANDIDATE_POOL = 4
FULL_LAMBDA0 = 1.0 / 16.0
FIRST_ONLY_LAMBDA0 = 1.0
BOUNDARY_CALLS = 2 * len(AIRPORT_COSTS) + 2


def _seed_value(sequence: np.random.SeedSequence) -> int:
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _cell_seeds(
    base_seed: int, repeat: int, budget_index: int
) -> dict[str, int]:
    root = np.random.SeedSequence([base_seed, repeat, budget_index])
    iid, component, relabel, orbit = root.spawn(4)
    return {
        "iid": _seed_value(iid),
        "component": _seed_value(component),
        "component_relabel": _seed_value(relabel),
        "orbit": _seed_value(orbit),
    }


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _coalition_sha256(coalitions: np.ndarray, sizes: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(coalitions.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(sizes, dtype="<i8").tobytes())
    digest.update(np.packbits(coalitions, axis=1).tobytes(order="C"))
    return digest.hexdigest()


def _ratio_coverage(
    coalitions: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
    num_players = coalitions.shape[1]
    expected_sizes, _, _ = inner_size_distribution(num_players)
    missing_inclusion = 0
    missing_exclusion = 0
    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    balanced_sizes = 0
    for size in expected_sizes:
        rows = coalitions[sizes == size]
        inclusion = rows.sum(axis=0, dtype=np.int64)
        exclusion = len(rows) - inclusion
        missing_inclusion += int(np.count_nonzero(inclusion == 0))
        missing_exclusion += int(np.count_nonzero(exclusion == 0))
        minimum_inclusion = min(minimum_inclusion, int(inclusion.min()))
        minimum_exclusion = min(minimum_exclusion, int(exclusion.min()))
        balanced_sizes += int(np.all(inclusion == inclusion[0]))
    return {
        "all_player_size_strata_covered": (
            missing_inclusion == 0 and missing_exclusion == 0
        ),
        "minimum_inclusion_count": int(minimum_inclusion),
        "minimum_exclusion_count": int(minimum_exclusion),
        "missing_inclusion_strata": int(missing_inclusion),
        "missing_exclusion_strata": int(missing_exclusion),
        "exactly_balanced_size_count": int(balanced_sizes),
        "total_inner_size_count": int(len(expected_sizes)),
    }


def _compact_design_diagnostics(value: Mapping[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, np.generic):
            compact[key] = item.item()
        elif isinstance(item, np.ndarray):
            compact[key] = item.tolist()
        else:
            compact[key] = item
    return compact


def _evaluate_design(
    method: str,
    design: Any,
    *,
    boundary: Any,
    truth: np.ndarray,
    inner_calls: int,
    design_seconds: float,
) -> dict[str, Any]:
    geometry_started = time.perf_counter()
    geometry = fixed_slice_moment_diagnostics(
        design.coalitions,
        design.sizes,
        require_all_inner_sizes=True,
    )
    geometry_seconds = time.perf_counter() - geometry_started
    coverage = _ratio_coverage(design.coalitions, design.sizes)
    if not coverage["all_player_size_strata_covered"]:
        raise RuntimeError(f"{method} lacks strict OFA ratio coverage")

    estimation_started = time.perf_counter()
    utilities = np.asarray(
        evaluate_airport(design.coalitions, AIRPORT_COSTS),
        dtype=np.float64,
    )
    estimate = np.asarray(
        estimate_official_ratio_ofa(
            design, utilities, boundary, missing="raise"
        ),
        dtype=np.float64,
    )
    estimation_seconds = time.perf_counter() - estimation_started
    if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
        raise RuntimeError(f"{method} returned an invalid estimate")
    repeat_rmse = float(np.sqrt(np.mean(np.square(estimate - truth))))
    actual_calls = len(design.coalitions) + BOUNDARY_CALLS
    if actual_calls != inner_calls + BOUNDARY_CALLS:
        raise RuntimeError(f"{method} physical-call accounting mismatch")
    return {
        "method": method,
        "label": METHOD_LABELS[method],
        "estimate": estimate.tolist(),
        "repeat_rmse": repeat_rmse,
        "actual_utility_calls": int(actual_calls),
        "efficiency_residual": float(estimate.sum() - truth.sum()),
        "size_schedule_sha256": _sha256_array(design.sizes),
        "coalition_design_sha256": _coalition_sha256(
            design.coalitions, design.sizes
        ),
        "coverage": coverage,
        "geometry": geometry,
        "design_method": design.method,
        "design_seconds": float(design_seconds),
        "geometry_seconds": float(geometry_seconds),
        "utility_and_estimation_seconds": float(estimation_seconds),
        "design_diagnostics": _compact_design_diagnostics(
            design.diagnostics
        ),
    }


def _run_cell(
    repeat: int,
    budget_index: int,
    inner_calls: int,
    *,
    base_seed: int,
    greedy_candidate_pool: int,
    orbit_candidate_pool: int,
    design_jobs: int,
    design_start_method: str,
) -> dict[str, Any]:
    num_players = len(AIRPORT_COSTS)
    truth = exact_airport_shapley(AIRPORT_COSTS)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        evaluate_airport(boundary_rows, AIRPORT_COSTS), num_players
    )
    seeds = _cell_seeds(base_seed, repeat, budget_index)
    methods: dict[str, Any] = {}
    cell_started = time.perf_counter()

    started = time.perf_counter()
    iid = iid_ofa_design(
        num_players,
        inner_calls,
        seed=seeds["iid"],
        compute_diagnostics=False,
    )
    methods["iid_ofa"] = _evaluate_design(
        "iid_ofa",
        iid,
        boundary=boundary,
        truth=truth,
        inner_calls=inner_calls,
        design_seconds=time.perf_counter() - started,
    )
    del iid

    component_specs = (
        ("first_only", FIRST_ONLY_LAMBDA0, 0.0),
        ("frame_only", 0.0, 1.0),
        ("full_inside", FULL_LAMBDA0, 1.0),
    )
    shared_schedule_hash: str | None = None
    shared_relabel_hash: str | None = None
    for method, mean_balance, second_moment_weight in component_specs:
        started = time.perf_counter()
        design = per_size_frame_coupled_design(
            num_players,
            inner_calls,
            seed=seeds["component"],
            candidate_pool=greedy_candidate_pool,
            mean_balance=mean_balance,
            second_moment_weight=second_moment_weight,
            mean_balance_mode="normalized",
            relabel_seed=seeds["component_relabel"],
            design_jobs=design_jobs,
            design_start_method=design_start_method,
            compute_frame_diagnostics=False,
        )
        result = _evaluate_design(
            method,
            design,
            boundary=boundary,
            truth=truth,
            inner_calls=inner_calls,
            design_seconds=time.perf_counter() - started,
        )
        relabel_hash = str(
            design.diagnostics["relabel_permutation_sha256"]
        )
        if shared_schedule_hash is None:
            shared_schedule_hash = result["size_schedule_sha256"]
            shared_relabel_hash = relabel_hash
        elif (
            result["size_schedule_sha256"] != shared_schedule_hash
            or relabel_hash != shared_relabel_hash
        ):
            raise RuntimeError(
                "component variants do not share schedule/relabeling"
            )
        methods[method] = result
        del design

    started = time.perf_counter()
    random_orbit, inside_orbit = paired_cyclic_orbit_designs(
        num_players,
        num_orbits=inner_calls // num_players,
        seed=seeds["orbit"],
        candidate_pool=orbit_candidate_pool,
    )
    paired_orbit_seconds = time.perf_counter() - started
    random_hash = random_orbit.diagnostics[
        "shared_candidate_pool_sha256"
    ]
    inside_hash = inside_orbit.diagnostics[
        "shared_candidate_pool_sha256"
    ]
    random_relabel = random_orbit.diagnostics[
        "shared_relabel_permutations_sha256"
    ]
    inside_relabel = inside_orbit.diagnostics[
        "shared_relabel_permutations_sha256"
    ]
    if (
        random_hash != inside_hash
        or random_relabel != inside_relabel
        or not np.array_equal(random_orbit.sizes, inside_orbit.sizes)
    ):
        raise RuntimeError("paired orbit controls are not shared")
    methods["random_orbit"] = _evaluate_design(
        "random_orbit",
        random_orbit,
        boundary=boundary,
        truth=truth,
        inner_calls=inner_calls,
        design_seconds=paired_orbit_seconds,
    )
    methods["inside_orbit"] = _evaluate_design(
        "inside_orbit",
        inside_orbit,
        boundary=boundary,
        truth=truth,
        inner_calls=inner_calls,
        design_seconds=paired_orbit_seconds,
    )

    return {
        "repeat": int(repeat),
        "budget_index": int(budget_index),
        "inner_utility_calls": int(inner_calls),
        "total_utility_calls": int(inner_calls + BOUNDARY_CALLS),
        "seeds": seeds,
        "paired_controls": {
            "component_shared_size_schedule_sha256": shared_schedule_hash,
            "component_shared_relabel_sha256": shared_relabel_hash,
            "orbit_shared_candidate_pool_sha256": random_hash,
            "orbit_shared_relabel_sha256": random_relabel,
        },
        "methods": methods,
        "cell_wall_seconds": float(time.perf_counter() - cell_started),
    }


def _bootstrap_interval(
    values: np.ndarray, samples: int, seed: int, *, rmse: bool
) -> list[float]:
    rng = np.random.default_rng(seed)
    repeats = len(values)
    indices = rng.integers(0, repeats, size=(samples, repeats))
    draws = values[indices].mean(axis=1)
    if rmse:
        draws = np.sqrt(draws)
    return np.quantile(draws, [0.025, 0.975]).astype(float).tolist()


def _paired_reduction_interval(
    numerator_mse: np.ndarray,
    denominator_mse: np.ndarray,
    samples: int,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    repeats = len(numerator_mse)
    indices = rng.integers(0, repeats, size=(samples, repeats))
    ratio = np.sqrt(
        numerator_mse[indices].mean(axis=1)
        / denominator_mse[indices].mean(axis=1)
    )
    return np.quantile(100.0 * (1.0 - ratio), [0.025, 0.975]).tolist()


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1 or len(x) < 2:
        return {
            "defined": False,
            "reason": "correlation needs paired one-dimensional samples",
            "pearson_log10": None,
            "spearman": None,
        }
    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(x <= 0.0)
        or np.any(y <= 0.0)
    ):
        return {
            "defined": False,
            "reason": "log correlation needs finite positive values",
            "pearson_log10": None,
            "spearman": None,
        }
    log_x = np.log10(x)
    log_y = np.log10(y)
    if np.ptp(log_x) == 0.0 or np.ptp(log_y) == 0.0:
        return {
            "defined": False,
            "reason": "correlation is undefined for a constant variable",
            "pearson_log10": None,
            "spearman": None,
        }
    return {
        "defined": True,
        "reason": None,
        "pearson_log10": float(
            stats.pearsonr(log_x, log_y).statistic
        ),
        "spearman": float(stats.spearmanr(x, y).statistic),
    }


def _aggregate(
    cells: list[dict[str, Any]],
    *,
    inner_budgets: tuple[int, ...],
    repeats: int,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    for budget_index, inner_calls in enumerate(inner_budgets):
        selected = sorted(
            (
                cell
                for cell in cells
                if int(cell["budget_index"]) == budget_index
            ),
            key=lambda cell: int(cell["repeat"]),
        )
        if len(selected) != repeats:
            raise RuntimeError("a budget cell has missing repeats")
        methods: dict[str, Any] = {}
        mse_by_method: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(METHOD_ORDER):
            rows = [cell["methods"][method] for cell in selected]
            estimates = np.asarray(
                [row["estimate"] for row in rows], dtype=np.float64
            )
            repeat_rmse = np.asarray(
                [row["repeat_rmse"] for row in rows], dtype=np.float64
            )
            mse = np.square(repeat_rmse)
            mse_by_method[method] = mse
            first = np.asarray(
                [row["geometry"]["first_moment_rms"] for row in rows]
            )
            second = np.asarray(
                [row["geometry"]["frame_frobenius_rms"] for row in rows]
            )
            methods[method] = {
                "label": METHOD_LABELS[method],
                "estimates": estimates.tolist(),
                "repeat_rmse": repeat_rmse.tolist(),
                "aggregate_rmse": float(np.sqrt(mse.mean())),
                "aggregate_rmse_bootstrap_95": _bootstrap_interval(
                    mse,
                    bootstrap_samples,
                    seed=71001 + 101 * budget_index + method_index,
                    rmse=True,
                ),
                "first_moment_rms_by_repeat": first.tolist(),
                "mean_first_moment_rms": float(first.mean()),
                "first_moment_mean_bootstrap_95": _bootstrap_interval(
                    first,
                    bootstrap_samples,
                    seed=72001 + 101 * budget_index + method_index,
                    rmse=False,
                ),
                "frame_frobenius_rms_by_repeat": second.tolist(),
                "mean_frame_frobenius_rms": float(second.mean()),
                "frame_frobenius_mean_bootstrap_95": _bootstrap_interval(
                    second,
                    bootstrap_samples,
                    seed=73001 + 101 * budget_index + method_index,
                    rmse=False,
                ),
            }

        def reduction(numerator: str, denominator: str, seed: int) -> dict[str, Any]:
            value = 100.0 * (
                1.0
                - np.sqrt(mse_by_method[numerator].mean())
                / np.sqrt(mse_by_method[denominator].mean())
            )
            return {
                "percent": float(value),
                "paired_bootstrap_95": _paired_reduction_interval(
                    mse_by_method[numerator],
                    mse_by_method[denominator],
                    bootstrap_samples,
                    seed,
                ),
            }

        results[str(inner_calls)] = {
            "inner_utility_calls": int(inner_calls),
            "total_utility_calls": int(inner_calls + BOUNDARY_CALLS),
            "methods": methods,
            "component_contributions": {
                "full_vs_first_only_rmse_reduction": reduction(
                    "full_inside", "first_only", 74001 + budget_index
                ),
                "full_vs_frame_only_rmse_reduction": reduction(
                    "full_inside", "frame_only", 75001 + budget_index
                ),
            },
            "orbit_frame_selection": {
                "inside_vs_random_rmse_reduction": reduction(
                    "inside_orbit", "random_orbit", 76001 + budget_index
                )
            },
        }

    points: list[dict[str, Any]] = []
    for cell in cells:
        for method in GEOMETRY_METHODS:
            row = cell["methods"][method]
            points.append(
                {
                    "method": method,
                    "repeat": int(cell["repeat"]),
                    "budget_index": int(cell["budget_index"]),
                    "total_utility_calls": int(cell["total_utility_calls"]),
                    "frame_frobenius_rms": float(
                        row["geometry"]["frame_frobenius_rms"]
                    ),
                    "repeat_rmse": float(row["repeat_rmse"]),
                }
            )
    x = np.asarray([point["frame_frobenius_rms"] for point in points])
    y = np.asarray([point["repeat_rmse"] for point in points])
    correlations: dict[str, Any] = {"overall": _correlation(x, y)}
    correlations["by_method"] = {}
    for method in GEOMETRY_METHODS:
        take = np.asarray([point["method"] == method for point in points])
        correlations["by_method"][method] = _correlation(x[take], y[take])

    log_x = np.log10(x)
    log_y = np.log10(y)
    residual_x = np.empty_like(log_x)
    residual_y = np.empty_like(log_y)
    for method in GEOMETRY_METHODS:
        for budget_index in range(len(inner_budgets)):
            take = np.asarray(
                [
                    point["method"] == method
                    and point["budget_index"] == budget_index
                    for point in points
                ]
            )
            residual_x[take] = log_x[take] - log_x[take].mean()
            residual_y[take] = log_y[take] - log_y[take].mean()
    residual = _correlation(
        np.power(10.0, residual_x), np.power(10.0, residual_y)
    )
    correlations["within_method_budget_log_residual"] = {
        "defined": residual["defined"],
        "reason": residual["reason"],
        "pearson": residual["pearson_log10"],
        "spearman": residual["spearman"],
        "interpretation": (
            "Association after removing each method-by-budget cell mean; "
            "three repeats per cell, so treat as descriptive."
        ),
    }
    return results, {"points": points, "correlations": correlations}


def _configuration(
    *,
    budget_multipliers: tuple[int, ...],
    repeats: int,
    base_seed: int,
    bootstrap_samples: int,
    repeat_processes: int,
    design_jobs: int,
    design_start_method: str,
    greedy_candidate_pool: int,
    orbit_candidate_pool: int,
) -> dict[str, Any]:
    inner_budgets = [len(AIRPORT_COSTS) * value for value in budget_multipliers]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "airport",
        "players": len(AIRPORT_COSTS),
        "budget_multipliers": list(budget_multipliers),
        "inner_utility_call_budgets": inner_budgets,
        "total_utility_call_budgets": [
            value + BOUNDARY_CALLS for value in inner_budgets
        ],
        "boundary_utility_calls": BOUNDARY_CALLS,
        "repeats": repeats,
        "base_seed": base_seed,
        "bootstrap_samples": bootstrap_samples,
        "repeat_processes": repeat_processes,
        "repeat_start_method": "spawn",
        "methods": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "strict_common_estimator": (
            "official OFA conditional-mean ratio, missing=raise"
        ),
        "greedy_candidate_pool": greedy_candidate_pool,
        "orbit_candidate_pool": orbit_candidate_pool,
        "full_lambda0": FULL_LAMBDA0,
        "full_effective_lambda": FULL_LAMBDA0 * 98.0 / 99.0,
        "first_only_score": "u^T m_s (positive scaling is immaterial)",
        "frame_only_score": "u^T A_s u",
        "full_score": "u^T A_s u + lambda_eff u^T m_s",
        "design_jobs": design_jobs,
        "design_start_method": design_start_method,
        "paired_component_controls": (
            "First/Frame/Full share systematic size schedule and relabeling"
        ),
        "paired_orbit_controls": (
            "Random/INSIDE share every K=4 pool, allocation, and relabeling"
        ),
        "geometry_aggregation": (
            "equal-size macro RMS over s=2,...,98"
        ),
        "iid_interpretation_caveat": (
            "OFA (IID) also differs in IID size draws versus the systematic "
            "size coupling used by the three Greedy variants, and it permits "
            "coalition repeats whereas Greedy avoids repeats within a slice "
            "until that slice is exhausted. Full-vs-First and Full-vs-Frame "
            "are the clean component contrasts."
        ),
        "utility_call_interpretation": (
            "The x-axis is the equivalent per-method call budget: inner "
            "coalitions plus 2n+2 boundary calls. Boundary utilities are "
            "physically evaluated once per joint experimental cell and then "
            "shared only as a runtime optimization."
        ),
    }


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "complete":
        raise ValueError("report status must be complete")
    configuration = report.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("report lacks configuration")
    if tuple(configuration.get("methods", ())) != METHOD_ORDER:
        raise ValueError("report method order is invalid")
    repeats = int(configuration["repeats"])
    inner_budgets = tuple(configuration["inner_utility_call_budgets"])
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    cells = report.get("raw_cells")
    if not isinstance(cells, list) or len(cells) != repeats * len(inner_budgets):
        raise ValueError("report has an incomplete raw cell grid")
    keys: set[tuple[int, int]] = set()
    for cell in cells:
        key = (int(cell["repeat"]), int(cell["budget_index"]))
        if key in keys:
            raise ValueError("report has duplicate raw cells")
        keys.add(key)
        methods = cell.get("methods")
        if not isinstance(methods, Mapping) or set(methods) != set(METHOD_ORDER):
            raise ValueError("raw cell has an invalid method set")
        paired = cell["paired_controls"]
        component_hashes = {
            methods[method]["size_schedule_sha256"]
            for method in ("first_only", "frame_only", "full_inside")
        }
        if len(component_hashes) != 1 or next(iter(component_hashes)) != paired[
            "component_shared_size_schedule_sha256"
        ]:
            raise ValueError("component schedule pairing is invalid")
        component_relabels = {
            methods[method]["design_diagnostics"][
                "relabel_permutation_sha256"
            ]
            for method in ("first_only", "frame_only", "full_inside")
        }
        if (
            len(component_relabels) != 1
            or next(iter(component_relabels))
            != paired["component_shared_relabel_sha256"]
        ):
            raise ValueError("component relabel pairing is invalid")
        orbit_hashes = {
            methods[method]["design_diagnostics"][
                "shared_candidate_pool_sha256"
            ]
            for method in ORBIT_METHODS
        }
        if len(orbit_hashes) != 1 or next(iter(orbit_hashes)) != paired[
            "orbit_shared_candidate_pool_sha256"
        ]:
            raise ValueError("orbit candidate pairing is invalid")
        orbit_relabels = {
            methods[method]["design_diagnostics"][
                "shared_relabel_permutations_sha256"
            ]
            for method in ORBIT_METHODS
        }
        orbit_schedules = {
            methods[method]["size_schedule_sha256"]
            for method in ORBIT_METHODS
        }
        if (
            len(orbit_relabels) != 1
            or next(iter(orbit_relabels))
            != paired["orbit_shared_relabel_sha256"]
            or len(orbit_schedules) != 1
        ):
            raise ValueError("orbit relabel/schedule pairing is invalid")
        expected_components = {
            "first_only": (["first_moment"], 0.0),
            "frame_only": (["second_moment"], 1.0),
            "full_inside": (["first_moment", "second_moment"], 1.0),
        }
        for method, (objectives, second_weight) in expected_components.items():
            diagnostics = methods[method]["design_diagnostics"]
            if (
                diagnostics.get("second_moment_scope") != "per_size"
                or diagnostics.get("objective_components") != objectives
                or not math.isclose(
                    float(diagnostics.get("second_moment_weight", math.nan)),
                    second_weight,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
            ):
                raise ValueError(f"{method} objective diagnostics are invalid")
        for method in METHOD_ORDER:
            row = methods[method]
            estimate = np.asarray(row["estimate"], dtype=np.float64)
            if estimate.shape != truth.shape or not np.all(np.isfinite(estimate)):
                raise ValueError(f"{method} estimate is invalid")
            expected_rmse = float(
                np.sqrt(np.mean(np.square(estimate - truth)))
            )
            if not math.isclose(
                float(row["repeat_rmse"]),
                expected_rmse,
                rel_tol=0.0,
                abs_tol=1e-14,
            ):
                raise ValueError(f"{method} RMSE is inconsistent")
            if not row["coverage"]["all_player_size_strata_covered"]:
                raise ValueError(f"{method} lacks ratio coverage")
            geometry = row["geometry"]
            if (
                not geometry["all_inner_sizes_present"]
                or geometry["observed_inner_size_count"] != 97
                or geometry["missing_sizes"]
            ):
                raise ValueError(f"{method} geometry is incomplete")
            if int(row["actual_utility_calls"]) != int(
                cell["total_utility_calls"]
            ):
                raise ValueError(f"{method} call accounting is invalid")


def _write_atomic(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_experiment(
    *,
    budget_multipliers: tuple[int, ...] = DEFAULT_BUDGET_MULTIPLIERS,
    repeats: int = 3,
    base_seed: int = 20260827,
    bootstrap_samples: int = 5000,
    repeat_processes: int = 3,
    design_jobs: int = 40,
    design_start_method: str = "spawn",
    greedy_candidate_pool: int = GREEDY_CANDIDATE_POOL,
    orbit_candidate_pool: int = ORBIT_CANDIDATE_POOL,
    output_path: Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    if (
        repeats < 2
        or bootstrap_samples < 1
        or repeat_processes < 1
        or design_jobs < 1
    ):
        raise ValueError(
            "repeats, bootstrap samples, repeat processes, and design jobs "
            "are invalid"
        )
    if design_start_method not in mp.get_all_start_methods():
        raise ValueError("unsupported multiprocessing start method")
    if not budget_multipliers or any(value < 500 for value in budget_multipliers):
        raise ValueError("Airport ablation multipliers must be at least 500")
    if any(
        left >= right
        for left, right in zip(budget_multipliers, budget_multipliers[1:])
    ):
        raise ValueError("budget multipliers must increase")
    configuration = _configuration(
        budget_multipliers=budget_multipliers,
        repeats=repeats,
        base_seed=base_seed,
        bootstrap_samples=bootstrap_samples,
        repeat_processes=repeat_processes,
        design_jobs=design_jobs,
        design_start_method=design_start_method,
        greedy_candidate_pool=greedy_candidate_pool,
        orbit_candidate_pool=orbit_candidate_pool,
    )
    truth = exact_airport_shapley(AIRPORT_COSTS)
    report: dict[str, Any]
    if output_path is not None and resume and output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        saved = dict(report.get("configuration", {}))
        current = dict(configuration)
        # Worker topology affects scheduling only; cell and fixed-slice seed
        # substreams are fixed, so interrupted runs may resume with a faster
        # topology without changing any realized design.
        for key in ("design_jobs", "repeat_processes", "repeat_start_method"):
            saved.pop(key, None)
            current.pop(key, None)
        # These two fields only clarify interpretation and do not identify a
        # stochastic protocol.  Accept checkpoints from the earlier wording.
        for key in ("iid_interpretation_caveat", "utility_call_interpretation"):
            saved.pop(key, None)
            current.pop(key, None)
        if report.get("experiment") != EXPERIMENT_ID or saved != current:
            raise ValueError("checkpoint configuration does not match")
        report["configuration"] = configuration
    else:
        report = {
            "status": "running",
            "experiment": EXPERIMENT_ID,
            "configuration": configuration,
            "game": {
                "dataset": "airport",
                "model": "100-player airport cost game",
                "cost_class_counts": list(AIRPORT_CLASS_COUNTS),
                "utility": "v(S)=max_{i in S} c_i; v(empty)=0",
                "fingerprint_sha256": game_fingerprint(),
            },
            "ground_truth": {
                "algorithm": "exact threshold-game decomposition",
                "values": truth.tolist(),
                "sum": float(truth.sum()),
                "efficiency_error": float(abs(truth.sum() - 10.0)),
            },
            "raw_cells": [],
            "results_by_inner_budget": {},
            "geometry_error_relation": {},
            "experiment_wall_seconds": 0.0,
        }
    completed = {
        (int(cell["repeat"]), int(cell["budget_index"]))
        for cell in report["raw_cells"]
    }
    inner_budgets = tuple(
        len(AIRPORT_COSTS) * value for value in budget_multipliers
    )
    pending = [
        (repeat, budget_index, inner_calls)
        for repeat in range(repeats)
        for budget_index, inner_calls in enumerate(inner_budgets)
        if (repeat, budget_index) not in completed
    ]
    run_started = time.perf_counter()
    prior_wall_seconds = float(report.get("experiment_wall_seconds", 0.0))
    completed_this_run = 0

    def merge_cell(cell: dict[str, Any]) -> None:
        nonlocal completed_this_run
        report["raw_cells"].append(cell)
        report["raw_cells"].sort(
            key=lambda item: (int(item["repeat"]), int(item["budget_index"]))
        )
        report["experiment_wall_seconds"] = float(
            prior_wall_seconds + time.perf_counter() - run_started
        )
        report["status"] = "running"
        if output_path is not None:
            _write_atomic(output_path, report)
        completed_this_run += 1
        print(
            f"completed cell {completed_this_run}/{len(pending)}: "
            f"repeat={int(cell['repeat']) + 1}, "
            f"inner_calls={int(cell['inner_utility_calls'])}",
            flush=True,
        )

    for budget_index in range(len(inner_budgets)):
        group = [task for task in pending if task[1] == budget_index]
        if not group:
            continue
        workers = min(repeat_processes, len(group))
        if workers <= 1:
            for repeat, current_budget_index, inner_calls in group:
                merge_cell(
                    _run_cell(
                        repeat,
                        current_budget_index,
                        inner_calls,
                        base_seed=base_seed,
                        greedy_candidate_pool=greedy_candidate_pool,
                        orbit_candidate_pool=orbit_candidate_pool,
                        design_jobs=design_jobs,
                        design_start_method=design_start_method,
                    )
                )
            continue
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = {
                executor.submit(
                    _run_cell,
                    repeat,
                    current_budget_index,
                    inner_calls,
                    base_seed=base_seed,
                    greedy_candidate_pool=greedy_candidate_pool,
                    orbit_candidate_pool=orbit_candidate_pool,
                    design_jobs=design_jobs,
                    design_start_method=design_start_method,
                ): (repeat, current_budget_index)
                for repeat, current_budget_index, inner_calls in group
            }
            for future in as_completed(futures):
                merge_cell(future.result())

    if len(report["raw_cells"]) != repeats * len(inner_budgets):
        raise RuntimeError("experiment ended with incomplete cells")
    results, relation = _aggregate(
        report["raw_cells"],
        inner_budgets=inner_budgets,
        repeats=repeats,
        bootstrap_samples=bootstrap_samples,
    )
    report["results_by_inner_budget"] = results
    report["geometry_error_relation"] = relation
    report["experiment_wall_seconds"] = float(
        prior_wall_seconds + time.perf_counter() - run_started
    )
    report["status"] = "complete"
    validate_report(report)
    if output_path is not None:
        _write_atomic(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-multipliers",
        nargs="+",
        type=int,
        default=DEFAULT_BUDGET_MULTIPLIERS,
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=20260827)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--repeat-processes", type=int, default=3)
    parser.add_argument("--design-jobs", type=int, default=40)
    parser.add_argument(
        "--design-start-method",
        choices=mp.get_all_start_methods(),
        default="spawn",
    )
    parser.add_argument("--greedy-candidate-pool", type=int, default=64)
    parser.add_argument("--orbit-candidate-pool", type=int, default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/json/airport_inside_ablations_k64_lambda1over16.json"),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_experiment(
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        base_seed=args.base_seed,
        bootstrap_samples=args.bootstrap_samples,
        repeat_processes=args.repeat_processes,
        design_jobs=args.design_jobs,
        design_start_method=args.design_start_method,
        greedy_candidate_pool=args.greedy_candidate_pool,
        orbit_candidate_pool=args.orbit_candidate_pool,
        output_path=args.output,
        resume=not args.no_resume,
    )
    print(
        f"saved {args.output} ({report['status']}, "
        f"{len(report['raw_cells'])} cells)",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Compare per-size Greedy + OFA ratio with global Greedy + linear.

This is a narrowly controlled extension of the completed second-moment-scope
ablation.  Its nine existing methods are copied verbatim (the per-size Greedy
method is renamed to make its linear estimator explicit).  Only one new
branch is evaluated:

``per-size frame design + official OFA conditional-mean ratio``.

For every repeat/budget cell the runner reconstructs the exact per-size
design used by the source report, checks the size-schedule and batch-wide
relabel hashes, re-evaluates the same utilities, and requires its linear
estimate to reproduce the source trace before accepting the ratio estimate.
``missing="raise"`` is used so no absent player/size conditional mean can be
silently replaced by zero.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
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
    _bootstrap_rmse_interval,
    _compact_json,
    _evaluate,
    _exact_truth,
    _game_metadata,
    _num_players,
    _seed_for,
    _sha256,
    _write_checkpoint,
    total_call_budgets,
)
from experiments.run_second_moment_scope_comparison import (
    EXPERIMENT_ID as SOURCE_EXPERIMENT_ID,
    METHOD_ORDER as SOURCE_METHOD_ORDER,
    _size_schedule_sha256,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    estimate_official_ratio_ofa,
    inner_size_distribution,
    per_size_frame_coupled_design,
)
from frame_ofa.geometry import centered_directions


METHOD_ORDER = (
    "inside_greedy_global",
    "inside_greedy_per_size_linear",
    "inside_greedy_per_size_ratio",
    "inside_orbit",
    "ofa_iid_linear",
    "ofa_iid_ratio",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)
METHOD_LABELS = {
    "inside_greedy_global": "INSIDE-Greedy (global, linear)",
    "inside_greedy_per_size_linear": (
        "INSIDE-Greedy (per-size, linear)"
    ),
    "inside_greedy_per_size_ratio": (
        "INSIDE-Greedy (per-size, OFA ratio)"
    ),
    "inside_orbit": "INSIDE-Orbit (OFA ratio)",
    "ofa_iid_linear": "OFA linear (IID)",
    "ofa_iid_ratio": "OFA ratio (IID)",
    "cc": "CC",
    "s_diff": "S-Diff",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}
SOURCE_TO_DESTINATION = {
    method: (
        "inside_greedy_per_size_linear"
        if method == "inside_greedy_per_size"
        else method
    )
    for method in SOURCE_METHOD_ORDER
}
COPIED_METHOD_ORDER = tuple(SOURCE_TO_DESTINATION.values())
RATIO_METHOD = "inside_greedy_per_size_ratio"
EXPERIMENT_ID = "analytic_inside_per_size_ratio_vs_global_linear"
DEFAULT_SOURCE_REPORTS = {
    "airport": Path(
        "results/json/airport_inside_global_vs_per_size_"
        "3repeats_50k_1m.json"
    ),
    "voting": Path(
        "results/json/voting_inside_global_vs_per_size_"
        "3repeats_25k_510k.json"
    ),
}


def _cell_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["repeat"]),
        str(row["method"]),
        int(row["budget_index"]),
    )


def _shared_greedy_seed(
    base_seed: int, repeat: int, budget_index: int
) -> int:
    """Use the same method-index-zero stream as the scope ablation."""
    return _seed_for(base_seed, repeat, budget_index, 0)


def _shared_relabel_seed(seed: int) -> int:
    """Reproduce :func:`paired_frame_scope_designs` relabel substream."""
    return int(
        np.random.SeedSequence([seed, 0x1A51DE]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _permutation_hash_from_seed(seed: int, num_players: int) -> str:
    permutation = np.random.default_rng(seed).permutation(num_players)
    encoded = np.ascontiguousarray(np.asarray(permutation, dtype="<i8"))
    return hashlib.sha256(encoded.tobytes(order="C")).hexdigest()


def _coverage_diagnostics(
    coalitions: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
    """Audit every conditional mean used by official OFA ratio."""
    num_players = coalitions.shape[1]
    expected_sizes, _, _ = inner_size_distribution(num_players)
    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    maximum_absolute_deviation = 0.0
    maximum_relative_deviation = 0.0
    maximum_first_moment_norm = 0.0
    absolute_deviations: list[float] = []
    slice_coefficients_of_variation: list[float] = []
    missing_inclusion_strata = 0
    missing_exclusion_strata = 0
    exact_balanced_sizes = 0
    count_by_size: list[dict[str, Any]] = []

    for size in expected_sizes:
        take = sizes == size
        rows = coalitions[take]
        count = len(rows)
        inclusion = rows.sum(axis=0, dtype=np.int64)
        exclusion = count - inclusion
        target = count * int(size) / num_players
        deviations = np.abs(inclusion - target)
        absolute_deviation = float(np.max(deviations))
        relative_deviation = absolute_deviation / max(target, 1.0)
        absolute_deviations.extend(deviations.astype(float).tolist())
        slice_coefficients_of_variation.append(
            float(np.std(inclusion, ddof=0) / max(target, 1.0))
        )
        missing_inclusion_strata += int(np.count_nonzero(inclusion == 0))
        missing_exclusion_strata += int(np.count_nonzero(exclusion == 0))
        first_moment_norm = (
            float(np.linalg.norm(centered_directions(rows).mean(axis=0)))
            if count
            else float("inf")
        )
        is_balanced = bool(
            count > 0 and np.all(inclusion == inclusion[0])
        )
        exact_balanced_sizes += int(is_balanced)
        minimum_inclusion = min(
            minimum_inclusion,
            int(inclusion.min(initial=np.iinfo(np.int64).max)),
        )
        minimum_exclusion = min(
            minimum_exclusion,
            int(exclusion.min(initial=np.iinfo(np.int64).max)),
        )
        maximum_absolute_deviation = max(
            maximum_absolute_deviation, absolute_deviation
        )
        maximum_relative_deviation = max(
            maximum_relative_deviation, relative_deviation
        )
        maximum_first_moment_norm = max(
            maximum_first_moment_norm, first_moment_norm
        )
        count_by_size.append(
            {
                "size": int(size),
                "coalitions": count,
                "minimum_inclusion_count": int(inclusion.min()),
                "maximum_inclusion_count": int(inclusion.max()),
                "minimum_exclusion_count": int(exclusion.min()),
                "maximum_exclusion_count": int(exclusion.max()),
                "exactly_1_balanced": is_balanced,
            }
        )

    all_covered = minimum_inclusion > 0 and minimum_exclusion > 0
    return {
        "all_player_size_strata_covered": bool(all_covered),
        "minimum_inclusion_count": int(minimum_inclusion),
        "minimum_exclusion_count": int(minimum_exclusion),
        "missing_inclusion_strata": missing_inclusion_strata,
        "missing_exclusion_strata": missing_exclusion_strata,
        "maximum_absolute_inclusion_count_deviation": (
            maximum_absolute_deviation
        ),
        "mean_absolute_inclusion_count_deviation": float(
            np.mean(absolute_deviations)
        ),
        "maximum_relative_inclusion_count_deviation": (
            maximum_relative_deviation
        ),
        "mean_slice_inclusion_count_coefficient_of_variation": float(
            np.mean(slice_coefficients_of_variation)
        ),
        "maximum_slice_inclusion_count_coefficient_of_variation": float(
            np.max(slice_coefficients_of_variation)
        ),
        "maximum_slice_first_moment_norm": maximum_first_moment_norm,
        "exactly_1_balanced_size_count": exact_balanced_sizes,
        "total_inner_size_count": len(expected_sizes),
        "counts_by_size": count_by_size,
    }


def _run_ratio_cell(
    dataset: str,
    repeat: int,
    budget_index: int,
    inner_calls: int,
    total_calls: int,
    base_seed: int,
    expected_linear_estimate: list[float],
    expected_schedule_hash: str,
    expected_relabel_hash: str,
) -> dict[str, Any]:
    """Reconstruct one source per-size design and add its ratio estimate."""
    seed = _shared_greedy_seed(base_seed, repeat, budget_index)
    relabel_seed = _shared_relabel_seed(seed)
    num_players = _num_players(dataset)
    cell_started = time.perf_counter()

    design_started = time.perf_counter()
    design = per_size_frame_coupled_design(
        num_players,
        inner_calls,
        seed=seed,
        candidate_pool=INSIDE_CANDIDATE_POOL,
        mean_balance=INSIDE_MEAN_BALANCE_LAMBDA0,
        mean_balance_mode=INSIDE_MEAN_BALANCE_MODE,
        relabel_seed=relabel_seed,
    )
    design_seconds = time.perf_counter() - design_started
    schedule_hash = _size_schedule_sha256(design.sizes)
    relabel_hash = str(
        design.diagnostics["relabel_permutation_sha256"]
    )
    if schedule_hash != expected_schedule_hash:
        raise RuntimeError("reconstructed per-size schedule hash disagrees")
    if relabel_hash != expected_relabel_hash:
        raise RuntimeError("reconstructed relabel hash disagrees")
    independently_derived_relabel_hash = _permutation_hash_from_seed(
        relabel_seed, num_players
    )
    if relabel_hash != independently_derived_relabel_hash:
        raise RuntimeError("relabel hash disagrees with its independent seed")

    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        _evaluate(dataset, boundary_rows), num_players
    )
    utilities = _evaluate(dataset, design.coalitions)
    linear = np.asarray(
        estimate_coupled(design, utilities, boundary, baseline="linear"),
        dtype=np.float64,
    )
    expected_linear = np.asarray(
        expected_linear_estimate, dtype=np.float64
    )
    if expected_linear.shape != (num_players,):
        raise RuntimeError("source per-size linear estimate has wrong shape")
    reconstruction_difference = float(
        np.max(np.abs(linear - expected_linear))
    )
    if not np.allclose(
        linear, expected_linear, rtol=0.0, atol=5e-14
    ):
        raise RuntimeError(
            "reconstructed per-size linear estimate disagrees with source: "
            f"max abs difference={reconstruction_difference:.3e}"
        )

    coverage = _coverage_diagnostics(
        design.coalitions, design.sizes
    )
    if not coverage["all_player_size_strata_covered"]:
        raise RuntimeError(
            "official ratio branch lacks an in/out player-size stratum"
        )
    values = np.asarray(
        estimate_official_ratio_ofa(
            design, utilities, boundary, missing="raise"
        ),
        dtype=np.float64,
    )
    if values.shape != (num_players,) or not np.all(np.isfinite(values)):
        raise RuntimeError("per-size ratio returned an invalid estimate")
    actual_calls = len(boundary_rows) + len(design.coalitions)
    if actual_calls != total_calls:
        raise RuntimeError("per-size ratio physical-call accounting mismatch")

    diagnostics = {
        "utility_evaluations": actual_calls,
        "boundary_utility_evaluations": len(boundary_rows),
        "inner_utility_evaluations": len(design.coalitions),
        "design_method": design.method,
        "second_moment_scope": "per_size",
        "estimator": "ofa_conditional_mean_ratio_missing_raise",
        "official_ratio_missing_policy": "raise",
        "candidate_pool": INSIDE_CANDIDATE_POOL,
        "mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
        "mean_balance_lambda0": INSIDE_MEAN_BALANCE_LAMBDA0,
        "mean_balance_effective_raw": float(
            design.diagnostics["mean_balance_effective_raw"]
        ),
        "shared_seed_stream": "historical inside_greedy method_index=0",
        "source_pair_design_reconstruction": True,
        "size_schedule_sha256": schedule_hash,
        "source_size_schedule_sha256": expected_schedule_hash,
        "relabel_seed": relabel_seed,
        "relabel_permutation_sha256": relabel_hash,
        "source_relabel_permutation_sha256": expected_relabel_hash,
        "independently_derived_relabel_permutation_sha256": (
            independently_derived_relabel_hash
        ),
        "linear_reconstruction_max_abs_difference": (
            reconstruction_difference
        ),
        "linear_reconstruction_tolerance": 5e-14,
        "coverage": coverage,
        "design_seconds": design_seconds,
        "design_diagnostics": _compact_json(design.diagnostics),
    }
    return {
        "repeat": repeat,
        "method": RATIO_METHOD,
        "budget_index": budget_index,
        "seed": seed,
        "inner_utility_call_budget": inner_calls,
        "target_total_utility_calls": total_calls,
        "actual_utility_calls": actual_calls,
        "estimate": values.tolist(),
        "elapsed_seconds": time.perf_counter() - cell_started,
        "diagnostics": diagnostics,
    }


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
    observed_budget = (
        int(row["inner_utility_call_budget"]),
        int(row["target_total_utility_calls"]),
    )
    expected_budget = (
        inner_budgets[budget_index],
        total_budgets[budget_index],
    )
    if observed_budget != expected_budget:
        raise ValueError("checkpoint cell budget disagrees with configuration")
    estimate = np.asarray(row.get("estimate", []), dtype=np.float64)
    if not np.all(np.isfinite(estimate)):
        raise ValueError("checkpoint contains a non-finite estimate")
    if method == RATIO_METHOD:
        expected_seed = _shared_greedy_seed(
            base_seed, repeat, budget_index
        )
        if int(row.get("seed", -1)) != expected_seed:
            raise ValueError("ratio checkpoint uses the wrong shared seed")
        diagnostics = row.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("ratio checkpoint lacks diagnostics")
        if diagnostics.get("second_moment_scope") != "per_size":
            raise ValueError("ratio checkpoint has the wrong scope")
        if diagnostics.get("estimator") != (
            "ofa_conditional_mean_ratio_missing_raise"
        ):
            raise ValueError("ratio checkpoint has the wrong estimator")
        coverage = diagnostics.get("coverage")
        if not isinstance(coverage, Mapping) or not coverage.get(
            "all_player_size_strata_covered"
        ):
            raise ValueError("ratio checkpoint lacks complete coverage")


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
    dict[tuple[int, int], dict[str, Any]],
    dict[str, Any],
]:
    """Copy the nine-method source and index its per-size linear cells."""
    source_path = path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"source report does not exist: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("status") != "complete":
        raise ValueError("source report must be complete")
    if source.get("experiment") != SOURCE_EXPERIMENT_ID:
        raise ValueError("source report is not the scope-ablation experiment")
    configuration = source.get("configuration")
    if not isinstance(configuration, Mapping):
        raise ValueError("source report lacks configuration")
    expected_configuration = {
        "dataset": dataset,
        "budget_multipliers": list(budget_multipliers),
        "inner_utility_call_budgets": list(inner_budgets),
        "total_call_budgets": list(total_budgets),
        "repeats": repeats,
        "base_seed": base_seed,
        "num_tasks_per_serial_baseline": num_tasks,
        "methods": list(SOURCE_METHOD_ORDER),
    }
    for key, expected in expected_configuration.items():
        if configuration.get(key) != expected:
            raise ValueError(f"source configuration.{key} is incompatible")
    source_truth = np.asarray(
        source.get("ground_truth", {}).get("values", []), dtype=np.float64
    )
    if source_truth.shape != truth.shape or not np.allclose(
        source_truth, truth, rtol=0.0, atol=1e-14
    ):
        raise ValueError("source ground truth disagrees with exact truth")

    raw_rows = source.get("raw_tasks")
    if not isinstance(raw_rows, list):
        raise ValueError("source report lacks raw task cells")
    expected_source_keys = {
        (repeat, method, budget_index)
        for repeat in range(repeats)
        for method in SOURCE_METHOD_ORDER
        for budget_index in range(len(inner_budgets))
    }
    seen_source: set[tuple[int, str, int]] = set()
    copied: list[dict[str, Any]] = []
    per_size_linear: dict[tuple[int, int], dict[str, Any]] = {}
    for source_row in raw_rows:
        source_method = str(source_row.get("method"))
        if source_method not in SOURCE_METHOD_ORDER:
            raise ValueError("source report contains an unknown method")
        repeat = int(source_row["repeat"])
        budget_index = int(source_row["budget_index"])
        source_key = (repeat, source_method, budget_index)
        if source_key in seen_source:
            raise ValueError("source report contains duplicate cells")
        seen_source.add(source_key)
        row = copy.deepcopy(dict(source_row))
        row["method"] = SOURCE_TO_DESTINATION[source_method]
        _validate_cell(
            row, inner_budgets, total_budgets, repeats, base_seed
        )
        copied.append(row)
        if source_method == "inside_greedy_per_size":
            diagnostics = row.get("diagnostics", {})
            if diagnostics.get("second_moment_scope") != "per_size":
                raise ValueError("source per-size row has the wrong scope")
            if diagnostics.get("estimator") != "coupled_linear":
                raise ValueError("source per-size row is not linear")
            schedule_hash = diagnostics.get("size_schedule_sha256")
            relabel_hash = diagnostics.get("relabel_permutation_sha256")
            if not schedule_hash or not relabel_hash:
                raise ValueError("source per-size row lacks design hashes")
            per_size_linear[(repeat, budget_index)] = row
    if seen_source != expected_source_keys:
        raise ValueError("source report does not contain every expected cell")
    if len(per_size_linear) != repeats * len(inner_budgets):
        raise ValueError("source report lacks per-size linear cells")

    provenance = {
        "path": str(source_path),
        "sha256": _sha256(source_path),
        "source_experiment": source.get("experiment"),
        "copied_methods": list(COPIED_METHOD_ORDER),
        "renamed_method": {
            "source": "inside_greedy_per_size",
            "destination": "inside_greedy_per_size_linear",
        },
        "recomputed_method": RATIO_METHOD,
        "copy_scope": "all nine source method-repeat-budget cells",
    }
    return copied, per_size_linear, provenance


def _aggregate(
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    grand_value = float(truth.sum())
    summaries: dict[str, Any] = {}
    for budget_index, (inner_calls, total_calls) in enumerate(
        zip(inner_budgets, total_budgets, strict=True)
    ):
        methods: dict[str, Any] = {}
        for method_index, method in enumerate(METHOD_ORDER):
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                    and int(row["budget_index"]) == budget_index
                ),
                key=lambda row: int(row["repeat"]),
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
                _seed_for(831247, 0, budget_index, method_index),
            )
            actual_calls = np.asarray(
                [row["actual_utility_calls"] for row in selected],
                dtype=np.float64,
            )
            methods[method] = {
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
            "methods": methods,
        }
    return summaries


def _validation_checks(
    rows: list[dict[str, Any]],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> dict[str, Any]:
    expected_keys = {
        (repeat, method, budget_index)
        for repeat in range(repeats)
        for method in METHOD_ORDER
        for budget_index in range(len(inner_budgets))
    }
    row_map = {_cell_key(row): row for row in rows}
    if set(row_map) != expected_keys or len(row_map) != len(rows):
        raise RuntimeError("final report has incomplete or duplicate cells")

    maximum_reconstruction_difference = 0.0
    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    for repeat in range(repeats):
        for budget_index, total_calls in enumerate(total_budgets):
            ratio = row_map[(repeat, RATIO_METHOD, budget_index)]
            linear = row_map[
                (repeat, "inside_greedy_per_size_linear", budget_index)
            ]
            global_row = row_map[
                (repeat, "inside_greedy_global", budget_index)
            ]
            ratio_diagnostics = ratio["diagnostics"]
            for paired in (linear, global_row):
                paired_diagnostics = paired["diagnostics"]
                if (
                    ratio_diagnostics["size_schedule_sha256"]
                    != paired_diagnostics["size_schedule_sha256"]
                ):
                    raise RuntimeError("paired methods use different schedules")
                if (
                    ratio_diagnostics["relabel_permutation_sha256"]
                    != paired_diagnostics["relabel_permutation_sha256"]
                ):
                    raise RuntimeError("paired methods use different relabels")
            if int(ratio["actual_utility_calls"]) != total_calls:
                raise RuntimeError("ratio cell does not consume target calls")
            difference = float(
                ratio_diagnostics[
                    "linear_reconstruction_max_abs_difference"
                ]
            )
            maximum_reconstruction_difference = max(
                maximum_reconstruction_difference, difference
            )
            coverage = ratio_diagnostics["coverage"]
            if not coverage["all_player_size_strata_covered"]:
                raise RuntimeError("ratio cell has incomplete coverage")
            minimum_inclusion = min(
                minimum_inclusion, int(coverage["minimum_inclusion_count"])
            )
            minimum_exclusion = min(
                minimum_exclusion, int(coverage["minimum_exclusion_count"])
            )
    if maximum_reconstruction_difference > 5e-14:
        raise RuntimeError("linear source reconstruction exceeded tolerance")
    return {
        "passed": True,
        "expected_method_repeat_budget_cells": len(expected_keys),
        "expected_budget_cells_per_method": len(inner_budgets),
        "ratio_design_matches_source_per_size_schedule_and_relabel": True,
        "ratio_design_matches_global_schedule_and_relabel": True,
        "ratio_all_player_size_strata_covered": True,
        "minimum_ratio_inclusion_count_across_all_cells": int(
            minimum_inclusion
        ),
        "minimum_ratio_exclusion_count_across_all_cells": int(
            minimum_exclusion
        ),
        "maximum_linear_reconstruction_absolute_difference": (
            maximum_reconstruction_difference
        ),
        "linear_reconstruction_tolerance": 5e-14,
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
        "processes": min(processes, repeats * len(inner_budgets)),
        "base_seed": base_seed,
        "bootstrap_samples": bootstrap_samples,
        "num_tasks_per_serial_baseline": num_tasks,
        "methods": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "comparison": {
            "primary_a": (
                "inside_greedy_per_size_ratio: per-size first/second "
                "moments + official OFA conditional-mean ratio"
            ),
            "primary_b": (
                "inside_greedy_global: per-size first moment + global "
                "radially weighted second moment + coupled linear estimator"
            ),
            "controlled_estimators": [
                "inside_greedy_per_size_linear",
                "inside_greedy_per_size_ratio",
            ],
            "controlled_geometry": [
                "inside_greedy_global",
                "inside_greedy_per_size_linear",
            ],
        },
        "inside_greedy": {
            "candidate_pool": INSIDE_CANDIDATE_POOL,
            "mean_balance_mode": INSIDE_MEAN_BALANCE_MODE,
            "mean_balance_lambda0": INSIDE_MEAN_BALANCE_LAMBDA0,
            "ratio_estimator": "official OFA conditional-mean ratio",
            "ratio_missing_policy": "raise",
            "ratio_balance_note": (
                "per-size Greedy need not be exactly 1-balanced; uniform "
                "batch-wide random relabeling preserves unbiasedness for "
                "fixed covered base-column counts"
            ),
            "shared_seed_rule": (
                "SeedSequence([base_seed, repeat, budget_index, 0])"
            ),
            "shared_relabel_rule": (
                "SeedSequence([shared_seed, 0x1A51DE]); independent of "
                "candidate draws"
            ),
        },
        "parallelism": "outer repeat-by-budget; up to 15 useful tasks",
        "checkpoint_granularity": "one ratio repeat-budget cell",
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
    copied_rows: list[dict[str, Any]],
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
        "raw_tasks": copy.deepcopy(copied_rows),
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
        raise ValueError("checkpoint belongs to another experiment")
    saved_configuration = copy.deepcopy(saved.get("configuration"))
    fresh_configuration = copy.deepcopy(fresh_report["configuration"])
    if not isinstance(saved_configuration, dict):
        raise ValueError("checkpoint configuration must be an object")
    saved_configuration.pop("processes", None)
    fresh_configuration.pop("processes", None)
    if saved_configuration != fresh_configuration:
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
        raise ValueError("checkpoint ground truth does not match")
    rows = saved.get("raw_tasks")
    if not isinstance(rows, list):
        raise ValueError("checkpoint raw_tasks must be a list")
    seen: set[tuple[int, str, int]] = set()
    for row in rows:
        _validate_cell(
            row, inner_budgets, total_budgets, repeats, base_seed
        )
        key = _cell_key(row)
        if key in seen:
            raise ValueError("checkpoint contains a duplicate cell")
        seen.add(key)
    expected_copied = {
        (repeat, method, budget_index)
        for repeat in range(repeats)
        for method in COPIED_METHOD_ORDER
        for budget_index in range(len(inner_budgets))
    }
    if not expected_copied.issubset(seen):
        raise ValueError("checkpoint lacks copied source cells")
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

    copied_rows, per_size_linear, provenance = _load_and_validate_source(
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
        copied_rows,
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
            raise ValueError("output path must differ from source report")
        _write_checkpoint(output_path, report)

    completed = {_cell_key(row) for row in report["raw_tasks"]}
    tasks: list[tuple[Any, ...]] = []
    for repeat in range(repeats):
        for budget_index, (inner_calls, total_calls) in enumerate(
            zip(inner_budgets, total_budgets, strict=True)
        ):
            key = (repeat, RATIO_METHOD, budget_index)
            if key in completed:
                continue
            source_row = per_size_linear[(repeat, budget_index)]
            diagnostics = source_row["diagnostics"]
            tasks.append(
                (
                    dataset,
                    repeat,
                    budget_index,
                    inner_calls,
                    total_calls,
                    base_seed,
                    source_row["estimate"],
                    diagnostics["size_schedule_sha256"],
                    diagnostics["relabel_permutation_sha256"],
                )
            )

    accumulated_wall = float(report.get("experiment_wall_seconds", 0.0))
    wall_started = time.perf_counter()

    def record(row: dict[str, Any]) -> None:
        nonlocal accumulated_wall, wall_started
        _validate_cell(
            row, inner_budgets, total_budgets, repeats, base_seed
        )
        key = _cell_key(row)
        if key in completed:
            raise RuntimeError("worker returned a duplicate ratio cell")
        report["raw_tasks"].append(row)
        report["raw_tasks"].sort(
            key=lambda item: (
                int(item["repeat"]),
                METHOD_ORDER.index(item["method"]),
                int(item["budget_index"]),
            )
        )
        completed.add(key)
        accumulated_wall += time.perf_counter() - wall_started
        report["experiment_wall_seconds"] = accumulated_wall
        report["status"] = "running"
        if output_path is not None:
            _write_checkpoint(output_path, report)
        wall_started = time.perf_counter()

    if processes == 1:
        for task in tasks:
            row = _run_ratio_cell(*task)
            record(row)
            print(
                f"completed ratio repeat={row['repeat'] + 1}, "
                f"budget={row['budget_index'] + 1}",
                flush=True,
            )
    elif tasks:
        context = mp.get_context("spawn")
        first_failure: BaseException | None = None
        with ProcessPoolExecutor(
            max_workers=min(processes, len(tasks)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_ratio_cell, *task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                try:
                    row = future.result()
                except BaseException as error:
                    if first_failure is None:
                        first_failure = error
                    print(
                        f"ratio task failure {index}/{len(tasks)}: "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                    continue
                record(row)
                print(
                    f"completed {index}/{len(tasks)} ratio tasks: "
                    f"repeat={row['repeat'] + 1}, "
                    f"budget={row['budget_index'] + 1}",
                    flush=True,
                )
        if first_failure is not None:
            raise RuntimeError(
                "one or more ratio tasks failed; completed tasks were "
                "checkpointed and the run can be resumed"
            ) from first_failure

    expected_cells = repeats * len(METHOD_ORDER) * len(inner_budgets)
    if len(report["raw_tasks"]) != expected_cells:
        raise RuntimeError("experiment ended with incomplete cells")
    report["results_by_inner_budget"] = _aggregate(
        report["raw_tasks"],
        truth,
        inner_budgets,
        total_budgets,
        repeats,
        bootstrap_samples,
    )
    report["validation"] = _validation_checks(
        report["raw_tasks"], inner_budgets, total_budgets, repeats
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
    parser.add_argument("--source-report", type=Path)
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
        "--no-resume", action="store_true", help="ignore an existing checkpoint"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    source = args.source_report or DEFAULT_SOURCE_REPORTS[args.dataset]
    output = args.output or Path(
        f"results/json/{args.dataset}_inside_per_size_ratio_vs_global_linear.json"
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

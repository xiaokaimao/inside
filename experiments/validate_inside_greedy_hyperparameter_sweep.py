"""Independent audit for INSIDE-Greedy hyperparameter-sweep reports.

The sweep runner stores convenient aggregates, but this validator deliberately
reconstructs the comparison from ``raw_cells``.  In particular it independently
checks exact game truth, physical-call budgets, registered seeds, Greedy size
schedules and relabels, strict ratio coverage, per-repeat RMSE, aggregate RMSE,
and every reported ratio.  Coverage failures remain visible and make a
configuration ineligible for selection; they are never silently dropped.

The pairing contract is intentionally asymmetric:

* all Greedy hyperparameters at one dataset/repeat/budget must share the exact
  same randomized-systematic size schedule and final player relabel;
* INSIDE-Orbit is a fixed independently randomized reference with the same
  physical utility-call budget.  Its complete-orbit allocation cannot and
  should not be claimed to share Greedy's row-wise size schedule or relabel.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.run_inside_greedy_hyperparameter_sweep import (
    EXPERIMENT_ID,
    ORBIT_CANDIDATE_POOL,
    REGISTERED_CANDIDATE_POOL,
    REGISTERED_LAMBDA0,
)
from experiments.validate_inside_comparison import (
    _exact_airport_truth,
    _exact_weighted_voting_truth,
)
from experiments.us_electoral_voting_game import (
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
)


ABSOLUTE_TOLERANCE = 5e-13
RATIO_TOLERANCE = 5e-13


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _integer(value: Any, *, path: str, minimum: int | None = None) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{path} must be at least {minimum}")
    return result


def _finite(value: Any, *, path: str, minimum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(math.isfinite(result), f"{path} must be finite")
    if minimum is not None:
        _require(result >= minimum, f"{path} must be at least {minimum}")
    return result


def _close(actual: Any, expected: float, *, path: str) -> float:
    result = _finite(actual, path=path)
    _require(
        math.isclose(
            result,
            float(expected),
            rel_tol=RATIO_TOLERANCE,
            abs_tol=ABSOLUTE_TOLERANCE,
        ),
        f"{path} disagrees with independent recomputation",
    )
    return result


def _configuration_id(lambda0: float, candidate_pool: int) -> str:
    return f"lambda0={lambda0:.12g}|K={candidate_pool}"


def _seed_for(
    base_seed: int, repeat: int, budget_index: int, method_index: int
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, repeat, budget_index, method_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _shared_relabel_seed(seed: int) -> int:
    return int(
        np.random.SeedSequence([seed, 0x1A51DE]).generate_state(
            1, dtype=np.uint32
        )[0]
    )


def _array_sha256(values: np.ndarray, *, dtype: str) -> str:
    encoded = np.ascontiguousarray(np.asarray(values, dtype=dtype))
    return sha256(encoded.tobytes(order="C")).hexdigest()


def _expected_size_schedule_hash(
    num_players: int, num_samples: int, seed: int
) -> str:
    """Independently replay only the public randomized-systematic schedule."""
    sizes = np.arange(2, num_players - 1, dtype=np.int64)
    weights = 1.0 / np.sqrt(sizes * (num_players - sizes))
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    offset = rng.random()
    points = (
        offset + np.arange(num_samples, dtype=np.float64) / num_samples
    ) % 1.0
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    sampled = sizes[np.searchsorted(cdf, points, side="right")]
    sampled = sampled[rng.permutation(num_samples)]
    return _array_sha256(sampled, dtype="<i8")


def _expected_relabel_hash(seed: int, num_players: int) -> tuple[int, str]:
    relabel_seed = _shared_relabel_seed(seed)
    permutation = np.random.default_rng(relabel_seed).permutation(num_players)
    return relabel_seed, _array_sha256(permutation, dtype="<i8")


def _dataset_truth(dataset: str) -> tuple[np.ndarray, float]:
    if dataset == "airport":
        return _exact_airport_truth(), 10.0
    if dataset == "voting":
        return (
            _exact_weighted_voting_truth(
                US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
            ),
            1.0,
        )
    raise ValueError("configuration.dataset must be 'airport' or 'voting'")


def _configurations(
    configuration: Mapping[str, Any],
) -> tuple[tuple[float, int], ...]:
    raw = configuration.get("evaluated_configurations")
    _require(isinstance(raw, list) and raw, "evaluated_configurations is empty")
    result: list[tuple[float, int]] = []
    for index, item in enumerate(raw):
        entry = _mapping(item, path=f"evaluated_configurations[{index}]")
        lambda0 = _finite(
            entry.get("lambda0"),
            path=f"evaluated_configurations[{index}].lambda0",
            minimum=0.0,
        )
        pool = _integer(
            entry.get("candidate_pool"),
            path=f"evaluated_configurations[{index}].candidate_pool",
            minimum=1,
        )
        result.append((lambda0, pool))
    _require(len(set(result)) == len(result), "evaluated configurations repeat")
    _require(
        (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL) in result,
        "registered default is absent from evaluated configurations",
    )
    return tuple(result)


def _requested_configurations(
    configuration: Mapping[str, Any],
) -> tuple[tuple[float, int], ...]:
    raw = configuration.get("requested_configurations")
    _require(isinstance(raw, list) and raw, "requested_configurations is empty")
    result: list[tuple[float, int]] = []
    for index, item in enumerate(raw):
        entry = _mapping(item, path=f"requested_configurations[{index}]")
        result.append(
            (
                _finite(
                    entry.get("lambda0"),
                    path=f"requested_configurations[{index}].lambda0",
                    minimum=0.0,
                ),
                _integer(
                    entry.get("candidate_pool"),
                    path=f"requested_configurations[{index}].candidate_pool",
                    minimum=1,
                ),
            )
        )
    _require(len(set(result)) == len(result), "requested configurations repeat")
    return tuple(result)


def _cell_key(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    method = cell.get("method")
    if method == "inside_orbit":
        return (
            method,
            _integer(cell.get("repeat"), path="raw_cell.repeat", minimum=0),
            _integer(
                cell.get("budget_index"),
                path="raw_cell.budget_index",
                minimum=0,
            ),
        )
    _require(method == "inside_greedy", "raw cell has an unknown method")
    return (
        method,
        _finite(cell.get("lambda0"), path="raw_cell.lambda0", minimum=0.0),
        _integer(
            cell.get("candidate_pool"),
            path="raw_cell.candidate_pool",
            minimum=1,
        ),
        _integer(cell.get("repeat"), path="raw_cell.repeat", minimum=0),
        _integer(
            cell.get("budget_index"),
            path="raw_cell.budget_index",
            minimum=0,
        ),
    )


def _aggregate(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = sum(cell["status"] != "ok" for cell in cells)
    successful = [cell for cell in cells if cell["status"] == "ok"]
    rmses = np.asarray([float(cell["rmse"]) for cell in successful])
    return {
        "registered_repeats": len(cells),
        "successful_repeats": len(successful),
        "coverage_failures": failures,
        "coverage_failure_rate": failures / len(cells),
        "aggregate_rmse": (
            float(np.sqrt(np.mean(np.square(rmses))))
            if failures == 0
            else None
        ),
        "mean_repeat_rmse": float(rmses.mean()) if len(rmses) else None,
        "rmse_by_repeat": [cell.get("rmse") for cell in cells],
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


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    _require(denominator >= 0.0, "RMSE denominator cannot be negative")
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return numerator / denominator


def _check_optional_number(actual: Any, expected: float | None, *, path: str) -> None:
    if expected is None:
        _require(actual is None, f"{path} must be null")
    else:
        _close(actual, expected, path=path)


def _check_aggregate(
    stored: Mapping[str, Any], expected: Mapping[str, Any], *, path: str
) -> None:
    for name in (
        "registered_repeats",
        "successful_repeats",
        "coverage_failures",
    ):
        _require(
            _integer(stored.get(name), path=f"{path}.{name}", minimum=0)
            == expected[name],
            f"{path}.{name} disagrees with raw cells",
        )
    _close(
        stored.get("coverage_failure_rate"),
        expected["coverage_failure_rate"],
        path=f"{path}.coverage_failure_rate",
    )
    for name in ("aggregate_rmse", "mean_repeat_rmse"):
        _check_optional_number(
            stored.get(name), expected[name], path=f"{path}.{name}"
        )
    _require(
        stored.get("rmse_by_repeat") == expected["rmse_by_repeat"],
        f"{path}.rmse_by_repeat disagrees with raw cells",
    )
    _close(
        stored.get("mean_design_seconds"),
        expected["mean_design_seconds"],
        path=f"{path}.mean_design_seconds",
    )
    _require(
        stored.get("actual_utility_calls_by_repeat")
        == expected["actual_utility_calls_by_repeat"],
        f"{path}.actual_utility_calls_by_repeat disagrees with raw cells",
    )


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one completed Airport or Voting sweep report."""
    _require(report.get("status") == "complete", "report is not complete")
    _require(report.get("experiment") == EXPERIMENT_ID, "wrong experiment id")
    configuration = _mapping(report.get("configuration"), path="configuration")
    dataset = str(configuration.get("dataset"))
    stage = str(configuration.get("stage"))
    _require(stage in {"screening", "validation"}, "invalid experiment stage")
    truth, efficiency_target = _dataset_truth(dataset)
    num_players = len(truth)

    ground_truth = _mapping(report.get("ground_truth"), path="ground_truth")
    stored_truth = np.asarray(ground_truth.get("values"), dtype=np.float64)
    _require(
        stored_truth.shape == truth.shape and np.all(np.isfinite(stored_truth)),
        "ground truth has wrong shape or non-finite values",
    )
    _require(
        np.allclose(stored_truth, truth, rtol=0.0, atol=ABSOLUTE_TOLERANCE),
        "ground truth disagrees with independent exact calculation",
    )
    _close(ground_truth.get("sum"), float(truth.sum()), path="ground_truth.sum")
    _close(
        ground_truth.get("efficiency_target"),
        efficiency_target,
        path="ground_truth.efficiency_target",
    )

    multipliers = tuple(
        _integer(value, path="configuration.budget_multipliers", minimum=1)
        for value in configuration.get("budget_multipliers", [])
    )
    _require(multipliers, "budget multiplier grid is empty")
    _require(
        all(left < right for left, right in zip(multipliers, multipliers[1:])),
        "budget multipliers are not strictly increasing",
    )
    expected_inner = tuple(num_players * value for value in multipliers)
    boundary_calls = 2 * num_players + 2
    expected_total = tuple(value + boundary_calls for value in expected_inner)
    _require(
        tuple(configuration.get("all_inner_utility_call_budgets", []))
        == expected_inner,
        "inner budgets disagree with n times multiplier",
    )
    _require(
        tuple(configuration.get("all_total_utility_call_budgets", []))
        == expected_total,
        "total budgets disagree with inner plus 2n+2",
    )
    _require(
        _integer(
            configuration.get("boundary_utility_calls"),
            path="configuration.boundary_utility_calls",
            minimum=1,
        )
        == boundary_calls,
        "boundary call count is wrong",
    )
    selected_indices = tuple(configuration.get("selected_budget_indices", []))
    _require(
        selected_indices
        and all(isinstance(value, int) for value in selected_indices)
        and tuple(sorted(set(selected_indices))) == selected_indices
        and all(0 <= value < len(multipliers) for value in selected_indices),
        "selected budget indices are invalid",
    )
    _require(
        tuple(configuration.get("selected_inner_utility_call_budgets", []))
        == tuple(expected_inner[index] for index in selected_indices),
        "selected inner budgets disagree with selected indices",
    )
    _require(
        tuple(configuration.get("selected_total_utility_call_budgets", []))
        == tuple(expected_total[index] for index in selected_indices),
        "selected total budgets disagree with selected indices",
    )

    repeats = _integer(configuration.get("repeats"), path="repeats", minimum=1)
    base_seed = _integer(configuration.get("base_seed"), path="base_seed", minimum=0)
    configs = _configurations(configuration)
    requested_configs = _requested_configurations(configuration)
    _require(
        set(configs)
        == set(requested_configs)
        | {(REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)},
        "evaluated configurations disagree with predeclared requests plus default",
    )
    _require(
        configuration.get("inside_greedy_design")
        == "per_size_frame_coupled_design",
        "Greedy design is not the formal per-size design",
    )
    _require(
        configuration.get("inside_greedy_estimator")
        == "ofa_conditional_mean_ratio_missing_raise",
        "Greedy does not use strict covered OFA ratio",
    )
    _require(
        configuration.get("inside_orbit_design")
        == "cyclic_orbit_frame_design",
        "Orbit design is not cyclic-orbit frame",
    )
    _require(
        configuration.get("inside_orbit_estimator")
        == "ofa_conditional_mean_ratio_strict_balanced",
        "Orbit does not use strict balanced OFA ratio",
    )

    raw = report.get("raw_cells")
    _require(isinstance(raw, list), "raw_cells must be a list")
    keys = [_cell_key(_mapping(cell, path="raw_cell")) for cell in raw]
    _require(len(keys) == len(set(keys)), "raw cells contain duplicates")
    expected_keys: set[tuple[Any, ...]] = set()
    for budget_index in selected_indices:
        for repeat in range(repeats):
            expected_keys.add(("inside_orbit", repeat, budget_index))
            for lambda0, pool in configs:
                expected_keys.add(
                    ("inside_greedy", lambda0, pool, repeat, budget_index)
                )
    _require(set(keys) == expected_keys, "raw cells are incomplete or unexpected")

    cells_by_key = {key: cell for key, cell in zip(keys, raw, strict=True)}
    schedule_hashes: dict[tuple[int, int], set[str]] = {}
    relabel_hashes: dict[tuple[int, int], set[str]] = {}
    expected_schedule_hashes: dict[tuple[int, int], str] = {}
    expected_relabels: dict[tuple[int, int], tuple[int, str]] = {}
    coverage_failures = 0
    for key, raw_cell in cells_by_key.items():
        cell = _mapping(raw_cell, path=f"raw_cells[{key}]")
        method = str(cell.get("method"))
        repeat = int(cell["repeat"])
        budget_index = int(cell["budget_index"])
        inner_calls = expected_inner[budget_index]
        total_calls = expected_total[budget_index]
        _require(cell.get("dataset") == dataset, "raw cell dataset is wrong")
        _require(int(cell.get("base_seed")) == base_seed, "raw cell base seed is wrong")
        _require(int(cell.get("inner_calls")) == inner_calls, "raw inner budget is wrong")
        _require(int(cell.get("total_calls")) == total_calls, "raw total budget is wrong")
        method_index = 0 if method == "inside_greedy" else 1
        seed = _seed_for(base_seed, repeat, budget_index, method_index)
        _require(int(cell.get("seed")) == seed, "raw cell seed is wrong")
        diagnostics = _mapping(cell.get("diagnostics"), path="cell.diagnostics")
        coverage = _mapping(diagnostics.get("coverage"), path="cell.coverage")
        status = str(cell.get("status"))

        if method == "inside_greedy":
            lambda0, pool = float(cell["lambda0"]), int(cell["candidate_pool"])
            _require(
                diagnostics.get("design_method") == "frame_coupled_per_size",
                "Greedy cell has the wrong design method",
            )
            _require(
                diagnostics.get("estimator")
                == "ofa_conditional_mean_ratio_missing_raise",
                "Greedy cell has the wrong estimator",
            )
            _require(
                diagnostics.get("second_moment_scope") == "per_size",
                "Greedy second moment is not per-size",
            )
            _require(
                diagnostics.get("mean_balance_mode") == "normalized",
                "Greedy lambda is not normalized",
            )
            _close(
                diagnostics.get("mean_balance_lambda0"),
                lambda0,
                path="cell.mean_balance_lambda0",
            )
            _require(
                int(diagnostics.get("candidate_pool")) == pool,
                "Greedy candidate pool is wrong",
            )
            group = (repeat, budget_index)
            if group not in expected_schedule_hashes:
                expected_schedule_hashes[group] = _expected_size_schedule_hash(
                    num_players, inner_calls, seed
                )
            expected_schedule_hash = expected_schedule_hashes[group]
            observed_schedule_hash = str(
                diagnostics.get("size_schedule_sha256")
            )
            _require(
                observed_schedule_hash == expected_schedule_hash,
                "Greedy size schedule hash is wrong",
            )
            if group not in expected_relabels:
                expected_relabels[group] = _expected_relabel_hash(
                    seed, num_players
                )
            expected_relabel_seed, expected_relabel_hash = expected_relabels[group]
            _require(
                int(diagnostics.get("relabel_seed")) == expected_relabel_seed,
                "Greedy relabel seed is wrong",
            )
            observed_relabel_hash = str(
                diagnostics.get("relabel_permutation_sha256")
            )
            _require(
                observed_relabel_hash == expected_relabel_hash,
                "Greedy relabel hash is wrong",
            )
            schedule_hashes.setdefault(group, set()).add(observed_schedule_hash)
            relabel_hashes.setdefault(group, set()).add(observed_relabel_hash)
        else:
            _require(
                diagnostics.get("design_method") == "cyclic_orbit_frame",
                "Orbit cell has the wrong design method",
            )
            _require(
                diagnostics.get("estimator")
                == "ofa_conditional_mean_ratio_strict_balanced",
                "Orbit cell has the wrong estimator",
            )
            _require(
                int(diagnostics.get("candidate_pool"))
                == ORBIT_CANDIDATE_POOL,
                "Orbit candidate pool is wrong",
            )
            _require(
                int(coverage.get("exactly_1_balanced_sizes", -1))
                == num_players - 3,
                "Orbit is not exactly 1-balanced at every inner size",
            )

        covered = bool(coverage.get("all_player_size_strata_covered"))
        if status == "coverage_failure":
            coverage_failures += 1
            _require(method == "inside_greedy", "Orbit coverage failure is invalid")
            _require(not covered, "coverage failure cell claims complete coverage")
            _require(cell.get("coverage_failure") is True, "failure flag is absent")
            _require(cell.get("estimate") is None and cell.get("rmse") is None,
                     "coverage failure must not contain an estimate")
            _require(int(cell.get("actual_utility_calls")) == 0,
                     "coverage failure must occur before utility calls")
            continue

        _require(status == "ok", "raw cell has an invalid status")
        _require(covered, "successful cell lacks strict ratio coverage")
        _require(
            int(coverage.get("missing_inner_sizes", -1)) == 0
            and
            int(coverage.get("missing_inclusion_strata", -1)) == 0
            and int(coverage.get("missing_exclusion_strata", -1)) == 0
            and int(coverage.get("minimum_inclusion_count", 0)) > 0
            and int(coverage.get("minimum_exclusion_count", 0)) > 0,
            "successful ratio cell has an uncovered in/out stratum",
        )
        _require(cell.get("coverage_failure") is False, "success has failure flag")
        _require(
            int(cell.get("planned_total_utility_calls")) == total_calls
            and int(cell.get("actual_utility_calls")) == total_calls,
            "successful cell violates physical-call parity",
        )
        _require(
            int(diagnostics.get("boundary_utility_evaluations")) == boundary_calls
            and int(diagnostics.get("inner_utility_evaluations")) == inner_calls
            and int(diagnostics.get("utility_evaluations")) == total_calls,
            "successful cell diagnostics have wrong call accounting",
        )
        estimate = np.asarray(cell.get("estimate"), dtype=np.float64)
        _require(
            estimate.shape == truth.shape and np.all(np.isfinite(estimate)),
            "successful cell estimate is invalid",
        )
        rmse = float(np.sqrt(np.mean(np.square(estimate - truth))))
        _close(cell.get("rmse"), rmse, path="raw_cell.rmse")

    _require(
        all(len(values) == 1 for values in schedule_hashes.values()),
        "Greedy hyperparameters do not share size schedules",
    )
    _require(
        all(len(values) == 1 for values in relabel_hashes.values()),
        "Greedy hyperparameters do not share player relabels",
    )

    results = _mapping(
        report.get("results_by_configuration"),
        path="results_by_configuration",
    )
    _require(
        set(results) == {_configuration_id(*item) for item in configs},
        "results_by_configuration keys disagree with evaluated grid",
    )
    recomputed: dict[str, Any] = {}
    for lambda0, pool in configs:
        config_id = _configuration_id(lambda0, pool)
        stored_config = _mapping(results[config_id], path=f"results.{config_id}")
        stored_budgets = _mapping(
            stored_config.get("budgets"), path=f"results.{config_id}.budgets"
        )
        _require(
            set(stored_budgets) == {str(index) for index in selected_indices},
            "stored configuration has wrong budget keys",
        )
        budget_summaries: dict[str, Any] = {}
        default_ratios: list[float] = []
        orbit_ratios: list[float] = []
        failures = 0
        design_seconds: list[float] = []
        for budget_index in selected_indices:
            candidate_cells = [
                cells_by_key[
                    ("inside_greedy", lambda0, pool, repeat, budget_index)
                ]
                for repeat in range(repeats)
            ]
            default_cells = [
                cells_by_key[
                    (
                        "inside_greedy",
                        REGISTERED_LAMBDA0,
                        REGISTERED_CANDIDATE_POOL,
                        repeat,
                        budget_index,
                    )
                ]
                for repeat in range(repeats)
            ]
            orbit_cells = [
                cells_by_key[("inside_orbit", repeat, budget_index)]
                for repeat in range(repeats)
            ]
            summary = _aggregate(candidate_cells)
            default_summary = _aggregate(default_cells)
            orbit_summary = _aggregate(orbit_cells)
            ratio_default = _ratio(
                summary["aggregate_rmse"], default_summary["aggregate_rmse"]
            )
            ratio_orbit = _ratio(
                summary["aggregate_rmse"], orbit_summary["aggregate_rmse"]
            )
            stored = _mapping(
                stored_budgets[str(budget_index)],
                path=f"results.{config_id}.budgets.{budget_index}",
            )
            _check_aggregate(
                stored,
                summary,
                path=f"results.{config_id}.budgets.{budget_index}",
            )
            _check_optional_number(
                stored.get("aggregate_rmse_ratio_to_registered_default"),
                ratio_default,
                path=(
                    f"results.{config_id}.budgets.{budget_index}."
                    "aggregate_rmse_ratio_to_registered_default"
                ),
            )
            _check_optional_number(
                stored.get("aggregate_rmse_ratio_to_inside_orbit"),
                ratio_orbit,
                path=(
                    f"results.{config_id}.budgets.{budget_index}."
                    "aggregate_rmse_ratio_to_inside_orbit"
                ),
            )
            budget_summaries[str(budget_index)] = {
                **summary,
                "aggregate_rmse_ratio_to_registered_default": ratio_default,
                "aggregate_rmse_ratio_to_inside_orbit": ratio_orbit,
                "default_aggregate_rmse": default_summary["aggregate_rmse"],
                "orbit_aggregate_rmse": orbit_summary["aggregate_rmse"],
                # Preserve the independently recomputed repeat-level values so
                # downstream uncertainty intervals can resample the repeat as
                # the experimental unit.  These are deliberately taken from
                # raw cells rather than copied from the runner's aggregates.
                "default_rmse_by_repeat": default_summary["rmse_by_repeat"],
                "orbit_rmse_by_repeat": orbit_summary["rmse_by_repeat"],
            }
            failures += int(summary["coverage_failures"])
            design_seconds.extend(summary["design_seconds_by_repeat"])
            if ratio_default is not None:
                default_ratios.append(ratio_default)
            if ratio_orbit is not None:
                orbit_ratios.append(ratio_orbit)
        eligible = failures == 0
        overall = {
            "coverage_failures": failures,
            "selection_eligible": eligible,
            "geometric_mean_rmse_ratio_to_registered_default": (
                float(np.exp(np.mean(np.log(default_ratios))))
                if eligible
                and len(default_ratios) == len(selected_indices)
                and all(value > 0.0 for value in default_ratios)
                else None
            ),
            "geometric_mean_rmse_ratio_to_inside_orbit": (
                float(np.exp(np.mean(np.log(orbit_ratios))))
                if eligible
                and len(orbit_ratios) == len(selected_indices)
                and all(value > 0.0 for value in orbit_ratios)
                else None
            ),
            "worst_budget_rmse_ratio_to_registered_default": (
                float(max(default_ratios)) if eligible else None
            ),
            "worst_budget_rmse_ratio_to_inside_orbit": (
                float(max(orbit_ratios)) if eligible else None
            ),
            "mean_design_seconds": float(np.mean(design_seconds)),
        }
        stored_overall = _mapping(
            stored_config.get("overall"), path=f"results.{config_id}.overall"
        )
        _require(
            int(stored_overall.get("coverage_failures")) == failures,
            "stored overall coverage failures are wrong",
        )
        _check_optional_number(
            stored_overall.get(
                "geometric_mean_rmse_ratio_to_registered_default"
            ),
            overall["geometric_mean_rmse_ratio_to_registered_default"],
            path=f"results.{config_id}.overall.ratio_to_default",
        )
        _check_optional_number(
            stored_overall.get("geometric_mean_rmse_ratio_to_inside_orbit"),
            overall["geometric_mean_rmse_ratio_to_inside_orbit"],
            path=f"results.{config_id}.overall.ratio_to_orbit",
        )
        recomputed[config_id] = {
            "lambda0": lambda0,
            "candidate_pool": pool,
            "budgets": budget_summaries,
            "overall": overall,
        }

    embedded = _mapping(report.get("validation"), path="validation")
    _require(embedded.get("passed") is True, "embedded runner validation failed")
    # Accept both equivalent registered representations: the full five-point
    # grid with indices 0 and 2 selected, or a compact two-point grid containing
    # only multipliers 500 and 2000.
    selected_multipliers = tuple(multipliers[index] for index in selected_indices)
    screening_selection_eligible = (
        stage == "screening" and selected_multipliers == (500, 2000)
    )
    # Seed separation is necessarily a two-report property.  A standalone
    # validation report is stage-eligible here; the selector subsequently
    # derives every screening/validation method seed and rejects any overlap.
    holdout_stage_eligible = stage == "validation"
    return {
        "status": "ready_to_select" if screening_selection_eligible else "validated",
        "passed": True,
        "dataset": dataset,
        "stage": stage,
        "num_players": num_players,
        "repeats": repeats,
        "base_seed": base_seed,
        "budget_multipliers": list(multipliers),
        "selected_budget_indices": list(selected_indices),
        "selected_budget_multipliers": list(selected_multipliers),
        "evaluated_configurations": [
            {"lambda0": value, "candidate_pool": pool}
            for value, pool in configs
        ],
        "requested_configurations": [
            {"lambda0": value, "candidate_pool": pool}
            for value, pool in requested_configs
        ],
        "coverage_failures": coverage_failures,
        "all_greedy_hyperparameters_share_size_schedules": True,
        "all_greedy_hyperparameters_share_relabels": True,
        "orbit_is_equal-call_but_independently_randomized": True,
        "all_rmse_and_aggregates_independently_recomputed": True,
        "screening_selection_eligible": screening_selection_eligible,
        "holdout_stage_eligible_pending_cross_report_seed_audit": (
            holdout_stage_eligible
        ),
        "recomputed_results_by_configuration": recomputed,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validation = validate_report(report)
    encoded = json.dumps(validation, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()

"""Independently audit the common data-valuation/analytic INSIDE reports.

This validator treats the retained 3 x n estimates as the audit source.  It
recomputes every stored accuracy/ranking/efficiency statistic that is present,
checks physical utility-call accounting, and reconstructs the analytic truth
for Airport and weighted voting without calling either production truth
routine.  Wine and Breast Cancer deliberately use high-budget Monte Carlo
references; for them we validate retained standard-error and call-budget
metadata instead of misrepresenting the references as exact.
"""

from __future__ import annotations

import argparse
from math import comb
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from experiments.airport_game import AIRPORT_CLASS_COUNTS, AIRPORT_COSTS
from experiments.plot_inside_comparison import METHOD_ORDER
from experiments.us_electoral_voting_game import (
    US_ELECTORAL_NAMES,
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
)


EXPECTED_REPEATS = 3
EXPECTED_BUDGETS = 5
LEGACY_GREEDY_CANDIDATE_POOL = 4
CURRENT_GREEDY_CANDIDATE_POOL = 64
EXPECTED_ORBIT_CANDIDATE_POOL = 4
LEGACY_NORMALIZED_LAMBDA0 = 1.0
CURRENT_NORMALIZED_LAMBDA0 = 1.0 / 16.0
HISTORICAL_K64_NORMALIZED_LAMBDA0 = 0.0
EXPECTED_LEGACY_RAW_MEAN_BALANCE = 0.1
FULL_BUDGET_METHODS = frozenset(METHOD_ORDER) - {"cc", "tmc_shapley"}
LEGACY_CANONICAL_INSIDE_EXPERIMENT_IDS = frozenset(
    {
        "analytic_inside_baseline_comparison_per_size_ratio",
        "wine_inside_greedy_orbit_baseline_comparison_per_size_ratio",
    }
)
HISTORICAL_K64_CANONICAL_INSIDE_EXPERIMENT_IDS = frozenset(
    {
        "analytic_inside_baseline_comparison_per_size_ratio_k64_lambda0",
        "wine_inside_greedy_orbit_baseline_comparison_per_size_ratio_k64_lambda0",
    }
)
CURRENT_K64_CANONICAL_INSIDE_EXPERIMENT_IDS = frozenset(
    {
        "analytic_inside_baseline_comparison_per_size_ratio_k64_lambda1over16",
        "wine_inside_greedy_orbit_baseline_comparison_per_size_ratio_k64_lambda1over16",
        "cancer_inside_greedy_orbit_baseline_comparison_per_size_ratio_k64_lambda1over16",
    }
)
# Both explicit K=64 protocols retain full hyperparameter diagnostics.  Keep
# this broader compatibility name because downstream audit code historically
# imported it when only the lambda0=0 protocol existed.
CURRENT_CANONICAL_INSIDE_EXPERIMENT_IDS = frozenset(
    HISTORICAL_K64_CANONICAL_INSIDE_EXPERIMENT_IDS
    | CURRENT_K64_CANONICAL_INSIDE_EXPERIMENT_IDS
)
CANONICAL_INSIDE_EXPERIMENT_IDS = frozenset(
    LEGACY_CANONICAL_INSIDE_EXPERIMENT_IDS
    | CURRENT_CANONICAL_INSIDE_EXPERIMENT_IDS
)
CURRENT_ANALYTIC_EXPERIMENT_ID = (
    "analytic_inside_baseline_comparison_per_size_ratio_k64_lambda1over16"
)
CURRENT_ANALYTIC_PROTOCOL_VERSION = "per_size_ratio_k64_lambda1over16"
ANALYTIC_PROTOCOL_VERSIONS = {
    "analytic_inside_baseline_comparison_per_size_ratio_k64_lambda0": (
        "per_size_ratio_k64_lambda0"
    ),
    CURRENT_ANALYTIC_EXPERIMENT_ID: CURRENT_ANALYTIC_PROTOCOL_VERSION,
}
MC_REFERENCE_DATASETS = frozenset({"wine", "cancer"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _finite(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(math.isfinite(result), f"{path} must be finite")
    return result


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, (int, np.integer)) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    result = int(value)
    _require(result >= minimum, f"{path} must be at least {minimum}")
    return result


def _close(
    stored: Any,
    expected: float,
    *,
    path: str,
    atol: float = 2e-13,
    rtol: float = 3e-11,
) -> float:
    value = _finite(stored, path=path)
    discrepancy = abs(value - float(expected))
    _require(
        math.isclose(value, float(expected), rel_tol=rtol, abs_tol=atol),
        f"{path} differs: stored={value:.17g}, recomputed={expected:.17g}",
    )
    return discrepancy


def _configured_methods(configuration: Mapping[str, Any]) -> tuple[str, ...]:
    raw = configuration.get("methods", configuration.get("method_order", ()))
    _require(isinstance(raw, (list, tuple)), "configuration methods must be a list")
    return tuple(str(value) for value in raw)


def _inside_balance_configuration(
    configuration: Mapping[str, Any],
) -> tuple[Any, str, Any]:
    """Return ``(candidate_pool, mode, coefficient)`` for either schema.

    New reports store an explicit dimensionless ``lambda0`` and normalized
    mode.  The legacy reports did not record a mode; preserving support for
    them lets the before/after comparison be audited without relabeling old
    raw ``mean_balance=0.1`` results as normalized.
    """
    inside = configuration.get("inside")
    inside_mapping = inside if isinstance(inside, Mapping) else configuration
    greedy = inside_mapping.get("greedy")
    greedy_mapping = greedy if isinstance(greedy, Mapping) else inside_mapping
    candidate_pool = greedy_mapping.get(
        "candidate_pool",
        inside_mapping.get("candidate_pool", configuration.get("candidate_pool")),
    )
    raw_mode = greedy_mapping.get(
        "mean_balance_mode",
        greedy_mapping.get(
            "greedy_mean_balance_mode", configuration.get("mean_balance_mode")
        ),
    )
    if raw_mode is None:
        coefficient = greedy_mapping.get(
            "greedy_mean_balance",
            greedy_mapping.get("mean_balance", configuration.get("mean_balance")),
        )
        return candidate_pool, "legacy_raw", coefficient
    mode = str(raw_mode)
    coefficient = greedy_mapping.get(
        "mean_balance_lambda0",
        greedy_mapping.get(
            "greedy_mean_balance_lambda0",
            configuration.get("mean_balance_lambda0"),
        ),
    )
    return candidate_pool, mode, coefficient


def _inside_orbit_candidate_pool(configuration: Mapping[str, Any]) -> Any:
    """Return an explicitly recorded Orbit pool without borrowing Greedy's.

    Historical flat reports used one shared ``candidate_pool`` field.  New
    nested reports deliberately split the Greedy and Orbit values; in that
    schema an absent Orbit value stays absent instead of falling back to the
    top-level Greedy value.
    """
    inside = configuration.get("inside")
    inside_mapping = inside if isinstance(inside, Mapping) else configuration
    nested = isinstance(inside_mapping.get("greedy"), Mapping) or isinstance(
        inside_mapping.get("orbit"), Mapping
    )
    if nested:
        orbit = inside_mapping.get("orbit")
        orbit_mapping = orbit if isinstance(orbit, Mapping) else {}
        return orbit_mapping.get(
            "candidate_pool", inside_mapping.get("orbit_candidate_pool")
        )
    return inside_mapping.get(
        "orbit_candidate_pool",
        inside_mapping.get("candidate_pool", configuration.get("candidate_pool")),
    )


def _canonical_protocol(experiment_id: str) -> tuple[int, float]:
    """Return the Greedy ``(K, lambda0)`` required by a canonical ID."""
    if experiment_id in CURRENT_K64_CANONICAL_INSIDE_EXPERIMENT_IDS:
        return CURRENT_GREEDY_CANDIDATE_POOL, CURRENT_NORMALIZED_LAMBDA0
    if experiment_id in HISTORICAL_K64_CANONICAL_INSIDE_EXPERIMENT_IDS:
        return (
            CURRENT_GREEDY_CANDIDATE_POOL,
            HISTORICAL_K64_NORMALIZED_LAMBDA0,
        )
    if experiment_id in LEGACY_CANONICAL_INSIDE_EXPERIMENT_IDS:
        return LEGACY_GREEDY_CANDIDATE_POOL, LEGACY_NORMALIZED_LAMBDA0
    raise ValueError(f"unknown canonical INSIDE experiment id: {experiment_id}")


def _dataset_kind(report: Mapping[str, Any]) -> str:
    configuration = _mapping(report.get("configuration"), path="configuration")
    candidate: Any = configuration.get("dataset")
    dataset = report.get("dataset")
    game = report.get("game")
    if isinstance(dataset, Mapping):
        candidate = dataset.get("dataset", candidate)
    elif isinstance(dataset, str):
        candidate = dataset
    if isinstance(game, Mapping):
        candidate = game.get("dataset", candidate)
        if candidate is None:
            model = str(game.get("model", "")).lower()
            if "airport" in model:
                candidate = "airport"
            elif "electoral" in model or "voting" in model:
                candidate = "voting"
    normalized = str(candidate or "").strip().lower()
    if normalized == "wine":
        return "wine"
    if normalized in {"cancer", "breast_cancer", "breast cancer"}:
        return "cancer"
    if "airport" in normalized:
        return "airport"
    if "vot" in normalized or "electoral" in normalized:
        return "voting"
    raise ValueError(f"unsupported or missing dataset identifier: {candidate!r}")


def _num_players(report: Mapping[str, Any], truth: np.ndarray) -> int:
    configuration = _mapping(report.get("configuration"), path="configuration")
    game = report.get("game")
    raw = configuration.get("num_players")
    if raw is None and isinstance(game, Mapping):
        raw = game.get("players")
    if raw is None:
        raw = len(truth)
    result = _integer(raw, path="num_players", minimum=2)
    _require(result == len(truth), "player count and ground-truth length disagree")
    return result


def _exact_airport_truth() -> np.ndarray:
    """Integrate the nested threshold games directly from their definition."""
    values = np.asarray(AIRPORT_COSTS, dtype=np.float64)
    result = np.zeros_like(values)
    previous = 0.0
    for level in np.unique(values[values > 0.0]):
        eligible = values >= level
        result[eligible] += (float(level) - previous) / int(eligible.sum())
        previous = float(level)
    return result


def _exact_weighted_voting_truth(
    weights: np.ndarray, quota: int
) -> np.ndarray:
    """Count pivotal predecessor coalitions by size and total weight.

    For player i and predecessor size k, each pivotal subset has probability
    ``1 / (n * C(n-1, k))`` in a uniformly random permutation.  Dynamic
    programming counts those subsets exactly using Python integers.
    """
    weights = np.asarray(weights, dtype=np.int64)
    n = len(weights)
    total_weight = int(weights.sum())
    result = np.zeros(n, dtype=np.float64)
    for excluded in range(n):
        dp = np.zeros((n, total_weight + 1), dtype=object)
        dp[0, 0] = 1
        players = 0
        running_sum = 0
        for index, raw_weight in enumerate(weights):
            if index == excluded:
                continue
            weight = int(raw_weight)
            for size in range(players, -1, -1):
                dp[size + 1, weight : weight + running_sum + 1] += dp[
                    size, : running_sum + 1
                ]
            players += 1
            running_sum += weight
        lower = max(0, quota - int(weights[excluded]))
        upper = min(quota - 1, running_sum)
        if lower <= upper:
            for size in range(n):
                pivotal = sum(int(value) for value in dp[size, lower : upper + 1])
                if pivotal:
                    result[excluded] += pivotal / (n * comb(n - 1, size))
    return result


def _validate_game_and_truth(
    report: Mapping[str, Any], dataset: str
) -> tuple[np.ndarray, int, float, dict[str, Any]]:
    ground_truth = _mapping(report.get("ground_truth"), path="ground_truth")
    truth = np.asarray(ground_truth.get("values"), dtype=np.float64)
    _require(truth.ndim == 1 and truth.size >= 2, "ground truth has wrong shape")
    _require(np.all(np.isfinite(truth)), "ground truth contains non-finite values")
    n = _num_players(report, truth)
    boundary = report.get("boundary")
    boundary_mapping = boundary if isinstance(boundary, Mapping) else {}

    if dataset == "airport":
        _require(n == 100, "the supplied airport game must have 100 players")
        game = _mapping(report.get("game"), path="game")
        counts = game.get("cost_class_counts", game.get("class_counts"))
        if counts is not None:
            _require(
                tuple(int(value) for value in counts) == AIRPORT_CLASS_COUNTS,
                "airport class counts disagree with the supplied game",
            )
        expected = _exact_airport_truth()
        target = 10.0
        kind = "analytic_exact"
    elif dataset == "voting":
        _require(n == len(US_ELECTORAL_WEIGHTS), "voting game must have 51 players")
        game = _mapping(report.get("game"), path="game")
        if "weights" in game:
            _require(
                np.array_equal(np.asarray(game["weights"]), US_ELECTORAL_WEIGHTS),
                "voting weights disagree with the supplied game",
            )
        if "quota" in game:
            _require(int(game["quota"]) == US_ELECTORAL_QUOTA, "voting quota is wrong")
        names = game.get("names", game.get("player_names"))
        if names is not None:
            _require(tuple(names) == US_ELECTORAL_NAMES, "voting names are wrong")
        expected = _exact_weighted_voting_truth(
            US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        )
        target = 1.0
        kind = "analytic_exact"
    elif dataset in MC_REFERENCE_DATASETS:
        expected = truth
        target = _finite(
            boundary_mapping.get("efficiency_target", truth.sum()),
            path="boundary.efficiency_target",
        )
        kind = "monte_carlo_with_standard_error"
        standard_errors = np.asarray(
            ground_truth.get("standard_errors"), dtype=np.float64
        )
        _require(
            standard_errors.shape == (n,),
            "Monte Carlo ground-truth SE has wrong shape",
        )
        _require(
            np.all(np.isfinite(standard_errors)) and np.all(standard_errors >= 0.0),
            "Monte Carlo ground-truth SE is invalid",
        )
        recomputed_rmse_se = float(np.sqrt(np.mean(np.square(standard_errors))))
        _close(
            ground_truth.get("rmse_standard_error"),
            recomputed_rmse_se,
            path="ground_truth.rmse_standard_error",
        )
        half_widths = np.asarray(
            ground_truth.get("simultaneous_half_widths"), dtype=np.float64
        )
        _require(
            half_widths.shape == (n,),
            "Monte Carlo simultaneous half-widths have wrong shape",
        )
        _require(
            np.all(np.isfinite(half_widths)) and np.all(half_widths >= 0.0),
            "Monte Carlo simultaneous half-widths are invalid",
        )
        _close(
            ground_truth.get("max_simultaneous_half_width"),
            float(half_widths.max()),
            path="ground_truth.max_simultaneous_half_width",
        )
        permutations = _integer(
            ground_truth.get("permutations"),
            path="ground_truth.permutations",
            minimum=2,
        )
        pairs = _integer(
            ground_truth.get("independent_pair_units"),
            path="ground_truth.independent_pair_units",
            minimum=1,
        )
        _require(
            permutations == 2 * pairs,
            "Monte Carlo permutation/pair counts disagree",
        )
        conceptual = _integer(
            ground_truth.get("conceptual_internal_prefix_calls"),
            path="ground_truth.conceptual_internal_prefix_calls",
            minimum=1,
        )
        physical = _integer(
            ground_truth.get("physical_internal_prefix_calls"),
            path="ground_truth.physical_internal_prefix_calls",
            minimum=1,
        )
        saved = _integer(
            ground_truth.get("boundary_reuse_saved_calls"),
            path="ground_truth.boundary_reuse_saved_calls",
        )
        _require(
            conceptual == permutations * (n - 1),
            "Monte Carlo conceptual ground-truth calls are wrong",
        )
        _require(
            physical + saved == conceptual,
            "Monte Carlo physical/saved ground-truth calls do not add up",
        )
        equivalent = ground_truth.get("permutation_path_equivalent_utility_calls")
        if equivalent is not None:
            _require(
                int(equivalent) == conceptual + 2,
                "Monte Carlo permutation-path-equivalent ground-truth calls are wrong",
            )
        shared_boundary = ground_truth.get("shared_boundary_calls_physically_evaluated")
        if shared_boundary is not None:
            _require(
                int(shared_boundary) == 2 * n + 2,
                "Monte Carlo shared ground-truth boundary-call count is wrong",
            )
    else:  # pragma: no cover - guarded by _dataset_kind
        raise ValueError(f"unsupported dataset kind: {dataset}")

    discrepancy = float(np.max(np.abs(truth - expected)))
    _require(discrepancy <= 2e-12, f"{dataset} ground truth is incorrect")
    _require(
        math.isclose(float(expected.sum()), target, rel_tol=2e-12, abs_tol=2e-12),
        f"{dataset} truth violates efficiency",
    )
    if "sum" in ground_truth:
        _close(ground_truth["sum"], float(expected.sum()), path="ground_truth.sum")
    if "efficiency_error" in ground_truth:
        _close(
            ground_truth["efficiency_error"],
            abs(float(expected.sum()) - target),
            path="ground_truth.efficiency_error",
        )
    return truth, n, target, {
        "kind": kind,
        "maximum_truth_discrepancy": discrepancy,
    }


def _budget_lists(
    configuration: Mapping[str, Any],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    raw_inner = configuration.get(
        "inner_utility_call_budgets", configuration.get("inner_budgets")
    )
    raw_total = configuration.get("total_call_budgets")
    _require(isinstance(raw_inner, (list, tuple)), "inner budgets must be a list")
    _require(isinstance(raw_total, (list, tuple)), "total budgets must be a list")
    inner = tuple(
        _integer(value, path=f"inner_budgets[{index}]", minimum=1)
        for index, value in enumerate(raw_inner)
    )
    total = tuple(
        _integer(value, path=f"total_budgets[{index}]", minimum=1)
        for index, value in enumerate(raw_total)
    )
    _require(len(inner) == EXPECTED_BUDGETS, "exactly five inner budgets are required")
    _require(len(total) == EXPECTED_BUDGETS, "exactly five total budgets are required")
    _require(all(a < b for a, b in zip(inner, inner[1:])), "inner budgets must increase")
    _require(all(a < b for a, b in zip(total, total[1:])), "total budgets must increase")
    return inner, total


def _actual_call_list(
    summary: Mapping[str, Any], *, path: str
) -> tuple[int, ...]:
    raw: Any = summary.get("actual_utility_calls_by_repeat")
    if raw is None:
        raw = summary.get("actual_utility_calls_per_estimate")
    if raw is None:
        raw = summary.get("utility_calls_per_estimate")
    if isinstance(raw, (int, np.integer)):
        raw = [int(raw)] * EXPECTED_REPEATS
    _require(
        isinstance(raw, (list, tuple)) and len(raw) == EXPECTED_REPEATS,
        f"{path} must retain three actual-call counts",
    )
    return tuple(
        _integer(value, path=f"{path}[{index}]", minimum=1)
        for index, value in enumerate(raw)
    )


def _validate_diagnostics(
    method: str,
    diagnostics: Any,
    *,
    actual: Sequence[int],
    target: int,
    n: int,
    path: str,
) -> int:
    if diagnostics is None:
        # The normalized call vector above remains mandatory.  Wine summaries
        # copied from an earlier fully-audited run may lack bulky diagnostics.
        return sum(target - value for value in actual)
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == EXPECTED_REPEATS,
        f"{path} must contain three diagnostic objects",
    )
    unused_total = 0
    for repeat, raw in enumerate(diagnostics):
        diagnostic = _mapping(raw, path=f"{path}[{repeat}]")
        diagnostic_path = f"{path}[{repeat}]"
        if "utility_evaluations" in diagnostic:
            recorded = _integer(
                diagnostic["utility_evaluations"],
                path=f"{diagnostic_path}.utility_evaluations",
            )
            _require(recorded == actual[repeat], f"{diagnostic_path} actual calls disagree")
        if "target_call_budget" in diagnostic:
            _require(
                _integer(
                    diagnostic["target_call_budget"],
                    path=f"{diagnostic_path}.target_call_budget",
                )
                == target,
                f"{diagnostic_path} target calls disagree",
            )
        if "target_total_utility_calls" in diagnostic:
            _require(
                int(diagnostic["target_total_utility_calls"]) == target,
                f"{diagnostic_path} total target calls disagree",
            )
        unused = target - actual[repeat]
        if "unused_calls" in diagnostic:
            _require(
                _integer(
                    diagnostic["unused_calls"], path=f"{diagnostic_path}.unused_calls"
                )
                == unused,
                f"{diagnostic_path} unused calls disagree",
            )
        if method == "tmc_shapley":
            boundary = diagnostic.get("boundary_utility_evaluations")
            prefixes = diagnostic.get("prefix_utility_evaluations")
            if boundary is not None and prefixes is not None:
                _require(
                    int(boundary) + int(prefixes) == actual[repeat],
                    f"{diagnostic_path} TMC call components do not add up",
                )
            remainder = diagnostic.get("budget_remainder_calls")
            saved = diagnostic.get("truncation_saved_calls")
            if remainder is not None and saved is not None:
                _require(
                    int(remainder) + int(saved) == unused,
                    f"{diagnostic_path} TMC unused-call components do not add up",
                )
        if method in {
            "inside_greedy",
            "inside_orbit",
            "ofa_iid_linear",
            "ofa_iid_ratio",
            "kernel_shap",
        }:
            boundary = diagnostic.get("boundary_utility_evaluations")
            interior = diagnostic.get("inner_utility_evaluations")
            if boundary is not None and interior is not None:
                _require(
                    int(boundary) + int(interior) == actual[repeat],
                    f"{diagnostic_path} boundary/interior calls do not add up",
                )
        if method == "s_diff" and "boundary_evaluations" in diagnostic:
            samples = diagnostic.get("num_utility_samples")
            if samples is not None:
                _require(
                    int(diagnostic["boundary_evaluations"]) + int(samples)
                    == actual[repeat],
                    f"{diagnostic_path} S-Diff calls do not add up",
                )
        unused_total += unused
    return unused_total


def _validate_canonical_inside_configuration(
    configuration: Mapping[str, Any],
    *,
    path: str,
    expected_greedy_candidate_pool: int,
    expected_lambda0: float,
    require_explicit_orbit_candidate_pool: bool,
) -> None:
    """Require the two fixed paper-facing INSIDE algorithm bundles."""
    inside = _mapping(configuration.get("inside"), path=f"{path}.inside")

    # The analytic runner historically records the same canonical bundle in a
    # flat ``inside.greedy_*`` schema, whereas the Wine runner/reducer records
    # nested ``inside.greedy`` and ``inside.orbit`` objects.  Accept both
    # serializations, but normalize them before applying exactly the same
    # algorithm-identity checks.
    nested = isinstance(inside.get("greedy"), Mapping) or isinstance(
        inside.get("orbit"), Mapping
    )
    if nested:
        greedy = _mapping(
            inside.get("greedy"), path=f"{path}.inside.greedy"
        )
        orbit = _mapping(
            inside.get("orbit"), path=f"{path}.inside.orbit"
        )
    else:
        greedy = {
            "design": inside.get("greedy_design"),
            "candidate_pool": inside.get(
                "greedy_candidate_pool", inside.get("candidate_pool")
            ),
            "second_moment_scope": inside.get(
                "greedy_second_moment_scope"
            ),
            "estimator": inside.get("greedy_estimator"),
            "ratio_missing_policy": inside.get(
                "greedy_ratio_missing_policy"
            ),
            "mean_balance_mode": inside.get("greedy_mean_balance_mode"),
            "mean_balance_lambda0": inside.get(
                "greedy_mean_balance_lambda0"
            ),
        }
        orbit = {
            "design": inside.get("orbit_design"),
            "candidate_pool": inside.get(
                "orbit_candidate_pool", inside.get("candidate_pool")
            ),
            "estimator": inside.get("orbit_estimator"),
        }
        if "orbit_second_moment_scope" in inside:
            orbit["second_moment_scope"] = inside.get(
                "orbit_second_moment_scope"
            )

    greedy_design = greedy.get("design")
    if expected_greedy_candidate_pool == CURRENT_GREEDY_CANDIDATE_POOL:
        _require(
            greedy_design is not None,
            "current canonical INSIDE-Greedy must record its design",
        )
    if greedy_design is not None:
        _require(
            "per_size_frame_coupled_design" in str(greedy_design),
            "canonical INSIDE-Greedy must use per_size_frame_coupled_design",
        )

    _require(
        _integer(
            greedy.get("candidate_pool"),
            path=f"{path}.inside.greedy.candidate_pool",
            minimum=1,
        )
        == expected_greedy_candidate_pool,
        "canonical INSIDE-Greedy candidate pool does not match its protocol",
    )

    _require(
        greedy.get("second_moment_scope") == "per_size",
        "canonical INSIDE-Greedy must use per-size second moments",
    )
    greedy_estimator = str(greedy.get("estimator", "")).lower()
    _require(
        "ratio" in greedy_estimator and "linear" not in greedy_estimator,
        "canonical INSIDE-Greedy must use the OFA ratio estimator",
    )
    _require(
        greedy.get("ratio_missing_policy") == "raise",
        "canonical INSIDE-Greedy must reject missing ratio strata",
    )
    _require(
        str(greedy.get("mean_balance_mode")) == "normalized",
        "canonical INSIDE-Greedy must use normalized mean balance",
    )
    _require(
        math.isclose(
            _finite(
                greedy.get("mean_balance_lambda0"),
                path=f"{path}.inside.greedy.mean_balance_lambda0",
            ),
            expected_lambda0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "canonical INSIDE-Greedy lambda0 does not match its protocol",
    )

    orbit_design = orbit.get("design")
    if expected_greedy_candidate_pool == CURRENT_GREEDY_CANDIDATE_POOL:
        _require(
            orbit_design is not None,
            "current canonical INSIDE-Orbit must record its design",
        )
    if orbit_design is not None:
        _require(
            "cyclic_orbit_frame_design" in str(orbit_design),
            "canonical INSIDE-Orbit must use cyclic_orbit_frame_design",
        )
    orbit_candidate_pool = orbit.get("candidate_pool")
    if require_explicit_orbit_candidate_pool:
        _require(
            orbit_candidate_pool is not None,
            "canonical analytic INSIDE-Orbit must record its candidate pool",
        )
    if orbit_candidate_pool is not None:
        _require(
            _integer(
                orbit_candidate_pool,
                path=f"{path}.inside.orbit.candidate_pool",
                minimum=1,
            )
            == EXPECTED_ORBIT_CANDIDATE_POOL,
            "canonical INSIDE-Orbit candidate pool must be K=4",
        )

    orbit_estimator = str(orbit.get("estimator", "")).lower()
    _require(
        "ratio" in orbit_estimator and "linear" not in orbit_estimator,
        "canonical INSIDE-Orbit must use the OFA ratio estimator",
    )
    if "second_moment_scope" in orbit:
        _require(
            orbit.get("second_moment_scope") == "per_size",
            "canonical INSIDE-Orbit must use per-size second moments",
        )


def _validate_canonical_inside_diagnostics(
    method: str,
    diagnostics: Any,
    *,
    path: str,
    expected_greedy_candidate_pool: int,
    expected_lambda0: float,
    require_hyperparameter_diagnostics: bool,
    num_players: int,
) -> None:
    """Reject a canonical label attached to an old design/estimator bundle."""
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == EXPECTED_REPEATS,
        f"{path} must retain three canonical diagnostic objects",
    )
    for repeat, raw in enumerate(diagnostics):
        diagnostic_path = f"{path}[{repeat}]"
        diagnostic = _mapping(raw, path=diagnostic_path)
        if method == "inside_greedy":
            _require(
                diagnostic.get("design_method") == "frame_coupled_per_size",
                "canonical INSIDE-Greedy must use the per-size Greedy design",
            )
            _require(
                diagnostic.get("second_moment_scope") == "per_size",
                "canonical INSIDE-Greedy diagnostics must identify per-size scope",
            )
            _require(
                diagnostic.get("estimator")
                == "ofa_conditional_mean_ratio_missing_raise",
                "canonical INSIDE-Greedy diagnostics must identify covered OFA ratio",
            )
            _require(
                diagnostic.get("official_ratio_missing_policy") == "raise",
                "canonical INSIDE-Greedy diagnostics must reject missing strata",
            )
            design_diagnostics = diagnostic.get("design_diagnostics")
            nested_design = (
                design_diagnostics
                if isinstance(design_diagnostics, Mapping)
                else {}
            )
            recorded_pool = diagnostic.get(
                "candidate_pool", nested_design.get("candidate_pool")
            )
            recorded_mode = diagnostic.get(
                "mean_balance_mode", nested_design.get("mean_balance_mode")
            )
            recorded_lambda0 = diagnostic.get(
                "mean_balance_lambda0",
                nested_design.get("mean_balance_lambda0"),
            )
            if require_hyperparameter_diagnostics:
                _require(
                    recorded_pool is not None
                    and recorded_mode is not None
                    and recorded_lambda0 is not None,
                    "current canonical INSIDE-Greedy diagnostics must retain K and lambda0",
                )
            if recorded_pool is not None:
                _require(
                    _integer(
                        recorded_pool,
                        path=f"{diagnostic_path}.candidate_pool",
                        minimum=1,
                    )
                    == expected_greedy_candidate_pool,
                    "canonical INSIDE-Greedy diagnostic candidate pool does not match its protocol",
                )
            if recorded_mode is not None:
                _require(
                    str(recorded_mode) == "normalized",
                    "canonical INSIDE-Greedy diagnostics must use normalized mean balance",
                )
            if recorded_lambda0 is not None:
                _require(
                    math.isclose(
                        _finite(
                            recorded_lambda0,
                            path=f"{diagnostic_path}.mean_balance_lambda0",
                        ),
                        expected_lambda0,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    ),
                    "canonical INSIDE-Greedy diagnostic lambda0 does not match its protocol",
                )
            if require_hyperparameter_diagnostics:
                expected_factor = (num_players - 2.0) / (
                    num_players - 1.0
                )
                expected_effective = expected_lambda0 * expected_factor
                recorded_effective = diagnostic.get(
                    "mean_balance_effective_raw",
                    nested_design.get("mean_balance_effective_raw"),
                )
                _require(
                    recorded_effective is not None,
                    "current canonical INSIDE-Greedy diagnostics must retain effective lambda",
                )
                _close(
                    recorded_effective,
                    expected_effective,
                    path=f"{diagnostic_path}.mean_balance_effective_raw",
                    atol=1e-15,
                    rtol=0.0,
                )
                expected_normalization = {
                    "mean_balance_mean_weight_squared": 1.0,
                    "mean_balance_dimension_correction": expected_factor,
                    "mean_balance_normalization_factor": expected_factor,
                    "mean_balance_effective_raw": expected_effective,
                }
                for key, expected in expected_normalization.items():
                    _require(
                        key in nested_design,
                        "current canonical INSIDE-Greedy nested diagnostics "
                        f"must retain {key}",
                    )
                    _close(
                        nested_design[key],
                        expected,
                        path=f"{diagnostic_path}.design_diagnostics.{key}",
                        atol=1e-15,
                        rtol=0.0,
                    )
            coverage = _mapping(
                diagnostic.get("coverage"),
                path=f"{diagnostic_path}.coverage",
            )
            _require(
                coverage.get("all_player_size_strata_covered") is True,
                "canonical INSIDE-Greedy requires complete player-size coverage",
            )
            _require(
                _integer(
                    coverage.get("minimum_inclusion_count"),
                    path=f"{diagnostic_path}.coverage.minimum_inclusion_count",
                    minimum=1,
                )
                >= 1,
                "canonical INSIDE-Greedy has an empty inclusion stratum",
            )
            _require(
                _integer(
                    coverage.get("minimum_exclusion_count"),
                    path=f"{diagnostic_path}.coverage.minimum_exclusion_count",
                    minimum=1,
                )
                >= 1,
                "canonical INSIDE-Greedy has an empty exclusion stratum",
            )
            for count_name in (
                "missing_inclusion_strata",
                "missing_exclusion_strata",
                "missing_inner_sizes",
            ):
                if count_name in coverage:
                    _require(
                        _integer(
                            coverage[count_name],
                            path=f"{diagnostic_path}.coverage.{count_name}",
                        )
                        == 0,
                        f"canonical INSIDE-Greedy coverage has nonzero {count_name}",
                    )
            if isinstance(design_diagnostics, Mapping):
                _require(
                    design_diagnostics.get("second_moment_scope") == "per_size",
                    "canonical INSIDE-Greedy nested design scope is not per-size",
                )
        elif method == "inside_orbit" and not diagnostic.get(
            "reused_from_source_report"
        ):
            _require(
                diagnostic.get("design_method") == "cyclic_orbit_frame",
                "canonical INSIDE-Orbit must use complete cyclic orbits",
            )
            _require(
                diagnostic.get("estimator")
                == "ofa_conditional_mean_ratio_strict_balanced",
                "canonical INSIDE-Orbit must use strict balanced OFA ratio",
            )


def _rank_metrics(estimates: np.ndarray, truth: np.ndarray) -> tuple[float, float]:
    correlations = [
        float(stats.spearmanr(estimate, truth).statistic) for estimate in estimates
    ]
    top_k = min(5, estimates.shape[1])
    truth_top = set(np.argsort(truth)[-top_k:])
    overlaps = [
        len(truth_top.intersection(np.argsort(estimate)[-top_k:])) / top_k
        for estimate in estimates
    ]
    return float(np.mean(correlations)), float(np.mean(overlaps))


def _validate_summary_metrics(
    summary: Mapping[str, Any],
    estimates: np.ndarray,
    truth: np.ndarray,
    *,
    efficiency_target: float,
    ground_truth_se: np.ndarray | None,
    path: str,
) -> float:
    errors = estimates - truth[None, :]
    per_repeat_mse = np.mean(np.square(errors), axis=1)
    per_repeat_rmse = np.sqrt(per_repeat_mse)
    aggregate_rmse = float(np.sqrt(per_repeat_mse.mean()))
    maximum = _close(
        summary.get("aggregate_rmse"),
        aggregate_rmse,
        path=f"{path}.aggregate_rmse",
    )
    values = {
        "mean_repeat_rmse": float(per_repeat_rmse.mean()),
        "std_repeat_rmse": float(per_repeat_rmse.std(ddof=1)),
        "bias_l2": float(np.linalg.norm(estimates.mean(axis=0) - truth)),
        "bias_rmse": float(np.sqrt(np.mean(np.square(estimates.mean(axis=0) - truth)))),
    }
    residuals = estimates.sum(axis=1) - efficiency_target
    values.update(
        {
            "mean_efficiency_residual": float(residuals.mean()),
            "max_absolute_efficiency_residual": float(np.max(np.abs(residuals))),
            "mean_efficiency_gap": float(np.mean(np.abs(residuals))),
            "max_efficiency_gap": float(np.max(np.abs(residuals))),
        }
    )
    for name, expected in values.items():
        if name in summary:
            maximum = max(
                maximum,
                _close(summary[name], expected, path=f"{path}.{name}"),
            )
    if ground_truth_se is not None and "noise_corrected_rmse" in summary:
        noise = float(np.mean(np.square(ground_truth_se)))
        corrected = math.sqrt(max(0.0, aggregate_rmse**2 - noise))
        maximum = max(
            maximum,
            _close(
                summary["noise_corrected_rmse"],
                corrected,
                path=f"{path}.noise_corrected_rmse",
            ),
        )
    if "mean_spearman" in summary or "mean_top5_overlap" in summary:
        mean_spearman, mean_top5 = _rank_metrics(estimates, truth)
        if "mean_spearman" in summary:
            maximum = max(
                maximum,
                _close(
                    summary["mean_spearman"],
                    mean_spearman,
                    path=f"{path}.mean_spearman",
                ),
            )
        if "mean_top5_overlap" in summary:
            maximum = max(
                maximum,
                _close(
                    summary["mean_top5_overlap"],
                    mean_top5,
                    path=f"{path}.mean_top5_overlap",
                ),
            )
    interval = np.asarray(summary.get("aggregate_rmse_bootstrap_95"), dtype=float)
    _require(interval.shape == (2,), f"{path} RMSE interval must have two endpoints")
    _require(
        np.all(np.isfinite(interval))
        and np.all(interval >= 0.0)
        and interval[0] <= interval[1],
        f"{path} RMSE interval is invalid",
    )
    return maximum


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Raise on any failed audit and otherwise return a durable summary."""
    _require(isinstance(report, Mapping), "report root must be an object")
    _require(report.get("status") == "complete", "report status must be complete")
    experiment_id = str(report.get("experiment", ""))
    canonical_inside = experiment_id in CANONICAL_INSIDE_EXPERIMENT_IDS
    current_canonical_inside = (
        experiment_id in CURRENT_CANONICAL_INSIDE_EXPERIMENT_IDS
    )
    if canonical_inside:
        expected_greedy_pool, expected_normalized_lambda0 = (
            _canonical_protocol(experiment_id)
        )
    else:
        expected_greedy_pool = LEGACY_GREEDY_CANDIDATE_POOL
        expected_normalized_lambda0 = LEGACY_NORMALIZED_LAMBDA0
    dataset = _dataset_kind(report)
    configuration = _mapping(report.get("configuration"), path="configuration")
    _require(
        _configured_methods(configuration) == METHOD_ORDER,
        "configuration method order must contain exactly the common eight methods",
    )
    _require(
        _integer(configuration.get("repeats"), path="configuration.repeats")
        == EXPECTED_REPEATS,
        "comparison must use exactly three repeats",
    )
    candidate_pool, mean_balance_mode, mean_balance_coefficient = (
        _inside_balance_configuration(configuration)
    )
    orbit_candidate_pool = _inside_orbit_candidate_pool(configuration)
    _require(
        _integer(candidate_pool, path="configuration.INSIDE.candidate_pool", minimum=1)
        == expected_greedy_pool,
        "INSIDE-Greedy candidate pool does not match the report protocol",
    )
    if mean_balance_mode == "normalized":
        _require(
            math.isclose(
                _finite(
                    mean_balance_coefficient,
                    path="configuration.INSIDE.mean_balance_lambda0",
                ),
                expected_normalized_lambda0,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "normalized INSIDE-Greedy lambda0 does not match the report protocol",
        )
    elif mean_balance_mode == "legacy_raw":
        _require(
            math.isclose(
                _finite(
                    mean_balance_coefficient,
                    path="configuration.INSIDE.mean_balance",
                ),
                EXPECTED_LEGACY_RAW_MEAN_BALANCE,
                rel_tol=0.0,
                abs_tol=1e-15,
            ),
            "legacy INSIDE-Greedy must use raw lambda=0.1",
        )
    else:
        raise ValueError(
            "INSIDE-Greedy mean_balance_mode must be explicit 'normalized' "
            "or an auditable legacy raw report"
        )
    if canonical_inside:
        if experiment_id in ANALYTIC_PROTOCOL_VERSIONS:
            _require(
                configuration.get("protocol_version")
                == ANALYTIC_PROTOCOL_VERSIONS[experiment_id],
                "canonical analytic INSIDE report has the wrong protocol version",
            )
        _validate_canonical_inside_configuration(
            configuration,
            path="configuration",
            expected_greedy_candidate_pool=expected_greedy_pool,
            expected_lambda0=expected_normalized_lambda0,
            require_explicit_orbit_candidate_pool=(
                experiment_id in ANALYTIC_PROTOCOL_VERSIONS
            ),
        )

    truth, n, efficiency_target, truth_audit = _validate_game_and_truth(
        report, dataset
    )
    inner_budgets, total_budgets = _budget_lists(configuration)
    boundary_calls = 2 * n + 2
    for index, (inner, total) in enumerate(zip(inner_budgets, total_budgets)):
        _require(
            total == inner + boundary_calls,
            f"budget {index} must include {boundary_calls} common boundary calls",
        )
    configured_boundary = configuration.get(
        "boundary_utility_calls_for_ofa_methods",
        configuration.get(
            "boundary_utility_calls_for_ratio_methods",
            configuration.get("boundary_utility_calls"),
        ),
    )
    if configured_boundary is not None:
        _require(int(configured_boundary) == boundary_calls, "configured boundary calls are wrong")
    boundary_metadata = report.get("boundary")
    if isinstance(boundary_metadata, Mapping) and "utility_calls" in boundary_metadata:
        _require(
            int(boundary_metadata["utility_calls"]) == boundary_calls,
            "boundary metadata has the wrong physical call count",
        )
    if dataset in MC_REFERENCE_DATASETS:
        gt_physical = int(report["ground_truth"]["physical_internal_prefix_calls"])
        _require(
            gt_physical > max(total_budgets),
            "Monte Carlo ground-truth budget must exceed every comparison budget",
        )

    results = _mapping(
        report.get("results_by_inner_budget"), path="results_by_inner_budget"
    )
    _require(len(results) == EXPECTED_BUDGETS, "results must contain five budgets")
    _require(
        set(results) == {str(value) for value in inner_budgets},
        "result budget keys differ from configuration",
    )
    ground_truth_se: np.ndarray | None = None
    if dataset in MC_REFERENCE_DATASETS:
        ground_truth_se = np.asarray(report["ground_truth"]["standard_errors"], dtype=float)

    max_metric_discrepancy = 0.0
    total_actual_calls = 0
    total_unused_calls = 0
    max_efficiency_residual = 0.0
    for inner, target in zip(inner_budgets, total_budgets):
        row_path = f"results_by_inner_budget.{inner}"
        row = _mapping(results[str(inner)], path=row_path)
        _require(int(row.get("inner_utility_calls")) == inner, f"{row_path} inner budget is wrong")
        _require(
            int(row.get("total_utility_calls_per_estimate")) == target,
            f"{row_path} total budget is wrong",
        )
        methods = _mapping(row.get("methods"), path=f"{row_path}.methods")
        _require(set(methods) == set(METHOD_ORDER), f"{row_path} method set is wrong")
        for method in METHOD_ORDER:
            path = f"{row_path}.methods.{method}"
            summary = _mapping(methods[method], path=path)
            estimates = np.asarray(summary.get("estimates"), dtype=np.float64)
            _require(
                estimates.shape == (EXPECTED_REPEATS, n),
                f"{path} must retain a 3 x {n} estimate matrix",
            )
            _require(np.all(np.isfinite(estimates)), f"{path} estimates are non-finite")
            max_metric_discrepancy = max(
                max_metric_discrepancy,
                _validate_summary_metrics(
                    summary,
                    estimates,
                    truth,
                    efficiency_target=efficiency_target,
                    ground_truth_se=ground_truth_se,
                    path=path,
                ),
            )
            max_efficiency_residual = max(
                max_efficiency_residual,
                float(np.max(np.abs(estimates.sum(axis=1) - efficiency_target))),
            )
            actual = _actual_call_list(
                summary, path=f"{path}.actual_utility_calls_by_repeat"
            )
            _require(all(value <= target for value in actual), f"{path} exceeds its call cap")
            if method in FULL_BUDGET_METHODS:
                _require(all(value == target for value in actual), f"{path} must spend its full budget")
            elif method == "cc":
                _require(
                    all(target - value in {0, 1} for value in actual),
                    f"{path} may leave at most one unpaired call unused",
                )
            if "mean_actual_utility_calls" in summary:
                max_metric_discrepancy = max(
                    max_metric_discrepancy,
                    _close(
                        summary["mean_actual_utility_calls"],
                        float(np.mean(actual)),
                        path=f"{path}.mean_actual_utility_calls",
                        atol=1e-10,
                        rtol=0.0,
                    ),
                )
            stored_target = summary.get(
                "target_total_utility_calls",
                summary.get("target_total_call_budget"),
            )
            if stored_target is not None:
                _require(int(stored_target) == target, f"{path} stored target is wrong")
            diagnostics = summary.get(
                "diagnostics_by_repeat", summary.get("cc_diagnostics_by_repeat")
            )
            total_unused_calls += _validate_diagnostics(
                method,
                diagnostics,
                actual=actual,
                target=target,
                n=n,
                path=f"{path}.diagnostics_by_repeat",
            )
            if canonical_inside and method in {
                "inside_greedy",
                "inside_orbit",
            }:
                _validate_canonical_inside_diagnostics(
                    method,
                    diagnostics,
                    path=f"{path}.diagnostics_by_repeat",
                    expected_greedy_candidate_pool=expected_greedy_pool,
                    expected_lambda0=expected_normalized_lambda0,
                    require_hyperparameter_diagnostics=(
                        current_canonical_inside
                    ),
                    num_players=n,
                )
            total_actual_calls += sum(actual)

    return {
        "status": "ready_to_share",
        "validated_report": report.get("experiment"),
        "dataset": dataset,
        "checks": {
            "common_method_order_exactly_eight": "passed",
            "inside_candidate_pool_matches_protocol": "passed",
            "inside_greedy_mean_balance_configuration": "passed",
            "canonical_inside_algorithm_identity": (
                "passed" if canonical_inside else "historical_not_applicable"
            ),
            "five_budget_points": "passed",
            "three_repeats_per_method_budget": "passed",
            "ground_truth": "passed",
            "ground_truth_uncertainty_metadata_if_mc": "passed",
            "rmse_and_retained_secondary_metrics_recomputed": "passed",
            "actual_calls_do_not_exceed_target": "passed",
            "non_tmc_methods_spend_full_budget_except_one_unpaired_cc_call": "passed",
            "available_method_diagnostic_call_accounting": "passed",
        },
        "ground_truth_audit": truth_audit,
        "inside_greedy_balance_audit": {
            "candidate_pool": int(candidate_pool),
            "mode": mean_balance_mode,
            "coefficient": float(mean_balance_coefficient),
        },
        "inside_candidate_pool_audit": {
            "greedy": int(candidate_pool),
            "orbit": (
                None
                if orbit_candidate_pool is None
                else int(orbit_candidate_pool)
            ),
        },
        "validated_shape": {
            "players": n,
            "methods": len(METHOD_ORDER),
            "budget_points": EXPECTED_BUDGETS,
            "repeats": EXPECTED_REPEATS,
            "per_repeat_estimates_checked": len(METHOD_ORDER)
            * EXPECTED_BUDGETS
            * EXPECTED_REPEATS,
            "scalar_player_estimates_checked": len(METHOD_ORDER)
            * EXPECTED_BUDGETS
            * EXPECTED_REPEATS
            * n,
        },
        "call_accounting": {
            "total_actual_utility_calls": total_actual_calls,
            "total_unused_utility_calls": total_unused_calls,
        },
        "maximum_discrepancies": {
            "stored_metric_absolute": max_metric_discrepancy,
            "sample_estimate_efficiency_residual": max_efficiency_residual,
        },
        "caveats": (
            [
                "This data-valuation experiment uses a high-budget Monte Carlo "
                "reference with retained standard errors; the reference is not "
                "exact."
            ]
            if dataset in MC_REFERENCE_DATASETS
            else []
        )
        + [
            "TMC-Shapley is evaluated at observed physical calls after truncation; saved calls are not silently reassigned.",
            "Three repeats provide the requested runtime-saving comparison but only coarse uncertainty intervals.",
        ],
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
    output = args.output or args.report.with_name(
        f"{args.report.stem}_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{validation['status']}: saved {output}")


if __name__ == "__main__":
    main()

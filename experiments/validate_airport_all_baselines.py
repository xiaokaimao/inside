"""Independently validate the Wine-style airport all-baseline report.

The validator deliberately reconstructs the analytic airport Shapley values,
RMSEs, and utility-call accounting from the retained per-repeat estimates and
diagnostics.  It does not trust the stored aggregate metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from experiments.airport_game import AIRPORT_CLASS_COUNTS, AIRPORT_COSTS


EXPECTED_METHOD_ORDER = (
    "frame_orbit_ratio",
    "official_ofa_fixed_ratio",
    "official_cc_basic",
    "s_diff",
    "diff",
    "group_testing",
    "kernel_shap_sampled",
    "gels_shapley",
    "tmc_shapley",
    "stratified_marginal_mc",
)
EXPECTED_REPEATS = 3
EXPECTED_BUDGET_POINTS = 5
NUM_PLAYERS = 100
RATIO_METHODS = {"frame_orbit_ratio", "official_ofa_fixed_ratio"}
STRICT_DIAGNOSTIC_METHODS = {
    "s_diff",
    "diff",
    "group_testing",
    "kernel_shap_sampled",
    "gels_shapley",
    "stratified_marginal_mc",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_float(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(math.isfinite(result), f"{path} must be finite")
    return result


def _integer(value: Any, *, path: str, minimum: int | None = None) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    result = int(value)
    if minimum is not None:
        _require(result >= minimum, f"{path} must be at least {minimum}")
    return result


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _independent_game_fingerprint() -> str:
    payload = {
        "class_counts": AIRPORT_CLASS_COUNTS,
        "costs": AIRPORT_COSTS.tolist(),
        "utility": "max coalition cost; empty coalition is zero",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _independent_exact_airport_shapley(costs: np.ndarray) -> np.ndarray:
    """Integrate the nested threshold games without calling production code."""
    values = np.asarray(costs, dtype=np.float64)
    _require(values.shape == (NUM_PLAYERS,), "airport costs have wrong shape")
    _require(np.all(np.isfinite(values)), "airport costs are not finite")
    _require(np.all(values >= 0.0), "airport costs must be nonnegative")

    result = np.zeros_like(values)
    previous = 0.0
    for level in np.unique(values[values > 0.0]):
        eligible = values >= level
        count = int(np.count_nonzero(eligible))
        _require(count > 0, "positive threshold has no eligible players")
        result[eligible] += (float(level) - previous) / count
        previous = float(level)
    return result


def _close(
    actual: Any,
    expected: float,
    *,
    path: str,
    atol: float = 2e-14,
    rtol: float = 2e-12,
) -> float:
    value = _finite_float(actual, path=path)
    discrepancy = abs(value - float(expected))
    _require(
        math.isclose(value, float(expected), rel_tol=rtol, abs_tol=atol),
        f"{path} differs: stored={value:.17g}, recomputed={expected:.17g}",
    )
    return discrepancy


def _validate_diagnostics(
    method: str,
    diagnostics: Mapping[str, Any],
    *,
    actual_calls: int,
    target_calls: int,
    inner_calls: int,
    path: str,
) -> int:
    diagnostic_actual = _integer(
        diagnostics.get("utility_evaluations"),
        path=f"{path}.utility_evaluations",
        minimum=0,
    )
    _require(
        diagnostic_actual == actual_calls,
        f"{path}.utility_evaluations disagrees with per-repeat actual calls",
    )

    if method in RATIO_METHODS:
        boundary = _integer(
            diagnostics.get("boundary_utility_evaluations"),
            path=f"{path}.boundary_utility_evaluations",
            minimum=0,
        )
        interior = _integer(
            diagnostics.get("inner_utility_evaluations"),
            path=f"{path}.inner_utility_evaluations",
            minimum=0,
        )
        _require(boundary == 2 * NUM_PLAYERS + 2, f"{path} has wrong boundary cost")
        _require(interior == inner_calls, f"{path} has wrong inner-call count")
        _require(boundary + interior == actual_calls, f"{path} calls do not add up")
        _require(actual_calls == target_calls, f"{path} must spend its full budget")
        return 0

    if method == "official_cc_basic":
        pairs = _integer(
            diagnostics.get("num_pairs"),
            path=f"{path}.num_pairs",
            minimum=0,
        )
        _require(2 * pairs == actual_calls, f"{path} pair calls do not add up")
        _require(actual_calls == target_calls, f"{path} must spend its full budget")
        return 0

    diagnostic_target = _integer(
        diagnostics.get("target_call_budget"),
        path=f"{path}.target_call_budget",
        minimum=0,
    )
    unused = _integer(
        diagnostics.get("unused_calls"),
        path=f"{path}.unused_calls",
        minimum=0,
    )
    _require(diagnostic_target == target_calls, f"{path} has wrong target budget")
    _require(
        diagnostic_actual + unused == target_calls,
        f"{path}: actual calls plus unused calls do not equal target",
    )

    if method == "tmc_shapley":
        boundary = _integer(
            diagnostics.get("boundary_utility_evaluations"),
            path=f"{path}.boundary_utility_evaluations",
            minimum=0,
        )
        prefixes = _integer(
            diagnostics.get("prefix_utility_evaluations"),
            path=f"{path}.prefix_utility_evaluations",
            minimum=0,
        )
        evaluated_prefixes = _integer(
            diagnostics.get("evaluated_prefixes"),
            path=f"{path}.evaluated_prefixes",
            minimum=0,
        )
        remainder = _integer(
            diagnostics.get("budget_remainder_calls"),
            path=f"{path}.budget_remainder_calls",
            minimum=0,
        )
        saved = _integer(
            diagnostics.get("truncation_saved_calls"),
            path=f"{path}.truncation_saved_calls",
            minimum=0,
        )
        _require(boundary == 2, f"{path} must use two endpoint calls")
        _require(prefixes == evaluated_prefixes, f"{path} prefix counts disagree")
        _require(boundary + prefixes == actual_calls, f"{path} calls do not add up")
        _require(remainder + saved == unused, f"{path} unused-call split is wrong")
        return unused

    _require(method in STRICT_DIAGNOSTIC_METHODS, f"unknown method {method}")
    if method in {"s_diff", "diff", "group_testing"}:
        boundary = _integer(
            diagnostics.get("boundary_evaluations"),
            path=f"{path}.boundary_evaluations",
            minimum=0,
        )
        _require(boundary == 2, f"{path} must use two boundary calls")
        sample_key = (
            "num_group_tests" if method == "group_testing" else "num_utility_samples"
        )
        samples = _integer(
            diagnostics.get(sample_key), path=f"{path}.{sample_key}", minimum=0
        )
        _require(boundary + samples == actual_calls, f"{path} calls do not add up")
    elif method in {"kernel_shap_sampled", "gels_shapley"}:
        boundary = _integer(
            diagnostics.get("boundary_utility_evaluations"),
            path=f"{path}.boundary_utility_evaluations",
            minimum=0,
        )
        interior = _integer(
            diagnostics.get("inner_utility_evaluations"),
            path=f"{path}.inner_utility_evaluations",
            minimum=0,
        )
        samples = _integer(
            diagnostics.get("num_samples"),
            path=f"{path}.num_samples",
            minimum=0,
        )
        _require(boundary == 2, f"{path} must use two boundary calls")
        _require(interior == samples, f"{path} sample counts disagree")
        _require(boundary + interior == actual_calls, f"{path} calls do not add up")
    else:
        marginal_samples = _integer(
            diagnostics.get("num_marginal_samples"),
            path=f"{path}.num_marginal_samples",
            minimum=0,
        )
        strata = _integer(
            diagnostics.get("num_strata"),
            path=f"{path}.num_strata",
            minimum=0,
        )
        _require(strata == NUM_PLAYERS**2, f"{path} has wrong stratum count")
        _require(2 * marginal_samples == actual_calls, f"{path} calls do not add up")
    return unused


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Return a serializable validation report or raise on a discrepancy."""
    _require(isinstance(report, Mapping), "report root must be an object")
    _require(report.get("status") == "complete", "experiment is not complete")
    _require(
        report.get("experiment") == "airport_game_wine_style_all_baselines",
        "unexpected experiment identifier",
    )

    game = _mapping(report.get("game"), path="game")
    _require(game.get("players") == NUM_PLAYERS, "game player count is not 100")
    _require(
        tuple(game.get("cost_class_counts", ())) == AIRPORT_CLASS_COUNTS,
        "airport class counts do not match the supplied game",
    )
    empirical_counts = tuple(
        int(np.count_nonzero(AIRPORT_COSTS == level)) for level in range(1, 11)
    )
    _require(empirical_counts == AIRPORT_CLASS_COUNTS, "local airport costs disagree")
    expected_fingerprint = _independent_game_fingerprint()
    _require(
        game.get("fingerprint_sha256") == expected_fingerprint,
        "game fingerprint does not match the supplied airport game",
    )

    configuration = _mapping(report.get("configuration"), path="configuration")
    methods = tuple(configuration.get("methods", ()))
    _require(
        methods == EXPECTED_METHOD_ORDER,
        "configuration must contain exactly the 10 Wine-main-figure methods",
    )
    repeats = _integer(configuration.get("repeats"), path="configuration.repeats")
    _require(repeats == EXPECTED_REPEATS, "configuration must use exactly 3 repeats")

    inner_budgets = tuple(configuration.get("inner_utility_call_budgets", ()))
    total_budgets = tuple(configuration.get("total_call_budgets", ()))
    multipliers = tuple(configuration.get("budget_multipliers", ()))
    _require(len(inner_budgets) == EXPECTED_BUDGET_POINTS, "expected 5 budgets")
    _require(len(total_budgets) == EXPECTED_BUDGET_POINTS, "expected 5 total budgets")
    _require(len(multipliers) == EXPECTED_BUDGET_POINTS, "expected 5 multipliers")
    inner_budgets = tuple(
        _integer(value, path=f"configuration.inner_utility_call_budgets.{index}", minimum=1)
        for index, value in enumerate(inner_budgets)
    )
    total_budgets = tuple(
        _integer(value, path=f"configuration.total_call_budgets.{index}", minimum=1)
        for index, value in enumerate(total_budgets)
    )
    multipliers = tuple(
        _integer(value, path=f"configuration.budget_multipliers.{index}", minimum=1)
        for index, value in enumerate(multipliers)
    )
    _require(
        all(left < right for left, right in zip(inner_budgets, inner_budgets[1:])),
        "inner budgets must be strictly increasing",
    )
    boundary_calls = 2 * NUM_PLAYERS + 2
    _require(
        configuration.get("boundary_utility_calls_for_ratio_methods") == boundary_calls,
        "ratio boundary-call count is incorrect",
    )
    for index, (multiplier, inner, total) in enumerate(
        zip(multipliers, inner_budgets, total_budgets, strict=True)
    ):
        _require(inner == NUM_PLAYERS * multiplier, f"budget {index} has wrong multiplier")
        _require(total == inner + boundary_calls, f"budget {index} has wrong total")

    truth = _independent_exact_airport_shapley(AIRPORT_COSTS)
    _require(truth.shape == (NUM_PLAYERS,), "analytic truth has wrong shape")
    _require(np.all(np.isfinite(truth)), "analytic truth is non-finite")
    _require(
        abs(float(truth.sum()) - float(AIRPORT_COSTS.max())) <= 2e-14,
        "analytic truth violates Shapley efficiency",
    )
    ground_truth = _mapping(report.get("ground_truth"), path="ground_truth")
    stored_truth = np.asarray(ground_truth.get("values"), dtype=np.float64)
    _require(stored_truth.shape == (NUM_PLAYERS,), "stored truth has wrong shape")
    _require(np.all(np.isfinite(stored_truth)), "stored truth is non-finite")
    truth_discrepancy = float(np.max(np.abs(stored_truth - truth)))
    _require(truth_discrepancy <= 2e-15, "stored ground truth is incorrect")
    truth_sum_discrepancy = _close(
        ground_truth.get("sum"), float(truth.sum()), path="ground_truth.sum"
    )
    efficiency_error_discrepancy = _close(
        ground_truth.get("efficiency_error"),
        abs(float(truth.sum()) - float(AIRPORT_COSTS.max())),
        path="ground_truth.efficiency_error",
    )

    results = _mapping(report.get("results_by_inner_budget"), path="results_by_inner_budget")
    _require(set(results) == {str(value) for value in inner_budgets}, "summary budget keys disagree")
    max_rmse_discrepancy = 0.0
    max_secondary_metric_discrepancy = 0.0
    max_efficiency_residual = 0.0
    total_actual_calls = 0
    total_unused_calls = 0
    for inner, target in zip(inner_budgets, total_budgets, strict=True):
        budget_path = f"results_by_inner_budget.{inner}"
        budget_summary = _mapping(results[str(inner)], path=budget_path)
        _require(budget_summary.get("inner_utility_calls") == inner, f"{budget_path} has wrong inner budget")
        _require(
            budget_summary.get("total_utility_calls_per_estimate") == target,
            f"{budget_path} has wrong target budget",
        )
        method_summaries = _mapping(budget_summary.get("methods"), path=f"{budget_path}.methods")
        _require(set(method_summaries) == set(EXPECTED_METHOD_ORDER), f"{budget_path} has wrong method set")

        for method in EXPECTED_METHOD_ORDER:
            method_path = f"{budget_path}.methods.{method}"
            summary = _mapping(method_summaries[method], path=method_path)
            # These are the unaggregated per-repeat estimates retained by the
            # runner.  Treat them as the audit source and recompute every
            # scalar summary below rather than trusting aggregate fields.
            estimates = np.asarray(summary.get("estimates"), dtype=np.float64)
            _require(estimates.shape == (EXPECTED_REPEATS, NUM_PLAYERS), f"{method_path} does not have 3x100 estimates")
            _require(np.all(np.isfinite(estimates)), f"{method_path} estimates are non-finite")

            per_repeat_mse = np.mean(np.square(estimates - truth[None, :]), axis=1)
            aggregate_rmse = float(np.sqrt(per_repeat_mse.mean()))
            mean_repeat_rmse = float(np.mean(np.sqrt(per_repeat_mse)))
            std_repeat_rmse = float(np.std(np.sqrt(per_repeat_mse), ddof=1))
            bias_l2 = float(np.linalg.norm(estimates.mean(axis=0) - truth))
            max_rmse_discrepancy = max(
                max_rmse_discrepancy,
                _close(summary.get("aggregate_rmse"), aggregate_rmse, path=f"{method_path}.aggregate_rmse"),
            )
            for name, expected in (
                ("mean_repeat_rmse", mean_repeat_rmse),
                ("std_repeat_rmse", std_repeat_rmse),
                ("bias_l2", bias_l2),
            ):
                max_secondary_metric_discrepancy = max(
                    max_secondary_metric_discrepancy,
                    _close(summary.get(name), expected, path=f"{method_path}.{name}"),
                )

            interval = np.asarray(summary.get("aggregate_rmse_bootstrap_95"), dtype=np.float64)
            _require(interval.shape == (2,), f"{method_path} bootstrap interval has wrong shape")
            _require(np.all(np.isfinite(interval)), f"{method_path} bootstrap interval is non-finite")
            _require(np.all(interval >= 0.0) and interval[0] <= interval[1], f"{method_path} bootstrap interval is invalid")

            stored_actual = summary.get("actual_utility_calls_by_repeat")
            _require(
                isinstance(stored_actual, list)
                and len(stored_actual) == EXPECTED_REPEATS,
                f"{method_path}.actual_utility_calls_by_repeat must have 3 entries",
            )
            actual_by_repeat = [
                _integer(
                    value,
                    path=f"{method_path}.actual_utility_calls_by_repeat.{repeat}",
                    minimum=0,
                )
                for repeat, value in enumerate(stored_actual)
            ]
            _require(
                all(value <= target for value in actual_by_repeat),
                f"{method_path} exceeds its utility-call cap",
            )
            _close(
                summary.get("mean_actual_utility_calls"),
                float(np.mean(actual_by_repeat)),
                path=f"{method_path}.mean_actual_utility_calls",
                atol=1e-12,
                rtol=0.0,
            )
            _require(summary.get("target_total_utility_calls") == target, f"{method_path} has wrong target")

            summary_diagnostics = summary.get("diagnostics_by_repeat")
            _require(isinstance(summary_diagnostics, list), f"{method_path}.diagnostics_by_repeat must be a list")
            _require(len(summary_diagnostics) == EXPECTED_REPEATS, f"{method_path} must have 3 diagnostics")
            for repeat, diagnostic_value in enumerate(summary_diagnostics):
                diagnostics = _mapping(diagnostic_value, path=f"{method_path}.diagnostics_by_repeat.{repeat}")
                unused = _validate_diagnostics(
                    method,
                    diagnostics,
                    actual_calls=actual_by_repeat[repeat],
                    target_calls=target,
                    inner_calls=inner,
                    path=f"{method_path}.diagnostics_by_repeat.{repeat}",
                )
                total_unused_calls += unused
            total_actual_calls += sum(actual_by_repeat)

            residuals = estimates.sum(axis=1) - float(AIRPORT_COSTS.max())
            max_efficiency_residual = max(
                max_efficiency_residual, float(np.max(np.abs(residuals)))
            )
            max_secondary_metric_discrepancy = max(
                max_secondary_metric_discrepancy,
                _close(
                    summary.get("mean_efficiency_residual"),
                    float(np.mean(residuals)),
                    path=f"{method_path}.mean_efficiency_residual",
                ),
                _close(
                    summary.get("max_absolute_efficiency_residual"),
                    float(np.max(np.abs(residuals))),
                    path=f"{method_path}.max_absolute_efficiency_residual",
                ),
            )

    return {
        "status": "ready_to_share",
        "validated_report": report.get("experiment"),
        "checks": {
            "airport_counts_and_fingerprint": "passed",
            "independent_exact_ground_truth": "passed",
            "wine_main_method_set_exactly_10": "passed",
            "five_budget_points": "passed",
            "three_repeats_per_method_budget": "passed",
            "all_estimates_shape_3_by_100_and_finite": "passed",
            "per_repeat_summary_records_complete": "passed",
            "rmse_independent_recomputation": "passed",
            "actual_calls_do_not_exceed_target": "passed",
            "ratio_and_cc_exact_call_use": "passed",
            "tmc_actual_plus_unused_call_accounting": "passed",
            "baseline_diagnostic_call_accounting": "passed",
        },
        "validated_shape": {
            "players": NUM_PLAYERS,
            "methods": len(EXPECTED_METHOD_ORDER),
            "budget_points": EXPECTED_BUDGET_POINTS,
            "repeats": EXPECTED_REPEATS,
            "summary_method_budget_cells": (
                len(EXPECTED_METHOD_ORDER) * EXPECTED_BUDGET_POINTS
            ),
            "per_repeat_estimates_checked": (
                len(EXPECTED_METHOD_ORDER)
                * EXPECTED_BUDGET_POINTS
                * EXPECTED_REPEATS
            ),
            "scalar_player_estimates_checked": (
                len(EXPECTED_METHOD_ORDER)
                * EXPECTED_BUDGET_POINTS
                * EXPECTED_REPEATS
                * NUM_PLAYERS
            ),
        },
        "call_accounting": {
            "total_actual_utility_calls": total_actual_calls,
            "total_unused_utility_calls_reported": total_unused_calls,
        },
        "maximum_discrepancies": {
            "stored_truth_absolute": truth_discrepancy,
            "stored_truth_sum_absolute": truth_sum_discrepancy,
            "stored_efficiency_error_absolute": efficiency_error_discrepancy,
            "aggregate_rmse_absolute": max_rmse_discrepancy,
            "secondary_summary_metric_absolute": max_secondary_metric_discrepancy,
            "sample_estimate_efficiency_residual": max_efficiency_residual,
        },
        "caveats": [
            "Efficiency residuals are recomputed and reported but are not a pass/fail condition because several source baselines deliberately omit efficiency projection.",
            "TMC-Shapley is plotted at observed physical calls; truncation savings remain unused and are not reassigned to additional permutations.",
            "Three repeats support the requested runtime-saving comparison but provide a coarse uncertainty estimate.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report",
        type=Path,
        nargs="?",
        default=Path("results/json/airport_100_all_baselines_3repeats_50k_1m.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/json/airport_100_all_baselines_3repeats_validation.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validation = validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{validation['status']}: saved {args.output}")


if __name__ == "__main__":
    main()

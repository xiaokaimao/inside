"""Independent consistency checks for the airport benchmark artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.airport_game import (
    AIRPORT_CLASS_COUNTS,
    AIRPORT_COSTS,
    METHOD_ORDER,
    exact_size_mean_utilities,
    game_fingerprint,
)
from frame_ofa import exact_airport_shapley


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    if report.get("status") != "complete":
        raise ValueError("experiment status is not complete")
    if report["game"]["fingerprint_sha256"] != game_fingerprint():
        raise ValueError("game fingerprint does not match the attached game")
    if tuple(report["game"]["cost_class_counts"]) != AIRPORT_CLASS_COUNTS:
        raise ValueError("cost-class counts do not match the attached game")

    truth = exact_airport_shapley(AIRPORT_COSTS)
    stored_truth = np.asarray(
        [row["exact_shapley"] for row in report["ground_truth"]["players"]],
        dtype=np.float64,
    )
    truth_error = float(np.max(np.abs(stored_truth - truth)))
    if truth_error > 2e-15:
        raise ValueError(f"stored exact values disagree by {truth_error}")
    efficiency_error = float(abs(stored_truth.sum() - AIRPORT_COSTS.max()))
    if efficiency_error > 1e-12:
        raise ValueError(f"exact values violate efficiency by {efficiency_error}")

    stored_profile = np.asarray(
        [
            row["exact_mean_utility"]
            for row in report["ground_truth"]["fixed_size_profile"]
        ],
        dtype=np.float64,
    )
    profile_error = float(
        np.max(
            np.abs(stored_profile - exact_size_mean_utilities(AIRPORT_COSTS))
        )
    )
    if profile_error > 2e-14:
        raise ValueError(f"fixed-size profile disagrees by {profile_error}")

    repeats = int(report["configuration"]["repeats"])
    inner_budgets = tuple(
        int(value)
        for value in report["configuration"]["inner_utility_call_budgets"]
    )
    boundary_calls = 2 * len(AIRPORT_COSTS) + 2
    if report["configuration"]["boundary_utility_calls"] != boundary_calls:
        raise ValueError("boundary-call count is incorrect")
    expected_physical = boundary_calls + len(METHOD_ORDER) * sum(inner_budgets)
    if (
        report["configuration"]["physical_experiment_utility_calls_per_repeat"]
        != expected_physical
    ):
        raise ValueError("physical per-repeat call count is incorrect")
    if len(report["raw_repeats"]) != repeats:
        raise ValueError("raw repeat count is incomplete")

    flat_rows = [
        row
        for repeat_report in report["raw_repeats"]
        for row in repeat_report["rows"]
    ]
    expected_rows = repeats * len(inner_budgets) * len(METHOD_ORDER)
    if len(flat_rows) != expected_rows:
        raise ValueError("raw method-budget rows are incomplete")

    max_rmse_recompute_error = 0.0
    max_estimate_efficiency_error = 0.0
    for inner_calls in inner_budgets:
        budget_summary = report["results_by_inner_budget"][str(inner_calls)]
        if budget_summary["total_utility_calls"] != inner_calls + boundary_calls:
            raise ValueError("logical call count is incorrect")
        if set(budget_summary["methods"]) != set(METHOD_ORDER):
            raise ValueError("method set is inconsistent")
        for method in METHOD_ORDER:
            selected = sorted(
                (
                    row
                    for row in flat_rows
                    if row["inner_utility_calls"] == inner_calls
                    and row["method"] == method
                ),
                key=lambda row: row["repeat"],
            )
            if len(selected) != repeats:
                raise ValueError("a method-budget cell has missing repeats")
            estimates = np.asarray(
                [row["estimate"] for row in selected], dtype=np.float64
            )
            if estimates.shape != (repeats, len(truth)):
                raise ValueError("estimate array has the wrong shape")
            if not np.all(np.isfinite(estimates)):
                raise ValueError("estimate array contains non-finite values")
            recomputed = float(
                np.sqrt(np.mean(np.square(estimates - truth[None, :])))
            )
            stored = float(
                budget_summary["methods"][method]["aggregate_rmse"]
            )
            max_rmse_recompute_error = max(
                max_rmse_recompute_error, abs(recomputed - stored)
            )
            max_estimate_efficiency_error = max(
                max_estimate_efficiency_error,
                float(np.max(np.abs(estimates.sum(axis=1) - 10.0))),
            )
    if max_rmse_recompute_error > 2e-15:
        raise ValueError(
            f"aggregate RMSE recomputation differs by {max_rmse_recompute_error}"
        )
    if max_estimate_efficiency_error > 5e-11:
        raise ValueError(
            "sample estimates violate efficiency by "
            f"{max_estimate_efficiency_error}"
        )

    return {
        "status": "ready_to_share",
        "checks": {
            "attached_configuration": "passed",
            "analytic_ground_truth": "passed",
            "ground_truth_efficiency": "passed",
            "fixed_size_profile": "passed",
            "logical_and_physical_call_accounting": "passed",
            "repeat_completeness": "passed",
            "aggregate_rmse_recomputation": "passed",
            "per_estimate_efficiency": "passed",
        },
        "maximum_discrepancies": {
            "stored_truth_absolute": truth_error,
            "ground_truth_efficiency": efficiency_error,
            "fixed_size_profile_absolute": profile_error,
            "aggregate_rmse_absolute": max_rmse_recompute_error,
            "sample_estimate_efficiency": max_estimate_efficiency_error,
        },
        "validated_shape": {
            "players": len(truth),
            "budget_points": len(inner_budgets),
            "methods": len(METHOD_ORDER),
            "repeats": repeats,
            "raw_method_budget_rows": len(flat_rows),
        },
        "caveats": [
            (
                "The airport game's large fixed-size mean component strongly "
                "favors exactly balanced batches; the main comparison does "
                "not isolate cyclic balance from candidate-pool frame search."
            ),
            (
                "The first two Batch-balanced points are not monotone; use "
                "their bootstrap intervals rather than interpreting one orbit "
                "as a stable low-budget ranking."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "report", type=Path, nargs="?", default=Path("results/json/airport_100_frame_ofa_greedy.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/json/airport_100_frame_ofa_greedy_validation.json"),
    )
    args = parser.parse_args()
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

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from experiments.summarize_plot_inside_greedy_fixed_k_lambda import (
    _geometric_mean,
    _safe_ratio,
    build_fixed_k_lambda_locked_selection,
    build_fixed_k_lambda_summary,
    plot_lambda_design_time_ratios,
    plot_lambda_rmse_ratios,
)


LAMBDA_VALUES = (0.0, 0.125, 0.25)


def _budget(
    rmse: float | None,
    design_seconds: float,
    *,
    failures: int = 0,
) -> dict:
    return {
        "aggregate_rmse": rmse,
        "mean_design_seconds": design_seconds,
        "coverage_failures": failures,
        "orbit_aggregate_rmse": 3.0,
    }


def _audit(dataset: str, *, fail_lambda: float | None = None) -> dict:
    if dataset == "airport":
        values = {
            0.0: ((2.0, 4.0), (2.0, 4.0)),
            0.125: ((1.0, 2.0), (8.0, 16.0)),
            0.25: ((1.5, 3.0), (16.0, 32.0)),
        }
        default_rmse = (0.8, 1.6)
        default_time = (0.5, 1.0)
    else:
        values = {
            0.0: ((1.0, 2.0), (1.0, 2.0)),
            0.125: ((2.0, 4.0), (4.0, 8.0)),
            0.25: ((0.75, 1.5), (8.0, 16.0)),
        }
        default_rmse = (0.5, 1.0)
        default_time = (0.25, 0.5)

    results = {}
    for lambda0, (rmses, times) in values.items():
        budgets = {}
        for budget_index in (0, 1):
            failed = fail_lambda == lambda0 and budget_index == 0
            budgets[str(budget_index)] = _budget(
                None if failed else rmses[budget_index],
                times[budget_index],
                failures=int(failed),
            )
        results[f"lambda0={lambda0:.12g}|K=64"] = {
            "lambda0": lambda0,
            "candidate_pool": 64,
            "budgets": budgets,
        }
    results["lambda0=1|K=4"] = {
        "lambda0": 1.0,
        "candidate_pool": 4,
        "budgets": {
            str(index): _budget(default_rmse[index], default_time[index])
            for index in (0, 1)
        },
    }
    return {
        "dataset": dataset,
        "stage": "screening",
        "repeats": 3,
        "base_seed": 20260825,
        "budget_multipliers": [500, 2000],
        "selected_budget_indices": [0, 1],
        "selected_budget_multipliers": [500, 2000],
        "evaluated_configurations": [
            *[
                {"lambda0": lambda0, "candidate_pool": 64}
                for lambda0 in LAMBDA_VALUES
            ],
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "all_rmse_and_aggregates_independently_recomputed": True,
        "all_greedy_hyperparameters_share_size_schedules": True,
        "all_greedy_hyperparameters_share_relabels": True,
        "recomputed_results_by_configuration": results,
    }


class FixedKLambdaSummaryTests(unittest.TestCase):
    def _paths(self, directory: str) -> tuple[Path, Path]:
        airport = Path(directory) / "airport.json"
        voting = Path(directory) / "voting.json"
        airport.write_text(
            json.dumps({"dataset": "airport", "raw_cells": [{"source": 1}]}),
            encoding="utf-8",
        )
        voting.write_text(
            json.dumps({"dataset": "voting", "raw_cells": [{"source": 2}]}),
            encoding="utf-8",
        )
        return airport, voting

    def test_exact_zero_ratio_is_valid(self) -> None:
        self.assertEqual(_safe_ratio(0.0, 2.0), 0.0)
        self.assertEqual(_safe_ratio(0.0, 0.0), 1.0)
        self.assertIsNone(_safe_ratio(1.0, 0.0))
        self.assertEqual(_geometric_mean([1.0, 0.0]), 0.0)

    def test_undefined_orbit_ratio_does_not_cancel_primary_selection(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            audits = [_audit("airport"), _audit("voting")]
            for audit in audits:
                for result in audit["recomputed_results_by_configuration"].values():
                    for budget in result["budgets"].values():
                        budget["orbit_aggregate_rmse"] = 0.0
            with patch(
                "experiments.summarize_plot_inside_greedy_fixed_k_lambda.validate_report",
                side_effect=audits,
            ):
                summary = build_fixed_k_lambda_summary(airport_path, voting_path)
            selection = build_fixed_k_lambda_locked_selection(summary)

        self.assertTrue(all(row["selection_eligible"] for row in summary["rows"]))
        self.assertEqual(selection["locked_configuration"]["lambda0"], 0.25)
        self.assertIsNone(
            summary["rows"][0]["rmse_ratio_to_inside_orbit"]["joint"]
        )

    def test_strict_raw_validation_and_equal_cell_aggregation(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            seen = []

            def validator(report: dict) -> dict:
                seen.append(report)
                return _audit(str(report["dataset"]))

            with patch(
                "experiments.summarize_plot_inside_greedy_fixed_k_lambda.validate_report",
                side_effect=validator,
            ):
                summary = build_fixed_k_lambda_summary(
                    airport_path, voting_path, fixed_k=64
                )

        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0]["raw_cells"], [{"source": 1}])
        self.assertEqual(summary["protocol"]["lambda0_values"], list(LAMBDA_VALUES))
        rows = {row["lambda0"]: row for row in summary["rows"]}
        self.assertAlmostEqual(
            rows[0.125]["rmse_ratio_to_lambda0_reference"]["airport"], 0.5
        )
        self.assertAlmostEqual(
            rows[0.125]["rmse_ratio_to_lambda0_reference"]["voting"], 2.0
        )
        # Equal weighting of 2 datasets x 2 budgets: sqrt(0.5 * 2) = 1.
        self.assertAlmostEqual(
            rows[0.125]["rmse_ratio_to_lambda0_reference"]["joint"], 1.0
        )
        self.assertAlmostEqual(
            rows[0.25]["rmse_ratio_to_lambda0_reference"]["joint"], 0.75
        )
        self.assertAlmostEqual(
            rows[0.125]["design_time_ratio_to_lambda0_reference"]["joint"],
            4.0,
        )
        self.assertTrue(
            summary["source_validation"]
            ["all_rmse_and_aggregates_independently_recomputed_from_raw_cells"]
        )

    def test_coverage_failure_is_retained_and_rmse_ratio_suppressed(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            audits = [_audit("airport", fail_lambda=0.25), _audit("voting")]
            with patch(
                "experiments.summarize_plot_inside_greedy_fixed_k_lambda.validate_report",
                side_effect=audits,
            ):
                summary = build_fixed_k_lambda_summary(airport_path, voting_path)

        row = next(row for row in summary["rows"] if row["lambda0"] == 0.25)
        self.assertEqual(row["candidate_coverage_failures"], 1)
        self.assertFalse(row["selection_eligible"])
        self.assertIsNone(row["rmse_ratio_to_lambda0_reference"]["airport"])
        self.assertIsNone(row["rmse_ratio_to_lambda0_reference"]["joint"])
        self.assertAlmostEqual(
            row["design_time_ratio_to_lambda0_reference"]["joint"], 8.0
        )

    def test_locked_selection_only_allows_fixed_k_lambda_candidates(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            with patch(
                "experiments.summarize_plot_inside_greedy_fixed_k_lambda.validate_report",
                side_effect=[_audit("airport"), _audit("voting")],
            ):
                summary = build_fixed_k_lambda_summary(airport_path, voting_path)
            selection = build_fixed_k_lambda_locked_selection(summary)

        self.assertEqual(selection["status"], "configuration_locked")
        self.assertEqual(selection["selection_rule"]["cell_count"], 4)
        self.assertEqual(selection["locked_configuration"]["candidate_pool"], 64)
        self.assertEqual(selection["locked_configuration"]["lambda0"], 0.25)
        default = next(
            row for row in selection["ranking"] if row["is_registered_default"]
        )
        self.assertTrue(default["selection_eligible"])
        self.assertFalse(default["eligible_for_lambda_lock"])
        self.assertEqual(
            {
                (row["lambda0"], row["candidate_pool"])
                for row in selection["ranking"]
            },
            {(0.0, 64), (0.125, 64), (0.25, 64), (1.0, 4)},
        )
        self.assertEqual(
            selection["screening_sources"],
            summary["source_validation"]["sources"],
        )
        self.assertIsNone(selection["holdout_validation"])

    def test_plot_exports_are_near_square(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            with patch(
                "experiments.summarize_plot_inside_greedy_fixed_k_lambda.validate_report",
                side_effect=[_audit("airport"), _audit("voting")],
            ):
                summary = build_fixed_k_lambda_summary(airport_path, voting_path)
            rmse_png, rmse_pdf = plot_lambda_rmse_ratios(
                summary, Path(directory) / "rmse.png"
            )
            time_png, time_pdf = plot_lambda_design_time_ratios(
                summary, Path(directory) / "time.png"
            )
            for path in (rmse_png, rmse_pdf, time_png, time_pdf):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 5_000)
            for path in (rmse_png, time_png):
                image = plt.imread(path)
                height, width = image.shape[:2]
                self.assertGreater(height / width, 0.80)
                self.assertLess(height / width, 1.20)

    def test_reference_zero_is_required(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            airport = _audit("airport")
            voting = _audit("voting")
            airport["evaluated_configurations"] = airport[
                "evaluated_configurations"
            ][1:]
            voting["evaluated_configurations"] = voting[
                "evaluated_configurations"
            ][1:]
            with patch(
                "experiments.summarize_plot_inside_greedy_fixed_k_lambda.validate_report",
                side_effect=[airport, voting],
            ):
                with self.assertRaisesRegex(ValueError, "reference is absent"):
                    build_fixed_k_lambda_summary(airport_path, voting_path)


if __name__ == "__main__":
    unittest.main()

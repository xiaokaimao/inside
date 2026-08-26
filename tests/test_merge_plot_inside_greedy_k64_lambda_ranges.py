from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from experiments.merge_plot_inside_greedy_k64_lambda_ranges import (
    FIXED_K,
    LOWER_LAMBDAS,
    MERGED_LAMBDAS,
    UPPER_LAMBDAS,
    build_merged_k64_lambda_summary,
    plot_merged_k64_lambda_summary,
)


def _budget(rmse: float) -> dict:
    return {
        "aggregate_rmse": rmse,
        "coverage_failures": 0,
    }


def _audit(dataset: str, range_name: str) -> dict:
    lambdas = LOWER_LAMBDAS if range_name == "lower" else UPPER_LAMBDAS
    base = (2.0, 4.0) if dataset == "airport" else (1.0, 2.0)
    factors = {value: 1.0 + value / 4.0 for value in MERGED_LAMBDAS}
    if dataset == "airport":
        factors[0.5] = 0.5
    else:
        factors[0.5] = 2.0
    results = {}
    for lambda0 in lambdas:
        results[f"lambda0={lambda0:.12g}|K=64"] = {
            "lambda0": lambda0,
            "candidate_pool": 64,
            "budgets": {
                str(index): _budget(base[index] * factors[lambda0])
                for index in (0, 1)
            },
        }
    results["lambda0=1|K=4"] = {
        "lambda0": 1.0,
        "candidate_pool": 4,
        "budgets": {str(index): _budget(base[index]) for index in (0, 1)},
    }
    return {
        "passed": True,
        "dataset": dataset,
        "stage": "screening",
        "repeats": 2,
        "base_seed": 20260911,
        "budget_multipliers": [500, 2000],
        "selected_budget_indices": [0, 1],
        "selected_budget_multipliers": [500, 2000],
        "evaluated_configurations": [
            *[
                {"lambda0": value, "candidate_pool": FIXED_K}
                for value in lambdas
            ],
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "all_rmse_and_aggregates_independently_recomputed": True,
        "all_greedy_hyperparameters_share_size_schedules": True,
        "all_greedy_hyperparameters_share_relabels": True,
        "recomputed_results_by_configuration": results,
    }


def _quarter_raw_cells(dataset: str) -> list[dict]:
    rows = []
    for repeat in range(2):
        for budget_index in (0, 1):
            calls = 100 * (500 if budget_index == 0 else 2000)
            estimate = [repeat + 0.1 * budget_index, 0.25]
            rows.append(
                {
                    "dataset": dataset,
                    "method": "inside_greedy",
                    "lambda0": 0.25,
                    "candidate_pool": 64,
                    "repeat": repeat,
                    "budget_index": budget_index,
                    "base_seed": 20260911,
                    "seed": 19_000 + repeat * 10 + budget_index,
                    "inner_calls": calls,
                    "total_calls": calls + 202,
                    "truth": [0.0, 0.25],
                    "status": "ok",
                    "coverage_failure": False,
                    "failure_reason": None,
                    "estimate": estimate,
                    "rmse": abs(estimate[0]) + 0.5,
                    "planned_total_utility_calls": calls + 202,
                    "actual_utility_calls": calls + 202,
                    "design_seconds": 10.0,
                    "utility_seconds": 0.01,
                    "diagnostics": {
                        "size_schedule_sha256": "a" * 64,
                        "relabel_permutation_sha256": "b" * 64,
                        "coverage": {"all_player_size_strata_covered": True},
                    },
                }
            )
    return rows


def _report(dataset: str, range_name: str) -> dict:
    lambdas = LOWER_LAMBDAS if range_name == "lower" else UPPER_LAMBDAS
    return {
        "status": "complete",
        "experiment": "inside_greedy_per_size_ratio_hyperparameter_sweep",
        "marker": f"{range_name}:{dataset}",
        "configuration": {
            "dataset": dataset,
            "stage": "screening",
            "repeats": 2,
            "base_seed": 20260911,
            "budget_multipliers": [500, 2000],
            "selected_budget_indices": [0, 1],
            "selected_inner_utility_call_budgets": [50_000, 200_000],
            "inside_greedy_design": "per_size_frame_coupled_design",
            "inside_greedy_estimator": "ofa_conditional_mean_ratio_missing_raise",
            "mean_balance_mode": "normalized",
            "processes": 64,
            "requested_configurations": [
                {"lambda0": value, "candidate_pool": 64} for value in lambdas
            ],
            "evaluated_configurations": [
                {"lambda0": value, "candidate_pool": 64} for value in lambdas
            ],
        },
        "raw_cells": _quarter_raw_cells(dataset),
    }


class MergeK64LambdaRangeTests(unittest.TestCase):
    def _fixture(self, directory: str) -> tuple[dict[str, Path], dict[str, dict]]:
        paths = {}
        reports = {}
        for range_name in ("lower", "upper"):
            for dataset in ("airport", "voting"):
                key = f"{range_name}_{dataset}"
                report = _report(dataset, range_name)
                path = Path(directory) / f"{key}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths[key] = path
                reports[key] = report
        return paths, reports

    @staticmethod
    def _validator(report: dict) -> dict:
        range_name, dataset = report["marker"].split(":")
        return _audit(dataset, range_name)

    def _build(self, paths: dict[str, Path]):
        return build_merged_k64_lambda_summary(
            paths["lower_airport"],
            paths["lower_voting"],
            paths["upper_airport"],
            paths["upper_voting"],
        )

    def test_four_independent_validations_exact_overlap_and_equal_cell_merge(self) -> None:
        with TemporaryDirectory() as directory:
            paths, _ = self._fixture(directory)
            with patch(
                "experiments.merge_plot_inside_greedy_k64_lambda_ranges.validate_report",
                side_effect=self._validator,
            ) as validator:
                summary = self._build(paths)

        self.assertEqual(validator.call_count, 4)
        self.assertEqual(summary["protocol"]["merged_lambda0_values"], list(MERGED_LAMBDAS))
        self.assertEqual(len(summary["rows"]), len(MERGED_LAMBDAS))
        quarter = next(row for row in summary["rows"] if row["lambda0"] == 0.25)
        self.assertEqual(quarter["source_range"], "both_exact_duplicate")
        for dataset in ("airport", "voting"):
            overlap = summary["source_validation"]["quarter_lambda_overlap"][dataset]
            self.assertTrue(overlap["passed"])
            self.assertEqual(overlap["cells_checked"], 4)
            self.assertTrue(overlap["estimates_exactly_equal"])
            self.assertTrue(overlap["design_hashes_exactly_equal"])
            self.assertTrue(overlap["physical_calls_exactly_equal"])
        half = next(row for row in summary["rows"] if row["lambda0"] == 0.5)
        # Four equal-weight cells have ratios 0.5, 0.5, 2, 2.
        self.assertAlmostEqual(
            half["joint_equal_cell_rmse_ratio_to_lambda0_zero"], 1.0
        )

    def test_overlap_estimate_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths, reports = self._fixture(directory)
            reports["upper_airport"]["raw_cells"][0]["estimate"][0] += 1.0
            paths["upper_airport"].write_text(
                json.dumps(reports["upper_airport"]), encoding="utf-8"
            )
            with patch(
                "experiments.merge_plot_inside_greedy_k64_lambda_ranges.validate_report",
                side_effect=self._validator,
            ):
                with self.assertRaisesRegex(ValueError, "differs on estimate"):
                    self._build(paths)

    def test_protocol_mismatch_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            paths, reports = self._fixture(directory)
            reports["upper_voting"]["configuration"]["processes"] = 32
            paths["upper_voting"].write_text(
                json.dumps(reports["upper_voting"]), encoding="utf-8"
            )
            with patch(
                "experiments.merge_plot_inside_greedy_k64_lambda_ranges.validate_report",
                side_effect=self._validator,
            ):
                with self.assertRaisesRegex(ValueError, "raw protocols"):
                    self._build(paths)

    def test_wrong_fixed_k_grid_is_rejected(self) -> None:
        def bad_validator(report: dict) -> dict:
            result = self._validator(report)
            if report["marker"] == "upper:airport":
                result = copy.deepcopy(result)
                result["evaluated_configurations"][0]["candidate_pool"] = 32
            return result

        with TemporaryDirectory() as directory:
            paths, _ = self._fixture(directory)
            with patch(
                "experiments.merge_plot_inside_greedy_k64_lambda_ranges.validate_report",
                side_effect=bad_validator,
            ):
                with self.assertRaisesRegex(ValueError, "wrong K=64 lambda grid"):
                    self._build(paths)

    def test_static_outputs_are_near_square(self) -> None:
        with TemporaryDirectory() as directory:
            paths, _ = self._fixture(directory)
            with patch(
                "experiments.merge_plot_inside_greedy_k64_lambda_ranges.validate_report",
                side_effect=self._validator,
            ):
                summary = self._build(paths)
            png, pdf = plot_merged_k64_lambda_summary(
                summary, Path(directory) / "lambda.png"
            )
            self.assertTrue(png.exists())
            self.assertTrue(pdf.exists())
            self.assertGreater(png.stat().st_size, 5_000)
            self.assertGreater(pdf.stat().st_size, 5_000)
            image = plt.imread(png)
            height, width = image.shape[:2]
            self.assertGreater(height / width, 0.80)
            self.assertLess(height / width, 1.20)


if __name__ == "__main__":
    unittest.main()

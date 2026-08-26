from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from experiments.validate_inside_greedy_fixed_k_lambda_holdout import (
    LOCKED_CONFIGURATION,
    build_fixed_k_lambda_holdout_validation,
    plot_locked_vs_zero_holdout,
)


def _rms(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _budget(
    values: list[float | None],
    orbit_values: list[float],
    *,
    failures: int = 0,
) -> dict:
    finite = [float(value) for value in values if value is not None]
    return {
        "aggregate_rmse": _rms(finite) if failures == 0 else None,
        "coverage_failures": failures,
        "rmse_by_repeat": values,
        "orbit_aggregate_rmse": _rms(orbit_values),
        "orbit_rmse_by_repeat": orbit_values,
    }


def _audit(
    dataset: str,
    *,
    stage: str = "validation",
    base_seed: int = 20261031,
    zero_failure: bool = False,
) -> dict:
    results = {}
    scales = (1.0, 2.0, 4.0)
    specifications = {
        (1.0 / 16.0, 64): [1.0, 3.0],
        (0.0, 64): [2.0, 6.0],
        (1.0, 4): [4.0, 12.0],
    }
    orbit_base = [5.0, 15.0]
    for (lambda0, pool), base_values in specifications.items():
        budgets = {}
        for budget_index, scale in enumerate(scales):
            values: list[float | None] = [
                value * scale for value in base_values
            ]
            failures = 0
            if (lambda0, pool) == (0.0, 64) and zero_failure and budget_index == 0:
                values[0] = None
                failures = 1
            budgets[str(budget_index)] = _budget(
                values,
                [value * scale for value in orbit_base],
                failures=failures,
            )
        results[f"lambda0={lambda0:.12g}|K={pool}"] = {
            "lambda0": lambda0,
            "candidate_pool": pool,
            "budgets": budgets,
        }
    return {
        "dataset": dataset,
        "stage": stage,
        "repeats": 2,
        "base_seed": base_seed,
        "budget_multipliers": [500, 2000, 10000],
        "selected_budget_indices": [0, 1, 2],
        "selected_budget_multipliers": [500, 2000, 10000],
        "evaluated_configurations": [
            {"lambda0": 1.0 / 16.0, "candidate_pool": 64},
            {"lambda0": 0.0, "candidate_pool": 64},
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "requested_configurations": [
            {"lambda0": 1.0 / 16.0, "candidate_pool": 64},
            {"lambda0": 0.0, "candidate_pool": 64},
        ],
        "all_rmse_and_aggregates_independently_recomputed": True,
        "holdout_stage_eligible_pending_cross_report_seed_audit": (
            stage == "validation"
        ),
        "recomputed_results_by_configuration": results,
    }


def _selection(*, lambda0: float = 1.0 / 16.0) -> dict:
    return {
        "status": "configuration_locked",
        "holdout_validation": None,
        "locked_configuration": {
            "lambda0": lambda0,
            "candidate_pool": 64,
        },
        "ranking": [
            {"lambda0": 0.0, "candidate_pool": 64},
            {"lambda0": 1.0 / 16.0, "candidate_pool": 64},
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "screening_contract": {
            "base_seed": 20260911,
            "repeats": 8,
        },
        "selection_rule": {"budget_indices": [0, 1]},
    }


class FixedKLambdaHoldoutTests(unittest.TestCase):
    def _paths(self, directory: str) -> tuple[Path, Path, Path]:
        root = Path(directory)
        selection = root / "selection.json"
        airport = root / "airport.json"
        voting = root / "voting.json"
        selection.write_text(json.dumps(_selection()), encoding="utf-8")
        airport.write_text(
            json.dumps(
                {
                    "dataset": "airport",
                    "configuration": {
                        "all_total_utility_call_budgets": [50202, 200202, 1000202]
                    },
                    "raw_cells": [{"source": "airport"}],
                }
            ),
            encoding="utf-8",
        )
        voting.write_text(
            json.dumps(
                {
                    "dataset": "voting",
                    "configuration": {
                        "all_total_utility_call_budgets": [25604, 102104, 510104]
                    },
                    "raw_cells": [{"source": "voting"}],
                }
            ),
            encoding="utf-8",
        )
        return selection, airport, voting

    def test_validates_locked_zero_default_and_orbit_with_paired_bootstrap(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            selection, airport, voting = self._paths(directory)
            with patch(
                "experiments.validate_inside_greedy_fixed_k_lambda_holdout.validate_report",
                side_effect=[_audit("airport"), _audit("voting")],
            ):
                result = build_fixed_k_lambda_holdout_validation(
                    selection, airport, voting
                )

        self.assertEqual(result["status"], "holdout_validated")
        self.assertEqual(
            (
                result["locked_configuration"]["lambda0"],
                result["locked_configuration"]["candidate_pool"],
            ),
            LOCKED_CONFIGURATION,
        )
        expected = {
            "lambda0_zero_k64": 0.5,
            "registered_default": 0.25,
            "inside_orbit": 0.2,
        }
        for name, ratio in expected.items():
            comparison = result["comparisons"][name]
            self.assertAlmostEqual(
                comparison["joint_equal_cell_geometric_mean_rmse_ratio"],
                ratio,
            )
            interval = comparison["paired_repeat_bootstrap"][
                "geometric_mean_ratio_95_percent_ci"
            ]
            self.assertAlmostEqual(interval[0], ratio)
            self.assertAlmostEqual(interval[1], ratio)
            self.assertEqual(
                comparison["paired_repeat_bootstrap"]["invalid_joint_draws"],
                0,
            )
        self.assertEqual(
            result["bootstrap"]["unit"],
            "repeat within each dataset-budget cell",
        )

    def test_coverage_failure_is_retained_without_discarding_other_comparisons(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            selection, airport, voting = self._paths(directory)
            with patch(
                "experiments.validate_inside_greedy_fixed_k_lambda_holdout.validate_report",
                side_effect=[
                    _audit("airport", zero_failure=True),
                    _audit("voting"),
                ],
            ):
                result = build_fixed_k_lambda_holdout_validation(
                    selection, airport, voting
                )

        self.assertEqual(result["status"], "holdout_inconclusive_coverage_failure")
        self.assertEqual(
            result["coverage_failures_by_method"]["lambda0_zero_k64"]["total"],
            1,
        )
        zero = result["comparisons"]["lambda0_zero_k64"]
        self.assertFalse(zero["complete_coverage"])
        self.assertIsNone(zero["joint_equal_cell_geometric_mean_rmse_ratio"])
        self.assertIsNone(zero["paired_repeat_bootstrap"])
        self.assertTrue(
            result["comparisons"]["registered_default"]["complete_coverage"]
        )
        self.assertIsNotNone(
            result["comparisons"]["registered_default"][
                "paired_repeat_bootstrap"
            ]
        )

    def test_rejects_a_selection_that_did_not_lock_one_sixteenth(self) -> None:
        with TemporaryDirectory() as directory:
            selection, airport, voting = self._paths(directory)
            selection.write_text(json.dumps(_selection(lambda0=0.0)), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "lambda0=1/16"):
                build_fixed_k_lambda_holdout_validation(
                    selection, airport, voting
                )

    def test_rejects_screening_seed_reuse(self) -> None:
        with TemporaryDirectory() as directory:
            selection, airport, voting = self._paths(directory)
            with patch(
                "experiments.validate_inside_greedy_fixed_k_lambda_holdout.validate_report",
                side_effect=[
                    _audit("airport", base_seed=20260911),
                    _audit("voting", base_seed=20260911),
                ],
            ):
                with self.assertRaisesRegex(ValueError, "reuses screening seed"):
                    build_fixed_k_lambda_holdout_validation(
                        selection, airport, voting
                    )

    def test_zero_comparison_plot_is_one_near_square_png_and_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            selection, airport, voting = self._paths(directory)
            with patch(
                "experiments.validate_inside_greedy_fixed_k_lambda_holdout.validate_report",
                side_effect=[_audit("airport"), _audit("voting")],
            ):
                result = build_fixed_k_lambda_holdout_validation(
                    selection, airport, voting
                )
            png, pdf = plot_locked_vs_zero_holdout(
                result, Path(directory) / "locked_vs_zero.png"
            )
            for path in (png, pdf):
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 5_000)
            image = plt.imread(png)
            height, width = image.shape[:2]
            self.assertGreater(height / width, 0.80)
            self.assertLess(height / width, 1.20)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.run_inside_greedy_hyperparameter_sweep import run_experiment
from experiments.select_plot_inside_greedy_hyperparameter_sweep import (
    attach_holdout_validation,
    build_joint_selection,
    plot_heatmaps,
)
from experiments.validate_inside_greedy_hyperparameter_sweep import (
    validate_report,
)


class SweepValidatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_experiment(
            dataset="voting",
            configurations=((1.0, 4),),
            stage="screening",
            budget_multipliers=(48,),
            budget_indices=(0,),
            repeats=1,
            base_seed=123,
            processes=1,
        )

    def test_independently_validates_small_real_report(self) -> None:
        result = validate_report(self.report)
        self.assertTrue(result["passed"])
        self.assertTrue(result["all_rmse_and_aggregates_independently_recomputed"])
        self.assertEqual(result["coverage_failures"], 0)

    def test_rejects_tampered_rmse(self) -> None:
        report = copy.deepcopy(self.report)
        first = next(cell for cell in report["raw_cells"] if cell["status"] == "ok")
        first["rmse"] *= 2.0
        with self.assertRaisesRegex(ValueError, "rmse"):
            validate_report(report)

    def test_rejects_tampered_schedule_hash(self) -> None:
        report = copy.deepcopy(self.report)
        greedy = next(
            cell for cell in report["raw_cells"] if cell["method"] == "inside_greedy"
        )
        greedy["diagnostics"]["size_schedule_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "schedule hash"):
            validate_report(report)

    def test_validation_stage_is_eligible_before_cross_report_seed_audit(self) -> None:
        report = run_experiment(
            dataset="voting",
            configurations=((1.0, 4),),
            stage="validation",
            budget_multipliers=(48,),
            repeats=1,
            base_seed=987654,
            processes=1,
        )
        # Stage eligibility must not trust a self-asserted runner flag.  The
        # selector derives both reports' method seeds and performs that audit.
        report["configuration"].pop(
            "uses_distinct_seed_from_registered_screening", None
        )
        result = validate_report(report)
        self.assertTrue(
            result["holdout_stage_eligible_pending_cross_report_seed_audit"]
        )


def _audit(dataset: str) -> dict:
    ratios = {
        "airport": {
            (0.5, 2): (0.80, 0.90),
            (0.25, 4): (1.20, 1.10),
            (1.0, 4): (1.0, 1.0),
        },
        "voting": {
            (0.5, 2): (1.10, 0.70),
            (0.25, 4): (1.15, 1.05),
            (1.0, 4): (1.0, 1.0),
        },
    }[dataset]
    results = {}
    for config, values in ratios.items():
        config_id = f"lambda0={config[0]:.12g}|K={config[1]}"
        results[config_id] = {
            "budgets": {
                str(index): {
                    "aggregate_rmse_ratio_to_registered_default": ratio,
                    "aggregate_rmse_ratio_to_inside_orbit": ratio * 1.1,
                }
                for index, ratio in zip((0, 2), values, strict=True)
            },
            "overall": {"coverage_failures": 0, "mean_design_seconds": config[1] / 10},
        }
    return {
        "dataset": dataset,
        "stage": "screening",
        "screening_selection_eligible": True,
        "budget_multipliers": [500, 1000, 2000, 5000, 10000],
        "selected_budget_indices": [0, 2],
        "selected_budget_multipliers": [500, 2000],
        "repeats": 3,
        "base_seed": 20260827,
        "evaluated_configurations": [
            {"lambda0": 0.5, "candidate_pool": 2},
            {"lambda0": 0.25, "candidate_pool": 4},
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "recomputed_results_by_configuration": results,
    }


class SweepSelectionPlotTests(unittest.TestCase):
    def test_joint_selection_equal_weights_four_cells_and_plots(self) -> None:
        audits = {dataset: _audit(dataset) for dataset in ("airport", "voting")}
        fake_sources = {
            dataset: {"path": dataset, "sha256": "0" * 64}
            for dataset in audits
        }
        with patch(
            "experiments.select_plot_inside_greedy_hyperparameter_sweep._validated_pair",
            return_value=({}, audits, fake_sources),
        ):
            selection = build_joint_selection(Path("airport"), Path("voting"))
        locked = selection["locked_configuration"]
        self.assertEqual((locked["lambda0"], locked["candidate_pool"]), (0.5, 2))
        expected = math.exp(sum(math.log(value) for value in (0.8, 0.9, 1.1, 0.7)) / 4)
        self.assertAlmostEqual(
            locked["joint_geometric_mean_rmse_ratio_to_default"], expected
        )
        self.assertEqual(selection["selection_rule"]["cell_count"], 4)

        with tempfile.TemporaryDirectory() as directory:
            png, pdf = plot_heatmaps(selection, Path(directory) / "heatmap.png")
            self.assertTrue(png.exists())
            self.assertTrue(pdf.exists())
            self.assertGreater(png.stat().st_size, 10_000)

    def test_holdout_allows_diagnostic_challenger_and_bootstraps_repeats(self) -> None:
        screening_audits = {
            dataset: _audit(dataset) for dataset in ("airport", "voting")
        }
        fake_sources = {
            dataset: {"path": dataset, "sha256": "0" * 64}
            for dataset in screening_audits
        }
        with patch(
            "experiments.select_plot_inside_greedy_hyperparameter_sweep._validated_pair",
            return_value=({}, screening_audits, fake_sources),
        ):
            selection = build_joint_selection(Path("airport"), Path("voting"))

        locked = (0.5, 2)
        default = (1.0, 4)
        diagnostic = (0.25, 4)

        def validation_audit(dataset: str) -> dict:
            results = {}
            for config in (locked, diagnostic, default):
                config_id = f"lambda0={config[0]:.12g}|K={config[1]}"
                budgets = {}
                for budget_index in (0, 1):
                    default_repeats = [2.0, 2.2, 1.8]
                    orbit_repeats = [1.6, 1.7, 1.5]
                    if config == locked:
                        candidate_repeats = [1.0, 1.1, 0.9]
                    elif config == default:
                        candidate_repeats = default_repeats
                    else:
                        candidate_repeats = [1.4, 1.5, 1.3]
                    candidate_rmse = math.sqrt(
                        sum(value * value for value in candidate_repeats) / 3
                    )
                    default_rmse = math.sqrt(
                        sum(value * value for value in default_repeats) / 3
                    )
                    orbit_rmse = math.sqrt(
                        sum(value * value for value in orbit_repeats) / 3
                    )
                    budgets[str(budget_index)] = {
                        "rmse_by_repeat": candidate_repeats,
                        "default_rmse_by_repeat": default_repeats,
                        "orbit_rmse_by_repeat": orbit_repeats,
                        "aggregate_rmse": candidate_rmse,
                        "default_aggregate_rmse": default_rmse,
                        "orbit_aggregate_rmse": orbit_rmse,
                        "aggregate_rmse_ratio_to_registered_default": (
                            candidate_rmse / default_rmse
                        ),
                        "aggregate_rmse_ratio_to_inside_orbit": (
                            candidate_rmse / orbit_rmse
                        ),
                    }
                results[config_id] = {"budgets": budgets}
            return {
                "dataset": dataset,
                "stage": "validation",
                "holdout_stage_eligible_pending_cross_report_seed_audit": True,
                "budget_multipliers": [500, 2000],
                "selected_budget_indices": [0, 1],
                "repeats": 3,
                "base_seed": 998877,
                "evaluated_configurations": [
                    {"lambda0": value, "candidate_pool": pool}
                    for value, pool in (locked, diagnostic, default)
                ],
                "requested_configurations": [
                    {"lambda0": value, "candidate_pool": pool}
                    for value, pool in (locked, diagnostic)
                ],
                "recomputed_results_by_configuration": results,
            }

        validation_audits = {
            dataset: validation_audit(dataset)
            for dataset in ("airport", "voting")
        }
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for dataset in ("airport", "voting"):
                path = Path(directory) / f"{dataset}.json"
                path.write_text(
                    '{"configuration":{"all_total_utility_call_budgets":[100,200]}}',
                    encoding="utf-8",
                )
                paths[dataset] = path
            with patch(
                "experiments.select_plot_inside_greedy_hyperparameter_sweep._validated_pair",
                return_value=({}, validation_audits, fake_sources),
            ):
                attached = attach_holdout_validation(
                    selection, paths["airport"], paths["voting"]
                )

        holdout = attached["holdout_validation"]
        self.assertTrue(holdout["main_curves_ignore_additional_challengers"])
        self.assertEqual(
            holdout["additional_predeclared_diagnostic_challengers"],
            [{"lambda0": 0.25, "candidate_pool": 4}],
        )
        self.assertEqual(holdout["bootstrap"]["joint_cell_count"], 4)
        for field in (
            "joint_geometric_mean_locked_ratio_to_default_bootstrap_95_percent_ci",
            "joint_geometric_mean_locked_ratio_to_orbit_bootstrap_95_percent_ci",
        ):
            lower, upper = holdout[field]
            self.assertLessEqual(lower, upper)
            self.assertTrue(math.isfinite(lower))
            self.assertTrue(math.isfinite(upper))
        point = holdout["curves"]["airport"][0]
        self.assertEqual(
            len(point["locked_ratio_to_default_bootstrap_95_percent_ci"]), 2
        )


if __name__ == "__main__":
    unittest.main()

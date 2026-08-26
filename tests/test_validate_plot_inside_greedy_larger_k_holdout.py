from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from experiments.validate_plot_inside_greedy_larger_k_holdout import (
    BOOTSTRAP_SAMPLES,
    _derived_method_seeds,
    _paired_bootstrap_ratios,
    _screening_paths,
    _seed_separation_audit,
    build_holdout_report,
    plot_dataset_holdout,
)


def _seed_audit(base_seed: int, *, validation: bool) -> dict:
    return {
        "base_seed": base_seed,
        "repeats": 20 if validation else 8,
        "selected_budget_indices": [0, 1, 2] if validation else [0, 1],
    }


def _validation_audit(dataset: str, *, fail_first_locked: bool = False) -> dict:
    results = {}
    specifications = {
        (0.0, 128): 1.0,
        (0.0, 64): 2.0,
        (1.0, 4): 4.0,
    }
    for configuration, value in specifications.items():
        budgets = {}
        for budget_index in (0, 1, 2):
            failure = int(
                fail_first_locked
                and configuration == (0.0, 128)
                and budget_index == 0
            )
            budget = {
                "coverage_failures": failure,
                "rmse_by_repeat": (
                    [None] * 20 if failure else [value] * 20
                ),
            }
            if configuration == (0.0, 128):
                budget["orbit_rmse_by_repeat"] = [5.0] * 20
            budgets[str(budget_index)] = budget
        results[f"lambda0={configuration[0]:.12g}|K={configuration[1]}"] = {
            "budgets": budgets
        }
    return {
        "dataset": dataset,
        "stage": "validation",
        "holdout_stage_eligible_pending_cross_report_seed_audit": True,
        "base_seed": 20261009,
        "repeats": 20,
        "num_players": 100 if dataset == "airport" else 51,
        "selected_budget_indices": [0, 1, 2],
        "selected_budget_multipliers": [500, 2000, 10000],
        "evaluated_configurations": [
            {"lambda0": 0.0, "candidate_pool": 128},
            {"lambda0": 0.0, "candidate_pool": 64},
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "requested_configurations": [
            {"lambda0": 0.0, "candidate_pool": 128},
            {"lambda0": 0.0, "candidate_pool": 64},
        ],
        "recomputed_results_by_configuration": results,
    }


def _screening_audits() -> dict:
    return {
        report_set: {
            dataset: _seed_audit(20260911, validation=False)
            for dataset in ("airport", "voting")
        }
        for report_set in ("base", "extension")
    }


class LargerKHoldoutTests(unittest.TestCase):
    def test_paired_bootstrap_preserves_exact_ratio(self) -> None:
        methods = {
            "inside_greedy_k128": np.ones(20),
            "inside_greedy_k64": np.full(20, 2.0),
            "registered_default": np.full(20, 4.0),
            "inside_orbit": np.full(20, 5.0),
        }
        draws = _paired_bootstrap_ratios(
            methods, rng=np.random.default_rng(123)
        )
        self.assertEqual(draws["inside_greedy_k64"].shape, (BOOTSTRAP_SAMPLES,))
        self.assertTrue(np.all(draws["inside_greedy_k64"] == 0.5))
        self.assertTrue(np.all(draws["registered_default"] == 0.25))
        self.assertTrue(np.all(draws["inside_orbit"] == 0.2))

    def test_seed_audit_checks_every_screening_report_and_rejects_overlap(self) -> None:
        screening = _screening_audits()
        validation = {
            dataset: _seed_audit(20261009, validation=True)
            for dataset in ("airport", "voting")
        }
        result = _seed_separation_audit(screening, validation)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["pairwise_intersection_counts"]), 8)
        self.assertTrue(
            all(value == 0 for value in result["pairwise_intersection_counts"].values())
        )
        validation["airport"] = _seed_audit(20260911, validation=False)
        with self.assertRaisesRegex(ValueError, "derived method seeds overlap"):
            _seed_separation_audit(screening, validation)

    def _build_with_fake_audits(
        self, directory: str, *, fail_first_locked: bool = False
    ) -> dict:
        locked = Path(directory) / "locked.json"
        airport = Path(directory) / "validation_airport.json"
        voting = Path(directory) / "validation_voting.json"
        locked.write_text(
            json.dumps(
                {
                    "screening_sources": {"synthetic": True},
                    "screening_overlap_merge_audit": {"passed": True},
                }
            ),
            encoding="utf-8",
        )
        airport.write_text(
            json.dumps(
                {
                    "configuration": {
                        "all_total_utility_call_budgets": [100, 200, 400]
                    }
                }
            ),
            encoding="utf-8",
        )
        voting.write_text(
            json.dumps(
                {
                    "configuration": {
                        "all_total_utility_call_budgets": [51, 102, 204]
                    }
                }
            ),
            encoding="utf-8",
        )
        reports = {
            "airport": json.loads(airport.read_text(encoding="utf-8")),
            "voting": json.loads(voting.read_text(encoding="utf-8")),
        }
        audits = {
            dataset: _validation_audit(
                dataset, fail_first_locked=fail_first_locked
            )
            for dataset in ("airport", "voting")
        }
        fake_screening_paths = {
            report_set: {
                dataset: Path(directory) / f"{report_set}_{dataset}.json"
                for dataset in ("airport", "voting")
            }
            for report_set in ("base", "extension")
        }
        with (
            patch(
                "experiments.validate_plot_inside_greedy_larger_k_holdout._revalidate_screening_lock",
                return_value=(fake_screening_paths, _screening_audits()),
            ),
            patch(
                "experiments.validate_plot_inside_greedy_larger_k_holdout._load_validated_pair",
                return_value=(reports, audits),
            ),
        ):
            return build_holdout_report(locked, airport, voting)

    def test_builds_six_cell_bootstrap_report_and_four_line_plots(self) -> None:
        with TemporaryDirectory() as directory:
            report = self._build_with_fake_audits(directory)
            self.assertEqual(report["status"], "holdout_validated")
            self.assertTrue(report["seed_separation_audit"]["passed"])
            self.assertEqual(
                report["registered_primary_comparison"]["six_cell_joint"][
                    "point_estimate"
                ],
                0.5,
            )
            joint = report["six_cell_joint_locked_ratios"]
            self.assertEqual(joint["inside_greedy_k64"]["point_estimate"], 0.5)
            self.assertEqual(joint["registered_default"]["point_estimate"], 0.25)
            self.assertEqual(joint["inside_orbit"]["point_estimate"], 0.2)
            self.assertEqual(
                joint["inside_greedy_k64"]["paired_bootstrap_95_percent_ci"],
                [0.5, 0.5],
            )
            self.assertEqual(
                len(report["datasets"]["airport"]["cells"]), 3
            )
            cell = report["datasets"]["airport"]["cells"][0]
            self.assertEqual(set(cell["methods"]), {
                "inside_greedy_k128",
                "inside_greedy_k64",
                "registered_default",
                "inside_orbit",
            })

            for dataset in ("airport", "voting"):
                png, pdf = plot_dataset_holdout(
                    report, dataset, Path(directory) / f"{dataset}.png"
                )
                self.assertTrue(png.exists())
                self.assertTrue(pdf.exists())
                image = plt.imread(png)
                height, width = image.shape[:2]
                self.assertGreater(height / width, 0.80)
                self.assertLess(height / width, 1.20)

    def test_coverage_failure_is_retained_and_invalidates_joint(self) -> None:
        with TemporaryDirectory() as directory:
            report = self._build_with_fake_audits(
                directory, fail_first_locked=True
            )
        self.assertEqual(report["status"], "holdout_coverage_failure")
        self.assertFalse(report["all_six_cells_eligible"])
        cell = report["datasets"]["airport"]["cells"][0]
        self.assertFalse(cell["eligible"])
        self.assertEqual(cell["coverage_failures"]["inside_greedy_k128"], 1)
        self.assertIsNone(
            report["six_cell_joint_locked_ratios"]["inside_greedy_k64"][
                "point_estimate"
            ]
        )

    def test_screening_hash_change_is_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source.json"
            source.write_text('{"value":1}', encoding="utf-8")
            import hashlib

            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            selection = {
                "screening_sources": {
                    report_set: {
                        dataset: {"path": str(source), "sha256": digest}
                        for dataset in ("airport", "voting")
                    }
                    for report_set in ("base", "extension")
                }
            }
            _screening_paths(selection)
            source.write_text('{"value":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash changed"):
                _screening_paths(selection)

    def test_derived_seed_count_has_no_internal_collision(self) -> None:
        audit = _seed_audit(20261009, validation=True)
        self.assertEqual(len(_derived_method_seeds(audit)), 20 * 3 * 2)


if __name__ == "__main__":
    unittest.main()

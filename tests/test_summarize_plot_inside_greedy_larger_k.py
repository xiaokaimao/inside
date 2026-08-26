from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from experiments.summarize_plot_inside_greedy_larger_k import (
    _geometric_mean,
    _safe_ratio,
    build_larger_k_summary,
    build_locked_selection_artifact,
    plot_design_time_ratios,
    plot_rmse_ratios,
)


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
        "orbit_aggregate_rmse": 1.5,
    }


def _audit(dataset: str, *, fail_k32: bool = False) -> dict:
    # The two datasets deliberately move in opposite directions at K=8 so the
    # test catches accidental player- or dataset-size weighting of the joint.
    if dataset == "airport":
        k8 = (2.0, 4.0)
        k16 = (1.0, 2.0)
    else:
        k8 = (1.0, 2.0)
        k16 = (2.0, 4.0)
    results = {
        "lambda0=0|K=8": {
            "lambda0": 0.0,
            "candidate_pool": 8,
            "budgets": {
                "0": _budget(k8[0], 0.5),
                "1": _budget(k8[1], 1.0),
            },
        },
        "lambda0=0|K=16": {
            "lambda0": 0.0,
            "candidate_pool": 16,
            "budgets": {
                "0": _budget(k16[0], 1.0),
                "1": _budget(k16[1], 2.0),
            },
        },
        "lambda0=0|K=32": {
            "lambda0": 0.0,
            "candidate_pool": 32,
            "budgets": {
                "0": _budget(None if fail_k32 else 0.8, 2.0, failures=int(fail_k32)),
                "1": _budget(1.6, 4.0),
            },
        },
        "lambda0=1|K=4": {
            "lambda0": 1.0,
            "candidate_pool": 4,
            "budgets": {
                "0": _budget(2.0, 0.25),
                "1": _budget(4.0, 0.5),
            },
        },
    }
    return {
        "dataset": dataset,
        "stage": "validation",
        "repeats": 3,
        "base_seed": 777,
        "budget_multipliers": [500, 2000],
        "selected_budget_indices": [0, 1],
        "selected_budget_multipliers": [500, 2000],
        "evaluated_configurations": [
            {"lambda0": 0.0, "candidate_pool": 8},
            {"lambda0": 0.0, "candidate_pool": 16},
            {"lambda0": 0.0, "candidate_pool": 32},
            {"lambda0": 1.0, "candidate_pool": 4},
        ],
        "all_rmse_and_aggregates_independently_recomputed": True,
        "all_greedy_hyperparameters_share_size_schedules": True,
        "all_greedy_hyperparameters_share_relabels": True,
        "recomputed_results_by_configuration": results,
    }


def _merge_audit(dataset: str, *, extension: bool) -> dict:
    if extension:
        values = {
            (0.0, 64): (0.8, 10.0),
            (0.0, 128): (0.7, 20.0),
            (1.0, 4): (2.0, 1.5),
        }
    else:
        values = {
            (0.0, 16): (1.0, 1.0),
            (0.0, 32): (0.9, 2.0),
            (0.0, 64): (0.8, 4.0),
            (1.0, 4): (2.0, 0.5),
        }
    results = {
        f"lambda0={lambda0:.12g}|K={pool}": {
            "lambda0": lambda0,
            "candidate_pool": pool,
            "budgets": {
                "0": _budget(rmse, seconds),
                "1": _budget(rmse, seconds * 2.0),
            },
        }
        for (lambda0, pool), (rmse, seconds) in values.items()
    }
    return {
        "dataset": dataset,
        "stage": "screening",
        "num_players": 100 if dataset == "airport" else 51,
        "repeats": 1,
        "base_seed": 20260911,
        "budget_multipliers": [500, 2000],
        "selected_budget_indices": [0, 1],
        "selected_budget_multipliers": [500, 2000],
        "evaluated_configurations": [
            {"lambda0": lambda0, "candidate_pool": pool}
            for lambda0, pool in values
        ],
        "all_rmse_and_aggregates_independently_recomputed": True,
        "all_greedy_hyperparameters_share_size_schedules": True,
        "all_greedy_hyperparameters_share_relabels": True,
        "recomputed_results_by_configuration": results,
    }


def _overlap_raw_cells(dataset: str, *, extension: bool) -> list[dict]:
    timing_scale = 2.0 if extension else 1.0
    cells = []
    for budget_index in (0, 1):
        for lambda0, pool, estimate, rmse, seed in (
            (0.0, 64, [0.4, 0.6], 0.8, 11 + budget_index),
            (1.0, 4, [0.2, 0.8], 2.0, 11 + budget_index),
        ):
            cells.append(
                {
                    "dataset": dataset,
                    "method": "inside_greedy",
                    "lambda0": lambda0,
                    "candidate_pool": pool,
                    "repeat": 0,
                    "budget_index": budget_index,
                    "seed": seed,
                    "status": "ok",
                    "estimate": estimate,
                    "rmse": rmse,
                    "actual_utility_calls": 500 * (budget_index + 1),
                    "planned_total_utility_calls": 500 * (budget_index + 1),
                    "coverage_failure": False,
                    "diagnostics": {
                        "size_schedule_sha256": chr(97 + budget_index) * 64,
                        "relabel_permutation_sha256": chr(98 + budget_index) * 64,
                        "coverage": {"all_player_size_strata_covered": True},
                    },
                    "design_seconds": (
                        timing_scale
                        * (4.0 if pool == 64 else 0.5)
                        * (budget_index + 1)
                    ),
                    "utility_seconds": timing_scale * 0.01 * (budget_index + 1),
                }
            )
        cells.append(
            {
                "dataset": dataset,
                "method": "inside_orbit",
                "repeat": 0,
                "budget_index": budget_index,
                "seed": 22 + budget_index,
                "status": "ok",
                "estimate": [0.3, 0.7],
                "rmse": 1.2,
                "actual_utility_calls": 500 * (budget_index + 1),
                "planned_total_utility_calls": 500 * (budget_index + 1),
                "coverage_failure": False,
                "diagnostics": {
                    "coverage": {"all_player_size_strata_covered": True}
                },
                "design_seconds": timing_scale * 2.0 * (budget_index + 1),
                "utility_seconds": timing_scale * 0.01 * (budget_index + 1),
            }
        )
    return cells


class LargerKSummarizerTests(unittest.TestCase):
    def test_exact_zero_rmse_is_preserved(self) -> None:
        self.assertEqual(_safe_ratio(0.0, 2.0), 0.0)
        self.assertEqual(_geometric_mean([1.0, 0.0]), 0.0)

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

    def test_reduces_validated_raw_cell_audits_and_equal_weights_cells(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            seen_reports = []

            def independent_validator(report: dict) -> dict:
                seen_reports.append(report)
                return _audit(str(report["dataset"]))

            with patch(
                "experiments.summarize_plot_inside_greedy_larger_k.validate_report",
                side_effect=independent_validator,
            ):
                summary = build_larger_k_summary(
                    airport_path, voting_path, reference="k16"
                )

        self.assertEqual(len(seen_reports), 2)
        self.assertEqual(seen_reports[0]["raw_cells"], [{"source": 1}])
        rows = {row["candidate_pool"]: row for row in summary["rows"]}
        self.assertAlmostEqual(rows[8]["rmse_ratio_to_reference"]["airport"], 2.0)
        self.assertAlmostEqual(rows[8]["rmse_ratio_to_reference"]["voting"], 0.5)
        # Equal weighting of four dataset-budget cells gives exactly one.
        self.assertAlmostEqual(rows[8]["rmse_ratio_to_reference"]["joint"], 1.0)
        self.assertAlmostEqual(
            rows[8]["design_time_ratio_to_reference"]["joint"], 0.5
        )
        self.assertAlmostEqual(rows[16]["rmse_ratio_to_reference"]["joint"], 1.0)
        self.assertTrue(
            summary["source_validation"][
                "all_rmse_and_aggregates_independently_recomputed_from_raw_cells"
            ]
        )

    def test_retains_coverage_failure_and_suppresses_affected_ratios(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            audits = [_audit("airport", fail_k32=True), _audit("voting")]
            with patch(
                "experiments.summarize_plot_inside_greedy_larger_k.validate_report",
                side_effect=audits,
            ):
                summary = build_larger_k_summary(airport_path, voting_path)

        row = next(row for row in summary["rows"] if row["candidate_pool"] == 32)
        self.assertFalse(row["selection_eligible"])
        self.assertEqual(row["coverage_failures"], 1)
        self.assertEqual(row["datasets"]["airport"]["coverage_failures"], 1)
        self.assertIsNone(row["rmse_ratio_to_reference"]["airport"])
        self.assertIsNone(row["rmse_ratio_to_reference"]["joint"])
        self.assertTrue(
            math.isfinite(row["design_time_ratio_to_reference"]["joint"])
        )

    def test_registered_default_reference_and_near_square_exports(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            audits = [_audit("airport"), _audit("voting")]
            with patch(
                "experiments.summarize_plot_inside_greedy_larger_k.validate_report",
                side_effect=audits,
            ):
                summary = build_larger_k_summary(
                    airport_path,
                    voting_path,
                    reference="registered-default",
                )
            rows = {row["candidate_pool"]: row for row in summary["rows"]}
            self.assertAlmostEqual(
                rows[16]["rmse_ratio_to_reference"]["airport"], 0.5
            )

            rmse_png, rmse_pdf = plot_rmse_ratios(
                summary, Path(directory) / "rmse.png"
            )
            time_png, time_pdf = plot_design_time_ratios(
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

    def _merge_paths(self, directory: str) -> tuple[Path, Path, Path, Path]:
        paths = []
        for report_set in ("base", "extension"):
            for dataset in ("airport", "voting"):
                path = Path(directory) / f"{report_set}_{dataset}.json"
                path.write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "report_set": report_set,
                            "raw_cells": _overlap_raw_cells(
                                dataset, extension=report_set == "extension"
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
        return paths[0], paths[1], paths[2], paths[3]

    def test_strictly_merges_extension_and_bridges_design_time_at_k64(self) -> None:
        with TemporaryDirectory() as directory:
            base_airport, base_voting, ext_airport, ext_voting = self._merge_paths(
                directory
            )

            def validator(report: dict) -> dict:
                return _merge_audit(
                    str(report["dataset"]),
                    extension=report["report_set"] == "extension",
                )

            with patch(
                "experiments.summarize_plot_inside_greedy_larger_k.validate_report",
                side_effect=validator,
            ):
                summary = build_larger_k_summary(
                    base_airport,
                    base_voting,
                    extension_airport_path=ext_airport,
                    extension_voting_path=ext_voting,
                )

        self.assertEqual(summary["protocol"]["k_values"], [16, 32, 64, 128])
        self.assertTrue(summary["protocol"]["extension_pair_merged"])
        overlap = summary["source_validation"]["overlap_merge_audit"]
        self.assertTrue(overlap["passed"])
        self.assertEqual(overlap["datasets"]["airport"]["raw_cells_compared"], 6)
        rows = {row["candidate_pool"]: row for row in summary["rows"]}
        self.assertEqual(rows[128]["source_report_set"], "extension")
        self.assertAlmostEqual(rows[128]["rmse_ratio_to_reference"]["joint"], 0.7)
        # Extension time 20 is first calibrated by base-K64/ext-K64 = 4/10;
        # then compared with base K16 time 1, giving a ratio of 8 rather than 20.
        self.assertAlmostEqual(
            rows[128]["design_time_ratio_to_reference"]["joint"], 8.0
        )
        point = rows[128]["datasets"]["airport"]["budget_cells"][0]
        self.assertEqual(point["design_time_ratio_method"], "K64-overlap-bridge")
        self.assertEqual(point["candidate_source"], "extension")
        self.assertEqual(point["reference_source"], "base")

        selection = build_locked_selection_artifact(summary)
        self.assertEqual(selection["status"], "configuration_locked")
        self.assertEqual(selection["selection_rule"]["cell_count"], 4)
        self.assertEqual(
            {
                (row["lambda0"], row["candidate_pool"])
                for row in selection["ranking"]
            },
            {(0.0, 16), (0.0, 32), (0.0, 64), (0.0, 128), (1.0, 4)},
        )
        self.assertEqual(
            (
                selection["locked_configuration"]["lambda0"],
                selection["locked_configuration"]["candidate_pool"],
            ),
            (0.0, 128),
        )
        self.assertEqual(
            selection["screening_sources"],
            summary["source_validation"]["sources"],
        )
        self.assertTrue(selection["screening_overlap_merge_audit"]["passed"])
        self.assertIsNone(selection["holdout_validation"])

    def test_merge_rejects_changed_overlap_estimate(self) -> None:
        with TemporaryDirectory() as directory:
            base_airport, base_voting, ext_airport, ext_voting = self._merge_paths(
                directory
            )
            extension = json.loads(ext_airport.read_text(encoding="utf-8"))
            extension["raw_cells"][0]["estimate"][0] += 1e-12
            ext_airport.write_text(json.dumps(extension), encoding="utf-8")

            def validator(report: dict) -> dict:
                return _merge_audit(
                    str(report["dataset"]),
                    extension=report["report_set"] == "extension",
                )

            with patch(
                "experiments.summarize_plot_inside_greedy_larger_k.validate_report",
                side_effect=validator,
            ):
                with self.assertRaisesRegex(ValueError, "overlap mismatch.*estimate"):
                    build_larger_k_summary(
                        base_airport,
                        base_voting,
                        extension_airport_path=ext_airport,
                        extension_voting_path=ext_voting,
                    )

    def test_merge_rejects_changed_overlap_hash(self) -> None:
        with TemporaryDirectory() as directory:
            base_airport, base_voting, ext_airport, ext_voting = self._merge_paths(
                directory
            )
            extension = json.loads(ext_voting.read_text(encoding="utf-8"))
            extension["raw_cells"][0]["diagnostics"][
                "size_schedule_sha256"
            ] = "f" * 64
            ext_voting.write_text(json.dumps(extension), encoding="utf-8")

            def validator(report: dict) -> dict:
                return _merge_audit(
                    str(report["dataset"]),
                    extension=report["report_set"] == "extension",
                )

            with patch(
                "experiments.summarize_plot_inside_greedy_larger_k.validate_report",
                side_effect=validator,
            ):
                with self.assertRaisesRegex(
                    ValueError, "overlap mismatch.*size_schedule_sha256"
                ):
                    build_larger_k_summary(
                        base_airport,
                        base_voting,
                        extension_airport_path=ext_airport,
                        extension_voting_path=ext_voting,
                    )

    def test_merge_requires_both_extension_reports(self) -> None:
        with TemporaryDirectory() as directory:
            airport_path, voting_path = self._paths(directory)
            with self.assertRaisesRegex(ValueError, "supplied together"):
                build_larger_k_summary(
                    airport_path,
                    voting_path,
                    extension_airport_path=airport_path,
                )


if __name__ == "__main__":
    unittest.main()

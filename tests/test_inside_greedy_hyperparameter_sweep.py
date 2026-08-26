from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.run_inside_greedy_hyperparameter_sweep import (
    REGISTERED_CANDIDATE_POOL,
    REGISTERED_LAMBDA0,
    _paired_metrics,
    _run_cell,
    configuration_grid,
    run_experiment,
    selected_budget_indices,
)
from frame_ofa import CoalitionDesign


class InsideGreedyHyperparameterSweepTests(unittest.TestCase):
    def test_grid_keeps_only_one_k1_cell_and_adds_registered_default(self) -> None:
        grid = configuration_grid(
            lambdas=(0.0, 0.25, 1.0, 4.0),
            candidate_pools=(1, 2, 4, 8, 16),
        )
        self.assertEqual(sum(pool == 1 for _, pool in grid), 1)
        self.assertIn((1.0, 1), grid)
        self.assertIn((REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL), grid)
        self.assertEqual(len(grid), 1 + 4 * 4)

    def test_screening_uses_registered_sparse_indices_or_all_custom_budgets(self) -> None:
        self.assertEqual(selected_budget_indices("screening", 5), (0, 2))
        self.assertEqual(selected_budget_indices("validation", 5), (0, 1, 2, 3, 4))
        self.assertEqual(selected_budget_indices("screening", 2), (0, 1))
        self.assertEqual(
            selected_budget_indices("screening", 5, explicit=(1, 4)),
            (1, 4),
        )

    def test_paired_metrics_match_on_repeat_and_budget(self) -> None:
        candidate = [
            {"repeat": 0, "budget_index": 0, "status": "ok", "rmse": 1.0},
            {"repeat": 0, "budget_index": 1, "status": "ok", "rmse": 10.0},
        ]
        reference = [
            {"repeat": 0, "budget_index": 0, "status": "ok", "rmse": 2.0},
            {"repeat": 0, "budget_index": 1, "status": "ok", "rmse": 5.0},
        ]
        paired = _paired_metrics(candidate, reference)
        self.assertEqual(paired["rmse_difference_by_repeat"], [-1.0, 5.0])
        self.assertEqual(paired["rmse_ratio_by_repeat"], [0.5, 2.0])
        self.assertEqual(paired["wins"], 1)
        self.assertEqual(paired["losses"], 1)

    def test_coverage_failure_is_retained_without_utility_evaluation(self) -> None:
        num_players = 51
        coalition = np.zeros((1, num_players), dtype=bool)
        coalition[0, :2] = True
        design = CoalitionDesign(
            coalitions=coalition,
            sizes=np.asarray([2], dtype=np.int64),
            method="frame_coupled_per_size",
            seed=1,
            diagnostics={"relabel_permutation_sha256": "test-hash"},
        )
        payload = {
            "dataset": "voting",
            "method": "inside_greedy",
            "repeat": 0,
            "budget_index": 0,
            "inner_calls": 100,
            "total_calls": 204,
            "base_seed": 7,
            "truth": [0.0] * num_players,
            "lambda0": 1.0,
            "candidate_pool": 4,
        }
        coverage = {
            "all_player_size_strata_covered": False,
            "minimum_inclusion_count": 0,
            "minimum_exclusion_count": 0,
            "missing_inclusion_strata": 1,
            "missing_exclusion_strata": 1,
        }
        with (
            patch(
                "experiments.run_inside_greedy_hyperparameter_sweep.per_size_frame_coupled_design",
                return_value=design,
            ),
            patch(
                "experiments.run_inside_greedy_hyperparameter_sweep._ratio_coverage_diagnostics",
                return_value=coverage,
            ),
            patch(
                "experiments.run_inside_greedy_hyperparameter_sweep._evaluate",
                side_effect=AssertionError("utility must not run after coverage failure"),
            ),
        ):
            cell = _run_cell(payload)
        self.assertEqual(cell["status"], "coverage_failure")
        self.assertTrue(cell["coverage_failure"])
        self.assertIsNone(cell["rmse"])
        self.assertEqual(cell["actual_utility_calls"], 0)
        self.assertIn("size_schedule_sha256", cell["diagnostics"])
        self.assertEqual(
            cell["diagnostics"]["relabel_permutation_sha256"], "test-hash"
        )

    def test_small_end_to_end_run_records_ratios_hash_pairing_and_resumes(self) -> None:
        configurations = ((1.0, 1), (0.0, 2), (1.0, 4))
        with TemporaryDirectory() as directory:
            output = Path(directory) / "sweep.json"
            report = run_experiment(
                dataset="voting",
                configurations=configurations,
                stage="screening",
                budget_multipliers=(50,),
                repeats=1,
                base_seed=123,
                processes=1,
                output_path=output,
            )
            serialized = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(serialized["status"], "complete")
            with patch(
                "experiments.run_inside_greedy_hyperparameter_sweep._run_cell",
                side_effect=AssertionError("resume must not rerun complete cells"),
            ):
                resumed = run_experiment(
                    dataset="voting",
                    configurations=configurations,
                    stage="screening",
                    budget_multipliers=(50,),
                    repeats=1,
                    base_seed=123,
                    processes=2,
                    output_path=output,
                    resume=True,
                )

        self.assertEqual(report["validation"]["passed"], True)
        self.assertEqual(
            report["validation"]["greedy_size_schedule_pairing_mismatches"], 0
        )
        self.assertEqual(
            report["validation"]["greedy_relabel_pairing_mismatches"], 0
        )
        self.assertEqual(resumed["status"], "complete")
        expected_cells = 1 + len(configurations)
        self.assertEqual(len(report["raw_cells"]), expected_cells)
        greedy_cells = [
            cell for cell in report["raw_cells"] if cell["method"] == "inside_greedy"
        ]
        default_cell = next(
            cell
            for cell in greedy_cells
            if cell["lambda0"] == REGISTERED_LAMBDA0
            and cell["candidate_pool"] == REGISTERED_CANDIDATE_POOL
        )
        self.assertEqual(default_cell["status"], "ok")
        self.assertTrue(
            all(
                (
                    cell["actual_utility_calls"] == cell["total_calls"]
                    if cell["status"] == "ok"
                    else cell["actual_utility_calls"] == 0
                )
                for cell in report["raw_cells"]
            )
        )
        self.assertTrue(
            all(
                "rmse_ratio_to_registered_default_same_repeat" in cell
                and "rmse_ratio_to_inside_orbit_same_repeat" in cell
                for cell in greedy_cells
            )
        )
        schedule_hashes = {
            cell["diagnostics"]["size_schedule_sha256"] for cell in greedy_cells
        }
        relabel_hashes = {
            cell["diagnostics"]["relabel_permutation_sha256"] for cell in greedy_cells
        }
        self.assertEqual(len(schedule_hashes), 1)
        self.assertEqual(len(relabel_hashes), 1)
        self.assertEqual(len(report["screening_ranking"]), len(configurations))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from experiments.diagnose_us_voting_frame_designs import (
    exact_size_win_probabilities,
)
from experiments.us_electoral_voting_game import (
    METHOD_LABELS,
    METHOD_ORDER,
    US_ELECTORAL_NAMES,
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
    game_fingerprint,
    run_experiment,
)
from frame_ofa import exact_shapley_shubik


class USElectoralVotingExperimentTests(unittest.TestCase):
    def test_exact_size_win_profile_on_symmetric_majority_game(self) -> None:
        np.testing.assert_array_equal(
            exact_size_win_probabilities(np.ones(3, dtype=int), 2),
            np.array([0.0, 0.0, 1.0, 1.0]),
        )

    def test_visible_method_names_explain_the_sampling_strategy(self) -> None:
        self.assertEqual(METHOD_LABELS["iid_linear"], "Random OFA")
        self.assertEqual(METHOD_LABELS["row_greedy"], "Greedy Frame-OFA")
        self.assertEqual(
            METHOD_LABELS["orbit_greedy"],
            "Batch-balanced Frame-OFA",
        )
        self.assertNotIn("orbit", METHOD_LABELS["orbit_greedy"].lower())

    def test_official_2024_2028_allocation_and_exact_truth(self) -> None:
        self.assertEqual(len(US_ELECTORAL_NAMES), 51)
        self.assertEqual(len(US_ELECTORAL_WEIGHTS), 51)
        self.assertEqual(int(US_ELECTORAL_WEIGHTS.sum()), 538)
        self.assertEqual(US_ELECTORAL_QUOTA, 270)
        self.assertEqual(len(game_fingerprint()), 64)

        truth = exact_shapley_shubik(
            US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        )
        self.assertAlmostEqual(float(truth.sum()), 1.0, places=14)
        self.assertAlmostEqual(truth[4], 0.108036833651899, places=14)
        self.assertAlmostEqual(truth[43], 0.077428257419069, places=14)
        for weight in np.unique(US_ELECTORAL_WEIGHTS):
            same_weight = truth[US_ELECTORAL_WEIGHTS == weight]
            self.assertLessEqual(float(np.ptp(same_weight)), 2e-15)
        ordered = [
            truth[US_ELECTORAL_WEIGHTS == weight][0]
            for weight in np.unique(US_ELECTORAL_WEIGHTS)
        ]
        self.assertTrue(np.all(np.diff(ordered) >= -2e-15))

    def test_small_end_to_end_report_uses_equal_call_budgets(self) -> None:
        report = run_experiment(
            budget_multipliers=(1,),
            repeats=2,
            candidate_pool=2,
            mean_balance=0.1,
            processes=1,
            base_seed=73,
            bootstrap_samples=20,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["game"]["fingerprint_sha256"], game_fingerprint())
        row = report["results_by_inner_budget"]["51"]
        self.assertEqual(row["inner_utility_calls"], 51)
        self.assertEqual(row["total_utility_calls"], 155)
        self.assertEqual(tuple(row["methods"]), METHOD_ORDER)
        for method in METHOD_ORDER:
            summary = row["methods"][method]
            self.assertGreater(summary["aggregate_rmse"], 0.0)
            self.assertEqual(
                len(summary["aggregate_rmse_bootstrap_95"]), 2
            )
            self.assertEqual(
                len(summary["rmse_change_vs_iid_bootstrap_95"]), 2
            )
        for repeat in report["raw_repeats"]:
            self.assertEqual(
                {item["total_utility_calls"] for item in repeat["rows"]},
                {155},
            )


if __name__ == "__main__":
    unittest.main()

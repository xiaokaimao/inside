from __future__ import annotations

from itertools import permutations
import unittest

import numpy as np

from experiments.airport_game import (
    AIRPORT_CLASS_COUNTS,
    AIRPORT_COSTS,
    METHOD_ORDER,
    exact_size_mean_utilities,
    game_fingerprint,
    run_experiment,
)
from frame_ofa import evaluate_airport, exact_airport_shapley
from experiments.validate_airport_report import validate_report


class AirportGameTests(unittest.TestCase):
    def test_attached_configuration_and_exact_truth(self) -> None:
        self.assertEqual(AIRPORT_CLASS_COUNTS, (8, 12, 6, 14, 8, 9, 13, 10, 10, 10))
        self.assertEqual(len(AIRPORT_COSTS), 100)
        self.assertEqual(len(game_fingerprint()), 64)
        truth = exact_airport_shapley(AIRPORT_COSTS)
        expected = np.cumsum(
            1.0 / np.array([100, 92, 80, 74, 60, 52, 43, 30, 20, 10])
        )
        for cost, value in enumerate(expected, start=1):
            np.testing.assert_allclose(
                truth[AIRPORT_COSTS == cost], value, rtol=0.0, atol=2e-15
            )
        self.assertAlmostEqual(float(truth.sum()), 10.0, places=13)

        size_means = exact_size_mean_utilities(AIRPORT_COSTS)
        self.assertEqual(size_means[0], 0.0)
        self.assertEqual(size_means[-1], 10.0)
        self.assertTrue(np.all(np.diff(size_means) >= -1e-14))

    def test_exact_formula_matches_all_permutations(self) -> None:
        costs = np.array([1.0, 1.0, 2.0, 3.0])
        totals = np.zeros(len(costs), dtype=np.float64)
        for order in permutations(range(len(costs))):
            previous = 0.0
            for player in order:
                current = max(previous, costs[player])
                totals[player] += current - previous
                previous = current
        brute_force = totals / 24.0
        np.testing.assert_allclose(
            exact_airport_shapley(costs),
            brute_force,
            rtol=0.0,
            atol=1e-15,
        )
        np.testing.assert_allclose(
            brute_force, [0.25, 0.25, 0.75, 1.75]
        )

    def test_utility_handles_empty_and_batched_coalitions(self) -> None:
        costs = np.array([1.0, 3.0, 2.0])
        coalitions = np.array(
            [[False, False, False], [True, False, True], [True, True, False]]
        )
        np.testing.assert_array_equal(
            evaluate_airport(coalitions, costs), [0.0, 2.0, 3.0]
        )
        self.assertEqual(
            float(evaluate_airport(coalitions[1], costs)), 2.0
        )

    def test_small_end_to_end_report_uses_equal_calls(self) -> None:
        report = run_experiment(
            budget_multipliers=(1,),
            repeats=2,
            candidate_pool=2,
            mean_balance=0.1,
            processes=1,
            base_seed=91,
            bootstrap_samples=20,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["game"]["fingerprint_sha256"], game_fingerprint())
        self.assertEqual(report["configuration"]["boundary_utility_calls"], 202)
        row = report["results_by_inner_budget"]["100"]
        self.assertEqual(row["inner_utility_calls"], 100)
        self.assertEqual(row["total_utility_calls"], 302)
        self.assertEqual(tuple(row["methods"]), METHOD_ORDER)
        for method in METHOD_ORDER:
            summary = row["methods"][method]
            self.assertTrue(np.isfinite(summary["aggregate_rmse"]))
            self.assertGreaterEqual(summary["aggregate_rmse"], 0.0)
        for repeat in report["raw_repeats"]:
            self.assertEqual(
                {item["total_utility_calls"] for item in repeat["rows"]},
                {302},
            )
        self.assertEqual(validate_report(report)["status"], "ready_to_share")


if __name__ == "__main__":
    unittest.main()

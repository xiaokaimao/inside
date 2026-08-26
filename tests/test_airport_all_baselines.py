from __future__ import annotations

import unittest

import numpy as np

from experiments.airport_all_baselines import (
    AirportGame,
    METHOD_ORDER,
    SerialAirportEvaluator,
    total_call_budgets,
)


class AirportAllBaselineProtocolTests(unittest.TestCase):
    def test_method_set_matches_wine_main_figure(self) -> None:
        self.assertEqual(
            METHOD_ORDER,
            (
                "frame_orbit_ratio",
                "official_ofa_fixed_ratio",
                "official_cc_basic",
                "s_diff",
                "diff",
                "group_testing",
                "kernel_shap_sampled",
                "gels_shapley",
                "tmc_shapley",
                "stratified_marginal_mc",
            ),
        )

    def test_default_scale_has_equal_total_call_caps(self) -> None:
        inner, total = total_call_budgets((500, 1000, 2000, 5000, 10000))
        self.assertEqual(inner, (50_000, 100_000, 200_000, 500_000, 1_000_000))
        self.assertEqual(total, (50_202, 100_202, 200_202, 500_202, 1_000_202))

    def test_budget_multipliers_must_be_strictly_increasing(self) -> None:
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            total_call_budgets((500, 500))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            total_call_budgets((1000, 500))

    def test_serial_evaluator_supports_vector_and_coarse_tasks(self) -> None:
        evaluator = SerialAirportEvaluator()
        rows = np.zeros((2, 100), dtype=bool)
        rows[0, 0] = True
        rows[1, -1] = True
        np.testing.assert_array_equal(evaluator.evaluate(rows), [1.0, 10.0])

        def task(game: AirportGame, index: int) -> float:
            coalition = np.zeros(100, dtype=bool)
            coalition[index] = True
            return game.evaluate(coalition)

        self.assertEqual(evaluator.run_game_tasks(task, [0, 99]), [1.0, 10.0])


if __name__ == "__main__":
    unittest.main()

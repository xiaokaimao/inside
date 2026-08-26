from __future__ import annotations

import unittest

import numpy as np

from experiments.debug_inside_greedy_lambda_scale import (
    converted_mean_balance,
)


class InsideGreedyLambdaScaleTests(unittest.TestCase):
    def test_conversion_matches_normalized_objective_scale(self) -> None:
        sizes = np.asarray([2, 3, 4, 5, 6], dtype=np.int64)
        effective, mean_weight_squared, correction = converted_mean_balance(
            sizes, num_players=8, lambda0=1.25
        )
        expected_mean = float(np.mean(sizes * (8 - sizes)))
        self.assertAlmostEqual(mean_weight_squared, expected_mean)
        self.assertAlmostEqual(correction, 1.0 - 1.0 / 7.0)
        self.assertAlmostEqual(
            effective, 1.25 * correction * expected_mean
        )

    def test_conversion_rejects_invalid_schedule_and_lambda(self) -> None:
        with self.assertRaises(ValueError):
            converted_mean_balance(np.asarray([]), 8, 1.0)
        with self.assertRaises(ValueError):
            converted_mean_balance(np.asarray([1, 2]), 8, 1.0)
        with self.assertRaises(ValueError):
            converted_mean_balance(np.asarray([2, 3]), 8, -1.0)


if __name__ == "__main__":
    unittest.main()

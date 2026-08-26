from __future__ import annotations

import math
import unittest

import numpy as np

from frame_ofa import evaluate_weighted_voting, exact_shapley_shubik


def _exhaustive_shapley(weights: np.ndarray, quota: int) -> np.ndarray:
    num_players = len(weights)
    table = np.zeros(1 << num_players, dtype=np.float64)
    for mask in range(1 << num_players):
        coalition = np.asarray(
            [bool(mask & (1 << index)) for index in range(num_players)]
        )
        table[mask] = evaluate_weighted_voting(
            coalition, weights, quota
        )

    values = np.zeros(num_players, dtype=np.float64)
    for player in range(num_players):
        bit = 1 << player
        for mask in range(1 << num_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            coefficient = 1.0 / (
                num_players * math.comb(num_players - 1, size)
            )
            values[player] += coefficient * (
                table[mask | bit] - table[mask]
            )
    return values


class WeightedVotingTests(unittest.TestCase):
    def test_known_shapley_shubik_games(self) -> None:
        np.testing.assert_allclose(
            exact_shapley_shubik(np.ones(3, dtype=int), 2),
            np.full(3, 1.0 / 3.0),
            atol=1e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            exact_shapley_shubik(np.array([2, 1, 1]), 3),
            np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]),
            atol=1e-15,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            exact_shapley_shubik(np.array([3, 1, 1]), 3),
            np.array([1.0, 0.0, 0.0]),
            atol=1e-15,
            rtol=0.0,
        )

    def test_inclusive_quota_boundary(self) -> None:
        weights = np.array([270, 0])
        self.assertEqual(
            evaluate_weighted_voting(np.array([True, False]), weights, 270),
            1.0,
        )
        np.testing.assert_array_equal(
            exact_shapley_shubik(weights, 270), np.array([1.0, 0.0])
        )

    def test_dp_matches_exhaustive_random_small_games(self) -> None:
        rng = np.random.default_rng(20260824)
        for num_players in range(2, 8):
            for _ in range(8):
                weights = rng.integers(0, 7, size=num_players)
                total = int(weights.sum())
                quota = int(rng.integers(1, total + 1)) if total else 1
                np.testing.assert_allclose(
                    exact_shapley_shubik(weights, quota),
                    _exhaustive_shapley(weights, quota),
                    atol=2e-15,
                    rtol=0.0,
                )

    def test_constant_games_and_input_validation(self) -> None:
        weights = np.array([4, 2, 1])
        np.testing.assert_array_equal(
            exact_shapley_shubik(weights, 0), np.zeros(3)
        )
        np.testing.assert_array_equal(
            exact_shapley_shubik(weights, 8), np.zeros(3)
        )
        with self.assertRaisesRegex(ValueError, "nonnegative integers"):
            exact_shapley_shubik(np.array([1.5, 2.0]), 2)
        with self.assertRaisesRegex(ValueError, "player count"):
            evaluate_weighted_voting(
                np.zeros((3, 4), dtype=bool), np.ones(3), 2
            )


if __name__ == "__main__":
    unittest.main()

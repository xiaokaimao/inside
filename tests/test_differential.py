from __future__ import annotations

import math
import unittest

import numpy as np

from frame_ofa.differential import (
    DifferentialCoverageError,
    aggregate_diff_samples,
    diff_layer_distribution,
    estimate_diff,
)
from frame_ofa.parallel import GameEvaluator


def _exact_shapley(table: np.ndarray, num_players: int) -> np.ndarray:
    values = np.zeros(num_players, dtype=np.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        player_bit = 1 << player
        for mask in range(1 << num_players):
            if mask & player_bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            values[player] += weight * (
                table[mask | player_bit] - table[mask]
            )
    return values


class _CountingAffineGame:
    def __init__(
        self,
        coefficients: np.ndarray,
        baseline: float = 0.0,
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.baseline = float(baseline)
        self.calls = 0

    def evaluate(self, coalition: np.ndarray) -> float:
        self.calls += 1
        return self.baseline + float(self.coefficients @ coalition)


class _SequentialEvaluator:
    def __init__(self, game: _CountingAffineGame) -> None:
        self.game = game
        self.task_counts: list[int] = []

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.game.evaluate(row) for row in coalitions],
            dtype=np.float64,
        )

    def run_game_tasks(self, task_func, payloads, *, chunksize=1):
        if chunksize < 1:
            raise AssertionError("invalid chunksize")
        self.task_counts.append(len(payloads))
        return [task_func(self.game, payload) for payload in payloads]


class DifferentialTests(unittest.TestCase):
    def test_paper_layer_distribution(self) -> None:
        layers, probabilities, normalization = diff_layer_distribution(4)
        np.testing.assert_array_equal(layers, [1, 2, 3])
        weights = np.asarray([1 / 3, 1 / 4, 1 / 3])
        self.assertAlmostEqual(normalization, float(weights.sum()))
        np.testing.assert_allclose(
            probabilities, weights / weights.sum(), atol=1e-15
        )
        np.testing.assert_allclose(probabilities, probabilities[::-1])

    def test_weighted_complete_four_player_samples_recover_exact_game(self) -> None:
        num_players = 4
        rng = np.random.default_rng(887)
        table = rng.normal(size=1 << num_players)
        # For n=4, p_k / C(4,k) is 1/11, 1/22, 1/11.
        # Repeating size-1/3 coalitions twice and size-2 coalitions once is
        # therefore an exact finite realization of Algorithm 1's law.
        masks = [
            mask
            for mask in range(1, (1 << num_players) - 1)
            for _ in range(2 if mask.bit_count() in (1, 3) else 1)
        ]
        coalitions = np.asarray(
            [
                [bool(mask & (1 << player)) for player in range(num_players)]
                for mask in masks
            ],
            dtype=bool,
        )
        utilities = np.asarray([table[mask] for mask in masks])
        values, matrix, counts = aggregate_diff_samples(
            coalitions,
            utilities,
            empty_utility=float(table[0]),
            full_utility=float(table[-1]),
        )
        np.testing.assert_allclose(
            values,
            _exact_shapley(table, num_players),
            atol=1e-13,
            rtol=1e-13,
        )
        np.testing.assert_allclose(matrix, -matrix.T, atol=0)
        np.testing.assert_array_equal(
            counts[~np.eye(num_players, dtype=bool)], np.full(12, 6)
        )

    def test_strict_coverage_failure(self) -> None:
        game = _CountingAffineGame(np.asarray([1.0, 2.0, 3.0, 4.0]))
        with self.assertRaises(DifferentialCoverageError) as caught:
            estimate_diff(
                _SequentialEvaluator(game),
                num_players=4,
                total_call_budget=3,
                seed=0,
                num_tasks=1,
            )
        self.assertGreater(caught.exception.missing_ordered_pairs, 0)
        self.assertEqual(caught.exception.total_ordered_pairs, 12)
        self.assertEqual(caught.exception.utility_evaluations, 3)
        self.assertEqual(game.calls, 3)

    def test_exact_call_count_coverage_and_efficiency(self) -> None:
        coefficients = np.asarray([1.0, -2.0, 0.5, 3.0, -0.25])
        game = _CountingAffineGame(coefficients, baseline=4.0)
        evaluator = _SequentialEvaluator(game)
        result = estimate_diff(
            evaluator,
            num_players=5,
            total_call_budget=1003,
            seed=91,
            num_tasks=17,
            task_chunksize=2,
        )
        diagnostics = result.diagnostics
        self.assertEqual(game.calls, 1003)
        self.assertEqual(diagnostics.utility_evaluations, 1003)
        self.assertEqual(diagnostics.target_call_budget, 1003)
        self.assertEqual(diagnostics.unused_calls, 0)
        self.assertEqual(diagnostics.num_utility_samples, 1001)
        self.assertEqual(diagnostics.boundary_evaluations, 2)
        self.assertEqual(diagnostics.num_tasks, 17)
        self.assertEqual(evaluator.task_counts, [17])
        self.assertEqual(int(diagnostics.layer_counts.sum()), 1001)
        self.assertEqual(diagnostics.ordered_pair_coverage, 1.0)
        self.assertEqual(diagnostics.missing_ordered_pairs, 0)
        self.assertGreater(diagnostics.minimum_ordered_pair_count, 0)
        self.assertAlmostEqual(
            float(result.values.sum()), float(coefficients.sum()), places=13
        )

    def test_seed_reproducibility(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        first = estimate_diff(
            _SequentialEvaluator(_CountingAffineGame(coefficients, 2.5)),
            5,
            503,
            12345,
            num_tasks=11,
        )
        second = estimate_diff(
            _SequentialEvaluator(_CountingAffineGame(coefficients, 2.5)),
            5,
            503,
            12345,
            num_tasks=11,
        )
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(
            first.diagnostics.layer_counts,
            second.diagnostics.layer_counts,
        )
        np.testing.assert_array_equal(
            first.diagnostics.pair_counts,
            second.diagnostics.pair_counts,
        )

    def test_spawn_game_evaluator(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        with GameEvaluator(
            _CountingAffineGame,
            {"coefficients": coefficients, "baseline": 1.25},
            n_jobs=2,
            start_method="spawn",
            worker_threads=1,
        ) as evaluator:
            result = estimate_diff(
                evaluator,
                num_players=5,
                total_call_budget=403,
                seed=54321,
                num_tasks=7,
            )
        self.assertEqual(result.values.shape, (5,))
        self.assertTrue(np.isfinite(result.values).all())
        self.assertEqual(result.diagnostics.utility_evaluations, 403)
        self.assertEqual(result.diagnostics.num_tasks, 7)
        self.assertEqual(result.diagnostics.ordered_pair_coverage, 1.0)

    def test_input_validation(self) -> None:
        evaluator = _SequentialEvaluator(
            _CountingAffineGame(np.asarray([1.0, 2.0]))
        )
        cases = (
            {"num_players": 1, "total_call_budget": 3, "seed": 0},
            {"num_players": 2, "total_call_budget": 2, "seed": 0},
            {"num_players": 2, "total_call_budget": 3, "seed": -1},
            {
                "num_players": 2,
                "total_call_budget": 3,
                "seed": 0,
                "num_tasks": 0,
            },
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_diff(evaluator, **kwargs)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

import numpy as np

from frame_ofa import (
    GameEvaluator,
    aggregate_basic_cc_samples,
    estimate_basic_cc,
)


def _exact_shapley(table: np.ndarray, num_players: int) -> np.ndarray:
    values = np.zeros(num_players, dtype=np.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        bit = 1 << player
        for mask in range(1 << num_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            values[player] += weight * (
                table[mask | bit] - table[mask]
            )
    return values


def _official_reference(
    coalitions: np.ndarray, differences: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Literal Algorithm-2 accumulator used as an independent oracle."""
    num_players = coalitions.shape[1]
    sums = np.zeros((num_players, num_players), dtype=np.float64)
    counts = np.zeros((num_players, num_players), dtype=np.int64)
    for coalition, difference in zip(coalitions, differences):
        size = int(coalition.sum())
        for player in range(num_players):
            if coalition[player]:
                sums[player, size - 1] += difference
                counts[player, size - 1] += 1
            elif num_players - size > 0:
                dual = num_players - size
                sums[player, dual - 1] -= difference
                counts[player, dual - 1] += 1
    means = np.zeros_like(sums)
    np.divide(sums, counts, out=means, where=counts > 0)
    return means.mean(axis=1), counts


class _CountingAdditiveGame:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.calls = 0

    def evaluate(self, coalition: np.ndarray) -> float:
        self.calls += 1
        return float(self.coefficients @ coalition)


class _SequentialTaskEvaluator:
    def __init__(self, game: _CountingAdditiveGame) -> None:
        self.game = game
        self.payload_counts: list[int] = []

    def run_game_tasks(
        self, task_func, payloads, *, chunksize: int = 1
    ):
        if chunksize < 1:
            raise AssertionError("invalid chunksize")
        self.payload_counts.append(len(payloads))
        return [task_func(self.game, payload) for payload in payloads]


class BasicComplementaryContributionTests(unittest.TestCase):
    def test_accumulator_matches_literal_official_reference(self) -> None:
        coalitions = np.asarray(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=bool,
        )
        differences = np.asarray([2.0, -1.0, 3.5, 7.0])
        expected_values, expected_counts = _official_reference(
            coalitions, differences
        )
        values, counts = aggregate_basic_cc_samples(
            coalitions, differences
        )
        np.testing.assert_array_equal(counts, expected_counts)
        np.testing.assert_allclose(values, expected_values, rtol=0, atol=0)

    def test_complete_slice_enumeration_recovers_exact_shapley(self) -> None:
        num_players = 4
        rng = np.random.default_rng(901)
        table = rng.normal(size=1 << num_players)
        coalitions = []
        differences = []
        full_mask = (1 << num_players) - 1
        for mask in range(1, 1 << num_players):
            coalition = np.asarray(
                [bool(mask & (1 << player)) for player in range(num_players)]
            )
            coalitions.append(coalition)
            differences.append(table[mask] - table[full_mask ^ mask])
        values, counts = aggregate_basic_cc_samples(
            np.asarray(coalitions, dtype=bool),
            np.asarray(differences, dtype=np.float64),
        )
        self.assertTrue(np.all(counts > 0))
        np.testing.assert_allclose(
            values,
            _exact_shapley(table, num_players),
            rtol=1e-13,
            atol=1e-13,
        )

    def test_missing_strata_are_zero_filled(self) -> None:
        coalitions = np.ones((1, 3), dtype=bool)
        values, counts = aggregate_basic_cc_samples(
            coalitions, np.asarray([6.0])
        )
        np.testing.assert_array_equal(
            counts,
            np.asarray(
                [[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.int64
            ),
        )
        # Only the full-coalition stratum is observed, so 6 / n = 2.
        np.testing.assert_array_equal(values, np.full(3, 2.0))

    def test_seed_reproducibility_and_exact_two_call_accounting(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        first_game = _CountingAdditiveGame(coefficients)
        first_evaluator = _SequentialTaskEvaluator(first_game)
        first = estimate_basic_cc(
            first_evaluator,
            num_players=5,
            num_pairs=97,
            seed=12345,
            num_tasks=11,
            task_chunksize=2,
        )
        second_game = _CountingAdditiveGame(coefficients)
        second_evaluator = _SequentialTaskEvaluator(second_game)
        second = estimate_basic_cc(
            second_evaluator,
            num_players=5,
            num_pairs=97,
            seed=12345,
            num_tasks=11,
            task_chunksize=2,
        )

        self.assertEqual(first_game.calls, 194)
        self.assertEqual(second_game.calls, 194)
        self.assertEqual(first.diagnostics.utility_evaluations, 194)
        self.assertEqual(first.diagnostics.num_pairs, 97)
        self.assertEqual(first.diagnostics.num_tasks, 11)
        self.assertEqual(first_evaluator.payload_counts, [11])
        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(
            first.diagnostics.stratum_counts,
            second.diagnostics.stratum_counts,
        )

    def test_process_game_evaluator_runs_coarse_cc_tasks(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        with GameEvaluator(
            _CountingAdditiveGame,
            {"coefficients": coefficients},
            n_jobs=2,
            chunksize=1,
            start_method="spawn",
            worker_threads=1,
        ) as evaluator:
            result = estimate_basic_cc(
                evaluator,
                num_players=5,
                num_pairs=31,
                seed=54321,
                num_tasks=7,
            )
        self.assertEqual(result.values.shape, (5,))
        self.assertTrue(np.isfinite(result.values).all())
        self.assertEqual(result.diagnostics.num_pairs, 31)
        self.assertEqual(result.diagnostics.utility_evaluations, 62)
        self.assertEqual(result.diagnostics.num_tasks, 7)

    def test_diagnostics_report_only_the_n_by_n_valid_cells(self) -> None:
        game = _CountingAdditiveGame(np.asarray([1.0, 2.0, 3.0]))
        result = estimate_basic_cc(
            _SequentialTaskEvaluator(game),
            num_players=3,
            num_pairs=1,
            seed=7,
            num_tasks=128,
        )
        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.num_tasks, 1)
        self.assertEqual(diagnostics.minimum_stratum_count, 0)
        self.assertEqual(diagnostics.minimum_positive_count, 1)
        self.assertEqual(
            diagnostics.missing_strata,
            int(np.count_nonzero(diagnostics.stratum_counts == 0)),
        )
        self.assertEqual(
            diagnostics.missing_stratum_fraction,
            diagnostics.missing_strata / 9,
        )
        self.assertEqual(diagnostics.sampling, "uniform_random_size_1_to_n")
        self.assertEqual(diagnostics.missing, "zero")
        self.assertFalse(diagnostics.efficiency_projection)

    def test_input_validation(self) -> None:
        evaluator = _SequentialTaskEvaluator(
            _CountingAdditiveGame(np.ones(2))
        )
        for kwargs in (
            {"num_players": 0, "num_pairs": 1, "seed": 0},
            {"num_players": 2, "num_pairs": 0, "seed": 0},
            {"num_players": 2, "num_pairs": 1, "seed": -1},
            {
                "num_players": 2,
                "num_pairs": 1,
                "seed": 0,
                "num_tasks": 0,
            },
        ):
            with self.assertRaises(ValueError):
                estimate_basic_cc(evaluator, **kwargs)


if __name__ == "__main__":
    unittest.main()

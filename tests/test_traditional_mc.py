from __future__ import annotations

import multiprocessing as mp
import unittest

import numpy as np

from frame_ofa.parallel import GameEvaluator
from frame_ofa.traditional_mc import (
    estimate_stratified_marginal_mc,
)


class _AdditiveGame:
    def __init__(self, weights: np.ndarray, constant: float = 0.0) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.constant = float(constant)
        self.calls = 0

    def evaluate(self, coalition: np.ndarray) -> float:
        self.calls += 1
        mask = np.asarray(coalition, dtype=bool)
        return self.constant + float(self.weights[mask].sum())


class _InteractionGame:
    def __init__(self, num_players: int) -> None:
        self.num_players = int(num_players)

    def evaluate(self, coalition: np.ndarray) -> float:
        mask = np.asarray(coalition, dtype=bool)
        linear = float(np.dot(np.arange(1, self.num_players + 1), mask))
        pair = 2.5 * float(mask[0] and mask[1])
        triple = -1.25 * float(mask[1] and mask[2] and mask[3])
        return linear + pair + triple


class _SerialTaskEvaluator:
    def __init__(self, game) -> None:
        self.game = game

    def run_game_tasks(self, task_func, payloads, *, chunksize=1):
        del chunksize
        return [task_func(self.game, payload) for payload in payloads]


def _make_additive_game(*, weights, constant=0.0):
    return _AdditiveGame(np.asarray(weights), constant=float(constant))


class StratifiedMarginalMCTests(unittest.TestCase):
    def test_additive_game_is_exact_and_calls_are_physical(self) -> None:
        weights = np.array([1.5, -0.25, 2.0, 0.75])
        game = _AdditiveGame(weights, constant=7.0)
        evaluator = _SerialTaskEvaluator(game)

        result = estimate_stratified_marginal_mc(
            evaluator,
            num_players=4,
            total_call_budget=33,
            seed=11,
            num_tasks=4,
        )

        np.testing.assert_allclose(result.values, weights, atol=1e-15)
        self.assertEqual(game.calls, 32)
        self.assertEqual(result.diagnostics.utility_evaluations, 32)
        self.assertEqual(result.diagnostics.unused_calls, 1)
        self.assertEqual(result.diagnostics.minimum_stratum_count, 1)
        self.assertEqual(result.diagnostics.maximum_stratum_count, 1)
        self.assertFalse(result.diagnostics.boundary_reuse)
        self.assertFalse(result.diagnostics.efficiency_projection)

    def test_integer_remainder_is_balanced_over_strata(self) -> None:
        n = 4
        marginal_samples = n * n + 5
        evaluator = _SerialTaskEvaluator(_AdditiveGame(np.ones(n)))
        result = estimate_stratified_marginal_mc(
            evaluator,
            num_players=n,
            total_call_budget=2 * marginal_samples,
            seed=23,
            num_tasks=3,
        )

        counts = result.diagnostics.stratum_counts
        self.assertEqual(int(counts.sum()), marginal_samples)
        self.assertEqual(int(np.count_nonzero(counts == 2)), 5)
        self.assertEqual(int(np.count_nonzero(counts == 1)), n * n - 5)
        self.assertEqual(result.diagnostics.minimum_stratum_count, 1)
        self.assertEqual(result.diagnostics.maximum_stratum_count, 2)

    def test_sampling_is_invariant_to_task_partition(self) -> None:
        n = 5
        budget = 2 * (3 * n * n + 7)
        first = estimate_stratified_marginal_mc(
            _SerialTaskEvaluator(_InteractionGame(n)),
            num_players=n,
            total_call_budget=budget,
            seed=2026,
            num_tasks=1,
        )
        second = estimate_stratified_marginal_mc(
            _SerialTaskEvaluator(_InteractionGame(n)),
            num_players=n,
            total_call_budget=budget,
            seed=2026,
            num_tasks=11,
        )

        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(
            first.diagnostics.stratum_counts,
            second.diagnostics.stratum_counts,
        )

    def test_game_evaluator_process_integration(self) -> None:
        weights = np.array([0.5, -1.0, 2.25])
        start_method = (
            "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        )
        with GameEvaluator(
            _make_additive_game,
            {"weights": weights, "constant": 3.0},
            n_jobs=2,
            start_method=start_method,
        ) as evaluator:
            result = estimate_stratified_marginal_mc(
                evaluator,
                num_players=3,
                total_call_budget=2 * 3 * 3,
                seed=5,
                num_tasks=5,
            )

        np.testing.assert_allclose(result.values, weights, atol=1e-15)
        self.assertEqual(result.diagnostics.utility_evaluations, 18)
        self.assertEqual(result.diagnostics.num_tasks, 5)

    def test_invalid_arguments_are_rejected(self) -> None:
        evaluator = _SerialTaskEvaluator(_AdditiveGame(np.ones(3)))
        invalid = (
            {"num_players": 0, "total_call_budget": 18, "seed": 1},
            {"num_players": 3, "total_call_budget": 17, "seed": 1},
            {"num_players": 3, "total_call_budget": 18, "seed": -1},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    estimate_stratified_marginal_mc(
                        evaluator,
                        **arguments,
                    )

        with self.assertRaises(ValueError):
            estimate_stratified_marginal_mc(
                evaluator,
                num_players=3,
                total_call_budget=18,
                seed=1,
                num_tasks=0,
            )


if __name__ == "__main__":
    unittest.main()

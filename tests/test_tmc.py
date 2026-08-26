from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from frame_ofa.parallel import GameEvaluator
from frame_ofa.tmc import (
    estimate_full_permutation_mc,
    estimate_tmc_shapley,
    full_permutation_contributions,
    full_permutations_for_budget,
    tmc_permutation_contributions,
)


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


class _CountingTableGame:
    def __init__(self, table: np.ndarray) -> None:
        self.table = np.asarray(table, dtype=np.float64)
        self.num_players = int(round(math.log2(len(self.table))))
        if len(self.table) != 1 << self.num_players:
            raise ValueError("table length must be a power of two")
        self.calls = 0

    def evaluate(self, coalition: np.ndarray) -> float:
        self.calls += 1
        row = np.asarray(coalition, dtype=bool)
        mask = int(row.astype(np.int64) @ (1 << np.arange(len(row))))
        return float(self.table[mask])


class _CountingAdditiveGame:
    def __init__(self, coefficients: np.ndarray, constant: float = 0.0) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.constant = float(constant)
        self.calls = 0

    def evaluate(self, coalition: np.ndarray) -> float:
        self.calls += 1
        return self.constant + float(self.coefficients @ coalition)


class _SequentialEvaluator:
    def __init__(self, game) -> None:
        self.game = game
        self.task_counts: list[int] = []

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.game.evaluate(row.copy()) for row in coalitions],
            dtype=np.float64,
        )

    def run_game_tasks(self, task_func, payloads, *, chunksize: int = 1):
        if chunksize < 1:
            raise AssertionError("invalid chunksize")
        self.task_counts.append(len(payloads))
        return [task_func(self.game, payload) for payload in payloads]


class FullPermutationMonteCarloTests(unittest.TestCase):
    def test_exhaustive_permutations_recover_arbitrary_game(self) -> None:
        num_players = 4
        table = np.random.default_rng(410).normal(size=1 << num_players)
        game = _CountingTableGame(table)
        empty = game.evaluate(np.zeros(num_players, dtype=bool))
        full = game.evaluate(np.ones(num_players, dtype=bool))

        contributions = [
            full_permutation_contributions(
                game,
                np.asarray(order, dtype=np.int64),
                empty,
                full,
            )
            for order in itertools.permutations(range(num_players))
        ]
        actual = np.mean(contributions, axis=0)

        np.testing.assert_allclose(
            actual,
            _exact_shapley(table, num_players),
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertEqual(
            game.calls,
            2 + math.factorial(num_players) * (num_players - 1),
        )

    def test_additive_game_is_exact_and_calls_match_budget_mapping(self) -> None:
        coefficients = np.asarray([1.25, -0.5, 3.0, 0.125, -2.0])
        target_budget = 153
        # 2 endpoints + 37 complete paths * 4 interior prefixes = 150.
        self.assertEqual(
            full_permutations_for_budget(5, target_budget),
            (37, 150, 3),
        )
        game = _CountingAdditiveGame(coefficients, constant=7.0)
        evaluator = _SequentialEvaluator(game)
        result = estimate_full_permutation_mc(
            evaluator,
            num_players=5,
            total_call_budget=target_budget,
            seed=1234,
            num_tasks=11,
            task_chunksize=2,
        )

        np.testing.assert_allclose(result.values, coefficients, atol=1e-15)
        self.assertEqual(game.calls, 150)
        self.assertEqual(result.diagnostics.num_permutations, 37)
        self.assertEqual(result.diagnostics.target_call_budget, target_budget)
        self.assertEqual(result.diagnostics.utility_evaluations, 150)
        self.assertEqual(result.diagnostics.unused_calls, 3)
        self.assertEqual(result.diagnostics.boundary_utility_evaluations, 2)
        self.assertEqual(result.diagnostics.interior_utility_evaluations, 148)
        self.assertEqual(result.diagnostics.num_tasks, 11)
        self.assertEqual(evaluator.task_counts, [11])
        self.assertFalse(result.diagnostics.truncation)
        self.assertFalse(result.diagnostics.efficiency_projection)
        self.assertTrue(result.diagnostics.boundary_reuse)

    def test_same_seed_is_reproducible_and_partition_invariant(self) -> None:
        num_players = 5
        table = np.random.default_rng(802).normal(size=1 << num_players)

        def run(num_tasks: int):
            game = _CountingTableGame(table)
            result = estimate_full_permutation_mc(
                _SequentialEvaluator(game),
                num_players=num_players,
                total_call_budget=2 + 97 * (num_players - 1),
                seed=98765,
                num_tasks=num_tasks,
            )
            return result, game.calls

        first, first_calls = run(7)
        second, second_calls = run(7)
        single_task, single_calls = run(1)

        np.testing.assert_array_equal(first.values, second.values)
        np.testing.assert_array_equal(
            first.diagnostics.sample_variance,
            second.diagnostics.sample_variance,
        )
        np.testing.assert_allclose(
            first.values,
            single_task.values,
            rtol=0,
            # Identical samples are summed in different task groupings.
            atol=5e-16,
        )
        self.assertEqual(first_calls, 390)
        self.assertEqual(second_calls, 390)
        self.assertEqual(single_calls, 390)

    def test_never_truncates_even_when_every_nonempty_prefix_is_full(self) -> None:
        num_players = 6
        table = np.ones(1 << num_players, dtype=np.float64)
        table[0] = 0.0
        game = _CountingTableGame(table)
        result = estimate_full_permutation_mc(
            _SequentialEvaluator(game),
            num_players=num_players,
            total_call_budget=2 + 13 * (num_players - 1),
            seed=4,
            num_tasks=4,
        )
        self.assertEqual(game.calls, 67)
        self.assertEqual(result.diagnostics.interior_utility_evaluations, 65)
        self.assertFalse(result.diagnostics.truncation)

    def test_one_player_uses_only_endpoints(self) -> None:
        game = _CountingAdditiveGame(np.asarray([2.75]), constant=-3.0)
        result = estimate_full_permutation_mc(
            _SequentialEvaluator(game),
            num_players=1,
            total_call_budget=9,
            seed=0,
        )
        np.testing.assert_array_equal(result.values, np.asarray([2.75]))
        self.assertEqual(game.calls, 2)
        self.assertEqual(result.diagnostics.num_permutations, 0)
        self.assertEqual(result.diagnostics.utility_evaluations, 2)
        self.assertEqual(result.diagnostics.unused_calls, 7)
        self.assertEqual(result.diagnostics.num_tasks, 0)

    def test_spawn_game_evaluator_runs_coarse_permutation_tasks(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        budget = 2 + 23 * (len(coefficients) - 1)
        with GameEvaluator(
            _CountingAdditiveGame,
            {"coefficients": coefficients, "constant": 1.0},
            n_jobs=2,
            chunksize=1,
            start_method="spawn",
            worker_threads=1,
        ) as evaluator:
            result = estimate_full_permutation_mc(
                evaluator,
                num_players=len(coefficients),
                total_call_budget=budget,
                seed=2048,
                num_tasks=7,
            )
        np.testing.assert_allclose(result.values, coefficients, atol=1e-15)
        self.assertEqual(result.diagnostics.utility_evaluations, budget)
        self.assertEqual(result.diagnostics.unused_calls, 0)
        self.assertEqual(result.diagnostics.num_tasks, 7)

    def test_invalid_inputs_fail_before_utility_use(self) -> None:
        game = _CountingAdditiveGame(np.ones(3))
        evaluator = _SequentialEvaluator(game)
        invalid = (
            {"num_players": 0, "total_call_budget": 10, "seed": 0},
            {"num_players": 3, "total_call_budget": 3, "seed": 0},
            {"num_players": 3, "total_call_budget": 10, "seed": -1},
            {
                "num_players": 3,
                "total_call_budget": 10,
                "seed": 0,
                "num_tasks": 0,
            },
            {
                "num_players": 3,
                "total_call_budget": 10,
                "seed": 0,
                "task_chunksize": 0,
            },
        )
        for arguments in invalid:
            with self.assertRaises(ValueError):
                estimate_full_permutation_mc(evaluator, **arguments)
        self.assertEqual(game.calls, 0)

    def test_explicit_permutation_validation(self) -> None:
        game = _CountingAdditiveGame(np.ones(3))
        for invalid in (
            np.asarray([], dtype=np.int64),
            np.asarray([[0, 1, 2]], dtype=np.int64),
            np.asarray([0, 0, 2], dtype=np.int64),
            np.asarray([0, 1, 3], dtype=np.int64),
        ):
            with self.assertRaises(ValueError):
                full_permutation_contributions(game, invalid, 0.0, 3.0)
        self.assertEqual(game.calls, 0)


class TMCShapleyTests(unittest.TestCase):
    def test_external_default_requires_six_consecutive_near_full_prefixes(
        self,
    ) -> None:
        num_players = 10
        table = np.ones(1 << num_players, dtype=np.float64)
        table[0] = 0.0
        game = _CountingTableGame(table)
        contributions, calls, truncated = tmc_permutation_contributions(
            game,
            np.arange(num_players),
            empty_utility=0.0,
            full_utility=1.0,
            truncation_tolerance=0.0,
            relative_tolerance=True,
            truncation_patience=5,
        )

        self.assertTrue(truncated)
        self.assertEqual(calls, 6)
        self.assertEqual(game.calls, 6)
        np.testing.assert_array_equal(
            contributions,
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )

    def test_fixed_permutation_budget_reports_truncation_savings(self) -> None:
        num_players = 10
        num_permutations = 20
        target_budget = 2 + num_permutations * num_players
        table = np.ones(1 << num_players, dtype=np.float64)
        table[0] = 0.0
        game = _CountingTableGame(table)
        evaluator = _SequentialEvaluator(game)
        result = estimate_tmc_shapley(
            evaluator,
            num_players=num_players,
            total_call_budget=target_budget,
            seed=47,
            num_tasks=7,
        )

        diagnostics = result.diagnostics
        self.assertEqual(diagnostics.num_permutations, num_permutations)
        self.assertEqual(diagnostics.evaluated_prefixes, 120)
        self.assertEqual(diagnostics.utility_evaluations, 122)
        self.assertEqual(diagnostics.unused_calls, 80)
        self.assertEqual(diagnostics.worst_case_reserved_calls, target_budget)
        self.assertEqual(diagnostics.budget_remainder_calls, 0)
        self.assertEqual(diagnostics.truncation_saved_calls, 80)
        self.assertEqual(diagnostics.truncated_permutations, 20)
        self.assertEqual(diagnostics.truncation_rate, 1.0)
        self.assertEqual(diagnostics.mean_prefix_length, 6.0)
        self.assertEqual(diagnostics.num_tasks, 7)
        self.assertEqual(evaluator.task_counts, [7])
        self.assertEqual(game.calls, 122)
        self.assertFalse(diagnostics.efficiency_projection)
        self.assertFalse(diagnostics.boundary_reuse)

    def test_untruncated_fixed_m_paths_recover_additive_game(self) -> None:
        coefficients = np.asarray([1.25, -0.5, 3.0, 0.125, -2.0])
        target_budget = 120
        # 2 endpoints + 23 paths * 5 prefix calls = 117; remainder is 3.
        game = _CountingAdditiveGame(coefficients, constant=7.0)
        result = estimate_tmc_shapley(
            _SequentialEvaluator(game),
            num_players=5,
            total_call_budget=target_budget,
            seed=1234,
            truncation_patience=5,
            num_tasks=11,
        )

        np.testing.assert_allclose(result.values, coefficients, atol=1e-15)
        self.assertEqual(result.diagnostics.num_permutations, 23)
        self.assertEqual(result.diagnostics.utility_evaluations, 117)
        self.assertEqual(result.diagnostics.unused_calls, 3)
        self.assertEqual(result.diagnostics.budget_remainder_calls, 3)
        self.assertEqual(result.diagnostics.truncation_saved_calls, 0)
        self.assertEqual(result.diagnostics.truncated_permutations, 0)
        self.assertEqual(game.calls, 117)

    def test_spawn_game_evaluator_runs_parallel_tmc_chunks(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        budget = 2 + 23 * len(coefficients)
        with GameEvaluator(
            _CountingAdditiveGame,
            {"coefficients": coefficients, "constant": 1.0},
            n_jobs=2,
            chunksize=1,
            start_method="spawn",
            worker_threads=1,
        ) as evaluator:
            result = estimate_tmc_shapley(
                evaluator,
                num_players=len(coefficients),
                total_call_budget=budget,
                seed=2048,
                truncation_patience=len(coefficients),
                num_tasks=7,
            )
        np.testing.assert_allclose(result.values, coefficients, atol=1e-15)
        self.assertEqual(result.diagnostics.utility_evaluations, budget)
        self.assertEqual(result.diagnostics.unused_calls, 0)
        self.assertEqual(result.diagnostics.num_tasks, 7)

    def test_relative_tolerance_scales_only_by_absolute_full_utility(self) -> None:
        # On this explicit order the first prefix is 0.05 below v(N)=10.
        table = np.zeros(8, dtype=np.float64)
        table[1] = 9.95
        table[3] = 9.95
        table[7] = 10.0

        relative_game = _CountingTableGame(table)
        _, relative_calls, _ = tmc_permutation_contributions(
            relative_game,
            np.asarray([0, 1, 2]),
            0.0,
            10.0,
            truncation_tolerance=0.01,
            relative_tolerance=True,
            truncation_patience=0,
        )
        absolute_game = _CountingTableGame(table)
        _, absolute_calls, _ = tmc_permutation_contributions(
            absolute_game,
            np.asarray([0, 1, 2]),
            0.0,
            10.0,
            truncation_tolerance=0.01,
            relative_tolerance=False,
            truncation_patience=0,
        )

        self.assertEqual(relative_calls, 1)
        self.assertEqual(absolute_calls, 3)

    def test_early_truncation_can_be_biased_despite_efficiency(self) -> None:
        # Every singleton already equals v(N), yet later positive and negative
        # marginals cancel.  TMC with patience=0 truncates at the singleton.
        table = np.asarray([0.0, 1.0, 1.0, 10.0, 1.0, -8.0, 2.0, 1.0])
        game = _CountingTableGame(table)
        estimates = []
        for order in itertools.permutations(range(3)):
            values, calls, truncated = tmc_permutation_contributions(
                game,
                np.asarray(order),
                0.0,
                1.0,
                truncation_tolerance=0.0,
                truncation_patience=0,
            )
            self.assertEqual(calls, 1)
            self.assertTrue(truncated)
            estimates.append(values)

        tmc_value = np.mean(estimates, axis=0)
        np.testing.assert_allclose(tmc_value, np.full(3, 1.0 / 3.0))
        np.testing.assert_allclose(
            _exact_shapley(table, 3), [0.0, 5.0, -4.0], atol=1e-15
        )
        self.assertAlmostEqual(float(tmc_value.sum()), 1.0)

    def test_tmc_invalid_inputs_fail_before_utility_use(self) -> None:
        game = _CountingAdditiveGame(np.ones(3))
        evaluator = _SequentialEvaluator(game)
        invalid = (
            {"num_players": 0, "total_call_budget": 10, "seed": 0},
            {"num_players": 3, "total_call_budget": 4, "seed": 0},
            {"num_players": 3, "total_call_budget": 10, "seed": -1},
            {
                "num_players": 3,
                "total_call_budget": 10,
                "seed": 0,
                "truncation_tolerance": -1e-3,
            },
            {
                "num_players": 3,
                "total_call_budget": 10,
                "seed": 0,
                "truncation_patience": -1,
            },
        )
        for arguments in invalid:
            with self.assertRaises(ValueError):
                estimate_tmc_shapley(evaluator, **arguments)
        self.assertEqual(game.calls, 0)


if __name__ == "__main__":
    unittest.main()

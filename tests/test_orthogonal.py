from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from frame_ofa import (GameEvaluator, estimate_orthogonal_shapley,
                       orthogonal_budget, orthogonal_permutations)
from frame_ofa.orthogonal import _frame_from_normals
from tests.test_tmc import _CountingAdditiveGame, _CountingTableGame, _SequentialEvaluator, _exact_shapley


class _NonlinearGame:
    def __init__(self, num_players):
        self.weights = np.linspace(-.7, 1.2, num_players)
        self.calls = 0

    def evaluate(self, coalition):
        self.calls += 1
        return 2.3 + np.sin(coalition @ self.weights) + .17 * float(coalition.sum()) ** 2


class OrthogonalTests(unittest.TestCase):
    def test_matches_executed_author_sampler_and_estimator(self):
        fixtures = json.loads((Path(__file__).resolve().parents[1] /
                               "third_party/shap_sampling/fixtures.json").read_text())
        for case in fixtures["cases"]:
            n, count, seed = case["num_players"], case["num_permutations"], case["seed"]
            with self.subTest(n=n, count=count):
                np.testing.assert_array_equal(orthogonal_permutations(n, count, seed), case["permutations"])
                game = _NonlinearGame(n)
                result = estimate_orthogonal_shapley(_SequentialEvaluator(game), n, 2 + count * (n - 1), seed)
                np.testing.assert_allclose(result.values, case["values"], rtol=1e-12, atol=1e-12)
                self.assertEqual(game.calls, result.diagnostics.utility_evaluations)

    def test_large_frame_is_orthonormal_and_zero_sum(self):
        for n in [2, 51, 142, 455]:
            with self.subTest(n=n):
                normal = np.random.default_rng(72).standard_normal((n - 1, n))
                frame = _frame_from_normals(normal)
                np.testing.assert_allclose(frame @ frame.T, np.eye(n - 1), atol=1e-13)
                np.testing.assert_allclose(frame.sum(axis=1), 0, atol=1e-13)
                # QR orientations match positive-residual Gram-Schmidt.
                self.assertTrue(np.all(np.sum(frame * normal, axis=1) > 0))

    def test_reverse_pairs_valid_permutations_and_global_rng_unchanged(self):
        np.random.seed(918)
        state = np.random.get_state()
        permutations = orthogonal_permutations(7, 28, 351)
        np.testing.assert_array_equal(permutations[14:], permutations[:14, ::-1])
        np.testing.assert_array_equal(np.sort(permutations), np.tile(np.arange(7), (28, 1)))
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_budget_allows_partial_frames_but_never_splits_pairs(self):
        for n in [2, 4, 51, 455]:
            for cap in [0, 2 * n - 1, 2 * n, 2 * n + 1, 228412]:
                plan = orthogonal_budget(n, cap)
                self.assertLessEqual(plan.utility_evaluations, cap)
                self.assertEqual(plan.utility_evaluations + plan.unused_calls, cap)
                if cap >= 2 * n:
                    self.assertEqual(plan.utility_evaluations, 2 + plan.num_permutations * (n - 1))
                    self.assertEqual(plan.num_permutations % 2, 0)
                    self.assertLess(plan.unused_calls, 2 * (n - 1))
        self.assertEqual(orthogonal_budget(455, 228412).num_full_blocks, 0)
        self.assertEqual(orthogonal_budget(455, 228412).tail_pairs, 251)

    def test_fail_before_evaluating_infeasible_or_invalid_request(self):
        game = _CountingAdditiveGame(np.ones(5))
        evaluator = _SequentialEvaluator(game)
        for kwargs in [{"total_call_budget": 9}, {"seed": -1}, {"num_tasks": 0},
                       {"task_chunksize": False}, {"num_players": 1.5}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                estimate_orthogonal_shapley(**(dict(evaluator=evaluator, num_players=5,
                    total_call_budget=100, seed=0) | kwargs))
        self.assertEqual(game.calls, 0)
        with self.assertRaises(ValueError):
            orthogonal_permutations(4, 3)

    def test_reverse_pair_is_exact_for_quadratic_game_with_nonzero_empty(self):
        n = 6
        rng = np.random.default_rng(74)
        linear, quadratic = rng.normal(size=n), np.triu(rng.normal(size=(n, n)), 1)
        masks = ((np.arange(1 << n)[:, None] >> np.arange(n)) & 1).astype(float)
        table = 9.2 + masks @ linear + np.einsum("bi,ij,bj->b", masks, quadratic, masks)
        game = _CountingTableGame(table)
        result = estimate_orthogonal_shapley(_SequentialEvaluator(game), n, 2 * n, 93)
        np.testing.assert_allclose(result.values, _exact_shapley(table, n), atol=1e-13)
        self.assertEqual(game.calls, 2 * n)

    def test_single_player_uses_two_endpoints(self):
        game = _CountingAdditiveGame(np.array([3.7]), constant=18)
        result = estimate_orthogonal_shapley(_SequentialEvaluator(game), 1, 29, 3)
        np.testing.assert_allclose(result.values, [3.7])
        self.assertEqual(game.calls, 2)
        self.assertEqual(result.diagnostics.unused_calls, 27)

    def test_spawn_and_task_partition_match_serial(self):
        n, cap, seed = 8, 160, 704
        a = estimate_orthogonal_shapley(_SequentialEvaluator(_NonlinearGame(n)), n, cap, seed, num_tasks=1)
        b = estimate_orthogonal_shapley(_SequentialEvaluator(_NonlinearGame(n)), n, cap, seed, num_tasks=17)
        with GameEvaluator(_NonlinearGame, {"num_players": n}, n_jobs=2) as evaluator:
            c = estimate_orthogonal_shapley(evaluator, n, cap, seed, num_tasks=5)
        np.testing.assert_allclose(a.values, b.values, atol=1e-13)
        np.testing.assert_allclose(a.values, c.values, atol=1e-13)
        self.assertAlmostEqual(c.values.sum(), c.diagnostics.full_utility - c.diagnostics.empty_utility)

    def test_nonlinear_estimate_is_unbiased_across_seeds(self):
        n = 4
        table = np.random.default_rng(244).normal(size=1 << n)
        estimates = [estimate_orthogonal_shapley(_SequentialEvaluator(_CountingTableGame(table)),
                     n, 2 + 8 * (n - 1), seed).values for seed in range(1200)]
        estimates = np.asarray(estimates)
        error = np.abs(estimates.mean(axis=0) - _exact_shapley(table, n))
        self.assertTrue(np.all(error < 5 * estimates.std(axis=0, ddof=1) / np.sqrt(len(estimates))))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest

import numpy as np

from frame_ofa.parallel import GameEvaluator
from frame_ofa.regression_baselines import (
    _solve_constrained_kernel_moments,
    _solve_gels_shapley_ratio,
    estimate_gels_shapley,
    estimate_kernel_shap,
    shapley_kernel_size_distribution,
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


def _population_moments(
    table: np.ndarray, num_players: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly integrate the KernelSHAP distribution by enumeration."""
    sizes, probabilities = shapley_kernel_size_distribution(num_players)
    del sizes
    gram = np.zeros((num_players, num_players), dtype=np.float64)
    rhs = np.zeros(num_players, dtype=np.float64)
    score_mean = np.zeros(num_players, dtype=np.float64)
    for mask in range(1, (1 << num_players) - 1):
        coalition = np.asarray(
            [bool(mask & (1 << player)) for player in range(num_players)]
        )
        size = int(coalition.sum())
        probability = probabilities[size - 1] / math.comb(
            num_players, size
        )
        row = coalition.astype(np.float64)
        utility = float(table[mask])
        gram += probability * np.outer(row, row)
        rhs += probability * row * (utility - table[0])
        score_mean += probability * row * utility
    return gram, rhs, score_mean


class _CountingAdditiveGame:
    def __init__(
        self, coefficients: np.ndarray, constant: float = 0.0
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.constant = float(constant)
        self.calls = 0

    def evaluate(self, coalition: np.ndarray) -> float:
        self.calls += 1
        return self.constant + float(self.coefficients @ coalition)


class _SpawnAdditiveGame:
    """Pickleable game factory used by the spawn integration test."""

    def __init__(
        self, coefficients: np.ndarray, constant: float = 0.0
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.constant = float(constant)

    def evaluate(self, coalition: np.ndarray) -> float:
        return self.constant + float(self.coefficients @ coalition)


class _SequentialTaskEvaluator:
    def __init__(self, game: _CountingAdditiveGame) -> None:
        self.game = game
        self.task_batch_sizes: list[int] = []

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        return np.asarray(
            [self.game.evaluate(row) for row in coalitions],
            dtype=np.float64,
        )

    def run_game_tasks(
        self, task_func, payloads, *, chunksize: int = 1
    ):
        if chunksize < 1:
            raise AssertionError("invalid chunksize")
        self.task_batch_sizes.append(len(payloads))
        return [task_func(self.game, payload) for payload in payloads]


class RegressionFormulaTests(unittest.TestCase):
    def test_size_distribution_is_normalized_symmetric_and_exact(self) -> None:
        num_players = 7
        sizes, probabilities = shapley_kernel_size_distribution(
            num_players
        )
        expected = 1.0 / (sizes * (num_players - sizes))
        expected /= expected.sum()
        np.testing.assert_array_equal(sizes, np.arange(1, num_players))
        np.testing.assert_allclose(probabilities, expected, rtol=0, atol=0)
        np.testing.assert_allclose(
            probabilities, probabilities[::-1], rtol=0, atol=0
        )
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)

    def test_exhaustive_population_moments_recover_arbitrary_games(
        self,
    ) -> None:
        for num_players in range(2, 7):
            rng = np.random.default_rng(8100 + num_players)
            table = rng.normal(size=1 << num_players)
            gram, rhs, score_mean = _population_moments(
                table, num_players
            )
            total = float(table[-1] - table[0])
            kernel = _solve_constrained_kernel_moments(
                gram, rhs, total, ridge=0.0
            )
            # Symmetry gives P(i in S)=1/2.  Therefore twice this
            # unconditional moment is the official conditional utility mean.
            gels = _solve_gels_shapley_ratio(
                2.0 * score_mean,
                np.ones(num_players, dtype=np.int64),
                total,
            )
            exact = _exact_shapley(table, num_players)
            np.testing.assert_allclose(
                kernel.values, exact, rtol=2e-13, atol=2e-13
            )
            np.testing.assert_allclose(
                gels, exact, rtol=2e-13, atol=2e-13
            )
            self.assertEqual(kernel.rank, num_players - 1)
            self.assertLess(kernel.condition, 1.0 + 1e-11)

    def test_additive_game_is_exact_for_sampled_kernel_regression(
        self,
    ) -> None:
        coefficients = np.asarray(
            [1.5, -0.25, 2.0, -3.0, 0.75, 4.5]
        )
        evaluator = _SequentialTaskEvaluator(
            _CountingAdditiveGame(coefficients, constant=91.0)
        )
        result = estimate_kernel_shap(
            evaluator,
            num_players=len(coefficients),
            total_call_budget=401,
            seed=171,
            num_tasks=9,
        )
        np.testing.assert_allclose(
            result.values, coefficients, rtol=2e-12, atol=2e-12
        )
        self.assertEqual(
            result.diagnostics.projected_gram_rank,
            len(coefficients) - 1,
        )

    def test_gels_ratio_population_identity_is_exact_for_additive_game(
        self,
    ) -> None:
        coefficients = np.asarray([3.0, -1.0, 0.25, 2.5, -4.0])
        constant = 137.0
        num_players = len(coefficients)
        table = np.empty(1 << num_players, dtype=np.float64)
        for mask in range(1 << num_players):
            coalition = np.asarray(
                [bool(mask & (1 << i)) for i in range(num_players)]
            )
            table[mask] = constant + coefficients @ coalition
        _, _, score_mean = _population_moments(table, num_players)
        values = _solve_gels_shapley_ratio(
            2.0 * score_mean,
            np.ones(num_players, dtype=np.int64),
            table[-1] - table[0],
        )
        np.testing.assert_allclose(
            values, coefficients, rtol=2e-13, atol=2e-13
        )

    def test_gels_ratio_matches_official_formula_with_zero_count(self) -> None:
        utility_sums = np.asarray([4.0, 0.0, 9.0, -2.0])
        inclusion_counts = np.asarray([2, 0, 3, 1], dtype=np.int64)
        total = 1.75

        official_denominators = inclusion_counts.copy()
        official_denominators[official_denominators == 0] = -1
        raw = (
            utility_sums
            / official_denominators
            * sum(1.0 / k for k in range(1, len(utility_sums)))
        )
        expected = raw + (total - raw.sum()) / len(raw)
        actual = _solve_gels_shapley_ratio(
            utility_sums, inclusion_counts, total
        )
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)

    def test_gels_ratio_is_invariant_to_a_common_utility_shift(self) -> None:
        utility_sums = np.asarray([11.0, 8.0, -3.0, 14.0, 2.0])
        inclusion_counts = np.asarray([7, 4, 3, 8, 2], dtype=np.int64)
        total = -2.25
        base = _solve_gels_shapley_ratio(
            utility_sums, inclusion_counts, total
        )
        shift = 137.0
        shifted = _solve_gels_shapley_ratio(
            utility_sums + shift * inclusion_counts,
            inclusion_counts,
            total,
        )
        np.testing.assert_allclose(shifted, base, rtol=0, atol=3e-13)

    def test_rank_deficient_kernel_solve_is_finite_and_efficient(
        self,
    ) -> None:
        num_players = 8
        row = np.asarray([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)
        gram = 17.0 * np.outer(row, row)
        rhs = 6.25 * row
        for ridge in (0.0, 1e-7):
            solved = _solve_constrained_kernel_moments(
                gram, rhs, total_value=-2.75, ridge=ridge
            )
            self.assertTrue(np.isfinite(solved.values).all())
            self.assertAlmostEqual(float(solved.values.sum()), -2.75)
            self.assertEqual(solved.rank, 1)
            self.assertEqual(solved.dimension, num_players - 1)
            self.assertAlmostEqual(solved.condition, 1.0)

    def test_kernel_solver_matches_full_kkt_reference(self) -> None:
        rng = np.random.default_rng(8191)
        num_players = 9
        design = rng.integers(0, 2, size=(200, num_players)).astype(
            np.float64
        )
        response = rng.normal(size=len(design))
        gram = design.T @ design
        rhs = design.T @ response
        total = -1.75
        for ridge in (0.0, 3e-5):
            system = np.zeros(
                (num_players + 1, num_players + 1),
                dtype=np.float64,
            )
            system[:num_players, :num_players] = gram + ridge * np.eye(
                num_players
            )
            system[:num_players, num_players] = 1.0
            system[num_players, :num_players] = 1.0
            target = np.concatenate((rhs, [total]))
            expected = np.linalg.solve(system, target)[:num_players]
            actual = _solve_constrained_kernel_moments(
                gram, rhs, total, ridge=ridge
            ).values
            np.testing.assert_allclose(
                actual, expected, rtol=2e-13, atol=2e-13
            )


class RegressionExecutionTests(unittest.TestCase):
    def test_exact_call_count_and_diagnostics_for_both_estimators(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        for estimator, expected_method, expected_solver in (
            (
                estimate_kernel_shap,
                "kernel_shap",
                "sampled_gram_constrained_wls_eigh_pseudoinverse",
            ),
            (
                estimate_gels_shapley,
                "gels_shapley",
                "official_algorithm_3_self_normalized_ratio",
            ),
        ):
            game = _CountingAdditiveGame(coefficients, constant=11.0)
            evaluator = _SequentialTaskEvaluator(game)
            result = estimator(
                evaluator,
                num_players=5,
                total_call_budget=83,
                seed=90210,
                num_tasks=7,
                task_chunksize=2,
            )
            diagnostics = result.diagnostics
            self.assertEqual(game.calls, 83)
            self.assertEqual(diagnostics.method, expected_method)
            self.assertEqual(diagnostics.solver, expected_solver)
            self.assertEqual(diagnostics.utility_evaluations, 83)
            self.assertEqual(diagnostics.target_call_budget, 83)
            self.assertEqual(diagnostics.unused_calls, 0)
            self.assertEqual(diagnostics.boundary_utility_evaluations, 2)
            self.assertEqual(diagnostics.inner_utility_evaluations, 81)
            self.assertEqual(diagnostics.num_samples, 81)
            self.assertEqual(diagnostics.num_tasks, 7)
            self.assertEqual(diagnostics.requested_num_tasks, 7)
            self.assertEqual(evaluator.task_batch_sizes, [7])
            self.assertEqual(int(diagnostics.layer_counts.sum()), 81)
            self.assertTrue(diagnostics.efficiency_constraint)
            self.assertAlmostEqual(
                float(result.values.sum()), coefficients.sum()
            )
            if estimator is estimate_gels_shapley:
                self.assertEqual(
                    int(diagnostics.inclusion_counts.sum()),
                    int(
                        np.dot(
                            np.arange(1, len(coefficients)),
                            diagnostics.layer_counts,
                        )
                    ),
                )
                self.assertEqual(diagnostics.zero_inclusion_players, 0)

    def test_fixed_seed_is_bitwise_reproducible(self) -> None:
        coefficients = np.asarray([1.0, -0.5, 2.25, 0.125, -3.0])
        for estimator in (estimate_kernel_shap, estimate_gels_shapley):
            results = []
            for _ in range(2):
                evaluator = _SequentialTaskEvaluator(
                    _CountingAdditiveGame(coefficients, constant=5.75)
                )
                results.append(
                    estimator(
                        evaluator,
                        num_players=5,
                        total_call_budget=257,
                        seed=7301,
                        num_tasks=13,
                        buffer_rows=17,
                    )
                )
            np.testing.assert_array_equal(
                results[0].values, results[1].values
            )
            np.testing.assert_array_equal(
                results[0].diagnostics.layer_counts,
                results[1].diagnostics.layer_counts,
            )
            np.testing.assert_array_equal(
                results[0].diagnostics.inclusion_counts,
                results[1].diagnostics.inclusion_counts,
            )

    def test_spawn_game_evaluator_runs_coarse_tasks(self) -> None:
        coefficients = np.asarray([0.5, -2.0, 1.25, 4.0, -0.75])
        with GameEvaluator(
            _SpawnAdditiveGame,
            {"coefficients": coefficients, "constant": 3.5},
            n_jobs=2,
            chunksize=1,
            start_method="spawn",
            worker_threads=1,
        ) as evaluator:
            kernel = estimate_kernel_shap(
                evaluator,
                num_players=5,
                total_call_budget=47,
                seed=808,
                num_tasks=5,
            )
            gels = estimate_gels_shapley(
                evaluator,
                num_players=5,
                total_call_budget=259,
                seed=809,
                num_tasks=128,
            )
        self.assertTrue(np.isfinite(kernel.values).all())
        self.assertEqual(kernel.diagnostics.utility_evaluations, 47)
        self.assertEqual(kernel.diagnostics.num_tasks, 5)
        self.assertAlmostEqual(float(kernel.values.sum()), coefficients.sum())
        self.assertTrue(np.isfinite(gels.values).all())
        self.assertEqual(gels.diagnostics.utility_evaluations, 259)
        self.assertEqual(gels.diagnostics.num_tasks, 128)
        self.assertAlmostEqual(float(gels.values.sum()), coefficients.sum())
        for result in (kernel, gels):
            self.assertTrue(np.isfinite(result.values).all())
            self.assertAlmostEqual(
                float(result.values.sum()), coefficients.sum()
            )

    def test_kernel_ridge_is_explicitly_reported(self) -> None:
        coefficients = np.asarray([1.0, 2.0, -1.0, 0.5])
        result = estimate_kernel_shap(
            _SequentialTaskEvaluator(
                _CountingAdditiveGame(coefficients)
            ),
            num_players=4,
            total_call_budget=23,
            seed=14,
            num_tasks=3,
            ridge=1e-6,
        )
        self.assertEqual(result.diagnostics.ridge, 1e-6)
        self.assertTrue(np.isfinite(result.values).all())

    def test_invalid_inputs_are_rejected_before_utility_calls(self) -> None:
        game = _CountingAdditiveGame(np.ones(3))
        evaluator = _SequentialTaskEvaluator(game)
        for estimator in (estimate_kernel_shap, estimate_gels_shapley):
            for kwargs in (
                {"num_players": 1, "total_call_budget": 3, "seed": 0},
                {"num_players": 3, "total_call_budget": 2, "seed": 0},
                {"num_players": 3, "total_call_budget": 3, "seed": -1},
                {
                    "num_players": 3,
                    "total_call_budget": 3,
                    "seed": 0,
                    "num_tasks": 0,
                },
            ):
                with self.assertRaises(ValueError):
                    estimator(evaluator, **kwargs)
        with self.assertRaises(ValueError):
            estimate_kernel_shap(
                evaluator,
                num_players=3,
                total_call_budget=3,
                seed=0,
                ridge=-1.0,
            )
        self.assertEqual(game.calls, 0)


if __name__ == "__main__":
    unittest.main()

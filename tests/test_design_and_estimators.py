from __future__ import annotations

import math
import time
import unittest
from itertools import permutations

import numpy as np

from frame_ofa import (
    CoalitionDesign,
    FrameOFAEstimator,
    GameEvaluator,
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    efficiency_projector,
    estimate_coupled,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    estimate_stratified,
    estimate_upstream_game,
    evaluate_boundary,
    frame_coupled_design,
    frame_diagnostics,
    iid_ofa_design,
    inner_frame_operator,
    inner_frame_target,
    inner_size_distribution,
    orbit_coupled_frame_design,
    stratified_frame_design,
)
from frame_ofa.geometry import (
    centered_directions,
    fixed_slice_frame_operator,
)
from frame_ofa.estimator import _weighted_direction_mean
from frame_ofa.design import (
    _cyclic_orbit,
    _cyclic_orbit_operator,
    _cyclic_orbit_signature,
    _cyclic_orbit_signatures,
    _minimum_one_counts,
    _random_relabel,
    _sample_iid_coalitions,
    _sample_uniform_coalition,
    _systematic_sizes,
)


class AdditiveGame:
    def __init__(self, coefficients: np.ndarray, constant: float) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.constant = float(constant)

    def __call__(self, coalition: np.ndarray) -> float:
        return self.constant + float(
            self.coefficients @ np.asarray(coalition, dtype=np.float64)
        )


class OfficialStyleAdditiveGame:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def evaluate(self, coalition: np.ndarray) -> float:
        return float(self.coefficients @ coalition)


def exact_shapley_table(
    table: np.ndarray, num_players: int
) -> np.ndarray:
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


class DesignAndEstimatorTests(unittest.TestCase):
    @staticmethod
    def _fixed_permutation_rng(permutation: tuple[int, ...]):
        class FixedPermutationRng:
            def permutation(self, length: int) -> np.ndarray:
                if length != len(permutation):
                    raise AssertionError("unexpected permutation length")
                return np.asarray(permutation, dtype=np.int64)

        return FixedPermutationRng()

    def test_minimum_one_allocation_is_squared_error_optimal(self) -> None:
        rng = np.random.default_rng(81)

        def positive_compositions(total: int, parts: int):
            if parts == 1:
                yield (total,)
                return
            for first in range(1, total - parts + 2):
                for rest in positive_compositions(
                    total - first, parts - 1
                ):
                    yield (first,) + rest

        for num_strata in range(1, 5):
            for total in range(num_strata, num_strata + 6):
                probabilities = rng.dirichlet(np.ones(num_strata))
                counts = _minimum_one_counts(total, probabilities)
                target = total * probabilities
                achieved = np.square(counts - target).sum()
                optimum = min(
                    np.square(np.asarray(candidate) - target).sum()
                    for candidate in positive_compositions(
                        total, num_strata
                    )
                )
                self.assertAlmostEqual(achieved, optimum, places=13)

    def test_random_relabeling_is_uniform_on_a_slice(self) -> None:
        num_players = 5
        base = np.array([[True, True, False, False, False]])
        counts: dict[tuple[bool, ...], int] = {}
        for permutation in permutations(range(num_players)):
            row = _random_relabel(
                base, self._fixed_permutation_rng(permutation)
            )[0]
            key = tuple(row.tolist())
            counts[key] = counts.get(key, 0) + 1
        self.assertEqual(len(counts), math.comb(num_players, 2))
        self.assertEqual(len(set(counts.values())), 1)

    def test_every_systematic_row_has_the_ofa_size_marginal(self) -> None:
        sizes, probabilities, _ = inner_size_distribution(8)
        num_rows = 7
        grid_size = 5000
        counts = np.zeros((num_rows, len(sizes)), dtype=np.int64)

        class OffsetRng:
            def __init__(self, offset: float) -> None:
                self.offset = offset

            def random(self) -> float:
                return self.offset

            def permutation(self, length: int) -> np.ndarray:
                return np.arange(length)

        for point in range(grid_size):
            sampled = _systematic_sizes(
                sizes,
                probabilities,
                num_rows,
                OffsetRng((point + 0.5) / grid_size),
            )
            for row, size in enumerate(sampled):
                counts[row, np.flatnonzero(sizes == size)[0]] += 1
        frequencies = counts / grid_size
        np.testing.assert_allclose(
            frequencies,
            np.broadcast_to(probabilities, frequencies.shape),
            atol=2.0 / grid_size,
            rtol=0.0,
        )

    def test_chunked_inner_frame_and_diagnostics_match_dense_formulas(
        self,
    ) -> None:
        rng = np.random.default_rng(903)
        num_players = 11
        sizes = rng.integers(2, num_players - 1, size=73)
        coalitions = _sample_iid_coalitions(
            num_players, sizes, rng, chunk_rows=9
        )
        directions = centered_directions(coalitions)
        _, _, normalizer = inner_size_distribution(num_players)
        radial_weights = np.sqrt(sizes * (num_players - sizes))
        dense_operator = (
            normalizer
            * np.einsum(
                "t,ti,tj->ij",
                radial_weights,
                directions,
                directions,
                optimize=True,
            )
            / len(coalitions)
        )

        target = inner_frame_target(num_players)
        residual = dense_operator - target
        eigenvalues = np.linalg.eigvalsh(residual)
        slice_mean_norms = np.asarray(
            [
                np.linalg.norm(
                    directions[sizes == size].mean(axis=0)
                )
                for size in np.unique(sizes)
            ]
        )
        dense_diagnostics = {
            "frobenius_discrepancy": float(
                np.linalg.norm(residual, ord="fro")
            ),
            "spectral_discrepancy": float(
                np.max(np.abs(eigenvalues))
            ),
            "trace_discrepancy": float(np.trace(residual)),
            "mean_direction_norm": float(
                np.linalg.norm(directions.mean(axis=0))
            ),
            "slice_mean_direction_rms": float(
                np.sqrt(np.mean(np.square(slice_mean_norms)))
            ),
            "slice_mean_direction_worst": float(
                np.max(slice_mean_norms)
            ),
        }

        for chunk_rows in (1, 7, 31, len(coalitions)):
            np.testing.assert_allclose(
                inner_frame_operator(
                    coalitions, chunk_rows=chunk_rows
                ),
                dense_operator,
                atol=2e-15,
                rtol=2e-15,
            )
            diagnostics = frame_diagnostics(
                coalitions, chunk_rows=chunk_rows
            )
            for key, expected in dense_diagnostics.items():
                self.assertAlmostEqual(
                    diagnostics[key], expected, places=13
                )

    def test_batched_iid_sampler_is_uniform_on_fixed_slice(self) -> None:
        num_players = 6
        size = 3
        num_samples = 200_000
        sizes = np.full(num_samples, size, dtype=np.int64)
        coalitions = _sample_iid_coalitions(
            num_players,
            sizes,
            np.random.default_rng(907),
            chunk_rows=17_000,
        )
        np.testing.assert_array_equal(
            coalitions.sum(axis=1), sizes
        )

        codes = (
            coalitions.astype(np.int64)
            @ (1 << np.arange(num_players, dtype=np.int64))
        )
        _, counts = np.unique(codes, return_counts=True)
        self.assertEqual(len(counts), math.comb(num_players, size))
        expected = num_samples / math.comb(num_players, size)
        # Five binomial standard deviations is a conservative distributional
        # check that remains deterministic under the fixed seed.
        standard_deviation = math.sqrt(
            num_samples
            * (1 / math.comb(num_players, size))
            * (1 - 1 / math.comb(num_players, size))
        )
        self.assertTrue(
            np.all(np.abs(counts - expected) <= 5 * standard_deviation)
        )

    def test_batched_iid_sampler_outpaces_rowwise_reference(self) -> None:
        num_players = 120
        num_samples = 10_000
        sizes = np.random.default_rng(911).integers(
            2, num_players - 1, size=num_samples
        )

        batched_times: list[float] = []
        rowwise_times: list[float] = []
        for repeat in range(2):
            rng = np.random.default_rng(919 + repeat)
            start = time.perf_counter()
            batched = _sample_iid_coalitions(
                num_players, sizes, rng
            )
            batched_times.append(time.perf_counter() - start)
            np.testing.assert_array_equal(
                batched.sum(axis=1), sizes
            )

            rng = np.random.default_rng(929 + repeat)
            start = time.perf_counter()
            rowwise = np.stack(
                [
                    _sample_uniform_coalition(
                        num_players, int(size_value), rng
                    )
                    for size_value in sizes
                ]
            )
            rowwise_times.append(time.perf_counter() - start)
            np.testing.assert_array_equal(
                rowwise.sum(axis=1), sizes
            )

        # Compare best-of-two timings to damp transient scheduler noise.  The
        # reference deliberately mirrors the former implementation.
        self.assertLess(min(batched_times), min(rowwise_times))

    def test_iid_design_seed_is_reproducible_after_batching(self) -> None:
        first = iid_ofa_design(10, 257, seed=937)
        second = iid_ofa_design(10, 257, seed=937)
        np.testing.assert_array_equal(first.sizes, second.sizes)
        np.testing.assert_array_equal(
            first.coalitions, second.coalitions
        )

    def test_coupled_design_preserves_row_sizes(self) -> None:
        design = frame_coupled_design(
            num_players=8,
            num_samples=31,
            seed=7,
            candidate_pool=20,
        )
        np.testing.assert_array_equal(
            design.coalitions.sum(axis=1), design.sizes
        )
        self.assertTrue(np.all(design.sizes >= 2))
        self.assertTrue(np.all(design.sizes <= 6))
        for size in np.unique(design.sizes):
            rows = design.coalitions[design.sizes == size]
            if len(rows) <= math.comb(8, int(size)):
                self.assertEqual(len(rows), len(np.unique(rows, axis=0)))

    def test_systematic_size_counts_are_floor_or_ceiling(self) -> None:
        num_players = 9
        num_samples = 41
        sizes, probabilities, _ = inner_size_distribution(num_players)
        expected = num_samples * probabilities
        for seed in range(20):
            design = frame_coupled_design(
                num_players=num_players,
                num_samples=num_samples,
                seed=seed,
                candidate_pool=1,
            )
            counts = np.asarray(
                [(design.sizes == size).sum() for size in sizes]
            )
            valid = (counts == np.floor(expected)) | (
                counts == np.ceil(expected)
            )
            self.assertTrue(np.all(valid))

    def test_additive_error_identity_for_coupled_estimator(self) -> None:
        rng = np.random.default_rng(123)
        num_players = 7
        coefficients = rng.normal(size=num_players)
        game = AdditiveGame(coefficients, constant=2.7)
        boundary = evaluate_boundary(game, num_players)
        design = frame_coupled_design(
            num_players, num_samples=23, seed=3, candidate_pool=24
        )
        utilities = np.asarray([game(row) for row in design.coalitions])
        estimate = estimate_coupled(design, utilities, boundary)
        predicted_error = (
            inner_frame_operator(design.coalitions)
            - inner_frame_target(num_players)
        ) @ coefficients
        np.testing.assert_allclose(
            estimate - coefficients, predicted_error, atol=3e-14
        )

    def test_n4_three_direction_design_is_additive_exact(self) -> None:
        coefficients = np.array([0.3, -1.2, 2.1, 0.8])
        game = AdditiveGame(coefficients, constant=-4.0)
        result = FrameOFAEstimator(
            num_players=4,
            num_samples=3,
            mode="coupled",
            seed=8,
            candidate_pool=6,
        ).estimate(game)
        np.testing.assert_allclose(
            result.values, coefficients, atol=3e-14
        )
        self.assertLess(
            result.design.diagnostics["frobenius_discrepancy"], 2e-14
        )

    def test_stratified_additive_error_identity(self) -> None:
        rng = np.random.default_rng(432)
        num_players = 6
        coefficients = rng.normal(size=num_players)
        game = AdditiveGame(coefficients, constant=1.3)
        design = stratified_frame_design(
            num_players, num_samples=19, seed=5, candidate_pool=16
        )
        boundary = evaluate_boundary(game, num_players)
        utilities = np.asarray([game(row) for row in design.coalitions])
        estimate = estimate_stratified(design, utilities, boundary)

        operator = np.zeros((num_players, num_players), dtype=np.float64)
        for size in range(2, num_players - 1):
            operator += fixed_slice_frame_operator(
                design.coalitions[design.sizes == size]
            )
        target = (
            (num_players - 3) / (num_players - 1)
        ) * efficiency_projector(num_players)
        np.testing.assert_allclose(
            estimate - coefficients,
            (operator - target) @ coefficients,
            atol=3e-14,
        )

    def test_general_baseline_additive_error_has_mean_term(self) -> None:
        num_players = 5
        constant = 7.0

        def game(_: np.ndarray) -> float:
            return constant

        design = stratified_frame_design(
            num_players, num_samples=4, seed=13, candidate_pool=10
        )
        boundary = evaluate_boundary(game, num_players)
        utilities = np.full(len(design.coalitions), constant)
        estimate = estimate_stratified(
            design, utilities, boundary, baseline="none"
        )
        directions = centered_directions(design.coalitions)
        predicted = np.zeros(num_players)
        for size in range(2, num_players - 1):
            take = design.sizes == size
            predicted += (
                np.sqrt(num_players / (size * (num_players - size)))
                * constant
                * directions[take].mean(axis=0)
            )
        np.testing.assert_allclose(estimate, predicted, atol=3e-14)
        self.assertGreater(np.linalg.norm(estimate), 1.0)

    def test_conditional_randomization_mse_formulas(self) -> None:
        num_players = 5
        projector = efficiency_projector(num_players)
        alpha = (num_players - 3) / (num_players - 1)
        coefficients = np.array([0.2, -0.7, 1.3, 0.1, -0.9])
        base = np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=bool,
        )
        sizes = base.sum(axis=1)
        base_coupled_error = (
            inner_frame_operator(base) - alpha * projector
        )
        coupled_rhs = (
            np.linalg.norm(projector @ coefficients) ** 2
            / (num_players - 1)
            * np.linalg.norm(base_coupled_error, ord="fro") ** 2
        )

        coupled_squared_errors = []
        slice_operators: dict[int, list[np.ndarray]] = {2: [], 3: []}
        for permutation in permutations(range(num_players)):
            relabeled = _random_relabel(
                base, self._fixed_permutation_rng(permutation)
            )
            error = (
                inner_frame_operator(relabeled) - alpha * projector
            ) @ coefficients
            coupled_squared_errors.append(float(error @ error))
            for size in (2, 3):
                slice_operators[size].append(
                    fixed_slice_frame_operator(relabeled[sizes == size])
                )
        self.assertAlmostEqual(
            float(np.mean(coupled_squared_errors)),
            coupled_rhs,
            places=13,
        )

        slice_target = projector / (num_players - 1)
        stratified_rhs = (
            np.linalg.norm(projector @ coefficients) ** 2
            / (num_players - 1)
            * sum(
                np.linalg.norm(
                    fixed_slice_frame_operator(base[sizes == size])
                    - slice_target,
                    ord="fro",
                )
                ** 2
                for size in (2, 3)
            )
        )
        stratified_squared_errors = [
            float(
                (
                    (operator_2 + operator_3 - alpha * projector)
                    @ coefficients
                )
                @ (
                    (operator_2 + operator_3 - alpha * projector)
                    @ coefficients
                )
            )
            for operator_2 in slice_operators[2]
            for operator_3 in slice_operators[3]
        ]
        self.assertAlmostEqual(
            float(np.mean(stratified_squared_errors)),
            stratified_rhs,
            places=13,
        )

    def test_frame_coupling_has_no_universal_mse_dominance(self) -> None:
        num_players = 5
        base = np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=bool,
        )
        sizes = base.sum(axis=1)
        boundary = evaluate_boundary(
            lambda _: 0.0, num_players
        )
        squared_errors = []
        for permutation in permutations(range(num_players)):
            relabeled = _random_relabel(
                base, self._fixed_permutation_rng(permutation)
            )
            design = CoalitionDesign(
                coalitions=relabeled,
                sizes=sizes,
                method="frame_coupled",
                seed=0,
            )
            utilities = np.where(sizes == 2, -1.0, 1.0)
            estimate = estimate_coupled(
                design, utilities, boundary, baseline="none"
            )
            squared_errors.append(float(estimate @ estimate))
        frame_mse = float(np.mean(squared_errors))
        iid_mse = 5.0 / 6.0
        self.assertAlmostEqual(frame_mse, 5.0 / 4.0, places=13)
        self.assertAlmostEqual(frame_mse / iid_mse, 1.5, places=13)

    def test_random_relabeling_is_unbiased_for_arbitrary_table_game(
        self,
    ) -> None:
        num_players = 5
        rng = np.random.default_rng(991)
        table = rng.normal(size=1 << num_players)

        def utility(row: np.ndarray) -> float:
            mask = sum(
                int(take) << player
                for player, take in enumerate(row)
            )
            return float(table[mask])

        truth = exact_shapley_table(table, num_players)
        boundary = evaluate_boundary(utility, num_players)
        base = np.array(
            [
                [0, 1, 0, 1, 0],
                [1, 0, 0, 1, 1],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 1, 1],
            ],
            dtype=bool,
        )
        sizes = base.sum(axis=1)
        coupled_estimates = []
        stratified_estimates = []
        for permutation in permutations(range(num_players)):
            relabeled = _random_relabel(
                base, self._fixed_permutation_rng(permutation)
            )
            utilities = np.asarray([utility(row) for row in relabeled])
            coupled_estimates.append(
                estimate_coupled(
                    CoalitionDesign(
                        relabeled, sizes, "frame_coupled", seed=0
                    ),
                    utilities,
                    boundary,
                )
            )
            stratified_estimates.append(
                estimate_stratified(
                    CoalitionDesign(
                        relabeled, sizes, "frame_stratified", seed=0
                    ),
                    utilities,
                    boundary,
                )
            )
        np.testing.assert_allclose(
            np.mean(coupled_estimates, axis=0), truth, atol=2e-14
        )
        np.testing.assert_allclose(
            np.mean(stratified_estimates, axis=0), truth, atol=2e-14
        )

    def test_frame_design_improves_exact_small_case_over_typical_iid(self) -> None:
        frame = frame_coupled_design(
            num_players=4,
            num_samples=3,
            seed=11,
            candidate_pool=6,
        )
        frame_error = frame.diagnostics["frobenius_discrepancy"]
        iid_errors = [
            iid_ofa_design(4, 3, seed).diagnostics[
                "frobenius_discrepancy"
            ]
            for seed in range(100)
        ]
        self.assertLess(frame_error, 2e-14)
        self.assertGreater(np.median(iid_errors), 0.2)

    def test_utility_count_includes_exact_boundary(self) -> None:
        game = AdditiveGame(np.arange(5, dtype=np.float64), constant=0.1)
        result = FrameOFAEstimator(
            num_players=5,
            num_samples=12,
            mode="stratified",
            seed=0,
            candidate_pool=10,
        ).estimate(game)
        self.assertEqual(result.utility_evaluations, 2 * 5 + 2 + 12)
        self.assertAlmostEqual(
            result.values.sum(),
            game(np.ones(5, dtype=bool))
            - game(np.zeros(5, dtype=bool)),
            places=13,
        )

    def test_cyclic_orbits_fix_all_ratio_denominators(self) -> None:
        num_players = 7
        num_orbits = 8
        design = cyclic_orbit_frame_design(
            num_players=num_players,
            num_orbits=num_orbits,
            seed=2,
            candidate_pool=20,
        )
        self.assertEqual(len(design.coalitions), num_players * num_orbits)
        for size in range(2, num_players - 1):
            rows = design.coalitions[design.sizes == size]
            orbit_count = len(rows) // num_players
            np.testing.assert_array_equal(
                rows.sum(axis=0),
                np.full(num_players, size * orbit_count),
            )
            np.testing.assert_array_equal(
                (~rows).sum(axis=0),
                np.full(
                    num_players, (num_players - size) * orbit_count
                ),
            )

    def test_fft_cyclic_operator_matches_direct_outer_products(self) -> None:
        rng = np.random.default_rng(29)
        for num_players in range(4, 11):
            for size in range(2, num_players - 1):
                base = np.zeros(num_players, dtype=bool)
                base[
                    rng.choice(num_players, size=size, replace=False)
                ] = True
                directions = centered_directions(_cyclic_orbit(base))
                direct = directions.T @ directions / num_players
                np.testing.assert_allclose(
                    _cyclic_orbit_operator(base), direct, atol=2e-14
                )

    def test_batched_fft_signatures_match_scalar_version(self) -> None:
        rng = np.random.default_rng(53)
        num_players = 11
        candidates = np.zeros((9, num_players), dtype=bool)
        for row in candidates:
            row[rng.choice(num_players, size=4, replace=False)] = True
        scalar = np.stack(
            [_cyclic_orbit_signature(row) for row in candidates]
        )
        np.testing.assert_allclose(
            _cyclic_orbit_signatures(candidates), scalar, atol=2e-14
        )

    def test_orbit_coupled_design_has_qstar_counts_and_balanced_orbits(
        self,
    ) -> None:
        num_players = 9
        num_orbits = 31
        design = orbit_coupled_frame_design(
            num_players=num_players,
            num_samples=num_players * num_orbits,
            seed=71,
            candidate_pool=12,
        )
        self.assertEqual(design.method, "orbit_coupled")
        self.assertEqual(
            design.coalitions.shape, (num_players * num_orbits, num_players)
        )

        orbit_sizes = design.sizes.reshape(num_orbits, num_players)
        self.assertTrue(
            np.all(orbit_sizes == orbit_sizes[:, :1])
        )
        orbit_rows = design.coalitions.reshape(
            num_orbits, num_players, num_players
        )
        np.testing.assert_array_equal(
            orbit_rows.sum(axis=1),
            np.repeat(
                orbit_sizes[:, :1], num_players, axis=1
            ),
        )

        sizes, probabilities, _ = inner_size_distribution(num_players)
        realized = np.asarray(
            [(orbit_sizes[:, 0] == size).sum() for size in sizes]
        )
        self.assertTrue(
            np.all(np.abs(realized - num_orbits * probabilities) <= 1.0)
        )

    def test_orbit_coupled_signature_diagnostics_match_materialized_rows(
        self,
    ) -> None:
        design = orbit_coupled_frame_design(
            num_players=8,
            num_samples=8 * 17,
            seed=79,
            candidate_pool=15,
        )
        direct = frame_diagnostics(design.coalitions)
        for key in (
            "frobenius_discrepancy",
            "spectral_discrepancy",
            "trace_discrepancy",
            "mean_direction_norm",
            "slice_mean_direction_rms",
            "slice_mean_direction_worst",
        ):
            self.assertAlmostEqual(
                design.diagnostics[key], direct[key], places=12
            )

    def test_orbit_coupled_works_with_linear_estimator(self) -> None:
        coefficients = np.asarray(
            [0.3, -0.6, 1.1, 0.2, -0.4, 0.8, -0.1]
        )
        game = AdditiveGame(coefficients, constant=0.7)
        design = orbit_coupled_frame_design(
            num_players=len(coefficients),
            num_samples=len(coefficients) * 13,
            seed=83,
            candidate_pool=20,
        )
        boundary = evaluate_boundary(game, len(coefficients))
        utilities = np.asarray([game(row) for row in design.coalitions])
        estimate = estimate_coupled(design, utilities, boundary)
        frame_error = (
            inner_frame_operator(design.coalitions)
            - inner_frame_target(len(coefficients))
        ) @ coefficients
        np.testing.assert_allclose(
            estimate - coefficients, frame_error, atol=3e-14
        )

    def test_chunked_weighted_direction_mean_matches_dense_formula(
        self,
    ) -> None:
        rng = np.random.default_rng(89)
        num_players = 13
        sizes = rng.integers(2, num_players - 1, size=41)
        coalitions = np.zeros((len(sizes), num_players), dtype=bool)
        for row, size in zip(coalitions, sizes):
            row[
                rng.choice(num_players, size=int(size), replace=False)
            ] = True
        residuals = rng.normal(size=len(coalitions))
        dense = np.mean(
            residuals[:, None] * centered_directions(coalitions), axis=0
        )
        for chunk_rows in (1, 3, 17, len(coalitions)):
            np.testing.assert_allclose(
                _weighted_direction_mean(
                    coalitions,
                    sizes,
                    residuals,
                    chunk_rows=chunk_rows,
                ),
                dense,
                atol=2e-15,
            )

    def test_orbit_coupled_validates_sample_multiple(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            orbit_coupled_frame_design(
                num_players=7,
                num_samples=20,
                seed=0,
            )

    def test_orbit_ratio_equals_stratified_linear_estimator(self) -> None:
        rng = np.random.default_rng(17)
        num_players = 6
        design = cyclic_orbit_frame_design(
            num_players=num_players,
            num_orbits=7,
            seed=4,
            candidate_pool=15,
        )
        # A deliberately non-additive table-valued game.
        table = rng.normal(size=1 << num_players)

        def utility(row: np.ndarray) -> float:
            mask = sum(
                int(take) << player
                for player, take in enumerate(row)
            )
            return float(table[mask])

        boundary = evaluate_boundary(utility, num_players)
        utilities = np.asarray([utility(row) for row in design.coalitions])
        ratio = estimate_ratio_ofa(design, utilities, boundary)
        official_ratio = estimate_official_ratio_ofa(
            design, utilities, boundary
        )
        linear = estimate_stratified(
            design, utilities, boundary, baseline="none"
        )
        np.testing.assert_allclose(ratio, linear, atol=3e-14)
        np.testing.assert_allclose(official_ratio, linear, atol=3e-14)

    def test_ratio_estimator_rejects_unbalanced_design(self) -> None:
        num_players = 4
        rows = np.array(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=bool,
        )
        design = CoalitionDesign(
            coalitions=rows,
            sizes=np.full(3, 2),
            method="custom",
            seed=0,
        )
        game = AdditiveGame(np.arange(4, dtype=float), constant=0.0)
        boundary = evaluate_boundary(game, num_players)
        utilities = np.asarray([game(row) for row in rows])
        with self.assertRaisesRegex(ValueError, "1-balanced"):
            estimate_ratio_ofa(design, utilities, boundary)

        wrong_width = CoalitionDesign(
            coalitions=np.ones((3, 5), dtype=bool),
            sizes=np.full(3, 5),
            method="custom",
            seed=0,
        )
        with self.assertRaisesRegex(ValueError, "player count"):
            estimate_ratio_ofa(
                wrong_width, np.zeros(3, dtype=float), boundary
            )

    def test_estimators_reject_incompatible_design_schemes(self) -> None:
        num_players = 5
        game = AdditiveGame(np.arange(5, dtype=float), constant=0.0)
        boundary = evaluate_boundary(game, num_players)
        stratified = stratified_frame_design(
            num_players, num_samples=8, seed=0
        )
        utilities = np.asarray([game(row) for row in stratified.coalitions])
        with self.assertRaisesRegex(ValueError, "coupled estimation"):
            estimate_coupled(stratified, utilities, boundary)

        coupled = frame_coupled_design(
            num_players, num_samples=8, seed=0
        )
        utilities = np.asarray([game(row) for row in coupled.coalitions])
        with self.assertRaisesRegex(ValueError, "stratified estimation"):
            estimate_stratified(coupled, utilities, boundary)

    def test_official_style_game_adapter(self) -> None:
        coefficients = np.array([1.2, -0.4, 0.7, 2.1])
        result = estimate_upstream_game(
            game_func=OfficialStyleAdditiveGame,
            game_args={"coefficients": coefficients},
            num_players=4,
            nue_avg=1,
            mode="coupled",
            seed=3,
            candidate_pool=6,
        )
        self.assertEqual(len(result.design.coalitions), 4)
        self.assertEqual(result.utility_evaluations, 14)

    def test_boundary_rows_and_serial_game_evaluator(self) -> None:
        coefficients = np.array([0.2, -0.4, 1.1, 0.7])
        rows = boundary_coalitions(4)
        with GameEvaluator(
            OfficialStyleAdditiveGame,
            {"coefficients": coefficients},
            n_jobs=1,
        ) as evaluator:
            utilities = evaluator.evaluate(rows)
        boundary = boundary_from_utilities(utilities, 4)
        direct = evaluate_boundary(
            AdditiveGame(coefficients, constant=0.0), 4
        )
        np.testing.assert_allclose(
            boundary.singletons, direct.singletons
        )
        np.testing.assert_allclose(
            boundary.leave_one_out, direct.leave_one_out
        )
        self.assertEqual(boundary.empty, direct.empty)
        self.assertEqual(boundary.full, direct.full)

        estimator = FrameOFAEstimator(
            num_players=4,
            num_samples=3,
            mode="coupled",
            seed=7,
            candidate_pool=6,
        )
        serial_result = estimator.estimate(
            AdditiveGame(coefficients, constant=0.0)
        )
        with GameEvaluator(
            OfficialStyleAdditiveGame,
            {"coefficients": coefficients},
            n_jobs=1,
        ) as evaluator:
            batch_result = estimator.estimate_with_evaluator(evaluator)
        np.testing.assert_allclose(
            batch_result.values, serial_result.values, atol=2e-15
        )
        self.assertEqual(
            batch_result.utility_evaluations,
            serial_result.utility_evaluations,
        )

    def test_invalid_configuration_fails_before_utility_use(self) -> None:
        calls = 0

        def utility(_: np.ndarray) -> float:
            nonlocal calls
            calls += 1
            return 0.0

        invalid_arguments = [
            {"num_players": 3, "num_samples": 1},
            {"num_players": 5, "num_samples": 0},
            {
                "num_players": 5,
                "num_samples": 4,
                "candidate_pool": 0,
            },
        ]
        for arguments in invalid_arguments:
            with self.assertRaises(ValueError):
                FrameOFAEstimator(**arguments).estimate(utility)
        self.assertEqual(calls, 0)

    def test_linear_endpoint_baseline_can_increase_variance(self) -> None:
        num_players = 6

        def grand_unanimity(row: np.ndarray) -> float:
            return 10.0 if np.all(row) else 0.0

        no_baseline = FrameOFAEstimator(
            num_players,
            17,
            mode="iid",
            seed=6,
            baseline="none",
        ).estimate(grand_unanimity)
        linear_baseline = FrameOFAEstimator(
            num_players,
            17,
            mode="iid",
            seed=6,
            baseline="linear",
        ).estimate(grand_unanimity)
        truth = np.full(num_players, 10.0 / num_players)
        np.testing.assert_allclose(no_baseline.values, truth, atol=2e-15)
        self.assertGreater(
            np.linalg.norm(linear_baseline.values - truth), 1e-3
        )


if __name__ == "__main__":
    unittest.main()

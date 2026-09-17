from __future__ import annotations

import math
import unittest
from itertools import combinations

import numpy as np

from frame_ofa import (
    centered_directions,
    efficiency_projector,
    evaluate_boundary,
    fixed_slice_moment_diagnostics,
    inner_size_distribution,
    shapley_boundary_vector,
)


def coalition_from_mask(mask: int, num_players: int) -> np.ndarray:
    return np.array(
        [bool(mask & (1 << player)) for player in range(num_players)],
        dtype=bool,
    )


def exact_shapley(table: np.ndarray, num_players: int) -> np.ndarray:
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
            values[player] += weight * (table[mask | bit] - table[mask])
    return values


class GeometricIdentityTests(unittest.TestCase):
    def test_direction_normalization(self) -> None:
        rng = np.random.default_rng(12)
        for num_players in range(4, 10):
            rows = []
            for size in range(1, num_players):
                row = np.zeros(num_players, dtype=bool)
                row[rng.choice(num_players, size=size, replace=False)] = True
                rows.append(row)
            directions = centered_directions(np.asarray(rows))
            np.testing.assert_allclose(
                directions.sum(axis=1), 0.0, atol=1e-14
            )
            np.testing.assert_allclose(
                np.linalg.norm(directions, axis=1), 1.0, atol=1e-14
            )

    def test_fixed_slice_tight_frame_identity(self) -> None:
        for num_players in range(4, 9):
            target = efficiency_projector(num_players) / (num_players - 1)
            for size in range(1, num_players):
                rows = np.zeros(
                    (math.comb(num_players, size), num_players), dtype=bool
                )
                for row, indices in zip(
                    rows, combinations(range(num_players), size)
                ):
                    row[list(indices)] = True
                directions = centered_directions(rows)
                empirical = directions.T @ directions / len(rows)
                np.testing.assert_allclose(empirical, target, atol=2e-14)
                np.testing.assert_allclose(
                    directions.mean(axis=0), 0.0, atol=2e-14
                )

    def test_fixed_slice_moment_diagnostics_match_exact_identities(self) -> None:
        num_players = 5
        rows = []
        sizes = []
        for size in range(2, num_players - 1):
            for indices in combinations(range(num_players), size):
                row = np.zeros(num_players, dtype=bool)
                row[list(indices)] = True
                rows.append(row)
                sizes.append(size)
        diagnostics = fixed_slice_moment_diagnostics(
            np.asarray(rows),
            np.asarray(sizes),
            require_all_inner_sizes=True,
        )
        self.assertEqual(diagnostics["missing_sizes"], [])
        self.assertEqual(diagnostics["first_moment_rms"], 0.0)
        self.assertLess(diagnostics["frame_frobenius_rms"], 2e-15)

    def test_single_row_slice_and_missing_slice_are_explicit(self) -> None:
        row = np.asarray([[1, 1, 0, 0, 0]], dtype=bool)
        diagnostics = fixed_slice_moment_diagnostics(row)
        self.assertAlmostEqual(diagnostics["first_moment_rms"], 1.0)
        self.assertAlmostEqual(
            diagnostics["frame_frobenius_rms"],
            math.sqrt(3.0 / 4.0),
        )
        self.assertEqual(diagnostics["missing_sizes"], [3])
        with self.assertRaisesRegex(ValueError, "missing"):
            fixed_slice_moment_diagnostics(
                row, require_all_inner_sizes=True
            )

    def test_ofa_probability_equalizes_radial_coefficient(self) -> None:
        for num_players in range(4, 20):
            sizes, probabilities, normalizer = inner_size_distribution(
                num_players
            )
            coefficients = np.sqrt(
                num_players / (sizes * (num_players - sizes))
            )
            ratios = coefficients / probabilities
            np.testing.assert_allclose(
                ratios,
                normalizer * np.sqrt(num_players),
                rtol=2e-15,
                atol=2e-15,
            )

    def test_exact_geometric_decomposition_with_ofa_boundary(self) -> None:
        rng = np.random.default_rng(91)
        for num_players in range(4, 8):
            table = rng.normal(size=1 << num_players)

            def utility(coalition: np.ndarray) -> float:
                mask = sum(
                    int(take) << player
                    for player, take in enumerate(coalition)
                )
                return float(table[mask])

            geometric = shapley_boundary_vector(
                evaluate_boundary(utility, num_players)
            )
            for size in range(2, num_players - 1):
                rows = []
                utilities = []
                for indices in combinations(range(num_players), size):
                    row = np.zeros(num_players, dtype=bool)
                    row[list(indices)] = True
                    rows.append(row)
                    utilities.append(utility(row))
                directions = centered_directions(np.asarray(rows))
                coefficient = np.sqrt(
                    num_players / (size * (num_players - size))
                )
                geometric += coefficient * np.mean(
                    np.asarray(utilities)[:, None] * directions, axis=0
                )

            np.testing.assert_allclose(
                geometric,
                exact_shapley(table, num_players),
                rtol=2e-13,
                atol=2e-13,
            )

    def test_boundary_compact_formula(self) -> None:
        rng = np.random.default_rng(4)
        num_players = 7
        table = rng.normal(size=1 << num_players)

        def utility(coalition: np.ndarray) -> float:
            mask = sum(
                int(take) << player for player, take in enumerate(coalition)
            )
            return float(table[mask])

        boundary = evaluate_boundary(utility, num_players)
        compact = (
            (boundary.full - boundary.empty) / num_players
            + (boundary.singletons - boundary.leave_one_out)
            / (num_players - 1)
            + (
                boundary.leave_one_out.sum()
                - boundary.singletons.sum()
            )
            / (num_players * (num_players - 1))
        )
        np.testing.assert_allclose(
            shapley_boundary_vector(boundary), compact, atol=2e-15
        )


if __name__ == "__main__":
    unittest.main()

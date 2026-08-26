from __future__ import annotations

import unittest

import numpy as np

from experiments.debug_inside_greedy_gap import (
    endpoint_size_profile,
    global_linear_with_size_profile,
)
from frame_ofa import (
    boundary_from_utilities,
    estimate_coupled,
    iid_ofa_design,
)


class InsideGreedyGapDiagnosticTests(unittest.TestCase):
    def test_generalized_endpoint_profile_matches_production_estimator(
        self,
    ) -> None:
        num_players = 8
        rng = np.random.default_rng(991)
        design = iid_ofa_design(
            num_players, 200, seed=17, compute_diagnostics=False
        )
        utilities = rng.normal(size=len(design.coalitions))
        boundary = boundary_from_utilities(
            rng.normal(size=2 * num_players + 2), num_players
        )
        expected = estimate_coupled(
            design, utilities, boundary, baseline="linear"
        )
        actual = global_linear_with_size_profile(
            design,
            utilities,
            boundary,
            endpoint_size_profile(boundary),
        )
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)

    def test_changing_size_profile_has_exact_paired_leakage_identity(
        self,
    ) -> None:
        num_players = 9
        design = iid_ofa_design(
            num_players, 300, seed=23, compute_diagnostics=False
        )
        rng = np.random.default_rng(997)
        utilities = rng.normal(size=len(design.coalitions))
        boundary = boundary_from_utilities(
            rng.normal(size=2 * num_players + 2), num_players
        )
        first_profile = endpoint_size_profile(boundary)
        second_profile = first_profile + np.linspace(
            -0.75, 0.5, num_players + 1
        )
        first = global_linear_with_size_profile(
            design, utilities, boundary, first_profile
        )
        second = global_linear_with_size_profile(
            design, utilities, boundary, second_profile
        )
        profile_only = global_linear_with_size_profile(
            design,
            second_profile[design.sizes] - first_profile[design.sizes],
            boundary_from_utilities(
                np.zeros(2 * num_players + 2), num_players
            ),
            np.zeros(num_players + 1),
        )
        np.testing.assert_allclose(
            first - second, profile_only, atol=3e-15, rtol=3e-15
        )


if __name__ == "__main__":
    unittest.main()

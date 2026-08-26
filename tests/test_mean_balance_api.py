from __future__ import annotations

import unittest

import numpy as np

from frame_ofa import (
    FrameOFAEstimator,
    estimate_upstream_game,
    per_size_frame_coupled_design,
)


class _OfficialAdditiveGame:
    def __init__(self, coefficients: np.ndarray) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)

    def evaluate(self, coalition: np.ndarray) -> float:
        return float(self.coefficients @ coalition)


class MeanBalanceApiTests(unittest.TestCase):
    def test_estimator_default_is_dimensionless_lambda0_one(self) -> None:
        estimator = FrameOFAEstimator(8, 31, candidate_pool=4)
        design = estimator.design()
        self.assertEqual(estimator.mean_balance, 1.0)
        self.assertEqual(estimator.mean_balance_mode, "normalized")
        self.assertEqual(
            design.diagnostics["mean_balance_mode"], "normalized"
        )
        self.assertAlmostEqual(
            design.diagnostics["mean_balance_lambda0"], 1.0
        )

    def test_estimator_can_reproduce_raw_design(self) -> None:
        estimator = FrameOFAEstimator(
            8,
            31,
            seed=17,
            candidate_pool=4,
            mean_balance=0.1,
            mean_balance_mode="raw",
        )
        design = estimator.design()
        self.assertEqual(design.diagnostics["mean_balance_mode"], "raw")
        self.assertAlmostEqual(
            design.diagnostics["mean_balance_effective_raw"], 0.1
        )

    def test_inside_greedy_uses_tuned_defaults_only_when_omitted(self) -> None:
        tuned = FrameOFAEstimator(
            8, 64, mode="inside_greedy", seed=7
        )
        self.assertEqual(tuned.candidate_pool, 64)
        self.assertEqual(tuned.mean_balance, 1.0 / 16.0)
        self.assertEqual(
            tuned.design().diagnostics["mean_balance_lambda0"],
            1.0 / 16.0,
        )

        explicit = FrameOFAEstimator(
            8,
            64,
            mode="inside_greedy",
            seed=7,
            candidate_pool=9,
            mean_balance=0.25,
        )
        self.assertEqual(explicit.candidate_pool, 9)
        self.assertEqual(explicit.mean_balance, 0.25)

    def test_other_modes_keep_historical_defaults(self) -> None:
        for mode in (
            "iid",
            "coupled",
            "stratified",
            "orbit",
        ):
            with self.subTest(mode=mode):
                estimator = FrameOFAEstimator(8, 31, mode=mode)
                self.assertEqual(estimator.candidate_pool, 32)
                self.assertEqual(estimator.mean_balance, 1.0)

        inside_orbit = FrameOFAEstimator(8, 31, mode="inside_orbit")
        self.assertEqual(inside_orbit.candidate_pool, 4)
        self.assertEqual(inside_orbit.mean_balance, 1.0)

    def test_coupled_omitted_defaults_are_bitwise_historical(self) -> None:
        omitted = FrameOFAEstimator(
            8, 31, mode="coupled", seed=13
        ).design()
        explicit = FrameOFAEstimator(
            8,
            31,
            mode="coupled",
            seed=13,
            candidate_pool=32,
            mean_balance=1.0,
        ).design()
        np.testing.assert_array_equal(omitted.sizes, explicit.sizes)
        np.testing.assert_array_equal(
            omitted.coalitions, explicit.coalitions
        )

    def test_direct_per_size_design_defaults_match_tuned_values(self) -> None:
        omitted = per_size_frame_coupled_design(8, 31, seed=29)
        explicit = per_size_frame_coupled_design(
            8,
            31,
            seed=29,
            candidate_pool=64,
            mean_balance=1.0 / 16.0,
        )
        np.testing.assert_array_equal(omitted.sizes, explicit.sizes)
        np.testing.assert_array_equal(
            omitted.coalitions, explicit.coalitions
        )
        self.assertEqual(
            omitted.diagnostics["mean_balance_lambda0"], 1.0 / 16.0
        )

    def test_upstream_adapter_forwards_raw_mode(self) -> None:
        result = estimate_upstream_game(
            game_func=_OfficialAdditiveGame,
            game_args={"coefficients": np.arange(8, dtype=np.float64)},
            num_players=8,
            nue_avg=5,
            mode="coupled",
            seed=19,
            candidate_pool=4,
            mean_balance=0.1,
            mean_balance_mode="raw",
            n_jobs=1,
        )
        self.assertEqual(result.design.diagnostics["mean_balance_mode"], "raw")
        self.assertAlmostEqual(
            result.design.diagnostics["mean_balance_effective_raw"], 0.1
        )

    def test_upstream_adapter_supports_both_formal_inside_modes(self) -> None:
        arguments = {
            "game_func": _OfficialAdditiveGame,
            "game_args": {"coefficients": np.arange(4, dtype=np.float64)},
            "num_players": 4,
            "seed": 23,
            "candidate_pool": 8,
            "n_jobs": 1,
        }
        greedy = estimate_upstream_game(
            **arguments, nue_avg=2, mode="inside_greedy"
        )
        orbit = estimate_upstream_game(
            **arguments, nue_avg=1, mode="inside_orbit"
        )

        self.assertEqual(greedy.design.method, "frame_coupled_per_size")
        self.assertEqual(
            greedy.design.diagnostics["inside_algorithm"], "INSIDE-Greedy"
        )
        self.assertEqual(greedy.utility_evaluations, 18)
        self.assertEqual(orbit.design.method, "cyclic_orbit_frame")
        self.assertEqual(
            orbit.design.diagnostics["inside_algorithm"], "INSIDE-Orbit"
        )
        self.assertEqual(orbit.utility_evaluations, 14)

    def test_upstream_adapter_applies_inside_greedy_tuned_defaults(self) -> None:
        arguments = {
            "game_func": _OfficialAdditiveGame,
            "game_args": {"coefficients": np.arange(8, dtype=np.float64)},
            "num_players": 8,
            "nue_avg": 8,
            "mode": "inside_greedy",
            "seed": 7,
            "n_jobs": 1,
        }
        omitted = estimate_upstream_game(**arguments)
        explicit = estimate_upstream_game(
            **arguments,
            candidate_pool=64,
            mean_balance=1.0 / 16.0,
        )
        np.testing.assert_array_equal(
            omitted.design.coalitions, explicit.design.coalitions
        )
        np.testing.assert_array_equal(omitted.values, explicit.values)
        self.assertEqual(
            omitted.design.diagnostics["mean_balance_lambda0"],
            1.0 / 16.0,
        )

    def test_upstream_adapter_preserves_coupled_defaults(self) -> None:
        arguments = {
            "game_func": _OfficialAdditiveGame,
            "game_args": {"coefficients": np.arange(8, dtype=np.float64)},
            "num_players": 8,
            "nue_avg": 4,
            "mode": "coupled",
            "seed": 11,
            "n_jobs": 1,
        }
        omitted = estimate_upstream_game(**arguments)
        explicit = estimate_upstream_game(
            **arguments, candidate_pool=32, mean_balance=1.0
        )
        np.testing.assert_array_equal(
            omitted.design.coalitions, explicit.design.coalitions
        )
        np.testing.assert_array_equal(omitted.values, explicit.values)
        self.assertEqual(
            omitted.design.diagnostics["mean_balance_lambda0"], 1.0
        )

    def test_invalid_mode_rejected_before_design(self) -> None:
        with self.assertRaisesRegex(ValueError, "mean_balance_mode"):
            FrameOFAEstimator(8, 31, mean_balance_mode="legacy")


if __name__ == "__main__":
    unittest.main()

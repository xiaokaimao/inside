from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from experiments.run_analytic_inside_comparison import (
    AnalyticGame,
    EXPERIMENT_ID,
    INSIDE_GREEDY_CANDIDATE_POOL,
    INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0,
    INSIDE_MEAN_BALANCE_LAMBDA0,
    INSIDE_ORBIT_CANDIDATE_POOL,
    LEGACY_GLOBAL_LINEAR_EXPERIMENT_ID,
    LEGACY_PER_SIZE_RATIO_EXPERIMENT_ID,
    METHOD_ORDER,
    PROTOCOL_VERSION,
    SerialAnalyticEvaluator,
    _exact_truth,
    _run_external_method,
    _run_ofa_method,
    default_output_path,
    run_experiment,
    total_call_budgets,
)
from experiments.validate_inside_comparison import (
    _inside_balance_configuration,
)
from frame_ofa import estimate_official_ratio_ofa


class AnalyticInsideComparisonTests(unittest.TestCase):
    def test_fixed_method_set_has_only_two_inside_algorithms(self) -> None:
        self.assertEqual(
            METHOD_ORDER,
            (
                "inside_greedy",
                "inside_orbit",
                "ofa_iid_linear",
                "ofa_iid_ratio",
                "cc",
                "s_diff",
                "kernel_shap",
                "tmc_shapley",
            ),
        )

    def test_default_budget_accounting_uses_analytic_player_count(self) -> None:
        airport_inner, airport_total = total_call_budgets(
            "airport", (500, 1000, 2000, 5000, 10000)
        )
        self.assertEqual(
            airport_inner,
            (50_000, 100_000, 200_000, 500_000, 1_000_000),
        )
        self.assertEqual(
            airport_total,
            (50_202, 100_202, 200_202, 500_202, 1_000_202),
        )

        voting_inner, voting_total = total_call_budgets(
            "voting", (500, 1000, 2000, 5000, 10000)
        )
        self.assertEqual(
            voting_inner,
            (25_500, 51_000, 102_000, 255_000, 510_000),
        )
        self.assertEqual(
            voting_total,
            (25_604, 51_104, 102_104, 255_104, 510_104),
        )

    def test_serial_evaluator_supports_both_games_and_coarse_tasks(self) -> None:
        airport = SerialAnalyticEvaluator("airport")
        rows = np.zeros((2, 100), dtype=bool)
        rows[0, 0] = True
        rows[1, -1] = True
        np.testing.assert_array_equal(airport.evaluate(rows), [1.0, 10.0])

        voting = SerialAnalyticEvaluator("voting")
        voting_rows = np.zeros((2, 51), dtype=bool)
        voting_rows[0, 4] = True
        voting_rows[1] = True
        np.testing.assert_array_equal(voting.evaluate(voting_rows), [0.0, 1.0])

        def task(game: AnalyticGame, full: bool) -> float:
            coalition = np.full(
                51 if game.dataset == "voting" else 100,
                full,
                dtype=bool,
            )
            return float(game.evaluate(coalition))

        self.assertEqual(voting.run_game_tasks(task, [False, True]), [0.0, 1.0])

    def test_both_ground_truths_are_exact_and_efficient(self) -> None:
        airport, airport_algorithm = _exact_truth("airport")
        voting, voting_algorithm = _exact_truth("voting")
        self.assertEqual(airport.shape, (100,))
        self.assertEqual(voting.shape, (51,))
        self.assertAlmostEqual(float(airport.sum()), 10.0, places=12)
        self.assertAlmostEqual(float(voting.sum()), 1.0, places=12)
        self.assertIn("threshold", airport_algorithm)
        self.assertIn("dynamic programming", voting_algorithm)

    def test_two_inside_names_have_the_fixed_estimator_pairing(self) -> None:
        # Forty-eight samples per player are also enough for every conditional
        # in/out mean needed by strict ratio aggregation in this deterministic
        # design/seed pair.
        greedy_inner = 51 * 48
        with patch(
            "experiments.run_analytic_inside_comparison."
            "estimate_official_ratio_ofa",
            wraps=estimate_official_ratio_ofa,
        ) as ratio_estimator:
            greedy, greedy_calls, greedy_diagnostics = _run_ofa_method(
                "voting",
                "inside_greedy",
                inner_calls=greedy_inner,
                total_calls=greedy_inner + 104,
                seed=11,
            )
        self.assertEqual(ratio_estimator.call_args.kwargs, {"missing": "raise"})
        self.assertEqual(greedy.shape, (51,))
        self.assertEqual(greedy_calls, greedy_inner + 104)
        self.assertEqual(greedy_diagnostics["candidate_pool"], 64)
        self.assertEqual(greedy_diagnostics["mean_balance_mode"], "normalized")
        self.assertEqual(greedy_diagnostics["mean_balance_lambda0"], 1.0 / 16.0)
        self.assertEqual(
            greedy_diagnostics["design_method"], "frame_coupled_per_size"
        )
        self.assertEqual(greedy_diagnostics["second_moment_scope"], "per_size")
        self.assertEqual(
            greedy_diagnostics["estimator"],
            "ofa_conditional_mean_ratio_missing_raise",
        )
        self.assertEqual(
            greedy_diagnostics["official_ratio_missing_policy"], "raise"
        )
        coverage = greedy_diagnostics["coverage"]
        self.assertTrue(coverage["all_player_size_strata_covered"])
        self.assertEqual(coverage["missing_inclusion_strata"], 0)
        self.assertEqual(coverage["missing_exclusion_strata"], 0)
        self.assertGreater(coverage["minimum_inclusion_count"], 0)
        self.assertGreater(coverage["minimum_exclusion_count"], 0)

        # Forty-eight orbits are the minimum needed to put one orbit in every
        # voting-game inner size (2, ..., 49).
        orbit_inner = 51 * 48
        orbit, orbit_calls, orbit_diagnostics = _run_ofa_method(
            "voting",
            "inside_orbit",
            inner_calls=orbit_inner,
            total_calls=orbit_inner + 104,
            seed=12,
        )
        self.assertEqual(orbit.shape, (51,))
        self.assertEqual(orbit_calls, orbit_inner + 104)
        self.assertEqual(orbit_diagnostics["candidate_pool"], 4)
        self.assertIsNone(orbit_diagnostics["mean_balance_mode"])
        self.assertIsNone(orbit_diagnostics["mean_balance_lambda0"])
        self.assertEqual(
            orbit_diagnostics["estimator"],
            "ofa_conditional_mean_ratio_strict_balanced",
        )

    def test_canonical_experiment_id_cannot_resume_legacy_semantics(self) -> None:
        self.assertNotEqual(EXPERIMENT_ID, LEGACY_GLOBAL_LINEAR_EXPERIMENT_ID)
        self.assertNotEqual(EXPERIMENT_ID, LEGACY_PER_SIZE_RATIO_EXPERIMENT_ID)
        self.assertIn("per_size_ratio", EXPERIMENT_ID)
        self.assertIn("k64_lambda1over16", EXPERIMENT_ID)
        self.assertEqual(PROTOCOL_VERSION, "per_size_ratio_k64_lambda1over16")
        self.assertEqual(INSIDE_GREEDY_CANDIDATE_POOL, 64)
        self.assertEqual(INSIDE_ORBIT_CANDIDATE_POOL, 4)
        self.assertEqual(INSIDE_GREEDY_MEAN_BALANCE_LAMBDA0, 1.0 / 16.0)
        self.assertEqual(INSIDE_MEAN_BALANCE_LAMBDA0, 1.0)
        self.assertEqual(
            default_output_path("airport"),
            Path(
                "results/airport_inside_baseline_comparison_"
                "per_size_ratio_k64_lambda1over16.json"
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.json"
            legacy_checkpoint = {
                "experiment": LEGACY_PER_SIZE_RATIO_EXPERIMENT_ID,
                "status": "running",
            }
            checkpoint.write_text(
                json.dumps(legacy_checkpoint),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different experiment"):
                run_experiment(
                    dataset="voting",
                    budget_multipliers=(48,),
                    repeats=2,
                    processes=1,
                    bootstrap_samples=10,
                    num_tasks=2,
                    output_path=checkpoint,
                )
            self.assertEqual(
                json.loads(checkpoint.read_text(encoding="utf-8")),
                legacy_checkpoint,
            )

    def test_inside_greedy_fails_instead_of_imputing_missing_ratio_cells(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "in/out observation"):
            _run_ofa_method(
                "voting",
                "inside_greedy",
                inner_calls=51,
                total_calls=155,
                seed=11,
            )

    def test_external_baselines_respect_the_same_physical_call_cap(self) -> None:
        target = 2_654
        for index, method in enumerate(
            ("cc", "s_diff", "kernel_shap", "tmc_shapley")
        ):
            values, actual, diagnostics = _run_external_method(
                "voting",
                method,
                total_calls=target,
                seed=100 + index,
                num_tasks=4,
            )
            self.assertEqual(values.shape, (51,))
            self.assertTrue(np.isfinite(values).all())
            self.assertLessEqual(actual, target)
            self.assertEqual(actual, diagnostics["utility_evaluations"])
            if method != "tmc_shapley":
                self.assertEqual(actual, target)

        # Basic CC consumes pairs, so an odd cap leaves exactly one call
        # unused instead of failing midway through an otherwise valid run.
        _, odd_actual, _ = _run_external_method(
            "voting", "cc", total_calls=2_655, seed=200, num_tasks=4
        )
        self.assertEqual(odd_actual, 2_654)

    def test_checkpoint_resume_skips_completed_method_repeat_tasks(self) -> None:
        truth, _ = _exact_truth("voting")

        def fake_task(
            dataset: str,
            repeat: int,
            method: str,
            inner_budgets: tuple[int, ...],
            total_budgets: tuple[int, ...],
            base_seed: int,
            num_tasks: int,
        ) -> dict[str, object]:
            del dataset, base_seed, num_tasks
            return {
                "repeat": repeat,
                "method": method,
                "rows": [
                    {
                        "repeat": repeat,
                        "method": method,
                        "seed": repeat,
                        "inner_utility_call_budget": inner,
                        "target_total_utility_calls": total,
                        "actual_utility_calls": total,
                        "estimate": truth.tolist(),
                        "elapsed_seconds": 0.0,
                        "diagnostics": {
                            "utility_evaluations": total,
                            **(
                                {
                                    "mean_balance_mode": "normalized",
                                    "mean_balance_lambda0": 1.0 / 16.0,
                                    "mean_balance_effective_raw": (
                                        (1.0 / 16.0) * (49.0 / 50.0)
                                    ),
                                    "candidate_pool": 64,
                                    "design_method": "frame_coupled_per_size",
                                    "second_moment_scope": "per_size",
                                    "estimator": (
                                        "ofa_conditional_mean_ratio_missing_raise"
                                    ),
                                    "official_ratio_missing_policy": "raise",
                                    "coverage": {
                                        "all_player_size_strata_covered": True,
                                        "minimum_inclusion_count": 1,
                                        "minimum_exclusion_count": 1,
                                        "missing_inclusion_strata": 0,
                                        "missing_exclusion_strata": 0,
                                        "missing_inner_sizes": 0,
                                    },
                                    "design_diagnostics": {
                                        "mean_balance_mean_weight_squared": 1.0,
                                        "mean_balance_dimension_correction": (
                                            49.0 / 50.0
                                        ),
                                        "mean_balance_normalization_factor": (
                                            49.0 / 50.0
                                        ),
                                        "mean_balance_effective_raw": (
                                            (1.0 / 16.0) * (49.0 / 50.0)
                                        ),
                                    },
                                }
                                if method == "inside_greedy"
                                else {}
                            ),
                        },
                    }
                    for inner, total in zip(
                        inner_budgets, total_budgets, strict=True
                    )
                ],
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "checkpoint.json"
            with patch(
                "experiments.run_analytic_inside_comparison."
                "_run_method_across_budgets",
                side_effect=fake_task,
            ) as worker:
                first = run_experiment(
                    dataset="voting",
                    budget_multipliers=(48,),
                    repeats=2,
                    processes=1,
                    bootstrap_samples=10,
                    num_tasks=2,
                    output_path=output,
                )
            self.assertEqual(worker.call_count, 2 * len(METHOD_ORDER))
            self.assertEqual(first["status"], "complete")
            self.assertTrue(output.exists())
            inside = first["configuration"]["inside"]
            self.assertEqual(set(inside), {"greedy", "orbit"})
            greedy = inside["greedy"]
            self.assertEqual(
                greedy["design"], "per_size_frame_coupled_design"
            )
            self.assertEqual(greedy["first_moment_scope"], "per_size")
            self.assertEqual(greedy["second_moment_scope"], "per_size")
            self.assertEqual(
                greedy["estimator"],
                "ofa_conditional_mean_ratio_missing_raise",
            )
            self.assertEqual(greedy["ratio_missing_policy"], "raise")
            self.assertEqual(greedy["candidate_pool"], 64)
            self.assertEqual(greedy["mean_balance_lambda0"], 1.0 / 16.0)
            orbit = inside["orbit"]
            self.assertEqual(orbit["design"], "cyclic_orbit_frame_design")
            self.assertEqual(orbit["candidate_pool"], 4)
            self.assertEqual(
                orbit["estimator"],
                "ofa_conditional_mean_ratio_strict_balanced",
            )
            self.assertEqual(
                _inside_balance_configuration(first["configuration"]),
                (64, "normalized", 1.0 / 16.0),
            )
            self.assertEqual(
                first["configuration"]["protocol_version"], PROTOCOL_VERSION
            )
            self.assertIn(
                "independent of worker scheduling",
                first["configuration"]["seed_semantics"],
            )

            with patch(
                "experiments.run_analytic_inside_comparison."
                "_run_method_across_budgets",
                side_effect=AssertionError("completed task was rerun"),
            ) as resumed_worker:
                second = run_experiment(
                    dataset="voting",
                    budget_multipliers=(48,),
                    repeats=2,
                    processes=2,
                    bootstrap_samples=10,
                    num_tasks=2,
                    output_path=output,
                )
            resumed_worker.assert_not_called()
            self.assertEqual(second["status"], "complete")
            self.assertEqual(len(second["raw_tasks"]), 2 * len(METHOD_ORDER))


if __name__ == "__main__":
    unittest.main()

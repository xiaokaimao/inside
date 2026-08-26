from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

from experiments.run_wine_inside_comparison import (
    BOUNDARY_CALLS,
    EXPERIMENT_ID,
    INNER_BUDGETS,
    INSIDE_GREEDY_CANDIDATE_POOL,
    METHOD_LABELS,
    MEAN_BALANCE_LAMBDA0,
    MEAN_BALANCE_MODE,
    NUM_PLAYERS,
    SOURCE_CANDIDATE_POOL,
    TOTAL_BUDGETS,
    _canonicalize_reused_summary,
    _inside_summary,
    _inside_seed,
    _merge_rows,
    parse_args,
    _ratio_coverage_diagnostics,
    _shard_key,
    _validate_inside_row,
)


def _row(repeat: int, budget_index: int) -> dict[str, object]:
    return {
        "repeat": repeat,
        "budget_index": budget_index,
        "seed": _inside_seed(repeat, budget_index),
        "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
        "inner_utility_calls": INNER_BUDGETS[budget_index],
        "total_utility_calls": TOTAL_BUDGETS[budget_index],
        "estimate": np.linspace(0.0, 1.0, NUM_PLAYERS).tolist(),
        "design_seconds": 1.0,
        "utility_and_estimation_seconds": 2.0,
        "elapsed_seconds": 3.0,
        "estimator": "ofa_conditional_mean_ratio_missing_raise",
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
            "frobenius_discrepancy": 0.25,
            "mean_balance_mode": "normalized",
            "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
            "mean_balance_effective_raw": MEAN_BALANCE_LAMBDA0
            * (1.0 - 1.0 / (NUM_PLAYERS - 1)),
            "second_moment_scope": "per_size",
        },
    }


class WineInsideComparisonTests(unittest.TestCase):
    def test_checkpoint_identity_records_normalized_lambda0(self) -> None:
        key = _shard_key("a" * 64, 0)
        self.assertEqual(key["experiment"], EXPERIMENT_ID)
        self.assertEqual(
            key["candidate_pool"], INSIDE_GREEDY_CANDIDATE_POOL
        )
        self.assertEqual(key["mean_balance_mode"], MEAN_BALANCE_MODE)
        self.assertEqual(
            key["mean_balance_lambda0"], MEAN_BALANCE_LAMBDA0
        )
        self.assertEqual(key["design"], "per_size_frame_coupled_design")
        self.assertEqual(key["second_moment_scope"], "per_size")
        self.assertEqual(
            key["estimator"],
            "estimate_official_ratio_ofa(missing='raise')",
        )

    def test_configured_and_immutable_source_candidate_pools_are_distinct(
        self,
    ) -> None:
        self.assertEqual(INSIDE_GREEDY_CANDIDATE_POOL, 64)
        self.assertEqual(SOURCE_CANDIDATE_POOL, 4)
        self.assertEqual(MEAN_BALANCE_LAMBDA0, 1.0 / 16.0)
        self.assertTrue(EXPERIMENT_ID.endswith("_k64_lambda1over16"))

    def test_default_output_is_versioned_away_from_lambda_zero(self) -> None:
        with patch.object(
            sys, "argv", ["run_wine_inside_comparison.py"]
        ):
            args = parse_args()
        self.assertIn("lambda1over16", args.output.name)
        self.assertNotIn("lambda0_", args.output.name)

    def test_lambda_one_over_sixteen_requires_matching_effective_weight(
        self,
    ) -> None:
        row = _row(0, 0)
        _validate_inside_row(row)
        row["design_diagnostics"]["mean_balance_effective_raw"] = 0.0
        with self.assertRaisesRegex(ValueError, "effective raw"):
            _validate_inside_row(row)

    def test_row_rejects_a_different_candidate_pool(self) -> None:
        row = _row(0, 0)
        row["candidate_pool"] = SOURCE_CANDIDATE_POOL
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            _validate_inside_row(row)

    def test_inside_summary_retains_configured_candidate_pool(self) -> None:
        rows = [_row(repeat, 0) for repeat in range(3)]
        summary = _inside_summary(
            rows,
            budget_index=0,
            ground_truth=np.zeros(NUM_PLAYERS),
            ground_truth_se=np.zeros(NUM_PLAYERS),
            efficiency_target=0.0,
        )
        self.assertIsNotNone(summary)
        assert summary is not None
        self.assertTrue(
            all(
                diagnostic["candidate_pool"]
                == INSIDE_GREEDY_CANDIDATE_POOL
                for diagnostic in summary["diagnostics_by_repeat"]
            )
        )

    def test_ratio_coverage_audit_and_checkpoint_rejection(self) -> None:
        coalitions = np.asarray(
            [
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [0, 1, 0, 1],
                [0, 0, 1, 1],
            ],
            dtype=bool,
        )
        complete = _ratio_coverage_diagnostics(
            coalitions, np.full(len(coalitions), 2, dtype=np.int64)
        )
        self.assertTrue(complete["all_player_size_strata_covered"])
        self.assertEqual(complete["minimum_inclusion_count"], 3)
        self.assertEqual(complete["minimum_exclusion_count"], 3)

        incomplete = _ratio_coverage_diagnostics(
            coalitions[:1], np.asarray([2], dtype=np.int64)
        )
        self.assertFalse(incomplete["all_player_size_strata_covered"])
        row = _row(0, 0)
        row["coverage"] = incomplete
        with self.assertRaisesRegex(ValueError, "complete ratio coverage"):
            _validate_inside_row(row)

    def test_merge_rows_orders_cells_and_rejects_conflicts(self) -> None:
        first = _row(1, 0)
        second = _row(0, 1)
        merged = _merge_rows([first], [second, first])
        self.assertEqual(
            [(row["repeat"], row["budget_index"]) for row in merged],
            [(0, 1), (1, 0)],
        )

        conflicting = dict(first)
        conflicting["estimate"] = np.ones(NUM_PLAYERS).tolist()
        with self.assertRaisesRegex(
            ValueError, "conflicting checkpoint rows"
        ):
            _merge_rows([first], [conflicting])

    def test_reused_summary_gets_uniform_call_accounting_aliases(
        self,
    ) -> None:
        total = TOTAL_BUDGETS[0]
        calls = [total - 10, total - 20, total - 30]
        source = {
            "estimates": np.zeros((3, NUM_PLAYERS)).tolist(),
            "actual_utility_calls_per_estimate": calls,
            "diagnostics_by_repeat": [{}, {}, {}],
        }
        result = _canonicalize_reused_summary(
            source,
            method="tmc_shapley",
            source_method="tmc_shapley",
            total_calls=total,
            source_path=Path("source.json"),
            source_sha256="a" * 64,
        )

        self.assertEqual(result["label"], METHOD_LABELS["tmc_shapley"])
        self.assertEqual(result["actual_utility_calls_by_repeat"], calls)
        self.assertEqual(result["unused_calls_per_estimate"], [10, 20, 30])
        self.assertTrue(
            all(
                diagnostic["target_total_utility_calls"] == total
                for diagnostic in result["diagnostics_by_repeat"]
            )
        )

    def test_budget_contract_counts_boundaries_once_per_estimate(
        self,
    ) -> None:
        self.assertEqual(BOUNDARY_CALLS, 2 * NUM_PLAYERS + 2)
        self.assertEqual(len(INNER_BUDGETS), 5)
        self.assertEqual(len(TOTAL_BUDGETS), 5)
        self.assertTrue(
            all(
                total == inner + BOUNDARY_CALLS
                for inner, total in zip(INNER_BUDGETS, TOTAL_BUDGETS)
            )
        )


if __name__ == "__main__":
    unittest.main()

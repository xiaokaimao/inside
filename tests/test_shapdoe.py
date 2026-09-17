from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from frame_ofa import (
    GameEvaluator, coa_field_order, component_orthogonal_array_design,
    estimate_shapdoe, latin_square_design, shapdoe_budget,
)
from frame_ofa.shapdoe import _field_tables
from tests.test_tmc import (
    _CountingAdditiveGame, _CountingTableGame, _SequentialEvaluator, _exact_shapley,
)


class ShapDoETests(unittest.TestCase):
    def test_latin_square_position_balance(self):
        for n in [1, 2, 3, 6, 10, 51]:
            design = latin_square_design(n, seed=5)
            expected = np.tile(np.arange(n), (n, 1))
            np.testing.assert_array_equal(np.sort(design, axis=1), expected)
            np.testing.assert_array_equal(np.sort(design, axis=0), expected.T)

    def test_coa_prime_and_prime_power_pair_position_balance(self):
        for n in [2, 3, 4, 5, 8, 9, 16, 25, 27]:
            design = component_orthogonal_array_design(n, 42)
            np.testing.assert_array_equal(np.sort(design, axis=1), np.tile(np.arange(n), (n*(n-1), 1)))
            expected = set((a, b) for a in range(n) for b in range(n) if a != b)
            for first, second in combinations(range(n), 2):
                self.assertEqual(set(zip(design[:, first], design[:, second])), expected)

    def test_polynomial_field_matches_official_documented_examples(self):
        self.assertEqual(_field_tables(8)[2], (1, 0, 1, 1))
        self.assertEqual(_field_tables(9)[2], (1, 1, 2))
        # In GF(9), x*x = -x-2 = 2x+1 for x^2+x+2, encoded as 7.
        self.assertEqual(_field_tables(9)[1][3, 3], 7)
        self.assertEqual(_field_tables(9)[0][3, 6], 0)

    def test_dummy_deletion_retains_valid_real_player_paths(self):
        for n, q in [(6, 7), (10, 11), (51, 53)]:
            self.assertEqual(coa_field_order(n), q)
            design = component_orthogonal_array_design(n, 7)
            np.testing.assert_array_equal(np.sort(design, axis=1), np.tile(np.arange(n), (q*(q-1), 1)))

    def test_complete_block_budget_and_actual_counter(self):
        for method in ["ls", "coa"]:
            for n in [1, 4, 6]:
                minimum = shapdoe_budget(n, 0, method).minimum_call_budget
                cap = minimum * 3 + 1
                game = _CountingAdditiveGame(np.arange(n), constant=4.75)
                result = estimate_shapdoe(_SequentialEvaluator(game), n, cap, 13, method=method, num_tasks=7)
                np.testing.assert_allclose(result.values, np.arange(n), atol=1e-13)
                self.assertEqual(game.calls, result.diagnostics.utility_evaluations)
                self.assertEqual(game.calls + result.diagnostics.unused_calls, cap)
                self.assertFalse(result.diagnostics.truncation)

    def test_infeasible_configuration_makes_no_calls(self):
        game = _CountingAdditiveGame(np.arange(4))
        evaluator = _SequentialEvaluator(game)
        for method in ["ls", "coa"]:
            minimum = shapdoe_budget(4, 0, method).minimum_call_budget
            with self.assertRaisesRegex(ValueError, "one complete design"):
                estimate_shapdoe(evaluator, 4, minimum-1, 0, method=method)
        self.assertEqual(game.calls, 0)

    def test_coa_recovers_arbitrary_three_player_game(self):
        table = np.random.default_rng(92).normal(size=8) + 5.0
        game = _CountingTableGame(table)
        result = estimate_shapdoe(_SequentialEvaluator(game), 3, 14, 11, method="coa")
        np.testing.assert_allclose(result.values, _exact_shapley(table, 3), atol=1e-14)
        self.assertIsNone(result.diagnostics.variance_of_mean)

    def test_coa_exact_quadratic_game_including_null_player_extension(self):
        for n in [4, 6, 8, 9]:
            rng = np.random.default_rng(n)
            linear = rng.normal(size=n)
            pairs = np.triu(rng.normal(size=(n, n)), 1)
            table = np.empty(2**n)
            for mask in range(2**n):
                row = ((mask >> np.arange(n)) & 1)
                table[mask] = 3.0 + linear @ row + row @ pairs @ row
            truth = linear + .5 * (pairs.sum(axis=0) + pairs.sum(axis=1))
            game = _CountingTableGame(table)
            cap = shapdoe_budget(n, 0, "coa").minimum_call_budget
            result = estimate_shapdoe(_SequentialEvaluator(game), n, cap, 19, method="coa")
            np.testing.assert_allclose(result.values, truth, atol=1e-13)

    def test_constant_shift_invariance_and_efficiency(self):
        table = np.random.default_rng(51).normal(size=64)
        for method in ["ls", "coa"]:
            cap = 2 + 3*shapdoe_budget(6, 0, method).calls_per_design
            outputs = [estimate_shapdoe(_SequentialEvaluator(_CountingTableGame(table+offset)),
                       6, cap, 10, method=method) for offset in [0, 11.25]]
            np.testing.assert_allclose(outputs[0].values, outputs[1].values, atol=1e-13)
            self.assertAlmostEqual(outputs[0].values.sum(), table[-1]-table[0])

    def test_task_partition_invariance_and_block_uncertainty(self):
        table = np.random.default_rng(91).normal(size=64)
        for method in ["ls", "coa"]:
            one = shapdoe_budget(6, 0, method)
            cap = 2 + 4*one.calls_per_design
            outputs = [estimate_shapdoe(_SequentialEvaluator(_CountingTableGame(table)),
                       6, cap, 11, method=method, num_tasks=tasks) for tasks in [1, 7, 128]]
            for output in outputs[1:]:
                np.testing.assert_allclose(output.values, outputs[0].values, atol=1e-14)
                np.testing.assert_allclose(output.diagnostics.variance_of_mean,
                                           outputs[0].diagnostics.variance_of_mean, atol=1e-14)

    def test_spawn_worker_equivalence(self):
        table = np.random.default_rng(91).normal(size=16)
        for method in ["ls", "coa"]:
            outputs = []
            for jobs in [1, 2]:
                with GameEvaluator(_CountingTableGame, {"table": table}, n_jobs=jobs) as evaluator:
                    outputs.append(estimate_shapdoe(evaluator, 4, 250, 11, method=method, num_tasks=7))
            np.testing.assert_allclose(outputs[0].values, outputs[1].values, atol=1e-14)

    def test_invalid_inputs(self):
        for n, budget, method in [(0, 20, "ls"), (4, -1, "ls"), (4, 20, "bad"), (4.5, 20, "ls"), (4, True, "ls")]:
            with self.assertRaises(ValueError):
                shapdoe_budget(n, budget, method)

    def test_official_source_fingerprints(self):
        import hashlib
        root = Path(__file__).resolve().parents[1] / "third_party/shapdoe"
        metadata = json.loads((root/"provenance.json").read_text())
        for filename, digest in metadata["files"].items():
            self.assertEqual(hashlib.sha256((root/filename).read_bytes()).hexdigest(), digest)

    def test_executed_official_r_design_and_estimator_fixtures(self):
        root = Path(__file__).resolve().parents[1] / "third_party/shapdoe/fixtures"
        for case, n, method in [("ls5", 5, "ls"), ("coa5", 5, "coa"),
                                ("coa8", 8, "coa"), ("coa9", 9, "coa"),
                                ("coa7_project6", 6, "coa")]:
            with self.subTest(case=case):
                draws = [np.fromstring(line, sep=",", dtype=np.int64)
                         for line in (root/f"{case}_draws.txt").read_text().splitlines()]
                if method == "ls":
                    draws[1] -= 1
                    draws[2] -= 1
                class Replay:
                    def __init__(self):
                        self.cursor = 0
                    def permutation(self, _):
                        result = draws[self.cursor].copy()
                        self.cursor += 1
                        return result
                expected_design = np.loadtxt(root/f"{case}_design.csv", delimiter=",", dtype=np.int64)
                expected_design = expected_design[expected_design < n].reshape(-1, n)
                expected_values = np.loadtxt(root/f"{case}_values.csv", delimiter=",").ravel()[:n]
                masks = np.arange(2**n)
                table = np.sin(.37*masks) + np.cos(.11*masks) + masks**2/10000 - 1
                with patch("frame_ofa.shapdoe.np.random.default_rng", side_effect=lambda *a, **k: Replay()):
                    design = (latin_square_design(n) if method == "ls"
                              else component_orthogonal_array_design(n))
                    np.testing.assert_array_equal(design, expected_design)
                    result = estimate_shapdoe(_SequentialEvaluator(_CountingTableGame(table)), n,
                        shapdoe_budget(n, 0, method).minimum_call_budget, 0, method=method, num_tasks=1)
                np.testing.assert_allclose(result.values, expected_values, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()

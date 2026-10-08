import unittest
import numpy as np
from frame_ofa.interaction_game import make_interaction_game, InteractionGame
from experiments.benchmark_synthetic import exact_shapley
from experiments.run_large_interaction import evaluate_boundary_batched
from frame_ofa import boundary_coalitions, boundary_from_utilities


class InteractionGameTests(unittest.TestCase):
    def test_closed_form_against_exhaustive_marginals(self):
        n = 7
        rows = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
        for degree in (2, 3):
            game = make_interaction_game(n, degree)
            table = game.evaluate(rows, batch_size=3)
            np.testing.assert_allclose(game.exact_shapley(), exact_shapley(table, n), atol=1e-12)
            self.assertAlmostEqual(game.exact_shapley().sum(), table[-1]-table[0])

    def test_paired_games_and_boundary_batches(self):
        pair = make_interaction_game(12, 2)
        cubic = make_interaction_game(12, 3)
        np.testing.assert_array_equal(pair.additive, cubic.additive)
        np.testing.assert_array_equal(pair.pairs, cubic.pairs)
        np.testing.assert_array_equal(pair.pair_weights, cubic.pair_weights)
        for game in (pair, cubic):
            expected = boundary_from_utilities(game.evaluate(boundary_coalitions(12)), 12)
            actual = evaluate_boundary_batched(game)
            np.testing.assert_allclose(actual.leave_one_out, expected.leave_one_out)
            np.testing.assert_allclose(actual.singletons, expected.singletons)
            self.assertAlmostEqual(actual.full, expected.full)
        b = evaluate_boundary_batched(pair)
        np.testing.assert_allclose(pair.exact_shapley(), (b.singletons + b.full-b.leave_one_out)/2)
        b = evaluate_boundary_batched(cubic)
        self.assertGreater(np.linalg.norm(cubic.exact_shapley()-(b.singletons+b.full-b.leave_one_out)/2), .01)

    def test_large_game_efficiency(self):
        game = make_interaction_game(5000, 3)
        self.assertEqual(game.pairs.shape, (20000, 2))
        self.assertEqual(game.triples.shape, (10000, 3))
        self.assertAlmostEqual(game.exact_shapley().sum(), game.evaluate(np.ones((1,5000)))[0])

    def test_invalid_binary_coalition(self):
        with self.assertRaises(ValueError):
            make_interaction_game(5).evaluate(np.full((2,5), .5))

    def test_runner_counts_and_resumes(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from experiments.run_large_interaction import parser, run
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'json' / 'test.json'
            args = parser().parse_args(['--players', '8', '--degrees', '2', '3',
                '--budgets', '20', '--methods', 'ofa_linear', '--repeats', '2',
                '--output', str(path)])
            report = run(args)
            self.assertEqual(len(report['cells']), 4)
            self.assertEqual(report['status'], 'complete')
            for cell in report['cells']:
                self.assertEqual(cell['utility_calls'], 20*8+2*8+2)
                self.assertTrue(np.isfinite(cell['rmse']))
            self.assertEqual(len(run(args)['cells']), 4)
            for summary in report['summary']:
                self.assertIsNotNone(summary['rmse']['std'])

"""Budget, exact-reference, and method adapter checks for the synthetic comparison."""
import unittest
import numpy as np
from experiments.run_interaction_baselines import METHODS, GameAdapter, run_task
from frame_ofa.interaction_game import make_interaction_game


class InteractionBaselineTests(unittest.TestCase):
    def test_scalar_batch_agreement(self):
        game=make_interaction_game(8)
        adapter=GameAdapter(game)
        rows=np.random.default_rng(12).integers(0,2,(12,8)).astype(bool)
        np.testing.assert_allclose([adapter.evaluate(z) for z in rows], game.evaluate(rows))

    def test_all_methods_use_same_truth_and_stay_within_budget(self):
        for method in METHODS:
            with self.subTest(method=method):
                cells=run_task(8,20,0,method,20260923,1)
                self.assertEqual(len(cells),2)
                for cell in cells:
                    self.assertLessEqual(cell['actual_calls'],178)
                    truth=make_interaction_game(8,cell['degree'],20260923+8).exact_shapley()
                    self.assertAlmostEqual(cell['rmse'],np.sqrt(np.mean((np.array(cell['estimate'])-truth)**2)))
                if method=='orthogonal':
                    self.assertLess(cells[0]['rmse'],1e-12)

    def test_extension_reuses_cells_without_modifying_source(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from experiments.run_interaction_baselines import parser, run
        with TemporaryDirectory() as folder:
            source=Path(folder)/'json'/'source.json'
            output=Path(folder)/'json'/'extended.json'
            common=['--players','8','--repeats','1','--methods','ofa','cc','--jobs','1']
            first=run(parser().parse_args(common+['--budgets','20','--output',str(source)]))
            original=source.read_bytes()
            extended=run(parser().parse_args(common+['--budgets','20','40',
                '--reuse-source',str(source),'--output',str(output)]))
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(extended['reuse']['cells'],4)
            self.assertEqual(len(extended['cells']),8)
            self.assertEqual(extended['cells'][:4],first['cells'])
            self.assertEqual(extended['status'],'complete')

    def test_sdiff_parallel_payloads_preserve_exact_results(self):
        serial=run_task(12,40,0,'s_diff',20260923,1,1)
        parallel=run_task(12,40,0,'s_diff',20260923,1,8)
        for a,b in zip(serial,parallel):
            np.testing.assert_array_equal(a['estimate'],b['estimate'])
            self.assertEqual(a['rmse'],b['rmse'])
            self.assertEqual(a['actual_calls'],b['actual_calls'])

    def test_degree_three_only_matches_original(self):
        for method in ('inside_greedy', 'inside_orbit', 'ofa', 'cc', 's_diff'):
            with self.subTest(method=method):
                original=run_task(8,20,0,method,20260923,1)[1]
                selected=run_task(8,20,0,method,20260923,1,degrees=(3,))
                self.assertEqual(len(selected),1)
                np.testing.assert_array_equal(original['estimate'],selected[0]['estimate'])
                self.assertEqual(original['seed'],selected[0]['seed'])

    def test_orbit_optional_diagnostics_preserve_design(self):
        from frame_ofa import cyclic_orbit_frame_design
        full=cyclic_orbit_frame_design(12,24,seed=4,candidate_pool=4)
        lean=cyclic_orbit_frame_design(12,24,seed=4,candidate_pool=4,compute_frame_diagnostics=False)
        np.testing.assert_array_equal(full.coalitions,lean.coalitions)
        np.testing.assert_array_equal(full.sizes,lean.sizes)

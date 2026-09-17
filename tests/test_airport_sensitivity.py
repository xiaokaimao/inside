import unittest
import numpy as np
from experiments.run_airport_sensitivity import configurations, summarize, LAMBDAS, POOLS


class AirportSensitivityTests(unittest.TestCase):
    def test_one_at_a_time_grid_shares_default_once(self):
        grid=configurations()
        self.assertEqual(len(grid),11)
        self.assertEqual(sum((lam,k)==(1/16,64) for lam,k in grid),1)
        self.assertEqual({lam for lam,k in grid if k==64},set(LAMBDAS))
        self.assertEqual({k for lam,k in grid if lam==1/16},set(POOLS))
        self.assertTrue(all(lam==1/16 or k==64 for lam,k in grid))

    def test_summary_is_mean_of_seed_metrics_and_sample_std(self):
        cells=[dict(repeat=i,lambda0=1/16,candidate_pool=64,
                    rmse=x,D1=2*x,D2=3*x,design_seconds=4*x)
               for i,x in enumerate([1.,2.,3.])]
        stats=summarize(cells)[0]['statistics']
        self.assertEqual(stats['rmse']['mean'],2.)
        self.assertEqual(stats['rmse']['std'],1.)
        self.assertEqual(stats['design_seconds']['std'],4.)
        self.assertNotEqual(stats['rmse']['mean'],np.sqrt(np.mean([1,4,9])))

    def test_normalization_uses_dataset_player_count(self):
        cell=dict(repeat=0,lambda0=1/16,candidate_pool=64,rmse=1,D1=2,D2=3,design_seconds=4)
        for n in (100,142,455):
            row=summarize([cell],n)[0]
            self.assertAlmostEqual(row['effective_lambda'],(1/16)*(n-2)/(n-1))

    def test_incomplete_summary_does_not_invent_sd(self):
        cell=dict(repeat=0,lambda0=1/16,candidate_pool=64,rmse=1,D1=2,D2=3,design_seconds=4)
        self.assertIsNone(summarize([cell])[0]['statistics']['rmse']['std'])

if __name__=='__main__':unittest.main()

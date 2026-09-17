import copy
import unittest
from experiments.run_airport_sensitivity import summarize, validate
from experiments.run_airport_inside_ablations import _cell_seeds


def report(n):
    seeds=_cell_seeds(20260915,0,0)
    cell={'repeat':0,'lambda0':1/16,'candidate_pool':64,'seeds':seeds,
        'actual_utility_calls':500*n+2*n+2,'coverage':{'all_player_size_strata_covered':True},
        'size_schedule_sha256':'shared-schedule','estimate':[1.]*n,'rmse':1.,
        'D1':.1,'D2':.2,'design_seconds':2.,
        'design_diagnostics':{'relabel_seed':seeds['component_relabel'],
            'relabel_permutation_sha256':'shared-relabel','mean_balance_effective_raw':(1/16)*(n-2)/(n-1),'design_jobs':16},
        'geometry':{'sizes':list(range(2,n-1)), 'first_moment_norm_by_size':[.1]*(n-3),
                    'frame_frobenius_by_size':[.2]*(n-3)}}
    return {'experiment':'svm_inside_one_at_a_time_sensitivity_v1',
        'configuration':{'players':n,'inner_calls':500*n,'base_seed':20260915,'design_jobs':16},
        'ground_truth':[0.]*n,'cells':[cell],'summary':summarize([cell],n)}


class SVMSensitivityTests(unittest.TestCase):
    def test_player_specific_budget_and_normalization(self):
        for n in (142,455):
            with self.subTest(n=n):validate(report(n),complete=False)

    def test_airport_normalization_is_rejected_for_svm(self):
        data=report(455)
        data['cells'][0]['design_diagnostics']['mean_balance_effective_raw']=(1/16)*98/99
        with self.assertRaises(AssertionError):validate(data,complete=False)

    def test_missing_size_or_incomplete_run_cannot_pass(self):
        data=report(142)
        with self.assertRaisesRegex(ValueError,'Expected 33'):validate(data)
        data['cells'][0]['geometry']['sizes'].pop()
        with self.assertRaisesRegex(ValueError,'Missing size'):validate(data,complete=False)

    def test_size_allocation_must_be_paired_across_configurations(self):
        data=report(142);cell=copy.deepcopy(data['cells'][0]);cell['candidate_pool']=8
        cell['size_schedule_sha256']='different';data['cells'].append(cell)
        with self.assertRaisesRegex(ValueError,'Unpaired size schedule'):validate(data,complete=False)

if __name__=='__main__':unittest.main()

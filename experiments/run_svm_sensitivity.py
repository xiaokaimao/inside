"""Paired Wine/Breast Cancer lambda/K sensitivity at the existing 500n budget."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from experiments.run_airport_sensitivity import configurations, summarize, validate, LAMBDAS, POOLS
from experiments.run_airport_inside_ablations import _cell_seeds, _compact_design_diagnostics, _sha256_array
from experiments.run_wine_inside_comparison import validate_reconstructed_dataset, _ratio_coverage_diagnostics
from experiments.sklearn_data import load_sklearn_train_test_split
from experiments.iris_sklearn_game import SklearnClassificationGame
from frame_ofa import (GameEvaluator, boundary_from_utilities, per_size_frame_coupled_design,
                       fixed_slice_moment_diagnostics, estimate_official_ratio_ofa)


def write(path, report):
    tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)


def run(dataset, jobs=16, utility_jobs=120):
    source=Path(f'results/json/{dataset}_inside_with_shapdoe_orthogonal_3repeats.json')
    original=json.loads(source.read_text())['source_report']['base_report']
    args,metadata=load_sklearn_train_test_split(dataset,test_size=.2,dataset_seed=2024)
    validate_reconstructed_dataset(original['dataset'],metadata)
    args=args|{'model':'rbf_svm','regularization':1.0}
    n=len(args['y_valued']);inner=500*n;base_seed=20260915
    truth=np.asarray(original['ground_truth']['values'])
    boundary_values=np.asarray(original['boundary_reconstruction']['utilities'])
    boundary=boundary_from_utilities(boundary_values,n)
    # Check model-specific special cases against the recorded game boundary.
    game=SklearnClassificationGame(**args)
    assert game.evaluate(np.zeros(n,dtype=bool))==boundary_values[0]
    cfg={'dataset':dataset,'players':n,'inner_calls':inner,'total_calls':inner+2*n+2,
         'base_seed':base_seed,'repeats':3,'lambda0_grid':list(LAMBDAS),'candidate_pool_grid':list(POOLS),
         'fixed_K':64,'fixed_lambda0':1/16,'design_jobs':jobs,'utility_jobs':utility_jobs,
         'start_method':'fork','cell_execution':'sequential',
         'lambda_parameterization':'normalized lambda0; effective coefficient = lambda0*(n-2)/(n-1)',
         'timing':'design construction wall seconds including worker overhead, excluding SVM utilities and geometry',
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
         'design_source_sha256':hashlib.sha256(Path('frame_ofa/design.py').read_bytes()).hexdigest(),
         'reference':'existing Monte Carlo reference; fixed across all settings',
         'boundary_policy':'recorded boundary utilities reused, 2n+2 calls included in equivalent per-estimate budget',
         'uncertainty':'sample SD across three estimator seeds, ddof=1; fixed split',
         'scope':'single dataset, fixed 500n interior-call budget'}
    output=Path(f'results/json/{dataset}_inside_sensitivity.json')
    report={'experiment':'svm_inside_one_at_a_time_sensitivity_v1','status':'running',
            'configuration':cfg,'dataset':metadata,'game':original['game'],
            'ground_truth':truth.tolist(),'reference_metadata':original['ground_truth'],
            'boundary_utilities':boundary_values.tolist(),
            'environment':{'python':platform.python_version(),'cpu_affinity_count':len(os.sched_getaffinity(0)),
              'threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}},
            'cells':[],'summary':[]}
    if output.exists():
        report=json.loads(output.read_text())
        if report['configuration']!=cfg:raise ValueError('Resume configuration mismatch')
        validate(report,complete=False)
    write(output,report)
    done={(c['repeat'],c['lambda0'],c['candidate_pool']) for c in report['cells']}
    with GameEvaluator(SklearnClassificationGame,args,n_jobs=utility_jobs,chunksize=64,
                       start_method='fork',worker_threads=1) as evaluator:
        for repeat in range(3):
            order=configurations();np.random.default_rng(np.random.SeedSequence([base_seed,repeat,913])).shuffle(order)
            seeds=_cell_seeds(base_seed,repeat,0)
            for lam,k in order:
                if (repeat,lam,k) in done:continue
                print(f'{dataset}: start {len(report["cells"])+1}/33 repeat={repeat} lambda={lam:g} K={k}',flush=True)
                start=time.perf_counter()
                design=per_size_frame_coupled_design(n,inner,seed=seeds['component'],candidate_pool=k,
                    mean_balance=lam,second_moment_weight=1.0,mean_balance_mode='normalized',
                    relabel_seed=seeds['component_relabel'],design_jobs=jobs,design_start_method='fork',
                    compute_frame_diagnostics=False)
                elapsed=time.perf_counter()-start
                print(f'{dataset}: design {elapsed:.2f}s; evaluating {inner} SVM utilities',flush=True)
                start=time.perf_counter();utilities=evaluator.evaluate(design.coalitions)
                utility_seconds=time.perf_counter()-start
                print(f'{dataset}: SVM utilities {utility_seconds:.2f}s; geometry/estimation',flush=True)
                start=time.perf_counter()
                geometry=fixed_slice_moment_diagnostics(design.coalitions,design.sizes,require_all_inner_sizes=True)
                coverage=_ratio_coverage_diagnostics(design.coalitions,design.sizes)
                estimate=np.asarray(estimate_official_ratio_ofa(design,utilities,boundary,missing='raise'))
                cell={'repeat':repeat,'lambda0':lam,'candidate_pool':k,'seeds':seeds,
                      'estimate':estimate.tolist(),'rmse':float(np.sqrt(np.mean((estimate-truth)**2))),
                      'D1':geometry['first_moment_rms'],'D2':geometry['frame_frobenius_rms'],
                      'design_seconds':elapsed,'utility_seconds':utility_seconds,
                      'geometry_and_estimation_seconds':time.perf_counter()-start,
                      'actual_utility_calls':inner+2*n+2,'new_physical_svm_queries':inner,
                      'coverage':coverage,'geometry':geometry,'size_schedule_sha256':_sha256_array(design.sizes),
                      'design_diagnostics':_compact_design_diagnostics(design.diagnostics),
                      'completed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
                report['cells'].append(cell);report['summary']=summarize(report['cells'],n)
                report['status']='complete' if len(report['cells'])==33 else 'running'
                validate(report,complete=False);write(output,report)
                print(f'{dataset}: done {len(report["cells"])}/33 RMSE={cell["rmse"]:.6g}',flush=True)
                del design,utilities
    validate(report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',nargs='+',choices=['wine','cancer'],default=['wine','cancer'])
    parser.add_argument('--design-jobs',type=int,default=16)
    parser.add_argument('--utility-jobs',type=int,default=120)
    a=parser.parse_args()
    for dataset in a.datasets:run(dataset,a.design_jobs,a.utility_jobs)

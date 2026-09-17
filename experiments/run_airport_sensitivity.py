"""Paired one-at-a-time Airport sensitivity; sequential, comparable timing.

Lambda denotes the normalized lambda0 interface used in the main experiment.
The effective coefficient is lambda0 * 98/99 for this 100-player game.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
from experiments.airport_game import AIRPORT_COSTS
from experiments.run_airport_inside_ablations import _cell_seeds, _evaluate_design
from frame_ofa import (boundary_coalitions, boundary_from_utilities,
                       evaluate_airport, exact_airport_shapley,
                       per_size_frame_coupled_design)

LAMBDAS = (1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0)
POOLS = (8, 16, 32, 64, 128)
EXPERIMENT = 'airport_inside_one_at_a_time_sensitivity_v1'


def configurations():
    return list(dict.fromkeys([(v, 64) for v in LAMBDAS] + [(1/16, k) for k in POOLS]))


def summarize(cells, players=100):
    rows = []
    for lam, k in configurations():
        selected = sorted((c for c in cells if c['lambda0'] == lam and c['candidate_pool'] == k), key=lambda c: c['repeat'])
        if not selected:
            continue
        row = {'lambda0': lam, 'effective_lambda': lam*(players-2)/(players-1), 'candidate_pool': k,
               'repeats': len(selected), 'statistics': {}}
        for metric in ('rmse', 'D1', 'D2', 'design_seconds'):
            values = [c[metric] for c in selected]
            row['statistics'][metric] = {'values': values, 'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)) if len(values)>1 else None}
        rows.append(row)
    return rows


def validate(report, complete=True):
    cfg = report['configuration']
    n = cfg['players']
    if report['experiment'] not in (EXPERIMENT, 'svm_inside_one_at_a_time_sensitivity_v1'):
        raise ValueError('Unexpected experiment')
    keys = set()
    schedule = {}
    relabel = {}
    for c in report['cells']:
        key = (c['repeat'], c['lambda0'], c['candidate_pool'])
        if key in keys or (key[1], key[2]) not in configurations() or not 0 <= key[0] < 3:
            raise ValueError('Duplicate or unexpected cell')
        keys.add(key)
        if c['actual_utility_calls'] != cfg['inner_calls'] + 2*n+2:
            raise ValueError('Unequal utility-call budgets')
        if not c['coverage']['all_player_size_strata_covered']:
            raise ValueError('Missing conditional-mean strata')
        expected_seeds = _cell_seeds(cfg['base_seed'], c['repeat'], 0)
        if c['seeds'] != expected_seeds:
            raise ValueError('Seed mismatch')
        diag = c['design_diagnostics']
        if diag['relabel_seed'] != expected_seeds['component_relabel']:
            raise ValueError('Relabel seed mismatch')
        fingerprint = relabel.setdefault(c['repeat'], diag['relabel_permutation_sha256'])
        if fingerprint != diag['relabel_permutation_sha256']:
            raise ValueError('Unpaired player relabeling')
        np.testing.assert_allclose(diag['mean_balance_effective_raw'], c['lambda0']*(n-2)/(n-1))
        if diag['design_jobs'] != cfg['design_jobs']:
            raise ValueError('Inconsistent design-time worker count')
        previous = schedule.setdefault(c['repeat'], c['size_schedule_sha256'])
        if previous != c['size_schedule_sha256']:
            raise ValueError('Unpaired size schedule')
        truth = np.asarray(report['ground_truth'])
        rmse = np.sqrt(np.mean((np.asarray(c['estimate'])-truth)**2))
        np.testing.assert_allclose(c['rmse'], rmse, rtol=1e-12)
        g=c['geometry']
        if g['sizes'] != list(range(2,n-1)):
            raise ValueError('Missing size in equal-size moment aggregate')
        for name, field in [('D1','first_moment_norm_by_size'),('D2','frame_frobenius_by_size')]:
            np.testing.assert_allclose(c[name], np.sqrt(np.mean(np.square(g[field]))), rtol=1e-12)
        if not all(np.isfinite(c[m]) and c[m]>=0 for m in ('rmse','D1','D2','design_seconds')):
            raise ValueError('Invalid metric')
    if complete and len(keys)!=33:
        raise ValueError(f'Expected 33 cells, found {len(keys)}')
    if report['summary'] != summarize(report['cells'], n):
        raise ValueError('Summary differs from raw cells')


def run(output: Path, inner_calls=50000, jobs=16, base_seed=20260915):
    cfg = {'players':100, 'inner_calls':inner_calls, 'total_calls':inner_calls+202,
           'base_seed':base_seed, 'repeats':3, 'lambda0_grid':list(LAMBDAS),
           'candidate_pool_grid':list(POOLS), 'fixed_K':64, 'fixed_lambda0':1/16,
           'design_jobs':jobs, 'start_method':'fork', 'cell_execution':'sequential',
           'lambda_parameterization':'normalized lambda0; effective coefficient = lambda0 * 98/99',
           'timing':'perf_counter wall seconds around design construction only, including worker overhead; excludes geometry, utilities and estimation',
           'seed_pairing':'same size schedule and relabel seed within each repeat; K changes do not guarantee nested candidate pools',
           'uncertainty':'mean +/- sample standard deviation across estimator seeds, ddof=1',
           'scope':'single fixed budget and one dataset; descriptive sensitivity, not parameter selection'}
    design_path = Path(__file__).resolve().parents[1] / 'frame_ofa/design.py'
    cfg['design_source_sha256'] = hashlib.sha256(design_path.read_bytes()).hexdigest()
    truth=exact_airport_shapley(AIRPORT_COSTS)
    report={'experiment':EXPERIMENT, 'status':'running', 'configuration':cfg,
            'environment':{'python':platform.python_version(), 'platform':platform.platform(),
                'cpu_affinity_count':len(os.sched_getaffinity(0)),
                'blas_threads':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS')}},
            'ground_truth':truth.tolist(), 'cells':[], 'summary':[]}
    if output.exists():
        report=json.loads(output.read_text())
        if report['configuration']!=cfg:
            raise ValueError('Resume configuration mismatch')
        validate(report, complete=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    boundary=boundary_from_utilities(evaluate_airport(boundary_coalitions(100),AIRPORT_COSTS),100)
    done={(c['repeat'],c['lambda0'],c['candidate_pool']) for c in report['cells']}
    # A fixed randomized order per repeat limits systematic timing order effects.
    for repeat in range(3):
        order=configurations()
        np.random.default_rng(np.random.SeedSequence([base_seed,repeat,913])).shuffle(order)
        seeds=_cell_seeds(base_seed,repeat,0)
        for lam,k in order:
            if (repeat,lam,k) in done:continue
            print(f'start {len(report["cells"])+1}/33 repeat={repeat} lambda0={lam:g} K={k}',flush=True)
            start=time.perf_counter()
            design=per_size_frame_coupled_design(100,inner_calls,seed=seeds['component'],
                candidate_pool=k,mean_balance=lam,second_moment_weight=1.0,
                mean_balance_mode='normalized',relabel_seed=seeds['component_relabel'],
                design_jobs=jobs,design_start_method='fork',compute_frame_diagnostics=False)
            elapsed=time.perf_counter()-start
            result=_evaluate_design('full_inside',design,boundary=boundary,truth=truth,
                                    inner_calls=inner_calls,design_seconds=elapsed)
            del design
            result.update(repeat=repeat,lambda0=lam,candidate_pool=k,seeds=seeds,
                          rmse=result['repeat_rmse'],D1=result['geometry']['first_moment_rms'],
                          D2=result['geometry']['frame_frobenius_rms'],completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            report['cells'].append(result)
            report['summary']=summarize(report['cells'])
            report['status']='complete' if len(report['cells'])==33 else 'running'
            validate(report,complete=False)
            temp=output.with_suffix('.json.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output)
            print(f'done RMSE={result["rmse"]:.6g} D1={result["D1"]:.6g} D2={result["D2"]:.6g} design={elapsed:.2f}s',flush=True)
    validate(report)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/json/airport_inside_sensitivity.json'))
    parser.add_argument('--inner-calls',type=int,default=50000)
    parser.add_argument('--design-jobs',type=int,default=16)
    args=parser.parse_args()
    run(args.output,args.inner_calls,args.design_jobs)

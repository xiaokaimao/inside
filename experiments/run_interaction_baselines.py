"""All paper RMSE methods on matched exact-reference interaction games."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
from time import perf_counter

import numpy as np
from frame_ofa import (iid_ofa_design, per_size_frame_coupled_design,
    cyclic_orbit_frame_design, estimate_ratio_ofa, estimate_official_ratio_ofa,
    estimate_basic_cc, estimate_sdiff, estimate_kernel_shap, estimate_tmc_shapley)
from frame_ofa.orthogonal import estimate_orthogonal_shapley
from frame_ofa.shapdoe import estimate_shapdoe
from frame_ofa.interaction_game import make_interaction_game
from experiments.run_large_interaction import evaluate_boundary_batched
from experiments.run_analytic_inside_comparison import _compact_json

METHODS = ('inside_greedy','inside_orbit','ofa','cc','s_diff','kernel_shap',
           'tmc_shapley','shapdoe_ls','orthogonal')


class GameAdapter:
    def __init__(self, game):
        self.game = game
    def evaluate(self, coalition):
        z = np.asarray(coalition, dtype=bool)
        if z.ndim == 2:
            return self.game.evaluate(z)
        g = self.game
        value = z @ g.additive
        value += (z[g.pairs[:,0]] & z[g.pairs[:,1]]) @ g.pair_weights
        value += (z[g.triples[:,0]] & z[g.triples[:,1]] & z[g.triples[:,2]]) @ g.triple_weights
        return float(value)


def _execute_payload(task):
    function, game, payload = task
    return function(game, payload)


class Evaluator:
    def __init__(self, game, workers=1):
        self.game = GameAdapter(game)
        self.workers = workers
    def evaluate(self, coalitions):
        return self.game.evaluate(coalitions)
    def run_game_tasks(self, function, payloads, *, chunksize=1):
        if self.workers == 1:
            return [function(self.game, payload) for payload in payloads]
        # Preserve payload seeds and ordered reduction exactly; only schedule changes.
        with ProcessPoolExecutor(self.workers, mp_context=mp.get_context('fork')) as pool:
            return list(pool.map(_execute_payload,
                ((function, self.game, payload) for payload in payloads), chunksize=chunksize))


def run_task(n, multiplier, repeat, method, base_seed, design_jobs, baseline_jobs=1, degrees=(2,3)):
    seed = int(np.random.SeedSequence([base_seed,n,multiplier,repeat,METHODS.index(method)]).generate_state(1)[0])
    inner = n * multiplier
    total = inner + 2*n + 2
    design = None
    start = perf_counter()
    if method == 'inside_greedy':
        design = per_size_frame_coupled_design(n,inner,seed=seed,candidate_pool=64,
            mean_balance=1/16,design_jobs=design_jobs,design_start_method='fork',
            compute_frame_diagnostics=False)
    elif method == 'inside_orbit':
        design = cyclic_orbit_frame_design(n, inner//n, seed=seed, candidate_pool=4, compute_frame_diagnostics=False)
    elif method == 'ofa':
        design = iid_ofa_design(n,inner,seed=seed,compute_diagnostics=False)
    design_seconds = perf_counter()-start
    cells=[]
    for degree in degrees:
        game = make_interaction_game(n,degree,20260923+n)
        truth = game.exact_shapley()
        start = perf_counter()
        diagnostics={}
        if design is not None:
            boundary = evaluate_boundary_batched(game)
            utilities = game.evaluate(design.coalitions)
            if method == 'inside_orbit':
                values = estimate_ratio_ofa(design,utilities,boundary)
            else:
                values = estimate_official_ratio_ofa(design,utilities,boundary,
                    missing='raise' if method=='inside_greedy' else 'zero')
            actual_calls = len(utilities)+2*n+2
        else:
            evaluator=Evaluator(game, workers=baseline_jobs if method=='s_diff' else 1)
            kwargs={'seed':seed,'num_tasks':32}
            if method=='cc':
                result=estimate_basic_cc(evaluator,n,total//2,**kwargs)
            elif method=='s_diff':
                result=estimate_sdiff(evaluator,n,total,**kwargs)
            elif method=='kernel_shap':
                result=estimate_kernel_shap(evaluator,n,total,ridge=1e-8,**kwargs)
            elif method=='tmc_shapley':
                result=estimate_tmc_shapley(evaluator,n,total,**kwargs)
            elif method=='shapdoe_ls':
                result=estimate_shapdoe(evaluator,n,total,method='ls',**kwargs)
            else:
                result=estimate_orthogonal_shapley(evaluator,n,total,**kwargs)
            values = result.values
            diagnostics=_compact_json(result.diagnostics)
            actual_calls=int(diagnostics['utility_evaluations'])
        values=np.asarray(values)
        if values.shape!=(n,) or not np.isfinite(values).all() or actual_calls>total:
            raise ValueError('invalid estimates or query budget exceeded')
        cells.append({'degree':degree,'method':method,'multiplier':multiplier,'repeat':repeat,
            'seed':seed,'target_calls':total,'actual_calls':actual_calls,
            'rmse':float(np.sqrt(np.mean((values-truth)**2))), 'estimate':values.tolist(),
            'design_seconds':design_seconds, 'total_seconds':design_seconds+perf_counter()-start,
            'diagnostics':diagnostics})
    return cells


def run(args):
    if args.players < 4 or args.repeats < 1 or min(args.budgets) < 1:
        raise ValueError('invalid players, repeats or budgets')
    if 'inside_orbit' in args.methods and min(args.budgets) < args.players - 3:
        raise ValueError('complete Orbit coverage needs at least n(n-3) inner queries')
    if 'shapdoe_ls' in args.methods and min(args.budgets)*args.players + 2*args.players+2 < args.players*(args.players-1)+2:
        raise ValueError('budget cannot fit one complete ShapDoE-LS block')
    degrees = args.degrees or [2,3]
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if v is not None}
    report={'configuration':cfg,'status':'running','games':[],'cells':[]}
    if args.output.exists():
        report=json.loads(args.output.read_text())
        if report['configuration']!=cfg:raise ValueError('resume configuration mismatch')
    elif args.reuse_source is not None:
        old=json.loads(args.reuse_source.read_text())
        if old['status']!='complete' or any(old['configuration'][k]!=cfg[k] for k in ('players','seed')):
            raise ValueError('reuse requires a complete report with matching game and estimator seeds')
        report['games']=[g for g in old['games'] if g['degree'] in degrees]
        report['cells']=[c for c in old['cells'] if c['method'] in args.methods
            and c['degree'] in degrees and c['multiplier'] in args.budgets and c['repeat'] < args.repeats]
        report['reuse']={'source':str(args.reuse_source),
            'sha256':hashlib.sha256(args.reuse_source.read_bytes()).hexdigest(),
            'cells':len(report['cells']),
            'note':'fixed-slice seed streams are invariant to design worker count; timings retain original scheduling'}
        for degree in degrees:
            truth=make_interaction_game(args.players,degree,20260923+args.players).exact_shapley()
            np.testing.assert_array_equal(truth,next(g['truth'] for g in report['games'] if g['degree']==degree))
            for cell in report['cells']:
                if cell['degree']==degree:
                    np.testing.assert_allclose(cell['rmse'],np.sqrt(np.mean((np.asarray(cell['estimate'])-truth)**2)))
    else:
        for degree in degrees:
            game=make_interaction_game(args.players,degree,20260923+args.players)
            path=args.output.parent.parent/'npz'/f'interaction_baselines_n{args.players}_degree{degree}.npz'
            path.parent.mkdir(parents=True,exist_ok=True);game.save(path)
            report['games'].append({'degree':degree,'path':str(path),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'truth':game.exact_shapley().tolist()})
    report['protocol'] = {
        'game_seed': 20260923 + args.players,
        'coarse_baseline_tasks': 32,
        'sdiff_task_workers': args.baseline_jobs or 1,
        'inside_coalition': {'candidate_pool': 64, 'normalized_lambda0': 1/16, 'missing': 'raise'},
        'inside_orbit': {'candidate_pool': 4, 'complete_orbits': True},
        'ofa_missing': 'zero (official behavior)',
        'kernel_shap_ridge': 1e-8,
        'tmc': {'relative_tolerance': 1e-3, 'patience': 5},
        'shapdoe': 'LS complete blocks only',
        'design_reuse': 'same design evaluated on requested degrees; full design time charged to each',
        'timing': 'concurrent throughput run, not a controlled wall-clock comparison',
        'uncertainty': 'mean and sample SD of per-seed RMSE; exact reference',
    }
    for entry in report['games']:
        if hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('game archive checksum mismatch')
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        temp=args.output.with_suffix('.tmp');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)
    save()
    done={(c['multiplier'],c['repeat'],c['method']) for c in report['cells']}
    tasks=[(args.players,b,r,m,args.seed,args.design_jobs,args.baseline_jobs or 1,degrees) for b in args.budgets for r in range(args.repeats) for m in args.methods if (b,r,m) not in done]
    with ProcessPoolExecutor(args.jobs,mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(run_task,*task):task for task in tasks}
        for future in as_completed(futures):
            cells=future.result();report['cells'].extend(cells);save()
            print(f'{len(report["cells"])}/{len(degrees)*len(args.budgets)*args.repeats*len(args.methods)}',futures[future][1:4],[(c['degree'],round(c['rmse'],7)) for c in cells],flush=True)
    expected={(d,b,r,m) for d in degrees for b in args.budgets for r in range(args.repeats) for m in args.methods}
    actual=[(c['degree'],c['multiplier'],c['repeat'],c['method']) for c in report['cells']]
    if set(actual)!=expected or len(actual)!=len(expected):
        raise ValueError('incomplete or duplicated final cell grid')
    report['status']='complete';save()
    return report


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--degrees',nargs='+',type=int,choices=[2,3])
    p.add_argument('--players',type=int,default=500)
    p.add_argument('--budgets',nargs='+',type=int,default=[500,1000,2000])
    p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--jobs',type=int,default=24)
    p.add_argument('--baseline-jobs',type=int)
    p.add_argument('--design-jobs',type=int,default=4)
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--reuse-source',type=Path)
    p.add_argument('--output',type=Path,default=Path('results/json/interaction_all_baselines_n500.json'))
    return p

if __name__=='__main__':run(parser().parse_args())

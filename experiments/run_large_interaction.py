"""Checkpointed exact-reference large-n interaction benchmark; no coalition enumeration."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from frame_ofa import (boundary_from_utilities, iid_ofa_design,
                      per_size_frame_coupled_design, estimate_coupled,
                      estimate_official_ratio_ofa)
from frame_ofa.interaction_game import make_interaction_game


def evaluate_boundary_batched(game):
    n = game.num_players
    utilities = []
    # Materialize at most 128 boundary coalitions, not a (2n+2)-by-n matrix.
    for start in range(0, 2 * n + 2, 128):
        ids = np.arange(start, min(start + 128, 2 * n + 2))
        rows = np.zeros((len(ids), n), dtype=bool)
        for k, i in enumerate(ids):
            if 1 <= i <= n:
                rows[k, i - 1] = True
            elif i >= n + 1:
                rows[k] = True
                if i > n + 1:
                    rows[k, i - n - 2] = False
        utilities.extend(game.evaluate(rows))
    return boundary_from_utilities(np.array(utilities), n)


def run(args):
    if args.jobs < 1 or args.max_memory_gib <= 0 or min(args.players) < 4:
        raise ValueError('require positive jobs/memory and at least four players')
    if args.repeats < 1 or min(args.budgets) < 1:
        raise ValueError('repeats and budget multipliers must be positive')
    cfg = vars(args) | {'output': str(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'configuration': cfg, 'status': 'running', 'games': [], 'cells': [], 'summary': []}
    if args.output.exists():
        report = json.loads(args.output.read_text())
        if report['configuration'] != cfg:
            raise ValueError('resume configuration mismatch; use a different output file')
    report['protocol'] = {
        'reference': 'exact sum of interaction coefficient divided by its degree',
        'inside_coalition': 'K=64, normalized lambda0=1/16, strict covered ratio',
        'ofa': 'IID, official ratio, missing conditional means initialized to zero',
        'ofa_linear': 'IID, coupled linear OFA identity',
        'utility_calls': 'interior queries + 2n+2 boundary calls, charged per cell',
        'timing': 'boundary evaluated once per game and charged to each cell; design includes process overhead',
        'uncertainty': 'sample SD (ddof=1) across estimator seeds on a fixed game',
        'normalized_rmse': 'RMSE divided by RMS of exact Shapley reference',
        'degree_pairing': 'same additive and pairwise coefficients for degrees 2 and 3',
        'environment': {'python': platform.python_version(), 'numpy': np.__version__},
    }
    def checkpoint():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(args.output)
    for n in args.players:
        for degree in args.degrees:
            seed = args.game_seed + n
            game = make_interaction_game(n, degree, seed)
            truth = game.exact_shapley()
            key = [n, degree]
            if not any(g['key'] == key for g in report['games']):
                path = args.output.parent.parent / 'npz' / f'{args.output.stem}_n{n}_degree{degree}.npz'
                path.parent.mkdir(parents=True, exist_ok=True)
                game.save(path)
                report['games'].append({'key': key, 'seed': seed, 'pairs': len(game.pairs),
                    'triples': len(game.triples), 'source': str(path),
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'reference': 'exact coefficient allocation, not Monte Carlo'})
                checkpoint()
            source = next(g for g in report['games'] if g['key'] == key)
            if hashlib.sha256(Path(source['source']).read_bytes()).hexdigest() != source['sha256']:
                raise ValueError('stored game archive hash mismatch')
            boundary = None
            for multiple in args.budgets:
                budget = multiple * n
                for repeat in range(args.repeats):
                    for method in args.methods:
                        cell_key = [n, degree, multiple, repeat, method]
                        if any(c['key'] == cell_key for c in report['cells']):
                            continue
                        print('start', cell_key, flush=True)
                        # Existing design APIs materialize a budget-by-n Boolean array.
                        # Reserve room for copies, aggregation, and worker operators.
                        estimated_bytes = 16 * budget * n + 8 * args.jobs * n * n
                        if estimated_bytes > args.max_memory_gib * 1024**3:
                            report['cells'].append({'key': cell_key, 'status': 'resource_limit',
                                'estimated_working_bytes': estimated_bytes,
                                'reason': 'dense design working-set estimate exceeds configured limit'})
                            checkpoint()
                            continue
                        start = perf_counter()
                        estimator_seed = int(np.random.SeedSequence([args.seed, n, multiple, repeat]).generate_state(1)[0])
                        if method == 'inside_coalition':
                            design = per_size_frame_coupled_design(n, budget, seed=estimator_seed,
                                candidate_pool=64, mean_balance=1/16, design_jobs=args.jobs,
                                compute_frame_diagnostics=False)
                        else:
                            design = iid_ofa_design(n, budget, seed=estimator_seed, compute_diagnostics=False)
                        design_seconds = perf_counter() - start
                        start = perf_counter()
                        if boundary is None:
                            boundary = evaluate_boundary_batched(game)
                            boundary_seconds = perf_counter() - start
                        start = perf_counter()
                        utilities = game.evaluate(design.coalitions)
                        utility_seconds = perf_counter() - start + boundary_seconds
                        start = perf_counter()
                        cell = {'key': cell_key, 'seed': estimator_seed, 'utility_calls': budget + 2*n + 2,
                                'design_seconds': design_seconds, 'utility_seconds': utility_seconds}
                        try:
                            if method == 'ofa_linear':
                                estimate = estimate_coupled(design, utilities, boundary)
                            else:
                                estimate = estimate_official_ratio_ofa(design, utilities, boundary,
                                    missing='raise' if method == 'inside_coalition' else 'zero')
                            rmse = float(np.sqrt(np.mean((estimate - truth)**2)))
                            cell.update(status='complete', rmse=rmse,
                                normalized_rmse=rmse / float(np.sqrt(np.mean(truth**2))),
                                estimate=estimate.tolist())
                        except ValueError as exc:
                            if 'missing in/out observations' not in str(exc):
                                raise
                            cell.update(status='insufficient_coverage', reason=str(exc))
                        cell['aggregation_seconds'] = perf_counter() - start
                        report['cells'].append(cell)
                        checkpoint()
                        print('done', cell.get('rmse', cell['status']), flush=True)
                        del design, utilities
    report['summary'] = []
    groups = sorted({tuple(c['key'][:3] + c['key'][4:]) for c in report['cells']})
    for group in groups:
        cells = [c for c in report['cells'] if tuple(c['key'][:3]+c['key'][4:]) == group and c['status']=='complete']
        row = {'key': group, 'completed_repeats': len(cells), 'requested_repeats': args.repeats}
        for metric in ('rmse', 'normalized_rmse', 'design_seconds', 'utility_seconds'):
            values = [c[metric] for c in cells]
            row[metric] = {'mean': float(np.mean(values)) if values else None,
                          'std': float(np.std(values, ddof=1)) if len(values)>1 else None}
        report['summary'].append(row)
    report['status'] = 'complete' if all(c['status']=='complete' for c in report['cells']) else 'complete_with_unavailable_cells'
    checkpoint()
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--players', nargs='+', type=int, default=[500, 1000, 5000])
    p.add_argument('--degrees', nargs='+', type=int, choices=[2,3], default=[2,3])
    p.add_argument('--budgets', nargs='+', type=int, default=[500,1000])
    p.add_argument('--methods', nargs='+', choices=['inside_coalition','ofa','ofa_linear'], default=['inside_coalition','ofa'])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--max-memory-gib', type=float, default=8.0)
    p.add_argument('--seed', type=int, default=20260923)
    p.add_argument('--game-seed', type=int, default=20260923)
    p.add_argument('--output', type=Path, default=Path('results/json/large_interaction.json'))
    return p

if __name__ == '__main__':
    run(parser().parse_args())

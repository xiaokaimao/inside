# Frame-OFA

This workspace contains a clean-room implementation of geometry-aware
coalition sampling for Shapley-specific OFA.

The official NeurIPS 2024 code is pinned unchanged in `ofa_upstream/` for
comparison.  It has no license file, package metadata, or tests, so the new
implementation lives separately in `frame_ofa/`.

## What is implemented

- `iid`: a linear, unbiased counterpart of Shapley-specific OFA.
- `coupled`: the same OFA marginal distribution for every coalition, with
  randomized systematic size sampling and greedy frame balancing.
- `orbit_coupled`: a scalable global-linear design using randomized
  systematic OFA sizes, complete cyclic orbits, FFT-greedy frame balancing,
  and uniform random relabeling.
- `stratified`: a minimum-one integer approximation to the OFA allocation,
  followed by independent
  frame balancing on each fixed-size Boolean slice.
- `orbit`: complete cyclic orbits.  These make every player/size in/out count
  deterministic and are therefore compatible with the official ratio
  aggregation.

All variants retain the official exact evaluation of sizes
`0, 1, n-1, n`, costing `2n+2` utility calls in addition to sampled inner
coalitions.

## Quick start

```python
import numpy as np
from frame_ofa import FrameOFAEstimator

weights = np.array([1.0, -0.5, 2.0, 0.3])

def utility(coalition):
    return float(weights @ coalition)

result = FrameOFAEstimator(
    num_players=4,
    num_samples=3,
    mode="coupled",
    seed=0,
    candidate_pool=16,
).estimate(utility)

print(result.values)
print(result.design.diagnostics)
print(result.utility_evaluations)
```

For `mode="orbit"`, `num_samples` denotes the number of complete orbits,
matching the official code's average query budget per player.  The number of
sampled inner coalitions is therefore `num_players * num_samples`.

The adapter accepts the upstream `game_func(**game_args).evaluate(coalition)`
interface:

```python
from frame_ofa import estimate_upstream_game

result = estimate_upstream_game(
    game_func=gameTraining,
    game_args=game_args,
    num_players=24,
    nue_avg=100,
    mode="coupled",
    seed=0,
)
```

Using the original game classes additionally requires their undeclared
research dependencies and import-path setup; the adapter itself is tested
against the interface contract.

Expensive utilities can be evaluated with a persistent process pool:

```python
from frame_ofa import FrameOFAEstimator, GameEvaluator

estimator = FrameOFAEstimator(
    num_players=24,
    num_samples=100,  # complete orbits in orbit mode
    mode="orbit",
)

with GameEvaluator(
    game_func,
    game_args,
    n_jobs=16,
    chunksize=32,
    start_method="spawn",
) as evaluator:
    result = estimator.estimate_with_evaluator(evaluator)
```

Use multiprocessing under an `if __name__ == "__main__"` guard.  Every worker
owns a private game and is restricted to one internal BLAS/Torch thread.

## Verification

Run the exact tests:

```bash
python -m unittest discover -s tests -v
```

Run the reproducible synthetic comparison:

```bash
python -m experiments.benchmark_synthetic \
  --players 8 --samples 40 --repeats 100
```

Run the Iris RBF-SVM data-valuation experiment:

```bash
conda run -n svmsv python -m experiments.iris_data_valuation \
  --game rbf_svm_accuracy \
  --gt-pairs 40000 \
  --budgets 1200 2400 4800 \
  --repeats 20 \
  --jobs 16
```

Run the 120-player, 128-process formal Iris experiment:

```bash
conda run -n svmsv python -m \
  experiments.iris_full_train_data_valuation \
  --budgets 60000 120000 240000 600000 1200000 \
  --coupled-design orbit_coupled \
  --jobs 128 \
  --output results/iris_full_train_rbf_svm_frame_ofa_60k_1p2m.json

conda run -n svmsv python -m \
  experiments.plot_iris_full_train_results \
  --input results/iris_full_train_rbf_svm_frame_ofa_60k_1p2m.json \
  --output results/iris_full_train_rbf_svm_rmse_60k_1p2m.png
```

The strict formula audit, theorem conditions, counterexamples, and algorithm
derivations are in [docs/THEORY_AUDIT.md](docs/THEORY_AUDIT.md).
The Iris protocol and results are in
[docs/IRIS_DATA_VALUATION.md](docs/IRIS_DATA_VALUATION.md).
The all-training-point SVM experiment and its call accounting are in
[docs/IRIS_FULL_TRAIN_SVM_EXPERIMENT.md](docs/IRIS_FULL_TRAIN_SVM_EXPERIMENT.md).

## Important limits

- Random relabeling proves marginal correctness and unbiasedness for the
  linear estimators; it does not restore independence.
- The IID concentration bound in the original OFA paper cannot be reused for
  a dependent batch without a new batch-level proof.
- With the endpoint-linear baseline (or exact first-moment balance), frame
  discrepancy exactly controls additive games.  It does not imply uniform
  improvement for arbitrary games; the audit includes an exact case where a
  coupled frame batch has 1.5 times the IID MSE.
- The endpoint-linear baseline is exact for additive games but need not reduce
  variance for every utility.
- `ofa_upstream/` is a local, ignored verification snapshot.  The official
  repository has no license file at the pinned commit, so do not redistribute
  it without permission.

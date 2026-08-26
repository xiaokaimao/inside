# INSIDE

**Intra-Size Coalition Design for Shapley Value Estimation**

This workspace contains a clean-room implementation of geometry-aware
coalition sampling for Shapley-specific OFA.

The official NeurIPS 2024 code is pinned unchanged in `ofa_upstream/` for
comparison.  It has no license file, package metadata, or tests, so the new
implementation lives separately in `frame_ofa/`.

## What is implemented

The formal public modes are `inside_greedy` and `inside_orbit`; their exact
algorithm bundles are fixed below.

The lower-level `iid`, `coupled`, `orbit_coupled`, `stratified`, and `orbit`
variants remain available for baseline comparisons, controlled ablations, and
reproduction of historical experiments.  In particular, legacy `coupled`
retains its global weighted-frame plus linear-estimator semantics; it is not
an alias for `inside_greedy`.

The package also includes:

- `official basic CC`: a deterministic, process-parallel adapter of Zhang et
  al.'s complementary-contribution Algorithm 2, including its official
  zero-fill behavior for missing player/size cells.
- auditable local ports of early-truncated TMC-Shapley,
  player/cardinality-stratified marginal MC, sampled-Gram KernelSHAP, linear
  GELS, Group Testing, Diff, and S-Diff.  These share one persistent
  `GameEvaluator` and expose exact physical-call diagnostics.

All variants retain the official exact evaluation of sizes
`0, 1, n-1, n`, costing `2n+2` utility calls in addition to sampled inner
coalitions.

## Canonical INSIDE algorithms

The paper-facing method names are intentionally limited to two algorithms:

- **INSIDE-Greedy** uses randomized-systematic OFA size allocation,
  per-size Greedy first/second-moment design, one uniform batch-wide player
  relabeling, and the official OFA conditional-mean ratio estimator.  Missing
  player/size in-or-out strata are rejected instead of zero-filled.  Its
  paper-facing configured defaults are candidate pool `K=64` and normalized
  first-moment weight `lambda0=1/16`.
- **INSIDE-Orbit** uses complete cyclic orbits within each allocated size and
  the strictly balanced OFA conditional-mean ratio estimator.  Its formal
  candidate-pool default is `K=4`, matching the benchmark protocol.

The global weighted `coupled` design and its linear estimator remain available
as implementation primitives and reproducible ablations, but they are not
called `INSIDE-Greedy` in final benchmark reports.

## Quick start

```python
import numpy as np
from frame_ofa import FrameOFAEstimator

weights = np.array([1.0, -0.5, 2.0, 0.3])

def utility(coalition):
    return float(weights @ coalition)

result = FrameOFAEstimator(
    num_players=4,
    num_samples=6,
    mode="inside_greedy",
    seed=0,
).estimate(utility)

print(result.values)
print(result.design.diagnostics)
print(result.utility_evaluations)
```

### Mean-balance normalization

For paper-facing `inside_greedy`, omitted hyperparameters resolve to
`candidate_pool=64`, `mean_balance=1/16`, and
`mean_balance_mode="normalized"`.  The direct
`per_size_frame_coupled_design` helper uses the same defaults.  Legacy/global
`coupled` mode keeps its historical `candidate_pool=32` and normalized
`mean_balance=1.0` defaults.  Explicit arguments always override these
mode-specific values.

`mean_balance` is the dimensionless coefficient \(\lambda_0\); the raw
coefficient used by a greedy objective is

\[
\lambda_{\mathrm{eff}}
=\lambda_0\left(1-\frac{1}{n-1}\right)
  \frac{1}{T}\sum_{t=1}^{T} w_t^2,
\]

where \(w_t^2=1\) for the formal per-size `inside_greedy` design and
\(w_t^2=s_t(n-s_t)\) for the legacy global radial `coupled` design.  Thus the
default scales with the realized size schedule instead of treating one raw
coefficient as comparable across player counts and budgets.  The resolved
value is recorded as `mean_balance_effective_raw` in the design diagnostics.

To reproduce a legacy run that used the old raw value directly, request raw
mode explicitly, for example:

```python
legacy = FrameOFAEstimator(
    num_players=24,
    num_samples=2400,
    mode="coupled",
    mean_balance=0.1,
    mean_balance_mode="raw",
)
```

The current paper-facing `inside_greedy` setting is \(\lambda_0=1/16\), so the
per-size Greedy score retains a small soft first-moment penalty; strict ratio
coverage is also enforced explicitly.  This value is an explicit experimental
configuration, not a theorem of universal optimality.  Historical
\(\lambda_0=0\) reports remain versioned separately.  `K=64` also has a
materially higher design-time cost than smaller candidate pools.

For `mode="inside_orbit"` and the legacy `mode="orbit"`, `num_samples`
denotes the number of complete orbits, matching the official code's average
query budget per player.  The number of sampled inner coalitions is therefore
`num_players * num_samples`.

The adapter accepts the upstream `game_func(**game_args).evaluate(coalition)`
interface:

```python
from frame_ofa import estimate_upstream_game

result = estimate_upstream_game(
    game_func=gameTraining,
    game_args=game_args,
    num_players=24,
    nue_avg=100,
    mode="inside_greedy",
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
    num_samples=100,  # complete orbits in inside_orbit mode
    mode="inside_orbit",
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

Run the corresponding 142-player Wine experiment. When `--budgets` is
omitted, the generic runner uses
`n * [500, 1000, 2000, 5000, 10000]`:

```bash
conda run -n svmsv python -m \
  experiments.full_train_data_valuation \
  --dataset wine \
  --gt-pairs 800000 \
  --jobs 128 \
  --coupled-design orbit_coupled \
  --output results/wine_full_train_rbf_svm_frame_ofa_71k_1p42m.json

conda run -n svmsv python -m \
  experiments.plot_full_train_results \
  --input results/wine_full_train_rbf_svm_frame_ofa_71k_1p42m.json

# Add official basic CC at the same five total-call coordinates, using
# 128 processes and the existing high-budget reference.
conda run -n svmsv python -m experiments.add_wine_cc_results \
  --input results/wine_full_train_rbf_svm_frame_ofa_71k_1p42m.json \
  --output \
    results/wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json \
  --jobs 128 \
  --cc-tasks 128

conda run -n svmsv python -m experiments.plot_full_train_results \
  --input \
    results/wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json \
  --output \
    results/wine_full_train_rbf_svm_rmse_with_cc_71k_1p42m.png

# Replace the historical mislabelled linear GELS curve non-destructively;
# all other completed method results are retained verbatim.
conda run -n svmsv python -m \
  experiments.replace_wine_linear_gels_with_official

# Add all compatible baselines from the read-only audit.  With the migrated
# checkpoint above, only official GELS-Shapley remains pending.
# Existing OFA/Frame-OFA/CC estimates are reduced to their first 3 stored
# repeats; each new method is run 3 times with 128 persistent workers.
conda run -n svmsv python -m experiments.add_wine_local_baselines \
  --input \
    results/wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json \
  --output \
    results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_71k_1p42m.json \
  --repeats 3 \
  --jobs 128 \
  --num-tasks 128

conda run -n svmsv python -m experiments.validate_full_train_report \
  results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_71k_1p42m.json \
  --expected-dataset wine \
  --expected-num-players 142 \
  --expected-repeats 3 \
  --expected-gt-pairs 800000 \
  --expected-jobs 128

conda run -n svmsv python -m \
  experiments.plot_wine_baseline_comparison \
  --input \
    results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_71k_1p42m.json

# Add the current paper-facing INSIDE-Greedy (K=64, lambda0=1/16, ratio)
# without altering the immutable baseline report. INSIDE-Orbit and the six
# external/OFA baselines are reused verbatim at the same call coordinates.
conda run -n svmsv python -m experiments.run_wine_inside_comparison \
  --input \
    results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_71k_1p42m.json \
  --output \
    results/wine_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_71k_1p42m.json \
  --repeat-processes 2 \
  --jobs-per-repeat 64

conda run -n svmsv python -m experiments.validate_inside_comparison \
  results/wine_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_71k_1p42m.json

conda run -n svmsv python -m experiments.plot_inside_comparison \
  results/wine_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_71k_1p42m.json
```

Run the corresponding 455-player Breast Cancer experiment with the same
budget multipliers and 128 processes:

```bash
conda run -n svmsv python -m \
  experiments.full_train_data_valuation \
  --dataset cancer \
  --gt-pairs 800000 \
  --jobs 128 \
  --coupled-design orbit_coupled \
  --output results/cancer_full_train_rbf_svm_frame_ofa_228k_4p55m.json

conda run -n svmsv python -m \
  experiments.plot_full_train_results \
  --input results/cancer_full_train_rbf_svm_frame_ofa_228k_4p55m.json
```

Run the balanced Digits experiment. It samples 100 observations from each of
the 10 classes before the 80/20 split, giving 800 training-point players and
200 fixed test observations. The first multiplier is 800 because the current
complete-orbit ratio estimator needs at least `n * (n - 3)` inner calls when
`n=800`:

```bash
conda run -n svmsv python -m \
  experiments.full_train_data_valuation \
  --dataset digits \
  --samples-per-class 100 \
  --gt-pairs 800000 \
  --budget-multipliers 800 1000 2000 5000 10000 \
  --jobs 128 \
  --coupled-design orbit_coupled \
  --output \
    results/digits_100_per_class_full_train_rbf_svm_frame_ofa_640k_8m.json

conda run -n svmsv python -m \
  experiments.plot_full_train_results \
  --input \
    results/digits_100_per_class_full_train_rbf_svm_frame_ofa_640k_8m.json
```

The strict formula audit, theorem conditions, counterexamples, and algorithm
derivations are in [docs/THEORY_AUDIT.md](docs/THEORY_AUDIT.md).
The Iris protocol and results are in
[docs/IRIS_DATA_VALUATION.md](docs/IRIS_DATA_VALUATION.md).
The all-training-point SVM experiment and its call accounting are in
[docs/IRIS_FULL_TRAIN_SVM_EXPERIMENT.md](docs/IRIS_FULL_TRAIN_SVM_EXPERIMENT.md).
The corresponding Wine protocol, high-budget ground truth, and results are in
[docs/WINE_FULL_TRAIN_SVM_EXPERIMENT.md](docs/WINE_FULL_TRAIN_SVM_EXPERIMENT.md).
The official basic-CC formula, pinned source, zero-count theorem caveat, and
same-call Wine comparison are in
[docs/BASIC_CC_BASELINE_AUDIT.md](docs/BASIC_CC_BASELINE_AUDIT.md).
The formula, target, call-accounting, exclusion, and external-source integrity
audit for the additional local baseline ports is in
[docs/INTEGRAL_BASELINES_AUDIT.md](docs/INTEGRAL_BASELINES_AUDIT.md).
The corresponding Breast Cancer protocol and results are in
[docs/CANCER_FULL_TRAIN_SVM_EXPERIMENT.md](docs/CANCER_FULL_TRAIN_SVM_EXPERIMENT.md).
The balanced Digits protocol and its complete-orbit budget constraint are in
[docs/DIGITS_100_PER_CLASS_SVM_EXPERIMENT.md](docs/DIGITS_100_PER_CLASS_SVM_EXPERIMENT.md).

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

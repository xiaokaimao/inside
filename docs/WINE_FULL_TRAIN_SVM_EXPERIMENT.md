# Wine full-training-set SVM data valuation

## Protocol

- Dataset: `sklearn.datasets.load_wine`, 178 observations and 13 features.
- Split: stratified 80/20 train/test split with seed 2024.
- Players: all 142 training observations; class counts `[47, 57, 38]`.
- Fixed test set: 36 observations; class counts `[12, 14, 10]`.
- Preprocessing: `StandardScaler` fitted on the training partition only and
  then applied to both partitions.
- Utility: fixed-test classification accuracy after fitting
  `sklearn.svm.SVC(C=1, kernel="rbf", gamma="scale")` to a coalition.
- Empty coalition: best constant-label test accuracy, (14/36).
- Single-class coalition: predict the coalition's sole observed class.
- Full-coalition utility: (35/36).

`gamma="scale"` is an estimator rule, so SVC recomputes its numeric gamma
from each coalition's training features. No model choice uses the test labels.

## High-budget Monte Carlo reference

Ground truth uses 800,000 independent antithetic units. Each unit averages a
uniform random permutation with its reverse. The conceptual permutation-path
budget is

\[
2+2(800{,}000)(141)=225{,}600{,}002
\]

utility calls, 158.8 times the largest method budget. Reusing the already
evaluated singleton and leave-one-out utilities is exact for this deterministic
game, reducing the physical computation to

\[
2(800{,}000)(139)+286=222{,}400{,}286
\]

calls. With 128 processes and one numerical-library thread per worker, the
streaming computation took 3,379.8 seconds (56.3 minutes).

Reference diagnostics:

- SE-RMSE across 142 coordinates: (3.327\times10^{-5}).
- Independent half-split RMSE: (3.236\times10^{-5}).
- Maximum Bonferroni simultaneous 95% half-width:
  (1.343\times10^{-4}).
- Efficiency:
  \(\sum_i\widehat\phi_i=7/12=v(N)-v(\varnothing)\), with numerical gap 0.

## Method budgets

The five inner budgets are

\[
142[500,1000,2000,5000,10000]
=
[71{,}000,142{,}000,284{,}000,710{,}000,1{,}420{,}000].
\]

The four OFA/Frame-OFA estimators also use the same (2n+2=286) exact boundary
calls, so the plotted total calls per estimate are

\[
[71{,}286,142{,}286,284{,}286,710{,}286,1{,}420{,}286].
\]

Each point has 10 independent method seeds. The two IID estimators reuse one
physical utility batch. Four logical estimators therefore require three
physical batches per repeat.

Official basic CC does not consume the OFA boundary batch. For an exact
same-total-call comparison it spends each complete plotted budget on
two-call complement pairs, giving `[35,643, 71,143, 142,143, 355,143,
710,143]` pairs. It therefore has no unused or silently added calls.

The methods are:

1. **OFA ratio (IID):** the official conditional-mean ratio aggregation on
   IID OFA samples.
2. **OFA linear (IID):** the global unbiased linear estimator on the same IID
   samples and utilities.
3. **Frame-OFA orbit-coupled linear:** randomized-systematic (q^\ast) orbit
   sizes, complete cyclic orbits, FFT-greedy frame balancing, uniform random
   relabeling, and global linear aggregation.
4. **Frame-OFA orbit ratio:** per-size ratio aggregation on complete cyclic
   orbits with frame-balanced bases.
5. **CC (official basic):** Algorithm 2 of Zhang et al. with IID uniform
   coalition size, a uniform coalition conditional on size, complement
   pairing, and the official zero-fill rule for an unobserved player/size
   cell. It has no warm-up or efficiency projection.

## Results

Lower RMSE is better. Values below are raw RMSE against the fixed high-budget
Monte Carlo reference.

| Total calls | OFA ratio IID | Frame orbit ratio | Ratio change | OFA linear IID | Frame orbit-coupled linear | Linear change |
|---:|---:|---:|---:|---:|---:|---:|
| 71,286 | 0.00063257 | 0.00061958 | 2.1% lower | 0.00299542 | 0.00062327 | 79.2% lower |
| 142,286 | 0.00047454 | 0.00042024 | 11.4% lower | 0.00213868 | 0.00045694 | 78.6% lower |
| 284,286 | 0.00032860 | 0.00030112 | 8.4% lower | 0.00151747 | 0.00031350 | 79.3% lower |
| 710,286 | 0.00020063 | 0.00018090 | 9.8% lower | 0.00095077 | 0.00020377 | 78.6% lower |
| 1,420,286 | 0.00015185 | 0.00013128 | 13.5% lower | 0.00069785 | 0.00014598 | 79.1% lower |

The added official-basic CC results are:

| Total calls | Basic CC RMSE | 95% bootstrap interval | vs. OFA ratio IID | Missing cells (mean) |
|---:|---:|---:|---:|---:|
| 71,286 | 0.00122259 | [0.00111626, 0.00133738] | 93.3% higher | 4.6 / 20,164 |
| 142,286 | 0.00062439 | [0.00060924, 0.00063839] | 31.6% higher | 0 |
| 284,286 | 0.00042759 | [0.00041176, 0.00044224] | 30.1% higher | 0 |
| 710,286 | 0.00027627 | [0.00026644, 0.00028554] | 37.7% higher | 0 |
| 1,420,286 | 0.00020083 | [0.00019542, 0.00020598] | 32.3% higher | 0 |

Basic CC is worse than OFA ratio at all five budgets on this fixed Wine game.
At the first point, official zero filling contributes a finite-budget bias:
the ten repeats contain `[9, 2, 5, 3, 4, 2, 3, 4, 9, 5]` missing cells, and
the mean efficiency gap is 0.01883. From the second point onward every cell
is observed, yet CC remains 30--38% worse than OFA ratio; that remaining gap
is an estimator-variance effect rather than a coverage artifact. No
efficiency correction was added because it would define a different method.

At the largest budget, the paired repeat-level difference

\[
\operatorname{RMSE}(\text{OFA ratio IID})
-\operatorname{RMSE}(\text{orbit ratio})
\]

is (2.076\times10^{-5}), with bootstrap 95% interval
([1.377\times10^{-5},2.899\times10^{-5}]). Orbit-ratio is better in all 10
paired repeats. At the smallest budget its paired interval includes zero, so
the observed 2.1% aggregate improvement is not conclusive with 10 repeats.

For IID-linear minus orbit-coupled-linear at the largest budget, the paired
difference is (5.496\times10^{-4}), with interval
([5.125\times10^{-4},5.874\times10^{-4}]); all 10 paired repeats favor the
coupled design.

The fitted log-log slopes over the five budgets are:

| Method | slope |
|---|---:|
| OFA ratio IID | -0.490 |
| Frame-OFA orbit ratio | -0.519 |
| OFA linear IID | -0.490 |
| Frame-OFA orbit-coupled linear | -0.488 |
| CC official basic | -0.580 |

All five curves decrease monotonically and are broadly consistent with a
(T^{-1/2}) Monte Carlo rate. For the original four methods, at 1,420,286
calls, noise-corrected RMSE is (1.482\times10^{-4}),
(1.270\times10^{-4}), (6.971\times10^{-4}), and
(1.421\times10^{-4}), respectively, so reference uncertainty does not
change their ranking. Basic CC's raw RMSE is about six times the reference
SE-RMSE at that point, so its observed gap is also larger than the reference
noise scale.

The very large linear-panel gain should be interpreted within estimator
families: global IID-linear is a high-variance Horvitz--Thompson-style
estimator on this game. It does not mean every IID OFA estimator is poor; the
practical ratio-to-ratio comparison is the left panel.

Bootstrap bands resample only the 10 method seeds and condition on the fixed
Monte Carlo reference. They do not include reference uncertainty and are
descriptive rather than a formal population-level claim. The experiment
estimates the fixed split's cooperative game; the 36-point test set is not a
population ground truth.

## Runtime and artifacts

At 1,420,000 inner calls, one utility batch took about 20.5--20.9 seconds.
Mean design time was 4.6 seconds for IID sampling and diagnostics, 2.5 seconds
for orbit-coupled linear, and 14.5 seconds for the per-size orbit-ratio
design. Across all budgets and repeats, the method experiment made 78.81
million physical inner utility calls.

The separate CC augmentation made exactly 26,284,300 additional physical
utility calls. With 128 processes and one numerical-library thread per
worker, a CC repeat took about 1.47, 2.29, 4.42, 10.82, and 21.44 seconds at
the five budgets. Every repeat used 128 coarse tasks.

- Raw method report:
  `results/wine_full_train_rbf_svm_frame_ofa_71k_1p42m.json`
- Raw report augmented with official basic CC:
  `results/wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json`
- Ground-truth report:
  `results/wine_full_train_rbf_svm_ground_truth_report.json`
- Ground-truth cache: `results/wine_full_train_rbf_svm_gt.npz`
- Figure: `results/wine_full_train_rbf_svm_rmse_71k_1p42m.png`
- Vector figure: `results/wine_full_train_rbf_svm_rmse_71k_1p42m.pdf`
- Figure including CC:
  `results/wine_full_train_rbf_svm_rmse_with_cc_71k_1p42m.png`
- Vector figure including CC:
  `results/wine_full_train_rbf_svm_rmse_with_cc_71k_1p42m.pdf`
- Basic CC formula, source, and finite-budget theorem audit:
  `docs/BASIC_CC_BASELINE_AUDIT.md`

## Twelve-method, three-repeat baseline comparison

The baseline audit added seven compatible methods: official GELS-Shapley
Algorithm 3, sampled-Gram KernelSHAP, Group Testing, Diff, S-Diff,
early-truncated TMC-Shapley, and player/cardinality-stratified marginal Monte
Carlo.  The earlier full-permutation MC curve and the mislabelled linear GELS
curve were removed rather than silently renamed.  To satisfy the requested
three-repeat protocol without discarding the existing expensive work, the
comparison reuses stored repeat indices
`[0, 1, 2]` for the five OFA/Frame-OFA/CC methods and recomputes their summaries
from the raw 3-by-142 estimates.  Every added baseline is run exactly three
times with new seeds.

The same five total physical-call caps and the same fixed 800,000-pair
reference are used throughout.  Lower is better:

| Method | 71,286 | 142,286 | 284,286 | 710,286 | 1,420,286 | log-log slope |
|---|---:|---:|---:|---:|---:|---:|
| Frame-OFA orbit ratio | 0.00063536 | **0.00041239** | **0.00027439** | **0.00017557** | **0.00013245** | -0.524 |
| S-Diff | 0.00064575 | 0.00045603 | 0.00029500 | 0.00019322 | 0.00014514 | -0.506 |
| Frame-OFA coupled linear | **0.00060522** | 0.00046715 | 0.00030051 | 0.00020521 | 0.00014794 | -0.479 |
| OFA ratio IID | 0.00061166 | 0.00046294 | 0.00030037 | 0.00019893 | 0.00015060 | -0.480 |
| CC official basic | 0.00126820 | 0.00061454 | 0.00043739 | 0.00027265 | 0.00020244 | -0.587 |
| Diff | 0.00144673 | 0.00109063 | 0.00071159 | 0.00046857 | 0.00033650 | -0.495 |
| GELS-Shapley | 0.00151249 | 0.00107175 | 0.00077738 | 0.00048341 | 0.00033817 | -0.500 |
| TMC-Shapley | 0.00239822 | 0.00190633 | 0.00157939 | 0.00145617 | 0.00142196 | -0.172 |
| Stratified marginal MC | 0.00266467 | 0.00170002 | 0.00125798 | 0.00076249 | 0.00054810 | -0.522 |
| OFA linear IID | 0.00287114 | 0.00215726 | 0.00150624 | 0.00091943 | 0.00070066 | -0.485 |
| KernelSHAP sampled Gram | 0.00366994 | 0.00260671 | 0.00195680 | 0.00124732 | 0.00093126 | -0.459 |
| Group Testing | 0.00730337 | 0.00480827 | 0.00368329 | 0.00227813 | 0.00167380 | -0.487 |

Frame-OFA orbit ratio is best at four of five caps; coupled-linear is best at
the smallest cap.  S-Diff is the strongest newly ported external baseline: it
ranks fourth at 71,286 calls and second at every larger cap.  Relative to
unstratified Diff, S-Diff reduces RMSE by 55.4--58.8% across the five caps.
Relative to OFA ratio IID it is 5.6% worse at the smallest cap, then 1.5%,
1.8%, 2.9%, and 3.6% better.  It nevertheless remains 1.6--9.6% worse than
Frame-OFA orbit ratio at every cap in this three-repeat run.

These top-method gaps should not be overinterpreted.  The bootstrap intervals
use only three repeats and are descriptive.  At the largest cap, for example,
the intervals are `[0.00012410, 0.00013795]` for Frame-OFA orbit ratio and
`[0.00014317, 0.00014746]` for S-Diff, while the fixed reference itself has
SE-RMSE `0.00003327`.  The curves estimate one fixed train/test game, not
performance over new dataset splits.

The GELS correction is material.  The deleted implementation used
\((2H_{n-1}/m)\sum_t\mathbf1\{i\in S_t\}v(S_t)\), which is the authors'
linear `simSHAP` estimator rather than GELS-Shapley.  Official Algorithm 3
uses the per-player ratio
\(H_{n-1}\sum_t\mathbf1\{i\in S_t\}v(S_t)/
\sum_t\mathbf1\{i\in S_t\}\) before the efficiency offset.  On this accuracy
game, that self-normalization removes utility-baseline noise from random
inclusion counts.  Its RMSE is 86.7--87.5% lower than the deleted curve across
the five budgets.  All 15 runs had zero missing player counts and satisfied
\(\sum_i C_i=\sum_s sN_s\).

All twelve curves decrease monotonically.  The non-TMC slopes remain broadly
near the ordinary Monte Carlo rate \(T^{-1/2}\).  TMC is the clear exception:
its actual-call slope is only -0.172 and its RMSE flattens near
\(1.4\times10^{-3}\), which is consistent with truncation bias becoming more
important than sampling variance.

TMC fixes the permutation count before sampling at
\(m=\lfloor(B-2)/142\rfloor\), giving `[502, 1002, 2002, 5002, 10002]` paths.
Its mean actual calls are `[26,131.7, 52,572.0, 104,678.7, 260,412.7,
513,978.0]`, and those are its plotted horizontal coordinates rather than the
larger caps in the table header.  About 97.4--98.1% of paths truncate, with a
mean visited prefix length of 51.4--52.5 out of 142.  The utility is test
accuracy on 36 rows, so the default tolerance \(10^{-3}|v(N)|\) is smaller
than one accuracy increment: in this experiment “near full” effectively means
exactly the same accuracy as the full set.  Consequently the observed
efficiency gaps are at floating-point zero, but this does not remove possible
coordinate-wise truncation bias.

The seven added methods made 50,185,059 physical utility calls.  Six methods
use every cap exactly.  TMC intentionally leaves its truncation savings unused
because adaptively adding paths until the budget is exhausted would introduce
a path-length stopping-time bias.  Across all repeats, recorded method time
sums to 803.5 seconds with 128 persistent workers and one numerical-library
thread per worker.

The augmented report passes the independent raw-estimate and call-accounting
validator.  The main comparison and linear-estimator ablation are exported as
two separate near-square figures with logarithmic axes.

- Twelve-method raw report:
  `results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_71k_1p42m.json`
- Twelve-method PNG:
  `results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_rmse_71k_1p42m.png`
- Twelve-method vector PDF:
  `results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_rmse_71k_1p42m.pdf`
- Linear-estimator PNG:
  `results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_linear_rmse_71k_1p42m.png`
- Linear-estimator vector PDF:
  `results/wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_3repeats_linear_rmse_71k_1p42m.pdf`
- External-baseline formula, target, and immutability audit:
  `docs/INTEGRAL_BASELINES_AUDIT.md`

## Current INSIDE-Greedy protocol

The twelve-method report above is retained as an immutable historical source;
its row-wise design used the earlier `K=4` configuration.  The current
paper-facing runner writes a separate, versioned report and evaluates only
INSIDE-Greedy with the explicitly configured protocol:

- per-size second-moment Greedy design;
- candidate pool `K=64`;
- normalized first-moment coefficient `lambda0=1/16`;
- official OFA conditional-mean ratio aggregation with missing strata treated
  as an error.

INSIDE-Orbit and the six retained OFA/external baselines are copied verbatim
from the source report, so this augmentation neither changes their estimates
nor reruns their utility calls.  The output path is
`results/wine_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_71k_1p42m.json`.
The unified INSIDE validator chooses the required `K` and `lambda0` from the
versioned experiment ID, so both the historical K4 report and the new K64
report remain auditable.

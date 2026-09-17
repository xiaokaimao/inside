# Breast-cancer full-training-set SVM data valuation

## Protocol

- Dataset: `sklearn.datasets.load_breast_cancer`, 569 observations and 30
  features.
- Split: stratified 80/20 train/test split with seed 2024.
- Players: all 455 training observations; class counts `[170, 285]`.
- Fixed test set: 114 observations; class counts `[42, 72]`.
- Preprocessing: `StandardScaler` fitted on the training partition only and
  then applied to both partitions.
- Utility: fixed-test classification accuracy after fitting
  `sklearn.svm.SVC(C=1, kernel="rbf", gamma="scale")` to a coalition.
- Empty coalition: best constant-label test accuracy, \(72/114=12/19\).
- Single-class coalition: predict the coalition's sole observed class.
- Full-coalition utility: \(110/114=55/57\).

`gamma="scale"` is an estimator rule, so SVC recomputes its numeric gamma
from each coalition's training features. No model choice uses the test
labels.

## Current INSIDE comparison (August 2026)

The current paper setting fixes **INSIDE-Greedy** to per-size first- and
second-moment design with \(K=64\), normalized
\(\lambda_0=1/16\), and the OFA conditional-mean ratio estimator.
**INSIDE-Orbit** uses complete cyclic orbits, \(K=4\), and the same OFA ratio
estimator. The common figure also includes OFA linear (IID), OFA ratio (IID),
CC, S-Diff, KernelSHAP, and TMC-Shapley.

All methods use five target total-call budgets

\[
[228{,}412,455{,}912,910{,}912,2{,}275{,}912,4{,}550{,}912]
\]

and three independent repeats. TMC-Shapley is plotted at its observed calls
after truncation; the other seven methods spend the full cap. Lower RMSE is
better.

| Method | 228,412 | 455,912 | 910,912 | 2,275,912 | 4,550,912 |
|---|---:|---:|---:|---:|---:|
| INSIDE-Greedy | 2.3445e-4 | 1.6691e-4 | 1.1539e-4 | 7.0135e-5 | 4.7809e-5 |
| INSIDE-Orbit | 2.6179e-4 | 1.7044e-4 | 1.1943e-4 | 7.1147e-5 | 5.0920e-5 |
| OFA linear (IID) | 9.9116e-4 | 6.9169e-4 | 5.0034e-4 | 3.1496e-4 | 2.3758e-4 |
| OFA ratio (IID) | 2.5044e-4 | 1.7126e-4 | 1.2170e-4 | 7.8673e-5 | 5.8045e-5 |
| CC | 8.3459e-4 | 4.8278e-4 | 2.5940e-4 | 1.4207e-4 | 9.9808e-5 |
| S-Diff | 2.0111e-4 | 1.4303e-4 | 9.9235e-5 | 6.4740e-5 | 4.6974e-5 |
| KernelSHAP | 1.2988e-3 | 9.3924e-4 | 6.4260e-4 | 4.0095e-4 | 2.9534e-4 |
| TMC-Shapley | 1.0568e-3 | 7.8569e-4 | 5.8501e-4 | 4.4220e-4 | 3.7041e-4 |

INSIDE-Greedy is lower than INSIDE-Orbit at every budget (1.4%--10.4%)
and lower than OFA ratio (IID) at every budget (2.5%--17.6%). S-Diff is the
lowest curve at all five points, although its final advantage over
INSIDE-Greedy is only 1.8%. The reference SE-RMSE is
\(1.8193\times10^{-5}\), 38.1% of INSIDE-Greedy's final raw RMSE,
so small final differences should not be interpreted as an exact ordering.
Three repeats also give only coarse method uncertainty.

The independently recomputing validator passed all protocol, call-accounting,
ground-truth, method-identity, and stored-metric checks. The complete run used
191,156,763 actual method utility calls and took 6,909.7 wall-clock seconds
(1.92 hours), reusing the immutable 800,000-pair reference and the first three
compatible repeats of the existing Orbit/OFA curves.

Current artifacts:

- Report:
  `results/json/cancer_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_228k_4p55m.json`
- Validation audit:
  `results/json/cancer_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_228k_4p55m_validation.json`
- Figure:
  `results/cancer_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_228k_4p55m_rmse.png`
- Vector figure:
  `results/pdf/cancer_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_228k_4p55m_rmse.pdf`

In the combined four-panel presentation, OFA linear is omitted from every
panel and S-Diff is omitted from the Cancer panel. S-Diff maintains a
pairwise state with \(O(n^2)\) space complexity, so its memory scaling is not
appropriate as a large-\(n\) comparison. This is a presentation choice: the
completed Cancer S-Diff measurements remain available in the raw report.

The remainder of this document records the earlier ten-repeat four-method
experiment that supplied the immutable reference and reusable Orbit/OFA
curves.

## High-budget Monte Carlo reference

Ground truth uses 800,000 independent antithetic units. Each unit averages a
uniform random permutation with its reverse. The conceptual permutation-path
budget is

\[
2+2(800{,}000)(454)=726{,}400{,}002
\]

utility calls, 159.6 times the largest plotted method budget. Reusing the
already evaluated singleton and leave-one-out utilities is exact for this
deterministic game, reducing the physical computation to

\[
2(800{,}000)(452)+912=723{,}200{,}912
\]

calls.
With 128 processes and one numerical-library thread per worker, the streaming
computation took 20,101.2 seconds (5 hours 35 minutes).

Reference diagnostics:

- SE-RMSE across 455 coordinates: \(1.8193\times10^{-5}\).
- Independent half-split RMSE: \(1.8992\times10^{-5}\).
- Maximum Bonferroni simultaneous 95% half-width:
  \(8.7239\times10^{-5}\).
- Efficiency:
  \(\sum_i\widehat\phi_i=0.3333333333333334\), versus
  \(1/3=v(N)-v(\varnothing)\), for a numerical gap of
  at most \(1.11\times10^{-16}\), depending on summation order.

## Method budgets

The five inner budgets are

\[
455[500,1000,2000,5000,10000]
=
[227{,}500,455{,}000,910{,}000,2{,}275{,}000,4{,}550{,}000].
\]

Every estimator also uses the same \(2n+2=912\) exact boundary calls, so the
plotted total calls per estimate are

\[
[228{,}412,455{,}912,910{,}912,2{,}275{,}912,4{,}550{,}912].
\]

Each point has 10 independent method seeds. The two IID estimators reuse one
physical utility batch. Four logical estimators therefore require three
physical batches per repeat.

The methods are:

1. **OFA ratio (IID):** the official conditional-mean ratio aggregation on
   IID OFA samples.
2. **OFA linear (IID):** the global unbiased linear estimator on the same IID
   samples and utilities.
3. **Frame-OFA orbit-coupled linear:** randomized-systematic \(q^\ast\) orbit
   sizes, complete cyclic orbits, FFT-greedy frame balancing, uniform random
   relabeling, and global linear aggregation.
4. **Frame-OFA orbit ratio:** per-size ratio aggregation on complete cyclic
   orbits with frame-balanced bases.

The minimum valid orbit-ratio inner budget is
\(n(n-3)=455\times452=205{,}660\), so all five requested budgets are valid.
At the first budget, however, most coalition-size strata receive only one
complete orbit (422 of 452 inner sizes under the implemented minimum-one
allocation); any low-budget gain or loss must therefore be interpreted from
the measured paired repeats, not inferred from frame geometry alone.

## Results

Lower RMSE is better. Values below are raw pooled RMSE against the fixed
high-budget Monte Carlo reference.

| Total calls | OFA ratio IID | Frame orbit ratio | Ratio change | OFA linear IID | Frame orbit-coupled linear | Linear change |
|---:|---:|---:|---:|---:|---:|---:|
| 228,412 | 0.00025257 | 0.00026607 | 5.3% higher | 0.00100147 | 0.00024207 | 75.8% lower |
| 455,912 | 0.00017259 | 0.00017168 | 0.5% lower | 0.00071053 | 0.00017367 | 75.6% lower |
| 910,912 | 0.00012557 | 0.00011803 | 6.0% lower | 0.00051353 | 0.00012386 | 75.9% lower |
| 2,275,912 | 0.00007950 | 0.00007268 | 8.6% lower | 0.00032354 | 0.00007874 | 75.7% lower |
| 4,550,912 | 0.00005824 | 0.00005098 | 12.5% lower | 0.00023531 | 0.00005811 | 75.3% lower |

At the largest budget, the paired repeat-level difference

\[
\operatorname{RMSE}(\text{OFA ratio IID})
-\operatorname{RMSE}(\text{Frame orbit ratio})
\]

is \(7.244\times10^{-6}\), with bootstrap 95% interval
\([6.018\times10^{-6},8.399\times10^{-6}]\). Frame orbit-ratio is better in
all 10 paired repeats. At the smallest budget the mean difference is instead
\(-1.335\times10^{-5}\), with interval
\([-2.110\times10^{-5},-4.679\times10^{-6}]\), and Frame wins only 1 of 10
repeats. Thus the low-budget loss is reproducible here, not just a pooled-RMSE
artifact. With only one complete orbit in 422 of 452 internal size slices,
frame balance does not guarantee favorable covariance for the non-additive
SVM utility residual. The method has no universal finite-budget dominance
theorem for arbitrary games.

For IID-linear minus orbit-coupled-linear at the largest budget, the paired
difference is \(1.771\times10^{-4}\), with interval
\([1.723\times10^{-4},1.815\times10^{-4}]\); all 10 paired repeats favor the
coupled design. This large gain is an estimator-family result: the global
IID-linear estimator is a high-variance Horvitz--Thompson-style estimator on
this game. The practical OFA comparison is the ratio-to-ratio panel.

The fitted log-log slopes over the five budgets are:

| Method | slope |
|---|---:|
| OFA ratio IID | -0.488 |
| Frame-OFA orbit ratio | -0.548 |
| OFA linear IID | -0.485 |
| Frame-OFA orbit-coupled linear | -0.480 |

All four curves decrease monotonically and are close to a
\(T^{-1/2}\) Monte Carlo rate. At 4,550,912 calls, subtracting the estimated
reference MSE gives noise-corrected RMSE values of
\(5.533\times10^{-5}\), \(4.762\times10^{-5}\),
\(2.346\times10^{-4}\), and \(5.519\times10^{-5}\), respectively, in the
same method order as the table.

As a direct reference-sensitivity check, the largest-budget aggregate RMSEs
against the two independent 400,000-pair half references are, respectively:

- ratio IID versus orbit ratio:
  \((6.130,5.461)\times10^{-5}\) and
  \((6.122,5.418)\times10^{-5}\);
- linear IID versus orbit-coupled linear:
  \((2.355,0.613)\times10^{-4}\) and
  \((2.366,0.609)\times10^{-4}\).

Both within-family rankings therefore survive either half reference, although
the bootstrap intervals below still condition on the fixed full reference.
The reference SE-RMSE is 35.7% of the best final raw RMSE, so it is not
negligible. A worst-case perturbation over the full coordinatewise Bonferroni
95% box can still reverse the final ratio ordering (but not the linear
ordering). The half-reference check and paired method-repeat interval support
the observed final ratio advantage; they do not constitute a worst-case
reference-error certificate.

Bootstrap bands resample only the 10 method seeds and condition on the fixed
Monte Carlo reference. They do not include reference uncertainty and are
descriptive rather than a formal population-level claim. The experiment
estimates the fixed split's cooperative game; the 114-point test set is not a
population ground truth.

## Runtime and artifacts

Across all five budgets and 10 repeats, the four logical estimators represent
336.70 million inner utility calls. Because the two IID estimators reuse the
same evaluated batch, the method run physically evaluated 252.525 million
inner coalitions, plus one shared set of 912 boundary utilities. At the
largest budget, mean utility time per physical batch was 131.9 seconds for the
shared IID batch, 137.8 seconds for orbit-coupled linear, and 133.3 seconds for
orbit ratio. Mean design time was 23.6, 9.9, and 44.6 seconds, respectively.
Summing the reported phases across all budgets gives about 8,952 seconds
(2.49 hours) for the method stage.

The run used a true 128-process `spawn` pool, with one BLAS/OpenMP thread per
worker. Utility submission was bounded to at most
`4 * jobs * chunksize` coalition rows at once, and each large design matrix
was released after its method finished; the formal run completed without swap
use or worker failure.

- Raw method report:
  `results/json/cancer_full_train_rbf_svm_frame_ofa_228k_4p55m.json`
- Ground-truth report:
  `results/json/cancer_full_train_rbf_svm_ground_truth_report.json`
- Ground-truth cache: `results/npz/cancer_full_train_rbf_svm_gt.npz`
- Ground-truth progress/moment cache:
  `results/npz/cancer_full_train_rbf_svm_gt.partial.npz`
- Figure: `results/cancer_full_train_rbf_svm_rmse_228k_4p55m.png`
- Vector figure:
  `results/pdf/cancer_full_train_rbf_svm_rmse_228k_4p55m.pdf`

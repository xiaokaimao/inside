# Iris full-training-set SVM data valuation

## Protocol

- Dataset: `sklearn.datasets.load_iris`.
- Split: stratified 80/20 split with seed 2024.
- Players: all 120 training observations, 40 from each class.
- Fixed test set: 30 observations, 10 from each class.
- Standardization: mean and standard deviation fitted on the 120 training
  observations only, then applied to both partitions.
- Utility: classification accuracy on the fixed test set after fitting
  `sklearn.svm.SVC(C=1, kernel="rbf", gamma="scale")` to a coalition.
- Empty coalition: best constant-label test accuracy, equal to \(1/3\).
- Single-class coalition: predict the sole observed class, also equal to
  \(1/3\) on this balanced test set.
- Full-coalition utility: \(0.933333\).

Because `gamma="scale"` is an estimator rule rather than a fixed numeric
gamma, SVC recomputes the numeric gamma from each coalition's training
features. No hyperparameter is selected using the test labels.

## Ground truth

The reference value uses 800,000 independent antithetic units. Each unit
draws one uniform permutation and pairs it with its reverse. The pair-average
Shapley vector is the independent unit used for the standard error.

For \(n=120\), the conceptual permutation-path budget is

\[
2 + 2(800{,}000)(119)=190{,}400{,}002
\]

utility calls. This is 103.48 times the previous 1,840,002-call reference.
Because the game is deterministic, every path reuses the already evaluated
singleton and leave-one-out boundary utilities. The actual computation was

\[
2(800{,}000)(117)+242=187{,}200{,}242
\]

SVM utility calls. This reuse is algebraically exact and was tested against
direct prefix evaluation on random table games.

The computation used 128 processes, one numerical-library thread per process,
and completed the ground-truth inner calls in 2,531.7 seconds (42.2 minutes).
Only small permutation shards and online moments were retained.

Ground-truth diagnostics:

- SE-RMSE across the 120 coordinates: \(3.280\times10^{-5}\).
- Independent half-split RMSE: \(3.065\times10^{-5}\).
- Maximum Bonferroni simultaneous 95% half-width:
  \(1.323\times10^{-4}\).
- Efficiency check:
  \(\sum_i\widehat\phi_i=0.600000=v(N)-v(\varnothing)\), up to
  \(2.2\times10^{-16}\).

## Fair call budgets

Every estimator includes the same \(2n+2=242\) exact boundary calls. A
complete cyclic orbit costs 120 inner calls. Orbit-ratio needs all
\(n-3=117\) inner sizes, so its minimum valid inner budget is
\(117\times120=14{,}040\).

The earlier diagnostic run used five total-call budgets

\[
14{,}282,\quad 21{,}842,\quad 36{,}242,\quad
60{,}242,\quad 96{,}242.
\]

The extended high-budget run uses

\[
120[500,1000,2000,5000,10000]+242
=
[60{,}242,120{,}242,240{,}242,600{,}242,1{,}200{,}242].
\]

Each point uses 10 independent method seeds. "Total calls per estimate" is
the logical cost of one estimator, including the shared boundary. It is not
the sum of all physical calls made to produce the complete four-method
benchmark. In particular, the two IID estimators reuse the same evaluated
coalitions.

The four methods in the extended run are:

1. **OFA ratio (IID):** official `OFA_fixed` conditional-mean ratio
   estimator with IID OFA size and coalition sampling.
2. **OFA linear (IID):** the unbiased linear estimator on exactly the same
   IID designs and utilities.
3. **Frame-OFA orbit-coupled linear:** the global linear estimator after
   randomized-systematic \(q^\ast\) size coupling, complete cyclic orbits,
   FFT-greedy global frame balancing, and one independent uniform relabeling.
   Every row retains the required \(Q^\ast\) marginal.
4. **Frame-OFA orbit ratio:** the ratio estimator on complete cyclic orbits
   with a minimum-one, per-size allocation and frame-balanced orbit bases.

The first two reuse one physical IID utility batch, but each still has the
same logical call budget when compared as an estimator.

## Extended high-budget results

Lower RMSE is better. These are raw RMSE values against the fixed
800,000-pair Monte Carlo reference.

| Total calls | OFA ratio (IID) | Frame orbit ratio | Orbit change vs. OFA | OFA linear (IID) | Frame orbit-coupled linear | Coupled change vs. IID |
|---:|---:|---:|---:|---:|---:|---:|
| 60,242 | 0.00070249 | 0.00063173 | 10.1% lower | 0.00334865 | 0.00065757 | 80.4% lower |
| 120,242 | 0.00049449 | 0.00044415 | 10.2% lower | 0.00227310 | 0.00048872 | 78.5% lower |
| 240,242 | 0.00034523 | 0.00030322 | 12.2% lower | 0.00166943 | 0.00033500 | 79.9% lower |
| 600,242 | 0.00023042 | 0.00017884 | 22.4% lower | 0.00102345 | 0.00021084 | 79.4% lower |
| 1,200,242 | 0.00015879 | 0.00012150 | 23.5% lower | 0.00072167 | 0.00015335 | 78.8% lower |

At 1,200,242 calls, the paired repeat-level RMSE difference is

\[
\operatorname{RMSE}(\text{OFA ratio IID})
-\operatorname{RMSE}(\text{orbit ratio})
=3.72\times10^{-5},
\]

with bootstrap 95% interval
\([3.06\times10^{-5},4.31\times10^{-5}]\). Orbit-ratio was better in all 10
paired repeats. For IID-linear minus orbit-coupled-linear, the corresponding
difference is \(5.67\times10^{-4}\), with interval
\([5.35\times10^{-4},5.99\times10^{-4}]\).

The log-log slopes over the five budgets are:

| Method | fitted slope |
|---|---:|
| OFA ratio (IID) | -0.491 |
| Frame-OFA orbit ratio | -0.554 |
| OFA linear (IID) | -0.509 |
| Frame-OFA orbit-coupled linear | -0.494 |

All are close to the Monte Carlo \(T^{-1/2}\) rate. At the largest budget,
the noise-corrected RMSE values are \(1.554\times10^{-4}\),
\(1.170\times10^{-4}\), \(7.209\times10^{-4}\), and
\(1.498\times10^{-4}\), respectively, so ground-truth uncertainty does not
explain the observed improvements.

The bootstrap bands resample only the 10 method repeats and condition on the
Monte Carlo reference. They are descriptive and do not incorporate
ground-truth uncertainty. The paired intervals above refer to the mean of
repeat-level RMSE differences, not to the ratio of pooled aggregate RMSEs.

The orbit-coupled linear and orbit-ratio estimators are not algebraically the
same. The former uses a global Horvitz--Thompson/OFA-linear weighting under
randomized \(q^\ast\) marginals; the latter normalizes conditional utility
means separately inside every coalition size.

## Earlier low-budget diagnostic

Lower RMSE is better.

| Total calls | OFA ratio (IID) | Frame orbit ratio | Orbit change vs. OFA | OFA linear (IID) | Frame coupled linear | Coupled change vs. IID |
|---:|---:|---:|---:|---:|---:|---:|
| 14,282 | 0.0016420 | 0.0018290 | 11.4% worse | 0.0068340 | 0.0061134 | 10.5% lower |
| 21,842 | 0.0011830 | 0.0011339 | 4.2% lower | 0.0053790 | 0.0048214 | 10.4% lower |
| 36,242 | 0.0009053 | 0.0008728 | 3.6% lower | 0.0042256 | 0.0038838 | 8.1% lower |
| 60,242 | 0.0006855 | 0.0006441 | 6.0% lower | 0.0031774 | 0.0030267 | 4.7% lower |
| 96,242 | 0.0005600 | 0.0005019 | 10.4% lower | 0.0025409 | 0.0023727 | 6.6% lower |

At 96,242 calls, the paired bootstrap 95% interval for

\[
\operatorname{RMSE}(\text{OFA ratio})
-\operatorname{RMSE}(\text{orbit ratio})
\]

is \([4.52\times10^{-5},7.38\times10^{-5}]\), and orbit-ratio was better in
all 10 paired repeats. The corresponding interval for IID-linear minus
coupled-linear is
\([9.64\times10^{-6},3.12\times10^{-4}]\).

Orbit-ratio is not uniformly superior. At the minimum valid budget, where
there is exactly one orbit per inner size, it is significantly worse than
official OFA in this experiment. Its advantage becomes clear only at larger
budgets; the paired interval first excludes zero in favor of orbit-ratio at
60,242 calls.

### Why the minimum-budget orbit result is worse

At size \(s\), \(R_s\) complete cyclic orbits give every player exactly

\[
N^+_{i,s}=sR_s,\qquad N^-_{i,s}=(n-s)R_s
\]

observations for the two conditional utility means in the ratio estimator.
The minimum budget has 117 total orbits and 117 inner sizes. The
minimum-one allocation therefore forces \(R_s=1\) for every size and consumes
the entire budget before the OFA allocation can be approximated.

For example, at size 2:

- the orbit design gives every player 2 included and 118 excluded
  observations;
- IID OFA has about 339.56 size-2 rows, giving expected counts 5.66 included
  and 333.90 excluded.

At size 60, the orbit design instead gives 60 observations on each side,
versus 43.47 expected under IID OFA. Thus the minimum-one constraint moves
budget away from the difficult rare side of edge sizes and toward central
sizes. At 180 orbits, the allocation is already much closer to its target:
size 2 receives 4 orbits versus a target of 4.35.

The observed minimum-budget loss is a variance effect rather than evidence of
an estimator bias:

| Method | empirical bias RMSE | empirical sampling RMSE |
|---|---:|---:|
| OFA ratio (IID) | 0.000541 | 0.001550 |
| Frame-OFA orbit ratio | 0.000522 | 0.001753 |

Random relabeling makes each fixed base coalition uniform on its size slice,
and deterministic orbit denominators make the ratio estimator unbiased.
However, the 120 rows inside one orbit are dependent. A tight or nearly tight
direction frame controls additive-game geometry; it does not force the
utility-weighted cross-orbit covariance to be negative for an arbitrary
retrained SVM game.

The low-budget geometric intuition does hold in the matched linear
comparison: coupled-linear is 10.5% better than IID-linear at the first
budget. It does not imply universal low-budget dominance for a different
ratio estimator with a binding all-sizes/full-orbits constraint.

The estimator choice is also consequential: both ratio estimators are much
more accurate here than both linear estimators. The clean geometry comparison
is therefore coupled-linear versus IID-linear, while the practical OFA
comparison is orbit-ratio versus IID-ratio.

## Extended-run runtime interpretation

At 1,200,000 inner calls, one SVM utility batch took 15.3--15.5 seconds with
128 processes. Mean design times recorded during the run were 25.9 seconds
for the original row-wise IID generator plus diagnostics, 2.1 seconds for the
scalable orbit-coupled linear design, and 12.4 seconds for the per-size
orbit-ratio design.

After this run, the IID generator and frame diagnostics were rewritten as
strictly uniform, chunked routines. On the same 120-player, 1,200,000-row
target they take 5.85 seconds and peak at about 227 MiB; the estimator's
direction accumulation is also chunked. These implementation changes do not
alter the completed experiment's samples or results.

The extended benchmark made 66.6 million physical inner utility calls:
22.2 million each for the shared IID batch, orbit-coupled-linear batch, and
orbit-ratio batch. Four estimators correspond to 88.8 million logical inner
calls because the two IID estimators reuse one batch.

## Earlier-run runtime interpretation

At the largest budget, one 96,000-inner-call utility batch took about
1.3 seconds with 128 processes. Mean design times were about:

- IID design and diagnostics: 2.05 seconds;
- frame-coupled design: 29.58 seconds;
- cyclic-orbit design: 1.01 seconds.

Thus the figure measures query efficiency, not total wall-clock efficiency.
The orbit method improves high-budget query RMSE without a large design-time
penalty. The current row-by-row coupled greedy design improves its matched IID
linear baseline but has a substantial non-utility design cost.

## Artifacts

- Extended raw report:
  `results/iris_full_train_rbf_svm_frame_ofa_60k_1p2m.json`
- Ground-truth cache: `results/iris_full_train_rbf_svm_gt.npz`
- Extended figure: `results/iris_full_train_rbf_svm_rmse_60k_1p2m.png`
- Extended vector figure:
  `results/iris_full_train_rbf_svm_rmse_60k_1p2m.pdf`
- Earlier diagnostic report:
  `results/iris_full_train_rbf_svm_frame_ofa.json`

# Basic Complementary Contributions baseline audit

## Primary sources and pinned code

- Paper: Jiayao Zhang, Qiheng Sun, Jinfei Liu, Li Xiong, Jian Pei, and
  Kui Ren, *Efficient Sampling Approaches to Shapley Value Approximation*,
  Proc. ACM Manag. Data 1(1), Article 48, 2023,
  <https://doi.org/10.1145/3588728>.
- Open full text: <https://par.nsf.gov/servlets/purl/10448773>.
- The paper's official repository:
  <https://github.com/ZJU-DIVER/ShapleyValueApproximation>.
- Audited MIT-licensed commit:
  `25e04b18433175dbc88abad547b319be904a5771` (2023-04-12).
- Basic parallel CC implementation: `cc_shap` and `_cc_shap_task` in
  `shapley/gtsv.py` at that commit.

The implementation in this project follows the paper's Algorithm 2 and the
official `cc_shap` accumulator, while replacing the upstream unseeded
`multiprocessing.Pool()` execution with the project's deterministic persistent
worker pool. The cooperative game itself is not changed.

## Exact estimator

For a coalition \(S\subseteq N\), define

\[
c(S)=v(S)-v(N\setminus S).
\]

For player \(i\) and coalition size \(k\), let

\[
\mu_{i,k}
=
\mathbb E\left[c(S)\mid i\in S,\ |S|=k\right].
\]

The complementary-contribution identity is

\[
\phi_i=\frac{1}{n}\sum_{k=1}^{n}\mu_{i,k}.
\]

One basic-CC pair sample draws \(J\) uniformly from
\(\{1,\ldots,n\}\), draws a uniform size-\(J\) coalition \(S\), and
evaluates both \(v(S)\) and \(v(N\setminus S)\). Players in \(S\) receive
\(+c(S)\) in stratum \(J\). Players outside \(S\) receive \(-c(S)\) in
stratum \(n-J\). Each player's estimate is the equally weighted average of
the \(n\) empirical stratum means.

Thus one pair updates every player but costs exactly **two utility calls** for
a generic data-valuation game. The paper's and official code's parameter
\(m\) counts complement pairs, not utility invocations.

## Finite-budget zero-count issue

Algorithm 2 has no warm-up step and its displayed division is undefined when
a player/size cell receives no sample. The official `cc_shap` code resolves
this by assigning a zero mean to every missing cell. That choice is reproduced
for the official-basic baseline; no hidden initialization or efficiency
projection is added.

For a fixed player and size \(k\), a pair hits that cell with probability

\[
p_k=
\begin{cases}
2k/n^2, & 1\le k<n,\\
1/n, & k=n.
\end{cases}
\]

With \(m\) independent pairs and official zero filling,

\[
\mathbb E[\widehat\mu_{i,k}]
=\mu_{i,k}\left[1-(1-p_k)^m\right].
\]

Consequently, the paper's finite-sample unbiasedness claim requires every
stratum count to be positive (or a design with prescribed positive counts).
The implemented official zero-fill estimator is generally biased at finite
\(m\), although the missing-cell bias vanishes asymptotically. A simple
counterexample is a two-player additive game with one pair: the expectation
of the zero-fill estimate is one half of the true Shapley vector.

Every experiment therefore records the number and fraction of missing cells,
the minimum cell count, and the unprojected efficiency gap.

## Equal-call Wine allocation

The existing OFA/Frame-OFA methods spend \(2n+2=286\) calls on exact boundary
coalitions and the remainder on internal coalitions. Basic CC does not use
those boundary values. To compare at the exact same total call coordinate,
CC spends the entire total on complement pairs:

| Total utility calls | Basic CC pairs |
|---:|---:|
| 71,286 | 35,643 |
| 142,286 | 71,143 |
| 284,286 | 142,143 |
| 710,286 | 355,143 |
| 1,420,286 | 710,143 |

All totals are even, so there is no unused call. At the first point the
expected number of missing player/size cells is about 4.26 out of
\(142^2=20{,}164\); the later expected counts are approximately 0.122,
\(1.07\times10^{-4}\), \(7.13\times10^{-14}\), and
\(3.63\times10^{-29}\).

## Read-only audit of `integral_shapley/src/core`

The supplied baseline directory was inspected without modification. Its
`src/core/cc_methods.py::cc_shapley_parallel` samples `num_MC` coalitions at
every size rather than drawing one random size per pair. It also ignores
missing cells with `nanmean`, catches utility failures with different rules,
and uses a different utility-function interface. It is therefore a useful
separate **stratified CC** implementation, but it is not the paper's basic CC
baseline and is not relabeled as one here.

## Wine experiment outcome

The official-basic estimator was run with 128 processes, 128 deterministic
coarse tasks per repeat, and 10 repeats at each same-total-call budget. Raw
RMSE against the existing 800,000-antithetic-pair reference was:

| Total calls | Basic CC RMSE | 95% repeat bootstrap interval | vs. OFA ratio IID |
|---:|---:|---:|---:|
| 71,286 | 0.00122259 | [0.00111626, 0.00133738] | 93.3% higher |
| 142,286 | 0.00062439 | [0.00060924, 0.00063839] | 31.6% higher |
| 284,286 | 0.00042759 | [0.00041176, 0.00044224] | 30.1% higher |
| 710,286 | 0.00027627 | [0.00026644, 0.00028554] | 37.7% higher |
| 1,420,286 | 0.00020083 | [0.00019542, 0.00020598] | 32.3% higher |

The first-budget repeats had `[9, 2, 5, 3, 4, 2, 3, 4, 9, 5]` missing
player/size cells, or 4.6 of 20,164 on average. Every later repeat had full
stratum coverage. Therefore zero filling explains part of the especially poor
first point, but it cannot explain the persistent gap after 142,286 calls;
the latter is an estimator-variance difference on this game. Basic CC is left
unprojected, as in the official implementation, and its mean efficiency gaps
fall from 0.01883 to 0.000472 across the five budgets.

The augmented raw report and log-scale figure are:

- `results/wine_full_train_rbf_svm_frame_ofa_with_cc_71k_1p42m.json`
- `results/wine_full_train_rbf_svm_rmse_with_cc_71k_1p42m.png`
- `results/wine_full_train_rbf_svm_rmse_with_cc_71k_1p42m.pdf`

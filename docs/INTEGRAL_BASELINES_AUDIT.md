# `integral_shapley/src/core` baseline audit and local ports

## Scope and immutability

The directory
`/home/maoxiaokai/python_project/integral_shapley/src/core` was treated as a
read-only source.  No experiment imports it at runtime.  Comparable formulas
were ported into `/home/maoxiaokai/python_project/frame/frame_ofa`, adapted to
the persistent `GameEvaluator`, and tested independently.  This avoids a
nested process pool and makes physical utility calls auditable.

The comparison target is the ordinary Shapley value of all 142 Wine training
examples for the fixed cooperative game

\[
v(S)=\operatorname{accuracy}\!\left(
  \operatorname{RBF\text{-}SVM trained on }S,
  D_{\rm test}
\right).
\]

Every plotted budget is a cap on actual invocations of this `v(S)`, not an
algorithm-specific notion of a sample.

## Included local ports

| Plot key | Local implementation | Statistical target | Physical calls at cap \(B\) |
|---|---|---|---:|
| `gels_shapley` | `frame_ofa/regression_baselines.py` | official GELS-Shapley Algorithm 3 self-normalized ratio plus efficiency offset | \(B\) |
| `kernel_shap_sampled` | `frame_ofa/regression_baselines.py` | sampled-Gram constrained KernelSHAP, ridge \(10^{-8}\) | \(B\) |
| `group_testing` | `frame_ofa/group_testing.py` | Jia et al. Algorithm 1 pairwise contrasts | \(B\) |
| `diff` | `frame_ofa/differential.py` | Pang et al. unstratified Diff, Algorithm 1 | \(B\) |
| `s_diff` | `frame_ofa/stratified_differential.py` | size-stratified S-Diff with strict pair coverage | \(B\) |
| `tmc_shapley` | `frame_ofa/tmc.py` | fixed-count, early-truncated TMC-Shapley | \(2+\sum_{t=1}^{\lfloor(B-2)/n\rfloor}L_t\), \(1\le L_t\le n\) |
| `stratified_marginal_mc` | `frame_ofa/traditional_mc.py` | player-by-cardinality stratified marginal Monte Carlo | \(2\lfloor B/2\rfloor\) |

All seven interfaces return `.values` and `.diagnostics`.  Their random seeds
are assigned at the sample, stratum, or coarse-task level as appropriate, and
the number of requested tasks is recorded.  Utility exceptions are propagated;
they are never silently replaced by zero.

## Formula checks

### Shared KernelSHAP/GELS coalition law

For \(s=1,\ldots,n-1\), both ports draw

\[
q_s=\frac{n}{2H_{n-1}s(n-s)},
\qquad S\mid |S|=s\sim\operatorname{Unif}\binom Ns.
\]

The KernelSHAP port accumulates the empirical moments

\[
\widehat G=\sum_t z_tz_t^\top,
\qquad
\widehat b=\sum_t z_t\{v(S_t)-v(\varnothing)\},
\]

and solves the sampled constrained least-squares problem under

\[
\mathbf 1^\top\phi=v(N)-v(\varnothing).
\]

The solve is performed in an orthonormal basis of \(\mathbf1^\perp\).  The
empirical Gram inverse means that this original sampled-Gram estimator is not
generally finite-sample unbiased.  The explicit \(10^{-8}\) ridge matches the
configuration of the external baseline; it is reported rather than hidden.

The user's external `gels_methods.py` is not the self-normalized Algorithm 3
estimator: its finite-batch formula is the linear Horvitz--Thompson estimator
also called `simSHAP` in the authors' code.  That mislabelled curve was removed
from the completed report.

The replacement is a clean-room implementation of Li and Yu's official
**GELS-Shapley**, checked against `watml/fastpvalue` commit
`34392c53f8d609aebb5e0c0e57c165411d291a46`, class `GELS_shapley`.  With

\[
A_i=\sum_{t=1}^m\mathbf1\{i\in S_t\}v(S_t),\qquad
C_i=\sum_{t=1}^m\mathbf1\{i\in S_t\},
\]

Algorithm 3 returns

\[
 r_i=H_{n-1}\frac{A_i}{C_i},
\qquad
\widehat\phi_i=r_i+
\frac{v(N)-v(\varnothing)-\sum_jr_j}{n}.
\]

The official code maps (C_i=0) to a raw ratio of zero; the port preserves
that fallback.  Conditional on positive counts the ratio is unbiased for its
conditional-mean target.  Unconditionally, the fallback leaves a strict
finite-sample bias of order (2^{-m}), negligible here but recorded rather
than hidden.  The two boundary calls and (m=B-2) inner calls give exactly
(B) physical calls.  Worker tasks return only additive (A_i,C_i) totals;
the parent computes one global ratio, so using 128 tasks does not change the
estimator.  The primary reference is the
[ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/df22a19686a558e74f038e6277a51f68-Paper-Conference.pdf),
and the authors' source is
[`watml/fastpvalue`](https://github.com/watml/fastpvalue/blob/34392c53f8d609aebb5e0c0e57c165411d291a46/utils/estimators.py#L801-L905).

### Group Testing

The port uses the paper's

\[
Z=2H_{n-1},\qquad
q(k)=\frac{1/k+1/(n-k)}{Z}.
\]

Each coalition supplies contrasts between included and excluded players; the
final common translation enforces efficiency without changing any estimated
pairwise difference.  The implementation was checked against Algorithm 1 in
the primary [AISTATS 2019 paper](https://proceedings.mlr.press/v89/jia19a.html).

### Diff and S-Diff

Diff samples \(k\) with

\[
p_k\propto\frac1{k(n-k)}
\]

and records, for every ordered cut pair \(i\in S,j\notin S\), the conditional
mean \(M_{ij}\) of \(v(S)-v(\varnothing)\).  It forms

\[
\widehat\Delta_{ij}=M_{ij}-M_{ji},
\qquad
\widehat\phi_i=
\frac{v(N)-v(\varnothing)}n+
\frac1n\sum_j\widehat\Delta_{ij}.
\]

No missing ordered pair is filled with zero: missing coverage raises
`DifferentialCoverageError`.

S-Diff stores the same quantities separately for every size layer and averages
the layer-wise contrasts.  Its state is \(O(n^3)\); at \(n=142\), the single
parent state is 34,117,488 bytes (32.54 MiB).  It checks the paper's
\(\lceil4n\log n\rceil\) sample precondition and guarantees positive coverage
for every layer/order-pair cell before recovery.

One qualification is essential: the external file's coverage fill is a
first-missing-pair greedy routine, not a literal implementation of Appendix
Algorithm 6.  The local diagnostics consequently say
`s_diff_external_strict_greedy_port`; no universal finite-sample unbiasedness
claim is made for that deterministic initialization.  The random remainder
has the paper's distribution.  The primary publication is
[Pang et al., PACM SIGMOD 2025](https://doi.org/10.1145/3709725).

### TMC-Shapley

For every sampled permutation \(\pi\), the port evaluates prefix marginals

\[
v(P_i^\pi\cup\{i\})-v(P_i^\pi).
\]

After each prefix it checks

\[
|v(P_j^\pi)-v(N)|\le 10^{-3}|v(N)|.
\]

The counter resets after a failed check, and the external source stops at
`counter > 5`: the default therefore requires six consecutive near-full
prefixes.  Unvisited suffix contributions stay zero.  No efficiency
projection is applied.  A complete path deliberately re-evaluates the full
prefix, matching the source, so its cost is \(n\) calls even though the two
boundary utilities were already measured.

The source interface fixes a permutation count rather than a physical-call
budget.  For cap \(B\), the local experiment therefore fixes

\[
m=\left\lfloor\frac{B-2}{n}\right\rfloor
\]

before observing any path, runs exactly those \(m\) permutations, and plots
the observed calls \(2+\sum_tL_t\).  Truncation savings are left unused.  This
conservative rule both guarantees the cap and avoids introducing an extra
length bias by adaptively adding permutations until a contribution-dependent
stopping time exhausts the budget.

This is a faithful port of the fixed-count adaptation present in the user's
baseline library, not a claim to reproduce the original DataShapley package's
additional cross-permutation convergence rule.  The port improves engineering
only: persistent workers, sample-indexed random seeds, finite-value checks,
and exception propagation.  The stopping distribution and returned raw
estimator are unchanged.

Early truncation is not generally unbiased.  In particular, a prefix utility
being near \(v(N)\) does not imply that every remaining marginal is small;
large positive and negative suffix marginals may cancel.  The tests include a
three-player game whose true value is \((0,5,-4)\) but whose patience-zero TMC
limit is \((1/3,1/3,1/3)\), even though both vectors satisfy efficiency.

### Stratified marginal Monte Carlo

For each player \(i\) and predecessor size \(s\), the port estimates

\[
\mu_{i,s}=\mathbb E_{S\subseteq N\setminus\{i\},\,|S|=s}
\left[v(S\cup\{i\})-v(S)\right],
\qquad
\widehat\phi_i=\frac1n\sum_{s=0}^{n-1}\widehat\mu_{i,s}.
\]

The number of two-call marginal observations is balanced over all \(n^2\)
player/size strata.  It needs at least \(2n^2\) calls; Wine's smallest cap
71,286 exceeds the 40,328-call minimum.  No efficiency projection is applied.

## Deliberately excluded entries

| External entry | Reason it is not a fair curve in this experiment |
|---|---|
| CoShap | The local file is a staged, non-official adaptation and does not faithfully implement the paper's `m_j` semantics. Its nominal sample budget is not a physical-call budget: at \(n=142\), the smallest initialization uses about 79.2k calls and already exceeds the first 71,286-call cap; `num_samples=71,286` was observed to use 112,673 calls. |
| DU-Shapley | The paper's target is a dataset-owner valuation proxy, not the fixed-game ordinary Shapley value of individual Wine training rows. The external “ML variant” reduces to the traditional stratified marginal estimator already included under its correct name. |
| G-Shapley | It follows a stochastic online softmax/logistic gradient-training path rather than evaluating the fixed RBF-SVM coalition utility. The estimand and model are different. |
| per-player traditional MC | Same ordinary target, but duplicated by the included all-player stratified marginal port. |
| exact Shapley | Requires exponentially many coalitions at \(n=142\). |

Exclusion is based on target and call-accounting compatibility, not observed
accuracy.

## External-source integrity record

The SHA-256 values below were checked after porting.  They match the hashes
recorded during the read-only audits:

| Source file | SHA-256 |
|---|---|
| official `watml/fastpvalue/utils/estimators.py` at commit `34392c53...` | `2eff99580289e3f2374a72b720baa0284a2fff71341be47d5b757c585ff6e2fa` |
| `tmc_shapley_methods.py` | `d64548f00ceded3660764b8cd02d646b2548f4e19d0bee53e474d53dac3cbac1` |
| `kernel_shap_methods.py` | `6b6ebd4f46093718a90e880a8d117d140816247f87bbc123764f89d4d90f6244` |
| `gels_methods.py` | `a7b4fd60f9a9d668150c19e8a33edf87a4d4db737f4db6faf280279c4060e0c9` |
| `group_testing_methods.py` | `077d4edf01ee888cb1a4f65f6ce3558fbe244f8fbd8c065da9885074a10aeb09` |
| `differential_matrix_methods.py` | `fbcf95f4e43c316a6ad0d57391591997d5bc9d6d7d9b683da3d8ba703c1ed5ec` |
| `traditional_methods.py` | `64a3bef868d0385949be03a8223e9416acca546882961d4b450932cb1a75829b` |
| `coshap_methods.py` | `c879c8c43b0522764c1a120c0c0b3f1e44e81a3069ad1613a55eec604049d59f` |
| `du_shapley_methods.py` | `8c61c444ed0f8c9008983e093d6b8208d77536993f1a7132909a7eab74d0f341` |
| `g_shapley_methods.py` | `c0aa2e28a11ece7755e2c6d432bba0ea087e98ee470983467c2c5942ab247d0a` |

Some of these files were already staged additions in the external repository.
Their pre-existing Git status was preserved; the hash check, rather than a
claim that the external worktree was clean, is the immutability evidence.

## Verification surface

The local tests cover:

- exact small cooperative games and exhaustive population identities;
- exact official GELS-Shapley ratio parity, including the zero-count fallback,
  constant-utility-shift invariance, and the identity
  \(\sum_i C_i=\sum_s sN_s\);
- additive-game recovery where the estimator should be exact;
- KernelSHAP KKT and rank-deficient solves;
- Group/Diff/S-Diff pair and layer distributions;
- strict Diff/S-Diff coverage failures;
- exact physical-call accounting and unused-call remainders;
- TMC's six-prefix stopping boundary, relative tolerance, fixed-count budget
  mapping, truncation savings, and an efficiency-preserving bias counterexample;
- fixed-seed reproducibility and invariance to task partitioning where the
  implementation promises it; and
- real `spawn`-based `GameEvaluator` execution.

The final experiment report is also independently revalidated from its raw
repeat-level estimates: RMSE, efficiency gaps, actual/unused calls, and every
repeat diagnostic must agree before plotting.

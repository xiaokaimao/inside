# Upstream OFA reference

The official research code was downloaded for verification from:

- Repository: https://github.com/watml/one-for-all
- Commit: `f5e9da3f6f35f5fee2de1c89dbd779ed25c505cd`
- Commit date: 2024-12-11
- Paper: *One Sample Fits All: Approximating All Probabilistic Values
  Simultaneously and Efficiently*, NeurIPS 2024

The local snapshot is stored unchanged in `ofa_upstream/` and excluded from
version control.

The upstream repository does not contain a license file as of the pinned
commit.  Consequently, the new implementation in `frame_ofa/` is kept as a
clean-room, paper-derived extension rather than copying and editing upstream
source.  Check redistribution rights with the upstream authors before
publishing a combined derivative repository.

To reproduce the local reference checkout:

```bash
git clone https://github.com/watml/one-for-all ofa_upstream
git -C ofa_upstream checkout f5e9da3f6f35f5fee2de1c89dbd779ed25c505cd
```

## Local mean-balance extension

`mean_balance` belongs to the local greedy frame design; it is not an
upstream-OFA parameter.  For paper-facing `mode="inside_greedy"`, the local
`FrameOFAEstimator` and upstream-style adapter resolve omitted hyperparameters
to `candidate_pool=64`, `mean_balance=1/16`, and
`mean_balance_mode="normalized"`.  The legacy/global default
`mode="coupled"` remains `candidate_pool=32`, normalized
`mean_balance=1.0`.  In normalized mode the input is a dimensionless
\(\lambda_0\), converted to the raw objective coefficient as

\[
\lambda_{\mathrm{eff}}
=\lambda_0\left(1-\frac{1}{n-1}\right)
  \frac{1}{T}\sum_{t=1}^{T} w_t^2,
\]

with \(w_t^2=s_t(n-s_t)\) for the radial coupled design and \(w_t^2=1\) for
the unweighted per-size design.  Historical local experiments that passed
`mean_balance=0.1` as the coefficient itself are reproduced with
`mean_balance=0.1, mean_balance_mode="raw"`.  For INSIDE-Greedy,
\(\lambda_0=1/16\) retains a small soft first-moment penalty while explicit
OFA ratio-coverage checks remain active.  This K64/1-over-16 setting is an
explicit paper-facing configuration, not a universal optimality claim; it
also costs more design time than smaller candidate pools.  Historical
K64/zero reports remain versioned separately.

The formal `mode="inside_orbit"` adapter default is `candidate_pool=4`,
matching the Airport, Voting, and Wine benchmark bundle.  The lower-level
legacy `mode="orbit"` retains its historical `candidate_pool=32` default.

## Official GELS-Shapley reference

The GELS-Shapley replacement was independently implemented from Algorithm 3
in Li and Yu, *Faster Approximation of Probabilistic and Distributional Values
via Least Squares* (ICLR 2024), and checked against the authors' repository:

- Repository: https://github.com/watml/fastpvalue
- Commit: `34392c53f8d609aebb5e0c0e57c165411d291a46`
- Source: `utils/estimators.py::GELS_shapley`
- Downloaded source SHA-256:
  `2eff99580289e3f2374a72b720baa0284a2fff71341be47d5b757c585ff6e2fa`
- Paper: https://openreview.net/forum?id=lvSMIsztka

That repository also has no license file at the verified commit.  Its source
was therefore used only for behavioral comparison; the local implementation
is a clean-room rewrite of the published formula.

## Complementary Contributions reference

The official code accompanying Zhang et al., *Efficient Sampling Approaches
to Shapley Value Approximation* (Proc. ACM Manag. Data, 2023) was audited at:

- Repository: https://github.com/ZJU-DIVER/ShapleyValueApproximation
- Commit: `25e04b18433175dbc88abad547b319be904a5771`
- Basic implementation: `shapley/gtsv.py::cc_shap`
- Paper: https://doi.org/10.1145/3588728
- License at the pinned commit: MIT

`frame_ofa/complementary.py` ports the basic CC sampling and accumulator into
the local Boolean-coalition and persistent-`GameEvaluator` interfaces. It does
not modify or import the user's separate `integral_shapley` baseline tree.

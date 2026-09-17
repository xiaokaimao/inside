# Orthogonal baseline: source, implementation and experiment protocol

## Paper and author code

Rory Mitchell, Joshua Cooper, Eibe Frank, Geoffrey Holmes. **Sampling
Permutations for Shapley Value Estimation.** JMLR **23**(43):1–46, 2022.
[Journal record](https://jmlr.org/papers/v23/21-0439.html),
[final paper, Section 4.2 and Algorithm 3](https://jmlr.org/papers/volume23/21-0439/21-0439.pdf).

The first author's public implementation is
[RAMitchell/shap_sampling](https://github.com/RAMitchell/shap_sampling), pinned to
commit `fae42e3aa332665f8bd60bcd0ac2cfbf0f8da4ac` (2021-04-25). Its commit author
is Rory Mitchell. `experiments.py` registers `Orthogonal` as
`algorithms.OrthogonalSphericalCodes()`; the sampler and generic permutation
estimator are in
[algorithms.py](https://github.com/RAMitchell/shap_sampling/blob/fae42e3aa332665f8bd60bcd0ac2cfbf0f8da4ac/algorithms.py).
This is author-maintained research code, predating the journal publication;
the JMLR article page does not separately link a software release.

No license file was present in the repository at this commit. The project
therefore retains source URLs and SHA-256 hashes in
[`provenance.json`](../third_party/shap_sampling/provenance.json), with an
independent mathematical implementation in
[`frame_ofa/orthogonal.py`](../frame_ofa/orthogonal.py). Upstream function bodies
were downloaded to a temporary directory for comparison and are not vendored.

## Sampling and budget

Each block draws Gaussian vectors, constructs up to `n-1` orthonormal
directions in the zero-sum hyperplane in `R^n`, sorts their coordinates into
permutations, and adds the reversed permutations. Independent blocks are
averaged through ordinary permutation marginal contributions.

The journal's Algorithm 3 describes a full basis with `2(n-1)` permutations.
The author's `_orthogonal_permutations` also accepts fewer directions in the
final basis. This baseline follows that behavior: it permits a partial basis
and always retains complete forward/reverse pairs. Each selected direction
has the same uniform spherical marginal, so this fixed, utility-independent
prefix selection preserves unbiasedness. Rows within a basis are dependent.
This method is distinct from Yang et al.'s component orthogonal array (COA).

For `n > 1` players and a total utility-call cap `B`, the local budget is:

```
number of pairs = floor((B - 2) / (2 * (n - 1)))
number of permutations = 2 * number of pairs
actual utility calls = 2 + number of permutations * (n - 1)
minimum feasible budget = 2 * n
```

The two endpoint utilities are computed once. Interior coalition evaluations
are counted whenever executed, including repeats; there is no hidden coalition
cache. Every permutation is evaluated to completion, with no TMC truncation or
post-hoc efficiency projection. Empty-set utility may be nonzero. The special
one-player case uses exactly two endpoint calls. Infeasible requests fail
before evaluating any coalition.

The author's `n_samples` argument is not this project's physical utility-call
counter. We use the existing comparison's total call caps and record actual
calls, unused calls, full orthogonal blocks, and the number of final partial
basis directions. At the smallest existing budgets:

| Dataset | Players | Call cap | Actual Orthogonal calls | Minimum calls |
|---|---:|---:|---:|---:|
| Airport | 100 | 50,202 | 50,096 | 200 |
| Voting | 51 | 25,604 | 25,602 | 102 |
| Wine | 142 | 71,286 | 71,066 | 284 |
| Cancer | 455 | 228,412 | 227,910 | 910 |

Here, Wine and Cancer players are valued training examples, as in the existing
data-valuation experiments; they are not input features. The game, training
split, scaler, model, and reference Shapley vector are reconstructed from the
source comparison. This integrates the estimator into the project's games;
it does not reproduce the paper's feature-attribution benchmark datasets.

## Numerical implementation and source agreement

The source implementation projects Gaussian vectors away from the constant
normal and uses modified Gram–Schmidt. The local implementation obtains the
same orthogonal frame by QR decomposition after prepending that constant
normal. Adjusting by the signs of `R`'s diagonal preserves Gram–Schmidt's
orientation. This avoids numerical loss of orthogonality at large dimensions.

Each block uses a local NumPy PCG64 generator with
`SeedSequence(seed, spawn_key=(block,))`. This makes sampling independent of
the number of workers and task partition. The author's global NumPy seed is
not numerically interchangeable with this local seed. For source comparison,
both implementations receive identical Gaussian draws.

[`verify_orthogonal_upstream.py`](../experiments/verify_orthogonal_upstream.py)
checks the source SHA-256, selects the original sampler and estimator function
ASTs, and executes their unchanged bodies. It removes Numba decorators and
excludes unrelated imports and experiment entry points. The resulting
[`fixtures.json`](../third_party/shap_sampling/fixtures.json) contains seven
executed cases: one pair, partial basis, full basis, multiple bases, and a
partial final basis, with 2–51 players. All permutation arrays agree exactly;
estimated values agree to `rtol=atol=1e-12` on a shifted nonlinear game.

Regenerate the source comparison with network access:

```bash
python -m experiments.verify_orthogonal_upstream --download
# Or use an already downloaded, hash-matching algorithms.py:
python -m experiments.verify_orthogonal_upstream --source /tmp/algorithms.py
```

Core tests also check orthogonality and zero-sum directions through 455
players, complete reverse pairing, call accounting, exact quadratic-game
values with nonzero empty utility, a nonlinear unbiasedness diagnostic,
single-player behavior, and serial/spawn agreement. Dependent permutations
are not reported as independent standard-error samples; experiment uncertainty
uses independent complete estimator repeats.

## API and comparison runner

```python
from frame_ofa import GameEvaluator, estimate_orthogonal_shapley, orthogonal_budget

plan = orthogonal_budget(num_players=100, total_call_budget=50_202)
# GameClass implements evaluate(boolean_coalition).
with GameEvaluator(GameClass, game_args, n_jobs=8) as evaluator:
    result = estimate_orthogonal_shapley(
        evaluator, 100, 50_202, seed=0, num_tasks=128,
    )
print(result.values)
print(result.diagnostics.utility_evaluations)
```

The separate runner consumes the existing `*_inside_with_shapdoe_3repeats.json`
reports and writes `*_inside_with_shapdoe_orthogonal_3repeats.json`. The source
report is embedded unchanged and fingerprinted. The validator checks that all
previous curves remain unchanged, recomputes metrics from saved estimates,
and verifies budgets, seeds, efficiency, and completion status before resume
or plotting. Each new budget has three independent seeds.

```bash
python -m experiments.add_orthogonal_baseline \
  --datasets airport voting wine cancer --jobs 40
python -m experiments.plot_orthogonal_comparison \
  results/json/airport_inside_with_shapdoe_orthogonal_3repeats.json
```

For a selected budget subset, use `--budget-indices 0 1 2`. Remaining feasible
points are `pending`. Plotting such reports requires `--allow-incomplete` and
marks the figure as partial. No missing RMSE is substituted with zero. The
runner resumes existing cells instead of recomputing them. Each new cell's
wall time includes its fresh pool, sampling, game calls, aggregation and pool
shutdown; inherited baseline timings are not remeasured. Plots compare RMSE
against actual utility calls, not elapsed time.

The combined report can remain partial because a source ShapDoE curve is
partial even after Orthogonal finishes; `orthogonal_status` records the new
method's completion separately.

## Checked comparison artifacts (2026-09-08)

| Dataset | Orthogonal budget points completed | Independent repeats | Status |
|---|---:|---:|---|
| Airport | 5 / 5 | 3 per point | Complete |
| Voting | 5 / 5 | 3 per point | Complete |
| Wine | 5 / 5 | 3 per point | Complete |
| Cancer | 1 / 5 | 3 at the first point | Partial; later points not run |

There are 48 saved Orthogonal repeat cells. Each output includes JSON
estimates and diagnostics, a recomputed `.validation.json` audit, and
PNG/PDF/CSV comparison artifacts. The completed comparisons are:

- [Airport figure](../results/airport_inside_with_shapdoe_orthogonal_3repeats.png)
- [Voting figure](../results/voting_inside_with_shapdoe_orthogonal_3repeats.png)
- [Wine figure](../results/wine_inside_with_shapdoe_orthogonal_3repeats.png)
- [Cancer partial figure](../results/cancer_inside_with_shapdoe_orthogonal_3repeats.png)

Original reports and the previous ShapDoE reports remain unchanged. The full
test suite passed with **388 tests and 77 subtests**; the 13 report/plot tests
were rerun successfully after the final CSV call-count export adjustment.
The full suite emitted the same 11 existing fork/constant-correlation warnings.

Resume the remaining Cancer Orthogonal points without recomputing the first:

```bash
python -m experiments.add_orthogonal_baseline --datasets cancer --jobs 40
python -m experiments.plot_orthogonal_comparison \
  results/json/cancer_inside_with_shapdoe_orthogonal_3repeats.json --allow-incomplete
```

The existing Cancer ShapDoE source still has pending LS points, so the combined
comparison remains partial until a completed source is supplied. A checkpoint
rejects a changed source fingerprint; use a new `--output-dir` when switching
to a different source report.

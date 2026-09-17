# ShapDoE baseline: Yang et al., JASA 2024

## Paper and official code

Liuqing Yang, Yongdao Zhou, Haoda Fu, Min-Qian Liu and Wei Zheng.
*Fast Approximation of the Shapley Values Based on Order-of-Addition
Experimental Designs*. Journal of the American Statistical Association
119(547), 2294–2304 (2024).
[DOI: 10.1080/01621459.2023.2257364](https://doi.org/10.1080/01621459.2023.2257364).

Crossref confirms the author order, volume, issue and pages. The online date
is 24 October 2023; the issue publication date is 2 July 2024. Thus the 2023
DOI suffix does not contradict the 2024 citation. The
[author preprint](https://arxiv.org/abs/2309.08923) supplies the accessible
algorithm and null-player extension (Sections 2.2, 3.1 and 3.3).

The **official author implementation is the R package
[ShapDoE 1.0.0 on CRAN](https://CRAN.R-project.org/package=ShapDoE)**,
published 30 April 2024. Its DESCRIPTION lists Liuqing Yang as author,
creator, copyright holder and maintainer and explicitly cites this paper.
It is licensed **MIT + file LICENSE**. The
[GitHub repository](https://github.com/cran/ShapDoE) is a read-only CRAN
mirror, not a separate author-maintained development repository.

The original `R/estsh.R`, `DESCRIPTION`, and `LICENSE` are retained byte for
byte in `third_party/shapdoe/`; `provenance.json` records the source archive
URL, release hash, file hashes and mirror commit
`c26f94633341aaaacfc6160a8589f6beb6a067a5`.
The full MIT permission text is retained alongside them.

## Implementation mapping

| Official R entry | Local Python entry |
|---|---|
| `onels`, `est.shls` | `latin_square_design`, `estimate_shapdoe(method="ls")` |
| `onecoa.prime`, `est.shcoa.prime` | Prime-field branch of `component_orthogonal_array_design`, `estimate_shapdoe(method="coa")` |
| `onecoa`, `est.shcoa` | Polynomial-field branch of the same COA implementation |

Both estimators average complete permutation marginal-contribution vectors.
LS randomizes the initial cyclic square's symbols, columns and rows using
the author's construction. COA constructs the affine arrays over GF(q).
For a prime power, arithmetic uses the finite field, **not** integer
arithmetic modulo q. The automatically selected primitive polynomials for
GF(8) and GF(9) match the official examples.

For a non-prime-power player count, COA adds null players up to the smallest
prime power q >= n, as described in the paper. Removing those null symbols
from each resulting path gives the identical real-player estimate. Repeated
projected permutations are retained with their original multiplicities.

This is a Python port; R is not required in production. NumPy's random-number
stream differs from R's. Equal numeric seeds do not promise identical designs
across languages. The unmodified author source was executed in a temporary
R 4.1.2 environment with gtools 3.9.2. Fixtures record the R sampling draws,
design matrices and original estimator outputs for LS(5), COA(5), COA(8),
COA(9), and COA(7) projected to six real players. Replaying the recorded
sampling draws in Python checks exact matrix equality and numerical estimator
agreement. `experiments/verify_shapdoe_r.R` regenerates those fixtures with R;
the normal Python tests consume them without requiring R. Other tests verify
combinatorial properties, exact small-game results, polynomial arithmetic,
shift invariance, physical call counts and serial/spawn equivalence.

## Budget and utility conventions

The R parameter `n` means **permutations**, whereas the local public API
accepts a **total coalition utility-call cap**. The local player count is
denoted by n below. After two shared endpoint evaluations, each complete
real-player path uses n-1 interior evaluations. Thus:

- LS: n permutations per block; block cost n(n-1).
- COA: q(q-1) permutations per block; block cost q(q-1)(n-1).
- Number of blocks: floor((B-2)/block_cost).
- Actual calls: 2 + blocks × block_cost; leftover calls are reported.

Only whole blocks are used. No partial COA/LS, IID remainder, truncation or
post-hoc efficiency projection is introduced. A budget that cannot fund a
block fails before evaluation in the API and becomes `budget_infeasible`
in the experiment report. Curves use **actual calls** as x coordinates.

The R implementation assumes v(empty)=0. Wine and Cancer use nonzero empty
accuracy. The port uses their actual v(empty), equivalently applying the
published estimator to v(S)-v(empty). Deterministic endpoint reuse and null
deletion preserve all real-player marginal contributions. Interior coalition
utilities are not memoized. As with the project's other cached-endpoint
baselines, this convention assumes a deterministic utility.

Uncertainty inside the estimator uses independent **whole designs** as
replicates. Dependent rows within a design are not treated as independent.
One block yields unavailable within-estimator variance (`None`), rather than
a misleading zero standard error. Experiment intervals use the three
independent experiment repeats, matching the existing comparison protocol.

| Dataset | Players n | COA order q | Minimum LS calls | Minimum COA calls |
|---|---:|---:|---:|---:|
| Airport | 100 | 101 | 9,902 | 999,902 |
| Voting | 51 | 53 | 2,552 | 137,802 |
| Wine | 142 | 149 | 20,024 | 3,109,334 |
| Cancer | 455 | 457 | 206,572 | 94,609,970 |

Consequently the existing five caps allow COA only at Airport's final point
and Voting's final two points. None of the current Wine/Cancer caps can fund
COA. LS is feasible at all existing points.

## Reproduction and result integrity

Use the project's experiment environment (NumPy, SciPy, scikit-learn,
Matplotlib and pytest; the existing `svmsv` environment includes them):

```bash
python -m pytest tests/test_shapdoe.py tests/test_add_shapdoe_baselines.py -q
python -m experiments.add_shapdoe_baselines --datasets airport voting wine cancer --jobs 40
python -m experiments.plot_shapdoe_comparison results/json/airport_inside_with_shapdoe_3repeats.json results/json/voting_inside_with_shapdoe_3repeats.json results/json/wine_inside_with_shapdoe_3repeats.json results/json/cancer_inside_with_shapdoe_3repeats.json
```

The runner writes `results/<dataset>_inside_with_shapdoe_3repeats.json`,
embedding the audited original eight-method report unchanged and appending
the two new methods to the comparison rows. Each completed
method/budget/repeat cell is saved atomically. Repeating the command resumes
the same source and seed configuration. The adjacent `.validation.json`
recomputes new RMSEs and budgets and audits the original report. The plotting
command writes PNG, PDF and CSV, retaining all original baselines.

For bounded runs, `--budget-indices 0 1 2` selects the first three points.
Feasible unselected points remain `pending` and the report is explicitly
`partial`. Plotting such reports requires `--allow-incomplete`, and the figure
is labeled as partial. Running the same dataset again without the selection
continues the remaining points from the saved cells.

The initial integration run completed all five LS points for Airport, Voting
and Wine, all feasible COA points, and the first three Cancer LS points, with
three repeats per completed point. Cancer's last two LS points are deferred
because of the measured SVM cost. To finish that report:

```bash
python -m experiments.add_shapdoe_baselines --datasets cancer --jobs 40
```

All new cells time pool startup, sampling, evaluations, aggregation and pool
shutdown. Old baseline timings are reused historical measurements and must
not be interpreted as a new jointly measured wall-clock comparison.
Airport/Voting use analytic truth; Wine/Cancer retain the original Monte
Carlo reference and its uncertainty. This integration does not claim a new
exact reference or statistically significant rankings from three repeats.

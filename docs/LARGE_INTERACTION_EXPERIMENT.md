# Large-n interaction games with exact Shapley references

## Game and reference

For binary coalition membership z, use

    v(z) = sum_i a_i z_i + sum_{(i,j) in E2} b_ij z_i z_j
           + sum_{(i,j,k) in E3} c_ijk z_i z_j z_k.

The exact reference allocates each interaction equally among its participants:

    phi_i = a_i + 1/2 sum_{e in E2: i in e} b_e
                  + 1/3 sum_{e in E3: i in e} c_e.

This follows because i is the last arriving participant in a degree-d term
with probability 1/d under a uniform permutation. It requires no enumeration
or Monte Carlo reference. The empty utility is zero. Signed coefficients
are intentional; these games need not be monotone.

Two matched game families are provided: degree 2, and degree 2+3. At each n,
they share exactly the additive and pairwise coefficients. The default graph
has 4n unique pairs and 2n unique triples, capped by the number of possible
terms for tiny test games. Edges are sampled uniformly without replacement.
The additive coefficients are N(0,1). Pair and triple coefficients have standard
deviations 1/sqrt(2|E2|/n) and 1/sqrt(3|E3|/n), respectively. This keeps each
interaction component's expected per-player Shapley variance independent of n.
This is a sparse large-n benchmark, not a dense all-pairs stress test.

The coefficient generator is independent of the design and estimator seeds.
Only the utility evaluator receives coefficients; estimators receive queried
utilities and coalition membership. The reference is used only for scoring.
NPZ archives contain every coefficient, edge, and exact reference, with hashes
recorded in JSON reports.

## Metrics and comparisons

Run n=500, 1000, 5000; use three estimator repeats for each fixed game. Interior
query budgets are multiples of n. Each method is charged another 2n+2 boundary
calls. Boundary values are reused computationally within a game; their query
count and first measured evaluation time are charged to every cell.
Timing separates design, utility evaluation, and aggregation. These are wall
seconds; utility timing with cached boundary work is an accounting estimate,
not an independently measured full end-to-end run.

INSIDE-Coalition uses the existing K=64, normalized lambda0=1/16 implementation
and strict covered ratio aggregation. OFA uses IID sampling and the official
ratio estimator's zero initialization for unobserved conditional means.
Optional `ofa_linear` uses the linear OFA estimator and is explicitly labeled
separately. Equal seeds do not imply equal within-size allocation for IID and
INSIDE. Insufficient INSIDE coverage is reported, never silently imputed.

RMSE is computed separately per estimator seed, then averaged; sample SD uses
ddof=1. Normalized RMSE divides by the RMS magnitude of the exact reference.
This avoids the instability of player-wise relative errors near zero. Three
seeds quantify estimator randomness on one fixed game, not variation across
random game instances. Replicate game seeds before making broad claims.

## Important limits

A quadratic game admits an additional specialized exact algorithm:

    phi_i = [v({i})-v(empty) + v(N)-v(N\\{i})]/2.

Thus degree-2 experiments are structured diagnostics, not evidence that a
generic estimator beats a quadratic-specific solver. Cubic terms break this
shortcut. Pairwise interactions in the original membership variables also do
not by themselves constitute a proof of the moment-error theorem.

The utility and reference need O(n+|E2|+|E3|) coefficient storage. Utility
batches are bounded to 128 rows. However, existing coalition-design routines
still materialize T-by-n arrays. Large n does not make INSIDE itself memory
linear. The runner conservatively estimates its working set and records
`resource_limit` cells above `--max-memory-gib` (default 8). This is a planning
estimate, not a measured peak or hard OS memory limit. At n=5000, 500n queries
alone require 12.5 GB for one Boolean design matrix, before working copies.
Do not launch the full grid on a small-memory machine. A streaming design and
aggregation implementation is needed for economical full-budget n=5000 runs.

INSIDE-Orbit is not included in this runner: the current full-size-coverage
implementation needs at least n(n-3) interior queries. Adding it at lower
budgets would not be a valid matched-budget comparison. Other existing
baselines can be added independently; the current experiment isolates the
INSIDE-Coalition/OFA comparison.

## Commands

From the repository root, with experiment dependencies installed:

```bash
# Fast exact-reference / evaluator scalability check (not paper accuracy evidence).
OPENBLAS_NUM_THREADS=1 python -m experiments.run_large_interaction \
  --players 500 1000 5000 --budgets 2 --methods ofa_linear --repeats 3 \
  --output results/json/large_interaction_scaling_smoke.json

# Moderate-budget real method comparison.
OPENBLAS_NUM_THREADS=1 python -m experiments.run_large_interaction \
  --players 500 --budgets 100 --repeats 3 --jobs 4 \
  --output results/json/large_interaction_n500_pilot.json

# Proposed accuracy grid; resource-limit cells are recorded explicitly.
OPENBLAS_NUM_THREADS=1 python -m experiments.run_large_interaction \
  --players 500 1000 5000 --budgets 500 1000 --repeats 3 --jobs 4
```

Each cell is checkpointed atomically. Repeating the identical command resumes
completed cells. For a changed memory limit, budget, seed or other protocol,
use a new output path. Source-game generation is deterministic.

## Completed pilot

n=500, 50,000 interior calls + 1,002 boundary calls; three estimator seeds.

| Game | Method | RMSE mean ± sample SD | Mean design seconds |
|---|---|---:|---:|
| Degree 2 | inside_coalition | 0.119540 ± 0.001549 | 71.90 |
| Degree 2 | ofa | 0.128508 ± 0.003214 | 0.31 |
| Degree 3 | inside_coalition | 0.132480 ± 0.003485 | 72.30 |
| Degree 3 | ofa | 0.141089 ± 0.003680 | 0.31 |

All 12 pilot cells completed with exact references. INSIDE coverage checks passed.
The separate n=500/1000/5000 smoke run completed 18 low-budget OFA-linear cells;
it tests evaluator/reference scalability, not full-budget INSIDE scalability.
Design costs are substantial relative to the cheap synthetic utility. No general
runtime superiority or statistical significance claim is made from this pilot.

## Full baseline RMSE comparison

`experiments.run_interaction_baselines` compares both matched n=500 games with
INSIDE-Coalition, INSIDE-Orbit, OFA, CC, S-Diff, KernelSHAP, TMC-Shapley,
ShapDoE-LS and Orthogonal. COA stays omitted, as in the main paper figure.
Budgets are 500n, 1000n and 2000n interior calls plus 2n+2 boundary calls;
three seeds give 162 method/game/budget/repeat cells. Complete LS blocks fit
at every budget. Both games share sampling seeds; OFA-family designs are
constructed once per method/budget/seed and evaluated on both games.
Each game is charged the full design time even when computation is shared.
Cross-method seeds are distinct and deterministically derived from the seed,
n, budget, repeat and method index. Coarse baseline tasks use 32 partitions.

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.run_interaction_baselines
python -m experiments.plot_interaction_baselines
```

The runner checkpoints after each paired-game task and resumes an identical
command. Methods run concurrently, so wall times from this report are not a
controlled efficiency benchmark. The plotted x coordinate is mean actual
calls, including incomplete-block savings and TMC truncation. Only the part
outside the common budget viewport is clipped, as in the existing RMSE plot.
The y coordinate is mean per-seed RMSE with one sample SD (ddof=1); exact zero
means are rejected by the logarithmic plotter rather than silently floored.

Orthogonal uses forward/reverse permutation pairs: for any quadratic game,
each pair divides every pairwise interaction equally. Its quadratic errors
are therefore floating-point residuals, not evidence of gradual statistical
convergence. The log plot retains these actual residuals even though they
make differences among the other quadratic-game methods visually smaller.
For cubic terms, reverse pairing alone does not yield an exact estimator.
TMC retains the existing relative-tolerance stopping rule; signed games do
not guarantee monotone convergence to the full-coalition utility, so its
truncation should not be interpreted as an accuracy guarantee.

## Completed full comparison

All 162 cells completed: 2 games × 9 methods × 3 budgets × 3 seeds.
All recorded RMSE values were independently recomputed from saved estimates and exact references.
No incomplete methods or repeats were excluded. The figure uses mean actual calls and mean ± sample SD.

At the largest budget (1,001,002-call cap):

| Method | Pairwise RMSE | Pairwise + third-order RMSE |
|---|---:|---:|
| INSIDE-Coalition | 0.0243716 ± 0.000887 | 0.0274944 ± 0.000579 |
| INSIDE-Orbit | 0.0253346 ± 0.00026 | 0.0280143 ± 0.000746 |
| OFA | 0.0284348 ± 0.00045 | 0.0312592 ± 0.00116 |
| CC | 0.0353293 ± 0.000884 | 0.0381084 ± 0.000596 |
| S-Diff | 0.0405353 ± 0.000799 | 0.0448562 ± 0.000798 |
| KernelSHAP | 0.0137675 ± 0.000367 | 0.0185771 ± 0.000318 |
| TMC-Shapley | 0.0111879 ± 0.000265 | 0.0152222 ± 0.000677 |
| ShapDoE-LS | 0.00632785 ± 0.000499 | 0.00943708 ± 0.000441 |
| Orthogonal | 2.84632e-16 ± 7.73e-18 | 0.00613006 ± 6.66e-05 |

Orthogonal is exact for quadratic games up to rounding; its machine-level error
is not a statistical convergence result. INSIDE is not the best method on these games.
Concurrency means recorded timing is not a controlled cross-method efficiency comparison.
PDF-only export follows the requested format; SVG/TIFF static-preflight export requirements do not apply.

## Current figure selection

At the user's request, the two-panel PDF displays only INSIDE-Coalition,
INSIDE-Orbit, OFA, CC and S-Diff. TMC-Shapley, KernelSHAP, ShapDoE-LS and
Orthogonal remain in the full experimental report and summary table. The
figure metadata records these presentation omissions; axes are rescaled to
the displayed methods. Mean ± sample SD and all three seeds are retained.

## Extended five-method budget grid

The current figure keeps INSIDE-Coalition, INSIDE-Orbit, OFA, CC and S-Diff.
Two additional budgets, 3000n and 4000n interior calls, extend the original
500n/1000n/2000n grid. At n=500 the total caps are 251,002, 501,002, 1,001,002,
1,501,002 and 2,001,002 calls. Each point still uses three seeds.

The original 90 matching cells are reused byte-for-byte in a separate report;
60 new cells are computed. The original nine-method report is preserved.
The extension initially used 16 and resumed with 32 independent fixed-size
design workers; the fixed-size
SeedSequence construction makes coalition designs invariant to worker count.
Timing remains unsuitable for controlled comparisons across worker settings.

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.run_interaction_baselines \
  --budgets 500 1000 2000 3000 4000 \
  --methods inside_greedy inside_orbit ofa cc s_diff \
  --jobs 12 --design-jobs 32 --baseline-jobs 8 \
  --reuse-source results/json/interaction_all_baselines_n500.json \
  --output results/json/interaction_selected_baselines_n500_extended.json
python -m experiments.plot_interaction_baselines \
  --source results/json/interaction_selected_baselines_n500_extended.json
```

The legend uses one row. Source hashes, reuse provenance and per-seed
estimates are retained in the extended JSON report.

The final 18 cells resumed with 12 outer workers and 8 S-Diff payload workers. S-Diff retains 32 payload partitions, identical seeds, and ordered reduction; serial/parallel estimates and counts were checked for bitwise equality. The report records execution-history boundaries, including the original 132 completed cells.

The extended grid is complete: 150 cells, comprising 90 unchanged reused cells and 60 new cells. All 150 RMSE values and budget counts were validated. The single-row legend and final PDF passed rendered checks.

## Pairwise + third-order scalability (n = 500, 1000, 2000)

The scalability run adds n=1000 and n=2000, degree 3 only. It retains
INSIDE-Coalition, INSIDE-Orbit, OFA, CC, and S-Diff, with three estimator seeds
on one fixed game per n. The interaction density remains 4n pairs and 2n
triples, with the same coefficient scaling and exact Shapley reference.

To accommodate complete Orbit coverage, compare interior query budgets
B/n² in {1, 2, 4}. Actual total caps include the 2n+2 boundary allowance:

| n | total utility-call caps |
|---|---|
| 500 (reused) | 251002, 501002, 1001002 |
| 1000 | 1002002, 2002002, 4002002 |
| 2000 | 4004002, 8004002, 16004002 |

Run/resume: `bash experiments/run_interaction_scalability.sh`.
The launcher uses `python` from the active environment; set `PYTHON_BIN` to
select another interpreter. External figure audits are optional: set
`FIGURE_QA` to the directory containing `audit_panel_alignment.py`,
`audit_pdf_text.py`, and `audit_figure_collisions.py` to enable them.
Logs: `results/logs/interaction_scalability.log`.
New checkpoints: `results/json/interaction_scalability_n{1000,2000}.json`.
Each new n has 45 cells; all 90 must finish before the plotting step.
The script then creates `results/pdf/interaction_scalability_rmse.pdf` and
its source-summary CSV and metadata JSON. These are planned outputs until
both reports have status `complete`; no partial result is presented as final.

The figure compares mean ± sample SD of per-seed RMSE in three n panels,
using a common y range and a one-row legend. The horizontal axis is actual
utility calls. This evaluates accuracy as n and query budget increase, not
linear-time computational scaling: the chosen budget itself grows as n².
Only one game realization per n is used; uncertainty describes estimator
randomness, not variation across games. Concurrent timings are not controlled
runtime comparisons.

Memory controls: at most three concurrent cells, 48 fixed-size design workers,
and 16 S-Diff payload workers. Orbit's optional dense diagnostics are disabled
without changing coalition generation; utility validation is batched. Small
regression tests check identical estimates for degree-3-only vs paired runs
and identical Orbit coalitions with/without diagnostics. The dense design API
still materializes B-by-n boolean arrays; this benchmark does not claim an
asymptotically memory-efficient implementation.

# INSIDE

This repository provides the implementation of the SIGMOD 2027 paper:

> **INSIDE: Intra-Size Coalition Design for Shapley Value Estimation**

INSIDE accelerates Shapley value estimation by designing which coalitions are
sampled within each coalition-size stratum. The implementation provides two
paper-facing variants: INSIDE-Coalition (`inside_greedy`) and INSIDE-Orbit (`inside_orbit`).

## Installation

INSIDE requires Python 3.10 or later.

```bash
git clone https://github.com/xiaokaimao/inside.git
cd inside
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[experiments,dev]"
```

## Quick start

```python
import numpy as np

from frame_ofa import FrameOFAEstimator


weights = np.array([1.0, -0.5, 2.0, 0.3])


def utility(coalition):
    return float(weights @ coalition)


result = FrameOFAEstimator(
    num_players=len(weights),
    num_samples=40,
    mode="inside_greedy",  # Use "inside_orbit" for the orbit variant.
    seed=0,
).estimate(utility)

print(result.values)
print(result.utility_evaluations)
```

## Introductory schematic

```bash
python -m experiments.plot_inside_intro
```

This exports only `results/pdf/inside_intro_figure.pdf`, at 180 × 58 mm,
using Matplotlib directly (no SVG or CairoSVG intermediate). The two-panel
schematic uses exact n=4, s=2 coalition geometry in panel (a): complementary
coalitions have opposite directions under a common orthographic projection.
Panel (b) is conceptual: six independent random directions on the continuous
unit sphere (seed 0, normalized Gaussian vectors, no selected batches) are
compared with the balanced directions of the existing six-coalition design.
The IID panel is not a literal sample from the discrete four-player slice.
The exact balance of this toy design is not a general finite-budget guarantee.
Dashed arrows indicate rear-facing directions. Unsampled markers and repeat
counts are omitted. The spheres restore the supplied original diagram's soft
upper-left radial gradient. A narrower, slightly lowered horizontal ellipse
uses a uniform light dashed line, without specular highlights or ground shadows.
Optional `--metadata PATH` records the coordinates and moment diagnostics.
For `--audit-alignment`, add the figure audit helper directory to `PYTHONPATH`;
this writes an alignment JSON report and no extra figure format.

## Run the synthetic experiment

```bash
python -m experiments.benchmark_synthetic \
  --players 8 \
  --samples 40 \
  --repeats 100
```

## Run the tests

```bash
OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q
```

## ShapDoE baseline (Yang et al., JASA 2024)

The author's MIT-licensed [ShapDoE R package](https://CRAN.R-project.org/package=ShapDoE)
is available through the Python API `estimate_shapdoe(..., method="ls")`
or `method="coa"`, using the same `GameEvaluator` interface as the other
baselines. The budget is a total utility-call cap; only complete LS/COA
designs are used. See [the source and protocol audit](docs/SHAPDOE_BASELINE_AUDIT.md).

With the experiment dependencies installed and the existing source reports:

```bash
python -m experiments.add_shapdoe_baselines --datasets airport voting wine cancer --jobs 40
python -m experiments.plot_shapdoe_comparison results/json/airport_inside_with_shapdoe_3repeats.json
```

The runner resumes completed cells and writes separate comparison reports.
COA points that cannot fit a complete design are explicitly marked unavailable.

## Orthogonal baseline (Mitchell et al., JMLR 2022)

The first author's [shap_sampling repository](https://github.com/RAMitchell/shap_sampling)
provides the Orthogonal implementation. `estimate_orthogonal_shapley` integrates
its spherical sampling method with the existing `GameEvaluator` interface.
The NumPy implementation is checked against executed author-code fixtures.
Like the author sampler, it permits a partial final orthogonal basis while
retaining forward/reverse permutation pairs. The minimum budget is `2*n`
utility calls. See [the implementation audit](docs/ORTHOGONAL_BASELINE_AUDIT.md).

```bash
python -m experiments.add_orthogonal_baseline --datasets airport voting wine cancer --jobs 40
python -m experiments.plot_orthogonal_comparison results/json/airport_inside_with_shapdoe_orthogonal_3repeats.json
```

This adds Orthogonal to separate reports that retain the original baselines
and ShapDoE results. Use `--budget-indices 0` for the first budget; partial
reports require `--allow-incomplete` when plotting.

The four-panel presentation includes ShapDoE-LS and Orthogonal, omits
ShapDoE-COA, and clips the left part of TMC-Shapley at the existing budget
viewport. RMSE uses a shared legend above the panels and shows the mean
± one sample standard deviation (`ddof=1`) across three estimator seeds.
AER and MER show mean curves without legends or uncertainty shading.
Metrics are computed separately for each seed before averaging. Raw
standard deviations remain in the metadata. Panel labels sit above the axes,
and gridlines span the complete plot area. No footnotes are drawn.
Choose `--metric rmse`, `--metric aer`, or `--metric mer`:

```bash
for figure_metric in rmse aer mer; do
  python -m experiments.plot_inside_four_panel \
    --voting results/json/voting_inside_with_shapdoe_orthogonal_3repeats.json \
    --airport results/json/airport_inside_with_shapdoe_orthogonal_3repeats.json \
    --wine results/json/wine_inside_with_shapdoe_orthogonal_3repeats.json \
    --cancer results/json/cancer_inside_with_shapdoe_orthogonal_3repeats.json \
    --metric "$figure_metric" \
    --output "results/pdf/inside_comparison_four_panel_ls_orthogonal_${figure_metric}.pdf"
done
```

Airport Figure 4 retains the three-panel moment diagnostic. D1 and D2 are
RMS discrepancies with equal weights over sizes 2–98. Panels (a,b) show
mean ± sample standard deviation across three estimator seeds. Panel (c)
shows all 45 run-level points, a pooled log–log OLS trend, and descriptive
Spearman correlation. See `docs/AIRPORT_FIGURE4_MOMENT_DISCREPANCIES.tex`
for the definitions, bounded interpretation, and caption.

```bash
python -m experiments.plot_airport_inside_ablations \
  results/json/airport_inside_ablations_k64_lambda1over16.json
```

This also exports the component and orbit ablations, whose RMSE uncertainty
continues to use the recorded bootstrap intervals. To run the Figure 4
alignment audit, add the nature-figure skill's `scripts` directory to
`PYTHONPATH` and pass `--audit-alignment`.

The lightweight Airport parameter sensitivity experiment varies the normalized
lambda at fixed K=64 and K at fixed lambda=1/16, with 3 paired repeats and
50,202 utility calls. Run `python -m experiments.run_airport_sensitivity`
with `OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1`, then
`python -m experiments.plot_airport_sensitivity`. See
`docs/AIRPORT_SENSITIVITY_ANALYSIS.md` for the protocol and observed results,
and `docs/AIRPORT_SENSITIVITY_APPENDIX.tex` for ready-to-integrate Appendix text.

To extend the same sensitivity grid to Wine and Breast Cancer, run:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.run_svm_sensitivity
python -m experiments.plot_svm_sensitivity
```

Both use 500n interior calls and the original fixed split/SVM/Monte Carlo
reference. Each dataset has 33 designs. See
`docs/SVM_SENSITIVITY_ANALYSIS.md` for pairing, timing, call-accounting, and
reference-uncertainty details. The source benchmark reports are unchanged.

For four standalone sensitivity plots per dataset, without panel tags and
with grids and in-axis legends:

```bash
python -m experiments.plot_sensitivity_separate --datasets wine cancer
```

Each dataset produces four PDF figures (`lambda_rmse`,
`lambda_moments`, `k_rmse`, `k_design_time`) in `results/pdf/`, with metadata in `results/json/`.

Result files are organized by format; see [the results index](results/README.md).

## Repository layout

- `frame_ofa/`: estimators, coalition designs, moment geometry, and baselines.
- `experiments/`: experiment runners (`run_*`), plotters (`plot_*`), validators,
  and historical diagnostic workflows. Run modules from the repository root.
- `tests/`: estimator, protocol, baseline parity, and plotting checks.
- `docs/`: dataset protocols, baseline audits, and manuscript-ready snippets.
- `third_party/`: licensed reference material, fixtures, and provenance;
  see `UPSTREAM.md` for attribution.
- `results/`: retained experiment reports and figures grouped by format.

Current workflows are documented above; historical diagnostics remain available
for reproducing prior experiments. Plotting recorded reports does not rerun
SVM training. The four-panel and sensitivity workflows support PDF-only output.

## Large-n synthetic interaction experiment

`frame_ofa.interaction_game` supplies sparse pairwise and cubic games with
closed-form Shapley values at n=500, 1000, and 5000. The checkpointed runner
is `python -m experiments.run_large_interaction`. See
[the protocol and runnable commands](docs/LARGE_INTERACTION_EXPERIMENT.md)
for exact-reference validation, paired game construction, pilot results,
and the current dense-design memory limit at large budgets.

For all nine methods used in the main RMSE figure on both n=500 interaction
games, run `python -m experiments.run_interaction_baselines`, followed by
`python -m experiments.plot_interaction_baselines`. The two-panel PDF uses
three budgets, three estimator seeds, and the same method colors and labels.

The current interaction-game figure shows five selected methods and five
budgets (up to 2,001,002 calls), with a single-row legend. See the
[extended-grid command](docs/LARGE_INTERACTION_EXPERIMENT.md#extended-five-method-budget-grid)
for reproducing the extension while reusing the original runs.

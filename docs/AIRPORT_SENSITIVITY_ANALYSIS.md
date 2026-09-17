# Airport parameter sensitivity protocol

This is a descriptive, one-at-a-time sensitivity analysis for the Appendix,
not a new hyperparameter-selection pass. It uses Airport's 100 players and
exact Shapley values. The budget is fixed before inspecting the new results:
50,000 interior utility evaluations plus 202 boundary evaluations per design.
This is the lowest budget in the existing Airport experiment.

- Lambda sweep: 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1, holding K=64.
- K sweep: 8, 16, 32, 64, 128, holding lambda0=1/16.
- Three estimator seeds, with base seed 20260915. The common default appears
  once, giving 11 configurations and 33 completed designs.
- Within each repeat, all settings share a size schedule and player-relabel
  seed. Different K values do not necessarily use nested candidate pools.
- All estimates use strict OFA conditional-mean ratios with complete
  inclusion/exclusion coverage. D1 and D2 use equal weights over sizes 2–98.
- Each curve is the arithmetic mean of per-seed metrics. Shading is ± one
  sample standard deviation (ddof=1), not a confidence interval.

## Parameter convention

The implementation's `mean_balance` parameter in normalized mode is lambda0.
For n=100, the effective coefficient multiplying the first-moment score is
lambda0 × 98/99. The horizontal axis uses the normalized input convention,
as in the existing main experiment. If the manuscript instead defines lambda
as the literal coefficient of the unnormalized score, state this conversion.

## Design-time measurement

Configurations run sequentially in a deterministic randomized order per seed.
Every design uses 16 worker processes (`fork`) with one BLAS/OpenMP thread.
The timer encloses the complete design-construction call, including worker
startup, but excludes moment diagnostics, utility evaluation, and estimation.
These are wall times on the current shared host, not hardware-independent
complexity estimates. Environment information and individual measurements
are saved in the JSON. No historical timing measurements are mixed into this
analysis.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.run_airport_sensitivity
python -m experiments.plot_airport_sensitivity
```

The runner checkpoints after every completed design and validates all
completed cells on resume. To audit final panel alignment, add nature-figure's
`scripts` directory to PYTHONPATH and pass `--audit-alignment` to the plotter.
The raw report is `results/json/airport_inside_sensitivity.json`; figures use the
same basename with PNG, PDF, and SVG extensions. Black outer rings mark the
pre-existing defaults. The results apply to this dataset and fixed budget;
they do not establish robustness over every budget or dataset.

## Observed results

At lambda0 = 1/32, 1/16, and 1/8, mean RMSE is respectively 0.005145, 0.005457, and 0.005494. The maximum is 6.8% above the minimum in this neighborhood; 1/16 is not an isolated optimum. This is descriptive stability at one budget, not a statistical equivalence claim. Across the full sweep, mean RMSE ranges from 0.005096 to 0.006151.

From lambda0=1/64 to 1, mean D1 falls from 0.02566 to 0.006016 while mean D2 rises from 0.02776 to 0.03458. This supports the intended first-/second-moment tradeoff, without assuming a monotone RMSE response.

K=128 reduces mean RMSE by 13.8% relative to K=64 (0.004703 versus 0.005457), while mean design time increases by 77.9% (10.72 versus 6.03 seconds). K=64 is not an observed accuracy-saturation point. RMSE is nonmonotone over K=8–64, so these runs do not show that K=64 dominates smaller candidate pools. Both D1 and D2 improve as K increases, but improved moments alone do not guarantee lower RMSE for every finite run.

| lambda0 | K | RMSE mean ± std | D1 mean | D2 mean | Design seconds mean ± std |
|---|---:|---:|---:|---:|---:|
| 0.015625 | 64 | 0.005347 ± 0.000868 | 0.025658 | 0.027764 | 6.020 ± 0.013 |
| 0.03125 | 64 | 0.005145 ± 0.000402 | 0.020521 | 0.027910 | 6.018 ± 0.020 |
| 0.0625 | 64 | 0.005457 ± 0.000068 | 0.015968 | 0.028226 | 6.026 ± 0.015 |
| 0.125 | 64 | 0.005494 ± 0.000250 | 0.012104 | 0.028789 | 6.162 ± 0.234 |
| 0.25 | 64 | 0.005213 ± 0.000520 | 0.009348 | 0.029902 | 6.031 ± 0.006 |
| 0.5 | 64 | 0.005096 ± 0.000939 | 0.007283 | 0.031823 | 6.000 ± 0.011 |
| 1 | 64 | 0.006151 ± 0.000864 | 0.006016 | 0.034576 | 6.057 ± 0.018 |
| 0.0625 | 8 | 0.005186 ± 0.000296 | 0.021739 | 0.034390 | 1.632 ± 0.019 |
| 0.0625 | 16 | 0.005439 ± 0.000118 | 0.018957 | 0.031961 | 2.240 ± 0.006 |
| 0.0625 | 32 | 0.005162 ± 0.000255 | 0.017286 | 0.029931 | 3.539 ± 0.039 |
| 0.0625 | 128 | 0.004703 ± 0.000155 | 0.014766 | 0.026719 | 10.719 ± 0.129 |

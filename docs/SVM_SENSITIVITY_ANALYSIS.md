# Wine and Breast Cancer parameter sensitivity

This extends the Airport one-at-a-time sensitivity protocol to the two SVM
data-valuation games. Each dataset uses 11 unique parameter settings and
three paired estimator seeds (33 designs); lambda0=1/16, K=64 is shared by
both sweeps and is evaluated only once per repeat.

| Dataset | Players | Held-out test points | Interior queries | Equivalent total calls |
|---|---:|---:|---:|---:|
| Wine | 142 | 36 | 71,000 | 71,286 |
| Breast Cancer | 455 | 114 | 227,500 | 228,412 |

The lowest existing budget, 500n interior queries, is fixed before observing
results. Lambda0 varies over 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, and 1 at K=64;
K varies over 8, 16, 32, 64, and 128 at lambda0=1/16. The effective first-moment
score coefficient is lambda0*(n-2)/(n-1); plotting labels follow the normalized
input convention used by the existing experiments.

## Data, utility and reference

The 80/20 stratified split (seed 2024) and full-training-partition standardizer
are reconstructed and verified against the existing source report. Utilities
are fixed-test accuracy from SVC(C=1, kernel='rbf', gamma='scale'). Empty
coalitions use the best constant-label test accuracy; single-class coalitions
predict their sole class. The recorded boundary utilities are reused and
2n+2 calls are counted in the equivalent per-estimate budget. New physical
SVM evaluations per configuration are the interior queries only.

The saved Monte Carlo Shapley reference is fixed across all configurations.
RMSE therefore measures discrepancy from that reference, not an exact truth.
Its standard errors and all reference metadata are copied into each output.
The plotted sample SD reflects estimator-seed variation only and does not
include reference uncertainty or data-split variability.

## Pairing and timing

Base seed 20260915 generates three independent repeat streams. Within a
repeat, the size schedule and final player relabeling are shared; different
candidate-pool sizes do not imply nested candidate sets. Every size 2 through
n-2 must be present, and strict conditional-mean inclusion/exclusion coverage
is checked. D1 and D2 are equal-size RMS moment discrepancies.

Configurations run sequentially in a deterministic randomized order. Design
construction uses 16 workers; SVM utility evaluation uses 120 workers, with
one BLAS/OpenMP thread each. Design timing encloses construction including
worker overhead, but excludes utilities, geometry diagnostics, and estimation.
Wine and Breast Cancer are run sequentially to avoid interference from these
two jobs. Shared-host wall times remain hardware- and load-dependent.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m experiments.run_svm_sensitivity
python -m experiments.plot_svm_sensitivity
```

Add nature-figure's `scripts` directory to PYTHONPATH and pass
`--audit-alignment` for the final plot alignment check. Completed cells are
checkpointed in `results/{wine,cancer}_inside_sensitivity.json`. Resuming
validates the configuration, source fingerprint, and all completed cells.
Figures and CSV/metadata exports use the corresponding dataset basename.
Raw results and findings are separate from the original benchmark reports.

## Completed results

### Wine

At lambda0=1/32, 1/16, 1/8, mean RMSE is 0.0006042, 0.0006438, 0.0006110; the maximum is 6.6% above the minimum. K=128 versus K=64 changes mean RMSE from 0.0006438 to 0.0006289 (2.3% reduction) and mean design time from 9.47s to 16.66s (76.1% increase).

| lambda0 | K | RMSE mean ± std | D1 mean | D2 mean | Design seconds mean ± std |
|---|---:|---:|---:|---:|---:|
| 0.015625 | 64 | 0.0006224 ± 0.0000086 | 0.028224 | 0.033175 | 9.49 ± 0.02 |
| 0.03125 | 64 | 0.0006042 ± 0.0000191 | 0.022908 | 0.033336 | 9.54 ± 0.02 |
| 0.0625 | 64 | 0.0006438 ± 0.0000506 | 0.017707 | 0.033643 | 9.47 ± 0.11 |
| 0.125 | 64 | 0.0006110 ± 0.0000402 | 0.013468 | 0.034267 | 9.62 ± 0.34 |
| 0.25 | 64 | 0.0006421 ± 0.0000408 | 0.010351 | 0.035359 | 9.50 ± 0.08 |
| 0.5 | 64 | 0.0006563 ± 0.0000532 | 0.008215 | 0.037152 | 9.70 ± 0.17 |
| 1 | 64 | 0.0006057 ± 0.0000322 | 0.006877 | 0.039498 | 9.51 ± 0.03 |
| 0.0625 | 8 | 0.0006134 ± 0.0000068 | 0.023235 | 0.038321 | 2.53 ± 0.04 |
| 0.0625 | 16 | 0.0006026 ± 0.0000313 | 0.020814 | 0.036529 | 3.57 ± 0.02 |
| 0.0625 | 32 | 0.0005964 ± 0.0000169 | 0.019003 | 0.034995 | 6.01 ± 0.59 |
| 0.0625 | 128 | 0.0006289 ± 0.0000305 | 0.016730 | 0.032451 | 16.66 ± 0.11 |

### Breast Cancer

At lambda0=1/32, 1/16, 1/8, mean RMSE is 0.0002351, 0.0002367, 0.0002402; the maximum is 2.1% above the minimum. K=128 versus K=64 changes mean RMSE from 0.0002367 to 0.0002356 (0.5% reduction) and mean design time from 74.60s to 130.76s (75.3% increase).

| lambda0 | K | RMSE mean ± std | D1 mean | D2 mean | Design seconds mean ± std |
|---|---:|---:|---:|---:|---:|
| 0.015625 | 64 | 0.0002415 ± 0.0000077 | 0.033290 | 0.043577 | 73.91 ± 0.36 |
| 0.03125 | 64 | 0.0002351 ± 0.0000064 | 0.027358 | 0.043721 | 74.51 ± 0.38 |
| 0.0625 | 64 | 0.0002367 ± 0.0000055 | 0.021538 | 0.044022 | 74.60 ± 0.79 |
| 0.125 | 64 | 0.0002402 ± 0.0000062 | 0.016880 | 0.044548 | 73.85 ± 0.33 |
| 0.25 | 64 | 0.0002466 ± 0.0000039 | 0.013750 | 0.045312 | 74.01 ± 0.32 |
| 0.5 | 64 | 0.0002517 ± 0.0000085 | 0.011990 | 0.046172 | 73.88 ± 0.33 |
| 1 | 64 | 0.0002347 ± 0.0000046 | 0.011224 | 0.046878 | 73.89 ± 0.44 |
| 0.0625 | 8 | 0.0002401 ± 0.0000100 | 0.027438 | 0.045519 | 24.84 ± 0.05 |
| 0.0625 | 16 | 0.0002418 ± 0.0000066 | 0.024844 | 0.044957 | 31.81 ± 0.18 |
| 0.0625 | 32 | 0.0002417 ± 0.0000090 | 0.022961 | 0.044462 | 46.29 ± 0.52 |
| 0.0625 | 128 | 0.0002356 ± 0.0000079 | 0.020402 | 0.043622 | 130.76 ± 0.51 |

Across both datasets, increasing lambda strengthens first-moment balance while weakening second-moment balance. The neighborhood 1/32–1/8 does not exhibit an isolated optimum at 1/16. K=128 provides only a small observed reduction in mean RMSE versus K=64 (2.3% for Wine, 0.5% for Cancer), at approximately 75–76% additional design time. Given three repeats and a fixed approximate reference, these differences are descriptive, not evidence of statistically reliable superiority or equivalence. Smaller pools are also competitive; the sweep does not establish K=64 as uniquely optimal.

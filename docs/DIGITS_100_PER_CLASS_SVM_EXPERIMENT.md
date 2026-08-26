# Digits 100-per-class full-training-set SVM data valuation

## Protocol

- Dataset: `sklearn.datasets.load_digits`, 1,797 observations, 64 flattened
  8-by-8 image pixels, and 10 classes.
- Balanced subset: before splitting, independently sample 100 observations
  without replacement from each class using NumPy RNG seed 2024. The resulting
  1,000 original row indices are stored in the report.
- Split: stratified 80/20 split of that subset with seed 2024.
- Players: all 800 training observations, exactly 80 from each class.
- Fixed test set: 200 observations, exactly 20 from each class.
- Feature filtering: identify zero-variance pixels from the training partition
  only. For this fixed split, pixels `[0, 32, 39, 56]` are removed, leaving 60
  features. No test row participates in this decision.
- Preprocessing: fit `StandardScaler` on the retained training features only,
  then apply the fitted transform to training and test partitions.
- Utility: fixed-test accuracy after fitting
  `sklearn.svm.SVC(C=1, kernel="rbf", gamma="scale")` to a coalition.
- Empty coalition: best constant-label test accuracy, \(20/200=0.1\).
- Single-class coalition: predict the coalition's sole observed class.
- Full-coalition utility for the fixed split: \(195/200=0.975\).

The `--samples-per-class` value, subset seed, selected original indices,
training-derived feature mask, and scaler parameters are recorded in the
experiment metadata. The value also enters the ground-truth cache key and the
default artifact stem, preventing accidental reuse across different subsets.

## Planned high-budget reference

The matching Wine and Cancer experiments use 800,000 independent antithetic
permutation pairs. With \(n=800\), the conceptual path budget is

\[
2+2(800{,}000)(799)=1{,}278{,}400{,}002
\]

utility calls. Exact reuse of singleton and leave-one-out boundary utilities
reduces physical work to

\[
2(800{,}000)(797)+1{,}602=1{,}275{,}201{,}602.
\]

Ground-truth precision and runtime will be filled from the completed report;
the call count alone is not a precision certificate.

## Method-budget constraint

The requested first inner budget \(n\times500=400{,}000\) is not valid for the
current complete-orbit ratio estimator. It requires at least one 800-row orbit
for each of the \(n-3=797\) internal coalition-size strata:

\[
n(n-3)=800\times797=637{,}600.
\]

To retain all four estimators at every plotted point, the smallest compatible
integer multiplier is 797. The executable command below uses the round value
800 and then keeps the requested 1,000, 2,000, 5,000, and 10,000 multipliers:

\[
[640{,}000,800{,}000,1{,}600{,}000,4{,}000{,}000,8{,}000{,}000]
\]

inner calls, or

\[
[641{,}602,801{,}602,1{,}601{,}602,4{,}001{,}602,8{,}001{,}602]
\]

total calls after the common \(2n+2=1{,}602\) boundary evaluations. Using 500
would require changing or omitting the orbit-ratio method at the first point,
so it would not be the same four-method comparison.

## Commands

Run in the `svmsv` conda environment with 128 worker processes and one
BLAS/OpenMP thread per worker:

```bash
conda run -n svmsv python -m \
  experiments.full_train_data_valuation \
  --dataset digits \
  --samples-per-class 100 \
  --gt-pairs 800000 \
  --budget-multipliers 800 1000 2000 5000 10000 \
  --jobs 128 \
  --coupled-design orbit_coupled \
  --output \
    results/digits_100_per_class_full_train_rbf_svm_frame_ofa_640k_8m.json

conda run -n svmsv python -m \
  experiments.plot_full_train_results \
  --input \
    results/digits_100_per_class_full_train_rbf_svm_frame_ofa_640k_8m.json \
  --output \
    results/digits_100_per_class_full_train_rbf_svm_rmse_640k_8m.png
```

Expected cache names when explicit paths are not supplied:

- `results/digits_100_per_class_full_train_rbf_svm_gt.npz`
- `results/digits_100_per_class_full_train_rbf_svm_gt.partial.npz`

## Results

Pending completion of the formal run. The final report must include reference
uncertainty, paired repeat intervals, efficiency checks, runtime, exact call
accounting, and a log-log RMSE plot. Lower RMSE is better.

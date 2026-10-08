# Results directory

| Directory | Contents |
|---|---|
| `pdf/` | 74 figure PDFs (including historical comparisons) |
| `json/` | Raw experiment reports, plot metadata, validation reports, and provenance |
| `csv/` | 17 tabular exports |
| `npz/` | 8 reference-value caches and recovery snapshots |
| `html/` | Interactive diagnostic report |

Current figures:
- [INSIDE introductory schematic](pdf/inside_intro_figure.pdf) — exact coalition map and conceptual random/balanced directions; PDF-only export.
- [Wine sensitivity: lambda / RMSE](pdf/wine_sensitivity_lambda_rmse.pdf)
- [Wine sensitivity: lambda / moments](pdf/wine_sensitivity_lambda_moments.pdf)
- [Wine sensitivity: K / RMSE](pdf/wine_sensitivity_k_rmse.pdf)
- [Wine sensitivity: K / design time](pdf/wine_sensitivity_k_design_time.pdf)
- [Cancer sensitivity: lambda / RMSE](pdf/cancer_sensitivity_lambda_rmse.pdf)
- [Cancer sensitivity: lambda / moments](pdf/cancer_sensitivity_lambda_moments.pdf)
- [Cancer sensitivity: K / RMSE](pdf/cancer_sensitivity_k_rmse.pdf)
- [Cancer sensitivity: K / design time](pdf/cancer_sensitivity_k_design_time.pdf)
- [Four-benchmark RMSE comparison](pdf/inside_comparison_four_panel_ls_orthogonal_rmse.pdf)
- [Airport moment diagnostics](pdf/airport_inside_ablations_k64_lambda1over16_geometry.pdf)

Experiment data and caches have been moved without changing their bytes.
Historical paths embedded inside JSON reports are retained as provenance;
[the migration manifest](json/organization_manifest.json) maps old paths to
current paths and records checksums and deletions. Restart/recovery files
remain under `json/checkpoints/` and the `json/*.json.checkpoints/` directories.
Partial NPZ snapshots are retained because their recovery value is not assumed
obsolete. Removed files are regenerable plot QA overlays/reports, completed-run
logs, and redundant ZIP bundles. No experimental records were discarded based
only on age or filename.

Large-n exact-reference interaction experiment:
- [Protocol and limitations](../docs/LARGE_INTERACTION_EXPERIMENT.md)
- [n=500,1000,5000 evaluator smoke run](json/large_interaction_scaling_smoke.json)
- [n=500 INSIDE-Coalition vs OFA pilot](json/large_interaction_n500_pilot.json)
- Coefficient archives and exact references: `npz/large_interaction_*.npz`.

- [Two-game selected-method RMSE PDF](pdf/interaction_all_baselines_n500_rmse.pdf)
- [Full-baseline raw results](json/interaction_all_baselines_n500.json)
- [Full-baseline mean/std table](csv/interaction_all_baselines_n500_rmse.csv)

- [Current five-budget comparison data](json/interaction_selected_baselines_n500_extended.json)
- [Current five-budget mean/std table](csv/interaction_selected_baselines_n500_extended_rmse.csv)

### Pairwise + third-order scalability

`json/interaction_scalability_n1000.json` and
`json/interaction_scalability_n2000.json` are resumable experiment checkpoints.
Read the `status` field before treating either as a complete result. The log is
`logs/interaction_scalability.log`. After both finish, the launcher generates
`pdf/interaction_scalability_rmse.pdf` and `csv/interaction_scalability_rmse.csv`,
including reused n=500 results at the same interior-budget/n² levels.

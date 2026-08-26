"""Build the canonical portable-report artifact for the voting diagnosis."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


REPORT_TITLE = "Why Batch-balanced Frame-OFA Beats Greedy Frame-OFA"


def _source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    generated_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_source = {"id": source_id, "label": label, "path": path}
    if Path(path).suffix == ".json":
        replay_query = f"SELECT * FROM read_json_auto('{path}')"
    else:
        replay_query = f"SELECT content FROM read_text('{path}')"
    canonical_source = {
        "id": source_id,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "sql": replay_query,
            "description": description,
            "executed_at": generated_at,
            "tables_used": [path],
        },
    }
    return manifest_source, canonical_source


def build_artifact(
    experiment: dict[str, Any],
    diagnostics: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    curve_rows: list[dict[str, Any]] = []
    method_names = {
        "iid_linear": "Random OFA",
        "row_greedy": "Greedy Frame-OFA",
        "orbit_greedy": "Batch-balanced Frame-OFA",
    }
    ordered_budgets = sorted(
        experiment["results_by_inner_budget"].values(),
        key=lambda row: row["total_utility_calls"],
    )
    for budget_index, row in enumerate(ordered_budgets):
        for method, visible_name in method_names.items():
            summary = row["methods"][method]
            curve_rows.append(
                {
                    "budget_index": budget_index + 1,
                    "total_calls": row["total_utility_calls"],
                    "inner_calls": row["inner_utility_calls"],
                    "method": visible_name,
                    "rmse": summary["aggregate_rmse"],
                    "rmse_change_vs_random": summary[
                        "rmse_change_vs_iid"
                    ],
                    "ci_lower": summary[
                        "aggregate_rmse_bootstrap_95"
                    ][0],
                    "ci_upper": summary[
                        "aggregate_rmse_bootstrap_95"
                    ][1],
                    "frame_frobenius": summary[
                        "mean_frame_frobenius_discrepancy"
                    ],
                    "design_seconds": summary["mean_design_seconds"],
                }
            )

    sensitivity_rows = []
    for row in diagnostics["sensitivity"]:
        name = row["configuration"]
        if row["family"] == "batch":
            visible_name = "Batch-balanced, K=64"
        else:
            visible_name = (
                f"Greedy, K={row['candidate_pool']}, "
                f"λ={row['mean_balance']:g}"
            )
        sensitivity_rows.append(
            {
                "configuration": visible_name,
                "family": (
                    "Batch-balanced"
                    if row["family"] == "batch"
                    else "Greedy"
                ),
                "candidate_pool": row["candidate_pool"],
                "mean_balance": row["mean_balance"],
                "rmse": row["aggregate_rmse"],
                "rmse_ci_lower": row["aggregate_rmse_bootstrap_95"][0],
                "rmse_ci_upper": row["aggregate_rmse_bootstrap_95"][1],
                "design_seconds": row["mean_design_seconds"],
                "frame_frobenius": row["mean_frobenius_discrepancy"],
                "slice_mean_direction_rms": row[
                    "mean_slice_direction_rms"
                ],
                "internal_key": name,
            }
        )

    extreme_size_rows = []
    for row in sorted(
        diagnostics["size_profile"],
        key=lambda item: abs(item["size_only_residual"]),
        reverse=True,
    )[:10]:
        extreme_size_rows.append(
            row
            | {
                "absolute_residual": abs(row["size_only_residual"]),
            }
        )

    hyperparameter_rows = [
        {
            "parameter": "candidate_pool (K)",
            "applies_to": "Both",
            "experiment_value": "64",
            "effect": (
                "Candidates scored at each greedy decision; time grows "
                "roughly linearly with K."
            ),
        },
        {
            "parameter": "mean_balance (λ)",
            "applies_to": "Greedy only",
            "experiment_value": "0.1",
            "effect": (
                "Soft penalty for fixed-size mean-direction imbalance; "
                "zero disables it."
            ),
        },
        {
            "parameter": "inner-call budget (T)",
            "applies_to": "Both",
            "experiment_value": "51–12,750",
            "effect": (
                "Batch-balanced requires T to be divisible by the 51 "
                "players; Greedy accepts arbitrary T."
            ),
        },
        {
            "parameter": "seed",
            "applies_to": "Both",
            "experiment_value": "20260824",
            "effect": (
                "Controls systematic size offset, candidate coalitions, "
                "tie breaks, and final relabeling."
            ),
        },
        {
            "parameter": "baseline",
            "applies_to": "Estimator",
            "experiment_value": "linear: s/51",
            "effect": (
                "A zero-attribution control variate; it is unbiased but "
                "fits this threshold game's size curve poorly."
            ),
        },
    ]

    mechanism_rows = [
        {
            "property": "Decision unit",
            "greedy": "One coalition",
            "batch_balanced": "One balanced batch of 51 coalitions",
        },
        {
            "property": "First-moment balance",
            "greedy": "Soft penalty λ",
            "batch_balanced": "Exact: direction sum is zero in every batch",
        },
        {
            "property": "Second-moment objective",
            "greedy": "Dense local frame-potential increment",
            "batch_balanced": "Orbit frame-potential increment via FFT",
        },
        {
            "property": "Design complexity",
            "greedy": "O(T K n²)",
            "batch_balanced": "Approximately O(T K log n)",
        },
        {
            "property": "Budget restriction",
            "greedy": "Any positive T",
            "batch_balanced": "T must be a multiple of n",
        },
    ]

    result_path = (
        "results/us_electoral_college_2024_frame_ofa_greedy.json"
    )
    diagnostic_path = (
        "results/us_electoral_college_frame_design_diagnostics.json"
    )
    design_path = "frame_ofa/design.py"
    estimator_path = "frame_ofa/estimator.py"
    source_specs = [
        _source(
            "voting_results",
            "30-repeat Electoral College experiment",
            result_path,
            "Loads RMSE, frame discrepancy, timing, and call-budget results.",
            generated_at,
        ),
        _source(
            "design_diagnostics",
            "Frame-design diagnostic sweep",
            diagnostic_path,
            "Loads exact size profile and the K/lambda sensitivity sweep.",
            generated_at,
        ),
        _source(
            "design_code",
            "Frame-OFA coalition-design implementation",
            design_path,
            "Reads the row-greedy and cyclic-batch construction code.",
            generated_at,
        ),
        _source(
            "estimator_code",
            "Frame-OFA linear estimator implementation",
            estimator_path,
            "Reads the baseline and coupled-estimator implementation.",
            generated_at,
        ),
    ]
    manifest_sources = [item[0] for item in source_specs]
    canonical_sources = [item[1] for item in source_specs]

    weighted_rms = diagnostics["size_residual_summary"]["ofa_weighted_rms"]
    main_low = ordered_budgets[0]["methods"]
    main_high = ordered_budgets[-1]["methods"]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": REPORT_TITLE,
            "description": (
                "Technical diagnosis of the statistical and computational "
                "gap between coalition-wise and batch-balanced Frame-OFA."
            ),
            "generatedAt": generated_at,
            "cards": [],
            "charts": [
                {
                    "id": "rmse_curve",
                    "title": "Shapley RMSE by utility-call budget",
                    "subtitle": (
                        "U.S. Electoral College voting game, 30 independent "
                        "designs per budget; lower is better."
                    ),
                    "type": "line",
                    "intent": "trend",
                    "question": (
                        "How does estimation error change with utility-call "
                        "budget for the three coalition designs?"
                    ),
                    "rationale": (
                        "A multi-series line chart shows error decay and the "
                        "persistent gap across eight ordered budgets."
                    ),
                    "dataset": "rmse_curve",
                    "sourceId": "voting_results",
                    "encodings": {
                        "x": {
                            "field": "total_calls",
                            "type": "quantitative",
                            "label": "Total utility calls",
                        },
                        "y": {
                            "field": "rmse",
                            "type": "quantitative",
                            "label": "RMSE",
                            "format": "number",
                        },
                        "color": {
                            "field": "method",
                            "type": "nominal",
                            "label": "Sampling method",
                        },
                        "tooltip": [
                            {
                                "field": "rmse_change_vs_random",
                                "type": "quantitative",
                                "label": "Change vs Random OFA",
                                "format": "percent",
                            },
                            {
                                "field": "frame_frobenius",
                                "type": "quantitative",
                                "label": "Frame discrepancy",
                                "format": "number",
                            },
                            {
                                "field": "design_seconds",
                                "type": "quantitative",
                                "label": "Mean design seconds",
                                "format": "number",
                            },
                        ],
                    },
                }
            ],
            "tables": [
                {
                    "id": "mechanism_table",
                    "title": "What the two Frame-OFA designs optimize",
                    "subtitle": (
                        "Batch-balanced is a stronger structured design, not "
                        "a lower-quality approximation of row greedy."
                    ),
                    "dataset": "mechanism_comparison",
                    "sourceId": "design_code",
                    "defaultSort": {"field": "property", "direction": "asc"},
                    "columns": [
                        {"field": "property", "label": "Property", "type": "text"},
                        {"field": "greedy", "label": "Greedy Frame-OFA", "type": "text"},
                        {
                            "field": "batch_balanced",
                            "label": "Batch-balanced Frame-OFA",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "size_residual_table",
                    "title": "Largest fixed-size baseline residuals",
                    "subtitle": (
                        "Exact coalition-size winning probabilities versus "
                        "the linear s/51 baseline."
                    ),
                    "dataset": "size_residual_extremes",
                    "sourceId": "design_diagnostics",
                    "defaultSort": {
                        "field": "absolute_residual",
                        "direction": "desc",
                    },
                    "columns": [
                        {"field": "size", "label": "Coalition size", "format": "number"},
                        {
                            "field": "winning_probability",
                            "label": "Exact win probability",
                            "format": "percent",
                        },
                        {
                            "field": "linear_baseline",
                            "label": "Linear baseline",
                            "format": "percent",
                        },
                        {
                            "field": "size_only_residual",
                            "label": "Residual",
                            "format": "percent",
                        },
                        {
                            "field": "absolute_residual",
                            "label": "Absolute residual",
                            "format": "percent",
                        },
                    ],
                },
                {
                    "id": "sensitivity_table",
                    "title": "Candidate-pool and balance sensitivity",
                    "subtitle": (
                        "2,550 inner calls, 20 independent seeds; aggregate "
                        "coordinate RMSE."
                    ),
                    "dataset": "sensitivity",
                    "sourceId": "design_diagnostics",
                    "defaultSort": {"field": "rmse", "direction": "asc"},
                    "columns": [
                        {"field": "configuration", "label": "Configuration", "type": "text"},
                        {"field": "rmse", "label": "RMSE", "format": "number"},
                        {
                            "field": "design_seconds",
                            "label": "Design seconds",
                            "format": "number",
                        },
                        {
                            "field": "frame_frobenius",
                            "label": "Frame discrepancy",
                            "format": "number",
                        },
                        {
                            "field": "slice_mean_direction_rms",
                            "label": "Slice mean imbalance",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "hyperparameter_table",
                    "title": "Practical controls",
                    "subtitle": "Visible and fixed choices in the current experiment.",
                    "dataset": "hyperparameters",
                    "sourceId": "design_code",
                    "defaultSort": {"field": "parameter", "direction": "asc"},
                    "columns": [
                        {"field": "parameter", "label": "Parameter", "type": "text"},
                        {"field": "applies_to", "label": "Applies to", "type": "text"},
                        {
                            "field": "experiment_value",
                            "label": "Experiment value",
                            "type": "text",
                        },
                        {"field": "effect", "label": "Effect", "type": "text"},
                    ],
                },
            ],
            "sources": manifest_sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": f"# {REPORT_TITLE}",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Technical summary\n\n"
                        "**The faster method is not a weakened version of the "
                        "slower method.** Batch-balanced Frame-OFA restricts "
                        "coalitions to complete 51-row cyclic batches. That "
                        "restriction makes every player's fixed-size inclusion "
                        "count exactly equal and removes a large variance term "
                        "that coalition-wise greedy only penalizes softly.\n\n"
                        f"Across the stored budgets, Greedy Frame-OFA reduced "
                        f"RMSE versus Random OFA but remained above the "
                        f"batch-balanced design. At the largest budget, the "
                        f"RMSE values were {main_high['row_greedy']['aggregate_rmse']:.5f} "
                        f"and {main_high['orbit_greedy']['aggregate_rmse']:.5f}, "
                        "respectively. The same cyclic symmetry also reduces "
                        "design complexity, so being faster and more accurate "
                        "in this game is not contradictory."
                    ),
                },
                {
                    "id": "main_finding",
                    "type": "markdown",
                    "sourceId": "voting_results",
                    "body": (
                        "## The accuracy gap persists across all eight budgets\n\n"
                        f"At the smallest budget, Greedy and Batch-balanced "
                        f"RMSE were {main_low['row_greedy']['aggregate_rmse']:.5f} "
                        f"and {main_low['orbit_greedy']['aggregate_rmse']:.5f}. "
                        "The batch method stayed lower as calls increased. "
                        "Read the chart as an empirical result for this voting "
                        "game, not a universal dominance theorem."
                    ),
                },
                {"id": "rmse_chart", "type": "chart", "chartId": "rmse_curve"},
                {
                    "id": "mechanism_intro",
                    "type": "markdown",
                    "sourceId": "design_code",
                    "body": (
                        "## Exact batch balance is stronger than a local soft penalty\n\n"
                        "Greedy Frame-OFA scores one candidate coalition at a "
                        "time using a second-moment frame term plus a weighted "
                        "mean-balance penalty. Batch-balanced Frame-OFA scores "
                        "an entire cyclic batch through FFT autocorrelation. "
                        "Every complete batch has direction sum exactly zero; "
                        "the row design has no such guarantee."
                    ),
                },
                {
                    "id": "mechanism_table_block",
                    "type": "table",
                    "tableId": "mechanism_table",
                },
                {
                    "id": "driver_finding",
                    "type": "markdown",
                    "sourceId": "design_diagnostics",
                    "body": (
                        "## The threshold game's nonlinear size profile is the main driver\n\n"
                        f"The estimator subtracts the linear baseline s/51, but "
                        f"the exact fixed-size winning curve leaves an OFA-weighted "
                        f"residual RMS of {weighted_rms:.4f}. A size-only residual "
                        "multiplied by an unbalanced direction sum leaks directly "
                        "into the Shapley estimate. Complete balanced batches "
                        "cancel every such fixed-size constant exactly."
                    ),
                },
                {
                    "id": "size_residual_table_block",
                    "type": "table",
                    "tableId": "size_residual_table",
                },
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The population is 51 jurisdiction blocs in the 538-vote "
                        "Electoral College game with winning quota 270. RMSE is "
                        "the square root of mean coordinate error against exact "
                        "Shapley–Shubik values. Each budget includes 104 exact "
                        "boundary calls. The main comparison uses 30 independent "
                        "designs per budget."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "design_code",
                    "body": (
                        "## How coalition-wise greedy works\n\n"
                        "For a sampled size s, it generates K unused candidates. "
                        "For unit direction u, accumulated weighted frame A, and "
                        "same-size direction sum m_s, it minimizes\n\n"
                        "`sqrt(s(n-s)) * uᵀ A u + λ * uᵀ m_s`.\n\n"
                        "The first term discourages squared correlation with "
                        "previous directions. The second only softly encourages "
                        "a zero same-size mean. The procedure is local, finite-"
                        "candidate, and utility-agnostic; it is not a global "
                        "optimizer of voting-game RMSE."
                    ),
                },
                {
                    "id": "sensitivity_finding",
                    "type": "markdown",
                    "sourceId": "design_diagnostics",
                    "body": (
                        "## Tuning K and λ does not close this voting-game gap\n\n"
                        "The controlled 20-seed sweep changes candidate-pool "
                        "size and mean-balance weight at 2,550 inner calls. "
                        "Larger K improves the frame surrogate but RMSE is not "
                        "monotone in K. The tested λ values alter first-moment "
                        "imbalance only slightly, while complete batches keep it "
                        "at zero by construction."
                    ),
                },
                {
                    "id": "sensitivity_table_block",
                    "type": "table",
                    "tableId": "sensitivity_table",
                },
                {
                    "id": "hyperparameters",
                    "type": "markdown",
                    "sourceId": "design_code",
                    "body": (
                        "## The main practical controls are K, λ, budget, and seed\n\n"
                        "Only Greedy Frame-OFA exposes λ because balanced batches "
                        "already enforce the corresponding constraint exactly. "
                        "The baseline is an estimator choice rather than a "
                        "coalition-design parameter, but it materially affects "
                        "variance in this threshold game."
                    ),
                },
                {
                    "id": "hyperparameter_table_block",
                    "type": "table",
                    "tableId": "hyperparameter_table",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and robustness\n\n"
                        "Batch-balanced budgets must be multiples of 51 and use "
                        "only T/51 size draws, so radial size coverage is coarser. "
                        "Its low-budget curve is heavy-tailed and need not be "
                        "monotone. The design family is cyclic and may be worse "
                        "for other high-order games. Both estimators retain the "
                        "correct OFA marginal after random relabeling, so the "
                        "observed finite-repeat bias is not evidence of theoretical "
                        "bias."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "- Use Batch-balanced Frame-OFA when the budget can be a "
                        "multiple of n and batch evaluation is acceptable.\n"
                        "- Use Greedy Frame-OFA for arbitrary budgets or sequential "
                        "selection; start with K=8–32 and λ between 0 and 0.1.\n"
                        "- Test a fixed-size mean baseline μ_s instead of s/n. If "
                        "Greedy then approaches Batch-balanced, it confirms that "
                        "size-only leakage is the dominant mechanism."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "Can a row design enforce exact per-size first moments "
                        "without requiring complete n-row batches? Can partial "
                        "orbits preserve most of the balance benefit at arbitrary "
                        "budgets? Those are the most promising extensions if the "
                        "goal is to combine Greedy's budget flexibility with "
                        "Batch-balanced accuracy."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "rmse_curve": curve_rows,
                "mechanism_comparison": mechanism_rows,
                "size_residual_extremes": extreme_size_rows,
                "sensitivity": sensitivity_rows,
                "hyperparameters": hyperparameter_rows,
            },
        },
        "sources": canonical_sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=Path,
        default=Path(
            "results/us_electoral_college_2024_frame_ofa_greedy.json"
        ),
    )
    parser.add_argument(
        "--diagnostics",
        type=Path,
        default=Path(
            "results/us_electoral_college_frame_design_diagnostics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/us_electoral_college_frame_design_report_artifact.json"
        ),
    )
    args = parser.parse_args()
    experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
    diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    artifact = build_artifact(experiment, diagnostics, generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

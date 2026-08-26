"""Build a bounded technical-report artifact for the INSIDE gap audit."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "inside_greedy_gap_report_artifact.json"

FORMAL_REPORTS = {
    "Wine": RESULTS / "wine_inside_comparison_3repeats_71k_1p42m.json",
    "Airport": RESULTS / "airport_inside_comparison_3repeats_50k_1m.json",
    "Voting": RESULTS / "voting_inside_comparison_3repeats_25k_510k.json",
}
DEBUG_REPORT = RESULTS / "inside_greedy_gap_debug.json"
SYNTHETIC_REPORT = RESULTS / "synthetic_n8_t40.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _formal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, path in FORMAL_REPORTS.items():
        report = _read(path)
        num_players = int(
            report["game"].get(
                "num_players", report["game"].get("players")
            )
        )
        for inner_key, result in sorted(
            report["results_by_inner_budget"].items(),
            key=lambda item: int(item[0]),
        ):
            methods = result["methods"]
            greedy = float(methods["inside_greedy"]["aggregate_rmse"])
            orbit = float(methods["inside_orbit"]["aggregate_rmse"])
            iid_linear = float(
                methods["ofa_iid_linear"]["aggregate_rmse"]
            )
            iid_ratio = float(
                methods["ofa_iid_ratio"]["aggregate_rmse"]
            )
            inner_calls = int(inner_key)
            rows.append(
                {
                    "dataset": dataset,
                    "players": num_players,
                    "budget_multiplier": inner_calls // num_players,
                    "inner_calls": inner_calls,
                    "total_calls": int(
                        result["total_utility_calls_per_estimate"]
                    ),
                    "greedy_rmse": greedy,
                    "orbit_rmse": orbit,
                    "iid_linear_rmse": iid_linear,
                    "iid_ratio_rmse": iid_ratio,
                    "greedy_to_orbit_ratio": greedy / orbit,
                    "linear_to_ratio_family_ratio": (
                        iid_linear / iid_ratio
                    ),
                    "greedy_gain_vs_iid_linear_percent": 100.0
                    * (1.0 - greedy / iid_linear),
                    "orbit_gain_vs_iid_ratio_percent": 100.0
                    * (1.0 - orbit / iid_ratio),
                    "greedy_over_iid_linear_factor": greedy / iid_linear,
                    "iid_linear_over_iid_ratio_factor": (
                        iid_linear / iid_ratio
                    ),
                    "iid_ratio_over_orbit_factor": iid_ratio / orbit,
                }
            )
    return rows


def _synthetic_rows() -> list[dict[str, Any]]:
    report = _read(SYNTHETIC_REPORT)
    rows: list[dict[str, Any]] = []
    labels = {"coupled": "INSIDE-Greedy", "orbit": "INSIDE-Orbit"}
    for game, summaries in report["games"].items():
        greedy = float(summaries["coupled"]["rmse"])
        orbit = float(summaries["orbit"]["rmse"])
        for method in ("coupled", "orbit"):
            rows.append(
                {
                    "game": game,
                    "method": labels[method],
                    "rmse": float(summaries[method]["rmse"]),
                    "players": int(report["configuration"]["players"]),
                    "inner_calls": int(report["configuration"]["samples"]),
                    "total_calls": int(summaries[method]["total_queries"]),
                    "repeats": int(report["configuration"]["repeats"]),
                    "candidate_pool": int(
                        report["configuration"]["candidate_pool"]
                    ),
                    "greedy_reduction_vs_orbit_percent": 100.0
                    * (1.0 - greedy / orbit),
                }
            )
    return rows


def _ablation_rows(debug: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in debug["comparison_table"]:
        dataset = str(item["dataset"]).title()
        multiplier = int(item["budget_multiplier"])
        scenario = f"{dataset} ×{multiplier}"
        rows.extend(
            [
                {
                    "scenario": scenario,
                    "dataset": dataset,
                    "budget_multiplier": multiplier,
                    "component": "Greedy: exact size-mean baseline",
                    "rmse_reduction_percent": -float(
                        item["greedy_oracle_change_percent"]
                    ),
                    "before_rmse": float(item["greedy_endpoint_rmse"]),
                    "after_rmse": float(
                        item["greedy_oracle_size_rmse"]
                    ),
                    "same_coalitions": "yes",
                },
                {
                    "scenario": scenario,
                    "dataset": dataset,
                    "budget_multiplier": multiplier,
                    "component": "Greedy: official ratio aggregation",
                    "rmse_reduction_percent": -float(
                        item["greedy_ratio_change_percent"]
                    ),
                    "before_rmse": float(item["greedy_endpoint_rmse"]),
                    "after_rmse": float(
                        item["greedy_official_ratio_rmse"]
                    ),
                    "same_coalitions": "yes",
                },
                {
                    "scenario": scenario,
                    "dataset": dataset,
                    "budget_multiplier": multiplier,
                    "component": "Same orbit: ratio instead of global HT",
                    "rmse_reduction_percent": -float(
                        item["same_qstar_orbit_ratio_change_percent"]
                    ),
                    "before_rmse": float(
                        item["qstar_orbit_global_rmse"]
                    ),
                    "after_rmse": float(
                        item["qstar_orbit_ratio_rmse"]
                    ),
                    "same_coalitions": "yes",
                },
            ]
        )
    return rows


def _size_profile_rows(debug: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, profile in debug["size_profiles"].items():
        rows.append(
            {
                "dataset": dataset.title(),
                "ofa_weighted_rms_mismatch": float(
                    profile["ofa_weighted_rms_endpoint_mismatch"]
                ),
                "ofa_weighted_mean_absolute_mismatch": float(
                    profile[
                        "ofa_weighted_mean_absolute_endpoint_mismatch"
                    ]
                ),
                "maximum_absolute_mismatch": float(
                    profile["maximum_absolute_endpoint_mismatch"]
                ),
            }
        )
    return rows


def _sensitivity_rows(debug: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in debug["sensitivity_summaries"]:
        diagnostics = item["mean_design_diagnostics"]
        rows.append(
            {
                "dataset": str(item["dataset"]).title(),
                "configuration": item["configuration"],
                "candidate_pool": int(item["candidate_pool"]),
                "mean_balance": float(item["mean_balance"]),
                "endpoint_rmse": float(item["aggregate_endpoint_rmse"]),
                "oracle_size_rmse": float(
                    item["aggregate_oracle_size_rmse"]
                ),
                "frame_frobenius": float(
                    diagnostics["frobenius_discrepancy"]
                ),
                "slice_mean_direction_rms": float(
                    diagnostics["slice_mean_direction_rms"]
                ),
            }
        )
    return rows


def _lambda_scale_rows() -> list[dict[str, Any]]:
    """Return the dimensionless-to-raw lambda conversion under inner q*."""
    rows: list[dict[str, Any]] = []
    for label, num_players in (
        ("Synthetic", 8),
        ("Voting", 51),
        ("Airport", 100),
        ("Wine", 142),
        ("Cancer", 455),
    ):
        sizes = range(2, num_players - 1)
        raw = [
            1.0 / math.sqrt(size * (num_players - size))
            for size in sizes
        ]
        normalizer = sum(raw)
        mean_squared_weight = sum(
            probability
            / normalizer
            * size
            * (num_players - size)
            for size, probability in zip(sizes, raw)
        )
        conversion = (
            1.0 - 1.0 / (num_players - 1)
        ) * mean_squared_weight
        rows.append(
            {
                "dataset": label,
                "players": num_players,
                "raw_lambda_for_lambda0_1": conversion,
                "lambda0_equivalent_of_raw_0_1": 0.1 / conversion,
            }
        )
    return rows


def _audit_rows(invariants: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "check": "Draft Eq. (12)/(13) vs row-greedy score",
            "status": "PASS",
            "evidence": "w_s u^T A u + lambda u^T m_s is implemented directly",
        },
        {
            "check": "Cyclic strict ratio vs slice-wise linear",
            "status": "PASS",
            "evidence": (
                f"maximum absolute difference "
                f"{invariants['max_abs_cyclic_ratio_vs_stratified_linear']:.3g}"
            ),
        },
        {
            "check": "Complete-orbit cancellation of any fixed size baseline",
            "status": "PASS",
            "evidence": (
                f"maximum absolute difference "
                f"{invariants['max_abs_qstar_orbit_endpoint_vs_oracle_size']:.3g}"
            ),
        },
        {
            "check": "Formal physical call accounting",
            "status": "PASS",
            "evidence": "all three stored validation reports are ready_to_share",
        },
        {
            "check": "Fixed lambda=0.1 gives exact first-moment balance",
            "status": "FAIL (method limitation)",
            "evidence": "row-wise objective is a soft, scale-sensitive penalty; m_s is nonzero",
        },
        {
            "check": "Greedy and Orbit curves isolate coalition design alone",
            "status": "FAIL (comparison caveat)",
            "evidence": "Greedy uses global HT; Orbit uses balanced fixed-size ratio",
        },
    ]


def build_artifact() -> dict[str, Any]:
    debug = _read(DEBUG_REPORT)
    formal = _formal_rows()
    synthetic = _synthetic_rows()
    ablation = _ablation_rows(debug)
    size_profiles = _size_profile_rows(debug)
    sensitivity = _sensitivity_rows(debug)
    lambda_scales = _lambda_scale_rows()
    invariants = debug["invariants"]
    generated_at = datetime.now(timezone.utc).isoformat()

    audit_checks = _audit_rows(invariants)

    formal_paths = [str(path.relative_to(ROOT)) for path in FORMAL_REPORTS.values()]

    def transformation_source(
        source_id: str,
        label: str,
        dataset_id: str,
        code: str,
        input_files: list[str],
        description: str,
        metric_definitions: list[str],
    ) -> dict[str, Any]:
        return {
            "id": source_id,
            "label": label,
            "path": "experiments/build_inside_greedy_gap_report.py",
            "query": {
                "id": source_id,
                "engine": "artifact_snapshot_sql",
                "language": "sql",
                "executed_at": generated_at,
                "sql": f"SELECT * FROM artifact.{dataset_id}",
                "description": description,
                "tables_used": [f"artifact.{dataset_id}"],
                "transformation_language": "python",
                "transformation_code": code,
                "input_files": input_files,
                "filters": [
                    "Use only completed stored experiment reports",
                    "Preserve every formal budget point",
                ],
                "metric_definitions": metric_definitions,
            },
        }

    sources = [
        transformation_source(
            "formal_transform",
            "Formal comparison transformation",
            "formal_gap",
            (
                "from experiments.build_inside_greedy_gap_report import _formal_rows\n"
                "rows = _formal_rows()"
            ),
            formal_paths,
            "Read all formal Wine, Airport, and Voting budget summaries and compute matched RMSE ratios.",
            [
                "greedy_to_orbit_ratio = INSIDE-Greedy aggregate RMSE / INSIDE-Orbit aggregate RMSE",
                "matched gain = 1 - method RMSE / estimator-matched IID RMSE",
            ],
        ),
        transformation_source(
            "debug_transform",
            "Paired-ablation transformation",
            "ablation_effects",
            (
                "from experiments.build_inside_greedy_gap_report import _ablation_rows, _read, DEBUG_REPORT\n"
                "rows = _ablation_rows(_read(DEBUG_REPORT))"
            ),
            ["results/inside_greedy_gap_debug.json"],
            "Read paired same-coalition ablations and express before/after changes as RMSE reductions.",
            [
                "RMSE reduction (%) = 100 * (1 - after RMSE / before RMSE)",
                "Every paired change reuses the identical realized coalition rows",
            ],
        ),
        transformation_source(
            "synthetic_transform",
            "Exact synthetic comparison transformation",
            "synthetic_results",
            (
                "from experiments.build_inside_greedy_gap_report import _synthetic_rows\n"
                "rows = _synthetic_rows()"
            ),
            ["results/synthetic_n8_t40.json"],
            "Read exact n=8 synthetic results for the true row-Greedy and cyclic-Orbit methods.",
            ["RMSE is aggregate coordinate RMSE over 100 independent repeats"],
        ),
        transformation_source(
            "audit_transform",
            "Code-audit checklist",
            "audit_checks",
            (
                "from experiments.build_inside_greedy_gap_report import _audit_rows, _read, DEBUG_REPORT\n"
                "rows = _audit_rows(_read(DEBUG_REPORT)['invariants'])"
            ),
            [
                "frame_ofa/design.py",
                "frame_ofa/estimator.py",
                "experiments/run_analytic_inside_comparison.py",
                "results/inside_greedy_gap_debug.json",
            ],
            "Assemble reviewed formula, implementation, invariant, and comparison checks.",
            ["PASS denotes exact identity or validated implementation behavior"],
        ),
        transformation_source(
            "size_profile_transform",
            "Exact size-profile mismatch transformation",
            "size_profiles",
            (
                "from experiments.build_inside_greedy_gap_report import _size_profile_rows, _read, DEBUG_REPORT\n"
                "rows = _size_profile_rows(_read(DEBUG_REPORT))"
            ),
            ["results/inside_greedy_gap_debug.json"],
            "Read exact fixed-size means and summarize mismatch from the endpoint-linear baseline.",
            ["q*-weighted RMS mismatch = sqrt(sum_s q_s * (mu_s - ell_s)^2)"],
        ),
        transformation_source(
            "sensitivity_transform",
            "Greedy K and lambda sensitivity transformation",
            "sensitivity",
            (
                "from experiments.build_inside_greedy_gap_report import _sensitivity_rows, _read, DEBUG_REPORT\n"
                "rows = _sensitivity_rows(_read(DEBUG_REPORT))"
            ),
            ["results/inside_greedy_gap_debug.json"],
            "Read the fixed-budget Greedy candidate-pool and raw-lambda sweep.",
            ["Aggregate RMSE = sqrt(mean over repeats of coordinate MSE)"],
        ),
        transformation_source(
            "lambda_transform",
            "Lambda-scale calculation",
            "lambda_scales",
            (
                "from experiments.build_inside_greedy_gap_report import _lambda_scale_rows\n"
                "rows = _lambda_scale_rows()"
            ),
            ["frame_ofa/design.py"],
            "Compute the raw first-moment penalty corresponding to a dimensionless lambda0 under inner q*.",
            [
                "lambda_eff = lambda0 * (1 - 1/(n-1)) * E_q*[s(n-s)]",
            ],
        ),
        {
            "id": "debug_results",
            "label": "Paired INSIDE gap ablation",
            "path": "results/inside_greedy_gap_debug.json",
        },
        {
            "id": "synthetic_results",
            "label": "Exact n=8 synthetic benchmark",
            "path": "results/synthetic_n8_t40.json",
        },
        *[
            {
                "id": f"formal_{dataset.lower()}",
                "label": f"Formal {dataset} INSIDE comparison",
                "path": str(path.relative_to(ROOT)),
            }
            for dataset, path in FORMAL_REPORTS.items()
        ],
        {
            "id": "design_code",
            "label": "Coalition-design implementation",
            "path": "frame_ofa/design.py",
        },
        {
            "id": "estimator_code",
            "label": "OFA estimator implementation",
            "path": "frame_ofa/estimator.py",
        },
        {
            "id": "debug_code",
            "label": "Reproducible paired diagnostic",
            "path": "experiments/debug_inside_greedy_gap.py",
        },
    ]

    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": "Why INSIDE-Greedy Looks Worse Than INSIDE-Orbit",
        "description": (
            "Code, formula, history, and paired-ablation audit of the "
            "INSIDE-Greedy versus INSIDE-Orbit RMSE gap."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": [],
        "charts": [
            {
                "id": "formal_gap_chart",
                "type": "line",
                "dataset": "formal_gap",
                "sourceId": "formal_transform",
                "title": "Greedy-to-Orbit RMSE ratio",
                "subtitle": (
                    "All formal points exceed 1, but this compares two "
                    "design-and-estimator bundles."
                ),
                "intent": "trend",
                "question": (
                    "How large is the formal RMSE gap across normalized budgets?"
                ),
                "rationale": (
                    "A normalized ratio compares datasets with different RMSE scales."
                ),
                "encodings": {
                    "x": {
                        "field": "budget_multiplier",
                        "type": "quantitative",
                        "label": "Inner calls / players",
                    },
                    "y": {
                        "field": "greedy_to_orbit_ratio",
                        "type": "quantitative",
                        "label": "Greedy RMSE / Orbit RMSE",
                        "format": "number",
                    },
                    "color": {
                        "field": "dataset",
                        "type": "nominal",
                        "label": "Dataset",
                    },
                    "tooltip": [
                        {
                            "field": "linear_to_ratio_family_ratio",
                            "type": "quantitative",
                            "label": "IID linear / IID ratio RMSE",
                            "format": "number",
                        },
                        {
                            "field": "greedy_gain_vs_iid_linear_percent",
                            "type": "quantitative",
                            "label": "Greedy gain vs matched IID linear",
                            "format": "number",
                        },
                    ],
                },
            },
            {
                "id": "ablation_chart",
                "type": "bar",
                "dataset": "ablation_effects",
                "sourceId": "debug_transform",
                "title": "RMSE change under same-coalition ablations",
                "subtitle": (
                    "Exact size centering or row-ratio transforms Greedy; changing only "
                    "global-versus-ratio aggregation on balanced rows does not."
                ),
                "intent": "comparison",
                "question": "Which factor explains the observed gap?",
                "rationale": (
                    "Paired rows isolate the baseline and aggregation effects."
                ),
                "encodings": {
                    "x": {
                        "field": "scenario",
                        "type": "nominal",
                        "label": "Dataset and budget multiplier",
                    },
                    "y": {
                        "field": "rmse_reduction_percent",
                        "type": "quantitative",
                        "label": "RMSE reduction (%)",
                        "format": "number",
                    },
                    "color": {
                        "field": "component",
                        "type": "nominal",
                        "label": "Paired change",
                    },
                    "tooltip": [
                        {
                            "field": "before_rmse",
                            "type": "quantitative",
                            "label": "Before RMSE",
                            "format": "number",
                        },
                        {
                            "field": "after_rmse",
                            "type": "quantitative",
                            "label": "After RMSE",
                            "format": "number",
                        },
                    ],
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
            },
            {
                "id": "synthetic_chart",
                "type": "bar",
                "dataset": "synthetic_results",
                "sourceId": "synthetic_transform",
                "title": "Exact n=8 synthetic-game RMSE",
                "subtitle": (
                    "Greedy wins when second-order coverage matters and the "
                    "size baseline is well aligned with low-order structure."
                ),
                "intent": "comparison",
                "question": "Has INSIDE-Greedy ever beaten INSIDE-Orbit?",
                "rationale": "The exact benchmark is the clean historical counterexample.",
                "encodings": {
                    "x": {
                        "field": "game",
                        "type": "nominal",
                        "label": "Synthetic game",
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
                        "label": "Method",
                    },
                },
                "options": {"orientation": "vertical", "grouping": "grouped"},
            },
        ],
        "tables": [
            {
                "id": "formal_table",
                "dataset": "formal_gap",
                "sourceId": "formal_transform",
                "title": "Formal comparison and matched-family controls",
                "subtitle": "Three repeats per point; lower RMSE is better.",
                "defaultSort": {
                    "field": "greedy_to_orbit_ratio",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "dataset", "label": "Dataset", "type": "text"},
                    {
                        "field": "budget_multiplier",
                        "label": "Calls / n",
                        "format": "number",
                    },
                    {
                        "field": "greedy_rmse",
                        "label": "Greedy RMSE",
                        "format": "number",
                    },
                    {
                        "field": "orbit_rmse",
                        "label": "Orbit RMSE",
                        "format": "number",
                    },
                    {
                        "field": "greedy_to_orbit_ratio",
                        "label": "Greedy / Orbit",
                        "format": "number",
                    },
                    {
                        "field": "greedy_gain_vs_iid_linear_percent",
                        "label": "Greedy gain vs IID linear (%)",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "audit_table",
                "dataset": "audit_checks",
                "sourceId": "audit_transform",
                "title": "Code and comparison audit",
                "subtitle": "A failed comparison check is a caveat, not a numerical crash.",
                "defaultSort": {"field": "status", "direction": "asc"},
                "columns": [
                    {"field": "check", "label": "Check", "type": "text"},
                    {"field": "status", "label": "Status", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                ],
            },
            {
                "id": "size_profile_table",
                "dataset": "size_profiles",
                "sourceId": "size_profile_transform",
                "title": "Endpoint baseline mismatch",
                "subtitle": "Mismatch is E[v(S) | |S|=s] minus endpoint interpolation.",
                "defaultSort": {
                    "field": "ofa_weighted_rms_mismatch",
                    "direction": "desc",
                },
                "columns": [
                    {"field": "dataset", "label": "Dataset", "type": "text"},
                    {
                        "field": "ofa_weighted_rms_mismatch",
                        "label": "q*-weighted RMS",
                        "format": "number",
                    },
                    {
                        "field": "maximum_absolute_mismatch",
                        "label": "Maximum absolute mismatch",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "sensitivity_table",
                "dataset": "sensitivity",
                "sourceId": "sensitivity_transform",
                "title": "Greedy K and lambda sensitivity",
                "subtitle": "Six repeats at inner calls = 100n; oracle means diagnostic only.",
                "defaultSort": {"field": "dataset", "direction": "asc"},
                "columns": [
                    {"field": "dataset", "label": "Dataset", "type": "text"},
                    {
                        "field": "configuration",
                        "label": "Configuration",
                        "type": "text",
                    },
                    {
                        "field": "endpoint_rmse",
                        "label": "Endpoint RMSE",
                        "format": "number",
                    },
                    {
                        "field": "oracle_size_rmse",
                        "label": "Exact-size RMSE",
                        "format": "number",
                    },
                    {
                        "field": "frame_frobenius",
                        "label": "Frame discrepancy",
                        "format": "number",
                    },
                    {
                        "field": "slice_mean_direction_rms",
                        "label": "Slice mean RMS",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "lambda_scale_table",
                "dataset": "lambda_scales",
                "sourceId": "lambda_transform",
                "title": "Raw lambda scale under the OFA size distribution",
                "subtitle": "Conversion for dimensionless lambda0 = 1 on inner sizes 2,...,n-2.",
                "defaultSort": {"field": "players", "direction": "asc"},
                "columns": [
                    {"field": "dataset", "label": "Reference", "type": "text"},
                    {"field": "players", "label": "Players", "format": "number"},
                    {
                        "field": "raw_lambda_for_lambda0_1",
                        "label": "Raw lambda for lambda0=1",
                        "format": "number",
                    },
                    {
                        "field": "lambda0_equivalent_of_raw_0_1",
                        "label": "Dimensionless value of raw 0.1",
                        "format": "number",
                    },
                ],
            },
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# Why INSIDE-Greedy Looks Worse Than INSIDE-Orbit",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": (
                    "## Technical summary\n\n"
                    "**No formula, call-budget, or estimator implementation bug was found.** "
                    "The apparent failure is primarily a method-bundle mismatch: INSIDE-Greedy "
                    "uses a global HT/linear estimator with an endpoint-linear size baseline, "
                    "while INSIDE-Orbit enforces exact fixed-size first-moment balance and uses "
                    "the equivalent balanced ratio/slice-wise estimator. Greedy still beats its "
                    "matched IID-linear baseline at every formal point."
                ),
            },
            {
                "id": "historical_finding",
                "type": "markdown",
                "sourceId": "synthetic_results",
                "body": (
                    "## Greedy has beaten Orbit before\n\n"
                    "The exact n=8, T=40 synthetic experiment is a real counterexample, not a "
                    "misnamed old curve. Across 100 repeats, Greedy reduces RMSE by 67.0% on "
                    "additive, 47.6% on pairwise, and 46.4% on degree-3 games. With only one "
                    "cyclic orbit per inner size, Orbit has poorer aggregate second-moment "
                    "coverage; Greedy can compensate across sizes."
                ),
            },
            {"id": "synthetic_chart_block", "type": "chart", "chartId": "synthetic_chart"},
            {
                "id": "formal_finding",
                "type": "markdown",
                "body": (
                    "## The formal large-game gap is real but not design-only\n\n"
                    "Orbit is lower at all 15 Wine, Airport, and Voting budget points. Yet the "
                    "matched controls show that Greedy improves on IID linear by 1.7%--25.9% "
                    "at every point. The IID-linear estimator itself is 4.6--5.0 times worse "
                    "than IID ratio on Wine, roughly 9.0--10.8 times worse on Airport, and "
                    "1.45--1.73 times worse on Voting. That family gap dominates the headline "
                    "Greedy/Orbit ratio."
                ),
            },
            {"id": "formal_gap_chart_block", "type": "chart", "chartId": "formal_gap_chart"},
            {"id": "formal_table_block", "type": "table", "tableId": "formal_table"},
            {
                "id": "root_cause",
                "type": "markdown",
                "sourceId": "debug_results",
                "body": (
                    "## The root cause is fixed-size mean leakage\n\n"
                    "Write `mu_s = E[v(S) | |S|=s]` and let `ell_s` be the endpoint baseline. "
                    "The global inner update decomposes as `C/T sum_s sum_t (v_t-mu_s)u_t + "
                    "C/T sum_s (mu_s-ell_s)m_s`, where `m_s=sum_t u_t`. Row-wise Greedy only "
                    "softly penalizes `m_s`; complete cyclic orbits make every `m_s=0` exactly. "
                    "Airport's q*-weighted endpoint mismatch is 5.575, and Voting's is 0.2143, "
                    "so the second term can dominate despite a small frame discrepancy."
                ),
            },
            {"id": "size_profile_table_block", "type": "table", "tableId": "size_profile_table"},
            {
                "id": "ablation_finding",
                "type": "markdown",
                "sourceId": "debug_results",
                "body": (
                    "## Same-coalition ablations isolate the mechanism\n\n"
                    "On identical Greedy rows, replacing only the endpoint baseline with the "
                    "exact fixed-size mean cuts Airport RMSE by about 88% and Voting RMSE by "
                    "30%--32%. The upstream row-ratio estimator gives almost the same gains, "
                    "although its random denominators generally sacrifice finite-sample "
                    "unbiasedness. On identical q*-marginal complete-orbit rows, switching only "
                    "global HT to strict ratio changes RMSE by less than 1%. Thus fixed-size "
                    "centering/cancellation explains most of the gap."
                ),
            },
            {"id": "ablation_chart_block", "type": "chart", "chartId": "ablation_chart"},
            {
                "id": "formula_audit",
                "type": "markdown",
                "sourceId": "design_code",
                "body": (
                    "## Formula and implementation audit\n\n"
                    "The direction normalization, q* distribution, systematic size marginal, "
                    "row-greedy score, random relabeling, exact boundary vector, global HT "
                    "coefficient, cyclic 1-balance, strict ratio formula, and physical call "
                    "accounting all match their stated identities. Strict ratio and slice-wise "
                    "linear agree to 5.6e-16 in the diagnostic. The issue is statistical: the "
                    "current named curves bind different estimators to different designs."
                ),
            },
            {"id": "audit_table_block", "type": "table", "tableId": "audit_table"},
            {
                "id": "sensitivity_finding",
                "type": "markdown",
                "sourceId": "debug_results",
                "body": (
                    "## K and lambda do not repair the missing constraint\n\n"
                    "Increasing K sharply improves second-moment frame discrepancy, but barely "
                    "changes the exact-size-centered RMSE or slice-mean imbalance. Changing "
                    "lambda from 0 to 1 yields only a modest Airport improvement and is "
                    "non-monotone on Voting. The raw Eq. (12) terms are scale-sensitive, so a "
                    "fixed lambda=0.1 is a weak soft penalty at large n; it is not a substitute "
                    "for exact first-moment balance. A dimensionless parameter can be converted "
                    "with `lambda_eff=lambda0*(1-1/(n-1))*mean_q*[s(n-s)]`."
                ),
            },
            {"id": "sensitivity_table_block", "type": "table", "tableId": "sensitivity_table"},
            {"id": "lambda_scale_table_block", "type": "table", "tableId": "lambda_scale_table"},
            {
                "id": "scope",
                "type": "markdown",
                "body": (
                    "## Scope and definitions\n\n"
                    "RMSE is coordinate RMSE against exact truth for Airport/Voting and the "
                    "stored high-budget truth for Wine. Formal results use three repeats per "
                    "budget. The paired ablation uses six repeats and exact truths. The exact "
                    "size-mean baseline is an oracle diagnostic and is not presented as a "
                    "deployable estimator. Lower RMSE is better."
                ),
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "debug_code",
                "body": (
                    "## Methodology\n\n"
                    "The audit first classified every historical JSON by the actual constructor "
                    "and estimator, excluding old `frame_coupled_linear` fields whose config says "
                    "`orbit_coupled`. It then checked identities against source and tests. Finally, "
                    "it reused each realized design across endpoint/oracle baselines and global/"
                    "ratio aggregation so each causal comparison changes one factor only."
                ),
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": (
                    "## Limitations and robustness\n\n"
                    "The oracle size mean is available only for the two analytic games. Wine's "
                    "same mechanism is inferred from matched IID families, not directly proven "
                    "with an exact learning curve. Formal results have only three repeats, though "
                    "the analytic gaps are much larger than their observed repeat variation. "
                    "Dependent designs have no universal MSE dominance for arbitrary high-order "
                    "games, and the synthetic counterexample should not be generalized beyond its "
                    "small-n, low-order setting."
                ),
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Do not interpret the current Greedy-versus-Orbit plot as a pure design "
                    "ablation; retain matched IID-linear and IID-ratio controls.\n"
                    "2. Keep current production code unchanged until choosing a statistical fix. "
                    "The clean options are exact/near-exact per-size first-moment balancing, "
                    "cross-fitted size baselines, or a normalized first-moment objective.\n"
                    "3. Re-run Wine after estimating a size-only baseline from an independent "
                    "pilot split; this tests the mechanism without using an oracle.\n"
                    "4. If Greedy remains a named main method, report both its global-linear and "
                    "a balanced/centered variant."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "Can partial orbits or dependent rounding enforce `m_s=0` at arbitrary "
                    "budgets? Can a pilot-estimated size baseline preserve unbiasedness through "
                    "cross-fitting while retaining Greedy's small-n second-moment advantage? "
                    "Those are the most direct paths to a stronger INSIDE-Greedy."
                ),
            },
        ],
    }

    # Keep canonical sources in the inventory and inline them on native
    # assets.  The current artifact validator requires the exact executable
    # transformation on each chart/table even when a sourceId is also present.
    sources_by_id = {source["id"]: source for source in sources}
    for asset in [*manifest["charts"], *manifest["tables"]]:
        asset["source"] = sources_by_id[asset["sourceId"]]

    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {
            "formal_gap": formal,
            "synthetic_results": synthetic,
            "ablation_effects": ablation,
            "size_profiles": size_profiles,
            "sensitivity": sensitivity,
            "lambda_scales": lambda_scales,
            "audit_checks": audit_checks,
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def main() -> None:
    artifact = build_artifact()
    OUTPUT.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {OUTPUT}")


if __name__ == "__main__":
    main()

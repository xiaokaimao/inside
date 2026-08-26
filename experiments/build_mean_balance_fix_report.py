"""Build the bounded MCP-report artifact for the mean-balance fix audit.

This is a pure post-processing command.  It reads the strictly validated JSON
created by :mod:`experiments.build_mean_balance_fix_summary`, derives all
reader-facing claims from those stored values, and writes an artifact payload.
It deliberately does not call MCP tools or render the artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.build_mean_balance_fix_summary import (
    DATASET_ORDER,
    _summarize_rows,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_LABELS = {
    "wine": "Wine",
    "airport": "Airport",
    "voting": "U.S. Electoral Voting",
}
SOURCE_SQL = "SELECT * FROM artifact.mean_balance_results"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "summary root must be an object")
    return value


def _finite(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(math.isfinite(result), f"{path} must be finite")
    return result


def _positive(value: Any, *, path: str) -> float:
    result = _finite(value, path=path)
    _require(result > 0.0, f"{path} must be positive")
    return result


def _integer(value: Any, *, path: str, minimum: int = 0) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    result = int(value)
    _require(result >= minimum, f"{path} must be at least {minimum}")
    return result


def _close(stored: Any, expected: float, *, path: str) -> float:
    value = _finite(stored, path=path)
    _require(
        math.isclose(value, expected, rel_tol=3e-13, abs_tol=3e-15),
        f"{path} differs from its recomputed value",
    )
    return value


def _artifact_path(raw: str | Path, *, path: str) -> str:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"{path} must resolve inside the project root") from error
    return relative.as_posix()


def _geometric_mean(values: Sequence[float]) -> float:
    _require(bool(values), "geometric mean requires at least one value")
    checked = [
        _positive(value, path=f"geometric_mean[{index}]")
        for index, value in enumerate(values)
    ]
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def _validate_summary(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _require(summary.get("status") == "ready_to_share", "summary is not ready")
    design = summary.get("validated_design")
    _require(isinstance(design, Mapping), "validated_design must be an object")
    _require(
        design.get("datasets") == list(DATASET_ORDER),
        "validated dataset order is wrong",
    )
    _require(
        _integer(
            design.get("budget_points_per_dataset"),
            path="validated_design.budget_points_per_dataset",
        )
        == 5,
        "each dataset must have five budget points",
    )
    _require(
        _integer(
            design.get("repeats_per_budget_method"),
            path="validated_design.repeats_per_budget_method",
        )
        == 3,
        "each method-budget point must have three repeats",
    )
    _require(
        _integer(
            design.get("total_budget_points"),
            path="validated_design.total_budget_points",
        )
        == 15,
        "the summary must contain 15 budget points",
    )

    identity = summary.get("method_identity")
    _require(isinstance(identity, Mapping), "method_identity must be an object")
    _require(
        identity.get("normalized_greedy")
        == {
            "method_key": "inside_greedy",
            "mean_balance_mode": "normalized",
            "lambda0": 1.0,
        },
        "normalized Greedy identity is wrong",
    )
    _require(
        identity.get("legacy_greedy")
        == {
            "method_key": "inside_greedy",
            "mean_balance_mode": "raw",
            "raw_lambda": 0.1,
        },
        "legacy Greedy identity is wrong",
    )
    _require(
        identity.get("orbit_reference") == {"method_key": "inside_orbit"},
        "Orbit identity is wrong",
    )

    datasets = summary.get("datasets")
    _require(isinstance(datasets, Mapping), "datasets must be an object")
    _require(set(datasets) == set(DATASET_ORDER), "dataset set is incomplete")
    rows: list[Mapping[str, Any]] = []
    for dataset in DATASET_ORDER:
        payload = datasets[dataset]
        _require(
            isinstance(payload, Mapping), f"datasets.{dataset} must be an object"
        )
        raw_rows = payload.get("rows")
        _require(
            isinstance(raw_rows, list) and len(raw_rows) == 5,
            f"datasets.{dataset}.rows must contain five rows",
        )
        previous_calls = 0
        for index, row in enumerate(raw_rows):
            _require(
                isinstance(row, Mapping),
                f"datasets.{dataset}.rows[{index}] must be an object",
            )
            _require(
                row.get("dataset") == dataset,
                f"datasets.{dataset}.rows[{index}] has the wrong identity",
            )
            _require(
                _integer(
                    row.get("budget_index"),
                    path=f"datasets.{dataset}.rows[{index}].budget_index",
                )
                == index,
                f"datasets.{dataset}.rows[{index}] has the wrong budget index",
            )
            calls = _integer(
                row.get("total_utility_calls"),
                path=f"datasets.{dataset}.rows[{index}].total_utility_calls",
                minimum=1,
            )
            _require(calls > previous_calls, f"{dataset} calls must increase")
            previous_calls = calls
            legacy_rmse = _positive(
                row.get("legacy_raw_lambda_0p1_greedy_rmse"),
                path=f"datasets.{dataset}.rows[{index}].legacy_rmse",
            )
            normalized_rmse = _positive(
                row.get("normalized_lambda0_1_greedy_rmse"),
                path=f"datasets.{dataset}.rows[{index}].normalized_rmse",
            )
            orbit_rmse = _positive(
                row.get("inside_orbit_rmse"),
                path=f"datasets.{dataset}.rows[{index}].orbit_rmse",
            )
            _close(
                row.get("normalized_to_legacy_rmse_ratio"),
                normalized_rmse / legacy_rmse,
                path=f"datasets.{dataset}.rows[{index}].new_old_ratio",
            )
            _close(
                row.get("normalized_vs_legacy_improvement_percent"),
                100.0 * (1.0 - normalized_rmse / legacy_rmse),
                path=f"datasets.{dataset}.rows[{index}].improvement_percent",
            )
            _close(
                row.get("normalized_to_orbit_rmse_ratio"),
                normalized_rmse / orbit_rmse,
                path=f"datasets.{dataset}.rows[{index}].new_orbit_ratio",
            )
            for prefix in ("legacy", "normalized"):
                slice_value = _positive(
                    row.get(f"{prefix}_slice_mean_direction_rms"),
                    path=(
                        f"datasets.{dataset}.rows[{index}]"
                        f".{prefix}_slice_mean_direction_rms"
                    ),
                )
                frobenius_value = _positive(
                    row.get(f"{prefix}_frobenius_discrepancy"),
                    path=(
                        f"datasets.{dataset}.rows[{index}]"
                        f".{prefix}_frobenius_discrepancy"
                    ),
                )
                del slice_value, frobenius_value
            _close(
                row.get(
                    "normalized_to_legacy_slice_mean_direction_rms_ratio"
                ),
                float(row["normalized_slice_mean_direction_rms"])
                / float(row["legacy_slice_mean_direction_rms"]),
                path=f"datasets.{dataset}.rows[{index}].slice_ratio",
            )
            _close(
                row.get(
                    "normalized_to_legacy_frobenius_discrepancy_ratio"
                ),
                float(row["normalized_frobenius_discrepancy"])
                / float(row["legacy_frobenius_discrepancy"]),
                path=f"datasets.{dataset}.rows[{index}].frobenius_ratio",
            )
            actual = row.get("actual_utility_calls_by_repeat")
            _require(
                isinstance(actual, Mapping),
                f"datasets.{dataset}.rows[{index}].actual calls are missing",
            )
            for method in ("legacy_greedy", "normalized_greedy", "inside_orbit"):
                values = actual.get(method)
                _require(
                    isinstance(values, list)
                    and len(values) == 3
                    and all(
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and value == calls
                        for value in values
                    ),
                    f"datasets.{dataset}.rows[{index}].{method} calls are wrong",
                )
            rows.append(row)

        stored_dataset_summary = payload.get("summary")
        _require(
            isinstance(stored_dataset_summary, Mapping),
            f"datasets.{dataset}.summary must be an object",
        )
        expected_dataset_summary = _summarize_rows(raw_rows)
        for field in (
            "normalized_to_legacy_geometric_mean_rmse_ratio",
            "normalized_to_orbit_geometric_mean_rmse_ratio",
        ):
            _close(
                stored_dataset_summary.get(field),
                float(expected_dataset_summary[field]),
                path=f"datasets.{dataset}.summary.{field}",
            )
        for field in (
            "budget_point_count",
            "improved_budget_point_count",
            "tied_budget_point_count",
            "worsened_budget_point_count",
        ):
            _require(
                stored_dataset_summary.get(field) == expected_dataset_summary[field],
                f"datasets.{dataset}.summary.{field} is inconsistent",
            )

    expected_overall = _summarize_rows(rows)
    stored_overall = summary.get("summary")
    _require(isinstance(stored_overall, Mapping), "summary.summary must be an object")
    for field in (
        "normalized_to_legacy_geometric_mean_rmse_ratio",
        "normalized_to_orbit_geometric_mean_rmse_ratio",
    ):
        _close(
            stored_overall.get(field),
            float(expected_overall[field]),
            path=f"summary.{field}",
        )
    for field in (
        "budget_point_count",
        "improved_budget_point_count",
        "tied_budget_point_count",
        "worsened_budget_point_count",
    ):
        _require(
            stored_overall.get(field) == expected_overall[field],
            f"summary.{field} is inconsistent",
        )
    return rows


def _result_rows(
    summary: Mapping[str, Any], validated_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    overall = _summarize_rows(validated_rows)
    overall_new_old = float(
        overall["normalized_to_legacy_geometric_mean_rmse_ratio"]
    )
    overall_new_orbit = float(
        overall["normalized_to_orbit_geometric_mean_rmse_ratio"]
    )
    new_beats_orbit = sum(
        float(row["normalized_to_orbit_rmse_ratio"]) < 1.0
        for row in validated_rows
    )
    result: list[dict[str, Any]] = []
    for row_order, raw in enumerate(validated_rows, start=1):
        dataset_key = str(raw["dataset"])
        actual = raw["actual_utility_calls_by_repeat"]
        new_old_ratio = float(raw["normalized_to_legacy_rmse_ratio"])
        result.append(
            {
                "row_order": row_order,
                "dataset": DATASET_LABELS[dataset_key],
                "dataset_key": dataset_key,
                "dataset_order": DATASET_ORDER.index(dataset_key) + 1,
                "budget_index": int(raw["budget_index"]) + 1,
                "total_utility_calls": int(raw["total_utility_calls"]),
                "repeats": 3,
                "legacy_greedy_rmse": float(
                    raw["legacy_raw_lambda_0p1_greedy_rmse"]
                ),
                "normalized_greedy_rmse": float(
                    raw["normalized_lambda0_1_greedy_rmse"]
                ),
                "inside_orbit_rmse": float(raw["inside_orbit_rmse"]),
                "normalized_to_legacy_rmse_ratio": new_old_ratio,
                "normalized_vs_legacy_improvement_rate": 1.0 - new_old_ratio,
                "normalized_vs_legacy_outcome": (
                    "improved"
                    if new_old_ratio < 1.0
                    else "tied"
                    if new_old_ratio == 1.0
                    else "worsened"
                ),
                "normalized_to_orbit_rmse_ratio": float(
                    raw["normalized_to_orbit_rmse_ratio"]
                ),
                "legacy_slice_mean_direction_rms": float(
                    raw["legacy_slice_mean_direction_rms"]
                ),
                "normalized_slice_mean_direction_rms": float(
                    raw["normalized_slice_mean_direction_rms"]
                ),
                "normalized_to_legacy_slice_mean_direction_rms_ratio": float(
                    raw[
                        "normalized_to_legacy_slice_mean_direction_rms_ratio"
                    ]
                ),
                "legacy_frobenius_discrepancy": float(
                    raw["legacy_frobenius_discrepancy"]
                ),
                "normalized_frobenius_discrepancy": float(
                    raw["normalized_frobenius_discrepancy"]
                ),
                "normalized_to_legacy_frobenius_discrepancy_ratio": float(
                    raw[
                        "normalized_to_legacy_frobenius_discrepancy_ratio"
                    ]
                ),
                "legacy_calls_repeat_1": int(actual["legacy_greedy"][0]),
                "legacy_calls_repeat_2": int(actual["legacy_greedy"][1]),
                "legacy_calls_repeat_3": int(actual["legacy_greedy"][2]),
                "normalized_calls_repeat_1": int(
                    actual["normalized_greedy"][0]
                ),
                "normalized_calls_repeat_2": int(
                    actual["normalized_greedy"][1]
                ),
                "normalized_calls_repeat_3": int(
                    actual["normalized_greedy"][2]
                ),
                "orbit_calls_repeat_1": int(actual["inside_orbit"][0]),
                "orbit_calls_repeat_2": int(actual["inside_orbit"][1]),
                "orbit_calls_repeat_3": int(actual["inside_orbit"][2]),
                "legacy_mean_balance_mode": "raw",
                "legacy_raw_lambda": 0.1,
                "normalized_mean_balance_mode": "normalized",
                "normalized_lambda0": 1.0,
                "overall_geometric_mean_new_old_ratio": overall_new_old,
                "overall_geometric_mean_improvement_rate": 1.0
                - overall_new_old,
                "overall_improved_point_count": int(
                    overall["improved_budget_point_count"]
                ),
                "overall_total_point_count": len(validated_rows),
                "overall_geometric_mean_new_orbit_ratio": overall_new_orbit,
                "overall_new_beats_orbit_point_count": new_beats_orbit,
            }
        )
    return result


def _ratio_phrase(ratio: float) -> str:
    change = abs(1.0 - ratio)
    if ratio < 1.0:
        return f"a {change:.1%} geometric-mean RMSE reduction"
    if ratio > 1.0:
        return f"a {change:.1%} geometric-mean RMSE increase"
    return "no geometric-mean RMSE change"


def _safe_input_paths(
    summary: Mapping[str, Any], summary_path: Path
) -> tuple[str, list[str]]:
    safe_summary = _artifact_path(summary_path, path="summary_path")
    datasets = summary["datasets"]
    inputs = [safe_summary]
    for dataset in DATASET_ORDER:
        payload = datasets[dataset]
        inputs.extend(
            [
                _artifact_path(
                    payload["normalized_report"],
                    path=f"datasets.{dataset}.normalized_report",
                ),
                _artifact_path(
                    payload["legacy_report"],
                    path=f"datasets.{dataset}.legacy_report",
                ),
            ]
        )
    _require(len(set(inputs)) == 7, "summary and six report paths must be distinct")
    return safe_summary, inputs


def _structural_check(artifact: Mapping[str, Any]) -> None:
    _require(artifact.get("surface") == "report", "artifact surface must be report")
    manifest = artifact.get("manifest")
    snapshot = artifact.get("snapshot")
    _require(isinstance(manifest, Mapping), "manifest must be an object")
    _require(isinstance(snapshot, Mapping), "snapshot must be an object")
    title = manifest.get("title")
    blocks = manifest.get("blocks")
    _require(isinstance(title, str) and title, "manifest title is missing")
    _require(isinstance(blocks, list) and blocks, "manifest blocks are missing")
    _require(
        blocks[0] == {"id": "title", "type": "markdown", "body": f"# {title}"},
        "first block must be the matching visible title",
    )
    _require(
        str(blocks[1].get("body", "")).startswith("## Technical summary"),
        "technical summary must immediately follow the title",
    )
    cards = manifest.get("cards")
    charts = manifest.get("charts")
    tables = manifest.get("tables")
    _require(isinstance(cards, list) and len(cards) == 3, "three cards are required")
    _require(isinstance(charts, list) and charts, "at least one chart is required")
    _require(isinstance(tables, list) and len(tables) == 1, "one table is required")
    datasets = snapshot.get("datasets")
    _require(isinstance(datasets, Mapping), "snapshot datasets are missing")
    _require(
        set(datasets) == {"mean_balance_results"}
        and len(datasets["mean_balance_results"]) == 15,
        "snapshot must contain the exact 15-row result dataset",
    )
    source = next(
        (
            item
            for item in manifest.get("sources", [])
            if item.get("id") == "mean_balance_transform"
        ),
        None,
    )
    _require(isinstance(source, Mapping), "canonical transformation source is missing")
    _require(
        source.get("query", {}).get("sql") == SOURCE_SQL,
        "canonical source SQL is not runnable artifact SQL",
    )
    source_ids = {item.get("id") for item in manifest.get("sources", [])}
    for asset in [*cards, *charts, *tables]:
        _require(
            asset.get("sourceId") in source_ids
            and isinstance(asset.get("source"), Mapping),
            f"{asset.get('id')} has no canonical provenance",
        )
    block_types = {block.get("type") for block in blocks}
    _require(
        {"markdown", "metric-strip", "chart", "table"}.issubset(block_types),
        "report reading path omits a required native block",
    )


def build_artifact(summary_path: Path) -> dict[str, Any]:
    summary = _read(summary_path)
    validated_rows = _validate_summary(summary)
    rows = _result_rows(summary, validated_rows)
    safe_summary_path, input_paths = _safe_input_paths(summary, summary_path)
    generated_at = datetime.now(timezone.utc).isoformat()
    overall = _summarize_rows(validated_rows)
    overall_new_old = float(
        overall["normalized_to_legacy_geometric_mean_rmse_ratio"]
    )
    overall_new_orbit = float(
        overall["normalized_to_orbit_geometric_mean_rmse_ratio"]
    )
    improved = int(overall["improved_budget_point_count"])
    total_points = len(validated_rows)
    new_beats_orbit = sum(
        float(row["normalized_to_orbit_rmse_ratio"]) < 1.0
        for row in validated_rows
    )
    best = overall["best_budget_point"]
    worst = overall["worst_budget_point"]
    slice_ratio = _geometric_mean(
        [
            float(row["normalized_to_legacy_slice_mean_direction_rms_ratio"])
            for row in validated_rows
        ]
    )
    frobenius_ratio = _geometric_mean(
        [
            float(row["normalized_to_legacy_frobenius_discrepancy_ratio"])
            for row in validated_rows
        ]
    )
    slice_improved = sum(
        float(row["normalized_to_legacy_slice_mean_direction_rms_ratio"]) < 1.0
        for row in validated_rows
    )
    frobenius_worsened = sum(
        float(row["normalized_to_legacy_frobenius_discrepancy_ratio"]) > 1.0
        for row in validated_rows
    )

    source = {
        "id": "mean_balance_transform",
        "label": "Validated mean-balance before/after transformation",
        "path": "experiments/build_mean_balance_fix_report.py",
        "query": {
            "id": "mean_balance_transform",
            "engine": "artifact_snapshot_sql",
            "language": "sql",
            "executed_at": generated_at,
            "sql": SOURCE_SQL,
            "description": (
                "Read all 15 matched Wine, Airport, and Voting budget points "
                "from the strictly validated before/after summary."
            ),
            "tables_used": ["artifact.mean_balance_results"],
            "transformation_language": "python",
            "transformation_code": (
                "from pathlib import Path\n"
                "from experiments.build_mean_balance_fix_report import "
                "_read, _result_rows, _validate_summary\n"
                f"summary = _read(Path({safe_summary_path!r}))\n"
                "rows = _result_rows(summary, _validate_summary(summary))"
            ),
            "input_files": input_paths,
            "filters": [
                "Datasets: Wine, Airport, and U.S. Electoral Voting",
                "Five matched utility-call budgets per dataset",
                "Exactly three repeats per method-budget point",
                "Normalized Greedy uses lambda0=1; legacy Greedy uses raw lambda=0.1",
            ],
            "metric_definitions": [
                (
                    "Aggregate RMSE = sqrt(mean over three repeats and all player "
                    "coordinates of squared Shapley error)"
                ),
                (
                    "Normalized/legacy ratio = normalized Greedy aggregate RMSE / "
                    "legacy Greedy aggregate RMSE"
                ),
                (
                    "RMSE improvement rate = 1 - normalized/legacy ratio; positive "
                    "values favor normalization"
                ),
                (
                    "Normalized/Orbit ratio = normalized Greedy aggregate RMSE / "
                    "INSIDE-Orbit aggregate RMSE"
                ),
                (
                    "Geometric-mean ratio = exp(mean of log pointwise RMSE ratios "
                    "across the stated points)"
                ),
                (
                    "Slice-mean RMS ratio and frame-Frobenius ratio use the "
                    "three-repeat mean design diagnostic at each point"
                ),
            ],
        },
    }
    raw_sources = [
        {
            "id": "combined_summary_input",
            "label": "Strictly validated three-dataset combined summary",
            "path": safe_summary_path,
        }
    ]
    for dataset in DATASET_ORDER:
        raw_sources.extend(
            [
                {
                    "id": f"{dataset}_normalized_input",
                    "label": f"{DATASET_LABELS[dataset]} normalized report",
                    "path": _artifact_path(
                        summary["datasets"][dataset]["normalized_report"],
                        path=f"datasets.{dataset}.normalized_report",
                    ),
                },
                {
                    "id": f"{dataset}_legacy_input",
                    "label": f"{DATASET_LABELS[dataset]} legacy report",
                    "path": _artifact_path(
                        summary["datasets"][dataset]["legacy_report"],
                        path=f"datasets.{dataset}.legacy_report",
                    ),
                },
            ]
        )
    sources = [source, *raw_sources]

    cards = [
        {
            "id": "overall_new_old_ratio",
            "dataset": "mean_balance_results",
            "sourceId": "mean_balance_transform",
            "filter": {"row_order": 1},
            "description": (
                "Geometric mean of normalized-to-legacy Greedy RMSE ratios "
                "over all 15 matched points; lower than 1 favors normalization."
            ),
            "metrics": [
                {
                    "label": "Geometric RMSE ratio",
                    "field": "overall_geometric_mean_new_old_ratio",
                    "format": "number",
                },
                {
                    "label": "RMSE improvement",
                    "field": "overall_geometric_mean_improvement_rate",
                    "format": "percent",
                    "signed": True,
                },
            ],
        },
        {
            "id": "improved_points",
            "dataset": "mean_balance_results",
            "sourceId": "mean_balance_transform",
            "filter": {"row_order": 1},
            "description": (
                "Matched budget points where normalized Greedy has lower RMSE "
                "than the legacy raw-lambda implementation."
            ),
            "metrics": [
                {
                    "label": "Improved budget points",
                    "field": "overall_improved_point_count",
                    "format": "number",
                },
                {
                    "label": "Total points",
                    "field": "overall_total_point_count",
                    "format": "number",
                },
            ],
        },
        {
            "id": "overall_new_orbit_ratio",
            "dataset": "mean_balance_results",
            "sourceId": "mean_balance_transform",
            "filter": {"row_order": 1},
            "description": (
                "Geometric mean of normalized Greedy RMSE divided by the "
                "INSIDE-Orbit RMSE at the same 15 budgets."
            ),
            "metrics": [
                {
                    "label": "Greedy / Orbit ratio",
                    "field": "overall_geometric_mean_new_orbit_ratio",
                    "format": "number",
                },
                {
                    "label": "Greedy wins",
                    "field": "overall_new_beats_orbit_point_count",
                    "format": "number",
                },
            ],
        },
    ]
    chart = {
        "id": "new_old_ratio_chart",
        "type": "line",
        "dataset": "mean_balance_results",
        "sourceId": "mean_balance_transform",
        "title": "Normalized-to-legacy Greedy RMSE ratio",
        "subtitle": "Fifteen matched budget points; values below 1 favor normalization.",
        "intent": "trend",
        "question": "Does normalized mean balancing improve Greedy across budgets?",
        "rationale": (
            "A ratio makes the three datasets comparable despite different RMSE scales."
        ),
        "encodings": {
            "x": {
                "field": "total_utility_calls",
                "type": "quantitative",
                "label": "Total utility calls",
                "format": "number",
            },
            "y": {
                "field": "normalized_to_legacy_rmse_ratio",
                "type": "quantitative",
                "label": "Normalized / legacy RMSE",
                "format": "number",
            },
            "color": {
                "field": "dataset",
                "type": "nominal",
                "label": "Dataset",
            },
            "tooltip": [
                {
                    "field": "normalized_vs_legacy_improvement_rate",
                    "type": "quantitative",
                    "label": "RMSE improvement",
                    "format": "percent",
                },
                {
                    "field": "normalized_to_orbit_rmse_ratio",
                    "type": "quantitative",
                    "label": "Normalized Greedy / Orbit",
                    "format": "number",
                },
                {
                    "field": "normalized_to_legacy_slice_mean_direction_rms_ratio",
                    "type": "quantitative",
                    "label": "Slice-mean RMS ratio",
                    "format": "number",
                },
            ],
        },
    }
    table = {
        "id": "exact_results_table",
        "dataset": "mean_balance_results",
        "sourceId": "mean_balance_transform",
        "title": "Exact matched-budget results and design diagnostics",
        "subtitle": "All 15 validated points; lower RMSE and diagnostic ratios are better.",
        "defaultSort": {"field": "row_order", "direction": "asc"},
        "columns": [
            {"field": "row_order", "label": "#", "format": "number"},
            {"field": "dataset", "label": "Dataset", "type": "text"},
            {
                "field": "total_utility_calls",
                "label": "Total calls",
                "format": "number",
            },
            {
                "field": "legacy_greedy_rmse",
                "label": "Legacy RMSE",
                "format": "number",
            },
            {
                "field": "normalized_greedy_rmse",
                "label": "Normalized RMSE",
                "format": "number",
            },
            {
                "field": "normalized_to_legacy_rmse_ratio",
                "label": "New / old",
                "format": "number",
            },
            {
                "field": "normalized_vs_legacy_improvement_rate",
                "label": "Improvement",
                "format": "percent",
            },
            {
                "field": "inside_orbit_rmse",
                "label": "Orbit RMSE",
                "format": "number",
            },
            {
                "field": "normalized_to_orbit_rmse_ratio",
                "label": "New / Orbit",
                "format": "number",
            },
            {
                "field": "legacy_slice_mean_direction_rms",
                "label": "Old slice RMS",
                "format": "number",
            },
            {
                "field": "normalized_slice_mean_direction_rms",
                "label": "New slice RMS",
                "format": "number",
            },
            {
                "field": "normalized_to_legacy_slice_mean_direction_rms_ratio",
                "label": "Slice RMS ratio",
                "format": "number",
            },
            {
                "field": "legacy_frobenius_discrepancy",
                "label": "Old frame Frobenius",
                "format": "number",
            },
            {
                "field": "normalized_frobenius_discrepancy",
                "label": "New frame Frobenius",
                "format": "number",
            },
            {
                "field": "normalized_to_legacy_frobenius_discrepancy_ratio",
                "label": "Frame ratio",
                "format": "number",
            },
        ],
    }

    title = "Normalized Mean-Balance Audit for INSIDE-Greedy"
    technical_summary = (
        "## Technical summary\n\n"
        f"Across {total_points} matched Wine, Airport, and Voting budget points, "
        f"normalized `lambda0=1` improves Greedy RMSE at {improved} points and yields "
        f"{_ratio_phrase(overall_new_old)} (new/old ratio {overall_new_old:.3f}). "
        f"Its geometric-mean RMSE ratio to INSIDE-Orbit is {overall_new_orbit:.3f}, "
        f"with Greedy lower at {new_beats_orbit}/{total_points} points. These are "
        "descriptive three-repeat results, not a universal dominance guarantee."
    )
    finding_body = (
        "## The normalized penalty changes Greedy RMSE across matched budgets\n\n"
        f"The best point is {DATASET_LABELS[str(best['dataset'])]} at "
        f"{int(best['total_utility_calls']):,} calls (new/old "
        f"{float(best['normalized_to_legacy_rmse_ratio']):.3f}); the worst is "
        f"{DATASET_LABELS[str(worst['dataset'])]} at "
        f"{int(worst['total_utility_calls']):,} calls (new/old "
        f"{float(worst['normalized_to_legacy_rmse_ratio']):.3f}). Read the line "
        "chart against 1: values below 1 favor normalized scaling, while values "
        "above 1 favor the legacy raw coefficient."
    )
    mechanism_body = (
        "## First-moment balance improves with a possible frame trade-off\n\n"
        f"Across the 15 points, the geometric-mean normalized/legacy ratio is "
        f"{slice_ratio:.3f} for `slice_mean_direction_rms` and {frobenius_ratio:.3f} "
        f"for frame Frobenius discrepancy. Slice-mean RMS is lower at "
        f"{slice_improved}/{total_points} points; frame Frobenius is higher at "
        f"{frobenius_worsened}/{total_points}. This directly audits the intended "
        "effect of normalization while exposing any second-moment coverage cost; "
        "it does not assume that either diagnostic alone determines RMSE."
    )
    exact_table_body = (
        "## Exact values and call-accounting evidence\n\n"
        "The table retains every matched point, all three RMSE values, both headline "
        "ratios, and the old/new design diagnostics. The source summary separately "
        "verified that Greedy and Orbit used each target call budget in all three repeats."
    )
    definitions_body = (
        "## Scope, formulas, and definitions\n\n"
        "For estimates `phi_hat^(r)` and reference `phi*`, aggregate RMSE is "
        "`sqrt[(1/(3n)) sum_r sum_i (phi_hat_i^(r)-phi_i*)^2]`; lower is better. "
        "The before/after ratio is `R_old = RMSE_normalized / RMSE_legacy`, and "
        "improvement is `1-R_old`. The Orbit ratio replaces the denominator with "
        "INSIDE-Orbit RMSE. Ratios are aggregated with `exp(mean(log R))`, so each "
        "dataset-budget point has equal multiplicative weight. Normalized Greedy uses "
        "`lambda_eff = lambda0 * (1-1/(n-1)) * mean_t[w_t^2]` with `lambda0=1`; "
        "legacy Greedy uses the unnormalized raw coefficient `0.1`. Slice-mean RMS "
        "measures first-moment imbalance across fixed-size slices; frame Frobenius "
        "measures second-moment deviation from the target frame operator."
    )
    methodology_body = (
        "## Validation and comparison methodology\n\n"
        "The upstream summary reruns the strict common validator for every normalized "
        "and legacy report, then enforces identical ground truth, five matching total-call "
        "budgets, the common eight-method identity, three retained estimate vectors per "
        "method-budget, finite recomputed RMSE, and physical call accounting. This report "
        "recomputes every displayed ratio from the three stored RMSE values and averages "
        "each design diagnostic over the same three repeats. Airport and Voting use exact "
        "analytic Shapley truth; Wine uses its audited high-budget Monte Carlo reference."
    )
    limitations_body = (
        "## Limitations, uncertainty, and robustness\n\n"
        "Three repeats give only coarse uncertainty, and this report does not attach a "
        "confidence interval to ratios or their geometric mean. Wine's reference has Monte "
        "Carlo error, although its uncertainty metadata and larger call budget were audited. "
        "The normalized and legacy runs are matched by dataset and budget, but the result is "
        "descriptive unless their random designs are analyzed as paired observations. A "
        "lower slice-mean diagnostic can coincide with a higher frame discrepancy, and no "
        "dependent coalition design universally dominates for arbitrary cooperative games."
    )
    recommendation = (
        "retain normalized `lambda0=1` as the default candidate"
        if overall_new_old < 1.0 and improved > total_points / 2
        else "keep normalized `lambda0=1` as an experimental candidate"
    )
    next_steps_body = (
        "## Recommended next steps\n\n"
        f"1. Based on this three-dataset evidence, {recommendation}; keep raw mode only "
        "for exact legacy reproduction.\n"
        "2. Increase repeats before making a general performance claim and report paired "
        "RMSE differences or bootstrap intervals at each budget.\n"
        "3. Continue reporting INSIDE-Orbit because normalization fixes coefficient scale, "
        "not the entire Greedy-versus-Orbit design difference.\n"
        "4. If tuning `lambda0`, select it on separate games or held-out budgets and retain "
        "both slice-mean and frame-discrepancy diagnostics."
    )
    further_questions_body = (
        "## Further questions\n\n"
        "Does the RMSE gain persist with 20 or more repeats and independent Wine truth "
        "replications? Which normalized `lambda0` best balances first- and second-moment "
        "coverage without dataset-specific tuning? Can partial-orbit or constrained Greedy "
        "designs enforce slice means exactly while preserving Greedy's frame advantage?"
    )

    manifest: dict[str, Any] = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": (
            "Strict before/after audit of normalized versus raw mean balancing for "
            "INSIDE-Greedy on Wine, Airport, and Voting."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "cards": cards,
        "charts": [chart],
        "tables": [table],
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "sourceId": "mean_balance_transform",
                "body": technical_summary,
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "overall_new_old_ratio",
                    "improved_points",
                    "overall_new_orbit_ratio",
                ],
            },
            {
                "id": "key_finding",
                "type": "markdown",
                "sourceId": "mean_balance_transform",
                "body": finding_body,
            },
            {
                "id": "ratio_chart_block",
                "type": "chart",
                "chartId": "new_old_ratio_chart",
            },
            {
                "id": "mechanism",
                "type": "markdown",
                "sourceId": "mean_balance_transform",
                "body": mechanism_body,
            },
            {
                "id": "exact_table_context",
                "type": "markdown",
                "sourceId": "mean_balance_transform",
                "body": exact_table_body,
            },
            {
                "id": "exact_table_block",
                "type": "table",
                "tableId": "exact_results_table",
            },
            {
                "id": "definitions",
                "type": "markdown",
                "body": definitions_body,
            },
            {
                "id": "methodology",
                "type": "markdown",
                "sourceId": "mean_balance_transform",
                "body": methodology_body,
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": limitations_body,
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "sourceId": "mean_balance_transform",
                "body": next_steps_body,
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": further_questions_body,
            },
        ],
    }
    for asset in [*cards, chart, table]:
        asset["source"] = source

    snapshot = {
        "version": 1,
        "status": "ready",
        "generatedAt": generated_at,
        "datasets": {"mean_balance_results": rows},
    }
    artifact = {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }
    _structural_check(artifact)
    return artifact


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifact = build_artifact(args.summary)
    output = args.output or args.summary.with_name(
        f"{args.summary.stem}_report_artifact.json"
    )
    _atomic_write(output, artifact)
    print(f"saved {output}")


if __name__ == "__main__":
    main()

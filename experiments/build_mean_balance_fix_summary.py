"""Build one strictly validated Wine/Airport/Voting mean-balance summary.

The command consumes the normalized and legacy reports for each benchmark.  It
does not trust their stored headline metrics in isolation: every input first
passes the common INSIDE report validator and the normalized-vs-legacy pair
validator used by :mod:`experiments.compare_inside_mean_balance_reports`.

No utility functions are evaluated here.  The output is a compact audit table
containing the three curves relevant to the mean-balance fix plus deterministic
cross-dataset summaries.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from experiments.compare_inside_mean_balance_reports import build_comparison
from experiments.plot_inside_comparison import _ordered_rows
from experiments.validate_inside_comparison import (
    EXPECTED_BUDGETS,
    EXPECTED_REPEATS,
    _actual_call_list,
)


DATASET_ORDER = ("wine", "airport", "voting")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(
        math.isfinite(result) and result > 0.0,
        f"{path} must be finite and positive",
    )
    return result


def _positive_integer(value: Any, *, path: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool),
        f"{path} must be an integer",
    )
    result = int(value)
    _require(result > 0, f"{path} must be positive")
    return result


def _geometric_mean(values: Sequence[float], *, path: str) -> float:
    _require(bool(values), f"{path} must not be empty")
    checked = [_positive(value, path=f"{path}[{index}]") for index, value in enumerate(values)]
    return math.exp(math.fsum(math.log(value) for value in checked) / len(checked))


def _point_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset": str(row["dataset"]),
        "total_utility_calls": int(row["total_utility_calls"]),
        "normalized_to_legacy_rmse_ratio": float(
            row["normalized_to_legacy_rmse_ratio"]
        ),
        "normalized_vs_legacy_improvement_percent": float(
            row["normalized_vs_legacy_improvement_percent"]
        ),
    }


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "summary rows must not be empty")
    legacy_ratios = [
        _positive(
            row["normalized_to_legacy_rmse_ratio"],
            path=f"rows[{index}].normalized_to_legacy_rmse_ratio",
        )
        for index, row in enumerate(rows)
    ]
    orbit_ratios = [
        _positive(
            row["normalized_to_orbit_rmse_ratio"],
            path=f"rows[{index}].normalized_to_orbit_rmse_ratio",
        )
        for index, row in enumerate(rows)
    ]
    legacy_geometric_mean = _geometric_mean(
        legacy_ratios, path="normalized_to_legacy_ratios"
    )
    orbit_geometric_mean = _geometric_mean(
        orbit_ratios, path="normalized_to_orbit_ratios"
    )
    improved = sum(ratio < 1.0 for ratio in legacy_ratios)
    tied = sum(ratio == 1.0 for ratio in legacy_ratios)
    worsened = len(legacy_ratios) - improved - tied
    best_index = min(range(len(rows)), key=legacy_ratios.__getitem__)
    worst_index = max(range(len(rows)), key=legacy_ratios.__getitem__)
    return {
        "budget_point_count": len(rows),
        "improved_budget_point_count": improved,
        "tied_budget_point_count": tied,
        "worsened_budget_point_count": worsened,
        "normalized_to_legacy_geometric_mean_rmse_ratio": legacy_geometric_mean,
        "normalized_vs_legacy_geometric_mean_improvement_percent": 100.0
        * (1.0 - legacy_geometric_mean),
        "normalized_to_orbit_geometric_mean_rmse_ratio": orbit_geometric_mean,
        "best_budget_point": _point_identity(rows[best_index]),
        "worst_budget_point": _point_identity(rows[worst_index]),
    }


def _method_calls(
    row: Mapping[str, Any],
    *,
    method: str,
    target: int,
    path: str,
) -> list[int]:
    methods = row.get("methods")
    _require(isinstance(methods, Mapping), f"{path}.methods must be an object")
    summary = methods.get(method)
    _require(
        isinstance(summary, Mapping),
        f"{path}.methods.{method} must be an object",
    )
    actual = _actual_call_list(
        summary,
        path=f"{path}.methods.{method}.actual_utility_calls_by_repeat",
    )
    _require(
        len(actual) == EXPECTED_REPEATS and all(value == target for value in actual),
        f"{path}.methods.{method} must spend the target in all three repeats",
    )
    return list(actual)


def _mean_design_diagnostics(
    row: Mapping[str, Any],
    *,
    method: str,
    path: str,
) -> dict[str, float]:
    methods = row.get("methods")
    _require(isinstance(methods, Mapping), f"{path}.methods must be an object")
    summary = methods.get(method)
    _require(
        isinstance(summary, Mapping),
        f"{path}.methods.{method} must be an object",
    )
    diagnostics = summary.get("diagnostics_by_repeat")
    _require(
        isinstance(diagnostics, list) and len(diagnostics) == EXPECTED_REPEATS,
        f"{path}.methods.{method}.diagnostics_by_repeat must contain three objects",
    )
    values = {
        "slice_mean_direction_rms": [],
        "frobenius_discrepancy": [],
    }
    for repeat, diagnostic in enumerate(diagnostics):
        _require(
            isinstance(diagnostic, Mapping),
            f"{path}.methods.{method}.diagnostics_by_repeat[{repeat}] must be an object",
        )
        design = diagnostic.get("design_diagnostics")
        _require(
            isinstance(design, Mapping),
            f"{path}.methods.{method}.diagnostics_by_repeat[{repeat}] has no design diagnostics",
        )
        for field in values:
            value = _positive(
                design.get(field),
                path=(
                    f"{path}.methods.{method}.diagnostics_by_repeat[{repeat}]"
                    f".design_diagnostics.{field}"
                ),
            )
            values[field].append(value)
    return {
        field: math.fsum(repeats) / EXPECTED_REPEATS
        for field, repeats in values.items()
    }


def _dataset_rows(
    dataset: str,
    comparison: Mapping[str, Any],
    normalized: Mapping[str, Any],
    legacy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    comparison_rows = comparison.get("rows")
    _require(
        isinstance(comparison_rows, list)
        and len(comparison_rows) == EXPECTED_BUDGETS,
        f"{dataset} comparison must contain exactly five budgets",
    )
    normalized_rows = _ordered_rows(normalized)
    legacy_rows = _ordered_rows(legacy)
    _require(
        len(normalized_rows) == len(legacy_rows) == len(comparison_rows),
        f"{dataset} report rows do not align",
    )

    result: list[dict[str, Any]] = []
    previous_calls = 0
    for index, (metrics, current, old) in enumerate(
        zip(comparison_rows, normalized_rows, legacy_rows, strict=True)
    ):
        _require(
            isinstance(metrics, Mapping),
            f"{dataset}.comparison.rows[{index}] must be an object",
        )
        target = _positive_integer(
            metrics.get("total_utility_calls"),
            path=f"{dataset}.rows[{index}].total_utility_calls",
        )
        _require(target > previous_calls, f"{dataset} budgets must strictly increase")
        previous_calls = target
        _require(
            int(current.get("total_utility_calls_per_estimate", -1)) == target
            and int(old.get("total_utility_calls_per_estimate", -1)) == target,
            f"{dataset} normalized and legacy target calls differ at row {index}",
        )

        legacy_rmse = _positive(
            metrics.get("legacy_raw_lambda_0p1_greedy_rmse"),
            path=f"{dataset}.rows[{index}].legacy_greedy_rmse",
        )
        normalized_rmse = _positive(
            metrics.get("normalized_lambda0_1_greedy_rmse"),
            path=f"{dataset}.rows[{index}].normalized_greedy_rmse",
        )
        orbit_rmse = _positive(
            metrics.get("inside_orbit_rmse"),
            path=f"{dataset}.rows[{index}].inside_orbit_rmse",
        )
        legacy_ratio = normalized_rmse / legacy_rmse
        orbit_ratio = normalized_rmse / orbit_rmse
        stored_change = float(
            metrics.get("normalized_vs_legacy_fractional_rmse_change", math.nan)
        )
        stored_orbit_ratio = float(
            metrics.get("normalized_greedy_to_orbit_rmse_ratio", math.nan)
        )
        _require(
            math.isfinite(stored_change)
            and math.isclose(
                stored_change, legacy_ratio - 1.0, rel_tol=2e-14, abs_tol=2e-14
            ),
            f"{dataset} row {index} has an inconsistent legacy ratio",
        )
        _require(
            math.isfinite(stored_orbit_ratio)
            and math.isclose(
                stored_orbit_ratio, orbit_ratio, rel_tol=2e-14, abs_tol=2e-14
            ),
            f"{dataset} row {index} has an inconsistent Orbit ratio",
        )

        legacy_design = _mean_design_diagnostics(
            old,
            method="inside_greedy",
            path=f"{dataset}.legacy.rows[{index}]",
        )
        normalized_design = _mean_design_diagnostics(
            current,
            method="inside_greedy",
            path=f"{dataset}.normalized.rows[{index}]",
        )

        row = {
            "dataset": dataset,
            "budget_index": index,
            "total_utility_calls": target,
            "legacy_raw_lambda_0p1_greedy_rmse": legacy_rmse,
            "normalized_lambda0_1_greedy_rmse": normalized_rmse,
            "inside_orbit_rmse": orbit_rmse,
            "normalized_to_legacy_rmse_ratio": legacy_ratio,
            "normalized_vs_legacy_improvement_percent": 100.0
            * (1.0 - legacy_ratio),
            "normalized_to_orbit_rmse_ratio": orbit_ratio,
            "legacy_slice_mean_direction_rms": legacy_design[
                "slice_mean_direction_rms"
            ],
            "normalized_slice_mean_direction_rms": normalized_design[
                "slice_mean_direction_rms"
            ],
            "normalized_to_legacy_slice_mean_direction_rms_ratio": (
                normalized_design["slice_mean_direction_rms"]
                / legacy_design["slice_mean_direction_rms"]
            ),
            "legacy_frobenius_discrepancy": legacy_design[
                "frobenius_discrepancy"
            ],
            "normalized_frobenius_discrepancy": normalized_design[
                "frobenius_discrepancy"
            ],
            "normalized_to_legacy_frobenius_discrepancy_ratio": (
                normalized_design["frobenius_discrepancy"]
                / legacy_design["frobenius_discrepancy"]
            ),
            "actual_utility_calls_by_repeat": {
                "legacy_greedy": _method_calls(
                    old,
                    method="inside_greedy",
                    target=target,
                    path=f"{dataset}.legacy.rows[{index}]",
                ),
                "normalized_greedy": _method_calls(
                    current,
                    method="inside_greedy",
                    target=target,
                    path=f"{dataset}.normalized.rows[{index}]",
                ),
                "inside_orbit": _method_calls(
                    current,
                    method="inside_orbit",
                    target=target,
                    path=f"{dataset}.normalized.rows[{index}]",
                ),
            },
        }
        result.append(row)
    return result


def build_summary(
    report_pairs: Mapping[
        str, tuple[Mapping[str, Any], Mapping[str, Any]]
    ],
    *,
    report_paths: Mapping[str, tuple[Path, Path]],
) -> dict[str, Any]:
    """Return one audited 15-point before/after summary.

    ``report_pairs`` values are ``(normalized, legacy)``.  Exactly Wine,
    Airport, and Voting must be supplied so a partial run cannot be mistaken
    for the requested three-benchmark result.
    """
    _require(
        set(report_pairs) == set(DATASET_ORDER),
        "report_pairs must contain exactly wine, airport, and voting",
    )
    _require(
        set(report_paths) == set(DATASET_ORDER),
        "report_paths must contain exactly wine, airport, and voting",
    )

    datasets: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    total_validated_calls = 0
    for dataset in DATASET_ORDER:
        normalized, legacy = report_pairs[dataset]
        normalized_path, legacy_path = report_paths[dataset]
        comparison = build_comparison(
            normalized,
            legacy,
            normalized_path=normalized_path,
            legacy_path=legacy_path,
        )
        validation = comparison.get("validation")
        _require(
            isinstance(validation, Mapping),
            f"{dataset} comparison has no validation audit",
        )
        normalized_audit = validation.get("normalized")
        legacy_audit = validation.get("legacy")
        _require(
            isinstance(normalized_audit, Mapping)
            and isinstance(legacy_audit, Mapping),
            f"{dataset} comparison has incomplete validation audits",
        )
        _require(
            normalized_audit.get("dataset") == dataset
            and legacy_audit.get("dataset") == dataset,
            f"{dataset} input pair has the wrong dataset identity",
        )
        _require(
            normalized_audit.get("inside_greedy_balance_audit")
            == {"mode": "normalized", "coefficient": 1.0},
            f"{dataset} normalized method identity is wrong",
        )
        _require(
            legacy_audit.get("inside_greedy_balance_audit")
            == {"mode": "legacy_raw", "coefficient": 0.1},
            f"{dataset} legacy method identity is wrong",
        )
        rows = _dataset_rows(dataset, comparison, normalized, legacy)
        all_rows.extend(rows)
        dataset_summary = _summarize_rows(rows)
        normalized_calls = normalized_audit.get("call_accounting")
        legacy_calls = legacy_audit.get("call_accounting")
        _require(
            isinstance(normalized_calls, Mapping)
            and isinstance(legacy_calls, Mapping),
            f"{dataset} validation has no call accounting",
        )
        normalized_total = _positive_integer(
            normalized_calls.get("total_actual_utility_calls"),
            path=f"{dataset}.normalized.total_actual_utility_calls",
        )
        legacy_total = _positive_integer(
            legacy_calls.get("total_actual_utility_calls"),
            path=f"{dataset}.legacy.total_actual_utility_calls",
        )
        total_validated_calls += normalized_total + legacy_total
        datasets[dataset] = {
            "normalized_report": str(normalized_path.resolve()),
            "legacy_report": str(legacy_path.resolve()),
            "rows": rows,
            "summary": dataset_summary,
            "validation": {
                "normalized_status": normalized_audit.get("status"),
                "legacy_status": legacy_audit.get("status"),
                "normalized_total_actual_utility_calls": normalized_total,
                "legacy_total_actual_utility_calls": legacy_total,
            },
        }

    _require(
        len(all_rows) == len(DATASET_ORDER) * EXPECTED_BUDGETS,
        "the combined summary must contain exactly 15 budget points",
    )
    return {
        "status": "ready_to_share",
        "experiment": "inside_greedy_normalized_mean_balance_three_dataset_summary",
        "method_identity": {
            "legacy_greedy": {
                "method_key": "inside_greedy",
                "mean_balance_mode": "raw",
                "raw_lambda": 0.1,
            },
            "normalized_greedy": {
                "method_key": "inside_greedy",
                "mean_balance_mode": "normalized",
                "lambda0": 1.0,
            },
            "orbit_reference": {"method_key": "inside_orbit"},
        },
        "validated_design": {
            "datasets": list(DATASET_ORDER),
            "budget_points_per_dataset": EXPECTED_BUDGETS,
            "repeats_per_budget_method": EXPECTED_REPEATS,
            "total_budget_points": len(all_rows),
            "normalized_and_legacy_total_actual_utility_calls_audited": (
                total_validated_calls
            ),
        },
        "datasets": datasets,
        "summary": _summarize_rows(all_rows),
    }


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
    for dataset in DATASET_ORDER:
        parser.add_argument(f"--{dataset}-normalized", type=Path, required=True)
        parser.add_argument(f"--{dataset}-legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_pairs: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    report_paths: dict[str, tuple[Path, Path]] = {}
    for dataset in DATASET_ORDER:
        normalized_path = getattr(args, f"{dataset}_normalized")
        legacy_path = getattr(args, f"{dataset}_legacy")
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
        legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
        report_pairs[dataset] = (normalized, legacy)
        report_paths[dataset] = (normalized_path, legacy_path)
    summary = build_summary(report_pairs, report_paths=report_paths)
    _atomic_write(args.output, summary)
    print(f"{summary['status']}: saved {args.output}")


if __name__ == "__main__":
    main()

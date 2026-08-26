"""Audit and plot legacy-vs-normalized INSIDE-Greedy reports.

This is a post-processing command: it performs no utility evaluation.  The
normalized report remains the canonical eight-method result, while the legacy
raw-lambda Greedy curve is overlaid solely as a controlled ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.plot_inside_comparison import (
    _dataset_name,
    _ordered_rows,
    plot_results,
    validate_plot_pair,
)
from experiments.validate_inside_comparison import validate_report


def comparison_rows(
    normalized: Mapping[str, Any], legacy: Mapping[str, Any]
) -> list[dict[str, float | int]]:
    """Return budget-wise Greedy before/after and Orbit reference values."""
    validate_plot_pair(normalized, legacy)
    normalized_rows = _ordered_rows(normalized)
    legacy_rows = _ordered_rows(legacy)
    rows: list[dict[str, float | int]] = []
    for current, previous in zip(
        normalized_rows, legacy_rows, strict=True
    ):
        current_rmse = float(
            current["methods"]["inside_greedy"]["aggregate_rmse"]
        )
        legacy_rmse = float(
            previous["methods"]["inside_greedy"]["aggregate_rmse"]
        )
        orbit_rmse = float(
            current["methods"]["inside_orbit"]["aggregate_rmse"]
        )
        rows.append(
            {
                "total_utility_calls": int(
                    current["total_utility_calls_per_estimate"]
                ),
                "legacy_raw_lambda_0p1_greedy_rmse": legacy_rmse,
                "normalized_lambda0_1_greedy_rmse": current_rmse,
                "inside_orbit_rmse": orbit_rmse,
                "normalized_vs_legacy_fractional_rmse_change": (
                    current_rmse / legacy_rmse - 1.0
                ),
                "normalized_greedy_to_orbit_rmse_ratio": (
                    current_rmse / orbit_rmse
                ),
            }
        )
    return rows


def build_comparison(
    normalized: Mapping[str, Any],
    legacy: Mapping[str, Any],
    *,
    normalized_path: Path,
    legacy_path: Path,
) -> dict[str, Any]:
    """Build a validated, provenance-preserving before/after summary."""
    normalized_audit = validate_report(normalized)
    legacy_audit = validate_report(legacy)
    if normalized_audit["inside_greedy_balance_audit"]["mode"] != "normalized":
        raise ValueError("the first report is not normalized INSIDE-Greedy")
    if legacy_audit["inside_greedy_balance_audit"]["mode"] != "legacy_raw":
        raise ValueError("the legacy report is not an old raw-lambda run")
    return {
        "status": "ready_to_share",
        "experiment": "inside_greedy_mean_balance_before_after",
        "dataset": _dataset_name(normalized),
        "normalized_report": str(normalized_path.resolve()),
        "legacy_report": str(legacy_path.resolve()),
        "method_identity": {
            "canonical_label": "INSIDE-Greedy",
            "normalized": {
                "mean_balance_mode": "normalized",
                "lambda0": 1.0,
            },
            "legacy_ablation": {
                "mean_balance_mode": "raw",
                "raw_lambda": 0.1,
            },
        },
        "rows": comparison_rows(normalized, legacy),
        "validation": {
            "normalized": normalized_audit,
            "legacy": legacy_audit,
        },
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
    parser.add_argument("normalized_report", type=Path)
    parser.add_argument("legacy_report", type=Path)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help="path without extension; defaults beside the normalized report",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    normalized = json.loads(
        args.normalized_report.read_text(encoding="utf-8")
    )
    legacy = json.loads(args.legacy_report.read_text(encoding="utf-8"))
    prefix = args.output_prefix or args.normalized_report.with_name(
        f"{args.normalized_report.stem}_before_after"
    )
    summary_path = prefix.with_suffix(".json")
    plot_path = prefix.with_suffix(".png")
    summary = build_comparison(
        normalized,
        legacy,
        normalized_path=args.normalized_report,
        legacy_path=args.legacy_report,
    )
    _atomic_write(summary_path, summary)
    png, pdf = plot_results(
        normalized,
        plot_path,
        legacy_report=legacy,
    )
    print(f"saved {summary_path}, {png}, and {pdf}")


if __name__ == "__main__":
    main()

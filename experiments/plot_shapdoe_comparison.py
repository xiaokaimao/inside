"""Plot the augmented baseline reports at observed coalition-call counts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np

from experiments.add_shapdoe_baselines import validate_report
from experiments.plot_inside_comparison import METHOD_ORDER, METHOD_STYLES, _calls_label


ADDED_STYLES = {
    "shapdoe_ls": {"label": "ShapDoE-LS", "color": "#7C3AED", "marker": "<", "linestyle": "--", "linewidth": 2.6},
    "shapdoe_coa": {"label": "ShapDoE-COA", "color": "#087F8C", "marker": ">", "linestyle": "-.", "linewidth": 2.6},
}


def _actual_calls(summary: Mapping[str, Any], row: Mapping[str, Any]) -> float:
    calls = summary.get("mean_actual_utility_calls")
    if calls is None:
        counts = summary.get("actual_utility_calls_by_repeat", summary.get("actual_utility_calls_per_estimate"))
        calls = float(np.mean(counts)) if counts is not None else row["total_utility_calls_per_estimate"]
    return float(calls)


def build_figure(report: Mapping[str, Any], *, allow_incomplete: bool = False) -> plt.Figure:
    validate_report(report, require_complete=not allow_incomplete)
    return _build_figure(report, ADDED_STYLES,
                         partial_note="Partial ShapDoE run; only completed budget points are shown")


def _build_figure(report: Mapping[str, Any], added_styles: Mapping[str, Any], *,
                  partial_note: str) -> plt.Figure:
    """Shared rendering after the caller has validated its report format."""
    figure, axis = plt.subplots(figsize=(9, 8.5))
    figure.subplots_adjust(left=.12, right=.97, bottom=.28, top=.93)
    styles = METHOD_STYLES | added_styles
    added_labels = {name: style["label"] for name, style in added_styles.items()}
    rows = sorted(report["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"])
    unavailable = []
    for method in (*METHOD_ORDER, *added_labels):
        points = []
        for row in rows:
            summary = row["methods"].get(method)
            if summary is None or summary.get("status") in {"budget_infeasible", "pending"}:
                continue
            calls = _actual_calls(summary, row)
            error = float(summary["aggregate_rmse"])
            if error <= 0:
                raise ValueError("zero RMSE needs a linear-axis figure; do not replace it with an arbitrary positive value")
            points.append((float(calls), error))
        if not points:
            if method in added_labels:
                unavailable.append(added_labels[method])
            continue
        style = styles[method]
        axis.plot(*np.asarray(points).T, **(style | {
            "markersize": style.get("markersize", 7.5),
            "markerfacecolor": style["color"] if method.startswith("inside_") else "white",
        }))
    axis.set(xscale="log", yscale="log", xlabel="Actual utility calls", ylabel="RMSE")
    axis.set_title(report["configuration"]["dataset"].capitalize(), loc="left", fontsize=17)
    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.grid(True, which="major", color="#D8DDE4", alpha=.8)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .09), ncol=3, fontsize=10)
    notes = []
    if unavailable:
        for name in unavailable:
            method = next(key for key, value in added_labels.items() if value == name)
            reason = ("no complete design fits these budgets" if all(
                row["methods"][method]["status"] == "budget_infeasible" for row in rows)
                else "no completed budget points")
            notes.append(f"{name}: {reason}")
    if report["status"] != "complete":
        notes.append(partial_note)
    if notes:
        figure.text(.5, .025, "\n".join(notes), ha="center", fontsize=9)
    return figure


def export_report(path: Path, *, allow_incomplete: bool = False) -> tuple[Path, Path, Path]:
    report = json.loads(path.read_text())
    figure = build_figure(report, allow_incomplete=allow_incomplete)
    return _export_artifacts(path, report, figure)


def _export_artifacts(path: Path, report: Mapping[str, Any], figure: plt.Figure) -> tuple[Path, Path, Path]:
    png, pdf, table = path.with_suffix(".png"), path.with_suffix(".pdf"), path.with_suffix(".csv")
    figure.savefig(png, dpi=220, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    with table.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "target_calls", "actual_calls", "rmse", "status", "minimum_call_budget"])
        for row in sorted(report["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"]):
            for method, summary in row["methods"].items():
                calls = (_actual_calls(summary, row) if summary.get("status", "complete") == "complete" else "")
                writer.writerow([method, row["total_utility_calls_per_estimate"],
                    calls, summary.get("aggregate_rmse", ""),
                    summary.get("status", "complete"), summary.get("minimum_call_budget", "")])
    return png, pdf, table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true", help="Label and plot only completed points in partial reports")
    args = parser.parse_args()
    for path in args.reports:
        print(export_report(path, allow_incomplete=args.allow_incomplete))


if __name__ == "__main__":
    main()

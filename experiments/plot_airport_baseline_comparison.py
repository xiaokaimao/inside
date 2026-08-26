"""Plot the Wine-style ten-method Airport comparison.

Chart contract
--------------
Question
    At the same physical utility-call scale, how does exact Shapley RMSE
    change for the ten methods used in the Wine main figure?
Form
    One near-square log-log line chart with five ordered budgets.  The focal
    Frame-OFA ratio line is emphasized; method families share palette roots
    and remain distinguishable by markers and line styles.
Surface
    Reproducible static Matplotlib PNG and PDF exports.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np

from experiments.airport_all_baselines import METHOD_ORDER


METHOD_STYLES: dict[str, dict[str, Any]] = {
    "official_ofa_fixed_ratio": {
        "label": "OFA ratio", "color": "#2563A6", "marker": "o",
        "linestyle": "-", "linewidth": 2.8,
    },
    "frame_orbit_ratio": {
        "label": "Frame-OFA ratio", "color": "#D97706", "marker": "D",
        "linestyle": "-", "linewidth": 4.2, "markerfacecolor": "#D97706",
    },
    "official_cc_basic": {
        "label": "CC", "color": "#5F6670", "marker": "X",
        "linestyle": ":", "linewidth": 2.5,
    },
    "stratified_marginal_mc": {
        "label": "Stratified MC", "color": "#8B929B", "marker": "v",
        "linestyle": "-.", "linewidth": 2.4,
    },
    "gels_shapley": {
        "label": "GELS-Shapley", "color": "#B48A00", "marker": "o",
        "linestyle": "-", "linewidth": 2.5,
    },
    "kernel_shap_sampled": {
        "label": "KernelSHAP", "color": "#B48A00", "marker": "s",
        "linestyle": "--", "linewidth": 2.5,
    },
    "group_testing": {
        "label": "Group Testing", "color": "#74831A", "marker": "^",
        "linestyle": "-.", "linewidth": 2.5,
    },
    "diff": {
        "label": "Diff", "color": "#74831A", "marker": "D",
        "linestyle": ":", "linewidth": 2.6,
    },
    "s_diff": {
        "label": "S-Diff", "color": "#74831A", "marker": "X",
        "linestyle": "-", "linewidth": 2.8,
    },
    "tmc_shapley": {
        "label": "TMC-Shapley", "color": "#383F48", "marker": "P",
        "linestyle": (0, (5, 2, 1, 2)), "linewidth": 2.5,
    },
}


def _ordered_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = list(report["results_by_inner_budget"].values())
    rows.sort(key=lambda row: row["total_utility_calls_per_estimate"])
    return rows


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "complete":
        raise ValueError("report status must be complete")
    if report["configuration"]["repeats"] != 3:
        raise ValueError("Wine-style comparison requires three repeats")
    if tuple(report["configuration"]["methods"]) != METHOD_ORDER:
        raise ValueError("method set or order differs from the Wine main figure")
    rows = _ordered_rows(report)
    if len(rows) != 5:
        raise ValueError("Wine-style comparison requires five budgets")
    target_calls = []
    actual_calls_by_method = {method: [] for method in METHOD_ORDER}
    for row in rows:
        target = float(row["total_utility_calls_per_estimate"])
        target_calls.append(target)
        if set(row["methods"]) != set(METHOD_ORDER):
            raise ValueError("a budget row has missing methods")
        for method in METHOD_ORDER:
            summary = row["methods"][method]
            rmse = float(summary["aggregate_rmse"])
            interval = summary["aggregate_rmse_bootstrap_95"]
            actual = float(summary["mean_actual_utility_calls"])
            estimates = np.asarray(summary["estimates"], dtype=np.float64)
            if not (math.isfinite(rmse) and rmse > 0.0):
                raise ValueError(f"{method} has invalid RMSE")
            if len(interval) != 2 or not (0.0 < interval[0] <= interval[1]):
                raise ValueError(f"{method} has invalid interval")
            if not (0.0 < actual <= target):
                raise ValueError(f"{method} has invalid actual calls")
            actual_calls_by_method[method].append(actual)
            if estimates.shape != (3, 100) or not np.all(np.isfinite(estimates)):
                raise ValueError(f"{method} has invalid estimates")
    if not np.all(np.diff(target_calls) > 0.0):
        raise ValueError("target call budgets must increase")
    for method, calls in actual_calls_by_method.items():
        if not np.all(np.diff(calls) > 0.0):
            raise ValueError(f"{method} actual call coordinates must increase")


def _series(
    rows: list[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calls = []
    rmse = []
    lower = []
    upper = []
    for row in rows:
        summary = row["methods"][method]
        calls.append(float(summary["mean_actual_utility_calls"]))
        rmse.append(float(summary["aggregate_rmse"]))
        lower.append(float(summary["aggregate_rmse_bootstrap_95"][0]))
        upper.append(float(summary["aggregate_rmse_bootstrap_95"][1]))
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (calls, rmse, lower, upper)
    )


def _calls_label(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    return f"{value / 1000:.3g}k"


def plot_results(
    report: Mapping[str, Any], output: Path
) -> tuple[Path, Path]:
    validate_report(report)
    if output.suffix.lower() != ".png":
        raise ValueError("output must be a PNG path")
    rows = _ordered_rows(report)
    target_calls = np.asarray(
        [row["total_utility_calls_per_estimate"] for row in rows],
        dtype=np.float64,
    )
    interval_lower = []
    interval_upper = []
    plotted_calls = []
    for method in METHOD_ORDER:
        calls, _rmse, lower, upper = _series(rows, method)
        plotted_calls.extend(calls.tolist())
        interval_lower.extend(lower.tolist())
        interval_upper.extend(upper.tolist())
    log_min = math.log10(min(interval_lower))
    log_max = math.log10(max(interval_upper))
    span = max(log_max - log_min, 0.25)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12.5,
            "axes.labelcolor": "#303640",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    figure, axis = plt.subplots(figsize=(9.0, 9.0))
    figure.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.26)
    axis.set_box_aspect(1.0)
    handles: dict[str, Any] = {}
    draw_order = tuple(
        method for method in METHOD_ORDER if method != "frame_orbit_ratio"
    ) + ("frame_orbit_ratio",)
    for method in draw_order:
        calls, rmse, _lower, _upper = _series(rows, method)
        style = METHOD_STYLES[method]
        (handle,) = axis.plot(
            calls,
            rmse,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=9.0 if method == "frame_orbit_ratio" else 7.0,
            markerfacecolor=style.get("markerfacecolor", style["color"]),
            markeredgewidth=1.25,
            markeredgecolor=(
                style["color"] if method == "frame_orbit_ratio" else "white"
            ),
            alpha=1.0 if method == "frame_orbit_ratio" else 0.86,
            zorder=5 if method == "frame_orbit_ratio" else 3,
        )
        handles[method] = handle

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(target_calls[0] * 0.80, max(plotted_calls) * 1.22)
    axis.set_ylim(
        10.0 ** (log_min - 0.08 * span),
        10.0 ** (log_max + 0.08 * span),
    )
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    axis.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(2.0, 3.0, 5.0, 7.0), numticks=20)
    )
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0e}")
    )
    axis.grid(True, which="major", color="#D8DDE4", linewidth=0.95, alpha=0.9)
    axis.grid(
        True, which="minor", axis="y", color="#EEF1F4", linewidth=0.65, alpha=0.8
    )
    axis.set_xlabel("Utility calls", fontsize=14.0)
    axis.set_ylabel("RMSE", fontsize=14.0)
    axis.tick_params(axis="both", which="major", labelsize=11.5, width=1.0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#59616C")
        spine.set_linewidth(1.15)
    axis.set_title(
        "Airport: Shapley RMSE",
        loc="left",
        fontsize=18.0,
        fontweight="semibold",
        color="#20252D",
        pad=13,
    )
    figure.legend(
        [handles[method] for method in METHOD_ORDER],
        [METHOD_STYLES[method]["label"] for method in METHOD_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=1.0,
        fontsize=10.5,
        borderpad=0.65,
        columnspacing=1.35,
        labelspacing=0.50,
        handlelength=2.5,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")
    figure.savefig(output, dpi=220, facecolor="white")
    figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)
    return output, pdf_output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/airport_100_all_baselines_3repeats_50k_1m.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/airport_100_all_baselines_3repeats_rmse_50k_1m.png"),
    )
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    png, pdf = plot_results(report, args.output)
    print(f"saved {png} and {pdf}")


if __name__ == "__main__":
    main()

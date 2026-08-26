"""Draw the ten-method Greedy-scope/estimator comparison.

Each dataset gets one near-square log-log panel.  The three controlled Greedy
branches and INSIDE-Orbit are visually prominent; all established baselines
remain on the same axes for an equal-call comparison.
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

from experiments.plot_inside_comparison import (
    _actual_calls,
    _calls_label,
    _dataset_name,
    _mapping,
    _ordered_rows,
    _positive,
)
from experiments.validate_greedy_estimator_comparison import METHOD_ORDER


METHOD_STYLES: dict[str, dict[str, Any]] = {
    "inside_greedy_global": {
        "label": "INSIDE-Greedy (global, linear)",
        "color": "#B45309",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 4.5,
        "markersize": 9.2,
        "filled": True,
        "zorder": 11,
    },
    "inside_greedy_per_size_linear": {
        "label": "INSIDE-Greedy (per-size, linear)",
        "color": "#EA580C",
        "marker": "o",
        "linestyle": (0, (6, 2)),
        "linewidth": 3.6,
        "markersize": 9.2,
        "filled": False,
        "zorder": 12,
    },
    "inside_greedy_per_size_ratio": {
        "label": "INSIDE-Greedy (per-size, ratio)",
        "color": "#B91C6C",
        "marker": "*",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "linewidth": 3.7,
        "markersize": 12.0,
        "filled": True,
        "zorder": 13,
    },
    "inside_orbit": {
        "label": "INSIDE-Orbit (ratio)",
        "color": "#7C2D12",
        "marker": "h",
        "linestyle": (0, (2, 1.6)),
        "linewidth": 3.5,
        "markersize": 9.1,
        "filled": True,
        "zorder": 9,
    },
    "ofa_iid_linear": {
        "label": "OFA linear",
        "color": "#2563A6",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.65,
    },
    "ofa_iid_ratio": {
        "label": "OFA ratio",
        "color": "#3B82C4",
        "marker": "o",
        "linestyle": "-.",
        "linewidth": 2.65,
    },
    "cc": {
        "label": "CC",
        "color": "#68707A",
        "marker": "X",
        "linestyle": ":",
        "linewidth": 2.55,
    },
    "s_diff": {
        "label": "S-Diff",
        "color": "#74831A",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.65,
    },
    "kernel_shap": {
        "label": "KernelSHAP",
        "color": "#AA8500",
        "marker": "v",
        "linestyle": "--",
        "linewidth": 2.65,
    },
    "tmc_shapley": {
        "label": "TMC-Shapley",
        "color": "#343B45",
        "marker": "P",
        "linestyle": (0, (5, 2, 1, 2)),
        "linewidth": 2.55,
    },
}

if tuple(METHOD_STYLES) != METHOD_ORDER:
    raise RuntimeError("plot styles must follow the ten-method order")


def _configured_methods(configuration: Mapping[str, Any]) -> tuple[str, ...]:
    raw = configuration.get("methods", configuration.get("method_order", ()))
    return tuple(raw) if isinstance(raw, (list, tuple)) else ()


def validate_plot_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "complete":
        raise ValueError("report status must be complete")
    configuration = _mapping(report.get("configuration"), path="configuration")
    if int(configuration.get("repeats", -1)) != 3:
        raise ValueError("comparison requires three repeats")
    if _configured_methods(configuration) != METHOD_ORDER:
        raise ValueError("report method order differs from the ten-method contract")
    rows = _ordered_rows(report)
    if len(rows) != 5:
        raise ValueError("comparison requires five budget points")
    for row_index, row in enumerate(rows):
        target = _positive(
            row.get("total_utility_calls_per_estimate"),
            path=f"row[{row_index}].total_utility_calls_per_estimate",
        )
        methods = _mapping(row.get("methods"), path=f"row[{row_index}].methods")
        if set(methods) != set(METHOD_ORDER):
            raise ValueError(f"row[{row_index}] has the wrong method set")
        for method in METHOD_ORDER:
            summary = _mapping(methods[method], path=f"row[{row_index}].{method}")
            _positive(summary.get("aggregate_rmse"), path=f"row[{row_index}].{method}.aggregate_rmse")
            interval = summary.get("aggregate_rmse_bootstrap_95")
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"{method} has no two-endpoint RMSE interval")
            lower = _positive(interval[0], path=f"{method}.interval.lower")
            upper = _positive(interval[1], path=f"{method}.interval.upper")
            if lower > upper:
                raise ValueError(f"{method} has a reversed RMSE interval")
            _actual_calls(summary, target, method=method)


def _series(
    rows: list[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calls: list[float] = []
    rmse: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        target = float(row["total_utility_calls_per_estimate"])
        summary = _mapping(row["methods"][method], path=f"methods.{method}")
        calls.append(_actual_calls(summary, target, method=method))
        rmse.append(float(summary["aggregate_rmse"]))
        interval = summary["aggregate_rmse_bootstrap_95"]
        lower.append(float(interval[0]))
        upper.append(float(interval[1]))
    return tuple(np.asarray(values, dtype=np.float64) for values in (calls, rmse, lower, upper))


def plot_results(report: Mapping[str, Any], output: Path) -> tuple[Path, Path]:
    validate_plot_report(report)
    if output.suffix.lower() != ".png":
        raise ValueError("output must be a PNG path")
    rows = _ordered_rows(report)
    target_calls = np.asarray(
        [float(row["total_utility_calls_per_estimate"]) for row in rows]
    )
    all_lower: list[float] = []
    all_upper: list[float] = []
    for method in METHOD_ORDER:
        _calls, _rmse, lower, upper = _series(rows, method)
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())
    log_min = math.log10(min(all_lower))
    log_max = math.log10(max(all_upper))
    log_span = max(log_max - log_min, 0.25)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.5,
            "axes.labelcolor": "#303640",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    figure, axis = plt.subplots(figsize=(9.0, 9.0))
    figure.subplots_adjust(left=0.115, right=0.98, top=0.92, bottom=0.305)
    axis.set_box_aspect(1.0)

    handles: dict[str, Any] = {}
    context = METHOD_ORDER[4:]
    focal = METHOD_ORDER[:4]
    for method in context + ("inside_orbit",) + METHOD_ORDER[:3]:
        calls, rmse, _lower, _upper = _series(rows, method)
        style = METHOD_STYLES[method]
        is_focal = method in focal
        filled = bool(style.get("filled", False))
        (handle,) = axis.plot(
            calls,
            rmse,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style.get("markersize", 7.5),
            markerfacecolor=style["color"] if filled else "white",
            markeredgecolor=style["color"],
            markeredgewidth=1.4,
            alpha=1.0 if is_focal else 0.86,
            zorder=style.get("zorder", 3),
            clip_on=True,
        )
        handles[method] = handle

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(target_calls[0] * 0.80, target_calls[-1] * 1.22)
    axis.set_ylim(
        10.0 ** (log_min - 0.08 * log_span),
        10.0 ** (log_max + 0.08 * log_span),
    )
    axis.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))
    axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=16))
    axis.yaxis.set_minor_locator(LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=30))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
    axis.grid(True, which="major", color="#D8DDE4", linewidth=1.0, alpha=0.92)
    axis.grid(True, which="minor", axis="y", color="#EEF1F4", linewidth=0.7, alpha=0.82)
    axis.set_xlabel("Utility calls", fontsize=15.5)
    axis.set_ylabel("RMSE", fontsize=15.5)
    axis.tick_params(axis="both", which="major", labelsize=12.5, width=1.1)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#555E69")
        spine.set_linewidth(1.3)
    axis.set_title(
        f"{_dataset_name(report)}: Shapley RMSE",
        loc="left",
        fontsize=19.5,
        fontweight="semibold",
        color="#20252D",
        pad=13,
    )
    figure.legend(
        [handles[method] for method in METHOD_ORDER],
        [METHOD_STYLES[method]["label"] for method in METHOD_ORDER],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.022),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=1.0,
        fontsize=9.7,
        borderpad=0.68,
        columnspacing=0.92,
        labelspacing=0.55,
        handlelength=2.35,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    figure.savefig(output, dpi=220, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    return output, pdf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output = args.output or args.report.with_name(f"{args.report.stem}_rmse.png")
    png, pdf = plot_results(report, output)
    print(f"saved {png} and {pdf}")


if __name__ == "__main__":
    main()

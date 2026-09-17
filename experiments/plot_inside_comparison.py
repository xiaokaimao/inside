"""Draw the common eight-method INSIDE comparison.

The plotting contract is intentionally shared by Wine, Breast Cancer,
Airport, and the U.S. Electoral College voting game.  Each input report
produces one near-square log--log figure; the two proposed INSIDE
implementations are visually prominent and the six baselines remain
secondary.  TMC-Shapley is positioned at its observed physical call count
after truncation.  Its raw early points are retained in the report and merely
clipped by the common target-budget viewport when they lie far to the left.
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


METHOD_ORDER = (
    "inside_greedy",
    "inside_orbit",
    "ofa_iid_linear",
    "ofa_iid_ratio",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "inside_greedy": {
        "label": "INSIDE-Greedy",
        "color": "#D97706",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 4.3,
        "markersize": 9.2,
        "zorder": 7,
    },
    "inside_orbit": {
        "label": "INSIDE-Orbit",
        "color": "#C2410C",
        "marker": "h",
        "linestyle": "-",
        "linewidth": 4.3,
        "markersize": 9.8,
        "zorder": 7,
    },
    "ofa_iid_linear": {
        "label": "OFA linear",
        "color": "#2563A6",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.7,
    },
    "ofa_iid_ratio": {
        "label": "OFA ratio",
        "color": "#3B82C4",
        "marker": "o",
        "linestyle": "-.",
        "linewidth": 2.7,
    },
    "cc": {
        "label": "CC",
        "color": "#68707A",
        "marker": "X",
        "linestyle": ":",
        "linewidth": 2.6,
    },
    "s_diff": {
        "label": "S-Diff",
        "color": "#74831A",
        "marker": "^",
        "linestyle": "-",
        "linewidth": 2.7,
    },
    "kernel_shap": {
        "label": "KernelSHAP",
        "color": "#AA8500",
        "marker": "v",
        "linestyle": "--",
        "linewidth": 2.7,
    },
    "tmc_shapley": {
        "label": "TMC-Shapley",
        "color": "#343B45",
        "marker": "P",
        "linestyle": (0, (5, 2, 1, 2)),
        "linewidth": 2.6,
    },
}

LEGACY_GREEDY_STYLE: dict[str, Any] = {
    "label": "Greedy (raw λ=.1)",
    "color": "#F3A64A",
    "marker": "d",
    "linestyle": "--",
    "linewidth": 2.7,
    "markersize": 7.4,
    "zorder": 4,
}

if tuple(METHOD_STYLES) != METHOD_ORDER:
    raise RuntimeError("plot styles must follow the common method order")


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _positive(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be finite and positive")
    return result


def _configured_methods(configuration: Mapping[str, Any]) -> tuple[str, ...]:
    raw = configuration.get("methods", configuration.get("method_order", ()))
    return tuple(raw) if isinstance(raw, (list, tuple)) else ()


def _ordered_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = _mapping(
        report.get("results_by_inner_budget"), path="results_by_inner_budget"
    )
    if len(raw) != 5:
        raise ValueError("the INSIDE comparison requires exactly five budgets")
    rows = [_mapping(row, path="budget row") for row in raw.values()]
    rows.sort(
        key=lambda row: _positive(
            row.get("total_utility_calls_per_estimate"),
            path="total_utility_calls_per_estimate",
        )
    )
    target = np.asarray(
        [float(row["total_utility_calls_per_estimate"]) for row in rows]
    )
    if not np.all(np.diff(target) > 0.0):
        raise ValueError("target utility-call budgets must increase")
    return rows


def _actual_calls(
    summary: Mapping[str, Any], target_calls: float, *, method: str
) -> float:
    if method != "tmc_shapley":
        # The seven non-truncating implementations are compared at the common
        # physical cap.  Their validator separately verifies full budget use.
        return target_calls
    if "mean_actual_utility_calls" in summary:
        actual = _positive(
            summary["mean_actual_utility_calls"],
            path=f"{method}.mean_actual_utility_calls",
        )
    else:
        values = summary.get(
            "actual_utility_calls_by_repeat",
            summary.get("actual_utility_calls_per_estimate"),
        )
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("TMC-Shapley must record observed physical calls")
        actual = float(np.mean(np.asarray(values, dtype=np.float64)))
    if actual > target_calls:
        raise ValueError("TMC-Shapley observed calls exceed its cap")
    return actual


def validate_plot_report(report: Mapping[str, Any]) -> None:
    """Validate only the fields needed to render the common comparison."""
    if report.get("status") != "complete":
        raise ValueError("report status must be complete")
    configuration = _mapping(report.get("configuration"), path="configuration")
    if int(configuration.get("repeats", -1)) != 3:
        raise ValueError("the common comparison requires three repeats")
    if _configured_methods(configuration) != METHOD_ORDER:
        raise ValueError("report method order differs from the common contract")
    rows = _ordered_rows(report)
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
            _positive(
                summary.get("aggregate_rmse"),
                path=f"row[{row_index}].{method}.aggregate_rmse",
            )
            interval = summary.get("aggregate_rmse_bootstrap_95")
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"{method} has no two-endpoint RMSE interval")
            lower = _positive(interval[0], path=f"{method}.interval.lower")
            upper = _positive(interval[1], path=f"{method}.interval.upper")
            if lower > upper:
                raise ValueError(f"{method} has a reversed RMSE interval")
            _actual_calls(summary, target, method=method)


def validate_plot_pair(
    report: Mapping[str, Any], legacy_report: Mapping[str, Any]
) -> None:
    """Require a normalized/legacy pair to describe the same benchmark."""
    validate_plot_report(report)
    validate_plot_report(legacy_report)
    if _dataset_name(report) != _dataset_name(legacy_report):
        raise ValueError("normalized and legacy reports use different datasets")
    current_rows = _ordered_rows(report)
    legacy_rows = _ordered_rows(legacy_report)
    current_calls = [
        int(row["total_utility_calls_per_estimate"]) for row in current_rows
    ]
    legacy_calls = [
        int(row["total_utility_calls_per_estimate"]) for row in legacy_rows
    ]
    if current_calls != legacy_calls:
        raise ValueError("normalized and legacy reports use different budgets")
    current_truth = np.asarray(
        _mapping(report.get("ground_truth"), path="ground_truth").get("values"),
        dtype=np.float64,
    )
    legacy_truth = np.asarray(
        _mapping(
            legacy_report.get("ground_truth"), path="legacy.ground_truth"
        ).get("values"),
        dtype=np.float64,
    )
    if (
        current_truth.shape != legacy_truth.shape
        or not np.allclose(current_truth, legacy_truth, rtol=0.0, atol=1e-14)
    ):
        raise ValueError("normalized and legacy reports use different ground truth")


def _series(
    rows: list[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calls: list[float] = []
    rmse: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        target = float(row["total_utility_calls_per_estimate"])
        summary = row["methods"][method]
        calls.append(_actual_calls(summary, target, method=method))
        rmse.append(float(summary["aggregate_rmse"]))
        interval = summary["aggregate_rmse_bootstrap_95"]
        lower.append(float(interval[0]))
        upper.append(float(interval[1]))
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (calls, rmse, lower, upper)
    )


def _dataset_name(report: Mapping[str, Any]) -> str:
    configuration = _mapping(report.get("configuration"), path="configuration")
    candidate: Any = configuration.get("dataset")
    dataset = report.get("dataset")
    game = report.get("game")
    if isinstance(dataset, Mapping):
        candidate = dataset.get("dataset", candidate)
    elif isinstance(dataset, str):
        candidate = dataset
    if isinstance(game, Mapping):
        candidate = game.get("dataset", candidate)
    normalized = str(candidate or "INSIDE").strip().lower()
    labels = {
        "wine": "Wine",
        "cancer": "Breast Cancer",
        "breast_cancer": "Breast Cancer",
        "breast cancer": "Breast Cancer",
        "airport": "Airport",
        "voting": "U.S. Electoral Voting",
        "us_electoral_voting": "U.S. Electoral Voting",
        "us_electoral_college": "U.S. Electoral Voting",
    }
    return labels.get(normalized, str(candidate or "INSIDE"))


def _calls_label(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}k"
    return f"{value:.0f}"


def plot_results(
    report: Mapping[str, Any],
    output: Path,
    *,
    legacy_report: Mapping[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save one PNG and one vector PDF and return their paths."""
    validate_plot_report(report)
    if legacy_report is not None:
        validate_plot_pair(report, legacy_report)
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
    legacy_rows: list[Mapping[str, Any]] | None = None
    if legacy_report is not None:
        legacy_rows = _ordered_rows(legacy_report)
        _calls, _rmse, lower, upper = _series(
            legacy_rows, "inside_greedy"
        )
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())
    log_min = math.log10(min(all_lower))
    log_max = math.log10(max(all_upper))
    log_span = max(log_max - log_min, 0.25)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13.0,
            "axes.labelcolor": "#303640",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )
    figure, axis = plt.subplots(figsize=(9.0, 9.0))
    figure.subplots_adjust(left=0.11, right=0.98, top=0.92, bottom=0.265)
    axis.set_box_aspect(1.0)

    handles: dict[str, Any] = {}
    if legacy_rows is not None:
        calls, rmse, _lower, _upper = _series(
            legacy_rows, "inside_greedy"
        )
        style = LEGACY_GREEDY_STYLE
        (handles["inside_greedy_legacy_raw"],) = axis.plot(
            calls,
            rmse,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.25,
            alpha=0.95,
            zorder=style["zorder"],
            clip_on=True,
        )
    draw_order = METHOD_ORDER[2:] + METHOD_ORDER[:2]
    for method in draw_order:
        calls, rmse, _lower, _upper = _series(rows, method)
        style = METHOD_STYLES[method]
        focal = method.startswith("inside_")
        (handle,) = axis.plot(
            calls,
            rmse,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style.get("markersize", 7.5),
            markerfacecolor=style["color"] if focal else "white",
            markeredgecolor=style["color"],
            markeredgewidth=1.35,
            alpha=1.0 if focal else 0.87,
            zorder=style.get("zorder", 3),
            clip_on=True,
        )
        handles[method] = handle

    axis.set_xscale("log")
    axis.set_yscale("log")
    # The viewport is defined by the common target budgets.  This is the only
    # clipping applied to TMC: very early actual-call points may lie left of
    # the panel, but no later/raw observation is removed from the series.
    axis.set_xlim(target_calls[0] * 0.80, target_calls[-1] * 1.22)
    axis.set_ylim(
        10.0 ** (log_min - 0.08 * log_span),
        10.0 ** (log_max + 0.08 * log_span),
    )
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    axis.xaxis.set_minor_locator(NullLocator())
    # Power-of-ten ticks alone can leave a sub-decade panel (notably Voting)
    # with just one numeric y label.  Labelling 1/2/5 on each decade keeps
    # the logarithmic scale explicit without making the axis dense.
    axis.yaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=16)
    )
    axis.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=30)
    )
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0e}"))
    axis.grid(True, which="major", color="#D8DDE4", linewidth=1.0, alpha=0.92)
    axis.grid(
        True, which="minor", axis="y", color="#EEF1F4", linewidth=0.7, alpha=0.82
    )
    axis.set_xlabel("Utility calls", fontsize=15.0)
    axis.set_ylabel("RMSE", fontsize=15.0)
    axis.tick_params(axis="both", which="major", labelsize=12.0, width=1.05)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#555E69")
        spine.set_linewidth(1.25)
    axis.set_title(
        f"{_dataset_name(report)}: Shapley RMSE",
        loc="left",
        fontsize=19.0,
        fontweight="semibold",
        color="#20252D",
        pad=13,
    )
    legend_order = list(METHOD_ORDER)
    if legacy_rows is not None:
        legend_order.insert(1, "inside_greedy_legacy_raw")
    figure.legend(
        [handles[method] for method in legend_order],
        [
            LEGACY_GREEDY_STYLE["label"]
            if method == "inside_greedy_legacy_raw"
            else METHOD_STYLES[method]["label"]
            for method in legend_order
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.027),
        ncol=3 if legacy_rows is not None else 4,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=1.0,
        fontsize=10.5,
        borderpad=0.72,
        columnspacing=1.25,
        labelspacing=0.62,
        handlelength=2.55,
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
    parser.add_argument(
        "--legacy-report",
        type=Path,
        help=(
            "optional legacy raw-lambda report; overlays its INSIDE-Greedy "
            "curve for a controlled before/after comparison"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    legacy_report = (
        json.loads(args.legacy_report.read_text(encoding="utf-8"))
        if args.legacy_report is not None
        else None
    )
    output = args.output or args.report.with_name(
        f"{args.report.stem}_rmse.png"
    )
    png, pdf = plot_results(
        report, output, legacy_report=legacy_report
    )
    print(f"saved {png} and {pdf}")


if __name__ == "__main__":
    main()

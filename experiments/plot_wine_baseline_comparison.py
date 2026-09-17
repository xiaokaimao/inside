"""Plot the three-repeat Wine comparison across all local baselines.

Chart contract
--------------
Question
    At the same physical utility-call scale, how does Shapley RMSE change for
    each of the twelve estimators in the completed Wine experiment?
Form
    Two separate near-square figures.  The main figure compares the practical
    Frame-OFA ratio estimator with all compatible external baselines; the two
    global-linear OFA variants are isolated in their own near-square figure.
Surface
    Reproducible static Matplotlib exports (PNG and PDF).

The plotter intentionally validates the narrow report contract produced by
``experiments.add_wine_local_baselines``.  It does not silently draw partial
checkpoints or comparisons with a different number of repeats.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.result_paths import by_format
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np


REUSED_METHODS = frozenset(
    {
        "official_ofa_fixed_ratio",
        "frame_orbit_ratio",
        "iid_linear_ofa",
        "frame_coupled_linear",
        "official_cc_basic",
    }
)

NEW_METHODS = frozenset(
    {
        "gels_shapley",
        "kernel_shap_sampled",
        "group_testing",
        "diff",
        "s_diff",
        "tmc_shapley",
        "stratified_marginal_mc",
    }
)

EXPECTED_METHODS = REUSED_METHODS | NEW_METHODS

MAIN_PLOT_ORDER = (
    "frame_orbit_ratio",
    "official_ofa_fixed_ratio",
    "official_cc_basic",
    "s_diff",
    "diff",
    "group_testing",
    "kernel_shap_sampled",
    "gels_shapley",
    "tmc_shapley",
    "stratified_marginal_mc",
)

LINEAR_PAIR_ORDER = (
    "iid_linear_ofa",
    "frame_coupled_linear",
)

PLOT_ORDER = MAIN_PLOT_ORDER + LINEAR_PAIR_ORDER

FOCAL_METHODS = frozenset(
    {"frame_orbit_ratio", "frame_coupled_linear"}
)


# Methods in the same estimator family share a palette root.  Line style,
# marker shape, fill, and stroke width provide non-color distinction.  The two
# Frame-OFA curves deliberately share the strongest orange treatment.
METHOD_STYLES: dict[str, dict[str, Any]] = {
    "official_ofa_fixed_ratio": {
        "label": "OFA ratio",
        "color": "#2563A6",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.8,
    },
    "frame_orbit_ratio": {
        "label": "Frame-OFA ratio",
        "color": "#D97706",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 4.2,
        "markerfacecolor": "#D97706",
    },
    "official_cc_basic": {
        "label": "CC",
        "color": "#5F6670",
        "marker": "X",
        "linestyle": ":",
        "linewidth": 2.5,
    },
    "stratified_marginal_mc": {
        "label": "Stratified MC",
        "color": "#8B929B",
        "marker": "v",
        "linestyle": "-.",
        "linewidth": 2.4,
    },
    "gels_shapley": {
        "label": "GELS-Shapley",
        "color": "#B48A00",
        "marker": "o",
        "linestyle": "-",
        "linewidth": 2.5,
    },
    "kernel_shap_sampled": {
        "label": "KernelSHAP",
        "color": "#B48A00",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.5,
    },
    "group_testing": {
        "label": "Group Testing",
        "color": "#74831A",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 2.5,
    },
    "diff": {
        "label": "Diff",
        "color": "#74831A",
        "marker": "D",
        "linestyle": ":",
        "linewidth": 2.6,
    },
    "s_diff": {
        "label": "S-Diff",
        "color": "#74831A",
        "marker": "X",
        "linestyle": "-",
        "linewidth": 2.8,
    },
    "iid_linear_ofa": {
        "label": "OFA (IID)",
        "color": "#2563A6",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 2.8,
    },
    "frame_coupled_linear": {
        "label": "Frame-OFA (coupled)",
        "color": "#D97706",
        "marker": "^",
        "linestyle": "--",
        "linewidth": 4.0,
        "markerfacecolor": "white",
    },
    "tmc_shapley": {
        "label": "TMC-Shapley",
        "color": "#383F48",
        "marker": "P",
        "linestyle": (0, (5, 2, 1, 2)),
        "linewidth": 2.5,
    },
}


def _plotted_methods() -> tuple[str, ...]:
    return PLOT_ORDER


if (
    len(_plotted_methods()) != len(set(_plotted_methods()))
    or set(_plotted_methods()) != EXPECTED_METHODS
):
    raise RuntimeError("panel definitions must cover each expected method once")

if set(METHOD_STYLES) != EXPECTED_METHODS:
    raise RuntimeError("every expected method must have exactly one style")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _positive_finite(value: Any, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return numeric


def _ordered_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_rows = _require_mapping(
        report.get("results_by_inner_budget"),
        "results_by_inner_budget",
    )
    if len(raw_rows) != 5:
        raise ValueError("the Wine comparison must contain exactly five budgets")
    rows = [_require_mapping(row, "budget row") for row in raw_rows.values()]
    rows.sort(
        key=lambda row: _positive_finite(
            row.get("total_utility_calls_per_estimate"),
            "total_utility_calls_per_estimate",
        )
    )
    totals = np.asarray(
        [
            _positive_finite(
                row.get("total_utility_calls_per_estimate"),
                "total_utility_calls_per_estimate",
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if not np.all(np.diff(totals) > 0.0):
        raise ValueError("total utility-call budgets must be strictly increasing")
    return rows


def _interval(summary: Mapping[str, Any], method: str) -> tuple[float, float]:
    raw = summary.get("aggregate_rmse_bootstrap_95")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError(
            f"{method}.aggregate_rmse_bootstrap_95 must have two endpoints"
        )
    lower = _positive_finite(raw[0], f"{method} bootstrap lower endpoint")
    upper = _positive_finite(raw[1], f"{method} bootstrap upper endpoint")
    if lower > upper:
        raise ValueError(f"{method} bootstrap endpoints are reversed")
    return lower, upper


def _x_value(
    row: Mapping[str, Any], method: str, summary: Mapping[str, Any]
) -> float:
    total = _positive_finite(
        row.get("total_utility_calls_per_estimate"),
        "total_utility_calls_per_estimate",
    )
    if method in NEW_METHODS and "mean_actual_utility_calls" in summary:
        actual = _positive_finite(
            summary["mean_actual_utility_calls"],
            f"{method}.mean_actual_utility_calls",
        )
        if actual > total:
            raise ValueError(
                f"{method}.mean_actual_utility_calls exceeds its call cap"
            )
        return actual
    return total


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate the completed, three-repeat, twelve-method plot contract."""
    if report.get("status") != "complete":
        raise ValueError("the comparison report must have status='complete'")
    configuration = _require_mapping(
        report.get("configuration"), "configuration"
    )
    if configuration.get("repeats") != 3:
        raise ValueError("the comparison report must use exactly three repeats")
    comparison = _require_mapping(
        configuration.get("baseline_comparison"),
        "configuration.baseline_comparison",
    )
    declared_new = comparison.get("new_methods")
    if not isinstance(declared_new, list) or set(declared_new) != NEW_METHODS:
        raise ValueError("baseline_comparison.new_methods is incomplete")
    if len(declared_new) != len(NEW_METHODS):
        raise ValueError("baseline_comparison.new_methods contains duplicates")

    rows = _ordered_rows(report)
    configured_totals = configuration.get("total_call_budgets")
    if configured_totals is not None:
        try:
            totals = [float(value) for value in configured_totals]
        except (TypeError, ValueError) as error:
            raise ValueError("configuration.total_call_budgets is invalid") from error
        row_totals = [
            float(row["total_utility_calls_per_estimate"]) for row in rows
        ]
        if sorted(totals) != row_totals:
            raise ValueError(
                "configuration.total_call_budgets disagrees with result rows"
            )

    for row_index, row in enumerate(rows):
        methods = _require_mapping(row.get("methods"), "budget-row methods")
        actual_methods = set(methods)
        if actual_methods != EXPECTED_METHODS:
            missing = sorted(EXPECTED_METHODS - actual_methods)
            unexpected = sorted(actual_methods - EXPECTED_METHODS)
            raise ValueError(
                f"budget row {row_index} method mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        for method in EXPECTED_METHODS:
            summary = _require_mapping(methods[method], f"methods.{method}")
            _positive_finite(
                summary.get("aggregate_rmse"), f"{method}.aggregate_rmse"
            )
            _interval(summary, method)
            _x_value(row, method, summary)

    for method in EXPECTED_METHODS:
        calls, _rmse, _lower, _upper = _series_unvalidated(rows, method)
        if not np.all(np.diff(calls) > 0.0):
            raise ValueError(f"{method} call coordinates must increase")


def _series_unvalidated(
    rows: list[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calls: list[float] = []
    rmse: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        methods = _require_mapping(row["methods"], "budget-row methods")
        summary = _require_mapping(methods[method], f"methods.{method}")
        interval_lower, interval_upper = _interval(summary, method)
        calls.append(_x_value(row, method, summary))
        rmse.append(_positive_finite(summary["aggregate_rmse"], method))
        lower.append(interval_lower)
        upper.append(interval_upper)
    return (
        np.asarray(calls, dtype=np.float64),
        np.asarray(rmse, dtype=np.float64),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def series(
    report: Mapping[str, Any], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return validated x, RMSE, lower, and upper arrays for one method."""
    validate_report(report)
    if method not in EXPECTED_METHODS:
        raise ValueError(f"unknown comparison method: {method}")
    return _series_unvalidated(_ordered_rows(report), method)


def _calls_label(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    return f"{value / 1000:.3g}k"


def _main_call_limits(
    target_calls: np.ndarray, plotted_calls: list[float]
) -> tuple[float, float]:
    """Crop TMC's early truncation tail without changing its call data."""
    return (
        float(target_calls[0] * 0.80),
        float(max(plotted_calls) * 1.22),
    )


def _dataset_label(report: Mapping[str, Any]) -> str:
    dataset = report.get("dataset")
    if isinstance(dataset, Mapping):
        raw = dataset.get("dataset", "wine")
    else:
        raw = "wine"
    return str(raw).replace("_", " ").replace("-", " ").title()


def _default_output(input_path: Path) -> Path:
    marker = "_3repeats_"
    if marker in input_path.stem:
        stem = input_path.stem.replace(marker, f"{marker}rmse_", 1)
    else:
        stem = f"{input_path.stem}_rmse"
    return by_format(input_path.with_name(stem + ".png"))


def _linear_output(main_output: Path) -> Path:
    marker = "_rmse_"
    if marker in main_output.stem:
        stem = main_output.stem.replace(marker, "_linear_rmse_", 1)
    else:
        stem = f"{main_output.stem}_linear"
    return main_output.with_name(stem + ".png")


def plot_results(
    report: Mapping[str, Any], output: Path
) -> tuple[Path, Path, Path, Path]:
    """Export separate near-square main and linear-comparison figures."""
    validate_report(report)
    if output.suffix.lower() != ".png":
        raise ValueError("the primary output path must have a .png suffix")

    rows = _ordered_rows(report)
    target_calls = np.asarray(
        [row["total_utility_calls_per_estimate"] for row in rows],
        dtype=np.float64,
    )
    main_call_values: list[float] = []
    main_lower: list[float] = []
    main_upper: list[float] = []
    for method in MAIN_PLOT_ORDER:
        calls, _rmse, lower, upper = _series_unvalidated(rows, method)
        main_call_values.extend(calls.tolist())
        main_lower.extend(lower.tolist())
        main_upper.extend(upper.tolist())
    log_min = math.log10(min(main_lower))
    log_max = math.log10(max(main_upper))
    span = max(log_max - log_min, 0.25)
    y_limits = (
        10.0 ** (log_min - 0.08 * span),
        10.0 ** (log_max + 0.08 * span),
    )

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
    figure.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.93,
        bottom=0.26,
    )
    axis.set_box_aspect(1.0)

    handles: dict[str, Any] = {}
    draw_order = tuple(
        method for method in MAIN_PLOT_ORDER if method not in FOCAL_METHODS
    ) + tuple(
        method for method in MAIN_PLOT_ORDER if method in FOCAL_METHODS
    )
    for method in draw_order:
        calls, rmse, _lower, _upper = _series_unvalidated(rows, method)
        style = METHOD_STYLES[method]
        (handle,) = axis.plot(
            calls,
            rmse,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=9.0 if method in FOCAL_METHODS else 7.0,
            markerfacecolor=style.get("markerfacecolor", style["color"]),
            markeredgewidth=1.25,
            markeredgecolor=(
                style["color"] if method in FOCAL_METHODS else "white"
            ),
            alpha=1.0 if method in FOCAL_METHODS else 0.86,
            zorder=5 if method in FOCAL_METHODS else 3,
        )
        handles[method] = handle

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*_main_call_limits(target_calls, main_call_values))
    axis.set_ylim(*y_limits)
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
    axis.grid(
        True,
        which="major",
        color="#D8DDE4",
        linewidth=0.95,
        alpha=0.9,
    )
    axis.grid(
        True,
        which="minor",
        axis="y",
        color="#EEF1F4",
        linewidth=0.65,
        alpha=0.8,
    )
    axis.set_xlabel("Utility calls", fontsize=14.0)
    axis.set_ylabel("RMSE", fontsize=14.0)
    axis.tick_params(axis="both", which="major", labelsize=11.5, width=1.0)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#59616C")
        spine.set_linewidth(1.15)
    axis.set_title(
        f"{_dataset_label(report)}: Shapley RMSE",
        loc="left",
        fontsize=18.0,
        fontweight="semibold",
        color="#20252D",
        pad=13,
    )

    figure.legend(
        [handles[method] for method in MAIN_PLOT_ORDER],
        [METHOD_STYLES[method]["label"] for method in MAIN_PLOT_ORDER],
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

    linear_figure, linear_axis = plt.subplots(figsize=(7.6, 7.6))
    linear_figure.subplots_adjust(
        left=0.14,
        right=0.96,
        top=0.91,
        bottom=0.13,
    )
    linear_axis.set_box_aspect(1.0)
    linear_lower: list[float] = []
    linear_upper: list[float] = []
    for method in LINEAR_PAIR_ORDER:
        calls, rmse, lower, upper = _series_unvalidated(rows, method)
        linear_lower.extend(lower.tolist())
        linear_upper.extend(upper.tolist())
        style = METHOD_STYLES[method]
        linear_axis.plot(
            calls,
            rmse,
            label=style["label"],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=7.2,
            markerfacecolor=style.get("markerfacecolor", style["color"]),
            markeredgewidth=1.1,
            markeredgecolor=style["color"],
            zorder=3,
        )
    linear_log_min = math.log10(min(linear_lower))
    linear_log_max = math.log10(max(linear_upper))
    linear_span = max(linear_log_max - linear_log_min, 0.25)
    linear_axis.set_xscale("log")
    linear_axis.set_yscale("log")
    linear_axis.set_xlim(target_calls[0] * 0.78, target_calls[-1] * 1.25)
    linear_axis.set_ylim(
        10.0 ** (linear_log_min - 0.09 * linear_span),
        10.0 ** (linear_log_max + 0.09 * linear_span),
    )
    linear_axis.set_xticks(target_calls[[0, 2, 4]])
    linear_axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    linear_axis.xaxis.set_minor_locator(NullLocator())
    linear_axis.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    linear_axis.yaxis.set_minor_locator(
        LogLocator(base=10, subs=(2.0, 5.0), numticks=10)
    )
    linear_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0e}")
    )
    linear_axis.grid(
        True,
        which="major",
        color="#D8DDE4",
        linewidth=0.9,
        alpha=0.9,
    )
    linear_axis.grid(
        True,
        which="minor",
        axis="y",
        color="#EEF1F4",
        linewidth=0.6,
        alpha=0.8,
    )
    linear_axis.set_title(
        f"{_dataset_label(report)}: Linear estimators",
        loc="left",
        fontsize=18.0,
        fontweight="semibold",
        color="#20252D",
        pad=13,
    )
    linear_axis.set_xlabel("Utility calls", fontsize=14.0)
    linear_axis.set_ylabel("RMSE", fontsize=14.0)
    linear_axis.tick_params(
        axis="both", which="major", labelsize=11.5, width=1.0
    )
    for spine in linear_axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#59616C")
        spine.set_linewidth(1.15)
    linear_axis.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=1.0,
        fontsize=10.0,
        borderpad=0.55,
        labelspacing=0.42,
        handlelength=2.5,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = by_format(output.with_suffix(".pdf"))
    linear_png_output = _linear_output(output)
    linear_pdf_output = by_format(linear_png_output.with_suffix(".pdf"))
    figure.savefig(output, dpi=220, facecolor="white")
    figure.savefig(pdf_output, facecolor="white")
    linear_figure.savefig(linear_png_output, dpi=220, facecolor="white")
    linear_figure.savefig(linear_pdf_output, facecolor="white")
    plt.close(figure)
    plt.close(linear_figure)
    return output, pdf_output, linear_png_output, linear_pdf_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/json/"
            "wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_"
            "3repeats_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "main PNG output path; main/linear PNG and PDF files are exported"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise ValueError(f"input report does not exist: {args.input}")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or _default_output(args.input)
    main_png, main_pdf, linear_png, linear_pdf = plot_results(report, output)
    print(
        f"Saved {main_png}, {main_pdf}, {linear_png}, and {linear_pdf}"
    )


if __name__ == "__main__":
    main()

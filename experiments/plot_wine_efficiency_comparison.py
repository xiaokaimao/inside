"""Plot Wine data-valuation RMSE against the audited estimated-time proxy.

This intentionally mirrors the Airport Efficiency figure.  The x-axis is an
estimated utility-dominated one-shot time, not measured end-to-end runtime.
INSIDE-Greedy includes its measured production design time.  S-Diff includes
the source pipeline wall time as a conservative upper-bound surcharge because
that source did not isolate its non-utility stages.  Other baseline overheads
remain zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np

from experiments.plot_airport_efficiency_comparison import (
    METHOD_STYLES,
    _legend_handle,
    _padded_log_limits,
    _plot_style,
    _scientific_label,
    _time_label,
)
from experiments.run_wine_efficiency_comparison import (
    EXPERIMENT_ID,
    METHOD_LABELS,
    METHOD_ORDER,
    validate_efficiency_report,
)


ARTIFACT_ID = "wine_estimated_time_efficiency_sdiff_wall_upper_bound"
DEFAULT_REPORT = Path("results/json/wine_efficiency_k64_lambda1over16.json")
DEFAULT_OUTPUT = Path("results/pdf/wine_efficiency_k64_lambda1over16.png")
# Matplotlib fills multi-column legends down each column.  This input order
# renders the requested method order across rows:
# Greedy, Orbit, OFA / CC, S-Diff, KernelSHAP / TMC-Shapley.
LEGEND_HANDLE_ORDER = (
    "inside_greedy",
    "cc",
    "tmc_shapley",
    "inside_orbit",
    "s_diff",
    "ofa",
    "kernel_shap",
)

if tuple(METHOD_STYLES) != METHOD_ORDER:
    raise RuntimeError("Wine and Airport Efficiency style orders disagree")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be positive and finite")
    return result


def _apply_rcparams() -> None:
    """Declare the publication font/export contract in this source file."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 20.0,
            "axes.labelcolor": "#303640",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _ordered_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("Wine Efficiency report must have five rows")
    ordered = sorted(rows, key=lambda row: int(row["budget_index"]))
    if [int(row["budget_index"]) for row in ordered] != list(range(5)):
        raise ValueError("Wine Efficiency budget indices are invalid")
    return ordered


def validate_plot_report(report: Mapping[str, Any]) -> None:
    """Validate every scalar and timing identity consumed by the figure."""
    validate_efficiency_report(report)
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("unexpected Wine Efficiency experiment")
    configuration = report["configuration"]
    if configuration.get("time_interpretation") != (
        "estimated utility-dominated one-shot time"
    ):
        raise ValueError("Wine report has the wrong time interpretation")
    if configuration.get("charged_extra_time_components") != [
        "INSIDE-Greedy production design time",
        (
            "S-Diff source complete 128-process pipeline wall time "
            "as a conservative upper-bound surcharge"
        ),
    ]:
        raise ValueError("Wine report has the wrong overhead policy")
    rows = _ordered_rows(report)
    for method in METHOD_ORDER:
        previous_time = 0.0
        for row_index, row in enumerate(rows):
            summary = row["methods"][method]
            if summary.get("label") != METHOD_LABELS[method]:
                raise ValueError(f"{method} has the wrong display label")
            mean_time = _finite_positive(
                summary.get("mean_estimated_total_seconds"),
                path=f"results[{row_index}].{method}.mean time",
            )
            time_range = summary.get("estimated_total_seconds_range")
            if not isinstance(time_range, list) or len(time_range) != 2:
                raise ValueError(f"{method} has an invalid time range")
            lower_time, upper_time = (
                _finite_positive(value, path=f"{method}.time range")
                for value in time_range
            )
            tolerance = 1e-12 * max(1.0, abs(mean_time))
            if not (
                lower_time - tolerance <= mean_time <= upper_time + tolerance
            ):
                raise ValueError(f"{method} mean time lies outside its range")
            if mean_time <= previous_time:
                raise ValueError(f"{method} time coordinates must increase")
            previous_time = mean_time
            _finite_positive(summary.get("aggregate_rmse"), path=f"{method}.RMSE")
            interval = summary.get("aggregate_rmse_bootstrap_95")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{method} has an invalid RMSE interval")
            lower_rmse, upper_rmse = (
                _finite_positive(value, path=f"{method}.RMSE interval")
                for value in interval
            )
            if lower_rmse > upper_rmse:
                raise ValueError(f"{method} RMSE interval is reversed")


def _series(
    rows: Sequence[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values: list[list[float]] = [[] for _ in range(6)]
    for row in rows:
        summary = row["methods"][method]
        time_range = summary["estimated_total_seconds_range"]
        interval = summary["aggregate_rmse_bootstrap_95"]
        point = (
            float(summary["mean_estimated_total_seconds"]),
            float(time_range[0]),
            float(time_range[1]),
            float(summary["aggregate_rmse"]),
            float(interval[0]),
            float(interval[1]),
        )
        for destination, item in zip(values, point, strict=True):
            destination.append(item)
    return tuple(np.asarray(items, dtype=np.float64) for items in values)


def _trim_grid_around_legend(figure: Any, axis: Any, legend: Any) -> None:
    """Keep the grid visible while preventing strokes beneath legend text."""

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer=renderer)
    axes_box = axis.get_window_extent(renderer=renderer)
    padding_pixels = 5.0 * figure.dpi / 72.0
    legend_x0 = legend_box.x0 - padding_pixels
    legend_x1 = legend_box.x1 + padding_pixels
    legend_y0 = legend_box.y0 - padding_pixels
    legend_y1 = legend_box.y1 + padding_pixels
    grid_x_stop = float(
        np.clip((legend_x0 - axes_box.x0) / axes_box.width, 0.0, 1.0)
    )
    grid_y_stop = float(
        np.clip((legend_y0 - axes_box.y0) / axes_box.height, 0.0, 1.0)
    )

    for tick in axis.xaxis.get_major_ticks():
        x_display = axis.transData.transform((tick.get_loc(), 1.0))[0]
        if legend_x0 <= x_display <= legend_x1:
            tick.gridline.set_ydata((0.0, grid_y_stop))
    for tick in (
        list(axis.yaxis.get_major_ticks())
        + list(axis.yaxis.get_minor_ticks())
    ):
        y_display = axis.transData.transform((1.0, tick.get_loc()))[1]
        if legend_y0 <= y_display <= legend_y1:
            tick.gridline.set_xdata((0.0, grid_x_stop))


def build_figure(report: Mapping[str, Any]) -> Any:
    validate_plot_report(report)
    rows = _ordered_rows(report)
    _apply_rcparams()
    figure, axis = plt.subplots(figsize=(8.9, 8.9))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.13, top=0.972)

    all_time_lower: list[float] = []
    all_time_upper: list[float] = []
    all_rmse_lower: list[float] = []
    all_rmse_upper: list[float] = []
    for method in METHOD_ORDER:
        times, time_lower, time_upper, rmse, rmse_lower, rmse_upper = _series(
            rows, method
        )
        style = METHOD_STYLES[method]
        all_time_lower.extend(time_lower.tolist())
        all_time_upper.extend(time_upper.tolist())
        all_rmse_lower.extend(rmse_lower.tolist())
        all_rmse_upper.extend(rmse_upper.tolist())
        axis.fill_between(
            times,
            rmse_lower,
            rmse_upper,
            color=style["color"],
            alpha=0.10 if style["filled"] else 0.065,
            linewidth=0.0,
            zorder=max(1, int(style["zorder"]) - 2),
        )
        left_error = np.maximum(times - time_lower, 0.0)
        right_error = np.maximum(time_upper - times, 0.0)
        if np.any(left_error > 0.0) or np.any(right_error > 0.0):
            axis.errorbar(
                times,
                rmse,
                xerr=np.vstack((left_error, right_error)),
                fmt="none",
                ecolor=style["color"],
                elinewidth=1.35,
                capsize=3.0,
                capthick=1.2,
                alpha=0.45,
                zorder=max(1, int(style["zorder"]) - 1),
            )
        axis.plot(
            times,
            rmse,
            label=METHOD_LABELS[method],
            **_plot_style(method),
        )

    if min(all_time_lower + all_time_upper) <= 0.0:
        raise ValueError("log-time axis requires strictly positive values")
    if min(all_rmse_lower + all_rmse_upper) <= 0.0:
        raise ValueError("log-RMSE axis requires strictly positive values")
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=18)
    )
    axis.xaxis.set_major_formatter(FuncFormatter(_time_label))
    axis.xaxis.set_minor_locator(NullLocator())
    axis.yaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=16)
    )
    axis.yaxis.set_major_formatter(FuncFormatter(_scientific_label))
    axis.yaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
            numticks=30,
        )
    )
    axis.set_xlim(
        *_padded_log_limits(all_time_lower, all_time_upper, fraction=0.055)
    )
    y_lower, y_upper = _padded_log_limits(
        all_rmse_lower, all_rmse_upper, fraction=0.08
    )
    # Reserve a data-free band for the in-axes legend.  This keeps the legend
    # genuinely clear of uncertainty fills and curves in the vector export.
    y_upper = max(y_upper, max(all_rmse_upper) * 2.25)
    axis.set_ylim(y_lower, y_upper)
    axis.set_xlabel("Estimated time (s)", fontsize=22.0, labelpad=9)
    axis.set_ylabel("RMSE", fontsize=22.0, labelpad=9)
    axis.grid(
        True, which="major", color="#D8DDE4", linewidth=1.0, alpha=0.94
    )
    axis.grid(
        True,
        which="minor",
        axis="y",
        color="#EEF1F4",
        linewidth=0.72,
        alpha=0.84,
    )
    axis.tick_params(axis="both", which="major", labelsize=15.5, width=1.15)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#515A66")
        spine.set_linewidth(1.35)
    axis.set_box_aspect(1.0)
    legend = axis.legend(
        handles=[_legend_handle(method) for method in LEGEND_HANDLE_ORDER],
        labels=[METHOD_LABELS[method] for method in LEGEND_HANDLE_ORDER],
        loc="upper right",
        bbox_to_anchor=(0.982, 0.982),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=0.96,
        fontsize=14.5,
        borderpad=0.55,
        labelspacing=0.35,
        columnspacing=0.8,
        handlelength=1.95,
        borderaxespad=0.0,
    )
    _trim_grid_around_legend(figure, axis, legend)
    _validate_figure_contract(figure)
    return figure


def _validate_figure_contract(figure: Any) -> None:
    if len(figure.axes) != 1 or figure.legends:
        raise RuntimeError("Wine Efficiency figure must contain one axes")
    axis = figure.axes[0]
    if axis.get_title():
        raise RuntimeError("Wine Efficiency figure must not contain a title")
    if axis.get_xscale() != "log" or axis.get_yscale() != "log":
        raise RuntimeError("Wine Efficiency figure must be log-log")
    if not all(spine.get_visible() for spine in axis.spines.values()):
        raise RuntimeError("Wine Efficiency figure must have a full axis box")
    legend = axis.get_legend()
    if legend is None:
        raise RuntimeError("Wine Efficiency legend must be inside the axes")
    labels = [text.get_text() for text in legend.get_texts()]
    if labels != [METHOD_LABELS[method] for method in LEGEND_HANDLE_ORDER]:
        raise RuntimeError("Wine Efficiency legend order is wrong")
    visible_lines = [
        line for line in axis.lines if not line.get_label().startswith("_")
    ]
    if len(visible_lines) != len(METHOD_ORDER):
        raise RuntimeError("Wine Efficiency figure has the wrong curve count")


def _plotted_values(report: Mapping[str, Any]) -> dict[str, Any]:
    rows = _ordered_rows(report)
    values: dict[str, Any] = {}
    for method in METHOD_ORDER:
        times, time_lower, time_upper, rmse, rmse_lower, rmse_upper = _series(
            rows, method
        )
        values[method] = {
            "label": METHOD_LABELS[method],
            "mean_estimated_total_seconds": times.tolist(),
            "estimated_total_seconds_range": np.column_stack(
                (time_lower, time_upper)
            ).tolist(),
            "aggregate_rmse": rmse.tolist(),
            "aggregate_rmse_bootstrap_95": np.column_stack(
                (rmse_lower, rmse_upper)
            ).tolist(),
            "mean_actual_utility_calls": [
                float(row["methods"][method]["mean_actual_utility_calls"])
                for row in rows
            ],
            "mean_charged_design_seconds": [
                float(row["methods"][method]["mean_charged_design_seconds"])
                for row in rows
            ],
            "mean_charged_sdiff_end_to_end_wall_upper_bound_seconds": [
                float(
                    row["methods"][method][
                        "mean_charged_sdiff_end_to_end_wall_upper_bound_seconds"
                    ]
                )
                for row in rows
            ],
            "mean_charged_extra_seconds": [
                float(row["methods"][method]["mean_charged_extra_seconds"])
                for row in rows
            ],
        }
    return values


def save_figure(
    report: Mapping[str, Any],
    output: Path,
    *,
    source_report: Path,
    alignment_json: Path | None = None,
    alignment_overlay: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    validate_plot_report(report)
    if output.suffix.lower() != ".png":
        raise ValueError("Wine Efficiency output must be a PNG path")
    if (alignment_json is None) != (alignment_overlay is None):
        raise ValueError("alignment paths must be provided together")
    output = output.resolve()
    source_report = source_report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    svg = output.with_suffix(".svg")
    metadata_path = output.with_suffix(".metadata.json")

    figure = build_figure(report)
    alignment_metadata: dict[str, Any] = {
        "status": "not_applicable_single_panel"
    }
    if alignment_json is not None and alignment_overlay is not None:
        try:
            from audit_panel_alignment import require_matplotlib_panel_alignment
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "alignment QA requested but audit_panel_alignment is not importable"
            ) from error
        alignment_json = alignment_json.resolve()
        alignment_overlay = alignment_overlay.resolve()
        alignment_json.parent.mkdir(parents=True, exist_ok=True)
        alignment_overlay.parent.mkdir(parents=True, exist_ok=True)
        alignment = require_matplotlib_panel_alignment(
            figure,
            json_out=alignment_json,
            overlay_svg=alignment_overlay,
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
            panel_ids=["wine"],
        )
        alignment_metadata = {
            "status": str(alignment.get("verdict")),
            "json": {
                "path": str(alignment_json),
                "sha256": _sha256(alignment_json),
            },
            "overlay_svg": {
                "path": str(alignment_overlay),
                "sha256": _sha256(alignment_overlay),
            },
        }
    figure.savefig(output, dpi=600, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)

    for artifact in (output, pdf, svg):
        if not artifact.is_file() or artifact.stat().st_size < 1_000:
            raise RuntimeError(f"figure output is missing/small: {artifact}")
    image = plt.imread(output)
    if image.ndim != 3 or min(image.shape[:2]) < 1_500:
        raise RuntimeError("Wine Efficiency PNG failed dimension validation")
    with pdf.open("rb") as stream:
        if stream.read(4) != b"%PDF":
            raise RuntimeError("Wine Efficiency PDF signature is invalid")

    metadata = {
        "status": "complete",
        "artifact": ARTIFACT_ID,
        "experiment": EXPERIMENT_ID,
        "dataset": "wine",
        "methods": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "layout": "1x1_near_square",
        "chart_title": None,
        "legend_location": "inside_axes_upper_right",
        "full_axis_box": True,
        "scales": {"x": "log", "y": "log"},
        "axis_semantics": {
            "x": "estimated utility-dominated one-shot time in seconds",
            "y": "aggregate RMSE against the high-budget Wine MC reference",
        },
        "time_proxy": {
            "formula": report["configuration"]["time_formula"],
            "utility_seconds_per_call": report["utility_calibration"][
                "median_scalar_seconds_per_call"
            ],
            "actual_call_policy": report["configuration"][
                "actual_calls_policy"
            ],
            "charged_extra_time_components": report["configuration"][
                "charged_extra_time_components"
            ],
            "utility_cost_caveat": report["configuration"][
                "utility_cost_caveat"
            ],
            "sdiff_surcharge_caveat": report["configuration"][
                "sdiff_surcharge_caveat"
            ],
        },
        "source_report": {
            "path": str(source_report),
            "sha256": _sha256(source_report),
        },
        "plotted_values": _plotted_values(report),
        "panel_alignment": alignment_metadata,
        "outputs": {
            "png": {"path": str(output), "sha256": _sha256(output)},
            "pdf": {"path": str(pdf), "sha256": _sha256(pdf)},
            "svg": {"path": str(svg), "sha256": _sha256(svg)},
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output, pdf, svg, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alignment-json", type=Path)
    parser.add_argument("--alignment-overlay", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    outputs = save_figure(
        report,
        args.output,
        source_report=args.report,
        alignment_json=args.alignment_json,
        alignment_overlay=args.alignment_overlay,
    )
    print("saved " + ", ".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()

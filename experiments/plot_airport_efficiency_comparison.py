"""Plot Airport RMSE against an audited estimated-time proxy.

The x-axis is not measured end-to-end wall-clock time.  It is reconstructed
from each method's physical utility calls and one scalar Airport-utility
calibration.  The clean production design time is additionally charged to
INSIDE-Greedy.  Every other method is assigned zero non-utility overhead, so
the comparison is deliberately conservative in favour of the baselines.
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
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np

from experiments.run_airport_efficiency_comparison import (
    EXPERIMENT_ID,
    METHOD_LABELS,
    METHOD_ORDER,
    validate_efficiency_report,
)


ARTIFACT_ID = "airport_estimated_time_efficiency"
DEFAULT_REPORT = Path("results/json/airport_efficiency_k64_lambda1over16.json")
DEFAULT_OUTPUT = Path("results/pdf/airport_efficiency_k64_lambda1over16.png")

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "inside_greedy": {
        "color": "#D97706",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 4.6,
        "markersize": 9.8,
        "filled": True,
        "zorder": 10,
    },
    "inside_orbit": {
        "color": "#C2410C",
        "marker": "h",
        "linestyle": "-",
        "linewidth": 4.4,
        "markersize": 10.4,
        "filled": True,
        "zorder": 9,
    },
    "ofa": {
        "color": "#2563A6",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 3.2,
        "markersize": 8.8,
        "filled": False,
        "zorder": 6,
    },
    "cc": {
        "color": "#69717C",
        "marker": "X",
        "linestyle": ":",
        "linewidth": 3.0,
        "markersize": 9.0,
        "filled": False,
        "zorder": 4,
    },
    "s_diff": {
        "color": "#63801C",
        "marker": "v",
        "linestyle": "-.",
        "linewidth": 3.1,
        "markersize": 9.1,
        "filled": False,
        "zorder": 5,
    },
    "kernel_shap": {
        "color": "#7C3AED",
        "marker": "s",
        "linestyle": "--",
        "linewidth": 3.0,
        "markersize": 8.6,
        "filled": False,
        "zorder": 3,
    },
    "tmc_shapley": {
        "color": "#343B45",
        "marker": "P",
        "linestyle": (0, (5, 2, 1, 2)),
        "linewidth": 3.0,
        "markersize": 9.0,
        "filled": False,
        "zorder": 7,
    },
}

if tuple(METHOD_STYLES) != METHOD_ORDER:
    raise RuntimeError("efficiency styles do not match the method order")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _finite_positive(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{path} must be positive and finite")
    return result


def _ordered_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("efficiency report must have five budget rows")
    ordered = sorted(rows, key=lambda row: int(row["budget_index"]))
    if [int(row["budget_index"]) for row in ordered] != list(range(5)):
        raise ValueError("efficiency budget indices are invalid")
    return ordered


def validate_plot_report(report: Mapping[str, Any]) -> None:
    """Validate the timing identity and every scalar consumed by the plot."""
    validate_efficiency_report(report)
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("unexpected efficiency experiment")
    configuration = report["configuration"]
    if configuration.get("time_interpretation") != (
        "estimated utility-dominated one-shot time"
    ):
        raise ValueError("the report does not contain the expected time proxy")
    if configuration.get("charged_nonutility_overhead") != [
        "INSIDE-Greedy production design time"
    ]:
        raise ValueError("unexpected non-utility overhead policy")

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
                lower_time - tolerance
                <= mean_time
                <= upper_time + tolerance
            ):
                raise ValueError(f"{method} mean time lies outside its range")
            if mean_time <= previous_time:
                raise ValueError(f"{method} time coordinates must increase")
            previous_time = mean_time

            _finite_positive(
                summary.get("aggregate_rmse"), path=f"{method}.RMSE"
            )
            interval = summary.get("aggregate_rmse_bootstrap_95")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{method} has an invalid RMSE interval")
            lower_rmse, upper_rmse = (
                _finite_positive(value, path=f"{method}.RMSE interval")
                for value in interval
            )
            if lower_rmse > upper_rmse:
                raise ValueError(f"{method} RMSE interval is reversed")


def _apply_rcparams() -> None:
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


def _time_label(value: float, _: int) -> str:
    if value >= 1_000.0:
        return f"{value / 1_000.0:g}k"
    if value >= 1.0:
        return f"{value:g}"
    return f"{value:.2g}"


def _scientific_label(value: float, _: int) -> str:
    return f"{value:.0e}"


def _plot_style(method: str) -> dict[str, Any]:
    style = METHOD_STYLES[method]
    return {
        "color": style["color"],
        "marker": style["marker"],
        "linestyle": style["linestyle"],
        "linewidth": style["linewidth"],
        "markersize": style["markersize"],
        "markerfacecolor": style["color"] if style["filled"] else "white",
        "markeredgecolor": style["color"],
        "markeredgewidth": 1.5,
        "zorder": style["zorder"],
        "clip_on": False,
    }


def _legend_handle(method: str) -> Line2D:
    return Line2D([], [], label=METHOD_LABELS[method], **_plot_style(method))


def _series(
    rows: Sequence[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    time_lower: list[float] = []
    time_upper: list[float] = []
    rmse: list[float] = []
    rmse_lower: list[float] = []
    rmse_upper: list[float] = []
    for row in rows:
        summary = row["methods"][method]
        times.append(float(summary["mean_estimated_total_seconds"]))
        time_range = summary["estimated_total_seconds_range"]
        time_lower.append(float(time_range[0]))
        time_upper.append(float(time_range[1]))
        rmse.append(float(summary["aggregate_rmse"]))
        interval = summary["aggregate_rmse_bootstrap_95"]
        rmse_lower.append(float(interval[0]))
        rmse_upper.append(float(interval[1]))
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            times,
            time_lower,
            time_upper,
            rmse,
            rmse_lower,
            rmse_upper,
        )
    )


def _padded_log_limits(
    lower: Sequence[float], upper: Sequence[float], *, fraction: float
) -> tuple[float, float]:
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if np.any(low <= 0.0) or np.any(high <= 0.0):
        raise ValueError("log limits require positive values")
    log_min = float(np.log10(low.min()))
    log_max = float(np.log10(high.max()))
    span = max(log_max - log_min, 0.25)
    return 10.0 ** (log_min - fraction * span), 10.0 ** (
        log_max + fraction * span
    )


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
    axis.set_ylim(
        *_padded_log_limits(all_rmse_lower, all_rmse_upper, fraction=0.08)
    )
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
    axis.legend(
        handles=[_legend_handle(method) for method in METHOD_ORDER],
        labels=[METHOD_LABELS[method] for method in METHOD_ORDER],
        loc="upper right",
        bbox_to_anchor=(0.982, 0.982),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=0.96,
        fontsize=15.5,
        borderpad=0.55,
        labelspacing=0.38,
        handlelength=2.35,
        borderaxespad=0.0,
    )
    _validate_figure_contract(figure)
    return figure


def _validate_figure_contract(figure: Any) -> None:
    if len(figure.axes) != 1 or figure.legends:
        raise RuntimeError("efficiency figure must contain one axes")
    axis = figure.axes[0]
    if axis.get_title():
        raise RuntimeError("efficiency figure must not contain a title")
    if axis.get_xscale() != "log" or axis.get_yscale() != "log":
        raise RuntimeError("efficiency figure must be log-log")
    if not all(spine.get_visible() for spine in axis.spines.values()):
        raise RuntimeError("efficiency figure must have a full axis box")
    legend = axis.get_legend()
    if legend is None:
        raise RuntimeError("efficiency legend must be inside the axes")
    labels = [text.get_text() for text in legend.get_texts()]
    if labels != [METHOD_LABELS[method] for method in METHOD_ORDER]:
        raise RuntimeError("efficiency legend order is wrong")
    visible_lines = [
        line for line in axis.lines if not line.get_label().startswith("_")
    ]
    if len(visible_lines) != len(METHOD_ORDER):
        raise RuntimeError("efficiency figure has the wrong curve count")


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
        }
    return values


def save_figure(
    report: Mapping[str, Any], output: Path, *, source_report: Path
) -> tuple[Path, Path, Path, Path]:
    validate_plot_report(report)
    if output.suffix.lower() != ".png":
        raise ValueError("efficiency output must be a PNG path")
    output = output.resolve()
    source_report = source_report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    svg = output.with_suffix(".svg")
    metadata_path = output.with_suffix(".metadata.json")

    figure = build_figure(report)
    figure.savefig(output, dpi=600, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)

    for artifact in (output, pdf, svg):
        if not artifact.is_file() or artifact.stat().st_size < 1_000:
            raise RuntimeError(f"figure output is missing/small: {artifact}")
    image = plt.imread(output)
    if image.ndim != 3 or min(image.shape[:2]) < 1_500:
        raise RuntimeError("efficiency PNG failed raster-dimension validation")
    with pdf.open("rb") as stream:
        if stream.read(4) != b"%PDF":
            raise RuntimeError("efficiency PDF signature is invalid")

    script = Path(__file__).resolve()
    metadata = {
        "status": "complete",
        "artifact": ARTIFACT_ID,
        "experiment": EXPERIMENT_ID,
        "dataset": "airport",
        "methods": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "layout": "1x1_near_square",
        "chart_title": None,
        "legend_location": "inside_axes",
        "full_axis_box": True,
        "scales": {"x": "log", "y": "log"},
        "axis_semantics": {
            "x": "estimated utility-dominated one-shot time in seconds",
            "y": "aggregate RMSE against exact Airport Shapley values",
        },
        "time_proxy": {
            "formula": report["configuration"]["time_formula"],
            "utility_seconds_per_call": report["utility_calibration"][
                "median_scalar_seconds_per_call"
            ],
            "actual_call_policy": report["configuration"][
                "actual_calls_policy"
            ],
            "charged_overhead": report["configuration"][
                "charged_nonutility_overhead"
            ],
            "omitted_overhead": report["configuration"][
                "omitted_nonutility_overhead"
            ],
            "caveat": report["configuration"]["baseline_favouring_caveat"],
            "horizontal_range": (
                "min/max over the three repeat-specific actual-call and "
                "charged-design times at the fixed median utility calibration"
            ),
        },
        "uncertainty": "repeat bootstrap 95% intervals for aggregate RMSE",
        "panel_alignment": "not_applicable_single_panel",
        "source_report": {
            "path": str(source_report),
            "sha256": _sha256(source_report),
        },
        "plotting_script": {"path": str(script), "sha256": _sha256(script)},
        "plotted_values": _plotted_values(report),
        "outputs": {
            "png": {"path": str(output), "sha256": _sha256(output)},
            "pdf": {"path": str(pdf), "sha256": _sha256(pdf)},
            "svg": {"path": str(svg), "sha256": _sha256(svg)},
        },
        "validation": {
            "source_schema_and_timing_identity": "passed",
            "plotted_scalar_validation": "passed",
            "figure_contract": "passed",
            "raster_and_pdf_signature": "passed",
            "output_hashes": "passed",
        },
    }
    _atomic_write_json(metadata_path, metadata)
    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, artifact in (("png", output), ("pdf", pdf), ("svg", svg)):
        if saved_metadata["outputs"][key]["sha256"] != _sha256(artifact):
            raise RuntimeError(f"{key} output hash does not validate")
    return output, pdf, svg, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", nargs="?", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = save_figure(report, args.output, source_report=report_path)
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()

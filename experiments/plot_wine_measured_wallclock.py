"""Plot Wine RMSE against freshly measured cold end-to-end wall time."""

from __future__ import annotations

import argparse
import csv
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
from experiments.run_wine_measured_wallclock import (
    EXPERIMENT_ID,
    METHOD_LABELS,
    METHOD_ORDER,
)


ARTIFACT_ID = "wine_fresh_measured_cold_wallclock_efficiency"
DEFAULT_REPORT = Path(
    "results/json/wine_measured_wallclock_k64_lambda1over16_"
    "3repeats_71k_1p42m.json"
)
DEFAULT_OUTPUT = Path("results/pdf/wine_measured_wallclock_efficiency.png")
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
    raise RuntimeError("measured Wine and shared Efficiency styles disagree")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
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


def _ordered_rows(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = report.get("results")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("measured Wine report must have five result rows")
    ordered = sorted(rows, key=lambda row: int(row["budget_index"]))
    if [int(row["budget_index"]) for row in ordered] != list(range(5)):
        raise ValueError("measured Wine budget indices are invalid")
    return ordered


def validate_plot_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "complete":
        raise ValueError("measured Wine report is not complete")
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("unexpected measured Wine experiment")
    configuration = report.get("configuration", {})
    if tuple(configuration.get("methods", ())) != METHOD_ORDER:
        raise ValueError("measured Wine method order is wrong")
    if int(configuration.get("repeats", -1)) != 3:
        raise ValueError("measured Wine report must contain three repeats")
    protocol = configuration.get("timing_protocol", {})
    if protocol.get("kind") != "fresh cold end-to-end wall clock per cell":
        raise ValueError("measured Wine report has the wrong timing protocol")
    cells = report.get("cells")
    if not isinstance(cells, list) or len(cells) != 105:
        raise ValueError("measured Wine report must contain 105 fresh cells")

    for method in METHOD_ORDER:
        previous = 0.0
        for row_index, row in enumerate(_ordered_rows(report)):
            summary = row.get("methods", {}).get(method)
            if not isinstance(summary, Mapping):
                raise ValueError(f"row {row_index} is missing {method}")
            if summary.get("label") != METHOD_LABELS[method]:
                raise ValueError(f"{method} has the wrong label")
            times = summary.get("cold_end_to_end_seconds_by_repeat")
            if not isinstance(times, list) or len(times) != 3:
                raise ValueError(f"{method} has invalid repeat timings")
            time_array = np.asarray(
                [_finite_positive(value, path=f"{method}.time") for value in times]
            )
            mean_time = _finite_positive(
                summary.get("mean_cold_end_to_end_seconds"),
                path=f"{method}.mean time",
            )
            if not math.isclose(
                mean_time,
                float(time_array.mean()),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{method} mean wall time is inconsistent")
            observed_range = summary.get("cold_end_to_end_seconds_range")
            if not isinstance(observed_range, list) or len(observed_range) != 2:
                raise ValueError(f"{method} has an invalid timing range")
            if not np.allclose(
                observed_range,
                [time_array.min(), time_array.max()],
                rtol=1e-12,
                atol=1e-12,
            ):
                raise ValueError(f"{method} timing range is inconsistent")
            if mean_time <= previous:
                raise ValueError(f"{method} wall time must increase with budget")
            previous = mean_time
            _finite_positive(summary.get("aggregate_rmse"), path=f"{method}.RMSE")
            interval = summary.get("aggregate_rmse_bootstrap_95")
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"{method} has an invalid RMSE interval")
            lower, upper = (
                _finite_positive(value, path=f"{method}.RMSE interval")
                for value in interval
            )
            if lower > upper:
                raise ValueError(f"{method} RMSE interval is reversed")


def _series(
    rows: Sequence[Mapping[str, Any]], method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points: list[list[float]] = [[] for _ in range(6)]
    for row in rows:
        summary = row["methods"][method]
        time_range = summary["cold_end_to_end_seconds_range"]
        rmse_range = summary["aggregate_rmse_bootstrap_95"]
        values = (
            float(summary["mean_cold_end_to_end_seconds"]),
            float(time_range[0]),
            float(time_range[1]),
            float(summary["aggregate_rmse"]),
            float(rmse_range[0]),
            float(rmse_range[1]),
        )
        for destination, value in zip(points, values, strict=True):
            destination.append(value)
    return tuple(np.asarray(values, dtype=np.float64) for values in points)


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


def build_figure(report: Mapping[str, Any]) -> Any:
    validate_plot_report(report)
    rows = _ordered_rows(report)
    _apply_rcparams()
    figure, axis = plt.subplots(figsize=(8.9, 8.9))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.13, top=0.972)

    all_x_low: list[float] = []
    all_x_high: list[float] = []
    all_y_low: list[float] = []
    all_y_high: list[float] = []
    for method in METHOD_ORDER:
        x, x_low, x_high, y, y_low, y_high = _series(rows, method)
        style = METHOD_STYLES[method]
        all_x_low.extend(x_low.tolist())
        all_x_high.extend(x_high.tolist())
        all_y_low.extend(y_low.tolist())
        all_y_high.extend(y_high.tolist())
        axis.fill_between(
            x,
            y_low,
            y_high,
            color=style["color"],
            alpha=0.10 if style["filled"] else 0.065,
            linewidth=0.0,
            zorder=max(1, int(style["zorder"]) - 2),
        )
        axis.errorbar(
            x,
            y,
            xerr=np.vstack((np.maximum(x - x_low, 0), np.maximum(x_high - x, 0))),
            fmt="none",
            ecolor=style["color"],
            elinewidth=1.35,
            capsize=3.0,
            capthick=1.2,
            alpha=0.45,
            zorder=max(1, int(style["zorder"]) - 1),
        )
        axis.plot(x, y, label=METHOD_LABELS[method], **_plot_style(method))

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
        LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=30)
    )
    axis.set_xlim(*_padded_log_limits(all_x_low, all_x_high, fraction=0.055))
    y_lower, y_upper = _padded_log_limits(
        all_y_low, all_y_high, fraction=0.08
    )
    axis.set_ylim(y_lower, max(y_upper, max(all_y_high) * 2.25))
    axis.set_xlabel("Measured wall time (s)", fontsize=22.0, labelpad=9)
    axis.set_ylabel("RMSE", fontsize=22.0, labelpad=9)
    axis.grid(False)
    axis.tick_params(axis="both", which="major", labelsize=15.5, width=1.15)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#515A66")
        spine.set_linewidth(1.35)
    axis.set_box_aspect(1.0)
    axis.legend(
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
    return figure


def save_figure(
    report: Mapping[str, Any],
    output: Path,
    *,
    source_report: Path,
    alignment_json: Path | None = None,
    alignment_overlay: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    validate_plot_report(report)
    if output.suffix.lower() != ".png":
        raise ValueError("output must be a PNG path")
    if (alignment_json is None) != (alignment_overlay is None):
        raise ValueError("alignment paths must be provided together")
    output = output.resolve()
    source_report = source_report.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    svg = output.with_suffix(".svg")
    csv_path = output.with_suffix(".csv")
    metadata_path = output.with_suffix(".metadata.json")

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "method",
                "label",
                "budget_index",
                "target_total_utility_calls",
                "mean_actual_utility_calls",
                "mean_cold_end_to_end_seconds",
                "min_cold_end_to_end_seconds",
                "max_cold_end_to_end_seconds",
                "mean_estimator_seconds",
                "aggregate_rmse",
                "rmse_bootstrap_95_low",
                "rmse_bootstrap_95_high",
            )
        )
        for row in _ordered_rows(report):
            for method in METHOD_ORDER:
                summary = row["methods"][method]
                time_range = summary["cold_end_to_end_seconds_range"]
                rmse_range = summary["aggregate_rmse_bootstrap_95"]
                writer.writerow(
                    (
                        method,
                        summary["label"],
                        row["budget_index"],
                        row["target_total_utility_calls"],
                        summary["mean_actual_utility_calls"],
                        summary["mean_cold_end_to_end_seconds"],
                        time_range[0],
                        time_range[1],
                        summary["mean_estimator_seconds"],
                        summary["aggregate_rmse"],
                        rmse_range[0],
                        rmse_range[1],
                    )
                )

    figure = build_figure(report)
    alignment_metadata: dict[str, Any] = {
        "status": "not_applicable_single_panel"
    }
    if alignment_json is not None and alignment_overlay is not None:
        from audit_panel_alignment import require_matplotlib_panel_alignment

        alignment = require_matplotlib_panel_alignment(
            figure,
            json_out=alignment_json.resolve(),
            overlay_svg=alignment_overlay.resolve(),
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
            panel_ids=["wine"],
        )
        alignment_metadata = {
            "status": str(alignment.get("verdict")),
            "json": str(alignment_json.resolve()),
            "overlay_svg": str(alignment_overlay.resolve()),
        }
    figure.savefig(output, dpi=600, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    figure.savefig(svg, facecolor="white")
    plt.close(figure)

    metadata = {
        "status": "complete",
        "artifact": ARTIFACT_ID,
        "figure_claim": (
            "Accuracy-efficiency is compared using fresh measured cold "
            "end-to-end runtime rather than a utility-call proxy."
        ),
        "archetype": "single-panel quantitative comparison",
        "methods": list(METHOD_ORDER),
        "layout": "1x1_near_square",
        "chart_title": None,
        "legend_location": "inside_axes_upper_right",
        "full_axis_box": True,
        "scales": {"x": "log", "y": "log"},
        "axis_semantics": {
            "x": "mean fresh cold end-to-end wall time in seconds",
            "x_interval": "minimum to maximum across three repeats",
            "y": "aggregate RMSE against the fixed high-budget reference",
            "y_interval": "repeat bootstrap 95% interval",
        },
        "statistics": {
            "repeat_unit": "independent method-budget run with a fresh pool",
            "repeats": 3,
            "x_center": "arithmetic mean wall time",
            "x_spread": "minimum to maximum repeat time",
            "y_center": "RMSE of the repeat-mean Shapley vector",
            "y_spread": "95% bootstrap interval over repeats",
        },
        "qa_notes": [
            (
                "The 8.9-inch square working export intentionally follows the "
                "requested near-square layout rather than a journal column width."
            ),
            (
                "PNG is the raster preview; PDF and SVG are the editable "
                "publication masters, so no TIFF preview is duplicated here."
            ),
            (
                "All plotted wall times and RMSE bounds pass explicit finite, "
                "strictly-positive validation before log scaling."
            ),
        ],
        "source_report": {
            "path": str(source_report),
            "sha256": _sha256(source_report),
        },
        "panel_alignment": alignment_metadata,
        "outputs": {
            "png": str(output),
            "pdf": str(pdf),
            "svg": str(svg),
            "csv": str(csv_path),
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output, pdf, svg, csv_path, metadata_path


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

"""Plot the publication-facing Airport INSIDE ablations.

The companion runner writes one auditable report containing three related
experiments.  This module renders them as three independent artifacts:

* a four-curve component ablation;
* a two-curve Random-Orbit/INSIDE-Orbit ablation;
* a 1x3 geometry audit (first moment, second moment, and D2/RMSE scatter).

The plotting contract intentionally keeps exact zeros exact.  In particular,
complete cyclic orbits have zero fixed-slice first-moment error; those points
are drawn at ``y=0`` on a linear axis and are never replaced by an epsilon.
Each PNG/PDF pair has a companion metadata JSON with source/output hashes and
the exact values used by the renderer.
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
from matplotlib.ticker import FuncFormatter, LogLocator, MaxNLocator, NullLocator
import numpy as np

from experiments.run_airport_inside_ablations import (
    COMPONENT_METHODS,
    EXPERIMENT_ID,
    GEOMETRY_METHODS,
    METHOD_LABELS as SOURCE_METHOD_LABELS,
    METHOD_ORDER,
    ORBIT_METHODS,
    PROTOCOL_VERSION,
    validate_report as validate_runner_report,
)


METHOD_LABELS = {**SOURCE_METHOD_LABELS, "full_inside": "INSIDE-Coalition"}

COMPONENT_ARTIFACT = "airport_inside_component_ablation"
ORBIT_ARTIFACT = "airport_inside_orbit_ablation"
GEOMETRY_ARTIFACT = "airport_inside_geometry_audit"

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "iid_ofa": {
        "color": "#2563A6",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 3.2,
        "markersize": 8.8,
        "filled": False,
        "zorder": 3,
    },
    "first_only": {
        "color": "#7C3AED",
        "marker": "^",
        "linestyle": "-.",
        "linewidth": 3.3,
        "markersize": 9.0,
        "filled": False,
        "zorder": 4,
    },
    "frame_only": {
        "color": "#16835A",
        "marker": "s",
        "linestyle": ":",
        "linewidth": 3.4,
        "markersize": 8.7,
        "filled": False,
        "zorder": 5,
    },
    "full_inside": {
        "color": "#D97706",
        "marker": "D",
        "linestyle": "-",
        "linewidth": 4.5,
        "markersize": 9.4,
        "filled": True,
        "zorder": 8,
    },
    "random_orbit": {
        "color": "#596574",
        "marker": "o",
        "linestyle": "--",
        "linewidth": 3.3,
        "markersize": 9.0,
        "filled": False,
        "zorder": 4,
    },
    "inside_orbit": {
        "color": "#C2410C",
        "marker": "h",
        "linestyle": "-",
        "linewidth": 4.5,
        "markersize": 10.0,
        "filled": True,
        "zorder": 8,
    },
}

if set(METHOD_STYLES) != set(METHOD_ORDER):
    raise RuntimeError("plot styles do not cover the Airport ablation methods")


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    return value


def _finite(value: Any, *, path: str, nonnegative: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{path} must be nonnegative")
    return result


def _positive(value: Any, *, path: str) -> float:
    result = _finite(value, path=path)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _interval(
    value: Any, *, path: str, allow_zero: bool
) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must have two endpoints")
    lower = _finite(value[0], path=f"{path}.lower", nonnegative=True)
    upper = _finite(value[1], path=f"{path}.upper", nonnegative=True)
    if not allow_zero and (lower <= 0.0 or upper <= 0.0):
        raise ValueError(f"{path} endpoints must be positive")
    if lower > upper:
        raise ValueError(f"{path} endpoints are reversed")
    return lower, upper


def _ordered_budget_rows(
    report: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    configuration = _mapping(report.get("configuration"), path="configuration")
    configured = tuple(
        int(value) for value in configuration.get("inner_utility_call_budgets", ())
    )
    if len(configured) != 5 or any(
        left >= right for left, right in zip(configured, configured[1:])
    ):
        raise ValueError("the Airport ablation requires five increasing budgets")
    raw = _mapping(
        report.get("results_by_inner_budget"),
        path="results_by_inner_budget",
    )
    observed = {int(key) for key in raw}
    if observed != set(configured):
        raise ValueError("aggregate budget keys do not match configuration")
    rows = [_mapping(raw[str(value)], path=f"budget[{value}]") for value in configured]
    calls = np.asarray(
        [
            _positive(
                row.get("total_utility_calls"),
                path=f"budget[{budget}].total_utility_calls",
            )
            for row, budget in zip(rows, configured, strict=True)
        ],
        dtype=np.float64,
    )
    if not np.all(np.diff(calls) > 0.0):
        raise ValueError("total utility-call coordinates must increase")
    return rows


def _raw_cells_by_key(
    report: Mapping[str, Any],
) -> dict[tuple[int, int], Mapping[str, Any]]:
    cells = report.get("raw_cells")
    if not isinstance(cells, list):
        raise ValueError("raw_cells must be a list")
    result: dict[tuple[int, int], Mapping[str, Any]] = {}
    for raw in cells:
        cell = _mapping(raw, path="raw cell")
        key = (int(cell["budget_index"]), int(cell["repeat"]))
        if key in result:
            raise ValueError(f"duplicate raw cell {key}")
        result[key] = cell
    return result


def validate_plot_report(report: Mapping[str, Any]) -> None:
    """Validate runner output and recompute every plotted aggregate."""
    validate_runner_report(report)
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("unexpected Airport ablation experiment id")
    configuration = _mapping(report.get("configuration"), path="configuration")
    if configuration.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("unexpected Airport ablation protocol version")
    if tuple(configuration.get("methods", ())) != METHOD_ORDER:
        raise ValueError("method order differs from the plotting contract")
    if configuration.get("geometry_aggregation") != (
        "equal-size macro RMS over s=2,...,98"
    ):
        raise ValueError("the geometry aggregation is not the audited macro RMS")
    repeats = int(configuration.get("repeats", -1))
    if repeats < 2:
        raise ValueError("at least two repeats are required")

    rows = _ordered_budget_rows(report)
    raw = _raw_cells_by_key(report)
    if len(raw) != repeats * len(rows):
        raise ValueError("raw cell grid is incomplete")

    for budget_index, row in enumerate(rows):
        methods = _mapping(row.get("methods"), path=f"budget[{budget_index}].methods")
        if set(methods) != set(METHOD_ORDER):
            raise ValueError(f"budget[{budget_index}] has the wrong method set")
        for method in METHOD_ORDER:
            summary = _mapping(
                methods[method], path=f"budget[{budget_index}].{method}"
            )
            if summary.get("label") != SOURCE_METHOD_LABELS[method]:
                raise ValueError(f"{method} has an unexpected display label")
            repeat_rows = [
                _mapping(
                    _mapping(
                        raw[(budget_index, repeat)].get("methods"),
                        path="raw methods",
                    ).get(method),
                    path=f"raw[{budget_index},{repeat}].{method}",
                )
                for repeat in range(repeats)
            ]
            observed_rmse = np.asarray(
                summary.get("repeat_rmse"), dtype=np.float64
            )
            expected_rmse = np.asarray(
                [item["repeat_rmse"] for item in repeat_rows], dtype=np.float64
            )
            if (
                observed_rmse.shape != (repeats,)
                or not np.all(np.isfinite(observed_rmse))
                or not np.all(observed_rmse > 0.0)
                or not np.allclose(
                    observed_rmse, expected_rmse, rtol=0.0, atol=1e-14
                )
            ):
                raise ValueError(f"{method} repeat RMSE is inconsistent")
            aggregate_rmse = _positive(
                summary.get("aggregate_rmse"),
                path=f"{method}.aggregate_rmse",
            )
            recomputed_rmse = float(np.sqrt(np.mean(np.square(expected_rmse))))
            if not math.isclose(
                aggregate_rmse, recomputed_rmse, rel_tol=0.0, abs_tol=1e-14
            ):
                raise ValueError(f"{method} aggregate RMSE is inconsistent")
            _interval(
                summary.get("aggregate_rmse_bootstrap_95"),
                path=f"{method}.aggregate_rmse_bootstrap_95",
                allow_zero=False,
            )

            for raw_key, mean_key, interval_key in (
                (
                    "first_moment_rms_by_repeat",
                    "mean_first_moment_rms",
                    "first_moment_mean_bootstrap_95",
                ),
                (
                    "frame_frobenius_rms_by_repeat",
                    "mean_frame_frobenius_rms",
                    "frame_frobenius_mean_bootstrap_95",
                ),
            ):
                observed = np.asarray(summary.get(raw_key), dtype=np.float64)
                geometry_field = (
                    "first_moment_rms"
                    if raw_key.startswith("first")
                    else "frame_frobenius_rms"
                )
                expected = np.asarray(
                    [item["geometry"][geometry_field] for item in repeat_rows],
                    dtype=np.float64,
                )
                if (
                    observed.shape != (repeats,)
                    or not np.all(np.isfinite(observed))
                    or np.any(observed < 0.0)
                    or not np.allclose(observed, expected, rtol=0.0, atol=1e-14)
                ):
                    raise ValueError(f"{method}.{raw_key} is inconsistent")
                stored_mean = _finite(
                    summary.get(mean_key), path=f"{method}.{mean_key}", nonnegative=True
                )
                if not math.isclose(
                    stored_mean,
                    float(expected.mean()),
                    rel_tol=0.0,
                    abs_tol=1e-14,
                ):
                    raise ValueError(f"{method}.{mean_key} is inconsistent")
                _interval(
                    summary.get(interval_key),
                    path=f"{method}.{interval_key}",
                    allow_zero=geometry_field == "first_moment_rms",
                )

        # Complete cyclic orbits must remain exactly at zero.  Requiring
        # equality rather than a tolerance prevents an unnoticed epsilon from
        # entering either the report or the plotting path.
        for method in ORBIT_METHODS:
            first = np.asarray(
                methods[method]["first_moment_rms_by_repeat"], dtype=np.float64
            )
            if not np.all(first == 0.0):
                raise ValueError(f"{method} first moment is not exactly zero")

    relation = _mapping(
        report.get("geometry_error_relation"), path="geometry_error_relation"
    )
    points = relation.get("points")
    expected_count = len(rows) * repeats * len(GEOMETRY_METHODS)
    if not isinstance(points, list) or len(points) != expected_count:
        raise ValueError("geometry-error scatter has an incomplete point grid")
    point_keys: set[tuple[str, int, int]] = set()
    for point_index, value in enumerate(points):
        point = _mapping(value, path=f"geometry point[{point_index}]")
        method = str(point.get("method"))
        budget_index = int(point.get("budget_index", -1))
        repeat = int(point.get("repeat", -1))
        key = (method, budget_index, repeat)
        if method not in GEOMETRY_METHODS or key in point_keys:
            raise ValueError("geometry-error scatter has an invalid/duplicate key")
        point_keys.add(key)
        source = raw[(budget_index, repeat)]["methods"][method]
        if not math.isclose(
            _positive(
                point.get("frame_frobenius_rms"),
                path=f"point[{point_index}].frame_frobenius_rms",
            ),
            float(source["geometry"]["frame_frobenius_rms"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError("scatter frame discrepancy is inconsistent")
        if not math.isclose(
            _positive(
                point.get("repeat_rmse"),
                path=f"point[{point_index}].repeat_rmse",
            ),
            float(source["repeat_rmse"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError("scatter RMSE is inconsistent")
        if int(point.get("total_utility_calls", -1)) != int(
            raw[(budget_index, repeat)]["total_utility_calls"]
        ):
            raise ValueError("scatter call accounting is inconsistent")
    expected_keys = {
        (method, budget_index, repeat)
        for method in GEOMETRY_METHODS
        for budget_index in range(len(rows))
        for repeat in range(repeats)
    }
    if point_keys != expected_keys:
        raise ValueError("geometry-error scatter keys are incomplete")


def _calls_label(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    if value >= 1_000:
        return f"{value / 1_000:.3g}k"
    return f"{value:.0f}"


def _scientific_label(value: float, _: int) -> str:
    if value == 0.0:
        return "0"
    return f"{value:.0e}"


def _apply_rcparams() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "font.size": 20.0,
            "axes.labelcolor": "#303640",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
        }
    )


def _box_axis(axis: Any) -> None:
    axis.tick_params(axis="both", which="major", labelsize=15.5, width=1.15)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#515A66")
        spine.set_linewidth(1.35)
    axis.set_box_aspect(1.0)


def _log_x_axis(axis: Any) -> None:
    axis.set_xscale("log")
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    axis.xaxis.set_minor_locator(NullLocator())


def _log_y_axis(axis: Any) -> None:
    axis.set_yscale("log")
    axis.yaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=16)
    )
    axis.yaxis.set_minor_locator(
        LogLocator(
            base=10,
            subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0),
            numticks=30,
        )
    )
    axis.yaxis.set_major_formatter(FuncFormatter(_scientific_label))


def _grid(axis: Any, *, logarithmic_y: bool) -> None:
    axis.grid(True, which="major", color="#D8DDE4", linewidth=1.0, alpha=0.94)
    if logarithmic_y:
        axis.grid(
            True,
            which="minor",
            axis="y",
            color="#EEF1F4",
            linewidth=0.72,
            alpha=0.84,
        )


def _style_for_plot(method: str) -> dict[str, Any]:
    style = METHOD_STYLES[method]
    return {
        "color": style["color"],
        "marker": style["marker"],
        "linestyle": style["linestyle"],
        "linewidth": style["linewidth"],
        "markersize": style["markersize"],
        "markerfacecolor": style["color"] if style["filled"] else "white",
        "markeredgecolor": style["color"],
        "markeredgewidth": 1.45,
        "zorder": style["zorder"],
        "clip_on": False,
    }


def _legend_handle(method: str) -> Line2D:
    style = _style_for_plot(method)
    return Line2D([], [], label=METHOD_LABELS[method], **style)


def _series(
    rows: Sequence[Mapping[str, Any]],
    method: str,
    *,
    value_key: str,
    interval_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calls: list[float] = []
    values: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        summary = row["methods"][method]
        calls.append(float(row["total_utility_calls"]))
        values.append(float(summary[value_key]))
        interval = summary[interval_key]
        lower.append(float(interval[0]))
        upper.append(float(interval[1]))
    return tuple(
        np.asarray(item, dtype=np.float64)
        for item in (calls, values, lower, upper)
    )


def _log_limits(lower: Sequence[float], upper: Sequence[float]) -> tuple[float, float]:
    low = np.asarray(lower, dtype=np.float64)
    high = np.asarray(upper, dtype=np.float64)
    if np.any(low <= 0.0) or np.any(high <= 0.0):
        raise ValueError("logarithmic limits require positive values")
    log_min = float(np.log10(low.min()))
    log_max = float(np.log10(high.max()))
    span = max(log_max - log_min, 0.25)
    return 10.0 ** (log_min - 0.08 * span), 10.0 ** (
        log_max + 0.08 * span
    )


def _draw_line(
    axis: Any,
    method: str,
    calls: np.ndarray,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Any:
    style = METHOD_STYLES[method]
    axis.fill_between(
        calls,
        lower,
        upper,
        color=style["color"],
        alpha=0.10 if style["filled"] else 0.075,
        linewidth=0.0,
        zorder=max(1, int(style["zorder"]) - 2),
    )
    (handle,) = axis.plot(
        calls,
        values,
        label=METHOD_LABELS[method],
        **_style_for_plot(method),
    )
    return handle


def _build_rmse_figure(
    report: Mapping[str, Any], methods: Sequence[str]
) -> Any:
    rows = _ordered_budget_rows(report)
    _apply_rcparams()
    figure, axis = plt.subplots(figsize=(8.9, 8.9))
    figure.subplots_adjust(left=0.145, right=0.975, bottom=0.125, top=0.970)
    all_lower: list[float] = []
    all_upper: list[float] = []
    handles: list[Any] = []
    for method in methods:
        calls, values, lower, upper = _series(
            rows,
            method,
            value_key="aggregate_rmse",
            interval_key="aggregate_rmse_bootstrap_95",
        )
        handles.append(_draw_line(axis, method, calls, values, lower, upper))
        all_lower.extend(lower.tolist())
        all_upper.extend(upper.tolist())

    target_calls = np.asarray(
        [float(row["total_utility_calls"]) for row in rows], dtype=np.float64
    )
    _log_x_axis(axis)
    _log_y_axis(axis)
    axis.set_xlim(target_calls[0] * 0.80, target_calls[-1] * 1.22)
    axis.set_ylim(*_log_limits(all_lower, all_upper))
    axis.set_xlabel("Utility calls", fontsize=22.0, labelpad=8)
    axis.set_ylabel("RMSE", fontsize=22.0, labelpad=9)
    _grid(axis, logarithmic_y=True)
    _box_axis(axis)
    axis.legend(
        handles=handles,
        labels=[METHOD_LABELS[method] for method in methods],
        loc="upper right",
        bbox_to_anchor=(0.982, 0.982),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=0.96,
        fontsize=18.0 if len(methods) == 4 else 19.0,
        borderpad=0.60,
        columnspacing=1.10,
        handlelength=2.30,
        borderaxespad=0.0,
    )
    return figure


def build_component_figure(report: Mapping[str, Any]) -> Any:
    """Build the four-curve component ablation figure."""
    validate_plot_report(report)
    figure = _build_rmse_figure(report, COMPONENT_METHODS)
    figure.axes[0].grid(False, which="both")
    _validate_single_figure_contract(figure, COMPONENT_METHODS)
    return figure


def build_orbit_figure(report: Mapping[str, Any]) -> Any:
    """Build the paired Random-Orbit/INSIDE-Orbit figure."""
    validate_plot_report(report)
    figure = _build_rmse_figure(report, ORBIT_METHODS)
    figure.axes[0].grid(False, which="both")
    _validate_single_figure_contract(figure, ORBIT_METHODS)
    return figure


def _panel_tag(axis: Any, text: str) -> None:
    axis.text(
        0.035,
        0.955,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=21.0,
        fontweight="semibold",
        color="#20252D",
        bbox={
            "boxstyle": "square,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.84,
        },
        zorder=12,
    )


def _geometry_series(
    rows: Sequence[Mapping[str, Any]], method: str, metric: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keys = {"first": "first_moment_rms_by_repeat",
            "frame": "frame_frobenius_rms_by_repeat"}
    if metric not in keys:
        raise ValueError("geometry metric must be 'first' or 'frame'")
    repeats = np.asarray([row["methods"][method][keys[metric]] for row in rows])
    means = repeats.mean(axis=1)
    std = repeats.std(axis=1, ddof=1)
    calls = np.asarray([row["total_utility_calls"] for row in rows])
    return calls, means, means - std, means + std


def _scatter_trend(report: Mapping[str, Any]) -> dict[str, Any]:
    from scipy.stats import spearmanr
    points = report["geometry_error_relation"]["points"]
    x = np.asarray([p["frame_frobenius_rms"] for p in points])
    y = np.asarray([p["repeat_rmse"] for p in points])
    slope, intercept = np.polyfit(np.log10(x), np.log10(y), 1)
    return {"n": len(x), "spearman_rho": float(spearmanr(x, y).statistic),
            "slope": float(slope), "intercept": float(intercept),
            "fit": "pooled OLS: log10(RMSE) = intercept + slope * log10(D2)",
            "x_range": [float(x.min()), float(x.max())]}


def build_geometry_figure(report: Mapping[str, Any]) -> Any:
    """Build the 1x3 first/frame/scatter geometry figure."""
    validate_plot_report(report)
    rows = _ordered_budget_rows(report)
    _apply_rcparams()
    figure, axes = plt.subplots(1, 3, figsize=(19.2, 7.15))
    figure.subplots_adjust(
        left=0.070, right=0.992, bottom=0.165, top=0.955, wspace=0.315
    )
    first_axis, frame_axis, scatter_axis = axes
    target_calls = np.asarray(
        [float(row["total_utility_calls"]) for row in rows], dtype=np.float64
    )

    first_upper: list[float] = []
    frame_lower: list[float] = []
    frame_upper: list[float] = []
    legend_handles: list[Any] = []
    for method in GEOMETRY_METHODS:
        calls, values, lower, upper = _geometry_series(rows, method, "first")
        legend_handles.append(
            _draw_line(first_axis, method, calls, values, lower, upper)
        )
        first_upper.extend(upper.tolist())
        calls, values, lower, upper = _geometry_series(rows, method, "frame")
        _draw_line(frame_axis, method, calls, values, lower, upper)
        frame_lower.extend(lower.tolist())
        frame_upper.extend(upper.tolist())

    # Keep all data values unchanged, including the exact orbit zeros.  A tiny
    # negative display margin lifts y=0 off the lower spine; only nonnegative
    # ticks are shown, so no negative metric values are implied.
    first_max = max(first_upper)
    if first_max <= 0.0:
        raise ValueError("all first-moment values are zero; linear range is undefined")
    locator = MaxNLocator(nbins=5, min_n_ticks=4)
    ticks = np.asarray(locator.tick_values(0.0, 1.08 * first_max))
    ticks = ticks[ticks >= 0.0]
    positive_ticks = ticks[ticks > 0.0]
    first_top = float(positive_ticks.max()) if positive_ticks.size else first_max
    first_axis.set_ylim(-0.035 * first_top, first_top * 1.18)
    first_axis.set_yticks(ticks[ticks <= first_top])
    first_axis.yaxis.set_major_formatter(FuncFormatter(_scientific_label))
    _log_x_axis(first_axis)
    first_axis.set_xlim(target_calls[0] * 0.80, target_calls[-1] * 1.22)
    first_axis.set_xlabel("Utility calls", fontsize=21.0, labelpad=8)
    first_axis.set_ylabel(r"$D_1$ (first-moment discrepancy)", fontsize=17.0, labelpad=8)
    _grid(first_axis, logarithmic_y=False)
    _box_axis(first_axis)
    _panel_tag(first_axis, "(a)")

    _log_x_axis(frame_axis)
    _log_y_axis(frame_axis)
    frame_axis.set_xlim(target_calls[0] * 0.80, target_calls[-1] * 1.22)
    frame_lo, frame_hi = _log_limits(frame_lower, frame_upper)
    frame_axis.set_ylim(frame_lo, frame_hi * 1.6)
    frame_axis.set_xlabel("Utility calls", fontsize=21.0, labelpad=8)
    frame_axis.set_ylabel(r"$D_2$ (second-moment discrepancy)", fontsize=17.0, labelpad=8)
    _grid(frame_axis, logarithmic_y=True)
    _box_axis(frame_axis)
    _panel_tag(frame_axis, "(b)")

    relation = report["geometry_error_relation"]
    points = relation["points"]
    all_x: list[float] = []
    all_y: list[float] = []
    for method in GEOMETRY_METHODS:
        selected = [point for point in points if point["method"] == method]
        selected.sort(key=lambda point: (point["budget_index"], point["repeat"]))
        x = np.asarray(
            [point["frame_frobenius_rms"] for point in selected],
            dtype=np.float64,
        )
        y = np.asarray([point["repeat_rmse"] for point in selected], dtype=np.float64)
        style = METHOD_STYLES[method]
        scatter_axis.scatter(
            x,
            y,
            label=METHOD_LABELS[method],
            color=style["color"] if style["filled"] else "white",
            edgecolor=style["color"],
            marker=style["marker"],
            s=82.0 if method != "inside_orbit" else 96.0,
            linewidth=1.55,
            alpha=0.88,
            zorder=style["zorder"],
        )
        all_x.extend(x.tolist())
        all_y.extend(y.tolist())
    trend = _scatter_trend(report)
    trend_x = np.geomspace(*trend["x_range"], 100)
    scatter_axis.plot(trend_x, 10 ** trend["intercept"] * trend_x ** trend["slope"],
                      color="#59616C", linestyle="--", linewidth=1.6, zorder=1)
    scatter_axis.text(0.97, 0.07,
                      f"Spearman ρ = {trend['spearman_rho']:.3f}" + "\nLog–log fitted trend",
                      transform=scatter_axis.transAxes, ha="right", va="bottom",
                      fontsize=13, color="#303640")
    scatter_axis.set_xscale("log")
    scatter_axis.set_yscale("log")
    scatter_axis.set_xlim(*_log_limits(all_x, all_x))
    scatter_axis.set_ylim(*_log_limits(all_y, all_y))
    scatter_axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=14)
    )
    scatter_axis.xaxis.set_minor_locator(
        LogLocator(base=10, subs=(3.0, 4.0, 6.0, 7.0, 8.0, 9.0), numticks=30)
    )
    scatter_axis.xaxis.set_major_formatter(FuncFormatter(_scientific_label))
    _log_y_axis(scatter_axis)
    scatter_axis.set_xlabel(r"$D_2$ (second-moment discrepancy)", fontsize=17.0, labelpad=8)
    scatter_axis.set_ylabel("RMSE", fontsize=21.0, labelpad=8)
    _grid(scatter_axis, logarithmic_y=True)
    _box_axis(scatter_axis)
    _panel_tag(scatter_axis, "(c)")

    first_axis.legend(
        handles=legend_handles,
        labels=[METHOD_LABELS[method] for method in GEOMETRY_METHODS],
        loc="upper right",
        bbox_to_anchor=(0.982, 0.982),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#59616C",
        framealpha=0.96,
        fontsize=16.5,
        borderpad=0.52,
        labelspacing=0.42,
        handlelength=2.20,
        borderaxespad=0.0,
    )
    for axis in axes:
        axis.grid(False, which="both")
    _validate_geometry_figure_contract(figure)
    return figure


def _validate_single_figure_contract(figure: Any, methods: Sequence[str]) -> None:
    if len(figure.axes) != 1 or figure.legends:
        raise RuntimeError("single-panel figure has an invalid axes/legend count")
    axis = figure.axes[0]
    legend = axis.get_legend()
    if legend is None:
        raise RuntimeError("single-panel legend is not inside the axes")
    if axis.get_title() or axis.get_xscale() != "log" or axis.get_yscale() != "log":
        raise RuntimeError("single-panel figure violates the no-title/log contract")
    if not all(spine.get_visible() for spine in axis.spines.values()):
        raise RuntimeError("single-panel figure does not have a full box")
    labels = [text.get_text() for text in legend.get_texts()]
    if labels != [METHOD_LABELS[method] for method in methods]:
        raise RuntimeError("single-panel legend order is incorrect")


def _validate_geometry_figure_contract(figure: Any) -> None:
    if len(figure.axes) != 3 or figure.legends:
        raise RuntimeError("geometry figure has an invalid axes/legend count")
    legends = [axis.get_legend() for axis in figure.axes]
    if legends[0] is None or any(legend is not None for legend in legends[1:]):
        raise RuntimeError("geometry legend is not inside the first panel")
    expected_scales = (("log", "linear"), ("log", "log"), ("log", "log"))
    for axis, scales in zip(figure.axes, expected_scales, strict=True):
        if axis.get_title() or (
            axis.get_xscale(), axis.get_yscale()
        ) != scales:
            raise RuntimeError("geometry panel violates its scale/title contract")
        if not all(spine.get_visible() for spine in axis.spines.values()):
            raise RuntimeError("geometry panel does not have a full box")
    orbit_lines = {
        line.get_label(): np.asarray(line.get_ydata(), dtype=np.float64)
        for line in figure.axes[0].lines
        if line.get_label() in {
            METHOD_LABELS["inside_orbit"], METHOD_LABELS["random_orbit"]
        }
    }
    inside_values = orbit_lines.get(METHOD_LABELS["inside_orbit"])
    if inside_values is None or not np.all(inside_values == 0.0):
        raise RuntimeError("INSIDE-Orbit zero first moment was not preserved")
    labels = [text.get_text() for text in legends[0].get_texts()]
    if labels != [METHOD_LABELS[method] for method in GEOMETRY_METHODS]:
        raise RuntimeError("geometry legend order is incorrect")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_metadata(source_report: Path | None) -> dict[str, Any] | None:
    if source_report is None:
        return None
    if not source_report.is_file():
        raise ValueError(f"source report does not exist: {source_report}")
    return {
        "path": str(source_report.resolve()),
        "sha256": _sha256(source_report),
    }


def _script_metadata() -> dict[str, str]:
    path = Path(__file__).resolve()
    return {"path": str(path), "sha256": _sha256(path)}


def _rmse_values_metadata(
    rows: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for method in methods:
        calls, point, lower, upper = _series(
            rows,
            method,
            value_key="aggregate_rmse",
            interval_key="aggregate_rmse_bootstrap_95",
        )
        values[method] = {
            "label": METHOD_LABELS[method],
            "total_utility_calls": calls.tolist(),
            "aggregate_rmse": point.tolist(),
            "bootstrap_95": np.column_stack((lower, upper)).tolist(),
        }
    return values


def _geometry_values_metadata(
    report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    for method in GEOMETRY_METHODS:
        calls, first, first_lower, first_upper = _geometry_series(
            rows, method, "first"
        )
        _calls, frame, frame_lower, frame_upper = _geometry_series(
            rows, method, "frame"
        )
        curves[method] = {
            "label": METHOD_LABELS[method],
            "total_utility_calls": calls.tolist(),
            "first_moment_rms": first.tolist(),
            "first_moment_mean_plus_minus_std": np.column_stack(
                (first_lower, first_upper)
            ).tolist(),
            "frame_frobenius_rms": frame.tolist(),
            "second_moment_mean_plus_minus_std": np.column_stack(
                (frame_lower, frame_upper)
            ).tolist(),
        }
    return {
        "curves": curves,
        "scatter_points": report["geometry_error_relation"]["points"],
        "correlations": report["geometry_error_relation"]["correlations"],
        "fitted_trend": _scatter_trend(report),
    }


def _write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _validate_saved_artifacts(
    png: Path, pdf: Path, metadata_path: Path
) -> None:
    if png.stat().st_size < 10_000:
        raise RuntimeError(f"PNG artifact is unexpectedly small: {png}")
    if pdf.stat().st_size < 1_000:
        raise RuntimeError(f"PDF artifact is unexpectedly small: {pdf}")
    image = plt.imread(png)
    if image.ndim != 3 or min(image.shape[:2]) < 800 or not np.all(np.isfinite(image)):
        raise RuntimeError(f"PNG artifact failed raster validation: {png}")
    with pdf.open("rb") as stream:
        if stream.read(4) != b"%PDF":
            raise RuntimeError(f"PDF artifact failed signature validation: {pdf}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "complete":
        raise RuntimeError("artifact metadata is not complete")
    outputs = metadata.get("outputs", {})
    for key, artifact in (("png", png), ("pdf", pdf)):
        recorded = outputs.get(key, {})
        if recorded.get("sha256") != _sha256(artifact):
            raise RuntimeError(f"{key.upper()} hash validation failed")


def _save_figure_set(
    figure: Any,
    output: Path,
    *,
    artifact: str,
    report: Mapping[str, Any],
    source_report: Path | None,
    methods: Sequence[str],
    plotted_values: Mapping[str, Any],
    layout: str,
    scales: Mapping[str, Any],
    interpretation: Mapping[str, Any] | None = None,
) -> tuple[Path, Path, Path]:
    if output.suffix.lower() != ".png":
        raise ValueError("figure output must be a PNG path")
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    metadata_path = output.with_suffix(".metadata.json")
    figure.savefig(output, dpi=600, facecolor="white")
    figure.savefig(output.with_suffix(".svg"), facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    metadata: dict[str, Any] = {
        "status": "complete",
        "artifact": artifact,
        "experiment": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "dataset": "airport",
        "methods": list(methods),
        "method_labels": {method: METHOD_LABELS[method] for method in methods},
        "layout": layout,
        "chart_title": None,
        "legend_location": "inside_axes",
        "full_axis_box": True,
        "scales": dict(scales),
        "source_report": _source_metadata(source_report),
        "plotting_script": _script_metadata(),
        "plotted_values": plotted_values,
        "outputs": {
            "png": {"path": str(output.resolve()), "sha256": _sha256(output)},
            "pdf": {"path": str(pdf.resolve()), "sha256": _sha256(pdf)},
        },
        "validation": {
            "source_schema": "passed",
            "plotted_aggregate_recomputation": "passed",
            "figure_contract": "passed",
            "output_hashes": "passed",
        },
    }
    if interpretation is not None:
        metadata["interpretation"] = dict(interpretation)
    _write_metadata(metadata_path, metadata)
    _validate_saved_artifacts(output, pdf, metadata_path)
    return output, pdf, metadata_path


def plot_all(
    report: Mapping[str, Any],
    output_prefix: Path,
    *,
    source_report: Path | None = None,
    audit_alignment: bool = False,
) -> dict[str, tuple[Path, Path, Path]]:
    """Render all three PNG/PDF/metadata sets."""
    validate_plot_report(report)
    rows = _ordered_budget_rows(report)
    base = output_prefix.parent / output_prefix.name

    component_output = base.with_name(base.name + "_component_rmse.png")
    component = _save_figure_set(
        build_component_figure(report),
        component_output,
        artifact=COMPONENT_ARTIFACT,
        report=report,
        source_report=source_report,
        methods=COMPONENT_METHODS,
        plotted_values=_rmse_values_metadata(rows, COMPONENT_METHODS),
        layout="1x1_near_square",
        scales={"x": "log", "y": "log"},
        interpretation={
            "comparison": (
                "OFA (IID), first-moment-only Greedy, frame-only Greedy, "
                "and the complete INSIDE objective"
            ),
            "uncertainty": "repeat bootstrap 95% intervals",
        },
    )

    orbit_output = base.with_name(base.name + "_orbit_rmse.png")
    orbit = _save_figure_set(
        build_orbit_figure(report),
        orbit_output,
        artifact=ORBIT_ARTIFACT,
        report=report,
        source_report=source_report,
        methods=ORBIT_METHODS,
        plotted_values=_rmse_values_metadata(rows, ORBIT_METHODS),
        layout="1x1_near_square",
        scales={"x": "log", "y": "log"},
        interpretation={
            "paired_control": (
                "Both methods share every K=4 candidate pool, size "
                "allocation, relabeling, complete-orbit balance, and OFA "
                "ratio estimator; only base-coalition selection differs."
            ),
            "uncertainty": "repeat bootstrap 95% intervals",
        },
    )

    geometry_output = base.with_name(base.name + "_geometry.png")
    geometry_figure = build_geometry_figure(report)
    if audit_alignment:
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(
            geometry_figure,
            json_out=geometry_output.with_suffix(".alignment.json"),
            overlay_svg=geometry_output.with_suffix(".alignment.svg"),
            tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    geometry = _save_figure_set(
        geometry_figure,
        geometry_output,
        artifact=GEOMETRY_ARTIFACT,
        report=report,
        source_report=source_report,
        methods=GEOMETRY_METHODS,
        plotted_values=_geometry_values_metadata(report, rows),
        layout="1x3_square_panels",
        scales={
            "first_moment": {"x": "log", "y": "linear"},
            "second_moment_discrepancy": {"x": "log", "y": "log"},
            "scatter": {"x": "log", "y": "log"},
        },
        interpretation={
            "geometry_aggregation": "equal-size macro RMS over s=2,...,98",
            "D1_definition": "sqrt(mean_s ||mean(u|s)||_2^2)",
            "D2_definition": "sqrt(mean_s ||mean(uu^T|s)-P/(n-1)||_F^2)",
            "uncertainty": "mean +/- sample std across 3 seeds, ddof=1",
            "theory_scope": "Empirical support for the second-moment component, not verification of the full theorem",

            "first_moment_zero_policy": (
                "Exact zeros are plotted at y=0 on a linear axis; no "
                "epsilon, clipping, or value substitution is used."
            ),
            "scatter_granularity": "method x budget x repeat",
            "scatter_caveat": (
                "The association is descriptive; utility budget drives both "
                "geometry and RMSE and does not establish universal causality."
            ),
        },
    )
    return {"component": component, "orbit": orbit, "geometry": geometry}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--audit-alignment", action="store_true",
                        help="Run nature-figure alignment checks (add its scripts directory to PYTHONPATH)")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        help=(
            "base path for outputs; defaults to the input path without its "
            ".json suffix"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    output_prefix = args.output_prefix or args.report.with_suffix("")
    outputs = plot_all(
        report, output_prefix, source_report=args.report, audit_alignment=args.audit_alignment
    )
    for name, paths in outputs.items():
        print(f"{name}: " + ", ".join(str(path) for path in paths), flush=True)


if __name__ == "__main__":
    main()

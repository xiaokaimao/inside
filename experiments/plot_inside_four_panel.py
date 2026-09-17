"""Draw publication-facing four-panel INSIDE accuracy comparisons.

The source reports retain the complete experiment, including omitted methods. This figure
uses a deliberately narrower presentation contract:

* OFA linear is hidden in every panel;
* augmented reports add ShapDoE-LS and Orthogonal; ShapDoE-COA is omitted;
* S-Diff is hidden only for Cancer because its quadratic pairwise state has
  poor memory scalability (this is not a claim that the recorded run OOMed);
* RMSE, AER, and MER use the same log--log layout and method presentation;
* all four panels use a full box; augmented comparisons put the shared legend
  above the panels for RMSE; AER and MER omit the legend;
* there is no figure or axes title.  Small tags above each panel identify the benchmark.

TMC-Shapley remains plotted at observed calls. Its off-screen polyline is
clipped to the common target-budget viewport, with no synthetic markers at
the boundary and no change to source estimates.

The companion metadata JSON makes these presentation-only omissions explicit
and preserves source-report hashes for provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import numpy as np

from experiments.result_paths import by_format

from experiments.add_orthogonal_baseline import (
    EXPERIMENT_ID as AUGMENTED_EXPERIMENT_ID,
    validate_report as validate_augmented_report,
)
from experiments.plot_orthogonal_comparison import ORTHOGONAL_STYLES

from experiments.plot_inside_comparison import (
    METHOD_ORDER,
    METHOD_STYLES,
    _actual_calls,
    _calls_label,
    _dataset_name,
    _ordered_rows,
    validate_plot_report,
)


PANEL_ORDER = ("voting", "airport", "wine", "cancer")
METRIC_ORDER = ("rmse", "aer", "mer")
METRIC_LABELS = {"rmse": "RMSE", "aer": "AER", "mer": "MER"}
METRIC_METADATA = {
    "rmse": {
        "definition": "sqrt(mean_player((estimate-reference)^2))",
        "repeat_aggregation": "arithmetic_mean_of_repeat_level_RMSE",
    },
    "aer": {
        "definition": (
            "mean_player(abs((estimate-reference)/reference))"
        ),
        "repeat_aggregation": "arithmetic_mean_of_repeat_level_AER",
    },
    "mer": {
        "definition": (
            "max_player(abs((estimate-reference)/reference))"
        ),
        "repeat_aggregation": "arithmetic_mean_of_repeat_level_MER",
    },
}
PANEL_TAGS = {
    "voting": "(a) U.S. Electoral Voting",
    "airport": "(b) Airport",
    "wine": "(c) Wine",
    "cancer": "(d) Breast Cancer",
}
EXPECTED_DATASET_NAMES = {
    "wine": "Wine",
    "airport": "Airport",
    "voting": "U.S. Electoral Voting",
    "cancer": "Breast Cancer",
}

# The legend is the union of the methods visible in at least one panel.
DISPLAY_METHOD_ORDER = tuple(
    method for method in METHOD_ORDER if method != "ofa_iid_linear"
)
CANCER_DISPLAY_METHOD_ORDER = tuple(
    method for method in DISPLAY_METHOD_ORDER if method != "s_diff"
)
DISPLAY_LABELS = {
    method: ("OFA" if method == "ofa_iid_ratio" else METHOD_STYLES[method]["label"])
    for method in DISPLAY_METHOD_ORDER
}
DISPLAY_LABELS["inside_greedy"] = "INSIDE-Coalition"
ADDED_METHOD_ORDER = ("shapdoe_ls", "orthogonal")
PLOT_METHOD_STYLES = METHOD_STYLES | {
    method: ORTHOGONAL_STYLES[method] for method in ADDED_METHOD_ORDER
}
DISPLAY_LABELS.update({method: PLOT_METHOD_STYLES[method]["label"] for method in ADDED_METHOD_ORDER})

OMISSION_METADATA: dict[str, dict[str, Any]] = {
    "ofa_iid_linear": {
        "scope": "all_panels",
        "reason_code": "presentation_baseline_removed",
        "reason": (
            "OFA linear is omitted from the four-panel presentation at the "
            "user's request; its completed measurements remain unchanged in "
            "the source reports."
        ),
    },
    "s_diff": {
        "scope": ["cancer"],
        "reason_code": "quadratic_state_memory_scalability",
        "reason": (
            "S-Diff is omitted from the Cancer panel because its O(n^2) "
            "pairwise state has poor memory scalability at larger n. The "
            "source report retains the completed Cancer measurements."
        ),
        "actual_oom_observed": False,
    },
}


def _plot_view(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """Expose source game/reference fields without editing the archived report."""
    if report.get("experiment") != AUGMENTED_EXPERIMENT_ID:
        return report
    return report["source_report"]["base_report"] | {
        "results_by_inner_budget": report["results_by_inner_budget"],
        "_added_plot_methods": ADDED_METHOD_ORDER,
    }


def _display_methods(panel: str, report: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    methods = CANCER_DISPLAY_METHOD_ORDER if panel == "cancer" else DISPLAY_METHOD_ORDER
    return methods + tuple(report.get("_added_plot_methods", ())) if report is not None else methods


def _legend_methods(reports: Mapping[str, Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(method for panel in PANEL_ORDER
                               for method in _display_methods(panel, _plot_view(reports[panel]))))


def _observed_calls(summary: Mapping[str, Any], target: float, method: str) -> float:
    if method in ADDED_METHOD_ORDER:
        return float(summary["mean_actual_utility_calls"])
    return _actual_calls(summary, target, method=method)


def validate_four_panel_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> None:
    """Validate report identity plus the complete source-report plot schema."""
    if set(reports) != set(PANEL_ORDER):
        missing = sorted(set(PANEL_ORDER) - set(reports))
        extra = sorted(set(reports) - set(PANEL_ORDER))
        raise ValueError(
            f"four-panel reports have missing={missing} and extra={extra}"
        )
    for panel in PANEL_ORDER:
        report = reports[panel]
        if report.get("experiment") == AUGMENTED_EXPERIMENT_ID:
            validate_augmented_report(report, require_complete=False)
            report = _plot_view(report)
        else:
            validate_plot_report(report)
        observed = _dataset_name(report)
        expected = EXPECTED_DATASET_NAMES[panel]
        if observed != expected:
            raise ValueError(
                f"{panel} panel received dataset {observed!r}, expected "
                f"{expected!r}"
            )


def _reference_values(report: Mapping[str, Any]) -> np.ndarray:
    ground_truth = report.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("ground_truth must be an object")
    truth = np.asarray(ground_truth.get("values"), dtype=np.float64)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("ground_truth.values must be a non-empty vector")
    if not np.all(np.isfinite(truth)):
        raise ValueError("ground_truth.values must be finite")
    return truth


def _relative_error_ratios(
    estimates: Any, truth: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-repeat AER and MER, with no denominator regularization."""
    reference = np.asarray(truth, dtype=np.float64)
    values = np.asarray(estimates, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("truth must be a non-empty vector")
    if not np.all(np.isfinite(reference)):
        raise ValueError("truth must be finite")
    zero_indices = np.flatnonzero(reference == 0.0)
    if zero_indices.size:
        preview = zero_indices[:8].tolist()
        raise ValueError(
            "AER/MER are undefined because reference Shapley values are "
            f"exactly zero at player indices {preview}"
        )
    if values.ndim != 2 or values.shape[1] != reference.size:
        raise ValueError(
            "estimates must have shape (repeats, number_of_players)"
        )
    if values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("estimates must contain finite repeat vectors")
    ratios = np.abs((values - reference[None, :]) / reference[None, :])
    if not np.all(np.isfinite(ratios)):
        raise ValueError(
            "AER/MER overflowed for a non-zero reference Shapley value"
        )
    return ratios.mean(axis=1), ratios.max(axis=1)


def _validate_relative_metric_report(report: Mapping[str, Any]) -> None:
    truth = _reference_values(report)
    if np.any(truth == 0.0):
        _relative_error_ratios(np.empty((1, truth.size)), truth)
    repeats = int(report["configuration"]["repeats"])
    for row_index, row in enumerate(_ordered_rows(report)):
        for method in (*METHOD_ORDER, *report.get("_added_plot_methods", ())):
            summary = row["methods"][method]
            if summary.get("status") in {"pending", "budget_infeasible"}:
                continue
            estimates = np.asarray(summary.get("estimates"), dtype=np.float64)
            if estimates.shape != (repeats, truth.size):
                raise ValueError(
                    f"row[{row_index}].{method}.estimates has shape "
                    f"{estimates.shape}, expected {(repeats, truth.size)}"
                )
            _relative_error_ratios(estimates, truth)


def _validate_metric(metric: str) -> str:
    normalized = str(metric).lower()
    if normalized not in METRIC_ORDER:
        raise ValueError(
            f"metric must be one of {METRIC_ORDER}, received {metric!r}"
        )
    return normalized


def _metric_series(
    report: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    method: str,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return calls and mean ± sample standard deviation across estimator repeats."""
    metric = _validate_metric(metric)
    rows = [row for row in rows if row["methods"][method].get("status")
            not in {"pending", "budget_infeasible"}]
    truth = _reference_values(report)
    calls: list[float] = []
    means: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    for row in rows:
        target = float(row["total_utility_calls_per_estimate"])
        summary = row["methods"][method]
        estimates = np.asarray(summary["estimates"], dtype=np.float64)
        if estimates.shape != (int(report["configuration"]["repeats"]), truth.size):
            raise ValueError("estimates must match configured repeats and reference players")
        if estimates.shape[0] < 2 or not np.all(np.isfinite(estimates)):
            raise ValueError("sample standard deviation requires at least two finite repeats")
        if metric == "rmse":
            per_repeat = np.sqrt(np.mean((estimates - truth[None, :]) ** 2, axis=1))
        else:
            aer, mer = _relative_error_ratios(estimates, truth)
            per_repeat = aer if metric == "aer" else mer
        std = float(np.std(per_repeat, ddof=1))
        calls.append(_observed_calls(summary, target, method))
        means.append(float(np.mean(per_repeat)))
        lower.append(means[-1] - std)
        upper.append(means[-1] + std)
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (calls, means, lower, upper)
    )


def _clip_series_to_viewport(calls: np.ndarray, values: np.ndarray,
                             limits: tuple[float, float]) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Clip the log-log polyline itself, retaining markers only on measured points.

    Axis clipping alone leaves off-screen PDF paths crossing adjacent text in
    geometry readers. Boundary intersections interpolate log coordinates to
    match exactly the original straight segments on logarithmic axes.
    """
    if not len(calls):
        return calls, values, []
    if not np.all(np.diff(calls) > 0):
        raise ValueError("utility calls must increase for viewport clipping")
    visible = (calls >= limits[0]) & (calls <= limits[1])
    x, y = calls[visible].tolist(), values[visible].tolist()
    markers = list(range(len(x)))
    for boundary, prepend in ((limits[0], True), (limits[1], False)):
        if calls[0] < boundary < calls[-1] and boundary not in x:
            value = float(np.exp(np.interp(np.log(boundary), np.log(calls), np.log(values))))
            if prepend:
                x.insert(0, boundary)
                y.insert(0, value)
                markers = [index + 1 for index in markers]
            else:
                x.append(boundary)
                y.append(value)
    return np.asarray(x), np.asarray(y), markers


def _panel_limits(
    report: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    methods: tuple[str, ...],
    metric: str,
) -> tuple[np.ndarray, tuple[float, float]]:
    target_calls = np.asarray(
        [float(row["total_utility_calls_per_estimate"]) for row in rows],
        dtype=np.float64,
    )
    lower: list[float] = []
    upper: list[float] = []
    for method in methods:
        _calls, values, method_lower, method_upper = _metric_series(
            report, rows, method, metric
        )
        if np.any(values <= 0):
            raise ValueError(f"{metric.upper()} means must be positive for a log axis")
        # A nonpositive lower SD bound cannot be shown on a log axis. Keep the
        # raw statistics in metadata and clip only the shading to the viewport.
        if metric == "rmse":
            lower.extend(method_lower[method_lower > 0].tolist())
            upper.extend(method_upper.tolist())
        else:
            upper.extend(values.tolist())
        lower.extend(values.tolist())
    log_min = math.log10(min(lower))
    log_max = math.log10(max(upper))
    log_span = max(log_max - log_min, 0.25)
    return target_calls, (
        10.0 ** (log_min - 0.08 * log_span),
        10.0 ** (log_max + (0.08) * log_span),
    )


def _draw_panel(
    axis: Any, report: Mapping[str, Any], panel: str, metric: str
) -> None:
    rows = _ordered_rows(report)
    methods = _display_methods(panel, report)
    target_calls, y_limits = _panel_limits(report, rows, methods, metric)
    x_limits = (target_calls[0] * .80, target_calls[-1] * 1.22)

    # Draw secondary baselines first and the two INSIDE curves last.
    draw_order = tuple(
        method for method in methods if not method.startswith("inside_")
    ) + tuple(method for method in methods if method.startswith("inside_"))
    for method in draw_order:
        calls, values, lower, upper = _metric_series(
            report, rows, method, metric
        )
        style = PLOT_METHOD_STYLES[method]
        focal = method.startswith("inside_")
        if metric == "rmse":
            band_calls, band_lower, _ = _clip_series_to_viewport(
                calls, np.maximum(lower, y_limits[0]), x_limits)
            _, band_upper, _ = _clip_series_to_viewport(calls, upper, x_limits)
            axis.fill_between(band_calls, band_lower, band_upper,
                              color=style["color"], alpha=0.16, linewidth=0, zorder=2)
            if len(band_calls) == 1:
                axis.vlines(band_calls, band_lower, band_upper,
                            color=style["color"], linewidth=1.5, alpha=0.7, zorder=3)
        calls, values, markers = _clip_series_to_viewport(calls, values, x_limits)
        axis.plot(
            calls,
            values,
            label=DISPLAY_LABELS[method],
            color=style["color"],
            marker=style["marker"],
            markevery=markers,
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style.get("markersize", 7.5),
            markerfacecolor=style["color"] if focal else "white",
            markeredgecolor=style["color"],
            markeredgewidth=1.25,
            alpha=1.0 if focal else 0.88,
            zorder=style.get("zorder", 3),
            clip_on=True,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
    axis.xaxis.set_minor_locator(NullLocator())
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
    axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:.0e}")
    )
    axis.grid(
        True, which="major", color="#D8DDE4", linewidth=1.0, alpha=0.92
    )
    axis.grid(
        True,
        which="minor",
        axis="y",
        color="#EEF1F4",
        linewidth=0.7,
        alpha=0.82,
    )
    axis.tick_params(axis="both", which="major", labelsize=11.5, width=1.05)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#555E69")
        spine.set_linewidth(1.25)
    axis.set_box_aspect(1.0)

    # Put panel identification above the axes so grids span the whole plot.
    axis.text(
        0.035,
        1.025,
        PANEL_TAGS[panel],
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.5,
        fontweight="semibold",
        color="#20252D",
        zorder=10,
    )


def _legend_handles(methods: tuple[str, ...] = DISPLAY_METHOD_ORDER) -> list[Line2D]:
    handles: list[Line2D] = []
    for method in methods:
        style = PLOT_METHOD_STYLES[method]
        focal = method.startswith("inside_")
        handles.append(
            Line2D(
                [],
                [],
                label=DISPLAY_LABELS[method],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                markersize=style.get("markersize", 7.5),
                markerfacecolor=(
                    style["color"] if focal else "white"
                ),
                markeredgecolor=style["color"],
                markeredgewidth=1.25,
            )
        )
    return handles


def build_four_panel_figure(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    metric: str = "rmse",
) -> Any:
    """Build, but do not save or close, the validated four-panel figure."""
    metric = _validate_metric(metric)
    validate_four_panel_reports(reports)
    reports = {panel: _plot_view(report) for panel, report in reports.items()}
    if metric != "rmse":
        for report in reports.values():
            _validate_relative_metric_report(report)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 13.0,
            "axes.labelcolor": "#303640",
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=(21.8, 6.2))
    figure.subplots_adjust(
        left=0.050,
        right=0.995,
        bottom=0.140,
        top=0.885 if metric == "rmse" else 0.975,
        wspace=0.220,
    )
    for axis, panel in zip(axes.flat, PANEL_ORDER):
        _draw_panel(axis, reports[panel], panel, metric)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    tick_bottom = min(axis.xaxis.get_tightbbox(renderer).y0 for axis in axes.flat)
    xlabel_y = tick_bottom / figure.bbox.height - 8.0 / (figure.get_figheight() * 72.0)
    figure.supxlabel(
        "Utility calls", x=0.525, y=xlabel_y, va="top",
        fontsize=15.0, color="#303640"
    )
    figure.supylabel(
        METRIC_LABELS[metric],
        x=0.012,
        y=0.500,
        fontsize=15.0,
        color="#303640",
    )
    handles = _legend_handles(_legend_methods(reports))
    if metric == "rmse":
        figure.legend(handles=handles, loc="center", bbox_to_anchor=(.525, .95),
                      ncol=len(handles), frameon=False, fontsize=11.2,
                      columnspacing=1.1, handlelength=2.2)
    return figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reference_diagnostics(report: Mapping[str, Any]) -> dict[str, Any]:
    truth = _reference_values(report)
    absolute = np.abs(truth)
    minimum_index = int(np.argmin(absolute))
    diagnostics: dict[str, Any] = {
        "number_of_players": int(truth.size),
        "exact_zero_count": int(np.count_nonzero(truth == 0.0)),
        "minimum_absolute_reference": float(absolute[minimum_index]),
        "minimum_absolute_reference_player_index_zero_based": minimum_index,
        "minimum_absolute_reference_signed_value": float(truth[minimum_index]),
        "absolute_reference_quantiles": {
            "q00": float(np.quantile(absolute, 0.00)),
            "q01": float(np.quantile(absolute, 0.01)),
            "q05": float(np.quantile(absolute, 0.05)),
            "q50": float(np.quantile(absolute, 0.50)),
            "q95": float(np.quantile(absolute, 0.95)),
            "q99": float(np.quantile(absolute, 0.99)),
            "q100": float(np.quantile(absolute, 1.00)),
        },
    }
    standard_errors = report["ground_truth"].get("standard_errors")
    if standard_errors is not None:
        errors = np.asarray(standard_errors, dtype=np.float64)
        if errors.shape == truth.shape and np.all(np.isfinite(errors)):
            diagnostics["reference_standard_error_at_minimum"] = float(
                errors[minimum_index]
            )
            diagnostics["standard_error_to_abs_reference_at_minimum"] = float(
                errors[minimum_index] / absolute[minimum_index]
            )
    return diagnostics


def _panel_metric_metadata(
    report: Mapping[str, Any], panel: str, metric: str
) -> dict[str, Any]:
    report = _plot_view(report)
    rows = _ordered_rows(report)
    methods = _display_methods(panel, report)
    x_limits = (rows[0]["total_utility_calls_per_estimate"] * .80,
                rows[-1]["total_utility_calls_per_estimate"] * 1.22)
    panel_metadata: dict[str, Any] = {
        "key": panel,
        "tag": PANEL_TAGS[panel],
        "methods_shown": list(methods),
        "x_limits": list(x_limits),
        "source_budget_points": len(rows),
        "point_counts": {},
    }
    for method in methods:
        calls, values, _, _ = _metric_series(report, rows, method, metric)
        panel_metadata["point_counts"][method] = {
            "completed": len(calls), "pending": len(rows) - len(calls),
            "left_of_viewport": int(np.sum(calls < x_limits[0])),
            "inside_viewport": int(np.sum((calls >= x_limits[0]) & (calls <= x_limits[1]))),
            "actual_calls": calls.tolist(), "plotted_values": values.tolist(),
        }
    if metric != "rmse":
        panel_metadata["reference_diagnostics"] = _reference_diagnostics(report)
    plotted: dict[str, Any] = {}
    for method in methods:
        calls, means, lower, upper = _metric_series(
            report, rows, method, metric
        )
        plotted[method] = {
            "utility_calls": calls.tolist(),
            "mean_over_repeats": means.tolist(),
            "sample_std_over_repeats": ((upper - lower) / 2).tolist(),
            "mean_minus_std": lower.tolist(),
            "mean_plus_std": upper.tolist(),
            "nonpositive_lower_bound_count": int(np.sum(lower <= 0)),
            "repeats": int(report["configuration"]["repeats"]),
        }
    panel_metadata["plotted_values"] = plotted
    if panel == "cancer" and metric != "rmse":
        panel_metadata["interpretation_caveat"] = (
            "Relative errors are highly sensitive to near-zero Monte Carlo "
            "reference attributions; use RMSE as the primary metric."
        )
    return panel_metadata


def _metadata(
    output: Path,
    pdf: Path,
    svg: Path,
    *,
    reports: Mapping[str, Mapping[str, Any]],
    metric: str,
    source_reports: Mapping[str, Path] | None,
    alignment_json: Path | None,
    alignment_overlay: Path | None,
    alignment_report: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metric = _validate_metric(metric)
    legend_methods = _legend_methods(reports)
    sources: dict[str, Any] = {}
    if source_reports is not None:
        sources = {
            panel: {
                "path": str(source_reports[panel].resolve()),
                "sha256": _sha256(source_reports[panel]),
            }
            for panel in PANEL_ORDER
        }
    alignment_metadata: dict[str, Any] = {"status": "not_run"}
    if alignment_report is not None:
        if alignment_json is None or alignment_overlay is None:
            raise ValueError("alignment artifact paths are incomplete")
        alignment_metadata = {
            "status": str(alignment_report.get("verdict")),
            "tolerance_pt": 1.5,
            "gutter_tolerance_pt": 1.5,
            "panel_label_anchor_check": (
                "not_requested because panel identifiers and dataset labels "
                "are combined in one in-axes tag"
            ),
            "json": {
                "path": str(alignment_json.resolve()),
                "sha256": _sha256(alignment_json),
            },
            "overlay_svg": {
                "path": str(alignment_overlay.resolve()),
                "sha256": _sha256(alignment_overlay),
            },
        }
    return {
        "status": "complete",
        "artifact": f"inside_{metric}_four_panel",
        "metric": {
            "key": metric,
            "display_label": METRIC_LABELS[metric],
            **METRIC_METADATA[metric],
            **(
                {
                    "ratio_scale": "proportion_not_percentage",
                    "denominator_policy": "strict_nonzero_no_epsilon",
                    "reference_source": "ground_truth.values",
                }
                if metric in {"aer", "mer"}
                else {}
            ),
        },
        "layout": "1x4",
        "chart_title": None,
        "axes_titles": False,
        "panel_identification": "above_axes_tags",
        "footnote_display": False,
        "method_display_names": {method: DISPLAY_LABELS[method] for method in legend_methods},
        "scales": {"x": "log", "y": "log"},
        "shared_legend": {
            "location": "above_panels" if metric == "rmse" else "none",
            "host_panel": None,
            "rendered_once": metric == "rmse",
            "method_order": list(legend_methods),
            "labels": [
                DISPLAY_LABELS[method]
                for method in legend_methods
            ],
        },
        "panel_alignment": alignment_metadata,
        "panels": [
            _panel_metric_metadata(reports[panel], panel, metric)
            for panel in PANEL_ORDER
        ],
        "presentation_omissions": OMISSION_METADATA | ({
            "shapdoe_coa": {"scope": "all_panels", "reason_code": "user_requested_removal",
                            "reason": "ShapDoE-COA is omitted at the user's request; source measurements are retained."}
        } if any(method in legend_methods for method in ADDED_METHOD_ORDER) else {}),
        "tmc_clipping": "Keep all observed points and clip the drawn line at the target-budget viewport; do not reposition or delete TMC-Shapley.",
        "uncertainty_display": {
            "center": "arithmetic mean of per-repeat metrics",
            "band": "mean plus/minus one sample standard deviation" if metric == "rmse" else "none (user requested)",
            "ddof": 1,
            "unit": "estimator seed, fixed data split and reference",
            "single_budget": "vertical standard deviation bar" if metric == "rmse" else "mean marker only",
            "log_axis_policy": "nonpositive lower bounds clipped to the y viewport for display only",
        },
        "source_reports": sources,
        "outputs": {
            kind: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for kind, path in (
                (("pdf", pdf),) if output.suffix.lower() == ".pdf"
                else (("png", output), ("pdf", pdf), ("svg", svg))
            )
        },
    }


def plot_four_panel_results(
    reports: Mapping[str, Mapping[str, Any]],
    output: Path,
    *,
    source_reports: Mapping[str, Path] | None = None,
    metric: str = "rmse",
    alignment_json: Path | None = None,
    alignment_overlay: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Save PDF and metadata; a legacy PNG request also exports PNG/SVG."""
    metric = _validate_metric(metric)
    if output.suffix.lower() not in {".pdf", ".png"}:
        raise ValueError("output must be a PDF or PNG path")
    if source_reports is not None and set(source_reports) != set(PANEL_ORDER):
        raise ValueError("source report paths must cover all four panels")
    if (alignment_json is None) != (alignment_overlay is None):
        raise ValueError(
            "alignment_json and alignment_overlay must be provided together"
        )
    figure = build_four_panel_figure(reports, metric=metric)
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf = output.with_suffix(".pdf")
    svg = output.with_suffix(".svg")
    metadata_path = by_format(output.with_suffix(".metadata.json"))
    alignment_report: Mapping[str, Any] | None = None
    if alignment_json is not None and alignment_overlay is not None:
        try:
            from audit_panel_alignment import (
                require_matplotlib_panel_alignment,
            )
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "alignment QA was requested but audit_panel_alignment is "
                "not importable; add its scripts directory to PYTHONPATH"
            ) from error
        alignment_json.parent.mkdir(parents=True, exist_ok=True)
        alignment_overlay.parent.mkdir(parents=True, exist_ok=True)
        alignment_report = require_matplotlib_panel_alignment(
            figure,
            json_out=alignment_json,
            overlay_svg=alignment_overlay,
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
            panel_ids=list(PANEL_ORDER),
        )
    if output.suffix.lower() == ".png":
        figure.savefig(output, dpi=600, facecolor="white")
        figure.savefig(svg, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    plt.close(figure)
    metadata = _metadata(
        output,
        pdf,
        svg,
        reports=reports,
        metric=metric,
        source_reports=source_reports,
        alignment_json=alignment_json,
        alignment_overlay=alignment_overlay,
        alignment_report=alignment_report,
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output, pdf, metadata_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wine", required=True, type=Path)
    parser.add_argument("--airport", required=True, type=Path)
    parser.add_argument("--voting", required=True, type=Path)
    parser.add_argument("--cancer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metric", choices=METRIC_ORDER, default="rmse")
    parser.add_argument("--alignment-json", type=Path)
    parser.add_argument("--alignment-overlay", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = {panel: getattr(args, panel) for panel in PANEL_ORDER}
    reports = {
        panel: json.loads(path.read_text(encoding="utf-8"))
        for panel, path in paths.items()
    }
    png, pdf, metadata = plot_four_panel_results(
        reports,
        args.output,
        source_reports=paths,
        metric=args.metric,
        alignment_json=args.alignment_json,
        alignment_overlay=args.alignment_overlay,
    )
    print(f"saved {png}, {pdf}, and {metadata}")


if __name__ == "__main__":
    main()

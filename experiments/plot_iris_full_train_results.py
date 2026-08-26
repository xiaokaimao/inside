"""Plot a full-training-set data-Shapley comparison.

The historical module name is retained for compatibility.  Dataset-specific
labels and call accounting are read from the experiment report, so the same
plotter can also render Wine and future full-training-set experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np


METHODS = {
    "official_cc_basic": {
        "label": "CC (official basic)",
        "color": "#4B5563",
        "marker": "X",
        "linestyle": ":",
    },
    "official_ofa_fixed_ratio": {
        "label": "OFA ratio (IID)",
        "color": "#2F6B9A",
        "marker": "o",
        "linestyle": "-",
    },
    "frame_orbit_ratio": {
        "label": "Frame-OFA orbit ratio",
        "color": "#D97706",
        "marker": "D",
        "linestyle": "--",
    },
    "iid_linear_ofa": {
        "label": "OFA linear (IID)",
        "color": "#2F6B9A",
        "marker": "s",
        "linestyle": "-",
    },
    "frame_coupled_linear": {
        "label": "Frame-OFA coupled linear",
        "color": "#D97706",
        "marker": "^",
        "linestyle": "--",
    },
}


def _dataset_display_name(report: dict) -> str:
    raw_name = report.get("dataset", {}).get("dataset", "dataset")
    return str(raw_name).replace("_", " ").replace("-", " ").title()


def _model_display_name(configuration: dict) -> str:
    explicit = configuration.get("model_label")
    if explicit:
        return str(explicit)
    model = str(configuration.get("model", "classifier"))
    lowered = model.lower()
    if "svc" in lowered and ("rbf" in lowered or "kernel='rbf'" in lowered):
        return "RBF-SVM"
    if "linearsvc" in lowered or "linear_svm" in lowered:
        return "linear SVM"
    if "logistic" in lowered or lowered == "lr":
        return "logistic regression"
    return model


def _plot_context(report: dict) -> dict[str, object]:
    configuration = report["configuration"]
    num_players = int(configuration["num_players"])
    n_test = configuration.get("n_test")
    if n_test is None:
        n_test = len(report.get("dataset", {}).get("test_labels", []))
    boundary_calls = int(
        report.get("boundary", {}).get(
            "utility_calls", 2 * num_players + 2
        )
    )
    return {
        "dataset": _dataset_display_name(report),
        "num_players": num_players,
        "n_test": int(n_test),
        "model": _model_display_name(configuration),
        "boundary_calls": boundary_calls,
    }


def _default_output(input_path: Path) -> Path:
    stem = input_path.stem
    if "_frame_ofa" in stem:
        stem = stem.replace("_frame_ofa", "_rmse", 1)
    else:
        stem = f"{stem}_rmse"
    return input_path.with_name(f"{stem}.png")


def _calls_label(value: float, _: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    return f"{value / 1000:.1f}k"


def _relative_change_label(reduction: float) -> str:
    """Describe a signed RMSE reduction without a misleading minus sign."""
    percent = 100.0 * float(reduction)
    direction = "lower" if percent >= 0.0 else "higher"
    return f"{abs(percent):.1f}% {direction}"


def _series(
    report: dict, method: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = sorted(
        report["results_by_inner_budget"].values(),
        key=lambda row: row["total_utility_calls_per_estimate"],
    )
    calls = np.asarray(
        [row["total_utility_calls_per_estimate"] for row in rows],
        dtype=np.float64,
    )
    rmse = np.asarray(
        [row["methods"][method]["aggregate_rmse"] for row in rows]
    )
    intervals = np.asarray(
        [
            row["methods"][method]["aggregate_rmse_bootstrap_95"]
            for row in rows
        ]
    )
    return calls, rmse, intervals[:, 0], intervals[:, 1]


def _has_method(report: dict, method: str) -> bool:
    rows = report.get("results_by_inner_budget", {}).values()
    return bool(rows) and all(
        method in row.get("methods", {}) for row in rows
    )


def plot_results(report: dict, output: Path) -> None:
    if report.get("status") != "complete":
        raise ValueError("the experiment report is not complete")
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.4, 5.4),
        constrained_layout=False,
    )
    coupled_label = (
        "Frame-OFA orbit-coupled linear"
        if report["configuration"].get("coupled_design")
        == "orbit_coupled"
        else METHODS["frame_coupled_linear"]["label"]
    )
    ratio_methods = ["official_ofa_fixed_ratio", "frame_orbit_ratio"]
    if _has_method(report, "official_cc_basic"):
        ratio_methods.insert(0, "official_cc_basic")
    panels = [
        (
            axes[0],
            "Stratified-mean / ratio estimators",
            ratio_methods,
        ),
        (
            axes[1],
            "Linear estimators",
            ["iid_linear_ofa", "frame_coupled_linear"],
        ),
    ]
    tick_calls: np.ndarray | None = None
    for axis, title, methods in panels:
        for method in methods:
            calls, rmse, lower, upper = _series(report, method)
            tick_calls = calls
            style = dict(METHODS[method])
            if method == "frame_coupled_linear":
                style["label"] = coupled_label
            axis.plot(
                calls,
                rmse,
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6.5,
                markeredgewidth=1.0,
                markeredgecolor="white",
                zorder=3,
            )
            axis.fill_between(
                calls,
                lower,
                upper,
                color=style["color"],
                alpha=0.12,
                linewidth=0,
                zorder=2,
            )
        axis.set_title(title, loc="left", fontsize=12.5, pad=10)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.yaxis.set_major_locator(
            LogLocator(
                base=10,
                subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0),
                numticks=10,
            )
        )
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:.1e}")
        )
        axis.margins(y=0.16)
        axis.set_xlabel("Total utility calls per estimate")
        axis.set_ylabel("RMSE vs. high-budget MC reference")
        axis.grid(
            True,
            which="major",
            color="#D9DEE5",
            linewidth=0.8,
            alpha=0.8,
        )
        axis.grid(
            True,
            which="minor",
            color="#EEF1F4",
            linewidth=0.5,
            alpha=0.55,
        )
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color("#6B7280")
        axis.spines["bottom"].set_color("#6B7280")
        axis.tick_params(colors="#374151")
        axis.margins(x=0.08)
        axis.legend(frameon=False, loc="upper right", fontsize=9.5)

    if tick_calls is None:
        raise ValueError("the report contains no budgets")
    for axis in axes:
        axis.set_xticks(tick_calls)
        axis.xaxis.set_major_formatter(FuncFormatter(_calls_label))
        axis.minorticks_off()

    ratio_rows = sorted(
        report["results_by_inner_budget"].values(),
        key=lambda row: row["total_utility_calls_per_estimate"],
    )
    final_ratio_gain = ratio_rows[-1][
        "orbit_rmse_reduction_vs_official_ratio"
    ]
    final_linear_gain = ratio_rows[-1][
        "frame_coupled_rmse_reduction_vs_iid_linear"
    ]
    ratio_calls, ratio_rmse, _, _ = _series(
        report, "frame_orbit_ratio"
    )
    linear_calls, linear_rmse, _, _ = _series(
        report, "frame_coupled_linear"
    )
    for axis, x, y, label in (
        (
            axes[0],
            ratio_calls[-1],
            ratio_rmse[-1],
            _relative_change_label(final_ratio_gain),
        ),
        (
            axes[1],
            linear_calls[-1],
            linear_rmse[-1],
            _relative_change_label(final_linear_gain),
        ),
    ):
        axis.annotate(
            label,
            xy=(x, y),
            xytext=(-8, 22),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=9,
            color="#9A4D00",
            arrowprops={
                "arrowstyle": "-",
                "color": "#9A4D00",
                "linewidth": 0.8,
            },
        )

    configuration = report["configuration"]
    context = _plot_context(report)
    gt_se = report["ground_truth"]["rmse_standard_error"]
    figure.suptitle(
        f"{context['dataset']} data Shapley RMSE vs. total utility calls",
        x=0.075,
        y=0.975,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color="#111827",
    )
    figure.text(
        0.075,
        0.925,
        (
            f"{context['num_players']} training points as players · "
            f"{context['n_test']} fixed test points · "
            f"{context['model']} accuracy · "
            f"{configuration['repeats']} repeats per budget · "
            f"{_calls_label(tick_calls[0], 0)}–"
            f"{_calls_label(tick_calls[-1], 0)} total calls"
        ),
        ha="left",
        va="top",
        fontsize=10.5,
        color="#4B5563",
    )
    cc_call_note = (
        f"OFA totals include {context['boundary_calls']} exact boundary "
        "calls; CC uses each total entirely as two-call complement pairs. "
        if _has_method(report, "official_cc_basic")
        else (
            "Total calls include "
            f"{context['boundary_calls']} exact boundary evaluations. "
        )
    )
    figure.text(
        0.075,
        0.02,
        (
            "Shaded bands: 95% repeat bootstrap intervals. "
            "Both axes use logarithmic scales. "
            f"{cc_call_note}"
            f"Ground-truth SE-RMSE: {gt_se:.2e}."
        ),
        ha="left",
        va="bottom",
        fontsize=9,
        color="#4B5563",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.84,
        bottom=0.16,
        wspace=0.24,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/iris_full_train_rbf_svm_frame_ofa.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "PNG output path; defaults to a dataset-specific name derived "
            "from --input"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    output = args.output or _default_output(args.input)
    plot_results(report, output)
    print(f"Saved {output} and {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()

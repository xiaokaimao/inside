"""Exact-ground-truth Frame-OFA benchmark on the 100-player airport game.

The player cost multiplicities reproduce the airport game in the supplied
paper excerpt.  Utility is the maximum runway requirement in a coalition,
with the empty coalition assigned zero.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from math import comb
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.us_electoral_voting_game import (
    METHOD_LABELS,
    METHOD_ORDER,
    _aggregate,
    _seed_for,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    evaluate_airport,
    exact_airport_shapley,
    frame_coupled_design,
    iid_ofa_design,
    orbit_coupled_frame_design,
)


AIRPORT_CLASS_COUNTS = (8, 12, 6, 14, 8, 9, 13, 10, 10, 10)
AIRPORT_COSTS = np.repeat(
    np.arange(1, 11, dtype=np.float64), AIRPORT_CLASS_COUNTS
)
METHOD_DESCRIPTIONS = {
    "iid_linear": "Independently sample every coalition.",
    "row_greedy": (
        "Choose each coalition from a candidate pool to best complement "
        "the coalitions already selected."
    ),
    "orbit_greedy": (
        "Construct balanced 100-coalition batches in which every player "
        "occupies every cyclic position once."
    ),
}


def game_fingerprint() -> str:
    payload = {
        "class_counts": AIRPORT_CLASS_COUNTS,
        "costs": AIRPORT_COSTS.tolist(),
        "utility": "max coalition cost; empty coalition is zero",
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def exact_size_mean_utilities(costs: np.ndarray) -> np.ndarray:
    """Return ``E[max_{i in S} c_i | |S|=s]`` for every size ``s``."""
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("costs must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("costs must be finite and nonnegative")
    num_players = len(values)
    levels = np.unique(values[values > 0.0])
    means = np.zeros(num_players + 1, dtype=np.float64)
    for size in range(1, num_players + 1):
        denominator = comb(num_players, size)
        for level_index, level in enumerate(levels):
            previous = 0.0 if level_index == 0 else float(levels[level_index - 1])
            lower_count = int(np.count_nonzero(values < level))
            no_eligible = (
                comb(lower_count, size) / denominator
                if size <= lower_count
                else 0.0
            )
            means[size] += (float(level) - previous) * (1.0 - no_eligible)
    return means


def _build_design(
    method: str,
    inner_calls: int,
    seed: int,
    candidate_pool: int,
    mean_balance: float,
):
    num_players = len(AIRPORT_COSTS)
    if method == "iid_linear":
        return iid_ofa_design(
            num_players, inner_calls, seed, compute_diagnostics=True
        )
    if method == "row_greedy":
        return frame_coupled_design(
            num_players,
            inner_calls,
            seed,
            candidate_pool,
            mean_balance,
            mean_balance_mode="raw",
        )
    if method == "orbit_greedy":
        return orbit_coupled_frame_design(
            num_players, inner_calls, seed, candidate_pool
        )
    raise ValueError(f"unknown method: {method}")


def _run_repeat(
    repeat: int,
    inner_budgets: tuple[int, ...],
    candidate_pool: int,
    mean_balance: float,
    base_seed: int,
) -> dict[str, Any]:
    num_players = len(AIRPORT_COSTS)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        evaluate_airport(boundary_rows, AIRPORT_COSTS), num_players
    )
    rows: list[dict[str, Any]] = []
    for budget_index, inner_calls in enumerate(inner_budgets):
        for method_index, method in enumerate(METHOD_ORDER):
            seed = _seed_for(
                base_seed, repeat, budget_index, method_index
            )
            design_start = time.perf_counter()
            design = _build_design(
                method,
                inner_calls,
                seed,
                candidate_pool,
                mean_balance,
            )
            design_seconds = time.perf_counter() - design_start

            estimate_start = time.perf_counter()
            utilities = evaluate_airport(design.coalitions, AIRPORT_COSTS)
            values = estimate_coupled(
                design, utilities, boundary, baseline="linear"
            )
            estimate_seconds = time.perf_counter() - estimate_start
            rows.append(
                {
                    "repeat": repeat,
                    "method": method,
                    "seed": seed,
                    "inner_utility_calls": inner_calls,
                    "total_utility_calls": inner_calls + len(boundary_rows),
                    "estimate": values.tolist(),
                    "design_seconds": design_seconds,
                    "evaluation_and_estimation_seconds": estimate_seconds,
                    "frame_frobenius_discrepancy": float(
                        design.diagnostics["frobenius_discrepancy"]
                    ),
                    "frame_spectral_discrepancy": float(
                        design.diagnostics["spectral_discrepancy"]
                    ),
                }
            )
    return {"repeat": repeat, "rows": rows}


def run_experiment(
    *,
    budget_multipliers: tuple[int, ...],
    repeats: int,
    candidate_pool: int,
    mean_balance: float,
    processes: int,
    base_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("at least two repeats are required")
    if candidate_pool < 1 or processes < 1:
        raise ValueError("candidate pool and processes must be positive")
    if mean_balance < 0:
        raise ValueError("mean balance must be nonnegative")
    if not budget_multipliers or any(
        multiplier < 1 for multiplier in budget_multipliers
    ):
        raise ValueError("budget multipliers must be positive")

    num_players = len(AIRPORT_COSTS)
    inner_budgets = tuple(
        num_players * multiplier for multiplier in budget_multipliers
    )
    truth_start = time.perf_counter()
    truth = exact_airport_shapley(AIRPORT_COSTS)
    size_means = exact_size_mean_utilities(AIRPORT_COSTS)
    truth_seconds = time.perf_counter() - truth_start
    efficiency_error = float(abs(truth.sum() - AIRPORT_COSTS.max()))
    if efficiency_error > 1e-12:
        raise RuntimeError(
            f"exact ground truth violates efficiency by {efficiency_error}"
        )

    tasks = [
        (
            repeat,
            inner_budgets,
            candidate_pool,
            mean_balance,
            base_seed,
        )
        for repeat in range(repeats)
    ]
    repeat_reports: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    if processes == 1:
        for task in tasks:
            repeat_reports.append(_run_repeat(*task))
            print(f"completed repeat {task[0] + 1}/{repeats}", flush=True)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(processes, repeats), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_repeat, *task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                repeat_reports.append(future.result())
                completed += 1
                print(
                    f"completed repeat {completed}/{repeats}", flush=True
                )
    experiment_wall_seconds = time.perf_counter() - wall_start
    repeat_reports.sort(key=lambda item: item["repeat"])

    players = []
    within_class_indices = np.zeros(10, dtype=np.int64)
    for player_index, (cost, value) in enumerate(
        zip(AIRPORT_COSTS, truth), start=1
    ):
        class_index = int(cost) - 1
        within_class_indices[class_index] += 1
        players.append(
            {
                "player": player_index,
                "cost_class": int(cost),
                "within_class_index": int(
                    within_class_indices[class_index]
                ),
                "exact_shapley": float(value),
            }
        )
    class_values = [
        {
            "cost_class": cost,
            "players": int(AIRPORT_CLASS_COUNTS[cost - 1]),
            "tail_players": int(np.count_nonzero(AIRPORT_COSTS >= cost)),
            "exact_shapley_per_player": float(truth[AIRPORT_COSTS == cost][0]),
        }
        for cost in range(1, 11)
    ]
    inner_sizes = np.arange(2, num_players - 1, dtype=np.int64)
    size_weights = 1.0 / np.sqrt(
        inner_sizes * (num_players - inner_sizes)
    )
    size_weights /= size_weights.sum()
    linear_size_baseline = (
        AIRPORT_COSTS.max() * np.arange(num_players + 1) / num_players
    )
    size_residuals = size_means - linear_size_baseline
    return {
        "status": "complete",
        "experiment": "airport_game_frame_ofa",
        "game": {
            "model": "100-player airport cost game",
            "players": num_players,
            "cost_class_counts": list(AIRPORT_CLASS_COUNTS),
            "costs": AIRPORT_COSTS.astype(int).tolist(),
            "utility": "v(S) = max_{i in S} c_i; v(empty) = 0",
            "grand_coalition_utility": float(AIRPORT_COSTS.max()),
            "fingerprint_sha256": game_fingerprint(),
            "source_note": "Configuration reproduced from the supplied paper excerpt.",
        },
        "configuration": {
            "budget_multipliers": list(budget_multipliers),
            "inner_utility_call_budgets": list(inner_budgets),
            "boundary_utility_calls": 2 * num_players + 2,
            "repeats": repeats,
            "candidate_pool": candidate_pool,
            "mean_balance": mean_balance,
            "mean_balance_mode": "raw",
            "processes": min(processes, repeats),
            "base_seed": base_seed,
            "bootstrap_samples": bootstrap_samples,
            "methods": METHOD_DESCRIPTIONS,
            "estimator_baseline": "linear",
            "logical_call_accounting": (
                "Each plotted method receives T inner calls plus the same "
                "202 boundary calls. Analytic ground truth is not counted."
            ),
            "physical_experiment_utility_calls_per_repeat": int(
                2 * num_players
                + 2
                + len(METHOD_ORDER) * sum(inner_budgets)
            ),
            "physical_experiment_utility_calls_total": int(
                repeats
                * (
                    2 * num_players
                    + 2
                    + len(METHOD_ORDER) * sum(inner_budgets)
                )
            ),
        },
        "ground_truth": {
            "algorithm": (
                "exact threshold-game decomposition: phi_i = "
                "sum_{k <= c_i} 1 / count(c_j >= k)"
            ),
            "computation_seconds": truth_seconds,
            "sum": float(truth.sum()),
            "efficiency_error": efficiency_error,
            "class_values": class_values,
            "players": players,
            "fixed_size_profile": [
                {
                    "coalition_size": size,
                    "exact_mean_utility": float(size_means[size]),
                    "linear_baseline": float(linear_size_baseline[size]),
                    "residual": float(size_residuals[size]),
                }
                for size in range(num_players + 1)
            ],
            "ofa_weighted_size_residual_rms": float(
                np.sqrt(
                    np.sum(size_weights * np.square(size_residuals[inner_sizes]))
                )
            ),
            "ofa_weighted_size_residual_mean_absolute": float(
                np.sum(size_weights * np.abs(size_residuals[inner_sizes]))
            ),
            "max_absolute_size_residual": float(
                np.max(np.abs(size_residuals[inner_sizes]))
            ),
        },
        "results_by_inner_budget": _aggregate(
            repeat_reports, truth, inner_budgets, bootstrap_samples
        ),
        "raw_repeats": repeat_reports,
        "experiment_wall_seconds": experiment_wall_seconds,
    }


def write_exact_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "cost_class",
                "players",
                "tail_players",
                "exact_shapley_per_player",
            ),
        )
        writer.writeheader()
        writer.writerows(report["ground_truth"]["class_values"])


def plot_report(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
    """Render the reviewed log-log comparison as a square static figure.

    Chart contract: an eight-point ordered call-budget comparison, one line
    per estimator, with 95% bootstrap uncertainty bands.  Blue identifies the
    IID reference; two orange tones plus marker/line-style changes distinguish
    the geometry-aware designs without relying on color alone.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator

    styles = {
        "iid_linear": {
            "color": "#4676B5",
            "marker": "o",
            "linestyle": "--",
            "markerfacecolor": "white",
        },
        "row_greedy": {
            "color": "#DF7401",
            "marker": "D",
            "linestyle": "-",
            "markerfacecolor": "#DF7401",
        },
        "orbit_greedy": {
            "color": "#A65300",
            "marker": "s",
            "linestyle": ":",
            "markerfacecolor": "white",
        },
    }
    rows = sorted(
        report["results_by_inner_budget"].values(),
        key=lambda row: row["total_utility_calls"],
    )
    figure, axis = plt.subplots(figsize=(8.8, 8.8))
    figure.subplots_adjust(left=0.14, right=0.97, top=0.87, bottom=0.20)
    axis.set_box_aspect(1.0)
    for method in METHOD_ORDER:
        calls = np.asarray(
            [row["total_utility_calls"] for row in rows], dtype=float
        )
        summaries = [row["methods"][method] for row in rows]
        rmse = np.asarray(
            [summary["aggregate_rmse"] for summary in summaries]
        )
        intervals = np.asarray(
            [
                summary["aggregate_rmse_bootstrap_95"]
                for summary in summaries
            ]
        )
        style = styles[method]
        axis.fill_between(
            calls,
            intervals[:, 0],
            intervals[:, 1],
            color=style["color"],
            alpha=0.10,
            linewidth=0.0,
        )
        axis.plot(
            calls,
            rmse,
            label=METHOD_LABELS[method],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=3.2,
            markersize=8.5,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.6,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    all_calls = np.asarray(
        [row["total_utility_calls"] for row in rows], dtype=float
    )
    axis.set_xlim(all_calls.min() * 0.82, all_calls.max() * 1.22)
    axis.xaxis.set_major_locator(
        LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12)
    )
    axis.xaxis.set_minor_locator(NullLocator())
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{value / 1000:g}k" if value >= 1000 else f"{value:g}"
            )
        )
    )
    axis.yaxis.set_major_locator(LogLocator(base=10, numticks=7))
    axis.set_xlabel("Total utility calls", fontsize=17)
    axis.set_ylabel("RMSE vs exact Shapley value", fontsize=17)
    figure.suptitle(
        "Airport game: Shapley RMSE",
        x=0.14,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#20252E",
    )
    axis.set_title(
        "100 planes · cost classes 1–10 · utility is coalition maximum",
        loc="left",
        fontsize=13,
        color="#5B6472",
        pad=16,
    )
    axis.grid(True, which="major", color="#D5DAE2", linewidth=1.0)
    axis.grid(True, which="minor", axis="y", color="#EDF0F4", linewidth=0.7)
    axis.tick_params(axis="both", labelsize=13, width=1.2, length=6)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#616A77")
        spine.set_linewidth(1.4)
    axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="#68717E",
        fontsize=12.5,
        handlelength=3.0,
    )
    figure.text(
        0.14,
        0.055,
        (
            f"95% bootstrap CI over "
            f"{report['configuration']['repeats']} independent designs.\n"
            "Total calls include 202 exact boundary coalitions."
        ),
        fontsize=10.8,
        color="#5B6472",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    pdf_output = output.with_suffix(".pdf")
    figure.savefig(output, dpi=220, facecolor="white")
    figure.savefig(pdf_output, facecolor="white")
    plt.close(figure)
    return output, pdf_output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-multipliers",
        nargs="+",
        type=int,
        default=(1, 2, 5, 10, 25, 50, 100, 250),
        help="inner calls are 100 times each multiplier",
    )
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--mean-balance", type=float, default=0.1)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260824)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/json/airport_100_frame_ofa_greedy.json"),
    )
    parser.add_argument("--plot", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_experiment(
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        candidate_pool=args.candidate_pool,
        mean_balance=args.mean_balance,
        processes=args.processes,
        base_seed=args.base_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_output = args.output.with_name(
        f"{args.output.stem}_exact_values.csv"
    )
    write_exact_csv(report, csv_output)
    plot_output = args.plot or args.output.with_suffix(".png")
    png_output, pdf_output = plot_report(report, plot_output)
    print(f"saved {args.output}")
    print(f"saved {csv_output}")
    print(f"saved {png_output} and {pdf_output}")


if __name__ == "__main__":
    main()

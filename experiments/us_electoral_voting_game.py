"""U.S. Electoral College weighted-voting experiment for Frame-OFA.

The stylized 51-player game treats every state and the District of Columbia as
one bloc.  A coalition wins when its 2024/2028 electoral-vote allocation is at
least 270.  Maine and Nebraska can split votes in the real election, so the
bloc game is deliberately an approximation of the legal allocation process.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    evaluate_weighted_voting,
    exact_shapley_shubik,
    frame_coupled_design,
    iid_ofa_design,
    orbit_coupled_frame_design,
)


NARA_ALLOCATION_URL = (
    "https://www.archives.gov/electoral-college/allocation"
)
US_ELECTORAL_QUOTA = 270
US_ELECTORAL_NAMES = (
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "District of Columbia",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming",
)
US_ELECTORAL_WEIGHTS = np.asarray(
    [
        9,
        3,
        11,
        6,
        54,
        10,
        7,
        3,
        3,
        30,
        16,
        4,
        4,
        19,
        11,
        6,
        6,
        8,
        8,
        4,
        10,
        11,
        15,
        10,
        6,
        10,
        4,
        5,
        6,
        4,
        14,
        5,
        28,
        16,
        3,
        17,
        7,
        8,
        19,
        4,
        9,
        3,
        11,
        40,
        6,
        3,
        13,
        12,
        4,
        10,
        3,
    ],
    dtype=np.int64,
)

METHOD_ORDER = ("iid_linear", "row_greedy", "orbit_greedy")
METHOD_LABELS = {
    "iid_linear": "Random OFA",
    "row_greedy": "Greedy Frame-OFA",
    "orbit_greedy": "Batch-balanced Frame-OFA",
}
METHOD_DESCRIPTIONS = {
    "iid_linear": "Independently sample every coalition.",
    "row_greedy": (
        "Choose each coalition from a candidate pool to best complement "
        "the coalitions already selected."
    ),
    "orbit_greedy": (
        "Construct balanced 51-coalition batches in which every player "
        "occupies every cyclic position once."
    ),
}


def game_fingerprint() -> str:
    payload = {
        "names": US_ELECTORAL_NAMES,
        "weights": US_ELECTORAL_WEIGHTS.tolist(),
        "quota": US_ELECTORAL_QUOTA,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seed_for(
    base_seed: int,
    repeat: int,
    budget_index: int,
    method_index: int,
) -> int:
    sequence = np.random.SeedSequence(
        [base_seed, repeat, budget_index, method_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _build_design(
    method: str,
    inner_calls: int,
    seed: int,
    candidate_pool: int,
    mean_balance: float,
):
    num_players = len(US_ELECTORAL_WEIGHTS)
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
    num_players = len(US_ELECTORAL_WEIGHTS)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        evaluate_weighted_voting(
            boundary_rows, US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        ),
        num_players,
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
            utilities = evaluate_weighted_voting(
                design.coalitions,
                US_ELECTORAL_WEIGHTS,
                US_ELECTORAL_QUOTA,
            )
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


def _bootstrap_rmse_interval(
    squared_error_by_repeat: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(seed)
    repeats = len(squared_error_by_repeat)
    indices = rng.integers(0, repeats, size=(samples, repeats))
    draws = np.sqrt(squared_error_by_repeat[indices].mean(axis=1))
    return tuple(
        float(value) for value in np.quantile(draws, [0.025, 0.975])
    )


def _bootstrap_relative_rmse_interval(
    method_squared_error: np.ndarray,
    iid_squared_error: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    """Bootstrap the aggregate RMSE change using aligned repeat draws."""
    rng = np.random.default_rng(seed)
    repeats = len(iid_squared_error)
    indices = rng.integers(0, repeats, size=(samples, repeats))
    method_draws = np.sqrt(method_squared_error[indices].mean(axis=1))
    iid_draws = np.sqrt(iid_squared_error[indices].mean(axis=1))
    changes = method_draws / iid_draws - 1.0
    return tuple(
        float(value) for value in np.quantile(changes, [0.025, 0.975])
    )


def _aggregate(
    repeat_reports: list[dict[str, Any]],
    truth: np.ndarray,
    inner_budgets: tuple[int, ...],
    bootstrap_samples: int,
) -> dict[str, Any]:
    flat_rows = [
        row
        for repeat_report in repeat_reports
        for row in repeat_report["rows"]
    ]
    summaries: dict[str, Any] = {}
    for budget_index, inner_calls in enumerate(inner_budgets):
        method_summaries: dict[str, Any] = {}
        squared_errors: dict[str, np.ndarray] = {}
        for method_index, method in enumerate(METHOD_ORDER):
            selected = sorted(
                (
                    row
                    for row in flat_rows
                    if row["inner_utility_calls"] == inner_calls
                    and row["method"] == method
                ),
                key=lambda row: row["repeat"],
            )
            estimates = np.asarray(
                [row["estimate"] for row in selected],
                dtype=np.float64,
            )
            errors = estimates - truth[None, :]
            squared_error_by_repeat = np.mean(
                np.square(errors), axis=1
            )
            squared_errors[method] = squared_error_by_repeat
            rmse = float(np.sqrt(squared_error_by_repeat.mean()))
            interval = _bootstrap_rmse_interval(
                squared_error_by_repeat,
                bootstrap_samples,
                _seed_for(90210, 0, budget_index, method_index),
            )
            method_summaries[method] = {
                "label": METHOD_LABELS[method],
                "aggregate_rmse": rmse,
                "aggregate_rmse_bootstrap_95": list(interval),
                "mean_repeat_rmse": float(
                    np.mean(np.sqrt(squared_error_by_repeat))
                ),
                "std_repeat_rmse": float(
                    np.std(
                        np.sqrt(squared_error_by_repeat), ddof=1
                    )
                ),
                "bias_l2": float(
                    np.linalg.norm(estimates.mean(axis=0) - truth)
                ),
                "mean_design_seconds": float(
                    np.mean([row["design_seconds"] for row in selected])
                ),
                "median_design_seconds": float(
                    np.median([row["design_seconds"] for row in selected])
                ),
                "mean_frame_frobenius_discrepancy": float(
                    np.mean(
                        [
                            row["frame_frobenius_discrepancy"]
                            for row in selected
                        ]
                    )
                ),
                "mean_frame_spectral_discrepancy": float(
                    np.mean(
                        [
                            row["frame_spectral_discrepancy"]
                            for row in selected
                        ]
                    )
                ),
            }
        iid_rmse = method_summaries["iid_linear"]["aggregate_rmse"]
        iid_squared_error = squared_errors["iid_linear"]
        for method_index, method in enumerate(METHOD_ORDER):
            method_summaries[method]["rmse_change_vs_iid"] = float(
                method_summaries[method]["aggregate_rmse"] / iid_rmse - 1.0
            )
            method_summaries[method][
                "rmse_change_vs_iid_bootstrap_95"
            ] = list(
                _bootstrap_relative_rmse_interval(
                    squared_errors[method],
                    iid_squared_error,
                    bootstrap_samples,
                    _seed_for(31991, 0, budget_index, method_index),
                )
            )
        summaries[str(inner_calls)] = {
            "inner_utility_calls": inner_calls,
            "total_utility_calls": inner_calls + 2 * len(truth) + 2,
            "methods": method_summaries,
        }
    return summaries


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

    num_players = len(US_ELECTORAL_WEIGHTS)
    inner_budgets = tuple(
        num_players * multiplier for multiplier in budget_multipliers
    )
    truth_start = time.perf_counter()
    truth = exact_shapley_shubik(
        US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
    )
    truth_seconds = time.perf_counter() - truth_start
    efficiency_error = float(abs(truth.sum() - 1.0))
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

    states = [
        {
            "state": name,
            "electoral_votes": int(weight),
            "exact_shapley_shubik": float(value),
        }
        for name, weight, value in zip(
            US_ELECTORAL_NAMES, US_ELECTORAL_WEIGHTS, truth
        )
    ]
    top_states = sorted(
        states,
        key=lambda row: row["exact_shapley_shubik"],
        reverse=True,
    )[:10]
    return {
        "status": "complete",
        "experiment": "us_electoral_college_weighted_voting_frame_ofa",
        "game": {
            "model": "51-jurisdiction bloc weighted voting game",
            "players": num_players,
            "total_electoral_votes": int(US_ELECTORAL_WEIGHTS.sum()),
            "inclusive_quota": US_ELECTORAL_QUOTA,
            "winning_rule": "sum(coalition electoral votes) >= 270",
            "allocation_period": "2024 and 2028 elections",
            "allocation_basis": "2020 Census",
            "source": NARA_ALLOCATION_URL,
            "fingerprint_sha256": game_fingerprint(),
            "modeling_note": (
                "Maine and Nebraska are treated as whole-state blocs even "
                "though the real Electoral College can split their votes."
            ),
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
        },
        "ground_truth": {
            "algorithm": (
                "exact cardinality-weight DP with leave-one-out polynomial "
                "division"
            ),
            "complexity": "O(n^2 * quota) time and O(n * quota) memory",
            "computation_seconds": truth_seconds,
            "sum": float(truth.sum()),
            "efficiency_error": efficiency_error,
            "states": states,
            "top_10": top_states,
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
                "state",
                "electoral_votes",
                "exact_shapley_shubik",
            ),
        )
        writer.writeheader()
        writer.writerows(report["ground_truth"]["states"])


def plot_report(report: dict[str, Any], output: Path) -> tuple[Path, Path]:
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
        "U.S. Electoral College: Shapley RMSE",
        x=0.14,
        y=0.965,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color="#20252E",
    )
    axis.set_title(
        "51 jurisdiction blocs · 538 votes · winning quota ≥270",
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
            "Total calls include 104 exact boundary coalitions."
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
        help="inner calls are 51 times each multiplier",
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
        default=Path(
            "results/us_electoral_college_2024_frame_ofa_greedy.json"
        ),
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

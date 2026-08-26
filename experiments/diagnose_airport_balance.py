"""Separate airport batch balance from candidate-pool frame search."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.airport_game import (
    AIRPORT_COSTS,
    exact_size_mean_utilities,
)
from experiments.us_electoral_voting_game import (
    _bootstrap_relative_rmse_interval,
    _bootstrap_rmse_interval,
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


FAMILIES = ("random", "greedy", "batch_k1", "batch_k64")


def _run_repeat(
    repeat: int,
    inner_budgets: tuple[int, ...],
    base_seed: int,
) -> dict[str, Any]:
    num_players = len(AIRPORT_COSTS)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        evaluate_airport(boundary_rows, AIRPORT_COSTS), num_players
    )
    size_means = exact_size_mean_utilities(AIRPORT_COSTS)
    rows: list[dict[str, Any]] = []
    for budget_index, inner_calls in enumerate(inner_budgets):
        common_batch_seed = _seed_for(
            base_seed, repeat, budget_index, 2
        )
        designs = (
            (
                "random",
                iid_ofa_design(
                    num_players,
                    inner_calls,
                    _seed_for(base_seed, repeat, budget_index, 0),
                    compute_diagnostics=True,
                ),
            ),
            (
                "greedy",
                frame_coupled_design(
                    num_players,
                    inner_calls,
                    _seed_for(base_seed, repeat, budget_index, 1),
                    64,
                    0.1,
                    mean_balance_mode="raw",
                ),
            ),
            (
                "batch_k1",
                orbit_coupled_frame_design(
                    num_players, inner_calls, common_batch_seed, 1
                ),
            ),
            (
                "batch_k64",
                orbit_coupled_frame_design(
                    num_players, inner_calls, common_batch_seed, 64
                ),
            ),
        )
        for family, design in designs:
            utilities = evaluate_airport(design.coalitions, AIRPORT_COSTS)
            linear = estimate_coupled(
                design, utilities, boundary, baseline="linear"
            )
            oracle = estimate_coupled(
                design,
                utilities - size_means[design.sizes],
                boundary,
                baseline="none",
            )
            rows.append(
                {
                    "repeat": repeat,
                    "inner_utility_calls": inner_calls,
                    "family": family,
                    "linear_estimate": linear.tolist(),
                    "oracle_size_baseline_estimate": oracle.tolist(),
                    "linear_oracle_max_absolute_difference": float(
                        np.max(np.abs(linear - oracle))
                    ),
                    "frame_frobenius_discrepancy": float(
                        design.diagnostics["frobenius_discrepancy"]
                    ),
                }
            )
    return {"repeat": repeat, "rows": rows}


def run_diagnostic(
    *,
    budget_multipliers: tuple[int, ...],
    repeats: int,
    processes: int,
    base_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if repeats < 2 or processes < 1:
        raise ValueError("at least two repeats and one process are required")
    num_players = len(AIRPORT_COSTS)
    inner_budgets = tuple(
        num_players * multiplier for multiplier in budget_multipliers
    )
    tasks = [(repeat, inner_budgets, base_seed) for repeat in range(repeats)]
    reports: list[dict[str, Any]] = []
    start = time.perf_counter()
    if processes == 1:
        for task in tasks:
            reports.append(_run_repeat(*task))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(processes, repeats), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_repeat, *task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                reports.append(future.result())
                completed += 1
                print(f"completed repeat {completed}/{repeats}", flush=True)
    reports.sort(key=lambda item: item["repeat"])
    flat_rows = [row for report in reports for row in report["rows"]]
    truth = exact_airport_shapley(AIRPORT_COSTS)

    summaries: dict[str, Any] = {}
    for budget_index, inner_calls in enumerate(inner_budgets):
        family_summaries: dict[str, Any] = {}
        squared_errors: dict[tuple[str, str], np.ndarray] = {}
        for family_index, family in enumerate(FAMILIES):
            selected = sorted(
                (
                    row
                    for row in flat_rows
                    if row["inner_utility_calls"] == inner_calls
                    and row["family"] == family
                ),
                key=lambda row: row["repeat"],
            )
            item: dict[str, Any] = {
                "mean_frame_frobenius_discrepancy": float(
                    np.mean(
                        [row["frame_frobenius_discrepancy"] for row in selected]
                    )
                ),
                "max_linear_oracle_difference": float(
                    np.max(
                        [
                            row["linear_oracle_max_absolute_difference"]
                            for row in selected
                        ]
                    )
                ),
            }
            for baseline_index, (baseline, field) in enumerate(
                (
                    ("linear", "linear_estimate"),
                    ("oracle_size", "oracle_size_baseline_estimate"),
                )
            ):
                estimates = np.asarray(
                    [row[field] for row in selected], dtype=np.float64
                )
                squared_error = np.mean(
                    np.square(estimates - truth[None, :]), axis=1
                )
                squared_errors[(family, baseline)] = squared_error
                rmse = float(np.sqrt(np.mean(squared_error)))
                interval = _bootstrap_rmse_interval(
                    squared_error,
                    bootstrap_samples,
                    _seed_for(
                        77123,
                        baseline_index,
                        budget_index,
                        family_index,
                    ),
                )
                item[baseline] = {
                    "aggregate_rmse": rmse,
                    "bootstrap_95": list(interval),
                    "squared_error_by_repeat": squared_error.tolist(),
                }
            family_summaries[family] = item
        batch_k1_error = squared_errors[("batch_k1", "linear")]
        batch_k64_error = squared_errors[("batch_k64", "linear")]
        batch_search_change = float(
            np.sqrt(batch_k64_error.mean())
            / np.sqrt(batch_k1_error.mean())
            - 1.0
        )
        batch_search_interval = _bootstrap_relative_rmse_interval(
            batch_k64_error,
            batch_k1_error,
            bootstrap_samples,
            _seed_for(88137, 0, budget_index, 0),
        )
        summaries[str(inner_calls)] = {
            "inner_utility_calls": inner_calls,
            "families": family_summaries,
            "batch_k64_rmse_change_vs_k1": batch_search_change,
            "batch_k64_rmse_change_vs_k1_bootstrap_95": list(
                batch_search_interval
            ),
        }
    return {
        "status": "complete",
        "diagnostic": "airport_batch_balance_vs_frame_search",
        "configuration": {
            "players": num_players,
            "inner_utility_call_budgets": list(inner_budgets),
            "repeats": repeats,
            "processes": min(processes, repeats),
            "base_seed": base_seed,
            "bootstrap_samples": bootstrap_samples,
            "greedy_candidate_pool": 64,
            "greedy_mean_balance": 0.1,
            "paired_batch_candidate_pools": [1, 64],
        },
        "interpretation_guardrail": (
            "The oracle size baseline is an analytic diagnostic, not charged "
            "as a deployable estimator. Similar batch K=1 and K=64 errors "
            "would attribute the main gain to exact batch balance rather than "
            "candidate-pool frame search."
        ),
        "results_by_inner_budget": summaries,
        "wall_seconds": time.perf_counter() - start,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-multipliers", nargs="+", type=int, default=(5, 25, 100)
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--processes", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=20260825)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/airport_100_balance_diagnostic.json"),
    )
    args = parser.parse_args()
    report = run_diagnostic(
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        processes=args.processes,
        base_seed=args.base_seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

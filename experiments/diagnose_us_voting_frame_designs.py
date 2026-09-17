"""Reproducible diagnostics for row- versus batch-balanced Frame-OFA."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

import numpy as np

from experiments.us_electoral_voting_game import (
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    evaluate_weighted_voting,
    exact_shapley_shubik,
    frame_coupled_design,
    inner_size_distribution,
    orbit_coupled_frame_design,
)


CONFIGURATIONS = (
    ("row_k1_lambda0.1", "row", 1, 0.1),
    ("row_k8_lambda0.1", "row", 8, 0.1),
    ("row_k32_lambda0.1", "row", 32, 0.1),
    ("row_k64_lambda0", "row", 64, 0.0),
    ("row_k64_lambda0.1", "row", 64, 0.1),
    ("row_k64_lambda1", "row", 64, 1.0),
    ("row_k128_lambda0.1", "row", 128, 0.1),
    ("batch_k64", "batch", 64, 0.0),
)


def exact_size_win_probabilities(
    weights: np.ndarray, quota: int
) -> np.ndarray:
    """Return exact ``P(v(S)=1 | |S|=s)`` for every coalition size."""
    integer_weights = np.asarray(weights, dtype=np.int64)
    num_players = len(integer_weights)
    total_weight = int(integer_weights.sum())
    counts = np.zeros(
        (num_players + 1, total_weight + 1), dtype=object
    )
    counts[0, 0] = 1
    processed = 0
    for weight_value in integer_weights:
        weight = int(weight_value)
        for size in range(processed, -1, -1):
            counts[size + 1, weight:] += counts[
                size, : total_weight + 1 - weight
            ]
        processed += 1
    probabilities = np.zeros(num_players + 1, dtype=np.float64)
    for size in range(num_players + 1):
        winning = int(sum(counts[size, quota:]))
        probabilities[size] = winning / math.comb(num_players, size)
    return probabilities


def _sensitivity_seed(base_seed: int, repeat: int) -> int:
    sequence = np.random.SeedSequence([base_seed, repeat])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _run_sensitivity_repeat(
    repeat: int,
    inner_calls: int,
    base_seed: int,
    truth: np.ndarray,
) -> dict[str, Any]:
    num_players = len(US_ELECTORAL_WEIGHTS)
    seed = _sensitivity_seed(base_seed, repeat)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        evaluate_weighted_voting(
            boundary_rows, US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        ),
        num_players,
    )
    results: list[dict[str, Any]] = []
    for name, family, candidate_pool, mean_balance in CONFIGURATIONS:
        start = time.perf_counter()
        if family == "row":
            design = frame_coupled_design(
                num_players,
                inner_calls,
                seed,
                candidate_pool,
                mean_balance,
                mean_balance_mode="raw",
            )
        else:
            design = orbit_coupled_frame_design(
                num_players, inner_calls, seed, candidate_pool
            )
        design_seconds = time.perf_counter() - start
        utilities = evaluate_weighted_voting(
            design.coalitions,
            US_ELECTORAL_WEIGHTS,
            US_ELECTORAL_QUOTA,
        )
        estimate = estimate_coupled(
            design, utilities, boundary, baseline="linear"
        )
        results.append(
            {
                "configuration": name,
                "family": family,
                "candidate_pool": candidate_pool,
                "mean_balance": mean_balance,
                "repeat": repeat,
                "seed": seed,
                "rmse": float(
                    np.sqrt(np.mean(np.square(estimate - truth)))
                ),
                "design_seconds": design_seconds,
                "frobenius_discrepancy": float(
                    design.diagnostics["frobenius_discrepancy"]
                ),
                "slice_mean_direction_rms": float(
                    design.diagnostics["slice_mean_direction_rms"]
                ),
            }
        )
    return {"repeat": repeat, "results": results}


def _bootstrap_interval(
    values: np.ndarray, seed: int, samples: int = 5000
) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    draws = np.sqrt(np.mean(np.square(values[indices]), axis=1))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def run_diagnostics(
    *,
    inner_calls: int,
    repeats: int,
    processes: int,
    base_seed: int,
) -> dict[str, Any]:
    num_players = len(US_ELECTORAL_WEIGHTS)
    if inner_calls % num_players:
        raise ValueError("inner calls must be divisible by 51")
    if repeats < 2 or processes < 1:
        raise ValueError("repeats must be >=2 and processes positive")
    truth = exact_shapley_shubik(
        US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
    )
    tasks = [
        (repeat, inner_calls, base_seed, truth)
        for repeat in range(repeats)
    ]
    raw: list[dict[str, Any]] = []
    if processes == 1:
        for task in tasks:
            raw.append(_run_sensitivity_repeat(*task))
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(processes, repeats), mp_context=context
        ) as executor:
            futures = [
                executor.submit(_run_sensitivity_repeat, *task)
                for task in tasks
            ]
            for future in as_completed(futures):
                raw.append(future.result())
    raw.sort(key=lambda item: item["repeat"])
    flat = [row for repeat in raw for row in repeat["results"]]
    summaries: list[dict[str, Any]] = []
    for config_index, (name, family, candidate_pool, mean_balance) in enumerate(
        CONFIGURATIONS
    ):
        selected = sorted(
            (row for row in flat if row["configuration"] == name),
            key=lambda row: row["repeat"],
        )
        repeat_rmse = np.asarray([row["rmse"] for row in selected])
        aggregate_rmse = float(np.sqrt(np.mean(np.square(repeat_rmse))))
        summaries.append(
            {
                "configuration": name,
                "family": family,
                "candidate_pool": candidate_pool,
                "mean_balance": mean_balance,
                "aggregate_rmse": aggregate_rmse,
                "aggregate_rmse_bootstrap_95": _bootstrap_interval(
                    repeat_rmse, base_seed + config_index
                ),
                "mean_design_seconds": float(
                    np.mean([row["design_seconds"] for row in selected])
                ),
                "mean_frobenius_discrepancy": float(
                    np.mean(
                        [row["frobenius_discrepancy"] for row in selected]
                    )
                ),
                "mean_slice_direction_rms": float(
                    np.mean(
                        [
                            row["slice_mean_direction_rms"]
                            for row in selected
                        ]
                    )
                ),
            }
        )

    win_probabilities = exact_size_win_probabilities(
        US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
    )
    inner_sizes, size_probabilities, _ = inner_size_distribution(num_players)
    linear_baseline = inner_sizes / num_players
    size_residual = win_probabilities[inner_sizes] - linear_baseline
    size_profile = [
        {
            "size": int(size),
            "winning_probability": float(win_probabilities[size]),
            "linear_baseline": float(size / num_players),
            "size_only_residual": float(
                win_probabilities[size] - size / num_players
            ),
            "ofa_size_probability": float(probability),
        }
        for size, probability in zip(inner_sizes, size_probabilities)
    ]
    return {
        "status": "complete",
        "game": {
            "players": num_players,
            "total_weight": int(US_ELECTORAL_WEIGHTS.sum()),
            "quota": US_ELECTORAL_QUOTA,
        },
        "configuration": {
            "inner_calls": inner_calls,
            "total_calls": inner_calls + 2 * num_players + 2,
            "repeats": repeats,
            "processes": min(processes, repeats),
            "base_seed": base_seed,
        },
        "size_profile": size_profile,
        "size_residual_summary": {
            "ofa_weighted_rms": float(
                np.sqrt(np.sum(size_probabilities * np.square(size_residual)))
            ),
            "ofa_weighted_mean_absolute": float(
                np.sum(size_probabilities * np.abs(size_residual))
            ),
            "maximum_absolute": float(np.max(np.abs(size_residual))),
        },
        "sensitivity": summaries,
        "raw_repeats": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inner-calls", type=int, default=2550)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260825)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/json/us_electoral_college_frame_design_diagnostics.json"
        ),
    )
    args = parser.parse_args()
    report = run_diagnostics(
        inner_calls=args.inner_calls,
        repeats=args.repeats,
        processes=args.processes,
        base_seed=args.base_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

"""Reproducible small-game comparison of OFA and Frame-OFA designs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from frame_ofa import FrameOFAEstimator


def coalition_mask(row: np.ndarray) -> int:
    return sum(int(take) << player for player, take in enumerate(row))


def exact_shapley(table: np.ndarray, num_players: int) -> np.ndarray:
    values = np.zeros(num_players, dtype=np.float64)
    denominator = math.factorial(num_players)
    for player in range(num_players):
        bit = 1 << player
        for mask in range(1 << num_players):
            if mask & bit:
                continue
            size = mask.bit_count()
            weight = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            values[player] += weight * (table[mask | bit] - table[mask])
    return values


def make_game(
    num_players: int, degree: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    additive = rng.normal(size=num_players)
    pairwise = rng.normal(
        scale=0.4, size=(num_players, num_players)
    )
    pairwise = np.triu(pairwise, k=1)
    triples: dict[tuple[int, int, int], float] = {}
    if degree >= 3:
        for i in range(num_players):
            for j in range(i + 1, num_players):
                for k in range(j + 1, num_players):
                    if rng.random() < 0.25:
                        triples[(i, j, k)] = float(
                            rng.normal(scale=0.2)
                        )

    table = np.zeros(1 << num_players, dtype=np.float64)
    for mask in range(1 << num_players):
        row = np.array(
            [bool(mask & (1 << i)) for i in range(num_players)]
        )
        value = 0.7 + additive @ row
        if degree >= 2:
            value += float(row @ pairwise @ row)
        if degree >= 3:
            value += sum(
                coefficient
                for indices, coefficient in triples.items()
                if all(row[index] for index in indices)
            )
        table[mask] = value
    return table, exact_shapley(table, num_players)


def run_case(
    table: np.ndarray,
    truth: np.ndarray,
    num_players: int,
    num_samples: int,
    repeats: int,
    candidate_pool: int,
) -> dict[str, dict[str, float]]:
    def utility(row: np.ndarray) -> float:
        return float(table[coalition_mask(row)])

    modes = ["iid", "coupled", "stratified"]
    if (
        num_samples % num_players == 0
        and num_samples // num_players >= num_players - 3
    ):
        modes.append("orbit")

    results: dict[str, dict[str, float]] = {}
    for mode in modes:
        errors = []
        estimates = []
        randomization_discrepancies = []
        aggregate_discrepancies = []
        query_counts = []
        budget = (
            num_samples // num_players if mode == "orbit" else num_samples
        )
        for seed in range(repeats):
            estimate = FrameOFAEstimator(
                num_players=num_players,
                num_samples=budget,
                mode=mode,
                seed=seed,
                candidate_pool=candidate_pool,
            ).estimate(utility)
            estimates.append(estimate.values)
            errors.append(np.linalg.norm(estimate.values - truth))
            query_counts.append(estimate.utility_evaluations)
            diagnostic = estimate.design.diagnostics
            randomization_discrepancy = diagnostic.get(
                "frobenius_discrepancy",
                diagnostic.get(
                    "slice_frobenius_root_sum_squares", np.nan
                ),
            )
            randomization_discrepancies.append(
                randomization_discrepancy
            )
            aggregate_discrepancies.append(
                diagnostic.get(
                    "aggregate_frobenius_discrepancy",
                    randomization_discrepancy,
                )
            )

        estimate_array = np.asarray(estimates)
        results[mode] = {
            "mean_l2_error": float(np.mean(errors)),
            "rmse": float(
                np.sqrt(
                    np.mean(np.square(estimate_array - truth[None, :]))
                )
            ),
            "bias_l2": float(
                np.linalg.norm(estimate_array.mean(axis=0) - truth)
            ),
            "mean_randomization_frame_discrepancy": float(
                np.nanmean(randomization_discrepancies)
            ),
            "mean_realized_aggregate_frame_discrepancy": float(
                np.nanmean(aggregate_discrepancies)
            ),
            "requested_inner_query_budget": int(num_samples),
            "sampled_inner_queries": int(
                query_counts[0] - (2 * num_players + 2)
            ),
            "total_queries": int(query_counts[0]),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--players", type=int, default=8)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=24)
    parser.add_argument("--game-seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.players > 16:
        raise ValueError("exhaustive truth is restricted to at most 16 players")
    if args.samples < args.players - 3:
        raise ValueError(
            "samples must cover all inner sizes for the stratified estimator"
        )

    report = {
        "configuration": vars(args) | {
            "output": str(args.output) if args.output else None
        },
        "games": {},
    }
    for degree, name in [(1, "additive"), (2, "pairwise"), (3, "degree3")]:
        table, truth = make_game(args.players, degree, args.game_seed)
        report["games"][name] = run_case(
            table,
            truth,
            args.players,
            args.samples,
            args.repeats,
            args.candidate_pool,
        )

    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

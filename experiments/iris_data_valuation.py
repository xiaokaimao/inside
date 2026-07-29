"""Iris data-valuation comparison: official OFA vs Frame-OFA."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from scipy import stats

from frame_ofa import (
    GameEvaluator,
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_coupled,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    frame_coupled_design,
    iid_ofa_design,
)

from .iris_game import IrisLogisticGame, load_official_iris_split
from .iris_sklearn_game import IrisSklearnGame


def antithetic_permutations(
    num_players: int, num_pairs: int, seed: int
) -> np.ndarray:
    """Generate adjacent permutation/reverse-permutation pairs."""
    rng = np.random.default_rng(seed)
    forward = np.stack(
        [rng.permutation(num_players) for _ in range(num_pairs)]
    )
    permutations = np.empty(
        (2 * num_pairs, num_players), dtype=np.int64
    )
    permutations[0::2] = forward
    permutations[1::2] = forward[:, ::-1]
    return permutations


def internal_prefix_coalitions(permutations: np.ndarray) -> np.ndarray:
    """Materialize prefix sizes 1,...,n-1 for every permutation."""
    num_trajectories, num_players = permutations.shape
    rows = np.zeros(
        (num_trajectories * (num_players - 1), num_players),
        dtype=bool,
    )
    cursor = 0
    for permutation in permutations:
        coalition = np.zeros(num_players, dtype=bool)
        for player in permutation[:-1]:
            coalition[player] = True
            rows[cursor] = coalition
            cursor += 1
    return rows


def permutation_contributions(
    permutations: np.ndarray,
    internal_utilities: np.ndarray,
    empty_utility: float,
    full_utility: float,
) -> np.ndarray:
    """Convert prefix utilities into one Shapley vector per trajectory."""
    num_trajectories, num_players = permutations.shape
    internal = np.asarray(internal_utilities, dtype=np.float64)
    if internal.shape != (num_trajectories * (num_players - 1),):
        raise ValueError("internal utility count does not match permutations")
    paths = np.empty(
        (num_trajectories, num_players + 1), dtype=np.float64
    )
    paths[:, 0] = empty_utility
    paths[:, -1] = full_utility
    paths[:, 1:-1] = internal.reshape(
        num_trajectories, num_players - 1
    )
    marginal_by_position = np.diff(paths, axis=1)
    contributions = np.empty_like(marginal_by_position)
    trajectory_indices = np.arange(num_trajectories)[:, None]
    contributions[trajectory_indices, permutations] = marginal_by_position
    return contributions


def ground_truth_summary(
    pair_contributions: np.ndarray, alpha: float = 0.05
) -> dict[str, Any]:
    """Summarize paired-permutation MC uncertainty."""
    num_pairs, num_players = pair_contributions.shape
    values = pair_contributions.mean(axis=0)
    standard_errors = (
        pair_contributions.std(axis=0, ddof=1) / math.sqrt(num_pairs)
    )
    critical = stats.t.ppf(
        1.0 - alpha / (2 * num_players), df=num_pairs - 1
    )
    simultaneous_half_widths = critical * standard_errors
    midpoint = num_pairs // 2
    first_half = pair_contributions[:midpoint].mean(axis=0)
    second_half = pair_contributions[midpoint:].mean(axis=0)
    half_split_rmse = 0.5 * float(
        np.sqrt(np.mean(np.square(first_half - second_half)))
    )
    return {
        "values": values,
        "standard_errors": standard_errors,
        "rmse_standard_error": float(
            np.sqrt(np.mean(np.square(standard_errors)))
        ),
        "simultaneous_half_widths": simultaneous_half_widths,
        "max_simultaneous_half_width": float(
            simultaneous_half_widths.max()
        ),
        "half_split_rmse": half_split_rmse,
    }


def missing_stratum_fraction(design: Any) -> float:
    """Fraction of official player/size/in-out conditional means unobserved."""
    num_players = design.coalitions.shape[1]
    missing = 0
    total = 0
    for size in range(2, num_players - 1):
        rows = design.coalitions[design.sizes == size]
        for player in range(num_players):
            included = rows[:, player] if len(rows) else np.zeros(0, bool)
            missing += int(not np.any(included))
            missing += int(not np.any(~included)) if len(rows) else 1
            total += 2
    return missing / total


def bootstrap_rmse_interval(
    squared_errors: np.ndarray,
    *,
    seed: int,
    num_bootstrap: int = 2000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    num_repeats = len(squared_errors)
    indices = rng.integers(
        0, num_repeats, size=(num_bootstrap, num_repeats)
    )
    bootstrap = np.sqrt(squared_errors[indices].mean(axis=(1, 2)))
    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
    return float(lower), float(upper)


def summarize_method(
    estimates: np.ndarray,
    ground_truth: np.ndarray,
    ground_truth_se: np.ndarray,
    efficiency_target: float,
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    errors = estimates - ground_truth[None, :]
    squared_errors = np.square(errors)
    raw_mse = float(squared_errors.mean())
    ground_truth_noise = float(np.square(ground_truth_se).mean())
    per_repeat_rmse = np.sqrt(squared_errors.mean(axis=1))
    interval = bootstrap_rmse_interval(
        squared_errors,
        seed=bootstrap_seed,
    )
    correlations = [
        float(stats.spearmanr(estimate, ground_truth).statistic)
        for estimate in estimates
    ]
    top_k = min(5, estimates.shape[1])
    truth_top = set(np.argsort(ground_truth)[-top_k:])
    top_overlaps = [
        len(truth_top.intersection(np.argsort(estimate)[-top_k:]))
        / top_k
        for estimate in estimates
    ]
    efficiency_gaps = np.abs(estimates.sum(axis=1) - efficiency_target)
    return {
        "aggregate_rmse": math.sqrt(raw_mse),
        "aggregate_rmse_bootstrap_95": list(interval),
        "noise_corrected_rmse": math.sqrt(
            max(0.0, raw_mse - ground_truth_noise)
        ),
        "mean_repeat_rmse": float(per_repeat_rmse.mean()),
        "std_repeat_rmse": float(per_repeat_rmse.std(ddof=1)),
        "bias_rmse": float(
            np.sqrt(
                np.mean(
                    np.square(estimates.mean(axis=0) - ground_truth)
                )
            )
        ),
        "mean_spearman": float(np.mean(correlations)),
        "mean_top5_overlap": float(np.mean(top_overlaps)),
        "mean_efficiency_gap": float(efficiency_gaps.mean()),
        "max_efficiency_gap": float(efficiency_gaps.max()),
        "estimates": estimates.tolist(),
    }


def paired_rmse_difference(
    first_estimates: np.ndarray,
    second_estimates: np.ndarray,
    ground_truth: np.ndarray,
    *,
    seed: int,
    num_bootstrap: int = 20_000,
) -> dict[str, float | list[float]]:
    """Bootstrap the mean paired per-repeat RMSE difference."""
    first_rmse = np.sqrt(
        np.mean(
            np.square(first_estimates - ground_truth[None, :]), axis=1
        )
    )
    second_rmse = np.sqrt(
        np.mean(
            np.square(second_estimates - ground_truth[None, :]), axis=1
        )
    )
    differences = first_rmse - second_rmse
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(differences),
        size=(num_bootstrap, len(differences)),
    )
    bootstrap_means = differences[indices].mean(axis=1)
    return {
        "mean_rmse_difference": float(differences.mean()),
        "bootstrap_95": np.quantile(
            bootstrap_means, [0.025, 0.975]
        ).tolist(),
        "fraction_first_rmse_larger": float(
            np.mean(differences > 0)
        ),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    game_args, dataset_metadata = load_official_iris_split(
        n_valued=24,
        n_performance=24,
        dataset_seed=args.dataset_seed,
    )
    num_players = len(game_args["y_valued"])
    if args.game == "torch_lr_cross_entropy":
        game_factory = IrisLogisticGame
        utility_name = "negative cross entropy"
        model_name = "float64 3-class linear logistic, one SGD epoch"
    else:
        game_factory = IrisSklearnGame
        model_by_game = {
            "sklearn_lr_accuracy": "lr",
            "linear_svm_accuracy": "linear_svm",
            "rbf_svm_accuracy": "rbf_svm",
        }
        game_args = game_args | {"model": model_by_game[args.game]}
        utility_name = "classification accuracy"
        model_name = args.game
    boundary_rows = boundary_coalitions(num_players)
    print(
        f"Iris game: n={num_players}, jobs={args.jobs}, "
        f"GT antithetic pairs={args.gt_pairs}",
        flush=True,
    )

    with GameEvaluator(
        game_factory,
        game_args,
        n_jobs=args.jobs,
        chunksize=args.chunksize,
        start_method=args.start_method,
        worker_threads=1,
    ) as evaluator:
        boundary_start = time.perf_counter()
        boundary_values = evaluator.evaluate(boundary_rows)
        boundary_seconds = time.perf_counter() - boundary_start
        boundary = boundary_from_utilities(
            boundary_values, num_players
        )

        permutations = antithetic_permutations(
            num_players, args.gt_pairs, args.gt_seed
        )
        prefix_rows = internal_prefix_coalitions(permutations)
        print(
            f"Ground truth: evaluating {len(prefix_rows):,} internal "
            "prefix coalitions...",
            flush=True,
        )
        ground_truth_start = time.perf_counter()
        prefix_utilities = evaluator.evaluate(prefix_rows)
        ground_truth_seconds = time.perf_counter() - ground_truth_start
        trajectory_contributions = permutation_contributions(
            permutations,
            prefix_utilities,
            boundary.empty,
            boundary.full,
        )
        pair_contributions = (
            trajectory_contributions[0::2]
            + trajectory_contributions[1::2]
        ) / 2
        ground_truth = ground_truth_summary(pair_contributions)
        print(
            "Ground truth complete: "
            f"SE-RMSE={ground_truth['rmse_standard_error']:.3e}, "
            f"half-split={ground_truth['half_split_rmse']:.3e}, "
            f"{ground_truth_seconds:.1f}s",
            flush=True,
        )

        results_by_budget: dict[str, Any] = {}
        for budget_index, inner_budget in enumerate(args.budgets):
            print(
                f"Budget {inner_budget}+{len(boundary_rows)}: "
                f"{args.repeats} repeats",
                flush=True,
            )
            estimates = {
                "official_ofa_fixed_ratio": [],
                "iid_linear_ofa": [],
                "frame_coupled": [],
            }
            if inner_budget % num_players == 0:
                estimates["frame_orbit_ratio"] = []
            timings = {
                method: {"design": [], "utility": []}
                for method in estimates
            }
            missing_fractions = []
            for repeat in range(args.repeats):
                seed = args.method_seed + budget_index * 100_000 + repeat
                start = time.perf_counter()
                iid_design = iid_ofa_design(
                    num_players, inner_budget, seed
                )
                iid_design_seconds = time.perf_counter() - start
                start = time.perf_counter()
                iid_utilities = evaluator.evaluate(
                    iid_design.coalitions
                )
                iid_utility_seconds = time.perf_counter() - start

                estimates["official_ofa_fixed_ratio"].append(
                    estimate_official_ratio_ofa(
                        iid_design,
                        iid_utilities,
                        boundary,
                        missing="zero",
                    )
                )
                estimates["iid_linear_ofa"].append(
                    estimate_coupled(
                        iid_design,
                        iid_utilities,
                        boundary,
                        baseline="linear",
                    )
                )
                timings["official_ofa_fixed_ratio"]["design"].append(
                    iid_design_seconds
                )
                timings["official_ofa_fixed_ratio"]["utility"].append(
                    iid_utility_seconds
                )
                timings["iid_linear_ofa"]["design"].append(
                    iid_design_seconds
                )
                timings["iid_linear_ofa"]["utility"].append(
                    iid_utility_seconds
                )
                missing_fractions.append(
                    missing_stratum_fraction(iid_design)
                )

                start = time.perf_counter()
                frame_design = frame_coupled_design(
                    num_players,
                    inner_budget,
                    seed=seed,
                    candidate_pool=args.candidate_pool,
                    mean_balance=args.mean_balance,
                )
                frame_design_seconds = time.perf_counter() - start
                start = time.perf_counter()
                frame_utilities = evaluator.evaluate(
                    frame_design.coalitions
                )
                frame_utility_seconds = time.perf_counter() - start
                estimates["frame_coupled"].append(
                    estimate_coupled(
                        frame_design,
                        frame_utilities,
                        boundary,
                        baseline="linear",
                    )
                )
                timings["frame_coupled"]["design"].append(
                    frame_design_seconds
                )
                timings["frame_coupled"]["utility"].append(
                    frame_utility_seconds
                )

                if "frame_orbit_ratio" in estimates:
                    start = time.perf_counter()
                    orbit_design = cyclic_orbit_frame_design(
                        num_players,
                        num_orbits=inner_budget // num_players,
                        seed=seed,
                        candidate_pool=args.candidate_pool,
                    )
                    orbit_design_seconds = time.perf_counter() - start
                    start = time.perf_counter()
                    orbit_utilities = evaluator.evaluate(
                        orbit_design.coalitions
                    )
                    orbit_utility_seconds = time.perf_counter() - start
                    estimates["frame_orbit_ratio"].append(
                        estimate_ratio_ofa(
                            orbit_design,
                            orbit_utilities,
                            boundary,
                        )
                    )
                    timings["frame_orbit_ratio"]["design"].append(
                        orbit_design_seconds
                    )
                    timings["frame_orbit_ratio"]["utility"].append(
                        orbit_utility_seconds
                    )
                if (repeat + 1) % max(1, args.repeats // 5) == 0:
                    print(
                        f"  completed {repeat + 1}/{args.repeats}",
                        flush=True,
                    )

            truth_values = ground_truth["values"]
            truth_se = ground_truth["standard_errors"]
            efficiency_target = boundary.full - boundary.empty
            summaries = {}
            for method_index, (method, values) in enumerate(
                estimates.items()
            ):
                array = np.asarray(values)
                summaries[method] = summarize_method(
                    array,
                    truth_values,
                    truth_se,
                    efficiency_target,
                    bootstrap_seed=(
                        args.method_seed
                        + budget_index * 100
                        + method_index
                    ),
                )
                summaries[method]["mean_design_seconds"] = float(
                    np.mean(timings[method]["design"])
                )
                summaries[method]["mean_utility_seconds"] = float(
                    np.mean(timings[method]["utility"])
                )

            official_rmse = summaries[
                "official_ofa_fixed_ratio"
            ]["aggregate_rmse"]
            iid_linear_rmse = summaries[
                "iid_linear_ofa"
            ]["aggregate_rmse"]
            frame_rmse = summaries["frame_coupled"]["aggregate_rmse"]
            orbit_rmse = (
                summaries["frame_orbit_ratio"]["aggregate_rmse"]
                if "frame_orbit_ratio" in summaries
                else None
            )
            estimate_arrays = {
                method: np.asarray(values)
                for method, values in estimates.items()
            }
            results_by_budget[str(inner_budget)] = {
                "inner_utility_calls": inner_budget,
                "boundary_utility_calls": len(boundary_rows),
                "total_utility_calls_per_estimate": (
                    inner_budget + len(boundary_rows)
                ),
                "official_missing_stratum_fraction_mean": float(
                    np.mean(missing_fractions)
                ),
                "methods": summaries,
                "frame_rmse_reduction_vs_official_ratio": (
                    (official_rmse - frame_rmse) / official_rmse
                ),
                "frame_rmse_reduction_vs_iid_linear": (
                    (iid_linear_rmse - frame_rmse) / iid_linear_rmse
                ),
                "paired_rmse_differences": {
                    "iid_linear_minus_frame": paired_rmse_difference(
                        estimate_arrays["iid_linear_ofa"],
                        estimate_arrays["frame_coupled"],
                        truth_values,
                        seed=args.method_seed + budget_index * 10 + 7,
                    ),
                    "frame_minus_official_ratio": paired_rmse_difference(
                        estimate_arrays["frame_coupled"],
                        estimate_arrays["official_ofa_fixed_ratio"],
                        truth_values,
                        seed=args.method_seed + budget_index * 10 + 8,
                    ),
                },
            }
            if orbit_rmse is not None:
                budget_result = results_by_budget[str(inner_budget)]
                budget_result[
                    "orbit_rmse_reduction_vs_official_ratio"
                ] = (official_rmse - orbit_rmse) / official_rmse
                budget_result["paired_rmse_differences"][
                    "official_ratio_minus_orbit"
                ] = paired_rmse_difference(
                    estimate_arrays["official_ofa_fixed_ratio"],
                    estimate_arrays["frame_orbit_ratio"],
                    truth_values,
                    seed=args.method_seed + budget_index * 10 + 9,
                )
            print(
                "  RMSE official/iid-linear/frame = "
                f"{official_rmse:.4e}/{iid_linear_rmse:.4e}/"
                f"{frame_rmse:.4e}",
                flush=True,
            )

    serial_equivalent_gt_calls = (
        2 + 2 * args.gt_pairs * (num_players - 1)
    )
    return {
        "configuration": {
            "environment_python": os.sys.version,
            "num_players": num_players,
            "n_performance": len(game_args["y_performance"]),
            "utility": utility_name,
            "model": model_name,
            "learning_rate": game_args["learning_rate"],
            "game_seed": game_args["game_seed"],
            "gt_pairs": args.gt_pairs,
            "gt_seed": args.gt_seed,
            "method_seed": args.method_seed,
            "budgets": args.budgets,
            "repeats": args.repeats,
            "candidate_pool": args.candidate_pool,
            "mean_balance": args.mean_balance,
            "jobs": args.jobs,
            "chunksize": args.chunksize,
            "start_method": args.start_method,
        },
        "dataset": dataset_metadata,
        "boundary": {
            "empty": boundary.empty,
            "full": boundary.full,
            "efficiency_target": boundary.full - boundary.empty,
            "utility_calls": len(boundary_rows),
            "wall_seconds": boundary_seconds,
        },
        "ground_truth": {
            "estimator": (
                "antithetic permutation Monte Carlo "
                "(permutation and reverse)"
            ),
            "independent_pair_units": args.gt_pairs,
            "permutations": 2 * args.gt_pairs,
            "serial_equivalent_utility_calls": (
                serial_equivalent_gt_calls
            ),
            "physical_internal_prefix_calls": len(prefix_rows),
            "wall_seconds": ground_truth_seconds,
            **{
                key: (
                    value.tolist()
                    if isinstance(value, np.ndarray)
                    else value
                )
                for key, value in ground_truth.items()
            },
        },
        "results_by_inner_budget": results_by_budget,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pairs", type=int, default=10_000)
    parser.add_argument(
        "--game",
        choices=[
            "torch_lr_cross_entropy",
            "sklearn_lr_accuracy",
            "linear_svm_accuracy",
            "rbf_svm_accuracy",
        ],
        default="rbf_svm_accuracy",
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[1200, 2400, 4800]
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--candidate-pool", type=int, default=16)
    parser.add_argument("--mean-balance", type=float, default=0.1)
    parser.add_argument("--dataset-seed", type=int, default=2024)
    parser.add_argument("--gt-seed", type=int, default=730_001)
    parser.add_argument("--method-seed", type=int, default=910_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/iris_logistic_frame_ofa.json"),
    )
    args = parser.parse_args()
    if args.gt_pairs < 2 or args.gt_pairs % 2:
        raise ValueError("gt-pairs must be an even integer of at least 2")
    if any(budget < 1 for budget in args.budgets):
        raise ValueError("all budgets must be positive")
    if args.repeats < 2:
        raise ValueError("repeats must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

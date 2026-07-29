"""Full-training-set Iris data valuation with RBF-SVM accuracy utility.

All 120 observations in a stratified 80% training split are Shapley players;
the remaining 30 observations form the fixed test set.  Ground truth is a
streaming antithetic permutation Monte Carlo estimate so tens of millions of
prefix utilities never need to be materialized at once.
"""

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
    orbit_coupled_frame_design,
)

from .iris_data_valuation import (
    missing_stratum_fraction,
    paired_rmse_difference,
    summarize_method,
)
from .iris_game import load_iris_train_test_split
from .iris_sklearn_game import IrisSklearnGame


def _permutation_path_contribution(
    game: Any,
    permutation: np.ndarray,
    *,
    empty_utility: float,
    full_utility: float,
    singletons: np.ndarray,
    leave_one_out: np.ndarray,
) -> np.ndarray:
    """Evaluate one path while reusing exact size-1 and size-(n-1) values."""
    permutation = np.asarray(permutation, dtype=np.int64)
    num_players = len(permutation)
    contribution = np.zeros(num_players, dtype=np.float64)
    coalition = np.zeros(num_players, dtype=bool)

    first = int(permutation[0])
    coalition[first] = True
    previous = float(singletons[first])
    contribution[first] = previous - empty_utility

    # Evaluate prefix sizes 2,...,n-2.  Boundary values supply sizes 1
    # and n-1 exactly for this deterministic game.
    for position in range(1, num_players - 2):
        player = int(permutation[position])
        coalition[player] = True
        current = float(game.evaluate(coalition))
        contribution[player] = current - previous
        previous = current

    penultimate = int(permutation[-2])
    last = int(permutation[-1])
    leave_last_out = float(leave_one_out[last])
    contribution[penultimate] = leave_last_out - previous
    contribution[last] = full_utility - leave_last_out
    return contribution


def _antithetic_pair_task(
    game: Any,
    payload: tuple[
        np.ndarray,
        float,
        float,
        np.ndarray,
        np.ndarray,
    ],
) -> np.ndarray:
    """Return pair-averaged Shapley vectors for a small permutation shard."""
    (
        forward_permutations,
        empty_utility,
        full_utility,
        singletons,
        leave_one_out,
    ) = payload
    pairs = np.empty(
        (len(forward_permutations), forward_permutations.shape[1]),
        dtype=np.float64,
    )
    for index, forward in enumerate(forward_permutations):
        forward_contribution = _permutation_path_contribution(
            game,
            forward,
            empty_utility=empty_utility,
            full_utility=full_utility,
            singletons=singletons,
            leave_one_out=leave_one_out,
        )
        reverse_contribution = _permutation_path_contribution(
            game,
            forward[::-1],
            empty_utility=empty_utility,
            full_utility=full_utility,
            singletons=singletons,
            leave_one_out=leave_one_out,
        )
        pairs[index] = (
            forward_contribution + reverse_contribution
        ) / 2
    return pairs


def summarize_pair_moments(
    *,
    count: int,
    total: np.ndarray,
    total_squares: np.ndarray,
    first_half_count: int,
    first_half_total: np.ndarray,
    second_half_count: int,
    second_half_total: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Recover the same uncertainty summary without retaining MC samples."""
    if count < 2:
        raise ValueError("at least two independent antithetic pairs are needed")
    if first_half_count < 1 or second_half_count < 1:
        raise ValueError("both half-split accumulators must be nonempty")
    values = np.asarray(total, dtype=np.float64) / count
    centered_squares = (
        np.asarray(total_squares, dtype=np.float64)
        - count * np.square(values)
    )
    sample_variances = np.maximum(centered_squares / (count - 1), 0.0)
    standard_errors = np.sqrt(sample_variances / count)
    num_players = len(values)
    critical = stats.t.ppf(
        1.0 - alpha / (2 * num_players), df=count - 1
    )
    simultaneous_half_widths = critical * standard_errors
    first_half = first_half_total / first_half_count
    second_half = second_half_total / second_half_count
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


def streaming_ground_truth(
    evaluator: GameEvaluator,
    *,
    num_players: int,
    num_pairs: int,
    seed: int,
    block_pairs: int,
    task_pairs: int,
    empty_utility: float,
    full_utility: float,
    singletons: np.ndarray,
    leave_one_out: np.ndarray,
    progress_cache: Path | None = None,
    progress_cache_key: dict[str, Any] | None = None,
    resume_progress: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimate Shapley values with bounded-memory antithetic permutations."""
    if num_pairs < 2 or num_pairs % 2:
        raise ValueError("num_pairs must be an even integer of at least two")
    if block_pairs < 1:
        raise ValueError("block_pairs must be positive")
    if task_pairs < 1:
        raise ValueError("task_pairs must be positive")

    rng = np.random.default_rng(seed)
    total = np.zeros(num_players, dtype=np.float64)
    total_squares = np.zeros(num_players, dtype=np.float64)
    first_half_total = np.zeros(num_players, dtype=np.float64)
    second_half_total = np.zeros(num_players, dtype=np.float64)
    first_half_count = 0
    second_half_count = 0
    completed = 0
    elapsed_before = 0.0
    if (
        resume_progress
        and progress_cache is not None
        and progress_cache.exists()
        and progress_cache_key is not None
    ):
        with np.load(progress_cache, allow_pickle=False) as saved:
            stored_key = json.loads(
                str(saved["cache_key"].item())
            )
            if stored_key == progress_cache_key:
                completed = int(saved["completed"].item())
                total = saved["total"].copy()
                total_squares = saved["total_squares"].copy()
                first_half_total = saved[
                    "first_half_total"
                ].copy()
                second_half_total = saved[
                    "second_half_total"
                ].copy()
                first_half_count = int(
                    saved["first_half_count"].item()
                )
                second_half_count = int(
                    saved["second_half_count"].item()
                )
                elapsed_before = float(
                    saved["elapsed_seconds"].item()
                )
                rng.bit_generator.state = json.loads(
                    str(saved["rng_state"].item())
                )
                print(
                    "  resumed ground truth at "
                    f"{completed:,}/{num_pairs:,} pairs",
                    flush=True,
                )
    physical_internal_calls = 2 * completed * (num_players - 3)
    conceptual_internal_calls = 2 * completed * (num_players - 1)
    peak_forward_permutation_bytes = 0
    split = num_pairs // 2
    next_report = 0.05 * (
        math.floor(completed / num_pairs / 0.05 + 1e-12) + 1
    )
    started = time.perf_counter()

    while completed < num_pairs:
        take = min(block_pairs, num_pairs - completed)
        forward = np.stack(
            [rng.permutation(num_players) for _ in range(take)]
        ).astype(np.int16, copy=False)
        peak_forward_permutation_bytes = max(
            peak_forward_permutation_bytes, forward.nbytes
        )
        payloads = [
            (
                forward[start : start + task_pairs],
                empty_utility,
                full_utility,
                singletons,
                leave_one_out,
            )
            for start in range(0, take, task_pairs)
        ]
        pair_shards = evaluator.run_game_tasks(
            _antithetic_pair_task,
            payloads,
            chunksize=1,
        )
        pairs = np.concatenate(pair_shards, axis=0)
        total += pairs.sum(axis=0)
        total_squares += np.square(pairs).sum(axis=0)

        block_start = completed
        block_end = completed + take
        first_stop = min(block_end, split)
        if first_stop > block_start:
            count = first_stop - block_start
            first_half_total += pairs[:count].sum(axis=0)
            first_half_count += count
        second_start = max(split, block_start)
        if block_end > second_start:
            local_start = second_start - block_start
            second_half_total += pairs[local_start:].sum(axis=0)
            second_half_count += block_end - second_start

        completed = block_end
        physical_internal_calls += 2 * take * (num_players - 3)
        conceptual_internal_calls += 2 * take * (num_players - 1)
        fraction = completed / num_pairs
        if fraction + 1e-12 >= next_report or completed == num_pairs:
            elapsed = elapsed_before + time.perf_counter() - started
            rate = physical_internal_calls / elapsed
            remaining = 2 * (num_pairs - completed) * (
                num_players - 3
            )
            eta = remaining / rate if rate > 0 else math.inf
            print(
                "  ground truth "
                f"{completed:,}/{num_pairs:,} pairs "
                f"({100 * fraction:.0f}%), "
                f"{rate:,.0f} utility/s, ETA {eta:.0f}s",
                flush=True,
            )
            while next_report <= fraction + 1e-12:
                next_report += 0.05
            if (
                progress_cache is not None
                and progress_cache_key is not None
            ):
                progress_cache.parent.mkdir(
                    parents=True, exist_ok=True
                )
                temporary = progress_cache.with_suffix(
                    progress_cache.suffix + ".tmp"
                )
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        cache_key=np.asarray(
                            json.dumps(
                                progress_cache_key, sort_keys=True
                            )
                        ),
                        completed=np.asarray(completed),
                        total=total,
                        total_squares=total_squares,
                        first_half_total=first_half_total,
                        second_half_total=second_half_total,
                        first_half_count=np.asarray(
                            first_half_count
                        ),
                        second_half_count=np.asarray(
                            second_half_count
                        ),
                        elapsed_seconds=np.asarray(elapsed),
                        rng_state=np.asarray(
                            json.dumps(rng.bit_generator.state)
                        ),
                    )
                temporary.replace(progress_cache)

    summary = summarize_pair_moments(
        count=num_pairs,
        total=total,
        total_squares=total_squares,
        first_half_count=first_half_count,
        first_half_total=first_half_total,
        second_half_count=second_half_count,
        second_half_total=second_half_total,
    )
    diagnostics = {
        "physical_internal_prefix_calls": physical_internal_calls,
        "conceptual_internal_prefix_calls": conceptual_internal_calls,
        "boundary_reuse_saved_calls": (
            conceptual_internal_calls - physical_internal_calls
        ),
        "peak_forward_permutation_bytes": (
            peak_forward_permutation_bytes
        ),
        "max_worker_tasks_per_outer_block": int(
            math.ceil(block_pairs / task_pairs)
        ),
        "wall_seconds": (
            elapsed_before + time.perf_counter() - started
        ),
    }
    return summary, diagnostics


def _array_json(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_ground_truth_cache(
    path: Path,
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
    cache_key: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cache_key=np.asarray(json.dumps(cache_key, sort_keys=True)),
        diagnostics=np.asarray(json.dumps(diagnostics, sort_keys=True)),
        values=summary["values"],
        standard_errors=summary["standard_errors"],
        rmse_standard_error=np.asarray(summary["rmse_standard_error"]),
        simultaneous_half_widths=summary["simultaneous_half_widths"],
        max_simultaneous_half_width=np.asarray(
            summary["max_simultaneous_half_width"]
        ),
        half_split_rmse=np.asarray(summary["half_split_rmse"]),
    )


def load_ground_truth_cache(
    path: Path, cache_key: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        stored_key = json.loads(str(saved["cache_key"].item()))
        if stored_key != cache_key:
            return None
        summary = {
            "values": saved["values"].copy(),
            "standard_errors": saved["standard_errors"].copy(),
            "rmse_standard_error": float(
                saved["rmse_standard_error"].item()
            ),
            "simultaneous_half_widths": (
                saved["simultaneous_half_widths"].copy()
            ),
            "max_simultaneous_half_width": float(
                saved["max_simultaneous_half_width"].item()
            ),
            "half_split_rmse": float(
                saved["half_split_rmse"].item()
            ),
        }
        diagnostics = json.loads(str(saved["diagnostics"].item()))
    diagnostics["loaded_from_cache"] = True
    return summary, diagnostics


def _write_checkpoint(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_array_json)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    game_args, dataset_metadata = load_iris_train_test_split(
        test_size=args.test_size,
        dataset_seed=args.dataset_seed,
    )
    game_args = game_args | {
        "model": "rbf_svm",
        "regularization": args.regularization,
    }
    num_players = len(game_args["y_valued"])
    boundary_rows = boundary_coalitions(num_players)
    boundary_calls = len(boundary_rows)
    minimum_orbit_calls = num_players * (num_players - 3)
    if any(budget % num_players for budget in args.budgets):
        raise ValueError("every inner budget must be divisible by n=120")
    if any(budget < minimum_orbit_calls for budget in args.budgets):
        raise ValueError(
            "orbit-ratio needs at least n(n-3) inner utility calls"
        )

    cache_key = {
        "dataset": "iris",
        "dataset_seed": args.dataset_seed,
        "test_size": args.test_size,
        "model": "sklearn.svm.SVC",
        "kernel": "rbf",
        "C": args.regularization,
        "gamma": "scale",
        "num_players": num_players,
        "num_pairs": args.gt_pairs,
        "seed": args.gt_seed,
        "pairing": "permutation_reverse",
    }
    print(
        f"Iris full-train valuation: {num_players} players, "
        f"{len(game_args['y_performance'])} test examples, "
        f"{args.jobs} processes",
        flush=True,
    )
    print(
        "Ground truth target: "
        f"{2 + 2 * args.gt_pairs * (num_players - 1):,} "
        "permutation-path equivalent calls",
        flush=True,
    )

    with GameEvaluator(
        IrisSklearnGame,
        game_args,
        n_jobs=args.jobs,
        chunksize=args.chunksize,
        start_method=args.start_method,
        worker_threads=1,
    ) as evaluator:
        boundary_started = time.perf_counter()
        boundary_values = evaluator.evaluate(boundary_rows)
        boundary_seconds = time.perf_counter() - boundary_started
        boundary = boundary_from_utilities(
            boundary_values, num_players
        )
        print(
            f"Boundary complete in {boundary_seconds:.2f}s: "
            f"v(empty)={boundary.empty:.6f}, "
            f"v(full)={boundary.full:.6f}",
            flush=True,
        )

        cached = (
            load_ground_truth_cache(args.ground_truth_cache, cache_key)
            if args.reuse_ground_truth
            else None
        )
        if cached is None:
            ground_truth, ground_truth_diagnostics = streaming_ground_truth(
                evaluator,
                num_players=num_players,
                num_pairs=args.gt_pairs,
                seed=args.gt_seed,
                block_pairs=args.gt_block_pairs,
                task_pairs=args.gt_task_pairs,
                empty_utility=boundary.empty,
                full_utility=boundary.full,
                singletons=boundary.singletons,
                leave_one_out=boundary.leave_one_out,
                progress_cache=args.ground_truth_progress_cache,
                progress_cache_key=cache_key,
                resume_progress=args.reuse_ground_truth,
            )
            ground_truth_diagnostics["loaded_from_cache"] = False
            save_ground_truth_cache(
                args.ground_truth_cache,
                ground_truth,
                ground_truth_diagnostics,
                cache_key,
            )
            print(
                f"Saved ground truth cache {args.ground_truth_cache}",
                flush=True,
            )
        else:
            ground_truth, ground_truth_diagnostics = cached
            print(
                f"Loaded ground truth cache {args.ground_truth_cache}",
                flush=True,
            )
        print(
            "Ground truth precision: "
            f"SE-RMSE={ground_truth['rmse_standard_error']:.3e}, "
            f"half-split={ground_truth['half_split_rmse']:.3e}, "
            "max simultaneous half-width="
            f"{ground_truth['max_simultaneous_half_width']:.3e}",
            flush=True,
        )

        report: dict[str, Any] = {
            "status": "running",
            "configuration": {
                "environment_python": os.sys.version,
                "num_players": num_players,
                "n_test": len(game_args["y_performance"]),
                "utility": "fixed-test classification accuracy",
                "model": (
                    "sklearn.svm.SVC(C="
                    f"{args.regularization}, kernel='rbf', gamma='scale')"
                ),
                "empty_coalition_rule": (
                    "best constant-label accuracy on the fixed test set"
                ),
                "single_class_rule": (
                    "predict the coalition's sole observed class"
                ),
                "gt_pairs": args.gt_pairs,
                "gt_seed": args.gt_seed,
                "gt_block_pairs": args.gt_block_pairs,
                "gt_task_pairs": args.gt_task_pairs,
                "ground_truth_progress_cache": str(
                    args.ground_truth_progress_cache
                ),
                "method_seed": args.method_seed,
                "inner_budgets": args.budgets,
                "total_call_budgets": [
                    budget + boundary_calls for budget in args.budgets
                ],
                "repeats": args.repeats,
                "candidate_pool": args.candidate_pool,
                "mean_balance": args.mean_balance,
                "coupled_design": args.coupled_design,
                "jobs": args.jobs,
                "worker_threads": 1,
                "chunksize": args.chunksize,
                "start_method": args.start_method,
            },
            "dataset": dataset_metadata,
            "boundary": {
                "empty": boundary.empty,
                "full": boundary.full,
                "efficiency_target": boundary.full - boundary.empty,
                "utility_calls": boundary_calls,
                "wall_seconds": boundary_seconds,
            },
            "ground_truth": {
                "estimator": (
                    "streaming antithetic permutation Monte Carlo "
                    "(independent permutation/reverse-permutation pairs)"
                ),
                "independent_pair_units": args.gt_pairs,
                "permutations": 2 * args.gt_pairs,
                "permutation_path_equivalent_utility_calls": (
                    2 + 2 * args.gt_pairs * (num_players - 1)
                ),
                "shared_boundary_calls_physically_evaluated": boundary_calls,
                **ground_truth_diagnostics,
                **{
                    key: _array_json(value)
                    for key, value in ground_truth.items()
                },
            },
            "results_by_inner_budget": {},
        }
        _write_checkpoint(args.output, report)
        if args.ground_truth_only:
            report["status"] = "ground_truth_complete"
            _write_checkpoint(args.output, report)
            return report

        truth_values = ground_truth["values"]
        truth_se = ground_truth["standard_errors"]
        efficiency_target = boundary.full - boundary.empty
        for budget_index, inner_budget in enumerate(args.budgets):
            total_calls = inner_budget + boundary_calls
            print(
                f"Budget {budget_index + 1}/{len(args.budgets)}: "
                f"{inner_budget:,}+{boundary_calls}={total_calls:,} "
                f"calls, {args.repeats} repeats",
                flush=True,
            )
            estimates: dict[str, list[np.ndarray]] = {
                "official_ofa_fixed_ratio": [],
                "iid_linear_ofa": [],
                "frame_coupled_linear": [],
                "frame_orbit_ratio": [],
            }
            timings = {
                method: {"design": [], "utility": []}
                for method in estimates
            }
            missing_fractions: list[float] = []
            for repeat in range(args.repeats):
                seed = (
                    args.method_seed
                    + budget_index * 100_000
                    + repeat
                )

                started = time.perf_counter()
                iid_design = iid_ofa_design(
                    num_players, inner_budget, seed
                )
                iid_design_seconds = time.perf_counter() - started
                started = time.perf_counter()
                iid_utilities = evaluator.evaluate(
                    iid_design.coalitions
                )
                iid_utility_seconds = time.perf_counter() - started
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
                for method in (
                    "official_ofa_fixed_ratio",
                    "iid_linear_ofa",
                ):
                    timings[method]["design"].append(iid_design_seconds)
                    timings[method]["utility"].append(
                        iid_utility_seconds
                    )
                missing_fractions.append(
                    missing_stratum_fraction(iid_design)
                )

                started = time.perf_counter()
                if args.coupled_design == "orbit_coupled":
                    frame_design = orbit_coupled_frame_design(
                        num_players,
                        inner_budget,
                        seed=seed,
                        candidate_pool=args.candidate_pool,
                    )
                else:
                    frame_design = frame_coupled_design(
                        num_players,
                        inner_budget,
                        seed=seed,
                        candidate_pool=args.candidate_pool,
                        mean_balance=args.mean_balance,
                    )
                frame_design_seconds = time.perf_counter() - started
                started = time.perf_counter()
                frame_utilities = evaluator.evaluate(
                    frame_design.coalitions
                )
                frame_utility_seconds = time.perf_counter() - started
                estimates["frame_coupled_linear"].append(
                    estimate_coupled(
                        frame_design,
                        frame_utilities,
                        boundary,
                        baseline="linear",
                    )
                )
                timings["frame_coupled_linear"]["design"].append(
                    frame_design_seconds
                )
                timings["frame_coupled_linear"]["utility"].append(
                    frame_utility_seconds
                )

                started = time.perf_counter()
                orbit_design = cyclic_orbit_frame_design(
                    num_players,
                    num_orbits=inner_budget // num_players,
                    seed=seed,
                    candidate_pool=args.candidate_pool,
                )
                orbit_design_seconds = time.perf_counter() - started
                started = time.perf_counter()
                orbit_utilities = evaluator.evaluate(
                    orbit_design.coalitions
                )
                orbit_utility_seconds = time.perf_counter() - started
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

                print(
                    f"  repeat {repeat + 1}/{args.repeats}: "
                    f"IID {iid_utility_seconds:.1f}s, "
                    f"coupled {frame_utility_seconds:.1f}s "
                    f"(design {frame_design_seconds:.1f}s), "
                    f"orbit {orbit_utility_seconds:.1f}s",
                    flush=True,
                )

            summaries: dict[str, Any] = {}
            estimate_arrays = {
                method: np.asarray(values)
                for method, values in estimates.items()
            }
            for method_index, (method, array) in enumerate(
                estimate_arrays.items()
            ):
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
            coupled_rmse = summaries[
                "frame_coupled_linear"
            ]["aggregate_rmse"]
            orbit_rmse = summaries[
                "frame_orbit_ratio"
            ]["aggregate_rmse"]
            iid_linear_rmse = summaries[
                "iid_linear_ofa"
            ]["aggregate_rmse"]
            budget_result = {
                "inner_utility_calls": inner_budget,
                "boundary_utility_calls": boundary_calls,
                "total_utility_calls_per_estimate": total_calls,
                "official_missing_stratum_fraction_mean": float(
                    np.mean(missing_fractions)
                ),
                "methods": summaries,
                "frame_coupled_rmse_reduction_vs_iid_linear": (
                    (iid_linear_rmse - coupled_rmse)
                    / iid_linear_rmse
                ),
                "orbit_rmse_reduction_vs_official_ratio": (
                    (official_rmse - orbit_rmse) / official_rmse
                ),
                "paired_rmse_differences": {
                    "iid_linear_minus_frame_coupled": (
                        paired_rmse_difference(
                            estimate_arrays["iid_linear_ofa"],
                            estimate_arrays["frame_coupled_linear"],
                            truth_values,
                            seed=args.method_seed
                            + budget_index * 10
                            + 7,
                        )
                    ),
                    "official_ratio_minus_orbit": (
                        paired_rmse_difference(
                            estimate_arrays[
                                "official_ofa_fixed_ratio"
                            ],
                            estimate_arrays["frame_orbit_ratio"],
                            truth_values,
                            seed=args.method_seed
                            + budget_index * 10
                            + 9,
                        )
                    ),
                },
            }
            report["results_by_inner_budget"][
                str(inner_budget)
            ] = budget_result
            _write_checkpoint(args.output, report)
            print(
                "  aggregate RMSE official / IID-linear / "
                "coupled-linear / orbit-ratio = "
                f"{official_rmse:.4e} / {iid_linear_rmse:.4e} / "
                f"{coupled_rmse:.4e} / {orbit_rmse:.4e}",
                flush=True,
            )

    report["status"] = "complete"
    _write_checkpoint(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pairs", type=int, default=800_000)
    parser.add_argument("--gt-block-pairs", type=int, default=1_024)
    parser.add_argument("--gt-task-pairs", type=int, default=8)
    parser.add_argument(
        "--budgets",
        type=int,
        nargs="+",
        default=[14_040, 21_600, 36_000, 60_000, 96_000],
    )
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=128)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--candidate-pool", type=int, default=4)
    parser.add_argument("--mean-balance", type=float, default=0.1)
    parser.add_argument(
        "--coupled-design",
        choices=("row_greedy", "orbit_coupled"),
        default="row_greedy",
        help=(
            "coalition coupling used by the Frame-OFA linear estimator; "
            "orbit_coupled is the scalable complete-orbit construction"
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--regularization", type=float, default=1.0)
    parser.add_argument("--dataset-seed", type=int, default=2024)
    parser.add_argument("--gt-seed", type=int, default=730_001)
    parser.add_argument("--method-seed", type=int, default=910_001)
    parser.add_argument(
        "--ground-truth-cache",
        type=Path,
        default=Path("results/iris_full_train_rbf_svm_gt.npz"),
    )
    parser.add_argument(
        "--ground-truth-progress-cache",
        type=Path,
        default=Path(
            "results/iris_full_train_rbf_svm_gt.partial.npz"
        ),
    )
    parser.add_argument(
        "--reuse-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ground-truth-only",
        action="store_true",
        help="stop after writing the ground-truth cache and report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/iris_full_train_rbf_svm_frame_ofa.json"
        ),
    )
    args = parser.parse_args()
    if args.gt_pairs < 2 or args.gt_pairs % 2:
        raise ValueError("gt-pairs must be an even integer of at least 2")
    if args.gt_block_pairs < 1:
        raise ValueError("gt-block-pairs must be positive")
    if args.gt_task_pairs < 1:
        raise ValueError("gt-task-pairs must be positive")
    if args.gt_task_pairs > args.gt_block_pairs:
        raise ValueError("gt-task-pairs cannot exceed gt-block-pairs")
    if len(args.budgets) != 5:
        raise ValueError("the formal experiment requires five budgets")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("budgets must be distinct")
    if any(budget < 1 for budget in args.budgets):
        raise ValueError("all budgets must be positive")
    if args.repeats < 2:
        raise ValueError("repeats must be at least two")
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    if not 0 < args.test_size < 1:
        raise ValueError("test-size must lie strictly between zero and one")
    return args


def main() -> None:
    args = parse_args()
    run_experiment(args)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

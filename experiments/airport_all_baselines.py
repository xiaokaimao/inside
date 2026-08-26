"""Wine-style all-baseline benchmark for the 100-player airport game."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import fields, is_dataclass
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np

from experiments.airport_game import (
    AIRPORT_CLASS_COUNTS,
    AIRPORT_COSTS,
    game_fingerprint,
)
from experiments.us_electoral_voting_game import (
    _bootstrap_rmse_interval,
    _seed_for,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_basic_cc,
    estimate_diff,
    estimate_gels_shapley,
    estimate_group_testing,
    estimate_kernel_shap,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    estimate_sdiff,
    estimate_stratified_marginal_mc,
    estimate_tmc_shapley,
    evaluate_airport,
    exact_airport_shapley,
    iid_ofa_design,
)


METHOD_ORDER = (
    "frame_orbit_ratio",
    "official_ofa_fixed_ratio",
    "official_cc_basic",
    "s_diff",
    "diff",
    "group_testing",
    "kernel_shap_sampled",
    "gels_shapley",
    "tmc_shapley",
    "stratified_marginal_mc",
)

METHOD_LABELS = {
    "frame_orbit_ratio": "Frame-OFA ratio",
    "official_ofa_fixed_ratio": "OFA ratio",
    "official_cc_basic": "CC",
    "s_diff": "S-Diff",
    "diff": "Diff",
    "group_testing": "Group Testing",
    "kernel_shap_sampled": "KernelSHAP",
    "gels_shapley": "GELS-Shapley",
    "tmc_shapley": "TMC-Shapley",
    "stratified_marginal_mc": "Stratified MC",
}


class AirportGame:
    """Scalar game interface used by locally ported baseline tasks."""

    def __init__(self, costs: np.ndarray = AIRPORT_COSTS) -> None:
        self.costs = np.asarray(costs, dtype=np.float64).copy()

    def evaluate(self, coalition: np.ndarray) -> float:
        row = np.asarray(coalition, dtype=bool)
        if row.shape != self.costs.shape:
            raise ValueError("coalition and costs disagree on player count")
        if not np.any(row):
            return 0.0
        return float(np.max(self.costs[row]))


class SerialAirportEvaluator:
    """Vectorized evaluator plus the coarse-task protocol, without nesting pools."""

    def __init__(self) -> None:
        self.game = AirportGame()

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        return np.asarray(
            evaluate_airport(coalitions, self.game.costs),
            dtype=np.float64,
        )

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]:
        if chunksize < 1:
            raise ValueError("chunksize must be positive")
        return [task_func(self.game, payload) for payload in payloads]


def _compact_json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _compact_json(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, np.ndarray):
        if value.size <= 256:
            return value.tolist()
        numeric = np.asarray(value)
        return {
            "omitted_dense_array": True,
            "shape": list(numeric.shape),
            "dtype": str(numeric.dtype),
            "minimum": float(numeric.min()),
            "maximum": float(numeric.max()),
            "sum": float(numeric.sum(dtype=np.float64)),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _compact_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_compact_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def total_call_budgets(
    budget_multipliers: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    num_players = len(AIRPORT_COSTS)
    if not budget_multipliers or any(value < 1 for value in budget_multipliers):
        raise ValueError("budget multipliers must be positive")
    if any(
        left >= right
        for left, right in zip(budget_multipliers, budget_multipliers[1:])
    ):
        raise ValueError("budget multipliers must be strictly increasing")
    inner = tuple(num_players * value for value in budget_multipliers)
    boundary_calls = 2 * num_players + 2
    return inner, tuple(value + boundary_calls for value in inner)


def _diagnostic_value(diagnostics: Any, name: str) -> Any:
    if isinstance(diagnostics, Mapping):
        return diagnostics[name]
    return getattr(diagnostics, name)


def _run_ratio_method(
    method: str,
    *,
    inner_calls: int,
    total_calls: int,
    seed: int,
    candidate_pool: int,
) -> tuple[np.ndarray, int, Any]:
    num_players = len(AIRPORT_COSTS)
    boundary_rows = boundary_coalitions(num_players)
    boundary = boundary_from_utilities(
        evaluate_airport(boundary_rows, AIRPORT_COSTS), num_players
    )
    if method == "official_ofa_fixed_ratio":
        design = iid_ofa_design(
            num_players, inner_calls, seed, compute_diagnostics=False
        )
        utilities = evaluate_airport(design.coalitions, AIRPORT_COSTS)
        values = estimate_official_ratio_ofa(
            design, utilities, boundary, missing="zero"
        )
    elif method == "frame_orbit_ratio":
        if inner_calls % num_players:
            raise ValueError("Frame-OFA ratio requires n-multiple inner calls")
        design = cyclic_orbit_frame_design(
            num_players,
            num_orbits=inner_calls // num_players,
            seed=seed,
            candidate_pool=candidate_pool,
        )
        utilities = evaluate_airport(design.coalitions, AIRPORT_COSTS)
        values = estimate_ratio_ofa(design, utilities, boundary)
    else:
        raise ValueError(f"not a ratio method: {method}")
    actual_calls = len(boundary_rows) + len(design.coalitions)
    if actual_calls != total_calls:
        raise RuntimeError("ratio-method physical call accounting mismatch")
    diagnostics = {
        "utility_evaluations": actual_calls,
        "boundary_utility_evaluations": len(boundary_rows),
        "inner_utility_evaluations": len(design.coalitions),
        "design_method": design.method,
        "design_diagnostics": _compact_json(design.diagnostics),
    }
    return values, actual_calls, diagnostics


def _run_external_method(
    method: str,
    *,
    total_calls: int,
    seed: int,
    num_tasks: int,
) -> tuple[np.ndarray, int, Any]:
    evaluator = SerialAirportEvaluator()
    num_players = len(AIRPORT_COSTS)
    if method == "official_cc_basic":
        result = estimate_basic_cc(
            evaluator,
            num_players,
            total_calls // 2,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "s_diff":
        result = estimate_sdiff(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "diff":
        result = estimate_diff(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "group_testing":
        result = estimate_group_testing(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "kernel_shap_sampled":
        result = estimate_kernel_shap(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
            ridge=1e-8,
        )
    elif method == "gels_shapley":
        result = estimate_gels_shapley(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "tmc_shapley":
        result = estimate_tmc_shapley(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    elif method == "stratified_marginal_mc":
        result = estimate_stratified_marginal_mc(
            evaluator,
            num_players,
            total_calls,
            seed,
            num_tasks=num_tasks,
        )
    else:
        raise ValueError(f"unsupported method: {method}")
    actual_calls = int(
        _diagnostic_value(result.diagnostics, "utility_evaluations")
    )
    if actual_calls > total_calls:
        raise RuntimeError("method exceeded its physical-call cap")
    return (
        np.asarray(result.values, dtype=np.float64),
        actual_calls,
        _compact_json(result.diagnostics),
    )


def _run_method_across_budgets(
    repeat: int,
    method: str,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    base_seed: int,
    candidate_pool: int,
    num_tasks: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    method_index = METHOD_ORDER.index(method)
    for budget_index, (inner_calls, total_calls) in enumerate(
        zip(inner_budgets, total_budgets)
    ):
        seed = _seed_for(
            base_seed, repeat, budget_index, method_index
        )
        started = time.perf_counter()
        if method in {"frame_orbit_ratio", "official_ofa_fixed_ratio"}:
            values, actual_calls, diagnostics = _run_ratio_method(
                method,
                inner_calls=inner_calls,
                total_calls=total_calls,
                seed=seed,
                candidate_pool=candidate_pool,
            )
        else:
            values, actual_calls, diagnostics = _run_external_method(
                method,
                total_calls=total_calls,
                seed=seed,
                num_tasks=num_tasks,
            )
        elapsed = time.perf_counter() - started
        if values.shape != (len(AIRPORT_COSTS),) or not np.all(
            np.isfinite(values)
        ):
            raise RuntimeError(f"{method} returned an invalid estimate")
        rows.append(
            {
                "repeat": repeat,
                "method": method,
                "seed": seed,
                "inner_utility_call_budget": inner_calls,
                "target_total_utility_calls": total_calls,
                "actual_utility_calls": actual_calls,
                "estimate": values.tolist(),
                "elapsed_seconds": elapsed,
                "diagnostics": diagnostics,
            }
        )
    return {"repeat": repeat, "method": method, "rows": rows}


def _aggregate(
    task_reports: list[dict[str, Any]],
    truth: np.ndarray,
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    flat_rows = [row for report in task_reports for row in report["rows"]]
    summaries: dict[str, Any] = {}
    for budget_index, (inner_calls, total_calls) in enumerate(
        zip(inner_budgets, total_budgets)
    ):
        method_summaries: dict[str, Any] = {}
        for method_index, method in enumerate(METHOD_ORDER):
            selected = sorted(
                (
                    row
                    for row in flat_rows
                    if row["target_total_utility_calls"] == total_calls
                    and row["method"] == method
                ),
                key=lambda row: row["repeat"],
            )
            if len(selected) != repeats:
                raise RuntimeError("method-budget cell has missing repeats")
            estimates = np.asarray(
                [row["estimate"] for row in selected], dtype=np.float64
            )
            errors = estimates - truth[None, :]
            squared_error = np.mean(np.square(errors), axis=1)
            interval = _bootstrap_rmse_interval(
                squared_error,
                bootstrap_samples,
                _seed_for(94127, 0, budget_index, method_index),
            )
            actual_calls = np.asarray(
                [row["actual_utility_calls"] for row in selected],
                dtype=np.float64,
            )
            method_summaries[method] = {
                "label": METHOD_LABELS[method],
                "estimates": estimates.tolist(),
                "aggregate_rmse": float(np.sqrt(squared_error.mean())),
                "aggregate_rmse_bootstrap_95": list(interval),
                "mean_repeat_rmse": float(
                    np.mean(np.sqrt(squared_error))
                ),
                "std_repeat_rmse": float(
                    np.std(np.sqrt(squared_error), ddof=1)
                ),
                "bias_l2": float(
                    np.linalg.norm(estimates.mean(axis=0) - truth)
                ),
                "mean_efficiency_residual": float(
                    np.mean(estimates.sum(axis=1) - 10.0)
                ),
                "max_absolute_efficiency_residual": float(
                    np.max(np.abs(estimates.sum(axis=1) - 10.0))
                ),
                "target_total_utility_calls": total_calls,
                "actual_utility_calls_by_repeat": actual_calls.astype(int).tolist(),
                "mean_actual_utility_calls": float(actual_calls.mean()),
                "mean_elapsed_seconds": float(
                    np.mean([row["elapsed_seconds"] for row in selected])
                ),
                "diagnostics_by_repeat": [
                    row["diagnostics"] for row in selected
                ],
            }
        summaries[str(inner_calls)] = {
            "inner_utility_calls": inner_calls,
            "total_utility_calls_per_estimate": total_calls,
            "methods": method_summaries,
        }
    return summaries


def run_experiment(
    *,
    budget_multipliers: tuple[int, ...],
    repeats: int,
    candidate_pool: int,
    processes: int,
    base_seed: int,
    bootstrap_samples: int,
    num_tasks: int,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("at least two repeats are required")
    if candidate_pool < 1 or processes < 1 or num_tasks < 1:
        raise ValueError("candidate pool, processes, and tasks must be positive")
    inner_budgets, total_budgets = total_call_budgets(budget_multipliers)
    num_players = len(AIRPORT_COSTS)
    if min(inner_budgets) < num_players * (num_players - 3):
        raise ValueError(
            "the smallest budget cannot cover every Frame-OFA ratio size"
        )
    if min(total_budgets) < 2 * num_players * num_players:
        raise ValueError(
            "the smallest budget cannot cover every Stratified-MC stratum"
        )

    truth = exact_airport_shapley(AIRPORT_COSTS)
    tasks = [
        (
            repeat,
            method,
            inner_budgets,
            total_budgets,
            base_seed,
            candidate_pool,
            num_tasks,
        )
        for repeat in range(repeats)
        for method in METHOD_ORDER
    ]
    reports: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    if processes == 1:
        for task in tasks:
            reports.append(_run_method_across_budgets(*task))
            print(
                f"completed repeat={task[0] + 1}, method={task[1]}",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(processes, len(tasks)), mp_context=context
        ) as executor:
            futures = [executor.submit(_run_method_across_budgets, *task) for task in tasks]
            completed = 0
            for future in as_completed(futures):
                report = future.result()
                reports.append(report)
                completed += 1
                print(
                    f"completed {completed}/{len(tasks)}: "
                    f"repeat={report['repeat'] + 1}, method={report['method']}",
                    flush=True,
                )
    reports.sort(key=lambda row: (row["repeat"], METHOD_ORDER.index(row["method"])))
    return {
        "status": "complete",
        "experiment": "airport_game_wine_style_all_baselines",
        "game": {
            "model": "100-player airport cost game",
            "players": num_players,
            "cost_class_counts": list(AIRPORT_CLASS_COUNTS),
            "utility": "v(S) = max_{i in S} c_i; v(empty) = 0",
            "fingerprint_sha256": game_fingerprint(),
        },
        "configuration": {
            "protocol": "Wine-main-figure-compatible method set and scale",
            "budget_multipliers": list(budget_multipliers),
            "inner_utility_call_budgets": list(inner_budgets),
            "total_call_budgets": list(total_budgets),
            "boundary_utility_calls_for_ratio_methods": 2 * num_players + 2,
            "repeats": repeats,
            "candidate_pool": candidate_pool,
            "processes": min(processes, len(tasks)),
            "base_seed": base_seed,
            "bootstrap_samples": bootstrap_samples,
            "num_tasks_per_serial_baseline": num_tasks,
            "methods": list(METHOD_ORDER),
            "method_labels": METHOD_LABELS,
            "call_budget_semantics": "total physical utility-call cap",
            "tmc_x_coordinate": "mean observed physical calls after truncation",
        },
        "ground_truth": {
            "algorithm": "exact threshold-game decomposition",
            "values": truth.tolist(),
            "sum": float(truth.sum()),
            "efficiency_error": float(abs(truth.sum() - 10.0)),
        },
        "results_by_inner_budget": _aggregate(
            reports,
            truth,
            inner_budgets,
            total_budgets,
            repeats,
            bootstrap_samples,
        ),
        "experiment_wall_seconds": time.perf_counter() - wall_start,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-multipliers",
        nargs="+",
        type=int,
        default=(500, 1000, 2000, 5000, 10000),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--candidate-pool", type=int, default=64)
    parser.add_argument("--processes", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=20260826)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--num-tasks", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/airport_100_all_baselines_3repeats_50k_1m.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_experiment(
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        candidate_pool=args.candidate_pool,
        processes=args.processes,
        base_seed=args.base_seed,
        bootstrap_samples=args.bootstrap_samples,
        num_tasks=args.num_tasks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_compact_json) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

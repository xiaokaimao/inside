"""Fresh, end-to-end wall-clock benchmark for the formal Wine methods.

Unlike the historical assembled report, this runner does not reuse any
method estimate, coalition utility, boundary utility, process pool, or timing
measurement.  Every ``method x budget x repeat`` cell creates a fresh utility
pool.  The cell timer starts before sampling/design and stops after the final
Shapley vector is available and the pool has shut down.

The already-audited high-budget Monte Carlo reference is reused only as an
accuracy target.  It is not part of any method's runtime.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.iris_data_valuation import summarize_method
from experiments.iris_sklearn_game import SklearnClassificationGame
from experiments.run_wine_inside_comparison import (
    BOUNDARY_CALLS,
    BUDGET_MULTIPLIERS,
    DATASET_SEED,
    INNER_BUDGETS,
    INSIDE_GREEDY_CANDIDATE_POOL,
    MEAN_BALANCE_LAMBDA0,
    MEAN_BALANCE_MODE,
    NUM_PLAYERS,
    TEST_SIZE,
    TOTAL_BUDGETS,
    validate_reconstructed_dataset,
)
from experiments.sklearn_data import load_sklearn_train_test_split
from frame_ofa import (
    GameEvaluator,
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_basic_cc,
    estimate_kernel_shap,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    estimate_sdiff,
    estimate_tmc_shapley,
    iid_ofa_design,
    per_size_frame_coupled_design,
)


EXPERIMENT_ID = "wine_fresh_measured_cold_end_to_end_wallclock"
DEFAULT_SOURCE = Path(
    "results/json/"
    "wine_inside_comparison_per_size_ratio_k64_lambda1over16_"
    "3repeats_71k_1p42m.json"
)
DEFAULT_OUTPUT = Path(
    "results/json/wine_measured_wallclock_k64_lambda1over16_"
    "3repeats_71k_1p42m.json"
)
METHOD_ORDER = (
    "inside_greedy",
    "inside_orbit",
    "ofa",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)
METHOD_LABELS = {
    "inside_greedy": "INSIDE-Greedy",
    "inside_orbit": "INSIDE-Orbit",
    "ofa": "OFA",
    "cc": "CC",
    "s_diff": "S-Diff",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}
METHOD_SEED_BASES = {
    "inside_greedy": 910_001,
    "inside_orbit": 910_001,
    "ofa": 910_001,
    "cc": 1_910_001,
    "kernel_shap": 12_910_001,
    "s_diff": 42_910_001,
    "tmc_shapley": 52_910_001,
}
ORBIT_CANDIDATE_POOL = 4
IMPLEMENTATION_REVISION = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


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


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_compact_json,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _diagnostic(diagnostics: Any, name: str) -> Any:
    if isinstance(diagnostics, Mapping):
        return diagnostics[name]
    return getattr(diagnostics, name)


def _method_seed(method: str, budget_index: int, repeat: int) -> int:
    return METHOD_SEED_BASES[method] + budget_index * 100_000 + repeat


def _fresh_game_args(source: Mapping[str, Any]) -> dict[str, Any]:
    game_args, metadata = load_sklearn_train_test_split(
        "wine", test_size=TEST_SIZE, dataset_seed=DATASET_SEED
    )
    validate_reconstructed_dataset(source["dataset"], metadata)
    configured = game_args | {
        "model": "rbf_svm",
        "regularization": 1.0,
    }
    if len(configured["y_valued"]) != NUM_PLAYERS:
        raise ValueError("reconstructed Wine game has the wrong player count")
    return configured


def _manual_ratio_method(
    method: str,
    evaluator: GameEvaluator,
    *,
    inner_budget: int,
    seed: int,
    design_jobs: int,
    start_method: str,
) -> tuple[np.ndarray, int, dict[str, Any], dict[str, float]]:
    design_started = time.perf_counter()
    if method == "inside_greedy":
        design = per_size_frame_coupled_design(
            NUM_PLAYERS,
            inner_budget,
            seed=seed,
            candidate_pool=INSIDE_GREEDY_CANDIDATE_POOL,
            mean_balance=MEAN_BALANCE_LAMBDA0,
            mean_balance_mode=MEAN_BALANCE_MODE,
            design_jobs=design_jobs,
            design_start_method=start_method,
            compute_frame_diagnostics=False,
        )
    elif method == "inside_orbit":
        if inner_budget % NUM_PLAYERS:
            raise ValueError("INSIDE-Orbit requires a whole number of orbits")
        design = cyclic_orbit_frame_design(
            NUM_PLAYERS,
            num_orbits=inner_budget // NUM_PLAYERS,
            seed=seed,
            candidate_pool=ORBIT_CANDIDATE_POOL,
        )
    elif method == "ofa":
        design = iid_ofa_design(
            NUM_PLAYERS,
            inner_budget,
            seed=seed,
            compute_diagnostics=False,
        )
    else:
        raise ValueError(f"not a manual ratio method: {method}")
    design_seconds = time.perf_counter() - design_started

    if method == "inside_greedy":
        if design.method != "frame_coupled_per_size":
            raise RuntimeError("INSIDE-Greedy returned the wrong design type")
        design_diagnostics = design.diagnostics
        if design_diagnostics.get("second_moment_scope") != "per_size":
            raise RuntimeError("INSIDE-Greedy is not using a per-size frame")
        if design_diagnostics.get("mean_balance_mode") != MEAN_BALANCE_MODE:
            raise RuntimeError("INSIDE-Greedy has the wrong lambda mode")
        if not math.isclose(
            float(design_diagnostics.get("mean_balance_lambda0", math.nan)),
            MEAN_BALANCE_LAMBDA0,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError("INSIDE-Greedy has the wrong lambda0")
        coverage = design_diagnostics.get("ratio_coverage", {})
        if coverage.get("all_player_size_strata_covered") is not True:
            raise RuntimeError("INSIDE-Greedy has an uncovered ratio stratum")
    elif method == "inside_orbit":
        if design.method != "cyclic_orbit_frame":
            raise RuntimeError("INSIDE-Orbit returned the wrong design type")
    elif design.method != "iid":
        raise RuntimeError("OFA returned the wrong design type")

    boundary_started = time.perf_counter()
    raw_boundary = evaluator.evaluate(boundary_coalitions(NUM_PLAYERS))
    boundary = boundary_from_utilities(raw_boundary, NUM_PLAYERS)
    boundary_seconds = time.perf_counter() - boundary_started

    utility_started = time.perf_counter()
    utilities = evaluator.evaluate(design.coalitions)
    utility_seconds = time.perf_counter() - utility_started

    aggregation_started = time.perf_counter()
    if method == "inside_orbit":
        # This dedicated aggregation verifies exact fixed-size 1-balance.
        values = estimate_ratio_ofa(design, utilities, boundary)
        estimator_name = "balanced_ratio_ofa"
    else:
        missing = "zero" if method == "ofa" else "raise"
        values = estimate_official_ratio_ofa(
            design,
            utilities,
            boundary,
            missing=missing,
        )
        estimator_name = f"official_ratio_ofa_missing_{missing}"
    aggregation_seconds = time.perf_counter() - aggregation_started
    diagnostics = {
        "design_method": design.method,
        "design_diagnostics": copy.deepcopy(design.diagnostics),
        "boundary_utility_evaluations": BOUNDARY_CALLS,
        "inner_utility_evaluations": inner_budget,
        "utility_evaluations": inner_budget + BOUNDARY_CALLS,
        "estimator": estimator_name,
    }
    stages = {
        "design_seconds": design_seconds,
        "boundary_seconds": boundary_seconds,
        "utility_seconds": utility_seconds,
        "aggregation_seconds": aggregation_seconds,
    }
    return values, inner_budget + BOUNDARY_CALLS, diagnostics, stages


def _run_fresh_cell(
    method: str,
    *,
    game_args: dict[str, Any],
    inner_budget: int,
    total_budget: int,
    seed: int,
    jobs: int,
    chunksize: int,
    num_tasks: int,
    design_jobs: int,
    start_method: str,
) -> dict[str, Any]:
    """Run one cell with a fresh pool and return cold end-to-end timing."""
    if method not in METHOD_ORDER:
        raise ValueError(f"unknown method: {method}")
    cell_started = time.perf_counter()
    context_entered = math.nan
    estimator_started = math.nan
    estimator_stopped = math.nan
    stages: dict[str, float] = {}
    diagnostics: Any

    with GameEvaluator(
        SklearnClassificationGame,
        game_args,
        n_jobs=jobs,
        chunksize=chunksize,
        start_method=start_method,
        worker_threads=1,
    ) as evaluator:
        context_entered = time.perf_counter()
        estimator_started = time.perf_counter()
        if method in {"inside_greedy", "inside_orbit", "ofa"}:
            values, actual_calls, diagnostics, stages = _manual_ratio_method(
                method,
                evaluator,
                inner_budget=inner_budget,
                seed=seed,
                design_jobs=design_jobs,
                start_method=start_method,
            )
        elif method == "cc":
            if total_budget % 2:
                raise ValueError("CC requires an even total call budget")
            result = estimate_basic_cc(
                evaluator,
                num_players=NUM_PLAYERS,
                num_pairs=total_budget // 2,
                seed=seed,
                num_tasks=num_tasks,
            )
            values = result.values
            diagnostics = result.diagnostics
            actual_calls = int(
                _diagnostic(result.diagnostics, "utility_evaluations")
            )
        elif method == "s_diff":
            result = estimate_sdiff(
                evaluator,
                NUM_PLAYERS,
                total_budget,
                seed,
                num_tasks=num_tasks,
            )
            values = result.values
            diagnostics = result.diagnostics
            actual_calls = int(
                _diagnostic(result.diagnostics, "utility_evaluations")
            )
        elif method == "kernel_shap":
            result = estimate_kernel_shap(
                evaluator,
                NUM_PLAYERS,
                total_budget,
                seed,
                num_tasks=num_tasks,
                ridge=1e-8,
            )
            values = result.values
            diagnostics = result.diagnostics
            actual_calls = int(
                _diagnostic(result.diagnostics, "utility_evaluations")
            )
        else:
            result = estimate_tmc_shapley(
                evaluator,
                NUM_PLAYERS,
                total_budget,
                seed,
                num_tasks=num_tasks,
            )
            values = result.values
            diagnostics = result.diagnostics
            actual_calls = int(
                _diagnostic(result.diagnostics, "utility_evaluations")
            )
        estimator_stopped = time.perf_counter()
    cell_stopped = time.perf_counter()

    estimate = np.asarray(values, dtype=np.float64)
    if estimate.shape != (NUM_PLAYERS,) or not np.all(np.isfinite(estimate)):
        raise RuntimeError(f"{method} returned an invalid Shapley vector")
    if actual_calls < 1 or actual_calls > total_budget:
        raise RuntimeError(f"{method} returned invalid utility-call accounting")
    if method != "tmc_shapley" and actual_calls != total_budget:
        raise RuntimeError(f"{method} did not consume its complete call budget")

    estimator_seconds = estimator_stopped - estimator_started
    cold_seconds = cell_stopped - cell_started
    shutdown_seconds = cell_stopped - estimator_stopped
    context_seconds = context_entered - cell_started
    stage_sum = sum(stages.values())
    if stages and stage_sum > estimator_seconds + 1e-8:
        raise RuntimeError("manual stage times exceed estimator wall time")
    return {
        "estimate": estimate.tolist(),
        "actual_utility_calls": int(actual_calls),
        "unused_calls": int(total_budget - actual_calls),
        "timing": {
            "cold_end_to_end_seconds": float(cold_seconds),
            "context_construction_seconds": float(context_seconds),
            "estimator_seconds": float(estimator_seconds),
            "pool_shutdown_seconds": float(shutdown_seconds),
            "unclassified_estimator_seconds": float(
                max(0.0, estimator_seconds - stage_sum)
            ),
            **{key: float(value) for key, value in stages.items()},
        },
        "diagnostics": _compact_json(diagnostics),
    }


def _source_contract(source: Mapping[str, Any]) -> None:
    configuration = source.get("configuration", {})
    dataset = source.get("dataset", {})
    game = source.get("game", {})
    if source.get("status") != "complete":
        raise ValueError("source report is not complete")
    if source.get("dataset", {}).get("dataset") != "wine":
        raise ValueError("source report is not Wine")
    if int(dataset.get("dataset_seed", -1)) != DATASET_SEED:
        raise ValueError("source report has the wrong dataset seed")
    if not math.isclose(
        float(dataset.get("test_size", math.nan)),
        TEST_SIZE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("source report has the wrong test size")
    if int(configuration.get("num_players", -1)) != NUM_PLAYERS:
        raise ValueError("source report has the wrong player count")
    if tuple(configuration.get("inner_budgets", ())) != INNER_BUDGETS:
        raise ValueError("source report has the wrong budgets")
    if tuple(configuration.get("total_call_budgets", ())) != TOTAL_BUDGETS:
        raise ValueError("source report has the wrong total call budgets")
    if game.get("model") != (
        "sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='scale')"
    ):
        raise ValueError("source report has the wrong model")
    if game.get("utility") != "fixed-test classification accuracy":
        raise ValueError("source report has the wrong utility")
    truth = np.asarray(source.get("ground_truth", {}).get("values"))
    truth_se = np.asarray(source.get("ground_truth", {}).get("standard_errors"))
    if truth.shape != (NUM_PLAYERS,) or truth_se.shape != (NUM_PLAYERS,):
        raise ValueError("source report has an invalid ground truth")


def _initial_report(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    methods: Sequence[str],
    repeats: int,
    jobs: int,
    chunksize: int,
    num_tasks: int,
    design_jobs: int,
    start_method: str,
) -> dict[str, Any]:
    return {
        "status": "running",
        "experiment": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "dataset": "wine",
            "num_players": NUM_PLAYERS,
            "test_size": TEST_SIZE,
            "dataset_seed": DATASET_SEED,
            "model": "sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='scale')",
            "utility": "fixed-test classification accuracy",
            "methods": list(methods),
            "method_labels": {
                method: METHOD_LABELS[method] for method in methods
            },
            "method_seed_bases": {
                method: METHOD_SEED_BASES[method] for method in methods
            },
            "budget_multipliers": list(BUDGET_MULTIPLIERS),
            "inner_budgets": list(INNER_BUDGETS),
            "total_call_budgets": list(TOTAL_BUDGETS),
            "repeats": repeats,
            "jobs": jobs,
            "chunksize": chunksize,
            "num_tasks": num_tasks,
            "design_jobs": design_jobs,
            "start_method": start_method,
            "worker_threads": 1,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "inside_greedy_candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
            "inside_greedy_lambda0": MEAN_BALANCE_LAMBDA0,
            "inside_greedy_mean_balance_mode": MEAN_BALANCE_MODE,
            "inside_greedy_second_moment_scope": "per_size",
            "inside_greedy_parallel_fixed_slice_rng": (
                "independent SeedSequence substreams"
            ),
            "inside_greedy_compute_frame_diagnostics": False,
            "inside_orbit_candidate_pool": ORBIT_CANDIDATE_POOL,
            "timing_protocol": {
                "kind": "fresh cold end-to-end wall clock per cell",
                "timer_start": "before method sampling/design",
                "timer_stop": (
                    "after final Shapley vector and fresh utility-pool shutdown"
                ),
                "included": [
                    "sampling/design",
                    "fresh process lazy startup",
                    "all physical utility evaluations",
                    "aggregation/linear solve/differential recovery",
                    "process-pool shutdown",
                ],
                "excluded": [
                    "dataset loading in the parent process",
                    "accuracy-summary construction",
                    "JSON serialization",
                    "plotting",
                ],
                "cross_cell_reuse": "none",
                "ground_truth_runtime_included": False,
            },
            "thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
            "host": platform.node(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
        "source": {
            "reference_report": str(source_path),
            "reference_report_sha256": _sha256(source_path),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "reused_items": [
                "fixed train/test split metadata",
                "audited high-budget Monte Carlo reference vector",
                "reference standard errors",
            ],
            "method_estimates_reused": False,
            "method_timings_reused": False,
            "coalition_utilities_reused": False,
        },
        "dataset": copy.deepcopy(source["dataset"]),
        "ground_truth": copy.deepcopy(source["ground_truth"]),
        "boundary_target": copy.deepcopy(source["boundary"]),
        "cells": [],
        "results": [],
    }


def _validate_resume(
    report: Mapping[str, Any],
    *,
    source_path: Path,
    methods: Sequence[str],
    repeats: int,
    jobs: int,
    chunksize: int,
    num_tasks: int,
    design_jobs: int,
    start_method: str,
) -> None:
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("existing output has a different experiment id")
    configuration = report.get("configuration", {})
    expected = {
        "methods": list(methods),
        "repeats": repeats,
        "jobs": jobs,
        "chunksize": chunksize,
        "num_tasks": num_tasks,
        "design_jobs": design_jobs,
        "start_method": start_method,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "inside_greedy_candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
        "inside_greedy_lambda0": MEAN_BALANCE_LAMBDA0,
        "inside_greedy_mean_balance_mode": MEAN_BALANCE_MODE,
        "inside_greedy_second_moment_scope": "per_size",
        "inside_greedy_compute_frame_diagnostics": False,
        "inside_orbit_candidate_pool": ORBIT_CANDIDATE_POOL,
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"existing output uses a different {key}")
    observed_hash = report.get("source", {}).get("reference_report_sha256")
    if observed_hash != _sha256(source_path):
        raise ValueError("existing output uses a different reference report")
    observed_runner = report.get("source", {}).get("runner_sha256")
    if observed_runner != _sha256(Path(__file__).resolve()):
        raise ValueError("runner source changed since the checkpoint was created")


def _validate_cells(
    cells: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str],
    repeats: int,
) -> None:
    keys: set[tuple[str, int, int]] = set()
    for cell in cells:
        method = str(cell.get("method"))
        budget_index = int(cell.get("budget_index", -1))
        repeat = int(cell.get("repeat", -1))
        key = (method, budget_index, repeat)
        if key in keys:
            raise ValueError(f"duplicate completed cell: {key}")
        keys.add(key)
        if method not in methods:
            raise ValueError(f"completed cell has an unconfigured method: {method}")
        if budget_index not in range(len(INNER_BUDGETS)):
            raise ValueError("completed cell has an invalid budget index")
        if repeat not in range(repeats):
            raise ValueError("completed cell has an invalid repeat index")
        if int(cell.get("seed", -1)) != _method_seed(
            method, budget_index, repeat
        ):
            raise ValueError("completed cell has the wrong method seed")
        if int(cell.get("inner_utility_calls", -1)) != INNER_BUDGETS[
            budget_index
        ]:
            raise ValueError("completed cell has the wrong inner budget")
        target = TOTAL_BUDGETS[budget_index]
        if int(cell.get("target_total_utility_calls", -1)) != target:
            raise ValueError("completed cell has the wrong total budget")
        actual = int(cell.get("actual_utility_calls", -1))
        unused = int(cell.get("unused_calls", -1))
        if actual < 1 or actual + unused != target:
            raise ValueError("completed cell has invalid call accounting")
        if method != "tmc_shapley" and actual != target:
            raise ValueError("a non-TMC cell did not use its whole budget")
        estimate = np.asarray(cell.get("estimate"), dtype=np.float64)
        if estimate.shape != (NUM_PLAYERS,) or not np.all(np.isfinite(estimate)):
            raise ValueError("completed cell has an invalid estimate")
        timing = cell.get("timing", {})
        components = [
            float(timing.get(name, math.nan))
            for name in (
                "cold_end_to_end_seconds",
                "context_construction_seconds",
                "estimator_seconds",
                "pool_shutdown_seconds",
                "unclassified_estimator_seconds",
            )
        ]
        if not all(math.isfinite(value) and value >= 0.0 for value in components):
            raise ValueError("completed cell has invalid timing values")
        cold, context, estimator, shutdown, _ = components
        if not math.isclose(
            cold,
            context + estimator + shutdown,
            rel_tol=0.0,
            abs_tol=5e-6,
        ):
            raise ValueError("completed cell timing decomposition is inconsistent")


def _summaries(
    report: Mapping[str, Any],
    *,
    methods: Sequence[str],
    repeats: int,
) -> list[dict[str, Any]]:
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    truth_se = np.asarray(
        report["ground_truth"]["standard_errors"], dtype=np.float64
    )
    efficiency_target = float(report["boundary_target"]["efficiency_target"])
    indexed = {
        (cell["method"], int(cell["budget_index"]), int(cell["repeat"])): cell
        for cell in report.get("cells", [])
    }
    rows: list[dict[str, Any]] = []
    for budget_index, (inner_budget, total_budget) in enumerate(
        zip(INNER_BUDGETS, TOTAL_BUDGETS, strict=True)
    ):
        row: dict[str, Any] = {
            "budget_index": budget_index,
            "inner_utility_calls": inner_budget,
            "target_total_utility_calls": total_budget,
            "methods": {},
        }
        for method_index, method in enumerate(methods):
            cells = [
                indexed.get((method, budget_index, repeat))
                for repeat in range(repeats)
            ]
            if any(cell is None for cell in cells):
                continue
            complete_cells = [cell for cell in cells if cell is not None]
            estimates = np.asarray(
                [cell["estimate"] for cell in complete_cells],
                dtype=np.float64,
            )
            summary = summarize_method(
                estimates,
                truth,
                truth_se,
                efficiency_target,
                bootstrap_seed=8_910_001 + 100 * budget_index + method_index,
            )
            # ``summarize_method`` intentionally uses the sample standard
            # deviation (ddof=1).  A one-repeat canary has no sample standard
            # deviation, so record zero rather than emitting non-JSON NaN.
            if repeats == 1:
                summary["std_repeat_rmse"] = 0.0
            actual_calls = [
                int(cell["actual_utility_calls"]) for cell in complete_cells
            ]
            cold = [
                float(cell["timing"]["cold_end_to_end_seconds"])
                for cell in complete_cells
            ]
            estimator = [
                float(cell["timing"]["estimator_seconds"])
                for cell in complete_cells
            ]
            summary.update(
                {
                    "label": METHOD_LABELS[method],
                    "actual_utility_calls_by_repeat": actual_calls,
                    "mean_actual_utility_calls": float(np.mean(actual_calls)),
                    "cold_end_to_end_seconds_by_repeat": cold,
                    "mean_cold_end_to_end_seconds": float(np.mean(cold)),
                    "median_cold_end_to_end_seconds": float(np.median(cold)),
                    "cold_end_to_end_seconds_range": [
                        float(np.min(cold)),
                        float(np.max(cold)),
                    ],
                    "estimator_seconds_by_repeat": estimator,
                    "mean_estimator_seconds": float(np.mean(estimator)),
                    "timing_stages_by_repeat": [
                        copy.deepcopy(cell["timing"])
                        for cell in complete_cells
                    ],
                }
            )
            row["methods"][method] = summary
        rows.append(row)
    return rows


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.start_method not in mp.get_all_start_methods():
        raise ValueError(f"unsupported start method: {args.start_method}")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    _source_contract(source)
    methods = tuple(args.methods)
    game_args = _fresh_game_args(source)

    if args.output.exists():
        if not args.resume:
            raise ValueError(
                f"output already exists; use --resume or a new path: {args.output}"
            )
        report = json.loads(args.output.read_text(encoding="utf-8"))
        _validate_resume(
            report,
            source_path=args.source,
            methods=methods,
            repeats=args.repeats,
            jobs=args.jobs,
            chunksize=args.chunksize,
            num_tasks=args.num_tasks,
            design_jobs=args.design_jobs,
            start_method=args.start_method,
        )
        _validate_cells(
            report.get("cells", []), methods=methods, repeats=args.repeats
        )
    else:
        report = _initial_report(
            source,
            source_path=args.source,
            methods=methods,
            repeats=args.repeats,
            jobs=args.jobs,
            chunksize=args.chunksize,
            num_tasks=args.num_tasks,
            design_jobs=args.design_jobs,
            start_method=args.start_method,
        )
        _atomic_write(args.output, report)

    completed = {
        (cell["method"], int(cell["budget_index"]), int(cell["repeat"]))
        for cell in report.get("cells", [])
    }
    total_cells = len(methods) * len(INNER_BUDGETS) * args.repeats
    run_started = time.perf_counter()
    new_cells = 0
    for budget_index, (inner_budget, total_budget) in enumerate(
        zip(INNER_BUDGETS, TOTAL_BUDGETS, strict=True)
    ):
        for repeat in range(args.repeats):
            # Rotate method order to reduce a fixed thermal/order advantage.
            shift = (budget_index * args.repeats + repeat) % len(methods)
            schedule = methods[shift:] + methods[:shift]
            for method in schedule:
                key = (method, budget_index, repeat)
                if key in completed:
                    continue
                seed = _method_seed(method, budget_index, repeat)
                print(
                    f"[{len(completed) + 1}/{total_cells}] {method}, "
                    f"budget={total_budget:,}, repeat={repeat + 1}/"
                    f"{args.repeats}, seed={seed}",
                    flush=True,
                )
                cell = _run_fresh_cell(
                    method,
                    game_args=game_args,
                    inner_budget=inner_budget,
                    total_budget=total_budget,
                    seed=seed,
                    jobs=args.jobs,
                    chunksize=args.chunksize,
                    num_tasks=args.num_tasks,
                    design_jobs=args.design_jobs,
                    start_method=args.start_method,
                )
                cell.update(
                    {
                        "method": method,
                        "label": METHOD_LABELS[method],
                        "budget_index": budget_index,
                        "repeat": repeat,
                        "seed": seed,
                        "inner_utility_calls": inner_budget,
                        "target_total_utility_calls": total_budget,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                report["cells"].append(cell)
                completed.add(key)
                new_cells += 1
                report["cells"].sort(
                    key=lambda item: (
                        int(item["budget_index"]),
                        int(item["repeat"]),
                        METHOD_ORDER.index(item["method"]),
                    )
                )
                _validate_cells(
                    report["cells"], methods=methods, repeats=args.repeats
                )
                report["results"] = _summaries(
                    report, methods=methods, repeats=args.repeats
                )
                report["progress"] = {
                    "completed_cells": len(completed),
                    "total_cells": total_cells,
                    "fraction": len(completed) / total_cells,
                    "new_cells_this_invocation": new_cells,
                    "invocation_wall_seconds": time.perf_counter() - run_started,
                }
                _atomic_write(args.output, report)
                print(
                    "  calls="
                    f"{cell['actual_utility_calls']:,}, cold wall="
                    f"{cell['timing']['cold_end_to_end_seconds']:.3f}s, "
                    "estimator="
                    f"{cell['timing']['estimator_seconds']:.3f}s",
                    flush=True,
                )

    report["status"] = "complete"
    report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["results"] = _summaries(
        report, methods=methods, repeats=args.repeats
    )
    report["progress"] = {
        "completed_cells": len(completed),
        "total_cells": total_cells,
        "fraction": 1.0,
        "new_cells_this_invocation": new_cells,
        "invocation_wall_seconds": time.perf_counter() - run_started,
    }
    _validate_cells(report["cells"], methods=methods, repeats=args.repeats)
    if len(report["cells"]) != total_cells:
        raise RuntimeError("benchmark ended with missing cells")
    _atomic_write(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--methods", nargs="+", choices=METHOD_ORDER, default=list(METHOD_ORDER)
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=128)
    parser.add_argument("--chunksize", type=int, default=512)
    parser.add_argument("--num-tasks", type=int, default=128)
    parser.add_argument("--design-jobs", type=int, default=128)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if not args.source.exists():
        raise ValueError(f"source report does not exist: {args.source}")
    if len(set(args.methods)) != len(args.methods):
        raise ValueError("--methods must be distinct")
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    if min(args.jobs, args.chunksize, args.num_tasks, args.design_jobs) < 1:
        raise ValueError("parallelism and chunksize arguments must be positive")
    return args


def main() -> None:
    args = parse_args()
    report = run_experiment(args)
    print(
        f"Wine measured wall-clock benchmark complete: {args.output} "
        f"({len(report['cells'])} cells)",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Run the canonical eight-method INSIDE comparison on Breast Cancer.

The expensive 800,000-pair Monte Carlo reference and three compatible curves
are reduced from the immutable ten-repeat Cancer report.  INSIDE-Greedy, CC,
S-Diff, KernelSHAP, and TMC-Shapley are evaluated afresh for three repeats.

The fine-grained Greedy objective is independent across fixed-size Boolean
slices.  This runner uses deterministic per-slice seed substreams and solves
those slices in parallel; worker count changes scheduling, not the resulting
coalitions.  Atomic method/repeat shards make the long run resumable after
every completed budget cell.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from experiments.iris_data_valuation import summarize_method
from experiments.iris_sklearn_game import SklearnClassificationGame
from experiments.sklearn_data import load_sklearn_train_test_split
from experiments.validate_full_train_report import (
    validate_report as validate_full_train_report,
)
from frame_ofa import (
    GameEvaluator,
    boundary_coalitions,
    boundary_from_utilities,
    estimate_basic_cc,
    estimate_official_ratio_ofa,
    per_size_frame_coupled_design,
)


EXPERIMENT_ID = (
    "cancer_inside_greedy_orbit_baseline_comparison_"
    "per_size_ratio_k64_lambda1over16"
)
DATASET = "cancer"
NUM_PLAYERS = 455
NUM_TEST = 114
REPEATS = 3
SOURCE_REPEATS = 10
SOURCE_ORBIT_CANDIDATE_POOL = 4
INSIDE_GREEDY_CANDIDATE_POOL = 64
MEAN_BALANCE_MODE = "normalized"
MEAN_BALANCE_LAMBDA0 = 1.0 / 16.0
DATASET_SEED = 2024
TEST_SIZE = 0.2
BOUNDARY_CALLS = 2 * NUM_PLAYERS + 2
BUDGET_MULTIPLIERS = (500, 1000, 2000, 5000, 10000)
INNER_BUDGETS = tuple(NUM_PLAYERS * value for value in BUDGET_MULTIPLIERS)
TOTAL_BUDGETS = tuple(value + BOUNDARY_CALLS for value in INNER_BUDGETS)
METHOD_SEED = 910_001
GROUND_TRUTH_SEED = 730_001
NUM_BASELINE_TASKS = 128

SOURCE_METHOD_MAPPING = {
    "inside_orbit": "frame_orbit_ratio",
    "ofa_iid_linear": "iid_linear_ofa",
    "ofa_iid_ratio": "official_ofa_fixed_ratio",
}
# Method indices used by the immutable source runner's bootstrap seeds.
SOURCE_BOOTSTRAP_METHOD_INDEX = {
    "official_ofa_fixed_ratio": 0,
    "iid_linear_ofa": 1,
    "frame_coupled_linear": 2,
    "frame_orbit_ratio": 3,
}
FRESH_METHODS = (
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
    "inside_greedy",
)
METHOD_ORDER = (
    "inside_greedy",
    "inside_orbit",
    "ofa_iid_linear",
    "ofa_iid_ratio",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)
METHOD_LABELS = {
    "inside_greedy": "INSIDE-Greedy",
    "inside_orbit": "INSIDE-Orbit",
    "ofa_iid_linear": "OFA linear (IID)",
    "ofa_iid_ratio": "OFA ratio (IID)",
    "cc": "CC",
    "s_diff": "S-Diff",
    "kernel_shap": "KernelSHAP",
    "tmc_shapley": "TMC-Shapley",
}
# Match the method-specific seed streams used by the current Wine baselines.
FRESH_BASE_SEEDS = {
    "cc": 1_910_001,
    "kernel_shap": 12_910_001,
    "s_diff": 42_910_001,
    "tmc_shapley": 52_910_001,
    "inside_greedy": METHOD_SEED,
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"cannot JSON-encode {type(value).__name__}")


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_vector(value: Any, length: int, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-{length} vector")
    return vector


def _diagnostic_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        result = asdict(value)
    elif isinstance(value, Mapping):
        result = dict(value)
    else:
        result = dict(vars(value))
    # This dense n-by-n CC state is useful while estimating but not in a
    # report; all of its relevant coverage summaries are retained separately.
    result.pop("stratum_counts", None)
    return result


def _method_seed(method: str, repeat: int, budget_index: int) -> int:
    return FRESH_BASE_SEEDS[method] + budget_index * 100_000 + repeat


def validate_source_report(report: Mapping[str, Any]) -> None:
    audit = validate_full_train_report(
        report,
        expected_dataset=DATASET,
        expected_num_players=NUM_PLAYERS,
        expected_repeats=SOURCE_REPEATS,
        expected_gt_pairs=800_000,
        expected_jobs=128,
        expected_inner_budgets=INNER_BUDGETS,
        expected_budget_multipliers=BUDGET_MULTIPLIERS,
    )
    if not audit["passed"]:
        raise ValueError(
            "immutable Cancer source failed its full audit: "
            + "; ".join(audit["errors"][:8])
        )
    configuration = report["configuration"]
    dataset = report["dataset"]
    ground_truth = report["ground_truth"]
    if dataset.get("dataset") != DATASET:
        raise ValueError("source is not the canonical Cancer dataset")
    if int(dataset.get("dataset_seed", -1)) != DATASET_SEED:
        raise ValueError("source Cancer split seed is wrong")
    if not math.isclose(
        float(dataset.get("test_size", math.nan)),
        TEST_SIZE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("source Cancer test size is wrong")
    if int(configuration.get("n_test", -1)) != NUM_TEST:
        raise ValueError("source Cancer test count is wrong")
    if int(configuration.get("candidate_pool", -1)) != 4:
        raise ValueError("source orbit candidate pool is not K=4")
    if "SVC(C=1.0, kernel='rbf', gamma='scale')" not in str(
        configuration.get("model")
    ):
        raise ValueError("source model is not the canonical RBF SVC")
    _finite_vector(ground_truth.get("values"), NUM_PLAYERS, name="truth")
    _finite_vector(
        ground_truth.get("standard_errors"),
        NUM_PLAYERS,
        name="truth standard errors",
    )
    if int(ground_truth.get("independent_pair_units", -1)) != 800_000:
        raise ValueError("source ground truth does not use 800,000 pairs")
    for inner in INNER_BUDGETS:
        methods = report["results_by_inner_budget"][str(inner)]["methods"]
        for source_method in SOURCE_METHOD_MAPPING.values():
            estimates = np.asarray(
                methods[source_method].get("estimates"), dtype=np.float64
            )
            if estimates.shape != (SOURCE_REPEATS, NUM_PLAYERS):
                raise ValueError(
                    f"source {source_method} has the wrong estimate shape"
                )
            if not np.all(np.isfinite(estimates)):
                raise ValueError(f"source {source_method} is non-finite")


def _validate_ground_truth_cache(
    path: Path, source_ground_truth: Mapping[str, Any]
) -> str:
    if not path.exists():
        raise ValueError(f"ground-truth cache does not exist: {path}")
    with np.load(path, allow_pickle=False) as cache:
        required = {
            "cache_key",
            "diagnostics",
            "values",
            "standard_errors",
            "rmse_standard_error",
            "simultaneous_half_widths",
            "max_simultaneous_half_width",
            "half_split_rmse",
        }
        missing = required - set(cache.files)
        if missing:
            raise ValueError(
                "ground-truth cache is missing fields: "
                + ", ".join(sorted(missing))
            )
        try:
            cache_key = json.loads(str(cache["cache_key"].item()))
            cache_diagnostics = json.loads(
                str(cache["diagnostics"].item())
            )
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "ground-truth cache metadata is malformed"
            ) from error
        expected_cache_key = {
            "C": 1.0,
            "dataset": DATASET,
            "dataset_seed": DATASET_SEED,
            "gamma": "scale",
            "kernel": "rbf",
            "model": "sklearn.svm.SVC",
            "num_pairs": 800_000,
            "num_players": NUM_PLAYERS,
            "pairing": "permutation_reverse",
            "seed": GROUND_TRUTH_SEED,
            "test_size": TEST_SIZE,
        }
        if cache_key != expected_cache_key:
            raise ValueError(
                "ground-truth cache identity is not the canonical Cancer "
                "RBF-SVM reference"
            )
        if not isinstance(cache_diagnostics, Mapping):
            raise ValueError("ground-truth cache diagnostics are malformed")
        for key in (
            "values",
            "standard_errors",
            "simultaneous_half_widths",
            "rmse_standard_error",
            "max_simultaneous_half_width",
            "half_split_rmse",
        ):
            if not np.array_equal(
                np.asarray(cache[key]),
                np.asarray(source_ground_truth[key]),
            ):
                raise ValueError(
                    f"ground-truth cache disagrees with source on {key}"
                )
        for key in (
            "boundary_reuse_saved_calls",
            "conceptual_internal_prefix_calls",
            "max_worker_tasks_per_outer_block",
            "peak_forward_permutation_bytes",
            "physical_internal_prefix_calls",
            "wall_seconds",
        ):
            if cache_diagnostics.get(key) != source_ground_truth.get(key):
                raise ValueError(
                    "ground-truth cache diagnostics disagree with source "
                    f"on {key}"
                )
    return _sha256(path)


def _validate_boundary_reconstruction(
    report: Mapping[str, Any],
) -> tuple[list[float], float]:
    reconstruction = report.get("boundary_reconstruction")
    if not isinstance(reconstruction, Mapping):
        raise ValueError("resume report lacks boundary reconstruction")
    utilities = _finite_vector(
        reconstruction.get("utilities"),
        BOUNDARY_CALLS,
        name="resume boundary utilities",
    )
    expected_hash = hashlib.sha256(utilities.tobytes()).hexdigest()
    if reconstruction.get("sha256_float64_bytes") != expected_hash:
        raise ValueError("resume boundary reconstruction hash differs")
    if int(reconstruction.get("physical_utility_calls", -1)) != BOUNDARY_CALLS:
        raise ValueError("resume boundary physical-call count is wrong")
    wall_seconds = float(reconstruction.get("wall_seconds", math.nan))
    if not math.isfinite(wall_seconds) or wall_seconds < 0.0:
        raise ValueError("resume boundary wall time is invalid")
    if reconstruction.get("source_empty_full_match") is not True:
        raise ValueError("resume boundary was not matched to the source")
    return utilities.tolist(), wall_seconds


def validate_reconstructed_dataset(
    stored: Mapping[str, Any], reconstructed: Mapping[str, Any]
) -> None:
    for key, stored_value in stored.items():
        if key not in reconstructed:
            raise ValueError(f"reconstructed dataset is missing {key}")
        actual = reconstructed[key]
        if key in {"standardization_mean", "standardization_std"}:
            if not np.allclose(
                np.asarray(stored_value, dtype=np.float64),
                np.asarray(actual, dtype=np.float64),
                rtol=0.0,
                atol=1e-14,
            ):
                raise ValueError(f"reconstructed dataset differs on {key}")
        elif stored_value != actual:
            raise ValueError(f"reconstructed dataset differs on {key}")


def _shard_key(
    source_sha256: str,
    method: str,
    repeat: int,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "experiment": EXPERIMENT_ID,
        "source_sha256": source_sha256,
        "method": method,
        "repeat": repeat,
        "inner_budgets": list(INNER_BUDGETS),
        "base_seed": FRESH_BASE_SEEDS[method],
        "num_baseline_tasks": NUM_BASELINE_TASKS,
    }
    if method == "inside_greedy":
        identity.update(
            {
                "design": "per_size_frame_coupled_design",
                "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                "mean_balance_mode": MEAN_BALANCE_MODE,
                "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                "fixed_slice_rng_partitioning": (
                    "independent_seedsequence_substreams"
                ),
                "second_moment_scope": "per_size",
                "estimator": "estimate_official_ratio_ofa(missing='raise')",
            }
        )
    return identity


def _validate_fresh_row(row: Mapping[str, Any]) -> None:
    method = str(row.get("method"))
    if method not in FRESH_METHODS:
        raise ValueError(f"unknown fresh method {method!r}")
    repeat = int(row.get("repeat", -1))
    budget_index = int(row.get("budget_index", -1))
    if repeat not in range(REPEATS) or budget_index not in range(5):
        raise ValueError("fresh row has an invalid repeat/budget")
    total = TOTAL_BUDGETS[budget_index]
    inner = INNER_BUDGETS[budget_index]
    if int(row.get("seed", -1)) != _method_seed(
        method, repeat, budget_index
    ):
        raise ValueError("fresh row seed is wrong")
    if int(row.get("target_total_calls", -1)) != total:
        raise ValueError("fresh row target-call cap is wrong")
    actual = int(row.get("actual_total_calls", -1))
    if actual < 0 or actual > total:
        raise ValueError("fresh row actual-call count is invalid")
    if int(row.get("unused_calls", -1)) != total - actual:
        raise ValueError("fresh row unused-call count is wrong")
    if method != "tmc_shapley" and actual != total:
        raise ValueError(f"{method} must spend its full call cap")
    _finite_vector(row.get("estimate"), NUM_PLAYERS, name=f"{method} estimate")
    elapsed = float(row.get("elapsed_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("fresh row elapsed time is invalid")
    diagnostics = row.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("fresh row diagnostics are missing")
    if int(diagnostics.get("utility_evaluations", actual)) != actual:
        raise ValueError("fresh-row diagnostic calls disagree")
    if method == "inside_greedy":
        if int(row.get("inner_utility_calls", -1)) != inner:
            raise ValueError("INSIDE-Greedy inner calls are wrong")
        design = row.get("design_diagnostics")
        coverage = row.get("coverage")
        if not isinstance(design, Mapping) or not isinstance(
            coverage, Mapping
        ):
            raise ValueError("INSIDE-Greedy diagnostics are incomplete")
        if design.get("second_moment_scope") != "per_size":
            raise ValueError("INSIDE-Greedy scope is not per-size")
        if design.get("fixed_slice_rng_partitioning") != (
            "independent_seedsequence_substreams"
        ):
            raise ValueError("INSIDE-Greedy fixed-slice RNG audit is wrong")
        expected = MEAN_BALANCE_LAMBDA0 * (NUM_PLAYERS - 2) / (
            NUM_PLAYERS - 1
        )
        if not math.isclose(
            float(design.get("mean_balance_effective_raw", math.nan)),
            expected,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("INSIDE-Greedy effective lambda is wrong")
        if coverage.get("all_player_size_strata_covered") is not True:
            raise ValueError("INSIDE-Greedy ratio strata are incomplete")


def _merge_rows(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for group in groups:
        for raw in group:
            row = copy.deepcopy(dict(raw))
            _validate_fresh_row(row)
            key = (
                str(row["method"]),
                int(row["repeat"]),
                int(row["budget_index"]),
            )
            previous = indexed.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting checkpoint rows for {key}")
            indexed[key] = row
    return [indexed[key] for key in sorted(indexed)]


def _load_shard(
    path: Path,
    *,
    source_sha256: str,
    method: str,
    repeat: int,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cache_key") != _shard_key(
        source_sha256, method, repeat
    ):
        raise ValueError(f"checkpoint identity differs: {path}")
    rows = _merge_rows(payload.get("rows", []))
    if any(
        row["method"] != method or int(row["repeat"]) != repeat
        for row in rows
    ):
        raise ValueError(f"checkpoint contains another method/repeat: {path}")
    return rows


def _evaluate_baseline(
    method: str,
    evaluator: GameEvaluator,
    *,
    total_call_budget: int,
    seed: int,
) -> Any:
    if method == "cc":
        if total_call_budget % 2:
            raise ValueError("CC requires an even total call cap")
        return estimate_basic_cc(
            evaluator,
            NUM_PLAYERS,
            total_call_budget // 2,
            seed,
            num_tasks=NUM_BASELINE_TASKS,
        )
    if method == "s_diff":
        from frame_ofa.stratified_differential import estimate_sdiff

        return estimate_sdiff(
            evaluator,
            NUM_PLAYERS,
            total_call_budget,
            seed,
            num_tasks=NUM_BASELINE_TASKS,
        )
    if method == "kernel_shap":
        from frame_ofa.regression_baselines import estimate_kernel_shap

        return estimate_kernel_shap(
            evaluator,
            NUM_PLAYERS,
            total_call_budget,
            seed,
            num_tasks=NUM_BASELINE_TASKS,
            ridge=1e-8,
        )
    if method == "tmc_shapley":
        from frame_ofa.tmc import estimate_tmc_shapley

        return estimate_tmc_shapley(
            evaluator,
            NUM_PLAYERS,
            total_call_budget,
            seed,
            num_tasks=NUM_BASELINE_TASKS,
        )
    raise ValueError(f"unsupported baseline method: {method}")


def _run_fresh_repeat(
    method: str,
    repeat: int,
    game_args: dict[str, Any],
    boundary_utilities: Sequence[float],
    source_sha256: str,
    shard_path: Path,
    initial_rows: Sequence[Mapping[str, Any]],
    utility_jobs: int,
    design_jobs: int,
    chunksize: int,
    start_method: str,
    design_start_method: str,
) -> dict[str, Any]:
    rows = _merge_rows(
        initial_rows,
        _load_shard(
            shard_path,
            source_sha256=source_sha256,
            method=method,
            repeat=repeat,
        ),
    )
    completed = {int(row["budget_index"]) for row in rows}
    boundary = boundary_from_utilities(
        np.asarray(boundary_utilities, dtype=np.float64), NUM_PLAYERS
    )

    def save(row: Mapping[str, Any]) -> None:
        nonlocal rows
        _validate_fresh_row(row)
        rows = _merge_rows(rows, [row])
        _atomic_write(
            shard_path,
            {
                "cache_key": _shard_key(
                    source_sha256, method, repeat
                ),
                "rows": rows,
            },
        )

    if method == "inside_greedy":
        for budget_index, (inner, total) in enumerate(
            zip(INNER_BUDGETS, TOTAL_BUDGETS)
        ):
            if budget_index in completed:
                continue
            seed = _method_seed(method, repeat, budget_index)
            elapsed_start = time.perf_counter()
            design_start = time.perf_counter()
            design = per_size_frame_coupled_design(
                NUM_PLAYERS,
                inner,
                seed=seed,
                candidate_pool=INSIDE_GREEDY_CANDIDATE_POOL,
                mean_balance=MEAN_BALANCE_LAMBDA0,
                mean_balance_mode=MEAN_BALANCE_MODE,
                design_jobs=design_jobs,
                design_start_method=design_start_method,
                compute_frame_diagnostics=False,
            )
            design_seconds = time.perf_counter() - design_start
            coverage = copy.deepcopy(
                design.diagnostics.get("ratio_coverage")
            )
            if not isinstance(coverage, Mapping) or not coverage.get(
                "all_player_size_strata_covered"
            ):
                raise ValueError(
                    "INSIDE-Greedy lacks complete player/size coverage"
                )
            utility_start = time.perf_counter()
            with GameEvaluator(
                SklearnClassificationGame,
                game_args,
                n_jobs=utility_jobs,
                chunksize=chunksize,
                start_method=start_method,
                worker_threads=1,
            ) as evaluator:
                utilities = evaluator.evaluate(design.coalitions)
            estimate = estimate_official_ratio_ofa(
                design, utilities, boundary, missing="raise"
            )
            utility_seconds = time.perf_counter() - utility_start
            diagnostic = {
                "utility_evaluations": total,
                "target_call_budget": total,
                "unused_calls": 0,
                "boundary_utility_evaluations": BOUNDARY_CALLS,
                "inner_utility_evaluations": inner,
                "design_method": "frame_coupled_per_size",
                "second_moment_scope": "per_size",
                "estimator": "ofa_conditional_mean_ratio_missing_raise",
                "official_ratio_missing_policy": "raise",
                "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                "mean_balance_mode": MEAN_BALANCE_MODE,
                "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                "mean_balance_effective_raw": design.diagnostics[
                    "mean_balance_effective_raw"
                ],
                "coverage": coverage,
                "design_diagnostics": copy.deepcopy(design.diagnostics),
                "design_seconds": design_seconds,
                "utility_and_estimation_seconds": utility_seconds,
            }
            row = {
                "method": method,
                "repeat": repeat,
                "budget_index": budget_index,
                "seed": seed,
                "inner_utility_calls": inner,
                "target_total_calls": total,
                "actual_total_calls": total,
                "unused_calls": 0,
                "estimate": np.asarray(estimate).tolist(),
                "elapsed_seconds": time.perf_counter() - elapsed_start,
                "design_seconds": design_seconds,
                "utility_seconds": utility_seconds,
                "coverage": coverage,
                "design_diagnostics": copy.deepcopy(design.diagnostics),
                "diagnostics": diagnostic,
            }
            save(row)
            del design, utilities
            print(
                f"Cancer {METHOD_LABELS[method]} repeat "
                f"{repeat + 1}/{REPEATS}, budget {budget_index + 1}/5 "
                f"complete ({design_seconds:.1f}s design, "
                f"{utility_seconds:.1f}s utility)",
                flush=True,
            )
    else:
        with GameEvaluator(
            SklearnClassificationGame,
            game_args,
            n_jobs=utility_jobs,
            chunksize=chunksize,
            start_method=start_method,
            worker_threads=1,
        ) as evaluator:
            for budget_index, total in enumerate(TOTAL_BUDGETS):
                if budget_index in completed:
                    continue
                seed = _method_seed(method, repeat, budget_index)
                started = time.perf_counter()
                result = _evaluate_baseline(
                    method,
                    evaluator,
                    total_call_budget=total,
                    seed=seed,
                )
                elapsed = time.perf_counter() - started
                diagnostics = _diagnostic_dict(result.diagnostics)
                actual = int(diagnostics["utility_evaluations"])
                row = {
                    "method": method,
                    "repeat": repeat,
                    "budget_index": budget_index,
                    "seed": seed,
                    "target_total_calls": total,
                    "actual_total_calls": actual,
                    "unused_calls": total - actual,
                    "estimate": np.asarray(result.values).tolist(),
                    "elapsed_seconds": elapsed,
                    "diagnostics": diagnostics,
                }
                save(row)
                print(
                    f"Cancer {METHOD_LABELS[method]} repeat "
                    f"{repeat + 1}/{REPEATS}, budget {budget_index + 1}/5 "
                    f"complete ({elapsed:.1f}s, {actual:,}/{total:,} calls)",
                    flush=True,
                )
    return {"method": method, "repeat": repeat, "rows": rows}


def _summary_common(
    estimates: np.ndarray,
    *,
    truth: np.ndarray,
    truth_se: np.ndarray,
    efficiency_target: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    return summarize_method(
        estimates,
        truth,
        truth_se,
        efficiency_target,
        bootstrap_seed=bootstrap_seed,
    )


def _reduced_source_summary(
    source_summary: Mapping[str, Any],
    *,
    method: str,
    source_method: str,
    budget_index: int,
    truth: np.ndarray,
    truth_se: np.ndarray,
    efficiency_target: float,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    estimates = np.asarray(source_summary["estimates"], dtype=np.float64)[
        :REPEATS
    ]
    bootstrap_seed = (
        METHOD_SEED
        + 100 * budget_index
        + SOURCE_BOOTSTRAP_METHOD_INDEX[source_method]
    )
    summary = _summary_common(
        estimates,
        truth=truth,
        truth_se=truth_se,
        efficiency_target=efficiency_target,
        bootstrap_seed=bootstrap_seed,
    )
    total = TOTAL_BUDGETS[budget_index]
    diagnostics = [
        {
            "repeat": repeat,
            "seed": METHOD_SEED + 100_000 * budget_index + repeat,
            "utility_evaluations": total,
            "target_call_budget": total,
            "unused_calls": 0,
            "boundary_utility_evaluations": BOUNDARY_CALLS,
            "inner_utility_evaluations": INNER_BUDGETS[budget_index],
            "reused_from_source_report": True,
            "source_method": source_method,
            "source_repeat_index": repeat,
        }
        for repeat in range(REPEATS)
    ]
    summary.update(
        {
            "label": METHOD_LABELS[method],
            "method_label": METHOD_LABELS[method],
            "target_total_utility_calls": total,
            "target_total_call_budget": total,
            "actual_utility_calls_by_repeat": [total] * REPEATS,
            "actual_utility_calls_per_estimate": [total] * REPEATS,
            "mean_actual_utility_calls": float(total),
            "unused_calls_per_estimate": [0] * REPEATS,
            "mean_unused_calls": 0.0,
            "diagnostics_by_repeat": diagnostics,
            "source_mean_design_seconds_over_10_repeats": source_summary.get(
                "mean_design_seconds"
            ),
            "source_mean_utility_seconds_over_10_repeats": source_summary.get(
                "mean_utility_seconds"
            ),
            "provenance": {
                "kind": "reduced_from_immutable_source",
                "source_report": str(source_path),
                "source_sha256": source_sha256,
                "source_method": source_method,
                "source_repeat_indices": list(range(REPEATS)),
                "all_three_repeat_metrics_recomputed": True,
                "recomputed_bootstrap_seed": bootstrap_seed,
            },
        }
    )
    return summary


def _fresh_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    budget_index: int,
    truth: np.ndarray,
    truth_se: np.ndarray,
    efficiency_target: float,
) -> dict[str, Any] | None:
    selected = sorted(
        (
            row
            for row in rows
            if row["method"] == method
            and int(row["budget_index"]) == budget_index
        ),
        key=lambda row: int(row["repeat"]),
    )
    if len(selected) != REPEATS:
        return None
    if [int(row["repeat"]) for row in selected] != list(range(REPEATS)):
        raise ValueError("fresh summary has duplicate/missing repeats")
    estimates = np.asarray(
        [row["estimate"] for row in selected], dtype=np.float64
    )
    summary = _summary_common(
        estimates,
        truth=truth,
        truth_se=truth_se,
        efficiency_target=efficiency_target,
        bootstrap_seed=(
            FRESH_BASE_SEEDS[method] + 700_000 + budget_index
        ),
    )
    total = TOTAL_BUDGETS[budget_index]
    actual = [int(row["actual_total_calls"]) for row in selected]
    elapsed = [float(row["elapsed_seconds"]) for row in selected]
    summary.update(
        {
            "label": METHOD_LABELS[method],
            "method_label": METHOD_LABELS[method],
            "target_total_utility_calls": total,
            "target_total_call_budget": total,
            "actual_utility_calls_by_repeat": actual,
            "actual_utility_calls_per_estimate": actual,
            "mean_actual_utility_calls": float(np.mean(actual)),
            "unused_calls_per_estimate": [
                total - value for value in actual
            ],
            "mean_unused_calls": float(
                np.mean([total - value for value in actual])
            ),
            "total_seconds_by_repeat": elapsed,
            "mean_total_seconds": float(np.mean(elapsed)),
            "diagnostics_by_repeat": [
                copy.deepcopy(row["diagnostics"]) for row in selected
            ],
            "provenance": {
                "kind": "newly_evaluated",
                "method": method,
                "repeat_indices": list(range(REPEATS)),
                "base_seed": FRESH_BASE_SEEDS[method],
            },
        }
    )
    if method == "inside_greedy":
        summary["mean_design_seconds"] = float(
            np.mean([row["design_seconds"] for row in selected])
        )
        summary["mean_utility_and_estimation_seconds"] = float(
            np.mean([row["utility_seconds"] for row in selected])
        )
    return summary


def build_report(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    ground_truth_cache: Path,
    ground_truth_cache_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    boundary_utilities: Sequence[float],
    boundary_wall_seconds: float,
    repeat_processes: int,
    jobs_per_repeat: int,
    sdiff_jobs: int,
    design_jobs_per_repeat: int,
    chunksize: int,
    start_method: str,
    design_start_method: str,
    experiment_wall_seconds: float,
) -> dict[str, Any]:
    merged = _merge_rows(rows)
    truth = _finite_vector(
        source["ground_truth"]["values"], NUM_PLAYERS, name="truth"
    )
    truth_se = _finite_vector(
        source["ground_truth"]["standard_errors"],
        NUM_PLAYERS,
        name="truth SE",
    )
    efficiency_target = float(source["boundary"]["efficiency_target"])
    results: dict[str, Any] = {}
    for budget_index, (inner, total) in enumerate(
        zip(INNER_BUDGETS, TOTAL_BUDGETS)
    ):
        source_methods = source["results_by_inner_budget"][str(inner)][
            "methods"
        ]
        methods: dict[str, Any] = {}
        for method, source_method in SOURCE_METHOD_MAPPING.items():
            methods[method] = _reduced_source_summary(
                source_methods[source_method],
                method=method,
                source_method=source_method,
                budget_index=budget_index,
                truth=truth,
                truth_se=truth_se,
                efficiency_target=efficiency_target,
                source_path=source_path,
                source_sha256=source_sha256,
            )
        for method in FRESH_METHODS:
            summary = _fresh_summary(
                merged,
                method=method,
                budget_index=budget_index,
                truth=truth,
                truth_se=truth_se,
                efficiency_target=efficiency_target,
            )
            if summary is not None:
                methods[method] = summary
        results[str(inner)] = {
            "inner_utility_calls": inner,
            "boundary_utility_calls": BOUNDARY_CALLS,
            "total_utility_calls_per_estimate": total,
            "methods": {
                method: methods[method]
                for method in METHOD_ORDER
                if method in methods
            },
        }
    total_cells = len(FRESH_METHODS) * REPEATS * len(INNER_BUDGETS)
    complete = len(merged) == total_cells
    boundary_array = np.asarray(boundary_utilities, dtype=np.float64)
    fresh_physical_calls = BOUNDARY_CALLS + sum(
        int(row["actual_total_calls"])
        - (BOUNDARY_CALLS if row["method"] == "inside_greedy" else 0)
        for row in merged
    )
    return {
        "status": "complete" if complete else "cancer_comparison_running",
        "experiment": EXPERIMENT_ID,
        "game": {
            "dataset": DATASET,
            "num_players": NUM_PLAYERS,
            "num_test": NUM_TEST,
            "model": "sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='scale')",
            "utility": "fixed-test classification accuracy",
            "empty_coalition_rule": (
                "best constant-label accuracy on the fixed test set"
            ),
            "single_class_rule": (
                "predict the coalition's sole observed class"
            ),
        },
        "dataset": copy.deepcopy(source["dataset"]),
        "boundary": copy.deepcopy(source["boundary"]),
        "boundary_reconstruction": {
            "utilities": boundary_array.tolist(),
            "sha256_float64_bytes": hashlib.sha256(
                boundary_array.tobytes()
            ).hexdigest(),
            "physical_utility_calls": BOUNDARY_CALLS,
            "wall_seconds": boundary_wall_seconds,
            "source_empty_full_match": True,
        },
        "ground_truth": copy.deepcopy(source["ground_truth"]),
        "configuration": {
            "dataset": DATASET,
            "methods": list(METHOD_ORDER),
            "method_labels": copy.deepcopy(METHOD_LABELS),
            "method_mapping": {
                "inside_greedy": "newly evaluated",
                "inside_orbit": "frame_orbit_ratio first three repeats",
                "ofa_iid_linear": "iid_linear_ofa first three repeats",
                "ofa_iid_ratio": (
                    "official_ofa_fixed_ratio first three repeats"
                ),
                "cc": "local strict port of official basic CC",
                "s_diff": "local strict S-Diff port",
                "kernel_shap": "sampled-Gram constrained KernelSHAP",
                "tmc_shapley": "copied TMC-Shapley stopping rule",
            },
            "inside": {
                "greedy": {
                    "sections": "4.1--4.3",
                    "design": "per_size_frame_coupled_design",
                    "first_moment_scope": "per_size",
                    "second_moment_scope": "per_size",
                    "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                    "mean_balance_mode": MEAN_BALANCE_MODE,
                    "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                    "fixed_slice_rng_partitioning": (
                        "independent_seedsequence_substreams"
                    ),
                    "estimator": "OFA conditional-mean ratio",
                    "ratio_missing_policy": "raise",
                },
                "orbit": {
                    "section": "4.4",
                    "design": "cyclic_orbit_frame_design",
                    "first_moment_scope": (
                        "per_size_exact_by_complete_orbits"
                    ),
                    "second_moment_scope": "per_size",
                    "candidate_pool": SOURCE_ORBIT_CANDIDATE_POOL,
                    "estimator": "OFA conditional-mean ratio",
                    "reused_source_method": "frame_orbit_ratio",
                },
            },
            "num_players": NUM_PLAYERS,
            "n_test": NUM_TEST,
            "repeats": REPEATS,
            "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
            "mean_balance_mode": MEAN_BALANCE_MODE,
            "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
            "protocol_version": "per_size_ratio_k64_lambda1over16",
            "budget_multipliers": list(BUDGET_MULTIPLIERS),
            "inner_budgets": list(INNER_BUDGETS),
            "total_call_budgets": list(TOTAL_BUDGETS),
            "boundary_utility_calls": BOUNDARY_CALLS,
            "call_budget_semantics": "total physical utility-call cap",
            "method_seed": METHOD_SEED,
            "fresh_method_base_seeds": copy.deepcopy(FRESH_BASE_SEEDS),
            "seed_semantics": (
                "base_seed[method] + 100000 * budget_index + repeat"
            ),
            "num_baseline_tasks": NUM_BASELINE_TASKS,
            "repeat_processes": repeat_processes,
            "jobs_per_repeat": jobs_per_repeat,
            "sdiff_jobs": sdiff_jobs,
            "design_jobs_per_repeat": design_jobs_per_repeat,
            "maximum_concurrent_utility_workers": max(
                repeat_processes * jobs_per_repeat, sdiff_jobs
            ),
            "maximum_concurrent_design_workers": (
                repeat_processes * design_jobs_per_repeat
            ),
            "parallel_reproducibility": (
                "worker counts change scheduling only; fixed-size designs "
                "use deterministic SeedSequence substreams"
            ),
            "chunksize": chunksize,
            "start_method": start_method,
            "design_start_method": design_start_method,
            "worker_threads": 1,
            "checkpoint_granularity": "one method-repeat-budget cell",
            "source_report": str(source_path),
            "source_report_sha256": source_sha256,
            "source_report_never_modified": True,
            "source_repeat_indices": list(range(REPEATS)),
            "ground_truth_cache": str(ground_truth_cache),
            "ground_truth_cache_sha256": ground_truth_cache_sha256,
            "ground_truth_reused_without_recomputation": True,
        },
        "fresh_progress": {
            "completed_cells": len(merged),
            "total_cells": total_cells,
            "fresh_stage_physical_utility_calls": fresh_physical_calls,
            "shared_boundary_calls_counted_once_for_execution": True,
            "rows": merged,
        },
        "results_by_inner_budget": results,
        "experiment_wall_seconds": experiment_wall_seconds,
    }


def _validate_resume_report(
    report: Mapping[str, Any],
    *,
    source_sha256: str,
    ground_truth_cache_sha256: str,
) -> list[dict[str, Any]]:
    if report.get("experiment") != EXPERIMENT_ID:
        raise ValueError("resume report has another experiment identity")
    configuration = report.get("configuration")
    progress = report.get("fresh_progress")
    if not isinstance(configuration, Mapping) or not isinstance(
        progress, Mapping
    ):
        raise ValueError("resume report lacks configuration/progress")
    expected = {
        "source_report_sha256": source_sha256,
        "num_players": NUM_PLAYERS,
        "repeats": REPEATS,
        "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
        "mean_balance_mode": MEAN_BALANCE_MODE,
        "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
        "inner_budgets": list(INNER_BUDGETS),
        "total_call_budgets": list(TOTAL_BUDGETS),
        "methods": list(METHOD_ORDER),
        "fresh_method_base_seeds": FRESH_BASE_SEEDS,
        "num_baseline_tasks": NUM_BASELINE_TASKS,
        "ground_truth_cache_sha256": ground_truth_cache_sha256,
    }
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"resume configuration differs on {key}")
    if report.get("status") not in {
        "cancer_comparison_running",
        "complete",
    }:
        raise ValueError("resume report has an invalid status")
    rows = _merge_rows(progress.get("rows", []))
    total_cells = len(FRESH_METHODS) * REPEATS * len(INNER_BUDGETS)
    if int(progress.get("completed_cells", -1)) != len(rows):
        raise ValueError("resume completed-cell count disagrees with rows")
    if int(progress.get("total_cells", -1)) != total_cells:
        raise ValueError("resume total-cell count is wrong")
    return rows


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    ground_truth_cache = args.ground_truth_cache.resolve()
    if source_path == output_path:
        raise ValueError("output must differ from the immutable source")
    if not source_path.exists():
        raise ValueError(f"source report does not exist: {source_path}")
    for name in (
        "repeat_processes",
        "jobs_per_repeat",
        "sdiff_jobs",
        "design_jobs_per_repeat",
        "boundary_jobs",
        "chunksize",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"{name} must be positive")
    if args.start_method not in mp.get_all_start_methods():
        raise ValueError(f"unsupported start method: {args.start_method}")
    if args.design_start_method not in mp.get_all_start_methods():
        raise ValueError(
            f"unsupported design start method: {args.design_start_method}"
        )
    if args.repeat_processes * args.jobs_per_repeat > 128:
        raise ValueError("repeat utility workers may not exceed 128")
    if args.repeat_processes * args.design_jobs_per_repeat > 128:
        raise ValueError("concurrent fixed-slice design workers exceed 128")
    if args.sdiff_jobs > 128:
        raise ValueError("S-Diff workers may not exceed 128")

    source_sha256 = _sha256(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_report(source)
    gt_cache_sha256 = _validate_ground_truth_cache(
        ground_truth_cache, source["ground_truth"]
    )
    game_args, metadata = load_sklearn_train_test_split(
        DATASET, test_size=TEST_SIZE, dataset_seed=DATASET_SEED
    )
    validate_reconstructed_dataset(source["dataset"], metadata)
    game_args = game_args | {
        "model": "rbf_svm",
        "regularization": 1.0,
    }

    existing: dict[str, Any] | None = None
    initial_rows: list[dict[str, Any]] = []
    if output_path.exists():
        if not args.resume:
            raise ValueError(f"output already exists: {output_path}")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        initial_rows = _validate_resume_report(
            existing,
            source_sha256=source_sha256,
            ground_truth_cache_sha256=gt_cache_sha256,
        )
        boundary_utilities, boundary_wall_seconds = (
            _validate_boundary_reconstruction(existing)
        )
        if existing.get("status") == "complete":
            from experiments.validate_inside_comparison import (
                validate_report as validate_inside_report,
            )

            validate_inside_report(existing)
            print(f"Cancer INSIDE report already complete: {output_path}")
            return existing

    started = time.perf_counter()
    prior_seconds = (
        float(existing.get("experiment_wall_seconds", 0.0))
        if existing is not None
        else 0.0
    )
    if existing is None:
        boundary_rows = boundary_coalitions(NUM_PLAYERS)
        boundary_start = time.perf_counter()
        with GameEvaluator(
            SklearnClassificationGame,
            game_args,
            n_jobs=args.boundary_jobs,
            chunksize=args.chunksize,
            start_method=args.start_method,
            worker_threads=1,
        ) as evaluator:
            boundary_array = evaluator.evaluate(boundary_rows)
        boundary_wall_seconds = time.perf_counter() - boundary_start
        boundary_utilities = _finite_vector(
            boundary_array,
            BOUNDARY_CALLS,
            name="recomputed boundary utilities",
        ).tolist()
    boundary = boundary_from_utilities(
        np.asarray(boundary_utilities, dtype=np.float64), NUM_PLAYERS
    )
    if not math.isclose(
        boundary.empty,
        float(source["boundary"]["empty"]),
        rel_tol=0.0,
        abs_tol=1e-14,
    ) or not math.isclose(
        boundary.full,
        float(source["boundary"]["full"]),
        rel_tol=0.0,
        abs_tol=1e-14,
    ):
        raise ValueError("recomputed Cancer boundary disagrees with source")

    def checkpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        report = build_report(
            source,
            source_path=source_path,
            source_sha256=source_sha256,
            ground_truth_cache=ground_truth_cache,
            ground_truth_cache_sha256=gt_cache_sha256,
            rows=rows,
            boundary_utilities=boundary_utilities,
            boundary_wall_seconds=boundary_wall_seconds,
            repeat_processes=min(args.repeat_processes, REPEATS),
            jobs_per_repeat=args.jobs_per_repeat,
            sdiff_jobs=args.sdiff_jobs,
            design_jobs_per_repeat=args.design_jobs_per_repeat,
            chunksize=args.chunksize,
            start_method=args.start_method,
            design_start_method=args.design_start_method,
            experiment_wall_seconds=(
                prior_seconds + time.perf_counter() - started
            ),
        )
        _atomic_write(output_path, report)
        return report

    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else Path(str(output_path) + ".checkpoints")
    )
    merged = _merge_rows(initial_rows)
    checkpoint(merged)
    for method in FRESH_METHODS:
        rows_by_repeat = {
            repeat: [
                row
                for row in merged
                if row["method"] == method
                and int(row["repeat"]) == repeat
            ]
            for repeat in range(REPEATS)
        }
        pending = [
            repeat
            for repeat in range(REPEATS)
            if len(rows_by_repeat[repeat]) < len(INNER_BUDGETS)
        ]
        if not pending:
            continue
        method_outer = 1 if method == "s_diff" else min(
            args.repeat_processes, len(pending)
        )
        method_jobs = (
            args.sdiff_jobs if method == "s_diff" else args.jobs_per_repeat
        )
        tasks = [
            (
                method,
                repeat,
                game_args,
                boundary_utilities,
                source_sha256,
                checkpoint_dir / f"{method}-repeat-{repeat:02d}.json",
                rows_by_repeat[repeat],
                method_jobs,
                args.design_jobs_per_repeat,
                args.chunksize,
                args.start_method,
                args.design_start_method,
            )
            for repeat in pending
        ]
        if method_outer <= 1:
            for task in tasks:
                completed = _run_fresh_repeat(*task)
                merged = _merge_rows(merged, completed["rows"])
                checkpoint(merged)
        else:
            context = mp.get_context(args.start_method)
            with ProcessPoolExecutor(
                max_workers=method_outer,
                mp_context=context,
            ) as executor:
                futures = {
                    executor.submit(_run_fresh_repeat, *task): task[1]
                    for task in tasks
                }
                for future in as_completed(futures):
                    repeat = futures[future]
                    completed = future.result()
                    merged = _merge_rows(merged, completed["rows"])
                    report = checkpoint(merged)
                    print(
                        f"Cancer {METHOD_LABELS[method]} repeat "
                        f"{repeat + 1}/{REPEATS} merged "
                        f"({report['fresh_progress']['completed_cells']}"
                        f"/{report['fresh_progress']['total_cells']} cells)",
                        flush=True,
                    )
    final = checkpoint(merged)
    if final["status"] != "complete":
        raise RuntimeError("Cancer comparison ended with missing cells")
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/json/cancer_full_train_rbf_svm_frame_ofa_228k_4p55m.json"
        ),
    )
    parser.add_argument(
        "--ground-truth-cache",
        type=Path,
        default=Path("results/npz/cancer_full_train_rbf_svm_gt.npz"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/json/cancer_inside_comparison_per_size_ratio_"
            "k64_lambda1over16_3repeats_228k_4p55m.json"
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--repeat-processes", type=int, default=3)
    parser.add_argument("--jobs-per-repeat", type=int, default=42)
    parser.add_argument("--sdiff-jobs", type=int, default=128)
    parser.add_argument("--design-jobs-per-repeat", type=int, default=40)
    parser.add_argument("--boundary-jobs", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=512)
    parser.add_argument(
        "--start-method",
        choices=mp.get_all_start_methods(),
        default="spawn",
    )
    parser.add_argument(
        "--design-start-method",
        choices=mp.get_all_start_methods(),
        default=("fork" if "fork" in mp.get_all_start_methods() else "spawn"),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_experiment(args)
    print(
        f"Saved {report['status']} Cancer INSIDE comparison to "
        f"{args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

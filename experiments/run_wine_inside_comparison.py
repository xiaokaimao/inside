"""Add the canonical INSIDE-Greedy result to the final Wine benchmark.

The final Wine report already contains the expensive baselines and the
cyclic-orbit ratio estimator.  This runner deliberately evaluates only the
fine-grained row-wise INSIDE design, then creates a new eight-method report:

* ``inside_greedy`` is the explicitly configured ``K=64, lambda0=1/16``
  ``per_size_frame_coupled_design`` followed by the OFA conditional-mean
  (ratio) estimator with strict player/size coverage;
* ``inside_orbit`` is the existing cyclic-orbit ratio result;
* the remaining six methods are copied verbatim from the immutable source
  report, with only uniform call-accounting/provenance aliases added.

Each repeat owns a persistent :class:`~frame_ofa.GameEvaluator` and writes an
atomic shard after every budget.  Consequently a killed run resumes without
re-evaluating completed INSIDE-Greedy cells.  The source report is never
modified and the output path is required to be different from it.
"""

from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from frame_ofa import (
    GameEvaluator,
    boundary_coalitions,
    boundary_from_utilities,
    estimate_official_ratio_ofa,
    inner_size_distribution,
    per_size_frame_coupled_design,
)


EXPERIMENT_ID = (
    "wine_inside_greedy_orbit_baseline_comparison_"
    "per_size_ratio_k64_lambda1over16"
)
NUM_PLAYERS = 142
NUM_TEST = 36
REPEATS = 3
SOURCE_CANDIDATE_POOL = 4
INSIDE_GREEDY_CANDIDATE_POOL = 64
SOURCE_LEGACY_MEAN_BALANCE = 0.1
MEAN_BALANCE_MODE = "normalized"
MEAN_BALANCE_LAMBDA0 = 1.0 / 16.0
DATASET_SEED = 2024
TEST_SIZE = 0.2
BOUNDARY_CALLS = 2 * NUM_PLAYERS + 2
BUDGET_MULTIPLIERS = (500, 1000, 2000, 5000, 10000)
INNER_BUDGETS = tuple(NUM_PLAYERS * value for value in BUDGET_MULTIPLIERS)
TOTAL_BUDGETS = tuple(value + BOUNDARY_CALLS for value in INNER_BUDGETS)
METHOD_SEED = 910001

METHOD_MAPPING = {
    "inside_orbit": "frame_orbit_ratio",
    "ofa_iid_linear": "iid_linear_ofa",
    "ofa_iid_ratio": "official_ofa_fixed_ratio",
    "cc": "official_cc_basic",
    "s_diff": "s_diff",
    "kernel_shap": "kernel_shap_sampled",
    "tmc_shapley": "tmc_shapley",
}
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


def _as_finite_vector(
    value: Any, length: int, *, name: str
) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite length-{length} vector")
    return vector


def _as_estimate_matrix(value: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (REPEATS, NUM_PLAYERS) or not np.all(
        np.isfinite(matrix)
    ):
        raise ValueError(
            f"{name} estimates must have shape "
            f"({REPEATS}, {NUM_PLAYERS})"
        )
    return matrix


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}")


def validate_source_report(report: Mapping[str, Any]) -> None:
    """Strictly validate the immutable Wine report used for augmentation."""
    _require_equal(report.get("status"), "complete", name="source status")
    dataset = report.get("dataset")
    configuration = report.get("configuration")
    ground_truth = report.get("ground_truth")
    boundary = report.get("boundary")
    results = report.get("results_by_inner_budget")
    if not all(
        isinstance(value, Mapping)
        for value in (
            dataset,
            configuration,
            ground_truth,
            boundary,
            results,
        )
    ):
        raise ValueError("source report is missing a required mapping")

    _require_equal(dataset.get("dataset"), "wine", name="dataset")
    _require_equal(
        dataset.get("split"),
        "stratified_train_test",
        name="dataset split",
    )
    _require_equal(
        dataset.get("dataset_seed"), DATASET_SEED, name="dataset seed"
    )
    if not math.isclose(
        float(dataset.get("test_size", math.nan)),
        TEST_SIZE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("Wine test size is not the canonical 0.2 split")
    _require_equal(
        len(dataset.get("train_labels", [])),
        NUM_PLAYERS,
        name="number of Wine training players",
    )
    _require_equal(
        len(dataset.get("test_labels", [])),
        NUM_TEST,
        name="number of Wine test observations",
    )
    _require_equal(
        sum(dataset.get("train_class_counts", [])),
        NUM_PLAYERS,
        name="Wine training class-count sum",
    )
    _require_equal(
        sum(dataset.get("test_class_counts", [])),
        NUM_TEST,
        name="Wine test class-count sum",
    )

    expected_configuration = {
        "num_players": NUM_PLAYERS,
        "n_test": NUM_TEST,
        "repeats": REPEATS,
        "candidate_pool": SOURCE_CANDIDATE_POOL,
        "budget_multipliers": list(BUDGET_MULTIPLIERS),
        "inner_budgets": list(INNER_BUDGETS),
        "total_call_budgets": list(TOTAL_BUDGETS),
        "method_seed": METHOD_SEED,
    }
    for key, expected in expected_configuration.items():
        _require_equal(
            configuration.get(key), expected, name=f"configuration.{key}"
        )
    if not math.isclose(
        float(configuration.get("mean_balance", math.nan)),
        SOURCE_LEGACY_MEAN_BALANCE,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("configuration.mean_balance must be 0.1")
    model = str(configuration.get("model", ""))
    if "SVC" not in model or "rbf" not in model:
        raise ValueError("source report is not the canonical RBF-SVM run")

    _require_equal(
        boundary.get("utility_calls"),
        BOUNDARY_CALLS,
        name="boundary utility calls",
    )
    empty = float(boundary.get("empty", math.nan))
    full = float(boundary.get("full", math.nan))
    target = float(boundary.get("efficiency_target", math.nan))
    if not all(math.isfinite(value) for value in (empty, full, target)):
        raise ValueError("source boundary values must be finite")
    if not math.isclose(target, full - empty, rel_tol=0.0, abs_tol=1e-14):
        raise ValueError("source efficiency target disagrees with boundaries")

    truth = _as_finite_vector(
        ground_truth.get("values"), NUM_PLAYERS, name="ground truth"
    )
    truth_se = _as_finite_vector(
        ground_truth.get("standard_errors"),
        NUM_PLAYERS,
        name="ground-truth standard errors",
    )
    if np.any(truth_se < 0):
        raise ValueError("ground-truth standard errors must be nonnegative")
    if not math.isclose(
        float(truth.sum()), target, rel_tol=0.0, abs_tol=1e-10
    ):
        raise ValueError("ground truth violates Shapley efficiency")
    _require_equal(
        ground_truth.get("independent_pair_units"),
        800_000,
        name="ground-truth antithetic pair count",
    )
    _require_equal(
        ground_truth.get("permutations"),
        1_600_000,
        name="ground-truth permutation count",
    )
    _require_equal(
        ground_truth.get("permutation_path_equivalent_utility_calls"),
        2 + 1_600_000 * (NUM_PLAYERS - 1),
        name="ground-truth permutation-path call count",
    )
    _require_equal(
        ground_truth.get("shared_boundary_calls_physically_evaluated"),
        BOUNDARY_CALLS,
        name="ground-truth shared boundary calls",
    )
    expected_rmse_se = float(np.sqrt(np.mean(np.square(truth_se))))
    if not math.isclose(
        float(ground_truth.get("rmse_standard_error", math.nan)),
        expected_rmse_se,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("ground-truth RMSE standard error is inconsistent")

    _require_equal(
        set(results),
        {str(value) for value in INNER_BUDGETS},
        name="source budget keys",
    )
    required_source_methods = set(METHOD_MAPPING.values())
    for inner, total in zip(INNER_BUDGETS, TOTAL_BUDGETS):
        budget_row = results[str(inner)]
        if not isinstance(budget_row, Mapping):
            raise ValueError(f"budget {inner} is not a mapping")
        _require_equal(
            budget_row.get("inner_utility_calls"),
            inner,
            name=f"budget {inner} inner calls",
        )
        _require_equal(
            budget_row.get("total_utility_calls_per_estimate"),
            total,
            name=f"budget {inner} total calls",
        )
        methods = budget_row.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError(f"budget {inner} has no method mapping")
        missing = required_source_methods - set(methods)
        if missing:
            raise ValueError(
                f"budget {inner} is missing source methods {sorted(missing)}"
            )
        for source_method in required_source_methods:
            summary = methods[source_method]
            if not isinstance(summary, Mapping):
                raise ValueError(
                    f"budget {inner}/{source_method} is not a mapping"
                )
            estimates = _as_estimate_matrix(
                summary.get("estimates"),
                name=f"budget {inner}/{source_method}",
            )
            recomputed_rmse = float(
                np.sqrt(np.mean(np.square(estimates - truth[None, :])))
            )
            if not math.isclose(
                float(summary.get("aggregate_rmse", math.nan)),
                recomputed_rmse,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"budget {inner}/{source_method} RMSE is inconsistent"
                )


def validate_reconstructed_dataset(
    source_metadata: Mapping[str, Any],
    reconstructed_metadata: Mapping[str, Any],
) -> None:
    """Verify that sklearn recreates every metadata field in the source."""
    for key, source_value in source_metadata.items():
        if key not in reconstructed_metadata:
            raise ValueError(f"reconstructed dataset is missing {key}")
        reconstructed = reconstructed_metadata[key]
        if key in {"standardization_mean", "standardization_std"}:
            if not np.allclose(
                np.asarray(source_value, dtype=np.float64),
                np.asarray(reconstructed, dtype=np.float64),
                rtol=0.0,
                atol=1e-14,
            ):
                raise ValueError(f"reconstructed dataset disagrees on {key}")
        elif source_value != reconstructed:
            raise ValueError(f"reconstructed dataset disagrees on {key}")


def _inside_seed(repeat: int, budget_index: int) -> int:
    return METHOD_SEED + budget_index * 100_000 + repeat


def _ratio_coverage_diagnostics(
    coalitions: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
    """Return the strict player/size coverage audit required by ratio OFA."""
    coalitions = np.asarray(coalitions, dtype=bool)
    sizes = np.asarray(sizes, dtype=np.int64)
    if coalitions.ndim != 2 or sizes.shape != (len(coalitions),):
        raise ValueError("coalitions and sizes have incompatible shapes")
    expected_sizes, _, _ = inner_size_distribution(coalitions.shape[1])
    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    missing_inclusion = 0
    missing_exclusion = 0
    observed_sizes = 0
    exactly_balanced_sizes = 0
    for size in expected_sizes:
        rows = coalitions[sizes == size]
        if not len(rows):
            inclusion = np.zeros(coalitions.shape[1], dtype=np.int64)
        else:
            observed_sizes += 1
            inclusion = rows.sum(axis=0, dtype=np.int64)
        exclusion = len(rows) - inclusion
        minimum_inclusion = min(minimum_inclusion, int(inclusion.min()))
        minimum_exclusion = min(minimum_exclusion, int(exclusion.min()))
        missing_inclusion += int(np.count_nonzero(inclusion == 0))
        missing_exclusion += int(np.count_nonzero(exclusion == 0))
        exactly_balanced_sizes += int(
            len(rows) > 0 and np.all(inclusion == inclusion[0])
        )
    return {
        "all_player_size_strata_covered": (
            missing_inclusion == 0 and missing_exclusion == 0
        ),
        "expected_inner_sizes": int(len(expected_sizes)),
        "observed_inner_sizes": observed_sizes,
        "missing_inner_sizes": int(len(expected_sizes) - observed_sizes),
        "minimum_inclusion_count": int(minimum_inclusion),
        "minimum_exclusion_count": int(minimum_exclusion),
        "missing_inclusion_strata": missing_inclusion,
        "missing_exclusion_strata": missing_exclusion,
        "exactly_1_balanced_sizes": exactly_balanced_sizes,
    }


def _shard_key(source_sha256: str, repeat: int) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "source_sha256": source_sha256,
        "repeat": repeat,
        "inner_budgets": list(INNER_BUDGETS),
        "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
        "mean_balance_mode": MEAN_BALANCE_MODE,
        "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
        "design": "per_size_frame_coupled_design",
        "second_moment_scope": "per_size",
        "estimator": "estimate_official_ratio_ofa(missing='raise')",
        "method_seed": METHOD_SEED,
    }


def _validate_inside_row(row: Mapping[str, Any]) -> None:
    repeat = int(row.get("repeat", -1))
    budget_index = int(row.get("budget_index", -1))
    if repeat not in range(REPEATS) or budget_index not in range(
        len(INNER_BUDGETS)
    ):
        raise ValueError("INSIDE-Greedy row has invalid repeat/budget index")
    inner = INNER_BUDGETS[budget_index]
    total = TOTAL_BUDGETS[budget_index]
    _require_equal(
        row.get("inner_utility_calls"), inner, name="row inner calls"
    )
    _require_equal(
        row.get("total_utility_calls"), total, name="row total calls"
    )
    _require_equal(
        row.get("seed"),
        _inside_seed(repeat, budget_index),
        name="row seed",
    )
    _require_equal(
        row.get("candidate_pool"),
        INSIDE_GREEDY_CANDIDATE_POOL,
        name="row candidate pool",
    )
    _as_finite_vector(
        row.get("estimate"), NUM_PLAYERS, name="INSIDE-Greedy estimate"
    )
    for key in (
        "design_seconds",
        "utility_and_estimation_seconds",
        "elapsed_seconds",
    ):
        value = float(row.get(key, math.nan))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"INSIDE-Greedy row has invalid {key}")
    diagnostics = row.get("design_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("INSIDE-Greedy row has no design diagnostics")
    _require_equal(
        diagnostics.get("mean_balance_mode"),
        MEAN_BALANCE_MODE,
        name="row mean-balance mode",
    )
    if not math.isclose(
        float(diagnostics.get("mean_balance_lambda0", math.nan)),
        MEAN_BALANCE_LAMBDA0,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "row mean-balance lambda0 violates the configured protocol"
        )
    effective = float(
        diagnostics.get("mean_balance_effective_raw", math.nan)
    )
    expected_effective = MEAN_BALANCE_LAMBDA0 * (
        1.0 - 1.0 / (NUM_PLAYERS - 1)
    )
    if not math.isfinite(effective) or not math.isclose(
        effective,
        expected_effective,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("row has the wrong effective raw mean balance")
    _require_equal(
        diagnostics.get("second_moment_scope"),
        "per_size",
        name="row second-moment scope",
    )
    _require_equal(
        row.get("estimator"),
        "ofa_conditional_mean_ratio_missing_raise",
        name="row estimator",
    )
    _require_equal(
        row.get("official_ratio_missing_policy"),
        "raise",
        name="row ratio missing policy",
    )
    coverage = row.get("coverage")
    if not isinstance(coverage, Mapping) or not coverage.get(
        "all_player_size_strata_covered"
    ):
        raise ValueError("INSIDE-Greedy row lacks complete ratio coverage")
    if (
        int(coverage.get("minimum_inclusion_count", 0)) <= 0
        or int(coverage.get("minimum_exclusion_count", 0)) <= 0
        or int(coverage.get("missing_inclusion_strata", 1)) != 0
        or int(coverage.get("missing_exclusion_strata", 1)) != 0
        or int(coverage.get("missing_inner_sizes", 1)) != 0
    ):
        raise ValueError("INSIDE-Greedy row has invalid ratio coverage counts")


def _merge_rows(*groups: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for group in groups:
        for raw_row in group:
            row = copy.deepcopy(dict(raw_row))
            _validate_inside_row(row)
            key = (int(row["repeat"]), int(row["budget_index"]))
            previous = indexed.get(key)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting checkpoint rows for {key}")
            indexed[key] = row
    return [indexed[key] for key in sorted(indexed)]


def _load_shard(
    path: Path, *, source_sha256: str, repeat: int
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require_equal(
        payload.get("cache_key"),
        _shard_key(source_sha256, repeat),
        name=f"checkpoint key for repeat {repeat}",
    )
    rows = _merge_rows(payload.get("rows", []))
    if any(int(row["repeat"]) != repeat for row in rows):
        raise ValueError(f"checkpoint {path} contains a different repeat")
    return rows


def _run_inside_repeat(
    repeat: int,
    game_args: dict[str, Any],
    boundary_utilities: Sequence[float],
    source_sha256: str,
    shard_path: Path,
    initial_rows: Sequence[Mapping[str, Any]],
    jobs_per_repeat: int,
    chunksize: int,
    start_method: str,
) -> dict[str, Any]:
    """Evaluate every missing budget for one repeat and checkpoint each."""
    if repeat not in range(REPEATS):
        raise ValueError("repeat index is out of range")
    if jobs_per_repeat < 1 or chunksize < 1:
        raise ValueError("jobs_per_repeat and chunksize must be positive")
    rows = _merge_rows(
        initial_rows,
        _load_shard(
            shard_path, source_sha256=source_sha256, repeat=repeat
        ),
    )
    completed = {int(row["budget_index"]) for row in rows}
    boundary = boundary_from_utilities(
        np.asarray(boundary_utilities, dtype=np.float64), NUM_PLAYERS
    )
    with GameEvaluator(
        SklearnClassificationGame,
        game_args,
        n_jobs=jobs_per_repeat,
        chunksize=chunksize,
        start_method=start_method,
        worker_threads=1,
    ) as evaluator:
        for budget_index, inner_budget in enumerate(INNER_BUDGETS):
            if budget_index in completed:
                continue
            seed = _inside_seed(repeat, budget_index)
            elapsed_start = time.perf_counter()
            design_start = time.perf_counter()
            design = per_size_frame_coupled_design(
                NUM_PLAYERS,
                inner_budget,
                seed=seed,
                candidate_pool=INSIDE_GREEDY_CANDIDATE_POOL,
                mean_balance=MEAN_BALANCE_LAMBDA0,
                mean_balance_mode=MEAN_BALANCE_MODE,
            )
            design_seconds = time.perf_counter() - design_start
            coverage = _ratio_coverage_diagnostics(
                design.coalitions, design.sizes
            )
            if not coverage["all_player_size_strata_covered"]:
                raise ValueError(
                    "INSIDE-Greedy lacks an in/out observation for at least "
                    "one player/size stratum"
                )
            utility_start = time.perf_counter()
            utilities = evaluator.evaluate(design.coalitions)
            estimate = estimate_official_ratio_ofa(
                design,
                utilities,
                boundary,
                missing="raise",
            )
            utility_seconds = time.perf_counter() - utility_start
            row = {
                "repeat": repeat,
                "budget_index": budget_index,
                "seed": seed,
                "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                "inner_utility_calls": inner_budget,
                "total_utility_calls": inner_budget + BOUNDARY_CALLS,
                "estimate": estimate.tolist(),
                "design_seconds": design_seconds,
                "utility_and_estimation_seconds": utility_seconds,
                "elapsed_seconds": time.perf_counter() - elapsed_start,
                "estimator": "ofa_conditional_mean_ratio_missing_raise",
                "official_ratio_missing_policy": "raise",
                "coverage": coverage,
                "design_diagnostics": copy.deepcopy(design.diagnostics),
            }
            _validate_inside_row(row)
            rows = _merge_rows(rows, [row])
            _atomic_write(
                shard_path,
                {
                    "cache_key": _shard_key(source_sha256, repeat),
                    "rows": rows,
                },
            )
            del design, utilities
            print(
                f"Wine INSIDE-Greedy repeat {repeat + 1}/{REPEATS}, "
                f"budget {budget_index + 1}/{len(INNER_BUDGETS)} complete "
                f"({design_seconds:.1f}s design, "
                f"{utility_seconds:.1f}s utility)",
                flush=True,
            )
    return {"repeat": repeat, "rows": rows}


def _canonicalize_reused_summary(
    summary: Mapping[str, Any],
    *,
    method: str,
    source_method: str,
    total_calls: int,
    source_path: Path,
    source_sha256: str,
) -> dict[str, Any]:
    canonical = copy.deepcopy(dict(summary))
    _as_estimate_matrix(
        canonical.get("estimates"), name=f"reused {source_method}"
    )
    raw_actual = canonical.get("actual_utility_calls_by_repeat")
    if raw_actual is None:
        raw_actual = canonical.get("actual_utility_calls_per_estimate")
    if raw_actual is None:
        scalar = canonical.get("utility_calls_per_estimate", total_calls)
        raw_actual = [scalar] * REPEATS
    actual = [int(value) for value in raw_actual]
    if len(actual) != REPEATS or any(
        value < 0 or value > total_calls for value in actual
    ):
        raise ValueError(f"invalid utility calls for reused {source_method}")
    unused = [total_calls - value for value in actual]
    canonical["label"] = METHOD_LABELS[method]
    canonical["method_label"] = METHOD_LABELS[method]
    canonical["target_total_utility_calls"] = total_calls
    canonical["target_total_call_budget"] = total_calls
    canonical["actual_utility_calls_by_repeat"] = actual
    canonical["actual_utility_calls_per_estimate"] = actual
    canonical["mean_actual_utility_calls"] = float(np.mean(actual))
    canonical["unused_calls_per_estimate"] = unused
    canonical["mean_unused_calls"] = float(np.mean(unused))
    if "mean_elapsed_seconds" not in canonical:
        elapsed = canonical.get("mean_total_seconds")
        canonical["mean_elapsed_seconds"] = (
            None if elapsed is None else float(elapsed)
        )
    source_diagnostics = canonical.get("diagnostics_by_repeat")
    if source_diagnostics is None:
        source_diagnostics = canonical.get("cc_diagnostics_by_repeat")
    if source_diagnostics is None:
        source_diagnostics = [{} for _ in range(REPEATS)]
    if len(source_diagnostics) != REPEATS:
        raise ValueError(f"invalid diagnostics for reused {source_method}")
    diagnostics: list[dict[str, Any]] = []
    for repeat, (raw, calls, unused_calls) in enumerate(
        zip(source_diagnostics, actual, unused)
    ):
        item = copy.deepcopy(dict(raw)) if isinstance(raw, Mapping) else {
            "source_diagnostics": copy.deepcopy(raw)
        }
        item.setdefault("utility_evaluations", calls)
        item["target_total_utility_calls"] = total_calls
        item["unused_calls"] = unused_calls
        item["repeat"] = repeat
        item["reused_from_source_report"] = True
        diagnostics.append(item)
    canonical["diagnostics_by_repeat"] = diagnostics
    canonical["provenance"] = {
        "kind": "reused_verbatim_estimates",
        "source_report": str(source_path),
        "source_sha256": source_sha256,
        "source_method": source_method,
        "renamed_to": method,
        "only_call_accounting_and_provenance_aliases_added": True,
    }
    return canonical


def _inside_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    budget_index: int,
    ground_truth: np.ndarray,
    ground_truth_se: np.ndarray,
    efficiency_target: float,
) -> dict[str, Any] | None:
    selected = sorted(
        (
            row
            for row in rows
            if int(row["budget_index"]) == budget_index
        ),
        key=lambda row: int(row["repeat"]),
    )
    if len(selected) != REPEATS:
        return None
    if [int(row["repeat"]) for row in selected] != list(range(REPEATS)):
        raise ValueError("INSIDE-Greedy budget has duplicate/missing repeats")
    estimates = np.asarray(
        [row["estimate"] for row in selected], dtype=np.float64
    )
    summary = summarize_method(
        estimates,
        ground_truth,
        ground_truth_se,
        efficiency_target,
        bootstrap_seed=METHOD_SEED + 700_000 + budget_index,
    )
    total_calls = TOTAL_BUDGETS[budget_index]
    actual = [total_calls] * REPEATS
    elapsed = [float(row["elapsed_seconds"]) for row in selected]
    summary.update(
        {
            "label": METHOD_LABELS["inside_greedy"],
            "method_label": METHOD_LABELS["inside_greedy"],
            "target_total_utility_calls": total_calls,
            "target_total_call_budget": total_calls,
            "actual_utility_calls_by_repeat": actual,
            "actual_utility_calls_per_estimate": actual,
            "mean_actual_utility_calls": float(total_calls),
            "unused_calls_per_estimate": [0] * REPEATS,
            "mean_unused_calls": 0.0,
            "total_seconds_by_repeat": elapsed,
            "mean_elapsed_seconds": float(np.mean(elapsed)),
            "mean_total_seconds": float(np.mean(elapsed)),
            "mean_design_seconds": float(
                np.mean([row["design_seconds"] for row in selected])
            ),
            "mean_utility_and_estimation_seconds": float(
                np.mean(
                    [
                        row["utility_and_estimation_seconds"]
                        for row in selected
                    ]
                )
            ),
            "diagnostics_by_repeat": [
                {
                    "repeat": int(row["repeat"]),
                    "seed": int(row["seed"]),
                    "candidate_pool": int(row["candidate_pool"]),
                    "utility_evaluations": total_calls,
                    "boundary_utility_evaluations": BOUNDARY_CALLS,
                    "inner_utility_evaluations": INNER_BUDGETS[
                        budget_index
                    ],
                    "target_total_utility_calls": total_calls,
                    "unused_calls": 0,
                    "design_method": "frame_coupled_per_size",
                    "second_moment_scope": "per_size",
                    "estimator": row["estimator"],
                    "official_ratio_missing_policy": row[
                        "official_ratio_missing_policy"
                    ],
                    "coverage": copy.deepcopy(row["coverage"]),
                    "mean_balance_mode": MEAN_BALANCE_MODE,
                    "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                    "mean_balance_effective_raw": float(
                        row["design_diagnostics"][
                            "mean_balance_effective_raw"
                        ]
                    ),
                    "design_diagnostics": copy.deepcopy(
                        row["design_diagnostics"]
                    ),
                    "design_seconds": float(row["design_seconds"]),
                    "utility_and_estimation_seconds": float(
                        row["utility_and_estimation_seconds"]
                    ),
                }
                for row in selected
            ],
            "provenance": {
                "kind": "newly_evaluated",
                "design": "per_size_frame_coupled_design (row-wise greedy)",
                "second_moment_scope": "per_size",
                "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                "mean_balance_mode": MEAN_BALANCE_MODE,
                "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                "estimator": "estimate_official_ratio_ofa",
                "ratio_missing_policy": "raise",
            },
        }
    )
    return summary


def build_report(
    source: Mapping[str, Any],
    *,
    source_path: Path,
    source_sha256: str,
    rows: Sequence[Mapping[str, Any]],
    boundary_utilities: Sequence[float],
    boundary_wall_seconds: float,
    repeat_processes: int,
    jobs_per_repeat: int,
    chunksize: int,
    start_method: str,
    experiment_wall_seconds: float,
) -> dict[str, Any]:
    """Build either a resumable partial report or the final eight-method one."""
    merged_rows = _merge_rows(rows)
    truth = _as_finite_vector(
        source["ground_truth"]["values"],
        NUM_PLAYERS,
        name="ground truth",
    )
    truth_se = _as_finite_vector(
        source["ground_truth"]["standard_errors"],
        NUM_PLAYERS,
        name="ground-truth standard errors",
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
        greedy = _inside_summary(
            merged_rows,
            budget_index=budget_index,
            ground_truth=truth,
            ground_truth_se=truth_se,
            efficiency_target=efficiency_target,
        )
        if greedy is not None:
            methods["inside_greedy"] = greedy
        for method in METHOD_ORDER[1:]:
            source_method = METHOD_MAPPING[method]
            methods[method] = _canonicalize_reused_summary(
                source_methods[source_method],
                method=method,
                source_method=source_method,
                total_calls=total,
                source_path=source_path,
                source_sha256=source_sha256,
            )
        results[str(inner)] = {
            "inner_utility_calls": inner,
            "boundary_utility_calls": BOUNDARY_CALLS,
            "total_utility_calls_per_estimate": total,
            "methods": methods,
        }
    complete = len(merged_rows) == REPEATS * len(INNER_BUDGETS)
    recomputed_boundary = np.asarray(boundary_utilities, dtype=np.float64)
    return {
        "status": "complete" if complete else "inside_greedy_running",
        "experiment": EXPERIMENT_ID,
        "game": {
            "dataset": "wine",
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
            "utilities": recomputed_boundary.tolist(),
            "sha256_float64_bytes": hashlib.sha256(
                recomputed_boundary.tobytes()
            ).hexdigest(),
            "physical_utility_calls": BOUNDARY_CALLS,
            "wall_seconds": boundary_wall_seconds,
            "source_empty_full_match": True,
        },
        "ground_truth": copy.deepcopy(source["ground_truth"]),
        "configuration": {
            "methods": list(METHOD_ORDER),
            "method_labels": copy.deepcopy(METHOD_LABELS),
            "method_mapping": {
                "inside_greedy": "newly evaluated",
                **copy.deepcopy(METHOD_MAPPING),
            },
            "inside": {
                "greedy": {
                    "sections": "4.1--4.3",
                    "design": (
                        "per_size_frame_coupled_design (row-wise greedy)"
                    ),
                    "second_moment_scope": "per_size",
                    "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                    "mean_balance_mode": MEAN_BALANCE_MODE,
                    "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                    "estimator": "OFA conditional-mean ratio",
                    "ratio_missing_policy": "raise",
                },
                "orbit": {
                    "section": "4.4",
                    "design": "cyclic_orbit_frame_design",
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
            "budget_multipliers": list(BUDGET_MULTIPLIERS),
            "inner_budgets": list(INNER_BUDGETS),
            "total_call_budgets": list(TOTAL_BUDGETS),
            "boundary_utility_calls": BOUNDARY_CALLS,
            "method_seed": METHOD_SEED,
            "seed_semantics": (
                "seed = method_seed + 100000 * budget_index + repeat; "
                "independent of repeat/utility worker scheduling"
            ),
            "call_budget_semantics": "total physical utility-call cap",
            "repeat_processes": repeat_processes,
            "jobs_per_repeat": jobs_per_repeat,
            "maximum_concurrent_utility_workers": (
                repeat_processes * jobs_per_repeat
            ),
            "parallel_reproducibility": (
                "repeat_processes and jobs_per_repeat change scheduling only"
            ),
            "chunksize": chunksize,
            "start_method": start_method,
            "worker_threads": 1,
            "checkpoint_granularity": "one repeat-budget cell",
            "source_report": str(source_path),
            "source_report_sha256": source_sha256,
            "source_report_never_modified": True,
        },
        "inside_greedy_progress": {
            "completed_cells": len(merged_rows),
            "total_cells": REPEATS * len(INNER_BUDGETS),
            "rows": merged_rows,
        },
        "results_by_inner_budget": results,
        "experiment_wall_seconds": experiment_wall_seconds,
    }


def _validate_resume_report(
    report: Mapping[str, Any], *, source_sha256: str
) -> list[dict[str, Any]]:
    _require_equal(
        report.get("experiment"), EXPERIMENT_ID, name="resume experiment"
    )
    configuration = report.get("configuration")
    progress = report.get("inside_greedy_progress")
    if not isinstance(configuration, Mapping) or not isinstance(
        progress, Mapping
    ):
        raise ValueError("resume report has no configuration/progress")
    expected = {
        "source_report_sha256": source_sha256,
        "num_players": NUM_PLAYERS,
        "repeats": REPEATS,
        "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
        "mean_balance_mode": MEAN_BALANCE_MODE,
        "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
        "inner_budgets": list(INNER_BUDGETS),
        "methods": list(METHOD_ORDER),
    }
    for key, value in expected.items():
        _require_equal(
            configuration.get(key), value, name=f"resume configuration.{key}"
        )
    inside = configuration.get("inside")
    greedy = inside.get("greedy") if isinstance(inside, Mapping) else None
    if not isinstance(greedy, Mapping):
        raise ValueError("resume report has no INSIDE-Greedy configuration")
    expected_greedy = {
        "design": "per_size_frame_coupled_design (row-wise greedy)",
        "second_moment_scope": "per_size",
        "estimator": "OFA conditional-mean ratio",
        "ratio_missing_policy": "raise",
    }
    for key, value in expected_greedy.items():
        _require_equal(
            greedy.get(key), value, name=f"resume INSIDE-Greedy.{key}"
        )
    return _merge_rows(progress.get("rows", []))


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    source_path = args.input.resolve()
    output_path = args.output.resolve()
    if source_path == output_path:
        raise ValueError("output must differ from the immutable source report")
    if not source_path.exists():
        raise ValueError(f"source report does not exist: {source_path}")
    if args.repeat_processes < 1 or args.jobs_per_repeat < 1:
        raise ValueError("repeat processes and jobs per repeat must be positive")
    if args.chunksize < 1:
        raise ValueError("chunksize must be positive")
    if args.start_method not in mp.get_all_start_methods():
        raise ValueError(f"unsupported start method: {args.start_method}")

    source_sha256 = _sha256(source_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_report(source)
    game_args, metadata = load_sklearn_train_test_split(
        "wine", test_size=TEST_SIZE, dataset_seed=DATASET_SEED
    )
    validate_reconstructed_dataset(source["dataset"], metadata)
    game_args = game_args | {
        "model": "rbf_svm",
        "regularization": 1.0,
    }
    _require_equal(
        len(game_args["y_valued"]), NUM_PLAYERS, name="reconstructed players"
    )

    initial_rows: list[dict[str, Any]] = []
    existing: dict[str, Any] | None = None
    if output_path.exists():
        if not args.resume:
            raise ValueError(f"output already exists: {output_path}")
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        initial_rows = _validate_resume_report(
            existing, source_sha256=source_sha256
        )
        if existing.get("status") == "complete":
            print(f"Wine INSIDE report already complete: {output_path}")
            return existing

    started = time.perf_counter()
    if existing is not None:
        boundary_utilities = existing["boundary_reconstruction"]["utilities"]
        boundary_wall_seconds = float(
            existing["boundary_reconstruction"]["wall_seconds"]
        )
    else:
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
            boundary_utilities_array = evaluator.evaluate(boundary_rows)
        boundary_wall_seconds = time.perf_counter() - boundary_start
        boundary_utilities = boundary_utilities_array.tolist()
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
        raise ValueError("recomputed Wine boundary disagrees with source")

    def checkpoint(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        prior_seconds = (
            float(existing.get("experiment_wall_seconds", 0.0))
            if existing is not None
            else 0.0
        )
        report = build_report(
            source,
            source_path=source_path,
            source_sha256=source_sha256,
            rows=rows,
            boundary_utilities=boundary_utilities,
            boundary_wall_seconds=boundary_wall_seconds,
            repeat_processes=min(args.repeat_processes, REPEATS),
            jobs_per_repeat=args.jobs_per_repeat,
            chunksize=args.chunksize,
            start_method=args.start_method,
            experiment_wall_seconds=(
                prior_seconds + time.perf_counter() - started
            ),
        )
        _atomic_write(output_path, report)
        return report

    merged_rows = _merge_rows(initial_rows)
    checkpoint(merged_rows)
    checkpoint_dir = (
        args.checkpoint_dir.resolve()
        if args.checkpoint_dir is not None
        else Path(str(output_path) + ".checkpoints")
    )
    rows_by_repeat = {
        repeat: [
            row for row in merged_rows if int(row["repeat"]) == repeat
        ]
        for repeat in range(REPEATS)
    }
    pending_repeats = [
        repeat
        for repeat in range(REPEATS)
        if len(rows_by_repeat[repeat]) < len(INNER_BUDGETS)
    ]
    tasks = [
        (
            repeat,
            game_args,
            boundary_utilities,
            source_sha256,
            checkpoint_dir / f"repeat-{repeat:02d}.json",
            rows_by_repeat[repeat],
            args.jobs_per_repeat,
            args.chunksize,
            args.start_method,
        )
        for repeat in pending_repeats
    ]
    if min(args.repeat_processes, len(tasks)) <= 1:
        for task in tasks:
            completed = _run_inside_repeat(*task)
            merged_rows = _merge_rows(merged_rows, completed["rows"])
            checkpoint(merged_rows)
    elif tasks:
        context = mp.get_context(args.start_method)
        with ProcessPoolExecutor(
            max_workers=min(args.repeat_processes, len(tasks)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_inside_repeat, *task): task[0]
                for task in tasks
            }
            for future in as_completed(futures):
                repeat = futures[future]
                completed = future.result()
                merged_rows = _merge_rows(merged_rows, completed["rows"])
                report = checkpoint(merged_rows)
                print(
                    f"Wine INSIDE-Greedy repeat {repeat + 1}/{REPEATS} "
                    f"merged ({report['inside_greedy_progress']['completed_cells']}"
                    f"/{REPEATS * len(INNER_BUDGETS)} cells)",
                    flush=True,
                )
    final = checkpoint(merged_rows)
    if final["status"] != "complete":
        raise RuntimeError("Wine INSIDE-Greedy run ended with missing cells")
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/json/"
            "wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_"
            "3repeats_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/json/"
            "wine_inside_comparison_per_size_ratio_k64_lambda1over16_"
            "3repeats_71k_1p42m.json"
        ),
    )
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--repeat-processes", type=int, default=3)
    parser.add_argument("--jobs-per-repeat", type=int, default=42)
    parser.add_argument("--boundary-jobs", type=int, default=42)
    parser.add_argument("--chunksize", type=int, default=512)
    parser.add_argument(
        "--start-method",
        choices=mp.get_all_start_methods(),
        default="spawn",
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
        f"Saved {report['status']} Wine INSIDE comparison to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

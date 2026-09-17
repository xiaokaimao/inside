"""Reproduce the Airport INSIDE-Greedy lambda-scale diagnostic.

This module is deliberately separate from :mod:`frame_ofa`: it does not
change the production design or estimator.  It reruns the first formal
Airport budget with the exact formal seeds under two settings:

* the published raw ``mean_balance=0.1``;
* a dimensionless ``lambda0=1`` converted to the raw Eq. (12)/(13) scale.

For a fixed size schedule, write ``w_t^2=s_t(n-s_t)`` and ``d=n-1``.  The
conversion used here is

    lambda_eff = lambda0 * (1 - 1/d) * mean_t(w_t^2).

It follows by normalizing the IID reference magnitudes

    E ||A-EA||_F^2 = (1 - 1/d) sum_t w_t^2,
    E sum_s ||m_s||_2^2 = T.

The exact realized size schedule is taken from the raw-lambda design and is
verified to be identical when the converted lambda is rerun with the same
seed.  Candidate selection is otherwise allowed to change, as intended.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.debug_inside_greedy_gap import (
    _boundary,
    _evaluate,
    _num_players,
    _rmse,
    _truth,
)
from experiments.run_analytic_inside_comparison import _seed_for
from frame_ofa import estimate_coupled, frame_coupled_design


DATASET = "airport"
FORMAL_BASE_SEED = 20260827
FORMAL_BUDGET_INDEX = 0
FORMAL_METHOD_INDEX = 0
FORMAL_BUDGET_MULTIPLIER = 500
FORMAL_REPEATS = 3
FORMAL_CANDIDATE_POOL = 4
RAW_MEAN_BALANCE = 0.1
NORMALIZED_LAMBDA0 = 1.0


def converted_mean_balance(
    sizes: np.ndarray,
    num_players: int,
    lambda0: float,
) -> tuple[float, float, float]:
    """Return ``(lambda_eff, mean(w^2), dimension_correction)``.

    ``lambda0`` is dimensionless.  ``lambda_eff`` is on the raw scale used by
    Eq. (12)/(13) and ``frame_coupled_design(mean_balance=...)``.
    """
    schedule = np.asarray(sizes, dtype=np.int64)
    if schedule.ndim != 1 or len(schedule) == 0:
        raise ValueError("sizes must be a nonempty one-dimensional schedule")
    if num_players < 4:
        raise ValueError("at least four players are required")
    if np.any(schedule < 2) or np.any(schedule > num_players - 2):
        raise ValueError("sizes must lie in the OFA inner range")
    if not np.isfinite(lambda0) or lambda0 < 0:
        raise ValueError("lambda0 must be finite and nonnegative")
    dimension = num_players - 1
    correction = 1.0 - 1.0 / dimension
    mean_weight_squared = float(
        np.mean(schedule * (num_players - schedule), dtype=np.float64)
    )
    effective = float(lambda0 * correction * mean_weight_squared)
    return effective, mean_weight_squared, correction


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _design_sha256(coalitions: np.ndarray) -> str:
    packed = np.packbits(np.asarray(coalitions, dtype=bool), axis=1)
    return _array_sha256(packed)


def _numeric_diagnostics(design: Any) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in design.diagnostics.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }


def _configuration_result(
    *,
    label: str,
    mean_balance: float,
    design: Any,
    utilities: np.ndarray,
    boundary: Any,
    truth: np.ndarray,
    boundary_calls: int,
) -> dict[str, Any]:
    estimate = estimate_coupled(
        design, utilities, boundary, baseline="linear"
    )
    return {
        "label": label,
        "mean_balance_raw": float(mean_balance),
        "estimator": "estimate_coupled(baseline='linear')",
        "inner_utility_calls": int(len(design.coalitions)),
        "boundary_utility_calls": int(boundary_calls),
        "total_utility_calls": int(len(design.coalitions) + boundary_calls),
        "rmse": _rmse(estimate, truth),
        "estimate": estimate.tolist(),
        "estimate_sha256_float64": _array_sha256(
            np.asarray(estimate, dtype=np.float64)
        ),
        "coalition_design_sha256_packbits": _design_sha256(
            design.coalitions
        ),
        "size_schedule_sha256_int64": _array_sha256(
            np.asarray(design.sizes, dtype=np.int64)
        ),
        "design_diagnostics": _numeric_diagnostics(design),
    }


def _run_repeat(repeat: int) -> dict[str, Any]:
    num_players = _num_players(DATASET)
    inner_calls = num_players * FORMAL_BUDGET_MULTIPLIER
    seed = _seed_for(
        FORMAL_BASE_SEED,
        repeat,
        FORMAL_BUDGET_INDEX,
        FORMAL_METHOD_INDEX,
    )
    truth = _truth(DATASET)
    boundary, boundary_calls = _boundary(DATASET)

    raw_started = time.perf_counter()
    raw_design = frame_coupled_design(
        num_players,
        inner_calls,
        seed=seed,
        candidate_pool=FORMAL_CANDIDATE_POOL,
        mean_balance=RAW_MEAN_BALANCE,
        mean_balance_mode="raw",
    )
    raw_design_seconds = time.perf_counter() - raw_started
    effective, mean_weight_squared, correction = converted_mean_balance(
        raw_design.sizes, num_players, NORMALIZED_LAMBDA0
    )

    converted_started = time.perf_counter()
    converted_design = frame_coupled_design(
        num_players,
        inner_calls,
        seed=seed,
        candidate_pool=FORMAL_CANDIDATE_POOL,
        mean_balance=effective,
        mean_balance_mode="raw",
    )
    converted_design_seconds = time.perf_counter() - converted_started
    same_schedule = bool(
        np.array_equal(raw_design.sizes, converted_design.sizes)
    )
    if not same_schedule:
        raise RuntimeError("changing lambda unexpectedly changed the size schedule")

    raw_utilities = _evaluate(DATASET, raw_design.coalitions)
    converted_utilities = _evaluate(DATASET, converted_design.coalitions)
    raw = _configuration_result(
        label="raw_lambda_0.1",
        mean_balance=RAW_MEAN_BALANCE,
        design=raw_design,
        utilities=raw_utilities,
        boundary=boundary,
        truth=truth,
        boundary_calls=boundary_calls,
    )
    converted = _configuration_result(
        label="normalized_lambda0_1",
        mean_balance=effective,
        design=converted_design,
        utilities=converted_utilities,
        boundary=boundary,
        truth=truth,
        boundary_calls=boundary_calls,
    )
    raw["design_seconds"] = raw_design_seconds
    converted["design_seconds"] = converted_design_seconds
    return {
        "repeat": repeat,
        "seed": seed,
        "schedule_mean_weight_squared": mean_weight_squared,
        "dimension_correction": correction,
        "normalized_lambda0": NORMALIZED_LAMBDA0,
        "converted_mean_balance_raw": effective,
        "same_size_schedule": same_schedule,
        "raw_lambda": raw,
        "converted_lambda": converted,
        "paired_change": {
            "rmse_change_percent": float(
                100.0 * (converted["rmse"] / raw["rmse"] - 1.0)
            ),
            "slice_mean_direction_rms_change_percent": float(
                100.0
                * (
                    converted["design_diagnostics"][
                        "slice_mean_direction_rms"
                    ]
                    / raw["design_diagnostics"][
                        "slice_mean_direction_rms"
                    ]
                    - 1.0
                )
            ),
            "frobenius_discrepancy_change_percent": float(
                100.0
                * (
                    converted["design_diagnostics"][
                        "frobenius_discrepancy"
                    ]
                    / raw["design_diagnostics"][
                        "frobenius_discrepancy"
                    ]
                    - 1.0
                )
            ),
        },
    }


def _aggregate(raw_repeats: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("raw_lambda", "converted_lambda"):
        rows = [repeat[field] for repeat in raw_repeats]
        repeat_rmse = np.asarray([row["rmse"] for row in rows])
        diagnostics = rows[0]["design_diagnostics"].keys()
        result[field] = {
            "repeat_rmse": repeat_rmse.tolist(),
            "aggregate_rmse": float(
                np.sqrt(np.mean(np.square(repeat_rmse)))
            ),
            "mean_repeat_rmse": float(np.mean(repeat_rmse)),
            "mean_design_seconds": float(
                np.mean([row["design_seconds"] for row in rows])
            ),
            "mean_design_diagnostics": {
                name: float(
                    np.mean([row["design_diagnostics"][name] for row in rows])
                )
                for name in diagnostics
            },
        }
    raw_rmse = result["raw_lambda"]["aggregate_rmse"]
    converted_rmse = result["converted_lambda"]["aggregate_rmse"]
    result["converted_vs_raw"] = {
        "aggregate_rmse_change_percent": float(
            100.0 * (converted_rmse / raw_rmse - 1.0)
        ),
        "aggregate_rmse_reduction_percent": float(
            100.0 * (1.0 - converted_rmse / raw_rmse)
        ),
    }
    return result


def _formal_report_match(
    report: Mapping[str, Any], formal_report_path: Path
) -> dict[str, Any]:
    formal_bytes = formal_report_path.read_bytes()
    formal = json.loads(formal_bytes)
    budget = str(report["configuration"]["inner_utility_calls"])
    summary = formal["results_by_inner_budget"][budget]["methods"][
        "inside_greedy"
    ]
    formal_estimates = np.asarray(summary["estimates"], dtype=np.float64)
    reproduced = np.asarray(
        [row["raw_lambda"]["estimate"] for row in report["raw_repeats"]],
        dtype=np.float64,
    )
    discrepancy = float(np.max(np.abs(formal_estimates - reproduced)))
    return {
        "formal_report": str(formal_report_path),
        "formal_report_sha256": hashlib.sha256(formal_bytes).hexdigest(),
        "source_method": "inside_greedy",
        "source_inner_budget": int(budget),
        "max_absolute_estimate_difference": discrepancy,
        "bitwise_equal_float64": bool(
            formal_estimates.tobytes() == reproduced.tobytes()
        ),
        "tolerance_1e-14_pass": bool(discrepancy <= 1e-14),
    }


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Validate retained rows and every derived aggregate."""
    configuration = report["configuration"]
    raw_repeats = report["raw_repeats"]
    if len(raw_repeats) != FORMAL_REPEATS:
        raise ValueError("lambda diagnostic must retain exactly three repeats")
    expected_seeds = [
        _seed_for(
            FORMAL_BASE_SEED,
            repeat,
            FORMAL_BUDGET_INDEX,
            FORMAL_METHOD_INDEX,
        )
        for repeat in range(FORMAL_REPEATS)
    ]
    if [row["seed"] for row in raw_repeats] != expected_seeds:
        raise ValueError("retained seeds do not match the formal seed schedule")
    num_players = int(configuration["num_players"])
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    for row in raw_repeats:
        if not row["same_size_schedule"]:
            raise ValueError("paired designs do not share the same size schedule")
        expected_lambda = (
            NORMALIZED_LAMBDA0
            * row["dimension_correction"]
            * row["schedule_mean_weight_squared"]
        )
        if not np.isclose(
            row["converted_mean_balance_raw"],
            expected_lambda,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("converted lambda is inconsistent")
        for field in ("raw_lambda", "converted_lambda"):
            item = row[field]
            estimate = np.asarray(item["estimate"], dtype=np.float64)
            if estimate.shape != (num_players,) or not np.all(
                np.isfinite(estimate)
            ):
                raise ValueError("retained estimate has invalid shape/values")
            recomputed = _rmse(estimate, truth)
            if not np.isclose(
                recomputed, item["rmse"], rtol=0.0, atol=1e-15
            ):
                raise ValueError("retained RMSE is inconsistent")
            if item["total_utility_calls"] != (
                configuration["inner_utility_calls"]
                + configuration["boundary_utility_calls"]
            ):
                raise ValueError("utility-call accounting is inconsistent")
    recomputed_aggregate = _aggregate(list(raw_repeats))
    for field in ("raw_lambda", "converted_lambda"):
        if not np.isclose(
            recomputed_aggregate[field]["aggregate_rmse"],
            report["aggregate"][field]["aggregate_rmse"],
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("aggregate RMSE is inconsistent")
    formal_match = report.get("formal_report_match")
    if formal_match is not None and not formal_match["tolerance_1e-14_pass"]:
        raise ValueError("raw lambda does not reproduce the formal report")
    return {
        "status": "ready_to_share",
        "checks": {
            "formal_seed_schedule": True,
            "same_size_schedule_within_pair": True,
            "lambda_conversion_formula": True,
            "repeat_rmse_recomputed": True,
            "aggregate_rmse_recomputed": True,
            "utility_call_accounting": True,
            "formal_report_match": formal_match is not None,
        },
    }


def run_experiment(*, processes: int = 3) -> dict[str, Any]:
    if processes < 1:
        raise ValueError("processes must be positive")
    num_players = _num_players(DATASET)
    truth = _truth(DATASET)
    _, boundary_calls = _boundary(DATASET)
    started = time.perf_counter()
    if processes == 1:
        raw_repeats = [_run_repeat(repeat) for repeat in range(FORMAL_REPEATS)]
    else:
        context = mp.get_context("spawn")
        raw_repeats = []
        with ProcessPoolExecutor(
            max_workers=min(processes, FORMAL_REPEATS),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_repeat, repeat): repeat
                for repeat in range(FORMAL_REPEATS)
            }
            for future in as_completed(futures):
                raw_repeats.append(future.result())
    raw_repeats.sort(key=lambda row: row["repeat"])
    report: dict[str, Any] = {
        "status": "complete",
        "experiment": "inside_greedy_lambda_scale_debug",
        "purpose": (
            "Diagnose whether raw mean_balance=0.1 is scale-compatible with "
            "the unnormalized Eq. (12)/(13) objective."
        ),
        "configuration": {
            "dataset": DATASET,
            "num_players": num_players,
            "budget_multiplier": FORMAL_BUDGET_MULTIPLIER,
            "inner_utility_calls": num_players * FORMAL_BUDGET_MULTIPLIER,
            "boundary_utility_calls": boundary_calls,
            "total_utility_calls_per_estimate": (
                num_players * FORMAL_BUDGET_MULTIPLIER + boundary_calls
            ),
            "repeats": FORMAL_REPEATS,
            "candidate_pool": FORMAL_CANDIDATE_POOL,
            "raw_mean_balance": RAW_MEAN_BALANCE,
            "normalized_lambda0": NORMALIZED_LAMBDA0,
            "formal_base_seed": FORMAL_BASE_SEED,
            "formal_budget_index": FORMAL_BUDGET_INDEX,
            "formal_method_index": FORMAL_METHOD_INDEX,
            "processes": min(processes, FORMAL_REPEATS),
        },
        "lambda_conversion": {
            "formula": (
                "lambda_eff = lambda0 * (1 - 1/(n-1)) * "
                "mean_t[s_t*(n-s_t)]"
            ),
            "reference_identities": [
                "E||A-EA||_F^2=(1-1/(n-1))*sum_t w_t^2",
                "E sum_s ||m_s||_2^2=T",
            ],
            "interpretation": (
                "lambda0 is dimensionless; mean_balance is the raw value "
                "consumed by the unchanged production design."
            ),
        },
        "ground_truth": {
            "kind": "exact_airport_threshold_decomposition",
            "values": truth.tolist(),
            "sum": float(truth.sum()),
            "sha256_float64": _array_sha256(truth),
        },
        "raw_repeats": raw_repeats,
        "aggregate": _aggregate(raw_repeats),
        "guardrails": [
            "No production frame_ofa code is modified by this diagnostic.",
            "lambda0=1 is derived from geometric reference scaling, not "
            "selected by minimizing Airport RMSE.",
            "This isolates lambda scaling but does not make Greedy and Orbit "
            "the same estimator."
        ],
        "elapsed_seconds": time.perf_counter() - started,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=3)
    parser.add_argument(
        "--formal-report",
        type=Path,
        default=Path(
            "results/json/airport_inside_comparison_3repeats_50k_1m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/json/inside_greedy_lambda_scale_debug.json"),
    )
    args = parser.parse_args()
    report = run_experiment(processes=args.processes)
    if args.formal_report.exists():
        report["formal_report_match"] = _formal_report_match(
            report, args.formal_report
        )
    report["validation"] = validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(report["aggregate"], indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

"""Paired ablations for the INSIDE-Greedy versus INSIDE-Orbit gap.

This diagnostic does not modify either production estimator.  It reuses the
same realized coalition design across several aggregation/control-variate
choices so that differences can be attributed to one factor at a time:

* endpoint-linear versus exact size-mean baselines on Greedy and IID rows;
* global HT versus fixed-size (ratio) aggregation on the same q*-marginal
  complete-orbit rows;
* strict ratio versus slice-wise linear aggregation on the formal cyclic
  INSIDE-Orbit rows (these should agree to floating-point precision);
* candidate-pool and mean-balance sensitivity for the Greedy design.

Airport and U.S. Electoral College voting games are used because both the
Shapley vector and every fixed-size utility mean are available exactly.
"""

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
from experiments.diagnose_us_voting_frame_designs import (
    exact_size_win_probabilities,
)
from experiments.us_electoral_voting_game import (
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
)
from frame_ofa import (
    CoalitionDesign,
    OFABoundary,
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_coupled,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    estimate_stratified,
    evaluate_airport,
    evaluate_weighted_voting,
    exact_airport_shapley,
    exact_shapley_shubik,
    frame_coupled_design,
    iid_ofa_design,
    inner_size_distribution,
    orbit_coupled_frame_design,
    shapley_boundary_vector,
)
from frame_ofa.estimator import _weighted_direction_mean


DATASETS = ("airport", "voting")
DEFAULT_BUDGET_MULTIPLIERS = (250, 500)
SENSITIVITY_CONFIGURATIONS = (
    ("k1_lambda0.1", 1, 0.1),
    ("k4_lambda0", 4, 0.0),
    ("k4_lambda0.1", 4, 0.1),
    ("k4_lambda1", 4, 1.0),
    ("k16_lambda0.1", 16, 0.1),
)


def _num_players(dataset: str) -> int:
    if dataset == "airport":
        return len(AIRPORT_COSTS)
    if dataset == "voting":
        return len(US_ELECTORAL_WEIGHTS)
    raise ValueError("dataset must be 'airport' or 'voting'")


def _evaluate(dataset: str, coalitions: np.ndarray) -> np.ndarray:
    if dataset == "airport":
        return np.asarray(
            evaluate_airport(coalitions, AIRPORT_COSTS), dtype=np.float64
        )
    if dataset == "voting":
        return np.asarray(
            evaluate_weighted_voting(
                coalitions, US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
            ),
            dtype=np.float64,
        )
    raise ValueError("dataset must be 'airport' or 'voting'")


def _truth(dataset: str) -> np.ndarray:
    if dataset == "airport":
        return exact_airport_shapley(AIRPORT_COSTS)
    if dataset == "voting":
        return exact_shapley_shubik(
            US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        )
    raise ValueError("dataset must be 'airport' or 'voting'")


def _exact_size_profile(dataset: str) -> np.ndarray:
    if dataset == "airport":
        return exact_size_mean_utilities(AIRPORT_COSTS)
    if dataset == "voting":
        return exact_size_win_probabilities(
            US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        )
    raise ValueError("dataset must be 'airport' or 'voting'")


def _boundary(dataset: str) -> tuple[OFABoundary, int]:
    num_players = _num_players(dataset)
    rows = boundary_coalitions(num_players)
    return (
        boundary_from_utilities(_evaluate(dataset, rows), num_players),
        len(rows),
    )


def endpoint_size_profile(boundary: OFABoundary) -> np.ndarray:
    """Return the endpoint-linear inner-direction control variate."""
    sizes = np.arange(boundary.num_players + 1, dtype=np.float64)
    return boundary.empty + sizes / boundary.num_players * (
        boundary.full - boundary.empty
    )


def global_linear_with_size_profile(
    design: CoalitionDesign,
    utilities: np.ndarray,
    boundary: OFABoundary,
    size_profile: np.ndarray,
) -> np.ndarray:
    """Evaluate the global OFA HT identity with an arbitrary size baseline.

    The formula is the production ``estimate_coupled`` formula, generalized
    from its two built-in baselines to a supplied vector ``b_s``.  A constant
    on one fixed inner-size slice has zero centered-direction contribution.
    Because the exact boundary terms are left unchanged, this remains
    unbiased whenever every row has the q* marginal and the supplied profile
    is fixed independently of the realized batch utilities.
    """
    values = np.asarray(utilities, dtype=np.float64)
    profile = np.asarray(size_profile, dtype=np.float64)
    num_players = boundary.num_players
    if design.coalitions.shape[1] != num_players:
        raise ValueError("design and boundary disagree on player count")
    if values.shape != (len(design.coalitions),):
        raise ValueError("utilities must contain one value per coalition")
    if profile.shape != (num_players + 1,):
        raise ValueError("size profile must contain n+1 values")
    _, _, normalizer = inner_size_distribution(num_players)
    residuals = values - profile[design.sizes]
    inner = normalizer * np.sqrt(num_players) * _weighted_direction_mean(
        design.coalitions, design.sizes, residuals
    )
    return shapley_boundary_vector(boundary) + inner


def _seed_for(
    base_seed: int,
    dataset: str,
    repeat: int,
    budget_index: int,
    family_index: int,
) -> int:
    dataset_index = DATASETS.index(dataset)
    sequence = np.random.SeedSequence(
        [base_seed, dataset_index, repeat, budget_index, family_index]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(estimate - truth))))


def _result_row(
    *,
    dataset: str,
    repeat: int,
    budget_multiplier: int,
    inner_calls: int,
    boundary_calls: int,
    method: str,
    design_family: str,
    estimator: str,
    baseline: str,
    estimate: np.ndarray,
    truth: np.ndarray,
    design: CoalitionDesign,
    seed: int,
    diagnostic_only: bool = False,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "repeat": repeat,
        "budget_multiplier": budget_multiplier,
        "inner_utility_calls": inner_calls,
        "total_utility_calls": inner_calls + boundary_calls,
        "method": method,
        "design_family": design_family,
        "estimator": estimator,
        "baseline": baseline,
        "seed": seed,
        "rmse": _rmse(estimate, truth),
        "diagnostic_only": diagnostic_only,
        "design_diagnostics": {
            key: float(value)
            for key, value in design.diagnostics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        },
    }


def _run_primary_repeat(
    dataset: str,
    repeat: int,
    budget_multipliers: tuple[int, ...],
    candidate_pool: int,
    mean_balance: float,
    base_seed: int,
) -> dict[str, Any]:
    num_players = _num_players(dataset)
    truth = _truth(dataset)
    exact_profile = _exact_size_profile(dataset)
    boundary, boundary_calls = _boundary(dataset)
    endpoint_profile = endpoint_size_profile(boundary)
    rows: list[dict[str, Any]] = []
    paired_checks: list[dict[str, Any]] = []

    for budget_index, multiplier in enumerate(budget_multipliers):
        inner_calls = num_players * multiplier
        seeds = {
            family: _seed_for(
                base_seed, dataset, repeat, budget_index, family_index
            )
            for family_index, family in enumerate(
                ("iid", "greedy", "cyclic", "qstar_orbit")
            )
        }

        iid = iid_ofa_design(
            num_players,
            inner_calls,
            seed=seeds["iid"],
            compute_diagnostics=True,
        )
        greedy = frame_coupled_design(
            num_players,
            inner_calls,
            seed=seeds["greedy"],
            candidate_pool=candidate_pool,
            mean_balance=mean_balance,
            mean_balance_mode="raw",
        )
        cyclic = cyclic_orbit_frame_design(
            num_players,
            num_orbits=multiplier,
            seed=seeds["cyclic"],
            candidate_pool=candidate_pool,
        )
        qstar_orbit = orbit_coupled_frame_design(
            num_players,
            inner_calls,
            seed=seeds["qstar_orbit"],
            candidate_pool=candidate_pool,
        )
        designs = {
            "iid": iid,
            "greedy": greedy,
            "cyclic": cyclic,
            "qstar_orbit": qstar_orbit,
        }
        utilities = {
            family: _evaluate(dataset, design.coalitions)
            for family, design in designs.items()
        }

        estimates: dict[str, tuple[np.ndarray, str, str, str, bool]] = {}
        for family in ("iid", "greedy"):
            endpoint = global_linear_with_size_profile(
                designs[family],
                utilities[family],
                boundary,
                endpoint_profile,
            )
            oracle = global_linear_with_size_profile(
                designs[family],
                utilities[family],
                boundary,
                exact_profile,
            )
            estimates[f"{family}_global_endpoint"] = (
                endpoint,
                family,
                "global_ht",
                "endpoint_linear",
                False,
            )
            estimates[f"{family}_global_oracle_size"] = (
                oracle,
                family,
                "global_ht",
                "exact_size_mean",
                False,
            )
            official_ratio = estimate_official_ratio_ofa(
                designs[family],
                utilities[family],
                boundary,
                missing="raise",
            )
            estimates[f"{family}_official_ratio"] = (
                official_ratio,
                family,
                "official_conditional_mean_ratio",
                "size_invariant",
                False,
            )
            paired_checks.append(
                {
                    "dataset": dataset,
                    "repeat": repeat,
                    "budget_multiplier": multiplier,
                    "design_family": family,
                    "size_only_leakage_rmse": float(
                        np.sqrt(np.mean(np.square(endpoint - oracle)))
                    ),
                }
            )

        cyclic_ratio = estimate_ratio_ofa(
            cyclic, utilities["cyclic"], boundary
        )
        cyclic_stratified = estimate_stratified(
            cyclic,
            utilities["cyclic"],
            boundary,
            baseline="linear",
        )
        estimates["cyclic_ratio"] = (
            cyclic_ratio,
            "cyclic",
            "strict_balanced_ratio",
            "size_invariant",
            False,
        )
        estimates["cyclic_stratified_linear"] = (
            cyclic_stratified,
            "cyclic",
            "slice_wise_linear",
            "endpoint_linear",
            False,
        )
        paired_checks.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "budget_multiplier": multiplier,
                "design_family": "cyclic",
                "ratio_vs_stratified_max_abs": float(
                    np.max(np.abs(cyclic_ratio - cyclic_stratified))
                ),
            }
        )

        # This is intentionally labeled diagnostic-only.  Cyclic INSIDE-Orbit
        # fixes rounded size counts, while the unweighted global HT identity
        # assumes q*-marginal rows.  It is included solely to show why the
        # estimator must match the design's size allocation.
        cyclic_global = global_linear_with_size_profile(
            cyclic, utilities["cyclic"], boundary, endpoint_profile
        )
        estimates["cyclic_global_endpoint_mismatched"] = (
            cyclic_global,
            "cyclic",
            "global_ht_with_fixed_rounded_size_counts",
            "endpoint_linear",
            True,
        )

        qstar_global = estimate_coupled(
            qstar_orbit,
            utilities["qstar_orbit"],
            boundary,
            baseline="linear",
        )
        qstar_global_oracle = global_linear_with_size_profile(
            qstar_orbit,
            utilities["qstar_orbit"],
            boundary,
            exact_profile,
        )
        qstar_ratio = estimate_ratio_ofa(
            qstar_orbit, utilities["qstar_orbit"], boundary
        )
        paired_checks.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "budget_multiplier": multiplier,
                "design_family": "qstar_orbit",
                "endpoint_vs_oracle_size_max_abs": float(
                    np.max(np.abs(qstar_global - qstar_global_oracle))
                ),
            }
        )
        estimates["qstar_orbit_global_endpoint"] = (
            qstar_global,
            "qstar_orbit",
            "global_ht",
            "endpoint_linear",
            False,
        )
        estimates["qstar_orbit_global_oracle_size"] = (
            qstar_global_oracle,
            "qstar_orbit",
            "global_ht",
            "exact_size_mean",
            False,
        )
        estimates["qstar_orbit_ratio"] = (
            qstar_ratio,
            "qstar_orbit",
            "strict_balanced_ratio",
            "size_invariant",
            False,
        )

        for method, (
            estimate,
            family,
            estimator,
            baseline,
            diagnostic_only,
        ) in estimates.items():
            rows.append(
                _result_row(
                    dataset=dataset,
                    repeat=repeat,
                    budget_multiplier=multiplier,
                    inner_calls=inner_calls,
                    boundary_calls=boundary_calls,
                    method=method,
                    design_family=family,
                    estimator=estimator,
                    baseline=baseline,
                    estimate=estimate,
                    truth=truth,
                    design=designs[family],
                    seed=seeds[family],
                    diagnostic_only=diagnostic_only,
                )
            )

    return {
        "dataset": dataset,
        "repeat": repeat,
        "rows": rows,
        "paired_checks": paired_checks,
    }


def _run_sensitivity_repeat(
    dataset: str,
    repeat: int,
    budget_multiplier: int,
    base_seed: int,
) -> dict[str, Any]:
    num_players = _num_players(dataset)
    inner_calls = num_players * budget_multiplier
    truth = _truth(dataset)
    exact_profile = _exact_size_profile(dataset)
    boundary, boundary_calls = _boundary(dataset)
    endpoint_profile = endpoint_size_profile(boundary)
    # Use the same deterministic seed schedule for every configuration.  This
    # is exact common-random-number pairing when K is unchanged; changing K
    # changes RNG consumption and therefore does not preserve the final
    # relabeling permutation.
    seed = _seed_for(base_seed, dataset, repeat, 99, 0)
    rows: list[dict[str, Any]] = []
    for configuration, candidate_pool, mean_balance in SENSITIVITY_CONFIGURATIONS:
        started = time.perf_counter()
        design = frame_coupled_design(
            num_players,
            inner_calls,
            seed=seed,
            candidate_pool=candidate_pool,
            mean_balance=mean_balance,
            mean_balance_mode="raw",
        )
        design_seconds = time.perf_counter() - started
        utilities = _evaluate(dataset, design.coalitions)
        endpoint = global_linear_with_size_profile(
            design, utilities, boundary, endpoint_profile
        )
        oracle = global_linear_with_size_profile(
            design, utilities, boundary, exact_profile
        )
        rows.append(
            {
                "dataset": dataset,
                "repeat": repeat,
                "budget_multiplier": budget_multiplier,
                "inner_utility_calls": inner_calls,
                "total_utility_calls": inner_calls + boundary_calls,
                "configuration": configuration,
                "candidate_pool": candidate_pool,
                "mean_balance": mean_balance,
                "seed": seed,
                "endpoint_rmse": _rmse(endpoint, truth),
                "oracle_size_rmse": _rmse(oracle, truth),
                "size_only_leakage_rmse": float(
                    np.sqrt(np.mean(np.square(endpoint - oracle)))
                ),
                "design_seconds": design_seconds,
                "design_diagnostics": {
                    key: float(value)
                    for key, value in design.diagnostics.items()
                    if isinstance(
                        value, (int, float, np.integer, np.floating)
                    )
                },
            }
        )
    return {"dataset": dataset, "repeat": repeat, "rows": rows}


def _aggregate_primary(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = [row for report in raw for row in report["rows"]]
    keys = sorted(
        {
            (row["dataset"], row["budget_multiplier"], row["method"])
            for row in flat
        }
    )
    summaries: list[dict[str, Any]] = []
    for dataset, multiplier, method in keys:
        selected = sorted(
            (
                row
                for row in flat
                if row["dataset"] == dataset
                and row["budget_multiplier"] == multiplier
                and row["method"] == method
            ),
            key=lambda row: row["repeat"],
        )
        repeat_rmse = np.asarray([row["rmse"] for row in selected])
        diagnostic_names = sorted(
            {
                key
                for row in selected
                for key in row["design_diagnostics"]
            }
        )
        summaries.append(
            {
                "dataset": dataset,
                "budget_multiplier": multiplier,
                "inner_utility_calls": selected[0]["inner_utility_calls"],
                "total_utility_calls": selected[0]["total_utility_calls"],
                "method": method,
                "design_family": selected[0]["design_family"],
                "estimator": selected[0]["estimator"],
                "baseline": selected[0]["baseline"],
                "diagnostic_only": selected[0]["diagnostic_only"],
                "aggregate_rmse": float(
                    np.sqrt(np.mean(np.square(repeat_rmse)))
                ),
                "mean_repeat_rmse": float(np.mean(repeat_rmse)),
                "repeat_rmse": repeat_rmse.tolist(),
                "mean_design_diagnostics": {
                    name: float(
                        np.mean(
                            [
                                row["design_diagnostics"].get(name, np.nan)
                                for row in selected
                            ]
                        )
                    )
                    for name in diagnostic_names
                },
            }
        )
    return summaries


def _aggregate_sensitivity(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = [row for report in raw for row in report["rows"]]
    summaries: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for configuration, candidate_pool, mean_balance in SENSITIVITY_CONFIGURATIONS:
            selected = sorted(
                (
                    row
                    for row in flat
                    if row["dataset"] == dataset
                    and row["configuration"] == configuration
                ),
                key=lambda row: row["repeat"],
            )
            if not selected:
                continue
            endpoint = np.asarray([row["endpoint_rmse"] for row in selected])
            oracle = np.asarray(
                [row["oracle_size_rmse"] for row in selected]
            )
            leakage = np.asarray(
                [row["size_only_leakage_rmse"] for row in selected]
            )
            diagnostics = selected[0]["design_diagnostics"].keys()
            summaries.append(
                {
                    "dataset": dataset,
                    "configuration": configuration,
                    "candidate_pool": candidate_pool,
                    "mean_balance": mean_balance,
                    "budget_multiplier": selected[0]["budget_multiplier"],
                    "aggregate_endpoint_rmse": float(
                        np.sqrt(np.mean(np.square(endpoint)))
                    ),
                    "aggregate_oracle_size_rmse": float(
                        np.sqrt(np.mean(np.square(oracle)))
                    ),
                    "aggregate_size_only_leakage_rmse": float(
                        np.sqrt(np.mean(np.square(leakage)))
                    ),
                    "mean_design_seconds": float(
                        np.mean([row["design_seconds"] for row in selected])
                    ),
                    "mean_design_diagnostics": {
                        name: float(
                            np.mean(
                                [row["design_diagnostics"][name] for row in selected]
                            )
                        )
                        for name in diagnostics
                    },
                }
            )
    return summaries


def _comparison_table(
    primary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["budget_multiplier"], row["method"]): row
        for row in primary
    }
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        multipliers = sorted(
            {
                multiplier
                for data, multiplier, _ in lookup
                if data == dataset
            }
        )
        for multiplier in multipliers:
            greedy_endpoint = lookup[
                (dataset, multiplier, "greedy_global_endpoint")
            ]["aggregate_rmse"]
            greedy_oracle = lookup[
                (dataset, multiplier, "greedy_global_oracle_size")
            ]["aggregate_rmse"]
            greedy_ratio = lookup[
                (dataset, multiplier, "greedy_official_ratio")
            ]["aggregate_rmse"]
            qstar_global = lookup[
                (dataset, multiplier, "qstar_orbit_global_endpoint")
            ]["aggregate_rmse"]
            qstar_ratio = lookup[
                (dataset, multiplier, "qstar_orbit_ratio")
            ]["aggregate_rmse"]
            cyclic_ratio = lookup[
                (dataset, multiplier, "cyclic_ratio")
            ]["aggregate_rmse"]
            rows.append(
                {
                    "dataset": dataset,
                    "budget_multiplier": multiplier,
                    "greedy_endpoint_rmse": greedy_endpoint,
                    "greedy_oracle_size_rmse": greedy_oracle,
                    "greedy_oracle_change_percent": 100.0
                    * (greedy_oracle / greedy_endpoint - 1.0),
                    "greedy_official_ratio_rmse": greedy_ratio,
                    "greedy_ratio_change_percent": 100.0
                    * (greedy_ratio / greedy_endpoint - 1.0),
                    "qstar_orbit_global_rmse": qstar_global,
                    "qstar_orbit_ratio_rmse": qstar_ratio,
                    "same_qstar_orbit_ratio_change_percent": 100.0
                    * (qstar_ratio / qstar_global - 1.0),
                    "formal_cyclic_ratio_rmse": cyclic_ratio,
                }
            )
    return rows


def run_debug_experiment(
    *,
    budget_multipliers: tuple[int, ...],
    repeats: int,
    sensitivity_repeats: int,
    sensitivity_multiplier: int,
    candidate_pool: int,
    mean_balance: float,
    processes: int,
    base_seed: int,
) -> dict[str, Any]:
    if repeats < 2 or sensitivity_repeats < 2:
        raise ValueError("repeat counts must be at least two")
    if processes < 1 or candidate_pool < 1:
        raise ValueError("processes and candidate pool must be positive")
    if mean_balance < 0:
        raise ValueError("mean balance must be nonnegative")
    if not budget_multipliers or any(value < 1 for value in budget_multipliers):
        raise ValueError("budget multipliers must be positive")
    for dataset in DATASETS:
        num_players = _num_players(dataset)
        _, probabilities, _ = inner_size_distribution(num_players)
        minimum_for_qstar_coverage = int(
            np.ceil(1.0 / float(probabilities.min()))
        )
        minimum_multiplier = max(
            num_players - 3, minimum_for_qstar_coverage
        )
        if min(budget_multipliers) < minimum_multiplier:
            raise ValueError(
                "every budget multiplier must support one cyclic orbit and "
                "one systematic q*-orbit for every inner size; need at least "
                f"{minimum_multiplier} for {dataset}"
            )

    primary_tasks = [
        (
            dataset,
            repeat,
            budget_multipliers,
            candidate_pool,
            mean_balance,
            base_seed,
        )
        for dataset in DATASETS
        for repeat in range(repeats)
    ]
    sensitivity_tasks = [
        (dataset, repeat, sensitivity_multiplier, base_seed + 1_000_003)
        for dataset in DATASETS
        for repeat in range(sensitivity_repeats)
    ]
    started = time.perf_counter()
    primary_raw: list[dict[str, Any]] = []
    sensitivity_raw: list[dict[str, Any]] = []
    if processes == 1:
        primary_raw = [_run_primary_repeat(*task) for task in primary_tasks]
        sensitivity_raw = [
            _run_sensitivity_repeat(*task) for task in sensitivity_tasks
        ]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(processes, len(primary_tasks)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_primary_repeat, *task): task[:2]
                for task in primary_tasks
            }
            for future in as_completed(futures):
                primary_raw.append(future.result())
        with ProcessPoolExecutor(
            max_workers=min(processes, len(sensitivity_tasks)),
            mp_context=context,
        ) as executor:
            futures = {
                executor.submit(_run_sensitivity_repeat, *task): task[:2]
                for task in sensitivity_tasks
            }
            for future in as_completed(futures):
                sensitivity_raw.append(future.result())

    primary_raw.sort(key=lambda row: (row["dataset"], row["repeat"]))
    sensitivity_raw.sort(key=lambda row: (row["dataset"], row["repeat"]))
    primary = _aggregate_primary(primary_raw)
    sensitivity = _aggregate_sensitivity(sensitivity_raw)
    paired_checks = [
        check for report in primary_raw for check in report["paired_checks"]
    ]
    max_ratio_identity_error = max(
        check["ratio_vs_stratified_max_abs"]
        for check in paired_checks
        if "ratio_vs_stratified_max_abs" in check
    )
    max_qstar_baseline_error = max(
        check["endpoint_vs_oracle_size_max_abs"]
        for check in paired_checks
        if "endpoint_vs_oracle_size_max_abs" in check
    )

    size_profiles: dict[str, Any] = {}
    for dataset in DATASETS:
        num_players = _num_players(dataset)
        boundary, _ = _boundary(dataset)
        exact = _exact_size_profile(dataset)
        endpoint = endpoint_size_profile(boundary)
        sizes, probabilities, _ = inner_size_distribution(num_players)
        mismatch = exact[sizes] - endpoint[sizes]
        size_profiles[dataset] = {
            "ofa_weighted_rms_endpoint_mismatch": float(
                np.sqrt(np.sum(probabilities * np.square(mismatch)))
            ),
            "ofa_weighted_mean_absolute_endpoint_mismatch": float(
                np.sum(probabilities * np.abs(mismatch))
            ),
            "maximum_absolute_endpoint_mismatch": float(
                np.max(np.abs(mismatch))
            ),
            "rows": [
                {
                    "size": int(size),
                    "ofa_probability": float(probability),
                    "exact_size_mean": float(exact[size]),
                    "endpoint_linear": float(endpoint[size]),
                    "mismatch": float(exact[size] - endpoint[size]),
                }
                for size, probability in zip(sizes, probabilities)
            ],
        }

    return {
        "status": "complete",
        "experiment": "inside_greedy_gap_paired_debug",
        "configuration": {
            "datasets": list(DATASETS),
            "budget_multipliers": list(budget_multipliers),
            "repeats": repeats,
            "sensitivity_repeats": sensitivity_repeats,
            "sensitivity_multiplier": sensitivity_multiplier,
            "candidate_pool": candidate_pool,
            "mean_balance": mean_balance,
            "processes": min(processes, len(primary_tasks)),
            "base_seed": base_seed,
            "sensitivity_configurations": [
                {
                    "configuration": name,
                    "candidate_pool": pool,
                    "mean_balance": balance,
                }
                for name, pool, balance in SENSITIVITY_CONFIGURATIONS
            ],
        },
        "definitions": {
            "greedy_endpoint": (
                "formal row-greedy q*-marginal design with the global HT "
                "estimator and endpoint-linear size baseline"
            ),
            "greedy_oracle_size": (
                "the identical Greedy rows and global HT estimator after "
                "subtracting the exact E[v(S)| |S|=s] control variate"
            ),
            "greedy_official_ratio": (
                "the identical Greedy rows aggregated with the upstream "
                "conditional-mean ratio estimator; all in/out strata are "
                "required, but the realized denominators are not balanced"
            ),
            "qstar_orbit_global_vs_ratio": (
                "the identical q*-marginal complete-orbit rows aggregated "
                "once globally and once by exact fixed-size strata"
            ),
            "cyclic_identity_check": (
                "strict balanced ratio and slice-wise linear aggregation on "
                "the identical formal INSIDE-Orbit rows"
            ),
        },
        "invariants": {
            "max_abs_cyclic_ratio_vs_stratified_linear": float(
                max_ratio_identity_error
            ),
            "cyclic_identity_tolerance_pass": bool(
                max_ratio_identity_error <= 1e-12
            ),
            "max_abs_qstar_orbit_endpoint_vs_oracle_size": float(
                max_qstar_baseline_error
            ),
            "qstar_orbit_size_baseline_cancellation_pass": bool(
                max_qstar_baseline_error <= 1e-12
            ),
        },
        "size_profiles": size_profiles,
        "comparison_table": _comparison_table(primary),
        "primary_summaries": primary,
        "sensitivity_summaries": sensitivity,
        "raw_primary": primary_raw,
        "raw_sensitivity": sensitivity_raw,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budget-multipliers",
        type=int,
        nargs="+",
        default=list(DEFAULT_BUDGET_MULTIPLIERS),
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--sensitivity-repeats", type=int, default=6)
    parser.add_argument("--sensitivity-multiplier", type=int, default=100)
    parser.add_argument("--candidate-pool", type=int, default=4)
    parser.add_argument("--mean-balance", type=float, default=0.1)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--base-seed", type=int, default=20260824)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/inside_greedy_gap_debug.json"),
    )
    args = parser.parse_args()
    report = run_debug_experiment(
        budget_multipliers=tuple(args.budget_multipliers),
        repeats=args.repeats,
        sensitivity_repeats=args.sensitivity_repeats,
        sensitivity_multiplier=args.sensitivity_multiplier,
        candidate_pool=args.candidate_pool,
        mean_balance=args.mean_balance,
        processes=args.processes,
        base_seed=args.base_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["comparison_table"], indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

"""Strictly summarize and plot the fixed-K INSIDE-Greedy lambda sweep.

The source reports are first passed through the independent raw-cell validator.
No runner headline aggregate is trusted directly.  The resulting figures answer
two deliberately separate questions at fixed candidate-pool size:

* does a non-zero ``lambda0`` improve RMSE relative to ``lambda0=0``?;
* how much Greedy design time does that change cost?

Every Airport/Voting-by-budget cell receives equal weight.  Coverage failures
are retained and suppress any RMSE aggregate that contains them.  Design time
is descriptive only.  The numerical lambda axis uses Matplotlib's symmetric-log
transform so that zero is displayed honestly alongside log-spaced positive
values.

``build_fixed_k_lambda_locked_selection`` emits the formal four-cell screening
artifact consumed by the existing held-out-validation workflow.  The registered
``(lambda0=1, K=4)`` method remains in its ranking as a baseline, but cannot win
the fixed-``K`` lambda search; the locked configuration is always one of the
predeclared ``K=64`` candidates.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.run_inside_greedy_hyperparameter_sweep import (
    REGISTERED_CANDIDATE_POOL,
    REGISTERED_LAMBDA0,
)
from experiments.validate_inside_greedy_hyperparameter_sweep import (
    _configuration_id,
    validate_report,
)


DATASETS = ("airport", "voting")
DATASET_LABELS = {
    "airport": "Airport",
    "voting": "Voting",
    "joint": "Joint",
}
DEFAULT_FIXED_K = 64
DEFAULT_REFERENCE_LAMBDA0 = 0.0
REGISTERED_DEFAULT = (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)

BLUE = "#2B6CB0"
ORANGE = "#C56A1A"
CHARCOAL = "#20252B"
MID_GREY = "#737B84"
LIGHT_GREY = "#D8DDE3"

SERIES_STYLE = {
    "airport": {
        "color": BLUE,
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": BLUE,
    },
    "voting": {
        "color": ORANGE,
        "marker": "^",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
    "joint": {
        "color": CHARCOAL,
        "marker": "D",
        "linestyle": "-.",
        "markerfacecolor": "white",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{path} does not contain an object")
    return value


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    top = float(numerator)
    bottom = float(denominator)
    if (
        not math.isfinite(top)
        or not math.isfinite(bottom)
        or top < 0.0
        or bottom < 0.0
    ):
        return None
    if bottom == 0.0:
        return 1.0 if top == 0.0 else None
    value = top / bottom
    return value if math.isfinite(value) and value >= 0.0 else None


def _geometric_mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.all(np.isfinite(array))
        or np.any(array < 0.0)
    ):
        return None
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _config_set(audit: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["evaluated_configurations"]
    }


def _load_validated_pair(
    airport_path: Path,
    voting_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    paths = {"airport": airport_path, "voting": voting_path}
    reports = {dataset: _read(path) for dataset, path in paths.items()}
    audits = {
        dataset: validate_report(report) for dataset, report in reports.items()
    }
    for dataset in DATASETS:
        _require(audits[dataset]["dataset"] == dataset, "dataset paths are swapped")
        _require(
            bool(audits[dataset]["all_rmse_and_aggregates_independently_recomputed"]),
            f"{dataset} report was not independently reconstructed",
        )
    first, second = audits["airport"], audits["voting"]
    for field in (
        "stage",
        "repeats",
        "base_seed",
        "budget_multipliers",
        "selected_budget_indices",
        "selected_budget_multipliers",
    ):
        _require(first[field] == second[field], f"datasets disagree on {field}")
    _require(
        _config_set(first) == _config_set(second),
        "Airport and Voting evaluate different Greedy configurations",
    )
    return reports, audits


def build_fixed_k_lambda_summary(
    airport_path: Path,
    voting_path: Path,
    *,
    fixed_k: int = DEFAULT_FIXED_K,
) -> dict[str, Any]:
    """Reduce independently reconstructed cells into a fixed-K lambda sweep."""
    _require(fixed_k >= 1, "fixed_k must be positive")
    reference_lambda0 = DEFAULT_REFERENCE_LAMBDA0
    reports, audits = _load_validated_pair(airport_path, voting_path)
    configurations = _config_set(audits["airport"])
    lambda_values = sorted(
        lambda0 for lambda0, pool in configurations if pool == fixed_k
    )
    _require(lambda_values, f"no K={fixed_k} configurations are present")
    reference_tuple = (float(reference_lambda0), int(fixed_k))
    _require(
        reference_tuple in configurations,
        f"lambda0={reference_lambda0:g},K={fixed_k} reference is absent",
    )
    _require(
        REGISTERED_DEFAULT in configurations,
        "registered default is absent from the validated reports",
    )

    selected_indices = [
        int(value) for value in audits["airport"]["selected_budget_indices"]
    ]
    selected_multipliers = [
        int(value)
        for value in audits["airport"]["selected_budget_multipliers"]
    ]
    expected_joint_cells = len(DATASETS) * len(selected_indices)

    def result_for(dataset: str, configuration: tuple[float, int]) -> Mapping[str, Any]:
        return audits[dataset]["recomputed_results_by_configuration"][
            _configuration_id(*configuration)
        ]

    rows: list[dict[str, Any]] = []
    for lambda0 in lambda_values:
        configuration = (float(lambda0), int(fixed_k))
        dataset_summaries: dict[str, Any] = {}
        joint_rmse_reference: list[float] = []
        joint_rmse_default: list[float] = []
        joint_rmse_orbit: list[float] = []
        joint_design_reference: list[float] = []
        candidate_failures_total = 0
        reference_failures_total = 0
        default_failures_total = 0

        for dataset in DATASETS:
            candidate = result_for(dataset, configuration)
            reference = result_for(dataset, reference_tuple)
            default = result_for(dataset, REGISTERED_DEFAULT)
            budget_rows: list[dict[str, Any]] = []
            dataset_rmse_reference: list[float] = []
            dataset_rmse_default: list[float] = []
            dataset_rmse_orbit: list[float] = []
            dataset_design_reference: list[float] = []
            candidate_failures = 0
            reference_failures = 0
            default_failures = 0

            for position, budget_index in enumerate(selected_indices):
                candidate_budget = candidate["budgets"][str(budget_index)]
                reference_budget = reference["budgets"][str(budget_index)]
                default_budget = default["budgets"][str(budget_index)]
                candidate_cell_failures = int(candidate_budget["coverage_failures"])
                reference_cell_failures = int(reference_budget["coverage_failures"])
                default_cell_failures = int(default_budget["coverage_failures"])
                candidate_failures += candidate_cell_failures
                reference_failures += reference_cell_failures
                default_failures += default_cell_failures

                ratio_reference = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    reference_budget["aggregate_rmse"],
                )
                ratio_default = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    default_budget["aggregate_rmse"],
                )
                ratio_orbit = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    candidate_budget["orbit_aggregate_rmse"],
                )
                if candidate_cell_failures or reference_cell_failures:
                    ratio_reference = None
                if candidate_cell_failures or default_cell_failures:
                    ratio_default = None
                if candidate_cell_failures:
                    ratio_orbit = None
                design_ratio = _safe_ratio(
                    candidate_budget["mean_design_seconds"],
                    reference_budget["mean_design_seconds"],
                )

                if ratio_reference is not None:
                    dataset_rmse_reference.append(ratio_reference)
                    joint_rmse_reference.append(ratio_reference)
                if ratio_default is not None:
                    dataset_rmse_default.append(ratio_default)
                    joint_rmse_default.append(ratio_default)
                if ratio_orbit is not None:
                    dataset_rmse_orbit.append(ratio_orbit)
                    joint_rmse_orbit.append(ratio_orbit)
                if design_ratio is not None:
                    dataset_design_reference.append(design_ratio)
                    joint_design_reference.append(design_ratio)

                budget_rows.append(
                    {
                        "budget_index": budget_index,
                        "budget_multiplier": selected_multipliers[position],
                        "candidate_aggregate_rmse": candidate_budget[
                            "aggregate_rmse"
                        ],
                        "reference_aggregate_rmse": reference_budget[
                            "aggregate_rmse"
                        ],
                        "registered_default_aggregate_rmse": default_budget[
                            "aggregate_rmse"
                        ],
                        "inside_orbit_aggregate_rmse": candidate_budget[
                            "orbit_aggregate_rmse"
                        ],
                        "rmse_ratio_to_lambda0_reference": ratio_reference,
                        "rmse_ratio_to_registered_default": ratio_default,
                        "rmse_ratio_to_inside_orbit": ratio_orbit,
                        "candidate_mean_design_seconds": candidate_budget[
                            "mean_design_seconds"
                        ],
                        "reference_mean_design_seconds": reference_budget[
                            "mean_design_seconds"
                        ],
                        "registered_default_mean_design_seconds": default_budget[
                            "mean_design_seconds"
                        ],
                        "design_time_ratio_to_lambda0_reference": design_ratio,
                        "candidate_coverage_failures": candidate_cell_failures,
                        "reference_coverage_failures": reference_cell_failures,
                        "registered_default_coverage_failures": (
                            default_cell_failures
                        ),
                        # The strict validator rejects an Orbit coverage failure.
                        "inside_orbit_coverage_failures": 0,
                    }
                )

            candidate_failures_total += candidate_failures
            reference_failures_total += reference_failures
            default_failures_total += default_failures
            full_reference = (
                candidate_failures == 0
                and reference_failures == 0
                and len(dataset_rmse_reference) == len(selected_indices)
            )
            full_default = (
                candidate_failures == 0
                and default_failures == 0
                and len(dataset_rmse_default) == len(selected_indices)
            )
            full_orbit = (
                candidate_failures == 0
                and len(dataset_rmse_orbit) == len(selected_indices)
            )
            dataset_summaries[dataset] = {
                "rmse_ratio_to_lambda0_reference": (
                    _geometric_mean(dataset_rmse_reference)
                    if full_reference
                    else None
                ),
                "rmse_ratio_to_registered_default": (
                    _geometric_mean(dataset_rmse_default) if full_default else None
                ),
                "rmse_ratio_to_inside_orbit": (
                    _geometric_mean(dataset_rmse_orbit) if full_orbit else None
                ),
                "design_time_ratio_to_lambda0_reference": (
                    _geometric_mean(dataset_design_reference)
                    if len(dataset_design_reference) == len(selected_indices)
                    else None
                ),
                "candidate_coverage_failures": candidate_failures,
                "reference_coverage_failures": reference_failures,
                "registered_default_coverage_failures": default_failures,
                "budget_cells": budget_rows,
            }

        reference_available = (
            candidate_failures_total == 0
            and reference_failures_total == 0
            and len(joint_rmse_reference) == expected_joint_cells
        )
        selection_eligible = (
            candidate_failures_total == 0
            and default_failures_total == 0
            and len(joint_rmse_default) == expected_joint_cells
        )
        rows.append(
            {
                "configuration_id": _configuration_id(*configuration),
                "lambda0": float(lambda0),
                "candidate_pool": int(fixed_k),
                "is_lambda0_reference": configuration == reference_tuple,
                "selection_eligible": selection_eligible,
                "reference_comparison_available": reference_available,
                "candidate_coverage_failures": candidate_failures_total,
                "reference_coverage_failures": reference_failures_total,
                "registered_default_coverage_failures": default_failures_total,
                "rmse_ratio_to_lambda0_reference": {
                    "airport": dataset_summaries["airport"][
                        "rmse_ratio_to_lambda0_reference"
                    ],
                    "voting": dataset_summaries["voting"][
                        "rmse_ratio_to_lambda0_reference"
                    ],
                    "joint": (
                        _geometric_mean(joint_rmse_reference)
                        if reference_available
                        else None
                    ),
                },
                "design_time_ratio_to_lambda0_reference": {
                    "airport": dataset_summaries["airport"][
                        "design_time_ratio_to_lambda0_reference"
                    ],
                    "voting": dataset_summaries["voting"][
                        "design_time_ratio_to_lambda0_reference"
                    ],
                    "joint": (
                        _geometric_mean(joint_design_reference)
                        if len(joint_design_reference) == expected_joint_cells
                        else None
                    ),
                },
                "rmse_ratio_to_registered_default": {
                    "airport": dataset_summaries["airport"][
                        "rmse_ratio_to_registered_default"
                    ],
                    "voting": dataset_summaries["voting"][
                        "rmse_ratio_to_registered_default"
                    ],
                    "joint": (
                        _geometric_mean(joint_rmse_default)
                        if selection_eligible
                        else None
                    ),
                },
                "rmse_ratio_to_inside_orbit": {
                    "airport": dataset_summaries["airport"][
                        "rmse_ratio_to_inside_orbit"
                    ],
                    "voting": dataset_summaries["voting"][
                        "rmse_ratio_to_inside_orbit"
                    ],
                    "joint": (
                        _geometric_mean(joint_rmse_orbit)
                        if (
                            candidate_failures_total == 0
                            and len(joint_rmse_orbit) == expected_joint_cells
                        )
                        else None
                    ),
                },
                "datasets": dataset_summaries,
            }
        )

    source_paths = {
        "airport": {
            "path": str(airport_path.resolve()),
            "sha256": _sha256(airport_path),
        },
        "voting": {
            "path": str(voting_path.resolve()),
            "sha256": _sha256(voting_path),
        },
    }
    return {
        "status": "validated_fixed_k_lambda_summary",
        "experiment": "inside_greedy_fixed_k_lambda_sweep",
        "fixed_candidate_pool": int(fixed_k),
        "reference": {
            "lambda0": float(reference_lambda0),
            "candidate_pool": int(fixed_k),
            "configuration_id": _configuration_id(*reference_tuple),
        },
        "aggregation": {
            "rmse_within_budget": "sqrt(mean repeat-level squared RMSE)",
            "dataset_ratio": "geometric mean over selected budgets",
            "joint_ratio": "equal-weight geometric mean over dataset-budget cells",
            "coverage_policy": (
                "retain every failure; suppress RMSE aggregates containing a failure"
            ),
            "design_time_is_descriptive_only": True,
            "design_time_comparison": "same-report wall time; utility time excluded",
        },
        "protocol": {
            "stage": audits["airport"]["stage"],
            "repeats": audits["airport"]["repeats"],
            "base_seed": audits["airport"]["base_seed"],
            "budget_indices": selected_indices,
            "budget_multipliers": selected_multipliers,
            "lambda0_values": lambda_values,
            "raw_cells_audited": {
                dataset: len(reports[dataset].get("raw_cells", []))
                for dataset in DATASETS
            },
        },
        "source_validation": {
            "all_rmse_and_aggregates_independently_recomputed_from_raw_cells": True,
            "coverage_failures_retained": True,
            "all_greedy_hyperparameters_share_size_schedules": all(
                bool(audits[dataset]["all_greedy_hyperparameters_share_size_schedules"])
                for dataset in DATASETS
            ),
            "all_greedy_hyperparameters_share_relabels": all(
                bool(audits[dataset]["all_greedy_hyperparameters_share_relabels"])
                for dataset in DATASETS
            ),
            "sources": source_paths,
        },
        "rows": rows,
    }


def build_fixed_k_lambda_locked_selection(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock lambda0 among the fixed-K rows with the registered four-cell rule."""
    protocol = summary["protocol"]
    _require(protocol["stage"] == "screening", "locking requires screening reports")
    budget_indices = [int(value) for value in protocol["budget_indices"]]
    cell_count = len(DATASETS) * len(budget_indices)
    _require(cell_count == 4, "locking requires 2 datasets x 2 budgets")
    _require(
        list(protocol["budget_multipliers"]) == [500, 2000],
        "registered locking requires budget multipliers 500 and 2000",
    )
    fixed_k = int(summary["fixed_candidate_pool"])
    fixed_rows = list(summary["rows"])
    _require(fixed_rows, "fixed-K lambda summary has no rows")

    ranking: list[dict[str, Any]] = []
    for row in fixed_rows:
        cell_default: dict[str, float | None] = {}
        cell_orbit: dict[str, float | None] = {}
        design_seconds: list[float] = []
        failures = 0
        for dataset in DATASETS:
            for point in row["datasets"][dataset]["budget_cells"]:
                key = f"{dataset}:budget_index={int(point['budget_index'])}"
                cell_default[key] = point["rmse_ratio_to_registered_default"]
                cell_orbit[key] = point["rmse_ratio_to_inside_orbit"]
                failures += int(point["candidate_coverage_failures"])
                failures += int(point["registered_default_coverage_failures"])
                failures += int(point["inside_orbit_coverage_failures"])
                design_seconds.append(float(point["candidate_mean_design_seconds"]))
        default_values = [
            float(value) for value in cell_default.values() if value is not None
        ]
        orbit_values = [
            float(value) for value in cell_orbit.values() if value is not None
        ]
        eligible = (
            failures == 0
            and len(default_values) == cell_count
        )
        ranking.append(
            {
                "configuration_id": row["configuration_id"],
                "lambda0": float(row["lambda0"]),
                "candidate_pool": fixed_k,
                "is_registered_default": False,
                "selection_eligible": eligible,
                "eligible_for_lambda_lock": eligible,
                "coverage_failures": failures,
                "dataset_geometric_mean_rmse_ratio_to_default": {
                    dataset: row["datasets"][dataset][
                        "rmse_ratio_to_registered_default"
                    ]
                    for dataset in DATASETS
                },
                "cell_rmse_ratios_to_default": cell_default,
                "cell_rmse_ratios_to_orbit": cell_orbit,
                "joint_geometric_mean_rmse_ratio_to_default": (
                    _geometric_mean(default_values) if eligible else None
                ),
                "joint_geometric_mean_rmse_ratio_to_orbit": (
                    _geometric_mean(orbit_values)
                    if len(orbit_values) == cell_count
                    else None
                ),
                "worst_cell_rmse_ratio_to_default": (
                    max(default_values) if eligible else None
                ),
                "worst_cell_rmse_ratio_to_orbit": (
                    max(orbit_values)
                    if len(orbit_values) == cell_count
                    else None
                ),
                "mean_design_seconds": (
                    float(np.mean(design_seconds))
                    if len(design_seconds) == cell_count
                    else None
                ),
            }
        )

    # Add the registered default as a comparable baseline and as an allowed
    # future validation configuration.  It is deliberately ineligible to win
    # a search whose declared candidate family is lambda0 at fixed K.
    template = fixed_rows[0]
    default_cells: dict[str, float | None] = {}
    default_orbit_cells: dict[str, float | None] = {}
    default_dataset_scores: dict[str, float | None] = {}
    default_design_seconds: list[float] = []
    default_failures = 0
    for dataset in DATASETS:
        dataset_default_cells: list[float] = []
        for point in template["datasets"][dataset]["budget_cells"]:
            key = f"{dataset}:budget_index={int(point['budget_index'])}"
            failures = int(point["registered_default_coverage_failures"])
            orbit_failures = int(point["inside_orbit_coverage_failures"])
            default_failures += failures + orbit_failures
            default_ratio = 1.0 if failures == 0 else None
            orbit_ratio = _safe_ratio(
                point["registered_default_aggregate_rmse"],
                point["inside_orbit_aggregate_rmse"],
            )
            if failures or orbit_failures:
                orbit_ratio = None
            default_cells[key] = default_ratio
            default_orbit_cells[key] = orbit_ratio
            if default_ratio is not None:
                dataset_default_cells.append(default_ratio)
            default_design_seconds.append(
                float(point["registered_default_mean_design_seconds"])
            )
        default_dataset_scores[dataset] = (
            _geometric_mean(dataset_default_cells)
            if len(dataset_default_cells) == len(budget_indices)
            else None
        )
    default_orbit_values = [
        float(value) for value in default_orbit_cells.values() if value is not None
    ]
    default_eligible = (
        default_failures == 0
        and len(default_cells) == cell_count
    )
    ranking.append(
        {
            "configuration_id": _configuration_id(*REGISTERED_DEFAULT),
            "lambda0": float(REGISTERED_DEFAULT[0]),
            "candidate_pool": int(REGISTERED_DEFAULT[1]),
            "is_registered_default": True,
            "selection_eligible": default_eligible,
            "eligible_for_lambda_lock": False,
            "coverage_failures": default_failures,
            "dataset_geometric_mean_rmse_ratio_to_default": default_dataset_scores,
            "cell_rmse_ratios_to_default": default_cells,
            "cell_rmse_ratios_to_orbit": default_orbit_cells,
            "joint_geometric_mean_rmse_ratio_to_default": (
                1.0 if default_eligible else None
            ),
            "joint_geometric_mean_rmse_ratio_to_orbit": (
                _geometric_mean(default_orbit_values)
                if len(default_orbit_values) == cell_count
                else None
            ),
            "worst_cell_rmse_ratio_to_default": 1.0 if default_eligible else None,
            "worst_cell_rmse_ratio_to_orbit": (
                max(default_orbit_values)
                if len(default_orbit_values) == cell_count
                else None
            ),
            "mean_design_seconds": (
                float(np.mean(default_design_seconds))
                if len(default_design_seconds) == cell_count
                else None
            ),
        }
    )

    candidates = [row for row in ranking if row["eligible_for_lambda_lock"]]
    _require(candidates, "every fixed-K lambda configuration failed coverage")
    candidates.sort(
        key=lambda row: (
            float(row["joint_geometric_mean_rmse_ratio_to_default"]),
            float(row["worst_cell_rmse_ratio_to_default"]),
            float(row["lambda0"]),
        )
    )
    best = candidates[0]
    candidate_rank = {
        str(row["configuration_id"]): index + 1
        for index, row in enumerate(candidates)
    }
    for row in ranking:
        row["rank_among_fixed_k_lambda_candidates"] = candidate_rank.get(
            str(row["configuration_id"])
        )
    ranking.sort(
        key=lambda row: (
            not bool(row["selection_eligible"]),
            math.inf
            if row["joint_geometric_mean_rmse_ratio_to_default"] is None
            else float(row["joint_geometric_mean_rmse_ratio_to_default"]),
            math.inf
            if row["worst_cell_rmse_ratio_to_default"] is None
            else float(row["worst_cell_rmse_ratio_to_default"]),
            int(row["candidate_pool"]),
            float(row["lambda0"]),
        )
    )
    locked_keys = (
        "configuration_id",
        "lambda0",
        "candidate_pool",
        "joint_geometric_mean_rmse_ratio_to_default",
        "joint_geometric_mean_rmse_ratio_to_orbit",
        "worst_cell_rmse_ratio_to_default",
        "worst_cell_rmse_ratio_to_orbit",
        "mean_design_seconds",
        "rank_among_fixed_k_lambda_candidates",
    )
    return {
        "status": "configuration_locked",
        "experiment": "inside_greedy_fixed_k_lambda_joint_selection",
        "selection_rule": {
            "candidate_constraint": f"candidate_pool K={fixed_k}",
            "primary": (
                "minimize the equal-weight mean log aggregate-RMSE ratio "
                "to the registered default across 2 datasets x 2 budgets"
            ),
            "zero_ratio_policy": "an exact zero is valid and ranks first",
            "coverage_policy": (
                "eliminate any candidate with a candidate, registered-default, "
                "or Orbit coverage failure"
            ),
            "tie_breaks": [
                "lower worst dataset-budget RMSE ratio to default",
                "smaller lambda0",
            ],
            "registered_default_is_comparator_not_a_lambda_candidate": True,
            "design_time_is_descriptive_not_a_selection_tie_break": True,
            "cell_count": cell_count,
            "datasets": list(DATASETS),
            "budget_indices": budget_indices,
            "budget_multipliers": list(protocol["budget_multipliers"]),
        },
        "screening_sources": summary["source_validation"]["sources"],
        "screening_contract": {
            "base_seed": protocol["base_seed"],
            "repeats": protocol["repeats"],
            "budget_multipliers": list(protocol["budget_multipliers"]),
            "all_rmse_and_aggregates_independently_recomputed": True,
            "greedy_size_schedules_and_relabels_paired_across_hyperparameters": True,
            "orbit_is_equal_call_but_independently_randomized": True,
        },
        "locked_configuration": {key: best[key] for key in locked_keys},
        "ranking": ranking,
        "holdout_validation": None,
    }


def _figure_setup(title: str, subtitle: str) -> tuple[Any, Any]:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 18,
            "xtick.labelsize": 12,
            "ytick.labelsize": 13,
            "legend.fontsize": 11.5,
        }
    )
    fig, axis = plt.subplots(figsize=(6.3, 6.0))
    fig.suptitle(title, x=0.14, y=0.975, ha="left", fontsize=18, weight="semibold")
    fig.text(
        0.14,
        0.927,
        subtitle,
        ha="left",
        va="top",
        fontsize=11.3,
        color=MID_GREY,
    )
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.19, top=0.79)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=1.0, alpha=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.45)
        spine.set_color(CHARCOAL)
    axis.tick_params(width=1.25, length=5, color=CHARCOAL)
    return fig, axis


def _lambda_label(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1_000 or abs(value) < 0.001:
        return f"{value:.0e}"
    return f"{value:g}"


def _set_lambda_axis(axis: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    lambdas = [float(row["lambda0"]) for row in rows]
    positive = [value for value in lambdas if value > 0.0]
    _require(positive, "a symlog lambda plot needs at least one positive value")
    linthresh = min(positive)
    axis.set_xscale("symlog", base=10, linthresh=linthresh, linscale=1.0)
    axis.set_xticks(lambdas)
    axis.set_xticklabels([_lambda_label(value) for value in lambdas])
    for label in axis.get_xticklabels():
        label.set_rotation(32)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    axis.set_xlabel(r"First-moment weight $\lambda_0$")


def _plot_series(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> tuple[list[float], list[tuple[str, float, int]]]:
    valid_values: list[float] = []
    failures: list[tuple[str, float, int]] = []
    for series in DATASETS + ("joint",):
        style = SERIES_STYLE[series]
        x_values: list[float] = []
        y_values: list[float] = []
        for row in rows:
            lambda0 = float(row["lambda0"])
            value = row[field][series]
            x_values.append(lambda0)
            if value is None:
                y_values.append(float("nan"))
                if field == "rmse_ratio_to_lambda0_reference":
                    if series == "joint":
                        count = int(row["candidate_coverage_failures"]) + int(
                            row["reference_coverage_failures"]
                        )
                    else:
                        dataset = row["datasets"][series]
                        count = int(dataset["candidate_coverage_failures"]) + int(
                            dataset["reference_coverage_failures"]
                        )
                    failures.append((series, lambda0, count))
            else:
                numeric = float(value)
                y_values.append(numeric)
                valid_values.append(numeric)
        axis.plot(
            x_values,
            y_values,
            label=DATASET_LABELS[series],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.8,
            markersize=7.5,
            linewidth=2.7,
        )
    return valid_values, failures


def _export(fig: Any, output_png: Path) -> tuple[Path, Path]:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_png, output_pdf


def plot_lambda_rmse_ratios(
    summary: Mapping[str, Any], output_png: Path
) -> tuple[Path, Path]:
    """Plot equal-cell RMSE ratios to lambda0=0 on an honest symlog axis."""
    rows = sorted(summary["rows"], key=lambda row: float(row["lambda0"]))
    fixed_k = int(summary["fixed_candidate_pool"])
    reference_lambda0 = float(summary["reference"]["lambda0"])
    fig, axis = _figure_setup(
        "Lambda sweep RMSE",
        rf"$K={fixed_k}$; ratio to $\lambda_0={reference_lambda0:g}$; symlog x, focused y; <1 better",
    )
    values, failures = _plot_series(
        axis, rows, field="rmse_ratio_to_lambda0_reference"
    )
    axis.axhline(1.0, color=MID_GREY, linewidth=1.7, linestyle=":", zorder=0)
    _set_lambda_axis(axis, rows)
    axis.set_ylabel("RMSE ratio")
    plotted = values + [1.0]
    lower, upper = min(plotted), max(plotted)
    span = max(upper - lower, 0.08)
    y_min = max(0.0, lower - 0.18 * span)
    y_max = upper + 0.28 * span
    axis.set_ylim(y_min, y_max)
    if failures:
        offsets = {"airport": 0.00, "voting": 0.035, "joint": 0.07}
        for series, lambda0, count in failures:
            y = y_max - (0.10 + offsets[series]) * span
            axis.scatter(
                [lambda0],
                [y],
                marker="X",
                s=78,
                linewidth=1.4,
                color=SERIES_STYLE[series]["color"],
                zorder=5,
            )
            axis.annotate(
                f"{count} fail" if count else "undefined",
                (lambda0, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9.3,
                color=SERIES_STYLE[series]["color"],
            )
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    return _export(fig, output_png)


def plot_lambda_design_time_ratios(
    summary: Mapping[str, Any], output_png: Path
) -> tuple[Path, Path]:
    """Plot descriptive design-time ratios to lambda0=0."""
    rows = sorted(summary["rows"], key=lambda row: float(row["lambda0"]))
    fixed_k = int(summary["fixed_candidate_pool"])
    reference_lambda0 = float(summary["reference"]["lambda0"])
    fig, axis = _figure_setup(
        "Lambda sweep design time",
        rf"$K={fixed_k}$; ratio to $\lambda_0={reference_lambda0:g}$; symlog x; utility time excluded",
    )
    values, _ = _plot_series(
        axis, rows, field="design_time_ratio_to_lambda0_reference"
    )
    axis.axhline(1.0, color=MID_GREY, linewidth=1.7, linestyle=":", zorder=0)
    _set_lambda_axis(axis, rows)
    axis.set_ylabel("Design-time ratio")
    axis.set_ylim(0.0, max(values + [1.0]) * 1.13)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.2,
    )
    return _export(fig, output_png)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport", required=True, type=Path)
    parser.add_argument("--voting", required=True, type=Path)
    parser.add_argument("--fixed-k", type=int, default=DEFAULT_FIXED_K)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--rmse-plot", required=True, type=Path)
    parser.add_argument("--time-plot", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_fixed_k_lambda_summary(
        args.airport,
        args.voting,
        fixed_k=args.fixed_k,
    )
    _atomic_write_json(args.output, summary)
    if args.selection_output is not None:
        _atomic_write_json(
            args.selection_output,
            build_fixed_k_lambda_locked_selection(summary),
        )
    plot_lambda_rmse_ratios(summary, args.rmse_plot)
    plot_lambda_design_time_ratios(summary, args.time_plot)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

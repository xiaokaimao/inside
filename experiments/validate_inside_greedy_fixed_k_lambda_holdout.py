"""Validate the locked fixed-``K`` lambda holdout without re-running selection.

This module is deliberately separate from the generic hyperparameter selector.
The fixed-``K`` screening artifact has a different candidate constraint and
selection rule, so feeding it to the generic ``--locked-selection`` CLI would
incorrectly compare two non-identical selection contracts.

The registered holdout compares the screening-locked
``(lambda0=1/16, K=64)`` configuration with ``(lambda0=0, K=64)``, the
registered ``(lambda0=1, K=4)`` default, and INSIDE-Orbit at multipliers
``500, 2000, 10000``.  Both reports are reconstructed from raw cells by the
independent sweep validator before any result is used.

Bootstrap uncertainty is paired within each dataset-budget cell: one resampled
repeat index is applied to every compared method in that cell.  Budget index is
part of the experiment's seed sequence, so different budget cells are
resampled independently and then combined draw-by-draw for the six-cell joint
summary.
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
    _seed_for,
    validate_report,
)


DATASETS = ("airport", "voting")
LOCKED_CONFIGURATION = (1.0 / 16.0, 64)
ZERO_CONFIGURATION = (0.0, 64)
REGISTERED_DEFAULT = (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)
HOLDOUT_MULTIPLIERS = (500, 2000, 10_000)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_BASE_SEED = 20261031

COMPARATORS = {
    "lambda0_zero_k64": ZERO_CONFIGURATION,
    "registered_default": REGISTERED_DEFAULT,
    "inside_orbit": None,
}

PLOT_STYLE = {
    "airport": {
        "label": "Airport",
        "color": "#2B6CB0",
        "marker": "o",
        "linestyle": "-",
        "markerfacecolor": "#2B6CB0",
    },
    "voting": {
        "label": "Voting",
        "color": "#C56A1A",
        "marker": "^",
        "linestyle": "--",
        "markerfacecolor": "white",
    },
}
CHARCOAL = "#20252B"
MID_GREY = "#737B84"
LIGHT_GREY = "#D8DDE3"


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


def _configuration_set(audit: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["evaluated_configurations"]
    }


def _requested_configuration_set(
    audit: Mapping[str, Any],
) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["requested_configurations"]
    }


def _method_seed_set(
    *, base_seed: int, repeats: int, budget_indices: Sequence[int]
) -> set[int]:
    return {
        _seed_for(base_seed, repeat, budget_index, method_index)
        for repeat in range(repeats)
        for budget_index in budget_indices
        for method_index in (0, 1)
    }


def _point_ratio(numerator: float, denominator: float) -> float | None:
    _require(
        math.isfinite(numerator)
        and math.isfinite(denominator)
        and numerator >= 0.0
        and denominator >= 0.0,
        "RMSE ratio inputs must be finite and nonnegative",
    )
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else None
    return numerator / denominator


def _geometric_mean(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        return None
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _repeat_matrix(
    budget_rows: Sequence[Mapping[str, Any]], *, path: str
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for budget_position, budget in enumerate(budget_rows):
        try:
            values = np.asarray(budget["rmse_by_repeat"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{path}.budget={budget_position} has a nonnumeric repeat array"
            ) from error
        _require(
            values.ndim == 1
            and values.size > 0
            and np.all(np.isfinite(values))
            and np.all(values >= 0.0),
            f"{path}.budget={budget_position} has an invalid repeat array",
        )
        rows.append(values)
    _require(
        len({row.size for row in rows}) == 1,
        f"{path} budgets have different repeat counts",
    )
    return np.vstack(rows)


def _cell_rms_draws(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Return one aggregate-RMSE draw per paired cell resample."""
    _require(values.ndim == 1, "cell values must be a repeat vector")
    _require(indices.ndim == 2, "cell indices must be sample x repeat")
    _require(
        indices.shape[1] == values.size,
        "cell draw size differs from the registered repeat count",
    )
    return np.sqrt(np.mean(np.square(values[indices]), axis=1))


def _ratio_draws(
    numerator: np.ndarray, denominator: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    _require(numerator.shape == denominator.shape, "bootstrap shapes disagree")
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    positive = denominator > 0.0
    result[positive] = numerator[positive] / denominator[positive]
    both_zero = (denominator == 0.0) & (numerator == 0.0)
    result[both_zero] = 1.0
    invalid = (denominator == 0.0) & (numerator > 0.0)
    return result, invalid


def _row_geometric_mean(values: np.ndarray) -> np.ndarray:
    _require(values.ndim == 2 and values.shape[1] > 0, "invalid draw matrix")
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = np.all(np.isfinite(values) & (values >= 0.0), axis=1)
    if not np.any(valid):
        return result
    subset = values[valid]
    zero = np.any(subset == 0.0, axis=1)
    reduced = np.zeros(subset.shape[0], dtype=np.float64)
    positive = ~zero
    if np.any(positive):
        reduced[positive] = np.exp(np.mean(np.log(subset[positive]), axis=1))
    result[valid] = reduced
    return result


def _interval(values: np.ndarray) -> list[float] | None:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return None
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return [float(lower), float(upper)]


def _bootstrap_summary(
    ratio_draws: np.ndarray, invalid: np.ndarray
) -> dict[str, Any]:
    _require(ratio_draws.ndim == 2, "ratio draws must be sample x budget")
    _require(invalid.shape == ratio_draws.shape, "invalid mask shape disagrees")
    geometric = _row_geometric_mean(ratio_draws)
    invalid_joint = ~np.isfinite(geometric)
    return {
        "budget_ratio_95_percent_ci": [
            _interval(ratio_draws[:, position])
            for position in range(ratio_draws.shape[1])
        ],
        "geometric_mean_ratio_95_percent_ci": _interval(geometric),
        "probability_locked_rmse_is_lower": (
            float(np.mean(geometric < 1.0))
            if not np.any(invalid_joint)
            else None
        ),
        "invalid_denominator_draws_by_budget": [
            int(value) for value in np.sum(invalid, axis=0)
        ],
        "invalid_joint_draws": int(np.sum(invalid_joint)),
    }


def _load_pair(
    airport_path: Path, voting_path: Path
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    paths = {"airport": airport_path, "voting": voting_path}
    reports = {dataset: _read(path) for dataset, path in paths.items()}
    audits = {
        dataset: validate_report(report) for dataset, report in reports.items()
    }
    for dataset in DATASETS:
        audit = audits[dataset]
        _require(audit["dataset"] == dataset, "dataset paths are swapped")
        _require(audit["stage"] == "validation", f"{dataset} is not validation")
        _require(
            bool(audit["all_rmse_and_aggregates_independently_recomputed"]),
            f"{dataset} was not reconstructed from raw cells",
        )
        _require(
            bool(audit["holdout_stage_eligible_pending_cross_report_seed_audit"]),
            f"{dataset} is not holdout-stage eligible",
        )
    first, second = audits["airport"], audits["voting"]
    for field in (
        "repeats",
        "base_seed",
        "budget_multipliers",
        "selected_budget_indices",
        "selected_budget_multipliers",
    ):
        _require(first[field] == second[field], f"datasets disagree on {field}")
    _require(
        _configuration_set(first) == _configuration_set(second),
        "datasets evaluate different configurations",
    )
    return reports, audits


def _method_budget_rows(
    audit: Mapping[str, Any], configuration: tuple[float, int] | None
) -> list[dict[str, Any]]:
    selected_indices = [int(value) for value in audit["selected_budget_indices"]]
    locked_result = audit["recomputed_results_by_configuration"][
        _configuration_id(*LOCKED_CONFIGURATION)
    ]
    if configuration is None:
        return [
            {
                "aggregate_rmse": locked_result["budgets"][str(index)][
                    "orbit_aggregate_rmse"
                ],
                "coverage_failures": 0,
                "rmse_by_repeat": locked_result["budgets"][str(index)][
                    "orbit_rmse_by_repeat"
                ],
            }
            for index in selected_indices
        ]
    result = audit["recomputed_results_by_configuration"][
        _configuration_id(*configuration)
    ]
    return [dict(result["budgets"][str(index)]) for index in selected_indices]


def build_fixed_k_lambda_holdout_validation(
    locked_selection_path: Path,
    airport_validation_path: Path,
    voting_validation_path: Path,
) -> dict[str, Any]:
    """Build the independently checked three-budget holdout artifact."""
    selection = _read(locked_selection_path)
    _require(
        selection.get("status") == "configuration_locked"
        and selection.get("holdout_validation") is None,
        "selection artifact is not a pre-holdout lock",
    )
    locked = selection.get("locked_configuration")
    _require(isinstance(locked, Mapping), "locked configuration is absent")
    locked_tuple = (float(locked["lambda0"]), int(locked["candidate_pool"]))
    _require(
        locked_tuple == LOCKED_CONFIGURATION,
        "holdout requires the screening-locked lambda0=1/16,K=64",
    )
    screened = {
        (float(row["lambda0"]), int(row["candidate_pool"]))
        for row in selection.get("ranking", [])
    }
    _require(
        {LOCKED_CONFIGURATION, ZERO_CONFIGURATION, REGISTERED_DEFAULT}.issubset(
            screened
        ),
        "screening artifact lacks a registered holdout comparator",
    )

    reports, audits = _load_pair(
        airport_validation_path, voting_validation_path
    )
    validation_configs = {
        LOCKED_CONFIGURATION, ZERO_CONFIGURATION, REGISTERED_DEFAULT
    }
    for dataset in DATASETS:
        audit = audits[dataset]
        _require(
            _configuration_set(audit) == validation_configs,
            f"{dataset} validation must contain exactly locked, zero, and default",
        )
        requested = _requested_configuration_set(audit)
        _require(
            {LOCKED_CONFIGURATION, ZERO_CONFIGURATION}.issubset(requested)
            and requested.issubset(validation_configs),
            f"{dataset} did not predeclare locked and zero comparators",
        )
        _require(
            audit["selected_budget_indices"]
            == list(range(len(audit["budget_multipliers"]))),
            f"{dataset} does not cover its full budget grid",
        )
    protocol = audits["airport"]
    _require(
        tuple(protocol["selected_budget_multipliers"]) == HOLDOUT_MULTIPLIERS,
        "holdout requires multipliers 500, 2000, 10000",
    )

    screening = selection.get("screening_contract")
    _require(isinstance(screening, Mapping), "screening contract is absent")
    screening_seed = int(screening["base_seed"])
    validation_seed = int(protocol["base_seed"])
    _require(validation_seed != screening_seed, "validation reuses screening seed")
    screening_indices = [
        int(value) for value in selection["selection_rule"]["budget_indices"]
    ]
    screening_seeds = _method_seed_set(
        base_seed=screening_seed,
        repeats=int(screening["repeats"]),
        budget_indices=screening_indices,
    )
    validation_seeds = _method_seed_set(
        base_seed=validation_seed,
        repeats=int(protocol["repeats"]),
        budget_indices=[int(value) for value in protocol["selected_budget_indices"]],
    )
    _require(
        not screening_seeds.intersection(validation_seeds),
        "screening and validation derived method seeds overlap",
    )

    method_configurations: dict[str, tuple[float, int] | None] = {
        "locked": LOCKED_CONFIGURATION,
        **COMPARATORS,
    }
    methods: dict[str, dict[str, list[dict[str, Any]]]] = {
        dataset: {
            method: _method_budget_rows(audits[dataset], configuration)
            for method, configuration in method_configurations.items()
        }
        for dataset in DATASETS
    }
    # All configuration entries must carry the one and only Orbit aggregate.
    for dataset in DATASETS:
        orbit_reference = methods[dataset]["inside_orbit"]
        for configuration in (
            LOCKED_CONFIGURATION,
            ZERO_CONFIGURATION,
            REGISTERED_DEFAULT,
        ):
            result = audits[dataset]["recomputed_results_by_configuration"][
                _configuration_id(*configuration)
            ]
            for position, budget_index in enumerate(
                audits[dataset]["selected_budget_indices"]
            ):
                budget = result["budgets"][str(budget_index)]
                _require(
                    budget["orbit_aggregate_rmse"]
                    == orbit_reference[position]["aggregate_rmse"]
                    and budget["orbit_rmse_by_repeat"]
                    == orbit_reference[position]["rmse_by_repeat"],
                    f"{dataset} Orbit reconstruction differs across configs",
                )

    total_calls: dict[str, list[int]] = {}
    for dataset in DATASETS:
        configuration = reports[dataset]["configuration"]
        all_totals = configuration["all_total_utility_call_budgets"]
        total_calls[dataset] = [
            int(all_totals[index])
            for index in audits[dataset]["selected_budget_indices"]
        ]

    coverage_by_method: dict[str, dict[str, Any]] = {}
    for method in method_configurations:
        per_dataset = {
            dataset: sum(
                int(row["coverage_failures"]) for row in methods[dataset][method]
            )
            for dataset in DATASETS
        }
        coverage_by_method[method] = {
            "by_dataset": per_dataset,
            "total": sum(per_dataset.values()),
        }

    repeats = int(protocol["repeats"])
    cell_indices: dict[tuple[str, int], np.ndarray] = {}
    for dataset_position, dataset in enumerate(DATASETS):
        for budget_index in protocol["selected_budget_indices"]:
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        BOOTSTRAP_BASE_SEED,
                        validation_seed,
                        dataset_position,
                        int(budget_index),
                    ]
                )
            )
            cell_indices[(dataset, int(budget_index))] = rng.integers(
                0, repeats, size=(BOOTSTRAP_SAMPLES, repeats)
            )

    comparisons: dict[str, Any] = {}
    for comparator_name in COMPARATORS:
        comparison_datasets: dict[str, Any] = {}
        joint_points: list[float | None] = []
        joint_draw_parts: list[np.ndarray] = []
        joint_invalid_parts: list[np.ndarray] = []
        comparison_complete = True
        for dataset in DATASETS:
            locked_rows = methods[dataset]["locked"]
            comparator_rows = methods[dataset][comparator_name]
            budget_cells: list[dict[str, Any]] = []
            cell_points: list[float | None] = []
            locked_failures = sum(
                int(row["coverage_failures"]) for row in locked_rows
            )
            comparator_failures = sum(
                int(row["coverage_failures"]) for row in comparator_rows
            )
            complete = locked_failures == 0 and comparator_failures == 0
            for position, budget_index in enumerate(
                audits[dataset]["selected_budget_indices"]
            ):
                locked_budget = locked_rows[position]
                comparator_budget = comparator_rows[position]
                ratio: float | None = None
                if (
                    int(locked_budget["coverage_failures"]) == 0
                    and int(comparator_budget["coverage_failures"]) == 0
                ):
                    ratio = _point_ratio(
                        float(locked_budget["aggregate_rmse"]),
                        float(comparator_budget["aggregate_rmse"]),
                    )
                cell_points.append(ratio)
                joint_points.append(ratio)
                budget_cells.append(
                    {
                        "budget_index": int(budget_index),
                        "budget_multiplier": int(
                            protocol["selected_budget_multipliers"][position]
                        ),
                        "total_utility_calls": total_calls[dataset][position],
                        "locked_aggregate_rmse": locked_budget["aggregate_rmse"],
                        "comparator_aggregate_rmse": comparator_budget[
                            "aggregate_rmse"
                        ],
                        "locked_over_comparator_rmse_ratio": ratio,
                        "locked_coverage_failures": int(
                            locked_budget["coverage_failures"]
                        ),
                        "comparator_coverage_failures": int(
                            comparator_budget["coverage_failures"]
                        ),
                    }
                )

            bootstrap: dict[str, Any] | None = None
            if complete:
                locked_matrix = _repeat_matrix(
                    locked_rows, path=f"{dataset}.locked"
                )
                comparator_matrix = _repeat_matrix(
                    comparator_rows, path=f"{dataset}.{comparator_name}"
                )
                _require(
                    locked_matrix.shape
                    == (len(HOLDOUT_MULTIPLIERS), repeats)
                    == comparator_matrix.shape,
                    f"{dataset} repeat matrices violate holdout dimensions",
                )
                locked_draw_columns: list[np.ndarray] = []
                comparator_draw_columns: list[np.ndarray] = []
                for position, budget_index in enumerate(
                    audits[dataset]["selected_budget_indices"]
                ):
                    indices = cell_indices[(dataset, int(budget_index))]
                    locked_draw_columns.append(
                        _cell_rms_draws(locked_matrix[position], indices)
                    )
                    comparator_draw_columns.append(
                        _cell_rms_draws(comparator_matrix[position], indices)
                    )
                locked_draws = np.column_stack(locked_draw_columns)
                comparator_draws = np.column_stack(comparator_draw_columns)
                ratio_draws, invalid = _ratio_draws(
                    locked_draws, comparator_draws
                )
                bootstrap = _bootstrap_summary(ratio_draws, invalid)
                for position, cell in enumerate(budget_cells):
                    cell["paired_repeat_bootstrap_95_percent_ci"] = bootstrap[
                        "budget_ratio_95_percent_ci"
                    ][position]
                    cell["invalid_denominator_bootstrap_draws"] = bootstrap[
                        "invalid_denominator_draws_by_budget"
                    ][position]
                joint_draw_parts.append(ratio_draws)
                joint_invalid_parts.append(invalid)
            else:
                comparison_complete = False
                for cell in budget_cells:
                    cell["paired_repeat_bootstrap_95_percent_ci"] = None
                    cell["invalid_denominator_bootstrap_draws"] = None

            comparison_datasets[dataset] = {
                "complete_coverage": complete,
                "locked_coverage_failures": locked_failures,
                "comparator_coverage_failures": comparator_failures,
                "geometric_mean_locked_over_comparator_rmse_ratio": (
                    _geometric_mean(cell_points) if complete else None
                ),
                "paired_repeat_bootstrap": bootstrap,
                "budget_cells": budget_cells,
            }

        joint_bootstrap: dict[str, Any] | None = None
        if comparison_complete and len(joint_draw_parts) == len(DATASETS):
            joint_matrix = np.concatenate(joint_draw_parts, axis=1)
            joint_invalid = np.concatenate(joint_invalid_parts, axis=1)
            joint_bootstrap = _bootstrap_summary(joint_matrix, joint_invalid)
        comparisons[comparator_name] = {
            "ratio_direction": "locked RMSE / comparator RMSE; below 1 is better",
            "complete_coverage": comparison_complete,
            "joint_equal_cell_geometric_mean_rmse_ratio": (
                _geometric_mean(joint_points) if comparison_complete else None
            ),
            "paired_repeat_bootstrap": joint_bootstrap,
            "datasets": comparison_datasets,
        }

    any_failures = any(
        item["total"] > 0 for item in coverage_by_method.values()
    )
    sources = {
        "locked_selection": {
            "path": str(locked_selection_path.resolve()),
            "sha256": _sha256(locked_selection_path),
        },
        "airport_validation": {
            "path": str(airport_validation_path.resolve()),
            "sha256": _sha256(airport_validation_path),
        },
        "voting_validation": {
            "path": str(voting_validation_path.resolve()),
            "sha256": _sha256(voting_validation_path),
        },
    }
    return {
        "status": (
            "holdout_inconclusive_coverage_failure"
            if any_failures
            else "holdout_validated"
        ),
        "experiment": "inside_greedy_fixed_k_lambda_holdout",
        "locked_configuration": {
            "lambda0": LOCKED_CONFIGURATION[0],
            "candidate_pool": LOCKED_CONFIGURATION[1],
            "configuration_id": _configuration_id(*LOCKED_CONFIGURATION),
        },
        "comparators": {
            "lambda0_zero_k64": {
                "lambda0": ZERO_CONFIGURATION[0],
                "candidate_pool": ZERO_CONFIGURATION[1],
            },
            "registered_default": {
                "lambda0": REGISTERED_DEFAULT[0],
                "candidate_pool": REGISTERED_DEFAULT[1],
            },
            "inside_orbit": {"estimator": "INSIDE-Orbit"},
        },
        "protocol": {
            "base_seed": validation_seed,
            "repeats": repeats,
            "budget_indices": list(protocol["selected_budget_indices"]),
            "budget_multipliers": list(protocol["selected_budget_multipliers"]),
            "screening_and_validation_seed_sets_disjoint": True,
            "raw_reports_independently_reconstructed": True,
            "equal_utility_call_comparisons": True,
        },
        "bootstrap": {
            "unit": "repeat within each dataset-budget cell",
            "pairing": (
                "same resampled repeat indices for all compared methods within "
                "one dataset-budget cell"
            ),
            "cell_resampling": (
                "dataset-budget cells independently resampled because budget "
                "index enters the experiment seed sequence"
            ),
            "joint_summary": "draw-wise equal-weight geometric mean over 6 cells",
            "samples": BOOTSTRAP_SAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "base_seed": BOOTSTRAP_BASE_SEED,
        },
        "coverage_failures_by_method": coverage_by_method,
        "comparisons": comparisons,
        "sources": sources,
    }


def plot_locked_vs_zero_holdout(
    validation: Mapping[str, Any], output_png: Path
) -> tuple[Path, Path]:
    """Plot held-out locked/zero RMSE ratios with paired-repeat intervals."""
    _require(
        output_png.suffix.lower() == ".png",
        "zero-comparison plot path must end in .png",
    )
    comparison = validation.get("comparisons", {}).get("lambda0_zero_k64")
    _require(isinstance(comparison, Mapping), "zero-lambda comparison is absent")
    _require(
        bool(comparison.get("complete_coverage")),
        "cannot plot an incomplete zero-lambda comparison",
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 17,
            "axes.titlesize": 19,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
        }
    )
    fig, axis = plt.subplots(figsize=(6.25, 5.95))
    fig.suptitle(
        r"Held-out $\lambda_0$ comparison",
        x=0.15,
        y=0.975,
        ha="left",
        fontsize=19,
        weight="semibold",
    )
    fig.text(
        0.15,
        0.925,
        r"$K=64$;  $(\lambda_0=1/16)/(\lambda_0=0)$;  <1 better",
        ha="left",
        va="top",
        fontsize=11.5,
        color=MID_GREY,
    )
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.80)

    plotted_values: list[float] = [1.0]
    expected_x = np.asarray(HOLDOUT_MULTIPLIERS, dtype=np.float64)
    for dataset in DATASETS:
        dataset_result = comparison["datasets"][dataset]
        cells = list(dataset_result["budget_cells"])
        _require(
            len(cells) == len(HOLDOUT_MULTIPLIERS),
            f"{dataset} zero comparison has the wrong budget count",
        )
        x = np.asarray(
            [float(cell["budget_multiplier"]) for cell in cells],
            dtype=np.float64,
        )
        _require(
            np.array_equal(x, expected_x),
            f"{dataset} zero comparison has the wrong budget multipliers",
        )
        y = np.asarray(
            [float(cell["locked_over_comparator_rmse_ratio"]) for cell in cells],
            dtype=np.float64,
        )
        intervals = [
            cell["paired_repeat_bootstrap_95_percent_ci"] for cell in cells
        ]
        _require(
            np.all(np.isfinite(y))
            and np.all(y >= 0.0)
            and all(
                isinstance(interval, Sequence)
                and len(interval) == 2
                and all(math.isfinite(float(value)) for value in interval)
                and float(interval[0]) <= float(interval[1])
                for interval in intervals
            ),
            f"{dataset} has invalid point estimates or intervals",
        )
        lower = np.asarray([float(interval[0]) for interval in intervals])
        upper = np.asarray([float(interval[1]) for interval in intervals])
        _require(
            np.all(lower >= 0.0),
            f"{dataset} bootstrap interval contains a negative ratio",
        )
        style = PLOT_STYLE[dataset]
        # Draw percentile intervals directly: unlike symmetric error bars,
        # this remains honest if a bootstrap interval does not contain its
        # original point estimate.
        axis.vlines(
            x,
            lower,
            upper,
            color=style["color"],
            linewidth=2.0,
            alpha=0.85,
            zorder=2,
        )
        cap_factor = 1.045
        for x_value, low, high in zip(x, lower, upper, strict=True):
            axis.plot(
                [x_value / cap_factor, x_value * cap_factor],
                [low, low],
                color=style["color"],
                linewidth=2.0,
                solid_capstyle="round",
                zorder=2,
            )
            axis.plot(
                [x_value / cap_factor, x_value * cap_factor],
                [high, high],
                color=style["color"],
                linewidth=2.0,
                solid_capstyle="round",
                zorder=2,
            )
        axis.plot(
            x,
            y,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.8,
            markersize=8.5,
            linewidth=3.0,
            zorder=3,
        )
        plotted_values.extend(y.tolist())
        plotted_values.extend(lower.tolist())
        plotted_values.extend(upper.tolist())

    axis.axhline(1.0, color=MID_GREY, linestyle=":", linewidth=1.8, zorder=1)
    axis.set_xscale("log")
    axis.set_xticks(expected_x)
    axis.set_xticklabels(["500", "2,000", "10,000"])
    axis.minorticks_off()
    axis.set_xlabel("Budget multiplier")
    axis.set_ylabel("RMSE ratio")
    lower_bound = min(plotted_values)
    upper_bound = max(plotted_values)
    span = max(upper_bound - lower_bound, 0.06)
    axis.set_ylim(max(0.0, lower_bound - 0.18 * span), upper_bound + 0.22 * span)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=1.0, alpha=0.85)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=2,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.5,
    )
    axis.tick_params(width=1.25, length=5, color=CHARCOAL)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.45)
        spine.set_color(CHARCOAL)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_png, output_pdf


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
    parser.add_argument("--locked-selection", required=True, type=Path)
    parser.add_argument("--airport-validation", required=True, type=Path)
    parser.add_argument("--voting-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--zero-comparison-plot",
        type=Path,
        help="optional PNG path; a same-stem PDF is also written",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_fixed_k_lambda_holdout_validation(
        args.locked_selection,
        args.airport_validation,
        args.voting_validation,
    )
    if args.zero_comparison_plot is not None:
        png, pdf = plot_locked_vs_zero_holdout(
            result, args.zero_comparison_plot
        )
        result["plots"] = {
            "locked_vs_lambda0_zero": [str(png), str(pdf)]
        }
    _atomic_write_json(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "locked": result["locked_configuration"],
                "joint_ratios": {
                    name: comparison[
                        "joint_equal_cell_geometric_mean_rmse_ratio"
                    ]
                    for name, comparison in result["comparisons"].items()
                },
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

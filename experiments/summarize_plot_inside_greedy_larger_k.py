"""Independently summarize and plot the fixed-lambda larger-K ablation.

The input reports are the JSON artifacts emitted by
``run_inside_greedy_hyperparameter_sweep``.  This module intentionally does
not consume their convenient headline aggregates.  It first runs the strict
report validator, which reconstructs exact-game RMSE, physical-call parity,
coverage, size schedules, relabels, and aggregates from ``raw_cells``.  The
larger-K summary is then reduced from those independently reconstructed
budget-level values.

Coverage failures are never dropped.  A dataset or joint RMSE ratio is absent
when any contributing repeat failed strict OFA-ratio coverage, while the
failure counts remain in the JSON summary and are marked on the RMSE figure.
Design time remains descriptive and never changes eligibility or selects K.

An optional second Airport/Voting pair extends the K grid.  The merge is
accepted only when its repeated K=64, registered-default, and INSIDE-Orbit raw
cells match the base reports in every deterministic field.  Elapsed timing is
calibrated through the duplicated K=64 cells.  ``--selection-output`` emits a
formal four-cell mean-log locked-selection artifact suitable for the existing
held-out-validation attachment workflow.
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
FIXED_LAMBDA0 = 0.0
K16_REFERENCE = (FIXED_LAMBDA0, 16)
MERGE_ANCHOR = (FIXED_LAMBDA0, 64)
EXTENSION_TARGET = (FIXED_LAMBDA0, 128)
REGISTERED_DEFAULT = (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)
REFERENCE_CHOICES = ("k16", "registered-default")
ALLOWED_OVERLAP_TIMING_DIFFERENCES = ("design_seconds", "utility_seconds")

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


def _safe_ratio(numerator: Any, denominator: Any) -> float | None:
    if numerator is None or denominator is None:
        return None
    top = float(numerator)
    bottom = float(denominator)
    if not math.isfinite(top) or not math.isfinite(bottom) or bottom <= 0.0:
        return None
    result = top / bottom
    return result if math.isfinite(result) and result >= 0.0 else None


def _config_set(audit: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["evaluated_configurations"]
    }


def _load_validated_pair(
    airport_path: Path, voting_path: Path
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
    airport, voting = audits["airport"], audits["voting"]
    for field in (
        "stage",
        "repeats",
        "base_seed",
        "budget_multipliers",
        "selected_budget_indices",
        "selected_budget_multipliers",
    ):
        _require(
            airport[field] == voting[field],
            f"Airport and Voting disagree on {field}",
        )
    fixed_k_sets = []
    for dataset in DATASETS:
        fixed_k_sets.append(
            {
                pool
                for lambda0, pool in _config_set(audits[dataset])
                if lambda0 == FIXED_LAMBDA0
            }
        )
    _require(
        fixed_k_sets[0] == fixed_k_sets[1] and fixed_k_sets[0],
        "Airport and Voting have different or empty lambda0=0 K grids",
    )
    return reports, audits


def _raw_cell_identity(cell: Mapping[str, Any]) -> tuple[Any, ...]:
    method = str(cell.get("method"))
    repeat = int(cell.get("repeat"))
    budget_index = int(cell.get("budget_index"))
    if method == "inside_orbit":
        return (method, repeat, budget_index)
    _require(method == "inside_greedy", "raw cell has an unknown method")
    return (
        method,
        float(cell.get("lambda0")),
        int(cell.get("candidate_pool")),
        repeat,
        budget_index,
    )


def _raw_cell_index(report: Mapping[str, Any]) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    raw = report.get("raw_cells")
    _require(isinstance(raw, list) and raw, "report raw_cells is empty")
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for position, value in enumerate(raw):
        _require(isinstance(value, Mapping), f"raw_cells[{position}] is not an object")
        key = _raw_cell_identity(value)
        _require(key not in result, f"duplicate raw-cell identity {key}")
        result[key] = value
    return result


def _deterministic_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only elapsed wall-clock fields from a raw-cell comparison."""
    return {
        key: value
        for key, value in cell.items()
        if key not in ALLOWED_OVERLAP_TIMING_DIFFERENCES
    }


def _first_difference(left: Any, right: Any, path: str = "raw_cell") -> str | None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return f"{path}.keys"
        for key in sorted(left_keys):
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}.length"
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = _first_difference(
                left_item, right_item, f"{path}[{index}]"
            )
            if difference is not None:
                return difference
        return None
    return None if left == right and type(left) is type(right) else path


def _protocol_signature(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: audit[field]
        for field in (
            "stage",
            "num_players",
            "repeats",
            "base_seed",
            "budget_multipliers",
            "selected_budget_indices",
            "selected_budget_multipliers",
        )
    }


def _strict_overlap_audit(
    base_reports: Mapping[str, Mapping[str, Any]],
    base_audits: Mapping[str, Mapping[str, Any]],
    extension_reports: Mapping[str, Mapping[str, Any]],
    extension_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require deterministic equality for every overlapping repeated cell."""
    result: dict[str, Any] = {}
    for dataset in DATASETS:
        _require(
            _protocol_signature(base_audits[dataset])
            == _protocol_signature(extension_audits[dataset]),
            f"{dataset} base and extension protocols differ",
        )
        overlap = _config_set(base_audits[dataset]).intersection(
            _config_set(extension_audits[dataset])
        )
        _require(
            MERGE_ANCHOR in overlap,
            f"{dataset} reports do not overlap at lambda0=0,K=64",
        )
        _require(
            REGISTERED_DEFAULT in overlap,
            f"{dataset} reports do not share the registered default",
        )
        base_index = _raw_cell_index(base_reports[dataset])
        extension_index = _raw_cell_index(extension_reports[dataset])

        def relevant(key: tuple[Any, ...]) -> bool:
            if key[0] == "inside_orbit":
                return True
            return (float(key[1]), int(key[2])) in overlap

        base_keys = {key for key in base_index if relevant(key)}
        extension_keys = {key for key in extension_index if relevant(key)}
        _require(
            base_keys == extension_keys,
            f"{dataset} overlap raw-cell identities differ",
        )
        timing_ratios: dict[str, list[float]] = {
            field: [] for field in ALLOWED_OVERLAP_TIMING_DIFFERENCES
        }
        for key in sorted(base_keys, key=repr):
            base_cell = base_index[key]
            extension_cell = extension_index[key]
            difference = _first_difference(
                _deterministic_cell(base_cell),
                _deterministic_cell(extension_cell),
            )
            _require(
                difference is None,
                f"{dataset} overlap mismatch at {key}: {difference}",
            )
            for field in ALLOWED_OVERLAP_TIMING_DIFFERENCES:
                base_time = float(base_cell[field])
                extension_time = float(extension_cell[field])
                if base_time > 0.0 and extension_time >= 0.0:
                    timing_ratios[field].append(extension_time / base_time)
        result[dataset] = {
            "overlapping_greedy_configurations": [
                {
                    "lambda0": lambda0,
                    "candidate_pool": pool,
                    "configuration_id": _configuration_id(lambda0, pool),
                }
                for lambda0, pool in sorted(overlap, key=lambda item: (item[1], item[0]))
            ],
            "inside_orbit_compared": True,
            "raw_cells_compared": len(base_keys),
            "deterministic_raw_cell_fields_exact": True,
            "estimate_rmse_seed_budget_coverage_and_hashes_exact": True,
            "allowed_elapsed_timing_differences": list(
                ALLOWED_OVERLAP_TIMING_DIFFERENCES
            ),
            "extension_over_base_timing_geometric_mean": {
                field: _geometric_mean(values)
                for field, values in timing_ratios.items()
            },
        }
    return {
        "passed": True,
        "merge_anchor": {
            "lambda0": MERGE_ANCHOR[0],
            "candidate_pool": MERGE_ANCHOR[1],
        },
        "datasets": result,
    }


def _reference_tuple(reference: str) -> tuple[float, int]:
    if reference == "k16":
        return K16_REFERENCE
    if reference == "registered-default":
        return REGISTERED_DEFAULT
    raise ValueError(f"reference must be one of {REFERENCE_CHOICES}")


def _reference_label(reference: str) -> str:
    if reference == "k16":
        return r"$K=16$"
    return r"default $(\lambda_0=1, K=4)$"


def build_larger_k_summary(
    airport_path: Path,
    voting_path: Path,
    *,
    extension_airport_path: Path | None = None,
    extension_voting_path: Path | None = None,
    reference: str = "k16",
) -> dict[str, Any]:
    """Recompute the fixed-``lambda0=0`` larger-K comparison.

    Dataset scores are geometric means over all selected budget cells.  The
    joint score equally weights every Airport/Voting-by-budget cell, rather
    than weighting datasets by player count or utility-call volume.
    """
    _require(
        (extension_airport_path is None) == (extension_voting_path is None),
        "extension Airport and Voting reports must be supplied together",
    )
    base_reports, base_audits = _load_validated_pair(airport_path, voting_path)
    report_sets: dict[str, dict[str, Mapping[str, Any]]] = {
        "base": base_reports,
    }
    audit_sets: dict[str, dict[str, dict[str, Any]]] = {"base": base_audits}
    overlap_audit: dict[str, Any] | None = None
    extension_enabled = extension_airport_path is not None
    if extension_enabled:
        assert extension_airport_path is not None
        assert extension_voting_path is not None
        extension_reports, extension_audits = _load_validated_pair(
            extension_airport_path, extension_voting_path
        )
        report_sets["extension"] = extension_reports
        audit_sets["extension"] = extension_audits
        overlap_audit = _strict_overlap_audit(
            base_reports,
            base_audits,
            extension_reports,
            extension_audits,
        )
        for dataset in DATASETS:
            base_fixed = {
                item for item in _config_set(base_audits[dataset]) if item[0] == 0.0
            }
            extension_fixed = {
                item
                for item in _config_set(extension_audits[dataset])
                if item[0] == 0.0
            }
            _require(
                {K16_REFERENCE, MERGE_ANCHOR}.issubset(base_fixed),
                f"{dataset} base report is missing K=16 or K=64",
            )
            _require(
                {MERGE_ANCHOR, EXTENSION_TARGET}.issubset(extension_fixed),
                f"{dataset} extension report is missing K=64 or K=128",
            )

    reference_tuple = _reference_tuple(reference)
    for dataset in DATASETS:
        _require(
            any(
                reference_tuple in _config_set(audits[dataset])
                for audits in audit_sets.values()
            ),
            f"{dataset} is missing the {reference} reference configuration",
        )

    k_sources: dict[int, str] = {}
    for source in ("base", "extension"):
        if source not in audit_sets:
            continue
        for lambda0, pool in _config_set(audit_sets[source]["airport"]):
            if lambda0 == FIXED_LAMBDA0 and pool not in k_sources:
                k_sources[pool] = source
    k_values = sorted(k_sources)
    if extension_enabled:
        _require(
            k_values == [16, 32, 64, 128],
            "merged larger-K grid must be exactly K=16,32,64,128",
        )
    selected_indices = [
        int(value) for value in base_audits["airport"]["selected_budget_indices"]
    ]
    selected_multipliers = [
        int(value)
        for value in base_audits["airport"]["selected_budget_multipliers"]
    ]

    def result_for(
        source: str, dataset: str, configuration: tuple[float, int]
    ) -> Mapping[str, Any]:
        config_id = _configuration_id(*configuration)
        return audit_sets[source][dataset][
            "recomputed_results_by_configuration"
        ][config_id]

    def source_for_reference(candidate_source: str, dataset: str) -> str:
        if reference_tuple in _config_set(audit_sets[candidate_source][dataset]):
            return candidate_source
        for source, audits in audit_sets.items():
            if reference_tuple in _config_set(audits[dataset]):
                return source
        raise ValueError(f"{dataset} has no source for reference {reference_tuple}")

    def scaled_design_seconds(
        source: str,
        dataset: str,
        budget_index: int,
        seconds: Any,
    ) -> tuple[float | None, str]:
        value = float(seconds)
        if not extension_enabled or source == "base":
            return value, "same-report"
        extension_anchor = result_for(
            "extension", dataset, MERGE_ANCHOR
        )["budgets"][str(budget_index)]["mean_design_seconds"]
        base_anchor = result_for("base", dataset, MERGE_ANCHOR)["budgets"][
            str(budget_index)
        ]["mean_design_seconds"]
        scale = _safe_ratio(base_anchor, extension_anchor)
        if scale is None:
            return None, "unavailable-K64-bridge"
        return value * scale, "K64-overlap-bridge-to-base-clock"

    rows: list[dict[str, Any]] = []
    for pool in k_values:
        config_tuple = (FIXED_LAMBDA0, pool)
        config_id = _configuration_id(*config_tuple)
        candidate_source = k_sources[pool]
        dataset_summaries: dict[str, Any] = {}
        joint_rmse_ratios: list[float] = []
        joint_default_ratios: list[float] = []
        joint_orbit_ratios: list[float] = []
        joint_design_ratios: list[float] = []
        total_failures = 0
        total_reference_failures = 0

        for dataset in DATASETS:
            reference_source = source_for_reference(candidate_source, dataset)
            candidate = result_for(candidate_source, dataset, config_tuple)
            baseline = result_for(reference_source, dataset, reference_tuple)
            budget_rows: list[dict[str, Any]] = []
            dataset_rmse_ratios: list[float] = []
            dataset_default_ratios: list[float] = []
            dataset_orbit_ratios: list[float] = []
            dataset_design_ratios: list[float] = []
            dataset_failures = 0
            dataset_reference_failures = 0
            for position, budget_index in enumerate(selected_indices):
                candidate_budget = candidate["budgets"][str(budget_index)]
                baseline_budget = baseline["budgets"][str(budget_index)]
                default_budget = result_for(
                    candidate_source, dataset, REGISTERED_DEFAULT
                )["budgets"][str(budget_index)]
                failures = int(candidate_budget["coverage_failures"])
                reference_failures = int(baseline_budget["coverage_failures"])
                default_failures = int(default_budget["coverage_failures"])
                dataset_failures += failures
                dataset_reference_failures += reference_failures
                rmse_ratio = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    baseline_budget["aggregate_rmse"],
                )
                default_ratio = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    default_budget["aggregate_rmse"],
                )
                orbit_ratio = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    candidate_budget["orbit_aggregate_rmse"],
                )
                candidate_design_scaled, candidate_time_method = scaled_design_seconds(
                    candidate_source,
                    dataset,
                    budget_index,
                    candidate_budget["mean_design_seconds"],
                )
                baseline_design_scaled, baseline_time_method = scaled_design_seconds(
                    reference_source,
                    dataset,
                    budget_index,
                    baseline_budget["mean_design_seconds"],
                )
                design_ratio = _safe_ratio(
                    candidate_design_scaled, baseline_design_scaled
                )
                design_ratio_method = (
                    "same-report"
                    if candidate_source == reference_source
                    else "K64-overlap-bridge"
                )
                if failures or reference_failures:
                    rmse_ratio = None
                if failures or default_failures:
                    default_ratio = None
                if failures:
                    orbit_ratio = None
                if rmse_ratio is not None:
                    dataset_rmse_ratios.append(rmse_ratio)
                    joint_rmse_ratios.append(rmse_ratio)
                if default_ratio is not None:
                    dataset_default_ratios.append(default_ratio)
                    joint_default_ratios.append(default_ratio)
                if orbit_ratio is not None:
                    dataset_orbit_ratios.append(orbit_ratio)
                    joint_orbit_ratios.append(orbit_ratio)
                if design_ratio is not None:
                    dataset_design_ratios.append(design_ratio)
                    joint_design_ratios.append(design_ratio)
                budget_rows.append(
                    {
                        "budget_index": budget_index,
                        "budget_multiplier": selected_multipliers[position],
                        "candidate_aggregate_rmse": candidate_budget[
                            "aggregate_rmse"
                        ],
                        "reference_aggregate_rmse": baseline_budget[
                            "aggregate_rmse"
                        ],
                        "rmse_ratio_to_reference": rmse_ratio,
                        "rmse_ratio_to_registered_default": default_ratio,
                        "rmse_ratio_to_inside_orbit": orbit_ratio,
                        "registered_default_aggregate_rmse": default_budget[
                            "aggregate_rmse"
                        ],
                        "inside_orbit_aggregate_rmse": candidate_budget[
                            "orbit_aggregate_rmse"
                        ],
                        "candidate_mean_design_seconds": candidate_budget[
                            "mean_design_seconds"
                        ],
                        "reference_mean_design_seconds": baseline_budget[
                            "mean_design_seconds"
                        ],
                        "candidate_mean_design_seconds_on_base_clock": (
                            candidate_design_scaled
                        ),
                        "reference_mean_design_seconds_on_base_clock": (
                            baseline_design_scaled
                        ),
                        "registered_default_mean_design_seconds": default_budget[
                            "mean_design_seconds"
                        ],
                        "design_time_ratio_to_reference": design_ratio,
                        "design_time_ratio_method": design_ratio_method,
                        "candidate_design_time_scaling": candidate_time_method,
                        "reference_design_time_scaling": baseline_time_method,
                        "candidate_source": candidate_source,
                        "reference_source": reference_source,
                        "coverage_failures": failures,
                        "reference_coverage_failures": reference_failures,
                        "registered_default_coverage_failures": default_failures,
                        "inside_orbit_coverage_failures": 0,
                    }
                )
            total_failures += dataset_failures
            total_reference_failures += dataset_reference_failures
            all_budget_rmse_valid = (
                dataset_failures == 0
                and dataset_reference_failures == 0
                and len(dataset_rmse_ratios) == len(selected_indices)
            )
            dataset_summaries[dataset] = {
                "rmse_ratio_to_reference": (
                    _geometric_mean(dataset_rmse_ratios)
                    if all_budget_rmse_valid
                    else None
                ),
                "design_time_ratio_to_reference": (
                    _geometric_mean(dataset_design_ratios)
                    if len(dataset_design_ratios) == len(selected_indices)
                    else None
                ),
                "rmse_ratio_to_registered_default": (
                    _geometric_mean(dataset_default_ratios)
                    if dataset_failures == 0
                    and len(dataset_default_ratios) == len(selected_indices)
                    else None
                ),
                "rmse_ratio_to_inside_orbit": (
                    _geometric_mean(dataset_orbit_ratios)
                    if dataset_failures == 0
                    and len(dataset_orbit_ratios) == len(selected_indices)
                    else None
                ),
                "coverage_failures": dataset_failures,
                "reference_coverage_failures": dataset_reference_failures,
                "budget_cells": budget_rows,
            }

        expected_joint_cells = len(DATASETS) * len(selected_indices)
        eligible = (
            total_failures == 0
            and total_reference_failures == 0
            and len(joint_rmse_ratios) == expected_joint_cells
        )
        rows.append(
            {
                "lambda0": FIXED_LAMBDA0,
                "candidate_pool": pool,
                "configuration_id": config_id,
                "source_report_set": candidate_source,
                "selection_eligible": eligible,
                "coverage_failures": total_failures,
                "reference_coverage_failures": total_reference_failures,
                "rmse_ratio_to_reference": {
                    "airport": dataset_summaries["airport"][
                        "rmse_ratio_to_reference"
                    ],
                    "voting": dataset_summaries["voting"][
                        "rmse_ratio_to_reference"
                    ],
                    "joint": (
                        _geometric_mean(joint_rmse_ratios) if eligible else None
                    ),
                },
                "design_time_ratio_to_reference": {
                    "airport": dataset_summaries["airport"][
                        "design_time_ratio_to_reference"
                    ],
                    "voting": dataset_summaries["voting"][
                        "design_time_ratio_to_reference"
                    ],
                    "joint": (
                        _geometric_mean(joint_design_ratios)
                        if len(joint_design_ratios) == expected_joint_cells
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
                        _geometric_mean(joint_default_ratios)
                        if eligible and len(joint_default_ratios) == expected_joint_cells
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
                        _geometric_mean(joint_orbit_ratios)
                        if eligible and len(joint_orbit_ratios) == expected_joint_cells
                        else None
                    ),
                },
                "datasets": dataset_summaries,
            }
        )

    if extension_enabled:
        assert extension_airport_path is not None
        assert extension_voting_path is not None
        source_paths: dict[str, Any] = {
            "base": {
                "airport": {
                    "path": str(airport_path.resolve()),
                    "sha256": _sha256(airport_path),
                },
                "voting": {
                    "path": str(voting_path.resolve()),
                    "sha256": _sha256(voting_path),
                },
            },
            "extension": {
                "airport": {
                    "path": str(extension_airport_path.resolve()),
                    "sha256": _sha256(extension_airport_path),
                },
                "voting": {
                    "path": str(extension_voting_path.resolve()),
                    "sha256": _sha256(extension_voting_path),
                },
            },
        }
        raw_cells_audited: dict[str, Any] = {
            source: {
                dataset: len(report_sets[source][dataset].get("raw_cells", []))
                for dataset in DATASETS
            }
            for source in ("base", "extension")
        }
    else:
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
        raw_cells_audited = {
            dataset: len(base_reports[dataset].get("raw_cells", []))
            for dataset in DATASETS
        }

    return {
        "status": "validated_larger_k_summary",
        "experiment": "inside_greedy_fixed_lambda0_larger_k",
        "fixed_lambda0": FIXED_LAMBDA0,
        "reference": {
            "choice": reference,
            "configuration_id": _configuration_id(*reference_tuple),
            "lambda0": reference_tuple[0],
            "candidate_pool": reference_tuple[1],
            "label": _reference_label(reference),
        },
        "aggregation": {
            "rmse_within_budget": "sqrt(mean repeat-level squared RMSE)",
            "dataset_ratio": "geometric mean over selected budgets",
            "joint_ratio": "equal-weight geometric mean over dataset-budget cells",
            "coverage_policy": (
                "retain every failure; suppress RMSE aggregates containing a failure"
            ),
            "design_time_is_descriptive_only": True,
            "cross_run_design_time_calibration": (
                "per-dataset-budget K64 overlap bridge to the base-report clock"
                if extension_enabled
                else "not needed"
            ),
        },
        "protocol": {
            "stage": base_audits["airport"]["stage"],
            "repeats": base_audits["airport"]["repeats"],
            "base_seed": base_audits["airport"]["base_seed"],
            "budget_indices": selected_indices,
            "budget_multipliers": selected_multipliers,
            "k_values": k_values,
            "raw_cells_audited": raw_cells_audited,
            "extension_pair_merged": extension_enabled,
        },
        "source_validation": {
            "all_rmse_and_aggregates_independently_recomputed_from_raw_cells": True,
            "coverage_failures_retained": True,
            "all_greedy_hyperparameters_share_size_schedules": all(
                bool(
                    audits[dataset][
                        "all_greedy_hyperparameters_share_size_schedules"
                    ]
                )
                for audits in audit_sets.values()
                for dataset in DATASETS
            ),
            "all_greedy_hyperparameters_share_relabels": all(
                bool(audits[dataset]["all_greedy_hyperparameters_share_relabels"])
                for audits in audit_sets.values()
                for dataset in DATASETS
            ),
            "overlap_merge_audit": overlap_audit,
            "sources": source_paths,
        },
        "rows": rows,
    }


def build_locked_selection_artifact(
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Lock one larger-K configuration using the registered four-cell rule.

    The returned structure intentionally matches the fields consumed by
    ``attach_holdout_validation``.  The registered default participates in
    the ranking even though it is not part of the fixed-``lambda0=0`` K curve.
    """
    protocol = summary["protocol"]
    budget_indices = [int(value) for value in protocol["budget_indices"]]
    cell_count = len(DATASETS) * len(budget_indices)
    _require(
        cell_count == 4,
        "registered larger-K locking requires 2 datasets x 2 budgets",
    )
    fixed_rows = list(summary["rows"])
    _require(fixed_rows, "larger-K summary has no fixed-lambda rows")

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
                failures += int(point["coverage_failures"])
                failures += int(point["registered_default_coverage_failures"])
                failures += int(point["inside_orbit_coverage_failures"])
                scaled_seconds = point[
                    "candidate_mean_design_seconds_on_base_clock"
                ]
                if scaled_seconds is not None:
                    design_seconds.append(float(scaled_seconds))
        default_values = [
            float(value) for value in cell_default.values() if value is not None
        ]
        orbit_values = [
            float(value) for value in cell_orbit.values() if value is not None
        ]
        eligible = (
            failures == 0
            and len(default_values) == cell_count
            and len(orbit_values) == cell_count
        )
        ranking.append(
            {
                "configuration_id": row["configuration_id"],
                "lambda0": float(row["lambda0"]),
                "candidate_pool": int(row["candidate_pool"]),
                "source_report_set": row.get("source_report_set", "base"),
                "is_registered_default": False,
                "selection_eligible": eligible,
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
                    _geometric_mean(orbit_values) if eligible else None
                ),
                "worst_cell_rmse_ratio_to_default": (
                    max(default_values) if eligible else None
                ),
                "worst_cell_rmse_ratio_to_orbit": (
                    max(orbit_values) if eligible else None
                ),
                "mean_design_seconds": (
                    float(np.mean(design_seconds))
                    if len(design_seconds) == cell_count
                    else None
                ),
            }
        )

    # Recover the registered-default row from the independently reconstructed
    # references stored in any fixed-K row.  Use the base-sourced row so its
    # descriptive elapsed time stays on the base report's clock.
    template = next(
        (row for row in fixed_rows if row.get("source_report_set", "base") == "base"),
        fixed_rows[0],
    )
    default_cells: dict[str, float | None] = {}
    default_orbit_cells: dict[str, float | None] = {}
    default_dataset_scores: dict[str, float | None] = {}
    default_design_seconds: list[float] = []
    default_failures = 0
    for dataset in DATASETS:
        dataset_orbit: list[float] = []
        for point in template["datasets"][dataset]["budget_cells"]:
            key = f"{dataset}:budget_index={int(point['budget_index'])}"
            failures = int(point["registered_default_coverage_failures"])
            orbit_failures = int(point["inside_orbit_coverage_failures"])
            default_failures += failures + orbit_failures
            default_cells[key] = 1.0 if failures == 0 else None
            ratio_orbit = _safe_ratio(
                point["registered_default_aggregate_rmse"],
                point["inside_orbit_aggregate_rmse"],
            )
            if failures or orbit_failures:
                ratio_orbit = None
            default_orbit_cells[key] = ratio_orbit
            if ratio_orbit is not None:
                dataset_orbit.append(ratio_orbit)
            default_design_seconds.append(
                float(point["registered_default_mean_design_seconds"])
            )
        default_dataset_scores[dataset] = (
            1.0
            if all(
                default_cells[
                    f"{dataset}:budget_index={int(point['budget_index'])}"
                ]
                is not None
                for point in template["datasets"][dataset]["budget_cells"]
            )
            else None
        )
    default_orbit_values = [
        float(value) for value in default_orbit_cells.values() if value is not None
    ]
    default_eligible = (
        default_failures == 0
        and len(default_orbit_values) == cell_count
        and len(default_cells) == cell_count
    )
    ranking.append(
        {
            "configuration_id": _configuration_id(*REGISTERED_DEFAULT),
            "lambda0": REGISTERED_DEFAULT[0],
            "candidate_pool": REGISTERED_DEFAULT[1],
            "source_report_set": "base",
            "is_registered_default": True,
            "selection_eligible": default_eligible,
            "coverage_failures": default_failures,
            "dataset_geometric_mean_rmse_ratio_to_default": (
                default_dataset_scores
            ),
            "cell_rmse_ratios_to_default": default_cells,
            "cell_rmse_ratios_to_orbit": default_orbit_cells,
            "joint_geometric_mean_rmse_ratio_to_default": (
                1.0 if default_eligible else None
            ),
            "joint_geometric_mean_rmse_ratio_to_orbit": (
                _geometric_mean(default_orbit_values)
                if default_eligible
                else None
            ),
            "worst_cell_rmse_ratio_to_default": (
                1.0 if default_eligible else None
            ),
            "worst_cell_rmse_ratio_to_orbit": (
                max(default_orbit_values) if default_eligible else None
            ),
            "mean_design_seconds": (
                float(np.mean(default_design_seconds))
                if len(default_design_seconds) == cell_count
                else None
            ),
        }
    )

    eligible_rows = [row for row in ranking if row["selection_eligible"]]
    _require(eligible_rows, "every larger-K screening configuration failed coverage")
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
    best = next(row for row in ranking if row["selection_eligible"])
    locked_keys = (
        "configuration_id",
        "lambda0",
        "candidate_pool",
        "source_report_set",
        "joint_geometric_mean_rmse_ratio_to_default",
        "joint_geometric_mean_rmse_ratio_to_orbit",
        "worst_cell_rmse_ratio_to_default",
        "worst_cell_rmse_ratio_to_orbit",
        "mean_design_seconds",
    )
    return {
        "status": "configuration_locked",
        "experiment": "inside_greedy_larger_k_joint_hyperparameter_selection",
        "selection_rule": {
            "primary": (
                "minimize the equal-weight mean log aggregate-RMSE ratio "
                "to the registered default across 2 datasets x 2 budgets"
            ),
            "coverage_policy": (
                "eliminate any configuration with a candidate, default, or "
                "Orbit coverage failure"
            ),
            "tie_breaks": [
                "lower worst dataset-budget RMSE ratio to default",
                "smaller candidate pool K",
                "smaller lambda0",
            ],
            "design_time_is_descriptive_not_a_selection_tie_break": True,
            "cell_count": cell_count,
            "datasets": list(DATASETS),
            "budget_indices": budget_indices,
            "budget_multipliers": list(protocol["budget_multipliers"]),
        },
        "screening_sources": summary["source_validation"]["sources"],
        "screening_overlap_merge_audit": summary["source_validation"][
            "overlap_merge_audit"
        ],
        "screening_contract": {
            "base_seed": protocol["base_seed"],
            "repeats": protocol["repeats"],
            "budget_multipliers": list(protocol["budget_multipliers"]),
            "all_rmse_and_aggregates_independently_recomputed": True,
            "greedy_size_schedules_and_relabels_paired_across_hyperparameters": True,
            "overlapping_report_cells_deterministically_identical": bool(
                summary["source_validation"]["overlap_merge_audit"]
                and summary["source_validation"]["overlap_merge_audit"]["passed"]
            ),
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
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11.5,
        }
    )
    fig, axis = plt.subplots(figsize=(6.3, 6.0))
    fig.suptitle(title, x=0.14, y=0.975, ha="left", fontsize=18, weight="semibold")
    fig.text(0.14, 0.927, subtitle, ha="left", va="top", fontsize=11.5, color=MID_GREY)
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.79)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=1.0, alpha=0.8)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.45)
        spine.set_color(CHARCOAL)
    axis.tick_params(width=1.25, length=5, color=CHARCOAL)
    return fig, axis


def _plot_series(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    field: str,
) -> tuple[list[float], list[tuple[str, int, int]]]:
    valid_values: list[float] = []
    failures: list[tuple[str, int, int]] = []
    for series in DATASETS + ("joint",):
        style = SERIES_STYLE[series]
        x_values: list[int] = []
        y_values: list[float] = []
        for row in rows:
            pool = int(row["candidate_pool"])
            value = row[field][series]
            x_values.append(pool)
            if value is None:
                y_values.append(float("nan"))
                if field == "rmse_ratio_to_reference":
                    if series == "joint":
                        count = int(row["coverage_failures"]) + int(
                            row["reference_coverage_failures"]
                        )
                    else:
                        dataset = row["datasets"][series]
                        count = int(dataset["coverage_failures"]) + int(
                            dataset["reference_coverage_failures"]
                        )
                    failures.append((series, pool, count))
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


def plot_rmse_ratios(
    summary: Mapping[str, Any], output_png: Path
) -> tuple[Path, Path]:
    """Plot Airport, Voting, and equal-cell joint RMSE ratios against K."""
    rows = summary["rows"]
    reference_label = str(summary["reference"]["label"])
    fig, axis = _figure_setup(
        "Larger-K RMSE",
        rf"$\lambda_0=0$; ratio to {reference_label}; <1 is better",
    )
    valid_values, failures = _plot_series(
        axis, rows, field="rmse_ratio_to_reference"
    )
    axis.axhline(1.0, color=MID_GREY, linewidth=1.7, linestyle=":", zorder=0)
    axis.set_xscale("log", base=2)
    k_values = [int(row["candidate_pool"]) for row in rows]
    axis.set_xticks(k_values, [str(value) for value in k_values])
    axis.set_xlabel("Candidate pool K")
    axis.set_ylabel("RMSE ratio")

    plotted = valid_values + [1.0]
    lower = min(plotted)
    upper = max(plotted)
    span = max(upper - lower, 0.08)
    y_min = max(0.0, lower - 0.18 * span)
    y_max = upper + 0.28 * span
    axis.set_ylim(y_min, y_max)
    if failures:
        offsets = {"airport": 0.00, "voting": 0.035, "joint": 0.07}
        for series, pool, count in failures:
            y = y_max - (0.10 + offsets[series]) * span
            axis.scatter(
                [pool],
                [y],
                marker="X",
                s=78,
                linewidth=1.4,
                color=SERIES_STYLE[series]["color"],
                zorder=5,
            )
            axis.annotate(
                f"{count} fail",
                (pool, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9.5,
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


def plot_design_time_ratios(
    summary: Mapping[str, Any], output_png: Path
) -> tuple[Path, Path]:
    """Plot descriptive Greedy-design wall-time ratios against K."""
    rows = summary["rows"]
    reference_label = str(summary["reference"]["label"])
    fig, axis = _figure_setup(
        "Larger-K design time",
        rf"$\lambda_0=0$; wall-time ratio to {reference_label}; utility time excluded",
    )
    valid_values, _ = _plot_series(
        axis, rows, field="design_time_ratio_to_reference"
    )
    axis.axhline(1.0, color=MID_GREY, linewidth=1.7, linestyle=":", zorder=0)
    axis.set_xscale("log", base=2)
    k_values = [int(row["candidate_pool"]) for row in rows]
    axis.set_xticks(k_values, [str(value) for value in k_values])
    axis.set_xlabel("Candidate pool K")
    axis.set_ylabel("Design-time ratio")
    upper = max(valid_values + [1.0])
    axis.set_ylim(0.0, upper * 1.13)
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
    parser.add_argument("--extension-airport", type=Path)
    parser.add_argument("--extension-voting", type=Path)
    parser.add_argument(
        "--reference", choices=REFERENCE_CHOICES, default="k16"
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--rmse-plot", required=True, type=Path)
    parser.add_argument("--time-plot", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = build_larger_k_summary(
        args.airport,
        args.voting,
        extension_airport_path=args.extension_airport,
        extension_voting_path=args.extension_voting,
        reference=args.reference,
    )
    _atomic_write_json(args.output, summary)
    if args.selection_output is not None:
        _atomic_write_json(
            args.selection_output,
            build_locked_selection_artifact(summary),
        )
    plot_rmse_ratios(summary, args.rmse_plot)
    plot_design_time_ratios(summary, args.time_plot)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Validate, merge, summarize, and plot the two K=64 lambda sweeps.

The lower-range and upper-range reports deliberately overlap at
``lambda0=1/4``.  That overlap is a reproducibility check, not a cell that may
be silently deduplicated.  Each of the four raw reports is first passed through
the existing independent hyperparameter-sweep validator.  The merger then
requires identical protocols and exact equality of every deterministic field
in the repeated quarter-lambda cells, including estimates, design hashes, and
physical utility-call counts.

The plotted quantity is the equal-weight geometric mean of aggregate-RMSE
ratios over Airport/Voting x selected-budget cells, relative to ``lambda0=0``.
Thus every dataset-budget cell has the same weight and values below one are
better.  The output is descriptive screening evidence; it does not re-lock the
paper-facing default or alter any runner/core setting.
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
    EXPERIMENT_ID,
    REGISTERED_CANDIDATE_POOL,
    REGISTERED_LAMBDA0,
)
from experiments.validate_inside_greedy_hyperparameter_sweep import (
    _configuration_id,
    validate_report,
)


DATASETS = ("airport", "voting")
RANGES = ("lower", "upper")
FIXED_K = 64
REFERENCE_LAMBDA0 = 0.0
OVERLAP_LAMBDA0 = 0.25
LOWER_LAMBDAS = (0.0, 1 / 128, 1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4)
UPPER_LAMBDAS = (1 / 4, 1 / 2, 1.0, 2.0)
MERGED_LAMBDAS = tuple(sorted(set(LOWER_LAMBDAS) | set(UPPER_LAMBDAS)))
REGISTERED_DEFAULT = (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)

CHARCOAL = "#20252B"
BLUE = "#2B6CB0"
ORANGE = "#C56A1A"
MID_GREY = "#737B84"
LIGHT_GREY = "#D8DDE3"

SERIES_STYLE = {
    "airport": {
        "label": "Airport",
        "color": BLUE,
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": BLUE,
    },
    "voting": {
        "label": "Voting",
        "color": ORANGE,
        "linestyle": "--",
        "marker": "^",
        "markerfacecolor": "white",
    },
    "joint": {
        "label": "Joint",
        "color": CHARCOAL,
        "linestyle": "-.",
        "marker": "D",
        "markerfacecolor": "white",
    },
}

# Wall-clock values are intentionally omitted: they need not be bitwise equal
# when an identical deterministic cell is rerun.  Everything that determines
# the estimate and call accounting is compared exactly.
OVERLAP_EXACT_FIELDS = (
    "dataset",
    "method",
    "lambda0",
    "candidate_pool",
    "repeat",
    "budget_index",
    "base_seed",
    "seed",
    "inner_calls",
    "total_calls",
    "truth",
    "status",
    "coverage_failure",
    "failure_reason",
    "estimate",
    "rmse",
    "planned_total_utility_calls",
    "actual_utility_calls",
    "diagnostics",
)
OVERLAP_HASH_FIELDS = (
    "size_schedule_sha256",
    "relabel_permutation_sha256",
)


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


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


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
    ratio = top / bottom
    return ratio if math.isfinite(ratio) and ratio >= 0.0 else None


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


def _configuration_set(audit: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["evaluated_configurations"]
    }


def _fixed_k_lambdas(audit: Mapping[str, Any]) -> set[float]:
    return {
        lambda0
        for lambda0, candidate_pool in _configuration_set(audit)
        if candidate_pool == FIXED_K
    }


def _protocol_snapshot(report: Mapping[str, Any]) -> dict[str, Any]:
    configuration = report.get("configuration")
    _require(isinstance(configuration, Mapping), "configuration is missing")
    # The two grids necessarily differ in these two declarations.  Every other
    # protocol field, including seeds, budgets, estimator, design, and process
    # count, must match exactly.
    ignored = {"dataset", "evaluated_configurations", "requested_configurations"}
    return {
        str(key): value
        for key, value in configuration.items()
        if key not in ignored
    }


def _validate_source(
    path: Path, *, dataset: str, range_name: str
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    report = _read(path)
    # This is deliberately the first semantic operation on a source report.
    audit = validate_report(report)
    _require(bool(audit.get("passed")), f"{range_name} {dataset} audit did not pass")
    _require(
        bool(audit.get("all_rmse_and_aggregates_independently_recomputed")),
        f"{range_name} {dataset} was not independently reconstructed",
    )
    _require(
        bool(audit.get("all_greedy_hyperparameters_share_size_schedules")),
        f"{range_name} {dataset} lacks paired Greedy size schedules",
    )
    _require(
        bool(audit.get("all_greedy_hyperparameters_share_relabels")),
        f"{range_name} {dataset} lacks paired Greedy relabels",
    )
    _require(audit.get("dataset") == dataset, f"{range_name} dataset paths are swapped")
    _require(report.get("experiment") == EXPERIMENT_ID, "wrong experiment protocol")
    configuration = report.get("configuration", {})
    _require(configuration.get("dataset") == dataset, "raw report dataset mismatch")
    expected_lambdas = set(LOWER_LAMBDAS if range_name == "lower" else UPPER_LAMBDAS)
    _require(
        _fixed_k_lambdas(audit) == expected_lambdas,
        f"{range_name} {dataset} has the wrong K={FIXED_K} lambda grid",
    )
    allowed = {(value, FIXED_K) for value in expected_lambdas} | {
        REGISTERED_DEFAULT
    }
    _require(
        _configuration_set(audit) == allowed,
        f"{range_name} {dataset} has unexpected K/configuration cells",
    )
    return report, audit


def _check_audit_protocol(
    left: Mapping[str, Any], right: Mapping[str, Any], *, context: str
) -> None:
    for field in (
        "stage",
        "repeats",
        "base_seed",
        "budget_multipliers",
        "selected_budget_indices",
        "selected_budget_multipliers",
    ):
        _require(left.get(field) == right.get(field), f"{context} disagrees on {field}")
    _require(left.get("stage") == "screening", f"{context} is not screening")


def _quarter_cells(report: Mapping[str, Any], audit: Mapping[str, Any]) -> dict[tuple[int, int], Mapping[str, Any]]:
    cells: dict[tuple[int, int], Mapping[str, Any]] = {}
    for raw in report.get("raw_cells", []):
        if (
            raw.get("method") == "inside_greedy"
            and int(raw.get("candidate_pool", -1)) == FIXED_K
            and float(raw.get("lambda0", math.nan)) == OVERLAP_LAMBDA0
        ):
            key = (int(raw["repeat"]), int(raw["budget_index"]))
            _require(key not in cells, f"duplicate quarter-lambda raw cell {key}")
            cells[key] = raw
    expected = {
        (repeat, int(budget_index))
        for repeat in range(int(audit["repeats"]))
        for budget_index in audit["selected_budget_indices"]
    }
    _require(set(cells) == expected, "quarter-lambda raw-cell coverage is incomplete")
    return cells


def _check_overlap(
    lower_report: Mapping[str, Any],
    lower_audit: Mapping[str, Any],
    upper_report: Mapping[str, Any],
    upper_audit: Mapping[str, Any],
    *,
    dataset: str,
) -> dict[str, Any]:
    lower = _quarter_cells(lower_report, lower_audit)
    upper = _quarter_cells(upper_report, upper_audit)
    _require(set(lower) == set(upper), f"{dataset} quarter-lambda cell keys differ")
    canonical_rows: list[dict[str, Any]] = []
    for key in sorted(lower):
        low_cell, high_cell = lower[key], upper[key]
        for field in OVERLAP_EXACT_FIELDS:
            _require(
                low_cell.get(field) == high_cell.get(field),
                f"{dataset} quarter-lambda cell {key} differs on {field}",
            )
        low_diagnostics = low_cell.get("diagnostics", {})
        high_diagnostics = high_cell.get("diagnostics", {})
        for field in OVERLAP_HASH_FIELDS:
            low_hash = low_diagnostics.get(field)
            high_hash = high_diagnostics.get(field)
            _require(
                isinstance(low_hash, str) and len(low_hash) == 64,
                f"{dataset} quarter-lambda cell {key} lacks {field}",
            )
            _require(
                low_hash == high_hash,
                f"{dataset} quarter-lambda cell {key} differs on {field}",
            )
        canonical_rows.append(
            {field: low_cell.get(field) for field in OVERLAP_EXACT_FIELDS}
        )
    digest = _json_sha256(canonical_rows)
    return {
        "passed": True,
        "lambda0": OVERLAP_LAMBDA0,
        "candidate_pool": FIXED_K,
        "cells_checked": len(canonical_rows),
        "exact_fields": list(OVERLAP_EXACT_FIELDS),
        "explicit_hash_fields": list(OVERLAP_HASH_FIELDS),
        "deterministic_cells_sha256": digest,
        "estimates_exactly_equal": True,
        "design_hashes_exactly_equal": True,
        "physical_calls_exactly_equal": True,
        "timing_fields_intentionally_excluded": [
            "design_seconds",
            "utility_seconds",
        ],
    }


def _result_for(audit: Mapping[str, Any], lambda0: float) -> Mapping[str, Any]:
    return audit["recomputed_results_by_configuration"][
        _configuration_id(lambda0, FIXED_K)
    ]


def build_merged_k64_lambda_summary(
    lower_airport_path: Path,
    lower_voting_path: Path,
    upper_airport_path: Path,
    upper_voting_path: Path,
) -> dict[str, Any]:
    """Build the strict, independently reconstructed 0-to-2 K=64 summary."""
    source_paths = {
        "lower": {"airport": lower_airport_path, "voting": lower_voting_path},
        "upper": {"airport": upper_airport_path, "voting": upper_voting_path},
    }
    reports: dict[str, dict[str, Mapping[str, Any]]] = {name: {} for name in RANGES}
    audits: dict[str, dict[str, dict[str, Any]]] = {name: {} for name in RANGES}
    for range_name in RANGES:
        for dataset in DATASETS:
            report, audit = _validate_source(
                source_paths[range_name][dataset],
                dataset=dataset,
                range_name=range_name,
            )
            reports[range_name][dataset] = report
            audits[range_name][dataset] = audit

    # First compare Airport and Voting inside each range, then compare the two
    # ranges inside each dataset.  The raw protocol snapshots make the latter
    # stricter than checking only the validator's headline fields.
    for range_name in RANGES:
        _check_audit_protocol(
            audits[range_name]["airport"],
            audits[range_name]["voting"],
            context=f"{range_name} datasets",
        )
        # Dataset-normalized budget multipliers and selected indices match via
        # the audited fields above.  Raw call counts may not: Airport has 100
        # players whereas Voting has 51, so n-scaled inner/boundary calls are
        # intentionally dataset specific.
    overlap_audits: dict[str, Any] = {}
    for dataset in DATASETS:
        _check_audit_protocol(
            audits["lower"][dataset],
            audits["upper"][dataset],
            context=f"{dataset} ranges",
        )
        _require(
            _protocol_snapshot(reports["lower"][dataset])
            == _protocol_snapshot(reports["upper"][dataset]),
            f"{dataset} ranges use different raw protocols",
        )
        overlap_audits[dataset] = _check_overlap(
            reports["lower"][dataset],
            audits["lower"][dataset],
            reports["upper"][dataset],
            audits["upper"][dataset],
            dataset=dataset,
        )

    reference_results = {
        dataset: _result_for(audits["lower"][dataset], REFERENCE_LAMBDA0)
        for dataset in DATASETS
    }
    budget_indices = [
        int(value) for value in audits["lower"]["airport"]["selected_budget_indices"]
    ]
    budget_multipliers = [
        int(value)
        for value in audits["lower"]["airport"]["selected_budget_multipliers"]
    ]
    expected_joint_cells = len(DATASETS) * len(budget_indices)
    rows: list[dict[str, Any]] = []

    for lambda0 in MERGED_LAMBDAS:
        range_name = "lower" if lambda0 <= OVERLAP_LAMBDA0 else "upper"
        source_range = "both_exact_duplicate" if lambda0 == OVERLAP_LAMBDA0 else range_name
        dataset_rows: dict[str, Any] = {}
        joint_ratios: list[float] = []
        total_failures = 0
        total_reference_failures = 0
        for dataset in DATASETS:
            candidate = _result_for(audits[range_name][dataset], lambda0)
            reference = reference_results[dataset]
            cell_rows: list[dict[str, Any]] = []
            dataset_ratios: list[float] = []
            dataset_failures = 0
            dataset_reference_failures = 0
            for position, budget_index in enumerate(budget_indices):
                candidate_budget = candidate["budgets"][str(budget_index)]
                reference_budget = reference["budgets"][str(budget_index)]
                candidate_failures = int(candidate_budget["coverage_failures"])
                reference_failures = int(reference_budget["coverage_failures"])
                dataset_failures += candidate_failures
                dataset_reference_failures += reference_failures
                ratio = _safe_ratio(
                    candidate_budget["aggregate_rmse"],
                    reference_budget["aggregate_rmse"],
                )
                if candidate_failures or reference_failures:
                    ratio = None
                if ratio is not None:
                    dataset_ratios.append(ratio)
                    joint_ratios.append(ratio)
                cell_rows.append(
                    {
                        "budget_index": budget_index,
                        "budget_multiplier": budget_multipliers[position],
                        "candidate_aggregate_rmse": candidate_budget["aggregate_rmse"],
                        "reference_aggregate_rmse": reference_budget["aggregate_rmse"],
                        "rmse_ratio_to_lambda0_zero": ratio,
                        "candidate_coverage_failures": candidate_failures,
                        "reference_coverage_failures": reference_failures,
                    }
                )
            total_failures += dataset_failures
            total_reference_failures += dataset_reference_failures
            complete = (
                dataset_failures == 0
                and dataset_reference_failures == 0
                and len(dataset_ratios) == len(budget_indices)
            )
            dataset_rows[dataset] = {
                "rmse_ratio_to_lambda0_zero": (
                    _geometric_mean(dataset_ratios) if complete else None
                ),
                "candidate_coverage_failures": dataset_failures,
                "reference_coverage_failures": dataset_reference_failures,
                "budget_cells": cell_rows,
            }
        complete_joint = (
            total_failures == 0
            and total_reference_failures == 0
            and len(joint_ratios) == expected_joint_cells
        )
        rows.append(
            {
                "configuration_id": _configuration_id(lambda0, FIXED_K),
                "lambda0": float(lambda0),
                "candidate_pool": FIXED_K,
                "source_range": source_range,
                "joint_equal_cell_rmse_ratio_to_lambda0_zero": (
                    _geometric_mean(joint_ratios) if complete_joint else None
                ),
                "candidate_coverage_failures": total_failures,
                "reference_coverage_failures": total_reference_failures,
                "datasets": dataset_rows,
            }
        )

    eligible = [
        row for row in rows if row["joint_equal_cell_rmse_ratio_to_lambda0_zero"] is not None
    ]
    _require(eligible, "all merged lambda configurations failed coverage")
    descriptive_best = min(
        eligible,
        key=lambda row: (
            float(row["joint_equal_cell_rmse_ratio_to_lambda0_zero"]),
            float(row["lambda0"]),
        ),
    )
    sources = {
        range_name: {
            dataset: {
                "path": str(source_paths[range_name][dataset].resolve()),
                "sha256": _sha256(source_paths[range_name][dataset]),
                "raw_cells_audited": len(reports[range_name][dataset].get("raw_cells", [])),
            }
            for dataset in DATASETS
        }
        for range_name in RANGES
    }
    first_audit = audits["lower"]["airport"]
    return {
        "status": "validated_merged_k64_lambda_summary",
        "experiment": "inside_greedy_k64_lambda_lower_upper_merged",
        "fixed_candidate_pool": FIXED_K,
        "reference_lambda0": REFERENCE_LAMBDA0,
        "protocol": {
            "source_experiment": EXPERIMENT_ID,
            "stage": first_audit["stage"],
            "repeats": int(first_audit["repeats"]),
            "base_seed": int(first_audit["base_seed"]),
            "budget_indices": budget_indices,
            "budget_multipliers": budget_multipliers,
            "lower_lambda0_values": list(LOWER_LAMBDAS),
            "upper_lambda0_values": list(UPPER_LAMBDAS),
            "merged_lambda0_values": list(MERGED_LAMBDAS),
        },
        "aggregation": {
            "within_budget_rmse": "sqrt(mean repeat-level squared RMSE)",
            "cell_ratio": "candidate aggregate RMSE / lambda0=0 aggregate RMSE",
            "joint_ratio": (
                "equal-weight geometric mean over Airport/Voting x selected budgets"
            ),
            "number_of_equal_weight_cells": expected_joint_cells,
            "interpretation": "less than 1 is better than lambda0=0",
            "coverage_policy": (
                "retain every failure and suppress any aggregate containing one"
            ),
        },
        "source_validation": {
            "independent_validator_calls": len(RANGES) * len(DATASETS),
            "all_reports_independently_recomputed_from_raw_cells": True,
            "raw_protocols_exactly_equal_between_ranges_within_each_dataset_except_lambda_grid": True,
            "quarter_lambda_overlap": overlap_audits,
            "sources": sources,
        },
        "descriptive_screening_minimum": {
            "lambda0": float(descriptive_best["lambda0"]),
            "candidate_pool": FIXED_K,
            "joint_equal_cell_rmse_ratio_to_lambda0_zero": descriptive_best[
                "joint_equal_cell_rmse_ratio_to_lambda0_zero"
            ],
            "not_a_holdout_claim": True,
        },
        "rows": rows,
    }


def _lambda_label(value: float) -> str:
    if value == 0.0:
        return "0"
    reciprocal = round(1.0 / value) if value < 1.0 else None
    if reciprocal is not None and math.isclose(value, 1.0 / reciprocal):
        return f"1/{reciprocal}"
    return f"{value:g}"


def plot_merged_k64_lambda_summary(
    summary: Mapping[str, Any], output_png: Path
) -> tuple[Path, Path]:
    """Export one near-square screening chart as PNG and PDF."""
    rows = sorted(summary["rows"], key=lambda row: float(row["lambda0"]))
    lambdas = [float(row["lambda0"]) for row in rows]
    series_values = {
        "airport": [
            row["datasets"]["airport"]["rmse_ratio_to_lambda0_zero"]
            for row in rows
        ],
        "voting": [
            row["datasets"]["voting"]["rmse_ratio_to_lambda0_zero"]
            for row in rows
        ],
        "joint": [
            row["joint_equal_cell_rmse_ratio_to_lambda0_zero"] for row in rows
        ],
    }
    valid = [
        float(value)
        for values in series_values.values()
        for value in values
        if value is not None
    ]
    _require(valid, "no finite RMSE ratio is available to plot")
    positives = [value for value in lambdas if value > 0.0]
    _require(positives, "the lambda plot needs at least one positive value")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 16,
            "xtick.labelsize": 12.5,
            "ytick.labelsize": 13.5,
        }
    )
    fig, axis = plt.subplots(figsize=(6.5, 6.5))
    fig.suptitle(
        "INSIDE-Greedy lambda sweep",
        x=0.145,
        y=0.975,
        ha="left",
        fontsize=18,
        weight="semibold",
        color=CHARCOAL,
    )
    protocol = summary["protocol"]
    fig.text(
        0.145,
        0.925,
        (
            rf"$K=64$; equal-weight Airport/Voting $\times$ budgets "
            rf"{protocol['budget_multipliers']}"
            "\n"
            rf"Ratio to $\lambda_0=0$; <1 better"
        ),
        ha="left",
        va="top",
        fontsize=11.2,
        color=MID_GREY,
    )
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.20, top=0.76)

    for series_name in ("airport", "voting", "joint"):
        style = SERIES_STYLE[series_name]
        plotted = [
            float(value) if value is not None else float("nan")
            for value in series_values[series_name]
        ]
        axis.plot(
            lambdas,
            plotted,
            label=style["label"],
            color=style["color"],
            linewidth=3.0,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8.0,
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=2.0,
        )
    axis.axhline(1.0, color=MID_GREY, linewidth=1.8, linestyle=":", zorder=0)
    axis.set_xscale(
        "symlog", base=2, linthresh=min(positives), linscale=1.0
    )
    # Matplotlib's automatic symlog margin can extend into a full negative-log
    # decade even though lambda is nonnegative, wasting nearly half the panel.
    # Keep only a small linear pad to the left of zero.
    axis.set_xlim(-0.15 * min(positives), 1.06 * max(positives))
    axis.set_xticks(lambdas)
    axis.set_xticklabels([_lambda_label(value) for value in lambdas])
    for label in axis.get_xticklabels():
        label.set_rotation(34)
        label.set_ha("right")
        label.set_rotation_mode("anchor")
    axis.set_xlabel(r"First-moment weight $\lambda_0$")
    axis.set_ylabel("Equal-cell RMSE ratio")
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=1.0, alpha=0.85)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_color(CHARCOAL)
    axis.tick_params(width=1.3, length=5, color=CHARCOAL)

    limits = valid + [1.0]
    lower, upper = min(limits), max(limits)
    span = max(upper - lower, 0.04)
    axis.set_ylim(max(0.0, lower - 0.22 * span), upper + 0.28 * span)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.0,
        fontsize=11.0,
    )
    for series_name, values in series_values.items():
        for lambda0, value in zip(lambdas, values, strict=True):
            if value is None:
                axis.scatter(
                    [lambda0],
                    [upper + 0.12 * span],
                    marker="X",
                    s=85,
                    color=SERIES_STYLE[series_name]["color"],
                )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_png, output_pdf


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lower-airport",
        type=Path,
        default=Path("results/json/inside_greedy_k64_lambda_fine_screen_airport.json"),
    )
    parser.add_argument(
        "--lower-voting",
        type=Path,
        default=Path("results/json/inside_greedy_k64_lambda_fine_screen_voting.json"),
    )
    parser.add_argument(
        "--upper-airport",
        type=Path,
        default=Path(
            "results/json/inside_greedy_k64_lambda_quarter_to_two_screen_airport.json"
        ),
    )
    parser.add_argument(
        "--upper-voting",
        type=Path,
        default=Path(
            "results/json/inside_greedy_k64_lambda_quarter_to_two_screen_voting.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/json/inside_greedy_k64_lambda_zero_to_two_summary.json"),
    )
    parser.add_argument(
        "--output-figure",
        type=Path,
        default=Path("results/pdf/inside_greedy_k64_lambda_zero_to_two_rmse.png"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_merged_k64_lambda_summary(
        args.lower_airport,
        args.lower_voting,
        args.upper_airport,
        args.upper_voting,
    )
    _write_json_atomic(args.output_json, summary)
    png, pdf = plot_merged_k64_lambda_summary(summary, args.output_figure)
    print(
        json.dumps(
            {
                "summary": str(args.output_json),
                "png": str(png),
                "pdf": str(pdf),
                "descriptive_screening_minimum": summary[
                    "descriptive_screening_minimum"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

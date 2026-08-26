"""Audit the ten-method Greedy-scope/estimator comparison.

The experiment adds one controlled branch to the established second-moment
scope report: the *same* per-size Greedy coalitions are evaluated with the OFA
conditional-mean ratio estimator.  This validator therefore checks both the
ordinary accuracy/call contract and the controls that make the estimator
comparison interpretable:

* global-linear, per-size-linear, and per-size-ratio use the same size schedule
  and final random relabeling in every budget/repeat cell;
* the ratio branch has at least one included and excluded observation for every
  player/size stratum; and
* recomputing the linear estimator while constructing the ratio branch agrees
  with the retained per-size-linear source estimate.

Exact Airport/voting truth, every retained RMSE, and physical utility-call
accounting are delegated to the independently tested common report validator.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from experiments.plot_inside_comparison import METHOD_ORDER as LEGACY_METHOD_ORDER
from experiments.validate_inside_comparison import (
    validate_report as validate_legacy_report,
)


METHOD_ORDER = (
    "inside_greedy_global",
    "inside_greedy_per_size_linear",
    "inside_greedy_per_size_ratio",
    "inside_orbit",
    "ofa_iid_linear",
    "ofa_iid_ratio",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)

EXPECTED_IDENTITIES = {
    "inside_greedy_global": {
        "second_moment_scope": "global_weighted",
        "design_method": "frame_coupled",
        "estimator": "coupled_linear",
    },
    "inside_greedy_per_size_linear": {
        "second_moment_scope": "per_size",
        "design_method": "frame_coupled_per_size",
        "estimator": "coupled_linear",
    },
    "inside_greedy_per_size_ratio": {
        "second_moment_scope": "per_size",
        "design_method": "frame_coupled_per_size",
        "estimator": "ofa_conditional_mean_ratio_missing_raise",
    },
}

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECONSTRUCTION_TOLERANCE = 5e-13


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _configured_methods(configuration: Mapping[str, Any]) -> tuple[str, ...]:
    raw = configuration.get("methods", configuration.get("method_order"))
    _require(isinstance(raw, (list, tuple)), "configuration methods must be a list")
    return tuple(str(value) for value in raw)


def _diagnostics(summary: Mapping[str, Any], *, path: str) -> list[Mapping[str, Any]]:
    raw = summary.get("diagnostics_by_repeat")
    _require(
        isinstance(raw, list) and len(raw) == 3,
        f"{path}.diagnostics_by_repeat must contain three objects",
    )
    return [
        _mapping(value, path=f"{path}.diagnostics_by_repeat[{index}]")
        for index, value in enumerate(raw)
    ]


def _nested_diagnostic_value(diagnostic: Mapping[str, Any], key: str) -> Any:
    nested = diagnostic.get("design_diagnostics")
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    return diagnostic.get(key, nested_mapping.get(key))


def _hash_value(diagnostic: Mapping[str, Any], key: str, *, path: str) -> str:
    aliases = (
        ("size_schedule_sha256_int64",)
        if key == "size_schedule_sha256"
        else ()
    )
    raw = _nested_diagnostic_value(diagnostic, key)
    if raw is None:
        for alias in aliases:
            raw = _nested_diagnostic_value(diagnostic, alias)
            if raw is not None:
                break
    result = str(raw or "").lower()
    _require(
        _HASH_PATTERN.fullmatch(result) is not None,
        f"{path} must retain a 64-character {key}",
    )
    return result


def _actual_calls(summary: Mapping[str, Any], *, path: str) -> tuple[int, ...]:
    raw = summary.get(
        "actual_utility_calls_by_repeat",
        summary.get("actual_utility_calls_per_estimate"),
    )
    _require(
        isinstance(raw, (list, tuple)) and len(raw) == 3,
        f"{path} must retain three actual-call counts",
    )
    result: list[int] = []
    for index, value in enumerate(raw):
        _require(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            f"{path}[{index}] must be an integer",
        )
        _require(int(value) > 0, f"{path}[{index}] must be positive")
        result.append(int(value))
    return tuple(result)


def _legacy_view(report: Mapping[str, Any], greedy_method: str) -> dict[str, Any]:
    """Substitute one controlled Greedy branch into the common 8-method view."""
    configuration = dict(_mapping(report.get("configuration"), path="configuration"))
    configuration["methods"] = list(LEGACY_METHOD_ORDER)
    if "method_order" in configuration:
        configuration["method_order"] = list(LEGACY_METHOD_ORDER)

    greedy_configuration = configuration.get("inside_greedy")
    if not isinstance(greedy_configuration, Mapping):
        greedy_configuration = configuration.get("inside")
    if isinstance(greedy_configuration, Mapping):
        configuration["inside"] = {
            "candidate_pool": greedy_configuration.get("candidate_pool"),
            "greedy_mean_balance_mode": greedy_configuration.get(
                "mean_balance_mode",
                greedy_configuration.get("greedy_mean_balance_mode"),
            ),
            "greedy_mean_balance_lambda0": greedy_configuration.get(
                "mean_balance_lambda0",
                greedy_configuration.get("greedy_mean_balance_lambda0"),
            ),
        }

    source_results = _mapping(
        report.get("results_by_inner_budget"), path="results_by_inner_budget"
    )
    results: dict[str, Any] = {}
    for budget, raw_row in source_results.items():
        row = dict(_mapping(raw_row, path=f"results_by_inner_budget.{budget}"))
        source_methods = _mapping(
            row.get("methods"), path=f"results_by_inner_budget.{budget}.methods"
        )
        row["methods"] = {
            method: source_methods[greedy_method]
            if method == "inside_greedy"
            else source_methods[method]
            for method in LEGACY_METHOD_ORDER
        }
        results[str(budget)] = row

    view = dict(report)
    view["configuration"] = configuration
    view["results_by_inner_budget"] = results
    return view


def _finite_nonnegative(value: Any, *, path: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(math.isfinite(result) and result >= 0.0, f"{path} must be finite and nonnegative")
    return result


def _validate_ratio_coverage(
    diagnostic: Mapping[str, Any], *, path: str
) -> tuple[int, int, float, float, float]:
    coverage = _mapping(diagnostic.get("coverage"), path=f"{path}.coverage")
    minimum_in = coverage.get("minimum_inclusion_count")
    minimum_out = coverage.get("minimum_exclusion_count")
    for name, value in (
        ("minimum_inclusion_count", minimum_in),
        ("minimum_exclusion_count", minimum_out),
    ):
        _require(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            f"{path}.coverage.{name} must be an integer",
        )
        _require(int(value) >= 1, f"{path}.coverage.{name} must be positive")
    _require(
        coverage.get("all_player_size_strata_covered") is True,
        f"{path}.coverage must confirm all player/size strata are covered",
    )
    for name in ("missing_inclusion_strata", "missing_exclusion_strata"):
        value = coverage.get(name)
        _require(
            isinstance(value, (int, np.integer)) and not isinstance(value, bool),
            f"{path}.coverage.{name} must be an integer",
        )
        _require(int(value) == 0, f"{path}.coverage.{name} must be zero")
    max_absolute = _finite_nonnegative(
        coverage.get("maximum_absolute_inclusion_count_deviation"),
        path=f"{path}.coverage.maximum_absolute_inclusion_count_deviation",
    )
    max_relative = _finite_nonnegative(
        coverage.get("maximum_relative_inclusion_count_deviation"),
        path=f"{path}.coverage.maximum_relative_inclusion_count_deviation",
    )
    reconstruction = _finite_nonnegative(
        diagnostic.get("linear_reconstruction_max_abs_difference"),
        path=f"{path}.linear_reconstruction_max_abs_difference",
    )
    _require(
        reconstruction <= _RECONSTRUCTION_TOLERANCE,
        f"{path} linear reconstruction differs by {reconstruction:.3g}",
    )
    return (
        int(minimum_in),
        int(minimum_out),
        max_absolute,
        max_relative,
        reconstruction,
    )


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Raise on a failed audit and return a compact durable result otherwise."""
    _require(isinstance(report, Mapping), "report root must be an object")
    _require(report.get("status") == "complete", "report status must be complete")
    configuration = _mapping(report.get("configuration"), path="configuration")
    _require(
        _configured_methods(configuration) == METHOD_ORDER,
        "configuration method order must contain exactly the ten comparison methods",
    )
    _require(int(configuration.get("repeats", -1)) == 3, "comparison requires three repeats")

    source_report = configuration.get("source_report")
    if isinstance(source_report, Mapping) and "sha256" in source_report:
        source_hash = str(source_report["sha256"]).lower()
        _require(
            _HASH_PATTERN.fullmatch(source_hash) is not None,
            "configuration.source_report.sha256 must be a SHA-256 hash",
        )

    results = _mapping(
        report.get("results_by_inner_budget"), path="results_by_inner_budget"
    )
    _require(len(results) == 5, "comparison must contain five budgets")

    matching_schedule_groups = 0
    matching_relabel_groups = 0
    coverage_cells = 0
    total_actual_calls = 0
    minimum_inclusion = math.inf
    minimum_exclusion = math.inf
    maximum_absolute_deviation = 0.0
    maximum_relative_deviation = 0.0
    maximum_reconstruction_difference = 0.0
    ratio_actual_calls = 0
    ratio_efficiency_residuals: list[float] = []
    truth_values = np.asarray(
        _mapping(report.get("ground_truth"), path="ground_truth").get("values"),
        dtype=np.float64,
    )
    _require(
        truth_values.ndim == 1 and np.all(np.isfinite(truth_values)),
        "ground_truth.values must be a finite vector",
    )
    efficiency_target = float(truth_values.sum())

    for budget, raw_row in results.items():
        row_path = f"results_by_inner_budget.{budget}"
        row = _mapping(raw_row, path=row_path)
        methods = _mapping(row.get("methods"), path=f"{row_path}.methods")
        _require(
            set(methods) == set(METHOD_ORDER),
            f"{row_path} has the wrong ten-method set",
        )

        controlled: dict[str, list[Mapping[str, Any]]] = {}
        for method, expected in EXPECTED_IDENTITIES.items():
            method_path = f"{row_path}.methods.{method}"
            summary = _mapping(methods[method], path=method_path)
            diagnostics = _diagnostics(summary, path=method_path)
            controlled[method] = diagnostics
            for repeat, diagnostic in enumerate(diagnostics):
                diagnostic_path = f"{method_path}.diagnostics_by_repeat[{repeat}]"
                for key, expected_value in expected.items():
                    observed = _nested_diagnostic_value(diagnostic, key)
                    _require(
                        str(observed) == expected_value,
                        f"{diagnostic_path}.{key} identifies {observed!r}, expected {expected_value!r}",
                    )
                if method == "inside_greedy_per_size_ratio":
                    values = _validate_ratio_coverage(diagnostic, path=diagnostic_path)
                    minimum_inclusion = min(minimum_inclusion, values[0])
                    minimum_exclusion = min(minimum_exclusion, values[1])
                    maximum_absolute_deviation = max(maximum_absolute_deviation, values[2])
                    maximum_relative_deviation = max(maximum_relative_deviation, values[3])
                    maximum_reconstruction_difference = max(
                        maximum_reconstruction_difference, values[4]
                    )
                    coverage_cells += 1

            if method == "inside_greedy_per_size_ratio":
                ratio_estimates = np.asarray(summary.get("estimates"), dtype=np.float64)
                _require(
                    ratio_estimates.shape == (3, truth_values.size)
                    and np.all(np.isfinite(ratio_estimates)),
                    f"{method_path}.estimates must be a finite 3 x {truth_values.size} matrix",
                )
                ratio_efficiency_residuals.extend(
                    (ratio_estimates.sum(axis=1) - efficiency_target).tolist()
                )

        for repeat in range(3):
            schedule_hashes = {
                _hash_value(
                    controlled[method][repeat],
                    "size_schedule_sha256",
                    path=f"{row_path}.{method}.repeat[{repeat}]",
                )
                for method in EXPECTED_IDENTITIES
            }
            _require(
                len(schedule_hashes) == 1,
                f"{row_path} repeat {repeat} uses different size schedules",
            )
            matching_schedule_groups += 1
            relabel_hashes = {
                _hash_value(
                    controlled[method][repeat],
                    "relabel_permutation_sha256",
                    path=f"{row_path}.{method}.repeat[{repeat}]",
                )
                for method in EXPECTED_IDENTITIES
            }
            _require(
                len(relabel_hashes) == 1,
                f"{row_path} repeat {repeat} uses different relabel permutations",
            )
            matching_relabel_groups += 1

        for method in METHOD_ORDER:
            summary = _mapping(methods[method], path=f"{row_path}.methods.{method}")
            calls = _actual_calls(
                summary,
                path=f"{row_path}.methods.{method}.actual_utility_calls_by_repeat",
            )
            total_actual_calls += sum(calls)
            if method == "inside_greedy_per_size_ratio":
                ratio_actual_calls += sum(calls)

    _require(coverage_cells == 15, "ratio coverage audit must inspect 15 cells")

    audits = {
        method: validate_legacy_report(_legacy_view(report, method))
        for method in EXPECTED_IDENTITIES
    }
    datasets = {audit["dataset"] for audit in audits.values()}
    _require(len(datasets) == 1, "controlled audit views disagree on dataset identity")
    dataset = datasets.pop()
    players = int(next(iter(audits.values()))["validated_shape"]["players"])
    metric_discrepancy = max(
        float(audit["maximum_discrepancies"]["stored_metric_absolute"])
        for audit in audits.values()
    )
    efficiency_residual = max(
        float(audit["maximum_discrepancies"]["sample_estimate_efficiency_residual"])
        for audit in audits.values()
    )

    return {
        "status": "ready_to_share",
        "validated_report": report.get("experiment"),
        "dataset": dataset,
        "checks": {
            "ten_method_identity_and_order": "passed",
            "five_budget_points": "passed",
            "three_repeats_per_method_budget": "passed",
            "controlled_design_and_estimator_identities": "passed",
            "same_size_schedule_for_three_controlled_branches": "passed",
            "same_relabel_permutation_for_three_controlled_branches": "passed",
            "ratio_all_player_size_strata_covered": "passed",
            "ratio_linear_reconstruction_matches_source": "passed",
            "independently_reconstructed_exact_ground_truth": "passed",
            "all_retained_rmse_metrics_recomputed": "passed",
            "physical_utility_call_accounting": "passed",
        },
        "ground_truth_audit": next(iter(audits.values()))["ground_truth_audit"],
        "method_identity_audit": EXPECTED_IDENTITIES,
        "controlled_randomization_audit": {
            "matching_size_schedule_sha256_groups_checked": matching_schedule_groups,
            "matching_relabel_permutation_sha256_groups_checked": matching_relabel_groups,
        },
        "ratio_coverage_audit": {
            "cells_checked": coverage_cells,
            "minimum_inclusion_count": int(minimum_inclusion),
            "minimum_exclusion_count": int(minimum_exclusion),
            "maximum_absolute_inclusion_count_deviation": maximum_absolute_deviation,
            "maximum_relative_inclusion_count_deviation": maximum_relative_deviation,
            "maximum_linear_reconstruction_abs_difference": maximum_reconstruction_difference,
            "reconstruction_tolerance": _RECONSTRUCTION_TOLERANCE,
        },
        "ratio_efficiency_audit": {
            "mean_signed_residual": float(np.mean(ratio_efficiency_residuals)),
            "mean_absolute_residual": float(np.mean(np.abs(ratio_efficiency_residuals))),
            "maximum_absolute_residual": float(np.max(np.abs(ratio_efficiency_residuals))),
        },
        "validated_shape": {
            "players": players,
            "methods": len(METHOD_ORDER),
            "budget_points": 5,
            "repeats": 3,
            "per_repeat_estimates_checked": len(METHOD_ORDER) * 5 * 3,
            "scalar_player_estimates_checked": len(METHOD_ORDER) * 5 * 3 * players,
        },
        "call_accounting": {
            "total_actual_utility_calls": total_actual_calls,
            "ratio_branch_actual_utility_calls": ratio_actual_calls,
        },
        "maximum_discrepancies": {
            "stored_metric_absolute": metric_discrepancy,
            "sample_estimate_efficiency_residual": efficiency_residual,
            "linear_reconstruction_absolute": maximum_reconstruction_difference,
        },
        "caveats": [
            "The per-size Greedy design is not exactly 1-balanced, so its official ratio estimate uses unequal observed denominators; complete coverage and uniform batch-wide relabeling preserve the estimator's symmetry, while the unequal denominators can still change finite-sample variance.",
            "Three repeats give only coarse uncertainty intervals; interpret small curve gaps cautiously.",
            "TMC-Shapley is positioned at observed physical calls after truncation.",
        ],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    validation = validate_report(report)
    output = args.output or args.report.with_name(f"{args.report.stem}_validation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{validation['status']}: saved {output}")


if __name__ == "__main__":
    main()

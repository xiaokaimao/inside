"""Audit the nine-method global-vs-per-size INSIDE comparison.

The scope experiment differs from the established eight-method report only
by replacing ``inside_greedy`` with two explicitly identified variants:

* ``inside_greedy_global`` uses the global radially weighted frame; and
* ``inside_greedy_per_size`` uses an unweighted operator for every size.

Both variants must retain the same systematic size schedule and final random
relabel permutation for each budget/repeat pair.  The validator first checks
these experimental-control invariants and then independently runs the
established exact-truth, metric, and physical-call audit once for each Greedy
variant.  This avoids weakening the old eight-method contract or duplicating
its analytic voting-game logic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np

from experiments.plot_inside_comparison import (
    METHOD_ORDER as LEGACY_METHOD_ORDER,
)
from experiments.validate_inside_comparison import (
    validate_report as validate_legacy_report,
)


METHOD_ORDER = (
    "inside_greedy_global",
    "inside_greedy_per_size",
    "inside_orbit",
    "ofa_iid_linear",
    "ofa_iid_ratio",
    "cc",
    "s_diff",
    "kernel_shap",
    "tmc_shapley",
)

EXPECTED_SCOPE = {
    "inside_greedy_global": "global_weighted",
    "inside_greedy_per_size": "per_size",
}

_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _configured_methods(configuration: Mapping[str, Any]) -> tuple[str, ...]:
    raw = configuration.get("methods", configuration.get("method_order"))
    _require(
        isinstance(raw, (list, tuple)),
        "configuration methods must be a list",
    )
    return tuple(str(value) for value in raw)


def _scope_from_diagnostic(diagnostic: Mapping[str, Any], *, path: str) -> str:
    nested = diagnostic.get("design_diagnostics")
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    value = diagnostic.get(
        "second_moment_scope", nested_mapping.get("second_moment_scope")
    )
    _require(value is not None, f"{path} has no second_moment_scope identity")
    return str(value)


def _schedule_hash(diagnostic: Mapping[str, Any], *, path: str) -> str:
    nested = diagnostic.get("design_diagnostics")
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    value = diagnostic.get(
        "size_schedule_sha256",
        diagnostic.get(
            "size_schedule_sha256_int64",
            nested_mapping.get(
                "size_schedule_sha256",
                nested_mapping.get("size_schedule_sha256_int64"),
            ),
        ),
    )
    result = str(value or "").lower()
    _require(
        _HASH_PATTERN.fullmatch(result) is not None,
        f"{path} must retain a 64-character size-schedule SHA-256",
    )
    return result


def _relabel_hash(diagnostic: Mapping[str, Any], *, path: str) -> str:
    nested = diagnostic.get("design_diagnostics")
    nested_mapping = nested if isinstance(nested, Mapping) else {}
    result = str(
        diagnostic.get(
            "relabel_permutation_sha256",
            nested_mapping.get("relabel_permutation_sha256", ""),
        )
        or ""
    ).lower()
    _require(
        _HASH_PATTERN.fullmatch(result) is not None,
        f"{path} must retain a 64-character relabel-permutation SHA-256",
    )
    return result


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


def _legacy_view(
    report: Mapping[str, Any], scope_method: str
) -> dict[str, Any]:
    """Build a shallow eight-method view for the established validator."""
    configuration = dict(
        _mapping(report.get("configuration"), path="configuration")
    )
    configuration["methods"] = list(LEGACY_METHOD_ORDER)
    if "method_order" in configuration:
        configuration["method_order"] = list(LEGACY_METHOD_ORDER)
    scope_configuration = configuration.get("inside_greedy")
    if isinstance(scope_configuration, Mapping):
        # The scope runner uses a clearer dedicated section.  Translate only
        # the shared controls needed by the established eight-method audit;
        # the scope identities themselves are checked on the untouched report
        # before either legacy view is constructed.
        configuration["inside"] = {
            "candidate_pool": scope_configuration.get("candidate_pool"),
            "greedy_mean_balance_mode": scope_configuration.get(
                "mean_balance_mode"
            ),
            "greedy_mean_balance_lambda0": scope_configuration.get(
                "mean_balance_lambda0"
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
        methods = {
            method: source_methods[scope_method]
            if method == "inside_greedy"
            else source_methods[method]
            for method in LEGACY_METHOD_ORDER
        }
        row["methods"] = methods
        results[str(budget)] = row

    view = dict(report)
    view["configuration"] = configuration
    view["results_by_inner_budget"] = results
    return view


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


def validate_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Raise on a failed audit and return a compact durable result otherwise."""
    _require(isinstance(report, Mapping), "report root must be an object")
    _require(report.get("status") == "complete", "report status must be complete")
    configuration = _mapping(report.get("configuration"), path="configuration")
    _require(
        _configured_methods(configuration) == METHOD_ORDER,
        "configuration method order must contain exactly the nine scope methods",
    )
    _require(
        int(configuration.get("repeats", -1)) == 3,
        "scope comparison must use exactly three repeats",
    )

    results = _mapping(
        report.get("results_by_inner_budget"), path="results_by_inner_budget"
    )
    _require(len(results) == 5, "scope comparison must contain five budgets")

    schedule_pairs_checked = 0
    relabel_pairs_checked = 0
    total_actual_calls = 0
    for budget, raw_row in results.items():
        row_path = f"results_by_inner_budget.{budget}"
        row = _mapping(raw_row, path=row_path)
        methods = _mapping(row.get("methods"), path=f"{row_path}.methods")
        _require(
            set(methods) == set(METHOD_ORDER),
            f"{row_path} has the wrong nine-method set",
        )

        scope_diagnostics: dict[str, list[Mapping[str, Any]]] = {}
        for method in EXPECTED_SCOPE:
            method_path = f"{row_path}.methods.{method}"
            summary = _mapping(methods[method], path=method_path)
            diagnostics = _diagnostics(summary, path=method_path)
            scope_diagnostics[method] = diagnostics
            for repeat, diagnostic in enumerate(diagnostics):
                diagnostic_path = f"{method_path}.diagnostics_by_repeat[{repeat}]"
                observed_scope = _scope_from_diagnostic(
                    diagnostic, path=diagnostic_path
                )
                _require(
                    observed_scope == EXPECTED_SCOPE[method],
                    f"{diagnostic_path} identifies {observed_scope!r}, expected "
                    f"{EXPECTED_SCOPE[method]!r}",
                )
                expected_design_method = (
                    "frame_coupled"
                    if method == "inside_greedy_global"
                    else "frame_coupled_per_size"
                )
                _require(
                    str(diagnostic.get("design_method"))
                    == expected_design_method,
                    f"{diagnostic_path} has the wrong design method identity",
                )
                estimator = diagnostic.get("estimator")
                if estimator is not None:
                    _require(
                        str(estimator) == "coupled_linear",
                        f"{diagnostic_path} must use the coupled-linear estimator",
                    )

        for repeat in range(3):
            global_hash = _schedule_hash(
                scope_diagnostics["inside_greedy_global"][repeat],
                path=(
                    f"{row_path}.methods.inside_greedy_global."
                    f"diagnostics_by_repeat[{repeat}]"
                ),
            )
            per_size_hash = _schedule_hash(
                scope_diagnostics["inside_greedy_per_size"][repeat],
                path=(
                    f"{row_path}.methods.inside_greedy_per_size."
                    f"diagnostics_by_repeat[{repeat}]"
                ),
            )
            _require(
                global_hash == per_size_hash,
                f"{row_path} repeat {repeat} uses different size schedules",
            )
            schedule_pairs_checked += 1
            global_relabel_hash = _relabel_hash(
                scope_diagnostics["inside_greedy_global"][repeat],
                path=(
                    f"{row_path}.methods.inside_greedy_global."
                    f"diagnostics_by_repeat[{repeat}]"
                ),
            )
            per_size_relabel_hash = _relabel_hash(
                scope_diagnostics["inside_greedy_per_size"][repeat],
                path=(
                    f"{row_path}.methods.inside_greedy_per_size."
                    f"diagnostics_by_repeat[{repeat}]"
                ),
            )
            _require(
                global_relabel_hash == per_size_relabel_hash,
                f"{row_path} repeat {repeat} uses different relabel permutations",
            )
            relabel_pairs_checked += 1

        for method in METHOD_ORDER:
            summary = _mapping(
                methods[method], path=f"{row_path}.methods.{method}"
            )
            total_actual_calls += sum(
                _actual_calls(
                    summary,
                    path=(
                        f"{row_path}.methods.{method}."
                        "actual_utility_calls_by_repeat"
                    ),
                )
            )

    # Reuse the independently tested exact truth, retained-estimate, metric,
    # and call-accounting audit.  Each pass substitutes exactly one scope
    # variant for the legacy ``inside_greedy`` identity, so all nine methods
    # are checked without changing the established eight-method semantics.
    global_audit = validate_legacy_report(
        _legacy_view(report, "inside_greedy_global")
    )
    per_size_audit = validate_legacy_report(
        _legacy_view(report, "inside_greedy_per_size")
    )
    _require(
        global_audit["dataset"] == per_size_audit["dataset"],
        "the two scope audit views disagree on dataset identity",
    )
    global_max = global_audit["maximum_discrepancies"]
    per_size_max = per_size_audit["maximum_discrepancies"]
    players = int(global_audit["validated_shape"]["players"])

    return {
        "status": "ready_to_share",
        "validated_report": report.get("experiment"),
        "dataset": global_audit["dataset"],
        "checks": {
            "nine_method_identity_and_order": "passed",
            "global_weighted_scope_identity": "passed",
            "per_size_scope_identity": "passed",
            "same_size_schedule_per_budget_repeat": "passed",
            "same_relabel_permutation_per_budget_repeat": "passed",
            "five_budget_points": "passed",
            "three_repeats_per_method_budget": "passed",
            "independently_reconstructed_exact_ground_truth": "passed",
            "all_retained_rmse_metrics_recomputed": "passed",
            "physical_utility_call_accounting": "passed",
        },
        "ground_truth_audit": global_audit["ground_truth_audit"],
        "method_identity_audit": {
            "inside_greedy_global": {
                "second_moment_scope": "global_weighted",
                "estimator": "coupled_linear",
            },
            "inside_greedy_per_size": {
                "second_moment_scope": "per_size",
                "estimator": "coupled_linear",
            },
            "controlled_components": [
                "outer size schedule",
                "seed",
                "candidate pool",
                "final relabel permutation",
                "coupled-linear estimator",
            ],
        },
        "size_schedule_audit": {
            "matching_sha256_pairs_checked": schedule_pairs_checked,
        },
        "relabel_permutation_audit": {
            "matching_sha256_pairs_checked": relabel_pairs_checked,
        },
        "validated_shape": {
            "players": players,
            "methods": len(METHOD_ORDER),
            "budget_points": 5,
            "repeats": 3,
            "per_repeat_estimates_checked": len(METHOD_ORDER) * 5 * 3,
            "scalar_player_estimates_checked": len(METHOD_ORDER)
            * 5
            * 3
            * players,
        },
        "call_accounting": {
            "total_actual_utility_calls": total_actual_calls,
        },
        "maximum_discrepancies": {
            "stored_metric_absolute": max(
                float(global_max["stored_metric_absolute"]),
                float(per_size_max["stored_metric_absolute"]),
            ),
            "sample_estimate_efficiency_residual": max(
                float(global_max["sample_estimate_efficiency_residual"]),
                float(per_size_max["sample_estimate_efficiency_residual"]),
            ),
        },
        "caveats": [
            (
                "Three repeats give only coarse uncertainty intervals; "
                "interpret small curve gaps cautiously."
            ),
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
    output = args.output or args.report.with_name(
        f"{args.report.stem}_validation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{validation['status']}: saved {output}")


if __name__ == "__main__":
    main()

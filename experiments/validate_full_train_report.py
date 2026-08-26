"""Read-only consistency audit for full-training-set experiment reports.

The experiment JSON files intentionally retain every repeat-level Shapley
estimate.  This module uses those raw estimates to recompute the headline
metrics and catches incomplete or internally inconsistent reports before a
plot or written conclusion is produced.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METHODS = (
    "official_ofa_fixed_ratio",
    "iid_linear_ofa",
    "frame_coupled_linear",
    "frame_orbit_ratio",
)
EFFICIENCY_CONSTRAINED_METHODS = (
    "iid_linear_ofa",
    "frame_coupled_linear",
    "frame_orbit_ratio",
)
CC_METHOD = "official_cc_basic"
NEW_EFFICIENCY_CONSTRAINED_METHODS = frozenset(
    {
        "gels_shapley",
        "kernel_shap_sampled",
        "group_testing",
        "diff",
        "s_diff",
    }
)


class _Audit:
    def __init__(self, *, rtol: float, atol: float) -> None:
        self.rtol = rtol
        self.atol = atol
        self.errors: list[str] = []

    def equal(self, path: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            self.errors.append(
                f"{path}: expected {expected!r}, found {actual!r}"
            )

    def close(self, path: str, actual: Any, expected: Any) -> None:
        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (TypeError, ValueError):
            self.errors.append(
                f"{path}: expected a finite number, found {actual!r}"
            )
            return
        if not (
            math.isfinite(actual_float)
            and math.isfinite(expected_float)
            and math.isclose(
                actual_float,
                expected_float,
                rel_tol=self.rtol,
                abs_tol=self.atol,
            )
        ):
            self.errors.append(
                f"{path}: expected {expected_float:.17g}, "
                f"found {actual_float:.17g}"
            )

    def require(self, condition: bool, message: str) -> None:
        if not condition:
            self.errors.append(message)


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return None


def _finite_vector(
    audit: _Audit,
    value: Any,
    *,
    path: str,
    length: int,
    nonnegative: bool = False,
) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        audit.errors.append(f"{path}: cannot convert to a numeric vector")
        return None
    if array.shape != (length,):
        audit.errors.append(
            f"{path}: expected shape ({length},), found {array.shape}"
        )
        return None
    if not np.isfinite(array).all():
        audit.errors.append(f"{path}: contains non-finite values")
        return None
    if nonnegative and np.any(array < 0):
        audit.errors.append(f"{path}: contains negative values")
        return None
    return array


def _finite_estimates(
    audit: _Audit,
    value: Any,
    *,
    path: str,
    repeats: int,
    num_players: int,
) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        audit.errors.append(f"{path}: cannot convert to a numeric matrix")
        return None
    expected_shape = (repeats, num_players)
    if array.shape != expected_shape:
        audit.errors.append(
            f"{path}: expected shape {expected_shape}, found {array.shape}"
        )
        return None
    if not np.isfinite(array).all():
        audit.errors.append(f"{path}: contains non-finite values")
        return None
    return array


def _nonnegative_integer_vector(
    audit: _Audit,
    value: Any,
    *,
    path: str,
    length: int,
) -> list[int] | None:
    """Return a strict JSON integer vector, rejecting bools and floats."""
    if not isinstance(value, list) or len(value) != length:
        shape = len(value) if isinstance(value, list) else type(value).__name__
        audit.errors.append(
            f"{path}: expected a list of {length} nonnegative integers, "
            f"found {shape!r}"
        )
        return None
    integers: list[int] = []
    for index, raw in enumerate(value):
        integer = _integer(raw)
        if integer is None or integer < 0:
            audit.errors.append(
                f"{path}.{index}: expected a nonnegative integer, "
                f"found {raw!r}"
            )
            return None
        integers.append(integer)
    return integers


def _validate_ground_truth(
    audit: _Audit,
    ground_truth: Mapping[str, Any],
    *,
    configuration: Mapping[str, Any],
    num_players: int,
    boundary_calls: int,
    efficiency_target: float,
    efficiency_tolerance: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    diagnostics: dict[str, Any] = {}
    values = _finite_vector(
        audit,
        ground_truth.get("values"),
        path="ground_truth.values",
        length=num_players,
    )
    standard_errors = _finite_vector(
        audit,
        ground_truth.get("standard_errors"),
        path="ground_truth.standard_errors",
        length=num_players,
        nonnegative=True,
    )
    simultaneous_widths = _finite_vector(
        audit,
        ground_truth.get("simultaneous_half_widths"),
        path="ground_truth.simultaneous_half_widths",
        length=num_players,
        nonnegative=True,
    )
    if standard_errors is not None:
        audit.close(
            "ground_truth.rmse_standard_error",
            ground_truth.get("rmse_standard_error"),
            np.sqrt(np.mean(np.square(standard_errors))),
        )
    if simultaneous_widths is not None:
        audit.close(
            "ground_truth.max_simultaneous_half_width",
            ground_truth.get("max_simultaneous_half_width"),
            simultaneous_widths.max(),
        )

    pairs = _integer(ground_truth.get("independent_pair_units"))
    configured_pairs = _integer(configuration.get("gt_pairs"))
    if pairs is None or pairs < 2:
        audit.errors.append(
            "ground_truth.independent_pair_units: expected an integer >= 2"
        )
    else:
        audit.equal("configuration.gt_pairs", configured_pairs, pairs)
        expected_permutations = 2 * pairs
        expected_conceptual = 2 * pairs * (num_players - 1)
        expected_physical = 2 * pairs * (num_players - 3)
        expected_saved = expected_conceptual - expected_physical
        expected_equivalent = 2 + expected_conceptual
        audit.equal(
            "ground_truth.permutations",
            ground_truth.get("permutations"),
            expected_permutations,
        )
        audit.equal(
            "ground_truth.conceptual_internal_prefix_calls",
            ground_truth.get("conceptual_internal_prefix_calls"),
            expected_conceptual,
        )
        audit.equal(
            "ground_truth.physical_internal_prefix_calls",
            ground_truth.get("physical_internal_prefix_calls"),
            expected_physical,
        )
        audit.equal(
            "ground_truth.boundary_reuse_saved_calls",
            ground_truth.get("boundary_reuse_saved_calls"),
            expected_saved,
        )
        audit.equal(
            "ground_truth.permutation_path_equivalent_utility_calls",
            ground_truth.get(
                "permutation_path_equivalent_utility_calls"
            ),
            expected_equivalent,
        )
        audit.equal(
            "ground_truth.shared_boundary_calls_physically_evaluated",
            ground_truth.get(
                "shared_boundary_calls_physically_evaluated"
            ),
            boundary_calls,
        )
        physical_total = expected_physical + boundary_calls
        diagnostics.update(
            {
                "pair_units": pairs,
                "permutations": expected_permutations,
                "physical_total_calls": physical_total,
                "permutation_path_equivalent_calls": expected_equivalent,
                "physical_to_equivalent_call_ratio": (
                    physical_total / expected_equivalent
                ),
                "boundary_reuse_saved_internal_calls": expected_saved,
            }
        )

    if values is not None:
        efficiency_gap = abs(float(values.sum()) - efficiency_target)
        diagnostics["efficiency_gap"] = efficiency_gap
        audit.require(
            efficiency_gap <= efficiency_tolerance,
            "ground_truth.values: efficiency gap "
            f"{efficiency_gap:.3e} exceeds "
            f"{efficiency_tolerance:.3e}",
        )
    return values, diagnostics


def _validate_method(
    audit: _Audit,
    summary: Mapping[str, Any],
    *,
    path: str,
    ground_truth: np.ndarray,
    repeats: int,
    num_players: int,
    efficiency_target: float,
) -> tuple[np.ndarray | None, dict[str, float]]:
    estimates = _finite_estimates(
        audit,
        summary.get("estimates"),
        path=f"{path}.estimates",
        repeats=repeats,
        num_players=num_players,
    )
    if estimates is None:
        return None, {}
    aggregate_rmse = float(
        np.sqrt(np.mean(np.square(estimates - ground_truth[None, :])))
    )
    efficiency_gaps = np.abs(
        estimates.sum(axis=1) - efficiency_target
    )
    mean_gap = float(efficiency_gaps.mean())
    max_gap = float(efficiency_gaps.max())
    audit.close(
        f"{path}.aggregate_rmse",
        summary.get("aggregate_rmse"),
        aggregate_rmse,
    )
    audit.close(
        f"{path}.mean_efficiency_gap",
        summary.get("mean_efficiency_gap"),
        mean_gap,
    )
    audit.close(
        f"{path}.max_efficiency_gap",
        summary.get("max_efficiency_gap"),
        max_gap,
    )
    return estimates, {
        "aggregate_rmse": aggregate_rmse,
        "mean_efficiency_gap": mean_gap,
        "max_efficiency_gap": max_gap,
    }


def _validate_cc_accounting(
    audit: _Audit,
    summary: Mapping[str, Any],
    *,
    budget_result: Mapping[str, Any],
    path: str,
    repeats: int,
    num_players: int,
    total_calls: int,
) -> dict[str, Any]:
    """Audit the official basic-CC pair and missing-cell accounting."""
    diagnostics: dict[str, Any] = {}
    pairs = _integer(summary.get("complementary_contributions_per_estimate"))
    calls = _integer(summary.get("utility_calls_per_estimate"))
    expected_pairs = total_calls // 2
    audit.require(
        total_calls % 2 == 0,
        f"{path}: total calls must be even for complementary pairs",
    )
    audit.equal(f"{path}.utility_calls_per_estimate", calls, total_calls)
    audit.equal(
        f"{path}.complementary_contributions_per_estimate",
        pairs,
        expected_pairs,
    )

    raw_fractions = summary.get("missing_stratum_fractions")
    fractions: np.ndarray | None = None
    try:
        candidate = np.asarray(raw_fractions, dtype=np.float64)
    except (TypeError, ValueError):
        audit.errors.append(
            f"{path}.missing_stratum_fractions: expected numeric values"
        )
    else:
        if candidate.shape != (repeats,):
            audit.errors.append(
                f"{path}.missing_stratum_fractions: expected shape "
                f"({repeats},), found {candidate.shape}"
            )
        elif (
            not np.isfinite(candidate).all()
            or np.any(candidate < 0)
            or np.any(candidate > 1)
        ):
            audit.errors.append(
                f"{path}.missing_stratum_fractions: expected finite values "
                "in [0, 1]"
            )
        else:
            fractions = candidate
            mean_fraction = float(candidate.mean())
            max_fraction = float(candidate.max())
            audit.close(
                f"{path}.mean_missing_stratum_fraction",
                summary.get("mean_missing_stratum_fraction"),
                mean_fraction,
            )
            audit.close(
                f"{path}.max_missing_stratum_fraction",
                summary.get("max_missing_stratum_fraction"),
                max_fraction,
            )
            audit.close(
                f"{path.rsplit('.methods.', 1)[0]}.cc_missing_stratum_fraction_mean",
                budget_result.get("cc_missing_stratum_fraction_mean"),
                mean_fraction,
            )
            diagnostics.update(
                {
                    "mean_missing_stratum_fraction": mean_fraction,
                    "max_missing_stratum_fraction": max_fraction,
                }
            )

    raw_repeat_diagnostics = summary.get("cc_diagnostics_by_repeat")
    if (
        not isinstance(raw_repeat_diagnostics, list)
        or len(raw_repeat_diagnostics) != repeats
    ):
        audit.errors.append(
            f"{path}.cc_diagnostics_by_repeat: expected {repeats} entries"
        )
    else:
        for repeat, entry in enumerate(raw_repeat_diagnostics):
            entry_path = f"{path}.cc_diagnostics_by_repeat.{repeat}"
            if not isinstance(entry, Mapping):
                audit.errors.append(f"{entry_path}: expected an object")
                continue
            audit.equal(
                f"{entry_path}.num_pairs",
                entry.get("num_pairs"),
                expected_pairs,
            )
            audit.equal(
                f"{entry_path}.utility_evaluations",
                entry.get("utility_evaluations"),
                total_calls,
            )
            missing = _integer(entry.get("missing_strata"))
            minimum_positive = _integer(entry.get("minimum_positive_count"))
            num_tasks = _integer(entry.get("num_tasks"))
            audit.require(
                missing is not None and 0 <= missing <= num_players**2,
                f"{entry_path}.missing_strata: expected an integer in "
                f"[0, {num_players**2}]",
            )
            audit.require(
                minimum_positive is not None and minimum_positive >= 0,
                f"{entry_path}.minimum_positive_count: expected a "
                "nonnegative integer",
            )
            audit.require(
                num_tasks is not None and num_tasks >= 1,
                f"{entry_path}.num_tasks: expected a positive integer",
            )
            if fractions is not None and missing is not None:
                audit.close(
                    f"{entry_path}.missing_strata_fraction",
                    fractions[repeat],
                    missing / (num_players**2),
                )
    return diagnostics


def _validate_new_method_accounting(
    audit: _Audit,
    summary: Mapping[str, Any],
    *,
    path: str,
    repeats: int,
    num_players: int,
    total_calls: int,
    requested_num_tasks: int | None,
    method: str,
) -> dict[str, Any]:
    """Audit strict total-call accounting for one locally ported method."""
    diagnostics: dict[str, Any] = {}
    target_calls = _integer(summary.get("target_total_call_budget"))
    audit.equal(
        f"{path}.target_total_call_budget", target_calls, total_calls
    )
    actual_calls = _nonnegative_integer_vector(
        audit,
        summary.get("actual_utility_calls_per_estimate"),
        path=f"{path}.actual_utility_calls_per_estimate",
        length=repeats,
    )
    unused_calls = _nonnegative_integer_vector(
        audit,
        summary.get("unused_calls_per_estimate"),
        path=f"{path}.unused_calls_per_estimate",
        length=repeats,
    )
    if actual_calls is not None:
        audit.close(
            f"{path}.mean_actual_utility_calls",
            summary.get("mean_actual_utility_calls"),
            np.mean(actual_calls),
        )
        audit.require(
            all(value <= total_calls for value in actual_calls),
            f"{path}.actual_utility_calls_per_estimate: calls exceed "
            f"the total-call cap {total_calls}",
        )
        diagnostics["actual_utility_calls_per_estimate"] = actual_calls
    if unused_calls is not None:
        audit.close(
            f"{path}.mean_unused_calls",
            summary.get("mean_unused_calls"),
            np.mean(unused_calls),
        )
        diagnostics["unused_calls_per_estimate"] = unused_calls
    if actual_calls is not None and unused_calls is not None:
        for repeat, (actual, unused) in enumerate(
            zip(actual_calls, unused_calls, strict=True)
        ):
            audit.require(
                actual + unused == total_calls,
                f"{path}.call_accounting.{repeat}: actual {actual} + "
                f"unused {unused} != cap {total_calls}",
            )

    raw_repeat_diagnostics = summary.get("diagnostics_by_repeat")
    if (
        not isinstance(raw_repeat_diagnostics, list)
        or len(raw_repeat_diagnostics) != repeats
    ):
        audit.errors.append(
            f"{path}.diagnostics_by_repeat: expected {repeats} entries"
        )
        return diagnostics

    for repeat, entry in enumerate(raw_repeat_diagnostics):
        entry_path = f"{path}.diagnostics_by_repeat.{repeat}"
        if not isinstance(entry, Mapping):
            audit.errors.append(f"{entry_path}: expected an object")
            continue
        audit.equal(
            f"{entry_path}.target_call_budget",
            _integer(entry.get("target_call_budget")),
            total_calls,
        )
        expected_actual = (
            actual_calls[repeat] if actual_calls is not None else None
        )
        expected_unused = (
            unused_calls[repeat] if unused_calls is not None else None
        )
        diagnostic_actual = _integer(entry.get("utility_evaluations"))
        diagnostic_unused = _integer(entry.get("unused_calls"))
        if expected_actual is not None:
            audit.equal(
                f"{entry_path}.utility_evaluations",
                diagnostic_actual,
                expected_actual,
            )
        elif diagnostic_actual is None or diagnostic_actual < 0:
            audit.errors.append(
                f"{entry_path}.utility_evaluations: expected a "
                "nonnegative integer"
            )
        if expected_unused is not None:
            audit.equal(
                f"{entry_path}.unused_calls",
                diagnostic_unused,
                expected_unused,
            )
        elif diagnostic_unused is None or diagnostic_unused < 0:
            audit.errors.append(
                f"{entry_path}.unused_calls: expected a nonnegative integer"
            )
        if diagnostic_actual is not None and diagnostic_unused is not None:
            audit.require(
                diagnostic_actual + diagnostic_unused == total_calls,
                f"{entry_path}: utility_evaluations {diagnostic_actual} + "
                f"unused_calls {diagnostic_unused} != cap {total_calls}",
            )
        diagnostic_players = _integer(entry.get("num_players"))
        audit.equal(
            f"{entry_path}.num_players",
            diagnostic_players,
            num_players,
        )
        diagnostic_tasks = _integer(entry.get("num_tasks"))
        audit.require(
            diagnostic_tasks is not None and diagnostic_tasks >= 1,
            f"{entry_path}.num_tasks: expected a positive integer",
        )
        if requested_num_tasks is not None:
            audit.equal(
                f"{entry_path}.requested_num_tasks",
                _integer(entry.get("requested_num_tasks")),
                requested_num_tasks,
            )
        if method == "gels_shapley":
            audit.equal(
                f"{entry_path}.method",
                entry.get("method"),
                "gels_shapley",
            )
            audit.equal(
                f"{entry_path}.solver",
                entry.get("solver"),
                "official_algorithm_3_self_normalized_ratio",
            )
            audit.equal(
                f"{entry_path}.boundary_utility_evaluations",
                _integer(entry.get("boundary_utility_evaluations")),
                2,
            )
            inner = _integer(entry.get("inner_utility_evaluations"))
            samples = _integer(entry.get("num_samples"))
            if diagnostic_actual is not None:
                audit.equal(
                    f"{entry_path}.inner_utility_evaluations",
                    inner,
                    diagnostic_actual - 2,
                )
                audit.equal(
                    f"{entry_path}.num_samples",
                    samples,
                    diagnostic_actual - 2,
                )
            layer_counts = _nonnegative_integer_vector(
                audit,
                entry.get("layer_counts"),
                path=f"{entry_path}.layer_counts",
                length=num_players - 1,
            )
            inclusion_counts = _nonnegative_integer_vector(
                audit,
                entry.get("inclusion_counts"),
                path=f"{entry_path}.inclusion_counts",
                length=num_players,
            )
            zero_players = _integer(entry.get("zero_inclusion_players"))
            if inclusion_counts is not None:
                audit.equal(
                    f"{entry_path}.zero_inclusion_players",
                    zero_players,
                    sum(value == 0 for value in inclusion_counts),
                )
            if layer_counts is not None and inclusion_counts is not None:
                weighted_layer_total = sum(
                    (size + 1) * count
                    for size, count in enumerate(layer_counts)
                )
                audit.equal(
                    f"{entry_path}.inclusion_count_identity",
                    sum(inclusion_counts),
                    weighted_layer_total,
                )
    return diagnostics


def validate_report(
    report: Mapping[str, Any],
    *,
    expected_status: str = "complete",
    expected_dataset: str | None = None,
    expected_num_players: int | None = None,
    expected_repeats: int | None = None,
    expected_gt_pairs: int | None = None,
    expected_jobs: int | None = None,
    expected_inner_budgets: Sequence[int] | None = None,
    expected_budget_multipliers: Sequence[int] | None = None,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    efficiency_tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Validate a parsed report and return serializable diagnostics."""
    audit = _Audit(rtol=rtol, atol=atol)
    if not isinstance(report, Mapping):
        return {
            "passed": False,
            "errors": ["report root must be a JSON object"],
            "diagnostics": {},
        }

    audit.equal("status", report.get("status"), expected_status)
    configuration = report.get("configuration")
    dataset = report.get("dataset")
    boundary = report.get("boundary")
    ground_truth = report.get("ground_truth")
    results = report.get("results_by_inner_budget")
    for path, value in (
        ("configuration", configuration),
        ("dataset", dataset),
        ("boundary", boundary),
        ("ground_truth", ground_truth),
        ("results_by_inner_budget", results),
    ):
        if not isinstance(value, Mapping):
            audit.errors.append(f"{path}: expected a JSON object")
    if audit.errors and not all(
        isinstance(value, Mapping)
        for value in (
            configuration,
            dataset,
            boundary,
            ground_truth,
            results,
        )
    ):
        return {
            "passed": False,
            "errors": audit.errors,
            "diagnostics": {},
        }
    assert isinstance(configuration, Mapping)
    assert isinstance(dataset, Mapping)
    assert isinstance(boundary, Mapping)
    assert isinstance(ground_truth, Mapping)
    assert isinstance(results, Mapping)

    num_players = _integer(configuration.get("num_players"))
    repeats = _integer(configuration.get("repeats"))
    if num_players is None or num_players < 4:
        audit.errors.append(
            "configuration.num_players: expected an integer >= 4"
        )
        num_players = 0
    if repeats is None or repeats < 1:
        audit.errors.append(
            "configuration.repeats: expected a positive integer"
        )
        repeats = 0
    if expected_num_players is not None:
        audit.equal(
            "configuration.num_players", num_players, expected_num_players
        )
    if expected_repeats is not None:
        audit.equal("configuration.repeats", repeats, expected_repeats)
    if expected_gt_pairs is not None:
        audit.equal(
            "configuration.gt_pairs",
            configuration.get("gt_pairs"),
            expected_gt_pairs,
        )
    if expected_jobs is not None:
        audit.equal(
            "configuration.jobs", configuration.get("jobs"), expected_jobs
        )
    dataset_name = dataset.get("dataset")
    if expected_dataset is not None:
        audit.equal("dataset.dataset", dataset_name, expected_dataset)

    raw_budgets = configuration.get("inner_budgets")
    raw_total_budgets = configuration.get("total_call_budgets")
    if not isinstance(raw_budgets, list) or not all(
        _integer(value) is not None and int(value) > 0
        for value in raw_budgets
    ):
        audit.errors.append(
            "configuration.inner_budgets: expected positive integers"
        )
        budgets: list[int] = []
    else:
        budgets = [int(value) for value in raw_budgets]
        audit.require(
            len(set(budgets)) == len(budgets),
            "configuration.inner_budgets: values must be distinct",
        )
    if expected_inner_budgets is not None:
        audit.equal(
            "configuration.inner_budgets",
            budgets,
            list(expected_inner_budgets),
        )
    multipliers = configuration.get("budget_multipliers")
    if multipliers is not None:
        if not isinstance(multipliers, list) or not all(
            _integer(value) is not None and int(value) > 0
            for value in multipliers
        ):
            audit.errors.append(
                "configuration.budget_multipliers: expected positive "
                "integers"
            )
        elif num_players:
            audit.equal(
                "configuration.inner_budgets_from_multipliers",
                budgets,
                [num_players * int(value) for value in multipliers],
            )
    if expected_budget_multipliers is not None:
        audit.equal(
            "configuration.budget_multipliers",
            multipliers,
            list(expected_budget_multipliers),
        )

    baseline_comparison = configuration.get("baseline_comparison")
    new_methods: tuple[str, ...] = ()
    requested_num_tasks: int | None = None
    if baseline_comparison is not None:
        if not isinstance(baseline_comparison, Mapping):
            audit.errors.append(
                "configuration.baseline_comparison: expected an object"
            )
        else:
            raw_new_methods = baseline_comparison.get("new_methods")
            if not isinstance(raw_new_methods, list) or not all(
                isinstance(method, str) and bool(method)
                for method in raw_new_methods
            ):
                audit.errors.append(
                    "configuration.baseline_comparison.new_methods: "
                    "expected a list of nonempty method keys"
                )
            else:
                new_methods = tuple(raw_new_methods)
                audit.require(
                    len(set(new_methods)) == len(new_methods),
                    "configuration.baseline_comparison.new_methods: "
                    "method keys must be distinct",
                )
                overlap = sorted(
                    set(new_methods).intersection((*METHODS, CC_METHOD))
                )
                audit.require(
                    not overlap,
                    "configuration.baseline_comparison.new_methods: "
                    f"existing methods cannot be redeclared: {overlap}",
                )
            raw_num_tasks = baseline_comparison.get("num_tasks")
            requested_num_tasks = _integer(raw_num_tasks)
            if requested_num_tasks is None or requested_num_tasks < 1:
                audit.errors.append(
                    "configuration.baseline_comparison.num_tasks: "
                    "expected a positive integer"
                )
                requested_num_tasks = None
            if "gels_shapley" in new_methods:
                gels_metadata = baseline_comparison.get("gels_shapley")
                gels_path = (
                    "configuration.baseline_comparison.gels_shapley"
                )
                if not isinstance(gels_metadata, Mapping):
                    audit.errors.append(f"{gels_path}: expected an object")
                else:
                    expected_gels_metadata = {
                        "paper_url": (
                            "https://openreview.net/forum?id=lvSMIsztka"
                        ),
                        "official_repository": (
                            "https://github.com/watml/fastpvalue"
                        ),
                        "verified_commit": (
                            "34392c53f8d609aebb5e0c0e57c165411d291a46"
                        ),
                        "downloaded_source_sha256": (
                            "2eff99580289e3f2374a72b720baa0284a2fff713"
                            "41be47d5b757c585ff6e2fa"
                        ),
                        "official_class": "GELS_shapley",
                        "paired_sampling": False,
                    }
                    for key, expected in expected_gels_metadata.items():
                        audit.equal(
                            f"{gels_path}.{key}",
                            gels_metadata.get(key),
                            expected,
                        )

    boundary_calls = 2 * num_players + 2 if num_players else 0
    audit.equal(
        "boundary.utility_calls",
        boundary.get("utility_calls"),
        boundary_calls,
    )
    try:
        empty = float(boundary.get("empty"))
        full = float(boundary.get("full"))
        efficiency_target = float(boundary.get("efficiency_target"))
    except (TypeError, ValueError):
        audit.errors.append(
            "boundary: empty, full, and efficiency_target must be numeric"
        )
        empty = full = efficiency_target = math.nan
    if all(math.isfinite(value) for value in (empty, full, efficiency_target)):
        audit.close(
            "boundary.efficiency_target", efficiency_target, full - empty
        )
    else:
        audit.errors.append(
            "boundary: empty, full, and efficiency_target must be finite"
        )
    expected_totals = [budget + boundary_calls for budget in budgets]
    audit.equal(
        "configuration.total_call_budgets",
        raw_total_budgets,
        expected_totals,
    )

    truth_values, gt_diagnostics = _validate_ground_truth(
        audit,
        ground_truth,
        configuration=configuration,
        num_players=num_players,
        boundary_calls=boundary_calls,
        efficiency_target=efficiency_target,
        efficiency_tolerance=efficiency_tolerance,
    )

    expected_result_keys = {str(value) for value in budgets}
    audit.equal(
        "results_by_inner_budget.keys",
        set(results),
        expected_result_keys,
    )
    cc_presence = []
    for budget in budgets:
        budget_result = results.get(str(budget))
        methods = (
            budget_result.get("methods")
            if isinstance(budget_result, Mapping)
            else None
        )
        cc_presence.append(
            isinstance(methods, Mapping) and CC_METHOD in methods
        )
    has_cc = bool(cc_presence) and all(cc_presence)
    if any(cc_presence) and not has_cc:
        audit.errors.append(
            f"results_by_inner_budget: {CC_METHOD} must be present at every "
            "budget or at none"
        )
    if has_cc:
        cc_configuration = configuration.get("cc_baseline")
        if not isinstance(cc_configuration, Mapping):
            audit.errors.append(
                "configuration.cc_baseline: expected an object when the CC "
                "method is present"
            )
        else:
            audit.equal(
                "configuration.cc_baseline.method_key",
                cc_configuration.get("method_key"),
                CC_METHOD,
            )
            audit.equal(
                "configuration.cc_baseline.missing_stratum_rule",
                cc_configuration.get("missing_stratum_rule"),
                "zero (matches official cc_shap)",
            )
    per_budget_diagnostics: dict[str, Any] = {}
    if truth_values is not None and repeats > 0:
        for budget in budgets:
            key = str(budget)
            budget_result = results.get(key)
            path = f"results_by_inner_budget.{key}"
            if not isinstance(budget_result, Mapping):
                audit.errors.append(f"{path}: expected a JSON object")
                continue
            audit.equal(
                f"{path}.inner_utility_calls",
                budget_result.get("inner_utility_calls"),
                budget,
            )
            audit.equal(
                f"{path}.boundary_utility_calls",
                budget_result.get("boundary_utility_calls"),
                boundary_calls,
            )
            audit.equal(
                f"{path}.total_utility_calls_per_estimate",
                budget_result.get("total_utility_calls_per_estimate"),
                budget + boundary_calls,
            )
            try:
                missing_fraction = float(
                    budget_result.get(
                        "official_missing_stratum_fraction_mean"
                    )
                )
            except (TypeError, ValueError):
                audit.errors.append(
                    f"{path}.official_missing_stratum_fraction_mean: "
                    "expected a finite number in [0, 1]"
                )
                missing_fraction = math.nan
            else:
                valid_missing_fraction = (
                    math.isfinite(missing_fraction)
                    and 0.0 <= missing_fraction <= 1.0
                )
                if not valid_missing_fraction:
                    audit.errors.append(
                        f"{path}.official_missing_stratum_fraction_mean: "
                        f"expected [0, 1], found {missing_fraction!r}"
                    )
            methods = budget_result.get("methods")
            if not isinstance(methods, Mapping):
                audit.errors.append(f"{path}.methods: expected an object")
                continue
            missing_methods = sorted(set(METHODS) - set(methods))
            if missing_methods:
                audit.errors.append(
                    f"{path}.methods: missing {missing_methods}"
                )
            missing_new_methods = sorted(set(new_methods) - set(methods))
            if missing_new_methods:
                audit.errors.append(
                    f"{path}.methods: missing configured new methods "
                    f"{missing_new_methods}"
                )
            method_arrays: dict[str, np.ndarray] = {}
            method_diagnostics: dict[str, dict[str, float]] = {}
            for method in METHODS:
                summary = methods.get(method)
                method_path = f"{path}.methods.{method}"
                if not isinstance(summary, Mapping):
                    continue
                estimates, diagnostics = _validate_method(
                    audit,
                    summary,
                    path=method_path,
                    ground_truth=truth_values,
                    repeats=repeats,
                    num_players=num_players,
                    efficiency_target=efficiency_target,
                )
                if estimates is not None:
                    method_arrays[method] = estimates
                    method_diagnostics[method] = diagnostics
                    if method in EFFICIENCY_CONSTRAINED_METHODS:
                        max_gap = diagnostics["max_efficiency_gap"]
                        audit.require(
                            max_gap <= efficiency_tolerance,
                            f"{method_path}.estimates: efficiency gap "
                            f"{max_gap:.3e} exceeds "
                            f"{efficiency_tolerance:.3e}",
                        )

            if has_cc:
                cc_summary = methods.get(CC_METHOD)
                cc_path = f"{path}.methods.{CC_METHOD}"
                if not isinstance(cc_summary, Mapping):
                    audit.errors.append(f"{cc_path}: expected an object")
                else:
                    estimates, diagnostics = _validate_method(
                        audit,
                        cc_summary,
                        path=cc_path,
                        ground_truth=truth_values,
                        repeats=repeats,
                        num_players=num_players,
                        efficiency_target=efficiency_target,
                    )
                    if estimates is not None:
                        method_arrays[CC_METHOD] = estimates
                        method_diagnostics[CC_METHOD] = diagnostics
                    cc_diagnostics = _validate_cc_accounting(
                        audit,
                        cc_summary,
                        budget_result=budget_result,
                        path=cc_path,
                        repeats=repeats,
                        num_players=num_players,
                        total_calls=budget + boundary_calls,
                    )
                    if CC_METHOD in method_diagnostics:
                        method_diagnostics[CC_METHOD].update(cc_diagnostics)

            for method in new_methods:
                summary = methods.get(method)
                method_path = f"{path}.methods.{method}"
                if not isinstance(summary, Mapping):
                    continue
                estimates, method_summary_diagnostics = _validate_method(
                    audit,
                    summary,
                    path=method_path,
                    ground_truth=truth_values,
                    repeats=repeats,
                    num_players=num_players,
                    efficiency_target=efficiency_target,
                )
                accounting_diagnostics = _validate_new_method_accounting(
                    audit,
                    summary,
                    path=method_path,
                    repeats=repeats,
                    num_players=num_players,
                    total_calls=budget + boundary_calls,
                    requested_num_tasks=requested_num_tasks,
                    method=method,
                )
                method_summary_diagnostics.update(accounting_diagnostics)
                if estimates is not None:
                    method_arrays[method] = estimates
                    method_diagnostics[method] = (
                        method_summary_diagnostics
                    )
                    if method in NEW_EFFICIENCY_CONSTRAINED_METHODS:
                        max_gap = method_summary_diagnostics[
                            "max_efficiency_gap"
                        ]
                        audit.require(
                            max_gap <= efficiency_tolerance,
                            f"{method_path}.estimates: efficiency gap "
                            f"{max_gap:.3e} exceeds "
                            f"{efficiency_tolerance:.3e}",
                        )

            needed = set(METHODS)
            if needed.issubset(method_diagnostics):
                official_rmse = method_diagnostics[
                    "official_ofa_fixed_ratio"
                ]["aggregate_rmse"]
                iid_rmse = method_diagnostics["iid_linear_ofa"][
                    "aggregate_rmse"
                ]
                coupled_rmse = method_diagnostics[
                    "frame_coupled_linear"
                ]["aggregate_rmse"]
                orbit_rmse = method_diagnostics["frame_orbit_ratio"][
                    "aggregate_rmse"
                ]
                if iid_rmse > 0:
                    expected_reduction = (
                        iid_rmse - coupled_rmse
                    ) / iid_rmse
                    audit.close(
                        f"{path}.frame_coupled_rmse_reduction_vs_iid_linear",
                        budget_result.get(
                            "frame_coupled_rmse_reduction_vs_iid_linear"
                        ),
                        expected_reduction,
                    )
                else:
                    audit.errors.append(
                        f"{path}: IID-linear RMSE is zero, so its relative "
                        "reduction is undefined"
                    )
                if official_rmse > 0:
                    if has_cc and CC_METHOD in method_diagnostics:
                        cc_rmse = method_diagnostics[CC_METHOD][
                            "aggregate_rmse"
                        ]
                        expected_cc_reduction = (
                            official_rmse - cc_rmse
                        ) / official_rmse
                        audit.close(
                            f"{path}.cc_rmse_reduction_vs_official_ratio",
                            budget_result.get(
                                "cc_rmse_reduction_vs_official_ratio"
                            ),
                            expected_cc_reduction,
                        )
                    expected_reduction = (
                        official_rmse - orbit_rmse
                    ) / official_rmse
                    audit.close(
                        f"{path}.orbit_rmse_reduction_vs_official_ratio",
                        budget_result.get(
                            "orbit_rmse_reduction_vs_official_ratio"
                        ),
                        expected_reduction,
                    )
                else:
                    audit.errors.append(
                        f"{path}: official-ratio RMSE is zero, so its "
                        "relative reduction is undefined"
                    )
            per_budget_diagnostics[key] = {
                "total_calls": budget + boundary_calls,
                "official_missing_stratum_fraction_mean": (
                    missing_fraction
                ),
                "methods": method_diagnostics,
            }

    diagnostics = {
        "status": report.get("status"),
        "dataset": dataset_name,
        "num_players": num_players,
        "repeats": repeats,
        "inner_budgets": budgets,
        "total_call_budgets": expected_totals,
        "boundary_calls": boundary_calls,
        "has_official_basic_cc": has_cc,
        "has_baseline_comparison": baseline_comparison is not None,
        "new_methods": list(new_methods),
        "ground_truth": gt_diagnostics,
        "results_by_inner_budget": per_budget_diagnostics,
    }
    return {
        "passed": not audit.errors,
        "errors": audit.errors,
        "diagnostics": diagnostics,
    }


def _text_output(path: Path, result: Mapping[str, Any]) -> str:
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    outcome = "PASS" if result.get("passed") else "FAIL"
    lines = [f"{outcome} {path}"]
    if diagnostics:
        lines.append(
            "  "
            f"status={diagnostics.get('status')} "
            f"dataset={diagnostics.get('dataset')} "
            f"n={diagnostics.get('num_players')} "
            f"repeats={diagnostics.get('repeats')}"
        )
        lines.append(
            "  total calls="
            + ", ".join(
                f"{int(value):,}"
                for value in diagnostics.get("total_call_budgets", [])
            )
        )
        gt = diagnostics.get("ground_truth", {})
        if isinstance(gt, Mapping) and gt:
            lines.append(
                "  ground truth: "
                f"pairs={int(gt.get('pair_units', 0)):,}, "
                f"physical calls={int(gt.get('physical_total_calls', 0)):,}, "
                f"efficiency gap={float(gt.get('efficiency_gap', math.nan)):.3e}"
            )
    errors = result.get("errors", [])
    if errors:
        lines.append(f"  {len(errors)} error(s):")
        lines.extend(f"    - {error}" for error in errors)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a completed full-train experiment JSON report."
    )
    parser.add_argument("report", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--expected-status", default="complete")
    parser.add_argument("--expected-dataset")
    parser.add_argument("--expected-num-players", type=int)
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--expected-gt-pairs", type=int)
    parser.add_argument("--expected-jobs", type=int)
    parser.add_argument("--expected-inner-budgets", type=int, nargs="+")
    parser.add_argument(
        "--expected-budget-multipliers", type=int, nargs="+"
    )
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument(
        "--efficiency-tolerance", type=float, default=1e-8
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        result = {
            "passed": False,
            "errors": [f"cannot read report: {error}"],
            "diagnostics": {},
        }
    else:
        result = validate_report(
            report,
            expected_status=args.expected_status,
            expected_dataset=args.expected_dataset,
            expected_num_players=args.expected_num_players,
            expected_repeats=args.expected_repeats,
            expected_gt_pairs=args.expected_gt_pairs,
            expected_jobs=args.expected_jobs,
            expected_inner_budgets=args.expected_inner_budgets,
            expected_budget_multipliers=(
                args.expected_budget_multipliers
            ),
            rtol=args.rtol,
            atol=args.atol,
            efficiency_tolerance=args.efficiency_tolerance,
        )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_text_output(args.report, result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()

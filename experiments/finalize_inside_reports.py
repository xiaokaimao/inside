"""Reduce the estimator-ablation report to the formal eight-method report.

The expensive Airport and Voting evaluations were retained in
``*_inside_per_size_ratio_vs_global_linear*.json``.  Those reports contain
three Greedy variants because they were built to answer an ablation question.
The paper-facing comparison has a smaller method contract:

* ``inside_greedy_per_size_ratio`` becomes ``inside_greedy``;
* ``inside_greedy_global`` and ``inside_greedy_per_size_linear`` are removed;
* INSIDE-Orbit and the six established baselines are copied unchanged.

This module performs only a lossless, deeply copied method selection.  It
does not rerun utilities, recompute estimates, or silently transplant summary
statistics.  Before writing, it checks that raw tasks and aggregate summaries
describe the same repeat-level estimates.  The resulting report is then
audited with :mod:`experiments.validate_inside_comparison`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from experiments.plot_inside_comparison import METHOD_ORDER as FORMAL_METHOD_ORDER
from experiments.run_analytic_inside_comparison import METHOD_LABELS
from experiments.run_per_size_ratio_comparison import METHOD_ORDER as SOURCE_METHOD_ORDER
from experiments.validate_inside_comparison import validate_report


SOURCE_GREEDY = "inside_greedy_per_size_ratio"
FORMAL_GREEDY = "inside_greedy"
DROPPED_METHODS = (
    "inside_greedy_global",
    "inside_greedy_per_size_linear",
)
EXPERIMENT_ID = "analytic_inside_baseline_comparison_per_size_ratio"

SOURCE_TO_FORMAL = {
    SOURCE_GREEDY: FORMAL_GREEDY,
    **{
        method: method
        for method in FORMAL_METHOD_ORDER
        if method != FORMAL_GREEDY
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{path} must be an object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (
        int(row["repeat"]),
        str(row["method"]),
        int(row["budget_index"]),
    )


def _source_contract(
    source: Mapping[str, Any],
) -> tuple[Mapping[str, Any], tuple[int, ...], tuple[int, ...], int]:
    _require(source.get("status") == "complete", "source report must be complete")
    configuration = _mapping(source.get("configuration"), path="configuration")
    methods = tuple(configuration.get("methods", ()))
    _require(
        methods == SOURCE_METHOD_ORDER,
        "source report must use the ten-method estimator-ablation contract",
    )
    repeats = int(configuration.get("repeats", -1))
    _require(repeats == 3, "source report must retain three repeats")
    inner = tuple(int(value) for value in configuration.get("inner_utility_call_budgets", ()))
    total = tuple(int(value) for value in configuration.get("total_call_budgets", ()))
    _require(len(inner) == 5 and len(total) == 5, "source report must contain five budgets")
    _require(
        len(inner) == len(total)
        and all(a < b for a, b in zip(inner, inner[1:]))
        and all(a < b for a, b in zip(total, total[1:])),
        "source budgets must be aligned and strictly increasing",
    )
    return configuration, inner, total, repeats


def _validate_source_rows(
    source: Mapping[str, Any],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    raw = source.get("raw_tasks")
    _require(isinstance(raw, list), "source report must retain raw_tasks")
    expected = {
        (repeat, method, budget_index)
        for repeat in range(repeats)
        for method in SOURCE_METHOD_ORDER
        for budget_index in range(len(inner_budgets))
    }
    rows: dict[tuple[int, str, int], Mapping[str, Any]] = {}
    for index, value in enumerate(raw):
        row = _mapping(value, path=f"raw_tasks[{index}]")
        key = _cell_key(row)
        _require(key not in rows, f"duplicate raw task {key}")
        _require(key in expected, f"unexpected raw task {key}")
        budget_index = key[2]
        _require(
            int(row.get("inner_utility_call_budget", -1))
            == inner_budgets[budget_index],
            f"raw task {key} has the wrong inner budget",
        )
        _require(
            int(row.get("target_total_utility_calls", -1))
            == total_budgets[budget_index],
            f"raw task {key} has the wrong total budget",
        )
        estimate = np.asarray(row.get("estimate"), dtype=np.float64)
        _require(
            estimate.ndim == 1 and estimate.size >= 2 and np.all(np.isfinite(estimate)),
            f"raw task {key} has an invalid estimate",
        )
        rows[key] = row
    _require(set(rows) == expected, "source report has incomplete raw tasks")
    return rows


def _validate_source_summaries(
    source: Mapping[str, Any],
    rows: Mapping[tuple[int, str, int], Mapping[str, Any]],
    inner_budgets: tuple[int, ...],
    total_budgets: tuple[int, ...],
    repeats: int,
) -> Mapping[str, Any]:
    results = _mapping(
        source.get("results_by_inner_budget"), path="results_by_inner_budget"
    )
    _require(
        set(results) == {str(value) for value in inner_budgets},
        "source result keys disagree with configured budgets",
    )
    for budget_index, (inner, total) in enumerate(
        zip(inner_budgets, total_budgets, strict=True)
    ):
        result = _mapping(results[str(inner)], path=f"results[{inner}]")
        _require(int(result.get("inner_utility_calls", -1)) == inner, "result inner budget differs")
        _require(
            int(result.get("total_utility_calls_per_estimate", -1)) == total,
            "result total budget differs",
        )
        methods = _mapping(result.get("methods"), path=f"results[{inner}].methods")
        _require(set(methods) == set(SOURCE_METHOD_ORDER), "source result method set differs")
        for method in SOURCE_METHOD_ORDER:
            summary = _mapping(methods[method], path=f"results[{inner}].{method}")
            raw_estimates = np.asarray(
                [rows[(repeat, method, budget_index)]["estimate"] for repeat in range(repeats)],
                dtype=np.float64,
            )
            summary_estimates = np.asarray(summary.get("estimates"), dtype=np.float64)
            _require(
                np.array_equal(summary_estimates, raw_estimates),
                f"{method} budget {inner} aggregate estimates differ from raw tasks",
            )
            raw_calls = [
                int(rows[(repeat, method, budget_index)]["actual_utility_calls"])
                for repeat in range(repeats)
            ]
            _require(
                list(summary.get("actual_utility_calls_by_repeat", ())) == raw_calls,
                f"{method} budget {inner} aggregate calls differ from raw tasks",
            )
    return results


def _formal_configuration(
    source_configuration: Mapping[str, Any],
    *,
    source_path: Path,
    source_hash: str,
    source_experiment: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    configuration = copy.deepcopy(dict(source_configuration))
    prior_source = copy.deepcopy(configuration.get("source_report"))
    configuration["methods"] = list(FORMAL_METHOD_ORDER)
    configuration["method_labels"] = copy.deepcopy(METHOD_LABELS)
    configuration.pop("comparison", None)
    prior_greedy = _mapping(
        configuration.pop("inside_greedy", {}), path="configuration.inside_greedy"
    )
    configuration["inside"] = {
        "greedy": {
            "candidate_pool": int(prior_greedy.get("candidate_pool", 4)),
            "mean_balance_mode": str(prior_greedy.get("mean_balance_mode", "normalized")),
            "mean_balance_lambda0": float(prior_greedy.get("mean_balance_lambda0", 1.0)),
            "first_moment_scope": "per_size",
            "second_moment_scope": "per_size",
            "estimator": "ofa_conditional_mean_ratio_missing_raise",
            "ratio_missing_policy": "raise",
            "random_relabeling": prior_greedy.get("shared_relabel_rule"),
        },
        "orbit": {
            "first_moment_scope": "per_size_exact_by_complete_orbits",
            "second_moment_scope": "per_size",
            "estimator": "ofa_conditional_mean_ratio_strict_balanced",
        },
    }
    # The common validator accepts this canonical name.  Retaining the older
    # key as an alias would leave two apparent sources of truth.
    boundary_calls = configuration.pop("boundary_utility_calls_for_greedy_methods", None)
    if boundary_calls is not None:
        configuration["boundary_utility_calls_for_ofa_methods"] = int(boundary_calls)
    configuration.pop("parallelism", None)
    configuration.pop("checkpoint_granularity", None)

    provenance = {
        "path": str(source_path.resolve()),
        "sha256": source_hash,
        "source_experiment": source_experiment,
        "transformation": "lossless method-set reduction; no utility evaluation or estimate recomputation",
        "retained_source_methods": list(SOURCE_TO_FORMAL),
        "renamed_method": {"source": SOURCE_GREEDY, "destination": FORMAL_GREEDY},
        "dropped_methods": list(DROPPED_METHODS),
        "parent_source_report": prior_source,
    }
    configuration["source_report"] = copy.deepcopy(provenance)
    configuration["report_construction"] = "deterministic deep-copy reduction"
    configuration["raw_task_schema"] = (
        "canonical grouped repeat-by-method task with five budget rows"
    )
    return configuration, provenance


def build_formal_report(
    source: Mapping[str, Any], *, source_path: Path, source_hash: str | None = None
) -> dict[str, Any]:
    """Build and independently validate the formal eight-method report."""
    configuration, inner, total, repeats = _source_contract(source)
    source_rows = _validate_source_rows(source, inner, total, repeats)
    source_results = _validate_source_summaries(
        source, source_rows, inner, total, repeats
    )
    digest = source_hash or _sha256(source_path)
    formal_configuration, provenance = _formal_configuration(
        configuration,
        source_path=source_path,
        source_hash=digest,
        source_experiment=source.get("experiment"),
    )

    # The ablation runner checkpoints one repeat-method-budget cell at a
    # time.  The canonical runner checkpoints one repeat-method task with all
    # five budget rows.  Re-group here so this formal report can itself be
    # passed to ``run_analytic_inside_comparison --reuse-non-greedy-from``.
    raw_tasks: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for method in FORMAL_METHOD_ORDER:
            source_method = SOURCE_GREEDY if method == FORMAL_GREEDY else method
            task_rows: list[dict[str, Any]] = []
            for budget_index in range(len(inner)):
                row = copy.deepcopy(
                    dict(source_rows[(repeat, source_method, budget_index)])
                )
                row["method"] = method
                task_rows.append(row)
            raw_tasks.append(
                {"repeat": repeat, "method": method, "rows": task_rows}
            )

    formal_results: dict[str, Any] = {}
    for inner_calls in inner:
        source_result = _mapping(source_results[str(inner_calls)], path="source result")
        result = {
            "inner_utility_calls": int(source_result["inner_utility_calls"]),
            "total_utility_calls_per_estimate": int(
                source_result["total_utility_calls_per_estimate"]
            ),
            "methods": {},
        }
        source_methods = _mapping(source_result["methods"], path="source methods")
        for method in FORMAL_METHOD_ORDER:
            source_method = SOURCE_GREEDY if method == FORMAL_GREEDY else method
            summary = copy.deepcopy(dict(source_methods[source_method]))
            summary["label"] = METHOD_LABELS[method]
            result["methods"][method] = summary
        formal_results[str(inner_calls)] = result

    report = {
        key: copy.deepcopy(value)
        for key, value in source.items()
        if key
        not in {
            "configuration",
            "experiment",
            "experiment_wall_seconds",
            "raw_tasks",
            "results_by_inner_budget",
            "validation",
        }
    }
    report.update(
        {
            "status": "complete",
            "experiment": EXPERIMENT_ID,
            "configuration": formal_configuration,
            "results_by_inner_budget": formal_results,
            "raw_tasks": raw_tasks,
            "provenance": copy.deepcopy(provenance),
        }
    )

    ratio_coverages = [
        _mapping(row.get("diagnostics"), path="INSIDE-Greedy diagnostics").get("coverage")
        for task in raw_tasks
        if task["method"] == FORMAL_GREEDY
        for row in task["rows"]
    ]
    _require(len(ratio_coverages) == repeats * len(inner), "wrong INSIDE-Greedy cell count")
    _require(
        all(
            isinstance(coverage, Mapping)
            and coverage.get("all_player_size_strata_covered") is True
            for coverage in ratio_coverages
        ),
        "formal INSIDE-Greedy has an uncovered ratio stratum",
    )
    structural_validation = {
        "passed": True,
        "source_sha256": digest,
        "source_method_repeat_budget_cells": repeats * len(SOURCE_METHOD_ORDER) * len(inner),
        "formal_grouped_repeat_method_tasks": repeats * len(FORMAL_METHOD_ORDER),
        "formal_method_repeat_budget_cells": repeats * len(FORMAL_METHOD_ORDER) * len(inner),
        "dropped_method_repeat_budget_cells": repeats * len(DROPPED_METHODS) * len(inner),
        "renamed_inside_greedy_cells": repeats * len(inner),
        "raw_tasks_and_aggregate_estimates_match": True,
        "inside_greedy_ratio_all_player_size_strata_covered": True,
        "estimates_recomputed": False,
    }
    report["validation"] = {
        "reduction": structural_validation,
        "common_eight_method_audit": validate_report(report),
    }
    # The embedded audit must not alter the result accepted by the same
    # validator when the serialized report is consumed later.
    final_audit = validate_report(report)
    _require(final_audit["status"] == "ready_to_share", "formal report audit failed")
    return report


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def finalize_report(source_path: Path, output_path: Path) -> dict[str, Any]:
    source_path = source_path.resolve()
    source = json.loads(source_path.read_text(encoding="utf-8"))
    report = build_formal_report(source, source_path=source_path)
    _write_json_atomic(output_path, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--validation-output",
        type=Path,
        help="optional standalone copy of the common eight-method audit",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = finalize_report(args.source, args.output)
    audit = report["validation"]["common_eight_method_audit"]
    if args.validation_output is not None:
        _write_json_atomic(args.validation_output, audit)
    print(
        f"{audit['status']}: saved {args.output} "
        f"({len(report['raw_tasks'])} grouped repeat-method tasks)"
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.finalize_inside_reports import (
    DROPPED_METHODS,
    EXPERIMENT_ID,
    FORMAL_GREEDY,
    FORMAL_METHOD_ORDER,
    SOURCE_GREEDY,
    SOURCE_METHOD_ORDER,
    build_formal_report,
    finalize_report,
)
from experiments.run_per_size_ratio_comparison import METHOD_LABELS as SOURCE_LABELS
from experiments.run_analytic_inside_comparison import (
    _exact_truth,
    _load_reused_non_greedy_tasks,
)
from experiments.validate_inside_comparison import validate_report
from tests.test_inside_plot_validate import _airport_report


def _ten_method_source() -> dict:
    formal = _airport_report()
    source = copy.deepcopy(formal)
    source["experiment"] = "analytic_inside_per_size_ratio_vs_global_linear"
    configuration = source["configuration"]
    configuration["methods"] = list(SOURCE_METHOD_ORDER)
    configuration["budget_multipliers"] = [1, 2, 3, 4, 5]
    configuration["base_seed"] = 1234
    configuration["num_tasks_per_serial_baseline"] = 1
    configuration["method_labels"] = copy.deepcopy(SOURCE_LABELS)
    configuration["inside_greedy"] = {
        "candidate_pool": 4,
        "mean_balance_mode": "normalized",
        "mean_balance_lambda0": 1.0,
        "ratio_estimator": "official OFA conditional-mean ratio",
        "ratio_missing_policy": "raise",
        "shared_relabel_rule": "deterministic test relabel",
    }
    configuration.pop("inside", None)
    configuration["source_report"] = {
        "path": "parent.json",
        "sha256": "parent-sha",
    }
    configuration["comparison"] = {"primary_a": SOURCE_GREEDY}

    for result in source["results_by_inner_budget"].values():
        original = result["methods"]
        greedy = original["inside_greedy"]
        expanded = {}
        for method in SOURCE_METHOD_ORDER:
            if method in {
                "inside_greedy_global",
                "inside_greedy_per_size_linear",
                SOURCE_GREEDY,
            }:
                summary = copy.deepcopy(greedy)
                summary["label"] = SOURCE_LABELS[method]
                if method == SOURCE_GREEDY:
                    for diagnostic in summary["diagnostics_by_repeat"]:
                        diagnostic.update(
                            {
                                "design_method": "frame_coupled_per_size",
                                "second_moment_scope": "per_size",
                                "estimator": "ofa_conditional_mean_ratio_missing_raise",
                                "official_ratio_missing_policy": "raise",
                                "coverage": {
                                    "all_player_size_strata_covered": True,
                                    "minimum_inclusion_count": 1,
                                    "minimum_exclusion_count": 1,
                                },
                                "design_diagnostics": {
                                    "second_moment_scope": "per_size"
                                },
                            }
                        )
                expanded[method] = summary
            else:
                summary = copy.deepcopy(original[method])
                if method == "inside_orbit":
                    for diagnostic in summary["diagnostics_by_repeat"]:
                        diagnostic.update(
                            {
                                "design_method": "cyclic_orbit_frame",
                                "estimator": (
                                    "ofa_conditional_mean_ratio_strict_balanced"
                                ),
                            }
                        )
                expanded[method] = summary
        result["methods"] = expanded

    raw_tasks = []
    inner_budgets = source["configuration"]["inner_utility_call_budgets"]
    for budget_index, inner in enumerate(inner_budgets):
        result = source["results_by_inner_budget"][str(inner)]
        total = result["total_utility_calls_per_estimate"]
        for method in SOURCE_METHOD_ORDER:
            summary = result["methods"][method]
            for repeat in range(3):
                raw_tasks.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "budget_index": budget_index,
                        "seed": 1000 + repeat * 100 + budget_index,
                        "inner_utility_call_budget": inner,
                        "target_total_utility_calls": total,
                        "actual_utility_calls": summary[
                            "actual_utility_calls_by_repeat"
                        ][repeat],
                        "estimate": copy.deepcopy(summary["estimates"][repeat]),
                        "elapsed_seconds": 0.0,
                        "diagnostics": copy.deepcopy(
                            summary["diagnostics_by_repeat"][repeat]
                        ),
                    }
                )
    source["raw_tasks"] = raw_tasks
    source["validation"] = {"passed": True, "source_only": True}
    return source


class FinalizeInsideReportsTests(unittest.TestCase):
    def test_reduces_to_formal_contract_without_mutating_source(self) -> None:
        source = _ten_method_source()
        original = copy.deepcopy(source)
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            report = build_formal_report(source, source_path=source_path)

        self.assertEqual(source, original)
        self.assertEqual(report["experiment"], EXPERIMENT_ID)
        self.assertEqual(
            report["configuration"]["methods"], list(FORMAL_METHOD_ORDER)
        )
        self.assertEqual(
            report["configuration"]["method_labels"][FORMAL_GREEDY],
            "INSIDE-Greedy",
        )
        self.assertEqual(
            report["configuration"]["method_labels"]["inside_orbit"],
            "INSIDE-Orbit",
        )
        greedy = report["configuration"]["inside"]["greedy"]
        self.assertEqual(greedy["second_moment_scope"], "per_size")
        self.assertEqual(
            greedy["estimator"], "ofa_conditional_mean_ratio_missing_raise"
        )
        self.assertEqual(len(report["raw_tasks"]), 3 * 8)
        self.assertTrue(
            all(len(task["rows"]) == 5 for task in report["raw_tasks"])
        )
        self.assertNotIn(
            DROPPED_METHODS[0], {task["method"] for task in report["raw_tasks"]}
        )
        self.assertNotIn(
            DROPPED_METHODS[1], {task["method"] for task in report["raw_tasks"]}
        )
        source_ratio = next(
            row for row in source["raw_tasks"] if row["method"] == SOURCE_GREEDY
        )
        formal_ratio_task = next(
            task
            for task in report["raw_tasks"]
            if task["method"] == FORMAL_GREEDY and task["repeat"] == 0
        )
        formal_ratio = formal_ratio_task["rows"][0]
        self.assertEqual(formal_ratio["estimate"], source_ratio["estimate"])
        self.assertEqual(formal_ratio["diagnostics"], source_ratio["diagnostics"])
        self.assertIsNot(formal_ratio["estimate"], source_ratio["estimate"])
        for result in report["results_by_inner_budget"].values():
            self.assertEqual(tuple(result["methods"]), FORMAL_METHOD_ORDER)
            self.assertEqual(
                result["methods"][FORMAL_GREEDY]["label"], "INSIDE-Greedy"
            )
            self.assertEqual(
                result["methods"]["inside_orbit"]["label"], "INSIDE-Orbit"
            )
        reduction = report["validation"]["reduction"]
        self.assertEqual(reduction["formal_grouped_repeat_method_tasks"], 24)
        self.assertEqual(reduction["formal_method_repeat_budget_cells"], 120)
        self.assertEqual(reduction["dropped_method_repeat_budget_cells"], 30)
        self.assertEqual(reduction["renamed_inside_greedy_cells"], 15)
        self.assertFalse(reduction["estimates_recomputed"])
        self.assertEqual(
            validate_report(report)["status"], "ready_to_share"
        )

    def test_finalize_writes_a_report_that_the_common_validator_accepts(self) -> None:
        source = _ten_method_source()
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "formal.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            finalize_report(source_path, output_path)
            serialized = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(validate_report(serialized)["status"], "ready_to_share")

    def test_formal_report_is_reusable_by_canonical_runner(self) -> None:
        source = _ten_method_source()
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "formal.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            report = finalize_report(source_path, output_path)
            configuration = report["configuration"]
            truth, _ = _exact_truth("airport")
            tasks, _ = _load_reused_non_greedy_tasks(
                output_path,
                dataset="airport",
                budget_multipliers=tuple(configuration["budget_multipliers"]),
                inner_budgets=tuple(
                    configuration["inner_utility_call_budgets"]
                ),
                total_budgets=tuple(configuration["total_call_budgets"]),
                repeats=configuration["repeats"],
                base_seed=configuration["base_seed"],
                num_tasks=configuration["num_tasks_per_serial_baseline"],
                truth=truth,
            )
        self.assertEqual(len(tasks), 3 * 7)
        self.assertTrue(all(len(task["rows"]) == 5 for task in tasks))
        self.assertNotIn(
            FORMAL_GREEDY, {task["method"] for task in tasks}
        )

    def test_rejects_duplicate_raw_cell(self) -> None:
        source = _ten_method_source()
        source["raw_tasks"].append(copy.deepcopy(source["raw_tasks"][0]))
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate raw task"):
                build_formal_report(source, source_path=source_path)

    def test_rejects_aggregate_that_differs_from_raw_tasks(self) -> None:
        source = _ten_method_source()
        first = source["results_by_inner_budget"]["100"]
        first["methods"][SOURCE_GREEDY]["estimates"][0][0] += 1.0
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differ from raw tasks"):
                build_formal_report(source, source_path=source_path)


if __name__ == "__main__":
    unittest.main()

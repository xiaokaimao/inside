from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.add_wine_local_baselines import (
    DEFAULT_NEW_METHODS,
    SOURCE_METHODS,
    _compact_json,
    _new_comparison_report,
)
from experiments.replace_wine_linear_gels_with_official import (
    migrate_report as migrate_gels_report,
)
from experiments.replace_wine_permutation_with_tmc import (
    LEGACY_TARGET_METHODS,
    migrate_report as migrate_tmc_report,
)


def _source_report() -> dict:
    n = 142
    repeats = 10
    boundary_calls = 2 * n + 2
    budgets = [n * scale for scale in (500, 1000, 2000, 5000, 10000)]
    truth = np.linspace(-0.01, 0.01, n)
    truth -= truth.mean()
    truth_se = np.full(n, 1e-5)
    base_error = np.linspace(-1e-3, 1e-3, n)
    base_error -= base_error.mean()
    results = {}
    for budget_index, budget in enumerate(budgets):
        methods = {}
        for method_index, method in enumerate(SOURCE_METHODS):
            estimates = np.asarray(
                [
                    truth
                    + base_error
                    * (method_index + 1)
                    * (repeat + 1)
                    / (budget_index + 1)
                    for repeat in range(repeats)
                ]
            )
            summary = {"estimates": estimates.tolist()}
            if method == "official_cc_basic":
                total = budget + boundary_calls
                missing = [repeat % 3 for repeat in range(repeats)]
                fractions = [value / n**2 for value in missing]
                summary.update(
                    {
                        "utility_calls_per_estimate": total,
                        "complementary_contributions_per_estimate": total // 2,
                        "missing_stratum_fractions": fractions,
                        "cc_diagnostics_by_repeat": [
                            {
                                "num_pairs": total // 2,
                                "utility_evaluations": total,
                                "missing_strata": missing[repeat],
                                "minimum_positive_count": 1,
                                "num_tasks": 128,
                            }
                            for repeat in range(repeats)
                        ],
                    }
                )
            methods[method] = summary
        results[str(budget)] = {
            "total_utility_calls_per_estimate": budget + boundary_calls,
            "methods": methods,
        }
    return {
        "status": "complete",
        "configuration": {
            "num_players": n,
            "repeats": repeats,
            "gt_pairs": 800_000,
            "inner_budgets": budgets,
            "total_call_budgets": [
                budget + boundary_calls for budget in budgets
            ],
            "cc_baseline": {
                "additional_physical_utility_calls": 12345,
            },
        },
        "dataset": {"dataset": "wine"},
        "boundary": {
            "utility_calls": boundary_calls,
            "efficiency_target": 0.0,
        },
        "ground_truth": {
            "values": truth.tolist(),
            "standard_errors": truth_se.tolist(),
        },
        "results_by_inner_budget": results,
    }


class WineLocalBaselineRunnerTests(unittest.TestCase):
    def test_default_method_set_contains_every_fair_local_port(self) -> None:
        self.assertEqual(
            set(DEFAULT_NEW_METHODS),
            {
                "gels_shapley",
                "kernel_shap_sampled",
                "group_testing",
                "diff",
                "s_diff",
                "tmc_shapley",
                "stratified_marginal_mc",
            },
        )

    def test_three_repeat_initialization_is_non_destructive(self) -> None:
        source = _source_report()
        original = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            source_path.write_text(
                json.dumps(source, sort_keys=True), encoding="utf-8"
            )
            report = _new_comparison_report(
                source,
                source_path=source_path,
                repeats=3,
                methods=DEFAULT_NEW_METHODS,
                seed=123,
                jobs=128,
                chunksize=512,
                num_tasks=128,
                start_method="spawn",
            )

        self.assertEqual(source, original)
        self.assertEqual(report["configuration"]["repeats"], 3)
        metadata = report["configuration"]["baseline_comparison"]
        self.assertEqual(metadata["source_repeat_indices"], [0, 1, 2])
        self.assertEqual(metadata["new_methods"], list(DEFAULT_NEW_METHODS))
        self.assertEqual(
            report["configuration"]["cc_baseline"][
                "additional_physical_utility_calls"
            ],
            0,
        )
        self.assertEqual(
            report["configuration"]["cc_baseline"][
                "source_report_additional_physical_utility_calls"
            ],
            12345,
        )
        for row in report["results_by_inner_budget"].values():
            self.assertEqual(set(row["methods"]), set(SOURCE_METHODS))
            for summary in row["methods"].values():
                self.assertEqual(
                    np.asarray(summary["estimates"]).shape,
                    (3, 142),
                )

    def test_dense_diagnostics_are_compacted_without_zero_bias(self) -> None:
        compact = _compact_json(np.arange(1, 302, dtype=np.int64))
        self.assertTrue(compact["omitted_dense_array"])
        self.assertEqual(compact["shape"], [301])
        self.assertEqual(compact["minimum"], 1.0)
        self.assertEqual(compact["maximum"], 301.0)
        self.assertEqual(compact["sum"], 301 * 302 / 2)

    def test_full_permutation_checkpoint_migration_is_non_destructive(self) -> None:
        old_new_methods = [
            "permutation_mc_full" if method == "tmc_shapley" else method
            for method in LEGACY_TARGET_METHODS
        ]
        report = {
            "status": "complete",
            "configuration": {
                "baseline_comparison": {
                    "new_methods": old_new_methods,
                    "method_labels": {},
                }
            },
            "results_by_inner_budget": {
                "100": {
                    "methods": {
                        method: {"sentinel": method}
                        for method in (*SOURCE_METHODS, *old_new_methods)
                    }
                }
            },
        }
        original = copy.deepcopy(report)
        migrated = migrate_tmc_report(
            report, source_path=Path("old.json")
        )

        self.assertEqual(report, original)
        self.assertEqual(migrated["status"], "baseline_augmentation_running")
        metadata = migrated["configuration"]["baseline_comparison"]
        self.assertEqual(
            metadata["new_methods"], list(LEGACY_TARGET_METHODS)
        )
        methods = migrated["results_by_inner_budget"]["100"]["methods"]
        self.assertNotIn("permutation_mc_full", methods)
        self.assertNotIn("tmc_shapley", methods)
        for method in old_new_methods:
            if method != "permutation_mc_full":
                self.assertEqual(methods[method], {"sentinel": method})

    def test_linear_gels_migration_preserves_all_other_results(self) -> None:
        methods_before = {
            method: {"sentinel": method}
            for method in (*SOURCE_METHODS, *LEGACY_TARGET_METHODS)
        }
        report = {
            "status": "complete",
            "configuration": {
                "baseline_comparison": {
                    "new_methods": list(LEGACY_TARGET_METHODS),
                    "method_labels": {},
                    "baseline_replacement": {"sentinel": "tmc"},
                }
            },
            "results_by_inner_budget": {
                "100": {"methods": methods_before}
            },
        }
        original = copy.deepcopy(report)
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "old.json"
            source_path.write_text(
                json.dumps(report, sort_keys=True), encoding="utf-8"
            )
            migrated = migrate_gels_report(
                report, source_path=source_path
            )

        self.assertEqual(report, original)
        self.assertEqual(migrated["status"], "baseline_augmentation_running")
        metadata = migrated["configuration"]["baseline_comparison"]
        self.assertEqual(metadata["new_methods"], list(DEFAULT_NEW_METHODS))
        self.assertEqual(metadata["baseline_replacement"], {"sentinel": "tmc"})
        replacement = metadata["gels_shapley_replacement"]
        self.assertEqual(replacement["removed_method"], "gels_linear")
        self.assertEqual(replacement["replacement_method"], "gels_shapley")
        migrated_methods = migrated["results_by_inner_budget"]["100"][
            "methods"
        ]
        self.assertNotIn("gels_linear", migrated_methods)
        self.assertNotIn("gels_shapley", migrated_methods)
        for method, summary in methods_before.items():
            if method != "gels_linear":
                self.assertEqual(migrated_methods[method], summary)


if __name__ == "__main__":
    unittest.main()

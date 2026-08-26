from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.plot_second_moment_scope_comparison import plot_results
from experiments.validate_second_moment_scope_comparison import (
    METHOD_ORDER,
    validate_report,
)
from tests.test_inside_plot_validate import _airport_report


def _scope_report() -> dict:
    report = _airport_report()
    report["experiment"] = "analytic_inside_second_moment_scope_ablation"
    configuration = report["configuration"]
    legacy_inside = configuration.pop("inside")
    configuration["inside_greedy"] = {
        "candidate_pool": legacy_inside["candidate_pool"],
        "mean_balance_mode": "normalized",
        "mean_balance_lambda0": 1.0,
        "estimator": "coupled linear",
        "global_second_moment": "one radial-weighted operator",
        "per_size_second_moment": "one unweighted operator per size",
        "first_moment": "per-size for both variants",
    }
    configuration["methods"] = list(METHOD_ORDER)
    configuration["method_labels"] = {
        method: method.replace("_", " ") for method in METHOD_ORDER
    }

    for budget_index, row in enumerate(
        report["results_by_inner_budget"].values()
    ):
        legacy = row["methods"].pop("inside_greedy")
        global_summary = deepcopy(legacy)
        per_size_summary = deepcopy(legacy)
        for method, summary, scope, design_method in (
            (
                "inside_greedy_global",
                global_summary,
                "global_weighted",
                "frame_coupled",
            ),
            (
                "inside_greedy_per_size",
                per_size_summary,
                "per_size",
                "frame_coupled_per_size",
            ),
        ):
            for repeat, diagnostic in enumerate(
                summary["diagnostics_by_repeat"]
            ):
                schedule_hash = f"{budget_index * 3 + repeat + 1:064x}"
                diagnostic.update(
                    {
                        "design_method": design_method,
                        "second_moment_scope": scope,
                        "estimator": "coupled_linear",
                        "size_schedule_sha256": schedule_hash,
                        "relabel_permutation_sha256": (
                            f"{1000 + budget_index * 3 + repeat:064x}"
                        ),
                        "design_diagnostics": {
                            "second_moment_scope": scope,
                        },
                    }
                )
            summary["method"] = method
        old_methods = row["methods"]
        row["methods"] = {
            "inside_greedy_global": global_summary,
            "inside_greedy_per_size": per_size_summary,
            **old_methods,
        }
    return report


class SecondMomentScopePlotValidateTests(unittest.TestCase):
    def test_validator_audits_all_nine_methods_and_schedule_pairs(self) -> None:
        validation = validate_report(_scope_report())
        self.assertEqual(validation["status"], "ready_to_share")
        self.assertEqual(validation["dataset"], "airport")
        self.assertEqual(validation["validated_shape"]["methods"], 9)
        self.assertEqual(
            validation["validated_shape"]["per_repeat_estimates_checked"],
            135,
        )
        self.assertEqual(
            validation["size_schedule_audit"][
                "matching_sha256_pairs_checked"
            ],
            15,
        )
        self.assertEqual(
            validation["relabel_permutation_audit"][
                "matching_sha256_pairs_checked"
            ],
            15,
        )

    def test_validator_rejects_scope_identity_tampering(self) -> None:
        report = _scope_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size"][
            "diagnostics_by_repeat"
        ][0]["second_moment_scope"] = "global_weighted"
        with self.assertRaisesRegex(ValueError, "identifies 'global_weighted'"):
            validate_report(report)

    def test_validator_rejects_different_size_schedules(self) -> None:
        report = _scope_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size"][
            "diagnostics_by_repeat"
        ][1]["size_schedule_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different size schedules"):
            validate_report(report)

    def test_validator_rejects_different_relabel_permutations(self) -> None:
        report = _scope_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size"][
            "diagnostics_by_repeat"
        ][1]["relabel_permutation_sha256"] = "f" * 64
        with self.assertRaisesRegex(
            ValueError, "different relabel permutations"
        ):
            validate_report(report)

    def test_validator_recomputes_scope_rmse(self) -> None:
        report = _scope_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size"]["aggregate_rmse"] *= 2.0
        with self.assertRaisesRegex(ValueError, "aggregate_rmse differs"):
            validate_report(report)

    def test_plot_is_one_near_square_png_and_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "airport_scope.png"
            png, pdf = plot_results(_scope_report(), output)
            self.assertEqual(png, output)
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertEqual(pdf, output.with_suffix(".pdf"))
            self.assertGreater(pdf.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()

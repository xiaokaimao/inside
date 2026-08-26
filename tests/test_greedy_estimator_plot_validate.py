from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.plot_greedy_estimator_comparison import plot_results
from experiments.validate_greedy_estimator_comparison import (
    METHOD_ORDER,
    validate_report,
)
from tests.test_second_moment_scope_plot_validate import _scope_report


def _comparison_report() -> dict:
    report = _scope_report()
    report["experiment"] = "analytic_inside_per_size_ratio_vs_global_linear"
    report["configuration"]["methods"] = list(METHOD_ORDER)
    report["configuration"]["method_labels"] = {
        method: method.replace("_", " ") for method in METHOD_ORDER
    }
    for row in report["results_by_inner_budget"].values():
        old_methods = row["methods"]
        per_size_linear = old_methods.pop("inside_greedy_per_size")
        per_size_ratio = deepcopy(per_size_linear)
        per_size_linear["method"] = "inside_greedy_per_size_linear"
        per_size_ratio["method"] = "inside_greedy_per_size_ratio"
        for diagnostic in per_size_ratio["diagnostics_by_repeat"]:
            diagnostic["estimator"] = "ofa_conditional_mean_ratio_missing_raise"
            diagnostic["coverage"] = {
                "minimum_inclusion_count": 2,
                "minimum_exclusion_count": 3,
                "maximum_absolute_inclusion_count_deviation": 1.25,
                "maximum_relative_inclusion_count_deviation": 0.2,
                "all_player_size_strata_covered": True,
                "missing_inclusion_strata": 0,
                "missing_exclusion_strata": 0,
            }
            diagnostic["linear_reconstruction_max_abs_difference"] = 0.0
        old_methods["inside_greedy_per_size_linear"] = per_size_linear
        old_methods["inside_greedy_per_size_ratio"] = per_size_ratio
        row["methods"] = {method: old_methods[method] for method in METHOD_ORDER}
    return report


class GreedyEstimatorPlotValidateTests(unittest.TestCase):
    def test_validator_audits_ten_methods_and_ratio_controls(self) -> None:
        validation = validate_report(_comparison_report())
        self.assertEqual(validation["status"], "ready_to_share")
        self.assertEqual(validation["dataset"], "airport")
        self.assertEqual(validation["validated_shape"]["methods"], 10)
        self.assertEqual(
            validation["validated_shape"]["per_repeat_estimates_checked"], 150
        )
        self.assertEqual(validation["ratio_coverage_audit"]["cells_checked"], 15)
        self.assertEqual(
            validation["controlled_randomization_audit"][
                "matching_size_schedule_sha256_groups_checked"
            ],
            15,
        )

    def test_validator_rejects_ratio_coverage_failure(self) -> None:
        report = _comparison_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size_ratio"]["diagnostics_by_repeat"][0][
            "coverage"
        ]["minimum_exclusion_count"] = 0
        with self.assertRaisesRegex(ValueError, "minimum_exclusion_count must be positive"):
            validate_report(report)

    def test_validator_rejects_ratio_schedule_mismatch(self) -> None:
        report = _comparison_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size_ratio"]["diagnostics_by_repeat"][1][
            "size_schedule_sha256"
        ] = "f" * 64
        with self.assertRaisesRegex(ValueError, "different size schedules"):
            validate_report(report)

    def test_validator_rejects_linear_reconstruction_failure(self) -> None:
        report = _comparison_report()
        row = report["results_by_inner_budget"]["100"]
        row["methods"]["inside_greedy_per_size_ratio"]["diagnostics_by_repeat"][2][
            "linear_reconstruction_max_abs_difference"
        ] = 1e-6
        with self.assertRaisesRegex(ValueError, "linear reconstruction differs"):
            validate_report(report)

    def test_plot_is_one_near_square_png_and_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "airport_ratio_comparison.png"
            png, pdf = plot_results(_comparison_report(), output)
            self.assertEqual(png, output)
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertEqual(pdf, output.with_suffix(".pdf"))
            self.assertGreater(pdf.stat().st_size, 1_000)


if __name__ == "__main__":
    unittest.main()

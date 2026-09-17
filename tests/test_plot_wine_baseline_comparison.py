import copy
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.plot_wine_baseline_comparison import (
    EXPECTED_METHODS,
    LINEAR_PAIR_ORDER,
    MAIN_PLOT_ORDER,
    METHOD_STYLES,
    NEW_METHODS,
    PLOT_ORDER,
    REUSED_METHODS,
    _default_output,
    _linear_output,
    _main_call_limits,
    _plotted_methods,
    plot_results,
    series,
    validate_report,
)


TOTAL_CALLS = [71_286, 142_286, 284_286, 710_286, 1_420_286]


def _synthetic_report() -> dict:
    report = {
        "status": "complete",
        "configuration": {
            "repeats": 3,
            "num_players": 142,
            "n_test": 36,
            "total_call_budgets": TOTAL_CALLS,
            "baseline_comparison": {
                "new_methods": sorted(NEW_METHODS),
            },
        },
        "dataset": {"dataset": "wine"},
        "results_by_inner_budget": {},
    }
    for budget_index, total_calls in enumerate(TOTAL_CALLS):
        methods = {}
        for method_index, method in enumerate(sorted(EXPECTED_METHODS)):
            rmse = 0.12 / (budget_index + 1) + 0.001 * method_index
            summary = {
                "aggregate_rmse": rmse,
                "aggregate_rmse_bootstrap_95": [
                    rmse * 0.88,
                    rmse * 1.14,
                ],
                # Reused methods must ignore this value.
                "mean_actual_utility_calls": total_calls - 900,
            }
            if method in NEW_METHODS:
                summary["mean_actual_utility_calls"] = (
                    total_calls - 100 - method_index
                )
            methods[method] = summary
        report["results_by_inner_budget"][str(total_calls - 286)] = {
            "total_utility_calls_per_estimate": total_calls,
            "methods": methods,
        }
    return report


class WineBaselineComparisonPlotTests(unittest.TestCase):
    def test_main_call_limits_crop_only_tmc_early_tail(self) -> None:
        plotted_calls = [26_132.0, 52_572.0, 104_679.0, *TOTAL_CALLS]
        left, right = _main_call_limits(TOTAL_CALLS, plotted_calls)

        self.assertEqual(left, TOTAL_CALLS[0] * 0.80)
        self.assertGreater(left, plotted_calls[1])
        self.assertGreater(right, max(plotted_calls))

    def test_main_and_linear_figures_partition_methods_exactly(self) -> None:
        plotted = _plotted_methods()
        self.assertEqual(plotted, PLOT_ORDER)
        self.assertEqual(len(plotted), 12)
        self.assertEqual(len(set(plotted)), 12)
        self.assertEqual(set(plotted), EXPECTED_METHODS)
        self.assertEqual(
            set(MAIN_PLOT_ORDER).intersection(LINEAR_PAIR_ORDER),
            set(),
        )
        self.assertEqual(
            set(MAIN_PLOT_ORDER).union(LINEAR_PAIR_ORDER),
            EXPECTED_METHODS,
        )
        self.assertEqual(
            LINEAR_PAIR_ORDER,
            ("iid_linear_ofa", "frame_coupled_linear"),
        )
        self.assertIn("tmc_shapley", NEW_METHODS)
        self.assertIn("tmc_shapley", MAIN_PLOT_ORDER)
        self.assertIn("gels_shapley", NEW_METHODS)
        self.assertIn("gels_shapley", MAIN_PLOT_ORDER)
        self.assertNotIn("gels_linear", EXPECTED_METHODS)
        self.assertNotIn("permutation_mc_full", EXPECTED_METHODS)
        self.assertEqual(METHOD_STYLES["tmc_shapley"]["label"], "TMC-Shapley")
        self.assertEqual(
            METHOD_STYLES["gels_shapley"]["label"], "GELS-Shapley"
        )

    def test_series_uses_actual_calls_only_for_new_methods(self) -> None:
        report = _synthetic_report()
        reused = next(iter(REUSED_METHODS))
        new = next(iter(NEW_METHODS))
        reused_calls, _, _, _ = series(report, reused)
        new_calls, _, lower, upper = series(report, new)

        self.assertEqual(reused_calls.tolist(), TOTAL_CALLS)
        method_index = sorted(EXPECTED_METHODS).index(new)
        self.assertEqual(
            new_calls.tolist(),
            [value - 100 - method_index for value in TOTAL_CALLS],
        )
        first_summary = next(
            iter(report["results_by_inner_budget"].values())
        )["methods"][new]
        self.assertEqual(
            [lower[0], upper[0]],
            first_summary["aggregate_rmse_bootstrap_95"],
        )

    def test_strict_validation_rejects_partial_or_wrong_repeat_report(self) -> None:
        report = _synthetic_report()
        report["configuration"]["repeats"] = 4
        with self.assertRaisesRegex(ValueError, "exactly three repeats"):
            validate_report(report)

        report = _synthetic_report()
        first_row = next(iter(report["results_by_inner_budget"].values()))
        del first_row["methods"][next(iter(EXPECTED_METHODS))]
        with self.assertRaisesRegex(ValueError, "method mismatch"):
            validate_report(report)

    def test_png_and_pdf_are_exported(self) -> None:
        report = _synthetic_report()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "comparison.png"
            main_png, main_pdf, linear_png, linear_pdf = plot_results(
                report, output
            )
            self.assertEqual(main_png, output)
            self.assertEqual(main_pdf, output.with_suffix(".pdf"))
            self.assertEqual(
                linear_png,
                Path(directory) / "comparison_linear.png",
            )
            self.assertEqual(linear_pdf, linear_png.with_suffix(".pdf"))
            for png_path, pdf_path in (
                (main_png, main_pdf),
                (linear_png, linear_pdf),
            ):
                self.assertGreater(png_path.stat().st_size, 10_000)
                self.assertGreater(pdf_path.stat().st_size, 1_000)
                self.assertEqual(
                    png_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n"
                )
                self.assertEqual(pdf_path.read_bytes()[:4], b"%PDF")
                width, height = struct.unpack(
                    ">II", png_path.read_bytes()[16:24]
                )
                self.assertGreaterEqual(width / height, 0.98)
                self.assertLessEqual(width / height, 1.02)

    def test_default_output_inserts_rmse_before_budget_range(self) -> None:
        source = Path(
            "results/json/"
            "wine_full_train_rbf_svm_all_baselines_3repeats_71k_1p42m.json"
        )
        self.assertEqual(
            _default_output(source),
            Path(
                "results/png/"
                "wine_full_train_rbf_svm_all_baselines_3repeats_"
                "rmse_71k_1p42m.png"
            ).resolve(),
        )
        self.assertEqual(
            _linear_output(_default_output(source)),
            Path(
                "results/png/"
                "wine_full_train_rbf_svm_all_baselines_3repeats_"
                "linear_rmse_71k_1p42m.png"
            ).resolve(),
        )


if __name__ == "__main__":
    unittest.main()

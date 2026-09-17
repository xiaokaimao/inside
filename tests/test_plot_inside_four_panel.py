from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import numpy as np

from experiments.plot_inside_comparison import METHOD_ORDER
from experiments.plot_inside_four_panel import (
    DISPLAY_METHOD_ORDER,
    PANEL_ORDER,
    PANEL_TAGS,
    _relative_error_ratios,
    _metric_series,
    build_four_panel_figure,
    plot_four_panel_results,
    validate_four_panel_reports,
)


def _report(dataset: str) -> dict:
    results = {}
    for budget_index, inner in enumerate((100, 200, 400, 800, 1600)):
        total = inner + 14
        methods = {}
        for method_index, method in enumerate(METHOD_ORDER):
            rmse = (method_index + 2) * 1e-3 / (budget_index + 1)
            actual = total - 5 if method == "tmc_shapley" else total
            estimates = [
                [1.0 + rmse, 2.0 - 2.0 * rmse, -4.0 + 4.0 * rmse],
                [1.0 + 2.0 * rmse, 2.0 + 2.0 * rmse, -4.0 - 4.0 * rmse],
                [1.0 - rmse, 2.0 + 4.0 * rmse, -4.0 + 8.0 * rmse],
            ]
            methods[method] = {
                "aggregate_rmse": rmse,
                "aggregate_rmse_bootstrap_95": [rmse * 0.8, rmse * 1.2],
                "mean_actual_utility_calls": actual,
                "estimates": estimates,
            }
        results[str(inner)] = {
            "inner_utility_calls": inner,
            "total_utility_calls_per_estimate": total,
            "methods": methods,
        }
    return {
        "status": "complete",
        "configuration": {
            "dataset": dataset,
            "methods": list(METHOD_ORDER),
            "repeats": 3,
        },
        "game": {"dataset": dataset},
        "ground_truth": {"values": [1.0, 2.0, -4.0]},
        "results_by_inner_budget": results,
    }


def _reports() -> dict[str, dict]:
    return {panel: _report(panel) for panel in PANEL_ORDER}


class InsideFourPanelPlotTests(unittest.TestCase):
    def test_panel_order_matches_publication_contract(self) -> None:
        self.assertEqual(
            PANEL_ORDER,
            ("voting", "airport", "wine", "cancer"),
        )

    def test_relative_error_formulas_are_computed_per_repeat(self) -> None:
        truth = np.asarray([2.0, -4.0])
        estimates = np.asarray([[3.0, -2.0], [1.0, -8.0]])
        aer, mer = _relative_error_ratios(estimates, truth)
        np.testing.assert_allclose(aer, [0.5, 0.75])
        np.testing.assert_allclose(mer, [0.5, 1.0])
        self.assertAlmostEqual(float(aer.mean()), 0.625)
        self.assertAlmostEqual(float(mer.mean()), 0.75)

    def test_mean_and_sample_std_are_across_seed_metrics(self) -> None:
        report = _report("wine")
        row = next(iter(report["results_by_inner_budget"].values()))
        method = "inside_greedy"
        # Reference [1,2,-4], repeat errors [0,0,0], [1,2,-4], [2,4,-8].
        row["methods"][method]["estimates"] = [[1,2,-4], [2,4,-8], [3,6,-12]]
        for metric, scale in (("rmse", np.sqrt(7)), ("aer", 1), ("mer", 1)):
            with self.subTest(metric=metric):
                _, mean, lower, upper = _metric_series(report, [row], method, metric)
                np.testing.assert_allclose(mean, [scale])
                np.testing.assert_allclose(lower, [0], atol=1e-14)
                np.testing.assert_allclose(upper, [2 * scale])

    def test_relative_error_rejects_exact_zero_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            _relative_error_ratios([[1.0, 2.0]], [0.0, 1.0])

    def test_contract_requires_each_named_dataset_once(self) -> None:
        reports = _reports()
        validate_four_panel_reports(reports)

        missing = dict(reports)
        del missing["cancer"]
        with self.assertRaisesRegex(ValueError, "missing=\\['cancer'\\]"):
            validate_four_panel_reports(missing)

        swapped = dict(reports)
        swapped["wine"] = reports["airport"]
        with self.assertRaisesRegex(ValueError, "wine panel received dataset"):
            validate_four_panel_reports(swapped)

    def test_figure_has_one_top_legend_and_requested_omissions(self) -> None:
        figure = build_four_panel_figure(_reports())
        try:
            self.assertEqual(len(figure.axes), 4)
            self.assertEqual(len(figure.legends), 1)
            legend = figure.legends[0]
            self.assertIsNotNone(legend)
            self.assertTrue(
                all(axis.get_legend() is None for axis in figure.axes[1:])
            )
            legend_labels = [text.get_text() for text in legend.get_texts()]
            self.assertEqual(
                legend_labels,
                [
                    "INSIDE-Coalition",
                    "INSIDE-Orbit",
                    "OFA",
                    "CC",
                    "S-Diff",
                    "KernelSHAP",
                    "TMC-Shapley",
                ],
            )
            self.assertNotIn("OFA linear", legend_labels)
            self.assertNotIn("OFA ratio", legend_labels)

            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            legend_box = legend.get_window_extent(renderer=renderer)
            self.assertGreater(legend_box.y0, max(
                axis.get_window_extent(renderer=renderer).y1 for axis in figure.axes
            ))

            panel_bottoms = [axis.get_position().y0 for axis in figure.axes]
            self.assertLess(max(panel_bottoms) - min(panel_bottoms), 1e-12)

            for axis, panel in zip(figure.axes, PANEL_ORDER):
                with self.subTest(panel=panel):
                    labels = [line.get_label() for line in axis.lines]
                    self.assertNotIn("OFA linear", labels)
                    if panel == "cancer":
                        self.assertNotIn("S-Diff", labels)
                        self.assertEqual(len(labels), 6)
                    else:
                        self.assertIn("S-Diff", labels)
                        self.assertEqual(len(labels), 7)
                    self.assertEqual(axis.get_title(), "")
                    self.assertEqual(axis.get_xscale(), "log")
                    self.assertEqual(axis.get_yscale(), "log")
                    self.assertEqual(axis.get_box_aspect(), 1.0)
                    self.assertTrue(
                        all(spine.get_visible() for spine in axis.spines.values())
                    )
                    self.assertEqual(
                        [text.get_text() for text in axis.texts],
                        [PANEL_TAGS[panel]],
                    )
        finally:
            plt.close(figure)

    def test_aer_and_mer_reuse_the_four_panel_presentation(self) -> None:
        for metric in ("aer", "mer"):
            with self.subTest(metric=metric):
                figure = build_four_panel_figure(_reports(), metric=metric)
                try:
                    self.assertEqual(figure._supylabel.get_text(), metric.upper())
                    self.assertEqual(len(figure.axes), 4)
                    self.assertEqual(len(figure.legends), 0)
                    self.assertIsNone(figure.axes[0].get_legend())
                    self.assertTrue(all(len(axis.collections) == 0 for axis in figure.axes))
                    self.assertTrue(
                        all(
                            axis.get_legend() is None
                            for axis in figure.axes[1:]
                        )
                    )
                    for axis, panel in zip(figure.axes, PANEL_ORDER):
                        labels = [line.get_label() for line in axis.lines]
                        self.assertNotIn("OFA linear", labels)
                        if panel == "cancer":
                            self.assertNotIn("S-Diff", labels)
                finally:
                    plt.close(figure)

    def test_export_writes_transparent_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "inside_four_panel.png"
            png, pdf, metadata_path = plot_four_panel_results(
                _reports(), output
            )
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertGreater(pdf.stat().st_size, 1_000)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["layout"], "1x4")
            self.assertIsNone(metadata["chart_title"])
            self.assertFalse(metadata["axes_titles"])
            self.assertEqual(
                metadata["shared_legend"]["method_order"],
                list(DISPLAY_METHOD_ORDER),
            )
            self.assertEqual(
                metadata["shared_legend"]["location"],
                "above_panels",
            )
            self.assertEqual(
                metadata["shared_legend"]["host_panel"], None
            )
            self.assertTrue(metadata["shared_legend"]["rendered_once"])
            self.assertIn("OFA", metadata["shared_legend"]["labels"])
            self.assertNotIn(
                "OFA ratio", metadata["shared_legend"]["labels"]
            )
            omissions = metadata["presentation_omissions"]
            self.assertEqual(omissions["ofa_iid_linear"]["scope"], "all_panels")
            self.assertEqual(omissions["s_diff"]["scope"], ["cancer"])
            self.assertEqual(
                omissions["s_diff"]["reason_code"],
                "quadratic_state_memory_scalability",
            )
            self.assertFalse(omissions["s_diff"]["actual_oom_observed"])
            cancer = next(
                panel for panel in metadata["panels"] if panel["key"] == "cancer"
            )
            self.assertNotIn("s_diff", cancer["methods_shown"])
            for artifact in ("png", "pdf", "svg"):
                self.assertEqual(
                    len(metadata["outputs"][artifact]["sha256"]), 64
                )

    def test_aer_export_records_formula_and_denominator_policy(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "inside_four_panel_aer.png"
            _png, _pdf, metadata_path = plot_four_panel_results(
                _reports(), output, metric="aer"
            )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["artifact"], "inside_aer_four_panel")
            self.assertEqual(metadata["metric"]["key"], "aer")
            self.assertEqual(
                metadata["metric"]["repeat_aggregation"],
                "arithmetic_mean_of_repeat_level_AER",
            )
            self.assertEqual(
                metadata["metric"]["denominator_policy"],
                "strict_nonzero_no_epsilon",
            )
            self.assertEqual(
                metadata["panels"][0]["reference_diagnostics"][
                    "exact_zero_count"
                ],
                0,
            )
            self.assertIn(
                "mean_over_repeats",
                metadata["panels"][0]["plotted_values"]["inside_greedy"],
            )


if __name__ == "__main__":
    unittest.main()

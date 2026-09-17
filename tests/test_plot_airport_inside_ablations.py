from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib.pyplot as plt
import numpy as np

from experiments.plot_airport_inside_ablations import (
    COMPONENT_ARTIFACT,
    METHOD_LABELS as DISPLAY_LABELS,
    _scatter_trend,
    _geometry_series,
    GEOMETRY_ARTIFACT,
    ORBIT_ARTIFACT,
    _sha256,
    build_component_figure,
    build_geometry_figure,
    build_orbit_figure,
    plot_all,
    validate_plot_report,
)
from experiments.run_airport_inside_ablations import (
    BOUNDARY_CALLS,
    COMPONENT_METHODS,
    EXPERIMENT_ID,
    GEOMETRY_METHODS,
    METHOD_LABELS,
    METHOD_ORDER,
    ORBIT_METHODS,
    PROTOCOL_VERSION,
    _aggregate,
    _configuration,
)


class AirportInsideAblationPlotTests(unittest.TestCase):
    def test_geometry_statistics_and_trend(self):
        report = self._report()
        rows = sorted(report["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"])
        _, mean, low, high = _geometry_series(rows, "full_inside", "first")
        raw = np.array([r["methods"]["full_inside"]["first_moment_rms_by_repeat"] for r in rows])
        np.testing.assert_allclose(mean, raw.mean(axis=1))
        np.testing.assert_allclose(high-low, 2*raw.std(axis=1, ddof=1))
        trend = _scatter_trend(report)
        self.assertEqual(trend["n"], 45)
        self.assertGreater(trend["slope"], 0)

    @staticmethod
    def _report() -> dict[str, object]:
        repeats = 3
        multipliers = (500, 1000, 2000, 5000, 10000)
        inner_budgets = tuple(100 * value for value in multipliers)
        truth = np.asarray([0.40, -0.20, 0.10, -0.30])
        error_scale = {
            "iid_ofa": 0.020,
            "first_only": 0.017,
            "frame_only": 0.015,
            "full_inside": 0.013,
            "random_orbit": 0.016,
            "inside_orbit": 0.012,
        }
        first_scale = {
            "iid_ofa": 0.060,
            "first_only": 0.020,
            "frame_only": 0.055,
            "full_inside": 0.025,
            "random_orbit": 0.0,
            "inside_orbit": 0.0,
        }
        frame_scale = {
            "iid_ofa": 0.120,
            "first_only": 0.110,
            "frame_only": 0.065,
            "full_inside": 0.060,
            "random_orbit": 0.100,
            "inside_orbit": 0.055,
        }
        objective = {
            "first_only": (["first_moment"], 0.0),
            "frame_only": (["second_moment"], 1.0),
            "full_inside": (["first_moment", "second_moment"], 1.0),
        }

        cells: list[dict[str, object]] = []
        for repeat in range(repeats):
            repeat_factor = 1.0 + 0.035 * repeat
            for budget_index, inner_calls in enumerate(inner_budgets):
                budget_factor = math.sqrt(inner_calls / inner_budgets[0])
                component_schedule = (
                    f"component-schedule-{repeat}-{budget_index}"
                )
                component_relabel = (
                    f"component-relabel-{repeat}-{budget_index}"
                )
                orbit_schedule = f"orbit-schedule-{repeat}-{budget_index}"
                orbit_pool = f"orbit-pool-{repeat}-{budget_index}"
                orbit_relabel = f"orbit-relabel-{repeat}-{budget_index}"
                methods: dict[str, object] = {}
                for method_index, method in enumerate(METHOD_ORDER):
                    scale = (
                        error_scale[method]
                        * repeat_factor
                        / budget_factor
                    )
                    error = scale * np.asarray([1.0, -0.7, 0.4, -0.2])
                    estimate = truth + error
                    repeat_rmse = float(np.sqrt(np.mean(np.square(error))))
                    first = (
                        first_scale[method]
                        * (1.0 + 0.018 * repeat + 0.001 * method_index)
                        / budget_factor
                    )
                    frame = (
                        frame_scale[method]
                        * (1.0 + 0.025 * repeat + 0.001 * method_index)
                        / budget_factor
                    )
                    if method in objective:
                        components, second_weight = objective[method]
                        diagnostics: dict[str, object] = {
                            "relabel_permutation_sha256": component_relabel,
                            "second_moment_scope": "per_size",
                            "objective_components": components,
                            "second_moment_weight": second_weight,
                        }
                        schedule_hash = component_schedule
                    elif method in ORBIT_METHODS:
                        diagnostics = {
                            "shared_candidate_pool_sha256": orbit_pool,
                            "shared_relabel_permutations_sha256": orbit_relabel,
                        }
                        schedule_hash = orbit_schedule
                    else:
                        diagnostics = {}
                        schedule_hash = (
                            f"iid-schedule-{repeat}-{budget_index}"
                        )
                    methods[method] = {
                        "method": method,
                        "label": METHOD_LABELS[method],
                        "estimate": estimate.tolist(),
                        "repeat_rmse": repeat_rmse,
                        "actual_utility_calls": inner_calls + BOUNDARY_CALLS,
                        "size_schedule_sha256": schedule_hash,
                        "coalition_design_sha256": (
                            f"coalition-{method}-{repeat}-{budget_index}"
                        ),
                        "coverage": {
                            "all_player_size_strata_covered": True,
                        },
                        "geometry": {
                            "aggregation": "equal_size_macro_rms",
                            "target": "P/(n-1)",
                            "all_inner_sizes_present": True,
                            "observed_inner_size_count": 97,
                            "missing_sizes": [],
                            "first_moment_rms": first,
                            "frame_frobenius_rms": frame,
                        },
                        "design_diagnostics": diagnostics,
                    }
                cells.append(
                    {
                        "repeat": repeat,
                        "budget_index": budget_index,
                        "inner_utility_calls": inner_calls,
                        "total_utility_calls": inner_calls + BOUNDARY_CALLS,
                        "paired_controls": {
                            "component_shared_size_schedule_sha256": (
                                component_schedule
                            ),
                            "component_shared_relabel_sha256": (
                                component_relabel
                            ),
                            "orbit_shared_candidate_pool_sha256": orbit_pool,
                            "orbit_shared_relabel_sha256": orbit_relabel,
                        },
                        "methods": methods,
                    }
                )

        results, relation = _aggregate(
            cells,
            inner_budgets=inner_budgets,
            repeats=repeats,
            bootstrap_samples=101,
        )
        configuration = _configuration(
            budget_multipliers=multipliers,
            repeats=repeats,
            base_seed=20260827,
            bootstrap_samples=101,
            repeat_processes=1,
            design_jobs=1,
            design_start_method="spawn",
            greedy_candidate_pool=64,
            orbit_candidate_pool=4,
        )
        return {
            "status": "complete",
            "experiment": EXPERIMENT_ID,
            "configuration": configuration,
            "ground_truth": {"values": truth.tolist()},
            "raw_cells": cells,
            "results_by_inner_budget": results,
            "geometry_error_relation": relation,
        }

    def test_three_figure_contracts(self) -> None:
        report = self._report()
        figures = (
            (
                build_component_figure(report),
                COMPONENT_METHODS,
                (("log", "log"),),
            ),
            (
                build_orbit_figure(report),
                ORBIT_METHODS,
                (("log", "log"),),
            ),
            (
                build_geometry_figure(report),
                GEOMETRY_METHODS,
                (("log", "linear"), ("log", "log"), ("log", "log")),
            ),
        )
        try:
            for figure, methods, expected_scales in figures:
                with self.subTest(methods=methods):
                    self.assertEqual(len(figure.axes), len(expected_scales))
                    self.assertEqual(len(figure.legends), 0)
                    legends = [
                        axis.get_legend()
                        for axis in figure.axes
                        if axis.get_legend() is not None
                    ]
                    self.assertEqual(len(legends), 1)
                    self.assertEqual(
                        [
                            text.get_text()
                            for text in legends[0].get_texts()
                        ],
                        [DISPLAY_LABELS[method] for method in methods],
                    )
                    for axis, scales in zip(
                        figure.axes, expected_scales, strict=True
                    ):
                        self.assertEqual(axis.get_title(), "")
                        self.assertEqual(
                            (axis.get_xscale(), axis.get_yscale()), scales
                        )
                        self.assertEqual(axis.get_box_aspect(), 1.0)
                        self.assertTrue(
                            all(
                                spine.get_visible()
                                for spine in axis.spines.values()
                            )
                        )
        finally:
            for figure, _methods, _scales in figures:
                plt.close(figure)

    def test_orbit_first_moment_remains_exact_zero(self) -> None:
        report = self._report()
        for row in report["results_by_inner_budget"].values():
            for method in ORBIT_METHODS:
                self.assertEqual(
                    row["methods"][method]["mean_first_moment_rms"], 0.0
                )
                self.assertTrue(
                    all(
                        value == 0.0
                        for value in row["methods"][method][
                            "first_moment_rms_by_repeat"
                        ]
                    )
                )

        figure = build_geometry_figure(report)
        try:
            inside_line = next(
                line
                for line in figure.axes[0].lines
                if line.get_label() == METHOD_LABELS["inside_orbit"]
            )
            np.testing.assert_array_equal(
                np.asarray(inside_line.get_ydata()), np.zeros(5)
            )
            self.assertLess(figure.axes[0].get_ylim()[0], 0.0)
            self.assertIn(0.0, figure.axes[0].get_yticks())
        finally:
            plt.close(figure)

    def test_exports_png_pdf_metadata_and_valid_hashes(self) -> None:
        report = self._report()
        expected_artifacts = {
            "component": COMPONENT_ARTIFACT,
            "orbit": ORBIT_ARTIFACT,
            "geometry": GEOMETRY_ARTIFACT,
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text(
                json.dumps(report, allow_nan=False), encoding="utf-8"
            )
            outputs = plot_all(
                report,
                root / "airport_ablation",
                source_report=source,
            )
            self.assertEqual(set(outputs), set(expected_artifacts))
            for key, (png, pdf, metadata_path) in outputs.items():
                with self.subTest(artifact=key):
                    self.assertGreater(png.stat().st_size, 10_000)
                    self.assertGreater(pdf.stat().st_size, 1_000)
                    metadata = json.loads(
                        metadata_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(
                        metadata["artifact"], expected_artifacts[key]
                    )
                    self.assertEqual(metadata["chart_title"], None)
                    self.assertEqual(
                        metadata["legend_location"], "inside_axes"
                    )
                    self.assertEqual(
                        metadata["source_report"]["sha256"],
                        _sha256(source),
                    )
                    self.assertEqual(
                        metadata["outputs"]["png"]["sha256"],
                        _sha256(png),
                    )
                    self.assertEqual(
                        metadata["outputs"]["pdf"]["sha256"],
                        _sha256(pdf),
                    )
                    self.assertEqual(
                        len(metadata["plotting_script"]["sha256"]), 64
                    )
                    self.assertTrue(
                        all(
                            value == "passed"
                            for value in metadata["validation"].values()
                        )
                    )
            geometry = json.loads(
                outputs["geometry"][2].read_text(encoding="utf-8")
            )
            self.assertIn(
                "no epsilon",
                geometry["interpretation"]["first_moment_zero_policy"],
            )

    def test_corrupted_aggregate_and_scatter_are_rejected(self) -> None:
        mutations = (
            (
                "aggregate",
                lambda report: report["results_by_inner_budget"]["50000"][
                    "methods"
                ]["full_inside"].__setitem__("aggregate_rmse", 9.0),
                "aggregate RMSE is inconsistent",
            ),
            (
                "scatter",
                lambda report: report["geometry_error_relation"]["points"][
                    0
                ].__setitem__("frame_frobenius_rms", 9.0),
                "scatter frame discrepancy is inconsistent",
            ),
        )
        for label, mutation, message in mutations:
            with self.subTest(label=label):
                report = copy.deepcopy(self._report())
                mutation(report)
                with self.assertRaisesRegex(ValueError, message):
                    validate_plot_report(report)


if __name__ == "__main__":
    unittest.main()

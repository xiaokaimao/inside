from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from experiments.add_shapdoe_baselines import (
    BASE_SEED, EXPERIMENT_ID, SOURCE_PATHS, _assemble, _digest, _seed,
    run_dataset, validate_report,
)
from experiments.plot_shapdoe_comparison import build_figure, export_report
from frame_ofa import shapdoe_budget


def _report() -> dict:
    source = json.loads(SOURCE_PATHS["airport"].read_text())
    report = {
        "experiment": EXPERIMENT_ID, "base_report": source, "cells": {},
        "source": {"canonical_json_sha256": _digest(source)},
        "configuration": {"dataset": "airport", "added_methods": ["ls", "coa"], "repeats": 3, "seed": BASE_SEED},
    }
    truth = np.asarray(source["ground_truth"]["values"])
    direction = np.linspace(-1, 1, len(truth))
    direction -= direction.mean()
    for index, row in enumerate(sorted(source["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"])):
        for method in ["ls", "coa"]:
            plan = shapdoe_budget(len(truth), row["total_utility_calls_per_estimate"], method)
            if plan.num_designs == 0:
                continue
            for repeat in range(3):
                diagnostic = {"budget": asdict(plan), "truncation": False, "efficiency_projection": False,
                              "boundary_reuse": True, "empty_utility": 0.0, "full_utility": 10.0}
                for field in ["utility_evaluations", "unused_calls", "num_designs", "num_permutations"]:
                    diagnostic[field] = getattr(plan, field)
                report["cells"][f"{method}/{index}/{repeat}"] = {
                    "seed": _seed(method, index, repeat), "elapsed_seconds": 1.0,
                    "estimate": (truth + direction * .001 * (repeat + 1)/(index + 1)).tolist(),
                    "diagnostics": diagnostic,
                }
    _assemble(report)
    return report


class ShapDoEIntegrationTests(unittest.TestCase):
    def test_augmented_report_keeps_original_curves_and_audits_metrics(self):
        report = _report()
        audit = validate_report(report)
        self.assertEqual(audit["completed_method_budget_cells"], 6)
        self.assertEqual(audit["infeasible_method_budget_cells"], 4)
        self.assertEqual(len(report["cells"]), 18)

    def test_validator_rejects_modified_baseline(self):
        report = _report()
        row = next(iter(report["results_by_inner_budget"].values()))
        row["methods"]["inside_greedy"]["aggregate_rmse"] *= 2
        with self.assertRaisesRegex(ValueError, "existing baseline"):
            validate_report(report)

    def test_validator_rejects_new_metric_and_call_corruption(self):
        report = _report()
        key = min(report["results_by_inner_budget"], key=int)
        corrupted = deepcopy(report)
        corrupted["results_by_inner_budget"][key]["methods"]["shapdoe_ls"]["aggregate_rmse"] *= 2
        with self.assertRaisesRegex(ValueError, "recomputation"):
            validate_report(corrupted)
        corrupted = deepcopy(report)
        corrupted["cells"]["ls/0/0"]["diagnostics"]["utility_evaluations"] += 1
        with self.assertRaisesRegex(ValueError, "call accounting"):
            validate_report(corrupted)

    def test_resume_checks_partial_cells_before_new_work(self):
        report = _report()
        del report["cells"]["ls/0/1"]
        del report["cells"]["ls/0/2"]
        _assemble(report)
        validate_report(report, require_complete=False)
        report["cells"]["ls/0/0"]["seed"] += 1
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_report(report, require_complete=False)

    def test_validator_rejects_false_infeasibility_and_changed_reference(self):
        report = _report()
        key = min(report["results_by_inner_budget"], key=int)
        report["results_by_inner_budget"][key]["methods"]["shapdoe_coa"]["minimum_call_budget"] -= 1
        with self.assertRaisesRegex(ValueError, "infeasibility"):
            validate_report(report)
        report = _report()
        report["source"]["canonical_json_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_report(report)

    def test_completed_resume_does_not_repeat_evaluations(self):
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            output = output_dir / "airport_inside_with_shapdoe_3repeats.json"
            report = _report()
            output.write_text(json.dumps(report))
            with patch("experiments.add_shapdoe_baselines.GameEvaluator", side_effect=AssertionError("unexpected recomputation")):
                result = run_dataset("airport", jobs=1, output_dir=output_dir)
            self.assertEqual(result, output)
            self.assertTrue(output.with_suffix(".validation.json").exists())

    def test_plot_uses_actual_calls_and_only_feasible_coa_points(self):
        report = _report()
        figure = build_figure(report)
        try:
            lines = {line.get_label(): line for line in figure.axes[0].lines}
            self.assertEqual(len(lines), 10)
            self.assertEqual(len(lines["ShapDoE-COA"].get_xdata()), 1)
            self.assertEqual(lines["ShapDoE-LS"].get_xdata()[0], 49502)
            self.assertEqual(figure.axes[0].get_xlabel(), "Actual utility calls")
        finally:
            plt.close(figure)
        with TemporaryDirectory() as temporary:
            path = Path(temporary)/"report.json"
            path.write_text(json.dumps(report))
            for artifact in export_report(path):
                self.assertGreater(artifact.stat().st_size, 100)

    def test_partial_run_is_explicit_and_requires_plot_opt_in(self):
        report = _report()
        for key in list(report["cells"]):
            if int(key.split("/")[1]) > 2:
                del report["cells"][key]
        _assemble(report)
        report["status"] = "partial"
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_figure(report)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "airport_inside_with_shapdoe_3repeats.json"
            output.write_text(json.dumps(report))
            with patch("experiments.add_shapdoe_baselines.GameEvaluator", side_effect=AssertionError("unexpected evaluation")):
                run_dataset("airport", jobs=1, output_dir=directory, budget_indices=(0, 1, 2))
            updated = json.loads(output.read_text())
            self.assertEqual(updated["status"], "partial")
            figure = build_figure(updated, allow_incomplete=True)
            try:
                self.assertTrue(any("Partial" in text.get_text() for text in figure.texts))
                ls = next(line for line in figure.axes[0].lines if line.get_label() == "ShapDoE-LS")
                self.assertEqual(len(ls.get_xdata()), 3)
            finally:
                plt.close(figure)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from experiments.add_orthogonal_baseline import run_dataset, validate_report
from experiments.plot_orthogonal_comparison import build_figure, export_report
from tests.test_add_shapdoe_baselines import _report as shapdoe_report


class OrthogonalIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = TemporaryDirectory()
        cls.directory = Path(cls.temporary.name)
        cls.source = cls.directory / "source.json"
        cls.source.write_text(json.dumps(shapdoe_report()))
        # Exercise real sampling, evaluation, checkpointing and validation.
        cls.output = run_dataset("airport", jobs=1, output_dir=cls.directory,
                                 source_path=cls.source, budget_indices=(0,))
        cls.report = json.loads(cls.output.read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_three_real_repeats_preserve_source_and_existing_methods(self):
        audit = validate_report(self.report, require_complete=False)
        self.assertEqual(audit["orthogonal_repeat_cells"], 3)
        self.assertEqual(self.report["source_report"], json.loads(self.source.read_text()))
        for key, row in self.report["source_report"]["results_by_inner_budget"].items():
            updated = self.report["results_by_inner_budget"][key]["methods"]
            self.assertEqual({k: v for k, v in updated.items() if k != "orthogonal"}, row["methods"])

    def test_resume_never_repeats_finished_cells(self):
        with patch("experiments.add_orthogonal_baseline.GameEvaluator", side_effect=AssertionError("unexpected work")):
            path = run_dataset("airport", jobs=1, output_dir=self.directory,
                               source_path=self.source, budget_indices=(0,))
        self.assertEqual(path, self.output)
        self.assertEqual(json.loads(path.read_text())["cells"], self.report["cells"])

    def test_corrupt_estimates_budgets_and_seeds_are_rejected(self):
        for field in ("seed", "calls", "estimate", "status", "source", "metric", "old_method"):
            with self.subTest(field=field):
                report = deepcopy(self.report)
                cell = report["cells"]["0/0"]
                row = report["results_by_inner_budget"][min(report["results_by_inner_budget"], key=int)]
                if field == "seed":
                    cell["seed"] += 1
                elif field == "calls":
                    cell["diagnostics"]["utility_evaluations"] += 1
                elif field == "estimate":
                    cell["estimate"][0] += .1
                elif field == "status":
                    report["status"] = "complete"
                elif field == "source":
                    report["source"]["canonical_json_sha256"] = "changed"
                elif field == "metric":
                    row["methods"]["orthogonal"]["aggregate_rmse"] *= 2
                else:
                    row["methods"]["shapdoe_ls"]["aggregate_rmse"] *= 2
                with self.assertRaises(ValueError):
                    validate_report(report, require_complete=False)

    def test_plot_adds_orthogonal_at_actual_calls_and_labels_pending_points(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            build_figure(self.report)
        figure = build_figure(self.report, allow_incomplete=True)
        try:
            lines = {line.get_label(): line for line in figure.axes[0].lines}
            self.assertEqual(len(lines), 11)
            self.assertEqual(list(lines["Orthogonal"].get_xdata()), [50096])
            self.assertTrue(any("Partial" in text.get_text() for text in figure.texts))
        finally:
            plt.close(figure)
        for artifact in export_report(self.output, allow_incomplete=True):
            self.assertGreater(artifact.stat().st_size, 100)
        with self.output.with_suffix(".csv").open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["status"] == "complete":
                self.assertGreater(float(row["actual_calls"]), 0)
            else:
                self.assertEqual(row["actual_calls"], "")

    def test_changed_source_is_rejected_before_evaluation(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / self.output.name
            output.write_text(self.output.read_text())
            changed = json.loads(self.source.read_text())
            changed["extra_metadata"] = "new source"
            source = directory / "changed.json"
            source.write_text(json.dumps(changed))
            with patch("experiments.add_orthogonal_baseline.GameEvaluator", side_effect=AssertionError("unexpected work")):
                with self.assertRaisesRegex(ValueError, "source changed"):
                    run_dataset("airport", jobs=1, output_dir=directory, source_path=source, budget_indices=(0,))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.build_mean_balance_fix_report import (
    ROOT,
    SOURCE_SQL,
    build_artifact,
)
from experiments.build_mean_balance_fix_summary import _summarize_rows


RATIOS = {
    "wine": (0.50, 0.75, 1.00, 1.25, 2.00),
    "airport": (0.40, 0.80, 0.90, 1.10, 1.50),
    "voting": (0.60, 0.70, 0.95, 1.05, 1.20),
}


def _summary() -> dict:
    datasets = {}
    all_rows = []
    for dataset, ratios in RATIOS.items():
        rows = []
        for index, (target, ratio) in enumerate(
            zip((100, 200, 300, 400, 500), ratios)
        ):
            legacy_rmse = 2.0
            normalized_rmse = legacy_rmse * ratio
            orbit_rmse = normalized_rmse / 0.8
            rows.append(
                {
                    "dataset": dataset,
                    "budget_index": index,
                    "total_utility_calls": target,
                    "legacy_raw_lambda_0p1_greedy_rmse": legacy_rmse,
                    "normalized_lambda0_1_greedy_rmse": normalized_rmse,
                    "inside_orbit_rmse": orbit_rmse,
                    "normalized_to_legacy_rmse_ratio": ratio,
                    "normalized_vs_legacy_improvement_percent": 100.0
                    * (1.0 - ratio),
                    "normalized_to_orbit_rmse_ratio": 0.8,
                    "legacy_slice_mean_direction_rms": 0.1,
                    "normalized_slice_mean_direction_rms": 0.05,
                    "normalized_to_legacy_slice_mean_direction_rms_ratio": 0.5,
                    "legacy_frobenius_discrepancy": 0.2,
                    "normalized_frobenius_discrepancy": 0.25,
                    "normalized_to_legacy_frobenius_discrepancy_ratio": 1.25,
                    "actual_utility_calls_by_repeat": {
                        "legacy_greedy": [target, target, target],
                        "normalized_greedy": [target, target, target],
                        "inside_orbit": [target, target, target],
                    },
                }
            )
        all_rows.extend(rows)
        datasets[dataset] = {
            "normalized_report": str(
                (ROOT / "results" / f"fixture_{dataset}_normalized.json").resolve()
            ),
            "legacy_report": str(
                (ROOT / "results" / f"fixture_{dataset}_legacy.json").resolve()
            ),
            "rows": rows,
            "summary": _summarize_rows(rows),
            "validation": {
                "normalized_status": "ready_to_share",
                "legacy_status": "ready_to_share",
                "normalized_total_actual_utility_calls": 1,
                "legacy_total_actual_utility_calls": 1,
            },
        }
    return {
        "status": "ready_to_share",
        "experiment": "fixture",
        "method_identity": {
            "legacy_greedy": {
                "method_key": "inside_greedy",
                "mean_balance_mode": "raw",
                "raw_lambda": 0.1,
            },
            "normalized_greedy": {
                "method_key": "inside_greedy",
                "mean_balance_mode": "normalized",
                "lambda0": 1.0,
            },
            "orbit_reference": {"method_key": "inside_orbit"},
        },
        "validated_design": {
            "datasets": ["wine", "airport", "voting"],
            "budget_points_per_dataset": 5,
            "repeats_per_budget_method": 3,
            "total_budget_points": 15,
            "normalized_and_legacy_total_actual_utility_calls_audited": 6,
        },
        "datasets": datasets,
        "summary": _summarize_rows(all_rows),
    }


class MeanBalanceFixReportTests(unittest.TestCase):
    def _write_summary(self, directory: str, summary: dict) -> Path:
        path = Path(directory) / "mean_balance_summary.json"
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def test_builds_dynamic_bounded_technical_report(self) -> None:
        with TemporaryDirectory(dir=ROOT / "results") as directory:
            path = self._write_summary(directory, _summary())
            artifact = build_artifact(path)

        manifest = artifact["manifest"]
        snapshot = artifact["snapshot"]
        rows = snapshot["datasets"]["mean_balance_results"]
        self.assertEqual(artifact["surface"], "report")
        self.assertEqual(
            manifest["blocks"][0]["body"], f"# {manifest['title']}"
        )
        self.assertTrue(
            manifest["blocks"][1]["body"].startswith("## Technical summary")
        )
        self.assertEqual(len(manifest["cards"]), 3)
        self.assertEqual(len(manifest["charts"]), 1)
        self.assertEqual(len(manifest["tables"]), 1)
        self.assertEqual(len(rows), 15)
        self.assertGreater(len(rows[0]), 30)
        self.assertEqual(
            manifest["sources"][0]["query"]["sql"], SOURCE_SQL
        )
        self.assertEqual(
            manifest["sources"][0]["query"]["tables_used"],
            ["artifact.mean_balance_results"],
        )
        self.assertEqual(
            len(manifest["sources"][0]["query"]["input_files"]), 7
        )
        self.assertTrue(
            all(
                not Path(value).is_absolute()
                for value in manifest["sources"][0]["query"]["input_files"]
            )
        )
        self.assertIn("8 points", manifest["blocks"][1]["body"])
        self.assertIn("0.500", manifest["blocks"][5]["body"])
        self.assertIn("1.250", manifest["blocks"][5]["body"])
        self.assertTrue(
            all(
                all(
                    value is None or isinstance(value, (str, int, float, bool))
                    for value in row.values()
                )
                for row in rows
            )
        )

    def test_narrative_and_cards_change_with_input_values(self) -> None:
        summary = _summary()
        for payload in summary["datasets"].values():
            for row in payload["rows"]:
                row["normalized_lambda0_1_greedy_rmse"] = 3.0
                row["normalized_to_legacy_rmse_ratio"] = 1.5
                row["normalized_vs_legacy_improvement_percent"] = -50.0
                row["inside_orbit_rmse"] = 2.0
                row["normalized_to_orbit_rmse_ratio"] = 1.5
            payload["summary"] = _summarize_rows(payload["rows"])
        summary["summary"] = _summarize_rows(
            [
                row
                for payload in summary["datasets"].values()
                for row in payload["rows"]
            ]
        )
        with TemporaryDirectory(dir=ROOT / "results") as directory:
            path = self._write_summary(directory, summary)
            artifact = build_artifact(path)

        technical_summary = artifact["manifest"]["blocks"][1]["body"]
        rows = artifact["snapshot"]["datasets"]["mean_balance_results"]
        self.assertIn("0 points", technical_summary)
        self.assertIn("RMSE increase", technical_summary)
        self.assertEqual(rows[0]["overall_improved_point_count"], 0)
        self.assertAlmostEqual(
            rows[0]["overall_geometric_mean_new_old_ratio"], 1.5
        )

    def test_rejects_tampered_stored_ratio(self) -> None:
        summary = deepcopy(_summary())
        summary["datasets"]["wine"]["rows"][0][
            "normalized_to_legacy_rmse_ratio"
        ] = 99.0
        with TemporaryDirectory(dir=ROOT / "results") as directory:
            path = self._write_summary(directory, summary)
            with self.assertRaisesRegex(ValueError, "recomputed value"):
                build_artifact(path)

    def test_rejects_source_paths_outside_project(self) -> None:
        summary = _summary()
        summary["datasets"]["wine"]["normalized_report"] = "/tmp/outside.json"
        with TemporaryDirectory(dir=ROOT / "results") as directory:
            path = self._write_summary(directory, summary)
            with self.assertRaisesRegex(ValueError, "inside the project root"):
                build_artifact(path)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.add_wine_cc_results import (
    METHOD_KEY,
    _initialize_or_resume_report,
    _validate_source_report,
    cc_call_plan,
)


def _source_report() -> dict:
    num_players = 142
    boundary_calls = 2 * num_players + 2
    budgets = [
        num_players * multiplier
        for multiplier in (500, 1000, 2000, 5000, 10000)
    ]
    totals = [budget + boundary_calls for budget in budgets]
    methods = {
        key: {
            "aggregate_rmse": 0.1,
            "estimates": np.zeros((10, num_players)).tolist(),
        }
        for key in (
            "official_ofa_fixed_ratio",
            "iid_linear_ofa",
            "frame_coupled_linear",
            "frame_orbit_ratio",
        )
    }
    return {
        "status": "complete",
        "configuration": {
            "num_players": num_players,
            "repeats": 10,
            "gt_pairs": 800_000,
            "inner_budgets": budgets,
            "total_call_budgets": totals,
        },
        "dataset": {"dataset": "wine"},
        "boundary": {
            "utility_calls": boundary_calls,
            "efficiency_target": 0.5,
        },
        "ground_truth": {
            "values": np.zeros(num_players).tolist(),
            "standard_errors": np.ones(num_players).tolist(),
        },
        "results_by_inner_budget": {
            str(budget): {
                "total_utility_calls_per_estimate": total,
                "methods": copy.deepcopy(methods),
            }
            for budget, total in zip(budgets, totals)
        },
    }


class WineCCAugmentationTests(unittest.TestCase):
    def test_same_total_call_plan_uses_two_calls_per_cc_sample(self) -> None:
        plan = cc_call_plan(_source_report())
        self.assertEqual(
            [item["total_utility_calls"] for item in plan],
            [71_286, 142_286, 284_286, 710_286, 1_420_286],
        )
        self.assertEqual(
            [item["complementary_contributions"] for item in plan],
            [35_643, 71_143, 142_143, 355_143, 710_143],
        )

    def test_source_validation_rejects_missing_original_method(self) -> None:
        report = _source_report()
        del report["results_by_inner_budget"]["71000"]["methods"][
            "frame_orbit_ratio"
        ]
        with self.assertRaisesRegex(ValueError, "missing source methods"):
            _validate_source_report(report)

    def test_initialization_is_non_destructive_and_resume_checks_source(self) -> None:
        source = _source_report()
        original = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            output_path = root / "with_cc.json"
            source_path.write_text(
                json.dumps(source, sort_keys=True), encoding="utf-8"
            )
            report = _initialize_or_resume_report(
                source=source,
                source_path=source_path,
                output_path=output_path,
                cc_seed=123,
                jobs=128,
                chunksize=512,
                start_method="spawn",
                cc_tasks=128,
            )
            self.assertEqual(source, original)
            self.assertEqual(report["status"], "cc_augmentation_running")
            metadata = report["configuration"]["cc_baseline"]
            self.assertEqual(metadata["method_key"], METHOD_KEY)
            self.assertEqual(
                metadata["additional_physical_utility_calls"],
                26_284_300,
            )

            resumed = _initialize_or_resume_report(
                source=source,
                source_path=source_path,
                output_path=output_path,
                cc_seed=123,
                jobs=128,
                chunksize=512,
                start_method="spawn",
                cc_tasks=128,
            )
            self.assertEqual(
                resumed["configuration"]["cc_baseline"], metadata
            )


if __name__ == "__main__":
    unittest.main()

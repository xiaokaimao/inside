from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.build_mean_balance_fix_summary import build_summary


RATIOS = {
    "wine": (0.50, 0.75, 1.00, 1.25, 2.00),
    "airport": (0.40, 0.80, 0.90, 1.10, 1.50),
    "voting": (0.60, 0.70, 0.95, 1.05, 1.20),
}


def _report(dataset: str) -> dict:
    results = {}
    for index, target in enumerate((100, 200, 300, 400, 500)):
        actual = [target, target, target]
        results[str(index)] = {
            "total_utility_calls_per_estimate": target,
            "methods": {
                "inside_greedy": {
                    "actual_utility_calls_by_repeat": actual,
                    "diagnostics_by_repeat": [
                        {
                            "design_diagnostics": {
                                "slice_mean_direction_rms": 0.1,
                                "frobenius_discrepancy": 0.2,
                            }
                        }
                        for _ in range(3)
                    ],
                },
                "inside_orbit": {
                    "actual_utility_calls_by_repeat": actual,
                },
            },
        }
    return {"dataset_key": dataset, "results_by_inner_budget": results}


def _comparison(
    normalized: dict,
    legacy: dict,
    *,
    normalized_path: Path,
    legacy_path: Path,
) -> dict:
    del legacy, normalized_path, legacy_path
    dataset = normalized["dataset_key"]
    rows = []
    for target, ratio in zip((100, 200, 300, 400, 500), RATIOS[dataset]):
        legacy_rmse = 2.0
        normalized_rmse = legacy_rmse * ratio
        orbit_rmse = normalized_rmse / 0.8
        rows.append(
            {
                "total_utility_calls": target,
                "legacy_raw_lambda_0p1_greedy_rmse": legacy_rmse,
                "normalized_lambda0_1_greedy_rmse": normalized_rmse,
                "inside_orbit_rmse": orbit_rmse,
                "normalized_vs_legacy_fractional_rmse_change": ratio - 1.0,
                "normalized_greedy_to_orbit_rmse_ratio": 0.8,
            }
        )
    return {
        "rows": rows,
        "validation": {
            "normalized": {
                "status": "ready_to_share",
                "dataset": dataset,
                "inside_greedy_balance_audit": {
                    "mode": "normalized",
                    "coefficient": 1.0,
                },
                "call_accounting": {"total_actual_utility_calls": 12_000},
            },
            "legacy": {
                "status": "ready_to_share",
                "dataset": dataset,
                "inside_greedy_balance_audit": {
                    "mode": "legacy_raw",
                    "coefficient": 0.1,
                },
                "call_accounting": {"total_actual_utility_calls": 12_000},
            },
        },
    }


def _inputs() -> tuple[dict, dict]:
    pairs = {
        dataset: (_report(dataset), _report(dataset)) for dataset in RATIOS
    }
    paths = {
        dataset: (
            Path(f"{dataset}_normalized.json"),
            Path(f"{dataset}_legacy.json"),
        )
        for dataset in RATIOS
    }
    return pairs, paths


class MeanBalanceFixSummaryTests(unittest.TestCase):
    def test_builds_fifteen_point_summary_and_geometric_aggregates(self) -> None:
        pairs, paths = _inputs()
        with patch(
            "experiments.build_mean_balance_fix_summary.build_comparison",
            side_effect=_comparison,
        ) as comparison:
            result = build_summary(pairs, report_paths=paths)

        self.assertEqual(comparison.call_count, 3)
        self.assertEqual(result["status"], "ready_to_share")
        self.assertEqual(result["validated_design"]["total_budget_points"], 15)
        self.assertEqual(
            result["validated_design"]["repeats_per_budget_method"], 3
        )
        self.assertEqual(result["summary"]["improved_budget_point_count"], 8)
        self.assertEqual(result["summary"]["tied_budget_point_count"], 1)
        self.assertEqual(result["summary"]["worsened_budget_point_count"], 6)
        expected = math.prod(
            ratio for ratios in RATIOS.values() for ratio in ratios
        ) ** (1.0 / 15.0)
        self.assertAlmostEqual(
            result["summary"][
                "normalized_to_legacy_geometric_mean_rmse_ratio"
            ],
            expected,
            places=14,
        )
        self.assertAlmostEqual(
            result["summary"]["normalized_to_orbit_geometric_mean_rmse_ratio"],
            0.8,
            places=14,
        )
        self.assertEqual(
            result["summary"]["best_budget_point"]["dataset"], "airport"
        )
        self.assertEqual(
            result["summary"]["worst_budget_point"]["dataset"], "wine"
        )
        wine_rows = result["datasets"]["wine"]["rows"]
        self.assertEqual(len(wine_rows), 5)
        self.assertEqual(
            wine_rows[0]["actual_utility_calls_by_repeat"]["normalized_greedy"],
            [100, 100, 100],
        )
        self.assertEqual(
            wine_rows[0][
                "normalized_to_legacy_slice_mean_direction_rms_ratio"
            ],
            1.0,
        )

    def test_rejects_wrong_dataset_identity(self) -> None:
        pairs, paths = _inputs()

        def wrong_dataset(*args, **kwargs):
            result = _comparison(*args, **kwargs)
            if result["validation"]["normalized"]["dataset"] == "wine":
                result["validation"]["normalized"]["dataset"] = "airport"
            return result

        with patch(
            "experiments.build_mean_balance_fix_summary.build_comparison",
            side_effect=wrong_dataset,
        ):
            with self.assertRaisesRegex(ValueError, "wrong dataset identity"):
                build_summary(pairs, report_paths=paths)

    def test_rejects_nonfinite_rmse_even_after_comparison(self) -> None:
        pairs, paths = _inputs()

        def nonfinite(*args, **kwargs):
            result = _comparison(*args, **kwargs)
            if result["validation"]["normalized"]["dataset"] == "wine":
                result["rows"][0]["normalized_lambda0_1_greedy_rmse"] = math.nan
            return result

        with patch(
            "experiments.build_mean_balance_fix_summary.build_comparison",
            side_effect=nonfinite,
        ):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                build_summary(pairs, report_paths=paths)

    def test_rejects_incomplete_repeat_call_spending(self) -> None:
        pairs, paths = _inputs()
        tampered = deepcopy(pairs)
        first = tampered["wine"][0]["results_by_inner_budget"]["0"]
        first["methods"]["inside_greedy"][
            "actual_utility_calls_by_repeat"
        ] = [99, 100, 100]
        with patch(
            "experiments.build_mean_balance_fix_summary.build_comparison",
            side_effect=_comparison,
        ):
            with self.assertRaisesRegex(ValueError, "spend the target"):
                build_summary(tampered, report_paths=paths)

    def test_requires_all_three_dataset_pairs(self) -> None:
        pairs, paths = _inputs()
        del pairs["voting"]
        with self.assertRaisesRegex(ValueError, "exactly wine, airport, and voting"):
            build_summary(pairs, report_paths=paths)


if __name__ == "__main__":
    unittest.main()

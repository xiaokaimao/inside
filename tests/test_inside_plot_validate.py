from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.compare_inside_mean_balance_reports import build_comparison
from experiments.plot_inside_comparison import (
    METHOD_ORDER,
    plot_results,
    validate_plot_pair,
)
from experiments.validate_inside_comparison import (
    CANONICAL_INSIDE_EXPERIMENT_IDS,
    CURRENT_CANONICAL_INSIDE_EXPERIMENT_IDS,
    CURRENT_K64_CANONICAL_INSIDE_EXPERIMENT_IDS,
    HISTORICAL_K64_CANONICAL_INSIDE_EXPERIMENT_IDS,
    LEGACY_CANONICAL_INSIDE_EXPERIMENT_IDS,
    _exact_airport_truth,
    _exact_weighted_voting_truth,
    validate_report,
)
from experiments.us_electoral_voting_game import (
    US_ELECTORAL_QUOTA,
    US_ELECTORAL_WEIGHTS,
)


def _airport_report() -> dict:
    truth = _exact_airport_truth()
    n = len(truth)
    inner = (100, 200, 300, 400, 500)
    total = tuple(value + 2 * n + 2 for value in inner)
    results = {}
    direction = np.linspace(-1.0, 1.0, n)
    direction -= direction.mean()
    for budget_index, (inner_calls, target) in enumerate(zip(inner, total)):
        summaries = {}
        for method_index, method in enumerate(METHOD_ORDER):
            scale = (method_index + 1) * 1e-4 / (budget_index + 1)
            estimates = np.asarray(
                [truth + scale * factor * direction for factor in (0.8, 1.0, 1.2)]
            )
            errors = estimates - truth[None, :]
            repeat_rmse = np.sqrt(np.mean(np.square(errors), axis=1))
            rmse = float(np.sqrt(np.mean(np.square(errors))))
            actual = target - 10 if method == "tmc_shapley" else target
            diagnostics = []
            for _ in range(3):
                diagnostic = {
                    "utility_evaluations": actual,
                    "target_call_budget": target,
                    "unused_calls": target - actual,
                }
                if method == "tmc_shapley":
                    diagnostic.update(
                        {
                            "boundary_utility_evaluations": 2,
                            "prefix_utility_evaluations": actual - 2,
                            "budget_remainder_calls": 4,
                            "truncation_saved_calls": 6,
                        }
                    )
                diagnostics.append(diagnostic)
            summaries[method] = {
                "estimates": estimates.tolist(),
                "aggregate_rmse": rmse,
                "aggregate_rmse_bootstrap_95": [
                    float(repeat_rmse.min()),
                    float(repeat_rmse.max()),
                ],
                "mean_repeat_rmse": float(repeat_rmse.mean()),
                "std_repeat_rmse": float(repeat_rmse.std(ddof=1)),
                "bias_l2": float(np.linalg.norm(estimates.mean(axis=0) - truth)),
                "mean_efficiency_residual": float(
                    np.mean(estimates.sum(axis=1) - 10.0)
                ),
                "max_absolute_efficiency_residual": float(
                    np.max(np.abs(estimates.sum(axis=1) - 10.0))
                ),
                "target_total_utility_calls": target,
                "actual_utility_calls_by_repeat": [actual] * 3,
                "mean_actual_utility_calls": float(actual),
                "diagnostics_by_repeat": diagnostics,
            }
        results[str(inner_calls)] = {
            "inner_utility_calls": inner_calls,
            "total_utility_calls_per_estimate": target,
            "methods": summaries,
        }
    return {
        "status": "complete",
        "experiment": "analytic_inside_baseline_comparison",
        "game": {
            "dataset": "airport",
            "players": n,
            "cost_class_counts": [8, 12, 6, 14, 8, 9, 13, 10, 10, 10],
        },
        "configuration": {
            "dataset": "airport",
            "methods": list(METHOD_ORDER),
            "repeats": 3,
            "inner_utility_call_budgets": list(inner),
            "total_call_budgets": list(total),
            "boundary_utility_calls_for_ofa_methods": 202,
            "inside": {"candidate_pool": 4, "greedy_mean_balance": 0.1},
        },
        "ground_truth": {
            "values": truth.tolist(),
            "sum": float(truth.sum()),
            "efficiency_error": abs(float(truth.sum()) - 10.0),
        },
        "results_by_inner_budget": results,
    }


def _canonical_airport_report() -> dict:
    report = _airport_report()
    report["experiment"] = "analytic_inside_baseline_comparison_per_size_ratio"
    self_ids = CANONICAL_INSIDE_EXPERIMENT_IDS
    if report["experiment"] not in self_ids:
        raise AssertionError("canonical test fixture uses an unknown experiment ID")
    report["configuration"]["inside"] = {
        "greedy": {
            "candidate_pool": 4,
            "mean_balance_mode": "normalized",
            "mean_balance_lambda0": 1.0,
            "second_moment_scope": "per_size",
            "estimator": "ofa_conditional_mean_ratio_missing_raise",
            "ratio_missing_policy": "raise",
        },
        "orbit": {
            "second_moment_scope": "per_size",
            "estimator": "ofa_conditional_mean_ratio_strict_balanced",
        },
    }
    for row in report["results_by_inner_budget"].values():
        greedy_diagnostics = row["methods"]["inside_greedy"][
            "diagnostics_by_repeat"
        ]
        orbit_diagnostics = row["methods"]["inside_orbit"][
            "diagnostics_by_repeat"
        ]
        for diagnostic in greedy_diagnostics:
            diagnostic.update(
                {
                    "design_method": "frame_coupled_per_size",
                    "second_moment_scope": "per_size",
                    "estimator": (
                        "ofa_conditional_mean_ratio_missing_raise"
                    ),
                    "official_ratio_missing_policy": "raise",
                    "coverage": {
                        "all_player_size_strata_covered": True,
                        "minimum_inclusion_count": 2,
                        "minimum_exclusion_count": 2,
                        "missing_inclusion_strata": 0,
                        "missing_exclusion_strata": 0,
                        "missing_inner_sizes": 0,
                    },
                    "design_diagnostics": {
                        "second_moment_scope": "per_size"
                    },
                }
            )
        for diagnostic in orbit_diagnostics:
            diagnostic.update(
                {
                    "design_method": "cyclic_orbit_frame",
                    "estimator": (
                        "ofa_conditional_mean_ratio_strict_balanced"
                    ),
                }
            )
    return report


def _current_canonical_airport_report(*, wine_id: bool = False) -> dict:
    report = _canonical_airport_report()
    report["experiment"] = (
        "wine_inside_greedy_orbit_baseline_comparison_per_size_ratio_"
        "k64_lambda0"
        if wine_id
        else "analytic_inside_baseline_comparison_per_size_ratio_k64_lambda0"
    )
    if report["experiment"] not in CURRENT_CANONICAL_INSIDE_EXPERIMENT_IDS:
        raise AssertionError("current canonical fixture uses an unknown ID")
    inside = report["configuration"]["inside"]
    inside["greedy"].update(
        {
            "design": "per_size_frame_coupled_design",
            "candidate_pool": 64,
            "mean_balance_lambda0": 0.0,
        }
    )
    inside["orbit"]["design"] = "cyclic_orbit_frame_design"
    if not wine_id:
        inside["orbit"]["candidate_pool"] = 4
        report["configuration"]["protocol_version"] = (
            "per_size_ratio_k64_lambda0"
        )
    for row in report["results_by_inner_budget"].values():
        diagnostics = row["methods"]["inside_greedy"][
            "diagnostics_by_repeat"
        ]
        for diagnostic in diagnostics:
            normalization_factor = 98.0 / 99.0
            diagnostic.update(
                {
                    "candidate_pool": 64,
                    "mean_balance_mode": "normalized",
                    "mean_balance_lambda0": 0.0,
                    "mean_balance_effective_raw": 0.0,
                }
            )
            diagnostic["design_diagnostics"].update(
                {
                    "candidate_pool": 64,
                    "mean_balance_mode": "normalized",
                    "mean_balance_lambda0": 0.0,
                    "mean_balance_mean_weight_squared": 1.0,
                    "mean_balance_dimension_correction": (
                        normalization_factor
                    ),
                    "mean_balance_normalization_factor": (
                        normalization_factor
                    ),
                    "mean_balance_effective_raw": 0.0,
                }
            )
        if wine_id:
            for diagnostic in row["methods"]["inside_orbit"][
                "diagnostics_by_repeat"
            ]:
                diagnostic.clear()
                diagnostic["reused_from_source_report"] = True
    return report


def _lambda_one_sixteenth_canonical_airport_report(
    *, wine_id: bool = False
) -> dict:
    report = _current_canonical_airport_report(wine_id=wine_id)
    report["experiment"] = (
        "wine_inside_greedy_orbit_baseline_comparison_per_size_ratio_"
        "k64_lambda1over16"
        if wine_id
        else (
            "analytic_inside_baseline_comparison_per_size_ratio_"
            "k64_lambda1over16"
        )
    )
    if report["experiment"] not in CURRENT_K64_CANONICAL_INSIDE_EXPERIMENT_IDS:
        raise AssertionError("lambda=1/16 fixture uses an unknown ID")
    report["configuration"]["inside"]["greedy"][
        "mean_balance_lambda0"
    ] = 1.0 / 16.0
    if not wine_id:
        report["configuration"]["protocol_version"] = (
            "per_size_ratio_k64_lambda1over16"
        )
    for row in report["results_by_inner_budget"].values():
        for diagnostic in row["methods"]["inside_greedy"][
            "diagnostics_by_repeat"
        ]:
            normalization_factor = 98.0 / 99.0
            effective = (1.0 / 16.0) * normalization_factor
            diagnostic["mean_balance_lambda0"] = 1.0 / 16.0
            diagnostic["mean_balance_effective_raw"] = effective
            diagnostic["design_diagnostics"]["mean_balance_lambda0"] = (
                1.0 / 16.0
            )
            diagnostic["design_diagnostics"].update(
                {
                    "mean_balance_mean_weight_squared": 1.0,
                    "mean_balance_dimension_correction": (
                        normalization_factor
                    ),
                    "mean_balance_normalization_factor": (
                        normalization_factor
                    ),
                    "mean_balance_effective_raw": effective,
                }
            )
    return report


class InsidePlotValidateTests(unittest.TestCase):
    def test_common_validator_recomputes_airport_report(self) -> None:
        validation = validate_report(_airport_report())
        self.assertEqual(validation["status"], "ready_to_share")
        self.assertEqual(validation["dataset"], "airport")
        self.assertEqual(
            validation["validated_shape"]["per_repeat_estimates_checked"],
            120,
        )

    def test_common_validator_accepts_explicit_normalized_lambda0(self) -> None:
        report = _airport_report()
        inside = report["configuration"]["inside"]
        inside.pop("greedy_mean_balance")
        inside["greedy_mean_balance_mode"] = "normalized"
        inside["greedy_mean_balance_lambda0"] = 1.0
        validation = validate_report(report)
        self.assertEqual(
            validation["inside_greedy_balance_audit"],
            {"candidate_pool": 4, "mode": "normalized", "coefficient": 1.0},
        )

    def test_canonical_validator_requires_the_fixed_inside_bundles(self) -> None:
        self.assertIn(
            "analytic_inside_baseline_comparison_per_size_ratio",
            LEGACY_CANONICAL_INSIDE_EXPERIMENT_IDS,
        )
        validation = validate_report(_canonical_airport_report())
        self.assertEqual(
            validation["checks"]["canonical_inside_algorithm_identity"],
            "passed",
        )

        mutations = (
            (
                "Greedy design",
                "design_method",
                "frame_coupled",
                "per-size Greedy design",
            ),
            (
                "Greedy scope",
                "second_moment_scope",
                "global_weighted",
                "per-size scope",
            ),
            (
                "Greedy estimator",
                "estimator",
                "coupled_linear",
                "covered OFA ratio",
            ),
        )
        for label, key, value, message in mutations:
            with self.subTest(label=label):
                report = _canonical_airport_report()
                first = report["results_by_inner_budget"]["100"]["methods"]
                first["inside_greedy"]["diagnostics_by_repeat"][0][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(report)

        report = _canonical_airport_report()
        first = report["results_by_inner_budget"]["100"]["methods"]
        first["inside_greedy"]["diagnostics_by_repeat"][0]["coverage"][
            "all_player_size_strata_covered"
        ] = False
        with self.assertRaisesRegex(ValueError, "complete player-size coverage"):
            validate_report(report)

        report = _canonical_airport_report()
        first = report["results_by_inner_budget"]["100"]["methods"]
        first["inside_orbit"]["diagnostics_by_repeat"][0]["estimator"] = (
            "coupled_linear"
        )
        with self.assertRaisesRegex(ValueError, "strict balanced OFA ratio"):
            validate_report(report)

    def test_historical_k64_lambda0_ids_still_validate_and_keep_orbit_k4(
        self,
    ) -> None:
        for wine_id in (False, True):
            with self.subTest(wine_id=wine_id):
                validation = validate_report(
                    _current_canonical_airport_report(wine_id=wine_id)
                )
                self.assertEqual(
                    validation["inside_greedy_balance_audit"],
                    {
                        "candidate_pool": 64,
                        "mode": "normalized",
                        "coefficient": 0.0,
                    },
                )
                self.assertEqual(
                    validation["checks"][
                        "inside_candidate_pool_matches_protocol"
                    ],
                    "passed",
                )

        report = _current_canonical_airport_report()
        report["configuration"]["inside"]["greedy"]["candidate_pool"] = 4
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            validate_report(report)

        report = _current_canonical_airport_report()
        report["configuration"]["inside"]["greedy"][
            "mean_balance_lambda0"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "lambda0"):
            validate_report(report)

        report = _current_canonical_airport_report()
        report["configuration"]["inside"]["orbit"]["candidate_pool"] = 64
        with self.assertRaisesRegex(ValueError, "Orbit candidate pool"):
            validate_report(report)

        report = _current_canonical_airport_report()
        report["configuration"]["protocol_version"] = "old_protocol"
        with self.assertRaisesRegex(ValueError, "protocol version"):
            validate_report(report)

        report = _lambda_one_sixteenth_canonical_airport_report()
        first = report["results_by_inner_budget"]["100"]["methods"]
        first["inside_greedy"]["diagnostics_by_repeat"][0][
            "mean_balance_effective_raw"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "effective_raw"):
            validate_report(report)

        report = _current_canonical_airport_report()
        first = report["results_by_inner_budget"]["100"]["methods"]
        first["inside_greedy"]["diagnostics_by_repeat"][0][
            "mean_balance_lambda0"
        ] = 1.0
        with self.assertRaisesRegex(ValueError, "diagnostic lambda0"):
            validate_report(report)

    def test_lambda_one_sixteenth_ids_are_versioned_and_strict(self) -> None:
        self.assertTrue(
            HISTORICAL_K64_CANONICAL_INSIDE_EXPERIMENT_IDS
            < CURRENT_CANONICAL_INSIDE_EXPERIMENT_IDS
        )
        for wine_id in (False, True):
            with self.subTest(wine_id=wine_id):
                validation = validate_report(
                    _lambda_one_sixteenth_canonical_airport_report(
                        wine_id=wine_id
                    )
                )
                self.assertEqual(
                    validation["inside_greedy_balance_audit"],
                    {
                        "candidate_pool": 64,
                        "mode": "normalized",
                        "coefficient": 1.0 / 16.0,
                    },
                )

        report = _lambda_one_sixteenth_canonical_airport_report()
        report["configuration"]["inside"]["greedy"][
            "mean_balance_lambda0"
        ] = 0.0
        with self.assertRaisesRegex(ValueError, "lambda0"):
            validate_report(report)

        report = _lambda_one_sixteenth_canonical_airport_report()
        report["configuration"]["protocol_version"] = (
            "per_size_ratio_k64_lambda0"
        )
        with self.assertRaisesRegex(ValueError, "protocol version"):
            validate_report(report)

    def test_canonical_validator_accepts_strict_flat_analytic_schema(self) -> None:
        report = _canonical_airport_report()
        report["configuration"]["inside"] = {
            "candidate_pool": 4,
            "greedy_mean_balance_mode": "normalized",
            "greedy_mean_balance_lambda0": 1.0,
            "greedy_design": "per_size_frame_coupled_design",
            "greedy_second_moment_scope": "per_size",
            "greedy_estimator": "OFA conditional-mean ratio",
            "greedy_ratio_missing_policy": "raise",
            "orbit_estimator": "OFA conditional-mean ratio",
        }
        validation = validate_report(report)
        self.assertEqual(
            validation["checks"]["canonical_inside_algorithm_identity"],
            "passed",
        )

        report["configuration"]["inside"]["greedy_design"] = (
            "global_weighted_frame_coupled_design"
        )
        with self.assertRaisesRegex(
            ValueError, "per_size_frame_coupled_design"
        ):
            validate_report(report)

    def test_common_validator_rejects_metric_tampering(self) -> None:
        report = _airport_report()
        first = report["results_by_inner_budget"]["100"]
        first["methods"]["inside_greedy"]["aggregate_rmse"] *= 2.0
        with self.assertRaisesRegex(ValueError, "aggregate_rmse differs"):
            validate_report(report)

    def test_common_validator_rejects_configuration_tampering(self) -> None:
        report = _airport_report()
        report["configuration"]["inside"]["candidate_pool"] = 64
        with self.assertRaisesRegex(ValueError, "candidate pool"):
            validate_report(report)

    def test_common_plot_is_single_square_png_and_pdf(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "airport.png"
            png, pdf = plot_results(_airport_report(), output)
            self.assertEqual(png, output)
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertEqual(pdf, output.with_suffix(".pdf"))
            self.assertGreater(pdf.stat().st_size, 1_000)

    def test_common_plot_can_overlay_legacy_raw_greedy(self) -> None:
        legacy = _airport_report()
        normalized = deepcopy(legacy)
        inside = normalized["configuration"]["inside"]
        inside.pop("greedy_mean_balance")
        inside["greedy_mean_balance_mode"] = "normalized"
        inside["greedy_mean_balance_lambda0"] = 1.0
        validate_plot_pair(normalized, legacy)
        with TemporaryDirectory() as directory:
            output = Path(directory) / "airport_before_after.png"
            png, pdf = plot_results(
                normalized, output, legacy_report=legacy
            )
            self.assertGreater(png.stat().st_size, 10_000)
            self.assertGreater(pdf.stat().st_size, 1_000)

    def test_before_after_summary_preserves_method_identity(self) -> None:
        legacy = _airport_report()
        normalized = deepcopy(legacy)
        inside = normalized["configuration"]["inside"]
        inside.pop("greedy_mean_balance")
        inside["greedy_mean_balance_mode"] = "normalized"
        inside["greedy_mean_balance_lambda0"] = 1.0
        summary = build_comparison(
            normalized,
            legacy,
            normalized_path=Path("normalized.json"),
            legacy_path=Path("legacy.json"),
        )
        self.assertEqual(summary["status"], "ready_to_share")
        self.assertEqual(
            summary["method_identity"]["canonical_label"],
            "INSIDE-Greedy",
        )
        self.assertEqual(len(summary["rows"]), 5)

    def test_independent_exact_voting_truth_is_efficient(self) -> None:
        values = _exact_weighted_voting_truth(
            US_ELECTORAL_WEIGHTS, US_ELECTORAL_QUOTA
        )
        self.assertEqual(values.shape, (51,))
        self.assertTrue(np.all(values > 0.0))
        self.assertAlmostEqual(float(values.sum()), 1.0, delta=2e-14)

    def test_plot_requires_tmc_actual_calls(self) -> None:
        report = deepcopy(_airport_report())
        summary = report["results_by_inner_budget"]["100"]["methods"]["tmc_shapley"]
        del summary["mean_actual_utility_calls"]
        del summary["actual_utility_calls_by_repeat"]
        with self.assertRaisesRegex(ValueError, "TMC-Shapley must record"):
            plot_results(report, Path("unused.png"))


if __name__ == "__main__":
    unittest.main()

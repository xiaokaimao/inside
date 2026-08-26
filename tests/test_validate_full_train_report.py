from __future__ import annotations

import copy
import unittest

import numpy as np

from experiments.validate_full_train_report import validate_report


def _summary(estimates: np.ndarray, truth: np.ndarray, target: float) -> dict:
    rmse = float(
        np.sqrt(np.mean(np.square(estimates - truth[None, :])))
    )
    gaps = np.abs(estimates.sum(axis=1) - target)
    return {
        "estimates": estimates.tolist(),
        "aggregate_rmse": rmse,
        "mean_efficiency_gap": float(gaps.mean()),
        "max_efficiency_gap": float(gaps.max()),
    }


def _valid_report() -> dict:
    num_players = 4
    repeats = 2
    pair_units = 10
    boundary_calls = 2 * num_players + 2
    multipliers = [2, 3, 4, 5, 6]
    budgets = [num_players * value for value in multipliers]
    truth = np.asarray([0.05, 0.10, 0.20, 0.25])
    target = float(truth.sum())
    iid_error = np.asarray(
        [[0.02, -0.01, -0.01, 0.0], [-0.01, 0.02, 0.0, -0.01]]
    )
    coupled_error = 0.5 * iid_error
    orbit_error = 0.25 * iid_error
    official_error = iid_error + 0.002
    arrays = {
        "official_ofa_fixed_ratio": truth + official_error,
        "iid_linear_ofa": truth + iid_error,
        "frame_coupled_linear": truth + coupled_error,
        "frame_orbit_ratio": truth + orbit_error,
    }
    methods = {
        method: _summary(estimates, truth, target)
        for method, estimates in arrays.items()
    }
    official_rmse = methods["official_ofa_fixed_ratio"][
        "aggregate_rmse"
    ]
    iid_rmse = methods["iid_linear_ofa"]["aggregate_rmse"]
    coupled_rmse = methods["frame_coupled_linear"]["aggregate_rmse"]
    orbit_rmse = methods["frame_orbit_ratio"]["aggregate_rmse"]
    results = {}
    for budget in budgets:
        results[str(budget)] = {
            "inner_utility_calls": budget,
            "boundary_utility_calls": boundary_calls,
            "total_utility_calls_per_estimate": budget + boundary_calls,
            "official_missing_stratum_fraction_mean": 0.0,
            "methods": copy.deepcopy(methods),
            "frame_coupled_rmse_reduction_vs_iid_linear": (
                iid_rmse - coupled_rmse
            )
            / iid_rmse,
            "orbit_rmse_reduction_vs_official_ratio": (
                official_rmse - orbit_rmse
            )
            / official_rmse,
        }
    standard_errors = np.asarray([0.01, 0.02, 0.01, 0.02])
    simultaneous_widths = 3.0 * standard_errors
    conceptual_calls = 2 * pair_units * (num_players - 1)
    physical_calls = 2 * pair_units * (num_players - 3)
    return {
        "status": "complete",
        "configuration": {
            "num_players": num_players,
            "repeats": repeats,
            "gt_pairs": pair_units,
            "jobs": 8,
            "inner_budgets": budgets,
            "budget_multipliers": multipliers,
            "total_call_budgets": [
                budget + boundary_calls for budget in budgets
            ],
        },
        "dataset": {"dataset": "synthetic"},
        "boundary": {
            "empty": 0.2,
            "full": 0.8,
            "efficiency_target": target,
            "utility_calls": boundary_calls,
        },
        "ground_truth": {
            "values": truth.tolist(),
            "standard_errors": standard_errors.tolist(),
            "rmse_standard_error": float(
                np.sqrt(np.mean(np.square(standard_errors)))
            ),
            "simultaneous_half_widths": (
                simultaneous_widths.tolist()
            ),
            "max_simultaneous_half_width": float(
                simultaneous_widths.max()
            ),
            "half_split_rmse": 0.01,
            "independent_pair_units": pair_units,
            "permutations": 2 * pair_units,
            "conceptual_internal_prefix_calls": conceptual_calls,
            "physical_internal_prefix_calls": physical_calls,
            "boundary_reuse_saved_calls": (
                conceptual_calls - physical_calls
            ),
            "permutation_path_equivalent_utility_calls": (
                2 + conceptual_calls
            ),
            "shared_boundary_calls_physically_evaluated": boundary_calls,
        },
        "results_by_inner_budget": results,
    }


def _with_official_basic_cc(report: dict) -> dict:
    report = copy.deepcopy(report)
    num_players = report["configuration"]["num_players"]
    repeats = report["configuration"]["repeats"]
    target = report["boundary"]["efficiency_target"]
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    report["configuration"]["cc_baseline"] = {
        "method_key": "official_cc_basic",
        "missing_stratum_rule": "zero (matches official cc_shap)",
    }
    for row in report["results_by_inner_budget"].values():
        total_calls = row["total_utility_calls_per_estimate"]
        estimates = np.repeat(truth[None, :], repeats, axis=0)
        estimates[:, 0] += 0.003
        cc = _summary(estimates, truth, target)
        missing = [repeat % 2 for repeat in range(repeats)]
        fractions = [value / (num_players**2) for value in missing]
        cc.update(
            {
                "utility_calls_per_estimate": total_calls,
                "complementary_contributions_per_estimate": (
                    total_calls // 2
                ),
                "missing_stratum_fractions": fractions,
                "mean_missing_stratum_fraction": float(
                    np.mean(fractions)
                ),
                "max_missing_stratum_fraction": float(
                    np.max(fractions)
                ),
                "cc_diagnostics_by_repeat": [
                    {
                        "num_pairs": total_calls // 2,
                        "utility_evaluations": total_calls,
                        "missing_strata": missing[repeat],
                        "minimum_positive_count": 1,
                        "num_tasks": 2,
                    }
                    for repeat in range(repeats)
                ],
            }
        )
        row["methods"]["official_cc_basic"] = cc
        row["cc_missing_stratum_fraction_mean"] = float(
            np.mean(fractions)
        )
        official_rmse = row["methods"]["official_ofa_fixed_ratio"][
            "aggregate_rmse"
        ]
        row["cc_rmse_reduction_vs_official_ratio"] = (
            official_rmse - cc["aggregate_rmse"]
        ) / official_rmse
    return report


def _with_new_baselines(report: dict) -> dict:
    report = copy.deepcopy(report)
    num_players = report["configuration"]["num_players"]
    repeats = report["configuration"]["repeats"]
    target = report["boundary"]["efficiency_target"]
    truth = np.asarray(report["ground_truth"]["values"], dtype=np.float64)
    new_methods = [
        "gels_shapley",
        "kernel_shap_sampled",
        "group_testing",
        "diff",
        "s_diff",
        "tmc_shapley",
        "stratified_marginal_mc",
    ]
    requested_num_tasks = 3
    report["configuration"]["baseline_comparison"] = {
        "new_methods": new_methods,
        "num_tasks": requested_num_tasks,
        "gels_shapley": {
            "paper_url": "https://openreview.net/forum?id=lvSMIsztka",
            "official_repository": "https://github.com/watml/fastpvalue",
            "verified_commit": (
                "34392c53f8d609aebb5e0c0e57c165411d291a46"
            ),
            "downloaded_source_sha256": (
                "2eff99580289e3f2374a72b720baa0284a2fff71341be47d5"
                "b757c585ff6e2fa"
            ),
            "official_class": "GELS_shapley",
            "paired_sampling": False,
        },
    }
    zero_sum_error = np.asarray([0.004, -0.003, -0.002, 0.001])
    unconstrained_error = np.asarray([0.004, 0.003, 0.002, 0.001])
    for row in report["results_by_inner_budget"].values():
        total_calls = row["total_utility_calls_per_estimate"]
        for method_index, method in enumerate(new_methods):
            error = (
                unconstrained_error
                if method in {"tmc_shapley", "stratified_marginal_mc"}
                else zero_sum_error
            )
            estimates = np.repeat(truth[None, :], repeats, axis=0)
            estimates += (method_index + 1) * error[None, :]
            summary = _summary(estimates, truth, target)
            # Exercise nonzero unused-call accounting as well as exact use.
            unused = [repeat % 2 for repeat in range(repeats)]
            actual = [total_calls - value for value in unused]
            repeat_diagnostics = []
            for repeat in range(repeats):
                entry = {
                    "num_players": num_players,
                    "target_call_budget": total_calls,
                    "utility_evaluations": actual[repeat],
                    "unused_calls": unused[repeat],
                    "num_tasks": 2,
                    "requested_num_tasks": requested_num_tasks,
                }
                if method == "gels_shapley":
                    inner = actual[repeat] - 2
                    entry.update(
                        {
                            "method": "gels_shapley",
                            "solver": (
                                "official_algorithm_3_"
                                "self_normalized_ratio"
                            ),
                            "boundary_utility_evaluations": 2,
                            "inner_utility_evaluations": inner,
                            "num_samples": inner,
                            "layer_counts": [inner, 0, 0],
                            "inclusion_counts": [inner, 0, 0, 0],
                            "zero_inclusion_players": 3,
                        }
                    )
                repeat_diagnostics.append(entry)
            summary.update(
                {
                    "target_total_call_budget": total_calls,
                    "actual_utility_calls_per_estimate": actual,
                    "unused_calls_per_estimate": unused,
                    "mean_actual_utility_calls": float(np.mean(actual)),
                    "mean_unused_calls": float(np.mean(unused)),
                    "diagnostics_by_repeat": repeat_diagnostics,
                }
            )
            row["methods"][method] = summary
    return report


class FullTrainReportValidatorTests(unittest.TestCase):
    def test_valid_report_recomputes_metrics_and_call_accounting(self) -> None:
        report = _valid_report()
        result = validate_report(
            report,
            expected_dataset="synthetic",
            expected_num_players=4,
            expected_repeats=2,
            expected_gt_pairs=10,
            expected_jobs=8,
            expected_inner_budgets=[8, 12, 16, 20, 24],
            expected_budget_multipliers=[2, 3, 4, 5, 6],
        )

        self.assertTrue(result["passed"], result["errors"])
        diagnostics = result["diagnostics"]
        self.assertEqual(
            diagnostics["ground_truth"]["physical_total_calls"], 30
        )
        self.assertEqual(
            diagnostics["ground_truth"][
                "permutation_path_equivalent_calls"
            ],
            62,
        )
        self.assertEqual(
            diagnostics["results_by_inner_budget"]["8"]["total_calls"],
            18,
        )

    def test_corrupt_metrics_efficiency_and_totals_are_reported(self) -> None:
        report = _valid_report()
        first = report["results_by_inner_budget"]["8"]
        first["total_utility_calls_per_estimate"] += 1
        first["methods"]["iid_linear_ofa"]["aggregate_rmse"] *= 2
        first["methods"]["frame_coupled_linear"]["estimates"][0][0] += 0.1
        first["orbit_rmse_reduction_vs_official_ratio"] = -12.0

        result = validate_report(report)

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn("total_utility_calls_per_estimate", errors)
        self.assertIn("iid_linear_ofa.aggregate_rmse", errors)
        self.assertIn("frame_coupled_linear.estimates: efficiency gap", errors)
        self.assertIn("orbit_rmse_reduction_vs_official_ratio", errors)

    def test_nonfinite_gt_calls_and_status_are_reported(self) -> None:
        report = _valid_report()
        report["status"] = "running"
        report["ground_truth"]["physical_internal_prefix_calls"] += 1
        report["ground_truth"]["values"][0] = float("nan")

        result = validate_report(report)

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn("status", errors)
        self.assertIn("physical_internal_prefix_calls", errors)
        self.assertIn("ground_truth.values: contains non-finite", errors)

    def test_bad_and_nonfinite_estimate_shapes_are_reported(self) -> None:
        report = _valid_report()
        methods = report["results_by_inner_budget"]["12"]["methods"]
        del methods["frame_orbit_ratio"]["estimates"][0]
        methods["iid_linear_ofa"]["estimates"][0][0] = float("inf")

        result = validate_report(report)

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn(
            "frame_orbit_ratio.estimates: expected shape (2, 4)", errors
        )
        self.assertIn(
            "iid_linear_ofa.estimates: contains non-finite", errors
        )

    def test_official_missing_fraction_is_bounded_but_gap_is_allowed(self) -> None:
        report = _valid_report()
        # The official fixed-ratio estimator need not obey efficiency when
        # sampled conditional-mean strata are missing.
        official = report["results_by_inner_budget"]["8"]["methods"][
            "official_ofa_fixed_ratio"
        ]
        self.assertGreater(official["max_efficiency_gap"], 0.0)
        self.assertTrue(validate_report(report)["passed"])

        report["results_by_inner_budget"]["8"][
            "official_missing_stratum_fraction_mean"
        ] = 1.1
        result = validate_report(report)
        self.assertFalse(result["passed"])
        self.assertIn(
            "official_missing_stratum_fraction_mean",
            "\n".join(result["errors"]),
        )

    def test_expected_protocol_arguments_are_enforced(self) -> None:
        result = validate_report(
            _valid_report(),
            expected_dataset="cancer",
            expected_num_players=455,
            expected_repeats=10,
            expected_gt_pairs=800_000,
            expected_jobs=128,
        )

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn("dataset.dataset", errors)
        self.assertIn("configuration.num_players", errors)
        self.assertIn("configuration.repeats", errors)
        self.assertIn("configuration.gt_pairs", errors)
        self.assertIn("configuration.jobs", errors)

    def test_optional_basic_cc_metrics_and_call_accounting_pass(self) -> None:
        result = validate_report(_with_official_basic_cc(_valid_report()))

        self.assertTrue(result["passed"], result["errors"])
        self.assertTrue(
            result["diagnostics"]["has_official_basic_cc"]
        )
        cc = result["diagnostics"]["results_by_inner_budget"]["8"][
            "methods"
        ]["official_cc_basic"]
        self.assertEqual(cc["max_missing_stratum_fraction"], 1 / 16)

    def test_corrupt_basic_cc_accounting_is_reported(self) -> None:
        report = _with_official_basic_cc(_valid_report())
        cc = report["results_by_inner_budget"]["8"]["methods"][
            "official_cc_basic"
        ]
        cc["utility_calls_per_estimate"] += 2
        cc["cc_diagnostics_by_repeat"][0]["missing_strata"] = 3

        result = validate_report(report)

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn("utility_calls_per_estimate", errors)
        self.assertIn("missing_strata_fraction", errors)

    def test_new_baselines_metrics_calls_and_diagnostics_pass(self) -> None:
        report = _with_new_baselines(_valid_report())

        result = validate_report(report)

        self.assertTrue(result["passed"], result["errors"])
        diagnostics = result["diagnostics"]
        self.assertTrue(diagnostics["has_baseline_comparison"])
        self.assertEqual(
            diagnostics["new_methods"],
            report["configuration"]["baseline_comparison"]["new_methods"],
        )
        method = diagnostics["results_by_inner_budget"]["8"]["methods"][
            "tmc_shapley"
        ]
        self.assertEqual(
            method["actual_utility_calls_per_estimate"], [18, 17]
        )
        # Both methods intentionally violate efficiency in the fixture; they
        # remain admissible because neither applies an efficiency projection.
        for unconstrained in ("tmc_shapley", "stratified_marginal_mc"):
            self.assertGreater(
                report["results_by_inner_budget"]["8"]["methods"][
                    unconstrained
                ]["max_efficiency_gap"],
                0.0,
            )

    def test_corrupt_new_baseline_is_strictly_reported(self) -> None:
        report = _with_new_baselines(_valid_report())
        first = report["results_by_inner_budget"]["8"]
        gels = first["methods"]["gels_shapley"]
        gels["aggregate_rmse"] *= 2
        gels["target_total_call_budget"] += 1
        gels["actual_utility_calls_per_estimate"][0] -= 1
        gels["mean_unused_calls"] += 1
        gels["diagnostics_by_repeat"][1]["utility_evaluations"] -= 1
        # GELS is efficiency constrained, so changing one coordinate must be
        # detected independently of the aggregate-RMSE corruption above.
        gels["estimates"][0][0] += 0.1
        del report["results_by_inner_budget"]["12"]["methods"]["diff"]

        result = validate_report(report)

        self.assertFalse(result["passed"])
        errors = "\n".join(result["errors"])
        self.assertIn("gels_shapley.aggregate_rmse", errors)
        self.assertIn("gels_shapley.target_total_call_budget", errors)
        self.assertIn("gels_shapley.call_accounting.0", errors)
        self.assertIn("gels_shapley.mean_unused_calls", errors)
        self.assertIn(
            "gels_shapley.diagnostics_by_repeat.1.utility_evaluations",
            errors,
        )
        self.assertIn("gels_shapley.estimates: efficiency gap", errors)
        self.assertIn("missing configured new methods ['diff']", errors)


if __name__ == "__main__":
    unittest.main()

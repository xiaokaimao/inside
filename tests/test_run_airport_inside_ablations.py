from __future__ import annotations

import copy
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.run_airport_inside_ablations import (
    BOUNDARY_CALLS,
    FULL_LAMBDA0,
    METHOD_LABELS,
    METHOD_ORDER,
    _aggregate,
    _cell_seeds,
    _configuration,
    _correlation,
    _run_cell,
    _sha256_array,
    validate_report,
)


class AirportInsideAblationRunnerTests(unittest.TestCase):
    def test_correlation_reports_undefined_without_inventing_epsilon(self) -> None:
        result = _correlation(
            np.asarray([0.0, 1.0]), np.asarray([1.0, 2.0])
        )
        self.assertFalse(result["defined"])
        self.assertIsNone(result["pearson_log10"])
        self.assertIsNone(result["spearman"])

    def test_cell_seed_partition_is_deterministic_and_locked(self) -> None:
        expected = {
            "iid": 651498517,
            "component": 3180925944,
            "component_relabel": 1428164462,
            "orbit": 644533757,
        }
        first = _cell_seeds(20260827, 0, 0)
        self.assertEqual(first, expected)
        self.assertEqual(_cell_seeds(20260827, 0, 0), expected)
        self.assertEqual(len(set(first.values())), len(first))
        self.assertNotEqual(first, _cell_seeds(20260827, 1, 0))
        self.assertNotEqual(first, _cell_seeds(20260827, 0, 1))

    def test_configuration_locks_airport_budget_and_ablation_contract(self) -> None:
        configuration = _configuration(
            budget_multipliers=(500, 1000),
            repeats=3,
            base_seed=20260827,
            bootstrap_samples=101,
            repeat_processes=3,
            design_jobs=7,
            design_start_method="spawn",
            greedy_candidate_pool=64,
            orbit_candidate_pool=4,
        )

        self.assertEqual(configuration["dataset"], "airport")
        self.assertEqual(configuration["players"], 100)
        self.assertEqual(
            configuration["inner_utility_call_budgets"],
            [50_000, 100_000],
        )
        self.assertEqual(
            configuration["total_utility_call_budgets"],
            [50_000 + BOUNDARY_CALLS, 100_000 + BOUNDARY_CALLS],
        )
        self.assertEqual(configuration["boundary_utility_calls"], 202)
        self.assertEqual(configuration["methods"], list(METHOD_ORDER))
        self.assertEqual(configuration["method_labels"], METHOD_LABELS)
        self.assertEqual(configuration["greedy_candidate_pool"], 64)
        self.assertEqual(configuration["orbit_candidate_pool"], 4)
        self.assertEqual(configuration["repeat_processes"], 3)
        self.assertEqual(configuration["full_lambda0"], FULL_LAMBDA0)
        self.assertAlmostEqual(
            configuration["full_effective_lambda"],
            FULL_LAMBDA0 * 98.0 / 99.0,
            places=15,
        )
        self.assertIn("missing=raise", configuration["strict_common_estimator"])
        self.assertIn("equal-size macro RMS", configuration["geometry_aggregation"])
        self.assertIn("IID size draws", configuration["iid_interpretation_caveat"])

    @staticmethod
    def _fake_design(
        method: str,
        sizes: tuple[int, ...],
        *,
        relabel_hash: str | None = None,
        candidate_hash: str | None = None,
        orbit_relabel_hash: str | None = None,
    ) -> SimpleNamespace:
        diagnostics: dict[str, object] = {}
        if relabel_hash is not None:
            diagnostics["relabel_permutation_sha256"] = relabel_hash
        if candidate_hash is not None:
            diagnostics["shared_candidate_pool_sha256"] = candidate_hash
        if orbit_relabel_hash is not None:
            diagnostics["shared_relabel_permutations_sha256"] = (
                orbit_relabel_hash
            )
        size_array = np.asarray(sizes, dtype=np.int64)
        return SimpleNamespace(
            coalitions=np.zeros((len(size_array), 100), dtype=bool),
            sizes=size_array,
            method=method,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _fake_evaluation(method: str, design: SimpleNamespace, **_: object):
        return {
            "method": method,
            "label": METHOD_LABELS[method],
            "size_schedule_sha256": _sha256_array(design.sizes),
        }

    def test_run_cell_passes_paired_component_seeds_and_objectives(self) -> None:
        iid_design = self._fake_design("iid", (2, 3, 2))

        def component_design(*_: object, **kwargs: object) -> SimpleNamespace:
            self.assertEqual(kwargs["candidate_pool"], 64)
            return self._fake_design(
                "frame_coupled_per_size",
                (2, 3, 2, 3),
                relabel_hash="component-relabel",
            )

        random_orbit = self._fake_design(
            "random_cyclic_orbit",
            (2, 2, 3, 3),
            candidate_hash="candidate-stream",
            orbit_relabel_hash="orbit-relabel",
        )
        inside_orbit = self._fake_design(
            "cyclic_orbit_frame",
            (2, 2, 3, 3),
            candidate_hash="candidate-stream",
            orbit_relabel_hash="orbit-relabel",
        )

        with (
            patch(
                "experiments.run_airport_inside_ablations.iid_ofa_design",
                return_value=iid_design,
            ) as iid_builder,
            patch(
                "experiments.run_airport_inside_ablations."
                "per_size_frame_coupled_design",
                side_effect=component_design,
            ) as component_builder,
            patch(
                "experiments.run_airport_inside_ablations."
                "paired_cyclic_orbit_designs",
                return_value=(random_orbit, inside_orbit),
            ) as orbit_builder,
            patch(
                "experiments.run_airport_inside_ablations._evaluate_design",
                side_effect=self._fake_evaluation,
            ),
        ):
            cell = _run_cell(
                1,
                2,
                50_000,
                base_seed=20260827,
                greedy_candidate_pool=64,
                orbit_candidate_pool=4,
                design_jobs=7,
                design_start_method="spawn",
            )

        seeds = _cell_seeds(20260827, 1, 2)
        self.assertEqual(iid_builder.call_args.kwargs["seed"], seeds["iid"])
        calls = component_builder.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [call.kwargs["seed"] for call in calls],
            [seeds["component"]] * 3,
        )
        self.assertEqual(
            [call.kwargs["relabel_seed"] for call in calls],
            [seeds["component_relabel"]] * 3,
        )
        self.assertEqual(
            [call.kwargs["mean_balance"] for call in calls],
            [1.0, 0.0, FULL_LAMBDA0],
        )
        self.assertEqual(
            [call.kwargs["second_moment_weight"] for call in calls],
            [0.0, 1.0, 1.0],
        )
        self.assertEqual(
            [call.kwargs["design_jobs"] for call in calls], [7, 7, 7]
        )
        self.assertEqual(orbit_builder.call_args.kwargs["seed"], seeds["orbit"])
        self.assertEqual(orbit_builder.call_args.kwargs["candidate_pool"], 4)
        self.assertEqual(set(cell["methods"]), set(METHOD_ORDER))
        self.assertEqual(
            cell["paired_controls"][
                "component_shared_relabel_sha256"
            ],
            "component-relabel",
        )
        self.assertEqual(
            cell["paired_controls"][
                "orbit_shared_candidate_pool_sha256"
            ],
            "candidate-stream",
        )

    def test_run_cell_rejects_unpaired_orbit_metadata(self) -> None:
        iid_design = self._fake_design("iid", (2, 3))
        component_design = self._fake_design(
            "frame_coupled_per_size",
            (2, 3),
            relabel_hash="component-relabel",
        )
        random_orbit = self._fake_design(
            "random_cyclic_orbit",
            (2, 2, 3, 3),
            candidate_hash="random-pool",
            orbit_relabel_hash="same-relabel",
        )
        inside_orbit = self._fake_design(
            "cyclic_orbit_frame",
            (2, 2, 3, 3),
            candidate_hash="greedy-pool",
            orbit_relabel_hash="same-relabel",
        )
        with (
            patch(
                "experiments.run_airport_inside_ablations.iid_ofa_design",
                return_value=iid_design,
            ),
            patch(
                "experiments.run_airport_inside_ablations."
                "per_size_frame_coupled_design",
                return_value=component_design,
            ),
            patch(
                "experiments.run_airport_inside_ablations."
                "paired_cyclic_orbit_designs",
                return_value=(random_orbit, inside_orbit),
            ),
            patch(
                "experiments.run_airport_inside_ablations._evaluate_design",
                side_effect=self._fake_evaluation,
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "paired orbit controls are not shared"
            ):
                _run_cell(
                    0,
                    0,
                    50_000,
                    base_seed=20260827,
                    greedy_candidate_pool=64,
                    orbit_candidate_pool=4,
                    design_jobs=1,
                    design_start_method="spawn",
                )

    @staticmethod
    def _aggregate_cells(
        *, repeats: int = 3, inner_budgets: tuple[int, ...] = (50_000, 100_000)
    ) -> list[dict[str, object]]:
        rmse_scale = {
            "iid_ofa": 1.40,
            "first_only": 1.20,
            "frame_only": 1.10,
            "full_inside": 1.00,
            "random_orbit": 1.30,
            "inside_orbit": 0.90,
        }
        cells: list[dict[str, object]] = []
        for repeat in range(repeats):
            for budget_index, inner_calls in enumerate(inner_budgets):
                base = 1.0 + 0.08 * repeat + 0.15 * budget_index
                methods: dict[str, object] = {}
                for method_index, method in enumerate(METHOD_ORDER):
                    rmse = rmse_scale[method] * base
                    methods[method] = {
                        "estimate": [rmse, -rmse],
                        "repeat_rmse": rmse,
                        "geometry": {
                            "first_moment_rms": (
                                0.01
                                + 0.002 * repeat
                                + 0.003 * budget_index
                                + 0.0005 * method_index
                            ),
                            "frame_frobenius_rms": (
                                0.04 * rmse + 0.0001 * method_index
                            ),
                        },
                    }
                cells.append(
                    {
                        "repeat": repeat,
                        "budget_index": budget_index,
                        "inner_utility_calls": inner_calls,
                        "total_utility_calls": inner_calls + BOUNDARY_CALLS,
                        "methods": methods,
                    }
                )
        return cells

    def test_aggregate_computes_pooled_rmse_and_paired_reductions(self) -> None:
        inner_budgets = (50_000, 100_000)
        cells = self._aggregate_cells(inner_budgets=inner_budgets)
        results, relation = _aggregate(
            cells,
            inner_budgets=inner_budgets,
            repeats=3,
            bootstrap_samples=101,
        )

        repeat_values = np.asarray([1.0, 1.08, 1.16])
        expected_full = float(np.sqrt(np.mean(np.square(repeat_values))))
        first = results["50000"]
        self.assertAlmostEqual(
            first["methods"]["full_inside"]["aggregate_rmse"],
            expected_full,
            places=14,
        )
        self.assertAlmostEqual(
            first["component_contributions"][
                "full_vs_first_only_rmse_reduction"
            ]["percent"],
            100.0 * (1.0 - 1.0 / 1.2),
            places=12,
        )
        self.assertAlmostEqual(
            first["component_contributions"][
                "full_vs_frame_only_rmse_reduction"
            ]["percent"],
            100.0 * (1.0 - 1.0 / 1.1),
            places=12,
        )
        self.assertAlmostEqual(
            first["orbit_frame_selection"][
                "inside_vs_random_rmse_reduction"
            ]["percent"],
            100.0 * (1.0 - 0.9 / 1.3),
            places=12,
        )
        self.assertEqual(
            len(relation["points"]),
            len(cells) * 3,
        )
        correlations = relation["correlations"]
        self.assertTrue(math.isfinite(correlations["overall"]["spearman"]))
        self.assertEqual(set(correlations["by_method"]), {
            "iid_ofa", "full_inside", "inside_orbit"
        })
        self.assertTrue(
            math.isfinite(
                correlations["within_method_budget_log_residual"][
                    "pearson"
                ]
            )
        )

    @staticmethod
    def _valid_fake_report() -> dict[str, object]:
        truth = np.asarray([0.25, -0.25], dtype=np.float64)
        total_calls = 50_000 + BOUNDARY_CALLS
        raw_cells: list[dict[str, object]] = []
        for repeat in range(2):
            methods: dict[str, object] = {}
            for method_index, method in enumerate(METHOD_ORDER):
                estimate = truth + np.asarray(
                    [0.01 + 0.001 * repeat, -0.002 * method_index]
                )
                rmse = float(
                    np.sqrt(np.mean(np.square(estimate - truth)))
                )
                if method in {"random_orbit", "inside_orbit"}:
                    design_diagnostics = {
                        "shared_candidate_pool_sha256": "orbit-pool",
                        "shared_relabel_permutations_sha256": "orbit-relabel",
                    }
                elif method in {"first_only", "frame_only", "full_inside"}:
                    objectives = {
                        "first_only": ["first_moment"],
                        "frame_only": ["second_moment"],
                        "full_inside": ["first_moment", "second_moment"],
                    }[method]
                    second_weight = 0.0 if method == "first_only" else 1.0
                    design_diagnostics = {
                        "relabel_permutation_sha256": "component-relabel",
                        "second_moment_scope": "per_size",
                        "objective_components": objectives,
                        "second_moment_weight": second_weight,
                    }
                else:
                    design_diagnostics = {}
                schedule_hash = (
                    "component-schedule"
                    if method in {"first_only", "frame_only", "full_inside"}
                    else (
                        "orbit-schedule"
                        if method in {"random_orbit", "inside_orbit"}
                        else f"{method}-schedule"
                    )
                )
                methods[method] = {
                    "estimate": estimate.tolist(),
                    "repeat_rmse": rmse,
                    "actual_utility_calls": total_calls,
                    "size_schedule_sha256": schedule_hash,
                    "coverage": {
                        "all_player_size_strata_covered": True,
                    },
                    "geometry": {
                        "all_inner_sizes_present": True,
                        "observed_inner_size_count": 97,
                        "missing_sizes": [],
                    },
                    "design_diagnostics": design_diagnostics,
                }
            raw_cells.append(
                {
                    "repeat": repeat,
                    "budget_index": 0,
                    "inner_utility_calls": 50_000,
                    "total_utility_calls": total_calls,
                    "paired_controls": {
                        "component_shared_size_schedule_sha256": (
                            "component-schedule"
                        ),
                        "component_shared_relabel_sha256": (
                            "component-relabel"
                        ),
                        "orbit_shared_candidate_pool_sha256": "orbit-pool",
                        "orbit_shared_relabel_sha256": "orbit-relabel",
                    },
                    "methods": methods,
                }
            )
        return {
            "status": "complete",
            "configuration": {
                "methods": list(METHOD_ORDER),
                "repeats": 2,
                "inner_utility_call_budgets": [50_000],
            },
            "ground_truth": {"values": truth.tolist()},
            "raw_cells": raw_cells,
        }

    def test_validate_report_accepts_complete_fake_grid(self) -> None:
        validate_report(self._valid_fake_report())

    def test_validate_report_rejects_component_schedule_mismatch(self) -> None:
        report = self._valid_fake_report()
        report["raw_cells"][0]["methods"]["frame_only"][
            "size_schedule_sha256"
        ] = "wrong-schedule"
        with self.assertRaisesRegex(ValueError, "component schedule pairing"):
            validate_report(report)

    def test_validate_report_rejects_orbit_candidate_mismatch(self) -> None:
        report = self._valid_fake_report()
        report["raw_cells"][0]["methods"]["inside_orbit"][
            "design_diagnostics"
        ]["shared_candidate_pool_sha256"] = "wrong-pool"
        with self.assertRaisesRegex(ValueError, "orbit candidate pairing"):
            validate_report(report)

    def test_validate_report_rejects_core_metric_and_accounting_errors(self) -> None:
        mutations = (
            (
                "RMSE",
                lambda report: report["raw_cells"][0]["methods"][
                    "full_inside"
                ].__setitem__("repeat_rmse", 999.0),
                "RMSE is inconsistent",
            ),
            (
                "coverage",
                lambda report: report["raw_cells"][0]["methods"][
                    "iid_ofa"
                ]["coverage"].__setitem__(
                    "all_player_size_strata_covered", False
                ),
                "lacks ratio coverage",
            ),
            (
                "geometry",
                lambda report: report["raw_cells"][0]["methods"][
                    "full_inside"
                ]["geometry"].__setitem__("missing_sizes", [2]),
                "geometry is incomplete",
            ),
            (
                "calls",
                lambda report: report["raw_cells"][0]["methods"][
                    "inside_orbit"
                ].__setitem__("actual_utility_calls", 49_999),
                "call accounting is invalid",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label):
                report = copy.deepcopy(self._valid_fake_report())
                mutate(report)
                with self.assertRaisesRegex(ValueError, message):
                    validate_report(report)


if __name__ == "__main__":
    unittest.main()

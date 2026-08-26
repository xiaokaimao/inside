from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.run_analytic_inside_comparison import (
    LEGACY_GLOBAL_LINEAR_EXPERIMENT_ID,
    METHOD_ORDER as SOURCE_METHOD_ORDER,
    _exact_truth,
    total_call_budgets,
)
from experiments.run_second_moment_scope_comparison import (
    METHOD_ORDER,
    _run_greedy_pair_cell,
    _shared_greedy_seed,
    run_experiment,
)


class SecondMomentScopeRunnerTests(unittest.TestCase):
    def test_two_scopes_share_the_historical_seed_and_size_schedule(self) -> None:
        rows = _run_greedy_pair_cell(
            "airport", 0, 0, 100, 302, 20260827
        )
        self.assertEqual(rows[0]["seed"], rows[1]["seed"])
        self.assertEqual(
            rows[0]["seed"], _shared_greedy_seed(20260827, 0, 0)
        )
        self.assertEqual(
            rows[0]["diagnostics"]["size_schedule_sha256"],
            rows[1]["diagnostics"]["size_schedule_sha256"],
        )
        self.assertEqual(
            rows[0]["diagnostics"]["relabel_permutation_sha256"],
            rows[1]["diagnostics"]["relabel_permutation_sha256"],
        )
        self.assertEqual(
            rows[0]["diagnostics"]["second_moment_scope"],
            "global_weighted",
        )
        self.assertEqual(
            rows[1]["diagnostics"]["second_moment_scope"], "per_size"
        )

    def test_small_audit_report_reuses_seven_methods_and_resumes(self) -> None:
        dataset = "airport"
        multipliers = (1, 2)
        repeats = 2
        base_seed = 20260827
        inner_budgets, total_budgets = total_call_budgets(
            dataset, multipliers
        )
        truth, _ = _exact_truth(dataset)

        source_tasks = []
        for repeat in range(repeats):
            for method_index, method in enumerate(SOURCE_METHOD_ORDER):
                rows = []
                for budget_index, (inner_calls, total_calls) in enumerate(
                    zip(inner_budgets, total_budgets, strict=True)
                ):
                    diagnostics = {"utility_evaluations": total_calls}
                    if method == "inside_greedy":
                        diagnostics.update(
                            {
                                "design_method": "frame_coupled",
                                "mean_balance_mode": "normalized",
                                "mean_balance_lambda0": 1.0,
                                "design_diagnostics": {
                                    "mean_balance_mean_weight_squared": 2.0
                                },
                            }
                        )
                    rows.append(
                        {
                            "repeat": repeat,
                            "method": method,
                            "seed": (
                                _shared_greedy_seed(
                                    base_seed, repeat, budget_index
                                )
                                if method == "inside_greedy"
                                else method_index * 100 + budget_index
                            ),
                            "inner_utility_call_budget": inner_calls,
                            "target_total_utility_calls": total_calls,
                            "actual_utility_calls": total_calls,
                            "estimate": truth.tolist(),
                            "elapsed_seconds": 0.0,
                            "diagnostics": diagnostics,
                        }
                    )
                source_tasks.append(
                    {"repeat": repeat, "method": method, "rows": rows}
                )

        source = {
            "status": "complete",
            "experiment": LEGACY_GLOBAL_LINEAR_EXPERIMENT_ID,
            "configuration": {
                "dataset": dataset,
                "budget_multipliers": list(multipliers),
                "inner_utility_call_budgets": list(inner_budgets),
                "total_call_budgets": list(total_budgets),
                "repeats": repeats,
                "base_seed": base_seed,
                "num_tasks_per_serial_baseline": 4,
                "methods": list(SOURCE_METHOD_ORDER),
                "inside": {
                    "candidate_pool": 4,
                    "greedy_mean_balance_mode": "normalized",
                    "greedy_mean_balance_lambda0": 1.0,
                    "greedy_estimator": "coupled linear",
                },
            },
            "ground_truth": {"values": truth.tolist()},
            "raw_tasks": source_tasks,
        }

        def fake_pair(
            dataset_value: str,
            repeat: int,
            budget_index: int,
            inner_calls: int,
            total_calls: int,
            seed_value: int,
        ) -> tuple[dict[str, object], dict[str, object]]:
            self.assertEqual(dataset_value, dataset)
            seed = _shared_greedy_seed(
                seed_value, repeat, budget_index
            )
            rows = []
            for method, scope in (
                ("inside_greedy_global", "global_weighted"),
                ("inside_greedy_per_size", "per_size"),
            ):
                rows.append(
                    {
                        "repeat": repeat,
                        "method": method,
                        "budget_index": budget_index,
                        "seed": seed,
                        "inner_utility_call_budget": inner_calls,
                        "target_total_utility_calls": total_calls,
                        "actual_utility_calls": total_calls,
                        "estimate": truth.tolist(),
                        "elapsed_seconds": 0.0,
                        "diagnostics": {
                            "second_moment_scope": scope,
                            "size_schedule_sha256": (
                                f"same-{repeat}-{budget_index}"
                            ),
                            "relabel_permutation_sha256": (
                                f"relabel-{repeat}-{budget_index}"
                            ),
                        },
                    }
                )
            return rows[0], rows[1]

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "output.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with patch(
                "experiments.run_second_moment_scope_comparison."
                "_run_greedy_pair_cell",
                side_effect=fake_pair,
            ):
                report = run_experiment(
                    dataset=dataset,
                    source_report=source_path,
                    budget_multipliers=multipliers,
                    repeats=repeats,
                    processes=1,
                    base_seed=base_seed,
                    bootstrap_samples=16,
                    num_tasks=4,
                    output_path=output_path,
                    resume=False,
                )
            self.assertEqual(report["status"], "complete")
            self.assertEqual(report["configuration"]["methods"], list(METHOD_ORDER))
            self.assertEqual(
                len(report["raw_tasks"]),
                repeats * len(METHOD_ORDER) * len(multipliers),
            )
            self.assertTrue(report["validation"]["passed"])
            self.assertTrue(
                report["validation"][
                    "global_vs_per_size_size_schedules_match"
                ]
            )
            self.assertEqual(
                report["validation"][
                    "historical_global_numerical_comparison"
                ][
                    "maximum_absolute_estimate_difference"
                ],
                0.0,
            )
            methods = next(
                iter(report["results_by_inner_budget"].values())
            )["methods"]
            self.assertEqual(set(methods), set(METHOD_ORDER))

            # A complete cell-granular checkpoint must resume without asking
            # the expensive worker for any cell again.
            with patch(
                "experiments.run_second_moment_scope_comparison."
                "_run_greedy_pair_cell",
                side_effect=AssertionError("worker should not run"),
            ):
                resumed = run_experiment(
                    dataset=dataset,
                    source_report=source_path,
                    budget_multipliers=multipliers,
                    repeats=repeats,
                    processes=2,
                    base_seed=base_seed,
                    bootstrap_samples=16,
                    num_tasks=4,
                    output_path=output_path,
                    resume=True,
                )
            self.assertEqual(resumed["status"], "complete")


if __name__ == "__main__":
    unittest.main()

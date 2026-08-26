from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.run_analytic_inside_comparison import (
    _exact_truth,
    total_call_budgets,
)
from experiments.run_per_size_ratio_comparison import (
    METHOD_ORDER,
    RATIO_METHOD,
    SOURCE_METHOD_ORDER,
    _run_ratio_cell,
    _shared_greedy_seed,
    _shared_relabel_seed,
    _size_schedule_sha256,
    run_experiment,
)
from experiments.run_second_moment_scope_comparison import (
    EXPERIMENT_ID as SOURCE_EXPERIMENT_ID,
)
from frame_ofa import (
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    per_size_frame_coupled_design,
)


class PerSizeRatioRunnerTests(unittest.TestCase):
    def test_ratio_worker_reconstructs_design_and_has_full_coverage(self) -> None:
        num_players = 6
        inner_calls = 600
        total_calls = inner_calls + 2 * num_players + 2
        base_seed = 20260827
        seed = _shared_greedy_seed(base_seed, 0, 0)
        relabel_seed = _shared_relabel_seed(seed)
        weights = np.linspace(0.1, 0.6, num_players)

        def evaluate(_: str, coalitions: np.ndarray) -> np.ndarray:
            return np.asarray(coalitions, dtype=np.float64) @ weights

        design = per_size_frame_coupled_design(
            num_players,
            inner_calls,
            seed=seed,
            candidate_pool=4,
            mean_balance=1.0,
            mean_balance_mode="normalized",
            relabel_seed=relabel_seed,
        )
        boundary_rows = boundary_coalitions(num_players)
        boundary = boundary_from_utilities(
            evaluate("tiny", boundary_rows), num_players
        )
        linear = estimate_coupled(
            design,
            evaluate("tiny", design.coalitions),
            boundary,
            baseline="linear",
        )
        schedule_hash = _size_schedule_sha256(design.sizes)
        relabel_hash = design.diagnostics["relabel_permutation_sha256"]

        with (
            patch(
                "experiments.run_per_size_ratio_comparison._num_players",
                return_value=num_players,
            ),
            patch(
                "experiments.run_per_size_ratio_comparison._evaluate",
                side_effect=evaluate,
            ),
        ):
            row = _run_ratio_cell(
                "tiny",
                0,
                0,
                inner_calls,
                total_calls,
                base_seed,
                np.asarray(linear).tolist(),
                schedule_hash,
                relabel_hash,
            )

        self.assertEqual(row["method"], RATIO_METHOD)
        self.assertEqual(row["actual_utility_calls"], total_calls)
        diagnostics = row["diagnostics"]
        self.assertEqual(
            diagnostics["linear_reconstruction_max_abs_difference"], 0.0
        )
        coverage = diagnostics["coverage"]
        self.assertTrue(coverage["all_player_size_strata_covered"])
        self.assertEqual(coverage["missing_inclusion_strata"], 0)
        self.assertEqual(coverage["missing_exclusion_strata"], 0)
        self.assertGreater(coverage["minimum_inclusion_count"], 0)
        self.assertGreater(coverage["minimum_exclusion_count"], 0)

    def test_extends_scope_report_and_resumes_cell_granularly(self) -> None:
        dataset = "airport"
        multipliers = (1, 2)
        repeats = 2
        base_seed = 20260827
        num_tasks = 4
        inner_budgets, total_budgets = total_call_budgets(
            dataset, multipliers
        )
        truth, _ = _exact_truth(dataset)

        raw_rows = []
        for repeat in range(repeats):
            for method in SOURCE_METHOD_ORDER:
                for budget_index, (inner_calls, total_calls) in enumerate(
                    zip(inner_budgets, total_budgets, strict=True)
                ):
                    pair_hash = f"pair-{repeat}-{budget_index}"
                    diagnostics = {"utility_evaluations": total_calls}
                    if method == "inside_greedy_global":
                        diagnostics.update(
                            {
                                "second_moment_scope": "global_weighted",
                                "estimator": "coupled_linear",
                                "size_schedule_sha256": pair_hash,
                                "relabel_permutation_sha256": pair_hash,
                            }
                        )
                    elif method == "inside_greedy_per_size":
                        diagnostics.update(
                            {
                                "second_moment_scope": "per_size",
                                "estimator": "coupled_linear",
                                "size_schedule_sha256": pair_hash,
                                "relabel_permutation_sha256": pair_hash,
                            }
                        )
                    raw_rows.append(
                        {
                            "repeat": repeat,
                            "method": method,
                            "budget_index": budget_index,
                            "seed": _shared_greedy_seed(
                                base_seed, repeat, budget_index
                            ),
                            "inner_utility_call_budget": inner_calls,
                            "target_total_utility_calls": total_calls,
                            "actual_utility_calls": total_calls,
                            "estimate": truth.tolist(),
                            "elapsed_seconds": 0.0,
                            "diagnostics": diagnostics,
                        }
                    )

        source = {
            "status": "complete",
            "experiment": SOURCE_EXPERIMENT_ID,
            "configuration": {
                "dataset": dataset,
                "budget_multipliers": list(multipliers),
                "inner_utility_call_budgets": list(inner_budgets),
                "total_call_budgets": list(total_budgets),
                "repeats": repeats,
                "base_seed": base_seed,
                "num_tasks_per_serial_baseline": num_tasks,
                "methods": list(SOURCE_METHOD_ORDER),
            },
            "ground_truth": {"values": truth.tolist()},
            "raw_tasks": raw_rows,
        }

        def fake_ratio(
            dataset_value: str,
            repeat: int,
            budget_index: int,
            inner_calls: int,
            total_calls: int,
            seed_value: int,
            expected_linear: list[float],
            schedule_hash: str,
            relabel_hash: str,
        ) -> dict[str, object]:
            self.assertEqual(dataset_value, dataset)
            self.assertEqual(schedule_hash, relabel_hash)
            return {
                "repeat": repeat,
                "method": RATIO_METHOD,
                "budget_index": budget_index,
                "seed": _shared_greedy_seed(
                    seed_value, repeat, budget_index
                ),
                "inner_utility_call_budget": inner_calls,
                "target_total_utility_calls": total_calls,
                "actual_utility_calls": total_calls,
                "estimate": expected_linear,
                "elapsed_seconds": 0.0,
                "diagnostics": {
                    "design_method": "frame_coupled_per_size",
                    "second_moment_scope": "per_size",
                    "estimator": (
                        "ofa_conditional_mean_ratio_missing_raise"
                    ),
                    "size_schedule_sha256": schedule_hash,
                    "relabel_permutation_sha256": relabel_hash,
                    "linear_reconstruction_max_abs_difference": 0.0,
                    "coverage": {
                        "all_player_size_strata_covered": True,
                        "minimum_inclusion_count": 1,
                        "minimum_exclusion_count": 1,
                        "missing_inclusion_strata": 0,
                        "missing_exclusion_strata": 0,
                        "maximum_absolute_inclusion_count_deviation": 0.0,
                        "maximum_relative_inclusion_count_deviation": 0.0,
                    },
                },
            }

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.json"
            output_path = Path(directory) / "output.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            with patch(
                "experiments.run_per_size_ratio_comparison._run_ratio_cell",
                side_effect=fake_ratio,
            ):
                report = run_experiment(
                    dataset=dataset,
                    source_report=source_path,
                    budget_multipliers=multipliers,
                    repeats=repeats,
                    processes=1,
                    base_seed=base_seed,
                    bootstrap_samples=16,
                    num_tasks=num_tasks,
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
            self.assertEqual(
                report["validation"][
                    "maximum_linear_reconstruction_absolute_difference"
                ],
                0.0,
            )

            with patch(
                "experiments.run_per_size_ratio_comparison._run_ratio_cell",
                side_effect=AssertionError("completed ratio cells must resume"),
            ):
                resumed = run_experiment(
                    dataset=dataset,
                    source_report=source_path,
                    budget_multipliers=multipliers,
                    repeats=repeats,
                    processes=2,
                    base_seed=base_seed,
                    bootstrap_samples=16,
                    num_tasks=num_tasks,
                    output_path=output_path,
                    resume=True,
                )
            self.assertEqual(resumed["status"], "complete")


if __name__ == "__main__":
    unittest.main()

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from experiments.run_cancer_inside_comparison import (
    BOUNDARY_CALLS,
    EXPERIMENT_ID,
    FRESH_METHODS,
    GROUND_TRUTH_SEED,
    INNER_BUDGETS,
    INSIDE_GREEDY_CANDIDATE_POOL,
    MEAN_BALANCE_LAMBDA0,
    METHOD_ORDER,
    METHOD_SEED,
    NUM_BASELINE_TASKS,
    NUM_PLAYERS,
    REPEATS,
    SOURCE_BOOTSTRAP_METHOD_INDEX,
    SOURCE_METHOD_MAPPING,
    TOTAL_BUDGETS,
    _method_seed,
    _reduced_source_summary,
    _shard_key,
    _validate_boundary_reconstruction,
    _validate_fresh_row,
    _validate_ground_truth_cache,
    _validate_resume_report,
    build_report,
)
from experiments.validate_inside_comparison import validate_report


def _truth_payload() -> dict[str, object]:
    target = 1.0 / 3.0
    raw = np.linspace(1.0, 2.0, NUM_PLAYERS)
    values = raw / raw.sum() * target
    standard_errors = np.linspace(1e-6, 2e-6, NUM_PLAYERS)
    half_widths = 3.0 * standard_errors
    permutations = 1_600_000
    conceptual = permutations * (NUM_PLAYERS - 1)
    saved = 3_200_000
    return {
        "values": values.tolist(),
        "standard_errors": standard_errors.tolist(),
        "rmse_standard_error": float(
            np.sqrt(np.mean(np.square(standard_errors)))
        ),
        "simultaneous_half_widths": half_widths.tolist(),
        "max_simultaneous_half_width": float(half_widths.max()),
        "half_split_rmse": 1.9e-5,
        "independent_pair_units": 800_000,
        "permutations": permutations,
        "conceptual_internal_prefix_calls": conceptual,
        "physical_internal_prefix_calls": conceptual - saved,
        "boundary_reuse_saved_calls": saved,
        "permutation_path_equivalent_utility_calls": conceptual + 2,
        "shared_boundary_calls_physically_evaluated": BOUNDARY_CALLS,
        "max_worker_tasks_per_outer_block": 128,
        "peak_forward_permutation_bytes": 931_840,
        "wall_seconds": 20_101.0,
    }


def _source_report() -> dict[str, object]:
    truth = _truth_payload()
    values = np.asarray(truth["values"], dtype=np.float64)
    direction = np.linspace(-1.0, 1.0, NUM_PLAYERS)
    direction -= direction.mean()
    results: dict[str, object] = {}
    for budget_index, (inner, total) in enumerate(
        zip(INNER_BUDGETS, TOTAL_BUDGETS)
    ):
        methods: dict[str, object] = {}
        for method_index, source_method in enumerate(
            SOURCE_BOOTSTRAP_METHOD_INDEX
        ):
            estimates = np.asarray(
                [
                    values
                    + direction
                    * (method_index + 1)
                    * (repeat + 1)
                    * 1e-7
                    / (budget_index + 1)
                    for repeat in range(10)
                ]
            )
            methods[source_method] = {
                "estimates": estimates.tolist(),
                "mean_design_seconds": 1.0,
                "mean_utility_seconds": 2.0,
            }
        results[str(inner)] = {
            "inner_utility_calls": inner,
            "boundary_utility_calls": BOUNDARY_CALLS,
            "total_utility_calls_per_estimate": total,
            "methods": methods,
        }
    return {
        "dataset": {"dataset": "cancer", "split": "stratified_train_test"},
        "boundary": {
            "empty": 0.63,
            "full": 0.63 + 1.0 / 3.0,
            "efficiency_target": 1.0 / 3.0,
            "utility_calls": BOUNDARY_CALLS,
        },
        "ground_truth": truth,
        "results_by_inner_budget": results,
    }


def _coverage() -> dict[str, object]:
    return {
        "all_player_size_strata_covered": True,
        "minimum_inclusion_count": 1,
        "minimum_exclusion_count": 1,
        "missing_inclusion_strata": 0,
        "missing_exclusion_strata": 0,
        "missing_inner_sizes": 0,
    }


def _fresh_row(method: str, repeat: int, budget_index: int) -> dict[str, object]:
    total = TOTAL_BUDGETS[budget_index]
    truth = np.asarray(_truth_payload()["values"], dtype=np.float64)
    direction = np.linspace(-1.0, 1.0, NUM_PLAYERS)
    direction -= direction.mean()
    estimate = truth + direction * (repeat + 1) * 1e-7
    actual = total - 7 if method == "tmc_shapley" else total
    diagnostics: dict[str, object] = {"utility_evaluations": actual}
    if method == "tmc_shapley":
        diagnostics.update(
            {
                "boundary_utility_evaluations": 2,
                "prefix_utility_evaluations": actual - 2,
            }
        )
    elif method == "s_diff":
        diagnostics.update(
            {"boundary_evaluations": 2, "num_utility_samples": total - 2}
        )
    elif method == "kernel_shap":
        diagnostics.update(
            {
                "boundary_utility_evaluations": 2,
                "inner_utility_evaluations": total - 2,
            }
        )
    row: dict[str, object] = {
        "method": method,
        "repeat": repeat,
        "budget_index": budget_index,
        "seed": _method_seed(method, repeat, budget_index),
        "target_total_calls": total,
        "actual_total_calls": actual,
        "unused_calls": total - actual,
        "estimate": estimate.tolist(),
        "elapsed_seconds": 3.0,
        "diagnostics": diagnostics,
    }
    if method == "inside_greedy":
        factor = (NUM_PLAYERS - 2.0) / (NUM_PLAYERS - 1.0)
        coverage = _coverage()
        design = {
            "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
            "mean_balance_mode": "normalized",
            "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
            "mean_balance_mean_weight_squared": 1.0,
            "mean_balance_dimension_correction": factor,
            "mean_balance_normalization_factor": factor,
            "mean_balance_effective_raw": MEAN_BALANCE_LAMBDA0 * factor,
            "second_moment_scope": "per_size",
            "fixed_slice_rng_partitioning": (
                "independent_seedsequence_substreams"
            ),
        }
        diagnostics.update(
            {
                "utility_evaluations": total,
                "target_call_budget": total,
                "unused_calls": 0,
                "boundary_utility_evaluations": BOUNDARY_CALLS,
                "inner_utility_evaluations": INNER_BUDGETS[budget_index],
                "design_method": "frame_coupled_per_size",
                "second_moment_scope": "per_size",
                "estimator": "ofa_conditional_mean_ratio_missing_raise",
                "official_ratio_missing_policy": "raise",
                "candidate_pool": INSIDE_GREEDY_CANDIDATE_POOL,
                "mean_balance_mode": "normalized",
                "mean_balance_lambda0": MEAN_BALANCE_LAMBDA0,
                "mean_balance_effective_raw": MEAN_BALANCE_LAMBDA0 * factor,
                "coverage": coverage,
                "design_diagnostics": design,
            }
        )
        row.update(
            {
                "inner_utility_calls": INNER_BUDGETS[budget_index],
                "design_seconds": 1.0,
                "utility_seconds": 2.0,
                "coverage": coverage,
                "design_diagnostics": design,
            }
        )
    return row


def _all_fresh_rows() -> list[dict[str, object]]:
    return [
        _fresh_row(method, repeat, budget_index)
        for method in FRESH_METHODS
        for repeat in range(REPEATS)
        for budget_index in range(len(INNER_BUDGETS))
    ]


class CancerInsideRunnerTests(unittest.TestCase):
    def test_checkpoint_identity_and_row_call_contract(self) -> None:
        key = _shard_key("a" * 64, "inside_greedy", 2)
        self.assertEqual(key["experiment"], EXPERIMENT_ID)
        self.assertEqual(key["candidate_pool"], 64)
        self.assertEqual(key["mean_balance_lambda0"], 1.0 / 16.0)
        self.assertEqual(key["num_baseline_tasks"], NUM_BASELINE_TASKS)
        for method in FRESH_METHODS:
            _validate_fresh_row(_fresh_row(method, 0, 0))
        bad = _fresh_row("cc", 0, 0)
        bad["actual_total_calls"] = TOTAL_BUDGETS[0] - 1
        bad["unused_calls"] = 1
        bad["diagnostics"]["utility_evaluations"] = TOTAL_BUDGETS[0] - 1
        with self.assertRaisesRegex(ValueError, "full call cap"):
            _validate_fresh_row(bad)

    def test_source_reduction_uses_original_method_specific_bootstrap_seeds(self) -> None:
        source = _source_report()
        truth = np.asarray(source["ground_truth"]["values"])
        truth_se = np.asarray(source["ground_truth"]["standard_errors"])
        seeds = []
        for method, source_method in SOURCE_METHOD_MAPPING.items():
            summary = _reduced_source_summary(
                source["results_by_inner_budget"][str(INNER_BUDGETS[2])][
                    "methods"
                ][source_method],
                method=method,
                source_method=source_method,
                budget_index=2,
                truth=truth,
                truth_se=truth_se,
                efficiency_target=1.0 / 3.0,
                source_path=Path("source.json"),
                source_sha256="a" * 64,
            )
            seeds.append(summary["provenance"]["recomputed_bootstrap_seed"])
            self.assertEqual(
                summary["diagnostics_by_repeat"][0]["seed"],
                METHOD_SEED + 200_000,
            )
        self.assertEqual(
            seeds,
            [
                METHOD_SEED + 200 + SOURCE_BOOTSTRAP_METHOD_INDEX[value]
                for value in SOURCE_METHOD_MAPPING.values()
            ],
        )
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_ground_truth_cache_requires_full_canonical_identity(self) -> None:
        truth = _truth_payload()
        cache_key = {
            "C": 1.0,
            "dataset": "cancer",
            "dataset_seed": 2024,
            "gamma": "scale",
            "kernel": "rbf",
            "model": "sklearn.svm.SVC",
            "num_pairs": 800_000,
            "num_players": NUM_PLAYERS,
            "pairing": "permutation_reverse",
            "seed": GROUND_TRUTH_SEED,
            "test_size": 0.2,
        }
        diagnostic_keys = (
            "boundary_reuse_saved_calls",
            "conceptual_internal_prefix_calls",
            "max_worker_tasks_per_outer_block",
            "peak_forward_permutation_bytes",
            "physical_internal_prefix_calls",
            "wall_seconds",
        )
        diagnostics = {key: truth[key] for key in diagnostic_keys}

        def write(path: Path, identity: dict[str, object]) -> None:
            np.savez(
                path,
                cache_key=json.dumps(identity),
                diagnostics=json.dumps(diagnostics),
                **{
                    key: truth[key]
                    for key in (
                        "values",
                        "standard_errors",
                        "rmse_standard_error",
                        "simultaneous_half_widths",
                        "max_simultaneous_half_width",
                        "half_split_rmse",
                    )
                },
            )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "gt.npz"
            write(path, cache_key)
            self.assertEqual(len(_validate_ground_truth_cache(path, truth)), 64)
            wrong = dict(cache_key)
            wrong["seed"] = GROUND_TRUTH_SEED + 1
            write(path, wrong)
            with self.assertRaisesRegex(ValueError, "identity"):
                _validate_ground_truth_cache(path, truth)

    def test_final_report_schema_call_accounting_and_common_validator(self) -> None:
        source = _source_report()
        rows = _all_fresh_rows()
        boundary = np.linspace(0.63, 0.63 + 1.0 / 3.0, BOUNDARY_CALLS)
        report = build_report(
            source,
            source_path=Path("source.json"),
            source_sha256="a" * 64,
            ground_truth_cache=Path("gt.npz"),
            ground_truth_cache_sha256="b" * 64,
            rows=rows,
            boundary_utilities=boundary,
            boundary_wall_seconds=4.0,
            repeat_processes=3,
            jobs_per_repeat=42,
            sdiff_jobs=128,
            design_jobs_per_repeat=40,
            chunksize=512,
            start_method="spawn",
            design_start_method="fork",
            experiment_wall_seconds=10.0,
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["configuration"]["methods"], list(METHOD_ORDER))
        self.assertEqual(
            report["configuration"]["protocol_version"],
            "per_size_ratio_k64_lambda1over16",
        )
        expected_physical = BOUNDARY_CALLS + sum(
            int(row["actual_total_calls"])
            - (BOUNDARY_CALLS if row["method"] == "inside_greedy" else 0)
            for row in rows
        )
        self.assertEqual(
            report["fresh_progress"]["fresh_stage_physical_utility_calls"],
            expected_physical,
        )
        self.assertEqual(validate_report(report)["status"], "ready_to_share")

        resumed = _validate_resume_report(
            report,
            source_sha256="a" * 64,
            ground_truth_cache_sha256="b" * 64,
        )
        self.assertEqual(len(resumed), len(rows))
        utilities, wall = _validate_boundary_reconstruction(report)
        self.assertEqual(len(utilities), BOUNDARY_CALLS)
        self.assertEqual(wall, 4.0)

        corrupted = deepcopy(report)
        corrupted["boundary_reconstruction"]["utilities"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "hash differs"):
            _validate_boundary_reconstruction(corrupted)

        corrupted = deepcopy(report)
        corrupted["fresh_progress"]["completed_cells"] -= 1
        with self.assertRaisesRegex(ValueError, "completed-cell"):
            _validate_resume_report(
                corrupted,
                source_sha256="a" * 64,
                ground_truth_cache_sha256="b" * 64,
            )


if __name__ == "__main__":
    unittest.main()

"""Evaluate cyclic-orbit ratio Frame-OFA and append it to an Iris report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from frame_ofa import (
    GameEvaluator,
    boundary_coalitions,
    boundary_from_utilities,
    cyclic_orbit_frame_design,
    estimate_ratio_ofa,
)

from .iris_data_valuation import (
    paired_rmse_difference,
    summarize_method,
)
from .iris_game import IrisLogisticGame, load_official_iris_split
from .iris_sklearn_game import IrisSklearnGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--chunksize", type=int, default=32)
    parser.add_argument("--start-method", default="spawn")
    args = parser.parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    configuration = report["configuration"]
    game_args, _ = load_official_iris_split(
        n_valued=configuration["num_players"],
        n_performance=configuration["n_performance"],
        dataset_seed=report["dataset"]["dataset_seed"],
    )
    model = configuration["model"]
    if model == "float64 3-class linear logistic, one SGD epoch":
        game_factory = IrisLogisticGame
    else:
        game_factory = IrisSklearnGame
        model_names = {
            "sklearn_lr_accuracy": "lr",
            "linear_svm_accuracy": "linear_svm",
            "rbf_svm_accuracy": "rbf_svm",
        }
        game_args = game_args | {"model": model_names[model]}

    num_players = configuration["num_players"]
    rows = boundary_coalitions(num_players)
    ground_truth = np.asarray(report["ground_truth"]["values"])
    ground_truth_se = np.asarray(
        report["ground_truth"]["standard_errors"]
    )
    method_seed = int(configuration["method_seed"])
    candidate_pool = int(configuration["candidate_pool"])
    efficiency_target = report["boundary"]["efficiency_target"]

    with GameEvaluator(
        game_factory,
        game_args,
        n_jobs=args.jobs,
        chunksize=args.chunksize,
        start_method=args.start_method,
        worker_threads=1,
    ) as evaluator:
        boundary = boundary_from_utilities(
            evaluator.evaluate(rows), num_players
        )
        if not np.isclose(
            boundary.full - boundary.empty, efficiency_target
        ):
            raise RuntimeError("recomputed boundary disagrees with report")

        for budget_index, (budget_key, result) in enumerate(
            report["results_by_inner_budget"].items()
        ):
            inner_budget = int(budget_key)
            if inner_budget % num_players:
                continue
            estimates = []
            design_seconds = []
            utility_seconds = []
            print(f"Orbit budget {inner_budget}", flush=True)
            for repeat in range(configuration["repeats"]):
                seed = method_seed + budget_index * 100_000 + repeat
                start = time.perf_counter()
                design = cyclic_orbit_frame_design(
                    num_players,
                    num_orbits=inner_budget // num_players,
                    seed=seed,
                    candidate_pool=candidate_pool,
                )
                design_seconds.append(time.perf_counter() - start)
                start = time.perf_counter()
                utilities = evaluator.evaluate(design.coalitions)
                utility_seconds.append(time.perf_counter() - start)
                estimates.append(
                    estimate_ratio_ofa(design, utilities, boundary)
                )

            estimate_array = np.asarray(estimates)
            summary = summarize_method(
                estimate_array,
                ground_truth,
                ground_truth_se,
                efficiency_target,
                bootstrap_seed=method_seed + budget_index * 100 + 3,
            )
            summary["mean_design_seconds"] = float(
                np.mean(design_seconds)
            )
            summary["mean_utility_seconds"] = float(
                np.mean(utility_seconds)
            )
            result["methods"]["frame_orbit_ratio"] = summary
            official = result["methods"]["official_ofa_fixed_ratio"]
            official_array = np.asarray(official["estimates"])
            official_rmse = official["aggregate_rmse"]
            orbit_rmse = summary["aggregate_rmse"]
            result["orbit_rmse_reduction_vs_official_ratio"] = (
                (official_rmse - orbit_rmse) / official_rmse
            )
            result.setdefault("paired_rmse_differences", {})[
                "official_ratio_minus_orbit"
            ] = paired_rmse_difference(
                official_array,
                estimate_array,
                ground_truth,
                seed=method_seed + budget_index * 10 + 9,
            )
            print(
                f"  official/orbit RMSE "
                f"{official_rmse:.4e}/{orbit_rmse:.4e}",
                flush=True,
            )

    args.result.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

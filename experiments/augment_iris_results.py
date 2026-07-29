"""Add derived paired-comparison diagnostics to an Iris result file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .iris_data_valuation import paired_rmse_difference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    report = json.loads(args.result.read_text(encoding="utf-8"))
    ground_truth = np.asarray(report["ground_truth"]["values"])
    method_seed = int(report["configuration"]["method_seed"])
    best_rmse = float("inf")

    for budget_index, (_, result) in enumerate(
        report["results_by_inner_budget"].items()
    ):
        methods = result["methods"]
        arrays = {
            name: np.asarray(summary["estimates"])
            for name, summary in methods.items()
        }
        paired = {
            "iid_linear_minus_frame": paired_rmse_difference(
                arrays["iid_linear_ofa"],
                arrays["frame_coupled"],
                ground_truth,
                seed=method_seed + budget_index * 10 + 7,
            ),
            "frame_minus_official_ratio": paired_rmse_difference(
                arrays["frame_coupled"],
                arrays["official_ofa_fixed_ratio"],
                ground_truth,
                seed=method_seed + budget_index * 10 + 8,
            ),
        }
        if "frame_orbit_ratio" in arrays:
            paired["official_ratio_minus_orbit"] = (
                paired_rmse_difference(
                    arrays["official_ofa_fixed_ratio"],
                    arrays["frame_orbit_ratio"],
                    ground_truth,
                    seed=method_seed + budget_index * 10 + 9,
                )
            )
        result["paired_rmse_differences"] = paired
        best_rmse = min(
            best_rmse,
            *(
                float(summary["aggregate_rmse"])
                for summary in methods.values()
            ),
        )

    ground_truth_report = report["ground_truth"]
    ground_truth_report["rmse_se_fraction_of_best_method_rmse"] = (
        ground_truth_report["rmse_standard_error"] / best_rmse
    )
    ground_truth_report[
        "max_simultaneous_half_width_fraction_of_best_method_rmse"
    ] = (
        ground_truth_report["max_simultaneous_half_width"] / best_rmse
    )
    ground_truth_report["parallel_calls_per_second"] = (
        ground_truth_report["physical_internal_prefix_calls"]
        / ground_truth_report["wall_seconds"]
    )
    args.result.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

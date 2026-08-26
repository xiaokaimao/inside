"""Create a non-destructive Wine checkpoint with TMC replacing full MC.

The completed comparison report contains expensive results for six other
locally ported baselines.  This migration preserves those values verbatim,
removes only ``permutation_mc_full``, and leaves ``tmc_shapley`` pending so
``experiments.add_wine_local_baselines`` can resume without recomputing the
other methods.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .add_wine_local_baselines import (
    TMC_CONFIGURATION,
    _write_checkpoint,
)


OLD_METHOD = "permutation_mc_full"
NEW_METHOD = "tmc_shapley"

# This is a historical migration used to create the now-immutable report that
# still contained the mislabelled linear estimator.  Freeze its target schema
# here so later corrections to DEFAULT_NEW_METHODS do not rewrite history.
LEGACY_TARGET_METHODS = (
    "gels_linear",
    "kernel_shap_sampled",
    "group_testing",
    "diff",
    "s_diff",
    "tmc_shapley",
    "stratified_marginal_mc",
)
LEGACY_METHOD_LABELS = {
    "gels_linear": "GELS (linear)",
    "kernel_shap_sampled": "KernelSHAP (sampled Gram)",
    "group_testing": "Group Testing",
    "diff": "Diff (differential matrix)",
    "s_diff": "S-Diff (external strict port)",
    "tmc_shapley": "TMC-Shapley",
    "stratified_marginal_mc": "Stratified marginal MC",
}


def migrate_report(report: dict[str, Any], *, source_path: Path) -> dict[str, Any]:
    """Return a migrated deep copy; never mutate the loaded old report."""
    migrated = copy.deepcopy(report)
    configuration = migrated.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("report configuration is missing")
    comparison = configuration.get("baseline_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("baseline_comparison metadata is missing")
    old_methods = comparison.get("new_methods")
    if not isinstance(old_methods, list) or OLD_METHOD not in old_methods:
        raise ValueError("source report does not declare permutation_mc_full")
    if NEW_METHOD in old_methods:
        raise ValueError("source report already declares tmc_shapley")

    expected_methods = list(LEGACY_TARGET_METHODS)
    if set(old_methods) - {OLD_METHOD} != set(expected_methods) - {NEW_METHOD}:
        raise ValueError("source report contains a different baseline set")
    comparison["new_methods"] = expected_methods
    comparison["method_labels"] = {
        method: LEGACY_METHOD_LABELS[method] for method in expected_methods
    }
    comparison["tmc_shapley"] = copy.deepcopy(TMC_CONFIGURATION)
    comparison["baseline_replacement"] = {
        "source_report": str(source_path),
        "removed_method": OLD_METHOD,
        "replacement_method": NEW_METHOD,
        "other_method_results_reused_verbatim": True,
    }

    results = migrated.get("results_by_inner_budget")
    if not isinstance(results, dict) or not results:
        raise ValueError("source report has no budget results")
    for budget, row in results.items():
        if not isinstance(row, dict) or not isinstance(row.get("methods"), dict):
            raise ValueError(f"budget {budget} has invalid method results")
        methods = row["methods"]
        if OLD_METHOD not in methods:
            raise ValueError(f"budget {budget} is missing {OLD_METHOD}")
        if NEW_METHOD in methods:
            raise ValueError(f"budget {budget} already contains {NEW_METHOD}")
        del methods[OLD_METHOD]

    migrated["status"] = "baseline_augmentation_running"
    return migrated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/"
            "wine_full_train_rbf_svm_all_baselines_3repeats_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "wine_full_train_rbf_svm_all_baselines_tmc_"
            "3repeats_71k_1p42m.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise ValueError(f"input report does not exist: {args.input}")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("output must differ from input")
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing output: {args.output}")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    migrated = migrate_report(report, source_path=args.input)
    _write_checkpoint(args.output, migrated)
    print(f"Saved resumable checkpoint {args.output}", flush=True)


if __name__ == "__main__":
    main()

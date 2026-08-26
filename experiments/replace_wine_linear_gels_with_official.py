"""Replace the mislabelled linear Wine curve with official GELS-Shapley.

The completed TMC report contains eleven expensive method results that remain
valid.  This non-destructive migration deep-copies that report, removes only
``gels_linear`` (which is algebraically simSHAP rather than GELS-Shapley), and
leaves ``gels_shapley`` pending for the resumable experiment runner.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from .add_wine_local_baselines import (
    DEFAULT_NEW_METHODS,
    GELS_SHAPLEY_CONFIGURATION,
    METHOD_LABELS,
    _source_digest,
    _write_checkpoint,
)


OLD_METHOD = "gels_linear"
NEW_METHOD = "gels_shapley"


def migrate_report(
    report: dict[str, Any], *, source_path: Path
) -> dict[str, Any]:
    """Return a migrated deep copy while preserving all other summaries."""
    migrated = copy.deepcopy(report)
    configuration = migrated.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("report configuration is missing")
    comparison = configuration.get("baseline_comparison")
    if not isinstance(comparison, dict):
        raise ValueError("baseline_comparison metadata is missing")
    old_methods = comparison.get("new_methods")
    if not isinstance(old_methods, list) or OLD_METHOD not in old_methods:
        raise ValueError("source report does not declare gels_linear")
    if NEW_METHOD in old_methods:
        raise ValueError("source report already declares gels_shapley")

    expected_methods = list(DEFAULT_NEW_METHODS)
    if set(old_methods) - {OLD_METHOD} != set(expected_methods) - {
        NEW_METHOD
    }:
        raise ValueError("source report contains a different baseline set")
    comparison["new_methods"] = expected_methods
    comparison["method_labels"] = {
        method: METHOD_LABELS[method] for method in expected_methods
    }
    comparison["gels_shapley"] = copy.deepcopy(
        GELS_SHAPLEY_CONFIGURATION
    )
    comparison["gels_shapley_replacement"] = {
        "source_report": str(source_path),
        "source_sha256": _source_digest(source_path),
        "removed_method": OLD_METHOD,
        "removed_method_identity": (
            "linear_horvitz_thompson_estimator_equivalent_to_simSHAP"
        ),
        "replacement_method": NEW_METHOD,
        "replacement_identity": (
            "Li_and_Yu_ICLR_2024_Algorithm_3_GELS-Shapley"
        ),
        "other_method_results_reused_verbatim": True,
    }

    results = migrated.get("results_by_inner_budget")
    if not isinstance(results, dict) or not results:
        raise ValueError("source report has no budget results")
    for budget, row in results.items():
        if not isinstance(row, dict) or not isinstance(
            row.get("methods"), dict
        ):
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
            "wine_full_train_rbf_svm_all_baselines_tmc_"
            "3repeats_71k_1p42m.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/"
            "wine_full_train_rbf_svm_all_baselines_gels_shapley_tmc_"
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
        raise ValueError(
            f"refusing to overwrite existing output: {args.output}"
        )
    report = json.loads(args.input.read_text(encoding="utf-8"))
    migrated = migrate_report(report, source_path=args.input)
    _write_checkpoint(args.output, migrated)
    print(f"Saved resumable checkpoint {args.output}", flush=True)


if __name__ == "__main__":
    main()

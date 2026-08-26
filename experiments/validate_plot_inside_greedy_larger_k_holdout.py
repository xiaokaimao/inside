"""Validate, bootstrap, and plot the locked larger-K holdout experiment.

The screening lock is reconstructed from all four source reports and compared
with the supplied locked-selection artifact.  Both validation reports are
then independently reconstructed from raw cells.  Derived Greedy/Orbit method
seeds must be disjoint from every screening report before any held-out result
is summarized.

The registered primary comparison is locked K=128 versus the predeclared K=64
diagnostic.  Paired-repeat bootstrap intervals are also reported versus the
registered default and INSIDE-Orbit for every dataset-budget cell and for the
equal-cell geometric mean across all six cells.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.summarize_plot_inside_greedy_larger_k import (
    DATASETS,
    _first_difference,
    _geometric_mean,
    _load_validated_pair,
    _sha256,
    build_larger_k_summary,
    build_locked_selection_artifact,
)
from experiments.validate_inside_greedy_hyperparameter_sweep import (
    _configuration_id,
    _seed_for,
)


LOCKED = (0.0, 128)
DIAGNOSTIC = (0.0, 64)
REGISTERED_DEFAULT = (1.0, 4)
VALIDATION_CONFIGS = {LOCKED, DIAGNOSTIC, REGISTERED_DEFAULT}
EXPECTED_VALIDATION_MULTIPLIERS = [500, 2000, 10000]
EXPECTED_VALIDATION_REPEATS = 20

BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_BASE_SEED = 20260826

METHOD_KEYS = (
    "inside_greedy_k128",
    "inside_greedy_k64",
    "registered_default",
    "inside_orbit",
)
COMPARATORS = (
    "inside_greedy_k64",
    "registered_default",
    "inside_orbit",
)
METHOD_LABELS = {
    "inside_greedy_k128": "K=128",
    "inside_greedy_k64": "K=64",
    "registered_default": "Default",
    "inside_orbit": "Orbit",
}
DATASET_LABELS = {"airport": "Airport", "voting": "Voting"}

BLUE = "#2B6CB0"
ORANGE = "#C56A1A"
CHARCOAL = "#20252B"
MID_GREY = "#747C85"
LIGHT_GREY = "#D8DDE3"
METHOD_STYLE = {
    "inside_greedy_k128": {
        "color": BLUE,
        "linestyle": "-",
        "marker": "o",
        "markerfacecolor": BLUE,
    },
    "inside_greedy_k64": {
        "color": ORANGE,
        "linestyle": "--",
        "marker": "^",
        "markerfacecolor": "white",
    },
    "registered_default": {
        "color": CHARCOAL,
        "linestyle": ":",
        "marker": "s",
        "markerfacecolor": "white",
    },
    "inside_orbit": {
        "color": MID_GREY,
        "linestyle": "-.",
        "marker": "D",
        "markerfacecolor": "white",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{path} does not contain an object")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
    temporary.replace(path)


def _screening_paths(
    selection: Mapping[str, Any],
) -> dict[str, dict[str, Path]]:
    sources = selection.get("screening_sources")
    _require(isinstance(sources, Mapping), "locked selection has no screening sources")
    result: dict[str, dict[str, Path]] = {}
    for report_set in ("base", "extension"):
        raw_set = sources.get(report_set)
        _require(isinstance(raw_set, Mapping), f"missing {report_set} sources")
        result[report_set] = {}
        for dataset in DATASETS:
            entry = raw_set.get(dataset)
            _require(isinstance(entry, Mapping), f"missing {report_set}/{dataset} source")
            path = Path(str(entry.get("path")))
            _require(path.is_file(), f"screening source does not exist: {path}")
            _require(
                _sha256(path) == str(entry.get("sha256")),
                f"screening source hash changed: {report_set}/{dataset}",
            )
            result[report_set][dataset] = path
    return result


def _revalidate_screening_lock(
    selection: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Path]], dict[str, dict[str, Mapping[str, Any]]]]:
    _require(selection.get("status") == "configuration_locked", "selection is not locked")
    locked = selection.get("locked_configuration")
    _require(isinstance(locked, Mapping), "locked configuration is absent")
    _require(
        (float(locked.get("lambda0")), int(locked.get("candidate_pool"))) == LOCKED,
        "locked selection is not lambda0=0,K=128",
    )
    paths = _screening_paths(selection)
    summary = build_larger_k_summary(
        paths["base"]["airport"],
        paths["base"]["voting"],
        extension_airport_path=paths["extension"]["airport"],
        extension_voting_path=paths["extension"]["voting"],
        reference="k16",
    )
    recomputed = build_locked_selection_artifact(summary)
    compared_fields = (
        "selection_rule",
        "screening_sources",
        "screening_overlap_merge_audit",
        "screening_contract",
        "locked_configuration",
        "ranking",
    )
    for field in compared_fields:
        difference = _first_difference(selection.get(field), recomputed.get(field), field)
        _require(difference is None, f"locked selection disagrees with screening: {difference}")

    audits: dict[str, dict[str, Mapping[str, Any]]] = {}
    for report_set in ("base", "extension"):
        _, pair_audits = _load_validated_pair(
            paths[report_set]["airport"], paths[report_set]["voting"]
        )
        for dataset in DATASETS:
            _require(
                pair_audits[dataset]["stage"] == "screening",
                f"{report_set}/{dataset} is not screening",
            )
        audits[report_set] = pair_audits
    return paths, audits


def _derived_method_seeds(audit: Mapping[str, Any]) -> set[int]:
    base_seed = int(audit["base_seed"])
    repeats = int(audit["repeats"])
    budget_indices = [int(value) for value in audit["selected_budget_indices"]]
    result = {
        _seed_for(base_seed, repeat, budget_index, method_index)
        for repeat in range(repeats)
        for budget_index in budget_indices
        for method_index in (0, 1)
    }
    expected = repeats * len(budget_indices) * 2
    _require(len(result) == expected, "derived method seeds collide within a report")
    return result


def _seed_digest(values: set[int]) -> str:
    encoded = np.asarray(sorted(values), dtype="<u8")
    return sha256(encoded.tobytes(order="C")).hexdigest()


def _seed_separation_audit(
    screening_audits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    validation_audits: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    screening_records = []
    validation_records = []
    screening_union: set[int] = set()
    validation_union: set[int] = set()
    screening_sets: dict[str, set[int]] = {}
    for report_set in ("base", "extension"):
        for dataset in DATASETS:
            label = f"{report_set}/{dataset}"
            seeds = _derived_method_seeds(screening_audits[report_set][dataset])
            screening_sets[label] = seeds
            screening_union.update(seeds)
            screening_records.append(
                {
                    "report": label,
                    "base_seed": screening_audits[report_set][dataset]["base_seed"],
                    "derived_seed_count": len(seeds),
                    "derived_seed_sha256": _seed_digest(seeds),
                }
            )
    validation_sets: dict[str, set[int]] = {}
    for dataset in DATASETS:
        seeds = _derived_method_seeds(validation_audits[dataset])
        validation_sets[dataset] = seeds
        validation_union.update(seeds)
        validation_records.append(
            {
                "report": dataset,
                "base_seed": validation_audits[dataset]["base_seed"],
                "derived_seed_count": len(seeds),
                "derived_seed_sha256": _seed_digest(seeds),
            }
        )
    pairwise = {}
    for screening_label, screening_seeds in screening_sets.items():
        for dataset, validation_seeds in validation_sets.items():
            key = f"{screening_label} vs validation/{dataset}"
            intersection = screening_seeds.intersection(validation_seeds)
            pairwise[key] = len(intersection)
            _require(not intersection, f"derived method seeds overlap: {key}")
    union_intersection = screening_union.intersection(validation_union)
    _require(not union_intersection, "screening and validation seed unions overlap")
    return {
        "passed": True,
        "method_indices": {"inside_greedy": 0, "inside_orbit": 1},
        "screening_reports": screening_records,
        "validation_reports": validation_records,
        "pairwise_intersection_counts": pairwise,
        "screening_union_count": len(screening_union),
        "screening_union_sha256": _seed_digest(screening_union),
        "validation_union_count": len(validation_union),
        "validation_union_sha256": _seed_digest(validation_union),
        "union_intersection_count": 0,
    }


def _config_set(audit: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["evaluated_configurations"]
    }


def _requested_set(audit: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in audit["requested_configurations"]
    }


def _repeat_array(values: Any, *, path: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} is not numeric") from error
    _require(result.ndim == 1 and result.size > 0, f"{path} is not one-dimensional")
    _require(np.all(np.isfinite(result)), f"{path} contains a failed/nonfinite repeat")
    _require(np.all(result >= 0.0), f"{path} contains negative RMSE")
    return result


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _bootstrap_interval(draws: np.ndarray) -> list[float]:
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return [float(lower), float(upper)]


def _paired_bootstrap_ratios(
    methods: Mapping[str, np.ndarray],
    *,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    locked = methods["inside_greedy_k128"]
    repeats = locked.size
    _require(
        all(values.size == repeats for values in methods.values()),
        "methods use different repeat counts",
    )
    indices = rng.integers(0, repeats, size=(BOOTSTRAP_SAMPLES, repeats))

    def draws(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.square(values[indices]), axis=1))

    locked_draws = draws(locked)
    result = {}
    for comparator in COMPARATORS:
        reference_draws = draws(methods[comparator])
        _require(
            np.all(reference_draws > 0.0),
            f"{comparator} has a zero bootstrap denominator",
        )
        result[comparator] = locked_draws / reference_draws
    return result


def _joint_geometric_draws(cell_draws: Sequence[np.ndarray]) -> np.ndarray:
    matrix = np.vstack(cell_draws)
    _require(
        matrix.shape[0] == 6
        and np.all(np.isfinite(matrix))
        and np.all(matrix >= 0.0),
        "joint bootstrap requires six finite nonnegative cells",
    )
    result = np.zeros(matrix.shape[1], dtype=np.float64)
    positive = np.all(matrix > 0.0, axis=0)
    result[positive] = np.exp(np.mean(np.log(matrix[:, positive]), axis=0))
    return result


def _total_budgets(
    report: Mapping[str, Any], selected_indices: Sequence[int]
) -> list[int]:
    configuration = report.get("configuration")
    _require(isinstance(configuration, Mapping), "validation configuration is absent")
    values = configuration.get("all_total_utility_call_budgets")
    _require(isinstance(values, list), "validation total-call budgets are absent")
    return [int(values[index]) for index in selected_indices]


def build_holdout_report(
    locked_selection_path: Path,
    airport_validation_path: Path,
    voting_validation_path: Path,
) -> dict[str, Any]:
    selection = _read(locked_selection_path)
    screening_paths, screening_audits = _revalidate_screening_lock(selection)
    validation_reports, validation_audits = _load_validated_pair(
        airport_validation_path, voting_validation_path
    )
    for dataset in DATASETS:
        audit = validation_audits[dataset]
        _require(audit["stage"] == "validation", f"{dataset} is not validation")
        _require(
            bool(audit["holdout_stage_eligible_pending_cross_report_seed_audit"]),
            f"{dataset} is not holdout-stage eligible",
        )
        _require(
            _config_set(audit) == VALIDATION_CONFIGS,
            f"{dataset} validation has the wrong configurations",
        )
        _require(
            _requested_set(audit) == {LOCKED, DIAGNOSTIC},
            f"{dataset} did not predeclare locked K128 and diagnostic K64",
        )
        _require(
            int(audit["repeats"]) == EXPECTED_VALIDATION_REPEATS,
            f"{dataset} validation does not use 20 repeats",
        )
        _require(
            list(audit["selected_budget_multipliers"])
            == EXPECTED_VALIDATION_MULTIPLIERS,
            f"{dataset} validation has the wrong three budgets",
        )
        _require(
            list(audit["selected_budget_indices"]) == [0, 1, 2],
            f"{dataset} validation does not use all three budget indices",
        )
    seed_audit = _seed_separation_audit(screening_audits, validation_audits)

    joint_draws: dict[str, list[np.ndarray]] = {
        comparator: [] for comparator in COMPARATORS
    }
    joint_points: dict[str, list[float]] = {
        comparator: [] for comparator in COMPARATORS
    }
    datasets_output: dict[str, Any] = {}
    validation_base_seed = int(validation_audits["airport"]["base_seed"])
    for dataset_position, dataset in enumerate(DATASETS):
        audit = validation_audits[dataset]
        results = audit["recomputed_results_by_configuration"]
        locked_result = results[_configuration_id(*LOCKED)]
        diagnostic_result = results[_configuration_id(*DIAGNOSTIC)]
        default_result = results[_configuration_id(*REGISTERED_DEFAULT)]
        selected_indices = [int(value) for value in audit["selected_budget_indices"]]
        total_budgets = _total_budgets(validation_reports[dataset], selected_indices)
        cells = []
        dataset_ratios: dict[str, list[float]] = {
            comparator: [] for comparator in COMPARATORS
        }
        for position, budget_index in enumerate(selected_indices):
            locked_budget = locked_result["budgets"][str(budget_index)]
            diagnostic_budget = diagnostic_result["budgets"][str(budget_index)]
            default_budget = default_result["budgets"][str(budget_index)]
            coverage = {
                "inside_greedy_k128": int(locked_budget["coverage_failures"]),
                "inside_greedy_k64": int(diagnostic_budget["coverage_failures"]),
                "registered_default": int(default_budget["coverage_failures"]),
                "inside_orbit": 0,
            }
            eligible = sum(coverage.values()) == 0
            cell: dict[str, Any] = {
                "budget_index": budget_index,
                "budget_multiplier": int(
                    audit["selected_budget_multipliers"][position]
                ),
                "total_utility_calls": total_budgets[position],
                "coverage_failures": coverage,
                "eligible": eligible,
                "methods": {},
                "locked_ratios": {},
            }
            if not eligible:
                for method in METHOD_KEYS:
                    cell["methods"][method] = {"aggregate_rmse": None}
                for comparator in COMPARATORS:
                    cell["locked_ratios"][comparator] = {
                        "point_estimate": None,
                        "paired_bootstrap_95_percent_ci": None,
                    }
                cells.append(cell)
                continue

            methods = {
                "inside_greedy_k128": _repeat_array(
                    locked_budget["rmse_by_repeat"],
                    path=f"{dataset}.{budget_index}.K128",
                ),
                "inside_greedy_k64": _repeat_array(
                    diagnostic_budget["rmse_by_repeat"],
                    path=f"{dataset}.{budget_index}.K64",
                ),
                "registered_default": _repeat_array(
                    default_budget["rmse_by_repeat"],
                    path=f"{dataset}.{budget_index}.default",
                ),
                "inside_orbit": _repeat_array(
                    locked_budget["orbit_rmse_by_repeat"],
                    path=f"{dataset}.{budget_index}.Orbit",
                ),
            }
            _require(
                all(values.size == EXPECTED_VALIDATION_REPEATS for values in methods.values()),
                f"{dataset} budget {budget_index} has the wrong repeat count",
            )
            aggregates = {method: _rms(values) for method, values in methods.items()}
            for method in METHOD_KEYS:
                cell["methods"][method] = {
                    "aggregate_rmse": aggregates[method],
                    "rmse_by_repeat": methods[method].tolist(),
                }
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        BOOTSTRAP_BASE_SEED,
                        validation_base_seed,
                        dataset_position,
                        budget_index,
                    ]
                )
            )
            draws = _paired_bootstrap_ratios(methods, rng=rng)
            for comparator in COMPARATORS:
                ratio = aggregates["inside_greedy_k128"] / aggregates[comparator]
                comparator_draws = draws[comparator]
                joint_points[comparator].append(ratio)
                joint_draws[comparator].append(comparator_draws)
                dataset_ratios[comparator].append(ratio)
                difference = methods["inside_greedy_k128"] - methods[comparator]
                cell["locked_ratios"][comparator] = {
                    "point_estimate": ratio,
                    "paired_bootstrap_95_percent_ci": _bootstrap_interval(
                        comparator_draws
                    ),
                    "paired_repeat_wins": int(np.count_nonzero(difference < 0.0)),
                    "paired_repeat_losses": int(np.count_nonzero(difference > 0.0)),
                    "paired_repeat_ties": int(np.count_nonzero(difference == 0.0)),
                }
            cells.append(cell)
        datasets_output[dataset] = {
            "num_players": int(audit["num_players"]),
            "all_cells_eligible": all(cell["eligible"] for cell in cells),
            "geometric_mean_locked_ratios": {
                comparator: (
                    _geometric_mean(values) if len(values) == 3 else None
                )
                for comparator, values in dataset_ratios.items()
            },
            "cells": cells,
        }

    all_cells_eligible = all(
        dataset["all_cells_eligible"] for dataset in datasets_output.values()
    )
    joint = {}
    for comparator in COMPARATORS:
        if all_cells_eligible and len(joint_draws[comparator]) == 6:
            draws = _joint_geometric_draws(joint_draws[comparator])
            joint[comparator] = {
                "point_estimate": _geometric_mean(joint_points[comparator]),
                "paired_bootstrap_95_percent_ci": _bootstrap_interval(draws),
            }
        else:
            joint[comparator] = {
                "point_estimate": None,
                "paired_bootstrap_95_percent_ci": None,
            }

    return {
        "status": (
            "holdout_validated" if all_cells_eligible else "holdout_coverage_failure"
        ),
        "experiment": "inside_greedy_larger_k_holdout",
        "registered_primary_comparison": {
            "locked": {"lambda0": LOCKED[0], "candidate_pool": LOCKED[1]},
            "predeclared_diagnostic": {
                "lambda0": DIAGNOSTIC[0],
                "candidate_pool": DIAGNOSTIC[1],
            },
            "six_cell_joint": joint["inside_greedy_k64"],
        },
        "locked_selection_source": {
            "path": str(locked_selection_path.resolve()),
            "sha256": _sha256(locked_selection_path),
        },
        "screening_revalidation": {
            "passed": True,
            "source_paths": {
                report_set: {
                    dataset: str(screening_paths[report_set][dataset].resolve())
                    for dataset in DATASETS
                }
                for report_set in ("base", "extension")
            },
            "source_hashes": selection["screening_sources"],
            "overlap_merge_audit": selection["screening_overlap_merge_audit"],
            "locked_configuration_recomputed": True,
        },
        "validation_sources": {
            "airport": {
                "path": str(airport_validation_path.resolve()),
                "sha256": _sha256(airport_validation_path),
            },
            "voting": {
                "path": str(voting_validation_path.resolve()),
                "sha256": _sha256(voting_validation_path),
            },
        },
        "validation_protocol": {
            "base_seed": validation_audits["airport"]["base_seed"],
            "repeats": EXPECTED_VALIDATION_REPEATS,
            "budget_multipliers": EXPECTED_VALIDATION_MULTIPLIERS,
            "all_rmse_and_aggregates_independently_recomputed_from_raw_cells": True,
            "locked_and_k64_predeclared_before_validation": True,
            "coverage_failures_retained": True,
        },
        "seed_separation_audit": seed_audit,
        "bootstrap": {
            "unit": "repeat within each dataset-budget cell",
            "paired_indices_across_all_four_methods": True,
            "cells_resampled_independently_for_joint_summary": True,
            "joint_aggregation": "equal-cell geometric mean across 6 cells",
            "samples": BOOTSTRAP_SAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "base_seed": BOOTSTRAP_BASE_SEED,
        },
        "all_six_cells_eligible": all_cells_eligible,
        "datasets": datasets_output,
        "six_cell_joint_locked_ratios": joint,
    }


def _format_calls(value: int) -> str:
    if value >= 1_000_000:
        scaled = value / 1_000_000
        return f"{scaled:g}M"
    if value >= 1_000:
        scaled = value / 1_000
        return f"{scaled:g}k"
    return str(value)


def plot_dataset_holdout(
    report: Mapping[str, Any], dataset: str, output_png: Path
) -> tuple[Path, Path]:
    _require(dataset in DATASETS, "unknown dataset")
    cells = report["datasets"][dataset]["cells"]
    calls = [int(cell["total_utility_calls"]) for cell in cells]
    fig, axis = plt.subplots(figsize=(6.3, 6.0))
    fig.suptitle(
        f"{DATASET_LABELS[dataset]} holdout",
        x=0.14,
        y=0.975,
        ha="left",
        fontsize=18,
        weight="semibold",
    )
    positive = True
    for method in METHOD_KEYS:
        values = [cell["methods"][method]["aggregate_rmse"] for cell in cells]
        numeric = [float(value) if value is not None else float("nan") for value in values]
        positive = positive and all(value > 0.0 for value in numeric if math.isfinite(value))
        style = METHOD_STYLE[method]
        axis.plot(
            calls,
            numeric,
            label=METHOD_LABELS[method],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markerfacecolor=style["markerfacecolor"],
            markeredgecolor=style["color"],
            markeredgewidth=1.7,
            markersize=7.5,
            linewidth=2.7,
        )
    scale_label = "log RMSE" if positive else "RMSE"
    fig.text(
        0.14,
        0.927,
        f"20 repeats; {scale_label}",
        ha="left",
        va="top",
        fontsize=11.5,
        color=MID_GREY,
    )
    axis.set_xscale("log")
    if positive:
        axis.set_yscale("log")
    axis.set_xticks(calls, [_format_calls(value) for value in calls])
    axis.set_xlabel("Utility calls", fontsize=15)
    axis.set_ylabel("RMSE", fontsize=15)
    axis.grid(axis="y", which="major", color=LIGHT_GREY, linewidth=1.0, alpha=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=13, width=1.25, length=5, color=CHARCOAL)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.45)
        spine.set_color(CHARCOAL)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=2,
        frameon=False,
        fontsize=11.5,
        handlelength=2.5,
        columnspacing=1.35,
    )
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.14, top=0.77)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_png, output_pdf


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--locked-selection", required=True, type=Path)
    parser.add_argument("--airport-validation", required=True, type=Path)
    parser.add_argument("--voting-validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--airport-plot", required=True, type=Path)
    parser.add_argument("--voting-plot", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_holdout_report(
        args.locked_selection,
        args.airport_validation,
        args.voting_validation,
    )
    _atomic_write_json(args.output, report)
    plot_dataset_holdout(report, "airport", args.airport_plot)
    plot_dataset_holdout(report, "voting", args.voting_plot)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

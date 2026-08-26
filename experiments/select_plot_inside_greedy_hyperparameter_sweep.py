"""Jointly select and plot INSIDE-Greedy sweep configurations.

Selection consumes independently validated Airport and Voting *screening*
reports.  Every dataset-by-budget aggregate is one equally weighted cell, so
the registered two-budget protocol has exactly four cells.  A configuration
with any ratio-coverage failure is eliminated.  Among the remaining settings,
the locked configuration minimizes

    mean_{dataset,budget} log(RMSE / registered-default RMSE).

The script can additionally consume two held-out validation reports.  They
must use seeds disjoint from screening and contain the locked configuration
plus the registered default.  Extra configurations that were declared in the
validation run are retained as diagnostic-only challengers; they never enter
the locked method's curves or main comparison.  Screening observations are
never included in held-out curves.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np

from experiments.run_inside_greedy_hyperparameter_sweep import (
    REGISTERED_CANDIDATE_POOL,
    REGISTERED_LAMBDA0,
)
from experiments.validate_inside_greedy_hyperparameter_sweep import (
    _configuration_id,
    _seed_for,
    validate_report,
)


DEFAULT = (REGISTERED_LAMBDA0, REGISTERED_CANDIDATE_POOL)
DATASETS = ("airport", "voting")
DATASET_LABELS = {
    "airport": "Airport",
    "voting": "U.S. Electoral Voting",
}
BLUE = "#2B6CB0"
ORANGE = "#C56A1A"
CHARCOAL = "#20252B"
GREY = "#777E87"
LIGHT_GREY = "#E4E7EB"
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "inside_ratio", [BLUE, "#F7F7F5", ORANGE]
)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_BASE_SEED = 20260825


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{path} does not contain an object")
    return value


def _config_set(validation: Mapping[str, Any]) -> set[tuple[float, int]]:
    return {
        (float(item["lambda0"]), int(item["candidate_pool"]))
        for item in validation["evaluated_configurations"]
    }


def _validated_pair(
    airport_path: Path,
    voting_path: Path,
    *,
    expected_stage: str,
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, Mapping[str, Any]],
    dict[str, dict[str, Any]],
]:
    paths = {"airport": airport_path, "voting": voting_path}
    reports = {dataset: _read(path) for dataset, path in paths.items()}
    audits = {
        dataset: validate_report(report) for dataset, report in reports.items()
    }
    for dataset in DATASETS:
        _require(audits[dataset]["dataset"] == dataset, "dataset paths are swapped")
        _require(
            audits[dataset]["stage"] == expected_stage,
            f"{dataset} report is not a {expected_stage} report",
        )
    first, second = audits["airport"], audits["voting"]
    _require(
        first["budget_multipliers"] == second["budget_multipliers"],
        "Airport and Voting use different budget-multiplier grids",
    )
    _require(
        first["selected_budget_indices"] == second["selected_budget_indices"],
        "Airport and Voting use different selected budget indices",
    )
    _require(
        first["repeats"] == second["repeats"],
        "Airport and Voting use different repeat counts",
    )
    _require(
        first["base_seed"] == second["base_seed"],
        "Airport and Voting use different base seeds",
    )
    _require(
        _config_set(first) == _config_set(second),
        "Airport and Voting evaluate different Greedy configurations",
    )
    sources = {
        dataset: {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
        }
        for dataset, path in paths.items()
    }
    return reports, audits, sources


def build_joint_selection(
    airport_path: Path, voting_path: Path
) -> dict[str, Any]:
    """Validate two screening reports and lock one configuration."""
    _, audits, sources = _validated_pair(
        airport_path, voting_path, expected_stage="screening"
    )
    for dataset in DATASETS:
        _require(
            bool(audits[dataset]["screening_selection_eligible"]),
            f"{dataset} report is not eligible for registered screening",
        )
    selected_indices = tuple(audits["airport"]["selected_budget_indices"])
    _require(
        len(selected_indices) == 2,
        "joint registered selection requires two budgets per dataset",
    )
    configs = sorted(_config_set(audits["airport"]), key=lambda item: (item[1], item[0]))
    rows: list[dict[str, Any]] = []
    for lambda0, pool in configs:
        config_id = _configuration_id(lambda0, pool)
        cell_ratios_default: dict[str, float | None] = {}
        cell_ratios_orbit: dict[str, float | None] = {}
        dataset_scores: dict[str, float | None] = {}
        coverage_failures = 0
        design_times: list[float] = []
        for dataset in DATASETS:
            result = audits[dataset]["recomputed_results_by_configuration"][config_id]
            per_dataset_default: list[float] = []
            for budget_index in selected_indices:
                budget = result["budgets"][str(budget_index)]
                key = f"{dataset}:budget_index={budget_index}"
                ratio_default = budget[
                    "aggregate_rmse_ratio_to_registered_default"
                ]
                ratio_orbit = budget["aggregate_rmse_ratio_to_inside_orbit"]
                cell_ratios_default[key] = ratio_default
                cell_ratios_orbit[key] = ratio_orbit
                if ratio_default is not None:
                    per_dataset_default.append(float(ratio_default))
            coverage_failures += int(result["overall"]["coverage_failures"])
            design_times.append(float(result["overall"]["mean_design_seconds"]))
            dataset_scores[dataset] = (
                float(np.exp(np.mean(np.log(per_dataset_default))))
                if len(per_dataset_default) == len(selected_indices)
                else None
            )
        eligible = coverage_failures == 0 and all(
            value is not None and value > 0.0
            for value in cell_ratios_default.values()
        )
        default_values = [float(value) for value in cell_ratios_default.values() if value is not None]
        orbit_values = [float(value) for value in cell_ratios_orbit.values() if value is not None]
        rows.append(
            {
                "configuration_id": config_id,
                "lambda0": lambda0,
                "candidate_pool": pool,
                "is_registered_default": (lambda0, pool) == DEFAULT,
                "selection_eligible": eligible,
                "coverage_failures": coverage_failures,
                "dataset_geometric_mean_rmse_ratio_to_default": dataset_scores,
                "cell_rmse_ratios_to_default": cell_ratios_default,
                "cell_rmse_ratios_to_orbit": cell_ratios_orbit,
                "joint_geometric_mean_rmse_ratio_to_default": (
                    float(np.exp(np.mean(np.log(default_values))))
                    if eligible
                    else None
                ),
                "joint_geometric_mean_rmse_ratio_to_orbit": (
                    float(np.exp(np.mean(np.log(orbit_values))))
                    if eligible and len(orbit_values) == 4
                    else None
                ),
                "worst_cell_rmse_ratio_to_default": (
                    float(max(default_values)) if eligible else None
                ),
                "worst_cell_rmse_ratio_to_orbit": (
                    float(max(orbit_values))
                    if eligible and len(orbit_values) == 4
                    else None
                ),
                # Equal-weight the two dataset-level means.  Runtime is
                # descriptive only and never enters the selection key.
                "mean_design_seconds": float(np.mean(design_times)),
            }
        )

    eligible_rows = [row for row in rows if row["selection_eligible"]]
    _require(eligible_rows, "every hyperparameter configuration failed coverage")
    eligible_rows.sort(
        key=lambda row: (
            float(row["joint_geometric_mean_rmse_ratio_to_default"]),
            float(row["worst_cell_rmse_ratio_to_default"]),
            int(row["candidate_pool"]),
            float(row["lambda0"]),
        )
    )
    best = eligible_rows[0]
    ranking = sorted(
        rows,
        key=lambda row: (
            not bool(row["selection_eligible"]),
            math.inf
            if row["joint_geometric_mean_rmse_ratio_to_default"] is None
            else float(row["joint_geometric_mean_rmse_ratio_to_default"]),
            math.inf
            if row["worst_cell_rmse_ratio_to_default"] is None
            else float(row["worst_cell_rmse_ratio_to_default"]),
            int(row["candidate_pool"]),
            float(row["lambda0"]),
        ),
    )
    return {
        "status": "configuration_locked",
        "experiment": "inside_greedy_joint_hyperparameter_selection",
        "selection_rule": {
            "primary": (
                "minimize the equal-weight mean log aggregate-RMSE ratio "
                "to the registered default across 2 datasets x 2 budgets"
            ),
            "coverage_policy": "eliminate any configuration with a coverage failure",
            "tie_breaks": [
                "lower worst dataset-budget RMSE ratio to default",
                "smaller candidate pool K",
                "smaller lambda0",
            ],
            "design_time_is_descriptive_not_a_selection_tie_break": True,
            "cell_count": 4,
            "datasets": list(DATASETS),
            "budget_indices": list(selected_indices),
            "budget_multipliers": audits["airport"][
                "selected_budget_multipliers"
            ],
        },
        "screening_sources": sources,
        "screening_contract": {
            "base_seed": audits["airport"]["base_seed"],
            "repeats": audits["airport"]["repeats"],
            "budget_multipliers": audits["airport"]["budget_multipliers"],
            "all_rmse_and_aggregates_independently_recomputed": True,
            "greedy_size_schedules_and_relabels_paired_across_hyperparameters": True,
            "orbit_is_equal-call_but_independently_randomized": True,
        },
        "locked_configuration": {
            key: best[key]
            for key in (
                "configuration_id",
                "lambda0",
                "candidate_pool",
                "joint_geometric_mean_rmse_ratio_to_default",
                "joint_geometric_mean_rmse_ratio_to_orbit",
                "worst_cell_rmse_ratio_to_default",
                "worst_cell_rmse_ratio_to_orbit",
                "mean_design_seconds",
            )
        },
        "ranking": ranking,
        "holdout_validation": None,
    }


def _method_seed_set(
    *, base_seed: int, repeats: int, budget_indices: Sequence[int]
) -> set[int]:
    return {
        _seed_for(base_seed, repeat, budget_index, method_index)
        for repeat in range(repeats)
        for budget_index in budget_indices
        for method_index in (0, 1)
    }


def _repeat_rmse_array(values: Any, *, path: str) -> np.ndarray:
    """Return one finite, nonnegative RMSE observation per repeat."""
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} is not a numeric repeat-level RMSE array") from error
    _require(result.ndim == 1 and result.size > 0, f"{path} must be one-dimensional")
    _require(np.all(np.isfinite(result)), f"{path} contains a failed/non-finite repeat")
    _require(np.all(result >= 0.0), f"{path} contains a negative RMSE")
    return result


def _root_mean_square(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _geometric_mean_nonnegative(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1
        and array.size > 0
        and np.all(np.isfinite(array))
        and np.all(array >= 0.0),
        "geometric-mean inputs must be finite and nonnegative",
    )
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _bootstrap_cell_ratio_draws(
    locked_rmse: np.ndarray,
    default_rmse: np.ndarray,
    orbit_rmse: np.ndarray,
    *,
    rng: np.random.Generator,
    samples: int = BOOTSTRAP_SAMPLES,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired repeat bootstrap of aggregate-RMSE ratios for one cell.

    The same resampled repeat indices are applied to all three methods.  This
    retains the registered repeat blocking and, for the two Greedy settings,
    their shared size schedule and relabeling.
    """
    _require(samples >= 1, "bootstrap sample count must be positive")
    repeats = locked_rmse.size
    _require(
        default_rmse.size == repeats and orbit_rmse.size == repeats,
        "bootstrap methods have different repeat counts",
    )
    indices = rng.integers(0, repeats, size=(samples, repeats))

    def draws(values: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean(np.square(values[indices]), axis=1))

    locked_draws = draws(locked_rmse)
    default_draws = draws(default_rmse)
    orbit_draws = draws(orbit_rmse)
    # A zero baseline RMSE makes a relative effect unidentified for bootstrap
    # samples containing only exact repeats.  Refuse to emit infinities/NaNs.
    _require(
        np.all(default_draws > 0.0),
        "registered-default bootstrap denominator is zero",
    )
    _require(
        np.all(orbit_draws > 0.0),
        "INSIDE-Orbit bootstrap denominator is zero",
    )
    return locked_draws / default_draws, locked_draws / orbit_draws


def _bootstrap_interval(draws: np.ndarray) -> list[float]:
    tail = (1.0 - BOOTSTRAP_CONFIDENCE_LEVEL) / 2.0
    lower, upper = np.quantile(draws, [tail, 1.0 - tail])
    return [float(lower), float(upper)]


def _joint_geometric_draws(cell_draws: Sequence[np.ndarray]) -> np.ndarray:
    _require(cell_draws, "joint bootstrap has no dataset-budget cells")
    matrix = np.vstack(cell_draws)
    _require(np.all(np.isfinite(matrix)) and np.all(matrix >= 0.0), "invalid draws")
    # Preserve an exact zero instead of replacing it by an arbitrary epsilon.
    result = np.zeros(matrix.shape[1], dtype=np.float64)
    positive = np.all(matrix > 0.0, axis=0)
    result[positive] = np.exp(np.mean(np.log(matrix[:, positive]), axis=0))
    return result


def attach_holdout_validation(
    selection: Mapping[str, Any],
    airport_path: Path,
    voting_path: Path,
    *,
    locked_selection_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach leakage-checked held-out curves to an already locked selection."""
    _, audits, sources = _validated_pair(
        airport_path, voting_path, expected_stage="validation"
    )
    result = json.loads(json.dumps(selection))
    locked = result["locked_configuration"]
    locked_tuple = (float(locked["lambda0"]), int(locked["candidate_pool"]))
    expected_configs = {DEFAULT, locked_tuple}
    screened_configs = {
        (float(row["lambda0"]), int(row["candidate_pool"]))
        for row in result["ranking"]
    }
    additional_configs: set[tuple[float, int]] | None = None
    for dataset in DATASETS:
        _require(
            bool(
                audits[dataset][
                    "holdout_stage_eligible_pending_cross_report_seed_audit"
                ]
            ),
            f"{dataset} report is not validation-stage eligible",
        )
        observed_configs = _config_set(audits[dataset])
        predeclared_configs = {
            (float(item["lambda0"]), int(item["candidate_pool"]))
            for item in audits[dataset]["requested_configurations"]
        }
        _require(
            expected_configs.issubset(observed_configs),
            f"{dataset} validation is missing locked best or registered default",
        )
        _require(
            observed_configs.issubset(screened_configs),
            f"{dataset} validation contains a challenger absent from screening",
        )
        extras = observed_configs - expected_configs
        _require(
            extras.issubset(predeclared_configs),
            f"{dataset} diagnostic challenger was not predeclared",
        )
        additional_configs = extras if additional_configs is None else additional_configs
        _require(
            extras == additional_configs,
            "Airport and Voting have different diagnostic challenger sets",
        )
        _require(
            audits[dataset]["selected_budget_indices"]
            == list(range(len(audits[dataset]["budget_multipliers"]))),
            f"{dataset} validation does not cover the full budget grid",
        )
    screening = result["screening_contract"]
    _require(
        audits["airport"]["base_seed"] != int(screening["base_seed"]),
        "validation base seed reuses screening",
    )
    screening_seeds = _method_seed_set(
        base_seed=int(screening["base_seed"]),
        repeats=int(screening["repeats"]),
        budget_indices=result["selection_rule"]["budget_indices"],
    )
    validation_seeds = _method_seed_set(
        base_seed=int(audits["airport"]["base_seed"]),
        repeats=int(audits["airport"]["repeats"]),
        budget_indices=audits["airport"]["selected_budget_indices"],
    )
    _require(
        not screening_seeds.intersection(validation_seeds),
        "screening and validation derived method seeds overlap",
    )

    curves: dict[str, Any] = {}
    joint_default_draws: list[np.ndarray] = []
    joint_orbit_draws: list[np.ndarray] = []
    joint_default_points: list[float] = []
    joint_orbit_points: list[float] = []
    validation_base_seed = int(audits["airport"]["base_seed"])
    for dataset_position, dataset in enumerate(DATASETS):
        locked_result = audits[dataset][
            "recomputed_results_by_configuration"
        ][_configuration_id(*locked_tuple)]
        default_result = audits[dataset][
            "recomputed_results_by_configuration"
        ][_configuration_id(*DEFAULT)]
        report_config = _read(
            airport_path if dataset == "airport" else voting_path
        )["configuration"]
        total_budgets = report_config["all_total_utility_call_budgets"]
        points: list[dict[str, Any]] = []
        for budget_index in audits[dataset]["selected_budget_indices"]:
            locked_budget = locked_result["budgets"][str(budget_index)]
            default_budget = default_result["budgets"][str(budget_index)]
            locked_repeats = _repeat_rmse_array(
                locked_budget["rmse_by_repeat"],
                path=f"{dataset}.locked.budget={budget_index}",
            )
            default_repeats = _repeat_rmse_array(
                default_budget["default_rmse_by_repeat"],
                path=f"{dataset}.default.budget={budget_index}",
            )
            orbit_repeats = _repeat_rmse_array(
                locked_budget["orbit_rmse_by_repeat"],
                path=f"{dataset}.orbit.budget={budget_index}",
            )
            _require(
                locked_repeats.size == int(audits[dataset]["repeats"]),
                f"{dataset} budget {budget_index} has wrong repeat count",
            )
            rng = np.random.default_rng(
                np.random.SeedSequence(
                    [
                        BOOTSTRAP_BASE_SEED,
                        validation_base_seed,
                        dataset_position,
                        int(budget_index),
                    ]
                )
            )
            default_draws, orbit_draws = _bootstrap_cell_ratio_draws(
                locked_repeats,
                default_repeats,
                orbit_repeats,
                rng=rng,
            )
            locked_ratio_default = float(
                locked_budget["aggregate_rmse_ratio_to_registered_default"]
            )
            locked_ratio_orbit = float(
                locked_budget["aggregate_rmse_ratio_to_inside_orbit"]
            )
            # The validator already recomputed these point estimates.  This
            # secondary check ensures the repeat arrays used by bootstrap are
            # exactly the arrays underlying those aggregates.
            _require(
                math.isclose(
                    locked_ratio_default,
                    _root_mean_square(locked_repeats)
                    / _root_mean_square(default_repeats),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                f"{dataset} default ratio disagrees with repeat arrays",
            )
            _require(
                math.isclose(
                    locked_ratio_orbit,
                    _root_mean_square(locked_repeats)
                    / _root_mean_square(orbit_repeats),
                    rel_tol=1e-12,
                    abs_tol=1e-15,
                ),
                f"{dataset} Orbit ratio disagrees with repeat arrays",
            )
            joint_default_draws.append(default_draws)
            joint_orbit_draws.append(orbit_draws)
            joint_default_points.append(locked_ratio_default)
            joint_orbit_points.append(locked_ratio_orbit)
            points.append(
                {
                    "budget_index": budget_index,
                    "total_utility_calls": int(total_budgets[budget_index]),
                    "locked_best_rmse": locked_budget["aggregate_rmse"],
                    "registered_default_rmse": default_budget["aggregate_rmse"],
                    "inside_orbit_rmse": locked_budget["orbit_aggregate_rmse"],
                    "locked_ratio_to_default": locked_ratio_default,
                    "locked_ratio_to_orbit": locked_ratio_orbit,
                    "locked_ratio_to_default_bootstrap_95_percent_ci": (
                        _bootstrap_interval(default_draws)
                    ),
                    "locked_ratio_to_orbit_bootstrap_95_percent_ci": (
                        _bootstrap_interval(orbit_draws)
                    ),
                }
            )
        curves[dataset] = points
    joint_default = _joint_geometric_draws(joint_default_draws)
    joint_orbit = _joint_geometric_draws(joint_orbit_draws)
    diagnostic_configs = [
        {"lambda0": lambda0, "candidate_pool": pool}
        for lambda0, pool in sorted(
            additional_configs or set(), key=lambda item: (item[1], item[0])
        )
    ]
    result["status"] = "holdout_validated"
    result["holdout_validation"] = {
        "sources": sources,
        "locked_selection_source": (
            dict(locked_selection_source)
            if locked_selection_source is not None
            else None
        ),
        "base_seed": audits["airport"]["base_seed"],
        "repeats": audits["airport"]["repeats"],
        "screening_and_validation_seed_sets_disjoint": True,
        "validation_configuration_matches_locked_selection": True,
        "temporal_precedence_is_not_inferred_from_file_timestamps": True,
        "locked_best_and_registered_default_present": True,
        "additional_predeclared_diagnostic_challengers": diagnostic_configs,
        "main_curves_ignore_additional_challengers": True,
        "bootstrap": {
            "unit": "repeat within each dataset-budget cell",
            "comparison_resampling": "paired indices across compared methods",
            "cells_resampled_independently_for_joint_summary": True,
            "samples": BOOTSTRAP_SAMPLES,
            "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "joint_cell_count": len(joint_default_draws),
        },
        "joint_geometric_mean_locked_ratio_to_default": (
            _geometric_mean_nonnegative(joint_default_points)
        ),
        "joint_geometric_mean_locked_ratio_to_default_bootstrap_95_percent_ci": (
            _bootstrap_interval(joint_default)
        ),
        "joint_geometric_mean_locked_ratio_to_orbit": (
            _geometric_mean_nonnegative(joint_orbit_points)
        ),
        "joint_geometric_mean_locked_ratio_to_orbit_bootstrap_95_percent_ci": (
            _bootstrap_interval(joint_orbit)
        ),
        "curves": curves,
    }
    return result


def _heatmap_matrix(
    selection: Mapping[str, Any], panel: str
) -> tuple[list[float], list[int], np.ndarray, np.ndarray]:
    rows = selection["ranking"]
    lambdas = sorted({float(row["lambda0"]) for row in rows})
    pools = sorted({int(row["candidate_pool"]) for row in rows})
    values = np.full((len(pools), len(lambdas)), np.nan)
    failures = np.zeros_like(values, dtype=bool)
    for row in rows:
        y = pools.index(int(row["candidate_pool"]))
        x = lambdas.index(float(row["lambda0"]))
        if panel in DATASETS:
            value = row["dataset_geometric_mean_rmse_ratio_to_default"][panel]
        else:
            value = row["joint_geometric_mean_rmse_ratio_to_default"]
        if not row["selection_eligible"] or value is None:
            failures[y, x] = True
        else:
            values[y, x] = float(value)
    return lambdas, pools, values, failures


def plot_heatmaps(selection: Mapping[str, Any], output_png: Path) -> tuple[Path, Path]:
    """Render the registered joint K-by-lambda screening score."""
    lambdas, pools, matrix, failures = _heatmap_matrix(selection, "joint")
    log_values = np.log(matrix[np.isfinite(matrix)])
    max_abs = float(np.max(np.abs(log_values)))
    max_abs = max(max_abs, 1e-6)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    fig, axis = plt.subplots(figsize=(6.6, 5.9), constrained_layout=True)
    locked = (
        float(selection["locked_configuration"]["lambda0"]),
        int(selection["locked_configuration"]["candidate_pool"]),
    )
    log_matrix = np.ma.masked_invalid(np.log(matrix))
    HEATMAP_CMAP.set_bad("#ECEFF2")
    image = axis.imshow(
        log_matrix,
        origin="lower",
        cmap=HEATMAP_CMAP,
        norm=norm,
        aspect="auto",
    )
    axis.set_xticks(range(len(lambdas)), [f"{value:g}" for value in lambdas])
    axis.set_yticks(range(len(pools)), [str(value) for value in pools])
    axis.set_xlabel(r"$\lambda_0$", fontsize=15)
    axis.set_ylabel("K", fontsize=15)
    axis.set_title("INSIDE-Greedy screening", fontsize=17, weight="semibold", pad=10)
    axis.tick_params(labelsize=13)
    for y, pool in enumerate(pools):
        for x, lambda0 in enumerate(lambdas):
            if failures[y, x]:
                axis.text(x, y, "FAIL", ha="center", va="center", fontsize=11)
            elif np.isfinite(matrix[y, x]):
                value = matrix[y, x]
                axis.text(
                    x,
                    y,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color=CHARCOAL,
                    weight=("bold" if (lambda0, pool) == locked else "normal"),
                )
            else:
                axis.text(x, y, "—", ha="center", va="center", color=GREY)
            if (lambda0, pool) == locked:
                axis.add_patch(
                    Rectangle(
                        (x - 0.48, y - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor=CHARCOAL,
                        linewidth=2.8,
                    )
                )
            if (lambda0, pool) == DEFAULT:
                axis.add_patch(
                    Rectangle(
                        (x - 0.39, y - 0.39),
                        0.78,
                        0.78,
                        fill=False,
                        edgecolor=CHARCOAL,
                        linewidth=2.0,
                        linestyle="--",
                    )
                )
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.35)
        spine.set_color(CHARCOAL)
    colorbar = fig.colorbar(image, ax=axis, shrink=0.88, pad=0.03)
    ticks = np.linspace(-max_abs, max_abs, 5)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{math.exp(value):.3f}" for value in ticks])
    colorbar.set_label(
        "Joint RMSE ratio to default  (<1 is better)",
        fontsize=12,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_png, output_pdf


def _plot_one_validation_curve(
    selection: Mapping[str, Any], dataset: str, output_png: Path
) -> tuple[Path, Path]:
    holdout = selection.get("holdout_validation")
    _require(isinstance(holdout, Mapping), "selection has no holdout validation")
    points = holdout["curves"][dataset]
    x = np.asarray([row["total_utility_calls"] for row in points], dtype=float)
    locked_values = np.asarray([row["locked_best_rmse"] for row in points], dtype=float)
    default_values = np.asarray(
        [row["registered_default_rmse"] for row in points], dtype=float
    )
    orbit_values = np.asarray([row["inside_orbit_rmse"] for row in points], dtype=float)
    locked = selection["locked_configuration"]
    locked_is_default = (
        float(locked["lambda0"]), int(locked["candidate_pool"])
    ) == DEFAULT

    fig, axis = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
    axis.plot(
        x,
        locked_values,
        color=BLUE,
        linewidth=3.0,
        marker="o",
        markersize=8,
        label=(
            "INSIDE-Greedy"
            if locked_is_default
            else (
                f"INSIDE-Greedy ($\\lambda_0$={locked['lambda0']:g}, "
                f"K={locked['candidate_pool']})"
            )
        ),
    )
    if not locked_is_default:
        axis.plot(
            x,
            default_values,
            color=GREY,
            linewidth=2.6,
            linestyle="--",
            marker="s",
            markersize=7,
            markerfacecolor="white",
            label=(
                f"INSIDE-Greedy ($\\lambda_0$={DEFAULT[0]:g}, "
                f"K={DEFAULT[1]})"
            ),
        )
    axis.plot(
        x,
        orbit_values,
        color=ORANGE,
        linewidth=2.8,
        linestyle="-.",
        marker="^",
        markersize=8,
        markerfacecolor="white",
        label="INSIDE-Orbit",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Total utility evaluations", fontsize=14)
    axis.set_ylabel("RMSE", fontsize=14)
    axis.set_title(
        DATASET_LABELS[dataset],
        fontsize=16,
        weight="semibold",
        pad=12,
    )
    axis.grid(True, which="major", color=LIGHT_GREY, linewidth=0.9)
    axis.grid(False, which="minor")
    axis.tick_params(axis="both", which="major", labelsize=12, width=1.1)
    axis.legend(frameon=False, fontsize=11, loc="best")
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_color(CHARCOAL)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf = output_png.with_suffix(".pdf")
    fig.savefig(output_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_png, output_pdf


def plot_validation_curves(
    selection: Mapping[str, Any], output_prefix: Path
) -> dict[str, list[str]]:
    outputs: dict[str, list[str]] = {}
    for dataset in DATASETS:
        png = output_prefix.with_name(f"{output_prefix.name}_{dataset}.png")
        png_path, pdf_path = _plot_one_validation_curve(selection, dataset, png)
        outputs[dataset] = [str(png_path), str(pdf_path)]
    return outputs


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--airport-screening", type=Path, required=True)
    parser.add_argument("--voting-screening", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heatmap", type=Path, required=True)
    parser.add_argument(
        "--locked-selection",
        type=Path,
        help=(
            "previously saved screening selection; required when attaching "
            "held-out validation"
        ),
    )
    parser.add_argument("--airport-validation", type=Path)
    parser.add_argument("--voting-validation", type=Path)
    parser.add_argument("--validation-plot-prefix", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recomputed_selection = build_joint_selection(
        args.airport_screening, args.voting_screening
    )
    locked_source: dict[str, Any] | None = None
    if args.locked_selection is None:
        selection = recomputed_selection
    else:
        selection_value = _read(args.locked_selection)
        selection = json.loads(json.dumps(selection_value))
        _require(
            selection.get("status") == "configuration_locked",
            "locked-selection artifact is not a pre-holdout selection",
        )
        for field in ("selection_rule", "screening_sources", "locked_configuration"):
            _require(
                selection.get(field) == recomputed_selection.get(field),
                f"locked-selection {field} disagrees with screening recomputation",
            )
        locked_source = {
            "path": str(args.locked_selection.resolve()),
            "sha256": _sha256(args.locked_selection),
        }
    validation_paths = (args.airport_validation, args.voting_validation)
    _require(
        (validation_paths[0] is None) == (validation_paths[1] is None),
        "provide both Airport and Voting validation reports or neither",
    )
    if validation_paths[0] is not None:
        _require(
            locked_source is not None,
            "held-out validation requires --locked-selection from screening",
        )
        selection = attach_holdout_validation(
            selection,
            validation_paths[0],
            validation_paths[1],  # type: ignore[arg-type]
            locked_selection_source=locked_source,
        )
    heatmap_png, heatmap_pdf = plot_heatmaps(selection, args.heatmap)
    selection["plots"] = {
        "screening_heatmaps": [str(heatmap_png), str(heatmap_pdf)]
    }
    if args.validation_plot_prefix is not None:
        _require(
            selection.get("holdout_validation") is not None,
            "validation plot prefix requires validation reports",
        )
        selection["plots"]["holdout_curves"] = plot_validation_curves(
            selection, args.validation_plot_prefix
        )
    _write_json(args.output, selection)
    locked = selection["locked_configuration"]
    print(
        f"locked {locked['configuration_id']}; joint ratio="
        f"{locked['joint_geometric_mean_rmse_ratio_to_default']:.6g}"
    )


if __name__ == "__main__":
    main()

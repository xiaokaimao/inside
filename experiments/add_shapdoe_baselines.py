"""Append Yang et al. LS/COA baselines to the current audited comparisons.

python -m experiments.add_shapdoe_baselines --datasets airport voting wine cancer --jobs 32

Original reports are embedded unchanged. New method/budget/repeat cells are
checkpointed atomically. COA points that cannot fund a complete design are
explicitly unavailable; no partial COA is passed off as the published method.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.iris_sklearn_game import SklearnClassificationGame
from experiments.run_analytic_inside_comparison import AnalyticGame
from experiments.run_wine_inside_comparison import validate_reconstructed_dataset
from experiments.sklearn_data import load_sklearn_train_test_split
from experiments.validate_inside_comparison import validate_report as validate_base_report
from frame_ofa import GameEvaluator, estimate_shapdoe, shapdoe_budget


SOURCE_PATHS = {
    "airport": Path("results/json/airport_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_50k_1m.json"),
    "voting": Path("results/json/voting_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_25k_510k.json"),
    "wine": Path("results/json/wine_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_71k_1p42m.json"),
    "cancer": Path("results/json/cancer_inside_comparison_per_size_ratio_k64_lambda1over16_3repeats_228k_4p55m.json"),
}
METHOD_LABELS = {"shapdoe_ls": "ShapDoE-LS", "shapdoe_coa": "ShapDoE-COA"}
EXPERIMENT_ID = "inside_with_shapdoe_1_0_0_v1"
BASE_SEED = 20260908


def _default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=_default,
                                    allow_nan=False).encode()).hexdigest()


def _atomic_write(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True,
                                        default=_default, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _dataset_name(source: Mapping[str, Any]) -> str:
    return str(source["game"]["dataset"])


def _make_game(source: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    dataset = _dataset_name(source)
    if dataset in {"airport", "voting"}:
        return AnalyticGame, {"dataset": dataset}
    metadata = source["dataset"]
    arguments, reconstructed = load_sklearn_train_test_split(
        dataset, test_size=metadata["test_size"], dataset_seed=metadata["dataset_seed"])
    validate_reconstructed_dataset(metadata, reconstructed)
    if source["game"]["model"] != "sklearn.svm.SVC(C=1.0, kernel='rbf', gamma='scale')":
        raise ValueError("unsupported source classification utility")
    return SklearnClassificationGame, arguments | {"model": "rbf_svm", "regularization": 1.0}


def _seed(method: str, budget_index: int, repeat: int) -> int:
    return BASE_SEED + (1000000 if method == "coa" else 0) + 1000 * budget_index + repeat


def _summary(cells: list[Mapping[str, Any]], truth: np.ndarray) -> dict[str, Any]:
    estimates = np.asarray([c["estimate"] for c in cells], dtype=np.float64)
    mse = np.mean((estimates - truth) ** 2, axis=1)
    rng = np.random.default_rng(1741)
    bootstrap = np.sqrt(mse[rng.integers(0, len(cells), (5000, len(cells)))].mean(axis=1))
    calls = [c["diagnostics"]["utility_evaluations"] for c in cells]
    return {
        "status": "complete", "estimates": estimates.tolist(),
        "aggregate_rmse": float(np.sqrt(mse.mean())),
        "aggregate_rmse_bootstrap_95": np.quantile(bootstrap, [.025, .975]).tolist(),
        "mean_repeat_rmse": float(np.sqrt(mse).mean()),
        "actual_utility_calls_by_repeat": calls,
        "mean_actual_utility_calls": float(np.mean(calls)),
        "mean_elapsed_seconds": float(np.mean([c["elapsed_seconds"] for c in cells])),
        "diagnostics_by_repeat": [c["diagnostics"] for c in cells],
    }


def _assemble(report: dict[str, Any]) -> None:
    source = report["base_report"]
    truth = np.asarray(source["ground_truth"]["values"], dtype=np.float64)
    assembled = copy.deepcopy(source["results_by_inner_budget"])
    completed = True
    for index, (key, row) in enumerate(sorted(assembled.items(), key=lambda item: int(item[0]))):
        cap = int(row["total_utility_calls_per_estimate"])
        for method in report["configuration"]["added_methods"]:
            name = "shapdoe_" + method
            plan = shapdoe_budget(len(truth), cap, method)
            if cap < plan.minimum_call_budget:
                summary = {"status": "budget_infeasible", "minimum_call_budget": plan.minimum_call_budget,
                           "field_order": plan.field_order, "reason": "one complete design exceeds the call cap"}
            else:
                cells = [report["cells"][f"{method}/{index}/{repeat}"]
                         for repeat in range(3) if f"{method}/{index}/{repeat}" in report["cells"]]
                if len(cells) == 3:
                    summary = _summary(cells, truth)
                else:
                    summary = {"status": "pending", "completed_repeats": len(cells)}
                    completed = False
            row["methods"][name] = {"label": METHOD_LABELS[name], **summary}
    report["results_by_inner_budget"] = assembled
    report["status"] = "complete" if completed else "running"


def validate_report(report: Mapping[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    """Recompute new metrics/budgets and validate the retained original report."""
    if report["experiment"] != EXPERIMENT_ID:
        raise ValueError("unsupported ShapDoE report version")
    source = report["base_report"]
    base_audit = validate_base_report(source)
    config = report["configuration"]
    if (config["added_methods"] != ["ls", "coa"] or config["repeats"] != 3
            or config["seed"] != BASE_SEED or config["dataset"] != _dataset_name(source)):
        raise ValueError("ShapDoE experiment configuration mismatch")
    if _digest(source) != report["source"]["canonical_json_sha256"]:
        raise ValueError("embedded source report fingerprint mismatch")
    if require_complete and report["status"] != "complete":
        raise ValueError("ShapDoE experiment is incomplete")
    truth = np.asarray(source["ground_truth"]["values"], dtype=np.float64)
    checked, infeasible = 0, 0
    allowed_cells = set()
    expected_keys = set(source["results_by_inner_budget"])
    if set(report["results_by_inner_budget"]) != expected_keys:
        raise ValueError("budget grid changed")
    for index, (key, original) in enumerate(sorted(source["results_by_inner_budget"].items(), key=lambda item: int(item[0]))):
        row = report["results_by_inner_budget"][key]
        if {k: v for k, v in row.items() if k != "methods"} != {k: v for k, v in original.items() if k != "methods"}:
            raise ValueError("source budget metadata changed")
        for name, summary in original["methods"].items():
            if summary != row["methods"][name]:
                raise ValueError("an existing baseline was modified")
        expected_methods = set(original["methods"]) | {"shapdoe_" + m for m in report["configuration"]["added_methods"]}
        if set(row["methods"]) != expected_methods:
            raise ValueError("unexpected method set")
        cap = int(row["total_utility_calls_per_estimate"])
        for method in report["configuration"]["added_methods"]:
            summary = row["methods"]["shapdoe_" + method]
            plan = shapdoe_budget(len(truth), cap, method)
            if cap < plan.minimum_call_budget:
                if summary["status"] != "budget_infeasible" or summary["minimum_call_budget"] != plan.minimum_call_budget:
                    raise ValueError("invalid infeasibility claim")
                infeasible += 1
                continue
            keys = [f"{method}/{index}/{repeat}" for repeat in range(3)]
            allowed_cells.update(keys)
            cells = [report["cells"][key] for key in keys if key in report["cells"]]
            for repeat, key in enumerate(keys):
                if key not in report["cells"]:
                    continue
                cell = report["cells"][key]
                diagnostic = cell["diagnostics"]
                if cell["seed"] != _seed(method, index, repeat) or diagnostic["budget"] != asdict(plan):
                    raise ValueError("seed or complete-design budget mismatch")
                for field in ["utility_evaluations", "unused_calls", "num_designs", "num_permutations"]:
                    if diagnostic[field] != getattr(plan, field):
                        raise ValueError("ShapDoE diagnostic call accounting mismatch")
                if diagnostic["truncation"] or diagnostic["efficiency_projection"] or not diagnostic["boundary_reuse"]:
                    raise ValueError("ShapDoE estimator identity changed")
                vector = np.asarray(cell["estimate"], dtype=np.float64)
                if vector.shape != truth.shape or not np.isfinite(vector).all():
                    raise ValueError("invalid estimate vector")
                target = diagnostic["full_utility"] - diagnostic["empty_utility"]
                if not np.isclose(target, truth.sum(), atol=1e-9, rtol=1e-9):
                    raise ValueError("utility endpoints disagree with the source game")
                if not np.isclose(vector.sum(), target, atol=1e-10, rtol=1e-10):
                    raise ValueError("complete paths violate efficiency")
                if not np.isfinite(cell["elapsed_seconds"]) or cell["elapsed_seconds"] < 0:
                    raise ValueError("invalid elapsed time")
            if len(cells) < 3:
                if require_complete or summary != {"label": METHOD_LABELS["shapdoe_" + method],
                        "status": "pending", "completed_repeats": len(cells)}:
                    raise ValueError("incomplete ShapDoE repeats")
                continue
            expected = _summary(cells, truth)
            if {k: summary[k] for k in expected} != expected:
                raise ValueError("stored ShapDoE metrics differ from recomputation")
            checked += 1
    if not set(report["cells"]).issubset(allowed_cells):
        raise ValueError("unexpected or infeasible checkpoint cells")
    return {"status": "passed", "completed_method_budget_cells": checked,
            "infeasible_method_budget_cells": infeasible, "base_report_audit": base_audit}


def run_dataset(dataset: str, *, jobs: int, output_dir: Path, source_path: Path | None = None,
                budget_indices: tuple[int, ...] = (0, 1, 2, 3, 4)) -> Path:
    if not budget_indices or not set(budget_indices).issubset(range(5)):
        raise ValueError("budget_indices must select indices from 0 through 4")
    source_path = source_path or SOURCE_PATHS[dataset]
    source = json.loads(source_path.read_text())
    validate_base_report(source)
    if _dataset_name(source) != dataset:
        raise ValueError("source belongs to a different dataset")
    output = output_dir / f"{dataset}_inside_with_shapdoe_3repeats.json"
    if output.resolve() == source_path.resolve():
        raise ValueError("output must differ from the source")
    fingerprint = _digest(source)
    if output.exists():
        report = json.loads(output.read_text())
        if report["source"]["canonical_json_sha256"] != fingerprint:
            raise ValueError("source changed since the checkpoint was created")
        validate_report(report, require_complete=False)
    else:
        report = {
            "experiment": EXPERIMENT_ID, "status": "running",
            "base_report": source, "cells": {},
            "source": {"path": str(source_path), "canonical_json_sha256": fingerprint,
                       "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
            "configuration": {"dataset": dataset, "added_methods": ["ls", "coa"],
                              "repeats": 3, "seed": BASE_SEED, "call_budget_semantics": "complete designs under physical utility-call cap",
                              "timing": "fresh pool setup, design, utility evaluation, aggregation and pool shutdown for each cell",
                              "original_curve_timings_are_not_remeasured": True},
            "official_source": json.loads(Path("third_party/shapdoe/provenance.json").read_text()),
        }
        _assemble(report)
        _atomic_write(output, report)
    game_factory, game_args = _make_game(source)
    n = len(source["ground_truth"]["values"])
    for index, row in enumerate(sorted(source["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"])):
        if index not in budget_indices:
            continue
        cap = int(row["total_utility_calls_per_estimate"])
        for method in ["ls", "coa"]:
            if cap < shapdoe_budget(n, cap, method).minimum_call_budget:
                continue
            for repeat in range(3):
                key = f"{method}/{index}/{repeat}"
                if key in report["cells"]:
                    continue
                seed = _seed(method, index, repeat)
                started = time.perf_counter()
                with GameEvaluator(game_factory, game_args, n_jobs=jobs, start_method="spawn") as evaluator:
                    result = estimate_shapdoe(evaluator, n, cap, seed, method=method,
                                             num_tasks=max(128, jobs * 4))
                report["cells"][key] = {"estimate": result.values.tolist(), "seed": seed,
                    "elapsed_seconds": time.perf_counter() - started,
                    "execution": {"jobs": jobs, "start_method": "spawn", "worker_threads": 1},
                    "diagnostics": json.loads(json.dumps(asdict(result.diagnostics), default=_default))}
                _assemble(report)
                _atomic_write(output, report)
                print(f"{dataset}: {key} calls={result.diagnostics.utility_evaluations} "
                      f"RMSE={np.sqrt(np.mean((result.values - source['ground_truth']['values'])**2)):.6g}", flush=True)
    _assemble(report)
    if report["status"] != "complete":
        report["status"] = "partial"
    report["last_requested_budget_indices"] = list(budget_indices)
    _atomic_write(output, report)
    audit = validate_report(report, require_complete=set(budget_indices) == set(range(5)))
    audit["report_status"] = report["status"]
    _atomic_write(output.with_suffix(".validation.json"), audit)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCE_PATHS), default=["airport", "voting", "wine", "cancer"])
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("results/json"))
    parser.add_argument("--source", type=Path, help="Override source for one dataset")
    parser.add_argument("--budget-indices", type=int, nargs="+", choices=range(5), default=list(range(5)),
                        help="Run only these budget points; other feasible points remain explicitly pending")
    args = parser.parse_args()
    if args.jobs < 1 or (args.source and len(args.datasets) != 1):
        parser.error("jobs must be positive; --source requires one dataset")
    for dataset in args.datasets:
        print(run_dataset(dataset, jobs=args.jobs, output_dir=args.output_dir, source_path=args.source,
                          budget_indices=tuple(args.budget_indices)), flush=True)


if __name__ == "__main__":
    main()

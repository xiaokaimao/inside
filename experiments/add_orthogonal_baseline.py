"""Append Mitchell et al.'s Orthogonal baseline to audited ShapDoE comparisons.

python -m experiments.add_orthogonal_baseline --datasets airport voting --jobs 16
The source report (including pending ShapDoE points) is retained unchanged.
New estimates are checkpointed separately and can resume by budget index.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from experiments.add_shapdoe_baselines import (
    _atomic_write, _default, _digest, _make_game, _summary,
    validate_report as validate_source,
)
from frame_ofa import GameEvaluator, estimate_orthogonal_shapley, orthogonal_budget

EXPERIMENT_ID = "inside_with_shapdoe_and_orthogonal_v1"
BASE_SEED = 20260909
SOURCE_PATHS = {name: Path(f"results/json/{name}_inside_with_shapdoe_3repeats.json")
                for name in ("airport", "voting", "wine", "cancer")}
PROVENANCE = Path(__file__).resolve().parents[1] / "third_party/shap_sampling/provenance.json"


def _seed(index: int, repeat: int) -> int:
    return BASE_SEED + 1000 * index + repeat


def _assemble(report: dict[str, Any]) -> None:
    source = report["source_report"]
    truth = np.asarray(source["base_report"]["ground_truth"]["values"])
    rows = copy.deepcopy(source["results_by_inner_budget"])
    complete = True
    for index, (key, row) in enumerate(sorted(rows.items(), key=lambda item: int(item[0]))):
        plan = orthogonal_budget(len(truth), row["total_utility_calls_per_estimate"])
        cells = [report["cells"][f"{index}/{repeat}"] for repeat in range(3)
                 if f"{index}/{repeat}" in report["cells"]]
        if plan.target_call_budget < plan.minimum_call_budget:
            summary = {"status": "budget_infeasible", "minimum_call_budget": plan.minimum_call_budget}
        elif len(cells) == 3:
            summary = _summary(cells, truth)
        else:
            summary = {"status": "pending", "completed_repeats": len(cells)}
            complete = False
        row["methods"]["orthogonal"] = {"label": "Orthogonal", **summary}
    report["results_by_inner_budget"] = rows
    report["orthogonal_status"] = "complete" if complete else "partial"
    report["status"] = "complete" if complete and source["status"] == "complete" else "partial"


def validate_report(report: Mapping[str, Any], *, require_complete: bool = True) -> dict[str, Any]:
    if report["experiment"] != EXPERIMENT_ID:
        raise ValueError("unsupported Orthogonal report version")
    source = report["source_report"]
    audit = validate_source(source, require_complete=False)
    if _digest(source) != report["source"]["canonical_json_sha256"]:
        raise ValueError("source report fingerprint mismatch")
    expected_config = {"dataset": source["configuration"]["dataset"], "added_methods": ["orthogonal"],
                       "repeats": 3, "seed": BASE_SEED,
                       "call_budget_semantics": "complete reverse pairs; partial final orthogonal basis allowed",
                       "timing": "fresh pool setup, sampling, utility evaluation, aggregation and pool shutdown for each cell"}
    if report["configuration"] != expected_config:
        raise ValueError("Orthogonal configuration changed")
    if report["official_source"] != json.loads(PROVENANCE.read_text()):
        raise ValueError("Orthogonal source provenance changed")
    truth = np.asarray(source["base_report"]["ground_truth"]["values"])
    allowed, checked = set(), 0
    for index, row in enumerate(sorted(source["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"])):
        plan = orthogonal_budget(len(truth), row["total_utility_calls_per_estimate"])
        if plan.target_call_budget < plan.minimum_call_budget:
            continue
        for repeat in range(3):
            key = f"{index}/{repeat}"
            allowed.add(key)
            if key not in report["cells"]:
                continue
            cell = report["cells"][key]
            diagnostic = cell["diagnostics"]
            if cell["seed"] != _seed(index, repeat) or diagnostic["budget"] != asdict(plan):
                raise ValueError("Orthogonal seed or budget mismatch")
            for field in ("utility_evaluations", "unused_calls", "num_permutations"):
                if diagnostic[field] != getattr(plan, field):
                    raise ValueError("Orthogonal call accounting mismatch")
            if (diagnostic["truncation"] or diagnostic["efficiency_projection"] or not diagnostic["boundary_reuse"]
                    or diagnostic["sampling"] != "orthogonal_spherical_codes_with_antipodes"
                    or not diagnostic["allows_partial_basis"]
                    or diagnostic["uncertainty_unit"] != "independent_estimator_repeats"):
                raise ValueError("Orthogonal estimator identity changed")
            values = np.asarray(cell["estimate"])
            target = diagnostic["full_utility"] - diagnostic["empty_utility"]
            if (values.shape != truth.shape or not np.isfinite(values).all()
                    or not np.isclose(target, truth.sum(), rtol=1e-9, atol=1e-9)
                    or not np.isclose(values.sum(), target, rtol=1e-10, atol=1e-10)):
                raise ValueError("Orthogonal estimate or endpoints violate efficiency")
            if not np.isfinite(cell["elapsed_seconds"]) or cell["elapsed_seconds"] < 0:
                raise ValueError("invalid Orthogonal elapsed time")
            checked += 1
    if not set(report["cells"]).issubset(allowed):
        raise ValueError("unexpected Orthogonal checkpoint cells")
    reconstructed = copy.deepcopy(dict(report))
    _assemble(reconstructed)
    if reconstructed["results_by_inner_budget"] != report["results_by_inner_budget"]:
        raise ValueError("stored metrics or existing baselines differ from recomputation")
    if any(reconstructed[k] != report[k] for k in ("status", "orthogonal_status")):
        raise ValueError("report completion status mismatch")
    if require_complete and report["status"] != "complete":
        raise ValueError("comparison is incomplete")
    return {"status": "passed", "orthogonal_repeat_cells": checked,
            "orthogonal_status": report["orthogonal_status"], "report_status": report["status"],
            "source_audit": audit}


def run_dataset(dataset: str, *, jobs: int, output_dir: Path,
                source_path: Path | None = None, budget_indices: tuple[int, ...] = (0, 1, 2, 3, 4)) -> Path:
    if jobs < 1 or not budget_indices or not set(budget_indices).issubset(range(5)):
        raise ValueError("jobs must be positive and budget_indices must select 0 through 4")
    source_path = source_path or SOURCE_PATHS[dataset]
    source = json.loads(source_path.read_text())
    validate_source(source, require_complete=False)
    if source["configuration"]["dataset"] != dataset:
        raise ValueError("source belongs to a different dataset")
    output = output_dir / f"{dataset}_inside_with_shapdoe_orthogonal_3repeats.json"
    if output.resolve() == source_path.resolve():
        raise ValueError("output must differ from source")
    fingerprint = _digest(source)
    if output.exists():
        report = json.loads(output.read_text())
        if report["source"]["canonical_json_sha256"] != fingerprint:
            raise ValueError("source changed since checkpoint creation")
        validate_report(report, require_complete=False)
    else:
        report = {"experiment": EXPERIMENT_ID, "source_report": source, "cells": {},
                  "source": {"path": str(source_path), "canonical_json_sha256": fingerprint,
                             "file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest()},
                  "configuration": {"dataset": dataset, "added_methods": ["orthogonal"],
                                    "repeats": 3, "seed": BASE_SEED,
                                    "call_budget_semantics": "complete reverse pairs; partial final orthogonal basis allowed",
                                    "timing": "fresh pool setup, sampling, utility evaluation, aggregation and pool shutdown for each cell"},
                  "official_source": json.loads(PROVENANCE.read_text())}
        _assemble(report)
        _atomic_write(output, report)
    base = source["base_report"]
    factory, arguments = _make_game(base)
    truth = np.asarray(base["ground_truth"]["values"])
    for index, row in enumerate(sorted(source["results_by_inner_budget"].values(), key=lambda r: r["inner_utility_calls"])):
        if index not in budget_indices:
            continue
        cap = row["total_utility_calls_per_estimate"]
        if cap < orthogonal_budget(len(truth), cap).minimum_call_budget:
            continue
        for repeat in range(3):
            key, seed = f"{index}/{repeat}", _seed(index, repeat)
            if key in report["cells"]:
                continue
            started = time.perf_counter()
            with GameEvaluator(factory, arguments, n_jobs=jobs, start_method="spawn") as evaluator:
                result = estimate_orthogonal_shapley(evaluator, len(truth), cap, seed, num_tasks=max(128, jobs * 4))
            report["cells"][key] = {"estimate": result.values.tolist(), "seed": seed,
                "elapsed_seconds": time.perf_counter() - started,
                "execution": {"jobs": jobs, "start_method": "spawn", "worker_threads": 1},
                "diagnostics": json.loads(json.dumps(asdict(result.diagnostics), default=_default))}
            _assemble(report)
            _atomic_write(output, report)
            print(f"{dataset}: orthogonal/{key} calls={result.diagnostics.utility_evaluations} "
                  f"RMSE={np.sqrt(np.mean((result.values - truth)**2)):.6g}", flush=True)
    report["last_requested_budget_indices"] = list(budget_indices)
    _atomic_write(output, report)
    _atomic_write(output.with_suffix(".validation.json"), validate_report(report, require_complete=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCE_PATHS), default=list(SOURCE_PATHS))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--source", type=Path, help="Override the augmented source report for one dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("results/json"))
    parser.add_argument("--budget-indices", type=int, nargs="+", choices=range(5), default=list(range(5)))
    args = parser.parse_args()
    if args.jobs < 1 or (args.source and len(args.datasets) != 1):
        parser.error("jobs must be positive; --source requires one dataset")
    for dataset in args.datasets:
        print(run_dataset(dataset, jobs=args.jobs, output_dir=args.output_dir,
                          source_path=args.source, budget_indices=tuple(args.budget_indices)), flush=True)


if __name__ == "__main__":
    main()

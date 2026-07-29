"""Microbenchmark alternative Iris coalition utilities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .iris_game import IrisLogisticGame, load_official_iris_split
from .iris_sklearn_game import IrisSklearnGame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=551)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    game_args, _ = load_official_iris_split()
    num_players = len(game_args["y_valued"])
    rng = np.random.default_rng(args.seed)
    rows = rng.random((args.calls, num_players)) < rng.uniform(
        0.05, 0.95, size=(args.calls, 1)
    )

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    factories = {
        "upstream_style_torch_lr_cross_entropy": (
            IrisLogisticGame,
            game_args,
        ),
        "sklearn_lr_accuracy": (
            IrisSklearnGame,
            game_args | {"model": "lr"},
        ),
        "sklearn_linear_svm_accuracy": (
            IrisSklearnGame,
            game_args | {"model": "linear_svm"},
        ),
        "sklearn_rbf_svm_accuracy": (
            IrisSklearnGame,
            game_args | {"model": "rbf_svm"},
        ),
    }
    report = {}
    for name, (factory, factory_args) in factories.items():
        game = factory(**factory_args)
        start = time.perf_counter()
        values = np.asarray([game.evaluate(row) for row in rows])
        seconds = time.perf_counter() - start
        report[name] = {
            "calls": args.calls,
            "seconds": seconds,
            "calls_per_second": args.calls / seconds,
            "utility_mean": float(values.mean()),
            "utility_std": float(values.std()),
        }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

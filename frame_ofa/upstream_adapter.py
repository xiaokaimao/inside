"""Adapter for utility classes used by the official OFA experiment code."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from .estimator import (
    EstimateResult,
    FrameOFAEstimator,
    boundary_coalitions,
    boundary_from_utilities,
    estimate_coupled,
    estimate_stratified,
)
from .parallel import evaluate_game_coalitions


def estimate_upstream_game(
    *,
    game_func: Callable[..., Any],
    game_args: dict[str, Any],
    num_players: int,
    nue_avg: int,
    mode: str = "coupled",
    seed: int = 0,
    candidate_pool: int = 32,
    baseline: str = "linear",
    mean_balance: float = 0.1,
    n_jobs: int = 1,
    chunksize: int = 16,
    start_method: str = "spawn",
    worker_threads: int = 1,
) -> EstimateResult:
    """Run Frame-OFA against an official-style ``game.evaluate`` object.

    In the upstream code, ``nue_avg`` is the average number of sampled inner
    utility evaluations per player.  Thus coupled/stratified modes receive
    ``nue_avg * num_players`` rows.  Orbit mode receives ``nue_avg`` complete
    n-row orbits and has the same sampled-query count.
    """
    if nue_avg < 1:
        raise ValueError("nue_avg must be positive")
    design_budget = (
        nue_avg if mode == "orbit" else nue_avg * num_players
    )
    estimator = FrameOFAEstimator(
        num_players=num_players,
        num_samples=design_budget,
        mode=mode,
        seed=seed,
        candidate_pool=candidate_pool,
        baseline=baseline,
        mean_balance=mean_balance,
    )
    design = estimator.design()
    boundary_rows = boundary_coalitions(num_players)
    all_rows = np.concatenate([boundary_rows, design.coalitions], axis=0)
    all_utilities = evaluate_game_coalitions(
        game_func,
        game_args,
        all_rows,
        n_jobs=n_jobs,
        chunksize=chunksize,
        start_method=start_method,
        worker_threads=worker_threads,
    )
    boundary_count = len(boundary_rows)
    boundary = boundary_from_utilities(
        all_utilities[:boundary_count], num_players
    )
    utilities = all_utilities[boundary_count:]
    if mode in {"stratified", "orbit"}:
        values = estimate_stratified(
            design, utilities, boundary, baseline=baseline
        )
    else:
        values = estimate_coupled(
            design, utilities, boundary, baseline=baseline
        )
    return EstimateResult(
        values=values,
        design=design,
        utilities=utilities,
        boundary=boundary,
        utility_evaluations=len(all_rows),
    )

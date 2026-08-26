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
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
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
    candidate_pool: int | None = None,
    baseline: str = "linear",
    mean_balance: float | None = None,
    mean_balance_mode: str = "normalized",
    n_jobs: int = 1,
    chunksize: int = 16,
    start_method: str = "spawn",
    worker_threads: int = 1,
) -> EstimateResult:
    """Run Frame-OFA against an official-style ``game.evaluate`` object.

    In the upstream code, ``nue_avg`` is the average number of sampled inner
    utility evaluations per player.  Thus non-orbit modes receive
    ``nue_avg * num_players`` rows.  ``orbit`` and ``inside_orbit`` receive
    ``nue_avg`` complete n-row orbits and have the same sampled-query count.
    Omitted design hyperparameters are resolved by :class:`FrameOFAEstimator`
    according to the selected mode.
    """
    if nue_avg < 1:
        raise ValueError("nue_avg must be positive")
    design_budget = (
        nue_avg
        if mode in {"orbit", "inside_orbit"}
        else nue_avg * num_players
    )
    estimator = FrameOFAEstimator(
        num_players=num_players,
        num_samples=design_budget,
        mode=mode,
        seed=seed,
        candidate_pool=candidate_pool,
        baseline=baseline,
        mean_balance=mean_balance,
        mean_balance_mode=mean_balance_mode,
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
    if mode == "inside_greedy":
        values = estimate_official_ratio_ofa(
            design, utilities, boundary, missing="raise"
        )
    elif mode == "inside_orbit":
        values = estimate_ratio_ofa(design, utilities, boundary)
    elif mode in {"stratified", "orbit"}:
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

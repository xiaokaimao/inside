"""Player-wise stratified marginal Monte Carlo for Shapley values.

This module adapts the estimator in
``integral_shapley/src/core/traditional_methods.py`` to the persistent
``GameEvaluator`` interface used by this project.  The source routine estimates
one player at a time by sampling the predecessor coalition separately at every
cardinality.  Here all players are handled in one call and a physical utility
call budget is allocated as evenly as possible over the ``n x n``
player/cardinality strata.

Unlike OFA, CC, and full-permutation Monte Carlo, a sampled marginal
contribution is not reused for another player: every observation evaluates
both ``v(S)`` and ``v(S union {i})`` and therefore costs exactly two utility
calls.  Empty and full coalitions are deliberately not cached, matching the
call semantics of the source estimator.  No efficiency projection is applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class StratifiedMarginalEvaluator(Protocol):
    """Part of :class:`frame_ofa.parallel.GameEvaluator` used here."""

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class StratifiedMarginalDiagnostics:
    """Exact sample allocation and physical utility-call accounting."""

    num_players: int
    target_call_budget: int
    utility_evaluations: int
    unused_calls: int
    num_marginal_samples: int
    num_strata: int
    minimum_stratum_count: int
    maximum_stratum_count: int
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    boundary_reuse: bool
    efficiency_projection: bool
    stratum_counts: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class StratifiedMarginalResult:
    """A player-wise stratified Shapley estimate and diagnostics."""

    values: np.ndarray
    diagnostics: StratifiedMarginalDiagnostics


@dataclass(frozen=True)
class _StratifiedMarginalPayload:
    num_players: int
    seed: int
    cell_indices: tuple[int, ...]
    cell_counts: tuple[int, ...]


def _cell_rng(seed: int, cell_index: int) -> np.random.Generator:
    """Key each stratum independently of task/process partitioning."""
    return np.random.default_rng(
        np.random.SeedSequence(seed, spawn_key=(cell_index,))
    )


def _evaluate_stratified_marginal_task(
    game: Any,
    payload: _StratifiedMarginalPayload,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate all samples assigned to a collection of disjoint strata."""
    n = payload.num_players
    all_players = np.arange(n, dtype=np.int64)
    sums = np.zeros(len(payload.cell_indices), dtype=np.float64)
    observed_counts = np.zeros(len(payload.cell_indices), dtype=np.int64)
    utility_evaluations = 0

    for result_index, (cell_index, requested_count) in enumerate(
        zip(payload.cell_indices, payload.cell_counts)
    ):
        player, layer = divmod(int(cell_index), n)
        candidates = all_players[all_players != player]
        rng = _cell_rng(payload.seed, int(cell_index))

        for _ in range(int(requested_count)):
            coalition = np.zeros(n, dtype=bool)
            if layer:
                predecessor = rng.choice(
                    candidates,
                    size=layer,
                    replace=False,
                )
                coalition[predecessor] = True
            utility_without = float(game.evaluate(coalition))
            coalition[player] = True
            utility_with = float(game.evaluate(coalition))
            sums[result_index] += utility_with - utility_without
            observed_counts[result_index] += 1
            utility_evaluations += 2

    return (
        np.asarray(payload.cell_indices, dtype=np.int64),
        sums,
        observed_counts,
        utility_evaluations,
    )


def _balanced_stratum_counts(
    *,
    num_players: int,
    num_marginal_samples: int,
    seed: int,
) -> np.ndarray:
    """Allocate samples so any two stratum counts differ by at most one."""
    num_strata = num_players * num_players
    base, remainder = divmod(num_marginal_samples, num_strata)
    counts = np.full(num_strata, base, dtype=np.int64)
    if remainder:
        # Randomizing the remainder avoids always favoring low-index players or
        # coalition sizes.  The allocation remains deterministic for ``seed``.
        rng = np.random.default_rng(
            np.random.SeedSequence(seed, spawn_key=(num_strata,))
        )
        extra_cells = rng.choice(
            num_strata,
            size=remainder,
            replace=False,
        )
        counts[extra_cells] += 1
    return counts.reshape(num_players, num_players)


def _make_payloads(
    *,
    counts: np.ndarray,
    seed: int,
    num_tasks: int,
) -> list[_StratifiedMarginalPayload]:
    """Distribute complete strata over tasks with balanced predicted work."""
    n = counts.shape[0]
    flat_counts = counts.reshape(-1)
    active_cells = np.flatnonzero(flat_counts)
    actual_tasks = min(num_tasks, len(active_cells))

    # Largest-first greedy bin packing balances calls even when the remainder
    # gives some strata one extra observation.  The cell-level RNG makes this
    # scheduling choice statistically irrelevant.
    ordered_cells = active_cells[
        np.lexsort((active_cells, -flat_counts[active_cells]))
    ]
    task_loads = np.zeros(actual_tasks, dtype=np.int64)
    task_cells: list[list[int]] = [[] for _ in range(actual_tasks)]
    for cell in ordered_cells:
        task = int(np.argmin(task_loads))
        task_cells[task].append(int(cell))
        task_loads[task] += int(flat_counts[cell])

    payloads: list[_StratifiedMarginalPayload] = []
    for cells in task_cells:
        payloads.append(
            _StratifiedMarginalPayload(
                num_players=n,
                seed=seed,
                cell_indices=tuple(cells),
                cell_counts=tuple(int(flat_counts[cell]) for cell in cells),
            )
        )
    return payloads


def estimate_stratified_marginal_mc(
    evaluator: StratifiedMarginalEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> StratifiedMarginalResult:
    """Estimate classic Shapley values with player-wise stratified MC.

    The target budget counts physical invocations of ``game.evaluate``.  Every
    marginal observation consumes two calls, so at most one call is unused.
    At least one observation per player/cardinality stratum is required; this
    matches the source estimator's positive ``num_MC`` contract and avoids an
    implicit zero-fill rule.
    """
    if num_players < 1:
        raise ValueError("num_players must be positive")
    if total_call_budget < 0:
        raise ValueError("total_call_budget must be nonnegative")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if task_chunksize < 1:
        raise ValueError("task_chunksize must be positive")

    num_marginal_samples = total_call_budget // 2
    num_strata = num_players * num_players
    if num_marginal_samples < num_strata:
        raise ValueError(
            "total_call_budget must fund at least one two-call marginal "
            "sample in every player/cardinality stratum; need at least "
            f"{2 * num_strata} calls"
        )

    planned_counts = _balanced_stratum_counts(
        num_players=num_players,
        num_marginal_samples=num_marginal_samples,
        seed=seed,
    )
    payloads = _make_payloads(
        counts=planned_counts,
        seed=seed,
        num_tasks=num_tasks,
    )
    task_results = evaluator.run_game_tasks(
        _evaluate_stratified_marginal_task,
        payloads,
        chunksize=task_chunksize,
    )
    if len(task_results) != len(payloads):
        raise RuntimeError("game evaluator returned the wrong number of tasks")

    sums = np.zeros(num_strata, dtype=np.float64)
    observed_counts = np.zeros(num_strata, dtype=np.int64)
    utility_evaluations = 0
    seen = np.zeros(num_strata, dtype=bool)
    for cell_indices, task_sums, task_counts, task_calls in task_results:
        cells = np.asarray(cell_indices, dtype=np.int64)
        values = np.asarray(task_sums, dtype=np.float64)
        counts = np.asarray(task_counts, dtype=np.int64)
        if (
            cells.ndim != 1
            or values.shape != cells.shape
            or counts.shape != cells.shape
        ):
            raise RuntimeError("a stratified-marginal worker returned invalid shapes")
        if np.any(cells < 0) or np.any(cells >= num_strata):
            raise RuntimeError(
                "a stratified-marginal worker returned an invalid stratum"
            )
        if np.any(seen[cells]):
            raise RuntimeError("a stratum was returned by more than one worker task")
        seen[cells] = True
        sums[cells] = values
        observed_counts[cells] = counts
        utility_evaluations += int(task_calls)

    flat_planned = planned_counts.reshape(-1)
    if not np.array_equal(observed_counts, flat_planned):
        raise RuntimeError("planned and observed stratum counts disagree")
    expected_calls = 2 * num_marginal_samples
    if utility_evaluations != expected_calls:
        raise RuntimeError("planned and observed utility calls disagree")

    stratum_means = (sums / observed_counts).reshape(
        num_players, num_players
    )
    values = stratum_means.mean(axis=1)
    diagnostics = StratifiedMarginalDiagnostics(
        num_players=num_players,
        target_call_budget=total_call_budget,
        utility_evaluations=utility_evaluations,
        unused_calls=total_call_budget - utility_evaluations,
        num_marginal_samples=num_marginal_samples,
        num_strata=num_strata,
        minimum_stratum_count=int(observed_counts.min()),
        maximum_stratum_count=int(observed_counts.max()),
        num_tasks=len(payloads),
        requested_num_tasks=num_tasks,
        sampling=(
            "uniform_predecessor_coalitions_balanced_over_"
            "player_and_cardinality"
        ),
        boundary_reuse=False,
        efficiency_projection=False,
        stratum_counts=observed_counts.reshape(
            num_players, num_players
        ),
    )
    return StratifiedMarginalResult(values, diagnostics)


__all__ = [
    "StratifiedMarginalDiagnostics",
    "StratifiedMarginalResult",
    "estimate_stratified_marginal_mc",
]

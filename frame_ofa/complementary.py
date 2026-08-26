"""Basic complementary-contribution sampling for Shapley values.

This module is a deterministic, game-interface adapter for Algorithm 2 in
Zhang et al., *Efficient Sampling Approaches to Shapley Value
Approximation* (SIGMOD 2023).  It follows the sampling and aggregation used
by the authors' basic ``cc_shap`` implementation:

* draw a size uniformly from ``{1, ..., n}``;
* draw a uniform coalition conditional on that size;
* evaluate the coalition and its complement;
* update the sampled size for included players and the complementary size
  for excluded players; and
* replace an unobserved player/size conditional mean by zero.

The implementation deliberately does not add boundary evaluations, warm-up
samples, or an efficiency projection.  One complementary-contribution pair
therefore costs exactly two utility evaluations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class CoarseGameTaskEvaluator(Protocol):
    """Subset of :class:`frame_ofa.parallel.GameEvaluator` used by CC."""

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class BasicCCDiagnostics:
    """Finite-budget coverage and exact utility-call accounting."""

    num_players: int
    num_pairs: int
    utility_evaluations: int
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    missing: str
    efficiency_projection: bool
    minimum_stratum_count: int
    minimum_positive_count: int
    missing_strata: int
    missing_stratum_fraction: float
    stratum_counts: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class BasicCCResult:
    """The basic-CC Shapley estimate and its sampling diagnostics."""

    values: np.ndarray
    diagnostics: BasicCCDiagnostics


@dataclass(frozen=True)
class _CCChunkPayload:
    num_players: int
    num_pairs: int
    seed_words: tuple[int, ...]


# ``run_game_tasks`` returns all task results as a list.  Limit the aggregate
# dense sum/count payload so a large player count cannot accidentally create
# one n-by-n pair of arrays per worker without bound.  For Wine (n=142), 128
# tasks use about 40 MiB and are not capped.
_MAX_AGGREGATE_TASK_RESULT_BYTES = 256 * 2**20


def _finalize_basic_cc(
    contribution_sums: np.ndarray,
    stratum_counts: np.ndarray,
) -> np.ndarray:
    """Apply the official zero-fill rule and average the n strata."""
    conditional_means = np.divide(
        contribution_sums,
        stratum_counts,
        out=np.zeros_like(contribution_sums, dtype=np.float64),
        where=stratum_counts > 0,
    )
    return conditional_means.mean(axis=1)


def aggregate_basic_cc_samples(
    coalitions: np.ndarray,
    complementary_differences: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate already-evaluated samples exactly as basic ``cc_shap``.

    Parameters
    ----------
    coalitions:
        A Boolean matrix whose rows are nonempty sampled coalitions.  Full
        coalitions are valid because Algorithm 2 draws sizes through ``n``.
    complementary_differences:
        One value ``v(S) - v(N\\S)`` per row.

    Returns
    -------
    values, counts:
        The zero-filled basic-CC estimate and the ``n x n`` matrix of
        player/size counts.  Column ``k-1`` is the stratum for coalitions of
        size ``k`` that contain the player.

    Notes
    -----
    This helper makes the accumulator independently testable against the
    paper pseudocode and the official implementation.  The parallel entry
    point below avoids materializing this coalition matrix for large runs.
    """
    rows = np.asarray(coalitions, dtype=bool)
    differences = np.asarray(
        complementary_differences, dtype=np.float64
    )
    if rows.ndim != 2:
        raise ValueError("coalitions must be a two-dimensional array")
    num_pairs, num_players = rows.shape
    if num_players < 1:
        raise ValueError("at least one player is required")
    if differences.shape != (num_pairs,):
        raise ValueError("one complementary difference is required per row")

    sizes = rows.sum(axis=1, dtype=np.int64)
    if np.any(sizes < 1) or np.any(sizes > num_players):
        raise ValueError("basic CC requires coalition sizes in {1, ..., n}")

    sums = np.zeros((num_players, num_players), dtype=np.float64)
    counts = np.zeros((num_players, num_players), dtype=np.int64)
    for coalition, size_value, difference in zip(
        rows, sizes, differences
    ):
        size = int(size_value)
        included = np.flatnonzero(coalition)
        sums[included, size - 1] += difference
        counts[included, size - 1] += 1

        complementary_size = num_players - size
        if complementary_size:
            excluded = np.flatnonzero(~coalition)
            sums[excluded, complementary_size - 1] -= difference
            counts[excluded, complementary_size - 1] += 1

    return _finalize_basic_cc(sums, counts), counts


def _evaluate_basic_cc_chunk(
    game: Any,
    payload: _CCChunkPayload,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample, evaluate, and aggregate one coarse CC worker task."""
    num_players = payload.num_players
    rng = np.random.default_rng(
        np.random.SeedSequence(payload.seed_words)
    )
    indices = np.arange(num_players)
    sums = np.zeros((num_players, num_players), dtype=np.float64)
    counts = np.zeros((num_players, num_players), dtype=np.int64)

    for _ in range(payload.num_pairs):
        size = int(rng.integers(1, num_players + 1))
        included = rng.choice(num_players, size=size, replace=False)
        coalition = np.zeros(num_players, dtype=bool)
        coalition[included] = True
        complement = ~coalition

        difference = float(game.evaluate(coalition)) - float(
            game.evaluate(complement)
        )
        sums[included, size - 1] += difference
        counts[included, size - 1] += 1

        complementary_size = num_players - size
        if complementary_size:
            excluded = indices[complement]
            sums[excluded, complementary_size - 1] -= difference
            counts[excluded, complementary_size - 1] += 1

    return sums, counts, payload.num_pairs


def _bounded_task_count(
    num_players: int,
    num_pairs: int,
    requested_num_tasks: int,
) -> int:
    bytes_per_task = (
        num_players
        * num_players
        * (
            np.dtype(np.float64).itemsize
            + np.dtype(np.int64).itemsize
        )
    )
    memory_limited_tasks = max(
        1, _MAX_AGGREGATE_TASK_RESULT_BYTES // bytes_per_task
    )
    return min(num_pairs, requested_num_tasks, memory_limited_tasks)


def estimate_basic_cc(
    evaluator: CoarseGameTaskEvaluator,
    num_players: int,
    num_pairs: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> BasicCCResult:
    """Estimate all Shapley values with official basic CC sampling.

    ``num_pairs`` is the number of complementary contributions, not the
    number of physical utility invocations.  The reported and actual call
    count is therefore ``2 * num_pairs``.

    Sampling happens inside coarse game-aware tasks, so a multi-process
    :class:`~frame_ofa.parallel.GameEvaluator` does not receive a giant
    coalition matrix through IPC.  A fixed seed, task count, and deterministic
    game produce bitwise-reproducible samples and aggregation.
    """
    if num_players < 1:
        raise ValueError("num_players must be positive")
    if num_pairs < 1:
        raise ValueError("num_pairs must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if task_chunksize < 1:
        raise ValueError("task_chunksize must be positive")

    actual_tasks = _bounded_task_count(
        num_players, num_pairs, num_tasks
    )
    base_pairs, remainder = divmod(num_pairs, actual_tasks)
    child_sequences = np.random.SeedSequence(seed).spawn(actual_tasks)
    payloads = [
        _CCChunkPayload(
            num_players=num_players,
            num_pairs=base_pairs + int(task < remainder),
            seed_words=tuple(
                int(word)
                for word in child_sequences[task].generate_state(
                    4, dtype=np.uint32
                )
            ),
        )
        for task in range(actual_tasks)
    ]

    chunk_results = evaluator.run_game_tasks(
        _evaluate_basic_cc_chunk,
        payloads,
        chunksize=task_chunksize,
    )
    if len(chunk_results) != actual_tasks:
        raise RuntimeError("game evaluator returned the wrong number of tasks")

    sums = np.zeros((num_players, num_players), dtype=np.float64)
    counts = np.zeros((num_players, num_players), dtype=np.int64)
    observed_pairs = 0
    for chunk_sums, chunk_counts, chunk_pairs in chunk_results:
        if chunk_sums.shape != sums.shape or chunk_counts.shape != counts.shape:
            raise RuntimeError("a basic-CC worker returned an invalid shape")
        sums += chunk_sums
        counts += chunk_counts
        observed_pairs += int(chunk_pairs)
    if observed_pairs != num_pairs:
        raise RuntimeError("basic-CC worker pair accounting disagrees")

    values = _finalize_basic_cc(sums, counts)
    missing_strata = int(np.count_nonzero(counts == 0))
    positive_counts = counts[counts > 0]
    diagnostics = BasicCCDiagnostics(
        num_players=num_players,
        num_pairs=num_pairs,
        utility_evaluations=2 * num_pairs,
        num_tasks=actual_tasks,
        requested_num_tasks=num_tasks,
        sampling="uniform_random_size_1_to_n",
        missing="zero",
        efficiency_projection=False,
        minimum_stratum_count=int(counts.min()),
        minimum_positive_count=(
            int(positive_counts.min()) if len(positive_counts) else 0
        ),
        missing_strata=missing_strata,
        missing_stratum_fraction=(
            missing_strata / float(num_players * num_players)
        ),
        stratum_counts=counts,
    )
    return BasicCCResult(values=values, diagnostics=diagnostics)


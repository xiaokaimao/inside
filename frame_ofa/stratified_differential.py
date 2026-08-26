"""Size-stratified differential-matrix estimation (S-Diff).

This module ports the external baseline's Algorithm-2 variant (called
``S-Diff`` in the paper's experiments) from Pang et al., "Shapley Value
Estimation Based on Differential Matrix" (SIGMOD 2025).  It is distinct from
the unstratified ``Diff`` estimator in ``frame_ofa.differential``.  The
partition initialization follows Algorithm 2; the greedy coverage routine is
the simpler first-missing-pair routine in the user's external baseline, not a
claim of a literal implementation of the paper's Algorithm 6.

The random phase is parallelized without making one O(n^3) state per worker.
Drawing all sizes IID and then forgetting their order is equivalent to first
drawing their multinomial layer counts and sampling uniformly within each
layer.  We use that equivalence to shard layers across coarse game tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Protocol

import numpy as np

from .differential import diff_layer_distribution


class StratifiedDifferentialEvaluator(Protocol):
    """Evaluator interface needed by :func:`estimate_sdiff`."""

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray: ...

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]: ...


class SDiffCoverageError(RuntimeError):
    """Raised when the strict S-Diff coverage precondition cannot be met."""

    def __init__(
        self,
        *,
        available_samples: int,
        required_samples: int,
        reason: str,
    ) -> None:
        self.available_samples = int(available_samples)
        self.required_samples = int(required_samples)
        self.reason = str(reason)
        super().__init__(
            "insufficient S-Diff utility-sample budget: "
            f"available={self.available_samples}, "
            f"required={self.required_samples}, reason={self.reason}"
        )


@dataclass(frozen=True)
class SDiffDiagnostics:
    """Coverage, phase allocation, solver, and exact call accounting."""

    algorithm: str
    num_players: int
    target_call_budget: int
    utility_evaluations: int
    unused_calls: int
    boundary_evaluations: int
    num_utility_samples: int
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    initialization: str
    unbiasedness_scope: str
    normalization: str
    efficiency_projection: bool
    solver: str
    coverage_precondition: str
    paper_minimum_utility_samples: int
    initial_partition_samples: int
    greedy_fill_samples: int
    initialization_samples: int
    random_phase_samples: int
    empty_utility: float
    full_utility: float
    covered_stratified_ordered_pairs: int
    total_stratified_ordered_pairs: int
    missing_stratified_ordered_pairs: int
    stratified_pair_coverage: float
    minimum_stratified_pair_count: int
    maximum_stratified_pair_count: int
    efficiency_residual: float
    state_bytes: int
    layer_sample_counts: np.ndarray = field(repr=False)
    layer_minimum_pair_counts: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class SDiffResult:
    """Estimated Shapley vector, differential matrix, and diagnostics."""

    values: np.ndarray
    diagnostics: SDiffDiagnostics
    differential_matrix: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class _LayerJob:
    layer: int
    num_samples: int
    seed_words: tuple[int, ...]


@dataclass(frozen=True)
class _SDiffPayload:
    num_players: int
    empty_utility: float
    jobs: tuple[_LayerJob, ...]


def sdiff_state_bytes(num_players: int) -> int:
    """Bytes for one float64-sum/uint32-count stratified state."""
    if num_players < 2:
        return 0
    entries = (num_players - 1) * num_players * num_players
    return entries * (
        np.dtype(np.float64).itemsize + np.dtype(np.uint32).itemsize
    )


def _update_sdiff_state(
    utility_sums: np.ndarray,
    pair_counts: np.ndarray,
    layer_sample_counts: np.ndarray,
    coalition: np.ndarray,
    centered_utility: float,
) -> None:
    row = np.asarray(coalition, dtype=bool)
    num_players = row.size
    layer = int(row.sum())
    if layer < 1 or layer >= num_players:
        raise ValueError("S-Diff samples must have sizes 1,...,n-1")
    included = np.flatnonzero(row)
    excluded = np.flatnonzero(~row)
    index = np.ix_(included, excluded)
    utility_sums[layer - 1][index] += float(centered_utility)
    pair_counts[layer - 1][index] += 1
    layer_sample_counts[layer - 1] += 1


def _update_coverage(
    pair_counts: np.ndarray,
    coalition: np.ndarray,
) -> None:
    row = np.asarray(coalition, dtype=bool)
    layer = int(row.sum())
    included = np.flatnonzero(row)
    excluded = np.flatnonzero(~row)
    pair_counts[layer - 1][np.ix_(included, excluded)] += 1


def _initial_sdiff_design(
    num_players: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, int]:
    """Build Algorithm-2 partitions and the external strict greedy fill."""
    coverage = np.zeros(
        (num_players - 1, num_players, num_players), dtype=np.uint16
    )
    coalitions: list[np.ndarray] = []

    for layer in range(1, num_players // 2 + 1):
        permutation = rng.permutation(num_players)
        num_blocks = int(math.ceil(num_players / layer))
        for block in range(num_blocks):
            positions = (
                np.arange(block * layer, block * layer + layer)
                % num_players
            )
            coalition = np.zeros(num_players, dtype=bool)
            coalition[permutation[positions]] = True
            complement = ~coalition
            coalitions.extend((coalition, complement))
            _update_coverage(coverage, coalition)
            _update_coverage(coverage, complement)
    partition_samples = len(coalitions)

    players = np.arange(num_players)
    off_diagonal = ~np.eye(num_players, dtype=bool)
    for layer in range(1, num_players // 2 + 1):
        while True:
            missing = np.flatnonzero(
                (coverage[layer - 1] == 0) & off_diagonal
            )
            if len(missing) == 0:
                break
            included_player, excluded_player = np.unravel_index(
                int(missing[0]), (num_players, num_players)
            )
            pool = players[
                (players != included_player)
                & (players != excluded_player)
            ]
            chosen = np.asarray([included_player], dtype=np.int64)
            if layer > 1:
                extras = rng.choice(
                    pool, size=layer - 1, replace=False
                )
                chosen = np.concatenate((chosen, extras))
            coalition = np.zeros(num_players, dtype=bool)
            coalition[chosen] = True
            complement = ~coalition
            coalitions.extend((coalition, complement))
            _update_coverage(coverage, coalition)
            _update_coverage(coverage, complement)

    off_counts = coverage[:, off_diagonal]
    if np.any(off_counts == 0):
        raise RuntimeError("S-Diff greedy initialization failed coverage")
    rows = np.asarray(coalitions, dtype=bool)
    return rows, partition_samples, len(coalitions) - partition_samples


def _recover_sdiff(
    utility_sums: np.ndarray,
    pair_counts: np.ndarray,
    total_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    num_layers, num_players, _ = utility_sums.shape
    if num_layers != num_players - 1:
        raise ValueError("S-Diff state must contain n-1 layers")
    off_diagonal = ~np.eye(num_players, dtype=bool)
    if np.any(pair_counts[:, off_diagonal] == 0):
        raise RuntimeError("strict S-Diff recovery requires full coverage")

    layer_mean_sum = np.zeros((num_players, num_players), dtype=np.float64)
    means = np.zeros((num_players, num_players), dtype=np.float64)
    for layer_index in range(num_layers):
        means.fill(0.0)
        np.divide(
            utility_sums[layer_index],
            pair_counts[layer_index],
            out=means,
            where=pair_counts[layer_index] > 0,
        )
        layer_mean_sum += means
    differential_matrix = (
        layer_mean_sum - layer_mean_sum.T
    ) / num_layers
    values = (
        float(total_value) / num_players
        + differential_matrix.sum(axis=1) / num_players
    )
    values += (float(total_value) - float(values.sum())) / num_players
    return values, differential_matrix


def aggregate_sdiff_samples(
    coalitions: np.ndarray,
    utilities: np.ndarray,
    *,
    empty_utility: float,
    full_utility: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate pre-evaluated S-Diff samples with strict full coverage."""
    rows = np.asarray(coalitions, dtype=bool)
    observed = np.asarray(utilities, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("coalitions must be a 2-D array with n >= 2")
    if observed.shape != (len(rows),):
        raise ValueError("one utility is required per coalition")
    num_players = rows.shape[1]
    utility_sums = np.zeros(
        (num_players - 1, num_players, num_players), dtype=np.float64
    )
    pair_counts = np.zeros(
        (num_players - 1, num_players, num_players), dtype=np.uint32
    )
    layer_counts = np.zeros(num_players - 1, dtype=np.int64)
    for coalition, utility in zip(rows, observed, strict=True):
        _update_sdiff_state(
            utility_sums,
            pair_counts,
            layer_counts,
            coalition,
            float(utility) - float(empty_utility),
        )
    values, matrix = _recover_sdiff(
        utility_sums,
        pair_counts,
        float(full_utility) - float(empty_utility),
    )
    return values, matrix, pair_counts


def _run_sdiff_payload(
    game: Any,
    payload: _SDiffPayload,
) -> list[tuple[int, np.ndarray, np.ndarray, int]]:
    """Evaluate several fixed-layer shards on one worker-local game."""
    results: list[tuple[int, np.ndarray, np.ndarray, int]] = []
    for job in payload.jobs:
        rng = np.random.default_rng(np.random.SeedSequence(job.seed_words))
        sums = np.zeros(
            (payload.num_players, payload.num_players), dtype=np.float64
        )
        counts = np.zeros(
            (payload.num_players, payload.num_players), dtype=np.uint32
        )
        for _ in range(job.num_samples):
            included = rng.choice(
                payload.num_players, size=job.layer, replace=False
            )
            coalition = np.zeros(payload.num_players, dtype=bool)
            coalition[included] = True
            excluded = np.flatnonzero(~coalition)
            centered_utility = (
                float(game.evaluate(coalition)) - payload.empty_utility
            )
            index = np.ix_(included, excluded)
            sums[index] += centered_utility
            counts[index] += 1
        results.append((job.layer, sums, counts, job.num_samples))
    return results


def _make_random_payloads(
    num_players: int,
    num_random_samples: int,
    num_tasks: int,
    empty_utility: float,
    count_sequence: np.random.SeedSequence,
    job_sequence: np.random.SeedSequence,
) -> tuple[list[_SDiffPayload], np.ndarray]:
    if num_random_samples == 0:
        return [], np.zeros(num_players - 1, dtype=np.int64)
    layers, probabilities, _ = diff_layer_distribution(num_players)
    count_rng = np.random.default_rng(count_sequence)
    layer_counts = count_rng.multinomial(
        num_random_samples, probabilities
    ).astype(np.int64)

    target_tasks = min(num_tasks, num_random_samples)
    target_shard_size = int(math.ceil(num_random_samples / target_tasks))
    shard_specs: list[tuple[int, int]] = []
    for layer, count in zip(layers, layer_counts, strict=True):
        remaining = int(count)
        while remaining:
            shard_size = min(remaining, target_shard_size)
            shard_specs.append((int(layer), shard_size))
            remaining -= shard_size

    child_sequences = job_sequence.spawn(len(shard_specs))
    jobs = [
        _LayerJob(
            layer=layer,
            num_samples=count,
            seed_words=tuple(
                int(word)
                for word in child.generate_state(4, dtype=np.uint32)
            ),
        )
        for (layer, count), child in zip(
            shard_specs, child_sequences, strict=True
        )
    ]

    actual_tasks = min(target_tasks, len(jobs))
    buckets: list[list[_LayerJob]] = [[] for _ in range(actual_tasks)]
    loads = np.zeros(actual_tasks, dtype=np.int64)
    for job in sorted(jobs, key=lambda candidate: -candidate.num_samples):
        bucket = int(np.argmin(loads))
        buckets[bucket].append(job)
        loads[bucket] += job.num_samples
    payloads = [
        _SDiffPayload(
            num_players=num_players,
            empty_utility=empty_utility,
            jobs=tuple(bucket),
        )
        for bucket in buckets
    ]
    return payloads, layer_counts


def estimate_sdiff(
    evaluator: StratifiedDifferentialEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> SDiffResult:
    """Estimate Shapley values with strict S-Diff (Algorithm 2).

    The method uses exactly ``total_call_budget`` physical calls: ``B-2``
    S-Diff coalition samples plus empty/full boundary calls.  Budgets below
    the paper's ``4 n log n`` precondition or below the realized strict
    initialization design are rejected before any utility is evaluated.
    """
    if num_players < 2:
        raise ValueError("S-Diff requires at least two players")
    if total_call_budget < 3:
        raise ValueError("total_call_budget must be at least three")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if task_chunksize < 1:
        raise ValueError("task_chunksize must be positive")

    num_samples = total_call_budget - 2
    paper_minimum = int(math.ceil(4.0 * num_players * math.log(num_players)))
    if num_samples < paper_minimum:
        raise SDiffCoverageError(
            available_samples=num_samples,
            required_samples=paper_minimum,
            reason="paper_4_n_log_n_precondition",
        )

    master_sequence = np.random.SeedSequence(seed)
    initial_sequence, count_sequence, job_sequence = master_sequence.spawn(3)
    initial_rows, partition_samples, greedy_samples = (
        _initial_sdiff_design(
            num_players, np.random.default_rng(initial_sequence)
        )
    )
    initialization_samples = len(initial_rows)
    if initialization_samples > num_samples:
        raise SDiffCoverageError(
            available_samples=num_samples,
            required_samples=initialization_samples,
            reason="realized_strict_initialization",
        )

    boundary = np.zeros((2, num_players), dtype=bool)
    boundary[1] = True
    boundary_values = np.asarray(
        evaluator.evaluate(boundary), dtype=np.float64
    )
    if boundary_values.shape != (2,):
        raise RuntimeError("evaluator returned invalid boundary utilities")
    empty_utility, full_utility = map(float, boundary_values)

    utility_sums = np.zeros(
        (num_players - 1, num_players, num_players), dtype=np.float64
    )
    pair_counts = np.zeros(
        (num_players - 1, num_players, num_players), dtype=np.uint32
    )
    layer_sample_counts = np.zeros(num_players - 1, dtype=np.int64)

    initial_utilities = np.asarray(
        evaluator.evaluate(initial_rows), dtype=np.float64
    )
    if initial_utilities.shape != (initialization_samples,):
        raise RuntimeError("evaluator returned invalid initialization values")
    for coalition, utility in zip(
        initial_rows, initial_utilities, strict=True
    ):
        _update_sdiff_state(
            utility_sums,
            pair_counts,
            layer_sample_counts,
            coalition,
            float(utility) - empty_utility,
        )

    random_samples = num_samples - initialization_samples
    payloads, allocated_layer_counts = _make_random_payloads(
        num_players,
        random_samples,
        num_tasks,
        empty_utility,
        count_sequence,
        job_sequence,
    )
    task_results = evaluator.run_game_tasks(
        _run_sdiff_payload,
        payloads,
        chunksize=task_chunksize,
    )
    if len(task_results) != len(payloads):
        raise RuntimeError("evaluator returned the wrong number of tasks")
    observed_random_samples = 0
    observed_random_layers = np.zeros(num_players - 1, dtype=np.int64)
    for task_result in task_results:
        for layer, sums, counts, shard_samples in task_result:
            if sums.shape != (num_players, num_players):
                raise RuntimeError("S-Diff worker returned invalid sums")
            if counts.shape != (num_players, num_players):
                raise RuntimeError("S-Diff worker returned invalid counts")
            utility_sums[layer - 1] += sums
            pair_counts[layer - 1] += counts
            layer_sample_counts[layer - 1] += shard_samples
            observed_random_layers[layer - 1] += shard_samples
            observed_random_samples += int(shard_samples)
    if observed_random_samples != random_samples:
        raise RuntimeError("S-Diff random sample accounting disagrees")
    if not np.array_equal(
        observed_random_layers, allocated_layer_counts
    ):
        raise RuntimeError("S-Diff random layer accounting disagrees")

    total_value = full_utility - empty_utility
    values, differential_matrix = _recover_sdiff(
        utility_sums, pair_counts, total_value
    )
    off_diagonal = ~np.eye(num_players, dtype=bool)
    off_counts = pair_counts[:, off_diagonal]
    total_stratified_pairs = (
        (num_players - 1) * num_players * (num_players - 1)
    )
    covered_stratified_pairs = int(np.count_nonzero(off_counts))
    layer_minimums = off_counts.min(axis=1).astype(np.int64)
    diagnostics = SDiffDiagnostics(
        algorithm="s_diff_external_strict_greedy_port",
        num_players=num_players,
        target_call_budget=total_call_budget,
        utility_evaluations=num_samples + 2,
        unused_calls=0,
        boundary_evaluations=2,
        num_utility_samples=num_samples,
        num_tasks=len(payloads),
        requested_num_tasks=num_tasks,
        sampling=(
            "paper_coverage_initialization_then_multinomial_equivalent_"
            "iid_p_k_proportional_to_1_over_k_times_n_minus_k"
        ),
        initialization=(
            "algorithm2_partition_plus_external_first_missing_pair_greedy"
        ),
        unbiasedness_scope=(
            "random_phase_matches_paper_distribution; no_universal_"
            "finite_sample_unbiasedness_claim_for_external_greedy_fill"
        ),
        normalization="utility_minus_empty",
        efficiency_projection=True,
        solver="stratified_differential_unweighted_ls_closed_form",
        coverage_precondition="every_layer_ordered_pair_count_positive",
        paper_minimum_utility_samples=paper_minimum,
        initial_partition_samples=partition_samples,
        greedy_fill_samples=greedy_samples,
        initialization_samples=initialization_samples,
        random_phase_samples=random_samples,
        empty_utility=empty_utility,
        full_utility=full_utility,
        covered_stratified_ordered_pairs=covered_stratified_pairs,
        total_stratified_ordered_pairs=total_stratified_pairs,
        missing_stratified_ordered_pairs=(
            total_stratified_pairs - covered_stratified_pairs
        ),
        stratified_pair_coverage=(
            covered_stratified_pairs / float(total_stratified_pairs)
        ),
        minimum_stratified_pair_count=int(off_counts.min()),
        maximum_stratified_pair_count=int(off_counts.max()),
        efficiency_residual=float(values.sum() - total_value),
        state_bytes=sdiff_state_bytes(num_players),
        layer_sample_counts=layer_sample_counts,
        layer_minimum_pair_counts=layer_minimums,
    )
    return SDiffResult(
        values=values,
        diagnostics=diagnostics,
        differential_matrix=differential_matrix,
    )


__all__ = [
    "SDiffCoverageError",
    "SDiffDiagnostics",
    "SDiffResult",
    "aggregate_sdiff_samples",
    "estimate_sdiff",
    "sdiff_state_bytes",
]

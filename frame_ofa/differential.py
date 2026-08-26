"""Unstratified differential-matrix Shapley estimation (Diff).

This is a clean-room implementation of Algorithm 1 (called ``Diff`` in the
experiments) from Pang et al., "Shapley Value Estimation Based on
Differential Matrix" (SIGMOD 2025).  It is deliberately not called S-Diff:
S-Diff is the distinct size-stratified Algorithm 2 and has O(n^3) state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class DifferentialEvaluator(Protocol):
    """Evaluator interface needed by :func:`estimate_diff`."""

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray: ...

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]: ...


class DifferentialCoverageError(RuntimeError):
    """Raised when Algorithm 1 cannot form every conditional mean."""

    def __init__(
        self,
        *,
        missing_ordered_pairs: int,
        total_ordered_pairs: int,
        utility_evaluations: int,
    ) -> None:
        self.missing_ordered_pairs = int(missing_ordered_pairs)
        self.total_ordered_pairs = int(total_ordered_pairs)
        self.utility_evaluations = int(utility_evaluations)
        super().__init__(
            "Diff requires every ordered player pair to be sampled at least "
            f"once; {self.missing_ordered_pairs} of "
            f"{self.total_ordered_pairs} pairs are missing after "
            f"{self.utility_evaluations} utility evaluations"
        )


@dataclass(frozen=True)
class DiffDiagnostics:
    """Sampling, pair coverage, solver, and exact call accounting."""

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
    normalization: str
    efficiency_projection: bool
    solver: str
    coverage_precondition: str
    empty_utility: float
    full_utility: float
    covered_ordered_pairs: int
    total_ordered_pairs: int
    missing_ordered_pairs: int
    ordered_pair_coverage: float
    minimum_ordered_pair_count: int
    maximum_ordered_pair_count: int
    sampled_layers: int
    missing_layers: int
    layer_coverage: float
    efficiency_residual: float
    layer_counts: np.ndarray = field(repr=False)
    pair_counts: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class DiffResult:
    """Estimated Shapley vector, differential matrix, and diagnostics."""

    values: np.ndarray
    diagnostics: DiffDiagnostics
    differential_matrix: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class _DiffPayload:
    num_players: int
    num_samples: int
    empty_utility: float
    seed_words: tuple[int, ...]


_MAX_AGGREGATE_RESULT_BYTES = 256 * 2**20


def diff_layer_distribution(
    num_players: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Algorithm-1 sizes and ``p_k proportional 1/[k(n-k)]``."""
    if num_players < 2:
        raise ValueError("Diff requires at least two players")
    layers = np.arange(1, num_players, dtype=np.int64)
    weights = 1.0 / (
        layers.astype(np.float64)
        * (num_players - layers).astype(np.float64)
    )
    normalization = float(weights.sum())
    probabilities = weights / normalization
    probabilities /= probabilities.sum()
    return layers, probabilities, normalization


def _recover_from_diff_state(
    utility_sums: np.ndarray,
    pair_counts: np.ndarray,
    total_value: float,
    *,
    utility_evaluations: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply Algorithm 1 and the complete-matrix LS closed form."""
    sums = np.asarray(utility_sums, dtype=np.float64)
    counts = np.asarray(pair_counts, dtype=np.int64)
    if sums.ndim != 2 or sums.shape[0] != sums.shape[1]:
        raise ValueError("utility_sums must be a square matrix")
    if counts.shape != sums.shape:
        raise ValueError("pair_counts must match utility_sums")
    num_players = sums.shape[0]
    off_diagonal = ~np.eye(num_players, dtype=bool)
    missing = int(np.count_nonzero((counts == 0) & off_diagonal))
    total_pairs = num_players * (num_players - 1)
    if missing:
        raise DifferentialCoverageError(
            missing_ordered_pairs=missing,
            total_ordered_pairs=total_pairs,
            utility_evaluations=utility_evaluations,
        )

    means = np.zeros_like(sums, dtype=np.float64)
    np.divide(sums, counts, out=means, where=counts > 0)
    differential_matrix = means - means.T
    values = (
        float(total_value) / num_players
        + differential_matrix.sum(axis=1) / num_players
    )
    # The matrix is anti-symmetric, so this is only a roundoff correction.
    values += (float(total_value) - float(values.sum())) / num_players
    return values, differential_matrix


def aggregate_diff_samples(
    coalitions: np.ndarray,
    utilities: np.ndarray,
    *,
    empty_utility: float,
    full_utility: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate pre-evaluated Algorithm-1 Diff samples.

    Returns the Shapley estimate, differential matrix, and ordered-pair count
    matrix.  Every off-diagonal count must be positive.
    """
    rows = np.asarray(coalitions, dtype=bool)
    observed = np.asarray(utilities, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("coalitions must be a 2-D array with n >= 2")
    if observed.shape != (len(rows),):
        raise ValueError("one utility is required per coalition")
    if len(rows) < 1:
        raise ValueError("at least one Diff sample is required")
    sizes = rows.sum(axis=1)
    if np.any(sizes < 1) or np.any(sizes >= rows.shape[1]):
        raise ValueError("Diff coalitions must have sizes 1,...,n-1")

    num_players = rows.shape[1]
    utility_sums = np.zeros((num_players, num_players), dtype=np.float64)
    pair_counts = np.zeros((num_players, num_players), dtype=np.int64)
    for coalition, utility in zip(rows, observed, strict=True):
        included = np.flatnonzero(coalition)
        excluded = np.flatnonzero(~coalition)
        index = np.ix_(included, excluded)
        utility_sums[index] += float(utility) - float(empty_utility)
        pair_counts[index] += 1
    values, matrix = _recover_from_diff_state(
        utility_sums,
        pair_counts,
        float(full_utility) - float(empty_utility),
        utility_evaluations=len(rows) + 2,
    )
    return values, matrix, pair_counts


def _run_diff_chunk(
    game: Any,
    payload: _DiffPayload,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample, evaluate, and aggregate one coarse Diff worker chunk."""
    rng = np.random.default_rng(np.random.SeedSequence(payload.seed_words))
    layers, probabilities, _ = diff_layer_distribution(payload.num_players)
    utility_sums = np.zeros(
        (payload.num_players, payload.num_players), dtype=np.float64
    )
    pair_counts = np.zeros(
        (payload.num_players, payload.num_players), dtype=np.int64
    )
    layer_counts = np.zeros(payload.num_players - 1, dtype=np.int64)

    for _ in range(payload.num_samples):
        layer = int(rng.choice(layers, p=probabilities))
        included = rng.choice(
            payload.num_players, size=layer, replace=False
        )
        coalition = np.zeros(payload.num_players, dtype=bool)
        coalition[included] = True
        excluded = np.flatnonzero(~coalition)
        centered_utility = (
            float(game.evaluate(coalition)) - payload.empty_utility
        )
        index = np.ix_(included, excluded)
        utility_sums[index] += centered_utility
        pair_counts[index] += 1
        layer_counts[layer - 1] += 1

    return utility_sums, pair_counts, layer_counts, payload.num_samples


def _bounded_task_count(
    num_players: int,
    num_samples: int,
    requested_num_tasks: int,
) -> int:
    bytes_per_result = num_players * num_players * (
        np.dtype(np.float64).itemsize + np.dtype(np.int64).itemsize
    )
    memory_limited = max(
        1, _MAX_AGGREGATE_RESULT_BYTES // max(bytes_per_result, 1)
    )
    return min(num_samples, requested_num_tasks, memory_limited)


def _split_diff_tasks(
    num_samples: int,
    num_tasks: int,
    seed: int,
    *,
    num_players: int,
    empty_utility: float,
) -> list[_DiffPayload]:
    actual_tasks = _bounded_task_count(
        num_players, num_samples, num_tasks
    )
    quotient, remainder = divmod(num_samples, actual_tasks)
    child_sequences = np.random.SeedSequence(seed).spawn(actual_tasks)
    return [
        _DiffPayload(
            num_players=num_players,
            num_samples=quotient + int(task_index < remainder),
            empty_utility=empty_utility,
            seed_words=tuple(
                int(word)
                for word in child_sequences[task_index].generate_state(
                    4, dtype=np.uint32
                )
            ),
        )
        for task_index in range(actual_tasks)
    ]


def estimate_diff(
    evaluator: DifferentialEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> DiffResult:
    """Estimate Shapley values with unstratified Diff (Algorithm 1).

    Exactly ``total_call_budget`` calls are used: ``B-2`` sampled coalition
    utilities and two boundary utilities.  Recovery is intentionally strict:
    if any ordered pair has zero qualifying samples, Algorithm 1's ratio is
    undefined and :class:`DifferentialCoverageError` is raised.
    """
    if num_players < 2:
        raise ValueError("Diff requires at least two players")
    if total_call_budget < 3:
        raise ValueError("total_call_budget must be at least three")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if task_chunksize < 1:
        raise ValueError("task_chunksize must be positive")

    boundary = np.zeros((2, num_players), dtype=bool)
    boundary[1] = True
    boundary_values = np.asarray(
        evaluator.evaluate(boundary), dtype=np.float64
    )
    if boundary_values.shape != (2,):
        raise RuntimeError("evaluator returned invalid boundary utilities")
    empty_utility, full_utility = map(float, boundary_values)

    num_samples = total_call_budget - 2
    payloads = _split_diff_tasks(
        num_samples,
        num_tasks,
        seed,
        num_players=num_players,
        empty_utility=empty_utility,
    )
    task_results = evaluator.run_game_tasks(
        _run_diff_chunk,
        payloads,
        chunksize=task_chunksize,
    )
    if len(task_results) != len(payloads):
        raise RuntimeError("evaluator returned the wrong number of tasks")

    utility_sums = np.zeros((num_players, num_players), dtype=np.float64)
    pair_counts = np.zeros((num_players, num_players), dtype=np.int64)
    layer_counts = np.zeros(num_players - 1, dtype=np.int64)
    observed_samples = 0
    for task_sums, task_pairs, task_layers, task_samples in task_results:
        if task_sums.shape != utility_sums.shape:
            raise RuntimeError("Diff worker returned invalid utility sums")
        if task_pairs.shape != pair_counts.shape:
            raise RuntimeError("Diff worker returned invalid pair counts")
        if task_layers.shape != layer_counts.shape:
            raise RuntimeError("Diff worker returned invalid layer counts")
        utility_sums += task_sums
        pair_counts += task_pairs
        layer_counts += task_layers
        observed_samples += int(task_samples)
    if observed_samples != num_samples:
        raise RuntimeError("Diff sample accounting disagrees")

    total_value = full_utility - empty_utility
    values, differential_matrix = _recover_from_diff_state(
        utility_sums,
        pair_counts,
        total_value,
        utility_evaluations=num_samples + 2,
    )
    off_diagonal_counts = pair_counts[
        ~np.eye(num_players, dtype=bool)
    ]
    covered_pairs = int(np.count_nonzero(off_diagonal_counts))
    total_pairs = num_players * (num_players - 1)
    sampled_layers = int(np.count_nonzero(layer_counts))
    diagnostics = DiffDiagnostics(
        algorithm="diff_pang2025_algorithm1",
        num_players=num_players,
        target_call_budget=total_call_budget,
        utility_evaluations=num_samples + 2,
        unused_calls=0,
        boundary_evaluations=2,
        num_utility_samples=num_samples,
        num_tasks=len(payloads),
        requested_num_tasks=num_tasks,
        sampling="p_k_proportional_to_1_over_k_times_n_minus_k",
        normalization="utility_minus_empty",
        efficiency_projection=True,
        solver="complete_differential_unweighted_ls_closed_form",
        coverage_precondition="every_ordered_pair_count_positive",
        empty_utility=empty_utility,
        full_utility=full_utility,
        covered_ordered_pairs=covered_pairs,
        total_ordered_pairs=total_pairs,
        missing_ordered_pairs=total_pairs - covered_pairs,
        ordered_pair_coverage=covered_pairs / float(total_pairs),
        minimum_ordered_pair_count=int(off_diagonal_counts.min()),
        maximum_ordered_pair_count=int(off_diagonal_counts.max()),
        sampled_layers=sampled_layers,
        missing_layers=(num_players - 1) - sampled_layers,
        layer_coverage=sampled_layers / float(num_players - 1),
        efficiency_residual=float(values.sum() - total_value),
        layer_counts=layer_counts,
        pair_counts=pair_counts,
    )
    return DiffResult(
        values=values,
        diagnostics=diagnostics,
        differential_matrix=differential_matrix,
    )


__all__ = [
    "DiffDiagnostics",
    "DiffResult",
    "DifferentialCoverageError",
    "aggregate_diff_samples",
    "diff_layer_distribution",
    "estimate_diff",
]

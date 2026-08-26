"""Jia et al.'s basic group-testing Shapley estimator.

This module is a clean-room port of Algorithm 1 in
"Towards Efficient Data Valuation Based on the Shapley Value" (AISTATS
2019).  One sampled coalition consumes one physical utility call.  Empty and
grand-coalition utilities consume two additional calls and are used for the
generic-game centering and efficiency constraint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class GroupTestingEvaluator(Protocol):
    """Evaluator interface needed by :func:`estimate_group_testing`."""

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray: ...

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class GroupTestingDiagnostics:
    """Sampling, coverage, solver, and exact call accounting."""

    algorithm: str
    num_players: int
    target_call_budget: int
    utility_evaluations: int
    unused_calls: int
    boundary_evaluations: int
    num_group_tests: int
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    normalization: str
    normalization_constant: float
    efficiency_projection: bool
    solver: str
    empty_utility: float
    full_utility: float
    sampled_layers: int
    missing_layers: int
    layer_coverage: float
    layer_counts: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class GroupTestingResult:
    """Estimated Shapley vector and group-testing diagnostics."""

    values: np.ndarray
    diagnostics: GroupTestingDiagnostics


@dataclass(frozen=True)
class _GroupTestingPayload:
    num_players: int
    num_samples: int
    empty_utility: float
    seed_words: tuple[int, ...]


def group_testing_layer_distribution(
    num_players: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Algorithm-1 sizes, probabilities, and ``Z``.

    The paper defines

    ``Z = 2 H_(n-1)`` and
    ``q(k) = (1/k + 1/(n-k)) / Z`` for ``k=1,...,n-1``.
    """
    if num_players < 2:
        raise ValueError("group testing requires at least two players")
    layers = np.arange(1, num_players, dtype=np.int64)
    normalization = 2.0 * float(
        np.sum(1.0 / np.arange(1, num_players, dtype=np.float64))
    )
    probabilities = (
        1.0 / layers.astype(np.float64)
        + 1.0 / (num_players - layers).astype(np.float64)
    ) / normalization
    # Avoid a platform-dependent final-bit drift in rng.choice's cumulative
    # probabilities while retaining the paper's exact distribution.
    probabilities /= probabilities.sum()
    return layers, probabilities, normalization


def _reconstruct_group_testing_values(
    raw_scores: np.ndarray,
    total_value: float,
) -> np.ndarray:
    """Enforce efficiency while preserving every estimated difference."""
    scores = np.asarray(raw_scores, dtype=np.float64)
    return scores + (float(total_value) - float(scores.sum())) / len(scores)


def aggregate_group_testing_samples(
    coalitions: np.ndarray,
    utilities: np.ndarray,
    *,
    empty_utility: float,
    full_utility: float,
) -> np.ndarray:
    """Aggregate pre-evaluated Algorithm-1 samples.

    This helper is primarily an independently testable statement of the
    estimator.  Callers are responsible for drawing each row with the
    Algorithm-1 size distribution.
    """
    rows = np.asarray(coalitions, dtype=bool)
    observed = np.asarray(utilities, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] < 2:
        raise ValueError("coalitions must be a 2-D array with n >= 2")
    if observed.shape != (len(rows),):
        raise ValueError("one utility is required per coalition")
    if len(rows) < 1:
        raise ValueError("at least one group test is required")
    sizes = rows.sum(axis=1)
    if np.any(sizes < 1) or np.any(sizes >= rows.shape[1]):
        raise ValueError("group-test coalitions must have sizes 1,...,n-1")

    _, _, normalization = group_testing_layer_distribution(rows.shape[1])
    centered = observed - float(empty_utility)
    raw_scores = normalization * np.sum(
        rows * centered[:, None], axis=0, dtype=np.float64
    ) / len(rows)
    return _reconstruct_group_testing_values(
        raw_scores, float(full_utility) - float(empty_utility)
    )


def _run_group_testing_chunk(
    game: Any,
    payload: _GroupTestingPayload,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample and evaluate one coarse worker chunk."""
    rng = np.random.default_rng(np.random.SeedSequence(payload.seed_words))
    layers, probabilities, normalization = (
        group_testing_layer_distribution(payload.num_players)
    )
    score_sums = np.zeros(payload.num_players, dtype=np.float64)
    layer_counts = np.zeros(payload.num_players - 1, dtype=np.int64)

    for _ in range(payload.num_samples):
        layer = int(rng.choice(layers, p=probabilities))
        included = rng.choice(
            payload.num_players, size=layer, replace=False
        )
        coalition = np.zeros(payload.num_players, dtype=bool)
        coalition[included] = True
        utility = float(game.evaluate(coalition))
        score_sums[included] += normalization * (
            utility - payload.empty_utility
        )
        layer_counts[layer - 1] += 1

    return score_sums, layer_counts, payload.num_samples


def _split_tasks(
    num_samples: int,
    num_tasks: int,
    seed: int,
    *,
    num_players: int,
    empty_utility: float,
) -> list[_GroupTestingPayload]:
    actual_tasks = min(num_samples, num_tasks)
    quotient, remainder = divmod(num_samples, actual_tasks)
    child_sequences = np.random.SeedSequence(seed).spawn(actual_tasks)
    return [
        _GroupTestingPayload(
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


def estimate_group_testing(
    evaluator: GroupTestingEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> GroupTestingResult:
    """Estimate every Shapley value with basic group testing.

    Exactly ``total_call_budget`` physical utility evaluations are used:
    ``total_call_budget - 2`` random group tests plus the empty and full
    coalition.  The returned closed-form solution is equivalent to satisfying
    all score-induced pairwise differences and the efficiency constraint.
    """
    if num_players < 2:
        raise ValueError("group testing requires at least two players")
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
    payloads = _split_tasks(
        num_samples,
        num_tasks,
        seed,
        num_players=num_players,
        empty_utility=empty_utility,
    )
    task_results = evaluator.run_game_tasks(
        _run_group_testing_chunk,
        payloads,
        chunksize=task_chunksize,
    )
    if len(task_results) != len(payloads):
        raise RuntimeError("evaluator returned the wrong number of tasks")

    score_sums = np.zeros(num_players, dtype=np.float64)
    layer_counts = np.zeros(num_players - 1, dtype=np.int64)
    observed_samples = 0
    for task_scores, task_layers, task_samples in task_results:
        if task_scores.shape != score_sums.shape:
            raise RuntimeError("group-testing worker returned invalid scores")
        if task_layers.shape != layer_counts.shape:
            raise RuntimeError("group-testing worker returned invalid layers")
        score_sums += task_scores
        layer_counts += task_layers
        observed_samples += int(task_samples)
    if observed_samples != num_samples:
        raise RuntimeError("group-testing sample accounting disagrees")

    raw_scores = score_sums / num_samples
    total_value = full_utility - empty_utility
    values = _reconstruct_group_testing_values(raw_scores, total_value)
    sampled_layers = int(np.count_nonzero(layer_counts))
    _, _, normalization = group_testing_layer_distribution(num_players)
    diagnostics = GroupTestingDiagnostics(
        algorithm="group_testing_jia2019_algorithm1",
        num_players=num_players,
        target_call_budget=total_call_budget,
        utility_evaluations=num_samples + 2,
        unused_calls=0,
        boundary_evaluations=2,
        num_group_tests=num_samples,
        num_tasks=len(payloads),
        requested_num_tasks=num_tasks,
        sampling="q_k_proportional_to_1_over_k_plus_1_over_n_minus_k",
        normalization="utility_minus_empty",
        normalization_constant=normalization,
        efficiency_projection=True,
        solver="closed_form_complete_score_differences",
        empty_utility=empty_utility,
        full_utility=full_utility,
        sampled_layers=sampled_layers,
        missing_layers=(num_players - 1) - sampled_layers,
        layer_coverage=sampled_layers / float(num_players - 1),
        layer_counts=layer_counts,
    )
    return GroupTestingResult(values=values, diagnostics=diagnostics)


__all__ = [
    "GroupTestingDiagnostics",
    "GroupTestingResult",
    "aggregate_group_testing_samples",
    "estimate_group_testing",
    "group_testing_layer_distribution",
]

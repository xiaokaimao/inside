r"""KernelSHAP and official GELS-Shapley baselines for cooperative games.

The estimators in this module use the persistent :class:`GameEvaluator`
interface rather than passing scikit-learn objects through a second process
pool.  Both methods draw an inner coalition by first sampling

.. math::

   q_s = \frac{n}{2 H_{n-1} s(n-s)},\qquad s=1,\ldots,n-1,

and then sampling uniformly on the size-``s`` Boolean slice.

``estimate_kernel_shap`` implements the original dataset-sampling version of
KernelSHAP: it solves the empirical constrained least-squares problem.  It is
consistent, but its random empirical Gram inverse does not provide a general
finite-sample unbiasedness guarantee.

``estimate_gels_shapley`` independently implements Algorithm 3 of Li and Yu
(ICLR 2024) and was checked against the authors' ``watml/fastpvalue`` code at
commit ``34392c53f8d609aebb5e0c0e57c165411d291a46``.  It forms a
self-normalized conditional utility mean for every player, multiplies it by
``H_{n-1}``, and adds the common offset required by Shapley efficiency.  The
authors' zero-count fallback is retained exactly: an unobserved player's raw
ranking score is zero before the common offset is applied.

For a deterministic game each method spends two calls on ``v(empty)`` and
``v(full)`` and every remaining call on one inner coalition.  No utility
failure is silently replaced by zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol

import numpy as np


class RegressionGameEvaluator(Protocol):
    """Part of :class:`frame_ofa.parallel.GameEvaluator` used here."""

    def evaluate(self, coalitions: np.ndarray) -> np.ndarray: ...

    def run_game_tasks(
        self,
        task_func: Callable[[Any, Any], Any],
        payloads: list[Any],
        *,
        chunksize: int = 1,
    ) -> list[Any]: ...


@dataclass(frozen=True)
class RegressionBaselineDiagnostics:
    """Sampling, solver, conditioning, and physical-call diagnostics."""

    method: str
    num_players: int
    utility_evaluations: int
    target_call_budget: int
    unused_calls: int
    boundary_utility_evaluations: int
    inner_utility_evaluations: int
    num_samples: int
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    solver: str
    finite_sample_bias: str
    utility_centering: str
    efficiency_constraint: bool
    ridge: float
    projected_gram_rank: int
    projected_gram_dimension: int
    projected_gram_condition: float
    empty_utility: float
    full_utility: float
    layer_counts: np.ndarray = field(repr=False)
    inclusion_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64), repr=False
    )
    zero_inclusion_players: int = 0


@dataclass(frozen=True)
class RegressionBaselineResult:
    """A Shapley estimate and its auditable run diagnostics."""

    values: np.ndarray
    diagnostics: RegressionBaselineDiagnostics


@dataclass(frozen=True)
class _RegressionChunkPayload:
    method: Literal["kernel_shap", "gels_shapley"]
    num_players: int
    num_samples: int
    empty_utility: float
    seed_words: tuple[int, ...]
    buffer_rows: int


@dataclass(frozen=True)
class _ProjectedSolve:
    values: np.ndarray
    rank: int
    dimension: int
    condition: float


# Bound the sum of dense per-task KernelSHAP Gram matrices returned to the
# parent.  Wine (n=142) retains all 128 requested tasks under this cap.
_MAX_AGGREGATE_TASK_RESULT_BYTES = 256 * 2**20


def shapley_kernel_size_distribution(
    num_players: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the normalized KernelSHAP/GELS inner-size distribution."""
    if num_players < 2:
        raise ValueError("at least two players are required")
    sizes = np.arange(1, num_players, dtype=np.int64)
    weights = 1.0 / (sizes * (num_players - sizes))
    probabilities = weights / weights.sum(dtype=np.float64)
    return sizes, probabilities.astype(np.float64, copy=False)


def _harmonic_number(order: int) -> float:
    if order < 1:
        return 0.0
    return float(
        np.sum(1.0 / np.arange(1, order + 1, dtype=np.float64))
    )


def _helmert_contrasts(num_players: int) -> np.ndarray:
    """An orthonormal basis for the efficiency subspace ``1^perp``."""
    basis = np.zeros((num_players, num_players - 1), dtype=np.float64)
    for column in range(num_players - 1):
        denominator = np.sqrt((column + 1) * (column + 2))
        basis[: column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    return basis


def _solve_constrained_kernel_moments(
    gram: np.ndarray,
    rhs: np.ndarray,
    total_value: float,
    *,
    ridge: float,
) -> _ProjectedSolve:
    """Solve constrained WLS robustly on an orthonormal contrast basis.

    ``ridge`` means ``ridge * ||phi||_2^2`` in the original constrained
    objective.  With zero ridge, an eigendecomposition supplies the
    minimum-norm solution when the sampled design is rank deficient.
    """
    gram = np.asarray(gram, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    num_players = rhs.size
    if gram.shape != (num_players, num_players):
        raise ValueError("gram and rhs have incompatible shapes")
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise ValueError("kernel moments must be finite")
    if not np.isfinite(total_value):
        raise ValueError("total_value must be finite")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")

    # Symmetrization removes harmless roundoff asymmetry from independent
    # task aggregation before the symmetric eigensolver is used.
    gram = (gram + gram.T) / 2.0
    contrasts = _helmert_contrasts(num_players)
    base = np.full(
        num_players, total_value / num_players, dtype=np.float64
    )
    projected = contrasts.T @ gram @ contrasts
    projected = (projected + projected.T) / 2.0
    projected_rhs = contrasts.T @ (rhs - gram @ base)

    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    maximum = float(max(np.max(eigenvalues, initial=0.0), 0.0))
    tolerance = (
        np.finfo(np.float64).eps
        * max(projected.shape[0], 1)
        * max(maximum, 1.0)
    )
    positive = eigenvalues > tolerance
    rank = int(np.count_nonzero(positive))
    if rank:
        minimum_positive = float(np.min(eigenvalues[positive]))
        condition = maximum / minimum_positive
    else:
        condition = float("inf")

    # A PSD Gram can acquire tiny negative eigenvalues from floating-point
    # summation.  Clipping them is preferable to turning a tiny roundoff
    # error into a large, sign-reversing inverse.
    nonnegative = np.maximum(eigenvalues, 0.0)
    transformed_rhs = eigenvectors.T @ projected_rhs
    if ridge > 0:
        coefficients = transformed_rhs / (nonnegative + ridge)
    else:
        coefficients = np.zeros_like(transformed_rhs)
        coefficients[positive] = (
            transformed_rhs[positive] / eigenvalues[positive]
        )
    values = base + contrasts @ (eigenvectors @ coefficients)
    # Make the exact efficiency constraint robust to final BLAS roundoff.
    values += (total_value - float(values.sum())) / num_players
    return _ProjectedSolve(
        values=values,
        rank=rank,
        dimension=num_players - 1,
        condition=float(condition),
    )


def _efficiency_correct(
    raw_scores: np.ndarray, total_value: float
) -> np.ndarray:
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    values = raw_scores + (
        total_value - float(raw_scores.sum())
    ) / raw_scores.size
    values += (total_value - float(values.sum())) / raw_scores.size
    return values


def _solve_gels_shapley_ratio(
    utility_sums: np.ndarray,
    inclusion_counts: np.ndarray,
    total_value: float,
) -> np.ndarray:
    """Apply the authors' Algorithm-3 ratio and efficiency offset.

    The official implementation replaces a zero denominator with ``-1``.
    Its corresponding numerator is necessarily zero, so this assigns a raw
    score of zero.  Keeping that behavior matters for exact source parity at
    very small budgets.
    """
    utility_sums = np.asarray(utility_sums, dtype=np.float64)
    inclusion_counts = np.asarray(inclusion_counts)
    if utility_sums.ndim != 1 or utility_sums.size < 2:
        raise ValueError("utility_sums must contain at least two players")
    if inclusion_counts.shape != utility_sums.shape:
        raise ValueError("utility_sums and inclusion_counts must align")
    if not np.isfinite(utility_sums).all() or not np.isfinite(total_value):
        raise ValueError("GELS moments must be finite")
    if not np.issubdtype(inclusion_counts.dtype, np.integer):
        if not np.equal(inclusion_counts, np.floor(inclusion_counts)).all():
            raise ValueError("inclusion_counts must be integers")
    counts = inclusion_counts.astype(np.int64, copy=True)
    if np.any(counts < 0):
        raise ValueError("inclusion_counts must be nonnegative")
    zero = counts == 0
    if np.any(utility_sums[zero] != 0.0):
        raise ValueError("a zero inclusion count must have zero utility sum")
    counts[zero] = -1
    conditional_means = utility_sums / counts
    raw_scores = (
        _harmonic_number(utility_sums.size - 1) * conditional_means
    )
    return _efficiency_correct(raw_scores, total_value)


def _evaluate_regression_chunk(
    game: Any,
    payload: _RegressionChunkPayload,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Sample and aggregate one deterministic coarse worker task."""
    num_players = payload.num_players
    rng = np.random.default_rng(
        np.random.SeedSequence(payload.seed_words)
    )
    sizes, probabilities = shapley_kernel_size_distribution(num_players)
    layer_counts = np.zeros(num_players - 1, dtype=np.int64)
    if payload.method == "kernel_shap":
        first = np.zeros((num_players, num_players), dtype=np.float64)
        second = np.zeros(num_players, dtype=np.float64)
    else:
        first = np.zeros(num_players, dtype=np.float64)
        second = np.zeros(num_players, dtype=np.int64)

    buffer_rows = min(payload.buffer_rows, payload.num_samples)
    rows = np.zeros((buffer_rows, num_players), dtype=bool)
    responses = np.empty(buffer_rows, dtype=np.float64)

    for start in range(0, payload.num_samples, buffer_rows):
        stop = min(start + buffer_rows, payload.num_samples)
        used = stop - start
        rows[:used].fill(False)
        for local_index in range(used):
            size = int(rng.choice(sizes, p=probabilities))
            included = rng.choice(
                num_players, size=size, replace=False
            )
            rows[local_index, included] = True
            utility = float(game.evaluate(rows[local_index]))
            if not np.isfinite(utility):
                raise ValueError("utility evaluations must be finite")
            responses[local_index] = utility
            layer_counts[size - 1] += 1

        design = rows[:used].astype(np.float64)
        if payload.method == "kernel_shap":
            centered = responses[:used] - payload.empty_utility
            first += design.T @ design
            second += design.T @ centered
        else:
            first += design.T @ responses[:used]
            second += rows[:used].sum(axis=0, dtype=np.int64)

    return first, second, layer_counts, payload.num_samples


def _bounded_task_count(
    num_players: int,
    num_samples: int,
    requested_num_tasks: int,
    method: Literal["kernel_shap", "gels_shapley"],
) -> int:
    if method == "kernel_shap":
        bytes_per_task = (
            num_players * num_players * np.dtype(np.float64).itemsize
            + num_players * np.dtype(np.float64).itemsize
            + (num_players - 1) * np.dtype(np.int64).itemsize
        )
        memory_limit = max(
            1, _MAX_AGGREGATE_TASK_RESULT_BYTES // bytes_per_task
        )
    else:
        memory_limit = requested_num_tasks
    return min(num_samples, requested_num_tasks, memory_limit)


def _run_regression_sampling(
    evaluator: RegressionGameEvaluator,
    num_players: int,
    num_samples: int,
    seed: int,
    *,
    method: Literal["kernel_shap", "gels_shapley"],
    empty_utility: float,
    num_tasks: int,
    task_chunksize: int,
    buffer_rows: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    actual_tasks = _bounded_task_count(
        num_players, num_samples, num_tasks, method
    )
    quotient, remainder = divmod(num_samples, actual_tasks)
    child_sequences = np.random.SeedSequence(seed).spawn(actual_tasks)
    payloads = [
        _RegressionChunkPayload(
            method=method,
            num_players=num_players,
            num_samples=quotient + int(index < remainder),
            empty_utility=float(empty_utility),
            seed_words=tuple(
                int(word)
                for word in child_sequences[index].generate_state(
                    4, dtype=np.uint32
                )
            ),
            buffer_rows=buffer_rows,
        )
        for index in range(actual_tasks)
    ]
    results = evaluator.run_game_tasks(
        _evaluate_regression_chunk,
        payloads,
        chunksize=task_chunksize,
    )

    if method == "kernel_shap":
        first = np.zeros((num_players, num_players), dtype=np.float64)
        second = np.zeros(num_players, dtype=np.float64)
    else:
        first = np.zeros(num_players, dtype=np.float64)
        second = np.zeros(num_players, dtype=np.int64)
    layer_counts = np.zeros(num_players - 1, dtype=np.int64)
    actual_evaluations = 0
    for task_first, task_second, task_counts, task_evaluations in results:
        first += task_first
        second += task_second
        layer_counts += task_counts
        actual_evaluations += int(task_evaluations)
    return first, second, layer_counts, actual_evaluations, actual_tasks


def _validate_public_inputs(
    num_players: int,
    total_call_budget: int,
    seed: int,
    num_tasks: int,
    task_chunksize: int,
    buffer_rows: int,
) -> None:
    if num_players < 2:
        raise ValueError("at least two players are required")
    if total_call_budget < 3:
        raise ValueError(
            "total_call_budget must cover two boundaries and one sample"
        )
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if task_chunksize < 1:
        raise ValueError("task_chunksize must be positive")
    if buffer_rows < 1:
        raise ValueError("buffer_rows must be positive")


def _evaluate_boundaries(
    evaluator: RegressionGameEvaluator, num_players: int
) -> tuple[float, float]:
    rows = np.vstack(
        (
            np.zeros(num_players, dtype=bool),
            np.ones(num_players, dtype=bool),
        )
    )
    utilities = np.asarray(evaluator.evaluate(rows), dtype=np.float64)
    if utilities.shape != (2,) or not np.isfinite(utilities).all():
        raise ValueError("boundary evaluator must return two finite utilities")
    return float(utilities[0]), float(utilities[1])


def estimate_kernel_shap(
    evaluator: RegressionGameEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    ridge: float = 0.0,
    task_chunksize: int = 1,
    buffer_rows: int = 256,
) -> RegressionBaselineResult:
    """Estimate Shapley values with sampled-Gram constrained KernelSHAP.

    The budget includes the two exact boundary calls.  The default has no
    hidden regularization; rank-deficient samples use a stable pseudoinverse
    on the efficiency subspace.  Passing ``ridge > 0`` explicitly adds an
    L2 penalty to the original coefficient vector.
    """
    _validate_public_inputs(
        num_players,
        total_call_budget,
        seed,
        num_tasks,
        task_chunksize,
        buffer_rows,
    )
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("ridge must be finite and nonnegative")
    empty_utility, full_utility = _evaluate_boundaries(
        evaluator, num_players
    )
    num_samples = total_call_budget - 2
    gram, rhs, layer_counts, actual_inner, actual_tasks = (
        _run_regression_sampling(
            evaluator,
            num_players,
            num_samples,
            seed,
            method="kernel_shap",
            empty_utility=empty_utility,
            num_tasks=num_tasks,
            task_chunksize=task_chunksize,
            buffer_rows=buffer_rows,
        )
    )
    if actual_inner != num_samples:
        raise RuntimeError("KernelSHAP physical call accounting mismatch")
    solved = _solve_constrained_kernel_moments(
        gram,
        rhs,
        full_utility - empty_utility,
        ridge=float(ridge),
    )
    actual_calls = actual_inner + 2
    diagnostics = RegressionBaselineDiagnostics(
        method="kernel_shap",
        num_players=num_players,
        utility_evaluations=actual_calls,
        target_call_budget=total_call_budget,
        unused_calls=total_call_budget - actual_calls,
        boundary_utility_evaluations=2,
        inner_utility_evaluations=actual_inner,
        num_samples=num_samples,
        num_tasks=actual_tasks,
        requested_num_tasks=num_tasks,
        sampling="iid_shapley_kernel_sizes_uniform_slice",
        solver="sampled_gram_constrained_wls_eigh_pseudoinverse",
        finite_sample_bias=(
            "consistent_finite_sample_unbiasedness_not_guaranteed"
        ),
        utility_centering="v(S)-v(empty)",
        efficiency_constraint=True,
        ridge=float(ridge),
        projected_gram_rank=solved.rank,
        projected_gram_dimension=solved.dimension,
        projected_gram_condition=solved.condition,
        empty_utility=empty_utility,
        full_utility=full_utility,
        layer_counts=layer_counts,
    )
    return RegressionBaselineResult(solved.values, diagnostics)


def estimate_gels_shapley(
    evaluator: RegressionGameEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
    buffer_rows: int = 256,
) -> RegressionBaselineResult:
    """Estimate Shapley values with official Algorithm-3 GELS-Shapley.

    For ``m = total_call_budget - 2`` inner samples, let

    ``A_i = sum_t 1{i in S_t} v(S_t)`` and
    ``C_i = sum_t 1{i in S_t}``.  The raw score is
    ``H_{n-1} A_i / C_i`` and a common offset enforces efficiency.

    The paper calls the estimate unbiased.  Strictly, the official zero-count
    fallback has an exponentially small finite-sample bias because
    ``P(C_i=0)=2^{-m}``; diagnostics expose whether that branch occurred.
    """
    _validate_public_inputs(
        num_players,
        total_call_budget,
        seed,
        num_tasks,
        task_chunksize,
        buffer_rows,
    )
    empty_utility, full_utility = _evaluate_boundaries(
        evaluator, num_players
    )
    num_samples = total_call_budget - 2
    utility_sums, inclusion_counts, layer_counts, actual_inner, actual_tasks = (
        _run_regression_sampling(
            evaluator,
            num_players,
            num_samples,
            seed,
            method="gels_shapley",
            empty_utility=empty_utility,
            num_tasks=num_tasks,
            task_chunksize=task_chunksize,
            buffer_rows=buffer_rows,
        )
    )
    if actual_inner != num_samples:
        raise RuntimeError("GELS physical call accounting mismatch")
    values = _solve_gels_shapley_ratio(
        utility_sums,
        inclusion_counts,
        full_utility - empty_utility,
    )
    actual_calls = actual_inner + 2
    diagnostics = RegressionBaselineDiagnostics(
        method="gels_shapley",
        num_players=num_players,
        utility_evaluations=actual_calls,
        target_call_budget=total_call_budget,
        unused_calls=total_call_budget - actual_calls,
        boundary_utility_evaluations=2,
        inner_utility_evaluations=actual_inner,
        num_samples=num_samples,
        num_tasks=actual_tasks,
        requested_num_tasks=num_tasks,
        sampling="iid_shapley_kernel_sizes_uniform_slice",
        solver="official_algorithm_3_self_normalized_ratio",
        finite_sample_bias=(
            "conditionally_unbiased_given_positive_inclusion_counts;"
            "official_zero_count_fallback_has_probability_2^-num_samples"
        ),
        utility_centering="none_official_algorithm_3",
        efficiency_constraint=True,
        ridge=0.0,
        projected_gram_rank=-1,
        projected_gram_dimension=num_players - 1,
        projected_gram_condition=float("nan"),
        empty_utility=empty_utility,
        full_utility=full_utility,
        layer_counts=layer_counts,
        inclusion_counts=inclusion_counts,
        zero_inclusion_players=int(np.count_nonzero(inclusion_counts == 0)),
    )
    return RegressionBaselineResult(values, diagnostics)


__all__ = [
    "RegressionBaselineDiagnostics",
    "RegressionBaselineResult",
    "estimate_gels_shapley",
    "estimate_kernel_shap",
    "shapley_kernel_size_distribution",
]

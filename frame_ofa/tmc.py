"""Permutation Monte Carlo and early-truncated TMC-Shapley.

The full-permutation estimator is the deliberately non-truncated variant:

* every sampled permutation produces one marginal contribution per player;
* no suffix is silently filled with zeros;
* no post-hoc efficiency projection is applied; and
* empty/full utilities are evaluated once and reused exactly.

For ``n > 1``, one complete permutation therefore costs ``n - 1`` interior
utility evaluations after two shared endpoint evaluations.  Given a target
budget ``B``, the estimator uses

``m = floor((B - 2) / (n - 1))``

complete permutations and reports the fewer than ``n - 1`` unused calls.
This differs from the copied baseline's default truncated TMC procedure,
whose physical call count is data dependent.

``estimate_tmc_shapley`` is a separate, faithful adaptation of that external
early-truncated baseline.  It evaluates every visited prefix (including the
full prefix on an untruncated path), stops after ``patience + 1`` consecutive
utilities within the configured tolerance of ``v(N)``, and leaves all
unvisited suffix contributions at zero.  It does not apply an efficiency
projection.  To compare methods under a strict physical-call cap without a
random stopping-time bias, it fixes the number of permutations using their
worst-case ``n``-call cost and reports the data-dependent truncation savings as
unused calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np


class PermutationGameEvaluator(Protocol):
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
class FullPermutationDiagnostics:
    """Exact sample, task, and physical utility-call accounting."""

    num_players: int
    num_permutations: int
    target_call_budget: int
    utility_evaluations: int
    unused_calls: int
    boundary_utility_evaluations: int
    interior_utility_evaluations: int
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    truncation: bool
    efficiency_projection: bool
    boundary_reuse: bool
    empty_utility: float
    full_utility: float
    sample_variance: np.ndarray = field(repr=False)
    variance_of_mean: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class FullPermutationResult:
    """A full-permutation estimate and its diagnostics."""

    values: np.ndarray
    diagnostics: FullPermutationDiagnostics


@dataclass(frozen=True)
class _PermutationChunkPayload:
    num_players: int
    first_permutation: int
    num_permutations: int
    seed: int
    empty_utility: float
    full_utility: float


def full_permutations_for_budget(
    num_players: int,
    total_call_budget: int,
) -> tuple[int, int, int]:
    """Map a physical-call budget to complete permutation samples.

    Returns ``(num_permutations, utility_evaluations, unused_calls)``.
    The two shared endpoint calls are included in ``utility_evaluations``.
    At least one complete permutation is required when ``num_players > 1``.
    For a one-player game, the two endpoints already identify the Shapley
    value, so ``num_permutations`` is reported as zero.
    """
    if num_players < 1:
        raise ValueError("num_players must be positive")
    if total_call_budget < 0:
        raise ValueError("total_call_budget must be nonnegative")
    if num_players == 1:
        if total_call_budget < 2:
            raise ValueError("a one-player game requires two endpoint calls")
        return 0, 2, total_call_budget - 2

    calls_per_permutation = num_players - 1
    if total_call_budget < 2 + calls_per_permutation:
        raise ValueError(
            "total_call_budget cannot fund one complete permutation"
        )
    num_permutations = (
        total_call_budget - 2
    ) // calls_per_permutation
    utility_evaluations = 2 + num_permutations * calls_per_permutation
    return (
        num_permutations,
        utility_evaluations,
        total_call_budget - utility_evaluations,
    )


def _validate_permutation(
    permutation: np.ndarray,
    num_players: int,
) -> np.ndarray:
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (num_players,):
        raise ValueError("permutation must contain one entry per player")
    if not np.array_equal(np.sort(order), np.arange(num_players)):
        raise ValueError("permutation must contain each player exactly once")
    return order


def _full_permutation_contributions_unchecked(
    game: Any,
    permutation: np.ndarray,
    empty_utility: float,
    full_utility: float,
) -> tuple[np.ndarray, int]:
    """Evaluate all interior prefixes of one already-validated order."""
    num_players = len(permutation)
    contributions = np.zeros(num_players, dtype=np.float64)
    if num_players == 1:
        contributions[int(permutation[0])] = (
            float(full_utility) - float(empty_utility)
        )
        return contributions, 0

    coalition = np.zeros(num_players, dtype=bool)
    previous_utility = float(empty_utility)
    for player_value in permutation[:-1]:
        player = int(player_value)
        coalition[player] = True
        current_utility = float(game.evaluate(coalition))
        contributions[player] = current_utility - previous_utility
        previous_utility = current_utility

    final_player = int(permutation[-1])
    contributions[final_player] = float(full_utility) - previous_utility
    return contributions, num_players - 1


def full_permutation_contributions(
    game: Any,
    permutation: np.ndarray,
    empty_utility: float,
    full_utility: float,
) -> np.ndarray:
    """Return the complete marginal vector along one explicit permutation.

    This public primitive is useful for exhaustive small-game validation.  It
    performs exactly ``n - 1`` calls to ``game.evaluate`` because the caller
    supplies the two endpoint utilities.
    """
    order = np.asarray(permutation, dtype=np.int64)
    if order.ndim != 1 or len(order) < 1:
        raise ValueError("permutation must be a nonempty one-dimensional array")
    order = _validate_permutation(order, len(order))
    contributions, _ = _full_permutation_contributions_unchecked(
        game,
        order,
        empty_utility,
        full_utility,
    )
    return contributions


def _permutation_rng(seed: int, sample_index: int) -> np.random.Generator:
    """Give each sample a seed independent of worker/task partitioning."""
    sequence = np.random.SeedSequence(
        entropy=seed,
        spawn_key=(sample_index,),
    )
    return np.random.default_rng(sequence)


def _evaluate_permutation_chunk(
    game: Any,
    payload: _PermutationChunkPayload,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Evaluate a contiguous, deterministically indexed sample chunk."""
    sums = np.zeros(payload.num_players, dtype=np.float64)
    squared_sums = np.zeros(payload.num_players, dtype=np.float64)
    utility_evaluations = 0

    stop = payload.first_permutation + payload.num_permutations
    for sample_index in range(payload.first_permutation, stop):
        permutation = _permutation_rng(
            payload.seed,
            sample_index,
        ).permutation(payload.num_players)
        contributions, sample_calls = (
            _full_permutation_contributions_unchecked(
                game,
                permutation,
                payload.empty_utility,
                payload.full_utility,
            )
        )
        sums += contributions
        squared_sums += contributions * contributions
        utility_evaluations += sample_calls

    return (
        sums,
        squared_sums,
        utility_evaluations,
        payload.num_permutations,
    )


def _permutation_payloads(
    *,
    num_players: int,
    num_permutations: int,
    seed: int,
    empty_utility: float,
    full_utility: float,
    num_tasks: int,
) -> list[_PermutationChunkPayload]:
    actual_tasks = min(num_tasks, num_permutations)
    base_samples, remainder = divmod(num_permutations, actual_tasks)
    payloads: list[_PermutationChunkPayload] = []
    first = 0
    for task in range(actual_tasks):
        task_samples = base_samples + int(task < remainder)
        payloads.append(
            _PermutationChunkPayload(
                num_players=num_players,
                first_permutation=first,
                num_permutations=task_samples,
                seed=seed,
                empty_utility=empty_utility,
                full_utility=full_utility,
            )
        )
        first += task_samples
    return payloads


def estimate_full_permutation_mc(
    evaluator: PermutationGameEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> FullPermutationResult:
    """Estimate Shapley values with non-truncated permutation Monte Carlo.

    ``total_call_budget`` is a target number of physical utility invocations,
    not a number of permutations.  Only complete permutations are used, so a
    remainder smaller than ``n - 1`` is intentionally left unused and exposed
    in ``diagnostics.unused_calls``.  The estimator never truncates a path or
    projects its result onto the efficiency hyperplane.

    Sampling and model fitting happen inside coarse game-aware tasks.  A
    sample's permutation is keyed by ``(seed, sample_index)``, so changing the
    process scheduling does not change which permutations are drawn.
    """
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    if task_chunksize < 1:
        raise ValueError("task_chunksize must be positive")

    (
        num_permutations,
        planned_utility_evaluations,
        unused_calls,
    ) = full_permutations_for_budget(num_players, total_call_budget)

    endpoints = np.zeros((2, num_players), dtype=bool)
    endpoints[1] = True
    endpoint_utilities = np.asarray(
        evaluator.evaluate(endpoints),
        dtype=np.float64,
    )
    if endpoint_utilities.shape != (2,):
        raise RuntimeError("game evaluator returned invalid endpoint utilities")
    empty_utility, full_utility = endpoint_utilities.tolist()

    if num_players == 1:
        values = np.asarray(
            [full_utility - empty_utility],
            dtype=np.float64,
        )
        zeros = np.zeros(1, dtype=np.float64)
        diagnostics = FullPermutationDiagnostics(
            num_players=1,
            num_permutations=0,
            target_call_budget=total_call_budget,
            utility_evaluations=2,
            unused_calls=unused_calls,
            boundary_utility_evaluations=2,
            interior_utility_evaluations=0,
            num_tasks=0,
            requested_num_tasks=num_tasks,
            sampling="exact_endpoints_one_player",
            truncation=False,
            efficiency_projection=False,
            boundary_reuse=True,
            empty_utility=empty_utility,
            full_utility=full_utility,
            sample_variance=zeros,
            variance_of_mean=zeros.copy(),
        )
        return FullPermutationResult(values, diagnostics)

    payloads = _permutation_payloads(
        num_players=num_players,
        num_permutations=num_permutations,
        seed=seed,
        empty_utility=empty_utility,
        full_utility=full_utility,
        num_tasks=num_tasks,
    )
    chunk_results = evaluator.run_game_tasks(
        _evaluate_permutation_chunk,
        payloads,
        chunksize=task_chunksize,
    )
    if len(chunk_results) != len(payloads):
        raise RuntimeError("game evaluator returned the wrong number of tasks")

    sums = np.zeros(num_players, dtype=np.float64)
    squared_sums = np.zeros(num_players, dtype=np.float64)
    interior_utility_evaluations = 0
    observed_permutations = 0
    for chunk_sums, chunk_squared_sums, chunk_calls, chunk_samples in (
        chunk_results
    ):
        if (
            np.shape(chunk_sums) != sums.shape
            or np.shape(chunk_squared_sums) != squared_sums.shape
        ):
            raise RuntimeError("a permutation worker returned an invalid shape")
        sums += chunk_sums
        squared_sums += chunk_squared_sums
        interior_utility_evaluations += int(chunk_calls)
        observed_permutations += int(chunk_samples)

    expected_interior_calls = num_permutations * (num_players - 1)
    if observed_permutations != num_permutations:
        raise RuntimeError("permutation worker sample accounting disagrees")
    if interior_utility_evaluations != expected_interior_calls:
        raise RuntimeError("permutation worker utility-call accounting disagrees")
    utility_evaluations = 2 + interior_utility_evaluations
    if utility_evaluations != planned_utility_evaluations:
        raise RuntimeError("planned and observed utility calls disagree")

    values = sums / float(num_permutations)
    if num_permutations > 1:
        sample_variance = (
            squared_sums
            - float(num_permutations) * values * values
        ) / float(num_permutations - 1)
        sample_variance = np.maximum(sample_variance, 0.0)
    else:
        sample_variance = np.zeros(num_players, dtype=np.float64)
    variance_of_mean = sample_variance / float(num_permutations)

    diagnostics = FullPermutationDiagnostics(
        num_players=num_players,
        num_permutations=num_permutations,
        target_call_budget=total_call_budget,
        utility_evaluations=utility_evaluations,
        unused_calls=unused_calls,
        boundary_utility_evaluations=2,
        interior_utility_evaluations=interior_utility_evaluations,
        num_tasks=len(payloads),
        requested_num_tasks=num_tasks,
        sampling="uniform_random_permutations_with_replacement",
        truncation=False,
        efficiency_projection=False,
        boundary_reuse=True,
        empty_utility=empty_utility,
        full_utility=full_utility,
        sample_variance=sample_variance,
        variance_of_mean=variance_of_mean,
    )
    return FullPermutationResult(values, diagnostics)


@dataclass(frozen=True)
class TMCShapleyDiagnostics:
    """Observed path, truncation, and physical-call accounting for TMC."""

    num_players: int
    num_permutations: int
    target_call_budget: int
    utility_evaluations: int
    unused_calls: int
    worst_case_reserved_calls: int
    budget_remainder_calls: int
    truncation_saved_calls: int
    boundary_utility_evaluations: int
    prefix_utility_evaluations: int
    evaluated_prefixes: int
    truncated_permutations: int
    truncation_rate: float
    mean_prefix_length: float
    num_tasks: int
    requested_num_tasks: int
    sampling: str
    truncation: bool
    truncation_tolerance_input: float
    truncation_tolerance: float
    relative_tolerance: bool
    truncation_patience: int
    efficiency_projection: bool
    boundary_reuse: bool
    empty_utility: float
    full_utility: float
    sample_variance: np.ndarray = field(repr=False)
    variance_of_mean: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class TMCShapleyResult:
    """An early-truncated TMC-Shapley estimate and diagnostics."""

    values: np.ndarray
    diagnostics: TMCShapleyDiagnostics


@dataclass(frozen=True)
class _TMCChunkPayload:
    num_players: int
    first_permutation: int
    num_permutations: int
    seed: int
    empty_utility: float
    full_utility: float
    truncation_tolerance: float
    truncation_patience: int


def _tmc_permutation_contributions_unchecked(
    game: Any,
    permutation: np.ndarray,
    empty_utility: float,
    full_utility: float,
    truncation_tolerance: float,
    truncation_patience: int,
) -> tuple[np.ndarray, int, bool]:
    """Run one path with the exact stopping rule of the external baseline."""
    num_players = len(permutation)
    contributions = np.zeros(num_players, dtype=np.float64)
    coalition = np.zeros(num_players, dtype=bool)
    previous_utility = float(empty_utility)
    near_full_count = 0

    for prefix_length, player_value in enumerate(permutation, start=1):
        player = int(player_value)
        coalition[player] = True
        current_utility = float(game.evaluate(coalition))
        if not np.isfinite(current_utility):
            raise RuntimeError("game returned a non-finite prefix utility")
        contributions[player] = current_utility - previous_utility
        previous_utility = current_utility

        if abs(float(full_utility) - current_utility) <= truncation_tolerance:
            near_full_count += 1
            # The source baseline uses ``counter > patience``.  Its default
            # patience of five therefore requires six consecutive matches.
            if near_full_count > truncation_patience:
                return contributions, prefix_length, True
        else:
            near_full_count = 0

    return contributions, num_players, False


def tmc_permutation_contributions(
    game: Any,
    permutation: np.ndarray,
    empty_utility: float,
    full_utility: float,
    *,
    truncation_tolerance: float = 1e-3,
    relative_tolerance: bool = True,
    truncation_patience: int = 5,
) -> tuple[np.ndarray, int, bool]:
    """Evaluate one explicit TMC path for validation and small examples.

    Returns ``(contributions, evaluated_prefixes, truncated)``.  As in the
    source baseline, an untruncated path evaluates all ``n`` prefixes even
    when the caller already supplied ``full_utility``.
    """
    order = np.asarray(permutation, dtype=np.int64)
    if order.ndim != 1 or len(order) < 1:
        raise ValueError("permutation must be a nonempty one-dimensional array")
    order = _validate_permutation(order, len(order))
    tolerance = _resolve_tmc_tolerance(
        truncation_tolerance,
        relative_tolerance=relative_tolerance,
        full_utility=full_utility,
    )
    patience = _validate_tmc_patience(truncation_patience)
    return _tmc_permutation_contributions_unchecked(
        game,
        order,
        empty_utility,
        full_utility,
        tolerance,
        patience,
    )


def _validate_tmc_patience(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError("truncation_patience must be a nonnegative integer")
    patience = int(value)
    if patience < 0:
        raise ValueError("truncation_patience must be a nonnegative integer")
    return patience


def _resolve_tmc_tolerance(
    truncation_tolerance: float,
    *,
    relative_tolerance: bool,
    full_utility: float,
) -> float:
    tolerance = float(truncation_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("truncation_tolerance must be finite and nonnegative")
    if relative_tolerance:
        tolerance *= abs(float(full_utility))
    return tolerance


def _evaluate_tmc_chunk(
    game: Any,
    payload: _TMCChunkPayload,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    sums = np.zeros(payload.num_players, dtype=np.float64)
    squared_sums = np.zeros(payload.num_players, dtype=np.float64)
    evaluated_prefixes = 0
    truncated_permutations = 0
    stop = payload.first_permutation + payload.num_permutations
    for sample_index in range(payload.first_permutation, stop):
        permutation = _permutation_rng(
            payload.seed,
            sample_index,
        ).permutation(payload.num_players)
        contributions, calls, truncated = (
            _tmc_permutation_contributions_unchecked(
                game,
                permutation,
                payload.empty_utility,
                payload.full_utility,
                payload.truncation_tolerance,
                payload.truncation_patience,
            )
        )
        sums += contributions
        squared_sums += contributions * contributions
        evaluated_prefixes += calls
        truncated_permutations += int(truncated)
    return (
        sums,
        squared_sums,
        evaluated_prefixes,
        truncated_permutations,
        payload.num_permutations,
    )


def _tmc_payloads(
    *,
    num_players: int,
    num_permutations: int,
    seed: int,
    empty_utility: float,
    full_utility: float,
    truncation_tolerance: float,
    truncation_patience: int,
    num_tasks: int,
) -> list[_TMCChunkPayload]:
    actual_tasks = min(num_tasks, num_permutations)
    base_samples, remainder = divmod(num_permutations, actual_tasks)
    payloads: list[_TMCChunkPayload] = []
    first = 0
    for task in range(actual_tasks):
        task_samples = base_samples + int(task < remainder)
        payloads.append(
            _TMCChunkPayload(
                num_players=num_players,
                first_permutation=first,
                num_permutations=task_samples,
                seed=seed,
                empty_utility=empty_utility,
                full_utility=full_utility,
                truncation_tolerance=truncation_tolerance,
                truncation_patience=truncation_patience,
            )
        )
        first += task_samples
    return payloads


def estimate_tmc_shapley(
    evaluator: PermutationGameEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    truncation_tolerance: float = 1e-3,
    relative_tolerance: bool = True,
    truncation_patience: int = 5,
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> TMCShapleyResult:
    """Estimate Data Shapley with the copied library's TMC stopping rule.

    The external function accepts a fixed number of permutations and only
    reports its data-dependent call count.  This adapter maps the experiment's
    strict physical-call cap ``B`` to the conservative fixed sample count

    ``m = floor((B - 2) / num_players)``.

    This preserves the source method's fixed-``m`` sampling and avoids the
    length bias that could arise from repeatedly adding paths until a random,
    contribution-dependent stopping time exhausts the call budget.  Truncation
    savings are intentionally left unused and reported explicitly.  The plot
    uses the observed physical calls rather than the cap for this method.

    Stopping early is an approximation: unvisited suffix contributions stay
    zero and no efficiency projection is applied, matching the source default.
    Utility exceptions and non-finite values are deliberately propagated
    instead of being silently converted to zero.
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
    tolerance_input = float(truncation_tolerance)
    if not np.isfinite(tolerance_input) or tolerance_input < 0.0:
        raise ValueError("truncation_tolerance must be finite and nonnegative")
    patience = _validate_tmc_patience(truncation_patience)
    if total_call_budget < 2:
        raise ValueError("TMC-Shapley requires two endpoint calls")
    if num_players > 1 and total_call_budget < num_players + 2:
        raise ValueError(
            "total_call_budget cannot fund one worst-case TMC permutation"
        )

    endpoints = np.zeros((2, num_players), dtype=bool)
    endpoints[1] = True
    endpoint_utilities = np.asarray(
        evaluator.evaluate(endpoints),
        dtype=np.float64,
    )
    if endpoint_utilities.shape != (2,):
        raise RuntimeError("game evaluator returned invalid endpoint utilities")
    if not np.isfinite(endpoint_utilities).all():
        raise RuntimeError("game evaluator returned non-finite endpoint utilities")
    empty_utility, full_utility = endpoint_utilities.tolist()
    tolerance = _resolve_tmc_tolerance(
        tolerance_input,
        relative_tolerance=relative_tolerance,
        full_utility=full_utility,
    )

    if num_players == 1:
        values = np.asarray([full_utility - empty_utility], dtype=np.float64)
        zeros = np.zeros(1, dtype=np.float64)
        diagnostics = TMCShapleyDiagnostics(
            num_players=1,
            num_permutations=0,
            target_call_budget=total_call_budget,
            utility_evaluations=2,
            unused_calls=total_call_budget - 2,
            worst_case_reserved_calls=2,
            budget_remainder_calls=total_call_budget - 2,
            truncation_saved_calls=0,
            boundary_utility_evaluations=2,
            prefix_utility_evaluations=0,
            evaluated_prefixes=0,
            truncated_permutations=0,
            truncation_rate=0.0,
            mean_prefix_length=0.0,
            num_tasks=0,
            requested_num_tasks=num_tasks,
            sampling="exact_endpoints_one_player",
            truncation=True,
            truncation_tolerance_input=tolerance_input,
            truncation_tolerance=tolerance,
            relative_tolerance=bool(relative_tolerance),
            truncation_patience=patience,
            efficiency_projection=False,
            boundary_reuse=False,
            empty_utility=empty_utility,
            full_utility=full_utility,
            sample_variance=zeros,
            variance_of_mean=zeros.copy(),
        )
        return TMCShapleyResult(values, diagnostics)

    num_permutations = (total_call_budget - 2) // num_players
    worst_case_reserved_calls = 2 + num_permutations * num_players
    budget_remainder_calls = total_call_budget - worst_case_reserved_calls
    payloads = _tmc_payloads(
        num_players=num_players,
        num_permutations=num_permutations,
        seed=seed,
        empty_utility=empty_utility,
        full_utility=full_utility,
        truncation_tolerance=tolerance,
        truncation_patience=patience,
        num_tasks=num_tasks,
    )
    chunk_results = evaluator.run_game_tasks(
        _evaluate_tmc_chunk,
        payloads,
        chunksize=task_chunksize,
    )
    if len(chunk_results) != len(payloads):
        raise RuntimeError("game evaluator returned the wrong number of tasks")

    sums = np.zeros(num_players, dtype=np.float64)
    squared_sums = np.zeros(num_players, dtype=np.float64)
    truncated_permutations = 0
    evaluated_prefixes = 0
    observed_permutations = 0
    for (
        chunk_sums,
        chunk_squared_sums,
        chunk_calls,
        chunk_truncated,
        chunk_permutations,
    ) in chunk_results:
        chunk_sum_array = np.asarray(chunk_sums, dtype=np.float64)
        chunk_squared_array = np.asarray(
            chunk_squared_sums, dtype=np.float64
        )
        if (
            chunk_sum_array.shape != sums.shape
            or chunk_squared_array.shape != squared_sums.shape
        ):
            raise RuntimeError("a TMC worker returned an invalid shape")
        if not (
            np.isfinite(chunk_sum_array).all()
            and np.isfinite(chunk_squared_array).all()
        ):
            raise RuntimeError("a TMC worker returned non-finite values")
        calls = int(chunk_calls)
        samples = int(chunk_permutations)
        truncated = int(chunk_truncated)
        if calls < samples or calls > samples * num_players:
            raise RuntimeError("TMC worker call accounting disagrees")
        if truncated < 0 or truncated > samples:
            raise RuntimeError("TMC worker truncation accounting disagrees")
        sums += chunk_sum_array
        squared_sums += chunk_squared_array
        evaluated_prefixes += calls
        truncated_permutations += truncated
        observed_permutations += samples
    if observed_permutations != num_permutations:
        raise RuntimeError("TMC worker sample accounting disagrees")
    values = sums / float(num_permutations)
    if num_permutations > 1:
        sample_variance = (
            squared_sums - float(num_permutations) * values * values
        ) / float(num_permutations - 1)
        sample_variance = np.maximum(sample_variance, 0.0)
    else:
        sample_variance = np.zeros(num_players, dtype=np.float64)
    variance_of_mean = sample_variance / float(num_permutations)
    utility_evaluations = 2 + evaluated_prefixes
    truncation_saved_calls = num_permutations * num_players - evaluated_prefixes
    unused_calls = total_call_budget - utility_evaluations
    if unused_calls != budget_remainder_calls + truncation_saved_calls:
        raise RuntimeError("TMC total call accounting disagrees")

    diagnostics = TMCShapleyDiagnostics(
        num_players=num_players,
        num_permutations=num_permutations,
        target_call_budget=total_call_budget,
        utility_evaluations=utility_evaluations,
        unused_calls=unused_calls,
        worst_case_reserved_calls=worst_case_reserved_calls,
        budget_remainder_calls=budget_remainder_calls,
        truncation_saved_calls=truncation_saved_calls,
        boundary_utility_evaluations=2,
        prefix_utility_evaluations=evaluated_prefixes,
        evaluated_prefixes=evaluated_prefixes,
        truncated_permutations=truncated_permutations,
        truncation_rate=(truncated_permutations / num_permutations),
        mean_prefix_length=(evaluated_prefixes / num_permutations),
        num_tasks=len(payloads),
        requested_num_tasks=num_tasks,
        sampling="uniform_random_permutations_with_replacement",
        truncation=True,
        truncation_tolerance_input=tolerance_input,
        truncation_tolerance=tolerance,
        relative_tolerance=bool(relative_tolerance),
        truncation_patience=patience,
        efficiency_projection=False,
        boundary_reuse=False,
        empty_utility=empty_utility,
        full_utility=full_utility,
        sample_variance=sample_variance,
        variance_of_mean=variance_of_mean,
    )
    return TMCShapleyResult(values, diagnostics)


__all__ = [
    "FullPermutationDiagnostics",
    "FullPermutationResult",
    "TMCShapleyDiagnostics",
    "TMCShapleyResult",
    "estimate_full_permutation_mc",
    "estimate_tmc_shapley",
    "full_permutation_contributions",
    "full_permutations_for_budget",
    "tmc_permutation_contributions",
]

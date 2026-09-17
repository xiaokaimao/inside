"""Orthogonal spherical-code sampling, Mitchell et al., JMLR 23(43), 2022.

Independent implementation of Section 4.2 / Algorithm 3, behaviorally checked
against RAMitchell/shap_sampling. Like the author's implementation, a final
partial orthogonal basis is allowed, always accompanied by its antipodes.
See docs/ORTHOGONAL_BASELINE_AUDIT.md for sources and adaptation details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tmc import PermutationGameEvaluator, _full_permutation_contributions_unchecked


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _frame_from_normals(normals: np.ndarray) -> np.ndarray:
    """Orthonormal rows in the zero-sum hyperplane, using positive-diagonal QR.

    Prepending the constant normal performs projection onto its orthogonal
    complement. Positive R diagonals recover the orientation of Gram-Schmidt
    applied to the same Gaussian vectors. Householder QR is more stable than
    one-pass Gram-Schmidt near the end of a large basis.
    """
    count, n = normals.shape
    if not 0 < count < n or not np.isfinite(normals).all():
        raise ValueError("need between 1 and n-1 finite Gaussian rows")
    matrix = np.column_stack((np.ones(n) / np.sqrt(n), normals.T))
    q, r = np.linalg.qr(matrix, mode="reduced")
    diagonal = np.diag(r)[1:]
    if np.any(np.abs(diagonal) <= np.finfo(float).eps * np.linalg.norm(matrix)):
        raise ValueError("degenerate Gaussian frame")
    return (q[:, 1:] * np.where(diagonal < 0, -1.0, 1.0)).T


def orthogonal_permutations(num_players: int, num_permutations: int, seed: int = 0) -> np.ndarray:
    """Sample an even number of zero-based permutations with reverse pairing.

    Independent Gaussian frames have at most n-1 rows. A budget may select
    fewer rows in the final frame, as in the author's _orthogonal_permutations.
    Output order is all forward permutations, then their reverses. Seeds are
    local to this function and do not alter NumPy's global random state.
    """
    n = _integer(num_players, "num_players")
    count = _integer(num_permutations, "num_permutations", 2)
    seed = _integer(seed, "seed", 0)
    if count % 2:
        raise ValueError("num_permutations must be even to retain reverse pairs")
    output = np.zeros((count, n), dtype=np.int64)
    if n == 1:
        return output
    pairs = count // 2
    for block, first in enumerate(range(0, pairs, n - 1)):
        stop = min(first + n - 1, pairs)
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(block,)))
        directions = _frame_from_normals(rng.standard_normal((stop - first, n)))
        output[first:stop] = np.argsort(directions, axis=1)
    output[pairs:] = output[:pairs, ::-1]
    return output


@dataclass(frozen=True)
class OrthogonalBudget:
    num_players: int
    target_call_budget: int
    minimum_call_budget: int
    permutations_per_full_block: int
    num_full_blocks: int
    tail_pairs: int
    num_permutations: int
    utility_evaluations: int
    unused_calls: int


def orthogonal_budget(num_players: int, total_call_budget: int) -> OrthogonalBudget:
    """Count complete reverse pairs, with the two endpoints evaluated once."""
    n = _integer(num_players, "num_players")
    cap = _integer(total_call_budget, "total_call_budget", 0)
    pairs = max(0, (cap - 2) // (2 * (n - 1))) if n > 1 else 0
    full, tail = divmod(pairs, n - 1) if n > 1 else (0, 0)
    calls = 2 + 2 * pairs * (n - 1) if cap >= 2 and (pairs or n == 1) else 0
    return OrthogonalBudget(n, cap, 2 * n, 2 * (n - 1), full, tail,
                            2 * pairs, calls, cap - calls)


@dataclass(frozen=True)
class OrthogonalDiagnostics:
    budget: OrthogonalBudget
    utility_evaluations: int
    unused_calls: int
    num_permutations: int
    num_tasks: int
    empty_utility: float
    full_utility: float
    sampling: str = "orthogonal_spherical_codes_with_antipodes"
    boundary_reuse: bool = True
    truncation: bool = False
    efficiency_projection: bool = False
    allows_partial_basis: bool = True
    uncertainty_unit: str = "independent_estimator_repeats"


@dataclass(frozen=True)
class OrthogonalResult:
    values: np.ndarray
    diagnostics: OrthogonalDiagnostics


@dataclass(frozen=True)
class _Payload:
    permutations: np.ndarray
    empty: float
    full: float


def _run_chunk(game: Any, payload: _Payload) -> tuple[np.ndarray, int]:
    total = np.zeros(payload.permutations.shape[1])
    calls = 0
    for permutation in payload.permutations:
        values, spent = _full_permutation_contributions_unchecked(
            game, permutation, payload.empty, payload.full)
        total += values
        calls += spent
    return total, calls


def estimate_orthogonal_shapley(
    evaluator: PermutationGameEvaluator, num_players: int,
    total_call_budget: int, seed: int, *, num_tasks: int = 128,
    task_chunksize: int = 1,
) -> OrthogonalResult:
    """Untruncated permutation Shapley estimator under a physical call cap.

    Marginally uniform permutations yield an unbiased estimate; permutations
    within a frame and their reverses are dependent. No IID permutation SE is
    reported. Compare independent seeds for uncertainty. Deterministic game
    endpoints are reused, preserving general nonzero-empty-set utilities.
    """
    plan = orthogonal_budget(num_players, total_call_budget)
    seed = _integer(seed, "seed", 0)
    num_tasks = _integer(num_tasks, "num_tasks")
    task_chunksize = _integer(task_chunksize, "task_chunksize")
    if total_call_budget < plan.minimum_call_budget:
        raise ValueError(f"Orthogonal needs at least {plan.minimum_call_budget} utility calls")
    n = plan.num_players
    endpoints = np.zeros((2, n), dtype=bool)
    endpoints[1] = True
    utilities = np.asarray(evaluator.evaluate(endpoints), dtype=float)
    if utilities.shape != (2,) or not np.isfinite(utilities).all():
        raise ValueError("evaluator must return two finite endpoint utilities")
    empty, full = map(float, utilities)
    payloads = []
    if n == 1:
        values = np.array([full - empty])
    else:
        permutations = orthogonal_permutations(n, plan.num_permutations, seed)
        payloads = [_Payload(chunk, empty, full) for chunk in np.array_split(
            permutations, min(num_tasks, plan.num_permutations))]
        results = evaluator.run_game_tasks(_run_chunk, payloads, chunksize=task_chunksize)
        if len(results) != len(payloads):
            raise RuntimeError("evaluator returned an incorrect number of tasks")
        total, calls = np.zeros(n), 2
        for partial, spent in results:
            total += partial
            calls += spent
        if calls != plan.utility_evaluations or not np.isfinite(total).all():
            raise RuntimeError("invalid Orthogonal utilities or call accounting")
        values = total / plan.num_permutations
    return OrthogonalResult(values, OrthogonalDiagnostics(
        plan, plan.utility_evaluations, plan.unused_calls, plan.num_permutations,
        len(payloads), empty, full))

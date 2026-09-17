"""Yang et al. (JASA 2024) LS/COA permutation-design Shapley baselines.

Port of the author's MIT-licensed ShapDoE 1.0.0 construction and estimator.
See docs/SHAPDOE_BASELINE_AUDIT.md and third_party/shapdoe/ for provenance.
NumPy and R seeds are not bitwise interchangeable. Only complete designs are
averaged; endpoint reuse and deletion of null players preserve their estimates
for deterministic games. The budget counts coalition utility evaluations.
"""

# MIT License — Copyright (c) 2024 ShapDoE authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from .tmc import (
    PermutationGameEvaluator,
    _full_permutation_contributions_unchecked,
)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _prime_power(value: int) -> tuple[int, int] | None:
    if value < 2:
        return None
    divisor = 2
    while divisor * divisor <= value and value % divisor:
        divisor += 1
    if divisor * divisor > value:
        return value, 1
    exponent, remainder = 0, value
    while remainder % divisor == 0:
        exponent += 1
        remainder //= divisor
    return (divisor, exponent) if remainder == 1 else None


def coa_field_order(num_players: int) -> int:
    """Smallest prime power >= n, using the paper's null-player extension."""
    order = max(2, _integer(num_players, "num_players"))
    while _prime_power(order) is None:
        order += 1
    return order


@lru_cache(maxsize=16)
def _field_tables(order: int) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Polynomial-basis GF(q), as in ShapDoE::onecoa (not integer mod q).

    Select a monic primitive polynomial deterministically. Its powers of x
    enumerate every nonzero field element, giving a checked multiplication
    table. Coefficients in the returned polynomial are highest degree first.
    """
    factorization = _prime_power(order)
    if factorization is None:
        raise ValueError("field order must be a prime power")
    prime, degree = factorization
    values = np.arange(order, dtype=np.int64)
    if degree == 1:
        return ((values[:, None] + values) % prime,
                (values[:, None] * values) % prime, ())
    powers = prime ** np.arange(degree, dtype=np.int64)
    digits = (values[:, None] // powers) % prime
    addition = np.zeros((order, order), dtype=np.int64)
    for k in range(degree):
        addition += ((digits[:, k, None] + digits[:, k]) % prime) * powers[k]
    for encoded in range(1, order):
        coefficients = digits[encoded]
        if coefficients[0] == 0:
            continue
        sequence, seen, value = [], set(), 1
        for _ in range(order - 1):
            if value == 0 or value in seen:
                break
            sequence.append(value)
            seen.add(value)
            current = digits[value]
            shifted = np.r_[0, current[:-1]] - current[-1] * coefficients
            value = int((shifted % prime) @ powers)
        if len(sequence) == order - 1 and value == 1:
            break
    else:
        raise RuntimeError("could not construct a primitive field polynomial")
    logarithm = np.zeros(order, dtype=np.int64)
    logarithm[sequence] = np.arange(order - 1)
    multiplication = np.zeros((order, order), dtype=np.int64)
    multiplication[1:, 1:] = np.asarray(sequence)[
        (logarithm[1:, None] + logarithm[None, 1:]) % (order - 1)
    ]
    polynomial = (1, *map(int, coefficients[::-1]))
    addition.flags.writeable = multiplication.flags.writeable = False
    return addition, multiplication, polynomial


def _design_rows(
    num_players: int, method: str, seed: int, block: int, start: int, stop: int,
) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(block,)))
    if method == "ls":
        # ShapDoE::onels: fixed first symbol, random remaining symbols,
        # left cyclic shifts, random columns, and fixed-first random rows.
        base = np.r_[0, rng.permutation(np.arange(1, num_players))]
        columns = rng.permutation(num_players)
        rows = np.r_[0, rng.permutation(np.arange(1, num_players))]
        return base[(rows[start:stop, None] + columns) % num_players]
    order = coa_field_order(num_players)
    addition, multiplication, _ = _field_tables(order)
    columns = np.r_[0, rng.permutation(np.arange(1, order))]
    indices = np.arange(start, stop)
    shifts, scales = indices // (order - 1), indices % (order - 1) + 1
    rows = addition[shifts[:, None], multiplication[scales[:, None], columns]]
    # The dummy players have zero marginal contribution. Removing them keeps
    # exactly the same real-player paths without evaluating unchanged sets.
    return rows[rows < num_players].reshape(stop - start, num_players)


def latin_square_design(num_players: int, seed: int = 0) -> np.ndarray:
    """One randomized LS; each row/column contains every zero-based player."""
    n = _integer(num_players, "num_players")
    return _design_rows(n, "ls", _integer(seed, "seed", 0), 0, 0, n)


def component_orthogonal_array_design(num_players: int, seed: int = 0) -> np.ndarray:
    """One COA, with null players removed when n is not a prime power.

    The resulting array has q(q-1) rows and n columns. For n < q it is the
    paper's projected design, not itself a strength-two COA on n symbols.
    Estimation generates this design in chunks instead of allocating it whole.
    """
    n = _integer(num_players, "num_players")
    order = coa_field_order(n)
    return _design_rows(n, "coa", _integer(seed, "seed", 0), 0, 0, order * (order - 1))


@dataclass(frozen=True)
class ShapDoEBudget:
    num_players: int
    method: str
    field_order: int
    permutations_per_design: int
    calls_per_design: int
    minimum_call_budget: int
    num_designs: int
    num_permutations: int
    target_call_budget: int
    utility_evaluations: int
    unused_calls: int


def shapdoe_budget(num_players: int, total_call_budget: int, method: str = "ls") -> ShapDoEBudget:
    """Plan complete blocks without making calls; zero blocks means infeasible."""
    n = _integer(num_players, "num_players")
    budget = _integer(total_call_budget, "total_call_budget", 0)
    if method not in {"ls", "coa"}:
        raise ValueError("method must be 'ls' or 'coa'")
    order = coa_field_order(n) if method == "coa" else n
    rows = n if method == "ls" else order * (order - 1)
    cost = rows * (n - 1)
    blocks = max(0, (budget - 2) // cost) if cost else 0
    actual = 2 + blocks * cost if budget >= 2 and (blocks or n == 1) else 0
    return ShapDoEBudget(n, method, order, rows, cost, 2 + cost, blocks,
                        blocks * rows, budget, actual, budget - actual)


@dataclass(frozen=True)
class ShapDoEDiagnostics:
    budget: ShapDoEBudget
    utility_evaluations: int
    unused_calls: int
    num_designs: int
    num_permutations: int
    num_tasks: int
    empty_utility: float
    full_utility: float
    null_players: int
    field_polynomial: tuple[int, ...]
    variance_of_mean: np.ndarray | None
    sampling: str
    boundary_reuse: bool = True
    truncation: bool = False
    efficiency_projection: bool = False
    uncertainty_unit: str = "independent_complete_design"
    implementation: str = "Python port of ShapDoE 1.0.0"


@dataclass(frozen=True)
class ShapDoEResult:
    values: np.ndarray
    diagnostics: ShapDoEDiagnostics


@dataclass(frozen=True)
class _Payload:
    plan: ShapDoEBudget
    seed: int
    first: int
    stop: int
    empty: float
    full: float


def _run_chunk(game: Any, payload: _Payload) -> tuple[dict[int, np.ndarray], int]:
    plan = payload.plan
    block_sums: dict[int, np.ndarray] = {}
    cursor, calls = payload.first, 0
    while cursor < payload.stop:
        block, start = divmod(cursor, plan.permutations_per_design)
        # Keep the temporary permutation matrix below roughly 1 MiB.
        count = min(payload.stop - cursor, plan.permutations_per_design - start,
                    max(1, 2**20 // (8 * plan.num_players)))
        permutations = _design_rows(plan.num_players, plan.method, payload.seed,
                                    block, start, start + count)
        sums = block_sums.setdefault(block, np.zeros(plan.num_players))
        for permutation in permutations:
            values, spent = _full_permutation_contributions_unchecked(
                game, permutation, payload.empty, payload.full)
            sums += values
            calls += spent
        cursor += count
    return block_sums, calls


def estimate_shapdoe(
    evaluator: PermutationGameEvaluator,
    num_players: int,
    total_call_budget: int,
    seed: int,
    *,
    method: str = "ls",
    num_tasks: int = 128,
    task_chunksize: int = 1,
) -> ShapDoEResult:
    """Estimate with complete LS/COA designs under a physical call cap.

    Infeasible budgets fail before any utility call. For general games use
    v(S)-v(empty), rather than ShapDoE's implicit v(empty)=0 convention.
    Only the two deterministic endpoints are cached. Within-design dependent
    rows are never treated as IID replicates for standard errors.
    """
    seed = _integer(seed, "seed", 0)
    num_tasks = _integer(num_tasks, "num_tasks")
    task_chunksize = _integer(task_chunksize, "task_chunksize")
    plan = shapdoe_budget(num_players, total_call_budget, method)
    n = plan.num_players
    if total_call_budget < plan.minimum_call_budget:
        raise ValueError(f"{method.upper()} needs at least {plan.minimum_call_budget} "
                         "utility calls for one complete design")
    endpoints = np.zeros((2, n), dtype=bool)
    endpoints[1] = True
    utilities = np.asarray(evaluator.evaluate(endpoints), dtype=np.float64)
    if utilities.shape != (2,) or not np.isfinite(utilities).all():
        raise ValueError("evaluator must return two finite endpoint utilities")
    empty, full = map(float, utilities)
    if n == 1:
        values, variance, payloads = np.array([full - empty]), np.zeros(1), []
    else:
        task_count = min(num_tasks, plan.num_permutations)
        edges = [i * plan.num_permutations // task_count for i in range(task_count + 1)]
        payloads = [_Payload(plan, seed, first, stop, empty, full)
                    for first, stop in zip(edges[:-1], edges[1:])]
        results = evaluator.run_game_tasks(_run_chunk, payloads, chunksize=task_chunksize)
        if len(results) != len(payloads):
            raise RuntimeError("evaluator returned an incorrect number of tasks")
        sums = np.zeros((plan.num_designs, n))
        calls = 2
        for partial, spent in results:
            calls += spent
            for block, subtotal in partial.items():
                sums[block] += subtotal
        if calls != plan.utility_evaluations or not np.isfinite(sums).all():
            raise RuntimeError("invalid ShapDoE utility values or call accounting")
        block_estimates = sums / plan.permutations_per_design
        values = block_estimates.mean(axis=0)
        variance = (block_estimates.var(axis=0, ddof=1) / plan.num_designs
                    if plan.num_designs > 1 else None)
    polynomial = _field_tables(plan.field_order)[2] if method == "coa" and n > 1 else ()
    diagnostics = ShapDoEDiagnostics(
        plan, plan.utility_evaluations, plan.unused_calls, plan.num_designs,
        plan.num_permutations, len(payloads), empty, full,
        plan.field_order - n if method == "coa" else 0, polynomial, variance,
        "randomized_latin_square" if method == "ls" else "randomized_component_orthogonal_array",
    )
    return ShapDoEResult(values, diagnostics)

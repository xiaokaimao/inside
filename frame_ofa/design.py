"""Coalition designs for Shapley-specific OFA."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb
from typing import Any

import numpy as np

from .geometry import (
    _inner_frame_statistics,
    centered_directions,
    efficiency_projector,
    fixed_slice_frame_operator,
    inner_frame_target,
)


@dataclass
class CoalitionDesign:
    """A realized batch of coalitions and its sampling metadata."""

    coalitions: np.ndarray
    sizes: np.ndarray
    method: str
    seed: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coalitions = np.asarray(self.coalitions, dtype=bool)
        self.sizes = np.asarray(self.sizes, dtype=np.int64)
        if self.coalitions.ndim != 2:
            raise ValueError("coalitions must be a two-dimensional Boolean array")
        if len(self.coalitions) != len(self.sizes):
            raise ValueError("coalitions and sizes must have the same length")
        if not np.array_equal(self.coalitions.sum(axis=1), self.sizes):
            raise ValueError("the recorded sizes do not match the coalitions")


def inner_size_distribution(
    num_players: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return Shapley-specific OFA probabilities on sizes 2, ..., n-2."""
    if num_players < 4:
        raise ValueError("at least 4 players are required")
    sizes = np.arange(2, num_players - 1, dtype=np.int64)
    unnormalized = 1.0 / np.sqrt(sizes * (num_players - sizes))
    normalizer = float(unnormalized.sum())
    return sizes, unnormalized / normalizer, normalizer


def _systematic_sizes(
    sizes: np.ndarray,
    probabilities: np.ndarray,
    num_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Couple size draws while preserving every draw's OFA marginal.

    If U is uniform on [0,1), then each (U+t/T) mod 1 is itself uniform.
    Applying the inverse CDF therefore gives every row the desired marginal,
    while the realized size counts are much more stable than IID counts.
    """
    offset = rng.random()
    points = (offset + np.arange(num_samples, dtype=np.float64) / num_samples) % 1.0
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.0
    sampled = sizes[np.searchsorted(cdf, points, side="right")]
    return sampled[rng.permutation(num_samples)]


def _sample_uniform_coalition(
    num_players: int, size: int, rng: np.random.Generator
) -> np.ndarray:
    coalition = np.zeros(num_players, dtype=bool)
    coalition[rng.choice(num_players, size=size, replace=False)] = True
    return coalition


def _sample_iid_coalitions(
    num_players: int,
    sampled_sizes: np.ndarray,
    rng: np.random.Generator,
    *,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Sample independent uniform coalitions conditional on their row sizes.

    For a row with ``r`` members still required and ``m`` player positions
    left, the next player is included with probability ``r / m``.  This
    sequential construction gives every fixed-size subset probability
    ``1 / binom(n, s)`` while replacing one Python ``rng.choice`` call per row
    by ``n`` vectorized random draws per chunk.
    """
    sampled_sizes = np.asarray(sampled_sizes, dtype=np.int64)
    if sampled_sizes.ndim != 1:
        raise ValueError("sampled_sizes must be one-dimensional")
    if np.any(sampled_sizes < 0) or np.any(sampled_sizes > num_players):
        raise ValueError("sampled sizes must lie between zero and n")
    if chunk_rows is None:
        # Temporary arrays are one float, one integer, and one Boolean per
        # row; this cap keeps them comfortably below the diagnostic buffers.
        chunk_rows = max(1, (8 * 2**20) // 17)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    coalitions = np.empty(
        (len(sampled_sizes), num_players), dtype=bool
    )
    for start in range(0, len(sampled_sizes), chunk_rows):
        stop = min(start + chunk_rows, len(sampled_sizes))
        remaining = sampled_sizes[start:stop].copy()
        for player in range(num_players):
            positions_left = num_players - player
            include = (
                rng.random(stop - start) * positions_left < remaining
            )
            coalitions[start:stop, player] = include
            remaining -= include
        if np.any(remaining):
            raise RuntimeError("failed to realize the requested coalition sizes")
    return coalitions


def _candidate_coalitions(
    num_players: int,
    size: int,
    candidate_pool: int,
    rng: np.random.Generator,
    excluded: set[tuple[int, ...]] | None = None,
) -> np.ndarray:
    total = comb(num_players, size)
    blocked = excluded if excluded is not None else set()
    if len(blocked) >= total:
        blocked = set()
    available = total - len(blocked)
    take = min(available, candidate_pool)
    if available <= candidate_pool:
        choices = [
            indices
            for indices in combinations(range(num_players), size)
            if indices not in blocked
        ]
        rows = np.zeros((len(choices), num_players), dtype=bool)
        for row, indices in zip(rows, choices):
            row[list(indices)] = True
        return rows

    seen: set[tuple[int, ...]] = set()
    while len(seen) < take:
        indices = tuple(sorted(rng.choice(num_players, size=size, replace=False).tolist()))
        if indices not in blocked:
            seen.add(indices)
    rows = np.zeros((take, num_players), dtype=bool)
    for row, indices in zip(rows, sorted(seen)):
        row[list(indices)] = True
    return rows


def _random_relabel(
    coalitions: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Apply one independent uniform player permutation to a whole design."""
    permutation = rng.permutation(coalitions.shape[1])
    relabeled = np.empty_like(coalitions)
    relabeled[:, permutation] = coalitions
    return relabeled


def _greedy_frame_rows(
    num_players: int,
    sampled_sizes: np.ndarray,
    rng: np.random.Generator,
    candidate_pool: int,
    *,
    radial_weights: bool,
    mean_balance: float,
) -> np.ndarray:
    """Greedily minimize the weighted frame potential.

    Conditional on the size sequence, the target cross-term with the
    efficiency projector and the self-outer-product terms are constants.
    Selecting the candidate minimizing u^T A u therefore greedily minimizes
    the exact Frobenius frame-discrepancy increment.
    """
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")

    if mean_balance < 0:
        raise ValueError("mean_balance must be nonnegative")
    operator_sum = np.zeros((num_players, num_players), dtype=np.float64)
    direction_sums: dict[int, np.ndarray] = {}
    used_by_size: dict[int, set[tuple[int, ...]]] = {}
    coalitions = np.zeros((len(sampled_sizes), num_players), dtype=bool)
    for row_index, size_value in enumerate(sampled_sizes):
        size = int(size_value)
        used = used_by_size.setdefault(size, set())
        if len(used) == comb(num_players, size):
            # Only repeat a coalition after its complete Boolean slice has
            # been exhausted.
            used.clear()
        candidates = _candidate_coalitions(
            num_players, size, candidate_pool, rng, excluded=used
        )
        directions = centered_directions(candidates)
        weight = (
            np.sqrt(size * (num_players - size)) if radial_weights else 1.0
        )
        scores = weight * np.einsum(
            "bi,ij,bj->b", directions, operator_sum, directions, optimize=True
        )
        direction_sum = direction_sums.setdefault(
            size, np.zeros(num_players, dtype=np.float64)
        )
        if mean_balance:
            scores += mean_balance * (directions @ direction_sum)
        best = np.flatnonzero(
            np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-14)
        )
        chosen = int(rng.choice(best))
        coalition = candidates[chosen]
        direction = directions[chosen]
        coalitions[row_index] = coalition
        used.add(tuple(np.flatnonzero(coalition).tolist()))

        operator_sum += weight * np.outer(direction, direction)
        direction_sum += direction
    return coalitions


def frame_diagnostics(
    coalitions: np.ndarray, *, chunk_rows: int | None = None
) -> dict[str, float]:
    """Compute coupled-OFA diagnostics with bounded float memory."""
    coalitions = np.asarray(coalitions, dtype=bool)
    if coalitions.ndim != 2:
        raise ValueError("coalitions must be a two-dimensional array")
    num_samples, num_players = coalitions.shape
    operator, slice_counts, slice_member_counts = _inner_frame_statistics(
        coalitions,
        chunk_rows=chunk_rows,
        collect_slice_means=True,
    )
    assert slice_counts is not None
    assert slice_member_counts is not None
    target = inner_frame_target(num_players)
    residual = operator - target
    eigenvalues = np.linalg.eigvalsh(residual)
    present = np.flatnonzero(slice_counts)
    sizes = present.astype(np.float64) + 2.0
    scales = np.sqrt(
        num_players / (sizes * (num_players - sizes))
    )
    slice_means = scales[:, None] * (
        slice_member_counts[present] / slice_counts[present, None]
        - sizes[:, None] / num_players
    )
    slice_mean_norms = np.linalg.norm(slice_means, axis=1)
    mean_direction = (
        slice_means * slice_counts[present, None]
    ).sum(axis=0) / num_samples
    return {
        "frobenius_discrepancy": float(np.linalg.norm(residual, ord="fro")),
        "spectral_discrepancy": float(np.max(np.abs(eigenvalues))),
        "trace_discrepancy": float(np.trace(residual)),
        "mean_direction_norm": float(np.linalg.norm(mean_direction)),
        "slice_mean_direction_rms": float(
            np.sqrt(np.mean(np.square(slice_mean_norms)))
        ),
        "slice_mean_direction_worst": float(max(slice_mean_norms)),
    }


def iid_ofa_design(
    num_players: int, num_samples: int, seed: int = 0
) -> CoalitionDesign:
    """Generate the independent Shapley-specific OFA design."""
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    sizes, probabilities, _ = inner_size_distribution(num_players)
    sampled_sizes = rng.choice(sizes, size=num_samples, p=probabilities)
    coalitions = _sample_iid_coalitions(
        num_players, sampled_sizes, rng
    )
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="iid",
        seed=seed,
        diagnostics=frame_diagnostics(coalitions),
    )


def frame_coupled_design(
    num_players: int,
    num_samples: int,
    seed: int = 0,
    candidate_pool: int = 32,
    mean_balance: float = 0.1,
) -> CoalitionDesign:
    """Generate a coupled batch with the exact OFA marginal for every row.

    Size draws use randomized systematic sampling.  Base coalitions then
    greedily reduce the weighted frame potential.  A final independent,
    uniform relabeling makes each row uniform on its fixed-size Boolean slice
    without changing any geometric relationship inside the batch.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    sizes, probabilities, _ = inner_size_distribution(num_players)
    sampled_sizes = _systematic_sizes(
        sizes, probabilities, num_samples, rng
    )
    base = _greedy_frame_rows(
        num_players,
        sampled_sizes,
        rng,
        candidate_pool,
        radial_weights=True,
        mean_balance=mean_balance,
    )
    coalitions = _random_relabel(base, rng)
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="frame_coupled",
        seed=seed,
        diagnostics=frame_diagnostics(coalitions),
    )


def _minimum_one_counts(
    num_samples: int, probabilities: np.ndarray
) -> np.ndarray:
    """Find a near-target integer allocation with one sample per stratum.

    The returned vector minimizes squared distance to ``num_samples * q``
    under the integer constraints ``counts >= 1`` and ``sum(counts) = T``.
    """
    num_strata = len(probabilities)
    if num_samples < num_strata:
        raise ValueError(
            "stratified estimation needs at least one sample for every inner size"
        )
    targets = num_samples * probabilities
    counts = np.maximum(1, np.floor(targets).astype(np.int64))
    while counts.sum() < num_samples:
        # Adding to the largest target deficit gives the largest reduction
        # in the separable squared-error objective.
        index = int(np.argmax(targets - counts))
        counts[index] += 1
    while counts.sum() > num_samples:
        removable = counts > 1
        if not np.any(removable):
            raise RuntimeError("failed to construct a minimum-one allocation")
        excess = np.where(removable, counts - targets, -np.inf)
        index = int(np.argmax(excess))
        counts[index] -= 1
    return counts


def _stratified_diagnostics(
    coalitions: np.ndarray, sizes: np.ndarray
) -> dict[str, float]:
    num_players = coalitions.shape[1]
    projector = efficiency_projector(num_players)
    target = projector / (num_players - 1)
    operators: list[np.ndarray] = []
    mean_norms: list[float] = []
    for size in np.unique(sizes):
        take = coalitions[sizes == size]
        operators.append(fixed_slice_frame_operator(take))
        mean_norms.append(
            float(np.linalg.norm(centered_directions(take).mean(axis=0)))
        )
    return _summarize_stratified_operators(
        operators, target, mean_norms
    )


def _summarize_stratified_operators(
    operators: list[np.ndarray],
    target: np.ndarray,
    mean_norms: list[float],
) -> dict[str, float]:
    """Summarize fixed-slice operators without materializing their rows."""
    squared = 0.0
    worst = 0.0
    aggregate = np.zeros_like(target)
    for operator in operators:
        discrepancy = float(np.linalg.norm(operator - target, ord="fro"))
        squared += discrepancy**2
        worst = max(worst, discrepancy)
        aggregate += operator
    mean_squared = float(np.sum(np.square(mean_norms)))
    mean_worst = float(max(mean_norms))
    aggregate_target = len(operators) * target
    return {
        "slice_frobenius_rms": float(
            np.sqrt(squared / len(operators))
        ),
        "slice_frobenius_root_sum_squares": float(np.sqrt(squared)),
        "slice_frobenius_worst": worst,
        "slice_mean_direction_rms": float(
            np.sqrt(mean_squared / len(operators))
        ),
        "slice_mean_direction_worst": mean_worst,
        "aggregate_frobenius_discrepancy": float(
            np.linalg.norm(aggregate - aggregate_target, ord="fro")
        ),
    }


def stratified_frame_design(
    num_players: int,
    num_samples: int,
    seed: int = 0,
    candidate_pool: int = 32,
    mean_balance: float = 0.1,
) -> CoalitionDesign:
    """Preallocate OFA-optimal counts and frame-balance every fixed-size slice."""
    rng = np.random.default_rng(seed)
    sizes, probabilities, _ = inner_size_distribution(num_players)
    counts = _minimum_one_counts(num_samples, probabilities)

    blocks: list[np.ndarray] = []
    recorded_sizes: list[np.ndarray] = []
    for size, count in zip(sizes, counts):
        size_sequence = np.full(count, size, dtype=np.int64)
        base = _greedy_frame_rows(
            num_players,
            size_sequence,
            rng,
            candidate_pool,
            radial_weights=False,
            mean_balance=mean_balance,
        )
        blocks.append(_random_relabel(base, rng))
        recorded_sizes.append(size_sequence)

    coalitions = np.concatenate(blocks, axis=0)
    sampled_sizes = np.concatenate(recorded_sizes)
    order = rng.permutation(num_samples)
    coalitions = coalitions[order]
    sampled_sizes = sampled_sizes[order]
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="frame_stratified",
        seed=seed,
        diagnostics=_stratified_diagnostics(coalitions, sampled_sizes),
    )


def _cyclic_orbit(base: np.ndarray) -> np.ndarray:
    """Return all n cyclic shifts of a base block."""
    num_players = len(base)
    return np.stack([np.roll(base, shift) for shift in range(num_players)])


def _cyclic_orbit_signature(base: np.ndarray) -> np.ndarray:
    """Return the first row of a cyclic orbit operator via FFT."""
    direction = centered_directions(base)[0]
    spectrum = np.fft.fft(direction)
    autocorrelation = np.fft.ifft(
        spectrum * np.conjugate(spectrum)
    ).real
    return autocorrelation / len(base)


def _cyclic_orbit_signatures(candidates: np.ndarray) -> np.ndarray:
    """Return cyclic-orbit signatures for a batch of equal-size candidates.

    This is the batched counterpart of :func:`_cyclic_orbit_signature`.  The
    FFT is taken along the player axis, so constructing a candidate pool costs
    ``O(K n log n)`` without materializing any of its ``K n`` shifted rows.
    """
    directions = centered_directions(candidates)
    spectra = np.fft.fft(directions, axis=1)
    autocorrelations = np.fft.ifft(
        spectra * np.conjugate(spectra), axis=1
    ).real
    return autocorrelations / candidates.shape[1]


def _cyclic_orbit_operator(base: np.ndarray) -> np.ndarray:
    """Compute a cyclic orbit's frame operator using FFT autocorrelation."""
    first_row = _cyclic_orbit_signature(base)
    return _circulant_from_first_row(first_row)


def _circulant_from_first_row(first_row: np.ndarray) -> np.ndarray:
    """Expand a circulant first row; used only for final diagnostics/tests."""
    indices = (
        np.arange(len(first_row))[None, :]
        - np.arange(len(first_row))[:, None]
    ) % len(first_row)
    operator = first_row[indices]
    return (operator + operator.T) / 2


def _orbit_coupled_diagnostics(
    weighted_signature_sum: np.ndarray,
    num_orbits: int,
    normalizer: float,
) -> dict[str, float]:
    """Summarize a coupled-orbit frame from its length-n sufficient statistic.

    A complete cyclic orbit has zero mean direction.  Its second-moment
    operator is circulant and is therefore fully represented by one row.
    Consequently this routine never constructs the ``T x n`` floating-point
    direction matrix used by :func:`frame_diagnostics`.
    """
    num_players = len(weighted_signature_sum)
    first_row = normalizer * weighted_signature_sum / num_orbits
    operator = _circulant_from_first_row(first_row)
    target = inner_frame_target(num_players)
    residual = operator - target
    eigenvalues = np.linalg.eigvalsh(residual)
    return {
        "frobenius_discrepancy": float(
            np.linalg.norm(residual, ord="fro")
        ),
        "spectral_discrepancy": float(np.max(np.abs(eigenvalues))),
        "trace_discrepancy": float(np.trace(residual)),
        "mean_direction_norm": 0.0,
        "slice_mean_direction_rms": 0.0,
        "slice_mean_direction_worst": 0.0,
    }


def orbit_coupled_frame_design(
    num_players: int,
    num_samples: int,
    seed: int = 0,
    candidate_pool: int = 32,
) -> CoalitionDesign:
    """Generate a scalable orbit-coupled linear OFA design.

    ``num_samples`` is the number of inner coalition evaluations and must be
    divisible by ``num_players``.  The design consists of complete cyclic
    orbits of length ``num_players``:

    * orbit sizes use randomized systematic sampling from the Shapley-specific
      OFA distribution ``q*``;
    * candidate base blocks are selected by an exact greedy reduction of the
      weighted frame potential, using only FFT autocorrelation signatures;
    * one independent uniform player relabeling makes every realized row
      uniform on its fixed-size Boolean slice.

    Thus every row has the same ``Q*`` marginal required by the coupled linear
    estimator, while rows within and across orbits are deliberately dependent.
    The greedy state and diagnostics use ``O(n)`` and ``O(n^2)`` floating-point
    memory respectively; no ``num_samples x n`` float array is formed.
    """
    if num_players < 4:
        raise ValueError("at least 4 players are required")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if num_samples % num_players:
        raise ValueError("num_samples must be divisible by num_players")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")

    num_orbits = num_samples // num_players
    seed_sequence = np.random.SeedSequence(seed)
    size_seed, candidate_seed, relabel_seed = seed_sequence.spawn(3)
    size_rng = np.random.default_rng(size_seed)
    candidate_rng = np.random.default_rng(candidate_seed)
    relabel_rng = np.random.default_rng(relabel_seed)

    sizes, probabilities, normalizer = inner_size_distribution(num_players)
    orbit_sizes = _systematic_sizes(
        sizes, probabilities, num_orbits, size_rng
    )

    # For an orbit at size s, its contribution to the coupled-HT operator is
    # w_s O(base), where w_s = sqrt(s(n-s)).  Every O(base) is circulant, so
    # minimizing the Frobenius increment is equivalent (up to the common
    # factor n) to minimizing the norm of the accumulated first row.
    weighted_signature_sum = np.zeros(num_players, dtype=np.float64)
    bases = np.zeros((num_orbits, num_players), dtype=bool)
    for orbit_index, size_value in enumerate(orbit_sizes):
        size = int(size_value)
        candidates = _candidate_coalitions(
            num_players, size, candidate_pool, candidate_rng
        )
        signatures = _cyclic_orbit_signatures(candidates)
        weight = np.sqrt(size * (num_players - size))
        scores = (
            2.0 * weight * (signatures @ weighted_signature_sum)
            + weight**2 * np.einsum(
                "bi,bi->b", signatures, signatures
            )
        )
        best = np.flatnonzero(
            np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-14)
        )
        chosen = int(candidate_rng.choice(best))
        bases[orbit_index] = candidates[chosen]
        weighted_signature_sum += weight * signatures[chosen]

    # Materialize Boolean rows only after the FFT-greedy pass.  Applying the
    # global relabeling through a precomputed source-index matrix avoids a
    # second num_samples x n Boolean allocation.  Chunking bounds temporary
    # memory independently of num_samples.
    permutation = relabel_rng.permutation(num_players)
    inverse_permutation = np.argsort(permutation)
    cyclic_indices = (
        np.arange(num_players)[None, :]
        - np.arange(num_players)[:, None]
    ) % num_players
    materialization_indices = cyclic_indices[:, inverse_permutation]
    coalitions = np.empty((num_samples, num_players), dtype=bool)
    coalition_orbits = coalitions.reshape(
        num_orbits, num_players, num_players
    )
    chunk_size = max(1, 2**20 // (num_players * num_players))
    for start in range(0, num_orbits, chunk_size):
        stop = min(start + chunk_size, num_orbits)
        coalition_orbits[start:stop] = bases[start:stop][
            :, materialization_indices
        ]

    sampled_sizes = np.repeat(orbit_sizes, num_players)
    diagnostics = _orbit_coupled_diagnostics(
        weighted_signature_sum, num_orbits, normalizer
    )
    diagnostics.update(
        {
            "num_orbits": int(num_orbits),
            "candidate_pool": int(candidate_pool),
        }
    )
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="orbit_coupled",
        seed=seed,
        diagnostics=diagnostics,
    )


def cyclic_orbit_frame_design(
    num_players: int,
    num_orbits: int,
    seed: int = 0,
    candidate_pool: int = 32,
) -> CoalitionDesign:
    """Build a ratio-OFA-compatible design from complete cyclic orbits.

    Every orbit contains n coalitions.  At size s, every player occurs exactly
    s times and is absent exactly n-s times per orbit, making all ratio
    denominators deterministic.  ``num_orbits`` therefore corresponds to the
    upstream code's average sampled-utility budget per player.
    """
    rng = np.random.default_rng(seed)
    sizes, probabilities, _ = inner_size_distribution(num_players)
    orbit_counts = _minimum_one_counts(num_orbits, probabilities)

    blocks: list[np.ndarray] = []
    recorded_sizes: list[np.ndarray] = []
    realized_operators: list[np.ndarray] = []
    target = efficiency_projector(num_players) / (num_players - 1)
    for size, count in zip(sizes, orbit_counts):
        # Every orbit operator is circulant, so its first row is a complete
        # representation.  Frobenius products of circulant matrices are n
        # times the inner products of their first rows.
        signature_sum = np.zeros(num_players, dtype=np.float64)
        slice_orbits: list[np.ndarray] = []
        for _ in range(int(count)):
            candidates = _candidate_coalitions(
                num_players, int(size), candidate_pool, rng
            )
            candidate_signatures = [
                _cyclic_orbit_signature(candidate)
                for candidate in candidates
            ]
            scores = np.asarray(
                [
                    num_players
                    * (
                        2 * np.dot(signature_sum, signature)
                        + np.dot(signature, signature)
                    )
                    for signature in candidate_signatures
                ]
            )
            best = np.flatnonzero(
                np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-14)
            )
            chosen = int(rng.choice(best))
            signature_sum += candidate_signatures[chosen]
            slice_orbits.append(_cyclic_orbit(candidates[chosen]))

        slice_rows = np.concatenate(slice_orbits, axis=0)
        base_operator = _circulant_from_first_row(
            signature_sum / int(count)
        )
        # A common relabeling preserves the optimized slice frame.  Track the
        # conjugated small operator directly so diagnostics do not multiply
        # all O(n * num_orbits) materialized directions.
        permutation = rng.permutation(num_players)
        relabeled_rows = np.empty_like(slice_rows)
        relabeled_rows[:, permutation] = slice_rows
        slice_rows = relabeled_rows
        relabeled_operator = np.empty_like(base_operator)
        relabeled_operator[np.ix_(permutation, permutation)] = base_operator
        realized_operators.append(relabeled_operator)
        blocks.append(slice_rows)
        recorded_sizes.append(
            np.full(len(slice_rows), size, dtype=np.int64)
        )

    coalitions = np.concatenate(blocks, axis=0)
    sampled_sizes = np.concatenate(recorded_sizes)
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="cyclic_orbit_frame",
        seed=seed,
        diagnostics=_summarize_stratified_operators(
            realized_operators,
            target,
            [0.0] * len(realized_operators),
        ),
    )

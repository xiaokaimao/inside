"""Coalition designs for Shapley-specific OFA."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
from itertools import combinations
from math import comb
import multiprocessing as mp
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
    return _relabel_with_permutation(coalitions, permutation)


def _relabel_with_permutation(
    coalitions: np.ndarray, permutation: np.ndarray
) -> np.ndarray:
    """Apply a supplied old-to-new player-label permutation."""
    permutation = np.asarray(permutation, dtype=np.int64)
    if permutation.shape != (coalitions.shape[1],) or not np.array_equal(
        np.sort(permutation), np.arange(coalitions.shape[1])
    ):
        raise ValueError("permutation must contain each player label once")
    relabeled = np.empty_like(coalitions)
    relabeled[:, permutation] = coalitions
    return relabeled


def _permutation_sha256(permutation: np.ndarray) -> str:
    encoded = np.ascontiguousarray(
        np.asarray(permutation, dtype="<i8")
    )
    return hashlib.sha256(encoded.tobytes(order="C")).hexdigest()


def _greedy_frame_rows(
    num_players: int,
    sampled_sizes: np.ndarray,
    rng: np.random.Generator,
    candidate_pool: int,
    *,
    radial_weights: bool,
    mean_balance: float,
    second_moment_weight: float = 1.0,
    second_moment_scope: str = "global",
) -> np.ndarray:
    """Greedily minimize a global or fixed-slice frame potential.

    For ``second_moment_scope="global"``, one operator accumulates all rows
    and the optional radial weight gives the coupled-linear OFA objective.
    For ``"per_size"``, each fixed-size slice has its own unweighted
    operator.  Conditional on the size sequence, target cross-terms with the
    efficiency projector and self-outer-product terms are constants, so the
    corresponding ``u^T A u`` score is the exact candidate-dependent part of
    the Frobenius-discrepancy increment.
    """
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")

    if mean_balance < 0:
        raise ValueError("mean_balance must be nonnegative")
    if (
        not np.isfinite(second_moment_weight)
        or second_moment_weight < 0
    ):
        raise ValueError(
            "second_moment_weight must be finite and nonnegative"
        )
    if second_moment_weight == 0 and mean_balance == 0:
        raise ValueError("at least one moment objective must be active")
    if second_moment_scope not in {"global", "per_size"}:
        raise ValueError(
            "second_moment_scope must be 'global' or 'per_size'"
        )
    if second_moment_scope == "per_size" and radial_weights:
        raise ValueError(
            "per-size second moments must use unweighted slice operators"
        )
    operator_sum = np.zeros((num_players, num_players), dtype=np.float64)
    operator_sums: dict[int, np.ndarray] = {}
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
            np.sqrt(size * (num_players - size))
            if radial_weights
            else 1.0
        )
        active_operator = (
            operator_sum
            if second_moment_scope == "global"
            else operator_sums.setdefault(
                size, np.zeros_like(operator_sum)
            )
        )
        if second_moment_weight:
            scores = second_moment_weight * weight * np.einsum(
                "bi,ij,bj->b",
                directions,
                active_operator,
                directions,
                optimize=True,
            )
        else:
            scores = np.zeros(len(candidates), dtype=np.float64)
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

        if second_moment_weight:
            active_operator += weight * np.outer(direction, direction)
        direction_sum += direction
    return coalitions


@dataclass(frozen=True)
class _FixedSliceGreedyTask:
    """One independent fixed-size component of a per-size design."""

    num_players: int
    size: int
    num_rows: int
    candidate_pool: int
    mean_balance: float
    second_moment_weight: float
    seed_words: tuple[int, ...]


def _run_fixed_slice_greedy_task(
    task: _FixedSliceGreedyTask,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Build one slice and return its exact first-moment counts.

    Per-size INSIDE has no state shared across coalition sizes.  Keeping this
    helper at module scope makes that mathematical decomposition directly
    usable by a spawn-based process pool without changing the public default
    (the historical single-stream implementation remains the default).
    """
    rng = np.random.default_rng(np.random.SeedSequence(task.seed_words))
    sampled_sizes = np.full(
        task.num_rows, task.size, dtype=np.int64
    )
    rows = _greedy_frame_rows(
        task.num_players,
        sampled_sizes,
        rng,
        task.candidate_pool,
        radial_weights=False,
        mean_balance=task.mean_balance,
        second_moment_weight=task.second_moment_weight,
        second_moment_scope="per_size",
    )
    member_counts = rows.sum(axis=0, dtype=np.int64)
    return task.size, rows, member_counts


def _parallel_fixed_slice_rows(
    num_players: int,
    sampled_sizes: np.ndarray,
    *,
    candidate_pool: int,
    mean_balance: float,
    second_moment_weight: float,
    seed_sequence: np.random.SeedSequence,
    design_jobs: int,
    start_method: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build independent fixed-size states with deterministic substreams."""
    if design_jobs < 1:
        raise ValueError("design_jobs must be positive")
    if start_method not in mp.get_all_start_methods():
        raise ValueError(f"unsupported design start method: {start_method}")

    sizes, _, _ = inner_size_distribution(num_players)
    counts = np.bincount(sampled_sizes, minlength=num_players)
    active_sizes = [int(size) for size in sizes if counts[int(size)] > 0]
    child_sequences = seed_sequence.spawn(len(active_sizes))
    tasks = [
        _FixedSliceGreedyTask(
            num_players=num_players,
            size=size,
            num_rows=int(counts[size]),
            candidate_pool=candidate_pool,
            mean_balance=mean_balance,
            second_moment_weight=second_moment_weight,
            seed_words=tuple(
                int(word)
                for word in child.generate_state(4, dtype=np.uint32)
            ),
        )
        for size, child in zip(
            active_sizes, child_sequences, strict=True
        )
    ]

    coalitions = np.empty(
        (len(sampled_sizes), num_players), dtype=bool
    )
    member_counts_by_size: dict[int, np.ndarray] = {}

    def consume(
        results: Any,
    ) -> None:
        for size, rows, member_counts in results:
            positions = np.flatnonzero(sampled_sizes == size)
            if rows.shape != (len(positions), num_players):
                raise RuntimeError(
                    "a fixed-slice design worker returned an invalid shape"
                )
            if member_counts.shape != (num_players,):
                raise RuntimeError(
                    "a fixed-slice design worker returned invalid counts"
                )
            coalitions[positions] = rows
            member_counts_by_size[size] = member_counts

    actual_jobs = min(design_jobs, len(tasks))
    if actual_jobs <= 1:
        consume(map(_run_fixed_slice_greedy_task, tasks))
    else:
        context = mp.get_context(start_method)
        with ProcessPoolExecutor(
            max_workers=actual_jobs,
            mp_context=context,
        ) as executor:
            consume(executor.map(_run_fixed_slice_greedy_task, tasks))

    missing_results = set(active_sizes) - set(member_counts_by_size)
    if missing_results:
        raise RuntimeError(
            "fixed-slice design workers omitted sizes "
            f"{sorted(missing_results)}"
        )

    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    missing_inclusion = 0
    missing_exclusion = 0
    exactly_balanced_sizes = 0
    for size in sizes:
        size_value = int(size)
        row_count = int(counts[size_value])
        inclusion = member_counts_by_size.get(
            size_value, np.zeros(num_players, dtype=np.int64)
        )
        exclusion = row_count - inclusion
        minimum_inclusion = min(
            minimum_inclusion, int(inclusion.min())
        )
        minimum_exclusion = min(
            minimum_exclusion, int(exclusion.min())
        )
        missing_inclusion += int(np.count_nonzero(inclusion == 0))
        missing_exclusion += int(np.count_nonzero(exclusion == 0))
        exactly_balanced_sizes += int(
            row_count > 0 and np.all(inclusion == inclusion[0])
        )
    coverage = {
        "all_player_size_strata_covered": (
            missing_inclusion == 0 and missing_exclusion == 0
        ),
        "expected_inner_sizes": int(len(sizes)),
        "observed_inner_sizes": int(len(active_sizes)),
        "missing_inner_sizes": int(len(sizes) - len(active_sizes)),
        "minimum_inclusion_count": int(minimum_inclusion),
        "minimum_exclusion_count": int(minimum_exclusion),
        "missing_inclusion_strata": int(missing_inclusion),
        "missing_exclusion_strata": int(missing_exclusion),
        "exactly_1_balanced_sizes": int(exactly_balanced_sizes),
    }
    return coalitions, coverage


def _resolve_mean_balance(
    num_players: int,
    sampled_sizes: np.ndarray,
    mean_balance: float,
    mean_balance_mode: str,
    *,
    radial_weights: bool,
) -> tuple[float, dict[str, float | str]]:
    r"""Convert a dimensionless first-moment weight to the raw objective scale.

    The greedy objective combines a second-moment frame term with the
    fixed-slice first-moment penalty.  Their unnormalized magnitudes depend on
    both dimension and, for the global radial design, the realized size
    schedule.  In ``"normalized"`` mode, ``mean_balance`` denotes the
    dimensionless :math:`\lambda_0` and is converted as

    .. math::

       \lambda_{\mathrm{eff}}
       = \lambda_0\left(1-\frac{1}{n-1}\right)
         \frac{1}{T}\sum_t w_t^2,

    where :math:`w_t^2=s_t(n-s_t)` for a radial design and one for an
    unweighted stratified design.  ``"raw"`` retains the legacy behavior in
    which ``mean_balance`` is used directly as
    :math:`\lambda_{\mathrm{eff}}`.
    """
    schedule = np.asarray(sampled_sizes, dtype=np.int64)
    if num_players < 4:
        raise ValueError("at least 4 players are required")
    if schedule.ndim != 1 or len(schedule) == 0:
        raise ValueError(
            "sampled_sizes must be a nonempty one-dimensional array"
        )
    if np.any(schedule < 2) or np.any(schedule > num_players - 2):
        raise ValueError("sampled sizes must lie in the OFA inner range")
    if not np.isfinite(mean_balance) or mean_balance < 0:
        raise ValueError("mean_balance must be finite and nonnegative")
    if mean_balance_mode not in {"normalized", "raw"}:
        raise ValueError("mean_balance_mode must be 'normalized' or 'raw'")

    dimension_correction = 1.0 - 1.0 / (num_players - 1)
    if radial_weights:
        weight_squared = schedule * (num_players - schedule)
        mean_weight_squared = float(
            np.mean(weight_squared, dtype=np.float64)
        )
    else:
        mean_weight_squared = 1.0
    normalization_factor = float(
        dimension_correction * mean_weight_squared
    )

    if mean_balance_mode == "normalized":
        lambda0 = float(mean_balance)
        effective_raw = float(lambda0 * normalization_factor)
    else:
        effective_raw = float(mean_balance)
        lambda0 = float(effective_raw / normalization_factor)

    diagnostics: dict[str, float | str] = {
        "mean_balance_mode": mean_balance_mode,
        "mean_balance_input": float(mean_balance),
        "mean_balance_lambda0": lambda0,
        "mean_balance_effective_raw": effective_raw,
        "mean_balance_normalization_factor": normalization_factor,
        "mean_balance_mean_weight_squared": mean_weight_squared,
        "mean_balance_dimension_correction": float(dimension_correction),
    }
    return effective_raw, diagnostics


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
    num_players: int,
    num_samples: int,
    seed: int = 0,
    *,
    compute_diagnostics: bool = True,
) -> CoalitionDesign:
    """Generate the independent Shapley-specific OFA design.

    Full frame diagnostics cost ``O(num_samples * num_players**2)``.  They
    remain enabled by default for backwards compatibility, but large data
    valuation experiments can disable them because they do not enter the
    estimator.
    """
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
        diagnostics=(
            frame_diagnostics(coalitions) if compute_diagnostics else {}
        ),
    )


def frame_coupled_design(
    num_players: int,
    num_samples: int,
    seed: int = 0,
    candidate_pool: int = 32,
    mean_balance: float = 1.0,
    *,
    mean_balance_mode: str = "normalized",
    relabel_seed: int | None = None,
) -> CoalitionDesign:
    """Generate a coupled batch with the exact OFA marginal for every row.

    Size draws use randomized systematic sampling.  Base coalitions then
    greedily reduce the weighted frame potential.  ``mean_balance`` is a
    dimensionless weight by default and is normalized using the realized size
    schedule.  Pass ``mean_balance_mode="raw"`` to reproduce the legacy
    objective, where the value is used directly.  A final independent,
    uniform batch-wide relabeling makes every row uniform on its fixed-size
    Boolean slice without changing geometric relationships inside the batch.
    ``relabel_seed`` can place that permutation on a separate RNG substream;
    its default ``None`` preserves the historical RNG path exactly.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    rng = np.random.default_rng(seed)
    sizes, probabilities, _ = inner_size_distribution(num_players)
    sampled_sizes = _systematic_sizes(
        sizes, probabilities, num_samples, rng
    )
    effective_mean_balance, balance_diagnostics = _resolve_mean_balance(
        num_players,
        sampled_sizes,
        mean_balance,
        mean_balance_mode,
        radial_weights=True,
    )
    base = _greedy_frame_rows(
        num_players,
        sampled_sizes,
        rng,
        candidate_pool,
        radial_weights=True,
        mean_balance=effective_mean_balance,
    )
    relabel_rng = (
        rng if relabel_seed is None else np.random.default_rng(relabel_seed)
    )
    permutation = relabel_rng.permutation(num_players)
    coalitions = _relabel_with_permutation(base, permutation)
    diagnostics: dict[str, Any] = frame_diagnostics(coalitions)
    diagnostics.update(balance_diagnostics)
    diagnostics["second_moment_scope"] = "global_weighted"
    diagnostics["relabel_seed"] = relabel_seed
    diagnostics["relabel_permutation_sha256"] = _permutation_sha256(
        permutation
    )
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="frame_coupled",
        seed=seed,
        diagnostics=diagnostics,
    )


def per_size_frame_coupled_design(
    num_players: int,
    num_samples: int,
    seed: int = 0,
    candidate_pool: int = 64,
    mean_balance: float = 1.0 / 16.0,
    *,
    second_moment_weight: float = 1.0,
    mean_balance_mode: str = "normalized",
    relabel_seed: int | None = None,
    design_jobs: int | None = None,
    design_start_method: str = "spawn",
    compute_frame_diagnostics: bool = True,
) -> CoalitionDesign:
    """Generate the design-only per-size counterpart of coupled Greedy.

    This function uses the randomized-systematic OFA size schedule and a
    batch-wide random relabeling, so every row retains the intended OFA
    sampling marginal.  Aggregation is chosen separately by the estimator;
    INSIDE-Greedy uses the covered official OFA ratio estimator.  Every
    fixed-size slice maintains a separate, unweighted second-moment operator.

    ``design_jobs=None`` preserves the historical single RNG stream exactly.
    Passing a positive ``design_jobs`` instead assigns deterministic child
    seed streams to the independent fixed-size problems and may solve them in
    parallel.  The resulting rows retain the same randomized-systematic size
    marginals and the same uniform fixed-slice marginals after the common
    random relabeling; only the otherwise-irrelevant coupling of candidate RNG
    draws *between different sizes* changes.  Results in this mode are
    invariant to the requested worker count.  Large experiments may set
    ``compute_frame_diagnostics=False`` because those diagnostics do not enter
    either the design objective or the OFA-ratio estimate.  Setting
    ``second_moment_weight=0`` gives the exact first-moment-only ablation;
    setting ``mean_balance=0`` gives the frame-only ablation.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if (
        not np.isfinite(second_moment_weight)
        or second_moment_weight < 0
    ):
        raise ValueError(
            "second_moment_weight must be finite and nonnegative"
        )
    if second_moment_weight == 0 and mean_balance == 0:
        raise ValueError("at least one moment objective must be active")
    sizes, probabilities, _ = inner_size_distribution(num_players)
    parallel_slices = design_jobs is not None
    if parallel_slices:
        if (
            isinstance(design_jobs, bool)
            or not isinstance(design_jobs, int)
            or design_jobs < 1
        ):
            raise ValueError("design_jobs must be a positive integer or None")
        root_sequence = np.random.SeedSequence(seed)
        size_sequence, slice_sequence, relabel_sequence = (
            root_sequence.spawn(3)
        )
        size_rng = np.random.default_rng(size_sequence)
        sampled_sizes = _systematic_sizes(
            sizes, probabilities, num_samples, size_rng
        )
    else:
        rng = np.random.default_rng(seed)
        sampled_sizes = _systematic_sizes(
            sizes, probabilities, num_samples, rng
        )
    effective_mean_balance, balance_diagnostics = _resolve_mean_balance(
        num_players,
        sampled_sizes,
        mean_balance,
        mean_balance_mode,
        radial_weights=False,
    )
    if parallel_slices:
        base, coverage = _parallel_fixed_slice_rows(
            num_players,
            sampled_sizes,
            candidate_pool=candidate_pool,
            mean_balance=effective_mean_balance,
            second_moment_weight=second_moment_weight,
            seed_sequence=slice_sequence,
            design_jobs=design_jobs,
            start_method=design_start_method,
        )
        relabel_rng = (
            np.random.default_rng(relabel_sequence)
            if relabel_seed is None
            else np.random.default_rng(relabel_seed)
        )
    else:
        base = _greedy_frame_rows(
            num_players,
            sampled_sizes,
            rng,
            candidate_pool,
            radial_weights=False,
            mean_balance=effective_mean_balance,
            second_moment_weight=second_moment_weight,
            second_moment_scope="per_size",
        )
        coverage = None
        relabel_rng = (
            rng
            if relabel_seed is None
            else np.random.default_rng(relabel_seed)
        )
    permutation = relabel_rng.permutation(num_players)
    coalitions = _relabel_with_permutation(base, permutation)
    diagnostics: dict[str, Any] = (
        frame_diagnostics(coalitions)
        if compute_frame_diagnostics
        else {}
    )
    diagnostics.update(balance_diagnostics)
    diagnostics["second_moment_scope"] = "per_size"
    diagnostics["second_moment_weight"] = float(second_moment_weight)
    diagnostics["objective_components"] = [
        component
        for component, active in (
            ("first_moment", effective_mean_balance > 0),
            ("second_moment", second_moment_weight > 0),
        )
        if active
    ]
    diagnostics["relabel_seed"] = relabel_seed
    diagnostics["relabel_permutation_sha256"] = _permutation_sha256(
        permutation
    )
    diagnostics["frame_diagnostics_computed"] = bool(
        compute_frame_diagnostics
    )
    diagnostics["fixed_slice_rng_partitioning"] = (
        "independent_seedsequence_substreams"
        if parallel_slices
        else "historical_shared_stream"
    )
    diagnostics["design_jobs"] = (
        None if design_jobs is None else int(design_jobs)
    )
    diagnostics["design_start_method"] = (
        None if not parallel_slices else design_start_method
    )
    if coverage is not None:
        diagnostics["ratio_coverage"] = coverage
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="frame_coupled_per_size",
        seed=seed,
        diagnostics=diagnostics,
    )


def paired_frame_scope_designs(
    num_players: int,
    num_samples: int,
    seed: int = 0,
    candidate_pool: int = 32,
    mean_balance: float = 1.0,
    *,
    mean_balance_mode: str = "normalized",
    relabel_seed: int | None = None,
) -> tuple[CoalitionDesign, CoalitionDesign]:
    """Generate a strict global-versus-per-size scope ablation pair.

    Both outputs use the same seed for their randomized-systematic size and
    candidate streams, plus a common *independent* seed for their final
    batch-wide random relabeling.  The common-random-number candidate streams
    can diverge only after a scope-dependent choice changes a blocked set or
    draw count.  The separate relabel substream prevents that divergence from
    changing player labels in a non-exchangeable game.

    When ``relabel_seed`` is omitted, a deterministic SeedSequence child with
    a fixed domain-separation salt is derived from ``seed``.  Existing
    standalone APIs retain their historical behavior because their own
    ``relabel_seed`` defaults remain ``None``.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    if relabel_seed is None:
        relabel_seed = int(
            np.random.SeedSequence([seed, 0x1A51DE]).generate_state(
                1, dtype=np.uint32
            )[0]
        )
    global_design = frame_coupled_design(
        num_players,
        num_samples,
        seed=seed,
        candidate_pool=candidate_pool,
        mean_balance=mean_balance,
        mean_balance_mode=mean_balance_mode,
        relabel_seed=relabel_seed,
    )
    per_size_design = per_size_frame_coupled_design(
        num_players,
        num_samples,
        seed=seed,
        candidate_pool=candidate_pool,
        mean_balance=mean_balance,
        mean_balance_mode=mean_balance_mode,
        relabel_seed=relabel_seed,
    )
    if not np.array_equal(global_design.sizes, per_size_design.sizes):
        raise RuntimeError("paired scope designs have different size schedules")
    permutation_hash = global_design.diagnostics[
        "relabel_permutation_sha256"
    ]
    if (
        permutation_hash
        != per_size_design.diagnostics["relabel_permutation_sha256"]
    ):
        raise RuntimeError("paired scope designs have different relabelings")
    for design in (global_design, per_size_design):
        design.diagnostics.update(
            {
                "paired_scope_ablation": True,
                "shared_size_schedule": True,
                "shared_final_relabel": True,
                "candidate_common_random_numbers": True,
                "relabel_seed": relabel_seed,
            }
        )
    return global_design, per_size_design


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
    mean_balance: float = 1.0,
    *,
    mean_balance_mode: str = "normalized",
) -> CoalitionDesign:
    """Preallocate OFA-optimal counts and frame-balance every fixed-size slice.

    Here the frame objective is unweighted within each slice, so normalized
    ``mean_balance`` uses only the dimension correction.  Use
    ``mean_balance_mode="raw"`` for the legacy unnormalized objective.
    """
    rng = np.random.default_rng(seed)
    sizes, probabilities, _ = inner_size_distribution(num_players)
    counts = _minimum_one_counts(num_samples, probabilities)
    sampled_schedule = np.repeat(sizes, counts)
    effective_mean_balance, balance_diagnostics = _resolve_mean_balance(
        num_players,
        sampled_schedule,
        mean_balance,
        mean_balance_mode,
        radial_weights=False,
    )

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
            mean_balance=effective_mean_balance,
        )
        blocks.append(_random_relabel(base, rng))
        recorded_sizes.append(size_sequence)

    coalitions = np.concatenate(blocks, axis=0)
    sampled_sizes = np.concatenate(recorded_sizes)
    order = rng.permutation(num_samples)
    coalitions = coalitions[order]
    sampled_sizes = sampled_sizes[order]
    diagnostics: dict[str, Any] = _stratified_diagnostics(
        coalitions, sampled_sizes
    )
    diagnostics.update(balance_diagnostics)
    return CoalitionDesign(
        coalitions=coalitions,
        sizes=sampled_sizes,
        method="frame_stratified",
        seed=seed,
        diagnostics=diagnostics,
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
    *,
    compute_frame_diagnostics: bool = True,
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
    target = (efficiency_projector(num_players) / (num_players - 1)
              if compute_frame_diagnostics else None)
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
        # A common relabeling preserves the optimized slice frame.  Track the
        # conjugated small operator directly so diagnostics do not multiply
        # all O(n * num_orbits) materialized directions.
        permutation = rng.permutation(num_players)
        relabeled_rows = np.empty_like(slice_rows)
        relabeled_rows[:, permutation] = slice_rows
        slice_rows = relabeled_rows
        if compute_frame_diagnostics:
            base_operator = _circulant_from_first_row(signature_sum / int(count))
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
        ) if compute_frame_diagnostics else {"frame_diagnostics_computed": False},
    )


def paired_cyclic_orbit_designs(
    num_players: int,
    num_orbits: int,
    seed: int = 0,
    candidate_pool: int = 4,
) -> tuple[CoalitionDesign, CoalitionDesign]:
    """Return a strict Random-Orbit/INSIDE-Orbit ablation pair.

    At every orbit, both designs see the exact same ``candidate_pool`` base
    coalitions.  Random-Orbit selects one uniformly, whereas INSIDE-Orbit
    selects the minimum per-size frame-potential increment.  Independent RNG
    substreams keep candidate generation, random selection, greedy tie
    breaking, and the shared per-size relabeling from perturbing one another.
    Consequently the selection rule is the pair's only algorithmic change.
    """
    if num_players < 4:
        raise ValueError("at least 4 players are required")
    if num_orbits < 1:
        raise ValueError("num_orbits must be positive")
    if candidate_pool < 1:
        raise ValueError("candidate_pool must be positive")

    sizes, probabilities, _ = inner_size_distribution(num_players)
    orbit_counts = _minimum_one_counts(num_orbits, probabilities)
    root = np.random.SeedSequence(seed)
    candidate_seed, random_seed, tie_seed, relabel_seed = root.spawn(4)
    candidate_rng = np.random.default_rng(candidate_seed)
    random_rng = np.random.default_rng(random_seed)
    tie_rng = np.random.default_rng(tie_seed)
    relabel_rng = np.random.default_rng(relabel_seed)

    random_blocks: list[np.ndarray] = []
    frame_blocks: list[np.ndarray] = []
    recorded_sizes: list[np.ndarray] = []
    random_operators: list[np.ndarray] = []
    frame_operators: list[np.ndarray] = []
    candidate_digest = hashlib.sha256()
    relabel_digest = hashlib.sha256()
    target = efficiency_projector(num_players) / (num_players - 1)

    for size_value, count_value in zip(
        sizes, orbit_counts, strict=True
    ):
        size = int(size_value)
        count = int(count_value)
        random_signature_sum = np.zeros(num_players, dtype=np.float64)
        frame_signature_sum = np.zeros(num_players, dtype=np.float64)
        random_orbits: list[np.ndarray] = []
        frame_orbits: list[np.ndarray] = []
        for orbit_index in range(count):
            candidates = _candidate_coalitions(
                num_players, size, candidate_pool, candidate_rng
            )
            signatures = _cyclic_orbit_signatures(candidates)
            candidate_digest.update(
                np.asarray([size, orbit_index], dtype="<i8").tobytes()
            )
            candidate_digest.update(
                np.ascontiguousarray(candidates, dtype=np.uint8).tobytes()
            )

            random_chosen = int(random_rng.integers(len(candidates)))
            scores = num_players * (
                2.0 * (signatures @ frame_signature_sum)
                + np.einsum("bi,bi->b", signatures, signatures)
            )
            best = np.flatnonzero(
                np.isclose(scores, scores.min(), rtol=1e-12, atol=1e-14)
            )
            frame_chosen = int(tie_rng.choice(best))

            random_signature_sum += signatures[random_chosen]
            frame_signature_sum += signatures[frame_chosen]
            random_orbits.append(_cyclic_orbit(candidates[random_chosen]))
            frame_orbits.append(_cyclic_orbit(candidates[frame_chosen]))

        permutation = relabel_rng.permutation(num_players)
        relabel_digest.update(
            np.ascontiguousarray(permutation, dtype="<i8").tobytes()
        )
        random_rows = _relabel_with_permutation(
            np.concatenate(random_orbits, axis=0), permutation
        )
        frame_rows = _relabel_with_permutation(
            np.concatenate(frame_orbits, axis=0), permutation
        )
        random_operator = _circulant_from_first_row(
            random_signature_sum / count
        )
        frame_operator = _circulant_from_first_row(
            frame_signature_sum / count
        )
        relabeled_random_operator = np.empty_like(random_operator)
        relabeled_frame_operator = np.empty_like(frame_operator)
        relabeled_random_operator[np.ix_(permutation, permutation)] = (
            random_operator
        )
        relabeled_frame_operator[np.ix_(permutation, permutation)] = (
            frame_operator
        )

        random_blocks.append(random_rows)
        frame_blocks.append(frame_rows)
        recorded_sizes.append(
            np.full(len(random_rows), size, dtype=np.int64)
        )
        random_operators.append(relabeled_random_operator)
        frame_operators.append(relabeled_frame_operator)

    sampled_sizes = np.concatenate(recorded_sizes)
    common = {
        "paired_orbit_ablation": True,
        "candidate_pool": int(candidate_pool),
        "num_orbits": int(num_orbits),
        "orbit_counts_by_size": orbit_counts.astype(int).tolist(),
        "shared_candidate_pools": True,
        "shared_candidate_pool_sha256": candidate_digest.hexdigest(),
        "shared_relabeling": True,
        "shared_relabel_permutations_sha256": relabel_digest.hexdigest(),
        "exact_first_moment_by_complete_orbits": True,
    }
    random_diagnostics: dict[str, Any] = _summarize_stratified_operators(
        random_operators, target, [0.0] * len(random_operators)
    )
    random_diagnostics.update(common)
    random_diagnostics["selection_rule"] = (
        "uniform_random_from_shared_candidate_pool"
    )
    frame_diagnostics_by_slice: dict[str, Any] = (
        _summarize_stratified_operators(
            frame_operators, target, [0.0] * len(frame_operators)
        )
    )
    frame_diagnostics_by_slice.update(common)
    frame_diagnostics_by_slice["selection_rule"] = (
        "minimum_per_size_frame_increment"
    )

    random_design = CoalitionDesign(
        coalitions=np.concatenate(random_blocks, axis=0),
        sizes=sampled_sizes.copy(),
        method="cyclic_orbit_random",
        seed=seed,
        diagnostics=random_diagnostics,
    )
    frame_design = CoalitionDesign(
        coalitions=np.concatenate(frame_blocks, axis=0),
        sizes=sampled_sizes.copy(),
        method="cyclic_orbit_frame",
        seed=seed,
        diagnostics=frame_diagnostics_by_slice,
    )
    return random_design, frame_design

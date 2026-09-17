"""Geometric identities used by Frame-OFA.

All routines operate on coalitions represented as Boolean rows.  The
normalized direction associated with a nontrivial coalition S is

    u_S = sqrt(n / (|S| (n-|S|))) (1_S - |S|/n 1).

It is a unit vector in the efficiency subspace 1^perp.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_DEFAULT_FLOAT_CHUNK_BYTES = 16 * 2**20


def efficiency_projector(num_players: int) -> np.ndarray:
    """Return the orthogonal projector onto the efficiency subspace."""
    if num_players < 2:
        raise ValueError("num_players must be at least 2")
    ones = np.ones((num_players, num_players), dtype=np.float64)
    return np.eye(num_players, dtype=np.float64) - ones / num_players


def centered_directions(coalitions: np.ndarray) -> np.ndarray:
    """Map Boolean coalition rows to their normalized efficiency directions."""
    coalitions = np.asarray(coalitions, dtype=bool)
    if coalitions.ndim == 1:
        coalitions = coalitions[None, :]
    if coalitions.ndim != 2:
        raise ValueError("coalitions must be a one- or two-dimensional array")

    num_players = coalitions.shape[1]
    sizes = coalitions.sum(axis=1).astype(np.float64)
    if np.any(sizes <= 0) or np.any(sizes >= num_players):
        raise ValueError("directions are defined only for nonempty, nonfull coalitions")

    centered = coalitions.astype(np.float64) - sizes[:, None] / num_players
    scales = np.sqrt(num_players / (sizes * (num_players - sizes)))
    return centered * scales[:, None]


def inner_frame_target(num_players: int) -> np.ndarray:
    """Target operator when OFA exactly evaluates sizes 0, 1, n-1, and n.

    There are n-3 randomly estimated inner sizes, each contributing
    P/(n-1) on an additive game.  Hence the inner target is
    (n-3)/(n-1) P, not P.
    """
    if num_players < 4:
        raise ValueError("the OFA inner-size construction requires at least 4 players")
    return ((num_players - 3) / (num_players - 1)) * efficiency_projector(num_players)


def _inner_frame_statistics(
    coalitions: np.ndarray,
    *,
    chunk_rows: int | None,
    collect_slice_means: bool,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Accumulate the inner frame and optional slice means in bounded memory.

    Only one ``chunk_rows x n`` floating-point buffer is live at a time.  In
    particular, this avoids both the full direction matrix and the second
    full weighted-direction matrix that a dense implementation would need.
    """
    coalitions = np.asarray(coalitions, dtype=bool)
    if coalitions.ndim != 2:
        raise ValueError("coalitions must be a two-dimensional array")
    num_samples, num_players = coalitions.shape
    if num_samples == 0:
        raise ValueError("at least one coalition is required")
    if num_players < 4:
        raise ValueError("the OFA inner-size construction requires at least 4 players")
    if chunk_rows is None:
        chunk_rows = max(
            1,
            _DEFAULT_FLOAT_CHUNK_BYTES
            // (num_players * np.dtype(np.float64).itemsize),
        )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    operator_sum = np.zeros(
        (num_players, num_players), dtype=np.float64
    )
    if collect_slice_means:
        slice_counts = np.zeros(num_players - 3, dtype=np.int64)
        slice_member_counts = np.zeros(
            (num_players - 3, num_players), dtype=np.float64
        )
    else:
        slice_counts = None
        slice_member_counts = None

    normalizer = np.reciprocal(
        np.sqrt(
            np.arange(2, num_players - 1, dtype=np.float64)
            * np.arange(num_players - 2, 1, -1, dtype=np.float64)
        )
    ).sum()

    for start in range(0, num_samples, chunk_rows):
        stop = min(start + chunk_rows, num_samples)
        # This is the only rows-by-players float allocation in the loop.
        boolean_rows = coalitions[start:stop]
        rows = boolean_rows.astype(np.float64)
        sizes = rows.sum(axis=1)
        if np.any(sizes < 2) or np.any(sizes > num_players - 2):
            raise ValueError(
                "all coalitions must have sizes in {2, ..., n-2}"
            )

        if collect_slice_means:
            assert slice_counts is not None
            assert slice_member_counts is not None
            size_indices = sizes.astype(np.int64) - 2
            slice_counts += np.bincount(
                size_indices, minlength=num_players - 3
            )
            for size_index in np.unique(size_indices):
                slice_member_counts[size_index] += boolean_rows[
                    size_indices == size_index
                ].sum(axis=0)

        # radial * ||normalization||^2 = n / sqrt(s(n-s)).
        # Center and scale this same buffer in place before its rank update.
        rows -= sizes[:, None] / num_players
        outer_weights = num_players / np.sqrt(
            sizes * (num_players - sizes)
        )
        rows *= np.sqrt(outer_weights)[:, None]
        operator_sum += rows.T @ rows

    return (
        normalizer * operator_sum / num_samples,
        slice_counts,
        slice_member_counts,
    )


def inner_frame_operator(
    coalitions: np.ndarray, *, chunk_rows: int | None = None
) -> np.ndarray:
    """Return the coupled-HT frame operator with bounded float memory.

    ``chunk_rows`` is primarily useful for verification and memory tuning.
    By default the only floating row buffer is capped near 16 MiB.
    """
    operator, _, _ = _inner_frame_statistics(
        coalitions,
        chunk_rows=chunk_rows,
        collect_slice_means=False,
    )
    return operator


def fixed_slice_frame_operator(coalitions: np.ndarray) -> np.ndarray:
    """Return the empirical second moment on a single Boolean slice."""
    coalitions = np.asarray(coalitions, dtype=bool)
    if coalitions.ndim != 2 or len(coalitions) == 0:
        raise ValueError("a nonempty two-dimensional coalition array is required")
    sizes = coalitions.sum(axis=1)
    if np.any(sizes != sizes[0]):
        raise ValueError("all coalitions must have the same size")
    directions = centered_directions(coalitions)
    return directions.T @ directions / len(directions)


def fixed_slice_moment_diagnostics(
    coalitions: np.ndarray,
    sizes: np.ndarray | None = None,
    *,
    require_all_inner_sizes: bool = False,
) -> dict[str, Any]:
    """Compute first- and second-moment errors on every fixed-size slice.

    The two headline summaries weight coalition sizes equally:

    ``first_moment_rms``
        ``sqrt(mean_s ||mean(u | s)||_2**2)``;

    ``frame_frobenius_rms``
        ``sqrt(mean_s ||mean(u u^T | s) - P/(n-1)||_F**2)``.

    Row-weighted counterparts are retained only as sensitivity diagnostics.
    Missing slices are never silently treated as zero-error observations.
    """
    rows = np.asarray(coalitions, dtype=bool)
    if rows.ndim != 2 or len(rows) == 0:
        raise ValueError(
            "coalitions must be a nonempty two-dimensional array"
        )
    num_samples, num_players = rows.shape
    observed_sizes = (
        rows.sum(axis=1, dtype=np.int64)
        if sizes is None
        else np.asarray(sizes, dtype=np.int64)
    )
    if observed_sizes.shape != (num_samples,):
        raise ValueError("sizes must contain one entry per coalition")
    if not np.array_equal(rows.sum(axis=1), observed_sizes):
        raise ValueError("sizes disagree with coalition cardinalities")
    if num_players < 4:
        raise ValueError("at least four players are required")
    if np.any(observed_sizes < 2) or np.any(
        observed_sizes > num_players - 2
    ):
        raise ValueError("all rows must use OFA inner sizes 2 through n-2")

    expected_sizes = np.arange(2, num_players - 1, dtype=np.int64)
    counts = np.bincount(
        observed_sizes, minlength=num_players + 1
    )[2 : num_players - 1]
    missing = expected_sizes[counts == 0]
    if require_all_inner_sizes and len(missing):
        raise ValueError(
            "fixed-slice diagnostics require every inner size; missing "
            f"{missing.tolist()}"
        )

    projector = efficiency_projector(num_players)
    target = projector / (num_players - 1)
    present_sizes: list[int] = []
    present_counts: list[int] = []
    first_norms: list[float] = []
    frame_discrepancies: list[float] = []
    for size, count in zip(expected_sizes, counts, strict=True):
        if count == 0:
            continue
        take = rows[observed_sizes == size]
        inclusion = take.sum(axis=0, dtype=np.int64)
        if np.all(inclusion * num_players == int(size) * int(count)):
            first_norm = 0.0
        else:
            centered_mean = (
                inclusion.astype(np.float64) / count
                - float(size) / num_players
            )
            scale = np.sqrt(
                num_players / (float(size) * (num_players - size))
            )
            first_norm = float(np.linalg.norm(scale * centered_mean))
        directions = centered_directions(take)
        operator = directions.T @ directions / count
        frame_discrepancy = float(
            np.linalg.norm(operator - target, ord="fro")
        )
        present_sizes.append(int(size))
        present_counts.append(int(count))
        first_norms.append(first_norm)
        frame_discrepancies.append(frame_discrepancy)

    first = np.asarray(first_norms, dtype=np.float64)
    second = np.asarray(frame_discrepancies, dtype=np.float64)
    weights = np.asarray(present_counts, dtype=np.float64)
    weights /= weights.sum()
    return {
        "aggregation": "equal_size_macro_rms",
        "target": "P/(n-1)",
        "expected_inner_size_count": int(len(expected_sizes)),
        "observed_inner_size_count": int(len(present_sizes)),
        "all_inner_sizes_present": len(missing) == 0,
        "missing_sizes": missing.astype(int).tolist(),
        "sizes": present_sizes,
        "counts": present_counts,
        "first_moment_norm_by_size": first.tolist(),
        "frame_frobenius_by_size": second.tolist(),
        "first_moment_rms": float(np.sqrt(np.mean(np.square(first)))),
        "frame_frobenius_rms": float(
            np.sqrt(np.mean(np.square(second)))
        ),
        "first_moment_worst": float(first.max()),
        "frame_frobenius_worst": float(second.max()),
        "row_weighted_first_moment_rms": float(
            np.sqrt(np.dot(weights, np.square(first)))
        ),
        "row_weighted_frame_frobenius_rms": float(
            np.sqrt(np.dot(weights, np.square(second)))
        ),
    }

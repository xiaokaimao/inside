"""Exact utilities and Shapley--Shubik values for weighted voting games."""

from __future__ import annotations

import math

import numpy as np


def _validated_weights(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("weights must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("weights must be finite")
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("weights must be nonnegative integers")
    return values.astype(np.int64, copy=True)


def evaluate_weighted_voting(
    coalitions: np.ndarray,
    weights: np.ndarray,
    quota: int,
) -> np.ndarray:
    """Evaluate ``1[sum_i w_i z_i >= quota]`` for one or many coalitions."""
    integer_weights = _validated_weights(weights)
    rows = np.asarray(coalitions, dtype=bool)
    was_vector = rows.ndim == 1
    if was_vector:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != len(integer_weights):
        raise ValueError("coalitions and weights disagree on player count")
    if not isinstance(quota, (int, np.integer)):
        raise ValueError("quota must be an integer")
    values = (rows @ integer_weights >= int(quota)).astype(np.float64)
    return values[0] if was_vector else values


def exact_shapley_shubik(
    weights: np.ndarray,
    quota: int,
) -> np.ndarray:
    """Compute exact Shapley--Shubik values by cardinality--weight DP.

    Let ``c_i(k, t)`` count size-``k`` coalitions excluding player ``i``
    whose total weight is ``t``.  Player ``i`` is pivotal exactly when
    ``quota - w_i <= t <= quota - 1``, so

    ``phi_i = sum_k sum_t c_i(k,t) / (n * comb(n-1,k))``.

    A single subset generating table is built for all players.  Dividing its
    generating polynomial by ``1 + z x**w_i`` recovers each leave-one-out
    table.  Tables are truncated at ``quota - 1`` because winning predecessor
    coalitions can never be pivotal.  Python-integer object arrays keep the
    subset counts exact instead of relying on a final normalization.
    """
    integer_weights = _validated_weights(weights)
    if not isinstance(quota, (int, np.integer)):
        raise ValueError("quota must be an integer")
    threshold = int(quota)
    num_players = len(integer_weights)
    if threshold <= 0 or threshold > int(integer_weights.sum()):
        return np.zeros(num_players, dtype=np.float64)

    # C[k, t] is the number of all-player coalitions of size k and losing
    # weight t.  Updating k in descending order enforces 0/1 inclusion.
    counts = np.zeros(
        (num_players + 1, threshold), dtype=object
    )
    counts[0, 0] = 1
    processed = 0
    for weight_value in integer_weights:
        weight = int(weight_value)
        if weight < threshold:
            for size in range(processed, -1, -1):
                counts[size + 1, weight:threshold] += counts[
                    size, : threshold - weight
                ]
        processed += 1

    coefficients = np.asarray(
        [
            1.0 / (num_players * math.comb(num_players - 1, size))
            for size in range(num_players)
        ],
        dtype=np.float64,
    )
    value_by_weight: dict[int, float] = {}
    for weight_value in np.unique(integer_weights):
        weight = int(weight_value)
        # C = D_i * (1 + z x**w_i).  Solve cardinalities from low to high.
        leave_one_out = np.zeros(
            (num_players, threshold), dtype=object
        )
        leave_one_out[0] = counts[0]
        for size in range(1, num_players):
            leave_one_out[size] = counts[size]
            if weight < threshold:
                leave_one_out[size, weight:threshold] -= leave_one_out[
                    size - 1, : threshold - weight
                ]

        pivotal_start = max(0, threshold - weight)
        pivotal_counts = np.asarray(
            [
                int(
                    sum(
                        leave_one_out[
                            size, pivotal_start:threshold
                        ]
                    )
                )
                for size in range(num_players)
            ],
            dtype=np.float64,
        )
        value_by_weight[weight] = float(
            coefficients @ pivotal_counts
        )

    return np.asarray(
        [value_by_weight[int(weight)] for weight in integer_weights],
        dtype=np.float64,
    )

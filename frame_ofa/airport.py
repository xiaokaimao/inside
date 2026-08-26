"""Utilities and exact Shapley values for airport games."""

from __future__ import annotations

import numpy as np


def _validated_costs(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("costs must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)):
        raise ValueError("costs must be finite")
    if np.any(values < 0.0):
        raise ValueError("costs must be nonnegative")
    return values.copy()


def evaluate_airport(
    coalitions: np.ndarray,
    costs: np.ndarray,
) -> np.ndarray:
    """Evaluate ``max_{i in S} c_i`` with ``v(empty)=0``."""
    values = _validated_costs(costs)
    rows = np.asarray(coalitions, dtype=bool)
    was_vector = rows.ndim == 1
    if was_vector:
        rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] != len(values):
        raise ValueError("coalitions and costs disagree on player count")
    utilities = np.max(np.where(rows, values[None, :], 0.0), axis=1)
    return utilities[0] if was_vector else utilities


def exact_airport_shapley(costs: np.ndarray) -> np.ndarray:
    """Return the exact Shapley cost allocation for an airport game.

    For distinct positive levels ``d_1 < ... < d_m``, write

    ``max_{i in S} c_i = sum_k (d_k-d_{k-1}) 1[S intersects H_k]``,

    where ``H_k = {i: c_i >= d_k}`` and ``d_0=0``.  Every player in
    ``H_k`` receives ``(d_k-d_{k-1}) / |H_k|`` in that threshold game.
    """
    values = _validated_costs(costs)
    result = np.zeros(len(values), dtype=np.float64)
    previous = 0.0
    for level in np.unique(values[values > 0.0]):
        eligible = values >= level
        result[eligible] += (float(level) - previous) / int(eligible.sum())
        previous = float(level)
    return result

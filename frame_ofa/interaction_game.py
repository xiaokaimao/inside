"""Sparse polynomial cooperative games with exact coefficient-based Shapley values."""
from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np


@dataclass(frozen=True)
class InteractionGame:
    additive: np.ndarray
    pairs: np.ndarray
    pair_weights: np.ndarray
    triples: np.ndarray
    triple_weights: np.ndarray

    def __post_init__(self):
        a = np.asarray(self.additive, dtype=float)
        if a.ndim != 1 or len(a) < 4 or not np.isfinite(a).all():
            raise ValueError('additive must be a finite vector with at least four players')
        object.__setattr__(self, 'additive', a.copy())
        for name, degree in [('pairs', 2), ('triples', 3)]:
            raw = np.asarray(getattr(self, name))
            indices = np.asarray(raw, dtype=np.int64)
            weight_name = 'pair_weights' if degree == 2 else 'triple_weights'
            weights = np.asarray(getattr(self, weight_name), dtype=float)
            if (weights.ndim != 1 or indices.shape != (len(weights), degree) or not np.array_equal(raw, indices)
                    or np.any(indices < 0) or np.any(indices >= len(a))
                    or np.any(np.diff(indices, axis=1) <= 0)
                    or not np.isfinite(weights).all()):
                raise ValueError('invalid interaction indices or coefficients')
            if len(np.unique(indices, axis=0)) != len(indices):
                raise ValueError('duplicate interactions')
            object.__setattr__(self, name, indices.copy())
            object.__setattr__(self, weight_name, weights.copy())

    @property
    def num_players(self):
        return len(self.additive)

    def exact_shapley(self):
        values = self.additive.copy()
        for indices, weights in [(self.pairs, self.pair_weights), (self.triples, self.triple_weights)]:
            for column in indices.T:
                np.add.at(values, column, weights / indices.shape[1])
        return values

    def evaluate(self, coalitions, batch_size=128):
        raw = np.asarray(coalitions)
        if raw.ndim != 2 or raw.shape[1] != self.num_players:
            raise ValueError('coalitions must be a binary matrix with n columns')
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        values = np.empty(len(raw))
        for start in range(0, len(raw), batch_size):
            chunk = raw[start:start + batch_size]
            if chunk.dtype != np.bool_ and not np.isin(chunk, [0, 1]).all():
                raise ValueError('coalitions must be a binary matrix with n columns')
            rows = chunk.astype(bool, copy=False)
            value = rows @ self.additive
            for indices, weights in [(self.pairs, self.pair_weights), (self.triples, self.triple_weights)]:
                active = np.ones((len(rows), len(indices)), dtype=bool)
                for column in indices.T:
                    active &= rows[:, column]
                value += active @ weights
            values[start:start + len(rows)] = value
        return values

    def save(self, path):
        np.savez_compressed(path, additive=self.additive, pairs=self.pairs,
                            pair_weights=self.pair_weights, triples=self.triples,
                            triple_weights=self.triple_weights, exact_shapley=self.exact_shapley())


def _edges(n, degree, count, rng):
    if not 0 <= count <= comb(n, degree):
        raise ValueError('interaction count exceeds possible unique interactions')
    if count > comb(n, degree) // 2:
        candidates = np.array(list(combinations(range(n), degree)), dtype=np.int64)
        return candidates[rng.choice(len(candidates), count, replace=False)]
    edges = set()
    while len(edges) < count:
        edges.add(tuple(sorted(rng.choice(n, degree, replace=False).tolist())))
    return np.asarray(sorted(edges), dtype=np.int64).reshape(-1, degree)


def make_interaction_game(n, degree=3, seed=20260923, pairs_per_player=4,
                          triples_per_player=2, pair_scale=1.0, triple_scale=1.0):
    """O(n) terms; signed weights scaled to constant expected per-player variance.

    Pairwise and cubic games with the same seed share additive and pair terms.
    Scales are coefficient scales before division by sqrt(expected incident degree).
    """
    if n < 4 or degree not in (2, 3):
        raise ValueError('require n >= 4 and degree 2 or 3')
    if pairs_per_player < 1 or triples_per_player < 0 or min(pair_scale, triple_scale) < 0:
        raise ValueError('invalid interaction density or scale')
    a_rng, p_rng, t_rng = map(np.random.default_rng, np.random.SeedSequence(seed).spawn(3))
    pair_count = min(n * pairs_per_player, comb(n, 2))
    triple_count = min(n * triples_per_player, comb(n, 3)) if degree == 3 else 0
    pairs = _edges(n, 2, pair_count, p_rng)
    triples = _edges(n, 3, triple_count, t_rng)
    return InteractionGame(a_rng.normal(size=n), pairs,
        p_rng.normal(scale=pair_scale / np.sqrt(2 * pair_count / n), size=pair_count), triples,
        t_rng.normal(scale=triple_scale / np.sqrt(max(3 * triple_count / n, 1)), size=triple_count))

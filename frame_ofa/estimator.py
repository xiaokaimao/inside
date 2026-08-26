"""Linear Shapley estimators driven by OFA and frame-coupled designs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .design import (
    CoalitionDesign,
    cyclic_orbit_frame_design,
    frame_coupled_design,
    iid_ofa_design,
    inner_size_distribution,
    per_size_frame_coupled_design,
    stratified_frame_design,
)
from .geometry import centered_directions

Utility = Callable[[np.ndarray], float]


class CoalitionBatchEvaluator(Protocol):
    def evaluate(self, coalitions: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class OFABoundary:
    """The 2n+2 exact evaluations used by the original OFA algorithm."""

    empty: float
    full: float
    singletons: np.ndarray
    leave_one_out: np.ndarray

    @property
    def num_players(self) -> int:
        return len(self.singletons)


@dataclass
class EstimateResult:
    values: np.ndarray
    design: CoalitionDesign
    utilities: np.ndarray
    boundary: OFABoundary
    utility_evaluations: int


def boundary_coalitions(num_players: int) -> np.ndarray:
    """Return the canonical 2n+2 OFA boundary coalitions."""
    if num_players < 4:
        raise ValueError("at least 4 players are required")
    rows = np.zeros((2 * num_players + 2, num_players), dtype=bool)
    rows[1 : num_players + 1] = np.eye(num_players, dtype=bool)
    rows[num_players + 1].fill(True)
    rows[num_players + 2 :] = ~np.eye(num_players, dtype=bool)
    return rows


def boundary_from_utilities(
    utilities: np.ndarray, num_players: int
) -> OFABoundary:
    """Decode utilities evaluated in :func:`boundary_coalitions` order."""
    values = np.asarray(utilities, dtype=np.float64)
    if values.shape != (2 * num_players + 2,):
        raise ValueError("boundary utilities must contain exactly 2n+2 values")
    return OFABoundary(
        empty=float(values[0]),
        full=float(values[num_players + 1]),
        singletons=values[1 : num_players + 1].copy(),
        leave_one_out=values[num_players + 2 :].copy(),
    )


def evaluate_boundary(utility: Utility, num_players: int) -> OFABoundary:
    """Evaluate sizes 0, 1, n-1, and n exactly."""
    rows = boundary_coalitions(num_players)
    return boundary_from_utilities(
        np.asarray([utility(row.copy()) for row in rows]),
        num_players,
    )


def shapley_boundary_vector(boundary: OFABoundary) -> np.ndarray:
    """Return the exact contributions of sizes 0, 1, n-1, and n."""
    num_players = boundary.num_players
    if len(boundary.leave_one_out) != num_players:
        raise ValueError("boundary arrays have inconsistent lengths")

    singleton_others = boundary.singletons.sum() - boundary.singletons
    leave_out_others = boundary.leave_one_out.sum() - boundary.leave_one_out
    values = np.full(
        num_players,
        (boundary.full - boundary.empty) / num_players,
        dtype=np.float64,
    )
    values += boundary.singletons / num_players
    values -= singleton_others / (num_players * (num_players - 1))
    values += leave_out_others / (num_players * (num_players - 1))
    values -= boundary.leave_one_out / num_players
    return values


def _size_baseline(
    boundary: OFABoundary, sizes: np.ndarray, baseline: str
) -> np.ndarray:
    if baseline == "none":
        return np.zeros_like(sizes, dtype=np.float64)
    if baseline == "linear":
        return boundary.empty + (sizes / boundary.num_players) * (
            boundary.full - boundary.empty
        )
    raise ValueError("baseline must be 'linear' or 'none'")


def _weighted_direction_mean(
    coalitions: np.ndarray,
    sizes: np.ndarray,
    residuals: np.ndarray,
    *,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Return ``mean_t residual_t u_t`` with bounded float memory.

    Directly calling :func:`centered_directions` on a million-row design would
    allocate a ``T x n`` float matrix.  Instead use

    ``u_t = scale_t (z_t - |S_t|/n 1)``

    and accumulate the Boolean matrix-vector product in chunks.  The default
    chunk cap keeps the temporary float conversion near 16 MiB.
    """
    coalitions = np.asarray(coalitions, dtype=bool)
    sizes = np.asarray(sizes, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    if coalitions.ndim != 2:
        raise ValueError("coalitions must be two-dimensional")
    num_samples, num_players = coalitions.shape
    if sizes.shape != (num_samples,) or residuals.shape != (num_samples,):
        raise ValueError("sizes and residuals must contain one value per row")
    if num_samples == 0:
        raise ValueError("at least one coalition is required")
    if np.any(sizes <= 0) or np.any(sizes >= num_players):
        raise ValueError("all coalition sizes must be nonempty and nonfull")
    if chunk_rows is None:
        chunk_rows = max(
            1, (16 * 2**20) // (num_players * np.dtype(np.float64).itemsize)
        )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")

    total = np.zeros(num_players, dtype=np.float64)
    center_total = 0.0
    for start in range(0, num_samples, chunk_rows):
        stop = min(start + chunk_rows, num_samples)
        chunk_sizes = sizes[start:stop]
        scales = np.sqrt(
            num_players
            / (chunk_sizes * (num_players - chunk_sizes))
        )
        weights = residuals[start:stop] * scales
        rows = coalitions[start:stop].astype(np.float64)
        total += rows.T @ weights
        center_total += float(
            weights @ (chunk_sizes / num_players)
        )
    total -= center_total
    return total / num_samples


def estimate_coupled(
    design: CoalitionDesign,
    utilities: np.ndarray,
    boundary: OFABoundary,
    *,
    baseline: str = "linear",
) -> np.ndarray:
    """Estimate Shapley values with the coupled linear OFA identity."""
    if design.method not in {
        "iid",
        "frame_coupled",
        "frame_coupled_per_size",
        "orbit_coupled",
    }:
        raise ValueError(
            "coupled estimation requires an IID, frame-coupled, or "
            "orbit-coupled OFA design"
        )
    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.shape != (len(design.coalitions),):
        raise ValueError("utilities must contain one scalar per coalition")
    if design.coalitions.shape[1] != boundary.num_players:
        raise ValueError("design and boundary disagree on the player count")

    num_players = boundary.num_players
    _, _, normalizer = inner_size_distribution(num_players)
    residuals = utilities - _size_baseline(
        boundary, design.sizes, baseline
    )
    inner = normalizer * np.sqrt(num_players) * _weighted_direction_mean(
        design.coalitions, design.sizes, residuals
    )
    return shapley_boundary_vector(boundary) + inner


def estimate_stratified(
    design: CoalitionDesign,
    utilities: np.ndarray,
    boundary: OFABoundary,
    *,
    baseline: str = "linear",
) -> np.ndarray:
    """Estimate every fixed-size expectation separately."""
    if design.method not in {"frame_stratified", "cyclic_orbit_frame"}:
        raise ValueError(
            "stratified estimation requires a stratified or orbit design"
        )
    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.shape != (len(design.coalitions),):
        raise ValueError("utilities must contain one scalar per coalition")
    if design.coalitions.shape[1] != boundary.num_players:
        raise ValueError("design and boundary disagree on the player count")

    num_players = boundary.num_players
    expected_sizes, _, _ = inner_size_distribution(num_players)
    present = np.unique(design.sizes)
    if not np.array_equal(present, expected_sizes):
        raise ValueError("stratified estimation requires every inner size")

    directions = centered_directions(design.coalitions)
    residuals = utilities - _size_baseline(
        boundary, design.sizes, baseline
    )
    values = shapley_boundary_vector(boundary)
    for size in expected_sizes:
        take = design.sizes == size
        coefficient = np.sqrt(num_players / (size * (num_players - size)))
        values += coefficient * np.mean(
            residuals[take, None] * directions[take], axis=0
        )
    return values


def estimate_ratio_ofa(
    design: CoalitionDesign,
    utilities: np.ndarray,
    boundary: OFABoundary,
) -> np.ndarray:
    """Reproduce official Shapley OFA conditional-mean aggregation.

    This function is intended for 1-balanced designs such as complete cyclic
    orbits.  It rejects missing player/size strata rather than silently
    substituting zero as the research code does.
    """
    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.shape != (len(design.coalitions),):
        raise ValueError("utilities must contain one scalar per coalition")
    num_players = boundary.num_players
    if design.coalitions.shape[1] != num_players:
        raise ValueError("design and boundary disagree on the player count")
    expected_sizes, _, _ = inner_size_distribution(num_players)
    if not np.array_equal(np.unique(design.sizes), expected_sizes):
        raise ValueError("ratio OFA requires every inner size")

    values = shapley_boundary_vector(boundary)
    for size in expected_sizes:
        take = design.sizes == size
        rows = design.coalitions[take]
        slice_utilities = utilities[take]
        inclusion_counts = rows.sum(axis=0, dtype=np.int64)
        if not np.all(inclusion_counts == inclusion_counts[0]):
            raise ValueError(
                f"ratio OFA requires a 1-balanced design at size {size}"
            )
        exclusion_counts = len(rows) - inclusion_counts
        missing_players = np.flatnonzero(
            (inclusion_counts == 0) | (exclusion_counts == 0)
        )
        if len(missing_players):
            raise ValueError(
                "missing in/out observations for player "
                f"{int(missing_players[0])}, size {size}"
            )
        inclusion_sums = np.einsum(
            "ij,i->j", rows, slice_utilities, optimize=True
        )
        total = float(slice_utilities.sum())
        values += (
            inclusion_sums / inclusion_counts
            - (total - inclusion_sums) / exclusion_counts
        ) / num_players
    return values


def estimate_official_ratio_ofa(
    design: CoalitionDesign,
    utilities: np.ndarray,
    boundary: OFABoundary,
    *,
    missing: str = "zero",
) -> np.ndarray:
    """Reproduce official ``OFA_fixed`` Shapley ratio aggregation.

    The upstream implementation initializes every unobserved player/size
    conditional mean to zero.  ``missing="zero"`` reproduces that behavior;
    ``missing="raise"`` makes insufficient coverage explicit.
    """
    if missing not in {"zero", "raise"}:
        raise ValueError("missing must be 'zero' or 'raise'")
    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.shape != (len(design.coalitions),):
        raise ValueError("utilities must contain one scalar per coalition")
    num_players = boundary.num_players
    if design.coalitions.shape[1] != num_players:
        raise ValueError("design and boundary disagree on the player count")
    expected_sizes, _, _ = inner_size_distribution(num_players)

    values = shapley_boundary_vector(boundary)
    for size in expected_sizes:
        take = design.sizes == size
        rows = design.coalitions[take]
        slice_utilities = utilities[take]
        inclusion_counts = rows.sum(axis=0, dtype=np.int64)
        exclusion_counts = len(rows) - inclusion_counts
        has_positive = inclusion_counts > 0
        has_negative = exclusion_counts > 0
        missing_players = np.flatnonzero(~has_positive | ~has_negative)
        if missing == "raise" and len(missing_players):
            raise ValueError(
                "missing in/out observations for player "
                f"{int(missing_players[0])}, size {size}"
            )
        inclusion_sums = np.einsum(
            "ij,i->j", rows, slice_utilities, optimize=True
        )
        positive = np.divide(
            inclusion_sums,
            inclusion_counts,
            out=np.zeros(num_players, dtype=np.float64),
            where=has_positive,
        )
        negative = np.divide(
            float(slice_utilities.sum()) - inclusion_sums,
            exclusion_counts,
            out=np.zeros(num_players, dtype=np.float64),
            where=has_negative,
        )
        values += (positive - negative) / num_players
    return values


def _require_inside_ratio_coverage(
    design: CoalitionDesign, *, strict_balance: bool
) -> dict[str, int | bool]:
    """Validate ratio strata before any potentially expensive utility call."""
    num_players = design.coalitions.shape[1]
    expected_sizes, _, _ = inner_size_distribution(num_players)
    if not np.array_equal(np.unique(design.sizes), expected_sizes):
        raise ValueError("INSIDE ratio estimation requires every inner size")

    minimum_inclusion = np.iinfo(np.int64).max
    minimum_exclusion = np.iinfo(np.int64).max
    balanced_sizes = 0
    for size in expected_sizes:
        rows = design.coalitions[design.sizes == size]
        inclusion = rows.sum(axis=0, dtype=np.int64)
        exclusion = len(rows) - inclusion
        if strict_balance and not np.all(inclusion == inclusion[0]):
            raise ValueError(
                f"INSIDE-Orbit requires a 1-balanced design at size {size}"
            )
        balanced_sizes += int(np.all(inclusion == inclusion[0]))
        missing = np.flatnonzero((inclusion == 0) | (exclusion == 0))
        if len(missing):
            raise ValueError(
                "missing in/out observations for player "
                f"{int(missing[0])}, size {size}"
            )
        minimum_inclusion = min(minimum_inclusion, int(inclusion.min()))
        minimum_exclusion = min(minimum_exclusion, int(exclusion.min()))

    return {
        "all_player_size_strata_covered": True,
        "minimum_inclusion_count": int(minimum_inclusion),
        "minimum_exclusion_count": int(minimum_exclusion),
        "exactly_1_balanced_size_count": balanced_sizes,
        "total_inner_size_count": len(expected_sizes),
    }


class FrameOFAEstimator:
    """End-to-end Shapley estimator for INSIDE and legacy OFA variants.

    The two paper-facing modes are ``"inside_greedy"`` (per-size Greedy
    design plus covered official OFA ratio aggregation) and
    ``"inside_orbit"`` (complete cyclic orbits plus strict balanced ratio
    aggregation).  The older ``iid``, ``coupled``, ``stratified``, and
    ``orbit`` modes retain their historical design and estimator semantics for
    reproducibility.

    ``num_samples`` counts coalition rows except in ``orbit`` and
    ``inside_orbit`` modes, where it counts complete ``num_players``-row
    orbits.  When omitted, ``inside_greedy`` resolves ``candidate_pool`` and
    normalized ``mean_balance`` to the configured values 64 and 1/16,
    respectively.  ``inside_orbit`` resolves its candidate pool to the
    benchmark value 4; lower-level legacy modes keep candidate pool 32.
    """

    def __init__(
        self,
        num_players: int,
        num_samples: int,
        *,
        mode: str = "coupled",
        seed: int = 0,
        candidate_pool: int | None = None,
        baseline: str = "linear",
        mean_balance: float | None = None,
        mean_balance_mode: str = "normalized",
    ) -> None:
        valid_modes = {
            "iid",
            "coupled",
            "stratified",
            "orbit",
            "inside_greedy",
            "inside_orbit",
        }
        if mode not in valid_modes:
            raise ValueError(
                "mode must be 'inside_greedy', 'inside_orbit', 'iid', "
                "'coupled', 'stratified', or 'orbit'"
            )
        if num_players < 4:
            raise ValueError("at least 4 players are required")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if candidate_pool is None:
            if mode == "inside_greedy":
                candidate_pool = 64
            elif mode == "inside_orbit":
                candidate_pool = 4
            else:
                candidate_pool = 32
        if mean_balance is None:
            mean_balance = 1.0 / 16.0 if mode == "inside_greedy" else 1.0
        if candidate_pool < 1:
            raise ValueError("candidate_pool must be positive")
        self.num_players = num_players
        self.num_samples = num_samples
        if mode in {
            "stratified",
            "orbit",
            "inside_greedy",
            "inside_orbit",
        } and num_samples < num_players - 3:
            unit = (
                "orbits" if mode in {"orbit", "inside_orbit"} else "samples"
            )
            raise ValueError(
                f"{mode} mode needs at least n-3={num_players - 3} {unit}"
            )
        self.mode = mode
        self.seed = seed
        self.candidate_pool = candidate_pool
        if mean_balance < 0:
            raise ValueError("mean_balance must be nonnegative")
        self.mean_balance = mean_balance
        if mean_balance_mode not in {"normalized", "raw"}:
            raise ValueError(
                "mean_balance_mode must be 'normalized' or 'raw'"
            )
        self.mean_balance_mode = mean_balance_mode
        if baseline not in {"linear", "none"}:
            raise ValueError("baseline must be 'linear' or 'none'")
        self.baseline = baseline

    def design(self) -> CoalitionDesign:
        if self.mode == "inside_greedy":
            design = per_size_frame_coupled_design(
                self.num_players,
                self.num_samples,
                self.seed,
                self.candidate_pool,
                self.mean_balance,
                mean_balance_mode=self.mean_balance_mode,
            )
            coverage = _require_inside_ratio_coverage(
                design, strict_balance=False
            )
            design.diagnostics.update(
                {
                    "inside_algorithm": "INSIDE-Greedy",
                    "inside_estimator": (
                        "ofa_conditional_mean_ratio_missing_raise"
                    ),
                    "official_ratio_missing_policy": "raise",
                    "ratio_coverage": coverage,
                }
            )
            return design
        if self.mode == "inside_orbit":
            design = cyclic_orbit_frame_design(
                self.num_players,
                self.num_samples,
                self.seed,
                self.candidate_pool,
            )
            coverage = _require_inside_ratio_coverage(
                design, strict_balance=True
            )
            design.diagnostics.update(
                {
                    "inside_algorithm": "INSIDE-Orbit",
                    "inside_estimator": (
                        "ofa_conditional_mean_ratio_strict_balanced"
                    ),
                    "official_ratio_missing_policy": "raise",
                    "ratio_coverage": coverage,
                }
            )
            return design
        if self.mode == "iid":
            return iid_ofa_design(
                self.num_players, self.num_samples, self.seed
            )
        if self.mode == "coupled":
            return frame_coupled_design(
                self.num_players,
                self.num_samples,
                self.seed,
                self.candidate_pool,
                self.mean_balance,
                mean_balance_mode=self.mean_balance_mode,
            )
        if self.mode == "stratified":
            return stratified_frame_design(
                self.num_players,
                self.num_samples,
                self.seed,
                self.candidate_pool,
                self.mean_balance,
                mean_balance_mode=self.mean_balance_mode,
            )
        return cyclic_orbit_frame_design(
            self.num_players,
            self.num_samples,
            self.seed,
            self.candidate_pool,
        )

    def _estimate_values(
        self,
        design: CoalitionDesign,
        utilities: np.ndarray,
        boundary: OFABoundary,
    ) -> np.ndarray:
        if self.mode == "inside_greedy":
            return estimate_official_ratio_ofa(
                design, utilities, boundary, missing="raise"
            )
        if self.mode == "inside_orbit":
            return estimate_ratio_ofa(design, utilities, boundary)
        if self.mode in {"stratified", "orbit"}:
            return estimate_stratified(
                design,
                utilities,
                boundary,
                baseline=self.baseline,
            )
        return estimate_coupled(
            design,
            utilities,
            boundary,
            baseline=self.baseline,
        )

    def estimate(self, utility: Utility) -> EstimateResult:
        # The design is utility-agnostic.  Construct it first so any invalid
        # configuration fails before an expensive boundary evaluation.
        design = self.design()
        boundary = evaluate_boundary(utility, self.num_players)
        utilities = np.asarray(
            [utility(row.copy()) for row in design.coalitions],
            dtype=np.float64,
        )
        values = self._estimate_values(design, utilities, boundary)
        return EstimateResult(
            values=values,
            design=design,
            utilities=utilities,
            boundary=boundary,
            utility_evaluations=(
                2 * self.num_players
                + 2
                + len(design.coalitions)
            ),
        )

    def estimate_with_evaluator(
        self, evaluator: CoalitionBatchEvaluator
    ) -> EstimateResult:
        """Evaluate boundary and inner coalitions through a reusable batch API."""
        design = self.design()
        boundary_rows = boundary_coalitions(self.num_players)
        all_rows = np.concatenate(
            [boundary_rows, design.coalitions], axis=0
        )
        all_utilities = np.asarray(
            evaluator.evaluate(all_rows), dtype=np.float64
        )
        if all_utilities.shape != (len(all_rows),):
            raise ValueError(
                "evaluator must return one scalar per coalition"
            )
        boundary_count = len(boundary_rows)
        boundary = boundary_from_utilities(
            all_utilities[:boundary_count], self.num_players
        )
        utilities = all_utilities[boundary_count:]
        values = self._estimate_values(design, utilities, boundary)
        return EstimateResult(
            values=values,
            design=design,
            utilities=utilities,
            boundary=boundary,
            utility_evaluations=len(all_rows),
        )

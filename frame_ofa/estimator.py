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
    if design.method not in {"iid", "frame_coupled", "orbit_coupled"}:
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
        rows = design.coalitions[design.sizes == size]
        slice_utilities = utilities[design.sizes == size]
        inclusion_counts = rows.sum(axis=0)
        if not np.all(inclusion_counts == inclusion_counts[0]):
            raise ValueError(
                f"ratio OFA requires a 1-balanced design at size {size}"
            )
        for player in range(num_players):
            included = rows[:, player]
            if not np.any(included) or np.all(included):
                raise ValueError(
                    f"missing in/out observations for player {player}, size {size}"
                )
            values[player] += (
                slice_utilities[included].mean()
                - slice_utilities[~included].mean()
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
        for player in range(num_players):
            included = rows[:, player] if len(rows) else np.zeros(0, dtype=bool)
            has_positive = bool(np.any(included))
            has_negative = bool(np.any(~included)) if len(rows) else False
            if missing == "raise" and (
                not has_positive or not has_negative
            ):
                raise ValueError(
                    f"missing in/out observations for player {player}, "
                    f"size {size}"
                )
            positive = (
                float(slice_utilities[included].mean())
                if has_positive
                else 0.0
            )
            negative = (
                float(slice_utilities[~included].mean())
                if has_negative
                else 0.0
            )
            values[player] += (positive - negative) / num_players
    return values


class FrameOFAEstimator:
    """End-to-end Shapley estimator with IID or frame-coupled sampling."""

    def __init__(
        self,
        num_players: int,
        num_samples: int,
        *,
        mode: str = "coupled",
        seed: int = 0,
        candidate_pool: int = 32,
        baseline: str = "linear",
        mean_balance: float = 0.1,
    ) -> None:
        if mode not in {"iid", "coupled", "stratified", "orbit"}:
            raise ValueError(
                "mode must be 'iid', 'coupled', 'stratified', or 'orbit'"
            )
        if num_players < 4:
            raise ValueError("at least 4 players are required")
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if candidate_pool < 1:
            raise ValueError("candidate_pool must be positive")
        self.num_players = num_players
        self.num_samples = num_samples
        if mode in {"stratified", "orbit"} and num_samples < num_players - 3:
            unit = "orbits" if mode == "orbit" else "samples"
            raise ValueError(
                f"{mode} mode needs at least n-3={num_players - 3} {unit}"
            )
        self.mode = mode
        self.seed = seed
        self.candidate_pool = candidate_pool
        if mean_balance < 0:
            raise ValueError("mean_balance must be nonnegative")
        self.mean_balance = mean_balance
        if baseline not in {"linear", "none"}:
            raise ValueError("baseline must be 'linear' or 'none'")
        self.baseline = baseline

    def design(self) -> CoalitionDesign:
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
            )
        if self.mode == "stratified":
            return stratified_frame_design(
                self.num_players,
                self.num_samples,
                self.seed,
                self.candidate_pool,
                self.mean_balance,
            )
        return cyclic_orbit_frame_design(
            self.num_players,
            self.num_samples,
            self.seed,
            self.candidate_pool,
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
        if self.mode in {"stratified", "orbit"}:
            values = estimate_stratified(
                design,
                utilities,
                boundary,
                baseline=self.baseline,
            )
        else:
            values = estimate_coupled(
                design,
                utilities,
                boundary,
                baseline=self.baseline,
            )
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
        if self.mode in {"stratified", "orbit"}:
            values = estimate_stratified(
                design,
                utilities,
                boundary,
                baseline=self.baseline,
            )
        else:
            values = estimate_coupled(
                design,
                utilities,
                boundary,
                baseline=self.baseline,
            )
        return EstimateResult(
            values=values,
            design=design,
            utilities=utilities,
            boundary=boundary,
            utility_evaluations=len(all_rows),
        )

"""Geometry-aware sampling for Shapley-specific OFA.

The implementation in this package is a clean-room extension built from the
mathematical description in Li and Yu (NeurIPS 2024).  The downloaded upstream
research code is kept separately in ``ofa_upstream`` for comparison.
"""

from .design import (
    CoalitionDesign,
    cyclic_orbit_frame_design,
    frame_coupled_design,
    frame_diagnostics,
    iid_ofa_design,
    inner_size_distribution,
    orbit_coupled_frame_design,
    stratified_frame_design,
)
from .estimator import (
    EstimateResult,
    FrameOFAEstimator,
    OFABoundary,
    boundary_coalitions,
    boundary_from_utilities,
    evaluate_boundary,
    estimate_coupled,
    estimate_official_ratio_ofa,
    estimate_ratio_ofa,
    estimate_stratified,
    shapley_boundary_vector,
)
from .geometry import (
    centered_directions,
    efficiency_projector,
    inner_frame_operator,
    inner_frame_target,
)
from .parallel import GameEvaluator, evaluate_game_coalitions
from .upstream_adapter import estimate_upstream_game

__all__ = [
    "CoalitionDesign",
    "EstimateResult",
    "FrameOFAEstimator",
    "GameEvaluator",
    "OFABoundary",
    "boundary_coalitions",
    "boundary_from_utilities",
    "centered_directions",
    "cyclic_orbit_frame_design",
    "efficiency_projector",
    "estimate_coupled",
    "estimate_official_ratio_ofa",
    "estimate_ratio_ofa",
    "estimate_stratified",
    "estimate_upstream_game",
    "evaluate_boundary",
    "evaluate_game_coalitions",
    "frame_coupled_design",
    "frame_diagnostics",
    "iid_ofa_design",
    "inner_frame_operator",
    "inner_frame_target",
    "inner_size_distribution",
    "orbit_coupled_frame_design",
    "shapley_boundary_vector",
    "stratified_frame_design",
]

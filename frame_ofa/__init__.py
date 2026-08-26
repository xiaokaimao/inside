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
    paired_frame_scope_designs,
    per_size_frame_coupled_design,
    stratified_frame_design,
)
from .airport import evaluate_airport, exact_airport_shapley
from .complementary import (
    BasicCCDiagnostics,
    BasicCCResult,
    aggregate_basic_cc_samples,
    estimate_basic_cc,
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
from .differential import (
    DifferentialCoverageError,
    DiffDiagnostics,
    DiffResult,
    estimate_diff,
)
from .group_testing import (
    GroupTestingDiagnostics,
    GroupTestingResult,
    estimate_group_testing,
)
from .regression_baselines import (
    RegressionBaselineDiagnostics,
    RegressionBaselineResult,
    estimate_gels_shapley,
    estimate_kernel_shap,
)
from .stratified_differential import (
    SDiffCoverageError,
    SDiffDiagnostics,
    SDiffResult,
    estimate_sdiff,
)
from .tmc import (
    FullPermutationDiagnostics,
    FullPermutationResult,
    TMCShapleyDiagnostics,
    TMCShapleyResult,
    estimate_full_permutation_mc,
    estimate_tmc_shapley,
)
from .traditional_mc import (
    StratifiedMarginalDiagnostics,
    StratifiedMarginalResult,
    estimate_stratified_marginal_mc,
)
from .upstream_adapter import estimate_upstream_game
from .weighted_voting import (
    evaluate_weighted_voting,
    exact_shapley_shubik,
)

__all__ = [
    "CoalitionDesign",
    "BasicCCDiagnostics",
    "BasicCCResult",
    "EstimateResult",
    "FrameOFAEstimator",
    "DifferentialCoverageError",
    "DiffDiagnostics",
    "DiffResult",
    "FullPermutationDiagnostics",
    "FullPermutationResult",
    "TMCShapleyDiagnostics",
    "TMCShapleyResult",
    "GameEvaluator",
    "GroupTestingDiagnostics",
    "GroupTestingResult",
    "RegressionBaselineDiagnostics",
    "RegressionBaselineResult",
    "OFABoundary",
    "SDiffCoverageError",
    "SDiffDiagnostics",
    "SDiffResult",
    "StratifiedMarginalDiagnostics",
    "StratifiedMarginalResult",
    "boundary_coalitions",
    "boundary_from_utilities",
    "centered_directions",
    "cyclic_orbit_frame_design",
    "efficiency_projector",
    "aggregate_basic_cc_samples",
    "estimate_basic_cc",
    "estimate_coupled",
    "estimate_diff",
    "estimate_full_permutation_mc",
    "estimate_tmc_shapley",
    "estimate_gels_shapley",
    "estimate_group_testing",
    "estimate_kernel_shap",
    "estimate_official_ratio_ofa",
    "estimate_ratio_ofa",
    "estimate_sdiff",
    "estimate_stratified",
    "estimate_stratified_marginal_mc",
    "estimate_upstream_game",
    "evaluate_airport",
    "evaluate_weighted_voting",
    "evaluate_boundary",
    "evaluate_game_coalitions",
    "frame_coupled_design",
    "frame_diagnostics",
    "exact_airport_shapley",
    "exact_shapley_shubik",
    "iid_ofa_design",
    "inner_frame_operator",
    "inner_frame_target",
    "inner_size_distribution",
    "orbit_coupled_frame_design",
    "paired_frame_scope_designs",
    "per_size_frame_coupled_design",
    "shapley_boundary_vector",
    "stratified_frame_design",
]

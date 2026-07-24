"""Virtual-population construction: plausible-patient cloud + prevalence weighting."""

from qsp_inference.vpop.diagnostics import (
    CoreResult,
    misspecification_ratio,
    perfect_model_null,
    conflict_ranking,
    duplicate_observables,
    ess_scaling,
    greedy_core,
    paired_effect_sizes,
    self_target_control,
)
from qsp_inference.vpop.weighting import (
    VPopResult,
    build_quantile_constraints,
    fit_prevalence_weights,
)
from qsp_inference.vpop.eigenbasis import (
    PriorMetricEigenbasis,
    fit_local_jacobian,
    whiten_sensitivity_rows,
    sensitivity_gram,
    prior_covariance,
    prior_metric_eigenbasis,
)
from qsp_inference.vpop.proposal import (
    EigenbasisPopulation,
    widen_on_identified,
    reachable_accept_fn,
)

__all__ = [
    "VPopResult",
    "build_quantile_constraints",
    "fit_prevalence_weights",
    "CoreResult",
    "conflict_ranking",
    "misspecification_ratio",
    "perfect_model_null",
    "duplicate_observables",
    "ess_scaling",
    "greedy_core",
    "paired_effect_sizes",
    "self_target_control",
    "PriorMetricEigenbasis",
    "fit_local_jacobian",
    "whiten_sensitivity_rows",
    "sensitivity_gram",
    "prior_covariance",
    "prior_metric_eigenbasis",
    "EigenbasisPopulation",
    "widen_on_identified",
    "reachable_accept_fn",
]

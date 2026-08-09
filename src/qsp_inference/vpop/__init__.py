"""Virtual-population inference.

The fit lives in :mod:`qsp_inference.vpop.fit` and is driven by pdac-build's
``workflows/vpop_fit.py``. Nothing on that path is re-exported here: it needs
jax and numpyro, and a package-level import would make an optional dependency
load on every ``import qsp_inference.vpop``. Import it directly.

What this namespace still carries is the fixed-cloud route: prevalence weighting
over a plausible-patient cloud, its reachability diagnostics, and the prior-metric
eigenbasis the hierarchical NPE runner uses.
"""

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

"""The fixed-cloud route, kept for the callers that still run it.

Prevalence weighting over a plausible-patient cloud (Allen 2016), its
reachability diagnostics, and the prior-metric eigenbasis the hierarchical NPE
runner uses. None of it is in ``docs/model-draft.tex``: it estimates a
population by reweighting a cloud that was drawn once, where the draft's model
fits the law the cloud is drawn from. The two answer different questions and
share no code.

It lived under :mod:`qsp_inference.vpop` and was moved out so that namespace is
the draft's fit and nothing else. Consumers in pdac-build: ``run_vpop.py``,
``plot_vpop_marginals.py``, ``hierarchical_runner.py``,
``scripts/compute_eigenbasis.py``, ``scripts/vpop_solver_control.py``.
"""

from qsp_inference.legacy.diagnostics import (
    CoreResult,
    conflict_ranking,
    duplicate_observables,
    ess_scaling,
    greedy_core,
    misspecification_ratio,
    paired_effect_sizes,
    perfect_model_null,
    self_target_control,
)
from qsp_inference.legacy.eigenbasis import (
    PriorMetricEigenbasis,
    fit_local_jacobian,
    prior_covariance,
    prior_metric_eigenbasis,
    sensitivity_gram,
    whiten_sensitivity_rows,
)
from qsp_inference.legacy.proposal import (
    EigenbasisPopulation,
    reachable_accept_fn,
    widen_on_identified,
)
from qsp_inference.legacy.weighting import (
    VPopResult,
    build_quantile_constraints,
    fit_prevalence_weights,
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

"""Virtual-population construction: plausible-patient cloud + prevalence weighting,
plus the closed-form summary likelihood and its NUTS fit (docs ch. 4b).

The ch. 4b modules need torch (and pyro-ppl to sample), so they are imported
under try/except like the rest of the package's optional-dep surface; import
``qsp_inference.vpop.summary_likelihood`` directly for a clear ImportError.
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

try:  # torch-only: the ch. 4b summary likelihood and its fit
    from qsp_inference.vpop.summary_likelihood import (
        StudyBlock,
        SummaryLikelihood,
        TargetAnchor,
        anchor_covariance,
        build_study_blocks,
        bvn_cdf,
        normal_score_correlation,
    )
    from qsp_inference.vpop.population_fit import (
        PopulationFit,
        PopulationPosterior,
        PopulationPrior,
        run_nuts,
    )

    _CH4B = [
        "StudyBlock",
        "SummaryLikelihood",
        "TargetAnchor",
        "anchor_covariance",
        "build_study_blocks",
        "bvn_cdf",
        "normal_score_correlation",
        "PopulationFit",
        "PopulationPosterior",
        "PopulationPrior",
        "run_nuts",
    ]
except ImportError:  # pragma: no cover - torch is an optional extra
    _CH4B = []

__all__ = _CH4B + [
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

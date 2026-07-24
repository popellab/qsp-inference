"""Prior generation and loading utilities.

Split by dependency weight, the same way :mod:`qsp_inference.inference` is.

The spec and pool objects import cleanly without torch, and that is deliberate
rather than incidental: a simulator asks this package which theta to run, and a
simulation host has no reason to carry a deep-learning stack. The CSV-only
sampler is pure numpy and works there. The composite copula prior does need
torch, and raises a clear ImportError from ``copula_prior`` if it is missing,
which is the honest failure: it is the *density* that needs the dependency, not
the idea of a pool.
"""

# Lightweight — no torch required. The single owner of (pi, pi_tilde) and of
# pool identity. Prefer these over calling the loaders directly: composing a
# prior by hand in a caller is what let the training cloud and its density
# drift apart.
from qsp_inference.priors.inference_prior import (
    PriorSpec,
    PriorPair,
    CsvIndependentPrior,
    build_prior_pair,
)
from qsp_inference.priors.theta_pool import (
    ThetaPoolSpec,
    get_theta_pool,
    theta_for_indices,
)

# Heavy — need torch. Guarded so a partial install (a simulation host, or CI
# without the sbi extra) can still import the objects above. If an optional dep
# is missing the symbols it provides are omitted from the package namespace;
# import the submodule directly for a clear ImportError message.
try:
    from qsp_inference.priors.load_sbi_priors import (
        load_prior,
        get_param_names,
        transform_lognormal_prior_to_normal,
    )
    from qsp_inference.priors.truncated_distributions import TruncatedDistribution
    from qsp_inference.priors.copula_prior import (
        GaussianCopulaPrior,
        load_copula_prior,
        load_copula_prior_log,
        load_composite_prior_log,
        compose_overlay_prior,
        load_overlay_prior_log,
        apply_derived_priors,
        load_derived_specs,
        temper_prior,
    )
except ImportError:  # pragma: no cover - exercised by torch-free installs
    pass

__all__ = [
    # Spec + pool (no torch)
    "PriorSpec",
    "PriorPair",
    "CsvIndependentPrior",
    "build_prior_pair",
    "ThetaPoolSpec",
    "get_theta_pool",
    "theta_for_indices",
    # Loaders and distributions (torch)
    "load_prior",
    "get_param_names",
    "transform_lognormal_prior_to_normal",
    "TruncatedDistribution",
    "GaussianCopulaPrior",
    "load_copula_prior",
    "load_copula_prior_log",
    "load_composite_prior_log",
    "compose_overlay_prior",
    "load_overlay_prior_log",
    "apply_derived_priors",
    "load_derived_specs",
    "temper_prior",
]

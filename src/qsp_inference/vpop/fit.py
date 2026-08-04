"""eq:pop through eq:post as one NumPyro model.

De-globalised from ``examples/toy_population_fit.py:population_model``, which
holds the settled decisions this keeps: the ``s``/``b_1`` alias is reported and
not reparameterised, ``Z`` is sampled in its own basis rather than orthonormalised
(that changes the prior), and the flat fit pins ``omega = omega_0`` rather than
zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = ["PopulationPrior", "Problem", "build_omega",
           "population_model"]


@dataclass(frozen=True)
class PopulationPrior:
    """Everything eq:mu through eq:auxprior needs, for one problem."""

    mu_0: jnp.ndarray            # (P,) prior centre, eq:mu
    L_sigma_1: jnp.ndarray       # (P, P) chol of the stage-1 covariance
    omega_0: jnp.ndarray         # (P,) prior widths
    measured: Tuple[int, ...] = ()   # j in M, the widths eq:omegameas applies to

    tau_s: float = 0.3           # eq:omegaassumed, the global level
    tau_u: float = 0.3           # eq:omegaassumed, the pattern across parameters
    tau_omega_measured: float = 0.2

    sigma_a: float = 0.5         # eq:abprior
    sigma_b: float = 0.5
    tau_beta: float = 0.15       # eq:betaprior, fixed rather than estimated
    n_beta: int = 0              # |S|, the declared subset carrying a free beta
    dim_z: int = 1               # columns of Z

    log_R_0: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))
    sigma_R: jnp.ndarray = field(default_factory=lambda: jnp.zeros(0))

    @property
    def n_params(self) -> int:
        return int(jnp.asarray(self.omega_0).shape[0])

    @property
    def assumed(self) -> Tuple[int, ...]:
        return tuple(j for j in range(self.n_params) if j not in set(self.measured))

    @property
    def n_aux(self) -> int:
        return int(jnp.asarray(self.log_R_0).shape[0])

    def __post_init__(self):
        if self.n_aux != int(jnp.asarray(self.sigma_R).shape[0]):
            raise ValueError("log_R_0 and sigma_R must have the same length")
        if not self.assumed:
            raise ValueError("every width is measured, so s and u have nothing to do")
        if self.n_beta == 1:
            raise ValueError(
                "eq:betaprior centres beta, so one free species gives beta = 0 "
                "identically: a parameter with no effect and a flat direction. "
                "Declare the whole shared-denominator set, or none of it."
            )


def build_omega(s, u_raw, log_omega_measured, prior: PopulationPrior):
    """``omega`` from its measured and assumed halves. eq:omegameas, eq:omegaassumed.

    ``u`` is centred so the global level lives in ``s`` alone.
    """
    log_omega = jnp.log(jnp.asarray(prior.omega_0))
    assumed = np.asarray(prior.assumed)
    log_omega = log_omega.at[assumed].add(s + (u_raw - jnp.mean(u_raw)))
    if prior.measured:
        log_omega = log_omega.at[np.asarray(prior.measured)].set(log_omega_measured)
    return jnp.exp(log_omega)


@dataclass(frozen=True)
class Problem:
    """The fixed side of the fit: everything ``tau_all`` needs that is not ``phi``."""

    mech: object                              # vpop.predict.Mechanism
    plans: Sequence[object]                   # vpop.resampling.BlockPlan
    specs_by_cohort: Mapping[str, Sequence[object]]
    refs: jnp.ndarray                         # c_r, one level per readout
    designs: Optional[Mapping[str, jnp.ndarray]] = None
    elig_fn: Optional[object] = None
    elig_at: Optional[Mapping[str, str]] = None
    mass_table: Optional[Mapping] = None

    def block_name(self, plan) -> str:
        return "+".join(plan.cohort_ids)


def population_model(prior: PopulationPrior, problem: Problem, V_chol,
                     observed=None, *, flat: bool = False, row_masks=None):
    """eq:pop through eq:post. ``observed=None`` draws from the prior predictive.

    ``flat`` is eq:phiflat: pin ``omega = omega_0`` and drop the spread terms.
    ``row_masks`` selects rows per block, which is how the width rows are held out.
    """
    import numpyro
    import numpyro.distributions as dist

    from qsp_inference.vpop.predict import tau_all

    P = prior.n_params
    mu_raw = numpyro.sample("mu_raw", dist.Normal(0.0, 1.0).expand([P]).to_event(1))
    mu = numpyro.deterministic("mu", prior.mu_0 + prior.L_sigma_1 @ mu_raw)

    if flat:
        # omega_0 and not zero: a point mass makes eq:V return zero, turns w^(c)
        # into a switch on the whole cohort, and collapses every location
        # functional onto one number.
        omega = numpyro.deterministic("omega", jnp.asarray(prior.omega_0))
    else:
        # s and b_1 are aliased in principle. Report the split; do not
        # reparameterise around it, and do not orthonormalise Z to avoid it:
        # iid on an orthonormal basis is a different prior from iid on a.
        s = numpyro.sample("s", dist.Normal(0.0, prior.tau_s))
        u_raw = numpyro.sample(
            "u_raw",
            dist.Normal(0.0, prior.tau_u).expand([len(prior.assumed)]).to_event(1))
        if prior.measured:
            idx = np.asarray(prior.measured)
            log_omega_measured = numpyro.sample(
                "log_omega_measured",
                dist.Normal(jnp.log(jnp.asarray(prior.omega_0)[idx]),
                            prior.tau_omega_measured).to_event(1))
        else:
            log_omega_measured = jnp.zeros(0)
        omega = numpyro.deterministic(
            "omega", build_omega(s, u_raw, log_omega_measured, prior))

    a = numpyro.sample("a", dist.Normal(0.0, prior.sigma_a)
                       .expand([prior.dim_z]).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, prior.sigma_b)
                       .expand([prior.dim_z]).to_event(1))

    if prior.n_beta:
        beta_raw = numpyro.sample(
            "beta_raw", dist.Normal(0.0, 1.0).expand([prior.n_beta]).to_event(1))
        beta_free = numpyro.deterministic(
            "beta_free", prior.tau_beta * (beta_raw - jnp.mean(beta_raw)))
    else:
        beta_free = jnp.zeros(0)

    if prior.n_aux:
        log_R = numpyro.sample(
            "log_R", dist.Normal(jnp.asarray(prior.log_R_0),
                                 jnp.asarray(prior.sigma_R)).to_event(1))
    else:
        log_R = None

    taus = tau_all(mu, omega, a, b, beta_free, problem.plans,
                   problem.specs_by_cohort, problem.refs, problem.mech,
                   log_R=log_R, designs=problem.designs,
                   mass_table=problem.mass_table,
                   elig_fn=problem.elig_fn, elig_at=problem.elig_at)

    for i, (plan, tau_B) in enumerate(zip(problem.plans, taus)):
        if row_masks is not None:
            tau_B = tau_B[np.asarray(row_masks[i])]
        numpyro.sample(
            f"T_{problem.block_name(plan)}",
            dist.MultivariateNormal(tau_B, scale_tril=V_chol[i]),
            obs=None if observed is None else observed[i])

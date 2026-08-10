"""eq:pop through eq:post as one NumPyro model.

Two settled decisions this keeps: the ``s``/``b_1`` alias is reported and not
reparameterised, and ``Z`` is sampled in its own basis rather than
orthonormalised, because iid on an orthonormal basis is a different prior.

eq:phiflat, the flat fit, is not here. It was carried as a ``flat=True`` branch
through this model, the metric and the conditioning report, plus row masks and a
``subset_V`` to hold the width rows out, and no driver ever ran it. Reinstating
it means writing the driver stage first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = ["PopulationPrior", "Problem", "build_omega", "site_spec",
           "phi_from_sites", "population_model"]


@dataclass(frozen=True)
class PopulationPrior:
    """Everything eq:mu through eq:auxprior needs, for one problem.

    **No field carries a default, deliberately.** Every one of these is a claim:
    ``tau_s`` says how far the spreads may move, ``n_beta`` says whether a
    mechanism discrepancy exists, ``pin_discrepancy`` says which model is being
    fitted. A default would let a caller assert one without writing it down, and
    a rubric number asserted by a dataclass is indistinguishable in the posterior
    from one somebody chose. The project states them; this class stores them.

    eq:omegameas is not here. It put a prior on ``log omega_j`` for the
    parameters whose between-patient spread a source had measured, and no source
    has: the set was empty on every corpus this has run, and its companion
    eq:omegashrink was never implemented at all. Its absence is what makes the
    ``s``/``b_1`` alias unconditional rather than merely unbroken, since a
    measured width is the only thing that would have separated them.
    """

    mu_0: jnp.ndarray            # (P,) prior centre, eq:mu
    L_sigma_1: jnp.ndarray       # (P, P) chol of the stage-1 covariance
    omega_0: jnp.ndarray         # (P,) prior widths

    tau_s: float                 # eq:omegaassumed, the global level
    tau_u: float                 # eq:omegaassumed, the pattern. See population_model.

    sigma_a: float               # eq:abprior
    sigma_b: float
    tau_beta: float              # eq:betaprior, fixed rather than estimated
    n_beta: int                  # |S|, the declared subset carrying a free beta
    dim_z: int                   # columns of Z

    log_R_0: jnp.ndarray         # eq:auxprior; zeros(0) declares no auxiliaries
    sigma_R: jnp.ndarray

    # The falsifiable baseline. With eq:disc off, a mismatch has nowhere to hide
    # and shows up as residual structure that can be read; with it on, 18 free
    # parameters can absorb most of one, so a good fit says little. The cost is
    # that real assay bias then lands on mu, so this configuration diagnoses and
    # does not ship. False is not the neutral choice it looks like: it turns the
    # discrepancy layer on, which is a modelling decision, so it is stated too.
    #
    pin_discrepancy: bool           # a = b = 0
    pin_aux: bool                   # log R at its prior centre

    # Columns of b held at zero while the rest of the measurement map stays free.
    # The intercept is the one this exists for. b_1 and s load identically on
    # every width row, so only their sum is identified, and the posterior splits
    # it in proportion to prior variance. Whatever that ratio is, nothing in the
    # data chose it, and b_1's share is discarded with the measurement map while
    # s's reaches eq:vpop. The pdac rubric is sigma_b 0.29 against tau_s 0.30, so
    # the split is near even at 48/52; it was 73/27 against s while sigma_b was
    # 0.5, which is what made this flag worth having.
    #
    # Pinning the intercept asserts there is no GLOBAL assay stretch, so a
    # measured width excess is population width. That is a claim a reader can
    # argue with, in place of an accident of two numbers picked separately. The
    # other columns stay free, so per-assay stretch is still absorbed: this
    # removes the one direction of b that is degenerate with s, not the layer.
    #
    # There is no equivalent for a. It has no twin in the population block, so
    # pinning a column of it would assert something the alias argument does not
    # support.
    pin_b_columns: Tuple[int, ...]

    # Parameters whose width is held at omega_0 exactly: outside s, outside u.
    #
    # How far a width may move is a continuous question tau_u already answers,
    # so this is NOT a second opinion on that. It is for a parameter that is not
    # a population width in the first place. initial_tumour_diameter is the one
    # this exists for: it is the stopping rule, the diameter evolve_to_diagnosis
    # integrates each patient to before anything is read out, so it sets WHEN a
    # patient is observed rather than how they behave. It also modulates its own
    # censoring, since a larger diameter takes longer to reach and is likelier to
    # be rejected as too slow.
    #
    # eq:pop carries no truncation term, so the fit is handed a pool conditioned
    # on surviving that gate and is not told. The cheapest way to recover the
    # spread the censoring removed is to widen the parameter that moves every
    # patient's readout time, which is what an unpinned fit does: x3.55 on its own
    # width while every other parameter contracts to x0.43-0.50.
    #
    # Pinning does not fix that. It denies the misfit its cheapest absorber and
    # makes it surface somewhere else, which is the point -- the artifact becomes
    # visible instead of being quietly priced into a width. The fix is eq:elig's
    # log Z(mu, omega), and when that lands this pin is what tests it: release it
    # and see whether the width stays put.
    fix_omega: Tuple[int, ...] = ()

    # Parameters whose centre is held at mu_0 exactly: outside mu_raw entirely.
    #
    # The companion to fix_omega, and for the same parameter, for a reason that
    # is stronger on mu than on omega. initial_tumour_diameter is the gate: it
    # is the stopping rule, so it is a statement about WHEN a patient was
    # observed, and a cohort that reports its readouts has already been measured
    # at whatever size it presented at. The distribution of that size is
    # clinical epidemiology, an input to the virtual population, not a quantity
    # the immune corpus is entitled to relitigate. Held here, it exists only to
    # spread virtual patients over disease duration, which is its job.
    #
    # Left free it is the corpus's cheapest absorber, because it is the one
    # parameter that moves every readout at once: it sets each patient's readout
    # time, so it has maximum leverage and no observable anchors it. Nothing in
    # the pdac corpus measures tumour size. An unpinned fit made it the 4th most
    # identified parameter of 271, pulling the centre from 3.2cm to 2.0cm with a
    # 95% interval excluding the prior centre, on evidence that is entirely
    # indirect.
    #
    # Holding mu at mu_0 is conditioning Sigma_1 on that coordinate, not just
    # dropping it, and the two agree only when the held row of Sigma_1 has no
    # off-diagonal. __post_init__ checks that rather than assuming it: the pdac
    # copula leaves this parameter uncorrelated with all 270 others, and a
    # corpus where that stops being true should fail loudly.
    fix_mu: Tuple[int, ...] = ()

    @property
    def n_params(self) -> int:
        return int(jnp.asarray(self.omega_0).shape[0])

    @property
    def free_omega(self) -> Tuple[int, ...]:
        """Parameters whose width the model samples. Declaration order."""
        fixed = set(self.fix_omega)
        return tuple(j for j in range(self.n_params) if j not in fixed)

    @property
    def free_mu(self) -> Tuple[int, ...]:
        """Parameters whose centre the model samples. Declaration order."""
        fixed = set(self.fix_mu)
        return tuple(j for j in range(self.n_params) if j not in fixed)

    @property
    def free_b_columns(self) -> Tuple[int, ...]:
        """Columns of b the model samples. Declaration order, not the pinned set."""
        pinned = set(self.pin_b_columns)
        return tuple(j for j in range(self.dim_z) if j not in pinned)

    @property
    def n_aux(self) -> int:
        return int(jnp.asarray(self.log_R_0).shape[0])

    def __post_init__(self):
        if self.n_aux != int(jnp.asarray(self.sigma_R).shape[0]):
            raise ValueError("log_R_0 and sigma_R must have the same length")
        bad = [j for j in self.pin_b_columns if not 0 <= j < self.dim_z]
        if bad:
            raise ValueError(
                f"pin_b_columns {bad} are not columns of Z, which has "
                f"{self.dim_z}"
            )
        if len(set(self.pin_b_columns)) != len(self.pin_b_columns):
            raise ValueError("pin_b_columns repeats a column")
        bad = [j for j in self.fix_omega if not 0 <= j < self.n_params]
        if bad:
            raise ValueError(
                f"fix_omega {bad} are not parameters; there are {self.n_params}"
            )
        if len(set(self.fix_omega)) != len(self.fix_omega):
            raise ValueError("fix_omega repeats a parameter")
        if len(self.fix_omega) == self.n_params:
            raise ValueError(
                "every width is held at omega_0, which leaves s and u with "
                "nothing to act on. Hold the ones that are not population "
                "widths, not all of them."
            )
        bad = [j for j in self.fix_mu if not 0 <= j < self.n_params]
        if bad:
            raise ValueError(
                f"fix_mu {bad} are not parameters; there are {self.n_params}"
            )
        if len(set(self.fix_mu)) != len(self.fix_mu):
            raise ValueError("fix_mu repeats a parameter")
        if len(self.fix_mu) == self.n_params:
            raise ValueError(
                "every centre is held at mu_0, which leaves nothing for the "
                "corpus to move. Hold the parameters that are not population "
                "centres, not all of them."
            )
        if self.fix_mu:
            # Dropping mu_raw[j] equals conditioning mu on mu_j = mu_0[j] only
            # when Sigma_1's row j is diagonal. See the field.
            L = np.asarray(self.L_sigma_1)
            C = L @ L.T
            for j in self.fix_mu:
                off = np.abs(np.delete(C[j], j))
                if off.max(initial=0.0) > 1e-10 * abs(C[j, j]):
                    raise ValueError(
                        f"fix_mu holds parameter {j} at mu_0, but Sigma_1 "
                        f"correlates it with others (largest off-diagonal "
                        f"{off.max():.3g} against variance {C[j, j]:.3g}). "
                        "Holding it would silently discard that correlation "
                        "rather than condition on it."
                    )
        if self.pin_discrepancy and self.pin_b_columns:
            raise ValueError(
                "pin_discrepancy already holds every column of b at zero, so "
                "pin_b_columns would assert the same thing twice. Pass one or "
                "the other."
            )
        if len(self.pin_b_columns) == self.dim_z:
            raise ValueError(
                "every column of b is pinned, which is pin_discrepancy for b "
                "written the long way. Say so with pin_discrepancy, or leave a "
                "column free."
            )
        if self.n_beta == 1:
            raise ValueError(
                "eq:betaprior centres beta, so one free species gives beta = 0 "
                "identically: a parameter with no effect and a flat direction. "
                "Declare the whole shared-denominator set, or none of it."
            )


def build_omega(s, u_raw, prior: PopulationPrior):
    """eq:omegaassumed. ``u`` is centred, so the global level lives in ``s`` alone.

    ``fix_omega`` parameters come back at ``omega_0``: they are held out of the
    centring as well as out of ``s``, so neither the level nor the pattern is
    estimated from a width that is not a population width. See the field.
    """
    omega_0 = jnp.asarray(prior.omega_0)
    if not prior.fix_omega:
        return omega_0 * jnp.exp(s + (u_raw - jnp.mean(u_raw)))
    free = np.zeros(prior.n_params, dtype=bool)
    free[np.asarray(prior.free_omega)] = True
    free = jnp.asarray(free)
    # Centre over the free entries only. Averaging in the held ones would put
    # their zeros into the mean and move the level s carries.
    centred = u_raw - jnp.sum(jnp.where(free, u_raw, 0.0)) / jnp.sum(free)
    return jnp.where(free, omega_0 * jnp.exp(s + centred), omega_0)


def site_spec(prior: "PopulationPrior"):
    """``[(name, zero-value, prior sd)]`` for every site the model samples.

    The single statement of what the latent space IS. A mass matrix, a Laplace
    metric and a conditioning report all need the site list, their shapes and
    their prior widths, and each one derived separately would be a copy that can
    disagree with the model without any symptom: numpyro accepts a mass matrix
    whose blocks belong to different parameters than it thinks.

    Order is declaration order, not numpyro's. Callers that need numpyro's own
    packing sort by name themselves, where it is visible.
    """
    # Renamed when a centre is held, for the same reason u_free is.
    if prior.fix_mu:
        out = [("mu_free", jnp.zeros(len(prior.free_mu)), 1.0)]
    else:
        out = [("mu_raw", jnp.zeros(prior.n_params), 1.0)]
    out.append(("s", jnp.zeros(()), prior.tau_s))
    # Renamed when a width is held, for the reason pin_b_columns is: a site whose
    # length changes with configuration under one name is what a mass matrix
    # cannot notice.
    if prior.fix_omega:
        out.append(("u_free", jnp.zeros(len(prior.free_omega)), prior.tau_u))
    else:
        out.append(("u_raw", jnp.zeros(prior.n_params), prior.tau_u))
    if not prior.pin_discrepancy:
        out.append(("a", jnp.zeros(prior.dim_z), prior.sigma_a))
        # Renamed when a column is pinned, rather than kept as "b" at a smaller
        # shape. A site whose length changes with configuration under one name is
        # exactly what a mass matrix cannot notice.
        if prior.pin_b_columns:
            out.append(("b_free", jnp.zeros(len(prior.free_b_columns)),
                        prior.sigma_b))
        else:
            out.append(("b", jnp.zeros(prior.dim_z), prior.sigma_b))
    if prior.n_beta:
        out.append(("beta_raw", jnp.zeros(prior.n_beta), 1.0))
    if prior.n_aux and not prior.pin_aux:
        out.append(("log_R", jnp.asarray(prior.log_R_0),
                    np.asarray(prior.sigma_R)))
    return out


def phi_from_sites(sites: Mapping[str, jnp.ndarray], prior: "PopulationPrior"):
    """``(mu, omega, a, b, beta_free, log_R)`` from the sampled sites.

    The deterministic half of :func:`population_model`, which calls it. Anything
    that has to differentiate the model's map without sampling it goes through
    here rather than rebuilding the map, so the two cannot drift apart.
    """
    if prior.fix_mu:
        mu_raw = jnp.zeros(prior.n_params).at[
            np.asarray(prior.free_mu)].set(sites["mu_free"])
    else:
        mu_raw = sites["mu_raw"]
    mu = jnp.asarray(prior.mu_0) + jnp.asarray(prior.L_sigma_1) @ mu_raw
    if prior.fix_omega:
        u_raw = jnp.zeros(prior.n_params).at[
            np.asarray(prior.free_omega)].set(sites["u_free"])
    else:
        u_raw = sites["u_raw"]
    omega = build_omega(sites["s"], u_raw, prior)
    if prior.pin_discrepancy:
        a = b = jnp.zeros(prior.dim_z)
    elif prior.pin_b_columns:
        a = sites["a"]
        b = jnp.zeros(prior.dim_z).at[
            np.asarray(prior.free_b_columns)].set(sites["b_free"])
    else:
        a, b = sites["a"], sites["b"]
    beta_free = (prior.tau_beta * (sites["beta_raw"] - jnp.mean(sites["beta_raw"]))
                 if prior.n_beta else jnp.zeros(0))
    if not prior.n_aux:
        log_R = None
    elif prior.pin_aux:
        log_R = jnp.asarray(prior.log_R_0)
    else:
        log_R = sites["log_R"]
    return mu, omega, a, b, beta_free, log_R


@dataclass(frozen=True)
class Problem:
    """The fixed side of the fit: everything ``tau_all`` needs that is not ``phi``."""

    mech: object                              # vpop.predict.Mechanism
    plans: Sequence[object]                   # vpop.blocks.BlockPlan
    specs_by_cohort: Mapping[str, Sequence[object]]
    refs: jnp.ndarray                         # c_r, one level per readout
    designs: Optional[Mapping[str, jnp.ndarray]] = None
    mass_table: Optional[Mapping] = None

    def block_name(self, plan) -> str:
        return "+".join(plan.cohort_ids)


def population_model(prior: PopulationPrior, problem: Problem, V_chol,
                     observed=None):
    """eq:pop through eq:post. ``observed=None`` draws from the prior predictive."""
    import numpyro
    import numpyro.distributions as dist

    from qsp_inference.vpop.predict import tau_all

    P = prior.n_params
    if prior.fix_mu:
        sites = {"mu_free": numpyro.sample(
            "mu_free",
            dist.Normal(0.0, 1.0).expand([len(prior.free_mu)]).to_event(1))}
        mu_raw = jnp.zeros(P).at[np.asarray(prior.free_mu)].set(
            sites["mu_free"])
    else:
        sites = {"mu_raw": numpyro.sample(
            "mu_raw", dist.Normal(0.0, 1.0).expand([P]).to_event(1))}
        mu_raw = sites["mu_raw"]
    mu = numpyro.deterministic("mu", prior.mu_0 + prior.L_sigma_1 @ mu_raw)

    # No bound factor here. eq:muprior is Gaussian in mu, and for a logit-margin
    # parameter mu is the logit of the median, so exp of it is never the thing
    # that has to stay under 1 and the prior is the Gaussian the draft states
    # rather than a truncated one. build_prior moves those marginals into that
    # coordinate with logit_median_coords.

    # s and b_1 are aliased in principle. Report the split; do not
    # reparameterise around it, and do not orthonormalise Z to avoid it:
    # iid on an orthonormal basis is a different prior from iid on a.
    sites["s"] = numpyro.sample("s", dist.Normal(0.0, prior.tau_s))
    # u is one number per parameter against however many scale rows the
    # corpus prints, so most of it is unidentified whatever tau_u is. The
    # prior is what decides between the two ways that can go wrong. Wide, and
    # the unidentified components sit at the prior and print a width profile
    # that looks individuated when the individuation is a prior draw. Pinned
    # to zero, and a direction the scale rows genuinely constrain cannot move
    # either. Small and free is neither: the constrained directions are pulled
    # off zero, the rest stay near it, and no rank cutoff has to be defended.
    #
    # tau_u should be set from how well a parameter's role is known, not from
    # the spacing of the role table. Spacing is a statement about the rubric;
    # the prior needs a statement about the assignment, and the elicitation
    # that made the assignments measures exactly that in its own disagreement.
    # A tau_u below that disagreement claims the roles are known better than
    # the panel knew them. The project derives the number and passes it.
    #
    # The prior is normal, so it shrinks uniformly and pulls on a constrained
    # direction too. That is the wrong trade if some width is expected to be
    # strongly identified, and a heavy tail would be the tool. None is, here,
    # so the sampling cost is not worth taking. It is a choice, not a default.
    #
    # The unidentified components stay at tau_u, so the posterior spread of u
    # is not by itself evidence. Report the pooling factor, posterior sd of
    # u_j over tau_u: near 1 means the corpus said nothing about that width.
    if prior.fix_omega:
        sites["u_free"] = numpyro.sample(
            "u_free",
            dist.Normal(0.0, prior.tau_u)
            .expand([len(prior.free_omega)]).to_event(1))
    else:
        sites["u_raw"] = numpyro.sample(
            "u_raw",
            dist.Normal(0.0, prior.tau_u).expand([P]).to_event(1))

    if not prior.pin_discrepancy:
        sites["a"] = numpyro.sample("a", dist.Normal(0.0, prior.sigma_a)
                                    .expand([prior.dim_z]).to_event(1))
        if prior.pin_b_columns:
            sites["b_free"] = numpyro.sample(
                "b_free", dist.Normal(0.0, prior.sigma_b)
                .expand([len(prior.free_b_columns)]).to_event(1))
        else:
            sites["b"] = numpyro.sample("b", dist.Normal(0.0, prior.sigma_b)
                                        .expand([prior.dim_z]).to_event(1))

    if prior.n_beta:
        sites["beta_raw"] = numpyro.sample(
            "beta_raw", dist.Normal(0.0, 1.0).expand([prior.n_beta]).to_event(1))

    if prior.n_aux and not prior.pin_aux:
        sites["log_R"] = numpyro.sample(
            "log_R", dist.Normal(jnp.asarray(prior.log_R_0),
                                 jnp.asarray(prior.sigma_R)).to_event(1))

    # Every site is sampled above and nothing is derived from one there: the map
    # from sites to phi is phi_from_sites, which the Laplace metric and the
    # conditioning report differentiate. Two copies of it would let a mass matrix
    # be built for a model that is not this one, with no symptom.
    _, omega, a, b, beta_free, log_R = phi_from_sites(sites, prior)
    numpyro.deterministic("omega", omega)
    if prior.pin_discrepancy:
        numpyro.deterministic("a", a)
        numpyro.deterministic("b", b)
    elif prior.pin_b_columns:
        # The full-width b, so a posterior always carries one under that name
        # whatever was pinned, with a zero sitting where the claim was made.
        numpyro.deterministic("b", b)
    if prior.n_beta:
        numpyro.deterministic("beta_free", beta_free)
    if prior.n_aux and prior.pin_aux:
        numpyro.deterministic("log_R", log_R)

    taus = tau_all(mu, omega, a, b, beta_free, problem.plans,
                   problem.specs_by_cohort, problem.refs, problem.mech,
                   log_R=log_R, designs=problem.designs,
                   mass_table=problem.mass_table)

    for i, (plan, tau_B) in enumerate(zip(problem.plans, taus)):
        numpyro.sample(
            f"T_{problem.block_name(plan)}",
            dist.MultivariateNormal(tau_B, scale_tril=V_chol[i]),
            obs=None if observed is None else observed[i])

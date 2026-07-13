"""Prevalence weighting: turn a plausible-patient cloud into a virtual population.

The standard VPop construction (plausible-patient generation + prevalence weighting;
Allen, Rieger & Musante 2016) inverts the usual QSP instinct. Rather than trying to
get each parameter's between-patient spread right *a priori* and pushing it forward,
we sample a large cloud of candidate patients from a deliberately generous population
prior, discard the biologically implausible ones, and then **weight the survivors so
that, as an ensemble, they reproduce the observed patient distributions**. No single
parameter's spread has to be known — the cohort just has to come out looking like the
real one.

Why max-entropy. Many weightings reproduce the same marginals; that non-uniqueness is
intrinsic to the problem, not a bug in the solver. We resolve it by taking the
weighting closest to uniform in KL — the least-committal ensemble consistent with the
data — which is the classic maximum-entropy / exponential-tilting answer:

    w_i  proportional to  exp(lambda . a_i)

where ``a_i`` stacks patient i's constraint features. Fitting is the smooth convex dual

    L(lambda) = logsumexp(A^T lambda) - lambda . p + (ridge/2) ||lambda||^2

with gradient ``A w - p + ridge * lambda``. The ridge term keeps the fit finite when the
constraints are not exactly satisfiable, which is the common case.

Constraints are **quantile bins of the observed data**, not moments. Binning on the
observed sample's own quantiles makes the target probabilities uniform (1/n_bins) by
construction and lets the fit match distributional *shape* — including bimodality,
which a mean-and-SD match would erase. The outer bins are open-ended so the cloud's
tails are scored rather than silently dropped.

The diagnostic that matters most is **support**, not fit. Reweighting cannot conjure a
patient the cloud does not contain: if the observed data put mass where no plausible
patient lives, the weights will strain toward a corner of the cloud and the effective
sample size collapses. That is a signal to widen the population prior (see the omega
prior source), not to push the optimiser harder. ``VPopResult`` reports both.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

__all__ = ["VPopResult", "fit_prevalence_weights", "build_quantile_constraints"]

# A bin holding fewer than this many cloud members cannot be weighted up to match the
# data; it is a support failure, not a fitting failure.
MIN_SUPPORT = 5


@dataclass
class VPopResult:
    """A fitted virtual population."""

    weights: np.ndarray
    """Per-patient prevalence weights over the plausible cloud; sums to 1."""

    ess: float
    """Effective sample size, 1 / sum(w^2). How many patients actually carry the cohort."""

    ess_fraction: float
    """ESS as a fraction of the cloud. Low means the fit leaned on a few patients."""

    converged: bool

    per_observable: pd.DataFrame
    """Per-observable fit quality (total-variation distance) and support deficiency."""

    tau2: float = 0.0
    """Model-discrepancy variance in observable space (0 when ``n_obs`` was not given).

    Estimated by method of moments from the residuals: the part of the model-vs-data gap
    that sampling noise alone CANNOT explain. ~0 = indistinguishable from a perfect model.
    A direct misspecification statistic, needing no null simulation."""

    support_deficient: list[str] = field(default_factory=list)
    """Observables where the observed data reach outside the cloud. Widen the prior."""

    def summary(self) -> str:
        worst = (
            self.per_observable.sort_values("tv_distance", ascending=False)
            .head(3)[["observable", "tv_distance"]]
            .to_string(index=False)
        )
        lines = [
            f"VPop: ESS {self.ess:.0f} / {len(self.weights)} ({self.ess_fraction:.1%})",
            f"converged={self.converged}",
            f"worst-fit observables:\n{worst}",
        ]
        if self.support_deficient:
            lines.append(
                "SUPPORT DEFICIENT (cloud does not reach the data; widen the population "
                f"prior rather than re-fitting): {', '.join(self.support_deficient)}"
            )
        return "\n".join(lines)


def build_quantile_constraints(
    sim_obs: np.ndarray,
    observed: dict[str, np.ndarray],
    obs_names: list[str],
    *,
    n_bins: int = 5,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int]], dict[str, int]]:
    """Bin each observable on the observed sample's quantiles.

    Returns ``(A, p, labels, support)``:
      * ``A`` (n_constraints, n_cloud) -- 1 where cloud member i lands in that bin
      * ``p`` (n_constraints,) -- observed probability of each bin
      * ``labels`` -- (observable, bin index) per constraint row
      * ``support`` -- observable -> smallest number of cloud members in any of its bins
    """
    if sim_obs.ndim != 2:
        raise ValueError(f"sim_obs must be (n_cloud, n_obs); got shape {sim_obs.shape}")
    if sim_obs.shape[1] != len(obs_names):
        raise ValueError(
            f"sim_obs has {sim_obs.shape[1]} columns but {len(obs_names)} obs_names"
        )
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")

    rows: list[np.ndarray] = []
    targets: list[float] = []
    labels: list[tuple[str, int]] = []
    support: dict[str, int] = {}

    inner = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # interior quantile levels

    for k, name in enumerate(obs_names):
        obs = np.asarray(observed[name], dtype=np.float64)
        obs = obs[np.isfinite(obs)]
        if obs.size == 0:
            raise ValueError(f"observable '{name}' has no finite observed samples")

        edges = np.quantile(obs, inner)
        # Degenerate (many ties) -> fewer usable bins; dedupe rather than emit
        # zero-width bins that no member can ever occupy.
        edges = np.unique(edges)

        sim = sim_obs[:, k]
        sim_bin = np.digitize(sim, edges)  # 0 .. len(edges)
        n_actual = len(edges) + 1

        counts = np.bincount(sim_bin, minlength=n_actual)
        support[name] = int(counts.min())

        # Target probabilities with a CONTINUITY CORRECTION for ties.
        #
        # A naive digitize of the observed sample shoves every value tied AT an edge
        # into one bin. When the observed sample has an atom at its own quantile --
        # common for bootstrap samples off a coarse published summary -- that skews
        # the target away from the nominal 1/n_bins (e.g. 0.444 instead of 0.500).
        #
        # That is not a harmless wobble. Two observables the model treats as the SAME
        # variable (a readout in two treatment arms the model cannot tell apart) then
        # receive CONTRADICTORY targets at the same threshold -- 44.4% below vs 50%
        # below. The dual has no solution, so the optimiser exploits floating-point
        # differences between the two columns, weights blow up, and ESS collapses while
        # the TV distance still looks perfect. It reads exactly like a joint conflict
        # and is not one.
        #
        # Splitting tied mass across the edge it sits on restores the nominal target.
        below = (obs[:, None] < edges[None, :]).sum(axis=0).astype(np.float64)
        tied = (obs[:, None] == edges[None, :]).sum(axis=0).astype(np.float64)
        cdf_at_edge = (below + 0.5 * tied) / len(obs)  # mid-rank ECDF
        bounds = np.concatenate([[0.0], cdf_at_edge, [1.0]])
        probs = np.diff(bounds)

        for b in range(n_actual):
            rows.append((sim_bin == b).astype(np.float64))
            targets.append(float(probs[b]))
            labels.append((name, b))

    A = np.vstack(rows)
    p = np.asarray(targets, dtype=np.float64)
    return A, p, labels, support


def fit_prevalence_weights(
    sim_obs: np.ndarray,
    observed: dict[str, np.ndarray],
    obs_names: list[str],
    *,
    n_bins: int = 5,
    ridge: float = 1e-3,
    max_iter: int = 500,
    n_obs: dict[str, int] | None = None,
) -> VPopResult:
    """Weight a plausible-patient cloud so it reproduces the observed distributions.

    Args:
        sim_obs: (n_cloud, n_obs) simulated observables for the plausible cloud.
        observed: observable name -> 1-D array of observed patient values. These are
            the across-patient ``samples`` the calibration targets already emit.
        obs_names: column order of ``sim_obs``.
        n_bins: quantile bins per observable. More bins match shape more finely and
            cost effective sample size.
        ridge: L2 on the dual variables. Larger = softer constraints, higher ESS,
            looser fit. Keeps the fit finite when the data are unreachable.
        max_iter: L-BFGS iterations.
        n_obs: observable name -> the REAL published sample size behind that target.
            **Supply this.** Without it the fit matches the observed proportions
            *exactly*, which treats a 6-patient proportion as exact truth and burns
            effective sample size reproducing its sampling noise -- see below.

    Returns:
        A :class:`VPopResult`. Read ``support_deficient`` before trusting the weights:
        a cloud that cannot reach the data cannot be reweighted onto it.

    Shrinkage (why ``n_obs`` matters more than it looks):
        A bin proportion estimated from n patients carries sampling noise of order
        ``sqrt(p(1-p)/n)``. Matching it *exactly* tilts the cloud toward that noise.
        Under a well-specified model the cloud's marginals are already right, so the
        tilt moves it AWAY from the truth and pays effective sample size for the
        privilege: with 46 observables at n=11, exact matching burns ~97% of the cloud
        reproducing noise. The n-dependence that produces is an artefact of the
        estimator, not a fact about virtual populations.

        Given ``n_obs`` we instead shrink each target toward the model's own marginal,

            p* = p_model + s * (p_hat - p_model),   s = tau^2 / (tau^2 + v)

        where ``v = p(1-p)/n`` is the target's sampling variance and ``tau^2`` is the
        model-discrepancy variance, estimated from the residuals by method of moments.
        A perfect model gives ``tau^2 = 0``, so ``s = 0``, so no tilt **at any n** --
        the ESS stays at N. A genuinely displaced model gives ``s -> 1`` and the fit
        tilts as hard as before, so misspecification is still caught.

        ``n`` then plays its proper role: it sets how much you TRUST a discrepancy (a
        6-patient target barely moves the cohort; a 700-patient one moves it a lot),
        rather than deciding how big the cohort is. ``tau^2`` is reported on the result
        and is itself a direct misspecification statistic.

        This is the same ``SEM + tau^2`` structure the submodel error models already use,
        one level up.
    """
    A, p_hat, labels, support = build_quantile_constraints(
        sim_obs, observed, obs_names, n_bins=n_bins
    )
    n_cloud = sim_obs.shape[0]

    # Shrink the target toward the model's own marginal by how much of the residual
    # sampling noise alone can explain.
    p_model = A @ np.full(n_cloud, 1.0 / n_cloud)
    tau2 = 0.0
    if n_obs is not None:
        v = np.array(
            [p_hat[i] * (1.0 - p_hat[i]) / max(int(n_obs[labels[i][0]]), 1)
             for i in range(len(p_hat))]
        )
        tau2 = float(max(0.0, np.mean((p_hat - p_model) ** 2) - np.mean(v)))
        s = tau2 / (tau2 + v) if tau2 > 0 else np.zeros_like(v)
        p = p_model + s * (p_hat - p_model)
    else:
        p = p_hat

    def objective(lam: np.ndarray) -> tuple[float, np.ndarray]:
        scores = A.T @ lam  # (n_cloud,)
        lse = logsumexp(scores)
        w = np.exp(scores - lse)
        loss = lse - float(lam @ p) + 0.5 * ridge * float(lam @ lam)
        grad = A @ w - p + ridge * lam
        return loss, grad

    res = minimize(
        objective,
        x0=np.zeros(A.shape[0]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    scores = A.T @ res.x
    weights = np.exp(scores - logsumexp(scores))

    ess = 1.0 / float(np.sum(weights**2))
    achieved = A @ weights

    per_obs = []
    for name in obs_names:
        idx = [i for i, (o, _) in enumerate(labels) if o == name]
        # Total-variation distance between the weighted cloud and the observed data
        # over this observable's bins: 0 = perfect, 1 = disjoint.
        tv = 0.5 * float(np.abs(achieved[idx] - p[idx]).sum())
        per_obs.append(
            {
                "observable": name,
                "tv_distance": tv,
                "min_bin_support": support[name],
                "support_deficient": support[name] < MIN_SUPPORT,
            }
        )
    per_observable = pd.DataFrame(per_obs)

    return VPopResult(
        tau2=tau2,
        weights=weights,
        ess=ess,
        ess_fraction=ess / n_cloud,
        converged=bool(res.success),
        per_observable=per_observable,
        support_deficient=per_observable.loc[
            per_observable["support_deficient"], "observable"
        ].tolist(),
    )

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
        obs_bin = np.digitize(obs, edges)
        n_actual = len(edges) + 1

        counts = np.bincount(sim_bin, minlength=n_actual)
        support[name] = int(counts.min())

        for b in range(n_actual):
            rows.append((sim_bin == b).astype(np.float64))
            targets.append(float((obs_bin == b).mean()))
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

    Returns:
        A :class:`VPopResult`. Read ``support_deficient`` before trusting the weights:
        a cloud that cannot reach the data cannot be reweighted onto it.
    """
    A, p, labels, support = build_quantile_constraints(
        sim_obs, observed, obs_names, n_bins=n_bins
    )
    n_cloud = sim_obs.shape[0]

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
        weights=weights,
        ess=ess,
        ess_fraction=ess / n_cloud,
        converged=bool(res.success),
        per_observable=per_observable,
        support_deficient=per_observable.loc[
            per_observable["support_deficient"], "observable"
        ].tolist(),
    )

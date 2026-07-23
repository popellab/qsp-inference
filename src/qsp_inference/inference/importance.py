"""Importance reweighting from a training proposal to the reporting prior.

Neural posterior estimation learns ``q(θ | x)`` from pairs simulated at θ drawn
from some distribution. That distribution has a purely computational job — cover
the region the posterior might occupy, keep the estimator well-trained — and it is
not the same object as the prior we want the reported posterior to be conditioned
on. Conflating the two is what forces a modeller to widen a prior for coverage and
then discover the population is over-dispersed, or narrow it for the population and
starve the training set.

Separating them costs one line of arithmetic. Draw θ from a proposal ``p̃``, report
under the anchored prior ``p`` by weighting each sample

    w(θ) ∝ p(θ) / p̃(θ),

which is ordinary self-normalized importance sampling. Both densities are available
in closed form for :class:`~qsp_inference.priors.copula_prior.GaussianCopulaPrior`,
so the weight is a subtraction of log-densities.

**Truncation.** When the proposal has been truncated — TSNPE keeps only the region
``S`` where the posterior density clears a quantile — its density is
``p̃_base(θ)·1[θ ∈ S] / Z`` with ``Z = P_{p̃_base}(S)``. ``Z`` does not depend on θ,
so under *self-normalized* weights it cancels exactly: evaluating ``log_prob`` on
the untruncated base object is correct, provided every sample being reweighted was
actually drawn from the truncated proposal and therefore lies in ``S``. The
normalizer would matter if these weights were used to compare densities across
rounds or to estimate a marginal likelihood; neither happens here, and both are
out of scope for this module.

**Relation to sbi's own importance weighting.** sbi implements this ratio already,
but in a different place and for a different purpose, so none of it substitutes for
what is here. ``NPE_B`` (``sbi/inference/trainers/npe/npe_b.py``) forms
``prior / proposal`` and applies it *inside the training loss* (Lueckmann,
Gonçalves et al. 2017), correcting the estimator as rounds proceed; its weights are
known to degenerate in high dimension, which is why NPE-C replaced it with the
atomic loss. ``ImportanceSamplingPosterior`` wraps a *potential function* and draws
via SIR with an oversampling factor — built for the NLE/NRE setting, and it returns
resampled draws rather than weights.

We are in neither case. Training uses ``force_first_round_loss=True``, so the flow
learns ``q(θ|x) ∝ p(x|θ)·p̃(θ)`` — the posterior with the *proposal* standing in as
the prior, uncorrected, which is exactly what avoids both NPE-B's weight variance
and NPE-C's atomic leakage. Recovering the posterior under the anchored prior is
then one multiplication at report time, ``q(θ|x)·p(θ)/p̃(θ)``, applied to the output
samples. sbi has no API for that because it assumes the object passed as ``prior=``
is the prior you intend to report under; declining that assumption is the point.
Weights (not resampled draws) are kept deliberately, so the ESS stays visible.

**Precision.** ``GaussianCopulaPrior.log_prob`` evaluates in float64 and casts to
float32 on return. Weights are formed as a difference of two such log-densities, so
in high dimension — where each log-density can reach several hundred in magnitude —
the difference carries roughly ``1e-4`` of absolute error, i.e. order ``0.01%`` on a
weight. That is immaterial for weighted medians and ESS, which is all this module
is used for. It would not be immaterial for a normalizing-constant estimate, which
is another reason not to use these weights for one.

**What ESS does and does not tell you.** The effective sample size reported here
measures the cost of the change of measure — how much of the sample survives the
reweighting — and nothing else. A low ESS means the proposal and the reporting
prior disagree about where the mass sits. Under prior-data conflict that is exactly
what you should expect, because the data have pulled the posterior into the prior's
tail; the ESS is then a *consequence* of a conflict, not evidence for one. Conflict
is diagnosed in observable space (a prior-predictive tail probability), never from
a weight-degeneracy number. Treat a low ESS as a statement about compute.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch

__all__ = [
    "ReweightResult",
    "log_importance_weights",
    "reweight_to_prior",
    "effective_sample_size",
    "weighted_quantile",
]


@dataclass
class ReweightResult:
    """Weights carrying samples drawn under a proposal onto a target prior.

    Attributes:
        log_weights: Unnormalized ``log p(θ) - log p̃(θ)``, shape ``(n,)``.
        weights: Self-normalized weights summing to 1, shape ``(n,)``.
        ess: Kish effective sample size, ``1 / Σ wᵢ²``.
        ess_fraction: ``ess / n``.
        max_weight: Largest normalized weight — a single sample carrying most of
            the mass is the failure mode a bare ESS can hide.
    """

    log_weights: np.ndarray
    weights: np.ndarray
    ess: float
    ess_fraction: float
    max_weight: float

    @property
    def n(self) -> int:
        return int(self.weights.size)

    @property
    def is_degenerate(self) -> bool:
        """True when too little of the sample survives to report a marginal.

        The 0.05 threshold is a reporting convention, not a test: below it the
        weighted quantiles are being carried by a handful of draws and should not
        be quoted. It says nothing about whether the model fits.
        """
        return self.ess_fraction < 0.05

    def summary(self) -> str:
        return (
            f"reweight: ESS {self.ess:.0f} / {self.n} ({self.ess_fraction:.1%}), "
            f"max weight {self.max_weight:.3g}"
            + ("  [degenerate]" if self.is_degenerate else "")
        )


def _as_numpy(theta) -> np.ndarray:
    if isinstance(theta, torch.Tensor):
        return theta.detach().cpu().numpy()
    return np.asarray(theta)


def _check_alignment(target, proposal, n_dim: int) -> None:
    """Fail loudly when the two priors are not over the same parameters.

    A silent column misalignment between two independently-built priors produces
    finite, plausible-looking weights that are wrong for every sample, so this is
    checked rather than assumed.
    """
    t_names = getattr(target, "param_names", None)
    p_names = getattr(proposal, "param_names", None)

    if t_names is not None and p_names is not None:
        if list(t_names) != list(p_names):
            t_set, p_set = set(t_names), set(p_names)
            only_t = sorted(t_set - p_set)[:5]
            only_p = sorted(p_set - t_set)[:5]
            detail = []
            if only_t:
                detail.append(f"only in target: {only_t}")
            if only_p:
                detail.append(f"only in proposal: {only_p}")
            if not detail:
                detail.append("same names, different order")
            raise ValueError(
                "target and proposal are not over the same parameters in the same "
                "order (" + "; ".join(detail) + "). Reweighting misaligned priors "
                "silently produces wrong weights."
            )
        if len(t_names) != n_dim:
            raise ValueError(
                f"theta has {n_dim} columns but the priors carry {len(t_names)} "
                "parameters."
            )


def log_importance_weights(
    theta,
    target,
    proposal,
    *,
    check_alignment: bool = True,
) -> np.ndarray:
    """Unnormalized log weights ``log p(θ) - log p̃(θ)``.

    Args:
        theta: Samples drawn from ``proposal``, shape ``(n, d)``. Must be in the
            space both distributions are defined on — log-space for the
            ``*_log`` prior loaders.
        target: The prior to report under (``p``). Needs ``log_prob``.
        proposal: The distribution ``theta`` was drawn from (``p̃``).
        check_alignment: Verify both carry the same ``param_names`` in the same
            order. Leave on unless the priors deliberately lack names.

    Returns:
        ``(n,)`` array of unnormalized log weights. Samples outside the target's
        support get ``-inf`` (weight zero), which is meaningful.

    Raises:
        ValueError: on parameter misalignment, or if any sample falls outside the
            *proposal's* support — that means ``theta`` was not drawn from
            ``proposal`` and the weights would be meaningless.
    """
    theta_np = _as_numpy(theta)
    if theta_np.ndim == 1:
        theta_np = theta_np[None, :]
    if theta_np.ndim != 2:
        raise ValueError(f"theta must be (n, d); got shape {theta_np.shape}")

    if check_alignment:
        _check_alignment(target, proposal, theta_np.shape[1])

    theta_t = torch.as_tensor(theta_np, dtype=torch.float32)
    with torch.no_grad():
        lp_target = _as_numpy(target.log_prob(theta_t)).astype(np.float64).ravel()
        lp_proposal = _as_numpy(proposal.log_prob(theta_t)).astype(np.float64).ravel()

    if not np.all(np.isfinite(lp_proposal)):
        n_bad = int((~np.isfinite(lp_proposal)).sum())
        raise ValueError(
            f"{n_bad} of {lp_proposal.size} samples have non-finite log-density "
            "under the proposal, so they cannot have been drawn from it. Check "
            "that theta and proposal correspond (and that both are in log-space "
            "if the priors are)."
        )

    # -inf under the target is legitimate: that sample simply carries no weight.
    log_w = lp_target - lp_proposal
    log_w[np.isneginf(lp_target)] = -np.inf
    return log_w


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size of self-normalized weights."""
    w = np.asarray(weights, dtype=np.float64)
    s2 = float(np.sum(w**2))
    return 0.0 if s2 <= 0 else 1.0 / s2


def reweight_to_prior(
    theta,
    target,
    proposal,
    *,
    check_alignment: bool = True,
    warn_below: float | None = 0.05,
) -> ReweightResult:
    """Reweight proposal-drawn samples onto the target prior.

    Args:
        theta: Samples from ``proposal``, shape ``(n, d)``.
        target: Prior to report under.
        proposal: Distribution ``theta`` was drawn from.
        check_alignment: Verify matching ``param_names``.
        warn_below: Emit a ``UserWarning`` when the ESS fraction falls below this.
            Set ``None`` to silence. This is a reporting guard — see the module
            docstring on why a low ESS is not a misspecification signal.

    Returns:
        A :class:`ReweightResult`.
    """
    log_w = log_importance_weights(
        theta, target, proposal, check_alignment=check_alignment
    )

    if not np.any(np.isfinite(log_w)):
        raise ValueError(
            "every sample has zero weight under the target prior — the proposal "
            "and target priors have disjoint support."
        )

    # Self-normalize in log-space; the shift makes the exponential safe and, being
    # a constant, leaves the normalized weights unchanged (this is also why a
    # truncated proposal's normalizer cancels).
    m = np.max(log_w[np.isfinite(log_w)])
    w = np.exp(log_w - m)
    w[~np.isfinite(log_w)] = 0.0
    total = float(np.sum(w))
    w = w / total

    ess = effective_sample_size(w)
    ess_fraction = ess / w.size

    if warn_below is not None and ess_fraction < warn_below:
        warnings.warn(
            f"importance reweight ESS is {ess:.0f}/{w.size} ({ess_fraction:.1%}) — "
            "the proposal and the reporting prior disagree about where the mass "
            "is. Weighted summaries are carried by few draws. This is a statement "
            "about compute, not evidence of misspecification.",
            UserWarning,
            stacklevel=2,
        )

    return ReweightResult(
        log_weights=log_w,
        weights=w,
        ess=ess,
        ess_fraction=ess_fraction,
        max_weight=float(np.max(w)),
    )


def weighted_quantile(
    x: np.ndarray, weights: np.ndarray, q: float | np.ndarray
) -> np.ndarray:
    """Quantiles of ``x`` under self-normalized ``weights``.

    Uses the standard cumulative-weight convention (interpolating on
    ``cumsum(w) - w/2``), which reduces to the usual empirical quantile when the
    weights are uniform.

    Args:
        x: Values, shape ``(n,)``.
        weights: Self-normalized weights, shape ``(n,)``.
        q: Quantile(s) in [0, 1].

    Returns:
        Array of quantiles, matching the shape of ``q``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if x.size != w.size:
        raise ValueError(f"x has {x.size} entries but weights has {w.size}")

    order = np.argsort(x)
    x_s, w_s = x[order], w[order]
    total = w_s.sum()
    if total <= 0:
        raise ValueError("weights sum to zero")
    w_s = w_s / total

    cum = np.cumsum(w_s) - 0.5 * w_s
    return np.interp(np.asarray(q, dtype=np.float64), cum, x_s)

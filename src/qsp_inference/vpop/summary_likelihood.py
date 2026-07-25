"""The closed-form summary likelihood for population inference (docs ch. 4b).

Chapter 4 fits the population by training an amortized density estimator on a
hyper-proposal. This module implements the alternative: the sampling law of the
observed summaries is available in closed form, so write it down and sample
``phi`` with NUTS.

The data are per-target quantile anchors ``Q_hat_j(p)`` at each target's real
published ``n`` (:mod:`qsp_inference.targets.anchors`). Their joint asymptotic
sampling law, for observables ``j, l`` summarised over the **same** ``n``
patients, is

    Cov(Q_hat_j(p_a), Q_hat_l(p_b))
        = [C_jl(p_a, p_b) - p_a p_b] / (n f_j(Q_j(p_a)) f_l(Q_l(p_b)))

with ``C_jl`` the copula of ``(X_j, X_l)`` under the population predictive.
Setting ``j = l`` gives ``C_jj(p_a, p_b) = min(p_a, p_b)`` and recovers the
familiar ``p_a (1 - p_b) / (f f)``. One formula covers the whole conditioning
vector, and the only new object it needs is the model's own rank dependence.

Three facts about this that were measured, not assumed (docs ch. 4b
§*It was checked*):

- The law holds down to ``n = 6``, but only **in log space**, and only at
  **three anchors** for small-``n`` targets. Hence the *working scale* contract
  below, and hence ``anchors.MIN_PATIENTS_PER_ANCHOR``.
- The density plug-in ``f_j`` is the weakest part. An error in it scales that
  observable's whole covariance, so it acts as a *weight* on how much the target
  counts, not as a bias on ``phi`` -- a precision-misallocation problem, not a
  correctness one. It is still the term to check first when coverage is off.
- 33 of 49 PDAC observables share a scenario and an ``n``: one assay panel on
  one set of biopsies. Factorising over those counts correlated readouts as
  independent, which is the overconfident direction, and it corrupts
  ``tau_eta`` (shared *sampling noise* gets absorbed as between-study
  *heterogeneity*). Hence :class:`StudyBlock`.

Working scale
-------------
Everything here lives on a monotone transform ``g`` of the raw observable, and
quantiles commute with monotone transforms, so nothing special is needed: supply
``predict`` returning ``g(x)`` and observed anchors already mapped through ``g``.
For QSP readouts ``g = log`` is the choice that was validated. The likelihood
never sees raw space.

What is live in ``phi`` and what is held fixed
---------------------------------------------
The mean model ``Q_j(phi)`` is always live: that is the whole forward map.
Everything in the *covariance* is a plug-in, and the recommended configuration
holds all of it fixed during a fit:

- ``rho_jl``, the normal-score correlation feeding ``C_jl``. The empirical copula
  is a step function of ``phi``, so a live version would put rank flips inside
  the NUTS potential. Set by :meth:`SummaryLikelihood.refresh_rho`.
- ``f_j``, the density plug-in, and with it the whole ``Sigma``. Leaving it live
  is a correct parametric density but it **biases the population spread low** by
  ``O(1/n)``, because ``Sigma`` scales with the population width and the
  log-determinant term therefore rewards a narrower population on its own. It is
  12% at ``n = 20``. :meth:`SummaryLikelihood.freeze_covariance_at` removes it
  and carries the measurement.

Both are computed **once, before the fit**, at whatever ``phi_0`` you have. There
is no iteration to converge and nothing here resembles the TSNPE truncation
rounds ch. 4b deletes. Measured on a curved forward map at ``n = 20``: a
``phi_0`` that is *twice too wide* moves ``phi_hat`` by 0.14 to 0.32 of one
sampling standard deviation and leaves the recovered spread unbiased (ratio
0.966 either way). So the plug-in is a weighting choice with a second-order cost,
and refitting once at ``phi_hat`` to confirm nothing moved is a **diagnostic**,
not a required pass.

Requires torch (the ``sbi`` extra). The emulator must be differentiable and
vectorised; see docs ch. 4b §*Common random numbers make it differentiable*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "TargetAnchor",
    "StudyBlock",
    "SummaryLikelihood",
    "build_study_blocks",
    "normal_score_correlation",
    "bvn_cdf",
    "anchor_covariance",
]

_DTYPE = torch.float64
_SQRT2 = float(np.sqrt(2.0))
_TWO_PI = float(2.0 * np.pi)

# Gauss-Legendre nodes for the bivariate-normal CDF quadrature. 32 is far more
# than the integrand needs (it is C-infinity on the interval after the sin
# substitution) and costs nothing at the block sizes here.
_GL_N, _GL_W = np.polynomial.legendre.leggauss(32)


def _std_normal_cdf(z: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


def bvn_cdf(z1: Tensor, z2: Tensor, rho: Tensor) -> Tensor:
    """Standard bivariate normal CDF ``Phi_2(z1, z2; rho)``, broadcast and
    differentiable in all three arguments.

    Uses the classical identity ``d/d rho Phi_2 = phi_2``, integrated with the
    ``r = sin(t)`` substitution that removes the ``(1 - r^2)^(-1/2)`` factor:

        Phi_2(z1, z2; rho) = Phi(z1) Phi(z2)
            + (1/2pi) * int_0^{asin rho} exp(-(z1^2 - 2 z1 z2 sin t + z2^2)
                                             / (2 cos^2 t)) dt

    The integrand is smooth on the whole interval, so fixed-node Gauss-Legendre
    is accurate to roughly machine precision away from ``|rho| = 1`` (where it is
    clamped). ``scipy`` has no differentiable equivalent, which is why this is
    written out rather than wrapped.
    """
    rho = rho.clamp(-0.999999, 0.999999)
    nodes = torch.as_tensor(_GL_N, dtype=z1.dtype, device=z1.device)
    weights = torch.as_tensor(_GL_W, dtype=z1.dtype, device=z1.device)

    upper = torch.asin(rho)[..., None]                   # (..., 1)
    t = 0.5 * upper * (nodes + 1.0)                      # (..., Nq)
    w = 0.5 * upper * weights                            # (..., Nq)

    s = torch.sin(t)
    cos2 = (1.0 - s * s).clamp_min(1e-300)
    a, b = z1[..., None], z2[..., None]
    integrand = torch.exp(-(a * a - 2.0 * s * a * b + b * b) / (2.0 * cos2))
    return _std_normal_cdf(z1) * _std_normal_cdf(z2) + (integrand * w).sum(-1) / _TWO_PI


def anchor_covariance(
    p: Tensor,
    col: Tensor,
    f: Tensor,
    rho: Tensor,
    n: float,
) -> Tensor:
    """The asymptotic covariance of a stacked anchor vector, one study block.

    Args:
        p: ``(M,)`` probability level of each row.
        col: ``(M,)`` long tensor, the observable each row belongs to (indices
            into ``rho``).
        f: ``(M,)`` population density at that row's quantile, on the working
            scale. Live in ``phi``.
        rho: ``(J, J)`` normal-score correlation of the observables. Held fixed.
        n: cohort size the block's summaries were computed over. Pass the
            effective size when Monte-Carlo error in ``Q(phi)`` is being folded
            in (see :class:`SummaryLikelihood`).

    Returns:
        ``(M, M)`` symmetric covariance.

    Within an observable the copula is exact (``min(p_a, p_b)``); across
    observables it is the Gaussian-copula approximation
    ``C_jl = Phi_2(z_{p_a}, z_{p_b}; rho_jl)``, which is smooth and cheap where
    the empirical copula is a step function. That approximation sits in a
    nuisance term of the covariance, not in the mean, so its error does not move
    ``phi`` the way an error in ``Q_j(phi)`` would.
    """
    pa, pb = p[:, None], p[None, :]
    same = col[:, None] == col[None, :]

    z = torch.special.ndtri(p)
    r = rho[col][:, col]
    c_cross = bvn_cdf(z[:, None].expand_as(pa * pb), z[None, :].expand_as(pa * pb), r)
    c = torch.where(same, torch.minimum(pa, pb), c_cross)

    cov = (c - pa * pb) / (float(n) * f[:, None] * f[None, :])
    return 0.5 * (cov + cov.T)


def normal_score_correlation(x: np.ndarray) -> np.ndarray:
    """``rho_jl`` for the copula block: the correlation of van-der-Waerden normal
    scores of a population sample ``x`` of shape ``(n_patients, J)``.

    This is the Gaussian-copula plug-in the covariance wants, and it is rank
    based, so it is invariant to the working-scale transform. Non-finite entries
    are ranked within their own column and the pairwise correlation is taken over
    rows finite in both columns.

    Estimate it on the *fitted* population, not on a deliberately generous
    proposal cloud: ``rho`` is manufactured by the width of the population it is
    evaluated on (docs ch. 4b), so a wide proposal reports an upper bound.
    """
    x = np.asarray(x, dtype=np.float64)
    n, j = x.shape
    from scipy.stats import norm, rankdata

    scores = np.full_like(x, np.nan)
    for c in range(j):
        ok = np.isfinite(x[:, c])
        m = int(ok.sum())
        if m < 3:
            continue
        r = rankdata(x[ok, c], method="average")
        scores[ok, c] = norm.ppf(r / (m + 1.0))

    rho = np.eye(j)
    for a in range(j):
        for b in range(a + 1, j):
            ok = np.isfinite(scores[:, a]) & np.isfinite(scores[:, b])
            if ok.sum() < 3:
                continue
            sa, sb = scores[ok, a], scores[ok, b]
            denom = sa.std() * sb.std()
            v = 0.0 if denom <= 0 else float(np.mean((sa - sa.mean()) * (sb - sb.mean())) / denom)
            rho[a, b] = rho[b, a] = np.clip(v, -0.999, 0.999)
    return rho


@dataclass(frozen=True)
class TargetAnchor:
    """One target's observed anchors, on the working scale.

    The unit of the observed data: one observable, measured on one cohort. The
    contract deliberately keeps ``cohort_id`` explicit rather than deriving it,
    because coupling is the assumption that manufactures correlation and the
    default when provenance is unknown must be *distinct* (docs ch. 4b).

    Attributes:
        obs_name: readable id, for reporting.
        obs_index: column of this observable in the emulator's output.
        cohort_id: the patient set. Targets sharing it share sampling noise and
            share the study offset ``eta_s``. Scenario + ``n`` is a working proxy
            when the targets carry no provenance column; two targets from one
            paper measured on *different* subsets would be wrongly coupled by it.
        n: real biological cohort size.
        p_levels: anchor probability levels, ascending.
        values: observed ``Q_hat(p)`` at those levels, working scale.
        sigma_c: per-anchor center uncertainty (matched-footing term 1), working
            scale. Scalar broadcasts.
        epsilon: the emulator's held-out residual SD for this observable on the
            working scale (matched-footing term 2). Uncertainty about the
            *model's prediction*, so it belongs here and nowhere else -- no noise
            injection, no M-draw averaging (docs ch. 4b §*Surrogate error*).
        feeds_spread: whether the spread anchors are genuine population
            variability. A ci95-derived target is center-only; its spread anchors
            are dropped rather than believed.
    """

    obs_name: str
    obs_index: int
    cohort_id: str
    n: int
    p_levels: tuple
    values: np.ndarray
    sigma_c: float | np.ndarray = 0.0
    epsilon: float = 0.0
    feeds_spread: bool = True


@dataclass
class StudyBlock:
    """The stacked anchor vector for one ``cohort_id``, with its noise terms.

    Rows are ``(observable, p)`` pairs concatenated over every target measured on
    this cohort. Studies are independent of each other, so the log-density is a
    sum over blocks; a target with no cohort mate is the 1x1 case.

    ``diag_var`` holds the two terms that are *not* properties of the patient
    sample and so stay diagonal: the per-target center uncertainty and the
    emulator residual. Only the quantile covariance, which is finite-sample noise
    on shared patients, gets off-diagonal structure.
    """

    cohort_id: str
    n: int
    study_index: int
    obs_names: list
    y: Tensor            # (M,) observed anchors, working scale
    p: Tensor            # (M,)
    col: Tensor          # (M,) long, emulator observable column per row
    delta_index: Tensor  # (M,) long, index into the global per-observable delta
    diag_var: Tensor     # (M,) sigma_c^2 + epsilon^2

    @property
    def size(self) -> int:
        return int(self.y.numel())


def build_study_blocks(
    targets: Sequence[TargetAnchor],
    *,
    drop_spread_from_center_only: bool = True,
    device=None,
) -> tuple[list, list, list]:
    """Group targets into per-cohort blocks.

    Args:
        targets: the observed data.
        drop_spread_from_center_only: for a target with ``feeds_spread=False``,
            keep only the median anchor. A ci95-derived interval is uncertainty
            about a *center*, and letting it act as population spread is the
            SEM-vs-SD conflation the omega layer exists to avoid.
        device: torch device for the emitted tensors.

    Returns:
        ``(blocks, cohort_ids, delta_names)`` -- ``cohort_ids`` indexes ``eta``
        and ``delta_names`` indexes ``delta``, both in the order the fit expects.
    """
    kept: list = []
    for t in targets:
        if drop_spread_from_center_only and not t.feeds_spread:
            ps = np.asarray(t.p_levels, dtype=np.float64)
            k = int(np.argmin(np.abs(ps - 0.5)))
            kept.append(
                TargetAnchor(
                    t.obs_name, t.obs_index, t.cohort_id, t.n,
                    (float(ps[k]),), np.asarray(t.values, dtype=np.float64)[k : k + 1],
                    np.broadcast_to(np.asarray(t.sigma_c, dtype=np.float64),
                                    ps.shape)[k : k + 1].copy(),
                    t.epsilon, t.feeds_spread,
                )
            )
        else:
            kept.append(t)

    cohort_ids = sorted({t.cohort_id for t in kept})
    delta_names = sorted({t.obs_name for t in kept})
    delta_pos = {name: i for i, name in enumerate(delta_names)}

    blocks: list = []
    for s, cid in enumerate(cohort_ids):
        members = [t for t in kept if t.cohort_id == cid]
        ns = {int(t.n) for t in members}
        if len(ns) > 1:
            raise ValueError(
                f"cohort '{cid}' mixes n={sorted(ns)}. A cohort is one patient set, "
                "so one n; split it or give the targets distinct cohort_ids."
            )
        y, p, col, didx, dvar, names = [], [], [], [], [], []
        for t in members:
            ps = np.asarray(t.p_levels, dtype=np.float64)
            vals = np.asarray(t.values, dtype=np.float64)
            if vals.shape != ps.shape:
                raise ValueError(
                    f"target '{t.obs_name}' has {vals.size} values for {ps.size} p_levels"
                )
            sc = np.broadcast_to(np.asarray(t.sigma_c, dtype=np.float64), ps.shape)
            y.append(vals)
            p.append(ps)
            col.append(np.full(ps.size, int(t.obs_index), dtype=np.int64))
            didx.append(np.full(ps.size, delta_pos[t.obs_name], dtype=np.int64))
            dvar.append(sc**2 + float(t.epsilon) ** 2)
            names.append(t.obs_name)

        kw = dict(dtype=_DTYPE, device=device)
        blocks.append(
            StudyBlock(
                cohort_id=cid,
                n=int(next(iter(ns))),
                study_index=s,
                obs_names=names,
                y=torch.tensor(np.concatenate(y), **kw),
                p=torch.tensor(np.concatenate(p), **kw),
                col=torch.tensor(np.concatenate(col), dtype=torch.long, device=device),
                delta_index=torch.tensor(np.concatenate(didx), dtype=torch.long, device=device),
                diag_var=torch.tensor(np.concatenate(dvar), **kw),
            )
        )
    return blocks, cohort_ids, delta_names


def _interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """Linear interpolation of ``fp`` at ``x``, with ``xp`` ascending.

    Saturates at the end knots rather than extrapolating. Differentiable in both
    ``xp`` and ``fp``; the bracket indices are piecewise constant, which is
    exactly the almost-everywhere differentiability the sorted quantile map has.
    """
    i1 = torch.searchsorted(xp.contiguous(), x.contiguous()).clamp(1, xp.numel() - 1)
    i0 = i1 - 1
    lo, hi = xp[i0], xp[i1]
    t = ((x - lo) / (hi - lo).clamp_min(1e-300)).clamp(0.0, 1.0)
    return fp[i0] + t * (fp[i1] - fp[i0])


def _weighted_quantile(x_sorted: Tensor, p: Tensor, w_sorted: Optional[Tensor]) -> Tensor:
    """Population quantiles of an already-sorted sample, optionally weighted.

    Plotting positions are ``(i + 0.5) / N`` unweighted and the weighted
    generalisation ``(cumsum(w) - w/2) / sum(w)``. The estimator choice is
    deliberately *not* :data:`~qsp_inference.targets.anchors.QUANTILE_METHOD`:
    that constant matches the observed anchor to the *cohort* summary, both at
    small ``n``, whereas this is a plug-in for the **population** quantile over
    thousands of simulated patients, where estimators differ by O(1/N).
    """
    if w_sorted is None:
        n = x_sorted.numel()
        u = (torch.arange(n, dtype=x_sorted.dtype, device=x_sorted.device) + 0.5) / n
    else:
        cw = torch.cumsum(w_sorted, 0)
        u = (cw - 0.5 * w_sorted) / cw[-1].clamp_min(1e-300)
    return _interp(p, u, x_sorted)


def _jittered_cholesky(cov: Tensor) -> Tensor:
    """Cholesky factor with escalating jitter.

    The anchor covariance is genuinely near-singular when two anchors sit close
    together or two observables are near-duplicates, so a failed factorisation is
    an expected event rather than a bug.
    """
    m = cov.shape[0]
    eye = torch.eye(m, dtype=cov.dtype, device=cov.device)
    scale = torch.diagonal(cov).mean().detach().clamp_min(1e-300)
    for k in range(-12, -3):
        chol, info = torch.linalg.cholesky_ex(cov + (10.0**k) * scale * eye)
        if int(info) == 0:
            return chol
    raise torch.linalg.LinAlgError(
        "anchor covariance is not positive definite even with 1e-4 relative jitter"
    )


def _mvn_logpdf(resid: Tensor, chol: Tensor) -> Tensor:
    """Zero-mean Gaussian log-density from a Cholesky factor."""
    sol = torch.cholesky_solve(resid[:, None], chol)[:, 0]
    logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return -0.5 * (resid @ sol + logdet + chol.shape[0] * float(np.log(_TWO_PI)))


@dataclass
class SummaryLikelihood:
    """``log p(observed anchors | phi)`` for a set of study blocks.

    The forward map is: draw a population from ``phi`` under **common random
    numbers**, push it through the emulator, read population quantiles and
    densities off the resulting cloud, and score the observed anchors against
    them under the covariance above. Holding ``z`` fixed is what makes
    ``phi -> Q(phi)`` a smooth deterministic function with no Monte-Carlo noise
    across MCMC steps.

    Args:
        blocks: from :func:`build_study_blocks`.
        predict: ``(N, P) log-theta -> (N, J)`` observables **on the working
            scale**, differentiable and vectorised. Wrap the emulator's native
            output space here (e.g. ``log(scale * sinh(asinh_pred))``).
        z: ``(N, P)`` fixed standard-normal draws. One row is one virtual
            patient; ``N`` sets the Monte-Carlo resolution of ``Q(phi)``.
        rho: ``(J, J)`` normal-score correlation, from
            :func:`normal_score_correlation`. ``None`` means independent
            observables, i.e. block-diagonal within each observable -- the
            overconfident default, allowed only for a single-observable cohort
            set.
        basis: optional ``(P, P)`` spread eigenbasis ``W`` so that
            ``log theta = mu + W (sigma_u * z)``. ``None`` is the identity, i.e.
            ``sigma_u`` is a per-parameter log-sd.
        bandwidth: quantile spacing ``h`` for the density plug-in
            ``f = 2h / (Q(p+h) - Q(p-h))``. Scalar, or one per emulator column.
            The measured sweet spot on the PDAC cloud is ~0.035, but this is not
            one global number to declare once: it wants a per-observable rule and
            a per-observable check, because at ``h = 0.035`` two heavy-tailed
            fold-change marginals had their asserted variance ~18% too large
            while well-behaved ones stayed within 2%.
        viability: optional ``(N, P) log-theta -> (N,)`` weight in ``(0, 1)``,
            the *smooth* viability probability rather than a hard filter. A hard
            filter changes the *membership* of the set being sorted, and sorting
            a variable-length set is where the non-smoothness actually bites --
            not the normalizer. See
            ``inference/restriction.py:RestrictionClassifier.score``.
        include_mc_error: fold the Monte-Carlo error of ``Q(phi)`` into the
            covariance by replacing ``n`` with ``1/(1/n + 1/N)``. Small (``n/N``)
            and conservative: under common random numbers that error is a fixed
            offset shared across ``phi`` rather than fresh noise, so this
            overstates it slightly.
    """

    blocks: list
    predict: Callable[[Tensor], Tensor]
    z: Tensor
    rho: Optional[Tensor] = None
    basis: Optional[Tensor] = None
    bandwidth: float | np.ndarray = 0.035
    viability: Optional[Callable[[Tensor], Tensor]] = None
    include_mc_error: bool = True

    _queries: list = field(init=False, repr=False, default_factory=list)
    _rows: list = field(init=False, repr=False, default_factory=list)
    _frozen_chol: Optional[list] = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("no study blocks")
        self.z = torch.as_tensor(self.z, dtype=_DTYPE)
        dev = self.z.device
        n_param = int(self.z.shape[1])
        if self.basis is not None:
            self.basis = torch.as_tensor(self.basis, dtype=_DTYPE)
            if self.basis.shape != (n_param, n_param):
                raise ValueError(f"basis must be ({n_param}, {n_param})")

        cols = sorted({int(c) for b in self.blocks for c in b.col.tolist()})
        n_col_max = max(cols) + 1

        h = np.broadcast_to(np.asarray(self.bandwidth, dtype=np.float64), (n_col_max,))
        if np.any(h <= 0) or np.any(h >= 0.5):
            raise ValueError("bandwidth must lie in (0, 0.5)")

        if self.rho is None:
            self.rho = torch.eye(n_col_max, dtype=_DTYPE, device=dev)
        else:
            self.rho = torch.as_tensor(self.rho, dtype=_DTYPE, device=dev)
            if self.rho.shape[0] < n_col_max:
                raise ValueError(
                    f"rho is {tuple(self.rho.shape)} but targets reference column {n_col_max - 1}"
                )

        # One query table per observable column: the anchor levels themselves and
        # the two density shoulders p +/- h, concatenated. Built once so a
        # log_prob call is a single emulator forward and a single sort, and the
        # per-block row lookups are plain gathers into the concatenated result.
        offsets, cursor = {}, 0
        for c in cols:
            ps = sorted({float(p) for b in self.blocks
                         for p, cc in zip(b.p.tolist(), b.col.tolist()) if int(cc) == c})
            ps = np.asarray(ps, dtype=np.float64)
            lo = np.clip(ps - h[c], 1e-6, 1.0 - 1e-6)
            hi = np.clip(ps + h[c], 1e-6, 1.0 - 1e-6)
            query = torch.tensor(np.concatenate([ps, lo, hi]), dtype=_DTYPE, device=dev)
            self._queries.append((int(c), query))
            offsets[c] = (cursor, {float(p): i for i, p in enumerate(ps)}, len(ps))
            cursor += query.numel()

        for b in self.blocks:
            at, at_lo, at_hi, h_row = [], [], [], []
            for p, c in zip(b.p.tolist(), b.col.tolist()):
                base, pos, k = offsets[int(c)]
                i = pos[float(p)]
                at.append(base + i)
                at_lo.append(base + k + i)
                at_hi.append(base + 2 * k + i)
                h_row.append(h[int(c)])
            self._rows.append(
                (
                    torch.tensor(at, dtype=torch.long, device=dev),
                    torch.tensor(at_lo, dtype=torch.long, device=dev),
                    torch.tensor(at_hi, dtype=torch.long, device=dev),
                    torch.tensor(h_row, dtype=_DTYPE, device=dev),
                )
            )

    # -- population --------------------------------------------------------

    @property
    def n_patients(self) -> int:
        return int(self.z.shape[0])

    @property
    def n_param(self) -> int:
        return int(self.z.shape[1])

    @property
    def n_studies(self) -> int:
        return len(self.blocks)

    def draw_population(self, mu: Tensor, sigma_u: Tensor) -> Tensor:
        """``log theta = mu + W (sigma_u * z)`` under the fixed common random
        numbers. Deterministic and smooth in ``phi`` by construction."""
        u = sigma_u * self.z
        if self.basis is not None:
            u = u @ self.basis.T
        return mu + u

    def population_observables(self, mu: Tensor, sigma_u: Tensor) -> Tensor:
        """``(N, J)`` working-scale observables for the population at ``phi``."""
        return self.predict(self.draw_population(mu, sigma_u))

    def _query_quantiles(self, x: Tensor, w: Optional[Tensor]) -> Tensor:
        """Evaluate the precomputed query table on a population cloud.

        One sort over the whole cloud, then a linear interpolation per observable.
        The result is concatenated in the same column order the constructor used,
        so the per-block index tensors gather straight out of it.
        """
        x_sorted, order = torch.sort(x, dim=0)
        parts = [
            _weighted_quantile(
                x_sorted[:, c], query, None if w is None else w[order[:, c]]
            )
            for c, query in self._queries
        ]
        return torch.cat(parts)

    def refresh_rho(self, mu: Tensor, sigma_u: Tensor) -> np.ndarray:
        """Re-estimate ``rho`` on the population at ``phi`` and install it.

        Call this between outer rounds, never inside the sampler: a live copula
        would put rank flips in the potential. Initialising ``rho`` from a
        deliberately generous proposal cloud overstates the cross-observable
        dependence, so one refresh at the fitted ``phi`` is worth doing.
        """
        with torch.no_grad():
            x = self.population_observables(mu, sigma_u).cpu().numpy()
        rho = normal_score_correlation(x)
        self.rho = torch.as_tensor(rho, dtype=_DTYPE, device=self.z.device)
        return rho

    # -- the covariance: parametric or plug-in -----------------------------

    def freeze_covariance_at(self, mu: Tensor, sigma_u: Tensor) -> "SummaryLikelihood":
        """Evaluate the anchor covariance once at ``phi`` and hold it fixed,
        turning the likelihood into a generalised-least-squares objective.

        **Do this.** The unfrozen form is a correct parametric density, but it
        biases the spread low, because ``Sigma`` scales with the population width
        (``f ~ 1/sigma``, so ``Sigma ~ sigma^2``) and the ``-0.5 log det Sigma``
        term therefore rewards a narrower population on its own. Measured on a
        Gaussian population with three anchors, fitting real ``n``-cohort sample
        quantiles, ``mean(sigma_hat)/sigma_true``:

        ====  ========  ========
         n     live     frozen
        ====  ========  ========
          20     0.878     0.995
          40     0.941     0.999
         120     0.963     1.001
        ====  ========  ========

        That is the same **low bias on the population spread**, at the same
        published ``n``, that :data:`~qsp_inference.targets.anchors.QUANTILE_METHOD`
        exists to remove on the summary side, arriving through a third door. It is
        an ``O(1/n)`` effect and it does not vanish at the ``n`` QSP targets carry.

        Freezing is not a fudge: it is the standard estimating-equation reading of
        this likelihood, in which ``Sigma`` is a **weight** saying how much each
        target counts and not part of the parametric family. It is also what makes
        docs ch. 4b's claim about the density plug-in literally true -- that an
        error in ``f_j`` is a precision-misallocation problem rather than a
        correctness one. With ``Sigma`` live, an error in ``f_j`` moves ``phi``.

        **Where to freeze barely matters, so freeze once and move on.** The
        obvious worry is circularity: ``Sigma`` needs the population you are
        fitting. Measured on a curved forward map at ``n = 20``, against a
        ``phi_0`` twice too wide, ``phi_hat`` moved by only 0.14 to 0.32 of one
        sampling standard deviation and the recovered spread was unchanged (0.966
        either way). A wrong plug-in misallocates precision across targets; it
        does not bias the answer. Refitting at ``phi_hat`` to check that is a
        diagnostic worth running once, not a loop to iterate to convergence.

        Freezing also makes every subsequent evaluation cheaper: the density, the
        copula quadrature and the Cholesky all leave the inner loop.
        """
        with torch.no_grad():
            self._frozen_chol = None
            log_theta = self.draw_population(mu, sigma_u)
            w = None if self.viability is None else self.viability(log_theta)
            q = self._query_quantiles(self.predict(log_theta), w)
            self._frozen_chol = [
                _jittered_cholesky(self._block_covariance(b, rows, q))
                for b, rows in zip(self.blocks, self._rows)
            ]
        return self

    def unfreeze_covariance(self) -> "SummaryLikelihood":
        """Go back to the fully parametric density. See the bias table above."""
        self._frozen_chol = None
        return self

    @property
    def covariance_is_frozen(self) -> bool:
        return self._frozen_chol is not None

    def _block_covariance(self, block: StudyBlock, rows: tuple, q: Tensor) -> Tensor:
        at, at_lo, at_hi, h_row = rows
        f = 2.0 * h_row / (q[at_hi] - q[at_lo]).clamp_min(1e-12)
        n_eff = block.n
        if self.include_mc_error:
            n_eff = 1.0 / (1.0 / block.n + 1.0 / self.n_patients)
        return anchor_covariance(block.p, block.col, f, self.rho, n_eff) + torch.diag(
            block.diag_var
        )

    # -- likelihood --------------------------------------------------------

    def log_prob(
        self,
        mu: Tensor,
        sigma_u: Tensor,
        eta: Optional[Tensor] = None,
        delta: Optional[Tensor] = None,
    ) -> Tensor:
        """Total log-likelihood of the observed anchors.

        Args:
            mu: ``(P,)`` population center in log-theta space.
            sigma_u: ``(P,)`` population spread, in the eigenbasis when ``basis``
                is set. Strictly positive.
            eta: ``(S,)`` per-study offsets, ordered as ``blocks``. The shared
                systematic shift.
            delta: ``(D,)`` per-observable offsets, ordered as ``delta_names``
                from :func:`build_study_blocks`. The structural discrepancy.
        """
        log_theta = self.draw_population(mu, sigma_u)
        x = self.predict(log_theta)
        w = None if self.viability is None else self.viability(log_theta)
        q = self._query_quantiles(x, w)

        # NaN guard, the same contract submodel/inference.py uses one level down:
        # an emulator asked for a theta far outside the pool can return non-finite
        # observables, and the sampler's job is to reject that trajectory. Catching
        # it here rather than at the Cholesky keeps a genuine non-PSD covariance --
        # which would be a bug in the formula -- loud.
        if not torch.isfinite(q).all():
            return torch.full((), -float("inf"), dtype=x.dtype, device=x.device)

        total = x.new_zeros(())
        for i, (b, rows) in enumerate(zip(self.blocks, self._rows)):
            chol = (
                self._frozen_chol[i]
                if self._frozen_chol is not None
                else _jittered_cholesky(self._block_covariance(b, rows, q))
            )
            resid = b.y - q[rows[0]]
            if eta is not None:
                resid = resid - eta[b.study_index]
            if delta is not None:
                resid = resid - delta[b.delta_index]
            total = total + _mvn_logpdf(resid, chol)
        return total

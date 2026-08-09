"""The population-spread (omega) eigenbasis: the identifiable-omega selector.

Per-parameter between-subject spread omega is structurally non-identifiable from
cohort summaries -- only a handful of *combination* directions carry signal, so a
hierarchical NLME fit that tried to infer one omega per parameter would set the
rest arbitrarily and over-disperse the population. The fix is to infer omega in a
low-dimensional identifiable subspace and let every other direction inherit its
anchored prior spread. This module builds that subspace.

Two operators meet here, in two different metrics:

- **The data side** -- a per-observable sensitivity ``J_i = d asinh(obs_i)/d log
  theta`` (a local linear fit), whitened by that observable's measurement/finite-n
  noise, stacked into the observation-precision-weighted Fisher ``G = sum_i
  J_i^T Sigma_obs,i^{-1} J_i`` (:func:`sensitivity_gram`). ``G`` says which
  parameter *combinations* the observables can resolve. It is computed once,
  offline, from a simulation pool; it does not depend on the prior.

- **The prior side** -- the population prior covariance ``Gamma = Cov_pi[log
  theta]`` (correlations from the anchored composite prior, marginals at the
  population omega scale). ``Gamma`` says how much spread the population actually
  has in each direction, so it is the natural *metric* on log-theta.

:func:`prior_metric_eigenbasis` combines them: it eigendecomposes ``G`` **in the
prior metric** (the active-subspace / likelihood-informed-subspace construction),
ranking directions by data information *per unit of population-prior variance*
rather than per unit of raw log-theta. Concretely it whitens with ``L = chol
(Gamma)``, eigendecomposes ``L^T G L``, and maps the directions back. The top
directions are where the data is informative relative to what the prior already
knows -- the honest identifiable subspace under the anchored base, and one that
respects the prior's correlations, not just its marginal scales.

The returned basis is used in the runner's rotated-spread draw ``log theta = mu +
W (sigma_u * z)`` with ``z ~ N(0, I)`` per direction. Because the eigenvectors are
orthonormal in the *whitened* space, ``sigma_u = 1`` on every direction
reproduces the population covariance ``Gamma`` exactly (``W W^T = Gamma``); the
inferred top-K directions then deviate from 1 only where the data moves them, and
the frozen complement rides ``sigma_u ~ 1`` (full anchored spread). This is why
the per-direction prior spread comes out ~1 by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

__all__ = [
    "fit_local_jacobian",
    "whiten_sensitivity_rows",
    "sensitivity_gram",
    "prior_covariance",
    "prior_metric_eigenbasis",
    "PriorMetricEigenbasis",
]


def fit_local_jacobian(
    log_theta: np.ndarray,
    y: np.ndarray,
    *,
    center: np.ndarray,
    scale: Optional[np.ndarray] = None,
    localize: bool = True,
    min_ess: float = 3000.0,
    bandwidths: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
):
    """Weighted linear sensitivity ``J = d y / d(log theta)`` at ``center``.

    Fits ``y ~ (log_theta - center) @ J^T`` by least squares. When ``localize`` is
    set the rows are Gaussian-kernel weighted toward ``center`` (bandwidth grown
    from ``bandwidths`` until the effective sample size clears ``min_ess``), so the
    fit is the *local* Jacobian at the operating point rather than a global secant.

    Args:
        log_theta: ``(n, P)`` log-parameter rows (the simulation pool).
        y: ``(n, n_obs)`` responses -- typically ``asinh(obs / scale)``, mean-centered.
        center: ``(P,)`` operating point in log-parameter space.
        scale: ``(P,)`` per-parameter scale for the localization distance (e.g. the
            pool's per-coordinate log-sd). Required when ``localize``; ignored otherwise.
        localize: local (kernel-weighted) fit vs a single global least squares.
        min_ess: target effective sample size for the localized fit.
        bandwidths: multipliers on the median squared distance, tried in order.

    Returns:
        ``(J, resid_std, ess, bw_mult)`` -- ``J`` is ``(n_obs, P)``; ``resid_std``
        is the per-observable held-out residual SD (a model-floor estimate);
        ``ess`` the effective sample size; ``bw_mult`` the chosen bandwidth
        multiplier (``None`` for a global fit).
    """
    log_theta = np.asarray(log_theta, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    Ac = log_theta - center

    if not localize:
        coef, _, _, _ = np.linalg.lstsq(Ac, y, rcond=None)
        resid = y - Ac @ coef
        return coef.T, resid.std(0), float(len(log_theta)), None

    if scale is None:
        raise ValueError("scale is required for a localized Jacobian fit")
    scale = np.asarray(scale, dtype=np.float64)
    d2 = ((Ac / scale) ** 2).sum(1)
    bw_mult = None
    w = None
    for bw_mult in bandwidths:
        bw = bw_mult * np.median(d2)
        w = np.exp(-d2 / (2 * bw))
        ess = w.sum() ** 2 / (w ** 2).sum()
        if ess >= min_ess:
            break
    ess = float(w.sum() ** 2 / (w ** 2).sum())
    ws = np.sqrt(w)[:, None]
    coef, _, _, _ = np.linalg.lstsq(Ac * ws, y * ws, rcond=None)
    resid = y - Ac @ coef
    return coef.T, resid.std(0), ess, bw_mult


def whiten_sensitivity_rows(
    J: np.ndarray,
    resid_floor: np.ndarray,
    sample_size,
    *,
    omega0: float,
    se_iqr_c: float,
):
    """Whiten sensitivity rows by each observable's spread-estimation noise.

    Returns ``J_i / sig_i`` per observable, where ``sig_i`` combines the
    finite-``n`` sampling noise of the spread estimate with a per-observable model
    floor::

        se_iqr_i = se_iqr_c * omega0 * ||J_i|| / sqrt(n_i)      # finite-n IQR noise
        sig_i    = sqrt(se_iqr_i^2 + resid_floor_i^2)

    ``se_iqr_c`` is ``SE(IQR)/sigma`` for the assumed within-cohort law (``1.573``
    for a Gaussian, since ``IQR ~ 1.349 sigma`` and ``SE(IQR) ~ 1.573 sigma /
    sqrt(n)``); ``omega0`` is a reference spread that cancels from the direction and
    only sets the relative weight of sampling noise vs the floor. Using each
    observable's *real* ``n_i`` down-weights low-``n`` targets so a 6-donor panel
    constrains the basis less than a 900-donor one.

    Stacking these rows and forming :func:`sensitivity_gram` gives the
    observation-precision-weighted Fisher ``sum_i J_i^T J_i / sig_i^2`` -- the
    correct object for the rotated, correlated-spread model the runner draws from
    (``theta = exp(mu + W(sigma_u * z))``, observable variance ``sum_k sigma_u,k^2
    (J_i . w_k)^2``).
    """
    J = np.asarray(J, dtype=np.float64)
    resid_floor = np.asarray(resid_floor, dtype=np.float64)
    Jn = np.linalg.norm(J, axis=1) + 1e-12
    n = np.clip(np.asarray(sample_size, dtype=np.float64), 1.0, None)
    se_iqr = se_iqr_c * omega0 * Jn / np.sqrt(n)
    sig = np.sqrt(se_iqr ** 2 + resid_floor ** 2)
    return J / sig[:, None]


def sensitivity_gram(rows: np.ndarray) -> np.ndarray:
    """The Gram ``rows^T @ rows`` -- the observation-precision-weighted Fisher.

    ``rows`` is the whitened sensitivity stacked across observables (and scenarios),
    ``(n_obs_total, P)``; the returned ``(P, P)`` matrix is symmetric PSD.
    """
    rows = np.asarray(rows, dtype=np.float64)
    G = rows.T @ rows
    return 0.5 * (G + G.T)


def prior_covariance(deviations: np.ndarray, *, ridge: float = 1e-6) -> np.ndarray:
    """Regularized covariance of log-parameter deviations from prior draws.

    ``deviations`` is ``(n_draws, P)`` centered log-parameter draws (already scaled
    to the population spread if an omega override applies). A relative ``ridge`` is
    added on the diagonal (``ridge * mean(diag)``) so the Cholesky in
    :func:`prior_metric_eigenbasis` is well-conditioned even when some directions
    are nearly degenerate.
    """
    dev = np.asarray(deviations, dtype=np.float64)
    Gamma = np.cov(dev, rowvar=False)
    Gamma = np.atleast_2d(Gamma)
    Gamma = 0.5 * (Gamma + Gamma.T)
    P = Gamma.shape[0]
    jitter = ridge * float(np.mean(np.diag(Gamma)))
    return Gamma + jitter * np.eye(P)


@dataclass
class PriorMetricEigenbasis:
    """The population-spread eigenbasis under the prior metric.

    Attributes:
        draw_matrix: ``W`` ``(P, P)``. The rotated-spread draw is ``log theta = mu +
            W @ (sigma_u * z)`` with ``z ~ N(0, I)``. Columns are the identifiable
            directions, eigenvalue-descending; ``sigma_u = 1`` on all of them
            reproduces the population covariance (``W W^T = Gamma``).
        project_matrix: ``W^{-T}`` ``(P, P)``. Recovers the spread coordinates of a
            deviation: ``u = deviations @ project_matrix`` gives ``u = W^{-1} dev``,
            so its per-column std is the population-prior spread per direction.
        sigma_u_prior: ``(P,)`` per-direction population-prior spread (~1 by
            construction; the small departures come from the ridge and finite draws).
        eigenvalues: ``(P,)`` whitened-Fisher spectrum, descending. The information
            each direction carries per unit of population-prior variance -- the
            honest identifiability ranking.
        n_directions: ``K``, how many leading directions are treated as inferred
            (the rest freeze at their anchored ``sigma_u_prior``).
    """

    draw_matrix: np.ndarray
    project_matrix: np.ndarray
    sigma_u_prior: np.ndarray
    eigenvalues: np.ndarray
    n_directions: int


def prior_metric_eigenbasis(
    gram: np.ndarray,
    prior_cov: np.ndarray,
    *,
    n_directions: Optional[int] = None,
) -> PriorMetricEigenbasis:
    """Eigendecompose the data Fisher ``gram`` in the population-prior metric.

    Implements the active-subspace / likelihood-informed-subspace construction:
    with ``Gamma = prior_cov = L L^T`` the population-prior covariance, it
    eigendecomposes the *prior-whitened* Fisher ``L^T G L = V~ Lambda V~^T`` and
    maps the orthonormal whitened directions back to log-parameter space as ``W =
    L V~`` (see the module docstring for why this is the right object).

    Args:
        gram: ``(P, P)`` observation-precision-weighted Fisher from
            :func:`sensitivity_gram` -- the data side, prior-independent.
        prior_cov: ``(P, P)`` population-prior covariance ``Cov_pi[log theta]`` from
            :func:`prior_covariance` -- the metric. Its correlations *and* marginal
            scales shape the directions.
        n_directions: how many leading directions to mark inferred (``K``); all
            ``P`` if ``None``.

    Returns:
        A :class:`PriorMetricEigenbasis`.
    """
    G = np.asarray(gram, dtype=np.float64)
    Gamma = np.asarray(prior_cov, dtype=np.float64)
    P = G.shape[0]
    if G.shape != (P, P) or Gamma.shape != (P, P):
        raise ValueError(f"gram {G.shape} and prior_cov {Gamma.shape} must both be (P, P)")

    L = np.linalg.cholesky(Gamma)                        # Gamma = L L^T, lower
    Gt = L.T @ G @ L                                     # whitened Fisher
    Gt = 0.5 * (Gt + Gt.T)
    evals, Vt = np.linalg.eigh(Gt)                       # ascending
    evals = evals[::-1]
    Vt = Vt[:, ::-1]                                     # orthonormal (whitened space)

    W = L @ Vt                                           # draw matrix, columns = L v~_k
    W_inv_t = np.linalg.solve(L.T, Vt)                   # L^{-T} V~ = W^{-T}
    # Per-direction population-prior spread: std of u = W^{-1} dev under the prior,
    # = sqrt(diag(W_inv_t^T Gamma W_inv_t)). ~1 by construction; keep it exact so
    # the ridge/rescale show through rather than being assumed away.
    sigma_u_prior = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", W_inv_t.T, Gamma, W_inv_t.T), 0.0, None))

    K = int(P if n_directions is None else min(n_directions, P))
    return PriorMetricEigenbasis(
        draw_matrix=W,
        project_matrix=W_inv_t,
        sigma_u_prior=sigma_u_prior,
        eigenvalues=evals,
        n_directions=K,
    )

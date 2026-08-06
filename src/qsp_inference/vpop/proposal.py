"""The population proposal ``pi~`` and its data-anchored truncation.

The population-inference loop (Ch. 4) separates a *scientific* distribution from a
*computational* one. The anchored population base ``pi`` is a log-normal in the
spread eigenbasis, ``log theta ~ N(mu, W diag(sigma_u^2) W^T)`` with ``sigma_u`` the
eigenbasis prior spread (so ``W W^T = Gamma_pop``). The **proposal** ``pi~`` is the
*same* object with ``sigma_u`` widened on the identified directions only -- wide
where the data can move the spread, unchanged (anchored) on the sloppy complement.
Reference simulations and cohort patients are drawn from ``pi~``; the reported
population is put back **under ``pi``** by the self-normalized importance weight
``w = pi/pi~`` (:mod:`qsp_inference.inference.importance`).

Two pieces live here:

- :class:`EigenbasisPopulation` -- the log-normal population/proposal. It samples
  ``theta`` and evaluates ``log pi(log theta)`` analytically (a Gaussian in
  log-space, since ``W`` is invertible), and carries ``param_names`` so it plugs
  straight into the importance reweight with the alignment checks on. Both ``pi``
  and ``pi~`` are instances; :func:`widen_on_identified` builds ``pi~`` from ``pi``.

- :func:`reachable_accept_fn` -- the round-to-round truncation. "Reachable" for a
  distribution-valued target is defined in **observable space against the data**:
  keep ``theta`` whose (emulated) observables land inside the observed envelope.
  Anchoring to the data -- not to the current fit -- is what keeps the truncation
  from feeding back into the very spread we infer (see the guide's reachable-set
  section). The envelope must be *generous* (the observed distribution's support,
  not the median+/-IQR core) or the population's legitimate tail patients get cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "EigenbasisPopulation",
    "widen_on_identified",
    "reachable_accept_fn",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


@dataclass
class EigenbasisPopulation:
    """A log-normal population (or proposal) in the spread eigenbasis.

    ``log theta ~ N(mu, W diag(sigma_u^2) W^T)``. With ``sigma_u`` the eigenbasis
    ``sigma_u_prior`` this is the anchored base ``pi`` (covariance = ``Gamma_pop``);
    with ``sigma_u`` widened on the identified directions it is the proposal ``pi~``
    (see :func:`widen_on_identified`).

    Attributes:
        mu: ``(P,)`` log-space center.
        draw_matrix: ``(P, P)`` ``W`` from
            :class:`~qsp_inference.vpop.eigenbasis.PriorMetricEigenbasis`.
        sigma_u: ``(P,)`` per-direction spread (top-K inferred/widened, rest anchored).
        param_names: ``P`` names, so the importance reweight can check alignment.
    """

    mu: np.ndarray
    draw_matrix: np.ndarray
    sigma_u: np.ndarray
    param_names: Sequence[str]
    _w_inv: np.ndarray = field(init=False, repr=False)
    _log_det_w: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64).ravel()
        self.draw_matrix = np.asarray(self.draw_matrix, dtype=np.float64)
        self.sigma_u = np.asarray(self.sigma_u, dtype=np.float64).ravel()
        self.param_names = list(self.param_names)
        P = self.mu.shape[0]
        if self.draw_matrix.shape != (P, P):
            raise ValueError(
                f"draw_matrix must be ({P}, {P}); got {self.draw_matrix.shape}"
            )
        if self.sigma_u.shape != (P,):
            raise ValueError(f"sigma_u must be ({P},); got {self.sigma_u.shape}")
        if len(self.param_names) != P:
            raise ValueError(
                f"param_names has {len(self.param_names)} entries; expected {P}"
            )
        if np.any(self.sigma_u <= 0) or np.any(~np.isfinite(self.sigma_u)):
            raise ValueError("sigma_u must be finite and positive")
        sign, logabsdet = np.linalg.slogdet(self.draw_matrix)
        if sign == 0 or not np.isfinite(logabsdet):
            raise ValueError("draw_matrix is singular; cannot form a density")
        self._w_inv = np.linalg.inv(self.draw_matrix)
        self._log_det_w = float(logabsdet)

    @property
    def n_params(self) -> int:
        return self.mu.shape[0]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` patients ``theta`` (original space), shape ``(n, P)``."""
        z = rng.standard_normal((int(n), self.n_params))
        log_theta = self.mu + (self.sigma_u * z) @ self.draw_matrix.T
        return np.exp(log_theta)

    def log_prob(self, log_theta) -> np.ndarray:
        """Log density at ``log_theta`` (log-space), shape ``(n,)``.

        Accepts a numpy array or a torch tensor of shape ``(n, P)`` (or ``(P,)``),
        so it satisfies the :mod:`~qsp_inference.inference.importance` contract
        (``log_prob`` on log-space draws). Analytic: ``u = W^{-1}(log_theta - mu)``,
        ``log p = -0.5[P log 2pi + 2 log|det W| + 2 sum_k log sigma_u,k
        + sum_k (u_k/sigma_u,k)^2]``.
        """
        lt = _to_numpy(log_theta)
        if lt.ndim == 1:
            lt = lt[None, :]
        if lt.ndim != 2 or lt.shape[1] != self.n_params:
            raise ValueError(
                f"log_theta must be (n, {self.n_params}); got {lt.shape}"
            )
        u = (lt - self.mu) @ self._w_inv.T                 # (n, P) = W^{-1}(lt-mu)
        quad = np.sum((u / self.sigma_u) ** 2, axis=1)     # (n,)
        const = self.n_params * _LOG_2PI + 2.0 * self._log_det_w
        const = const + 2.0 * float(np.sum(np.log(self.sigma_u)))
        return -0.5 * (const + quad)


def widen_on_identified(
    base: EigenbasisPopulation, factor: float, n_directions: int
) -> EigenbasisPopulation:
    """Build the proposal ``pi~`` by widening ``base`` on the top-K directions.

    The eigenbasis directions are eigenvalue-descending, so the leading
    ``n_directions`` are the identified ones. ``pi~`` scales their ``sigma_u`` by
    ``factor`` (> 1) and leaves the sloppy complement anchored -- wide only where
    the data can move the spread. Same ``mu``/``draw_matrix``/``param_names``, so
    ``base`` (as ``pi``) and the result (as ``pi~``) reweight against each other
    with no misalignment.
    """
    if factor <= 0 or not np.isfinite(factor):
        raise ValueError("factor must be finite and positive")
    K = int(min(max(n_directions, 0), base.n_params))
    sigma_u = base.sigma_u.copy()
    sigma_u[:K] = sigma_u[:K] * float(factor)
    return EigenbasisPopulation(
        mu=base.mu,
        draw_matrix=base.draw_matrix,
        sigma_u=sigma_u,
        param_names=base.param_names,
    )


def reachable_accept_fn(
    predict: Callable[[np.ndarray, Sequence[str]], np.ndarray],
    param_names: Sequence[str],
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    min_fraction: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Truncation accept-fn: keep ``theta`` whose emulated observables are reachable.

    Returns ``accept(theta) -> bool mask`` for ``theta`` in *original* space (the
    restriction-resample convention). Each ``theta`` is emulated
    (``predict(log theta, param_names)``) and kept when at least
    ``min_fraction`` of its observables fall inside the observed envelope
    ``[lo, hi]``; a non-finite prediction counts as outside. Generosity lives in the
    envelope width, not in ``min_fraction`` -- pass the observed distribution's
    support so tail patients survive.

    Args:
        predict: ``(log_theta (n, P), param_names) -> obs (n, n_obs)`` -- the
            round's emulator ``predict``.
        param_names: parameter order ``predict`` expects.
        lo, hi: ``(n_obs,)`` observed envelope, in the emulator's output space.
        min_fraction: fraction of observables that must be inside to keep a
            ``theta`` (default 1.0 = all, with a generous envelope).
    """
    lo = np.asarray(lo, dtype=np.float64).ravel()
    hi = np.asarray(hi, dtype=np.float64).ravel()
    if lo.shape != hi.shape:
        raise ValueError(f"lo {lo.shape} and hi {hi.shape} must match")
    if np.any(hi < lo):
        raise ValueError("hi must be >= lo elementwise")
    names = list(param_names)
    frac = float(min_fraction)

    def accept(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        log_theta = np.log(np.clip(theta, 1e-300, None))
        obs = np.asarray(predict(log_theta, names), dtype=np.float64)
        if obs.shape[1] != lo.shape[0]:
            raise ValueError(
                f"predict returned {obs.shape[1]} observables; envelope has {lo.shape[0]}"
            )
        inside = np.isfinite(obs) & (obs >= lo) & (obs <= hi)
        return inside.mean(axis=1) >= frac

    return accept

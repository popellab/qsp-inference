"""eq:gcorr: the emulator's residual against the simulator, as an affine map.

The surrogate errs by an amount that varies with where in parameter space it is
read, so removing only the mean of that error leaves the part that moves across a
cloud, and that part is what a spread statistic reports. ``Gamma`` is that
dependence. It corrects the emulator's Jacobian and not only its level, which is
why an under-dispersed surrogate can be fixed by a correction carrying no
multiplicative term: under-dispersion is a residual that grows with ``theta``.

Fitted in the transform the net was trained in, so a caller passes
``psi(g-hat)`` and ``psi(g)`` and never the physical quantity. ``P`` exceeds the
number of solves a correction can afford, hence ridge; the penalty is not a free
choice and :func:`gcorr_path` exists to make it on held-out residual.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

__all__ = ["GCorr", "fit_gcorr", "gcorr_path", "apply_gcorr",
           "save_gcorr", "load_gcorr", "reorder_gcorr"]


@dataclass(frozen=True)
class GCorr:
    """``alpha`` and ``Gamma`` of eq:gcorr, with the pool they were fitted over."""

    alpha: np.ndarray            # (Q,)
    Gamma: np.ndarray            # (Q, P)
    mu_0: np.ndarray             # (P,) the centre the fit is expanded about
    lam: float                   # ridge penalty, on standardised predictors
    param_names: Tuple[str, ...]
    target_names: Tuple[str, ...]
    n_fit: int
    #: sd of ``log theta`` over the fitting pool, per parameter. Not used in the
    #: correction, carried because it is the width the outer loop of eq:gcorr
    #: compares ``||mu - mu_0||`` against.
    x_sd: np.ndarray


def _design(log_theta: np.ndarray, mu_0: np.ndarray):
    """Standardised, centred predictors, plus what it takes to undo both."""
    X = np.asarray(log_theta, dtype=float) - np.asarray(mu_0, dtype=float)[None, :]
    # A constant column carries no information and its sd is zero, so it is held
    # at zero rather than divided by: the ridge would otherwise see 0/0.
    s = X.std(axis=0)
    s = np.where(s > 0, s, 1.0)
    Xs = X / s[None, :]
    return Xs, Xs.mean(axis=0), s


def _solve(Xc: np.ndarray, Rc: np.ndarray, lams: Sequence[float]) -> np.ndarray:
    """``(L, P, Q)`` ridge coefficients over a penalty path, one SVD for all."""
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    UtR = U.T @ Rc
    out = np.empty((len(lams), Xc.shape[1], Rc.shape[1]))
    for k, lam in enumerate(lams):
        out[k] = Vt.T @ ((S / (S ** 2 + lam))[:, None] * UtR)
    return out


def fit_gcorr(log_theta: np.ndarray, psi_hat: np.ndarray, psi_sim: np.ndarray,
              mu_0: np.ndarray, *, lam: float,
              param_names: Sequence[str] = (),
              target_names: Sequence[str] = ()) -> GCorr:
    """Ridge-fit eq:gcorr's residual over solved patients.

    ``psi_hat`` and ``psi_sim`` are ``(n, Q)`` in the net's transform. ``lam``
    penalises ``Gamma`` on standardised predictors and leaves ``alpha`` free,
    since an intercept shrunk toward zero would leave a level error behind.
    """
    R = np.asarray(psi_hat, dtype=float) - np.asarray(psi_sim, dtype=float)
    if not np.isfinite(R).all():
        raise ValueError("residual is not finite; filter the rows before fitting")
    Xs, xbar, s = _design(log_theta, mu_0)
    rbar = R.mean(axis=0)
    B = _solve(Xs - xbar[None, :], R - rbar[None, :], [lam])[0]   # (P, Q)
    return GCorr(
        alpha=rbar - xbar @ B,
        Gamma=(B / s[:, None]).T,
        mu_0=np.asarray(mu_0, dtype=float),
        lam=float(lam),
        param_names=tuple(param_names),
        target_names=tuple(target_names),
        n_fit=int(R.shape[0]),
        x_sd=s,
    )


def gcorr_path(log_theta: np.ndarray, psi_hat: np.ndarray, psi_sim: np.ndarray,
               mu_0: np.ndarray, lams: Sequence[float], *,
               folds: int = 5, seed: int = 0) -> dict:
    """K-fold held-out score for each penalty in ``lams``.

    The score is held-out residual variance after the correction over the same
    before it, averaged across outputs. Per output rather than pooled: the
    species span decades and a pooled figure would be one of them. ``alpha`` is
    refitted inside every fold, so 1.0 means ``Gamma`` earned nothing over the
    constant, and above 1.0 means it cost.
    """
    R = np.asarray(psi_hat, dtype=float) - np.asarray(psi_sim, dtype=float)
    Xs, _, _ = _design(log_theta, mu_0)
    n = R.shape[0]
    order = np.random.default_rng(seed).permutation(n)
    cut = np.array_split(order, folds)

    num = np.zeros((len(lams), R.shape[1]))
    den = np.zeros(R.shape[1])
    for f in range(folds):
        te = cut[f]
        tr = np.concatenate([cut[g] for g in range(folds) if g != f])
        xb, rb = Xs[tr].mean(axis=0), R[tr].mean(axis=0)
        B = _solve(Xs[tr] - xb[None, :], R[tr] - rb[None, :], lams)   # (L, P, Q)
        # The fold's own alpha, so the baseline is the constant correction and
        # the score reports Gamma alone.
        base = R[te] - rb[None, :]
        den += (base ** 2).sum(axis=0)
        for k in range(len(lams)):
            num[k] += ((base - (Xs[te] - xb[None, :]) @ B[k]) ** 2).sum(axis=0)
    ratio = num / np.where(den > 0, den, np.nan)
    return {"lams": list(map(float, lams)),
            "score": ratio.mean(axis=1).tolist(),
            "per_output": ratio.tolist()}


def apply_gcorr(psi_hat, log_theta, alpha, Gamma, mu_0):
    """eq:gcorr applied: ``psi(g-hat) - alpha - Gamma (theta - mu_0)``.

    Backend-agnostic. The arrays carry the arithmetic, so this is the same
    expression under numpy for the fit's own diagnostics and under jax inside the
    sampler.
    """
    return psi_hat - alpha - (log_theta - mu_0) @ Gamma.T


def reorder_gcorr(g: GCorr, param_names: Sequence[str]) -> GCorr:
    """Permute ``Gamma``'s columns and ``mu_0`` into ``param_names`` order.

    ``Gamma`` is indexed by parameter, so it travels with the same permutation
    ``reorder_arm`` applies to the first layer. Applying one without the other is
    silent: every shape still matches.
    """
    have = list(g.param_names)
    if have == list(param_names):
        return g
    if set(have) != set(param_names):
        missing = sorted(set(param_names) - set(have))
        raise ValueError(
            f"correction was fitted on a different parameter set; "
            f"{len(missing)} absent, e.g. {missing[:5]}"
        )
    at = {n: i for i, n in enumerate(have)}
    perm = np.array([at[n] for n in param_names])
    return GCorr(alpha=g.alpha, Gamma=g.Gamma[:, perm], mu_0=g.mu_0[perm],
                 lam=g.lam, param_names=tuple(param_names),
                 target_names=g.target_names, n_fit=g.n_fit, x_sd=g.x_sd[perm])


def save_gcorr(path: str | Path, g: GCorr) -> None:
    np.savez(Path(path), alpha=g.alpha, Gamma=g.Gamma, mu_0=g.mu_0,
             lam=np.asarray(g.lam), n_fit=np.asarray(g.n_fit), x_sd=g.x_sd,
             param_names=np.asarray(g.param_names),
             target_names=np.asarray(g.target_names))


def load_gcorr(path: str | Path) -> GCorr:
    d = np.load(Path(path), allow_pickle=False)
    return GCorr(alpha=d["alpha"], Gamma=d["Gamma"], mu_0=d["mu_0"],
                 lam=float(d["lam"]), n_fit=int(d["n_fit"]), x_sd=d["x_sd"],
                 param_names=tuple(d["param_names"].tolist()),
                 target_names=tuple(d["target_names"].tolist()))

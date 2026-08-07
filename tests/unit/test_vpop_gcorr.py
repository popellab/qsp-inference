"""eq:gcorr recovers an affine residual, and travels with the arm it corrects."""
from __future__ import annotations

import numpy as np
import pytest

from qsp_inference.vpop.gcorr import (
    apply_gcorr,
    fit_gcorr,
    gcorr_path,
    load_gcorr,
    reorder_gcorr,
    save_gcorr,
)

N, P, Q = 400, 12, 4
PARAMS = [f"p{i}" for i in range(P)]
TARGETS = [f"S{i}@0" for i in range(Q)]


def _problem(seed=0, noise=0.0):
    """An emulator whose residual is exactly ``alpha + Gamma (theta - mu_0)``."""
    rng = np.random.default_rng(seed)
    mu_0 = rng.standard_normal(P)
    log_theta = mu_0 + rng.standard_normal((N, P)) * rng.uniform(0.2, 2.0, P)
    alpha = rng.standard_normal(Q)
    Gamma = rng.standard_normal((Q, P)) * 0.3
    psi_sim = rng.standard_normal((N, Q)) * 2.0
    psi_hat = psi_sim + alpha + (log_theta - mu_0) @ Gamma.T
    if noise:
        psi_hat = psi_hat + rng.standard_normal((N, Q)) * noise
    return log_theta, psi_hat, psi_sim, mu_0, alpha, Gamma


def test_recovers_a_known_affine_residual():
    log_theta, psi_hat, psi_sim, mu_0, alpha, Gamma = _problem()
    g = fit_gcorr(log_theta, psi_hat, psi_sim, mu_0, lam=1e-10,
                  param_names=PARAMS, target_names=TARGETS)
    np.testing.assert_allclose(g.alpha, alpha, atol=1e-8)
    np.testing.assert_allclose(g.Gamma, Gamma, atol=1e-8)
    np.testing.assert_allclose(
        apply_gcorr(psi_hat, log_theta, g.alpha, g.Gamma, mu_0), psi_sim, atol=1e-8)


def test_gamma_carries_the_spread_and_alpha_alone_does_not():
    """The point of Gamma: a level correction leaves the shrinkage behind."""
    log_theta, psi_hat, psi_sim, mu_0, _, _ = _problem(seed=3)
    level_only = psi_hat - (psi_hat - psi_sim).mean(axis=0)
    g = fit_gcorr(log_theta, psi_hat, psi_sim, mu_0, lam=1e-10)
    full = apply_gcorr(psi_hat, log_theta, g.alpha, g.Gamma, mu_0)
    # Location is fixed by either; spread is fixed only by the full correction.
    assert abs(level_only.mean() - psi_sim.mean()) < 1e-8
    assert abs(level_only.std(0) / psi_sim.std(0) - 1).max() > 0.1
    assert abs(full.std(0) / psi_sim.std(0) - 1).max() < 1e-8


def test_path_score_reaches_the_irreducible_floor_and_one():
    """At no penalty only the noise survives; at a huge one, alpha stands alone."""
    noise = 0.5
    log_theta, psi_hat, psi_sim, mu_0, _, _ = _problem(seed=1, noise=noise)
    floor = (noise ** 2 / (psi_hat - psi_sim).var(axis=0)).mean()
    out = gcorr_path(log_theta, psi_hat, psi_sim, mu_0, [1e-6, 1e12], folds=4)
    assert out["score"][0] == pytest.approx(floor, rel=0.15)
    assert out["score"][1] == pytest.approx(1.0, abs=1e-3)


def test_path_penalises_a_gamma_fitted_to_noise():
    """Pure noise has no theta dependence, so held-out Gamma must cost."""
    rng = np.random.default_rng(7)
    mu_0 = np.zeros(P)
    log_theta = rng.standard_normal((60, P))
    psi_sim = rng.standard_normal((60, Q))
    psi_hat = psi_sim + rng.standard_normal((60, Q))
    out = gcorr_path(log_theta, psi_hat, psi_sim, mu_0, [1e-6, 1e12], folds=4)
    assert out["score"][0] > out["score"][1]


def test_reorder_permutes_gamma_with_its_parameters():
    log_theta, psi_hat, psi_sim, mu_0, _, _ = _problem(seed=2)
    g = fit_gcorr(log_theta, psi_hat, psi_sim, mu_0, lam=1.0,
                  param_names=PARAMS, target_names=TARGETS)
    perm = np.random.default_rng(0).permutation(P)
    h = reorder_gcorr(g, [PARAMS[i] for i in perm])
    np.testing.assert_allclose(
        apply_gcorr(psi_hat, log_theta, g.alpha, g.Gamma, g.mu_0),
        apply_gcorr(psi_hat, log_theta[:, perm], h.alpha, h.Gamma, h.mu_0),
        atol=1e-10)


def test_reorder_refuses_a_different_parameter_set():
    g = fit_gcorr(*_problem(seed=4)[:4], lam=1.0, param_names=PARAMS,
                  target_names=TARGETS)
    with pytest.raises(ValueError, match="different parameter set"):
        reorder_gcorr(g, PARAMS[:-1] + ["not_a_parameter"])


def test_round_trips_through_disk(tmp_path):
    g = fit_gcorr(*_problem(seed=5)[:4], lam=2.5, param_names=PARAMS,
                  target_names=TARGETS)
    save_gcorr(tmp_path / "g.npz", g)
    h = load_gcorr(tmp_path / "g.npz")
    np.testing.assert_allclose(h.alpha, g.alpha)
    np.testing.assert_allclose(h.Gamma, g.Gamma)
    np.testing.assert_allclose(h.mu_0, g.mu_0)
    assert (h.lam, h.n_fit) == (g.lam, g.n_fit)
    assert list(h.param_names) == PARAMS and list(h.target_names) == TARGETS


def test_non_finite_residual_is_refused():
    log_theta, psi_hat, psi_sim, mu_0, _, _ = _problem(seed=6)
    psi_hat = psi_hat.copy()
    psi_hat[3, 1] = np.inf
    with pytest.raises(ValueError, match="not finite"):
        fit_gcorr(log_theta, psi_hat, psi_sim, mu_0, lam=1.0)

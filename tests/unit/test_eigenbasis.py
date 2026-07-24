"""Unit tests for the population-spread eigenbasis under the prior metric.

The construction eigendecomposes the data Fisher G in the population-prior metric
Gamma (active subspace / LIS). The properties that must hold:

- sigma_u = 1 on every direction reproduces the population covariance (W W^T = Gamma);
- the per-direction prior spread comes out ~1 by construction;
- the metric changes which directions rank first (prior-aware, not raw log-theta);
- the identity-metric case recovers the ordinary eigendecomposition of G.
"""
import numpy as np
import pytest

from qsp_inference.vpop.eigenbasis import (
    fit_local_jacobian,
    whiten_sensitivity_rows,
    sensitivity_gram,
    prior_covariance,
    prior_metric_eigenbasis,
)


def _random_spd(P, rng, cond=5.0):
    A = rng.normal(size=(P, P))
    Q, _ = np.linalg.qr(A)
    eig = np.linspace(1.0, cond, P)
    return (Q * eig) @ Q.T


def test_draw_matrix_reproduces_prior_covariance():
    """W W^T == Gamma: a flat sigma_u=1 draw has exactly the population covariance."""
    rng = np.random.default_rng(0)
    P = 6
    Gamma = _random_spd(P, rng)
    G = _random_spd(P, rng, cond=20.0)
    eb = prior_metric_eigenbasis(G, Gamma)
    np.testing.assert_allclose(eb.draw_matrix @ eb.draw_matrix.T, Gamma, atol=1e-8)


def test_sigma_u_prior_is_unit():
    """The per-direction population-prior spread is ~1 by construction."""
    rng = np.random.default_rng(1)
    P = 8
    Gamma = _random_spd(P, rng)
    G = _random_spd(P, rng, cond=50.0)
    eb = prior_metric_eigenbasis(G, Gamma)
    np.testing.assert_allclose(eb.sigma_u_prior, np.ones(P), atol=1e-6)


def test_project_matrix_is_draw_inverse_transpose():
    """u = dev @ W^{-T} recovers the spread coordinates: W^{-1} W = I."""
    rng = np.random.default_rng(2)
    P = 5
    Gamma = _random_spd(P, rng)
    G = _random_spd(P, rng)
    eb = prior_metric_eigenbasis(G, Gamma)
    # project_matrix = W^{-T}, so project_matrix.T @ draw_matrix = I.
    np.testing.assert_allclose(eb.project_matrix.T @ eb.draw_matrix, np.eye(P), atol=1e-8)


def test_empirical_sigma_u_prior_unit_via_projection():
    """Draw from the population prior, project through W^{-T}: unit per-direction std."""
    rng = np.random.default_rng(3)
    P = 4
    Gamma = _random_spd(P, rng)
    G = _random_spd(P, rng, cond=30.0)
    eb = prior_metric_eigenbasis(G, Gamma)
    L = np.linalg.cholesky(Gamma)
    dev = rng.standard_normal((200000, P)) @ L.T       # Cov(dev) = Gamma
    u = dev @ eb.project_matrix
    np.testing.assert_allclose(u.std(0), np.ones(P), rtol=0.02)


def test_identity_metric_recovers_plain_eigendecomposition():
    """With Gamma = I the whitened Fisher is G itself: directions are G's eigenvectors,
    eigenvalues are G's, descending."""
    rng = np.random.default_rng(4)
    P = 6
    G = _random_spd(P, rng, cond=40.0)
    eb = prior_metric_eigenbasis(G, np.eye(P))
    ev_true = np.sort(np.linalg.eigvalsh(G))[::-1]
    np.testing.assert_allclose(eb.eigenvalues, ev_true, atol=1e-8)
    # W columns are eigenvectors of G (up to sign): G W = W diag(eigs).
    np.testing.assert_allclose(G @ eb.draw_matrix,
                               eb.draw_matrix * eb.eigenvalues[None, :], atol=1e-7)


def test_metric_changes_direction_ranking():
    """The prior metric re-ranks directions. A direction the data resolves well but
    the prior already pins tightly should lose rank to one the prior leaves wide."""
    # Data resolves axis 0 and axis 1 comparably (G ~ diag(1, 0.9)).
    G = np.diag([1.0, 0.9])
    # Prior pins axis 0 tightly (var 0.01) but leaves axis 1 wide (var 1.0).
    Gamma = np.diag([0.01, 1.0])
    eb = prior_metric_eigenbasis(G, Gamma)
    # Whitened Fisher eigenvalues: axis0 -> 1*0.01 = 0.01, axis1 -> 0.9*1.0 = 0.9.
    # So the top direction is axis 1 (wide prior), not axis 0 (tight prior).
    top = eb.draw_matrix[:, 0]
    assert abs(top[1]) > abs(top[0])         # leading direction points along axis 1
    # Under the identity metric axis 0 would have led (G eigenvalue 1 > 0.9).
    eb_id = prior_metric_eigenbasis(G, np.eye(2))
    top_id = eb_id.draw_matrix[:, 0]
    assert abs(top_id[0]) > abs(top_id[1])


def test_n_directions_clips():
    rng = np.random.default_rng(5)
    P = 7
    eb = prior_metric_eigenbasis(_random_spd(P, rng), _random_spd(P, rng), n_directions=3)
    assert eb.n_directions == 3
    eb_all = prior_metric_eigenbasis(_random_spd(P, rng), _random_spd(P, rng), n_directions=99)
    assert eb_all.n_directions == P


def test_shape_validation():
    with pytest.raises(ValueError, match="must both be"):
        prior_metric_eigenbasis(np.eye(3), np.eye(4))


def test_prior_covariance_is_regularized_and_symmetric():
    rng = np.random.default_rng(6)
    dev = rng.standard_normal((5000, 4))
    Gamma = prior_covariance(dev, ridge=1e-3)
    np.testing.assert_allclose(Gamma, Gamma.T)
    # Positive definite (Cholesky succeeds).
    np.linalg.cholesky(Gamma)


def test_sensitivity_gram_and_whitening():
    """whiten_sensitivity_rows + sensitivity_gram build a symmetric PSD Fisher, and
    a larger real n gives that observable more weight (smaller sig -> larger row)."""
    rng = np.random.default_rng(7)
    J = rng.normal(size=(3, 5))
    floor = np.full(3, 0.1)
    rows_small_n = whiten_sensitivity_rows(J, floor, [6, 6, 6], omega0=0.4, se_iqr_c=1.573)
    rows_big_n = whiten_sensitivity_rows(J, floor, [900, 900, 900], omega0=0.4, se_iqr_c=1.573)
    # Bigger n -> smaller sampling noise -> larger whitened rows (more weight).
    assert np.linalg.norm(rows_big_n) > np.linalg.norm(rows_small_n)
    G = sensitivity_gram(rows_big_n)
    np.testing.assert_allclose(G, G.T)
    assert (np.linalg.eigvalsh(G) >= -1e-10).all()


def test_fit_local_jacobian_recovers_linear_map():
    """A globally linear response is recovered exactly by the (global) fit."""
    rng = np.random.default_rng(8)
    P, n_obs, n = 4, 3, 2000
    Jtrue = rng.normal(size=(n_obs, P))
    log_theta = rng.normal(size=(n, P))
    center = log_theta.mean(0)
    y = (log_theta - center) @ Jtrue.T
    J, resid_std, ess, bw = fit_local_jacobian(log_theta, y, center=center, localize=False)
    np.testing.assert_allclose(J, Jtrue, atol=1e-8)
    assert resid_std.shape == (n_obs,) and (resid_std < 1e-8).all()
    assert bw is None and ess == n


def test_fit_local_jacobian_localizes():
    rng = np.random.default_rng(9)
    P, n = 3, 20000
    log_theta = rng.normal(size=(n, P))
    center = np.zeros(P)
    sd = log_theta.std(0)
    y = np.sin(log_theta[:, :1])          # nonlinear -> local fit differs from global
    J, resid_std, ess, bw = fit_local_jacobian(
        log_theta, y, center=center, scale=sd, localize=True, min_ess=3000
    )
    assert ess >= 3000 and bw is not None
    assert J.shape == (1, P)

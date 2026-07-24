"""Tests for the population proposal π̃ and its data-anchored truncation."""

import numpy as np
import pytest

from qsp_inference.vpop import (
    EigenbasisPopulation,
    widen_on_identified,
    reachable_accept_fn,
)
from qsp_inference.inference.importance import reweight_to_prior


def _random_basis(P=5, seed=0):
    """A random invertible draw matrix W, center mu, and per-direction spread."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((P, P))
    W += P * np.eye(P)                       # keep it well-conditioned / invertible
    mu = rng.standard_normal(P)
    sigma_u = 0.5 + rng.random(P)            # positive
    names = [f"p{j}" for j in range(P)]
    return mu, W, sigma_u, names


def test_sample_reproduces_the_log_covariance():
    mu, W, sigma_u, names = _random_basis(P=4, seed=1)
    pop = EigenbasisPopulation(mu, W, sigma_u, names)
    theta = pop.sample(400_000, np.random.default_rng(7))
    emp = np.cov(np.log(theta), rowvar=False)
    want = W @ np.diag(sigma_u**2) @ W.T
    assert np.linalg.norm(emp - want) / np.linalg.norm(want) < 0.02


def test_log_prob_matches_multivariate_normal():
    scipy_stats = pytest.importorskip("scipy.stats")
    mu, W, sigma_u, names = _random_basis(P=5, seed=2)
    pop = EigenbasisPopulation(mu, W, sigma_u, names)
    cov = W @ np.diag(sigma_u**2) @ W.T
    rng = np.random.default_rng(3)
    log_theta = rng.multivariate_normal(mu, cov, size=200)
    got = pop.log_prob(log_theta)
    ref = scipy_stats.multivariate_normal(mean=mu, cov=cov).logpdf(log_theta)
    assert np.allclose(got, ref, atol=1e-8)


def test_log_prob_accepts_1d_and_torch():
    torch = pytest.importorskip("torch")
    mu, W, sigma_u, names = _random_basis(P=3, seed=4)
    pop = EigenbasisPopulation(mu, W, sigma_u, names)
    row = mu + 0.1
    a = pop.log_prob(row)                       # 1-D array
    b = pop.log_prob(torch.tensor(row[None, :]))  # (1, P) tensor
    assert a.shape == (1,) and np.allclose(a, b)


def test_reweight_to_self_is_uniform():
    """π reweighted onto itself → uniform weights, ESS ≈ n."""
    mu, W, sigma_u, names = _random_basis(P=4, seed=5)
    pop = EigenbasisPopulation(mu, W, sigma_u, names)
    theta = pop.sample(5000, np.random.default_rng(9))
    res = reweight_to_prior(np.log(theta), pop, pop)
    assert res.ess_fraction > 0.999
    assert np.allclose(res.weights, 1.0 / theta.shape[0])


def test_reweight_wide_proposal_to_narrow_prior_loses_ess():
    """Draw from the widened π̃, report under π → ESS drops below n."""
    mu, W, sigma_u, names = _random_basis(P=4, seed=6)
    prior = EigenbasisPopulation(mu, W, sigma_u, names)
    proposal = widen_on_identified(prior, factor=3.0, n_directions=4)
    theta = proposal.sample(20_000, np.random.default_rng(11))
    res = reweight_to_prior(np.log(theta), prior, proposal, warn_below=None)
    assert 0.0 < res.ess_fraction < 0.95         # reweight is doing real work
    assert abs(res.weights.sum() - 1.0) < 1e-9


def test_widen_scales_only_the_identified_directions():
    mu, W, sigma_u, names = _random_basis(P=6, seed=7)
    prior = EigenbasisPopulation(mu, W, sigma_u, names)
    K = 2
    tilde = widen_on_identified(prior, factor=4.0, n_directions=K)
    assert np.allclose(tilde.sigma_u[:K], 4.0 * sigma_u[:K])
    assert np.allclose(tilde.sigma_u[K:], sigma_u[K:])   # sloppy complement anchored
    # same geometry → reweightable without misalignment
    assert tilde.param_names == prior.param_names


def test_reachable_accept_keeps_in_envelope_rejects_out_and_nan():
    # predict(log θ) = log θ (identity in obs space) so we control the envelope.
    def predict(log_theta, param_names):
        return np.asarray(log_theta, dtype=float)

    names = ["a", "b"]
    lo = np.array([-1.0, -1.0])
    hi = np.array([1.0, 1.0])
    accept = reachable_accept_fn(predict, names, lo, hi)
    theta = np.exp(np.array([
        [0.0, 0.0],     # inside
        [5.0, 0.0],     # one obs outside → reject (min_fraction=1.0)
        [-0.5, 0.9],    # inside
    ]))
    mask = accept(theta)
    assert mask.tolist() == [True, False, True]

    # non-finite prediction counts as outside
    def predict_nan(log_theta, param_names):
        out = np.asarray(log_theta, dtype=float).copy()
        out[0, 0] = np.nan
        return out
    accept_nan = reachable_accept_fn(predict_nan, names, lo, hi)
    assert accept_nan(np.exp(np.zeros((1, 2)))).tolist() == [False]


def test_reachable_accept_min_fraction_tolerates_one_miss():
    def predict(log_theta, param_names):
        return np.asarray(log_theta, dtype=float)
    names = ["a", "b", "c"]
    lo = np.full(3, -1.0)
    hi = np.full(3, 1.0)
    accept = reachable_accept_fn(predict, names, lo, hi, min_fraction=0.6)
    # 2 of 3 inside → fraction 0.667 ≥ 0.6 → kept
    theta = np.exp(np.array([[0.0, 0.0, 9.0]]))
    assert accept(theta).tolist() == [True]


def test_misaligned_reweight_raises():
    mu, W, sigma_u, names = _random_basis(P=3, seed=8)
    prior = EigenbasisPopulation(mu, W, sigma_u, names)
    proposal = EigenbasisPopulation(mu, W, sigma_u, ["x", "y", "z"])
    theta = proposal.sample(50, np.random.default_rng(1))
    with pytest.raises(ValueError, match="same parameters"):
        reweight_to_prior(np.log(theta), prior, proposal)


@pytest.mark.parametrize("bad", [
    dict(draw_matrix=np.zeros((3, 3))),          # singular
    dict(sigma_u=np.array([1.0, -1.0, 1.0])),    # non-positive
    dict(sigma_u=np.array([1.0, 1.0])),          # wrong length
    dict(param_names=["a", "b"]),                # wrong count
])
def test_construction_validation(bad):
    mu = np.zeros(3)
    kw = dict(mu=mu, draw_matrix=np.eye(3) + 0.1,
              sigma_u=np.ones(3), param_names=["a", "b", "c"])
    kw.update(bad)
    with pytest.raises(ValueError):
        EigenbasisPopulation(**kw)

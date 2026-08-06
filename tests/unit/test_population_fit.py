"""Unit tests for the NUTS population fit (docs ch. 4b, Part II).

The likelihood itself is tested in ``test_summary_likelihood.py``. These tests
cover the prior, the packing onto one unconstrained vector, and that the sampler
actually runs end to end and recovers a population it generated.
"""
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.stats import norm  # noqa: E402

from qsp_inference.vpop.population_fit import (  # noqa: E402
    PopulationPosterior,
    PopulationPrior,
    run_nuts,
)
from qsp_inference.vpop.summary_likelihood import (  # noqa: E402
    SummaryLikelihood,
    TargetAnchor,
    build_study_blocks,
)

PS = (0.25, 0.5, 0.75)


def _problem(n_param=3, n_obs=3, n_patients=2000, seed=0):
    rng = np.random.default_rng(seed)
    A = torch.tensor(rng.normal(size=(n_param, n_obs)), dtype=torch.float64)
    z = torch.tensor(rng.standard_normal((n_patients, n_param)), dtype=torch.float64)
    mu = torch.tensor(rng.normal(size=n_param), dtype=torch.float64)
    sigma = torch.tensor(rng.uniform(0.4, 0.8, size=n_param), dtype=torch.float64)
    m = (mu @ A).numpy()
    s = np.sqrt(((sigma[:, None] ** 2) * A**2).sum(0).numpy())
    q = np.array([[m[j] + s[j] * norm.ppf(p) for p in PS] for j in range(n_obs)])
    tgts = [TargetAnchor(f"o{j}", j, "s1", 40, PS, q[j], epsilon=0.02) for j in range(n_obs)]
    blocks, cohort_ids, delta_names = build_study_blocks(tgts)
    lik = SummaryLikelihood(
        blocks=blocks, predict=lambda lt: lt @ A, z=z, bandwidth=0.04
    ).freeze_covariance_at(mu, sigma)
    prior = PopulationPrior(
        mu_loc=mu.numpy(), mu_scale=np.full(n_param, 1.0),
        omega=sigma.numpy(), omega_scale=0.5,
    )
    post = PopulationPosterior(lik, prior, len(cohort_ids), len(delta_names))
    return post, mu, sigma, cohort_ids, delta_names, A, m, s


def test_pack_and_unpack_round_trip():
    post, mu, sigma, _, _, _, _, _ = _problem()
    phi = post.pack(tau_eta=0.2, tau_delta=0.3)
    assert phi.numel() == post.size == 2 * 3 + 1 + 3 + 2
    v = post.unpack(phi)
    assert torch.allclose(v["mu"], mu)
    assert torch.allclose(v["sigma_u"], sigma)
    assert float(v["tau_eta"]) == pytest.approx(0.2)
    assert float(v["tau_delta"]) == pytest.approx(0.3)
    # non-centered: zero raw offsets mean zero offsets regardless of tau
    assert torch.allclose(v["eta"], torch.zeros_like(v["eta"]))
    assert torch.allclose(v["delta"], torch.zeros_like(v["delta"]))


def test_non_centering_scales_the_offsets():
    post, _, _, _, _, _, _, _ = _problem()
    phi = post.pack(tau_eta=0.5, tau_delta=0.25)
    phi = phi.clone()
    phi[2 * 3] = 2.0          # eta_raw[0]
    phi[2 * 3 + 1] = -1.0     # delta_raw[0]
    v = post.unpack(phi)
    assert float(v["eta"][0]) == pytest.approx(1.0)
    assert float(v["delta"][0]) == pytest.approx(-0.25)


def test_prior_peaks_at_its_own_center():
    post, mu, sigma, _, _, _, _, _ = _problem()
    prior = post.prior
    zeros = torch.zeros(1, dtype=torch.float64)
    at_center = prior.log_prob(
        mu, torch.log(sigma), zeros, torch.zeros(3, dtype=torch.float64),
        torch.tensor(prior.tau_eta_loc), torch.tensor(prior.tau_delta_loc),
    )
    off = prior.log_prob(
        mu + 0.5, torch.log(sigma), zeros, torch.zeros(3, dtype=torch.float64),
        torch.tensor(prior.tau_eta_loc), torch.tensor(prior.tau_delta_loc),
    )
    assert float(at_center) > float(off)


def test_prior_rejects_a_mismatched_shape():
    with pytest.raises(ValueError, match="must all be"):
        PopulationPrior(mu_loc=np.zeros(3), mu_scale=np.ones(3), omega=np.ones(4))


def test_posterior_rejects_a_prior_of_the_wrong_width():
    post, _, _, _, _, _, _, _ = _problem()
    bad = PopulationPrior(mu_loc=np.zeros(5), mu_scale=np.ones(5), omega=np.ones(5))
    with pytest.raises(ValueError, match="parameters but the likelihood"):
        PopulationPosterior(post.likelihood, bad, post.n_studies, post.n_delta)


def test_potential_is_finite_and_negates_the_log_posterior():
    post, _, _, _, _, _, _, _ = _problem()
    phi = post.pack()
    lp = post.log_prob(phi)
    assert torch.isfinite(lp)
    assert float(post.potential({"phi": phi})) == pytest.approx(-float(lp))


def test_potential_returns_inf_rather_than_raising_on_a_broken_forward():
    post, _, _, _, _, _, _, _ = _problem()
    post.likelihood.predict = lambda lt: lt @ torch.full((3, 3), float("nan"), dtype=torch.float64)
    post.likelihood.unfreeze_covariance()
    assert math.isinf(float(post.potential({"phi": post.pack()})))


def test_run_nuts_recovers_the_population_it_generated():
    """End to end and small: the observed anchors ARE this population's quantiles,
    so the posterior has to sit on the working-scale population that made them.
    Checked in observable space rather than parameter space, because the linear
    map is not injective and only the pushed-forward population is identified."""
    post, mu, sigma, cohort_ids, delta_names, A, m, s = _problem(n_patients=3000, seed=3)
    fit = run_nuts(
        post, num_samples=120, warmup_steps=120, max_tree_depth=5,
        cohort_ids=cohort_ids, delta_names=delta_names,
        progress_bar=False, seed=0,
    )
    assert fit.n_samples == 120
    assert fit.mu.shape == (120, 3)
    assert fit.eta.shape == (120, 1)
    assert fit.delta.shape == (120, 3)
    assert fit.cohort_ids == cohort_ids and fit.delta_names == delta_names

    mu_hat = torch.tensor(np.median(fit.mu, axis=0))
    sig_hat = np.median(fit.sigma_u, axis=0)
    m_hat = (mu_hat @ A).numpy()
    s_hat = np.sqrt(((sig_hat[:, None] ** 2) * A.numpy() ** 2).sum(0))
    assert np.abs(m_hat - m).max() < 0.25 * s.min()
    assert np.abs(s_hat / s - 1.0).max() < 0.35


def test_spread_movement_is_zero_when_the_data_are_silent():
    """The reporting readout that replaces K selection. At the prior center with
    no data pull, every direction reads zero movement."""
    post, mu, sigma, _, _, _, _, _ = _problem()
    from qsp_inference.vpop.population_fit import PopulationFit

    fit = PopulationFit(
        mu=np.tile(mu.numpy(), (5, 1)),
        sigma_u=np.tile(sigma.numpy(), (5, 1)),
        eta=np.zeros((5, 1)), delta=np.zeros((5, 3)),
        tau_eta=np.zeros(5), tau_delta=np.zeros(5),
        param_names=["a", "b", "c"], cohort_ids=["s1"], delta_names=["o0", "o1", "o2"],
        diagnostics={},
    )
    assert np.allclose(fit.spread_movement(post.prior), 0.0)

"""Unit tests for the closed-form summary likelihood (docs ch. 4b, Part II).

The tests are organised around the three claims the chapter makes, in the order
it makes them: the covariance formula is right, the mean model is right, and
targets that share patients get a block rather than a product.

A linear "emulator" ``predict(log_theta) = log_theta @ A`` is used throughout, so
the population predictive on the working scale is exactly Gaussian and every
quantity the module estimates has a closed form to compare against.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from scipy.stats import multivariate_normal, norm  # noqa: E402

from qsp_inference.vpop.summary_likelihood import (  # noqa: E402
    SummaryLikelihood,
    TargetAnchor,
    anchor_covariance,
    build_study_blocks,
    bvn_cdf,
    normal_score_correlation,
)

PS = (0.25, 0.5, 0.75)


def _linear_setup(n_param=4, n_obs=3, n_patients=6000, seed=0):
    """A population whose working-scale marginals are exactly Gaussian."""
    rng = np.random.default_rng(seed)
    A = torch.tensor(rng.normal(size=(n_param, n_obs)), dtype=torch.float64)
    z = torch.tensor(rng.standard_normal((n_patients, n_param)), dtype=torch.float64)
    mu = torch.tensor(rng.normal(size=n_param), dtype=torch.float64)
    sigma = torch.tensor(rng.uniform(0.3, 0.9, size=n_param), dtype=torch.float64)
    m = (mu @ A).numpy()
    s = np.sqrt(((sigma[:, None] ** 2) * A**2).sum(0).numpy())
    return A, z, mu, sigma, m, s


def _true_anchors(m, s, ps=PS):
    return np.array([[m[j] + s[j] * norm.ppf(p) for p in ps] for j in range(len(m))])


# --------------------------------------------------------------------------
# the covariance
# --------------------------------------------------------------------------


def test_bvn_cdf_matches_scipy():
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(150):
        z1, z2 = rng.normal(size=2) * 1.8
        r = float(rng.uniform(-0.98, 0.98))
        ref = float(multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf([z1, z2]))
        got = float(bvn_cdf(torch.tensor(z1), torch.tensor(z2), torch.tensor(r)))
        worst = max(worst, abs(ref - got))
    assert worst < 1e-7


def test_bvn_cdf_is_independent_at_zero_correlation():
    z = torch.tensor([-1.0, 0.0, 1.5], dtype=torch.float64)
    got = bvn_cdf(z, z, torch.zeros(3, dtype=torch.float64))
    ref = torch.distributions.Normal(0.0, 1.0).cdf(z) ** 2
    assert torch.allclose(got, ref, atol=1e-10)


def test_bvn_cdf_is_differentiable_in_all_arguments():
    z1 = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    z2 = torch.tensor(-0.9, dtype=torch.float64, requires_grad=True)
    r = torch.tensor(0.6, dtype=torch.float64, requires_grad=True)
    bvn_cdf(z1, z2, r).backward()
    for g in (z1.grad, z2.grad, r.grad):
        assert torch.isfinite(g)
    # d/drho Phi_2 is the bivariate density, which is the identity the quadrature
    # is built on; check it rather than just finiteness.
    dens = float(multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]]).pdf([0.4, -0.9]))
    assert float(r.grad) == pytest.approx(dens, rel=1e-6)


def test_within_observable_block_is_the_textbook_form():
    p = torch.tensor(PS, dtype=torch.float64)
    f = torch.tensor([0.31, 0.40, 0.31], dtype=torch.float64)
    cov = anchor_covariance(p, torch.zeros(3, dtype=torch.long), f, torch.eye(1, dtype=torch.float64), 10.0)
    pa, pb = p[:, None], p[None, :]
    ref = (torch.minimum(pa, pb) - pa * pb) / (10.0 * f[:, None] * f[None, :])
    assert torch.allclose(cov, ref, atol=1e-14)


def test_uncorrelated_observables_give_a_block_diagonal_covariance():
    p = torch.tensor([0.25, 0.75, 0.25, 0.75], dtype=torch.float64)
    col = torch.tensor([0, 0, 1, 1])
    f = torch.tensor([0.3, 0.3, 0.5, 0.5], dtype=torch.float64)
    cov = anchor_covariance(p, col, f, torch.eye(2, dtype=torch.float64), 10.0)
    assert torch.allclose(cov[:2, 2:], torch.zeros(2, 2, dtype=torch.float64), atol=1e-12)
    assert cov[0, 1] != 0.0


def test_perfectly_correlated_observables_reproduce_the_within_form():
    """rho -> 1 must collapse the cross block onto ``min(p_a, p_b)``. This is the
    consistency check between the two halves of the copula formula: the exact
    within-observable branch and the Gaussian-copula cross branch have to agree
    where they meet. Convergence is ``O(sqrt(1 - rho))``, a property of the
    Gaussian copula rather than of the quadrature."""
    p = torch.tensor([0.25, 0.75, 0.25, 0.75], dtype=torch.float64)
    col = torch.tensor([0, 0, 1, 1])
    f = torch.tensor([0.3, 0.3, 0.3, 0.3], dtype=torch.float64)
    errs = []
    for r in (0.99, 0.9999, 0.999999):
        rho = torch.tensor([[1.0, r], [r, 1.0]], dtype=torch.float64)
        cov = anchor_covariance(p, col, f, rho, 10.0)
        errs.append(float((cov[:2, 2:] - cov[:2, :2]).abs().max()))
    assert errs[0] > errs[1] > errs[2]
    assert errs[-1] < 1e-3


def test_covariance_matches_the_measured_sampling_law_of_real_cohorts():
    """The load-bearing claim of ch. 4b: at a real published ``n``, the asserted
    covariance of a stacked anchor vector matches the covariance actually seen
    when cohorts are drawn as ROWS of the cloud, so a study's observables share
    patients. Reported as the Mahalanobis ``D^2/k``, 1.00 being exact."""
    A, z, mu, sigma, m, s = _linear_setup(n_obs=3, n_patients=20000, seed=5)
    lik = _likelihood(A, z, m, s, n=40, bandwidth=0.03)
    block = lik.blocks[0]

    f = torch.tensor(
        [norm.pdf(norm.ppf(p)) / s[j] for j in range(len(s)) for p in PS],
        dtype=torch.float64,
    )
    cov = anchor_covariance(block.p, block.col, f, lik.rho, 40.0).numpy()

    rng = np.random.default_rng(11)
    x = lik.population_observables(mu, sigma).detach().numpy()
    rows = rng.integers(0, x.shape[0], size=(4000, 40))
    draws = np.stack(
        [np.quantile(x[rows, j], PS, axis=1, method="normal_unbiased") for j in range(x.shape[1])],
        axis=0,
    )  # (J, k, B)
    draws = draws.transpose(2, 0, 1).reshape(4000, -1)

    centred = draws - draws.mean(0)
    d2 = np.einsum("bi,ij,bj->b", centred, np.linalg.inv(cov), centred)
    assert 0.90 < d2.mean() / cov.shape[0] < 1.10


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------


def _targets(m, s, n=30, cohort="s1", **kw):
    q = _true_anchors(m, s)
    return [TargetAnchor(f"o{j}", j, cohort, n, PS, q[j], **kw) for j in range(len(m))]


def _likelihood(A, z, m, s, *, n=30, bandwidth=0.035, rho=None, **kw):
    blocks, _, _ = build_study_blocks(_targets(m, s, n=n))
    return SummaryLikelihood(
        blocks=blocks, predict=lambda lt: lt @ A, z=z, bandwidth=bandwidth,
        rho=rho, include_mc_error=False, **kw
    )


def test_targets_sharing_a_cohort_form_one_block():
    _, _, _, _, m, s = _linear_setup(n_obs=3)
    tgts = _targets(m, s, n=10, cohort="gvax") + _targets(m, s, n=113, cohort="baseline")
    blocks, cohort_ids, delta_names = build_study_blocks(tgts)
    assert cohort_ids == ["baseline", "gvax"]
    assert delta_names == ["o0", "o1", "o2"]
    assert [b.size for b in blocks] == [9, 9]
    # delta is per observable and therefore SHARED across the two studies.
    assert torch.equal(blocks[0].delta_index, blocks[1].delta_index)
    assert [b.n for b in blocks] == [113, 10]


def test_a_cohort_cannot_mix_sample_sizes():
    _, _, _, _, m, s = _linear_setup(n_obs=2)
    tgts = _targets(m, s, n=10, cohort="c") + _targets(m, s, n=11, cohort="c")
    with pytest.raises(ValueError, match="mixes n="):
        build_study_blocks(tgts)


def test_center_only_targets_keep_only_their_median_anchor():
    """A ci95-derived interval is uncertainty about a center. Letting its spread
    anchors act as population variability is the SEM-vs-SD conflation."""
    _, _, _, _, m, s = _linear_setup(n_obs=1)
    q = _true_anchors(m, s)[0]
    t = TargetAnchor("o0", 0, "c", 12, PS, q, feeds_spread=False)
    blocks, _, _ = build_study_blocks([t])
    assert blocks[0].size == 1
    assert float(blocks[0].p[0]) == 0.5
    assert float(blocks[0].y[0]) == pytest.approx(q[1])

    kept, _, _ = build_study_blocks([t], drop_spread_from_center_only=False)
    assert kept[0].size == 3


def test_diagonal_variance_carries_center_and_surrogate_terms():
    """Neither is a property of the patient sample, so both stay diagonal."""
    _, _, _, _, m, s = _linear_setup(n_obs=1)
    tgts = _targets(m, s, sigma_c=0.1, epsilon=0.2)
    blocks, _, _ = build_study_blocks(tgts)
    assert torch.allclose(blocks[0].diag_var, torch.full((3,), 0.05, dtype=torch.float64))


def test_value_and_level_count_must_agree():
    with pytest.raises(ValueError, match="values for"):
        build_study_blocks([TargetAnchor("o", 0, "c", 10, PS, np.zeros(2))])


# --------------------------------------------------------------------------
# the mean model
# --------------------------------------------------------------------------


def test_population_quantiles_match_the_analytic_population():
    A, z, mu, sigma, m, s = _linear_setup(n_patients=40000, seed=7)
    lik = _likelihood(A, z, m, s)
    q = lik._query_quantiles(lik.population_observables(mu, sigma), None)
    at = lik._rows[0][0]
    got = q[at].detach().numpy()
    assert np.abs(got - _true_anchors(m, s).ravel()).max() < 0.02


def test_density_plugin_matches_the_analytic_density():
    A, z, mu, sigma, m, s = _linear_setup(n_patients=40000, seed=7)
    lik = _likelihood(A, z, m, s, bandwidth=0.03)
    q = lik._query_quantiles(lik.population_observables(mu, sigma), None)
    _, lo, hi, h = lik._rows[0]
    f_est = (2.0 * h / (q[hi] - q[lo])).detach().numpy()
    f_true = np.array([norm.pdf(norm.ppf(p)) / s[j] for j in range(len(s)) for p in PS])
    assert np.abs(f_est / f_true - 1.0).max() < 0.06


def test_log_prob_is_finite_and_differentiable_in_phi():
    A, z, mu, sigma, m, s = _linear_setup()
    lik = _likelihood(A, z, m, s)
    mu_v = mu.clone().requires_grad_(True)
    sig_v = sigma.clone().requires_grad_(True)
    lik.log_prob(mu_v, sig_v).backward()
    assert torch.isfinite(mu_v.grad).all() and torch.isfinite(sig_v.grad).all()
    assert torch.abs(sig_v.grad).max() > 0


def test_common_random_numbers_make_log_prob_deterministic():
    A, z, mu, sigma, m, s = _linear_setup()
    lik = _likelihood(A, z, m, s)
    a = float(lik.log_prob(mu, sigma))
    b = float(lik.log_prob(mu, sigma))
    assert a == b


def test_log_prob_peaks_near_the_generating_center():
    """The mean model must be right: shifting mu away from truth must cost."""
    A, z, mu, sigma, m, s = _linear_setup(n_patients=20000, seed=2)
    lik = _likelihood(A, z, m, s, n=60)
    best = float(lik.log_prob(mu, sigma))
    for shift in (-0.25, 0.25):
        assert float(lik.log_prob(mu + shift, sigma)) < best


def test_offsets_shift_the_residual():
    A, z, mu, sigma, m, s = _linear_setup()
    tgts = _targets(m, s, cohort="a") + _targets(m, s, cohort="b")
    blocks, cohort_ids, delta_names = build_study_blocks(tgts)
    lik = SummaryLikelihood(blocks=blocks, predict=lambda lt: lt @ A, z=z, include_mc_error=False)
    base = float(lik.log_prob(mu, sigma))
    eta = torch.tensor([0.3, 0.0], dtype=torch.float64)
    assert float(lik.log_prob(mu, sigma, eta=eta)) < base
    delta = torch.zeros(len(delta_names), dtype=torch.float64)
    delta[0] = 0.3
    assert float(lik.log_prob(mu, sigma, delta=delta)) < base


def test_viability_weight_moves_the_population_quantiles():
    """A smooth weight, not a hard filter: the patient set stays fixed so the
    sorted quantile map stays differentiable."""
    A, z, mu, sigma, m, s = _linear_setup(n_patients=20000, seed=4)
    lik = _likelihood(A, z, m, s)
    q_flat = lik._query_quantiles(lik.population_observables(mu, sigma), None)

    def viability(log_theta):
        return torch.sigmoid(-2.0 * log_theta[:, 0])

    lik.viability = viability
    lt = lik.draw_population(mu, sigma)
    q_w = lik._query_quantiles(lik.predict(lt), viability(lt))
    assert not torch.allclose(q_flat, q_w, atol=1e-3)
    assert torch.isfinite(lik.log_prob(mu, sigma))


# --------------------------------------------------------------------------
# the covariance plug-in, and why it is frozen
# --------------------------------------------------------------------------


def test_a_live_covariance_pulls_the_spread_low():
    """The mechanism behind ``freeze_covariance_at``, isolated.

    The observed anchors here ARE the population quantiles at ``phi_true``, so
    the residual is exactly zero and every gradient must come from the
    log-determinant. ``Sigma`` scales with the population width (``f ~ 1/sigma``,
    so ``Sigma ~ sigma^2``), so a live covariance reports a large negative score
    on ``log sigma_u``: the likelihood wants a narrower population for no reason
    connected to the data. That is a low bias on the population spread, which is
    the one direction ch. 4 designs matched footing to prevent.
    """
    A, z, mu, sigma, m, s = _linear_setup(n_obs=3, n_patients=40000, seed=6)
    lik = _likelihood(A, z, m, s, n=20)

    def score(like):
        ls = torch.log(sigma).clone().requires_grad_(True)
        like.log_prob(mu, torch.exp(ls)).backward()
        return float(ls.grad.sum())

    live = score(lik)
    frozen = score(lik.freeze_covariance_at(mu, sigma))
    assert live < -5.0
    assert abs(frozen) < 0.1 * abs(live)


def test_freezing_holds_the_covariance_while_the_mean_moves():
    A, z, mu, sigma, m, s = _linear_setup(n_obs=2, n_patients=20000, seed=8)
    lik = _likelihood(A, z, m, s, n=20)
    assert not lik.covariance_is_frozen
    live_wide = float(lik.log_prob(mu, sigma * 1.5))

    lik.freeze_covariance_at(mu, sigma)
    assert lik.covariance_is_frozen
    frozen_wide = float(lik.log_prob(mu, sigma * 1.5))
    assert frozen_wide != pytest.approx(live_wide, rel=1e-6)

    # the mean model is untouched by freezing
    at = lik._rows[0][0]
    q = lik._query_quantiles(lik.population_observables(mu, sigma), None)
    lik.unfreeze_covariance()
    assert not lik.covariance_is_frozen
    q2 = lik._query_quantiles(lik.population_observables(mu, sigma), None)
    assert torch.allclose(q[at], q2[at])


def test_frozen_log_prob_stays_differentiable():
    A, z, mu, sigma, m, s = _linear_setup()
    lik = _likelihood(A, z, m, s).freeze_covariance_at(mu, sigma)
    mu_v = mu.clone().requires_grad_(True)
    sig_v = sigma.clone().requires_grad_(True)
    lik.log_prob(mu_v, sig_v).backward()
    assert torch.isfinite(mu_v.grad).all() and torch.isfinite(sig_v.grad).all()


# --------------------------------------------------------------------------
# the copula block, and what ignoring it costs
# --------------------------------------------------------------------------


def test_normal_score_correlation_recovers_a_known_dependence():
    rng = np.random.default_rng(0)
    r = 0.7
    x = rng.multivariate_normal([0, 0], [[1, r], [r, 1]], size=20000)
    rho = normal_score_correlation(x)
    assert rho[0, 1] == pytest.approx(r, abs=0.03)
    # rank based, so a monotone working-scale transform must not move it
    rho2 = normal_score_correlation(np.exp(x))
    assert rho2[0, 1] == pytest.approx(rho[0, 1], abs=1e-9)


def test_refresh_rho_reads_the_population_at_phi():
    A, z, mu, sigma, m, s = _linear_setup(n_obs=2, n_patients=20000, seed=9)
    lik = _likelihood(A, z, m, s)
    assert float(lik.rho[0, 1]) == 0.0
    rho = lik.refresh_rho(mu, sigma)
    expected = float((A[:, 0] * A[:, 1] * sigma**2).sum() / (s[0] * s[1]))
    assert rho[0, 1] == pytest.approx(expected, abs=0.03)
    assert float(lik.rho[0, 1]) == pytest.approx(rho[0, 1])


def test_shared_sampling_noise_absorbs_a_common_shift():
    """What the block actually buys, stated as an experiment.

    Two readouts on one set of 10 biopsies. If their sampling errors are
    correlated, a cohort that drew high patients reads high on *both*, so a
    common shift is cheap to explain and a differential one is expensive.
    Factorising over targets gets both wrong, and the common-shift half is the
    one that corrupts ``tau_eta``: shared sampling noise gets absorbed as
    between-study heterogeneity.
    """
    p = torch.tensor(PS * 2, dtype=torch.float64)
    col = torch.tensor([0, 0, 0, 1, 1, 1])
    f = torch.tensor([0.31, 0.40, 0.31] * 2, dtype=torch.float64)
    common = torch.ones(6, dtype=torch.float64)
    differential = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=torch.float64)

    def cost(r, direction):
        rho = torch.tensor([[1.0, r], [r, 1.0]], dtype=torch.float64)
        cov = anchor_covariance(p, col, f, rho, 10.0)
        return float(direction @ torch.linalg.solve(cov, direction))

    assert cost(0.9, common) < cost(0.0, common)
    assert cost(0.9, differential) > cost(0.0, differential)


def test_the_copula_block_is_wired_through_to_the_likelihood():
    A, z, mu, sigma, m, s = _linear_setup(n_obs=2, n_patients=20000, seed=9)
    rho = torch.tensor([[1.0, 0.9], [0.9, 1.0]], dtype=torch.float64)
    indep = _likelihood(A, z, m, s, n=10)
    coupled = _likelihood(A, z, m, s, n=10, rho=rho)
    assert float(indep.log_prob(mu, sigma)) != pytest.approx(
        float(coupled.log_prob(mu, sigma)), rel=1e-6
    )
    # the residual is the same object either way; only the weighting changed
    q_i = indep._query_quantiles(indep.population_observables(mu, sigma), None)
    q_c = coupled._query_quantiles(coupled.population_observables(mu, sigma), None)
    assert torch.allclose(q_i, q_c)

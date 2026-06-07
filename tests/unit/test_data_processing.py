"""Unit tests for add_observation_noise (parametric + empirical paths)."""
import numpy as np
import pytest

from qsp_inference.inference.data_processing import add_observation_noise


def _ci_from_samples(s):
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5)), float(np.median(s))


def test_backward_compat_no_bootstrap_is_deterministic():
    """bootstrap_samples=None reproduces the legacy parametric behavior exactly."""
    rng = np.random.default_rng(0)
    x = rng.lognormal(0, 0.1, size=(500, 2))
    lo = np.array([0.5, 0.5]); hi = np.array([2.0, 2.0]); med = np.array([1.0, 1.0])
    a = add_observation_noise(x, lo, hi, med, seed=7)
    b = add_observation_noise(x, lo, hi, med, seed=7)
    np.testing.assert_array_equal(a, b)
    assert a.shape == x.shape
    # noise actually perturbed the data
    assert not np.allclose(a, x)


def test_empirical_constant_predictive_recovers_bootstrap():
    """With x pinned at the bootstrap median, multiplicative empirical noise makes
    x_noisy a resample of the bootstrap, so its percentiles match the bootstrap."""
    rng = np.random.default_rng(1)
    boot = rng.lognormal(mean=np.log(3.0), sigma=0.6, size=20000)  # positive, skewed
    med_b = np.median(boot)
    x = np.full((8000, 1), med_b)
    lo, hi, med = _ci_from_samples(boot)
    out = add_observation_noise(
        x, np.array([lo]), np.array([hi]), np.array([med]),
        seed=3, bootstrap_samples=[boot],
    )[:, 0]
    # percentiles of the noised predictive should track the bootstrap's
    for q in (2.5, 25, 50, 75, 97.5):
        assert np.percentile(out, q) == pytest.approx(np.percentile(boot, q), rel=0.08)


def test_empirical_beats_parametric_on_skewed_bootstrap():
    """For a strongly right-skewed bootstrap (median != geomean(lo,hi)), the
    empirical path reproduces the observed spread far better than the parametric
    lognormal refit of the CI endpoints."""
    rng = np.random.default_rng(2)
    # heavy right skew: most mass low, long upper tail
    boot = np.concatenate([rng.lognormal(np.log(2.0), 0.3, 15000),
                           rng.lognormal(np.log(8.0), 0.9, 5000)])
    lo, hi, med = _ci_from_samples(boot)
    x = np.full((8000, 1), med)
    emp = add_observation_noise(x, np.array([lo]), np.array([hi]), np.array([med]),
                                seed=5, bootstrap_samples=[boot])[:, 0]
    par = add_observation_noise(x, np.array([lo]), np.array([hi]), np.array([med]),
                                seed=5, bootstrap_samples=None)[:, 0]
    # error in reproducing the true 97.5th percentile of the observation
    truth_hi = np.percentile(boot, 97.5)
    emp_err = abs(np.percentile(emp, 97.5) - truth_hi) / truth_hi
    par_err = abs(np.percentile(par, 97.5) - truth_hi) / truth_hi
    assert emp_err < par_err


def test_empirical_additive_for_signed_data():
    """When the predictive column has non-positive values, empirical noise is
    applied additively (residuals around the bootstrap median)."""
    rng = np.random.default_rng(4)
    boot = rng.normal(0.0, 1.0, size=10000)  # centered, can be negative
    x = np.full((5000, 1), 0.0)
    lo, hi, med = _ci_from_samples(boot)
    out = add_observation_noise(x, np.array([lo]), np.array([hi]), np.array([med]),
                                seed=6, bootstrap_samples=[boot])[:, 0]
    # additive around med_b ~ 0 -> spread matches bootstrap sd
    assert np.std(out) == pytest.approx(np.std(boot), rel=0.1)


def test_too_few_bootstrap_falls_back_to_parametric():
    """An observable with < 5 bootstrap samples uses the parametric path; one with
    enough uses the empirical path. Mixed list is handled per-observable."""
    rng = np.random.default_rng(8)
    boot_full = rng.lognormal(0, 0.5, size=10000)
    x = np.full((4000, 2), 1.0)
    lo = np.array([0.4, 0.4]); hi = np.array([2.5, 2.5]); med = np.array([1.0, 1.0])
    out = add_observation_noise(
        x, lo, hi, med, seed=9,
        bootstrap_samples=[np.array([1.0, 1.1, 0.9]), boot_full],  # col0 too few, col1 ok
    )
    # col1 (empirical) should track the bootstrap spread; col0 (parametric) still noised
    assert np.std(out[:, 1]) == pytest.approx(np.std(boot_full), rel=0.15)
    assert not np.allclose(out[:, 0], x[:, 0])

"""Unit tests for fit_scenario_transform / ScenarioTransform.

The per-scenario processing atom behind amortized inference on QSP summaries:
noise-inject the training observables, fit copula quantiles on the *noisy*
training set, and carry every other array into that same standard-normal space.
"""
import warnings

import numpy as np
import pytest

from qsp_inference.inference.data_processing import (
    ScenarioTransform,
    fit_scenario_transform,
)


def _synthetic_scenario(n_train=4000, d=3, seed=0):
    """Positive lognormal training observables + matching CI95/median summaries."""
    rng = np.random.default_rng(seed)
    x_train = rng.lognormal(mean=0.0, sigma=0.4, size=(n_train, d))
    lo = np.full(d, np.exp(-0.8))
    hi = np.full(d, np.exp(0.8))
    med = np.ones(d)
    return x_train, lo, hi, med


def _fit(x_train, lo, hi, med, **kw):
    """fit_scenario_transform on the parametric path (bootstrap_samples=None)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # parametric fallback warns; not under test here
        return fit_scenario_transform(
            x_train, ci95_lower=lo, ci95_upper=hi, medians=med,
            bootstrap_samples=None, **kw,
        )


def test_returns_transform_and_matching_shape():
    x_train, lo, hi, med = _synthetic_scenario()
    x_t, xf = _fit(x_train, lo, hi, med, noise_seed=7)
    assert isinstance(xf, ScenarioTransform)
    assert x_t.shape == x_train.shape


def test_training_marginals_are_standard_normal():
    """The copula transform is fit on the training set, so its own transformed
    marginals are ~N(0, 1) per column."""
    x_train, lo, hi, med = _synthetic_scenario(n_train=8000)
    x_t, _ = _fit(x_train, lo, hi, med, noise_seed=1)
    assert np.all(np.abs(x_t.mean(axis=0)) < 0.05)
    assert np.all(np.abs(x_t.std(axis=0) - 1.0) < 0.05)


def test_deterministic_in_noise_seed():
    x_train, lo, hi, med = _synthetic_scenario()
    a_t, a = _fit(x_train, lo, hi, med, noise_seed=42)
    b_t, b = _fit(x_train, lo, hi, med, noise_seed=42)
    np.testing.assert_array_equal(a_t, b_t)
    np.testing.assert_array_equal(a.quantiles, b.quantiles)


def test_different_noise_seed_changes_result():
    x_train, lo, hi, med = _synthetic_scenario()
    a_t, _ = _fit(x_train, lo, hi, med, noise_seed=1)
    b_t, _ = _fit(x_train, lo, hi, med, noise_seed=2)
    assert not np.allclose(a_t, b_t)


def test_returned_train_equals_transform_of_noisy_train():
    """The returned x_train_t IS transform() applied to the same noisy training
    data the quantiles were fit on: fitting and applying share one gauge."""
    from qsp_inference.inference.data_processing import add_observation_noise
    from qsp_inference.inference.gaussian_copula_transform import (
        transform_to_normal_from_array,
    )
    x_train, lo, hi, med = _synthetic_scenario()
    x_t, xf = _fit(x_train, lo, hi, med, noise_seed=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x_noisy = add_observation_noise(
            x_train, lo, hi, med, bootstrap_samples=None, seed=5,
        )
    np.testing.assert_allclose(
        x_t, transform_to_normal_from_array(x_noisy, xf.quantiles), rtol=0, atol=0,
    )


def test_transform_vector_matches_batch_row():
    x_train, lo, hi, med = _synthetic_scenario()
    _, xf = _fit(x_train, lo, hi, med, noise_seed=0)
    v = np.array([0.7, 1.0, 1.4])
    np.testing.assert_array_equal(
        xf.transform_vector(v), xf.transform(v[np.newaxis, :])[0],
    )


def test_quantiles_fit_on_train_not_applied_array():
    """transform() uses the *training* quantiles: a shifted held-out array is NOT
    forced back to standard normal (which is what leakage would look like)."""
    x_train, lo, hi, med = _synthetic_scenario(n_train=8000)
    _, xf = _fit(x_train, lo, hi, med, noise_seed=3)
    # A test set shifted well above the training median lands high in latent
    # space, not re-centered at zero.
    x_test = np.full((2000, 3), np.exp(1.2))
    z = xf.transform(x_test)
    assert np.all(z.mean(axis=0) > 1.0)


def test_empirical_path_runs_without_warning():
    """With per-observable bootstrap samples the empirical noise path is used and
    no parametric-fallback warning is emitted."""
    x_train, lo, hi, med = _synthetic_scenario(d=2)
    rng = np.random.default_rng(11)
    boots = [rng.lognormal(0.0, 0.3, size=400) for _ in range(2)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning fails the test
        x_t, xf = fit_scenario_transform(
            x_train, ci95_lower=lo, ci95_upper=hi, medians=med,
            bootstrap_samples=boots, noise_seed=9,
        )
    assert x_t.shape == x_train.shape
    assert xf.quantiles.shape[1] == x_train.shape[1]


def test_scenario_transform_is_frozen():
    xf = ScenarioTransform(quantiles=np.zeros((10, 2)))
    with pytest.raises((AttributeError, TypeError)):
        xf.quantiles = np.ones((10, 2))

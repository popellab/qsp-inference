"""Unit tests for inflate_cloud -- folding a surrogate error budget into a
predictive null.

A deterministic surrogate (e.g. a GBM emulator) produces an artificially tight
null; scoring an observation against it charges the surrogate's approximation
error as model misspecification. inflate_cloud convolves each observable's
predictive with independent N(0, noise_sd) so the null widens by the budget and
the downstream discrepancy test becomes conservative wrt that error.
"""
import numpy as np
import pytest

from qsp_inference.inference.predictive_checks import (
    inflate_cloud,
    quantile_vpc,
)


def _iqr(a, axis=0):
    q75, q25 = np.percentile(a, [75, 25], axis=axis)
    return q75 - q25


def test_raw_space_adds_variance_by_budget():
    """In raw space the per-column variance grows by ~noise_sd^2."""
    rng = np.random.default_rng(0)
    cloud = rng.normal(scale=1.0, size=(200000, 2))
    out = inflate_cloud(cloud, noise_sd=np.array([0.5, 2.0]), seed=1)
    var = out.var(axis=0)
    # Var[f + eps] = Var[f] + noise_sd^2 (independent injection).
    assert var[0] == pytest.approx(1.0 + 0.5**2, rel=0.03)
    assert var[1] == pytest.approx(1.0 + 2.0**2, rel=0.03)


def test_widens_never_narrows():
    """Inflation only ever increases spread, per observable."""
    rng = np.random.default_rng(1)
    cloud = rng.normal(scale=1.5, size=(50000, 3))
    out = inflate_cloud(cloud, noise_sd=0.7, seed=2)
    assert (out.std(axis=0) >= cloud.std(axis=0) - 1e-9).all()


def test_zero_and_nonfinite_budget_is_noop_column():
    """A zero / NaN / negative budget leaves that observable untouched."""
    rng = np.random.default_rng(2)
    cloud = rng.normal(size=(1000, 4))
    out = inflate_cloud(cloud, noise_sd=np.array([0.0, np.nan, -1.0, 1.0]), seed=3)
    # columns 0,1,2 unchanged; column 3 changed.
    np.testing.assert_array_equal(out[:, :3], cloud[:, :3])
    assert not np.allclose(out[:, 3], cloud[:, 3])


def test_all_zero_budget_returns_copy():
    rng = np.random.default_rng(3)
    cloud = rng.normal(size=(100, 2))
    out = inflate_cloud(cloud, noise_sd=0.0)
    np.testing.assert_array_equal(out, cloud)
    assert out is not cloud  # a copy, not the original


def test_asinh_space_scales_with_magnitude():
    """asinh-space noise maps to raw spread that tracks the observable's scale:
    the same asinh budget yields larger raw SD for a larger-magnitude column."""
    scale = np.array([1.0, 100.0])
    cloud = np.column_stack([
        np.full(40000, 1.0),      # small-magnitude observable
        np.full(40000, 100.0),    # large-magnitude observable, same asinh budget
    ])
    out = inflate_cloud(cloud, noise_sd=0.3, asinh_scale=scale, seed=4)
    sd = out.std(axis=0)
    assert sd[1] > 10 * sd[0]     # raw noise rides the observable's magnitude


def test_asinh_preserves_nan():
    cloud = np.array([[np.nan, 1.0], [2.0, np.nan]])
    out = inflate_cloud(cloud, noise_sd=0.5, asinh_scale=np.array([1.0, 1.0]), seed=5)
    assert np.isnan(out[0, 0]) and np.isnan(out[1, 1])
    assert np.isfinite(out[0, 1]) and np.isfinite(out[1, 0])


def test_inflated_null_stops_false_overdispersion_label():
    """The end-to-end point. A surrogate whose cloud is biased tight relative to
    the true data reads as under-dispersed (spread_verdict 'over'); inflating the
    null by the true error budget restores 'ok'."""
    rng = np.random.default_rng(4)
    # Truth (what produced the observed anchors) has scale 1.
    truth = rng.normal(scale=1.0, size=(40000, 1))
    q25, q50, q75 = np.percentile(truth, [25, 50, 75])
    anchors = [[(0.25, q25), (0.5, q50), (0.75, q75)]]
    # Surrogate cloud is too tight (its deterministic predictions lost spread the
    # residual budget quantifies): scale 0.6 so that adding a 0.8-SD error budget
    # in raw space recovers scale sqrt(0.6^2 + 0.8^2) = 1.
    tight = rng.normal(scale=0.6, size=(40000, 1))

    raw = quantile_vpc(tight, anchors, cohort_size=100, seed=7)
    assert (raw["spread_verdict"] == "over").all()      # false under-dispersion

    fixed_cloud = inflate_cloud(tight, noise_sd=0.8, seed=8)
    fixed = quantile_vpc(fixed_cloud, anchors, cohort_size=100, seed=7)
    assert (fixed["spread_verdict"] == "ok").all()      # budget restores honesty


def test_shape_validation():
    with pytest.raises(ValueError, match="cloud must be 2-D"):
        inflate_cloud(np.zeros(10), 0.5)
    with pytest.raises(ValueError, match="noise_sd must be scalar"):
        inflate_cloud(np.zeros((10, 3)), np.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="asinh_scale must be"):
        inflate_cloud(np.zeros((10, 2)), 0.5, asinh_scale=np.array([1.0]))
    with pytest.raises(ValueError, match="finite and positive"):
        inflate_cloud(np.zeros((10, 2)), 0.5, asinh_scale=np.array([1.0, -1.0]))


def test_deterministic_seed():
    rng = np.random.default_rng(5)
    cloud = rng.normal(size=(500, 2))
    a = inflate_cloud(cloud, noise_sd=0.4, seed=11)
    b = inflate_cloud(cloud, noise_sd=0.4, seed=11)
    np.testing.assert_array_equal(a, b)

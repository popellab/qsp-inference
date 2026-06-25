"""Tail behavior of the array Gaussian-copula transform.

Locks in linear z-extrapolation beyond the empirical support (instead of the
old searchsorted+clip that pinned every out-of-support value to
``norm.ppf(1-1e-10) ≈ ±6.36``). See gaussian_copula_transform.py.
"""
import numpy as np
from scipy import stats

from qsp_inference.inference.gaussian_copula_transform import (
    compute_quantiles_from_array,
    transform_to_normal_from_array,
)

CLIP_Z = stats.norm.ppf(1 - 1e-10)  # ~6.36, the old saturation value


def _fit(seed=0, n=10000):
    rng = np.random.default_rng(seed)
    x = rng.lognormal(0.0, 1.0, size=(n, 1))
    q = compute_quantiles_from_array(x, 1000)
    return x, q


def test_in_support_marginals_are_standard_normal():
    x, q = _fit()
    z = transform_to_normal_from_array(x, q)
    assert abs(z.mean()) < 0.05
    assert abs(z.std() - 1.0) < 0.05
    # 1000-point grid → extreme grid scores ~±3.29, so in-sample stays bounded.
    assert z.max() < 3.5 and z.min() > -3.5


def test_monotonic():
    x, q = _fit()
    probe = np.linspace(x.min() * 0.1, x.max() * 5, 200).reshape(-1, 1)
    z = transform_to_normal_from_array(probe, q)[:, 0]
    assert np.all(np.diff(z) >= -1e-9), "transform must be monotone non-decreasing"


def test_out_of_support_extrapolates_not_clips():
    x, q = _fit()
    xmax = x.max()
    # Three distinct magnitudes above the training max.
    probes = np.array([[xmax * 1.5], [xmax * 3.0], [xmax * 10.0]])
    z = transform_to_normal_from_array(probes, q)[:, 0]
    # Strictly increasing — the old clip mapped all three to the SAME +6.36.
    assert z[0] < z[1] < z[2]
    # Each is past the in-support boundary (~+3.29) but a distinct finite value.
    assert z[0] > 3.29
    assert np.all(np.isfinite(z))
    # The nearest probe is far below the old saturation pin.
    assert z[0] < CLIP_Z


def test_lower_tail_extrapolates():
    x, q = _fit()
    probes = np.array([[x.min() * 0.5], [x.min() * 0.1]])
    z = transform_to_normal_from_array(probes, q)[:, 0]
    assert z[1] < z[0] < -3.29
    assert np.all(np.isfinite(z))


def test_degenerate_constant_feature_maps_to_zero():
    x = np.full((100, 1), 3.0)
    q = compute_quantiles_from_array(x, 1000)
    z = transform_to_normal_from_array(np.array([[3.0], [5.0], [1.0]]), q)
    assert np.allclose(z, 0.0)

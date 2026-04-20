"""Sanity tests for compute_per_param_calibration.

Generating process for a "perfectly calibrated" synthetic posterior:
    offset_i ~ N(0, post_sd)                 # calibrated residual
    samples[s, i, j] ~ N(theta_test[i,j] + offset[i,j], post_sd)
Empirically: post_mean ≈ theta + offset, post_std ≈ post_sd, so
    z = (post_mean - theta) / post_std ≈ offset / post_sd ~ N(0, 1)
and coverage95 is the Gaussian tail probability.
"""
import pytest

# torch is required transitively by qsp_inference.inference (via diagnostics.py).
# CI installs .[dev,submodel] without the sbi extra, so skip this whole module
# when torch isn't available.
pytest.importorskip("torch")

import numpy as np
import torch

from qsp_inference.inference import compute_per_param_calibration


def _calibrated_posterior(n_samples, n_test, n_params, post_sd, rng):
    theta = rng.normal(size=(n_test, n_params))
    offsets = rng.normal(scale=post_sd, size=(n_test, n_params))
    samples = rng.normal(
        loc=(theta + offsets)[np.newaxis, :, :],
        scale=post_sd,
        size=(n_samples, n_test, n_params),
    )
    return theta, samples


def test_perfectly_calibrated_gaussian():
    rng = np.random.default_rng(0)
    post_sd = 0.5
    theta, samples = _calibrated_posterior(4000, 2000, 3, post_sd, rng)

    out = compute_per_param_calibration(samples, theta)

    assert out["zscore_mean"].shape == (3,)
    np.testing.assert_allclose(out["zscore_mean"], 0.0, atol=0.1)
    np.testing.assert_allclose(out["zscore_std"], 1.0, atol=0.1)
    np.testing.assert_allclose(out["coverage95"], 0.95, atol=0.02)
    np.testing.assert_allclose(out["post_sd"], post_sd, rtol=0.05)


def test_biased_posterior_flags_via_zscore_mean():
    rng = np.random.default_rng(1)
    n_samples, n_test, n_params = 4000, 1500, 2
    post_sd = 0.5
    theta = rng.normal(size=(n_test, n_params))
    offsets = rng.normal(scale=post_sd, size=(n_test, n_params))
    # Shift param 1 upward by a constant bias of 1 posterior-SD.
    offsets[:, 1] += 1.0 * post_sd
    samples = rng.normal(
        loc=(theta + offsets)[np.newaxis, :, :],
        scale=post_sd,
        size=(n_samples, n_test, n_params),
    )

    out = compute_per_param_calibration(samples, theta)

    # Param 0 unbiased; param 1 has z-mean ≈ +1 and collapsed coverage.
    np.testing.assert_allclose(out["zscore_mean"][0], 0.0, atol=0.15)
    assert out["zscore_mean"][1] > 0.7
    assert out["coverage95"][0] > 0.9
    assert out["coverage95"][1] < out["coverage95"][0] - 0.1


def test_overconfident_posterior_flags_via_zscore_std():
    rng = np.random.default_rng(2)
    n_samples, n_test, n_params = 4000, 2000, 2
    # Offsets drawn with true width 1.0 but posterior claims width 0.4 for param 1
    theta = rng.normal(size=(n_test, n_params))
    true_offsets = rng.normal(scale=1.0, size=(n_test, n_params))
    claimed_sd = np.array([1.0, 0.4])  # calibrated, overconfident
    samples = rng.normal(
        loc=(theta + true_offsets)[np.newaxis, :, :],
        scale=claimed_sd[np.newaxis, np.newaxis, :],
        size=(n_samples, n_test, n_params),
    )

    out = compute_per_param_calibration(samples, theta)

    # Param 0 calibrated; param 1 overconfident → z-std > 1 and coverage << 0.95.
    np.testing.assert_allclose(out["zscore_std"][0], 1.0, atol=0.1)
    assert out["zscore_std"][1] > 1.5
    assert out["coverage95"][0] > 0.9
    assert out["coverage95"][1] < 0.7


def test_accepts_torch_inputs_and_precomputed_zscores():
    rng = np.random.default_rng(3)
    theta, samples = _calibrated_posterior(1000, 400, 2, 0.5, rng)

    post_mean = samples.mean(axis=0)
    post_std = samples.std(axis=0)
    z_scores = (post_mean - theta) / (post_std + 1e-10)

    out_np = compute_per_param_calibration(samples, theta, z_scores=z_scores)
    out_torch = compute_per_param_calibration(
        torch.from_numpy(samples),
        torch.from_numpy(theta),
        z_scores=torch.from_numpy(z_scores),
    )

    for key in ("zscore_mean", "zscore_std", "coverage95", "post_sd"):
        np.testing.assert_allclose(out_np[key], out_torch[key])

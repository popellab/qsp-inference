"""Unit tests for expected_coverage (weighted TARP).

The joint complement to weighted_sbc: under a calibrated joint posterior the TARP
credibility values are Uniform(0,1); an over-confident posterior undercovers
(ATC < 0), a conservative one overcovers (ATC > 0).
"""
import numpy as np
import pytest

from qsp_inference.inference.sbc import CoverageResult, expected_coverage


def _make(scale, *, n_rep=500, L=300, d=4, truth_sd=1.0, seed=0):
    """Per replicate: mu random, truth ~ N(mu, truth_sd), draws ~ N(mu, scale)."""
    r = np.random.default_rng(seed)
    mu = r.normal(size=(n_rep, d))
    truth = r.normal(mu, truth_sd)
    draws = r.normal(mu[:, None, :], scale, size=(n_rep, L, d))
    return draws, truth


def test_calibrated_is_ok():
    draws, truth = _make(1.0, seed=1)
    res = expected_coverage(draws, truth, seed=3)
    assert isinstance(res, CoverageResult)
    assert res.verdict == "ok"
    assert res.ks_p > 0.05
    assert abs(res.atc) < 0.03


def test_overconfident_undercovers():
    draws, truth = _make(0.4, seed=1)  # posterior narrower than the truth spread
    res = expected_coverage(draws, truth, seed=3)
    assert res.verdict == "overconfident"
    assert res.atc < 0
    assert res.ks_p < 0.05


def test_conservative_overcovers():
    draws, truth = _make(2.5, seed=1)  # posterior wider than the truth spread
    res = expected_coverage(draws, truth, seed=3)
    assert res.verdict == "conservative"
    assert res.atc > 0


def test_uniform_weights_equal_none():
    draws, truth = _make(1.0, seed=2)
    a = expected_coverage(draws, truth, seed=5)
    b = expected_coverage(draws, truth, weights=np.ones(draws.shape[:2]), seed=5)
    np.testing.assert_allclose(a.credibility, b.credibility)


def test_weights_shift_the_reading():
    """Down-weighting the tail draws narrows the effective posterior, so a
    correctly-spread posterior reads over-confident once reweighted."""
    draws, truth = _make(1.0, seed=4)
    # weight each draw by exp(-||draw - mean||): concentrates mass at the center
    centers = draws.mean(axis=1, keepdims=True)
    w = np.exp(-2.0 * np.linalg.norm(draws - centers, axis=2))
    res = expected_coverage(draws, truth, weights=w, seed=3)
    assert res.atc < expected_coverage(draws, truth, seed=3).atc


def test_ecp_curve_shape():
    draws, truth = _make(1.0, seed=1)
    res = expected_coverage(draws, truth, n_alpha=50, seed=3)
    assert res.alpha.shape == res.ecp.shape == (50,)
    assert res.alpha[0] == 0.0 and res.alpha[-1] == 1.0
    assert np.all(np.diff(res.ecp) >= -1e-9)  # ECP is a CDF: non-decreasing
    assert res.credibility.shape == (500,)


def test_shape_validation():
    with pytest.raises(ValueError, match="n_rep, L, d"):
        expected_coverage(np.zeros((10, 3)), np.zeros((10, 3)))
    with pytest.raises(ValueError, match="theta_star must be"):
        expected_coverage(np.zeros((10, 5, 3)), np.zeros((10, 2)))
    with pytest.raises(ValueError, match="at least 2 replicates"):
        expected_coverage(np.zeros((1, 5, 3)), np.zeros((1, 3)))

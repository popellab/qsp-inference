"""Unit tests for the quantile-anchor summary contract (qsp_inference.targets).

The generalization of the hierarchical summary from a single IQR scale to the
quantile-anchor form maple's ObservedDistribution reports. Tests target the pure
core (anchors_from_sources / cohort_quantiles) so no maple I/O is needed.
"""
import numpy as np
import pytest

from qsp_inference.targets import (
    ObservedAnchors,
    anchors_from_sources,
    cohort_quantiles,
)
from qsp_inference.targets.anchors import _resolvable_grid


class _StubOD:
    """Minimal duck-type of a maple ObservedDistribution."""

    def __init__(self, pairs, feeds):
        self._pairs = pairs
        self.feeds_population_spread = feeds

    def _anchor_pairs(self):
        return list(self._pairs)


# --- anchors_from_sources ---------------------------------------------------

def test_samples_branch_uses_empirical_quantiles():
    rng = np.random.default_rng(0)
    arr = rng.normal(10.0, 2.0, size=5000)
    anchors, n_s = anchors_from_sources(
        ["obs0"], [arr], np.array([10.0]), np.array([6.0]), np.array([14.0]),
    )
    assert n_s == 1
    a = anchors[0]
    assert a.source == "samples" and a.feeds_spread is True
    assert a.p_levels == (0.25, 0.5, 0.75)
    np.testing.assert_allclose(a.values, np.quantile(arr, [0.25, 0.5, 0.75]))


def test_ci95_fallback_when_no_samples():
    """No samples -> lognormal expansion off the median + ci95, flagged NOT
    population spread (a center/measurement interval)."""
    med, lo, hi = 100.0, 50.0, 200.0
    anchors, n_s = anchors_from_sources(
        ["obs0"], [None], np.array([med]), np.array([lo]), np.array([hi]),
    )
    assert n_s == 0
    a = anchors[0]
    assert a.source == "ci95" and a.feeds_spread is False
    assert a.values[1] == pytest.approx(med)           # median anchor exact (z=0)
    assert a.values[0] * a.values[2] == pytest.approx(med * med, rel=1e-6)  # log-symmetric


def test_observed_distribution_takes_precedence():
    """An ObservedDistribution wins over samples, and its own p-levels (deciles
    here) are used verbatim."""
    od = _StubOD([(0.1, 1.0), (0.5, 5.0), (0.9, 9.0)], feeds=True)
    arr = np.random.default_rng(1).normal(size=1000)
    anchors, _ = anchors_from_sources(
        ["obs0"], [arr], np.array([5.0]), np.array([1.0]), np.array([9.0]),
        observed_distributions=[od],
    )
    a = anchors[0]
    assert a.source == "observed_distribution"
    assert a.p_levels == (0.1, 0.5, 0.9)
    np.testing.assert_allclose(a.values, [1.0, 5.0, 9.0])
    assert a.feeds_spread is True


def test_custom_grid_deciles_from_samples():
    rng = np.random.default_rng(2)
    arr = rng.normal(size=8000)
    grid = (0.1, 0.5, 0.9)
    anchors, _ = anchors_from_sources(
        ["o"], [arr], np.array([0.0]), np.array([-2.0]), np.array([2.0]),
        quantiles=grid,
    )
    assert anchors[0].p_levels == grid
    np.testing.assert_allclose(anchors[0].values, np.quantile(arr, grid), atol=1e-9)


def test_too_few_samples_falls_back_to_ci95():
    anchors, n_s = anchors_from_sources(
        ["o"], [np.array([1.0, 2.0])], np.array([5.0]),
        np.array([2.0]), np.array([12.0]), min_samples=4,
    )
    assert n_s == 0 and anchors[0].source == "ci95"


def test_as_pairs_matches_quantile_vpc_form():
    a = ObservedAnchors((0.25, 0.5, 0.75), np.array([1.0, 2.0, 3.0]), True, "samples")
    assert a.as_pairs() == [(0.25, 1.0), (0.5, 2.0), (0.75, 3.0)]


def test_bad_quantiles_raise():
    with pytest.raises(ValueError, match="quantiles must lie in"):
        anchors_from_sources(["o"], [None], np.array([1.0]),
                             np.array([0.5]), np.array([2.0]), quantiles=(0.0, 0.5))


# --- finite-sample handling (n-subsample + grid clip) -----------------------

def test_resolvable_grid_clips_by_n():
    # n=2 can resolve only the median out of the quartiles; n=6 keeps all.
    assert _resolvable_grid((0.25, 0.5, 0.75), 2) == (0.5,)
    assert _resolvable_grid((0.25, 0.5, 0.75), 6) == (0.25, 0.5, 0.75)
    # deciles need a larger n to resolve the 0.1/0.9 tails.
    assert 0.1 not in _resolvable_grid((0.1, 0.5, 0.9), 6)
    assert _resolvable_grid((0.1, 0.5, 0.9), 20) == (0.1, 0.5, 0.9)
    # n=None leaves the grid untouched.
    assert _resolvable_grid((0.1, 0.9), None) == (0.1, 0.9)


def test_n_grid_clip_applied_to_samples_branch():
    rng = np.random.default_rng(10)
    arr = rng.normal(size=5000)
    anchors, _ = anchors_from_sources(
        ["o"], [arr], np.array([0.0]), np.array([-2.0]), np.array([2.0]),
        quantiles=(0.1, 0.5, 0.9), n=6,
    )
    # n=6 cannot resolve the 0.1/0.9 deciles -> only the median survives.
    assert anchors[0].p_levels == (0.5,)


def test_observed_at_n_is_noisier_than_full_pool():
    """The n-subsampled observed differs from the denoised full-pool quantile
    and varies with the seed (finite-sample noise), while n=None reproduces the
    population quantile."""
    rng = np.random.default_rng(11)
    arr = rng.normal(size=20000)
    full, _ = anchors_from_sources(
        ["o"], [arr], np.array([0.0]), np.array([-2.0]), np.array([2.0]),
        quantiles=(0.25, 0.5, 0.75), n=None,
    )
    at_n_a, _ = anchors_from_sources(
        ["o"], [arr], np.array([0.0]), np.array([-2.0]), np.array([2.0]),
        quantiles=(0.25, 0.5, 0.75), n=30, seed=1,
    )
    at_n_b, _ = anchors_from_sources(
        ["o"], [arr], np.array([0.0]), np.array([-2.0]), np.array([2.0]),
        quantiles=(0.25, 0.5, 0.75), n=30, seed=2,
    )
    # full-pool ~ population quantiles; n=30 draw deviates from them...
    np.testing.assert_allclose(full[0].values, np.quantile(arr, [0.25, 0.5, 0.75]))
    assert not np.allclose(at_n_a[0].values, full[0].values, atol=1e-3)
    # ...and different seeds give different observed draws.
    assert not np.allclose(at_n_a[0].values, at_n_b[0].values)


def test_observed_at_n_is_seed_reproducible():
    rng = np.random.default_rng(12)
    arr = rng.normal(size=5000)
    a1, _ = anchors_from_sources(["o"], [arr], np.array([0.0]),
                                 np.array([-2.0]), np.array([2.0]), n=20, seed=7)
    a2, _ = anchors_from_sources(["o"], [arr], np.array([0.0]),
                                 np.array([-2.0]), np.array([2.0]), n=20, seed=7)
    np.testing.assert_array_equal(a1[0].values, a2[0].values)


def test_n_per_observable_sequence():
    rng = np.random.default_rng(13)
    arrs = [rng.normal(size=5000), rng.normal(size=5000)]
    anchors, _ = anchors_from_sources(
        ["a", "b"], arrs, np.array([0.0, 0.0]), np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]), quantiles=(0.1, 0.5, 0.9), n=[2, 50],
    )
    assert anchors[0].p_levels == (0.5,)              # n=2 -> median only
    assert anchors[1].p_levels == (0.1, 0.5, 0.9)     # n=50 -> full grid


def test_observed_distribution_ignores_n():
    """Reported anchors are already n-statistics -> taken verbatim, no clip/draw."""
    od = _StubOD([(0.05, -2.0), (0.5, 0.0), (0.95, 2.0)], feeds=True)
    anchors, _ = anchors_from_sources(
        ["o"], [np.random.default_rng(0).normal(size=1000)],
        np.array([0.0]), np.array([-2.0]), np.array([2.0]),
        observed_distributions=[od], n=3,   # n=3 would clip 0.05/0.95 if applied
    )
    assert anchors[0].p_levels == (0.05, 0.5, 0.95)


# --- cohort_quantiles -------------------------------------------------------

def test_cohort_quantiles_shape_and_values():
    """Uniform {0.25,0.5,0.75} grid over 2 observables -> 6 summary columns; the
    per-cohort quantiles match a direct nanpercentile."""
    rng = np.random.default_rng(3)
    n_cohorts, n_cohort, n_obs = 5, 200, 2
    x = rng.normal(size=(n_cohorts * n_cohort, n_obs))
    grids = [(0.25, 0.5, 0.75), (0.25, 0.5, 0.75)]
    summary, valid = cohort_quantiles(x, n_cohorts, n_cohort, grids, min_patients=10)
    assert summary.shape == (n_cohorts, 6)
    assert valid.all()
    xr = x.reshape(n_cohorts, n_cohort, n_obs)
    np.testing.assert_allclose(summary[:, 0], np.nanpercentile(xr[:, :, 0], 25, axis=1))
    np.testing.assert_allclose(summary[:, 4], np.nanpercentile(xr[:, :, 1], 50, axis=1))


def test_cohort_quantiles_ragged_widths():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(3 * 50, 2))
    grids = [(0.5,), (0.1, 0.5, 0.9)]     # 1 + 3 = 4 columns
    summary, _ = cohort_quantiles(x, 3, 50, grids, min_patients=5)
    assert summary.shape == (3, 4)


def test_cohort_quantiles_marks_sparse_cohorts_invalid():
    x = np.full((2 * 10, 1), np.nan)
    x[:5, 0] = 1.0  # only 5 finite patients
    summary, valid = cohort_quantiles(x, 2, 10, [(0.5,)], min_patients=6)
    assert not valid.any()


def test_cohort_quantiles_shape_mismatch_raises():
    with pytest.raises(ValueError, match="anchor_p_levels has"):
        cohort_quantiles(np.zeros((10, 2)), 2, 5, [(0.5,)], min_patients=1)

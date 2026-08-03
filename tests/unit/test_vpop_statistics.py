"""Unit tests for the Beta order-statistic kernel (qsp_inference.vpop.statistics).

The claim under test is that eq:smoothq returns E[what the cohort printed], so the
checks are against simulated n-samples rather than against a population quantile.
"""
import numpy as np
import pytest

from qsp_inference.vpop.statistics import (
    QUANTILE_CONVENTIONS,
    expected_quantile,
    order_statistic_mass,
)

N_CLOUD = 200_000
REP = 100_000


@pytest.fixture(scope="module")
def clouds():
    """Log-scale clouds of three shapes; the kernel should not care which."""
    rng = np.random.default_rng(0)
    return {
        "normal": np.sort(rng.normal(0.0, 1.0, N_CLOUD)),
        "skewed": np.sort(rng.gumbel(0.0, 1.0, N_CLOUD)),
        "heavy": np.sort(rng.standard_t(3, N_CLOUD)),
    }


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(1)
    return {
        "normal": rng.normal(0.0, 1.0, (REP, 16)),
        "skewed": rng.gumbel(0.0, 1.0, (REP, 16)),
        "heavy": rng.standard_t(3, (REP, 16)),
    }


class TestMass:
    def test_sums_to_one(self):
        w = np.random.default_rng(2).exponential(1.0, 5_000)
        for kappa in (1, 3, 12, 20):
            assert order_statistic_mass(w, kappa, 20).sum() == pytest.approx(1.0)

    def test_mean_rank_is_kappa_over_n_plus_one(self):
        """The kernel's centre, which is what shifts a small-n quartile inward."""
        n, w = 9, np.ones(20_000)
        mid = (np.arange(20_000) + 0.5) / 20_000
        for kappa in (3, 5, 7):
            m = order_statistic_mass(w, kappa, n)
            assert mid @ m == pytest.approx(kappa / (n + 1), abs=2e-4)

    def test_weights_shift_the_mass(self):
        """Eligibility zeroing the lower half moves the kernel onto the upper."""
        w = np.ones(1_000)
        w[:500] = 0.0
        m = order_statistic_mass(w, 5, 9)
        assert m[:500].sum() == pytest.approx(0.0, abs=1e-12)
        assert m[500:].sum() == pytest.approx(1.0)


class TestExpectedQuantile:
    @pytest.mark.parametrize("shape", ["normal", "skewed", "heavy"])
    @pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
    def test_matches_the_simulated_sample_statistic(self, clouds, draws, shape, p):
        got = expected_quantile(clouds[shape], np.ones(N_CLOUD), p, 16)
        want = np.quantile(draws[shape], p, axis=1).mean()
        assert got == pytest.approx(want, abs=0.02)

    def test_small_n_quartiles_sit_inside_the_population(self, clouds):
        """The effect the whole construction exists for: inward on both sides."""
        cloud, w = clouds["normal"], np.ones(N_CLOUD)
        lo = expected_quantile(cloud, w, 0.25, 9)
        hi = expected_quantile(cloud, w, 0.75, 9)
        pop_lo, pop_hi = np.quantile(cloud, [0.25, 0.75])
        assert pop_lo < lo < 0 < hi < pop_hi
        assert (hi - lo) / (pop_hi - pop_lo) == pytest.approx(0.845, abs=0.02)

    def test_large_n_recovers_the_population_quantile(self, clouds):
        cloud, w = clouds["normal"], np.ones(N_CLOUD)
        for p in (0.25, 0.5, 0.75):
            got = expected_quantile(cloud, w, p, 5_000)
            assert got == pytest.approx(np.quantile(cloud, p), abs=0.01)

    def test_the_median_is_not_shifted(self, clouds):
        cloud, w = clouds["normal"], np.ones(N_CLOUD)
        for n in (9, 25, 101):
            assert expected_quantile(cloud, w, 0.5, n) == pytest.approx(0.0, abs=0.01)

    def test_conventions_disagree_at_small_n(self, clouds):
        """The unrecorded choice, worth about 0.4 sampling SE at n=9."""
        cloud, w = clouds["normal"], np.ones(N_CLOUD)
        t7 = expected_quantile(cloud, w, 0.25, 9, "type7")
        t6 = expected_quantile(cloud, w, 0.25, 9, "type6")
        assert t6 < t7                       # type 6 targets 0.25, type 7 targets 0.30
        assert abs(t7 - t6) == pytest.approx(0.18, abs=0.03)

    def test_conventions_agree_at_large_n(self, clouds):
        cloud, w = clouds["normal"], np.ones(N_CLOUD)
        vals = [expected_quantile(cloud, w, 0.25, 500, c) for c in QUANTILE_CONVENTIONS]
        assert max(vals) - min(vals) < 0.01

    def test_kappa_is_clamped_to_the_sample(self, clouds):
        """p=0 and p=1 must land on the first and last order statistic, not off the end."""
        cloud, w = clouds["normal"], np.ones(N_CLOUD)
        assert np.isfinite(expected_quantile(cloud, w, 0.0, 6))
        assert np.isfinite(expected_quantile(cloud, w, 1.0, 6))
        assert expected_quantile(cloud, w, 0.0, 6) < expected_quantile(cloud, w, 1.0, 6)
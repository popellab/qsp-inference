"""Unit tests for the row functionals (qsp_inference.vpop.statistics).

The claim under test is that each row returns E[what the cohort printed], so the
checks are against simulated n-samples rather than against a population value.
"""
import jax
import numpy as np
import pytest

from qsp_inference.vpop.statistics import (
    QUANTILE_CONVENTIONS,
    bootstrap_design,
    expected_quantile,
    extreme_row,
    iqr_row,
    mean_row,
    order_statistic_mass,
    sd_row,
    se_row,
)

N_CLOUD = 200_000
REP = 100_000
SHAPES = ["normal", "skewed", "heavy", "lognorm"]


@pytest.fixture(autouse=True)
def _x64():
    """Per test, not at import: submodel.inference turns x64 off mid-session."""
    jax.config.update("jax_enable_x64", True)


def _draw(rng, shape, size):
    return {
        "normal": lambda: rng.normal(0.0, 1.0, size),
        "skewed": lambda: rng.gumbel(0.0, 1.0, size),
        "heavy": lambda: rng.standard_t(3, size),
        "lognorm": lambda: rng.lognormal(0.0, 1.0, size),
    }[shape]()


@pytest.fixture(scope="module")
def clouds():
    rng = np.random.default_rng(0)
    return {s: np.sort(_draw(rng, s, N_CLOUD)) for s in SHAPES}


@pytest.fixture(scope="module")
def draws():
    rng = np.random.default_rng(1)
    return {s: _draw(rng, s, (REP, 16)) for s in SHAPES}


@pytest.fixture(scope="module")
def ones():
    return np.ones(N_CLOUD)


class TestMass:
    def test_sums_to_one(self):
        w = np.random.default_rng(2).exponential(1.0, 5_000)
        for kappa in (1, 3, 12, 20):
            assert float(order_statistic_mass(w, kappa, 20).sum()) == pytest.approx(1.0)

    def test_mean_rank_is_kappa_over_n_plus_one(self):
        n, w = 9, np.ones(20_000)
        mid = (np.arange(20_000) + 0.5) / 20_000
        for kappa in (3, 5, 7):
            m = np.asarray(order_statistic_mass(w, kappa, n))
            assert mid @ m == pytest.approx(kappa / (n + 1), abs=2e-4)

    def test_weights_shift_the_mass(self):
        w = np.ones(1_000)
        w[:500] = 0.0
        m = np.asarray(order_statistic_mass(w, 5, 9))
        assert m[:500].sum() == pytest.approx(0.0, abs=1e-12)
        assert m[500:].sum() == pytest.approx(1.0)


class TestQuantileRows:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
    def test_matches_the_simulated_sample_statistic(self, clouds, draws, ones, shape, p):
        got = float(expected_quantile(clouds[shape], ones, p, 16))
        want = np.quantile(draws[shape], p, axis=1).mean()
        assert got == pytest.approx(want, rel=0.02, abs=0.02)

    def test_small_n_quartiles_sit_inside_the_population(self, clouds, ones):
        cloud = clouds["normal"]
        lo = float(expected_quantile(cloud, ones, 0.25, 9))
        hi = float(expected_quantile(cloud, ones, 0.75, 9))
        pop_lo, pop_hi = np.quantile(cloud, [0.25, 0.75])
        assert pop_lo < lo < 0 < hi < pop_hi
        assert (hi - lo) / (pop_hi - pop_lo) == pytest.approx(0.845, abs=0.02)

    def test_large_n_recovers_the_population_quantile(self, clouds, ones):
        cloud = clouds["normal"]
        for p in (0.25, 0.5, 0.75):
            assert float(expected_quantile(cloud, ones, p, 5_000)) == pytest.approx(
                np.quantile(cloud, p), abs=0.01
            )

    def test_conventions_disagree_at_small_n(self, clouds, ones):
        cloud = clouds["normal"]
        t7 = float(expected_quantile(cloud, ones, 0.25, 9, "type7"))
        t6 = float(expected_quantile(cloud, ones, 0.25, 9, "type6"))
        assert t6 < t7
        assert abs(t7 - t6) == pytest.approx(0.18, abs=0.03)

    def test_conventions_agree_at_large_n(self, clouds, ones):
        vals = [float(expected_quantile(clouds["normal"], ones, 0.25, 500, c))
                for c in QUANTILE_CONVENTIONS]
        assert max(vals) - min(vals) < 0.01


class TestMeanAndIqr:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_mean_needs_no_correction(self, clouds, ones, shape):
        assert float(mean_row(clouds[shape], ones)) == pytest.approx(
            clouds[shape].mean(), rel=1e-9
        )

    @pytest.mark.parametrize("shape", SHAPES)
    def test_iqr_matches_the_simulated_sample_iqr(self, clouds, draws, ones, shape):
        q = np.quantile(draws[shape], [0.25, 0.75], axis=1)
        want = (q[1] - q[0]).mean()
        assert float(iqr_row(clouds[shape], ones, 16)) == pytest.approx(
            want, rel=0.03
        )

    def test_iqr_is_narrower_than_the_population_at_small_n(self, clouds, ones):
        cloud = clouds["normal"]
        pop = float(np.diff(np.quantile(cloud, [0.25, 0.75]))[0])
        assert float(iqr_row(cloud, ones, 12)) / pop == pytest.approx(0.88, abs=0.03)


class TestExtremes:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("upper", [False, True])
    def test_matches_the_simulated_sample_extreme(self, clouds, draws, ones,
                                                  shape, upper):
        want = (draws[shape].max(axis=1) if upper
                else draws[shape].min(axis=1)).mean()
        got = float(extreme_row(clouds[shape], ones, 16, upper))
        assert got == pytest.approx(want, abs=0.15 * np.std(draws[shape]))

    def test_n_of_one_is_the_mean(self, clouds, ones):
        # Beta(1,1) is uniform, so both endpoints put equal mass on every member.
        for upper in (False, True):
            assert float(extreme_row(clouds["normal"], ones, 1, upper)) == \
                pytest.approx(float(mean_row(clouds["normal"], ones)), rel=1e-9)

    def test_the_extremes_spread_with_n(self, clouds, ones):
        cloud = clouds["normal"]
        lo = [float(extreme_row(cloud, ones, n, False)) for n in (8, 200)]
        hi = [float(extreme_row(cloud, ones, n, True)) for n in (8, 200)]
        assert lo[1] < lo[0] < hi[0] < hi[1]


class TestMomentRows:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("n", [8, 21])
    def test_sd_matches_the_simulated_sample_sd(self, clouds, ones, shape, n):
        rng = np.random.default_rng(5)
        want = _draw(rng, shape, (60_000, n)).std(axis=1, ddof=1).mean()
        u = bootstrap_design(jax.random.PRNGKey(0), n, n_boot=4_000)
        assert float(sd_row(clouds[shape], ones, u)) == pytest.approx(want, rel=0.05)

    def test_sd_is_biased_low_and_more_so_when_skewed(self, clouds, ones):
        """The reason there is no closed form: the gap depends on the shape."""
        u = bootstrap_design(jax.random.PRNGKey(1), 8, n_boot=8_000)
        ratios = {
            s: float(sd_row(clouds[s], ones, u)) / clouds[s].std()
            for s in ("normal", "lognorm")
        }
        assert ratios["normal"] == pytest.approx(0.96, abs=0.04)
        assert ratios["lognorm"] < 0.85
        assert ratios["lognorm"] < ratios["normal"]

    def test_se_is_the_sd_over_root_n(self, clouds, ones):
        u = bootstrap_design(jax.random.PRNGKey(2), 11, n_boot=4_000)
        cloud = clouds["normal"]
        assert float(se_row(cloud, ones, u, 11)) == pytest.approx(
            float(sd_row(cloud, ones, u)) / np.sqrt(11), rel=1e-9
        )

    def test_log_forms_are_the_log_of_the_row(self, clouds, ones):
        u = bootstrap_design(jax.random.PRNGKey(3), 12, n_boot=2_000)
        cloud = clouds["normal"]
        assert float(sd_row(cloud, ones, u, log=True)) == pytest.approx(
            np.log(float(sd_row(cloud, ones, u))), rel=1e-9
        )

    def test_design_is_frozen_across_calls(self, clouds, ones):
        """Same design, same answer: no Monte Carlo noise in the gradient."""
        u = bootstrap_design(jax.random.PRNGKey(4), 10, n_boot=1_000)
        cloud = clouds["normal"]
        assert float(sd_row(cloud, ones, u)) == float(sd_row(cloud, ones, u))


class TestDifferentiability:
    def test_rows_are_differentiable_in_the_cloud(self, ones):
        """tau has to have a gradient; the hard quantile is what this replaces."""
        cloud = np.sort(np.random.default_rng(6).normal(0, 1, 4_000))
        w = np.ones(4_000)
        u = bootstrap_design(jax.random.PRNGKey(7), 9, n_boot=200)
        for fn in (
            lambda x: expected_quantile(x, w, 0.25, 9),
            lambda x: iqr_row(x, w, 9),
            lambda x: mean_row(x, w),
            lambda x: sd_row(x, w, u),
        ):
            g = jax.grad(lambda x: fn(x).sum() if fn(x).ndim else fn(x))(cloud)
            assert np.all(np.isfinite(g))
            assert np.any(np.asarray(g) != 0.0)

    def test_quantile_row_is_differentiable_in_the_weights(self):
        cloud = np.sort(np.random.default_rng(8).normal(0, 1, 4_000))
        g = jax.grad(lambda w: expected_quantile(cloud, w, 0.25, 9))(np.ones(4_000))
        assert np.all(np.isfinite(g))
        assert np.any(np.asarray(g) != 0.0)
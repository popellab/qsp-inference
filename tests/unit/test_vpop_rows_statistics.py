"""Unit tests for the row functionals (qsp_inference.vpop.rows).

The claim under test is that each row returns E[what the cohort printed], so the
checks are against simulated n-samples rather than against a population value.
"""
import jax
import numpy as np
import pytest

from qsp_inference.vpop.rows import (
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


class TestMass:
    def test_sums_to_one(self):
        for kappa in (1, 3, 12, 20):
            assert float(order_statistic_mass(5_000, kappa, 20).sum()) \
                == pytest.approx(1.0)

    def test_mean_rank_is_kappa_over_n_plus_one(self):
        n = 9
        mid = (np.arange(20_000) + 0.5) / 20_000
        for kappa in (3, 5, 7):
            m = np.asarray(order_statistic_mass(20_000, kappa, n))
            assert mid @ m == pytest.approx(kappa / (n + 1), abs=2e-4)

    def test_the_mass_reads_the_cloud_only_through_its_size(self):
        """No weights, so it is a constant of (N, kappa, n) and is built once."""
        a = np.asarray(order_statistic_mass(1_000, 5, 9))
        b = np.asarray(order_statistic_mass(1_000, 5, 9))
        assert np.array_equal(a, b)
        assert a.shape == (1_000,)
        assert np.all(np.diff(np.cumsum(a)) >= 0)


class TestQuantileRows:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("p", [0.25, 0.5, 0.75])
    def test_matches_the_simulated_sample_statistic(self, clouds, draws, shape, p):
        got = float(expected_quantile(clouds[shape], p, 16))
        want = np.quantile(draws[shape], p, axis=1).mean()
        assert got == pytest.approx(want, rel=0.02, abs=0.02)

    def test_small_n_quartiles_sit_inside_the_population(self, clouds):
        cloud = clouds["normal"]
        lo = float(expected_quantile(cloud, 0.25, 9))
        hi = float(expected_quantile(cloud, 0.75, 9))
        pop_lo, pop_hi = np.quantile(cloud, [0.25, 0.75])
        assert pop_lo < lo < 0 < hi < pop_hi
        assert (hi - lo) / (pop_hi - pop_lo) == pytest.approx(0.845, abs=0.02)

    def test_large_n_recovers_the_population_quantile(self, clouds):
        cloud = clouds["normal"]
        for p in (0.25, 0.5, 0.75):
            assert float(expected_quantile(cloud, p, 5_000)) == pytest.approx(
                np.quantile(cloud, p), abs=0.01
            )

    def test_conventions_disagree_at_small_n(self, clouds):
        cloud = clouds["normal"]
        t7 = float(expected_quantile(cloud, 0.25, 9, "type7"))
        t6 = float(expected_quantile(cloud, 0.25, 9, "type6"))
        assert t6 < t7
        assert abs(t7 - t6) == pytest.approx(0.18, abs=0.03)

    def test_conventions_agree_at_large_n(self, clouds):
        vals = [float(expected_quantile(clouds["normal"], 0.25, 500, c))
                for c in QUANTILE_CONVENTIONS]
        assert max(vals) - min(vals) < 0.01

    def test_every_declarable_convention_is_implemented(self):
        """A target may declare any name ``rows`` accepts, and the kernel runs it.

        These drifted apart once: ``rows`` took five names and the kernel had
        three, so a target declaring ``type2`` loaded clean and raised inside the
        sampler.
        """
        from qsp_inference.vpop.rows import NUMPY_QUANTILE_METHOD

        assert set(NUMPY_QUANTILE_METHOD) == set(QUANTILE_CONVENTIONS)

    @pytest.mark.parametrize("conv, p, n, want", [
        # Hyndman-Fan h, hand-checked. type2 averages at a discontinuity, which
        # the caller expresses as a half-weight between the two order statistics.
        ("type7", 0.25, 9, 3.0),
        ("type6", 0.25, 9, 2.5),
        ("type4", 0.25, 9, 2.25),
        ("type8", 0.25, 9, 2.6666666666666665),
        ("type2", 0.25, 9, 3.0),      # np = 2.25, not integer -> ceil
        ("type2", 0.25, 8, 2.5),      # np = 2, integer -> average x_(2), x_(3)
        ("type2", 0.5, 6, 3.5),       # np = 3, integer -> average x_(3), x_(4)
        ("type2", 0.75, 6, 5.0),      # np = 4.5, not integer -> ceil
    ])
    def test_h_matches_hyndman_fan(self, conv, p, n, want):
        assert QUANTILE_CONVENTIONS[conv](p, n) == pytest.approx(want)


class TestMeanAndIqr:
    @pytest.mark.parametrize("shape", SHAPES)
    def test_mean_needs_no_correction(self, clouds, shape):
        assert float(mean_row(clouds[shape])) == pytest.approx(
            clouds[shape].mean(), rel=1e-9
        )

    @pytest.mark.parametrize("shape", SHAPES)
    def test_iqr_matches_the_simulated_sample_iqr(self, clouds, draws, shape):
        q = np.quantile(draws[shape], [0.25, 0.75], axis=1)
        want = (q[1] - q[0]).mean()
        assert float(iqr_row(clouds[shape], 16)) == pytest.approx(
            want, rel=0.03
        )

    def test_iqr_is_narrower_than_the_population_at_small_n(self, clouds):
        cloud = clouds["normal"]
        pop = float(np.diff(np.quantile(cloud, [0.25, 0.75]))[0])
        assert float(iqr_row(cloud, 12)) / pop == pytest.approx(0.88, abs=0.03)


class TestExtremes:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("upper", [False, True])
    def test_matches_the_simulated_sample_extreme(self, clouds, draws,
                                                  shape, upper):
        want = (draws[shape].max(axis=1) if upper
                else draws[shape].min(axis=1)).mean()
        got = float(extreme_row(clouds[shape], 16, upper))
        assert got == pytest.approx(want, abs=0.15 * np.std(draws[shape]))

    def test_n_of_one_is_the_mean(self, clouds):
        # Beta(1,1) is uniform, so both endpoints put equal mass on every member.
        for upper in (False, True):
            assert float(extreme_row(clouds["normal"], 1, upper)) == \
                pytest.approx(float(mean_row(clouds["normal"])), rel=1e-9)

    def test_the_extremes_spread_with_n(self, clouds):
        cloud = clouds["normal"]
        lo = [float(extreme_row(cloud, n, False)) for n in (8, 200)]
        hi = [float(extreme_row(cloud, n, True)) for n in (8, 200)]
        assert lo[1] < lo[0] < hi[0] < hi[1]


class TestMomentRows:
    @pytest.mark.parametrize("shape", SHAPES)
    @pytest.mark.parametrize("n", [8, 21])
    def test_sd_matches_the_simulated_sample_sd(self, clouds, shape, n):
        rng = np.random.default_rng(5)
        want = _draw(rng, shape, (60_000, n)).std(axis=1, ddof=1).mean()
        u = bootstrap_design(jax.random.PRNGKey(0), n, n_boot=4_000)
        assert float(sd_row(clouds[shape], u)) == pytest.approx(want, rel=0.05)

    def test_sd_is_biased_low_and_more_so_when_skewed(self, clouds):
        """The reason there is no closed form: the gap depends on the shape."""
        u = bootstrap_design(jax.random.PRNGKey(1), 8, n_boot=8_000)
        ratios = {
            s: float(sd_row(clouds[s], u)) / clouds[s].std()
            for s in ("normal", "lognorm")
        }
        assert ratios["normal"] == pytest.approx(0.96, abs=0.04)
        assert ratios["lognorm"] < 0.85
        assert ratios["lognorm"] < ratios["normal"]

    def test_se_is_the_sd_over_root_n(self, clouds):
        u = bootstrap_design(jax.random.PRNGKey(2), 11, n_boot=4_000)
        cloud = clouds["normal"]
        assert float(se_row(cloud, u, 11)) == pytest.approx(
            float(sd_row(cloud, u)) / np.sqrt(11), rel=1e-9
        )

    def test_log_forms_take_the_expectation_inside_the_log(self, clouds):
        """eq:obs's mean is E[log s], which sits BELOW log E[s] by the Jensen gap.

        The row printed a log, so the expectation belongs there. Taking it
        outside reports a larger number, and on the width rows that is the only
        evidence about omega.
        """
        u = bootstrap_design(jax.random.PRNGKey(3), 12, n_boot=2_000)
        cloud = clouds["normal"]
        inside = float(sd_row(cloud, u, log=True))
        outside = np.log(float(sd_row(cloud, u)))
        assert inside < outside
        # Jensen's gap is second order in the sampling CV, so it is small and
        # strictly positive rather than a different quantity altogether.
        assert outside - inside == pytest.approx(0.0, abs=0.2)

    def test_design_is_frozen_across_calls(self, clouds):
        """Same design, same answer: no Monte Carlo noise in the gradient."""
        u = bootstrap_design(jax.random.PRNGKey(4), 10, n_boot=1_000)
        cloud = clouds["normal"]
        assert float(sd_row(cloud, u)) == float(sd_row(cloud, u))


class TestDifferentiability:
    def test_rows_are_differentiable_in_the_cloud(self):
        """tau has to have a gradient; the hard quantile is what this replaces."""
        cloud = np.sort(np.random.default_rng(6).normal(0, 1, 4_000))
        w = np.ones(4_000)
        u = bootstrap_design(jax.random.PRNGKey(7), 9, n_boot=200)
        for fn in (
            lambda x: expected_quantile(x, 0.25, 9),
            lambda x: iqr_row(x, 9),
            lambda x: mean_row(x),
            lambda x: sd_row(x, u),
        ):
            g = jax.grad(lambda x: fn(x).sum() if fn(x).ndim else fn(x))(cloud)
            assert np.all(np.isfinite(g))
            assert np.any(np.asarray(g) != 0.0)

def test_mass_edges_stay_inside_the_unit_interval():
    """betainc is nan just outside [0, 1].

    The edges used to be ``cumsum(w) / sum(w)``, which are probabilities by
    definition and not by arithmetic: the two reduce in different orders, so the
    last one could land an ulp over and return nan. Platform-dependent, which is
    how it passed on arm64 and failed on x86. Unweighted, linspace gives the
    edges exactly and there is nothing left to clip.
    """
    jax.config.update("jax_enable_x64", True)
    for n_cloud in (999, 5_000, 65_536):
        for kappa in (1, 3, 12, 20):
            got = float(order_statistic_mass(n_cloud, kappa, 20).sum())
            assert np.isfinite(got), f"N={n_cloud} kappa={kappa} gave {got}"
            assert got == pytest.approx(1.0)

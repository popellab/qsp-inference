"""Tests for weighted simulation-based calibration.

The substantive tests are built on a conjugate normal model where every object
the gate touches is available in closed form, so a failure is a bug in the gate
and not a Monte Carlo accident:

    prior     pi        = N(0, 1)
    proposal  pi_tilde  = N(0, T)          (exactly temper_prior's effect in 1D)
    data      x | theta = N(theta, s^2)

The posterior under the proposal is N(m_t, v_t) with v_t = 1/(1/T + 1/s^2), and
multiplying it by pi/pi_tilde gives back the posterior under the prior, N(m, v)
with v = 1/(1 + 1/s^2), exactly. So a run that draws from the proposal posterior
and reweights is a run whose reported posterior is exact, and its weighted PIT
must be uniform. The same draws left unweighted must not be.

The default s = 3 puts the model in the weakly-identified regime this package
actually operates in: the likelihood barely moves the prior, so the proposal's
extra width survives into the posterior and forgetting to reweight is visible.
At s = 1 the same omission is worth only about 8% in posterior scale, and the
gate does not see it at 400 replicates. That is a property of the problem rather
than of the gate. Where the data determine theta, the prior trained under stops
mattering, which is also why only the unidentified directions need any of this.
"""

import numpy as np
import pytest
from scipy import stats

from qsp_inference.inference.sbc import (
    SBCResult,
    plot_sbc_ecdf,
    uniform_band_halfwidth,
    weighted_pit,
    weighted_sbc,
)


# ---------------------------------------------------------------------------
# conjugate harness
# ---------------------------------------------------------------------------
def _conjugate_sbc(
    n_rep=400,
    n_draws=2000,
    temperature=4.0,
    s=3.0,
    weighted=True,
    sd_scale=1.0,
    mean_shift=0.0,
    seed=0,
):
    """Run the conjugate SBC above and return (pit, ess, theta_star).

    ``sd_scale`` and ``mean_shift`` corrupt the reported posterior deliberately,
    which is how the detection tests are built.
    """
    rng = np.random.default_rng(seed)
    v_t = 1.0 / (1.0 / temperature + 1.0 / s**2)  # posterior var under pi_tilde

    theta_star = rng.normal(0.0, 1.0, size=n_rep)  # theta* ~ pi, as SBC requires
    x = theta_star + rng.normal(0.0, s, size=n_rep)
    m_t = v_t * x / s**2

    pit = np.empty((n_rep, 1))
    ess = np.empty(n_rep)
    for i in range(n_rep):
        draws = rng.normal(m_t[i] + mean_shift, sd_scale * np.sqrt(v_t), size=(n_draws, 1))
        if weighted:
            # log pi - log pi_tilde, up to a constant that self-normalizes away.
            lw = -0.5 * draws[:, 0] ** 2 * (1.0 - 1.0 / temperature)
            w = np.exp(lw - lw.max())
            w /= w.sum()
            ess[i] = 1.0 / np.sum(w**2)
        else:
            w = None
            ess[i] = n_draws
        pit[i] = weighted_pit(draws, theta_star[i : i + 1], w)
    return pit, ess, theta_star[:, None]


# ---------------------------------------------------------------------------
# the statistic
# ---------------------------------------------------------------------------
class TestWeightedPIT:
    def test_uniform_weights_reduce_to_the_midrank(self):
        rng = np.random.default_rng(1)
        draws = rng.normal(size=(37, 3))
        truth = rng.normal(size=3)

        got = weighted_pit(draws, truth)
        r = (draws < truth[None, :]).sum(axis=0)
        expected = (r + 0.5) / (draws.shape[0] + 1)

        assert got == pytest.approx(expected)

    def test_explicit_weights_equal_to_uniform_match_no_weights(self):
        rng = np.random.default_rng(2)
        draws = rng.normal(size=(50, 2))
        truth = rng.normal(size=2)
        w = np.full(50, 1.0 / 50)
        assert weighted_pit(draws, truth, w) == pytest.approx(weighted_pit(draws, truth))

    def test_unnormalized_weights_are_normalized(self):
        rng = np.random.default_rng(3)
        draws = rng.normal(size=(40, 2))
        truth = rng.normal(size=2)
        w = rng.uniform(0.1, 2.0, size=40)
        assert weighted_pit(draws, truth, w) == pytest.approx(weighted_pit(draws, truth, w * 137.0))

    def test_weight_concentrated_on_one_draw(self):
        """m -> 1, so the PIT collapses to (F + 1/2)/2: 0.25 or 0.75."""
        draws = np.array([[0.0], [10.0]])
        w = np.array([1.0, 1e-300])
        assert weighted_pit(draws, np.array([5.0]), w)[0] == pytest.approx(0.75, abs=1e-6)
        assert weighted_pit(draws, np.array([-5.0]), w)[0] == pytest.approx(0.25, abs=1e-6)

    def test_never_hits_the_boundary(self):
        """Truth below every draw, and above every draw."""
        draws = np.zeros((100, 1))
        assert 0.0 < weighted_pit(draws + 1.0, np.array([0.0]))[0] < 1.0
        assert 0.0 < weighted_pit(draws - 1.0, np.array([0.0]))[0] < 1.0

    def test_ties_split_evenly(self):
        draws = np.array([[0.0], [1.0], [1.0], [2.0]])
        # one strictly below, two tied: F = 0.25 + 0.5*0.5 = 0.5
        assert weighted_pit(draws, np.array([1.0]))[0] == pytest.approx((4 * 0.5 + 0.5) / 5)

    def test_uniform_under_the_null(self):
        """theta* and draws from the same distribution: PIT is uniform."""
        rng = np.random.default_rng(4)
        pit = np.array(
            [weighted_pit(rng.normal(size=(500, 1)), rng.normal(size=1)) for _ in range(600)]
        ).ravel()
        assert stats.kstest(pit, "uniform").pvalue > 0.01
        assert pit.mean() == pytest.approx(0.5, abs=0.03)

    def test_rejects_shape_mismatches(self):
        with pytest.raises(ValueError, match="theta_star has"):
            weighted_pit(np.zeros((10, 3)), np.zeros(2))
        with pytest.raises(ValueError, match="weights has"):
            weighted_pit(np.zeros((10, 2)), np.zeros(2), np.ones(9))
        with pytest.raises(ValueError, match="non-negative"):
            weighted_pit(np.zeros((3, 1)), np.zeros(1), np.array([1.0, -1.0, 1.0]))
        with pytest.raises(ValueError, match="positive finite"):
            weighted_pit(np.zeros((3, 1)), np.zeros(1), np.zeros(3))


# ---------------------------------------------------------------------------
# the point of the module: the weighted path is checked, the unweighted is not
# ---------------------------------------------------------------------------
class TestDecoupledPathIsWhatGetsChecked:
    def test_reweighted_draws_are_calibrated(self):
        pit, ess, theta_star = _conjugate_sbc(weighted=True, seed=101)
        res = weighted_sbc(pit, ess=ess, theta_star=theta_star, param_names=["theta"])
        assert res.gate(min_replicates=100).passed, res.summary()

    def test_calibrated_case_is_unbiased_across_seeds(self):
        """The single-seed test above is a spot check; this is the claim.

        Under the null each z is standard normal, so a single seed lands beyond
        2 about 9% of the time for one moment or the other. Averaging over seeds
        is what distinguishes an unlucky draw from a biased statistic: the mean
        of ten z's has standard error 0.32, so a systematic offset from the
        continuity correction or from self-normalized weights would show.
        """
        zm, zv = [], []
        for seed in range(100, 110):
            pit, ess, ts = _conjugate_sbc(n_rep=300, n_draws=800, weighted=True, seed=seed)
            p = weighted_sbc(pit, ess=ess, theta_star=ts).parameters[0]
            zm.append(p.z_mean)
            zv.append(p.z_variance)
        assert abs(np.mean(zm)) < 1.0, f"mean z_mean = {np.mean(zm):.3f}"
        assert abs(np.mean(zv)) < 1.0, f"mean z_variance = {np.mean(zv):.3f}"

    def test_resolution_is_set_by_ess_not_by_draw_count(self):
        """The PIT stops improving once the weights stop carrying more draws."""
        zs = []
        for n_draws in (500, 2000, 8000):
            pit, ess, ts = _conjugate_sbc(n_draws=n_draws, weighted=True, seed=11)
            zs.append(weighted_sbc(pit, ess=ess, theta_star=ts).parameters[0].z_variance)
        # Same replicates, same theta*; more draws only refines each F-hat, so
        # the verdict must not wander as the draw count grows.
        assert max(zs) - min(zs) < 0.5, zs

    def test_the_same_draws_unweighted_are_not(self):
        """Training on pi_tilde and forgetting to reweight is exactly this."""
        pit, ess, theta_star = _conjugate_sbc(weighted=False, seed=11)
        res = weighted_sbc(pit, ess=ess, theta_star=theta_star, param_names=["theta"])
        gate = res.gate(min_replicates=100)
        assert not gate.passed
        assert not gate.inconclusive
        assert gate.failing == ["theta"]
        # The proposal posterior is wider than the true one, so the truth lands
        # in the middle too often.
        assert res.by_name("theta").shape == "wide"
        assert res.by_name("theta").variance < 1.0 / 12.0

    def test_higher_temperature_costs_effective_sample_size(self):
        med = []
        for T in (1.5, 4.0, 12.0):
            _, ess, _ = _conjugate_sbc(n_rep=60, n_draws=1000, temperature=T, seed=5)
            med.append(float(np.median(ess)))
        assert med == sorted(med, reverse=True)


class TestDetectsMiscalibration:
    """Corruption tests run at T = 1, so the weights are uniform and the knobs
    act on the reported posterior directly.

    At T > 1 the reweighting partially repairs a widened posterior (the weight
    is itself a normal factor, so it caps how diffuse the reweighted sample can
    be), which would confound "does the gate detect a bad posterior" with "does
    reweighting work". Those are separate questions and the class above owns the
    second one.
    """

    def test_overconfident_posterior_reads_as_narrow(self):
        pit, ess, theta_star = _conjugate_sbc(temperature=1.0, sd_scale=0.6, seed=21)
        res = weighted_sbc(pit, ess=ess, theta_star=theta_star, param_names=["theta"])
        assert res.by_name("theta").shape == "narrow"
        assert res.by_name("theta").variance > 1.0 / 12.0
        assert not res.gate(min_replicates=100).passed

    def test_underconfident_posterior_reads_as_wide(self):
        pit, ess, theta_star = _conjugate_sbc(temperature=1.0, sd_scale=1.8, seed=22)
        res = weighted_sbc(pit, ess=ess, theta_star=theta_star, param_names=["theta"])
        assert res.by_name("theta").shape == "wide"
        assert res.by_name("theta").variance < 1.0 / 12.0
        assert not res.gate(min_replicates=100).passed

    def test_posterior_shifted_low_reads_as_biased_low(self):
        """Draws pushed down, so theta* sits above them: PIT mean > 1/2."""
        pit, ess, theta_star = _conjugate_sbc(temperature=1.0, mean_shift=-0.7, seed=23)
        res = weighted_sbc(pit, ess=ess, theta_star=theta_star, param_names=["theta"])
        p = res.by_name("theta")
        assert p.mean > 0.5
        assert p.shape == "biased low"
        assert not res.gate(min_replicates=100).passed

    def test_posterior_shifted_high_reads_as_biased_high(self):
        pit, ess, theta_star = _conjugate_sbc(temperature=1.0, mean_shift=0.7, seed=24)
        p = weighted_sbc(pit, ess=ess, theta_star=theta_star, param_names=["theta"]).by_name(
            "theta"
        )
        assert p.mean < 0.5
        assert p.shape == "biased high"

    def test_the_moment_tests_carry_cases_ks_alone_misses(self):
        """Why the gate is not a bare KS test.

        A posterior 25% too wide is a modest, purely second-moment defect. KS
        sits near its critical value; the variance test does not.
        """
        pit, ess, ts = _conjugate_sbc(temperature=1.0, sd_scale=1.25, n_rep=400, seed=25)
        p = weighted_sbc(pit, ess=ess, theta_star=ts, param_names=["theta"]).by_name("theta")
        assert p.ks_pvalue > 0.01, f"pick a subtler corruption: ks_p={p.ks_pvalue:.3g}"
        assert p.pvalue_variance < 1e-3
        assert p.pvalue < 0.01


# ---------------------------------------------------------------------------
# assembly, multiplicity, gating
# ---------------------------------------------------------------------------
def _uniform_pit(n_rep, n_par, seed=0):
    return np.random.default_rng(seed).uniform(size=(n_rep, n_par))


class TestAssembly:
    def test_default_names(self):
        res = weighted_sbc(_uniform_pit(50, 3))
        assert res.param_names == ["p0", "p1", "p2"]
        assert res.n_replicates == 50 and res.n_params == 3

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError, match="n_replicates, n_params"):
            weighted_sbc(np.zeros(10))
        with pytest.raises(ValueError, match="at least 2 replicates"):
            weighted_sbc(np.zeros((1, 2)))
        with pytest.raises(ValueError, match="must lie in"):
            weighted_sbc(np.full((10, 1), 1.5))
        with pytest.raises(ValueError, match="param_names has"):
            weighted_sbc(_uniform_pit(20, 3), param_names=["a", "b"])
        with pytest.raises(ValueError, match="ess has"):
            weighted_sbc(_uniform_pit(20, 2), ess=np.ones(19))
        with pytest.raises(ValueError, match="theta_star has shape"):
            weighted_sbc(_uniform_pit(20, 2), theta_star=np.zeros((20, 3)))

    def test_median_ess_is_none_without_weights(self):
        assert weighted_sbc(_uniform_pit(30, 2)).median_ess is None
        assert weighted_sbc(_uniform_pit(30, 2), ess=np.full(30, 7.0)).median_ess == 7.0

    def test_table_is_sorted_worst_first(self):
        pd = pytest.importorskip("pandas")
        pit = _uniform_pit(200, 4, seed=7)
        pit[:, 2] = np.random.default_rng(8).beta(0.4, 0.4, size=200)  # U-shaped
        df = weighted_sbc(pit).table()
        assert isinstance(df, pd.DataFrame)
        assert df.loc[0, "parameter"] == "p2"


class TestUninformativeParameters:
    def test_pinned_parameter_is_excluded(self):
        pit = _uniform_pit(200, 3, seed=9)
        theta_star = np.random.default_rng(10).normal(size=(200, 3))
        theta_star[:, 1] = 4.2  # pinned: no variation for SBC to check
        res = weighted_sbc(pit, theta_star=theta_star)
        assert [p.informative for p in res.parameters] == [True, False, True]
        assert np.isnan(res.by_name("p1").pvalue_adjusted)
        assert res.gate(min_replicates=100).passed
        assert any("uninformative" in r for r in res.gate(min_replicates=100).reasons)

    def test_uninformative_parameters_do_not_dilute_the_correction(self):
        """A real failure must survive being surrounded by pinned parameters."""
        rng = np.random.default_rng(12)
        n_par = 60
        pit = rng.uniform(size=(300, n_par))
        pit[:, 0] = rng.beta(0.35, 0.35, size=300)  # genuinely U-shaped
        theta_star = np.zeros((300, n_par))
        theta_star[:, 0] = rng.normal(size=300)
        theta_star[:, 1] = rng.normal(size=300)

        res = weighted_sbc(pit, theta_star=theta_star)
        gate = res.gate(min_replicates=100)
        assert not gate.passed
        assert gate.failing == ["p0"]

    def test_all_uninformative_is_inconclusive(self):
        res = weighted_sbc(_uniform_pit(200, 2), theta_star=np.ones((200, 2)))
        gate = res.gate(min_replicates=100)
        assert gate.inconclusive and not gate.passed


class TestGate:
    def test_too_few_replicates_is_inconclusive_not_a_pass(self):
        gate = weighted_sbc(_uniform_pit(20, 2)).gate(min_replicates=100)
        assert gate.inconclusive and not gate.passed
        assert any("power" in r for r in gate.reasons)

    def test_low_ess_is_inconclusive(self):
        res = weighted_sbc(_uniform_pit(300, 2), ess=np.full(300, 5.0))
        gate = res.gate(min_replicates=100, min_ess=20.0)
        assert gate.inconclusive and not gate.passed
        assert any("ESS" in r for r in gate.reasons)

    def test_bool_protocol_tracks_passed(self):
        good = weighted_sbc(_uniform_pit(300, 2, seed=13)).gate(min_replicates=100)
        assert bool(good) is good.passed is True

    def test_summary_strings_render(self):
        res = weighted_sbc(_uniform_pit(300, 3, seed=14), ess=np.full(300, 500.0))
        assert "weighted SBC" in res.summary()
        assert "PASSED" in res.gate(min_replicates=100).summary()

    def test_alpha_controls_strictness(self):
        pit = _uniform_pit(300, 5, seed=15)
        pit[:, 1] = stats.beta(0.85, 0.85).rvs(300, random_state=16)  # mildly U
        res = weighted_sbc(pit)
        assert len(res.gate(alpha=0.5).failing) >= len(res.gate(alpha=1e-6).failing)


class TestBenjaminiHochberg:
    def test_adjusted_never_below_raw_and_is_monotone(self):
        rng = np.random.default_rng(17)
        pit = rng.uniform(size=(200, 12))
        pit[:, 3] = rng.beta(0.3, 0.3, size=200)
        res = weighted_sbc(pit)
        raw = np.array([p.pvalue for p in res.parameters])
        adj = np.array([p.pvalue_adjusted for p in res.parameters])
        assert np.all(adj >= raw - 1e-12)
        assert np.all(adj <= 1.0)
        o = np.argsort(raw)
        assert np.all(np.diff(adj[o]) >= -1e-12)

    def test_single_parameter_is_unadjusted(self):
        res = weighted_sbc(_uniform_pit(100, 1, seed=18))
        p = res.parameters[0]
        assert p.pvalue_adjusted == pytest.approx(p.pvalue)

    def test_combined_pvalue_is_bonferroni_over_the_three_tests(self):
        res = weighted_sbc(_uniform_pit(200, 3, seed=26))
        for p in res.parameters:
            assert p.pvalue == pytest.approx(
                min(3.0 * min(p.ks_pvalue, p.pvalue_mean, p.pvalue_variance), 1.0)
            )


class TestBandAndPlot:
    def test_band_matches_the_kolmogorov_quantile(self):
        assert uniform_band_halfwidth(100) == pytest.approx(1.3581 / 10.0, rel=1e-3)
        assert uniform_band_halfwidth(400) < uniform_band_halfwidth(100)
        with pytest.raises(ValueError):
            uniform_band_halfwidth(0)

    def test_plot_returns_axes_for_every_parameter(self):
        plt = pytest.importorskip("matplotlib.pyplot")
        res = weighted_sbc(_uniform_pit(150, 3, seed=19))
        fig, axes = plot_sbc_ecdf(res, figsize=(6, 4))
        assert isinstance(res, SBCResult)
        assert axes.size >= 3
        plt.close(fig)

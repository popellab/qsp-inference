"""Unit tests for the h_r channel: which coordinate each readout travels in.

A LOGIT_KINDS readout is carried as a log-odds so eq:disc acts where the
quantity's upper bound cannot be crossed; everything else stays on a log. The
claims here are that the two links are inverses where a fraction lives, that
eq:disc cannot leave (0, 1) on the logit channel for any kappa, and that a mixed
cloud inverts column by column rather than under one link for all of it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.predict import (
    Mechanism,
    apply_map,
    natural_cloud,
    reference_levels,
    tau_from_readouts,
)
from qsp_inference.vpop.blocks import BlockPlan, DrawGroup
from qsp_inference.vpop.rows import (
    LOGIT_KINDS,
    SCALE_BY_KIND,
    RowSpec,
    hard_row,
    logit_link,
    natural_from_logit,
    to_scale,
)

N = 2_000
DENS, FRAC = "r_density", "r_fraction"


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _g(vartheta, scenario):
    return jnp.exp(vartheta)


def _h(y, log_R=None):
    """A density on a log channel, a fraction on a log-odds one."""
    dens = jnp.log(y[0, :, 0])
    frac = logit_link(jax.nn.sigmoid(y[0, :, 1]), jnp)
    return jnp.stack([dens, frac], axis=-1)


@pytest.fixture
def mech():
    z = jnp.asarray(np.random.default_rng(0).standard_normal((N, 2)))
    return Mechanism(
        L_R=jnp.eye(2), z=z,
        Z_a=jnp.eye(2), Z_b=jnp.eye(2),
        readouts=(DENS, FRAC), n_species=2, n_scenarios=1,
        beta_species=jnp.zeros(0, dtype=int), g_fn=_g, h_fn=_h,
        logit_readout=np.array([False, True]),
    )


class TestLinks:
    def test_kinds_agree_with_the_comparison_scale(self):
        # One policy, read two ways. A kind carried on a log-odds and compared on
        # a logit is the same claim about where the quantity can be; if these
        # ever disagreed, V would standardise a different quantity from the one
        # eq:disc moved.
        assert LOGIT_KINDS == {k for k, v in SCALE_BY_KIND.items()
                               if v == "logit"}
        assert "fraction" in LOGIT_KINDS and "percentage" in LOGIT_KINDS
        assert "density" not in LOGIT_KINDS and "count" not in LOGIT_KINDS

    @pytest.mark.parametrize("x", [5e-4, 0.0014, 0.1, 0.5, 0.886, 0.99])
    def test_round_trip_is_exact_where_a_fraction_lives(self, x):
        assert float(natural_from_logit(logit_link(np.array(x), np), np)) == \
            pytest.approx(x, rel=1e-12)

    def test_inverse_is_onto_the_open_interval(self):
        # The whole point: whatever eq:disc does to the channel, the value that
        # comes back is a fraction. -800/+800 are far past anything a kappa
        # reaches, and both branches stay finite.
        for v in (-800.0, -40.0, 0.0, 40.0, 800.0):
            out = float(natural_from_logit(np.array(v), np))
            assert 0.0 <= out <= 1.0
            assert np.isfinite(out)

    def test_inverse_has_a_finite_gradient_in_both_tails(self):
        # 1/(1+exp(-x)) is right in value and gives inf/inf here under reverse
        # mode, which is the bug the |x| form exists to avoid.
        for v in (-800.0, 800.0):
            g = jax.grad(lambda t: natural_from_logit(t, jnp))(jnp.array(v))
            assert np.isfinite(float(g))


class TestDiscrepancyCannotLeaveTheInterval:
    @pytest.mark.parametrize("log_kappa", [-2.0, -0.5, 0.0, 0.615, 2.0, 5.0])
    def test_any_kappa_keeps_a_fraction_bounded(self, log_kappa):
        # 0.615 is the kappa measured on the pdac corpus, where a log channel put
        # stromal_fraction's smooth max at 1.105 and the logit tangent turned a
        # 7% overshoot into a 132-sigma residual.
        x = logit_link(np.linspace(0.68, 0.918, 64), np)
        c = float(np.median(x))
        mapped = apply_map(jnp.asarray(x), jnp.zeros(1), jnp.array([log_kappa]),
                           c, jnp.zeros((64, 1)), jnp.ones((64, 1)))
        frac = np.asarray(natural_from_logit(mapped, jnp))
        # Never ABOVE 1, which is the claim. Not "strictly below": a sigmoid
        # saturates to exactly 1.0 in float64 past about 37, so a large enough
        # kappa reaches the endpoint. What that costs is bounded, and the next
        # assertion is the bound -- against 112.9 on the log channel.
        assert frac.max() <= 1.0
        assert frac.min() >= 0.0
        spec = RowSpec(FRAC, "c", "quantile", 0.5, 40, p=0.5, scale="logit")
        on_scale = np.asarray(to_scale(frac, spec, np))
        assert np.isfinite(on_scale).all()
        assert on_scale.max() <= float(to_scale(np.array(1.0), spec, np)) + 1e-9

    def test_a_log_channel_does_not_stay_bounded(self):
        # The defect being fixed, stated as a test so the contrast is not folded
        # into a comment. Same cloud, same kappa, carried on a log instead.
        x = np.log(np.linspace(0.68, 0.918, 64))
        c = float(np.median(x))
        mapped = apply_map(jnp.asarray(x), jnp.zeros(1), jnp.array([0.615]),
                           c, jnp.zeros((64, 1)), jnp.ones((64, 1)))
        assert float(jnp.exp(mapped).max()) > 1.0


class TestNaturalCloud:
    def test_each_column_inverts_under_its_own_link(self, mech):
        x = jnp.asarray([[np.log(3.0), logit_link(np.array(0.25), np)],
                         [np.log(7.0), logit_link(np.array(0.80), np)]])
        out = np.asarray(natural_cloud(x, mech))
        assert out[:, 0] == pytest.approx([3.0, 7.0], rel=1e-12)
        assert out[:, 1] == pytest.approx([0.25, 0.80], rel=1e-12)

    def test_cols_selects_the_mask_too(self, mech):
        # A cohort reports a handful of the M readouts and is mapped on those
        # alone, so the mask has to be taken through the same selection. Passing
        # the fraction column as if it were column 0 is the mistake this catches.
        x = jnp.asarray([[logit_link(np.array(0.25), np)]])
        assert float(np.asarray(natural_cloud(x, mech, cols=(1,)))[0, 0]) == \
            pytest.approx(0.25, rel=1e-12)
        assert float(np.asarray(natural_cloud(x, mech, cols=(0,)))[0, 0]) == \
            pytest.approx(float(jnp.exp(x[0, 0])), rel=1e-12)

    def test_no_mask_is_the_all_log_path(self, mech):
        from dataclasses import replace
        bare = replace(mech, logit_readout=None)
        x = jnp.asarray([[0.5, -0.5]])
        assert np.asarray(natural_cloud(x, bare)) == \
            pytest.approx(np.asarray(jnp.exp(x)), rel=1e-12)


class TestEndToEnd:
    """tau lands on the row's comparison scale, and eq:disc is the identity at 0."""

    def _rows(self):
        return {"c": [
            RowSpec(DENS, "c", "quantile", 1.0, 40, p=0.5, scale="asinh",
                    scale_ref=5.0),
            RowSpec(FRAC, "c", "quantile", 0.5, 40, p=0.5, scale="logit"),
        ]}

    def _tau(self, mech, a, b):
        specs = self._rows()
        group = DrawGroup(("c",), 40, {"c": tuple(range(40))})
        plans = [BlockPlan(cohort_ids=("c",), groups=(group,))]
        mu, om = jnp.zeros(2), jnp.full((2,), 0.4)
        x = mech.h_fn(jnp.stack([mech.g_fn(
            mu[None, :] + mech.zL * om[None, :], 0)]))
        refs = reference_levels(mu, om, mech)
        return np.asarray(jnp.concatenate(tau_from_readouts(
            x, a, b, plans, specs, refs, mech))), x

    def test_identity_at_zero_discrepancy(self, mech):
        # The plug-in is the no-discrepancy point, so tau there is the row
        # functional of the unmapped cloud on its own comparison scale.
        tau, x = self._tau(mech, jnp.zeros(2), jnp.zeros(2))
        nat = np.asarray(natural_cloud(x, mech))
        for k, spec in enumerate(self._rows()["c"]):
            direct = hard_row(spec, nat[:, k])
            assert tau[k] == pytest.approx(direct, abs=0.05)

    def test_the_fraction_row_is_reported_in_logit(self, mech):
        tau, _ = self._tau(mech, jnp.zeros(2), jnp.zeros(2))
        # A logit-scale row's tau is a log-odds, so pushing it back through the
        # sigmoid has to give something inside the interval rather than the
        # fraction itself already being what tau reports.
        assert 0.0 < float(natural_from_logit(np.array(tau[1]), np)) < 1.0

    @pytest.mark.parametrize("log_kappa", [0.615, 2.0, 4.0])
    def test_the_fraction_row_stays_finite_under_any_kappa(self, mech, log_kappa):
        b = jnp.array([0.0, log_kappa])
        tau, _ = self._tau(mech, jnp.zeros(2), b)
        assert np.isfinite(tau).all()
        # to_scale's tangent bounds what a fraction row can report: below 1 in
        # natural units, so at most at_hi + LOGIT_MARGIN * slope on the scale.
        assert float(tau[1]) < float(to_scale(np.array(1.0), self._rows()["c"][1],
                                              np)) + 1e-9

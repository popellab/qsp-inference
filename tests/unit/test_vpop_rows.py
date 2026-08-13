"""Unit tests for the corpus row layer (qsp_inference.vpop.rows).

Two evaluators that must stay distinct: hard_row reproduces the source's own
estimator, tau_row predicts it.
"""
import dataclasses
import jax
import numpy as np
import pytest

from qsp_inference.vpop import rows as st
from qsp_inference.vpop.rows import (
    SCALE_STATS,
    by_cohort,
    hard_row,
    hard_rows_fn,
    row_specs,
    tau_row,
)


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _target(cohort_id, stats, n_evaluable=None, convention=None, kind="density"):
    ed = {"observed_distribution": {"statistics": stats, "spread_source": "across_patient"}}
    if convention:
        ed["observed_distribution"]["quantile_convention"] = convention
    if n_evaluable:
        ed["n_evaluable"] = n_evaluable
    out = {"cohort_id": cohort_id, "empirical_data": ed}
    if kind is not None:
        out["observable"] = {"readout": {"quantity_kind": kind}}
    return out


QUARTILES = [{"stat": "quantile", "p": 0.25, "value": 46.0},
             {"stat": "quantile", "p": 0.5, "value": 100.0},
             {"stat": "quantile", "p": 0.75, "value": 186.0}]
MEAN_SD = [{"stat": "mean", "value": 10.0}, {"stat": "sd", "value": 4.0}]


class TestRowSpecs:
    def test_one_row_per_printed_statistic(self):
        specs = row_specs({"t": _target("c", QUARTILES)}, {"c": 113})
        assert [s.stat for s in specs] == ["quantile"] * 3
        assert [s.p for s in specs] == [0.25, 0.5, 0.75]
        assert [s.value for s in specs] == [46.0, 100.0, 186.0]
        assert all(s.n == 113 for s in specs)

    def test_n_evaluable_beats_the_cohort_size(self):
        specs = row_specs({"t": _target("c", QUARTILES, n_evaluable=6)}, {"c": 9})
        assert all(s.n == 6 for s in specs)

    def test_missing_n_raises(self):
        with pytest.raises(ValueError, match="no n for cohort"):
            row_specs({"t": _target("ghost", QUARTILES)}, {"c": 9})

    def test_unsupported_statistics_raise_and_are_named(self):
        t = _target("c", [{"stat": "quantile", "p": 0.5, "value": 1.0},
                          {"stat": "ci95_lo", "value": 0.1},
                          {"stat": "ci95_hi", "value": 9.0}])
        with pytest.raises(ValueError, match=r"t/ci95_hi, t/ci95_lo"):
            row_specs({"t": t}, {"c": 10})

    def test_min_and_max_are_order_statistic_rows(self):
        t = _target("c", [{"stat": "quantile", "p": 0.5, "value": 1.0},
                          {"stat": "min", "value": 0.1},
                          {"stat": "max", "value": 9.0}])
        specs = row_specs({"t": t}, {"c": 10})
        by_stat = {s.stat: s for s in specs}
        assert set(by_stat) == {"quantile", "min", "max"}
        # One endpoint is not a width, so neither is held out by the flat fit.
        assert not any(s.is_scale or s.scale == "log" for s in specs)
        # All three are order statistics, so a monotone transform passes through
        # the expectation exactly and none of them needs a bootstrap design.
        assert all(s.commutes for s in specs)

    def test_exclude_drops_a_row_by_name(self):
        t = _target("c", [{"stat": "quantile", "p": 0.5, "value": 1.0},
                          {"stat": "ci95_lo", "value": 0.1},
                          {"stat": "ci95_hi", "value": 9.0}])
        specs = row_specs({"t": t}, {"c": 10},
                          exclude=[("t", "ci95_lo"), ("t", "ci95_hi")])
        assert [s.stat for s in specs] == ["quantile"]

    def test_exclude_matching_nothing_raises(self):
        # A corpus edit that removes the row must not leave the exclusion silently
        # covering nothing, or the reason recorded for it stops applying.
        with pytest.raises(ValueError, match=r"t/iqr"):
            row_specs({"t": _target("c", MEAN_SD)}, {"c": 20},
                      exclude=[("t", "sd"), ("t", "iqr")])

    def test_scale_rows_take_log_whatever_the_quantity_is(self):
        # A width is positive by construction and its sampling distribution is
        # right-skewed, whether the underlying quantity is a density or a
        # fraction. asinh would put it in the linear regime, since scale_ref is
        # the size of the level rather than of the spread.
        for kind in ("density", "fraction", "foldchange"):
            specs = row_specs({"t": _target("c", MEAN_SD, kind=kind)}, {"c": 20})
            by_stat = {s.stat: s for s in specs}
            assert by_stat["sd"].is_scale and by_stat["sd"].scale == "log"
            assert not by_stat["mean"].is_scale

    def test_log_scale_rows_can_be_turned_off(self):
        specs = row_specs({"t": _target("c", MEAN_SD)}, {"c": 20}, log_scale_rows=False)
        assert all(s.scale != "log" for s in specs if s.is_scale)

    def test_location_scale_comes_from_the_quantity_kind(self):
        for kind, want in [("foldchange", "asinh"), ("ratio", "asinh"),
                           ("density", "asinh"), ("concentration", "asinh"),
                           ("time", "asinh"), ("fraction", "logit")]:
            specs = row_specs({"t": _target("c", QUARTILES, kind=kind)}, {"c": 20})
            assert {s.scale for s in specs} == {want}, kind

    def test_a_kind_with_no_scale_raises(self):
        # Raw is a claim, not the absence of one, so it is not a fallback.
        with pytest.raises(ValueError, match=r"no scale.*t \(luminosity\)"):
            row_specs({"t": _target("c", QUARTILES, kind="luminosity")}, {"c": 20})
        with pytest.raises(ValueError, match=r"declares none"):
            row_specs({"t": _target("c", QUARTILES, kind=None)}, {"c": 20})

    def test_scale_ref_is_the_level_and_is_shared_across_a_targets_rows(self):
        # The median of the printed LOCATION values: 46, 100, 186 -> 100. The sd
        # is excluded, since a scale set from a width would put every location
        # row in asinh's linear regime.
        stats = QUARTILES + [{"stat": "sd", "value": 0.001}]
        specs = row_specs({"t": _target("c", stats, kind="density")}, {"c": 20})
        assert {s.scale_ref for s in specs} == {100.0}

    def test_only_the_mean_needs_a_design_among_location_rows(self):
        stats = [{"stat": "quantile", "p": 0.5, "value": 1.0},
                 {"stat": "mean", "value": 1.0}]
        specs = row_specs({"t": _target("c", stats, kind="density")}, {"c": 20})
        by_stat = {s.stat: s for s in specs}
        assert by_stat["quantile"].commutes
        assert not by_stat["mean"].commutes

    def test_unrecorded_convention_falls_back_and_is_flagged(self):
        specs = row_specs({"t": _target("c", QUARTILES)}, {"c": 113})
        assert all(s.convention == "type7" and not s.convention_recorded for s in specs)

    def test_recorded_convention_wins(self):
        specs = row_specs({"t": _target("c", QUARTILES, convention="type6")}, {"c": 113},
                          default_convention="type7")
        assert all(s.convention == "type6" and s.convention_recorded for s in specs)

    def test_scale_stats_come_from_the_schema(self):
        assert SCALE_STATS == {"sd", "se", "iqr", "cv", "range"}


class TestByCohort:
    def test_groups_and_preserves_order(self):
        specs = row_specs(
            {"a": _target("c1", QUARTILES), "b": _target("c2", MEAN_SD),
             "c": _target("c1", MEAN_SD)},
            {"c1": 50, "c2": 20},
        )
        bc = by_cohort(specs)
        assert set(bc) == {"c1", "c2"}
        assert len(bc["c1"]) == 5 and len(bc["c2"]) == 2
        assert [s.target_id for s in bc["c1"]] == ["a", "a", "a", "c", "c"]


class TestHardRow:
    @pytest.fixture
    def values(self):
        return np.random.default_rng(0).lognormal(0, 1, 9)

    def test_reproduces_numpy(self, values):
        (q,) = row_specs({"t": _target("c", [{"stat": "quantile", "p": 0.25,
                                              "value": 1.0}])}, {"c": 9})
        # On raw, so this checks the estimator and not the scale on top of it.
        raw = dataclasses.replace(q, scale="raw")
        assert hard_row(raw, values) == pytest.approx(np.quantile(values, 0.25))
        # And the scale is applied pathwise, which for an order statistic is
        # exact: g(q_p(v)) = q_p(g(v)).
        assert hard_row(q, values) == pytest.approx(
            np.arcsinh(np.quantile(values, 0.25) / q.scale_ref))

    def test_convention_changes_the_answer(self, values):
        t7 = row_specs({"t": _target("c", [{"stat": "quantile", "p": 0.25, "value": 1.0}])},
                       {"c": 9})[0]
        t6 = row_specs({"t": _target("c", [{"stat": "quantile", "p": 0.25, "value": 1.0}],
                                     convention="type6")}, {"c": 9})[0]
        assert hard_row(t6, values) != pytest.approx(hard_row(t7, values))

    def test_moment_and_width_rows(self, values):
        specs = {s.stat: s for s in row_specs(
            {"t": _target("c", MEAN_SD + [{"stat": "se", "value": 1.0},
                                          {"stat": "iqr", "value": 2.0}])}, {"c": 9})}
        # A location row takes its quantity's scale; scale_ref is the median of
        # the target's printed location values, which here is the mean row's 10.
        assert specs["mean"].scale == "asinh" and specs["mean"].scale_ref == 10.0
        assert hard_row(specs["mean"], values) == pytest.approx(
            np.arcsinh(values.mean() / 10.0))
        # Width rows stay on log whatever the quantity is.
        assert hard_row(specs["sd"], values) == pytest.approx(
            np.log(values.std(ddof=1)))
        assert hard_row(specs["se"], values) == pytest.approx(
            np.log(values.std(ddof=1) / 3.0))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        assert hard_row(specs["iqr"], values) == pytest.approx(np.log(q75 - q25))


class TestHardRowsFn:
    def test_returns_K_c_in_spec_order(self):
        specs = row_specs({"a": _target("c", QUARTILES), "b": _target("c", MEAN_SD)},
                          {"c": 12})
        cloud = {"a": np.arange(100.0), "b": np.arange(100.0) * 2}
        fn = hard_rows_fn(by_cohort(specs), cloud)
        got = fn("c", np.arange(12))
        assert got.shape == (5,)
        # Order, not values: each entry is its own spec's row, on its own scale.
        by_spec = by_cohort(specs)["c"]
        assert got[1] == pytest.approx(hard_row(by_spec[1], np.arange(12.0)))
        assert got[3] == pytest.approx(hard_row(by_spec[3], np.arange(12.0) * 2))


class TestTauRow:
    @pytest.fixture
    def cloud(self):
        return np.sort(np.random.default_rng(1).normal(0, 1, 50_000))

    def test_quantile_and_mean_dispatch(self, cloud):
        x = cloud
        specs = {s.stat: s for s in row_specs(
            {"t": _target("c", [{"stat": "quantile", "p": 0.25, "value": 1.0},
                                {"stat": "mean", "value": 1.0}])}, {"c": 9})}
        # The quantile row reads the transformed cloud, so it is expected_quantile
        # of g(x) and not g of expected_quantile.
        gx = np.asarray(st.to_scale(x, specs["quantile"], np))
        assert float(tau_row(specs["quantile"], x)) == pytest.approx(
            float(st.expected_quantile(gx, 0.25, 9)))
        # A transformed mean does not commute, so tau is E[g(mhat)] and not
        # g(E[mhat]). It takes that as an expansion in the cloud's own moments
        # rather than over the frozen design, so it needs no design and reads no
        # ordering -- see mean_row_scaled.
        assert float(tau_row(specs["mean"], x)) == pytest.approx(
            float(st.mean_row_scaled(x, specs["mean"])))
        raw = dataclasses.replace(specs["mean"], scale="raw")
        assert float(tau_row(raw, x)) == pytest.approx(float(st.mean_row(x)))

    def test_a_logged_iqr_row_is_E_log_iqr_and_needs_a_design(self, cloud):
        """Not log E[IQR]: the row printed a log, so the expectation goes there."""
        import jax

        x = cloud
        (spec,) = row_specs({"t": _target("c", [{"stat": "iqr", "value": 2.0},
                                                {"stat": "quantile", "p": 0.5,
                                                 "value": 1.0}])}, {"c": 12})[:1]
        with pytest.raises(ValueError, match="needs a bootstrap design"):
            tau_row(spec, x)
        u = st.bootstrap_design(jax.random.PRNGKey(0), 12, n_boot=2_000)
        got = float(tau_row(spec, x, design=u))
        assert got < float(np.log(st.iqr_row(x, 12)))

    def test_moment_rows_need_a_design(self, cloud):
        x = cloud
        (spec,) = [s for s in row_specs({"t": _target("c", MEAN_SD)}, {"c": 20})
                   if s.stat == "sd"]
        with pytest.raises(ValueError, match="needs a bootstrap design"):
            tau_row(spec, x)
        u = st.bootstrap_design(jax.random.PRNGKey(0), 20, n_boot=200)
        assert np.isfinite(float(tau_row(spec, x, u)))

    def test_tau_predicts_what_hard_row_computes(self, cloud):
        """The two evaluators must agree in expectation; that is the whole claim."""
        x = cloud
        (spec,) = row_specs({"t": _target("c", [{"stat": "quantile", "p": 0.25,
                                                 "value": 1.0}])}, {"c": 9})
        rng = np.random.default_rng(2)
        draws = np.asarray(x)[rng.integers(0, len(x), size=(20_000, 9))]
        want = np.array([hard_row(spec, d) for d in draws]).mean()
        assert float(tau_row(spec, x)) == pytest.approx(want, abs=0.02)


class TestScales:
    """eq:obs's coordinate: what each row is compared on, and that both sides agree."""

    def _spec(self, stat, scale, **kw):
        return st.RowSpec("t", "c", stat, 1.0, 9, scale=scale, **kw)

    def test_asinh_is_log_like_above_the_reference_and_linear_below(self):
        spec = self._spec("quantile", "asinh", p=0.5, scale_ref=10.0)
        # Far above: asinh(x/s) -> log(2x/s), so a decade in x is a decade in the
        # transform. This is the property that tames a fold change's right tail.
        big = st.to_scale(np.array([1e6, 1e7]), spec, np)
        assert np.isclose(big[1] - big[0], np.log(10.0), atol=1e-5)
        # Far below: linear, so zero is an ordinary value and needs no clip.
        assert np.isclose(float(st.to_scale(np.array(0.0), spec, np)), 0.0)
        assert np.isfinite(float(st.to_scale(np.array(0.0), spec, np)))
        # Defined for negatives, which `time` needs: a doubling time is negative
        # when the tumour shrinks.
        assert np.isclose(float(st.to_scale(np.array(-30.0), spec, np)),
                          -float(st.to_scale(np.array(30.0), spec, np)))

    def test_asinh_tames_a_tail_log_would_leave_unbounded(self):
        # The failure this exists for: a raw fold change whose cloud reaches 1e24
        # against data at 20 gives a row variance no metric can use.
        spec = self._spec("quantile", "asinh", p=0.5, scale_ref=6.6)
        assert float(st.to_scale(np.array(1e24), spec, np)) < 60.0

    def test_logit_is_exact_where_the_data_lives(self):
        spec = self._spec("quantile", "logit", p=0.5)
        # The tangent continuation must not touch anything real. 0.886 is the
        # largest fraction this corpus prints.
        for x in (1e-6, 0.0014, 0.5, 0.8864):
            assert float(st.to_scale(np.array(x), spec, np)) == pytest.approx(
                np.log(x / (1.0 - x)), rel=1e-12)

    def test_logit_stays_differentiable_past_one(self):
        # A fraction above 1 contradicts the declared kind, but it must report
        # itself as a large residual rather than kill the chain. A clip would
        # invent a value and flatten the gradient, and a flat gradient under a
        # downstream sqrt is the 0 * inf that puts NaN in a Jacobian while the
        # forward pass stays finite.
        import jax
        import jax.numpy as jnp
        spec = self._spec("quantile", "logit", p=0.5)
        grad = jax.grad(lambda y: st.to_scale(y, spec, jnp))
        for x in (0.9, 0.999, 1.0, 1.1, 2.0):
            assert np.isfinite(float(st.to_scale(np.array(x), spec, np)))
            assert np.isfinite(float(grad(jnp.array(x))))
        # Monotone across the handover, and the violation is visible.
        assert (float(st.to_scale(np.array(1.1), spec, np))
                > float(st.to_scale(np.array(0.9), spec, np)) + 50)

    def test_logit_is_C1_at_the_handover(self):
        spec = self._spec("quantile", "logit", p=0.5)
        hi = 1.0 - st.LOGIT_MARGIN
        lo = float(st.to_scale(np.array(hi - 1e-9), spec, np))
        up = float(st.to_scale(np.array(hi + 1e-9), spec, np))
        assert up - lo == pytest.approx(2e-9 / (hi * (1.0 - hi)), rel=1e-3)

    def test_both_sides_use_one_function(self):
        # hard_row and tau_row must land in the same coordinate or V describes a
        # different quantity from the residual it standardises.
        import jax.numpy as jnp
        for scale in ("raw", "log", "asinh", "logit"):
            spec = self._spec("quantile", scale, p=0.5, scale_ref=2.0)
            x = np.array(0.3)
            assert np.isclose(float(st.to_scale(x, spec, np)),
                              float(st.to_scale(jnp.asarray(x), spec, jnp)))

    def test_an_unknown_scale_raises_rather_than_passing_through(self):
        spec = self._spec("quantile", "sqrt", p=0.5)
        with pytest.raises(ValueError, match="unknown scale"):
            st.to_scale(np.array(1.0), spec, np)

    @pytest.mark.parametrize("scale,ref", [("asinh", 100.0), ("logit", 1.0),
                                           ("log", 1.0)])
    def test_an_order_statistic_row_reads_the_transformed_cloud(self, scale, ref):
        # What makes this exact for the 106 order-statistic rows: g is monotone
        # increasing, so it carries the quantile function of x to that of g(x)
        # and the SAME Beta kernel gives E[g(q_p)] off the transformed cloud.
        rng = np.random.default_rng(0)
        cloud = np.sort(rng.uniform(0.05, 0.95, 400)
                        * (100.0 if scale == "asinh" else 1.0))
        spec = self._spec("quantile", scale, p=0.5, scale_ref=ref)
        raw = st.RowSpec("t", "c", "quantile", 1.0, 9, p=0.5, scale="raw")
        got = float(tau_row(spec, cloud))
        want = float(tau_row(raw, np.asarray(st.to_scale(cloud, spec, np))))
        assert np.isclose(got, want, rtol=1e-12)

    def test_transforming_after_the_expectation_would_be_wrong(self):
        # The trap: g(E[q_p]) is not E[g(q_p)]. Applying g to tau_row's raw
        # output instead of to the cloud misses by the Jensen gap, which is what
        # eq:stat's expectation is defined to include.
        rng = np.random.default_rng(3)
        cloud = np.sort(rng.lognormal(0.0, 1.0, 5_000))
        spec = self._spec("quantile", "asinh", p=0.25, scale_ref=1.0)
        raw = st.RowSpec("t", "c", "quantile", 1.0, 9, p=0.25, scale="raw")
        outside = float(st.to_scale(np.asarray(tau_row(raw, cloud)), spec, np))
        assert not np.isclose(float(tau_row(spec, cloud)), outside, rtol=1e-3)

    def test_tau_predicts_hard_row_in_expectation_on_a_transformed_row(self):
        # The claim eq:stat makes, checked on a transformed row rather than only
        # on a raw one: tau_row is the mean of hard_row over n-samples.
        rng = np.random.default_rng(4)
        cloud = np.sort(rng.lognormal(0.0, 1.0, 20_000))
        spec = self._spec("quantile", "asinh", p=0.25, scale_ref=1.0)
        draws = cloud[rng.integers(0, len(cloud), size=(20_000, 9))]
        want = np.array([hard_row(spec, d) for d in draws]).mean()
        assert float(tau_row(spec, cloud)) == pytest.approx(want, abs=0.02)

    def test_hard_row_agrees_with_the_transform_of_the_untransformed_row(self):
        rng = np.random.default_rng(1)
        v = rng.uniform(1.0, 50.0, 40)
        spec = self._spec("quantile", "asinh", p=0.75, scale_ref=10.0)
        raw = st.RowSpec("t", "c", "quantile", 1.0, 9, p=0.75, scale="raw")
        assert np.isclose(hard_row(spec, v),
                          float(st.to_scale(np.array(hard_row(raw, v)), spec, np)))

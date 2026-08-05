"""Unit tests for the corpus row layer (qsp_inference.vpop.rows).

Two evaluators that must stay distinct: hard_row reproduces the source's own
estimator, tau_row predicts it.
"""
import jax
import numpy as np
import pytest

from qsp_inference.vpop import statistics as st
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


def _target(cohort_id, stats, n_evaluable=None, convention=None):
    ed = {"observed_distribution": {"statistics": stats, "spread_source": "across_patient"}}
    if convention:
        ed["observed_distribution"]["quantile_convention"] = convention
    if n_evaluable:
        ed["n_evaluable"] = n_evaluable
    return {"cohort_id": cohort_id, "empirical_data": ed}


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
        assert not any(s.is_scale or s.log for s in specs)

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

    def test_scale_rows_are_logged_and_location_rows_are_not(self):
        specs = row_specs({"t": _target("c", MEAN_SD)}, {"c": 20})
        by_stat = {s.stat: s for s in specs}
        assert by_stat["sd"].is_scale and by_stat["sd"].log
        assert not by_stat["mean"].is_scale and not by_stat["mean"].log

    def test_log_scale_rows_can_be_turned_off(self):
        specs = row_specs({"t": _target("c", MEAN_SD)}, {"c": 20}, log_scale_rows=False)
        assert not any(s.log for s in specs)

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
        assert hard_row(q, values) == pytest.approx(np.quantile(values, 0.25))

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
        assert hard_row(specs["mean"], values) == pytest.approx(values.mean())
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
        assert got[1] == pytest.approx(np.quantile(np.arange(12.0), 0.5))
        assert got[3] == pytest.approx(np.arange(12.0).mean() * 2)


class TestTauRow:
    @pytest.fixture
    def cloud(self):
        return np.sort(np.random.default_rng(1).normal(0, 1, 50_000)), np.ones(50_000)

    def test_quantile_and_mean_dispatch(self, cloud):
        x, w = cloud
        specs = {s.stat: s for s in row_specs(
            {"t": _target("c", [{"stat": "quantile", "p": 0.25, "value": 1.0},
                                {"stat": "mean", "value": 1.0}])}, {"c": 9})}
        assert float(tau_row(specs["quantile"], x, w)) == pytest.approx(
            float(st.expected_quantile(x, w, 0.25, 9)))
        assert float(tau_row(specs["mean"], x, w)) == pytest.approx(float(st.mean_row(x, w)))

    def test_iqr_is_logged_when_the_row_is(self, cloud):
        x, w = cloud
        (spec,) = row_specs({"t": _target("c", [{"stat": "iqr", "value": 2.0},
                                                {"stat": "quantile", "p": 0.5,
                                                 "value": 1.0}])}, {"c": 12})[:1]
        assert float(tau_row(spec, x, w)) == pytest.approx(
            float(np.log(st.iqr_row(x, w, 12))))

    def test_moment_rows_need_a_design(self, cloud):
        x, w = cloud
        (spec,) = [s for s in row_specs({"t": _target("c", MEAN_SD)}, {"c": 20})
                   if s.stat == "sd"]
        with pytest.raises(ValueError, match="needs a bootstrap design"):
            tau_row(spec, x, w)
        u = st.bootstrap_design(jax.random.PRNGKey(0), 20, n_boot=200)
        assert np.isfinite(float(tau_row(spec, x, w, u)))

    def test_tau_predicts_what_hard_row_computes(self, cloud):
        """The two evaluators must agree in expectation; that is the whole claim."""
        x, w = cloud
        (spec,) = row_specs({"t": _target("c", [{"stat": "quantile", "p": 0.25,
                                                 "value": 1.0}])}, {"c": 9})
        rng = np.random.default_rng(2)
        draws = np.asarray(x)[rng.integers(0, len(x), size=(20_000, 9))]
        want = np.array([hard_row(spec, d) for d in draws]).mean()
        assert float(tau_row(spec, x, w)) == pytest.approx(want, abs=0.02)

"""Unit tests for the width gate (qsp_inference.vpop.gates)."""
import numpy as np
import pytest

from qsp_inference.vpop.gates import WidthGate, WidthRow, paired_width_rows
from qsp_inference.vpop.rows import RowSpec

TAU_S, SIGMA_B = 0.3, 0.5
BUDGET = float(np.hypot(TAU_S, SIGMA_B))


def _row(shortfall, sd=1.0, n=20, label="t/sd"):
    return WidthRow(label=label, cohort_id="c", n=n, observed=shortfall,
                    predicted=0.0, sd=sd, logged=True)


def _gate(*shortfalls, sd=1.0):
    return WidthGate(tuple(_row(s, sd) for s in shortfalls), BUDGET)


class TestRow:
    def test_shortfall_is_observed_minus_predicted(self):
        r = WidthRow("t/sd", "c", 20, observed=2.5, predicted=1.0, sd=0.5,
                     logged=True)
        assert r.shortfall == pytest.approx(1.5)
        assert r.z == pytest.approx(3.0)

    def test_a_zero_error_bar_gives_nan_rather_than_inf(self):
        # A row with no bootstrap spread cannot be scored, and reporting inf
        # would make it the worst row in every listing.
        assert np.isnan(WidthRow("t/sd", "c", 20, 1.0, 0.0, 0.0, True).z)


class TestGate:
    def test_positive_shortfall_means_the_cloud_is_too_narrow(self):
        # The source reports more spread than the model can produce.
        assert _gate(0.4, 0.6).mean_shortfall > 0

    def test_the_budget_is_the_prior_sd_of_s_plus_b1(self):
        assert _gate(0.0).budget == pytest.approx(0.583, abs=1e-3)

    def test_shortfall_is_read_against_the_budget(self):
        g = _gate(BUDGET, BUDGET)
        assert g.prior_sd == pytest.approx(1.0)

    def test_sign_does_not_change_reachability(self):
        # Too wide is as far from right as too narrow; the gate reports the
        # distance, and which way to move it is the fit's business.
        assert _gate(-0.4).prior_sd == _gate(0.4).prior_sd

    def test_cancelling_rows_do_not_read_as_agreement(self):
        # A mean of zero over two rows that disagree is not a pass, so max |z|
        # has to be reported beside it.
        g = _gate(3.0, -3.0)
        assert g.mean_shortfall == pytest.approx(0.0)
        assert g.max_abs_z == pytest.approx(3.0)

    def test_unreachable_past_two_prior_sd(self):
        assert "unreachable" in _gate(2.5 * BUDGET).verdict

    def test_reachable_but_flagged_when_a_row_is_far_out(self):
        v = _gate(0.1, sd=0.01).verdict
        assert "reachable" in v and "dominate" in v

    def test_reachable_when_close_and_well_scored(self):
        assert _gate(0.05, -0.05).verdict == "reachable"

    def test_no_width_rows_is_reported_and_not_a_pass(self):
        g = WidthGate((), BUDGET)
        assert g.verdict == "no width rows in this problem"
        assert g.mean_shortfall == 0.0 and g.max_abs_z == 0.0


def _q(tid, p, value, n=20, cohort="c"):
    return RowSpec(target_id=tid, cohort_id=cohort, stat="quantile", value=value,
                   n=n, p=p)


def _spec(tid, stat, value, n=20, cohort="c"):
    return RowSpec(target_id=tid, cohort_id=cohort, stat=stat, value=value, n=n,
                   log=True)


class TestPairedWidths:
    # q25 = 1, q75 = 3 predicted; the source printed 1 and 5.
    SPECS = [_q("t", 0.25, 1.0), _q("t", 0.75, 5.0)]
    TAU = np.array([1.0, 3.0])
    OBS = np.array([1.0, 5.0])
    V = np.array([[0.04, 0.0], [0.0, 0.04]])

    def test_a_quartile_pair_becomes_one_width_row(self):
        (row,) = paired_width_rows(self.SPECS, self.TAU, self.OBS, self.V)
        assert row.derived and row.logged
        assert row.label == "t/q0.25-q0.75"
        assert row.predicted == pytest.approx(np.log(2.0))
        assert row.observed == pytest.approx(np.log(4.0))
        # The source reports twice the spread the model can produce.
        assert row.shortfall == pytest.approx(np.log(2.0))

    def test_error_bar_is_the_contrast_variance_on_the_log_scale(self):
        (row,) = paired_width_rows(self.SPECS, self.TAU, self.OBS, self.V)
        # Var(q75 - q25) = 0.04 + 0.04 - 0 = 0.08, delta-method'd at w_pred = 2.
        assert row.sd == pytest.approx(np.sqrt(0.08) / 2.0)

    def test_correlation_between_the_two_rows_narrows_the_bar(self):
        V = np.array([[0.04, 0.03], [0.03, 0.04]])
        (row,) = paired_width_rows(self.SPECS, self.TAU, self.OBS, V)
        assert row.sd == pytest.approx(np.sqrt(0.02) / 2.0)

    def test_offset_positions_a_cohort_inside_its_block(self):
        tau = np.array([9.0, 9.0, 1.0, 3.0])
        obs = np.array([9.0, 9.0, 1.0, 5.0])
        V = np.eye(4) * 0.04
        (row,) = paired_width_rows(self.SPECS, tau, obs, V, offset=2)
        assert row.predicted == pytest.approx(np.log(2.0))

    def test_a_printed_iqr_is_not_also_derived_from_its_quartiles(self):
        # Same number twice would double its weight in the mean shortfall.
        specs = [*self.SPECS, _spec("t", "iqr", 4.0)]
        tau = np.array([1.0, 3.0, np.log(2.0)])
        obs = np.array([1.0, 5.0, np.log(4.0)])
        assert paired_width_rows(specs, tau, obs, np.eye(3) * 0.04) == []

    def test_only_the_widest_symmetric_pair_is_taken(self):
        # q25/q50/q75 is one width of these patients, not three overlapping ones.
        specs = [_q("t", 0.25, 1.0), _q("t", 0.5, 2.0), _q("t", 0.75, 5.0),
                 _q("t", 0.1, 0.5), _q("t", 0.9, 8.0)]
        tau = np.array([1.0, 2.0, 3.0, 0.5, 6.0])
        obs = np.array([1.0, 2.0, 5.0, 0.5, 8.0])
        (row,) = paired_width_rows(specs, tau, obs, np.eye(5) * 0.04)
        assert row.label == "t/q0.1-q0.9"

    def test_an_unpaired_quantile_contributes_nothing(self):
        specs = [_q("t", 0.5, 2.0), _q("t", 0.75, 5.0)]
        assert paired_width_rows(specs, np.array([2.0, 3.0]),
                                 np.array([2.0, 5.0]), np.eye(2) * 0.04) == []

    def test_each_target_gets_its_own_row(self):
        specs = [_q("a", 0.25, 1.0), _q("a", 0.75, 5.0),
                 _q("b", 0.25, 1.0), _q("b", 0.75, 3.0)]
        rows = paired_width_rows(specs, np.array([1.0, 3.0, 1.0, 3.0]),
                                 np.array([1.0, 5.0, 1.0, 3.0]),
                                 np.eye(4) * 0.04)
        assert [r.label for r in rows] == ["a/q0.25-q0.75", "b/q0.25-q0.75"]

    def test_a_non_positive_printed_width_is_refused(self):
        specs = [_q("t", 0.25, 5.0), _q("t", 0.75, 1.0)]
        with pytest.raises(ValueError, match="corpus error"):
            paired_width_rows(specs, self.TAU, np.array([5.0, 1.0]), self.V)

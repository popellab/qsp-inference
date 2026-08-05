"""Unit tests for the width gate (qsp_inference.vpop.gates)."""
import numpy as np
import pytest

from qsp_inference.vpop.gates import WidthGate, WidthRow

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

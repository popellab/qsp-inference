"""Prevalence weighting: does a reweighted cloud actually become the target population?"""

from __future__ import annotations

import numpy as np
import pytest

from qsp_inference.vpop import build_quantile_constraints, fit_prevalence_weights


def _weighted_quantile(x, w, q):
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w) - 0.5 * w
    return np.interp(q, cdf / w.sum(), x)


def test_reweighting_recovers_a_shifted_population():
    """The cloud is centred wrong and too wide; the data should pull it into place."""
    rng = np.random.default_rng(0)
    # Truth: patients centred at 2.0, sd 0.3. Observed = 200 of them.
    observed = {"y": rng.normal(2.0, 0.3, size=200)}
    # Cloud: a generous, badly-centred prior.
    cloud = rng.normal(1.0, 1.5, size=20_000)[:, None]

    res = fit_prevalence_weights(cloud, observed, ["y"], n_bins=6)

    assert res.converged
    med = _weighted_quantile(cloud[:, 0], res.weights, 0.5)
    assert med == pytest.approx(2.0, abs=0.15)
    # and the spread, not just the centre
    lo, hi = _weighted_quantile(cloud[:, 0], res.weights, [0.25, 0.75])
    assert (hi - lo) == pytest.approx(0.40, abs=0.15)  # IQR of N(2, 0.3) ~ 0.405
    assert res.per_observable["tv_distance"].max() < 0.05


def test_reweighting_recovers_a_BIMODAL_population():
    """The claim in unit_panel_spread_design.md 13.5.

    A parametric (mu, omega) lognormal cannot express responder/non-responder
    structure -- it invents a fat middle no patient occupies. A reweighted cloud is
    nonparametric in the population shape, so it should find both modes and leave the
    middle empty.
    """
    rng = np.random.default_rng(1)
    responders = rng.normal(1.0, 0.15, size=120)
    non_responders = rng.normal(4.0, 0.15, size=80)
    observed = {"y": np.concatenate([responders, non_responders])}

    cloud = rng.uniform(0.0, 5.0, size=40_000)[:, None]
    res = fit_prevalence_weights(cloud, observed, ["y"], n_bins=10)

    assert res.converged
    y, w = cloud[:, 0], res.weights
    mass = lambda lo, hi: w[(y >= lo) & (y < hi)].sum()  # noqa: E731

    near_mode_1 = mass(0.5, 1.5)
    near_mode_2 = mass(3.5, 4.5)
    the_middle = mass(2.0, 3.0)

    # Both modes carry mass, in roughly the 60/40 split of the data...
    assert near_mode_1 > 0.4
    assert near_mode_2 > 0.25
    # ...and the middle, which no real patient occupies, stays thin. It is not exactly
    # zero because one quantile bin necessarily straddles the empty gap (its edges are
    # interpolated across it) and legitimately holds the lower tail of mode 2; uniform
    # cloud members inside that bin pick up some weight. The claim being pinned here is
    # the one that matters: no FAT invented middle, which is what a single (mu, omega)
    # lognormal produces by construction.
    assert the_middle < 0.10
    assert the_middle < near_mode_1 / 5
    assert the_middle < near_mode_2 / 3
    assert res.per_observable["tv_distance"].max() < 0.05


def test_support_deficiency_is_flagged_not_silently_fitted():
    """Reweighting cannot conjure a patient the cloud does not contain.

    Every cloud member falls below every observed quantile edge, so the constraint rows
    are constant (all-ones or all-zeros) and no reweighting can move them. The right
    behaviour is to say so -- flag the support failure and report the bad fit -- rather
    than to strain the weights onto a corner of the cloud and call it converged. The
    fix for this state is a wider population prior (see the omega prior source), not a
    harder push on the optimiser.
    """
    rng = np.random.default_rng(2)
    observed = {"y": rng.normal(10.0, 0.2, size=100)}  # data live far from the cloud
    cloud = rng.normal(0.0, 1.0, size=5_000)[:, None]

    res = fit_prevalence_weights(cloud, observed, ["y"], n_bins=5)

    assert "y" in res.support_deficient
    assert "SUPPORT DEFICIENT" in res.summary()
    # The failure is surfaced as a bad fit, loudly...
    assert res.per_observable["tv_distance"].max() > 0.5
    # ...and NOT hidden by collapsing onto a handful of members. Weights stay ~uniform.
    assert res.ess_fraction > 0.9


def test_ess_falls_as_constraints_tighten():
    rng = np.random.default_rng(3)
    observed = {"y": rng.normal(1.5, 0.2, size=200)}
    cloud = rng.normal(0.0, 2.0, size=20_000)[:, None]

    loose = fit_prevalence_weights(cloud, observed, ["y"], n_bins=3, ridge=1e-1)
    tight = fit_prevalence_weights(cloud, observed, ["y"], n_bins=12, ridge=1e-6)

    assert tight.ess < loose.ess
    assert tight.per_observable["tv_distance"].max() <= loose.per_observable["tv_distance"].max()


def test_multiple_observables_are_matched_jointly():
    rng = np.random.default_rng(4)
    n = 30_000
    a = rng.normal(0.0, 1.0, size=n)
    b = rng.normal(0.0, 1.0, size=n)
    cloud = np.column_stack([a, b])
    observed = {
        "a": rng.normal(0.8, 0.3, size=150),
        "b": rng.normal(-0.5, 0.3, size=150),
    }

    res = fit_prevalence_weights(cloud, observed, ["a", "b"], n_bins=5)

    assert res.converged
    assert _weighted_quantile(a, res.weights, 0.5) == pytest.approx(0.8, abs=0.2)
    assert _weighted_quantile(b, res.weights, 0.5) == pytest.approx(-0.5, abs=0.2)
    assert set(res.per_observable["observable"]) == {"a", "b"}


def test_weights_are_a_probability_vector():
    rng = np.random.default_rng(5)
    cloud = rng.normal(size=(2_000, 1))
    res = fit_prevalence_weights(cloud, {"y": rng.normal(size=50)}, ["y"])
    assert res.weights.shape == (2_000,)
    assert np.all(res.weights >= 0)
    assert res.weights.sum() == pytest.approx(1.0)


# --- input validation --------------------------------------------------------


def test_constraint_targets_sum_to_one_per_observable():
    rng = np.random.default_rng(6)
    cloud = rng.normal(size=(500, 1))
    _, p, labels, _ = build_quantile_constraints(
        cloud, {"y": rng.normal(size=100)}, ["y"], n_bins=4
    )
    assert p.sum() == pytest.approx(1.0)
    assert len(labels) == len(p)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="obs_names"):
        fit_prevalence_weights(np.zeros((10, 2)), {"y": np.ones(5)}, ["y"])


def test_no_finite_observed_samples_rejected():
    with pytest.raises(ValueError, match="no finite observed samples"):
        fit_prevalence_weights(np.zeros((10, 1)), {"y": np.array([np.nan])}, ["y"])

"""Unit tests for the NLME predictive checks (prediction discrepancy + VPC).

The invariants that matter: under a correctly specified model the discrepancies
are calibrated (uniform pd, large p), and each check fires on the miss it is
meant to see -- center for prediction_discrepancy, center AND spread for the VPC.
"""
import numpy as np
import pytest

from qsp_inference.inference.predictive_checks import (
    iqr,
    population_vpc,
    prediction_discrepancy,
)


# --- prediction_discrepancy -------------------------------------------------

def test_pd_correct_model_is_calibrated():
    """Observations drawn FROM the predictive have ~uniform pd, so almost none
    are flagged at the 0.05 two-sided level (a false-positive rate near 5%)."""
    rng = np.random.default_rng(0)
    n_obs = 200
    predictive = rng.normal(size=(4000, n_obs))
    observed = rng.normal(size=n_obs)  # same law as the predictive columns
    df = prediction_discrepancy(predictive, observed)
    flagged = (df["p_value"] < 0.05).mean()
    assert flagged < 0.12  # ~5% expected, allow MC slack


def test_pd_flags_an_extreme_observation():
    rng = np.random.default_rng(1)
    predictive = rng.normal(size=(4000, 3))
    observed = np.array([0.0, 0.0, 8.0])  # obs 2 is far in the upper tail
    df = prediction_discrepancy(predictive, observed).set_index("observable")
    assert df.loc["obs_2", "p_value"] < 0.01
    assert df.loc["obs_2", "tail"] == "upper"
    assert df.loc["obs_0", "p_value"] > 0.1


def test_pd_weights_shift_the_null():
    """Reweighting the predictive changes the rank: an observation at the
    unweighted median sits in the tail once weight concentrates elsewhere."""
    x = np.linspace(-3, 3, 5000).reshape(-1, 1)
    observed = np.array([0.0])
    unweighted = prediction_discrepancy(x, observed).iloc[0]
    assert abs(unweighted["pd"] - 0.5) < 0.02
    # Weight mass onto the high end -> 0.0 is now low in the reweighted predictive.
    w = np.exp(2.0 * x[:, 0])
    weighted = prediction_discrepancy(x, observed, weights=w).iloc[0]
    assert weighted["pd"] < 0.2


def test_pd_dict_observed_and_names():
    predictive = np.zeros((10, 2))
    df = prediction_discrepancy(
        predictive, {"a": 0.0, "b": 1.0}, observable_names=["a", "b"]
    )
    assert set(df["observable"]) == {"a", "b"}


def test_pd_nan_column_is_nan_not_crash():
    predictive = np.full((100, 2), np.nan)
    predictive[:, 0] = np.random.default_rng(0).normal(size=100)
    df = prediction_discrepancy(predictive, np.array([0.0, 0.0])).set_index("observable")
    assert np.isnan(df.loc["obs_1", "p_value"])
    assert np.isfinite(df.loc["obs_0", "p_value"])


# --- population_vpc ---------------------------------------------------------

def _population_cloud(n_sim=20000, sigma=1.0, seed=0):
    """A one-observable population-predictive cloud: individuals ~ N(0, sigma)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=(n_sim, 1))


def test_vpc_correct_model_center_and_spread_ok():
    """Observed summary drawn from a cohort of the SAME population is consistent
    on both arms."""
    cloud = _population_cloud(sigma=1.0)
    rng = np.random.default_rng(1)
    cohort = rng.normal(0.0, 1.0, size=60)
    df = population_vpc(
        cloud, [np.median(cohort)], [iqr(cohort)], cohort_size=60, seed=2
    ).iloc[0]
    assert df["median_p"] > 0.05
    assert df["iqr_p"] > 0.05
    assert df["spread_verdict"] == "ok"


def test_vpc_detects_over_dispersion():
    """Model spread wider than the data: the observed IQR sits BELOW the
    predictive IQR distribution -> spread 'under', a random-effects (omega) miss
    that the fixed-effects pd cannot see."""
    cloud = _population_cloud(sigma=3.0)          # model: wide population
    rng = np.random.default_rng(3)
    cohort = rng.normal(0.0, 1.0, size=80)        # data: narrow
    df = population_vpc(
        cloud, [np.median(cohort)], [iqr(cohort)], cohort_size=80, seed=4
    ).iloc[0]
    assert df["iqr_p"] < 0.05
    assert df["spread_verdict"] == "under"
    # center is fine (both centered at 0), so this is a pure spread miss
    assert df["median_p"] > 0.05


def test_vpc_detects_center_miss():
    cloud = _population_cloud(sigma=1.0)
    rng = np.random.default_rng(5)
    cohort = rng.normal(0.0, 1.0, size=80)
    df = population_vpc(
        cloud, [4.0], [iqr(cohort)], cohort_size=80, seed=6
    ).iloc[0]
    assert df["median_p"] < 0.01


def test_vpc_small_n_widens_bands():
    """A smaller cohort gives a wider predictive summary band, so the same
    observed IQR is less extreme (larger or equal p) at small n."""
    cloud = _population_cloud(sigma=2.0)
    rng = np.random.default_rng(7)
    cohort = rng.normal(0.0, 1.0, size=200)
    obs_med, obs_iqr = np.median(cohort), iqr(cohort)
    p_small = population_vpc(cloud, [obs_med], [obs_iqr], cohort_size=8, seed=8).iloc[0]["iqr_p"]
    p_large = population_vpc(cloud, [obs_med], [obs_iqr], cohort_size=200, seed=8).iloc[0]["iqr_p"]
    assert p_small >= p_large


def test_vpc_per_observable_cohort_sizes():
    cloud = np.concatenate([_population_cloud(seed=10), _population_cloud(seed=11)], axis=1)
    df = population_vpc(
        cloud, [0.0, 0.0], [1.35, 1.35], cohort_size=[10, 100],
        observable_names=["low_n", "high_n"], seed=9,
    )
    assert set(df["observable"]) == {"low_n", "high_n"}


def test_iqr_matches_numpy():
    a = np.random.default_rng(0).normal(size=1000)
    assert np.isclose(iqr(a), np.percentile(a, 75) - np.percentile(a, 25))

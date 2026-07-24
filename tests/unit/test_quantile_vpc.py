"""Unit tests for quantile_vpc -- the general quantile-anchor population VPC.

The mixed-effects VPC on the maple ObservedDistribution contract: median/IQR is
the {0.25, 0.5, 0.75} special case. A center anchor (p~0.5) probes the fixed
effect; tail anchors probe the random-effect spread (omega). The per-observable
spread verdict reads over/under-dispersion off the widest anchor pair.
"""
import numpy as np
import pytest

from qsp_inference.inference.predictive_checks import population_vpc, quantile_vpc


def _quartile_anchors(cloud, cohort_size):
    """The {0.25, 0.5, 0.75} anchor spec computed from a reference cloud -- the
    'observed' distribution a correctly-specified model should reproduce."""
    q25, q50, q75 = np.percentile(cloud, [25, 50, 75], axis=0)
    return [[(0.25, q25[i]), (0.5, q50[i]), (0.75, q75[i])] for i in range(cloud.shape[1])]


def test_well_specified_is_ok():
    """Observed anchors drawn from the same law as the cloud -> no conflict."""
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(20000, 3))
    obs = rng.normal(size=(20000, 3))          # same population, independent draw
    anchors = _quartile_anchors(obs, 50)
    df = quantile_vpc(cloud, anchors, cohort_size=50, seed=1)
    assert set(df["observable"]) == {"obs_0", "obs_1", "obs_2"}
    assert (df["p_value"] > 0.01).all()
    assert (df["spread_verdict"] == "ok").all()
    # p=0.5 is a center anchor; 0.25/0.75 are spread anchors.
    kinds = df.set_index(["observable", "p"])["kind"]
    assert kinds[("obs_0", 0.5)] == "center"
    assert kinds[("obs_0", 0.25)] == "spread"


def test_over_dispersed_model_reads_under():
    """Model spread wider than the data: observed IQR sits below the predictive
    -> spread_verdict 'under' (over-dispersed)."""
    rng = np.random.default_rng(1)
    cloud = rng.normal(scale=3.0, size=(20000, 1))   # model too wide
    obs_pop = rng.normal(scale=1.0, size=(20000, 1))  # data narrower
    anchors = _quartile_anchors(obs_pop, 100)
    df = quantile_vpc(cloud, anchors, cohort_size=100, seed=2)
    assert (df["spread_verdict"] == "under").all()
    # the tail anchors are the ones that fire, not the (matched) center.
    tail = df[df["kind"] == "spread"]
    assert (tail["p_value"] < 0.05).all()


def test_under_dispersed_model_reads_over():
    rng = np.random.default_rng(2)
    cloud = rng.normal(scale=0.5, size=(20000, 1))    # model too narrow
    obs_pop = rng.normal(scale=2.0, size=(20000, 1))   # data wider
    anchors = _quartile_anchors(obs_pop, 100)
    df = quantile_vpc(cloud, anchors, cohort_size=100, seed=3)
    assert (df["spread_verdict"] == "over").all()


def test_center_shift_fires_center_anchor():
    """A shifted center with matched spread: the p=0.5 anchor conflicts, the
    spread verdict stays ok."""
    rng = np.random.default_rng(3)
    cloud = rng.normal(loc=0.0, scale=1.0, size=(20000, 1))
    obs_pop = rng.normal(loc=5.0, scale=1.0, size=(20000, 1))  # shifted center
    anchors = _quartile_anchors(obs_pop, 100)
    df = quantile_vpc(cloud, anchors, cohort_size=100, seed=4)
    center = df[df["kind"] == "center"]
    assert (center["p_value"] < 0.05).all()
    assert (df["spread_verdict"] == "ok").all()  # width matches; only center moved


def test_reduces_to_population_vpc():
    """With the {0.25, 0.5, 0.75} anchors, quantile_vpc's median/IQR verdict must
    match population_vpc on the same cloud and observed summary."""
    rng = np.random.default_rng(4)
    cloud = rng.normal(size=(15000, 2))
    obs_med = np.array([0.0, 0.2])
    obs_iqr = np.array([1.35, 1.35])  # ~ IQR of a standard normal
    q25 = obs_med - obs_iqr / 2.0
    q75 = obs_med + obs_iqr / 2.0
    anchors = [[(0.25, q25[i]), (0.5, obs_med[i]), (0.75, q75[i])] for i in range(2)]

    pv = population_vpc(cloud, obs_med, obs_iqr, cohort_size=40, seed=7).set_index("observable")
    qv = quantile_vpc(cloud, anchors, cohort_size=40, seed=7)
    # The width verdict is the IQR verdict; check they agree per observable.
    for name in ["obs_0", "obs_1"]:
        qv_verdict = qv[qv["observable"] == name]["spread_verdict"].iloc[0]
        assert qv_verdict == pv.loc[name, "spread_verdict"]


def test_ragged_anchors_and_center_only():
    """Observables may carry different anchor sets; a single-anchor (center-only)
    observable gets spread_verdict 'center_only' and no width test."""
    rng = np.random.default_rng(5)
    cloud = rng.normal(size=(8000, 2))
    anchors = [
        [(0.5, 0.0)],                                   # center-only
        [(0.1, -1.28), (0.5, 0.0), (0.9, 1.28)],        # deciles + median
    ]
    df = quantile_vpc(cloud, anchors, cohort_size=30, seed=6)
    v0 = df[df["observable"] == "obs_0"]
    assert len(v0) == 1 and v0["spread_verdict"].iloc[0] == "center_only"
    v1 = df[df["observable"] == "obs_1"]
    assert len(v1) == 3
    assert set(v1["kind"]) == {"center", "spread"}  # 0.5 center, 0.1/0.9 spread


def test_per_target_cohort_size_sequence():
    rng = np.random.default_rng(6)
    cloud = rng.normal(size=(10000, 2))
    anchors = _quartile_anchors(rng.normal(size=(10000, 2)), 10)
    df = quantile_vpc(cloud, anchors, cohort_size=[6, 900], seed=8)
    assert len(df) == 6  # 3 anchors x 2 observables


def test_nan_column_is_labeled_not_crash():
    cloud = np.full((100, 1), np.nan)
    anchors = [[(0.25, 0.0), (0.5, 0.0), (0.75, 0.0)]]
    df = quantile_vpc(cloud, anchors, cohort_size=50)
    assert (df["spread_verdict"] == "nan").all()
    assert df["pd"].isna().all()


def test_shape_validation():
    with pytest.raises(ValueError, match="cloud must be 2-D"):
        quantile_vpc(np.zeros(10), [[(0.5, 0.0)]], cohort_size=5)
    with pytest.raises(ValueError, match="anchors has"):
        quantile_vpc(np.zeros((10, 2)), [[(0.5, 0.0)]], cohort_size=5)
    with pytest.raises(ValueError, match="p must be in"):
        quantile_vpc(np.zeros((10, 1)), [[(1.5, 0.0)]], cohort_size=5)
    with pytest.raises(ValueError, match="cohort_size must be >= 2"):
        quantile_vpc(np.zeros((10, 1)), [[(0.5, 0.0)]], cohort_size=1)

"""Unit tests for LOO-PIT calibration and the marginal misfit labeler.

pit_calibration / loo_pit: the PIT-uniformity calibration check and its
dispersion reading. label_marginal_conflict: the redesign's step 2 -> step 3
decision (consistent / prior_limited / structural) from two predictive samples.
"""
import numpy as np

from qsp_inference.inference.predictive_checks import (
    label_marginal_conflict,
    loo_pit,
    pit_calibration,
)


# --- pit_calibration --------------------------------------------------------

def test_uniform_pits_are_calibrated():
    pit = np.linspace(0.001, 0.999, 500)  # ~Uniform(0,1)
    cal = pit_calibration(pit)
    assert cal["ks_p"] > 0.05
    assert cal["dispersion"] == "ok"
    assert abs(cal["mean"] - 0.5) < 0.02
    assert abs(cal["var"] - 1.0 / 12.0) < 0.005


def test_pits_bunched_at_center_read_over_dispersed():
    """PITs concentrated near 1/2 (variance < 1/12) => predictive too wide."""
    rng = np.random.default_rng(0)
    pit = np.clip(0.5 + 0.05 * rng.standard_normal(1000), 0, 1)
    cal = pit_calibration(pit)
    assert cal["var"] < 1.0 / 12.0
    assert cal["dispersion"] == "over"
    assert cal["ks_p"] < 0.05


def test_pits_piled_at_ends_read_under_dispersed():
    """PITs piling at 0 and 1 (variance > 1/12) => predictive too narrow."""
    pit = np.concatenate([np.zeros(400), np.ones(400)])
    cal = pit_calibration(pit)
    assert cal["var"] > 1.0 / 12.0
    assert cal["dispersion"] == "under"


def test_empty_pit_is_nan_not_crash():
    cal = pit_calibration([])
    assert cal["n"] == 0 and np.isnan(cal["ks_stat"])


def test_loo_pit_correct_model_is_calibrated():
    """Observations from the predictive give uniform PITs -> calibrated."""
    rng = np.random.default_rng(1)
    n_obs = 300
    predictive = rng.normal(size=(4000, n_obs))
    observed = rng.normal(size=n_obs)
    pit_df, cal = loo_pit(predictive, observed)
    assert cal["ks_p"] > 0.05
    assert cal["dispersion"] == "ok"
    # the per-observable PIT is the pd column
    assert "pd" in pit_df.columns and len(pit_df) == n_obs


def test_loo_pit_detects_under_dispersed_predictive():
    """A predictive narrower than the truth: observations fall in its tails, PITs
    pile at the ends -> under-dispersed."""
    rng = np.random.default_rng(2)
    n_obs = 300
    predictive = rng.normal(scale=0.3, size=(4000, n_obs))  # too narrow
    observed = rng.normal(scale=1.0, size=n_obs)            # true spread wider
    _, cal = loo_pit(predictive, observed)
    assert cal["dispersion"] == "under"
    assert cal["ks_p"] < 0.05


# --- label_marginal_conflict ------------------------------------------------

def _prior_predictive(n=5000, n_obs=3, seed=0):
    return np.random.default_rng(seed).normal(size=(n, n_obs))


def test_consistent_when_no_prior_conflict():
    prior = _prior_predictive()
    observed = np.array([0.0, 0.1, -0.1])  # all near the prior-predictive center
    df = label_marginal_conflict(prior, observed).set_index("observable")
    assert (df["label"] == "consistent").all()


def test_conflict_unresolved_without_resim():
    """A prior conflict with no resimulation is not assumed structural (the OOD
    caveat): it is left unresolved until the sampler has tried."""
    prior = _prior_predictive()
    observed = np.array([0.0, 0.0, 8.0])  # obs_2 far in the prior tail
    df = label_marginal_conflict(prior, observed).set_index("observable")
    assert df.loc["obs_2", "label"] == "conflict_unresolved"
    assert df.loc["obs_0", "label"] == "consistent"


def test_prior_limited_when_resim_reaches():
    """Conflict under the prior, but the posterior-at-x_obs resim reaches the
    observation -> prior_limited (re-anchor)."""
    prior = _prior_predictive()
    observed = np.array([0.0, 0.0, 8.0])
    # resim for obs_2 is centered at 8 (a plausible theta reaches it)
    resim = np.random.default_rng(9).normal(size=(4000, 3))
    resim[:, 2] += 8.0
    df = label_marginal_conflict(prior, observed, posterior_resim=resim).set_index("observable")
    assert df.loc["obs_2", "label"] == "prior_limited"


def test_structural_when_resim_still_misses():
    """Conflict under the prior AND the resim still cannot reach it -> structural
    (mechanism problem)."""
    prior = _prior_predictive()
    observed = np.array([0.0, 0.0, 8.0])
    # resim still centered near 0: the model's best attempt cannot produce 8
    resim = np.random.default_rng(10).normal(size=(4000, 3))
    df = label_marginal_conflict(prior, observed, posterior_resim=resim).set_index("observable")
    assert df.loc["obs_2", "label"] == "structural"


def test_structural_sorts_first():
    prior = _prior_predictive()
    observed = np.array([0.0, 0.0, 8.0])
    resim = np.random.default_rng(11).normal(size=(4000, 3))
    df = label_marginal_conflict(prior, observed, posterior_resim=resim)
    assert df.iloc[0]["label"] == "structural"

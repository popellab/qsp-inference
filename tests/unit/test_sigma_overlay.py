"""Tests for the composite-prior sigma-overlay (cloud-generation prior).

Covers ``compose_overlay_prior`` (the pure recomposition) and
``load_overlay_prior_log`` (the disk wrapper that pulls population sigma from the
``population`` block). The overlay keeps center mu, overrides sigma with the
population sigma where declared, pins opted-out params (sigma -> ~0, decoupled),
and preserves the surviving copula correlations.
"""

import math

import numpy as np
import pytest

pytest.importorskip("torch")  # copula_prior pulls in torch (sbi extra)
from scipy import stats

from qsp_inference.priors.copula_prior import (
    GaussianCopulaPrior,
    compose_overlay_prior,
    load_overlay_prior_log,
)


def _center(names, mus, sigmas, R=None):
    """A log-space center prior (normal marginals) with a known copula."""
    marg = [stats.norm(loc=mu, scale=s) for mu, s in zip(mus, sigmas)]
    return GaussianCopulaPrior(marginals=marg, correlation=R, param_names=names)


def _corr_lognormal_samples(sigma, rho=0.7, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1]
    return {"k_a": list(np.exp(sigma * z[:, 0])), "k_b": list(np.exp(sigma * z[:, 1]))}


def test_overlay_pin_collapses_preserves_mu_and_surviving_copula():
    names = ["a", "b", "c"]
    mus = [0.0, 1.0, -2.0]
    sigmas = [0.5, 0.8, 1.2]
    # a-b, b-c, a-c all correlated; pinning b must drop a-b/b-c but keep a-c.
    R = np.array([[1.0, 0.6, 0.5], [0.6, 1.0, 0.4], [0.5, 0.4, 1.0]])
    center = _center(names, mus, sigmas, R)

    prior, out_names = compose_overlay_prior(center, pin_params=["b"], pin_sigma=1e-3)
    assert out_names == names

    s = prior.sample((8000,)).cpu().numpy()
    # centers never move; pinned param has negligible spread.
    assert s[:, 1].mean() == pytest.approx(1.0, abs=0.02)
    assert s[:, 1].std() < 5e-3
    # unpinned params keep center mu + sigma.
    assert s[:, 0].mean() == pytest.approx(0.0, abs=0.05)
    assert s[:, 0].std() == pytest.approx(0.5, rel=0.1)
    assert s[:, 2].std() == pytest.approx(1.2, rel=0.1)

    # pinned param decoupled; correlation among survivors preserved.
    assert prior._R[1, 0] == 0.0 and prior._R[0, 1] == 0.0
    assert prior._R[1, 1] == 1.0
    assert prior._R[0, 2] == pytest.approx(0.5)


def test_overlay_population_sigma_override_and_pin_precedence():
    names = ["a", "b"]
    center = _center(names, [0.0, 0.0], [0.5, 0.5])

    prior, _ = compose_overlay_prior(
        center,
        population_sigma={"a": 1.5, "b": 1.5},
        pin_params=["b"],
        pin_sigma=1e-3,
    )
    s = prior.sample((8000,)).cpu().numpy()
    assert s[:, 0].std() == pytest.approx(1.5, rel=0.1)  # population widened a
    assert s[:, 1].std() < 5e-3  # pin beat the population override for b


def test_overlay_rejects_non_log_prior():
    # lognormal marginals are NOT log-space; the overlay must refuse them rather
    # than misread lognormal moments as a normal loc/scale.
    center = GaussianCopulaPrior(marginals=[stats.lognorm(s=0.5, scale=1.0)], param_names=["a"])
    with pytest.raises(ValueError, match="log-space"):
        compose_overlay_prior(center, pin_params=["a"])


def test_load_overlay_prior_log_end_to_end(tmp_path):
    from qsp_inference.audit.report import _write_submodel_priors

    targets = {"k_a": ["t1"], "k_b": ["t1"]}
    center = {"compA": _corr_lognormal_samples(0.1, seed=1)}
    pop = {"compA": _corr_lognormal_samples(0.5, seed=1)}  # ~5x wider population spread
    yaml_out = tmp_path / "sp.yaml"
    _write_submodel_priors(center, targets, {}, yaml_out, population_samples_by_component=pop)

    csv_out = tmp_path / "priors.csv"
    csv_out.write_text(
        "name,distribution,dist_param1,dist_param2\n"
        "k_a,lognormal,0.0,0.3\n"
        "k_b,lognormal,0.0,0.3\n"
        "k_pin,lognormal,0.0,0.9\n"
    )

    prior, names = load_overlay_prior_log(yaml_out, csv_out, pin_params=["k_pin"])
    assert names == ["k_a", "k_b", "k_pin"]
    idx = {n: i for i, n in enumerate(names)}

    s = prior.sample((6000,)).cpu().numpy()
    # population block (~0.5) overrode the YAML center sigma (~0.1) for k_a/k_b.
    assert s[:, idx["k_a"]].std() > 0.3
    # CSV-only param pinned to its center.
    assert s[:, idx["k_pin"]].std() < 5e-3

    # With the population block disabled, k_a falls back to its narrow center sigma.
    prior2, _ = load_overlay_prior_log(yaml_out, csv_out, use_population_block=False)
    s2 = prior2.sample((6000,)).cpu().numpy()
    assert s2[:, idx["k_a"]].std() < 0.25


def test_load_overlay_prior_log_vary_params_allowlist(tmp_path):
    """``vary_params`` is the allowlist form: everything NOT listed is pinned.

    Same fixture as the end-to-end test, but instead of naming the pins we name
    the single param allowed to vary (k_a) and assert the other two collapse —
    including a param (k_b) that carries a population sigma, proving pins win
    over the population override when a param is outside the allowlist.
    """
    from qsp_inference.audit.report import _write_submodel_priors

    targets = {"k_a": ["t1"], "k_b": ["t1"]}
    center = {"compA": _corr_lognormal_samples(0.1, seed=1)}
    pop = {"compA": _corr_lognormal_samples(0.5, seed=1)}
    yaml_out = tmp_path / "sp.yaml"
    _write_submodel_priors(center, targets, {}, yaml_out, population_samples_by_component=pop)

    csv_out = tmp_path / "priors.csv"
    csv_out.write_text(
        "name,distribution,dist_param1,dist_param2\n"
        "k_a,lognormal,0.0,0.3\n"
        "k_b,lognormal,0.0,0.3\n"
        "k_pin,lognormal,0.0,0.9\n"
    )

    prior, names = load_overlay_prior_log(yaml_out, csv_out, vary_params=["k_a"])
    idx = {n: i for i, n in enumerate(names)}
    s = prior.sample((6000,)).cpu().numpy()
    # k_a varies (population sigma applied); k_b and k_pin pinned despite k_b's
    # population override — outside the allowlist ⇒ pinned.
    assert s[:, idx["k_a"]].std() > 0.3
    assert s[:, idx["k_b"]].std() < 5e-3
    assert s[:, idx["k_pin"]].std() < 5e-3

    # vary_params unions with explicit pin_params (a redundant pin here is a no-op).
    prior3, _ = load_overlay_prior_log(
        yaml_out, csv_out, vary_params=["k_a", "k_b"], pin_params=["k_b"]
    )
    s3 = prior3.sample((6000,)).cpu().numpy()
    assert s3[:, idx["k_a"]].std() > 0.3
    assert s3[:, idx["k_b"]].std() < 5e-3  # explicit pin beats allowlist membership

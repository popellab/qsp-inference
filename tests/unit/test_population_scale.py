"""Tests for the population-scale (Option A) submodel export.

Covers:
- ``_population_obs_from_distribution``: observed_distribution -> population observation
  likelihood, with heterogeneity_transfer omega-only widening.
- ``_write_submodel_priors``: parallel ``population`` block (marginals + own copula).
"""

import math
import types

import numpy as np
import pytest
import yaml

from maple.core.calibration.enums import HeterogeneityTransfer
from maple.core.calibration.shared_models import (
    ExperimentalUnitType,
    MomentSpread,
    ObservedDistribution,
    SpreadSource,
)
from qsp_inference.submodel.inference import (
    HETEROGENEITY_SIGMA,
    _population_obs_from_distribution,
)
from qsp_inference.audit.report import _write_submodel_priors


def _entry(od):
    return types.SimpleNamespace(observed_distribution=od)


def _target(het):
    return types.SimpleNamespace(
        primary_data_source=types.SimpleNamespace(
            source_relevance=types.SimpleNamespace(heterogeneity_transfer=het)
        )
    )


def _il12_like_od():
    # k_IL12_sec: mean 20 ng/mL, CV 0.5 lognormal, across 15 donors.
    return ObservedDistribution(
        moments=MomentSpread(center=20.0, scale=0.5, scale_type="cv", shape="lognormal"),
        spread_source=SpreadSource.BIOLOGICAL_EXPERIMENTAL,
        n_biological=15,
        experimental_unit_type=ExperimentalUnitType.BIOLOGICAL,
    )


def test_population_obs_none_without_population_spread():
    od = ObservedDistribution(
        moments=MomentSpread(center=1.0, scale=0.2, scale_type="sd", shape="normal"),
        spread_source=SpreadSource.CENTER_ONLY,
    )
    assert _population_obs_from_distribution(_entry(od), _target(None)) is None


def test_population_obs_none_when_no_distribution():
    assert _population_obs_from_distribution(_entry(None), _target(None)) is None


def test_population_obs_sigma_no_widening():
    # heterogeneity_transfer absent -> sigma is the raw observable log-spread (~sigma_ln(cv)).
    value, sigma, family = _population_obs_from_distribution(_entry(_il12_like_od()), _target(None))
    assert family == "lognormal"
    assert value == pytest.approx(20.0 / math.sqrt(1.25), rel=1e-3)  # lognormal median
    assert sigma == pytest.approx(math.sqrt(math.log(1.25)), rel=1e-3)  # ~0.472


def test_population_obs_heterogeneity_widening():
    # moderate transfer widens the spread in quadrature (omega-only).
    _, sigma_none, _ = _population_obs_from_distribution(_entry(_il12_like_od()), _target(None))
    _, sigma_mod, _ = _population_obs_from_distribution(
        _entry(_il12_like_od()), _target(HeterogeneityTransfer.MODERATE)
    )
    w = HETEROGENEITY_SIGMA["moderate"]
    assert sigma_mod == pytest.approx(math.sqrt(sigma_none**2 + w**2), rel=1e-3)
    # high transfer (target population) -> no widening
    _, sigma_high, _ = _population_obs_from_distribution(
        _entry(_il12_like_od()), _target(HeterogeneityTransfer.HIGH)
    )
    assert sigma_high == pytest.approx(sigma_none, rel=1e-3)
    assert sigma_mod > sigma_high


# ---------------------------------------------------------------------------
# Write-layer: parallel population block with its own copula
# ---------------------------------------------------------------------------


def _corr_lognormal_samples(sigma, rho=0.7, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 2))
    z[:, 1] = rho * z[:, 0] + math.sqrt(1 - rho**2) * z[:, 1]
    return {"k_a": list(np.exp(sigma * z[:, 0])), "k_b": list(np.exp(sigma * z[:, 1]))}


def test_write_submodel_priors_center_only(tmp_path):
    center = {"compA": _corr_lognormal_samples(0.1)}
    out = tmp_path / "sp.yaml"
    _write_submodel_priors(center, {"k_a": ["t1"], "k_b": ["t1"]}, {}, out)
    y = yaml.safe_load(out.read_text())
    assert set(y) == {"metadata", "parameters", "copula"}
    assert "population" not in y


def test_write_submodel_priors_population_block(tmp_path):
    targets = {"k_a": ["t1"], "k_b": ["t1"]}
    center = {"compA": _corr_lognormal_samples(0.1, seed=1)}
    pop = {"compA": _corr_lognormal_samples(0.5, seed=1)}  # ~5x wider, same correlation
    out = tmp_path / "sp.yaml"
    _write_submodel_priors(center, targets, {}, out, population_samples_by_component=pop)
    y = yaml.safe_load(out.read_text())

    assert "population" in y
    assert y["metadata"]["has_population_block"] is True
    assert "copula" in y["population"]  # full copula, not marginals-only

    cm = {p["name"]: p["marginal"] for p in y["parameters"]}
    pm = {p["name"]: p["marginal"] for p in y["population"]["parameters"]}
    # population marginal is wider than the center marginal
    assert pm["k_a"]["cv"] > cm["k_a"]["cv"] * 3

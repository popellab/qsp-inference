"""Unit tests for the shared per-target resolver (qsp_inference.targets.resolver).

The single seam that turns a raw observed_distribution mapping into a validated
maple ObservedDistribution, shared by the anchor (observed-data) and omega
(prior) halves of the contract.
"""
import pytest

from qsp_inference.targets import (
    parse_observed_distribution,
    population_n_biological,
)

pytest.importorskip("maple")


def _moments_od(center=10.0, scale=4.0, spread="across_patient", n=8):
    od = {
        "moments": {
            "center": center, "center_type": "median",
            "scale": scale, "scale_type": "iqr", "shape": "lognormal",
        },
        "spread_source": spread,
    }
    if spread in ("across_patient", "biological_experimental"):
        od["n_biological"] = n
        od["experimental_unit_type"] = "biological"
    return od


def test_parse_none_and_bad():
    assert parse_observed_distribution(None) is None
    assert parse_observed_distribution({"garbage": 1}) is None


def test_parse_moments_roundtrips_to_model():
    m = parse_observed_distribution(_moments_od())
    assert m is not None
    assert m.feeds_population_spread is True
    assert m.n_biological == 8
    ps = [p for p, _ in m._anchor_pairs()]
    assert ps == [0.25, 0.5, 0.75]


def test_parse_idempotent_on_model():
    m = parse_observed_distribution(_moments_od())
    assert parse_observed_distribution(m) is m


def test_population_n_takes_max_over_feeding():
    ods = [
        _moments_od(n=6),
        _moments_od(n=20),
        _moments_od(spread="center_only"),  # not population-feeding -> ignored
    ]
    assert population_n_biological(ods) == 20


def test_population_n_none_when_nothing_feeds():
    assert population_n_biological([_moments_od(spread="center_only")]) is None
    assert population_n_biological([]) is None
    assert population_n_biological([None, {"bad": 1}]) is None

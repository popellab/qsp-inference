"""Tests for ``qsp_inference.auxiliary.config``."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from qsp_inference.auxiliary.config import (
    AuxiliaryBasePrior,
    AuxiliaryConfig,
    AuxiliaryGroupSpec,
    load_auxiliary_config,
)


def _serum_to_tumor_spec() -> AuxiliaryGroupSpec:
    return AuxiliaryGroupSpec(
        description="Serum:tumor concentration ratio",
        base_prior=AuxiliaryBasePrior(distribution="lognormal", mu=0.0, sigma=0.7),
        member_deviation_sigma=0.3,
    )


def test_minimal_config_round_trips() -> None:
    cfg = AuxiliaryConfig(groups={"serum_to_tumor": _serum_to_tumor_spec()})
    spec = cfg.get("serum_to_tumor")
    assert spec is not None
    assert spec.member_deviation_sigma == 0.3
    assert spec.base_prior.distribution == "lognormal"


def test_empty_config_default() -> None:
    cfg = AuxiliaryConfig()
    assert cfg.groups == {}
    assert cfg.get("anything") is None


def test_member_deviation_sigma_defaults_to_zero() -> None:
    spec = AuxiliaryGroupSpec(
        base_prior=AuxiliaryBasePrior(distribution="lognormal", mu=0.0, sigma=0.5),
    )
    assert spec.member_deviation_sigma == 0.0


def test_base_prior_rejects_unknown_distribution() -> None:
    with pytest.raises(ValidationError):
        AuxiliaryBasePrior(distribution="cauchy", mu=0.0, sigma=0.5)


def test_base_prior_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValidationError):
        AuxiliaryBasePrior(distribution="normal", mu=0.0, sigma=0.0)
    with pytest.raises(ValidationError):
        AuxiliaryBasePrior(distribution="normal", mu=0.0, sigma=-0.1)


def test_member_deviation_sigma_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        AuxiliaryGroupSpec(
            base_prior=AuxiliaryBasePrior(distribution="lognormal", mu=0.0, sigma=0.5),
            member_deviation_sigma=-0.1,
        )


def test_extra_fields_forbidden_on_group_spec() -> None:
    with pytest.raises(ValidationError):
        AuxiliaryGroupSpec(
            base_prior=AuxiliaryBasePrior(distribution="lognormal", mu=0.0, sigma=0.5),
            tau=0.3,  # type: ignore[call-arg]
        )


def test_extra_fields_forbidden_on_top_level() -> None:
    with pytest.raises(ValidationError):
        AuxiliaryConfig(
            groups={"serum_to_tumor": _serum_to_tumor_spec()},
            extra_field="boom",  # type: ignore[call-arg]
        )


def test_load_returns_empty_when_file_missing(tmp_path: Path) -> None:
    cfg = load_auxiliary_config(tmp_path / "does_not_exist.yaml")
    assert cfg.groups == {}


def test_load_returns_empty_when_file_blank(tmp_path: Path) -> None:
    blank = tmp_path / "blank.yaml"
    blank.write_text("")
    cfg = load_auxiliary_config(blank)
    assert cfg.groups == {}


def test_load_parses_full_yaml(tmp_path: Path) -> None:
    yaml_text = yaml.safe_dump(
        {
            "groups": {
                "serum_to_tumor": {
                    "description": "Serum:tumor ratio",
                    "base_prior": {
                        "distribution": "lognormal",
                        "mu": 0.0,
                        "sigma": 0.7,
                    },
                    "member_deviation_sigma": 0.3,
                },
                "ihc_to_concentration": {
                    "base_prior": {
                        "distribution": "normal",
                        "mu": 1.0,
                        "sigma": 0.4,
                    },
                },
            }
        }
    )
    path = tmp_path / "auxiliary_config.yaml"
    path.write_text(yaml_text)

    cfg = load_auxiliary_config(path)
    assert set(cfg.groups) == {"serum_to_tumor", "ihc_to_concentration"}
    serum = cfg.get("serum_to_tumor")
    assert serum is not None
    assert serum.member_deviation_sigma == 0.3
    ihc = cfg.get("ihc_to_concentration")
    assert ihc is not None
    assert ihc.base_prior.distribution == "normal"
    assert ihc.member_deviation_sigma == 0.0


def test_load_rejects_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "groups": {
                    "g": {
                        "base_prior": {
                            "distribution": "lognormal",
                            "mu": 0.0,
                            "sigma": 0.5,
                        },
                        "members": ["k_a", "k_b"],  # not allowed
                    }
                }
            }
        )
    )
    with pytest.raises(ValidationError):
        load_auxiliary_config(path)

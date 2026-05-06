"""Tests for ``qsp_inference.auxiliary.discovery``."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from qsp_inference.auxiliary.config import (
    AuxiliaryBasePrior,
    AuxiliaryConfig,
    AuxiliaryGroupSpec,
)
from qsp_inference.auxiliary.discovery import (
    AuxiliaryRegistry,
    discover_auxiliary_members,
)


def _config_with(*group_names: str) -> AuxiliaryConfig:
    return AuxiliaryConfig(
        groups={
            name: AuxiliaryGroupSpec(
                base_prior=AuxiliaryBasePrior(
                    distribution="lognormal", mu=0.0, sigma=0.5
                ),
                member_deviation_sigma=0.2,
            )
            for name in group_names
        }
    )


def _write_cal_target(
    path: Path,
    *,
    aux: list[dict] | None,
    extra_observable: dict | None = None,
) -> None:
    """Write a minimal cal-target YAML containing only the fields the
    discovery walker reads. (Discovery does not validate the rest of the
    cal-target schema — that's maple's job at authoring time.)"""
    observable: dict = {"code": "def compute_observable(t, s, c, u): pass", "units": "nM"}
    if aux is not None:
        observable["auxiliary_parameters"] = aux
    if extra_observable:
        observable.update(extra_observable)
    path.write_text(yaml.safe_dump({"observable": observable}))


def test_empty_walk_returns_empty_registry(tmp_path: Path) -> None:
    reg = discover_auxiliary_members([tmp_path], _config_with())
    assert reg.is_empty
    assert reg.member_names == ()


def test_missing_root_is_silently_skipped(tmp_path: Path) -> None:
    reg = discover_auxiliary_members(
        [tmp_path / "nope", tmp_path], _config_with()
    )
    assert reg.is_empty


def test_single_member_collected(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "tgfb.yaml",
        aux=[
            {
                "name": "f_serum_to_tumor_TGFb",
                "group": "serum_to_tumor",
                "biological_basis": (
                    "Serum:tumor concentration ratio for TGFb; observable.code "
                    "multiplies V_T.TGFb by this factor."
                ),
                "units": "dimensionless",
            }
        ],
    )
    reg = discover_auxiliary_members([tmp_path], _config_with("serum_to_tumor"))
    assert reg.member_names == ("f_serum_to_tumor_TGFb",)
    member = reg.members["f_serum_to_tumor_TGFb"]
    assert member.group == "serum_to_tumor"
    assert member.units == "dimensionless"
    assert member.references == (tmp_path / "tgfb.yaml",)
    assert len(member.biological_basis_per_reference) == 1


def test_multiple_groups_sorted_deterministically(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "b.yaml",
        aux=[{"name": "f_b", "group": "g_b", "biological_basis": "x" * 25}],
    )
    _write_cal_target(
        tmp_path / "a.yaml",
        aux=[{"name": "f_a", "group": "g_a", "biological_basis": "x" * 25}],
    )
    reg = discover_auxiliary_members([tmp_path], _config_with("g_a", "g_b"))
    assert reg.group_names == ("g_a", "g_b")
    assert reg.member_names == ("f_a", "f_b")


def test_same_name_across_targets_merges_references(tmp_path: Path) -> None:
    aux = [
        {
            "name": "f_serum_to_tumor_TGFb",
            "group": "serum_to_tumor",
            "biological_basis": "TGFb bridge as described in cal target 1.",
            "units": "dimensionless",
        }
    ]
    aux2 = [
        {
            "name": "f_serum_to_tumor_TGFb",
            "group": "serum_to_tumor",
            "biological_basis": "Different rationale text in cal target 2.",
            "units": "dimensionless",
        }
    ]
    _write_cal_target(tmp_path / "t1.yaml", aux=aux)
    _write_cal_target(tmp_path / "t2.yaml", aux=aux2)

    reg = discover_auxiliary_members([tmp_path], _config_with("serum_to_tumor"))
    member = reg.members["f_serum_to_tumor_TGFb"]
    assert len(member.references) == 2
    assert len(member.biological_basis_per_reference) == 2


def test_same_name_conflicting_groups_raises(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "t1.yaml",
        aux=[{"name": "f_x", "group": "g_a", "biological_basis": "x" * 25}],
    )
    _write_cal_target(
        tmp_path / "t2.yaml",
        aux=[{"name": "f_x", "group": "g_b", "biological_basis": "y" * 25}],
    )
    with pytest.raises(ValueError, match="conflicting groups"):
        discover_auxiliary_members([tmp_path], _config_with("g_a", "g_b"))


def test_same_name_conflicting_units_raises(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "t1.yaml",
        aux=[
            {
                "name": "f_x",
                "group": "g",
                "biological_basis": "x" * 25,
                "units": "dimensionless",
            }
        ],
    )
    _write_cal_target(
        tmp_path / "t2.yaml",
        aux=[
            {
                "name": "f_x",
                "group": "g",
                "biological_basis": "y" * 25,
                "units": "nanomolar",
            }
        ],
    )
    with pytest.raises(ValueError, match="conflicting units"):
        discover_auxiliary_members([tmp_path], _config_with("g"))


def test_undeclared_group_raises(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "t.yaml",
        aux=[{"name": "f_x", "group": "missing_group", "biological_basis": "x" * 25}],
    )
    with pytest.raises(ValueError, match="not declared in auxiliary_config"):
        discover_auxiliary_members([tmp_path], _config_with("other_group"))


def test_missing_name_raises(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "t.yaml",
        aux=[{"group": "g", "biological_basis": "x" * 25}],
    )
    with pytest.raises(ValueError, match="missing 'name'"):
        discover_auxiliary_members([tmp_path], _config_with("g"))


def test_missing_group_raises(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "t.yaml",
        aux=[{"name": "f_x", "biological_basis": "x" * 25}],
    )
    with pytest.raises(ValueError, match="missing 'group'"):
        discover_auxiliary_members([tmp_path], _config_with("g"))


def test_non_calibration_yaml_skipped(tmp_path: Path) -> None:
    # Random YAML files in the same tree (e.g., scenario configs) without
    # an `observable` key should be ignored gracefully.
    (tmp_path / "scenario.yaml").write_text(
        yaml.safe_dump({"name": "baseline", "stop_time": 1})
    )
    _write_cal_target(
        tmp_path / "real.yaml",
        aux=[{"name": "f_x", "group": "g", "biological_basis": "x" * 25}],
    )
    reg = discover_auxiliary_members([tmp_path], _config_with("g"))
    assert reg.member_names == ("f_x",)


def test_observable_without_aux_skipped(tmp_path: Path) -> None:
    _write_cal_target(tmp_path / "no_aux.yaml", aux=None)
    reg = discover_auxiliary_members([tmp_path], _config_with())
    assert reg.is_empty


def test_aux_must_be_list(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump(
            {
                "observable": {
                    "code": "x",
                    "units": "nM",
                    "auxiliary_parameters": {"name": "f"},  # not a list
                }
            }
        )
    )
    with pytest.raises(ValueError, match="must be a list"):
        discover_auxiliary_members([tmp_path], _config_with())


def test_registry_helpers(tmp_path: Path) -> None:
    _write_cal_target(
        tmp_path / "t.yaml",
        aux=[
            {"name": "f_b", "group": "g", "biological_basis": "x" * 25},
            {"name": "f_a", "group": "g", "biological_basis": "y" * 25},
        ],
    )
    reg: AuxiliaryRegistry = discover_auxiliary_members(
        [tmp_path], _config_with("g")
    )
    assert reg.group_names == ("g",)
    assert tuple(m.name for m in reg.members_in("g")) == ("f_a", "f_b")
    assert reg.group_spec("g").member_deviation_sigma == 0.2
    with pytest.raises(KeyError):
        reg.group_spec("missing")


def test_file_root_accepted(tmp_path: Path) -> None:
    target = tmp_path / "single.yaml"
    _write_cal_target(
        target,
        aux=[{"name": "f_x", "group": "g", "biological_basis": "x" * 25}],
    )
    reg = discover_auxiliary_members([target], _config_with("g"))
    assert reg.member_names == ("f_x",)

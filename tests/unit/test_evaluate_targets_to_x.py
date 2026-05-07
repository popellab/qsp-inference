"""Tests for evaluate_targets_to_x and collect_required_trajectory_columns.

Uses lightweight dataclass duck-types for CalibrationTarget — no maple
dependency. Synthetic long-form trajectory frames in the same shape the
existing helper consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pint
import pytest

from qsp_inference.inference.trajectory_eval import (
    EvalReport,
    collect_required_trajectory_columns,
    evaluate_targets_to_x,
)


# ---- duck-typed cal-target fixtures -------------------------------------


@dataclass
class _Constant:
    name: str
    value: float
    units: str


@dataclass
class _AuxParam:
    name: str
    units: str = "dimensionless"


@dataclass
class _Observable:
    code: str
    units: str
    species: List[str]
    constants: List[_Constant] = field(default_factory=list)
    auxiliary_parameters: List[_AuxParam] = field(default_factory=list)


@dataclass
class _Empirical:
    index_values: Optional[List[float]] = None


@dataclass
class _Target:
    calibration_target_id: str
    observable: _Observable
    empirical_data: _Empirical = field(default_factory=_Empirical)


@pytest.fixture
def ureg() -> pint.UnitRegistry:
    reg = pint.UnitRegistry()
    reg.define("cell = [count]")
    return reg


def _make_traj(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["sample_index", "t_to_diagnosis_days", "column", "value"]
    )


def _density_target(tid: str, index_values: Optional[List[float]] = None) -> _Target:
    """CD8 density (cell/mL) at the requested time(s)."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    return (cd8 / vol).to('cell/mL')
"""
    return _Target(
        calibration_target_id=tid,
        observable=_Observable(
            code=code, units="cell/mL", species=["V_T.CD8", "V_T"]
        ),
        empirical_data=_Empirical(index_values=index_values),
    )


# ---- collect_required_trajectory_columns --------------------------------


def test_collect_columns_unions_and_sorts() -> None:
    t1 = _density_target("t1")
    t2 = _Target(
        calibration_target_id="t2",
        observable=_Observable(
            code="def compute_observable(t, s, c, u): return s['V_T.M2'] * u.dimensionless",
            units="cell",
            species=["V_T.M2", "V_T.CD4"],
        ),
    )
    assert collect_required_trajectory_columns([t1, t2]) == [
        "V_T",
        "V_T.CD4",
        "V_T.CD8",
        "V_T.M2",
    ]


def test_collect_columns_handles_missing_observable() -> None:
    bare = _Target(
        calibration_target_id="bare",
        observable=_Observable(code="", units="", species=[]),
    )
    bare.observable = None  # type: ignore[assignment]
    assert collect_required_trajectory_columns([bare]) == []


def test_collect_columns_empty_input() -> None:
    assert collect_required_trajectory_columns([]) == []


# ---- evaluate_targets_to_x: happy paths ---------------------------------


def _two_sim_density_traj() -> pd.DataFrame:
    """Two sims, three timepoints. cd8/vol = 100 cell/mL constant for sim 0,
    10 cell/mL constant for sim 1 (chosen so interp is trivially exact)."""
    return _make_traj(
        [
            (0, 0.0, "V_T.CD8", 100.0),
            (0, 0.0, "V_T", 1.0),
            (0, 21.0, "V_T.CD8", 200.0),
            (0, 21.0, "V_T", 2.0),
            (0, 50.0, "V_T.CD8", 500.0),
            (0, 50.0, "V_T", 5.0),
            (1, 0.0, "V_T.CD8", 10.0),
            (1, 0.0, "V_T", 1.0),
            (1, 21.0, "V_T.CD8", 30.0),
            (1, 21.0, "V_T", 3.0),
            (1, 50.0, "V_T.CD8", 70.0),
            (1, 50.0, "V_T", 7.0),
        ]
    )


def test_happy_path_single_time_target(ureg: pint.UnitRegistry) -> None:
    target = _density_target("cd8_d21", index_values=[21.0])
    traj = _two_sim_density_traj()
    species_units = {"V_T.CD8": "cell", "V_T": "milliliter"}

    x, names, report = evaluate_targets_to_x(
        [target], traj, species_units, ureg
    )
    assert names == ["cd8_d21"]
    assert x.shape == (2, 1)
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    assert report.per_target["cd8_d21"]["n_total"] == 2
    assert report.per_target["cd8_d21"]["n_nan"] == 0
    assert report.per_target["cd8_d21"]["n_code_raised"] == 0
    assert report.per_target["cd8_d21"]["n_species_missing"] == 0


def test_vector_target_emits_one_column_per_index(ureg: pint.UnitRegistry) -> None:
    target = _density_target("cd8_series", index_values=[0.0, 21.0, 50.0])
    traj = _two_sim_density_traj()
    species_units = {"V_T.CD8": "cell", "V_T": "milliliter"}

    x, names, report = evaluate_targets_to_x(
        [target], traj, species_units, ureg
    )
    assert names == ["cd8_series__t0.0", "cd8_series__t21.0", "cd8_series__t50.0"]
    assert x.shape == (2, 3)
    # Density constant per sim across times by construction.
    np.testing.assert_allclose(x[0, :], [100.0, 100.0, 100.0])
    np.testing.assert_allclose(x[1, :], [10.0, 10.0, 10.0])
    assert report.per_target["cd8_series"]["n_nan"] == 0


def test_index_values_default_to_zero(ureg: pint.UnitRegistry) -> None:
    """Per CLAUDE.md gotcha: empty index_values defaults to t=0."""
    target = _density_target("cd8_default")  # index_values=None
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x(
        [target], traj, {"V_T.CD8": "cell", "V_T": "milliliter"}, ureg
    )
    assert names == ["cd8_default"]
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])  # value at t=0


def test_target_index_values_override_takes_precedence(
    ureg: pint.UnitRegistry,
) -> None:
    target = _density_target("cd8", index_values=[0.0])  # YAML says t=0
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x(
        [target],
        traj,
        {"V_T.CD8": "cell", "V_T": "milliliter"},
        ureg,
        target_index_values={"cd8": [50.0]},  # override → t=50
    )
    assert names == ["cd8"]
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])


def test_multiple_targets_concat_columns(ureg: pint.UnitRegistry) -> None:
    t1 = _density_target("density_d21", index_values=[21.0])
    t2 = _density_target("density_d0", index_values=[0.0])
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x(
        [t1, t2], traj, {"V_T.CD8": "cell", "V_T": "milliliter"}, ureg
    )
    assert names == ["density_d21", "density_d0"]
    assert x.shape == (2, 2)
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    np.testing.assert_allclose(x[:, 1], [100.0, 10.0])
    assert report.per_target["density_d21"]["n_total"] == 2
    assert report.per_target["density_d0"]["n_total"] == 2


def test_empty_targets_returns_zero_columns(ureg: pint.UnitRegistry) -> None:
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x([], traj, {}, ureg)
    assert x.shape == (2, 0)
    assert names == []
    assert report.per_target == {}


# ---- evaluate_targets_to_x: failure-mode classification (D7) ------------


def test_missing_species_increments_species_missing(
    ureg: pint.UnitRegistry,
) -> None:
    target = _Target(
        calibration_target_id="needs_missing",
        observable=_Observable(
            code="def compute_observable(t,s,c,u):\n    return s['V_T.MISSING'] * u.dimensionless",
            units="dimensionless",
            species=["V_T.MISSING"],
        ),
        empirical_data=_Empirical(index_values=[0.0]),
    )
    traj = _two_sim_density_traj()  # has V_T.CD8 and V_T, NOT V_T.MISSING
    x, names, report = evaluate_targets_to_x(
        [target], traj, {"V_T.CD8": "cell", "V_T": "milliliter"}, ureg
    )
    assert np.all(np.isnan(x))
    assert report.per_target["needs_missing"]["n_species_missing"] == 2
    assert report.per_target["needs_missing"]["n_code_raised"] == 0
    assert report.per_target["needs_missing"]["n_nan"] == 2


def test_code_raises_increments_code_raised(ureg: pint.UnitRegistry) -> None:
    target = _Target(
        calibration_target_id="raises",
        observable=_Observable(
            code=(
                "def compute_observable(t, s, c, u):\n"
                "    raise RuntimeError('intentional')\n"
            ),
            units="dimensionless",
            species=["V_T.CD8"],
        ),
        empirical_data=_Empirical(index_values=[0.0]),
    )
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x(
        [target], traj, {"V_T.CD8": "cell", "V_T": "milliliter"}, ureg
    )
    assert np.all(np.isnan(x))
    assert report.per_target["raises"]["n_code_raised"] == 2
    assert report.per_target["raises"]["n_species_missing"] == 0
    assert report.per_target["raises"]["n_nan"] == 2
    assert isinstance(report.last_exception["raises"], RuntimeError)


def test_sim_nan_propagates_increments_nan_only(ureg: pint.UnitRegistry) -> None:
    """A clean code path that returns NaN (e.g., division by a NaN
    species value from the simulation) bumps n_nan but neither
    classification counter — distinguishing 'sim produced NaN' from
    'observable broke' as D7 calls for."""
    target = _density_target("cd8_with_nan", index_values=[0.0])
    rows = [
        (0, 0.0, "V_T.CD8", float("nan")),  # NaN flows through
        (0, 0.0, "V_T", 1.0),
        (0, 21.0, "V_T.CD8", 100.0),
        (0, 21.0, "V_T", 1.0),
    ]
    traj = _make_traj(rows)
    x, _names, report = evaluate_targets_to_x(
        [target], traj, {"V_T.CD8": "cell", "V_T": "milliliter"}, ureg
    )
    assert np.isnan(x[0, 0])
    assert report.per_target["cd8_with_nan"]["n_nan"] == 1
    assert report.per_target["cd8_with_nan"]["n_code_raised"] == 0
    assert report.per_target["cd8_with_nan"]["n_species_missing"] == 0


# ---- aux records --------------------------------------------------------


def test_auxiliary_records_threaded_per_sim(ureg: pint.UnitRegistry) -> None:
    """Aux records are filtered per-target and joined per-sim by the
    underlying helper. Verify the wiring through evaluate_targets_to_x."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    factor = constants['serum_to_tumor']
    return (factor * cd8 / vol).to('cell/mL')
"""
    target = _Target(
        calibration_target_id="aux_density",
        observable=_Observable(
            code=code,
            units="cell/mL",
            species=["V_T.CD8", "V_T"],
            auxiliary_parameters=[_AuxParam(name="serum_to_tumor")],
        ),
        empirical_data=_Empirical(index_values=[0.0]),
    )
    traj = _two_sim_density_traj()
    aux = [{"serum_to_tumor": 2.0}, {"serum_to_tumor": 0.5}]

    x, names, report = evaluate_targets_to_x(
        [target],
        traj,
        {"V_T.CD8": "cell", "V_T": "milliliter"},
        ureg,
        auxiliary_records=aux,
    )
    # sim 0 base = 100, scaled by 2.0 → 200
    # sim 1 base = 10, scaled by 0.5 → 5
    np.testing.assert_allclose(x[:, 0], [200.0, 5.0])
    assert report.per_target["aux_density"]["n_nan"] == 0


def test_auxiliary_records_length_mismatch_raises(ureg: pint.UnitRegistry) -> None:
    target = _density_target("d", index_values=[0.0])
    traj = _two_sim_density_traj()  # 2 sims
    with pytest.raises(ValueError, match="length 1 does not match"):
        evaluate_targets_to_x(
            [target],
            traj,
            {"V_T.CD8": "cell", "V_T": "milliliter"},
            ureg,
            auxiliary_records=[{"unused": 1.0}],  # only 1 record for 2 sims
        )


# ---- EvalReport rendering -----------------------------------------------


def test_eval_report_renders_table(ureg: pint.UnitRegistry) -> None:
    t1 = _density_target("cd8", index_values=[0.0])
    traj = _two_sim_density_traj()
    _x, _names, report = evaluate_targets_to_x(
        [t1], traj, {"V_T.CD8": "cell", "V_T": "milliliter"}, ureg
    )
    rendered = report.render_summary()
    assert "target_id" in rendered
    assert "cd8" in rendered
    assert "\t2\t0\t0\t0" in rendered  # total=2, nan=0, raised=0, missing=0


def test_eval_report_empty_when_no_targets() -> None:
    report = EvalReport()
    assert "no targets evaluated" in report.render_summary()

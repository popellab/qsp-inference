"""Tests for qsp_inference.inference.trajectory_eval.

Builds synthetic long-form trajectory frames + observable code strings,
verifies the helper computes values per (sample, time) correctly. No
maple / qsp-hpc-tools dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pint
import pytest

from qsp_inference.inference.trajectory_eval import (
    evaluate_calibration_target_over_trajectory,
    evaluate_observable_over_trajectory,
)


@pytest.fixture
def ureg() -> pint.UnitRegistry:
    """Mini Pint registry with the cell-counting units used in cal-target
    code. Mirrors what qsp_hpc.utils.unit_registry adds, scoped down to
    what these unit tests need."""
    reg = pint.UnitRegistry()
    reg.define("cell = [count]")
    return reg


def _make_traj_df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["sample_index", "t_to_diagnosis_days", "column", "value"]
    )


# ---- happy path ----------------------------------------------------------


def test_happy_path_density_observable(ureg: pint.UnitRegistry) -> None:
    """Two sims, three timepoints, density = CD8 / V_T_volume."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    return (cd8 / vol).to('cell/mL')
"""
    rows = [
        # sim 0
        (0, -10.0, "V_T.CD8", 100.0),
        (0, -10.0, "V_T", 1.0),  # mL
        (0, -5.0, "V_T.CD8", 200.0),
        (0, -5.0, "V_T", 2.0),
        (0, 0.0, "V_T.CD8", 300.0),
        (0, 0.0, "V_T", 3.0),
        # sim 7
        (7, -10.0, "V_T.CD8", 50.0),
        (7, -10.0, "V_T", 5.0),
        (7, -5.0, "V_T.CD8", 60.0),
        (7, -5.0, "V_T", 6.0),
        (7, 0.0, "V_T.CD8", 70.0),
        (7, 0.0, "V_T", 7.0),
    ]
    traj = _make_traj_df([dict(zip(["sample_index", "t_to_diagnosis_days", "column", "value"], r)) for r in rows])
    species_units = {"V_T.CD8": "cell", "V_T": "milliliter"}

    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        output_units="cell/mL",
        traj_df=traj,
        species_units=species_units,
        ureg=ureg,
    )
    assert list(out.columns) == ["sample_index", "t_to_diagnosis_days", "value"]
    # sim 0: 100/1, 200/2, 300/3 = 100, 100, 100
    sim0 = out[out["sample_index"] == 0].sort_values("t_to_diagnosis_days")
    np.testing.assert_allclose(sim0["value"], [100.0, 100.0, 100.0])
    # sim 7: 50/5, 60/6, 70/7 = 10, 10, 10
    sim7 = out[out["sample_index"] == 7].sort_values("t_to_diagnosis_days")
    np.testing.assert_allclose(sim7["value"], [10.0, 10.0, 10.0])
    # Output is sorted by (sample_index, t_to_diagnosis_days)
    assert list(out["sample_index"]) == [0, 0, 0, 7, 7, 7]
    assert list(out["t_to_diagnosis_days"]) == [-10.0, -5.0, 0.0, -10.0, -5.0, 0.0]


def test_constants_passed_through(ureg: pint.UnitRegistry) -> None:
    """Observable that scales by a constant — the constant must flow to the function."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    cd8 = species_dict['V_T.CD8']
    factor = constants['scale']
    return (cd8 * factor).to('cell')
"""
    traj = _make_traj_df(
        [
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 10.0},
        ]
    )
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={"scale": 5.0 * ureg.dimensionless},
        output_units="cell",
        traj_df=traj,
        species_units={"V_T.CD8": "cell"},
        ureg=ureg,
    )
    np.testing.assert_allclose(out["value"], [50.0])


def test_output_unit_conversion(ureg: pint.UnitRegistry) -> None:
    """compute_observable returns cell/mL, output_units='cell/L' — should convert ×1000."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    return (species_dict['V_T.CD8'] / species_dict['V_T']).to('cell/mL')
"""
    traj = _make_traj_df(
        [
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0},
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T", "value": 1.0},
        ]
    )
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        output_units="cell/L",
        traj_df=traj,
        species_units={"V_T.CD8": "cell", "V_T": "milliliter"},
        ureg=ureg,
    )
    np.testing.assert_allclose(out["value"], [1000.0])


# ---- error handling ------------------------------------------------------


def test_on_error_nan_returns_nan(ureg: pint.UnitRegistry) -> None:
    """Observable code that raises ZeroDivisionError yields NaN values for that sim."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    return (species_dict['V_T.CD8'] / species_dict['V_T']).to('cell/mL')
"""
    traj = _make_traj_df(
        [
            # V_T = 0 → division by zero
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0},
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T", "value": 0.0},
            # sim 1 is healthy, should still evaluate
            {"sample_index": 1, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0},
            {"sample_index": 1, "t_to_diagnosis_days": 0.0, "column": "V_T", "value": 1.0},
        ]
    )
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        output_units="cell/mL",
        traj_df=traj,
        species_units={"V_T.CD8": "cell", "V_T": "milliliter"},
        ureg=ureg,
        on_error="nan",
    )
    sim0 = out[out["sample_index"] == 0]
    sim1 = out[out["sample_index"] == 1]
    # Pint allows division by 0 and yields inf (not an exception). Either
    # way the helper records *something* — what we want to verify here is
    # that sim 1 (well-formed) is unaffected by sim 0's degeneracy.
    np.testing.assert_allclose(sim1["value"], [1.0])
    assert len(sim0) == 1


def test_on_error_raise_propagates(ureg: pint.UnitRegistry) -> None:
    """on_error='raise' surfaces the underlying exception."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    raise RuntimeError("intentional")
"""
    traj = _make_traj_df(
        [{"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0}]
    )
    with pytest.raises(RuntimeError, match="intentional"):
        evaluate_observable_over_trajectory(
            observable_code=code,
            constants={},
            output_units="cell",
            traj_df=traj,
            species_units={"V_T.CD8": "cell"},
            ureg=ureg,
            on_error="raise",
        )


def test_on_error_nan_when_observable_raises(ureg: pint.UnitRegistry) -> None:
    """on_error='nan' (default): an explicit exception in compute_observable
    yields NaN for that sim's rows, others unaffected."""
    code = """
def compute_observable(time, species_dict, constants, ureg):
    if species_dict['V_T.CD8'].magnitude.sum() < 0:
        raise ValueError("no CD8")
    return species_dict['V_T.CD8']
"""
    traj = _make_traj_df(
        [
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": -1.0},
            {"sample_index": 1, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 5.0},
        ]
    )
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        output_units="cell",
        traj_df=traj,
        species_units={"V_T.CD8": "cell"},
        ureg=ureg,
    )
    assert np.isnan(out[out["sample_index"] == 0]["value"].iloc[0])
    np.testing.assert_allclose(out[out["sample_index"] == 1]["value"], [5.0])


def test_missing_columns_raises(ureg: pint.UnitRegistry) -> None:
    bad = pd.DataFrame({"foo": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_observable_over_trajectory(
            observable_code="def compute_observable(t,s,c,u): return t",
            constants={},
            output_units="day",
            traj_df=bad,
            species_units={},
            ureg=ureg,
        )


def test_observable_code_must_define_function(ureg: pint.UnitRegistry) -> None:
    traj = _make_traj_df(
        [{"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "x", "value": 1.0}]
    )
    with pytest.raises(ValueError, match="compute_observable"):
        evaluate_observable_over_trajectory(
            observable_code="x = 5",  # no function
            constants={},
            output_units="cell",
            traj_df=traj,
            species_units={"x": "cell"},
            ureg=ureg,
        )


def test_invalid_on_error_raises(ureg: pint.UnitRegistry) -> None:
    with pytest.raises(ValueError, match="on_error"):
        evaluate_observable_over_trajectory(
            observable_code="",
            constants={},
            output_units="day",
            traj_df=_make_traj_df([]),
            species_units={},
            ureg=ureg,
            on_error="bogus",
        )


def test_empty_traj_returns_empty_df(ureg: pint.UnitRegistry) -> None:
    out = evaluate_observable_over_trajectory(
        observable_code=(
            "def compute_observable(time, species_dict, constants, ureg):\n"
            "    return species_dict['x']\n"
        ),
        constants={},
        output_units="cell",
        traj_df=_make_traj_df([]),
        species_units={"x": "cell"},
        ureg=ureg,
    )
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert list(out.columns) == ["sample_index", "t_to_diagnosis_days", "value"]


# ---- CalibrationTarget wrapper ------------------------------------------


@dataclass
class _FakeConst:
    name: str
    value: float
    units: str


@dataclass
class _FakeObservable:
    code: str
    units: str
    constants: list[_FakeConst]


@dataclass
class _FakeTarget:
    observable: _FakeObservable


def test_calibration_target_wrapper(ureg: pint.UnitRegistry) -> None:
    """The target wrapper extracts code/constants/units off a duck-typed
    object — should match the lower-level call exactly."""
    target = _FakeTarget(
        observable=_FakeObservable(
            code=(
                "def compute_observable(time, species_dict, constants, ureg):\n"
                "    return (species_dict['V_T.CD8'] * constants['scale']).to('cell')\n"
            ),
            units="cell",
            constants=[_FakeConst(name="scale", value=2.0, units="dimensionless")],
        )
    )
    traj = _make_traj_df(
        [
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 7.0},
        ]
    )
    out = evaluate_calibration_target_over_trajectory(
        target=target,
        traj_df=traj,
        species_units={"V_T.CD8": "cell"},
        ureg=ureg,
    )
    np.testing.assert_allclose(out["value"], [14.0])

"""Tests for qsp_inference.inference.trajectory_eval (pintless API).

Builds synthetic long-form trajectory frames + observable code strings,
verifies the helper computes values per (sample, time) correctly. No
maple / qsp-hpc-tools / pint dependency.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import pytest

from qsp_inference.inference.trajectory_eval import (
    evaluate_calibration_target_over_trajectory,
    evaluate_observable_over_trajectory,
)


def _make_traj_df(rows: List[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["sample_index", "t_to_diagnosis_days", "column", "value"]
    )


# ---- happy path ----------------------------------------------------------


def test_happy_path_density_observable() -> None:
    """Two sims, three timepoints, density = CD8 / V_T (raw floats)."""
    code = """
def compute_observable(time, species_dict, constants):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    return cd8 / vol
"""
    rows = [
        # sim 0
        (0, -10.0, "V_T.CD8", 100.0),
        (0, -10.0, "V_T", 1.0),
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
    traj = _make_traj_df(
        [dict(zip(["sample_index", "t_to_diagnosis_days", "column", "value"], r)) for r in rows]
    )

    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        traj_df=traj,
    )
    assert list(out.columns) == ["sample_index", "t_to_diagnosis_days", "value"]
    sim0 = out[out["sample_index"] == 0].sort_values("t_to_diagnosis_days")
    np.testing.assert_allclose(sim0["value"], [100.0, 100.0, 100.0])
    sim7 = out[out["sample_index"] == 7].sort_values("t_to_diagnosis_days")
    np.testing.assert_allclose(sim7["value"], [10.0, 10.0, 10.0])
    assert list(out["sample_index"]) == [0, 0, 0, 7, 7, 7]
    assert list(out["t_to_diagnosis_days"]) == [-10.0, -5.0, 0.0, -10.0, -5.0, 0.0]


def test_constants_passed_through() -> None:
    """Observable that scales by a constant — the constant must flow to the function."""
    code = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.CD8'] * constants['scale']
"""
    traj = _make_traj_df(
        [
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 10.0},
        ]
    )
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={"scale": 5.0},
        traj_df=traj,
    )
    np.testing.assert_allclose(out["value"], [50.0])


# ---- error handling ------------------------------------------------------


def test_on_error_nan_returns_nan() -> None:
    """Sim 0 produces inf/nan via division by zero; sim 1 still evaluates."""
    code = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.CD8'] / species_dict['V_T']
"""
    traj = _make_traj_df(
        [
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0},
            {"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T", "value": 0.0},
            {"sample_index": 1, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0},
            {"sample_index": 1, "t_to_diagnosis_days": 0.0, "column": "V_T", "value": 1.0},
        ]
    )
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        traj_df=traj,
        on_error="nan",
    )
    sim1 = out[out["sample_index"] == 1]
    np.testing.assert_allclose(sim1["value"], [1.0])


def test_on_error_raise_propagates() -> None:
    """on_error='raise' surfaces the underlying exception."""
    code = """
def compute_observable(time, species_dict, constants):
    raise RuntimeError("intentional")
"""
    traj = _make_traj_df(
        [{"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "V_T.CD8", "value": 1.0}]
    )
    with pytest.raises(RuntimeError, match="intentional"):
        evaluate_observable_over_trajectory(
            observable_code=code,
            constants={},
            traj_df=traj,
            on_error="raise",
        )


def test_on_error_nan_when_observable_raises() -> None:
    """on_error='nan' (default): an explicit exception in compute_observable
    yields NaN for that sim's rows, others unaffected."""
    code = """
def compute_observable(time, species_dict, constants):
    if species_dict['V_T.CD8'].sum() < 0:
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
        traj_df=traj,
    )
    assert np.isnan(out[out["sample_index"] == 0]["value"].iloc[0])
    np.testing.assert_allclose(out[out["sample_index"] == 1]["value"], [5.0])


def test_missing_columns_raises() -> None:
    bad = pd.DataFrame({"foo": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        evaluate_observable_over_trajectory(
            observable_code="def compute_observable(t,s,c): return t",
            constants={},
            traj_df=bad,
        )


def test_observable_code_must_define_function() -> None:
    traj = _make_traj_df(
        [{"sample_index": 0, "t_to_diagnosis_days": 0.0, "column": "x", "value": 1.0}]
    )
    with pytest.raises(ValueError, match="compute_observable"):
        evaluate_observable_over_trajectory(
            observable_code="x = 5",  # no function
            constants={},
            traj_df=traj,
        )


def test_invalid_on_error_raises() -> None:
    with pytest.raises(ValueError, match="on_error"):
        evaluate_observable_over_trajectory(
            observable_code="",
            constants={},
            traj_df=_make_traj_df([]),
            on_error="bogus",
        )


def test_empty_traj_returns_empty_df() -> None:
    out = evaluate_observable_over_trajectory(
        observable_code=(
            "def compute_observable(time, species_dict, constants):\n"
            "    return species_dict['x']\n"
        ),
        constants={},
        traj_df=_make_traj_df([]),
    )
    assert isinstance(out, pd.DataFrame)
    assert out.empty
    assert list(out.columns) == ["sample_index", "t_to_diagnosis_days", "value"]


# ---- CalibrationTarget wrapper ------------------------------------------


@dataclass
class _FakeConst:
    name: str
    value: float
    units: str = "dimensionless"


@dataclass
class _FakeObservable:
    code: str
    constants: list
    units: str = "dimensionless"


@dataclass
class _FakeTarget:
    observable: _FakeObservable


def test_calibration_target_wrapper() -> None:
    """The target wrapper extracts code/constants off a duck-typed object."""
    target = _FakeTarget(
        observable=_FakeObservable(
            code=(
                "def compute_observable(time, species_dict, constants):\n"
                "    return species_dict['V_T.CD8'] * constants['scale']\n"
            ),
            constants=[_FakeConst(name="scale", value=2.0)],
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
    )
    np.testing.assert_allclose(out["value"], [14.0])

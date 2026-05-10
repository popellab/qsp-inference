"""Tests for ``auxiliary_per_sim`` plumbing in ``trajectory_eval`` (pintless API)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from qsp_inference.inference.trajectory_eval import (
    evaluate_calibration_target_over_trajectory,
    evaluate_observable_over_trajectory,
)


def _traj(sample_index_values=(0, 1)) -> pd.DataFrame:
    rows = []
    for s in sample_index_values:
        for t in (-2.0, 0.0):
            rows.append(
                {
                    "sample_index": s,
                    "t_to_diagnosis_days": t,
                    "column": "V_T.X",
                    "value": 1.0 + s + 0.5 * t,
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# evaluate_observable_over_trajectory
# ----------------------------------------------------------------------


def test_aux_per_sim_threads_into_observable() -> None:
    """Different per-sim aux values yield different observables."""
    code = """
def compute_observable(time, species_dict, constants):
    x = species_dict['V_T.X']
    f = constants['f_serum_to_tumor']
    return x * f
"""
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        traj_df=_traj((0, 1)),
        auxiliary_per_sim={
            0: {"f_serum_to_tumor": 2.0},
            1: {"f_serum_to_tumor": 5.0},
        },
    )
    sim0 = out[out["sample_index"] == 0].sort_values("t_to_diagnosis_days")
    sim1 = out[out["sample_index"] == 1].sort_values("t_to_diagnosis_days")
    # sim 0 species values 0.0 (t=-2), 1.0 (t=0); * 2.0 = 0.0, 2.0
    np.testing.assert_allclose(sim0["value"].to_numpy(), [0.0, 2.0])
    # sim 1 species values 1.0 (t=-2), 2.0 (t=0); * 5.0 = 5.0, 10.0
    np.testing.assert_allclose(sim1["value"].to_numpy(), [5.0, 10.0])


def test_aux_per_sim_default_none_preserves_legacy_behavior() -> None:
    code = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.X']
"""
    out = evaluate_observable_over_trajectory(
        observable_code=code,
        constants={},
        traj_df=_traj((0,)),
    )
    assert len(out) == 2
    np.testing.assert_allclose(
        out.sort_values("t_to_diagnosis_days")["value"].to_numpy(), [0.0, 1.0]
    )


def test_aux_collision_with_constants_raises() -> None:
    code = "def compute_observable(t, s, c): return s['V_T.X']"
    with pytest.raises(ValueError, match="collide with observable.constants"):
        evaluate_observable_over_trajectory(
            observable_code=code,
            constants={"f_x": 1.0},
            traj_df=_traj((0,)),
            auxiliary_per_sim={0: {"f_x": 2.0}},
        )


def test_aux_missing_for_sample_falls_back_to_constants() -> None:
    """Missing sample_index in auxiliary_per_sim ⇒ evaluate with constants
    only. With on_error='raise', a missing aux key surfaces as a KeyError
    from the observable code (not a silent NaN)."""
    code = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.X'] * constants['f']
"""
    with pytest.raises(KeyError):
        evaluate_observable_over_trajectory(
            observable_code=code,
            constants={},
            traj_df=_traj((0, 1)),
            auxiliary_per_sim={0: {"f": 1.0}},
            on_error="raise",
        )


# ----------------------------------------------------------------------
# evaluate_calibration_target_over_trajectory (sugar wrapper)
# ----------------------------------------------------------------------


@dataclass
class _ConstStub:
    name: str
    value: float
    units: str = "dimensionless"


@dataclass
class _AuxStub:
    name: str
    units: str = "dimensionless"


@dataclass
class _ObservableStub:
    code: str
    units: str = "dimensionless"
    constants: list = field(default_factory=list)
    auxiliary_parameters: list = field(default_factory=list)


@dataclass
class _TargetStub:
    observable: _ObservableStub


def test_calibration_target_wrapper_filters_per_target_aux() -> None:
    """Registry has aux f1, f2; target only declares f1 — only f1 is
    threaded into observable.code."""
    code = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.X'] * constants['f1']
"""
    target = _TargetStub(
        observable=_ObservableStub(
            code=code,
            auxiliary_parameters=[_AuxStub(name="f1")],
        )
    )
    records = [
        {"f1": 2.0, "f2": 99.0},
        {"f1": 5.0, "f2": -7.0},
    ]
    out = evaluate_calibration_target_over_trajectory(
        target,
        traj_df=_traj((0, 1)),
        auxiliary_records=records,
    )
    sim0 = out[out["sample_index"] == 0].sort_values("t_to_diagnosis_days")
    sim1 = out[out["sample_index"] == 1].sort_values("t_to_diagnosis_days")
    np.testing.assert_allclose(sim0["value"].to_numpy(), [0.0, 2.0])
    np.testing.assert_allclose(sim1["value"].to_numpy(), [5.0, 10.0])


def test_calibration_target_wrapper_no_aux_ignores_records() -> None:
    """Target without auxiliary_parameters silently ignores records."""
    code = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.X']
"""
    target = _TargetStub(observable=_ObservableStub(code=code))
    out = evaluate_calibration_target_over_trajectory(
        target,
        traj_df=_traj((0,)),
        auxiliary_records=[{"unused": 1.0}],
    )
    assert len(out) == 2


def test_calibration_target_wrapper_record_count_mismatch() -> None:
    target = _TargetStub(
        observable=_ObservableStub(
            code="def compute_observable(t, s, c): return s['V_T.X']",
            auxiliary_parameters=[_AuxStub(name="f1")],
        )
    )
    with pytest.raises(ValueError, match="does not match"):
        evaluate_calibration_target_over_trajectory(
            target,
            traj_df=_traj((0, 1)),
            auxiliary_records=[{"f1": 1.0}],  # only 1 record, 2 sims
        )


def test_calibration_target_wrapper_missing_declared_aux_in_record() -> None:
    target = _TargetStub(
        observable=_ObservableStub(
            code="def compute_observable(t, s, c): return s['V_T.X']",
            auxiliary_parameters=[_AuxStub(name="f1")],
        )
    )
    with pytest.raises(ValueError, match="missing"):
        evaluate_calibration_target_over_trajectory(
            target,
            traj_df=_traj((0,)),
            auxiliary_records=[{"f2": 1.0}],
        )

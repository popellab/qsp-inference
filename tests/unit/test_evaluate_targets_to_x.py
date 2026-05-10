"""Tests for evaluate_targets_to_x and collect_required_trajectory_columns
(pintless API).

Uses lightweight dataclass duck-types for CalibrationTarget — no maple
or pint dependency. Synthetic long-form trajectory frames in the same
shape the existing helper consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
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
    units: str = "dimensionless"


@dataclass
class _AuxParam:
    name: str
    units: str = "dimensionless"


@dataclass
class _Observable:
    code: str
    species: List[str]
    units: str = "dimensionless"
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


def _make_traj(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["sample_index", "t_to_diagnosis_days", "column", "value"]
    )


def _density_target(tid: str, index_values: Optional[List[float]] = None) -> _Target:
    """CD8 density (cell/mL) at the requested time(s) — raw float arithmetic."""
    code = """
def compute_observable(time, species_dict, constants):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    return cd8 / vol
"""
    return _Target(
        calibration_target_id=tid,
        observable=_Observable(
            code=code, species=["V_T.CD8", "V_T"]
        ),
        empirical_data=_Empirical(index_values=index_values),
    )


# ---- collect_required_trajectory_columns --------------------------------


def test_collect_columns_unions_and_sorts() -> None:
    t1 = _density_target("t1")
    t2 = _Target(
        calibration_target_id="t2",
        observable=_Observable(
            code="def compute_observable(t, s, c): return s['V_T.M2']",
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
        observable=_Observable(code="", species=[]),
    )
    bare.observable = None  # type: ignore[assignment]
    assert collect_required_trajectory_columns([bare]) == []


def test_collect_columns_empty_input() -> None:
    assert collect_required_trajectory_columns([]) == []


# ---- evaluate_targets_to_x: happy paths ---------------------------------


def _two_sim_density_traj() -> pd.DataFrame:
    """Two sims, three timepoints. cd8/vol = 100 for sim 0, 10 for sim 1."""
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


def test_happy_path_single_time_target() -> None:
    target = _density_target("cd8_d21", index_values=[21.0])
    traj = _two_sim_density_traj()

    x, names, report = evaluate_targets_to_x([target], traj)
    assert names == ["cd8_d21"]
    assert x.shape == (2, 1)
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    assert report.per_target["cd8_d21"]["n_total"] == 2
    assert report.per_target["cd8_d21"]["n_nan"] == 0
    assert report.per_target["cd8_d21"]["n_code_raised"] == 0
    assert report.per_target["cd8_d21"]["n_species_missing"] == 0


def test_vector_target_emits_one_column_per_index() -> None:
    target = _density_target("cd8_series", index_values=[0.0, 21.0, 50.0])
    traj = _two_sim_density_traj()

    x, names, report = evaluate_targets_to_x([target], traj)
    assert names == ["cd8_series__t0.0", "cd8_series__t21.0", "cd8_series__t50.0"]
    assert x.shape == (2, 3)
    np.testing.assert_allclose(x[0, :], [100.0, 100.0, 100.0])
    np.testing.assert_allclose(x[1, :], [10.0, 10.0, 10.0])
    assert report.per_target["cd8_series"]["n_nan"] == 0


def test_index_values_default_to_zero() -> None:
    """Per CLAUDE.md gotcha: empty index_values defaults to t=0."""
    target = _density_target("cd8_default")  # index_values=None
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x([target], traj)
    assert names == ["cd8_default"]
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])


def test_target_index_values_override_takes_precedence() -> None:
    target = _density_target("cd8", index_values=[0.0])  # YAML says t=0
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x(
        [target],
        traj,
        target_index_values={"cd8": [50.0]},  # override → t=50
    )
    assert names == ["cd8"]
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])


def test_multiple_targets_concat_columns() -> None:
    t1 = _density_target("density_d21", index_values=[21.0])
    t2 = _density_target("density_d0", index_values=[0.0])
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x([t1, t2], traj)
    assert names == ["density_d21", "density_d0"]
    assert x.shape == (2, 2)
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    np.testing.assert_allclose(x[:, 1], [100.0, 10.0])
    assert report.per_target["density_d21"]["n_total"] == 2
    assert report.per_target["density_d0"]["n_total"] == 2


def test_empty_targets_returns_zero_columns() -> None:
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x([], traj)
    assert x.shape == (2, 0)
    assert names == []
    assert report.per_target == {}


# ---- evaluate_targets_to_x: failure-mode classification (D7) ------------


def test_missing_species_increments_species_missing() -> None:
    target = _Target(
        calibration_target_id="needs_missing",
        observable=_Observable(
            code="def compute_observable(t,s,c):\n    return s['V_T.MISSING']",
            species=["V_T.MISSING"],
        ),
        empirical_data=_Empirical(index_values=[0.0]),
    )
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x([target], traj)
    assert np.all(np.isnan(x))
    assert report.per_target["needs_missing"]["n_species_missing"] == 2
    assert report.per_target["needs_missing"]["n_code_raised"] == 0
    assert report.per_target["needs_missing"]["n_nan"] == 2


def test_code_raises_increments_code_raised() -> None:
    target = _Target(
        calibration_target_id="raises",
        observable=_Observable(
            code=(
                "def compute_observable(t, s, c):\n"
                "    raise RuntimeError('intentional')\n"
            ),
            species=["V_T.CD8"],
        ),
        empirical_data=_Empirical(index_values=[0.0]),
    )
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x([target], traj)
    assert np.all(np.isnan(x))
    assert report.per_target["raises"]["n_code_raised"] == 2
    assert report.per_target["raises"]["n_species_missing"] == 0
    assert report.per_target["raises"]["n_nan"] == 2
    assert isinstance(report.last_exception["raises"], RuntimeError)


def test_sim_nan_propagates_increments_nan_only() -> None:
    """A clean code path that returns NaN bumps n_nan but neither
    classification counter."""
    target = _density_target("cd8_with_nan", index_values=[0.0])
    rows = [
        (0, 0.0, "V_T.CD8", float("nan")),
        (0, 0.0, "V_T", 1.0),
        (0, 21.0, "V_T.CD8", 100.0),
        (0, 21.0, "V_T", 1.0),
    ]
    traj = _make_traj(rows)
    x, _names, report = evaluate_targets_to_x([target], traj)
    assert np.isnan(x[0, 0])
    assert report.per_target["cd8_with_nan"]["n_nan"] == 1
    assert report.per_target["cd8_with_nan"]["n_code_raised"] == 0
    assert report.per_target["cd8_with_nan"]["n_species_missing"] == 0


# ---- aux records --------------------------------------------------------


def test_auxiliary_records_threaded_per_sim() -> None:
    code = """
def compute_observable(time, species_dict, constants):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    factor = constants['serum_to_tumor']
    return factor * cd8 / vol
"""
    target = _Target(
        calibration_target_id="aux_density",
        observable=_Observable(
            code=code,
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
        auxiliary_records=aux,
    )
    np.testing.assert_allclose(x[:, 0], [200.0, 5.0])
    assert report.per_target["aux_density"]["n_nan"] == 0


def test_auxiliary_records_length_mismatch_raises() -> None:
    target = _density_target("d", index_values=[0.0])
    traj = _two_sim_density_traj()  # 2 sims
    with pytest.raises(ValueError, match="length 1 does not match"):
        evaluate_targets_to_x(
            [target],
            traj,
            auxiliary_records=[{"unused": 1.0}],
        )


# ---- EvalReport rendering -----------------------------------------------


def test_eval_report_renders_table() -> None:
    t1 = _density_target("cd8", index_values=[0.0])
    traj = _two_sim_density_traj()
    _x, _names, report = evaluate_targets_to_x([t1], traj)
    rendered = report.render_summary()
    assert "target_id" in rendered
    assert "cd8" in rendered
    assert "\t2\t0\t0\t0" in rendered


def test_eval_report_empty_when_no_targets() -> None:
    report = EvalReport()
    assert "no targets evaluated" in report.render_summary()

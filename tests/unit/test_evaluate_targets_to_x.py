"""Tests for evaluate_targets_to_x and collect_required_trajectory_columns
(pintless API).

Uses lightweight dataclass duck-types for CalibrationTarget — no maple
model import (the shared Pint registry is still used for readout_time unit
conversion inside the reducer). Synthetic long-form trajectory frames in the
same shape the existing helper consumes.

The time-series → scalar reduction lives on ``observable``: exactly one of
``readout_time`` (+ ``readout_time_unit``) or ``reduce_observable`` is set.
The removed ``empirical_data.index_values`` vector index is gone, so every
target contributes exactly one feature column.
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
    readout_time: Optional[float] = None
    readout_time_unit: Optional[str] = None
    reduce_observable: Optional[str] = None


@dataclass
class _Target:
    calibration_target_id: str
    observable: _Observable


def _make_traj(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows, columns=["sample_index", "t_to_diagnosis_days", "column", "value"]
    )


_DENSITY_CODE = """
def compute_observable(time, species_dict, constants):
    cd8 = species_dict['V_T.CD8']
    vol = species_dict['V_T']
    return cd8 / vol
"""

# cd8 count alone — varies over the time axis, unlike the density (which is
# constant by construction in the fixture below), so peak/AUC reductions and
# readout-time overrides are actually observable.
_CD8_COUNT_CODE = """
def compute_observable(time, species_dict, constants):
    return species_dict['V_T.CD8']
"""


def _density_target(
    tid: str,
    readout_time: Optional[float] = 0.0,
    readout_time_unit: Optional[str] = "day",
) -> _Target:
    """CD8 density (cell/mL) read at ``readout_time`` — raw float arithmetic."""
    return _Target(
        calibration_target_id=tid,
        observable=_Observable(
            code=_DENSITY_CODE,
            species=["V_T.CD8", "V_T"],
            readout_time=readout_time,
            readout_time_unit=readout_time_unit,
        ),
    )


def _cd8_count_target(
    tid: str,
    readout_time: Optional[float] = None,
    readout_time_unit: Optional[str] = None,
    reduce_observable: Optional[str] = None,
) -> _Target:
    return _Target(
        calibration_target_id=tid,
        observable=_Observable(
            code=_CD8_COUNT_CODE,
            species=["V_T.CD8"],
            readout_time=readout_time,
            readout_time_unit=readout_time_unit,
            reduce_observable=reduce_observable,
        ),
    )


# ---- collect_required_trajectory_columns --------------------------------


def test_collect_columns_unions_and_sorts() -> None:
    t1 = _density_target("t1")
    t2 = _Target(
        calibration_target_id="t2",
        observable=_Observable(
            code="def compute_observable(t, s, c): return s['V_T.M2']",
            species=["V_T.M2", "V_T.CD4"],
            readout_time=0.0,
            readout_time_unit="day",
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
    target = _density_target("cd8_d21", readout_time=21.0)
    traj = _two_sim_density_traj()

    x, names, report = evaluate_targets_to_x([target], traj)
    assert names == ["cd8_d21"]
    assert x.shape == (2, 1)
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    assert report.per_target["cd8_d21"]["n_total"] == 2
    assert report.per_target["cd8_d21"]["n_nan"] == 0
    assert report.per_target["cd8_d21"]["n_code_raised"] == 0
    assert report.per_target["cd8_d21"]["n_species_missing"] == 0


def test_readout_time_selects_the_declared_time() -> None:
    """A non-baseline readout picks that point on the series, not t=0."""
    at_0 = _cd8_count_target("cd8_t0", readout_time=0.0, readout_time_unit="day")
    at_50 = _cd8_count_target("cd8_t50", readout_time=50.0, readout_time_unit="day")
    traj = _two_sim_density_traj()

    x, names, _ = evaluate_targets_to_x([at_0, at_50], traj)
    assert names == ["cd8_t0", "cd8_t50"]
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    np.testing.assert_allclose(x[:, 1], [500.0, 70.0])


def test_readout_time_unit_converted_to_canonical_days() -> None:
    """readout_time is in readout_time_unit; the time axis is in days."""
    in_days = _cd8_count_target("in_days", readout_time=21.0, readout_time_unit="day")
    in_hours = _cd8_count_target(
        "in_hours", readout_time=21.0 * 24.0, readout_time_unit="hour"
    )
    traj = _two_sim_density_traj()

    x, _names, _ = evaluate_targets_to_x([in_days, in_hours], traj)
    np.testing.assert_allclose(x[:, 0], [200.0, 30.0])
    np.testing.assert_allclose(x[:, 1], x[:, 0])


def test_readout_time_zero_reads_baseline() -> None:
    """A baseline snapshot states readout_time: 0.0 explicitly."""
    target = _cd8_count_target("baseline", readout_time=0.0, readout_time_unit="day")
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x([target], traj)
    assert names == ["baseline"]
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])


def test_reduce_observable_peak() -> None:
    """A peak/Cmax measurement reduces the whole series, not one time."""
    target = _cd8_count_target(
        "cd8_peak",
        reduce_observable=(
            "def reduce_observable(time, series):\n"
            "    import numpy as np\n"
            "    return float(np.max(series))\n"
        ),
    )
    traj = _two_sim_density_traj()

    x, names, report = evaluate_targets_to_x([target], traj)
    assert names == ["cd8_peak"]
    assert x.shape == (2, 1)
    np.testing.assert_allclose(x[:, 0], [500.0, 70.0])
    assert report.per_target["cd8_peak"]["n_nan"] == 0


def _trapezoid(series: list[float], time: list[float]) -> float:
    """Trapezoidal AUC, spelled without np.trapz / np.trapezoid so the test
    is agnostic to the numpy 1.x → 2.x rename."""
    s = np.asarray(series, dtype=float)
    t = np.asarray(time, dtype=float)
    return float(np.sum(np.diff(t) * (s[:-1] + s[1:]) / 2.0))


def test_reduce_observable_auc() -> None:
    """AUC reduction consumes both the time axis and the series."""
    target = _cd8_count_target(
        "cd8_auc",
        reduce_observable=(
            "def reduce_observable(time, series):\n"
            "    import numpy as np\n"
            "    return float(\n"
            "        np.sum(np.diff(time) * (series[:-1] + series[1:]) / 2.0)\n"
            "    )\n"
        ),
    )
    traj = _two_sim_density_traj()

    x, _names, _ = evaluate_targets_to_x([target], traj)
    expected = [
        _trapezoid([100.0, 200.0, 500.0], [0.0, 21.0, 50.0]),
        _trapezoid([10.0, 30.0, 70.0], [0.0, 21.0, 50.0]),
    ]
    np.testing.assert_allclose(x[:, 0], expected)


def test_reduce_observable_raising_counts_as_code_raised() -> None:
    target = _cd8_count_target(
        "bad_reduce",
        reduce_observable=(
            "def reduce_observable(time, series):\n"
            "    raise RuntimeError('bad reduction')\n"
        ),
    )
    traj = _two_sim_density_traj()

    x, _names, report = evaluate_targets_to_x([target], traj)
    assert np.all(np.isnan(x))
    assert report.per_target["bad_reduce"]["n_code_raised"] == 2
    assert report.per_target["bad_reduce"]["n_nan"] == 2
    assert isinstance(report.last_exception["bad_reduce"], RuntimeError)


def test_reduce_observable_missing_function_raises() -> None:
    target = _cd8_count_target(
        "no_fn", reduce_observable="def not_the_reducer(time, series):\n    return 1.0\n"
    )
    traj = _two_sim_density_traj()
    with pytest.raises(ValueError, match="reduce_observable"):
        evaluate_targets_to_x([target], traj)


def test_neither_reduction_set_raises() -> None:
    """No implicit t=0 default anymore — a missing choice is an error."""
    target = _cd8_count_target("undeclared")
    traj = _two_sim_density_traj()
    with pytest.raises(ValueError, match="neither readout_time nor reduce_observable"):
        evaluate_targets_to_x([target], traj)


def test_both_reductions_set_raises() -> None:
    target = _cd8_count_target(
        "both",
        readout_time=0.0,
        readout_time_unit="day",
        reduce_observable=(
            "def reduce_observable(time, series):\n    return float(series[0])\n"
        ),
    )
    traj = _two_sim_density_traj()
    with pytest.raises(ValueError, match="both readout_time and reduce_observable"):
        evaluate_targets_to_x([target], traj)


def test_readout_time_without_unit_raises() -> None:
    target = _cd8_count_target("no_unit", readout_time=21.0, readout_time_unit=None)
    traj = _two_sim_density_traj()
    with pytest.raises(ValueError, match="readout_time_unit"):
        evaluate_targets_to_x([target], traj)


def test_target_readout_times_override_takes_precedence() -> None:
    """The surviving override forces a value-at-a-time readout (days)."""
    target = _cd8_count_target("cd8", readout_time=0.0, readout_time_unit="day")
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x(
        [target],
        traj,
        target_readout_times={"cd8": 50.0},  # override → t=50 days
    )
    assert names == ["cd8"]
    np.testing.assert_allclose(x[:, 0], [500.0, 70.0])


def test_target_readout_times_override_supersedes_reduce_observable() -> None:
    target = _cd8_count_target(
        "cd8_peak",
        reduce_observable=(
            "def reduce_observable(time, series):\n"
            "    import numpy as np\n"
            "    return float(np.max(series))\n"
        ),
    )
    traj = _two_sim_density_traj()
    x, _names, _ = evaluate_targets_to_x(
        [target], traj, target_readout_times={"cd8_peak": 0.0}
    )
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])


def test_multiple_targets_concat_columns() -> None:
    t1 = _density_target("density_d21", readout_time=21.0)
    t2 = _density_target("density_d0", readout_time=0.0)
    traj = _two_sim_density_traj()
    x, names, report = evaluate_targets_to_x([t1, t2], traj)
    assert names == ["density_d21", "density_d0"]
    assert x.shape == (2, 2)
    np.testing.assert_allclose(x[:, 0], [100.0, 10.0])
    np.testing.assert_allclose(x[:, 1], [100.0, 10.0])
    assert report.per_target["density_d21"]["n_total"] == 2
    assert report.per_target["density_d0"]["n_total"] == 2


def test_each_target_emits_exactly_one_column() -> None:
    """No vector index dimension: N targets → N columns, always."""
    targets = [
        _density_target("a", readout_time=0.0),
        _cd8_count_target(
            "b",
            reduce_observable=(
                "def reduce_observable(time, series):\n"
                "    import numpy as np\n"
                "    return float(np.max(series))\n"
            ),
        ),
        _density_target("c", readout_time=50.0),
    ]
    traj = _two_sim_density_traj()
    x, names, _ = evaluate_targets_to_x(targets, traj)
    assert names == ["a", "b", "c"]
    assert x.shape == (2, 3)


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
            readout_time=0.0,
            readout_time_unit="day",
        ),
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
            readout_time=0.0,
            readout_time_unit="day",
        ),
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
    target = _density_target("cd8_with_nan", readout_time=0.0)
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
            readout_time=0.0,
            readout_time_unit="day",
        ),
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
    target = _density_target("d", readout_time=0.0)
    traj = _two_sim_density_traj()  # 2 sims
    with pytest.raises(ValueError, match="length 1 does not match"):
        evaluate_targets_to_x(
            [target],
            traj,
            auxiliary_records=[{"unused": 1.0}],
        )


# ---- EvalReport rendering -----------------------------------------------


def test_eval_report_renders_table() -> None:
    t1 = _density_target("cd8", readout_time=0.0)
    traj = _two_sim_density_traj()
    _x, _names, report = evaluate_targets_to_x([t1], traj)
    rendered = report.render_summary()
    assert "target_id" in rendered
    assert "cd8" in rendered
    assert "\t2\t0\t0\t0" in rendered


def test_eval_report_empty_when_no_targets() -> None:
    report = EvalReport()
    assert "no targets evaluated" in report.render_summary()

"""Evaluate a CalibrationTarget's observable function over a burn-in
trajectory DataFrame.

Pairs with ``qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long``
upstream (long-form ``[sample_index, t_to_diagnosis_days, column, value]``)
and a posterior-predictive plotter downstream (one panel per cal target,
trajectory band ending in observed value at t=0).

The observable code's normal signature is
``compute_observable(time, species_dict, constants, ureg) -> Pint
Quantity array of length len(time)``. This module compiles it once and
loops over ``sample_index`` groups in the trajectory frame, evaluating
the function on each sim's full time series.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


def evaluate_observable_over_trajectory(
    *,
    observable_code: str,
    constants: Mapping[str, Any],
    output_units: str,
    traj_df: pd.DataFrame,
    species_units: Mapping[str, str],
    ureg: Any,
    time_units: str = "day",
    on_error: str = "nan",
    auxiliary_per_sim: Optional[Mapping[int, Mapping[str, Any]]] = None,
) -> pd.DataFrame:
    """Evaluate ``compute_observable`` per sim, collecting time-resolved values.

    Args:
        observable_code: Python source defining
            ``compute_observable(time, species_dict, constants, ureg)`` —
            the same string that lives in
            ``CalibrationTarget.observable.code``. Must return a Pint
            Quantity array with ``len(time)`` entries.
        constants: Mapping of constant name → Pint Quantity. Built by the
            caller from ``CalibrationTarget.observable.constants``.
        output_units: Pint-parseable target units. The evaluated result
            is converted to these before its magnitude is recorded —
            errors loudly on dimensionality mismatch.
        traj_df: Long-form trajectory frame produced by
            ``qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long``,
            with columns ``[sample_index, t_to_diagnosis_days, column,
            value]``. Each ``column`` is a species or compartment name
            from the QSP model (e.g. ``"V_T.CD8"``, ``"V_T"``).
        species_units: Mapping of model column name → Pint unit string.
            Sourced from ``model_structure.json`` / ``species_units.json``
            on the pdac-build side. Columns absent from this map are
            wrapped as ``ureg.dimensionless`` — this is the right default
            for compartment volumes that already carry their unit
            elsewhere, but signals to the cal-target author that any
            species without an entry will silently flow through unitless.
        ureg: Pint UnitRegistry the cal target was authored against.
        time_units: Pint unit for ``t_to_diagnosis_days``. Default
            ``"day"`` matches the column name; override only if you
            relabel the trajectory time axis.
        on_error: ``"nan"`` (default) returns NaN for that sample on any
            evaluation failure (mirrors the derive_test_stats_worker
            behavior). ``"raise"`` propagates the underlying exception —
            useful for debugging a broken observable.
        auxiliary_per_sim: Optional mapping ``{sample_index: {name: Pint
            Quantity, ...}}`` of measurement-bridging auxiliary parameters
            (see :mod:`qsp_inference.auxiliary`). Each per-sim record is
            merged into ``constants`` before invoking the observable, so
            ``observable.code`` accesses auxiliary draws via the same
            ``constants`` dict. ``None`` (default) means no auxiliary
            parameters — preserves the prior behavior. A sample_index
            absent from the mapping evaluates with ``constants`` only
            (treats the auxiliary contribution as missing — caller's
            responsibility to provide draws for every sample they want
            evaluated). An auxiliary name colliding with a fixed-constant
            name raises ``ValueError`` (regardless of ``on_error``).

    Returns:
        Long-form DataFrame with columns
        ``[sample_index, t_to_diagnosis_days, value]``. ``value`` is the
        observable in ``output_units`` (magnitude only — caller owns the
        unit metadata). Rows are ordered by ``(sample_index,
        t_to_diagnosis_days)`` ascending.

    Raises:
        ValueError: ``observable_code`` doesn't define ``compute_observable``,
            or required columns are missing from ``traj_df``.
        Pint dimensionality / unit errors propagate when
            ``on_error="raise"``.
    """
    if on_error not in ("nan", "raise"):
        raise ValueError(f"on_error must be 'nan' or 'raise'; got {on_error!r}")

    required_cols = {"sample_index", "t_to_diagnosis_days", "column", "value"}
    missing = required_cols - set(traj_df.columns)
    if missing:
        raise ValueError(f"traj_df missing required columns: {sorted(missing)}")

    # Compile the cal target's observable code once. Loose namespace
    # mirrors maple's validator + qsp_hpc derive worker.
    namespace: dict[str, Any] = {"np": np, "numpy": np, "ureg": ureg}
    exec(observable_code, namespace)
    compute_fn = namespace.get("compute_observable")
    if compute_fn is None:
        raise ValueError(
            "observable_code did not define a callable named "
            "'compute_observable'"
        )

    time_q_unit = ureg(time_units)
    out_rows: list[pd.DataFrame] = []
    for sample_idx, sim_df in traj_df.groupby("sample_index", sort=True):
        # Pivot the long-form rows to a (n_t × n_columns) wide table so
        # we can hand each column array to the observable as a Pint
        # Quantity. ``aggfunc="first"`` is safe — there is exactly one
        # row per (t, column) by construction in the upstream assembler.
        wide = (
            sim_df.pivot_table(
                index="t_to_diagnosis_days",
                columns="column",
                values="value",
                aggfunc="first",
            )
            .sort_index()
        )
        time_q = wide.index.to_numpy(dtype=np.float64) * time_q_unit

        species_dict: dict[str, Any] = {}
        for col in wide.columns:
            arr = wide[col].to_numpy(dtype=np.float64)
            unit = species_units.get(col)
            if unit is None:
                species_dict[col] = arr * ureg.dimensionless
            else:
                species_dict[col] = arr * ureg(unit)

        if auxiliary_per_sim is not None:
            aux_record = auxiliary_per_sim.get(int(sample_idx), {})
            collisions = set(aux_record).intersection(constants)
            if collisions:
                raise ValueError(
                    f"Auxiliary parameter name(s) collide with observable.constants: "
                    f"{sorted(collisions)}. Rename the AuxiliaryParameter or the "
                    f"ObservableConstant so the observable.code namespace is unambiguous."
                )
            sim_constants: dict[str, Any] = {**constants, **aux_record}
        else:
            sim_constants = dict(constants)

        try:
            result = compute_fn(time_q, species_dict, sim_constants, ureg)
            if not hasattr(result, "to"):
                raise TypeError(
                    "compute_observable returned a non-Pint object; "
                    "expected a Quantity array"
                )
            magnitude = np.asarray(result.to(output_units).magnitude, dtype=np.float64)
            if magnitude.shape != time_q.magnitude.shape:
                raise ValueError(
                    f"compute_observable returned array of shape "
                    f"{magnitude.shape}, expected {time_q.magnitude.shape} "
                    "(same length as time axis)"
                )
        except Exception:
            if on_error == "raise":
                raise
            magnitude = np.full(len(wide.index), np.nan, dtype=np.float64)

        out_rows.append(
            pd.DataFrame(
                {
                    "sample_index": np.full(
                        len(wide.index), int(sample_idx), dtype=np.int64
                    ),
                    "t_to_diagnosis_days": wide.index.to_numpy(dtype=np.float64),
                    "value": magnitude,
                }
            )
        )

    if not out_rows:
        return pd.DataFrame(
            columns=["sample_index", "t_to_diagnosis_days", "value"]
        )
    out = pd.concat(out_rows, ignore_index=True)
    return out.sort_values(["sample_index", "t_to_diagnosis_days"]).reset_index(
        drop=True
    )


def evaluate_calibration_target_over_trajectory(
    target: Any,
    traj_df: pd.DataFrame,
    species_units: Mapping[str, str],
    ureg: Any,
    *,
    time_units: str = "day",
    on_error: str = "nan",
    auxiliary_records: Optional[Iterable[Mapping[str, float]]] = None,
) -> pd.DataFrame:
    """Sugar wrapper that pulls observable code/constants/units off a
    ``maple.CalibrationTarget`` (or duck-typed equivalent).

    The pure dependency on the underlying
    :func:`evaluate_observable_over_trajectory` keeps this module free
    of a hard maple import — pass any object exposing
    ``target.observable.code`` / ``.units`` / ``.constants[i].name`` /
    ``.constants[i].value`` / ``.constants[i].units`` (and optionally
    ``.auxiliary_parameters[i].name`` / ``.units`` for the auxiliary
    plumbing below).

    ``auxiliary_records``, when provided, is a sequence of per-sim
    ``{name: float}`` dicts (one per ``sample_index`` in ascending order
    — i.e. the ordering produced by
    :meth:`qsp_inference.auxiliary.HierarchicalAuxiliaryPrior.sample_as_records`).
    The wrapper filters each record to only the auxiliary parameters
    declared on this calibration target's observable, attaches the
    declared Pint units, and threads them through as
    ``auxiliary_per_sim``. A target with no
    ``observable.auxiliary_parameters`` simply ignores the records.
    """
    obs = target.observable
    constants = {c.name: c.value * ureg(c.units) for c in obs.constants}

    aux_per_sim: Optional[dict[int, dict[str, Any]]] = None
    declared_aux = getattr(obs, "auxiliary_parameters", None) or []
    if auxiliary_records is not None and declared_aux:
        aux_units = {a.name: getattr(a, "units", "dimensionless") for a in declared_aux}
        sample_indices = sorted(
            {int(s) for s in traj_df["sample_index"].unique()}
        )
        records_list = list(auxiliary_records)
        if len(records_list) != len(sample_indices):
            raise ValueError(
                f"auxiliary_records length {len(records_list)} does not match "
                f"the number of sample indices in traj_df ({len(sample_indices)}). "
                f"Provide one record per sim, in ascending sample_index order."
            )
        aux_per_sim = {}
        for sample_idx, record in zip(sample_indices, records_list):
            filtered: dict[str, Any] = {}
            for name, units in aux_units.items():
                if name not in record:
                    raise ValueError(
                        f"auxiliary_records[{sample_idx}] is missing "
                        f"declared auxiliary parameter '{name}' for this "
                        f"calibration target."
                    )
                filtered[name] = float(record[name]) * ureg(units)
            aux_per_sim[sample_idx] = filtered

    return evaluate_observable_over_trajectory(
        observable_code=obs.code,
        constants=constants,
        output_units=obs.units,
        traj_df=traj_df,
        species_units=species_units,
        ureg=ureg,
        time_units=time_units,
        on_error=on_error,
        auxiliary_per_sim=aux_per_sim,
    )


__all__ = [
    "evaluate_observable_over_trajectory",
    "evaluate_calibration_target_over_trajectory",
]

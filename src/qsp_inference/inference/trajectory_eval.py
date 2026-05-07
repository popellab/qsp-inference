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

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

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


def _target_id(target: Any) -> str:
    """Best-effort stable identifier for a CalibrationTarget-shaped object.

    Prefers ``calibration_target_id`` (set during maple unpacking), falls
    back to ``name`` / ``target_id`` / ``id``, then to a sentinel. Used
    only for column naming and EvalReport keys — collisions just merge
    counters under the duplicate name, which is the right behavior for
    a misconfigured cal-target dir anyway.
    """
    for attr in ("calibration_target_id", "target_id", "name", "id"):
        val = getattr(target, attr, None)
        if val:
            return str(val)
    return "<unnamed-target>"


def _resolve_index_values(
    target: Any, override: Optional[Sequence[float]]
) -> list[float]:
    """Pick the time points to interpolate to for one target.

    Override wins if non-empty, else ``target.empirical_data.index_values``,
    else ``[0.0]`` (the pdac-build default for index-less scalar targets,
    per the CLAUDE.md gotcha that an unset index defaults to t=0).
    """
    if override:
        return [float(v) for v in override]
    empirical = getattr(target, "empirical_data", None)
    declared = getattr(empirical, "index_values", None) if empirical else None
    if declared:
        return [float(v) for v in declared]
    return [0.0]


@dataclass
class EvalReport:
    """Per-target failure-mode counters for ``evaluate_targets_to_x``.

    Attributes:
        per_target: ``{target_id: Counter}`` with keys ``n_total``,
            ``n_nan``, ``n_code_raised``, ``n_species_missing``. ``n_nan``
            counts sims whose interpolated value at any requested
            ``index_values`` came out NaN regardless of cause; the other
            counters break that down. Sum can exceed ``n_nan`` because a
            single sim can hit multiple buckets in principle, but in
            practice they're disjoint at evaluation time (we stop at
            the first failure mode for that sim/target).
        last_exception: ``{target_id: Exception | None}`` — keeps the
            most recent exception per target for log triage.
    """

    per_target: dict[str, Counter] = field(default_factory=dict)
    last_exception: dict[str, Optional[Exception]] = field(default_factory=dict)

    def _bucket(self, target_id: str) -> Counter:
        if target_id not in self.per_target:
            self.per_target[target_id] = Counter()
            self.last_exception[target_id] = None
        return self.per_target[target_id]

    def render_summary(self) -> str:
        """Human-readable table for log surfaces (sbi_runner step 7b)."""
        if not self.per_target:
            return "EvalReport: no targets evaluated."
        rows = ["target_id\ttotal\tnan\tcode_raised\tspecies_missing"]
        for tid in sorted(self.per_target):
            c = self.per_target[tid]
            rows.append(
                f"{tid}\t{c['n_total']}\t{c['n_nan']}"
                f"\t{c['n_code_raised']}\t{c['n_species_missing']}"
            )
        return "\n".join(rows)


def collect_required_trajectory_columns(
    calibration_targets: Sequence[Any],
) -> list[str]:
    """Union of ``observable.species`` across all cal targets — the
    minimal set of trajectory columns needed to evaluate every target.

    Pass this to a trajectory loader (or as the ``traj_columns`` arg to
    ``CppSimulator.run_hpc`` once the qsp-hpc-tools side lands) when the
    upstream binaries dump every species but the cal targets only
    reference a few. Promoted from pdac-build's
    ``workflows/calibration_trajectory_pipeline.py`` so downstream
    callers don't need a pdac-build dependency.
    """
    cols: set[str] = set()
    for target in calibration_targets:
        obs = getattr(target, "observable", None)
        if obs is None:
            continue
        for col in getattr(obs, "species", []) or []:
            cols.add(col)
    return sorted(cols)


def evaluate_targets_to_x(
    targets: Sequence[Any],
    traj_df: pd.DataFrame,
    species_units: Mapping[str, str],
    ureg: Any,
    *,
    auxiliary_records: Optional[Iterable[Mapping[str, float]]] = None,
    target_index_values: Optional[Mapping[str, Sequence[float]]] = None,
    time_units: str = "day",
) -> tuple[np.ndarray, list[str], EvalReport]:
    """Evaluate ``targets`` over ``traj_df`` and aggregate to a feature
    matrix suitable for SBI training.

    For each target:

    1. Calls :func:`evaluate_calibration_target_over_trajectory` on the
       full trajectory frame to get a per-(sample_index, time) Pint
       observable series, with ``on_error="raise"`` so failure modes
       can be classified per-sim rather than silently NaN-filled.
    2. Linearly interpolates each sim's series to the target's
       requested ``index_values`` (overridden by
       ``target_index_values[target_id]`` if present; otherwise pulled
       from ``target.empirical_data.index_values``; default ``[0.0]``).
    3. Emits one column per index value, named
       ``"{target_id}__t{idx_value}"`` for vector targets, or
       ``"{target_id}"`` for scalar (single-index) targets — keeps SBI
       column names readable when index_values has length 1.

    Failure-mode classification (per-sim, per-target):

    - ``KeyError`` from a missing species in ``species_dict`` →
      ``n_species_missing``.
    - Any other ``Exception`` from ``code`` or unit conversion →
      ``n_code_raised`` and the exception is stored in
      ``EvalReport.last_exception[target_id]``.
    - Either case yields NaN for that sim/target's columns.
    - ``n_nan`` independently counts sims whose interpolated value
      came out NaN, which also catches sim-side NaN propagation
      through a successful ``code`` call (the third failure mode in
      D7's three-way split).

    Args:
        targets: Sequence of CalibrationTarget-shaped objects (anything
            duck-typed against the observable / empirical_data fields
            consumed by :func:`evaluate_calibration_target_over_trajectory`).
        traj_df: Long-form trajectory frame. Same schema as the
            existing helpers: ``[sample_index, t_to_diagnosis_days,
            column, value]``. ``sample_index`` ascending order
            determines the row order of the output matrix.
        species_units: Per-species canonical unit string. Passed
            through to the underlying helper.
        ureg: Pint UnitRegistry.
        auxiliary_records: Optional sequence of per-sim auxiliary
            records (one ``{name: float}`` dict per ``sample_index``,
            in ascending order). Filtered per-target inside.
        target_index_values: Optional ``{target_id: [t1, t2, ...]}``
            override. Falls back to each target's YAML-declared
            ``empirical_data.index_values`` (or ``[0.0]`` if unset).
        time_units: Pint unit for the trajectory's time axis. Default
            ``"day"`` matches the standard column name.

    Returns:
        ``(x, names, report)``:

        - ``x``: ``(n_sims, n_columns)`` float64 ndarray. ``n_sims`` is
          the count of distinct sample_indices in ``traj_df`` (sorted
          ascending). ``n_columns`` is the sum of ``len(index_values)``
          across targets.
        - ``names``: column names matching ``x``'s column axis.
        - ``report``: :class:`EvalReport` with per-target counters.
    """
    if not targets:
        sample_indices = sorted({int(s) for s in traj_df["sample_index"].unique()})
        return (
            np.zeros((len(sample_indices), 0), dtype=np.float64),
            [],
            EvalReport(),
        )

    sample_indices = sorted({int(s) for s in traj_df["sample_index"].unique()})
    n_sims = len(sample_indices)
    sample_idx_to_row = {sid: row for row, sid in enumerate(sample_indices)}

    aux_records_list: Optional[list[Mapping[str, float]]] = (
        list(auxiliary_records) if auxiliary_records is not None else None
    )
    if aux_records_list is not None and len(aux_records_list) != n_sims:
        raise ValueError(
            f"auxiliary_records length {len(aux_records_list)} does not match "
            f"the number of sample indices in traj_df ({n_sims}). Provide one "
            f"record per sim, in ascending sample_index order."
        )

    overrides = target_index_values or {}
    report = EvalReport()
    columns: list[np.ndarray] = []
    names: list[str] = []

    for target in targets:
        tid = _target_id(target)
        bucket = report._bucket(tid)
        idx_values = _resolve_index_values(target, overrides.get(tid))

        column_block = np.full((n_sims, len(idx_values)), np.nan, dtype=np.float64)

        # Run the existing per-target helper once on the full frame,
        # with on_error="raise" so we can classify failure modes per
        # sim. The helper itself loops over sample_indices internally,
        # so we re-call it per-sim to scope the try/except. Per-sim
        # invocation pays the same pivot cost the bulk path would have
        # paid, in a tighter loop — acceptable for unit-test scale and
        # for production where N_targets ≪ N_sims.
        for sample_idx in sample_indices:
            sim_traj = traj_df[traj_df["sample_index"] == sample_idx]
            row = sample_idx_to_row[sample_idx]
            bucket["n_total"] += 1

            sim_aux: Optional[list[Mapping[str, float]]] = None
            if aux_records_list is not None:
                sim_aux = [aux_records_list[row]]

            try:
                result_df = evaluate_calibration_target_over_trajectory(
                    target,
                    sim_traj,
                    species_units,
                    ureg,
                    time_units=time_units,
                    on_error="raise",
                    auxiliary_records=sim_aux,
                )
            except KeyError as e:
                bucket["n_species_missing"] += 1
                report.last_exception[tid] = e
                bucket["n_nan"] += 1
                continue
            except Exception as e:  # noqa: BLE001 — classifying everything else
                bucket["n_code_raised"] += 1
                report.last_exception[tid] = e
                bucket["n_nan"] += 1
                continue

            t = result_df["t_to_diagnosis_days"].to_numpy(dtype=np.float64)
            v = result_df["value"].to_numpy(dtype=np.float64)
            order = np.argsort(t)
            t, v = t[order], v[order]

            interp = np.interp(np.asarray(idx_values, dtype=np.float64), t, v)
            column_block[row, :] = interp

            if np.any(np.isnan(interp)):
                bucket["n_nan"] += 1

        # Stash columns + names. Single-index targets keep the bare
        # target_id; vector targets append a ``__t{val}`` suffix so
        # downstream code can recover the per-time pairing.
        if len(idx_values) == 1:
            columns.append(column_block[:, 0])
            names.append(tid)
        else:
            for k, idx in enumerate(idx_values):
                columns.append(column_block[:, k])
                names.append(f"{tid}__t{idx}")

    if columns:
        x = np.column_stack(columns)
    else:
        x = np.zeros((n_sims, 0), dtype=np.float64)
    return x, names, report


__all__ = [
    "EvalReport",
    "collect_required_trajectory_columns",
    "evaluate_observable_over_trajectory",
    "evaluate_calibration_target_over_trajectory",
    "evaluate_targets_to_x",
]

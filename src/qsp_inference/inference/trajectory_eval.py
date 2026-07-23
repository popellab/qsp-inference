"""Evaluate a CalibrationTarget's observable function over a burn-in
trajectory DataFrame.

Pairs with ``qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long``
upstream (long-form ``[sample_index, t_to_diagnosis_days, column, value]``)
and a posterior-predictive plotter downstream (one panel per cal target,
trajectory band ending in observed value at t=0).

The observable code's signature is
``compute_observable(time, species_dict, constants) -> ndarray of length
len(time)``. Inputs and outputs are raw floats in canonical model units
— Pint was dropped from the calibration-target observable interface in
the pintless rollout (PR1 maple, PR2 qsp-hpc-tools, PR3 pdac-build,
PR4 here).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def pivot_traj_df_per_sim(
    traj_df: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    """Pre-pivot a long-form trajectory frame into ``{sample_index: wide_df}``.

    The wide frame is indexed by ``t_to_diagnosis_days`` (sorted) with one
    column per QSP column (species/compartment/rule). Hoisting this out of
    :func:`evaluate_observable_over_trajectory` lets a caller evaluating many
    cal targets on the same trajectory pay the pivot cost once instead of
    N_targets times.
    """
    required_cols = {"sample_index", "t_to_diagnosis_days", "column", "value"}
    missing = required_cols - set(traj_df.columns)
    if missing:
        raise ValueError(f"traj_df missing required columns: {sorted(missing)}")

    out: dict[int, pd.DataFrame] = {}
    for sample_idx, sim_df in traj_df.groupby("sample_index", sort=True):
        wide = (
            sim_df.pivot_table(
                index="t_to_diagnosis_days",
                columns="column",
                values="value",
                aggfunc="first",
            )
            .sort_index()
        )
        out[int(sample_idx)] = wide
    return out


def evaluate_observable_over_trajectory(
    *,
    observable_code: str,
    constants: Mapping[str, float],
    traj_df: Optional[pd.DataFrame] = None,
    wide_per_sim: Optional[Mapping[int, pd.DataFrame]] = None,
    on_error: str = "nan",
    auxiliary_per_sim: Optional[Mapping[int, Mapping[str, float]]] = None,
    parameter_per_sim: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> pd.DataFrame:
    """Evaluate ``compute_observable`` per sim, collecting time-resolved values.

    Args:
        observable_code: Python source defining
            ``compute_observable(time, species_dict, constants)`` — the
            same string that lives in ``CalibrationTarget.observable.code``.
            Must return a float ndarray with ``len(time)`` entries, in
            the target's declared output units.
        constants: Mapping of constant name → raw float (already in the
            units the observable expects). Built by the caller from
            ``CalibrationTarget.observable.constants``.
        traj_df: Long-form trajectory frame produced by
            ``qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long``,
            with columns ``[sample_index, t_to_diagnosis_days, column,
            value]``. Each ``column`` is a species or compartment name
            from the QSP model (e.g. ``"V_T.CD8"``, ``"V_T"``); values
            are raw floats in canonical model units.
        on_error: ``"nan"`` (default) returns NaN for that sample on any
            evaluation failure (mirrors the derive_test_stats_worker
            behavior). ``"raise"`` propagates the underlying exception —
            useful for debugging a broken observable.
        auxiliary_per_sim: Optional mapping ``{sample_index: {name:
            float, ...}}`` of measurement-bridging auxiliary parameters
            (see :mod:`qsp_inference.auxiliary`). Each per-sim record is
            merged into ``constants`` before invoking the observable, so
            ``observable.code`` accesses auxiliary draws via the same
            ``constants`` dict. ``None`` (default) means no auxiliary
            parameters. An auxiliary name colliding with a fixed-constant
            name raises ``ValueError`` (regardless of ``on_error``).
        parameter_per_sim: Optional mapping ``{sample_index: {name:
            float, ...}}`` of per-sim QSP parameter values to thread
            into ``species_dict``. Used when a cal target declares a
            varied prior parameter (e.g. ``rho_collagen``) under
            ``observable.species`` — the long-form trajectory parquet
            doesn't carry parameter columns, so the values must come from
            the caller's theta matrix. Each value is broadcast to the
            sim's time axis (``v * np.ones(len(time))``) so cal-target
            code that indexes/slices behaves identically to a real
            species. A parameter name colliding with a real species
            column or a constant name raises ``ValueError``.

    Returns:
        Long-form DataFrame with columns
        ``[sample_index, t_to_diagnosis_days, value]``. ``value`` is the
        raw float observable in the target's declared output units. Rows
        are ordered by ``(sample_index, t_to_diagnosis_days)`` ascending.

    Raises:
        ValueError: ``observable_code`` doesn't define ``compute_observable``,
            or required columns are missing from ``traj_df``.
        Exceptions from ``compute_observable`` propagate when
            ``on_error="raise"``.
    """
    if on_error not in ("nan", "raise"):
        raise ValueError(f"on_error must be 'nan' or 'raise'; got {on_error!r}")

    if wide_per_sim is None:
        if traj_df is None:
            raise ValueError("provide either traj_df or wide_per_sim")
        wide_per_sim = pivot_traj_df_per_sim(traj_df)
    elif traj_df is not None:
        raise ValueError("pass exactly one of traj_df / wide_per_sim, not both")

    # Compile the cal target's observable code once. Loose namespace
    # mirrors maple's validator + qsp_hpc derive worker.
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    exec(observable_code, namespace)
    compute_fn = namespace.get("compute_observable")
    if compute_fn is None:
        raise ValueError(
            "observable_code did not define a callable named "
            "'compute_observable'"
        )

    out_rows: list[pd.DataFrame] = []
    for sample_idx in sorted(wide_per_sim):
        wide = wide_per_sim[sample_idx]
        time = wide.index.to_numpy(dtype=np.float64)

        species_dict: dict[str, np.ndarray] = {
            col: wide[col].to_numpy(dtype=np.float64) for col in wide.columns
        }

        if auxiliary_per_sim is not None:
            aux_record = auxiliary_per_sim.get(int(sample_idx), {})
            collisions = set(aux_record).intersection(constants)
            if collisions:
                raise ValueError(
                    f"Auxiliary parameter name(s) collide with observable.constants: "
                    f"{sorted(collisions)}. Rename the AuxiliaryParameter or the "
                    f"ObservableConstant so the observable.code namespace is unambiguous."
                )
            sim_constants: dict[str, float] = {**constants, **aux_record}
        else:
            sim_constants = dict(constants)

        if parameter_per_sim is not None:
            param_record = parameter_per_sim.get(int(sample_idx), {})
            param_collisions_species = set(param_record).intersection(species_dict)
            if param_collisions_species:
                raise ValueError(
                    f"Parameter name(s) collide with real species in traj_df: "
                    f"{sorted(param_collisions_species)}. Drop the entry from "
                    f"parameter_per_sim or rename the QSP parameter so the "
                    f"observable.code species_dict namespace is unambiguous."
                )
            param_collisions_const = set(param_record).intersection(sim_constants)
            if param_collisions_const:
                raise ValueError(
                    f"Parameter name(s) collide with observable.constants / "
                    f"auxiliary records: {sorted(param_collisions_const)}. Rename "
                    f"the ObservableConstant or the QSP parameter."
                )
            ones = np.ones(len(wide.index), dtype=np.float64)
            for pname, pval in param_record.items():
                species_dict[pname] = ones * float(pval)

        try:
            result = compute_fn(time, species_dict, sim_constants)
            value = np.asarray(result, dtype=np.float64)
            if value.shape != time.shape:
                raise ValueError(
                    f"compute_observable returned array of shape "
                    f"{value.shape}, expected {time.shape} "
                    "(same length as time axis)"
                )
        except Exception:
            if on_error == "raise":
                raise
            value = np.full(len(wide.index), np.nan, dtype=np.float64)

        out_rows.append(
            pd.DataFrame(
                {
                    "sample_index": np.full(
                        len(wide.index), int(sample_idx), dtype=np.int64
                    ),
                    "t_to_diagnosis_days": wide.index.to_numpy(dtype=np.float64),
                    "value": value,
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
    traj_df: Optional[pd.DataFrame] = None,
    *,
    wide_per_sim: Optional[Mapping[int, pd.DataFrame]] = None,
    on_error: str = "nan",
    auxiliary_records: Optional[Iterable[Mapping[str, float]]] = None,
    parameter_records: Optional[Iterable[Mapping[str, float]]] = None,
) -> pd.DataFrame:
    """Sugar wrapper that pulls observable code/constants off a
    ``maple.CalibrationTarget`` (or duck-typed equivalent).

    The pure dependency on the underlying
    :func:`evaluate_observable_over_trajectory` keeps this module free
    of a hard maple import — pass any object exposing
    ``target.observable.code`` / ``.constants[i].name`` /
    ``.constants[i].value`` (and optionally
    ``.auxiliary_parameters[i].name`` for the auxiliary plumbing below).

    ``auxiliary_records``, when provided, is a sequence of per-sim
    ``{name: float}`` dicts (one per ``sample_index`` in ascending order
    — i.e. the ordering produced by
    :meth:`qsp_inference.auxiliary.HierarchicalAuxiliaryPrior.sample_as_records`).
    The wrapper filters each record to only the auxiliary parameters
    declared on this calibration target's observable and threads them
    through as ``auxiliary_per_sim``. A target with no
    ``observable.auxiliary_parameters`` simply ignores the records.
    """
    obs = target.observable
    constants = {c.name: float(c.value) for c in obs.constants}

    if wide_per_sim is None and traj_df is None:
        raise ValueError("provide either traj_df or wide_per_sim")
    if wide_per_sim is not None and traj_df is not None:
        raise ValueError("pass exactly one of traj_df / wide_per_sim, not both")

    def _sample_indices() -> list[int]:
        if wide_per_sim is not None:
            return sorted(int(k) for k in wide_per_sim)
        return sorted({int(s) for s in traj_df["sample_index"].unique()})

    aux_per_sim: Optional[dict[int, dict[str, float]]] = None
    declared_aux = getattr(obs, "auxiliary_parameters", None) or []
    if auxiliary_records is not None and declared_aux:
        aux_names = [a.name for a in declared_aux]
        sample_indices = _sample_indices()
        records_list = list(auxiliary_records)
        if len(records_list) != len(sample_indices):
            raise ValueError(
                f"auxiliary_records length {len(records_list)} does not match "
                f"the number of sample indices in traj_df ({len(sample_indices)}). "
                f"Provide one record per sim, in ascending sample_index order."
            )
        aux_per_sim = {}
        for sample_idx, record in zip(sample_indices, records_list):
            filtered: dict[str, float] = {}
            for name in aux_names:
                if name not in record:
                    raise ValueError(
                        f"auxiliary_records[{sample_idx}] is missing "
                        f"declared auxiliary parameter '{name}' for this "
                        f"calibration target."
                    )
                filtered[name] = float(record[name])
            aux_per_sim[sample_idx] = filtered

    # Per-sim parameter threading: cal targets that declare a varied
    # prior parameter under observable.species (e.g. rho_collagen) need
    # the parameter's per-sim value injected into species_dict because
    # the long-form trajectory parquet doesn't carry parameter columns.
    # Filter parameter_records per-target by the species list so
    # unrelated params don't pollute the observable's namespace.
    param_per_sim: Optional[dict[int, dict[str, float]]] = None
    declared_species = list(getattr(obs, "species", None) or [])
    if parameter_records is not None and declared_species:
        sample_indices_p = _sample_indices()
        param_records_list = list(parameter_records)
        if len(param_records_list) != len(sample_indices_p):
            raise ValueError(
                f"parameter_records length {len(param_records_list)} does not "
                f"match the number of sample indices in traj_df "
                f"({len(sample_indices_p)}). Provide one record per sim, in "
                f"ascending sample_index order."
            )
        # Subset to params declared as species on this target AND present
        # in the records; missing ones flow through (the underlying eval
        # will surface KeyError naturally when observable.code accesses
        # them).
        relevant = [s for s in declared_species if s in (param_records_list[0] if param_records_list else {})]
        if relevant:
            param_per_sim = {}
            for sid, rec in zip(sample_indices_p, param_records_list):
                filtered_p: dict[str, float] = {}
                for pname in relevant:
                    if pname not in rec:
                        raise ValueError(
                            f"parameter_records[{sid}] missing declared "
                            f"species/parameter '{pname}'."
                        )
                    filtered_p[pname] = float(rec[pname])
                param_per_sim[sid] = filtered_p

    return evaluate_observable_over_trajectory(
        observable_code=obs.code,
        constants=constants,
        traj_df=traj_df,
        wide_per_sim=wide_per_sim,
        on_error=on_error,
        auxiliary_per_sim=aux_per_sim,
        parameter_per_sim=param_per_sim,
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


def _readout_time_days(target: Any) -> Optional[float]:
    """Canonical-day readout time for a target, or ``None`` if the
    observable reduces via ``reduce_observable`` instead.

    ``observable.readout_time`` is expressed in ``observable.readout_time_unit``
    (a Pint unit, e.g. ``"day"``); the trajectory time axis is in canonical
    days, so we convert with the shared maple registry. The import is local
    so this module keeps no hard maple dependency for the pure-eval path.
    """
    obs = target.observable
    readout_time = getattr(obs, "readout_time", None)
    if readout_time is None:
        return None
    unit = getattr(obs, "readout_time_unit", None)
    if not unit:
        raise ValueError(
            "observable.readout_time is set but observable.readout_time_unit "
            "is missing; cannot convert the readout time to canonical days."
        )
    from maple.core.unit_registry import ureg

    return float((float(readout_time) * ureg(unit)).to("day").magnitude)


def _compile_reduce_observable(code: str):
    """Compile a ``reduce_observable(time, series) -> float`` from source."""
    namespace: dict[str, Any] = {"np": np, "numpy": np}
    exec(code, namespace)
    fn = namespace.get("reduce_observable")
    if fn is None:
        raise ValueError(
            "observable.reduce_observable source did not define a callable "
            "named 'reduce_observable'"
        )
    return fn


def _make_reducer(target: Any, override_time: Optional[float]):
    """Build the series → scalar reducer for one target.

    The maple ``Observable`` contract sets exactly one of ``readout_time``
    (evaluate the series at a fixed sim time) or ``reduce_observable``
    (a ``reduce_observable(time, series) -> float`` function for peak / AUC /
    final-value style measurements). ``override_time`` (canonical days), when
    provided, forces a value-at-a-time readout regardless of which the
    observable declares — the only override that still makes sense now that
    the vector index dimension is gone.

    Returns a callable ``reduce(time, series) -> float``. Resolution of the
    readout time / compilation of ``reduce_observable`` happens once here, not
    per-sim. Raises ``ValueError`` if the observable sets neither or both.
    """
    if override_time is not None:
        forced = float(override_time)

        def _reduce_override(t: np.ndarray, v: np.ndarray) -> float:
            return float(np.interp(forced, t, v))

        return _reduce_override

    obs = target.observable
    readout_days = _readout_time_days(target)
    reduce_src = getattr(obs, "reduce_observable", None)

    if readout_days is not None and reduce_src is not None:
        raise ValueError(
            "observable sets both readout_time and reduce_observable; the "
            "maple contract requires exactly one."
        )
    if readout_days is not None:

        def _reduce_at_time(t: np.ndarray, v: np.ndarray) -> float:
            return float(np.interp(readout_days, t, v))

        return _reduce_at_time
    if reduce_src is not None:
        fn = _compile_reduce_observable(reduce_src)

        def _reduce_fn(t: np.ndarray, v: np.ndarray) -> float:
            return float(fn(t, v))

        return _reduce_fn

    raise ValueError(
        "observable sets neither readout_time nor reduce_observable; the "
        "maple contract requires exactly one."
    )


@dataclass
class EvalReport:
    """Per-target failure-mode counters for ``evaluate_targets_to_x``.

    Attributes:
        per_target: ``{target_id: Counter}`` with keys ``n_total``,
            ``n_nan``, ``n_code_raised``, ``n_species_missing``. ``n_nan``
            counts sims whose reduced scalar value came out NaN regardless
            of cause; the other
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
    *,
    auxiliary_records: Optional[Iterable[Mapping[str, float]]] = None,
    parameter_records: Optional[Iterable[Mapping[str, float]]] = None,
    target_readout_times: Optional[Mapping[str, float]] = None,
) -> tuple[np.ndarray, list[str], EvalReport]:
    """Evaluate ``targets`` over ``traj_df`` and aggregate to a feature
    matrix suitable for SBI training.

    For each target:

    1. Calls :func:`evaluate_calibration_target_over_trajectory` on the
       full trajectory frame to get a per-(sample_index, time)
       observable series, with ``on_error="raise"`` so failure modes
       can be classified per-sim rather than silently NaN-filled.
    2. Reduces each sim's series to a single scalar per the target's
       ``observable`` reduction contract: interpolating at
       ``observable.readout_time`` (converted to canonical days via the
       shared maple Pint registry) when set, or applying the
       ``observable.reduce_observable(time, series) -> float`` function
       when set. Exactly one is set. A per-target entry in
       ``target_readout_times`` overrides this with a forced
       value-at-a-time readout (canonical days).
    3. Emits exactly one column per target, named ``"{target_id}"`` —
       the time-series → scalar reduction now lives on the observable
       (maple removed the vector index dimension), so every target
       contributes a single feature column.

    Failure-mode classification (per-sim, per-target):

    - ``KeyError`` from a missing species in ``species_dict`` →
      ``n_species_missing``.
    - Any other ``Exception`` from ``code`` (or from the reduction) →
      ``n_code_raised`` and the exception is stored in
      ``EvalReport.last_exception[target_id]``.
    - Either case yields NaN for that sim/target's column.
    - ``n_nan`` independently counts sims whose reduced scalar value
      came out NaN, which also catches sim-side NaN propagation
      through a successful ``code`` call (the third failure mode in
      D7's three-way split).

    Args:
        targets: Sequence of CalibrationTarget-shaped objects (anything
            duck-typed against the observable fields consumed by
            :func:`evaluate_calibration_target_over_trajectory` plus the
            ``observable.readout_time`` / ``observable.reduce_observable``
            reduction contract).
        traj_df: Long-form trajectory frame. Same schema as the
            existing helpers: ``[sample_index, t_to_diagnosis_days,
            column, value]``. ``sample_index`` ascending order
            determines the row order of the output matrix.
        auxiliary_records: Optional sequence of per-sim auxiliary
            records (one ``{name: float}`` dict per ``sample_index``,
            in ascending order). Filtered per-target inside.
        parameter_records: Optional sequence of per-sim QSP parameter
            records (one ``{name: float}`` dict per ``sample_index``,
            in ascending order). Filtered per-target by the target's
            declared ``observable.species`` list inside.
        target_readout_times: Optional ``{target_id: t_days}`` override.
            Forces a value-at-a-time readout at ``t_days`` (canonical
            days) for that target, superseding whichever reduction its
            observable declares. Omit to use the observable's own
            ``readout_time`` / ``reduce_observable``.

    Returns:
        ``(x, names, report)``:

        - ``x``: ``(n_sims, n_targets)`` float64 ndarray. ``n_sims`` is
          the count of distinct sample_indices in ``traj_df`` (sorted
          ascending). One column per target (in input order).
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

    param_records_list: Optional[list[Mapping[str, float]]] = (
        list(parameter_records) if parameter_records is not None else None
    )
    if param_records_list is not None and len(param_records_list) != n_sims:
        raise ValueError(
            f"parameter_records length {len(param_records_list)} does not match "
            f"the number of sample indices in traj_df ({n_sims}). Provide one "
            f"record per sim, in ascending sample_index order."
        )

    overrides = target_readout_times or {}
    report = EvalReport()
    columns: list[np.ndarray] = []
    names: list[str] = []

    for target in targets:
        tid = _target_id(target)
        bucket = report._bucket(tid)
        # Resolve the readout time / compile reduce_observable once per
        # target rather than once per sim.
        reducer = _make_reducer(target, overrides.get(tid))

        column = np.full(n_sims, np.nan, dtype=np.float64)

        # Run the existing per-target helper once per sim so the
        # try/except can scope per-sim failure-mode classification. Per-sim
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
            sim_param: Optional[list[Mapping[str, float]]] = None
            if param_records_list is not None:
                sim_param = [param_records_list[row]]

            try:
                result_df = evaluate_calibration_target_over_trajectory(
                    target,
                    sim_traj,
                    on_error="raise",
                    auxiliary_records=sim_aux,
                    parameter_records=sim_param,
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

            try:
                scalar = reducer(t, v)
            except Exception as e:  # noqa: BLE001 — reduce_observable is user code
                bucket["n_code_raised"] += 1
                report.last_exception[tid] = e
                bucket["n_nan"] += 1
                continue

            column[row] = scalar
            if np.isnan(scalar):
                bucket["n_nan"] += 1

        # One scalar column per target — the observable owns the
        # time-series → scalar reduction, so there is no vector index
        # suffix to emit anymore.
        columns.append(column)
        names.append(tid)

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
    "pivot_traj_df_per_sim",
]

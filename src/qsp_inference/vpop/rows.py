"""The rows a corpus contributes: one per statistic a source printed.

Two evaluators, and they are not the same function. :func:`hard_row` is what the
source's own software computed from its ``n_c`` patients, and resampling it builds
``V^boot``. :func:`tau_row` is the model's prediction of that number, which is its
expectation over an ``n_c``-sample and is smooth in ``phi``.

Scale rows are the schema's ``WIDTH_STATS | SAMPLING_WIDTH_STATS``. The draft's
location half is their complement, which is not the schema's ``LOCATION_STATS``:
that one excludes quantiles because it answers a different question.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from maple.core.calibration.shared_models import SAMPLING_WIDTH_STATS, WIDTH_STATS

__all__ = ["SCALE_STATS", "SUPPORTED_STATS", "NUMPY_QUANTILE_METHOD", "RowSpec",
           "row_specs", "hard_row", "hard_rows_fn", "tau_row", "by_cohort"]

#: Rows describing a width. The flat fit holds these out and ``b`` loads only on
#: them, so they are all ``omega`` has.
SCALE_STATS = frozenset(s.value for s in (WIDTH_STATS | SAMPLING_WIDTH_STATS))

#: Statistics with an evaluator. Anything else is corpus work, not a silent drop.
SUPPORTED_STATS = frozenset({"quantile", "mean", "sd", "se", "iqr"})

#: Estimator convention -> numpy's name for it.
NUMPY_QUANTILE_METHOD = {
    "type2": "averaged_inverted_cdf",
    "type4": "interpolated_inverted_cdf",
    "type6": "weibull",
    "type7": "linear",
    "type8": "median_unbiased",
}


@dataclass(frozen=True)
class RowSpec:
    """One statistic a source printed, and what the model must compute to match it."""

    target_id: str
    cohort_id: str
    stat: str
    value: float                  # the printed number, one entry of T-hat
    n: int                        # patients behind it: n_evaluable, else n_c
    p: Optional[float] = None
    convention: str = "type7"
    convention_recorded: bool = False
    log: bool = False             # whether the row enters the likelihood as its log

    @property
    def is_scale(self) -> bool:
        return self.stat in SCALE_STATS

    @property
    def label(self) -> str:
        return f"{self.target_id}/" + (
            f"q{self.p:g}" if self.stat == "quantile" else self.stat
        )


def row_specs(
    targets: Mapping[str, Dict[str, Any]],
    n_of: Mapping[str, int],
    *,
    default_convention: str = "type7",
    log_scale_rows: bool = True,
) -> List[RowSpec]:
    """Every printed statistic as a row, by target then by the source's own order.

    ``n_of`` maps cohort to ``n_c``; a target's ``n_evaluable`` wins where it
    declares one. An unrecorded ``quantile_convention`` falls back to
    ``default_convention`` and is flagged, so the count of assumed ones is
    reportable rather than invisible.

    Raises on a statistic with no evaluator: a silently missing row is a silently
    reweighted corpus.
    """
    out: List[RowSpec] = []
    unsupported: List[str] = []

    for tid in sorted(targets):
        ed = targets[tid].get("empirical_data") or {}
        od = ed.get("observed_distribution") or {}
        cohort_id = targets[tid].get("cohort_id")
        n = ed.get("n_evaluable") or n_of.get(cohort_id)
        if n is None:
            raise ValueError(f"{tid}: no n for cohort {cohort_id!r}")

        recorded = od.get("quantile_convention")
        for entry in od.get("statistics") or []:
            stat = entry.get("stat")
            if stat not in SUPPORTED_STATS:
                unsupported.append(f"{tid}/{stat}")
                continue
            out.append(RowSpec(
                target_id=tid,
                cohort_id=cohort_id,
                stat=stat,
                value=float(entry["value"]),
                n=int(n),
                p=entry.get("p"),
                convention=recorded or default_convention,
                convention_recorded=recorded is not None,
                log=log_scale_rows and stat in SCALE_STATS,
            ))

    if unsupported:
        raise ValueError(
            f"{len(unsupported)} printed statistics have no evaluator: "
            + ", ".join(sorted(unsupported))
        )
    return out


def by_cohort(specs: Sequence[RowSpec]) -> Dict[str, List[RowSpec]]:
    """Group rows by cohort, preserving order. ``K_c`` is the length of each list."""
    out: Dict[str, List[RowSpec]] = {}
    for spec in specs:
        out.setdefault(spec.cohort_id, []).append(spec)
    return out


def hard_row(spec: RowSpec, values: np.ndarray) -> float:
    """What the source's own software computed, from its ``n_c`` patients.

    Numpy and not smooth, on purpose: this is the observation being resampled to
    build ``V^boot``, so it has to reproduce the estimator rather than the model's
    prediction of it.
    """
    v = np.asarray(values, dtype=float)
    method = NUMPY_QUANTILE_METHOD[spec.convention]

    if spec.stat == "quantile":
        out = np.quantile(v, spec.p, method=method)
    elif spec.stat == "mean":
        out = v.mean()
    elif spec.stat == "iqr":
        q25, q75 = np.quantile(v, [0.25, 0.75], method=method)
        out = q75 - q25
    elif spec.stat == "sd":
        out = v.std(ddof=1)
    elif spec.stat == "se":
        out = v.std(ddof=1) / np.sqrt(v.shape[0])
    else:  # pragma: no cover - row_specs rejects these
        raise ValueError(f"no evaluator for {spec.stat!r}")

    return float(np.log(max(out, 1e-30)) if spec.log else out)


def hard_rows_fn(
    specs_by_cohort: Mapping[str, Sequence[RowSpec]],
    readout_of: Mapping[str, np.ndarray],
):
    """The ``rows_fn(cohort_id, indices)`` callback :func:`bootstrap_V` expects.

    ``readout_of`` maps a target to its cloud of predicted per-patient values, all
    sharing the patient axis so one index means one simulated patient everywhere.
    """
    def rows_fn(cohort_id: str, indices: np.ndarray) -> np.ndarray:
        idx = np.asarray(indices)
        return np.array([
            hard_row(spec, readout_of[spec.target_id][idx])
            for spec in specs_by_cohort[cohort_id]
        ])

    return rows_fn


def tau_row(spec: RowSpec, cloud_sorted, w, design=None):
    """The model's prediction of the printed number: its expectation over ``n_c``.

    ``design`` is the frozen bootstrap design, needed only by the moment rows.
    """
    from qsp_inference.vpop import statistics as st

    if spec.stat == "quantile":
        out = st.expected_quantile(cloud_sorted, w, spec.p, spec.n, spec.convention)
    elif spec.stat == "mean":
        out = st.mean_row(cloud_sorted, w)
    elif spec.stat == "iqr":
        return st.iqr_row(cloud_sorted, w, spec.n, spec.convention, log=spec.log)
    elif spec.stat == "sd":
        return st.sd_row(cloud_sorted, w, _need(design, spec), log=spec.log)
    elif spec.stat == "se":
        return st.se_row(cloud_sorted, w, _need(design, spec), spec.n, log=spec.log)
    else:  # pragma: no cover - row_specs rejects these
        raise ValueError(f"no evaluator for {spec.stat!r}")

    import jax.numpy as jnp
    return jnp.log(jnp.clip(out, 1e-30, None)) if spec.log else out


def _need(design, spec: RowSpec):
    if design is None:
        raise ValueError(f"{spec.label}: a {spec.stat} row needs a bootstrap design")
    return design

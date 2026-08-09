"""The rows a corpus contributes, and the two ways each one is evaluated.

A row is one statistic a source printed. It has two evaluators, and they are not
the same function. :func:`hard_row` is what the source's own software computed
from its ``n_c`` patients, and resampling it builds ``V^boot``. :func:`tau_row`
is the model's prediction of that number, which is its expectation over an
``n_c``-sample (eq:stat) and is smooth in ``phi``.

Order statistics get that expectation in closed form. Sort ``n`` draws from any
distribution and the rank of the k-th in it is ``Beta(k, n-k+1)`` whatever the
shape, so the expectation is a Beta-weighted average of the cloud (eq:smoothq).
That covers quantile rows, and an interquartile range as the difference of two.

Moments do not: ``E[s]/sigma`` runs from 0.77 to 0.96 at ``n=8`` depending on
shape, and logging does not stabilise it. Those rows are bootstrapped at ``phi``
on a frozen design instead.

Scale rows are the schema's ``WIDTH_STATS | SAMPLING_WIDTH_STATS``. The draft's
location half is their complement, which is not the schema's ``LOCATION_STATS``:
that one excludes quantiles because it answers a different question.

Every functional here reads an unweighted cloud. eq:elig would make the cloud
weighted per cohort, and no corpus declares a criterion; see ``predict``.

Requires ``jax_enable_x64``. The Beta kernel differences a CDF across cloud
members, so at ``N`` in the hundreds of thousands each mass is order ``1e-5`` and
float32 cumulative sums lose it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import betainc

from maple.core.calibration.shared_models import SAMPLING_WIDTH_STATS, WIDTH_STATS

__all__ = [
    # the corpus side
    "SCALE_STATS", "SUPPORTED_STATS", "NUMPY_QUANTILE_METHOD", "RowSpec",
    "row_specs", "by_cohort", "hard_row", "hard_rows_fn",
    # the model side
    "QUANTILE_CONVENTIONS", "order_statistic_mass", "quantile_mass",
    "expected_quantile", "extreme_row", "mean_row", "iqr_row",
    "bootstrap_design", "bootstrap_row", "sd_row", "se_row", "tau_row",
]

#: Rows describing a width. ``b`` loads only on them, so they are all ``omega``
#: has.
SCALE_STATS = frozenset(s.value for s in (WIDTH_STATS | SAMPLING_WIDTH_STATS))

#: Statistics with an evaluator. Anything else is corpus work, not a silent drop.
#: ``min``/``max`` are order statistics 1 and ``n``; the schema reads the pair as a
#: printed range. They stay out of ``SCALE_STATS`` because one endpoint alone is
#: not a width.
SUPPORTED_STATS = frozenset({"quantile", "mean", "sd", "se", "iqr", "min", "max"})

#: Estimator convention -> numpy's name for it.
NUMPY_QUANTILE_METHOD = {
    "type2": "averaged_inverted_cdf",
    "type4": "interpolated_inverted_cdf",
    "type6": "weibull",
    "type7": "linear",
    "type8": "median_unbiased",
}

#: Which order statistic an estimator selects for the p-quantile of n points.
#: Named for the Hyndman-Fan types. No paper records which its software used.
QUANTILE_CONVENTIONS: Dict[str, Callable[[float, int], float]] = {
    "type7": lambda p, n: (n - 1) * p + 1,  # R, numpy, pandas (default)
    "type6": lambda p, n: (n + 1) * p,      # SPSS, Minitab
    "type4": lambda p, n: n * p,            # linear interpolation of the ecdf
    "type8": lambda p, n: (n + 1 / 3) * p + 1 / 3,  # median-unbiased
    # Inverted ecdf, averaged at a discontinuity. Discontinuous in p, but the
    # caller mixes the two order statistics around h with weight frac, so the
    # averaging case is just h = np + 1/2.
    "type2": lambda p, n: (n * p + 0.5 if float(n * p).is_integer()
                           else math.ceil(n * p)),
}


# ------------------------------------------------------------ what a source said


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
    exclude: Sequence[tuple] = (),
) -> List[RowSpec]:
    """Every printed statistic as a row, by target then by the source's own order.

    ``n_of`` maps cohort to ``n_c``; a target's ``n_evaluable`` wins where it
    declares one. An unrecorded ``quantile_convention`` falls back to
    ``default_convention`` and is flagged, so the count of assumed ones is
    reportable rather than invisible.

    Raises on a statistic with no evaluator: a silently missing row is a silently
    reweighted corpus. ``exclude`` is the escape, as ``(target_id, stat)`` pairs,
    for a number the source did not print: a value the corpus derived under an
    assumption is not evidence, and fitting it feeds the assumption back in.
    """
    out: List[RowSpec] = []
    unsupported: List[str] = []
    skip = {tuple(e) for e in exclude}
    unused = set(skip)

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
            if (tid, stat) in skip:
                unused.discard((tid, stat))
                continue
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
    if unused:
        # An exclusion that matches nothing is a corpus edit the caller has not
        # noticed: the row it names is gone, or was renamed, and the reason the
        # caller recorded no longer applies to anything.
        raise ValueError(
            "these exclusions match no printed statistic: "
            + ", ".join(f"{t}/{s}" for t, s in sorted(unused))
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
    elif spec.stat == "min":
        out = v.min()
    elif spec.stat == "max":
        out = v.max()
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


# ------------------------------------------------- what the model predicts of it


def _require_x64() -> None:
    """Fail loudly rather than degrade. ``submodel.inference`` turns x64 off globally."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "vpop.rows needs jax_enable_x64. The Beta kernel differences a CDF "
            "across cloud members, so each mass is order 1/N and float32 loses "
            "it. Set jax.config.update('jax_enable_x64', True)."
        )


def order_statistic_mass(n_cloud: int, kappa: int, n: int):
    """``Beta(kappa, n-kappa+1)`` mass on each cloud member's slice of [0, 1].

    Member ``i`` of an unweighted cloud spans cumulative weight
    ``[i/N, (i+1)/N]``, so its mass is the CDF's increment across that span.
    Sums to one without normalising, and reads none of ``phi``: the cloud enters
    only through its size, which is why the masses a corpus needs are built once
    outside the gradient.
    """
    _require_x64()
    edges = jnp.linspace(0.0, 1.0, int(n_cloud) + 1)
    return jnp.diff(betainc(float(kappa), float(n - kappa + 1), edges))


def quantile_mass(n_cloud: int, p: float, n: int, convention: str = "type7"):
    """The Beta weights ``E[q_p]`` applies to the sorted cloud.

    A non-integer order statistic is what the estimator interpolates between, and
    the estimator is linear in the two, so the two Beta masses mix with the same
    weights. Depends only on ``(n_cloud, p, n, convention)``, so every row sharing
    those shares the vector.
    """
    lo, frac = _position(p, n, convention)
    mass = (1.0 - frac) * order_statistic_mass(n_cloud, lo, n)
    if frac > 0:
        mass = mass + frac * order_statistic_mass(n_cloud, min(lo + 1, n), n)
    return mass


def _position(p: float, n: int, convention: str):
    """The convention's order-statistic position, split into index and fraction."""
    h = QUANTILE_CONVENTIONS[convention](p, n)
    h = min(max(h, 1.0), float(n))
    lo = int(h // 1)
    return lo, h - lo


def expected_quantile(x_sorted, p: float, n: int, convention: str = "type7",
                      mass=None):
    """``E[q_p]`` over an ``n``-sample from the cloud. eq:smoothq.

    ``x_sorted`` is ascending. ``mass`` reuses a vector from :func:`quantile_mass`.
    """
    x_sorted = jnp.asarray(x_sorted)
    if mass is None:
        mass = quantile_mass(x_sorted.shape[0], p, n, convention)
    return x_sorted @ mass


def extreme_row(x_sorted, n: int, upper: bool):
    """A reported minimum or maximum: order statistic 1 or ``n``, exactly.

    A printed range is the pair, and each endpoint is an order statistic like any
    other, so no convention applies and nothing is interpolated. The expectation
    still reads the whole cloud, but the Beta mass concentrates on one tail, which
    is where the surrogate is least accurate.
    """
    x_sorted = jnp.asarray(x_sorted)
    return x_sorted @ order_statistic_mass(x_sorted.shape[0], n if upper else 1, n)


def mean_row(x_sorted):
    """A reported mean. ``E[sample mean] = population mean``, so no correction."""
    return jnp.mean(jnp.asarray(x_sorted))


def iqr_row(x_sorted, n: int, convention: str = "type7", log: bool = False,
            u=None):
    """A reported interquartile range: ``E[IQR]``, or ``E[log IQR]`` when logged.

    eq:obs's mean is the expectation of the number the source printed, and a
    logged row printed a log, so the expectation has to be taken there. Taking it
    outside instead reports ``log E[IQR]``, which is larger by the Jensen gap: an
    IQR from ``n=9`` carries about a 30% sampling CV, worth ~0.045 log units on
    the rows that are the only evidence about ``omega``.

    ``E[IQR]`` is exact by linearity of the expectation over the two order
    statistics. ``E[log IQR]`` is not, so it needs the frozen bootstrap design
    ``u``, the same one the moment rows use.
    """
    if not log:
        return (expected_quantile(x_sorted, 0.75, n, convention)
                - expected_quantile(x_sorted, 0.25, n, convention))
    if u is None:
        raise ValueError(
            "a logged iqr row needs a bootstrap design: E[log IQR] has no closed "
            "form, and log E[IQR] is a different quantity")

    lo25, f25 = _position(0.25, n, convention)
    lo75, f75 = _position(0.75, n, convention)

    def _log_width(v):
        # A replicate IS the printed sample, so its quantiles are the ordinary
        # ones.
        vs = jnp.sort(v)
        width = _interp(vs, lo75, f75, n) - _interp(vs, lo25, f25, n)
        return jnp.log(jnp.clip(width, 1e-30, None))

    return bootstrap_row(x_sorted, u, _log_width)


def _interp(v_sorted, lo: int, frac: float, n: int):
    """The estimator between two order statistics of one replicate."""
    a = v_sorted[lo - 1]
    b = v_sorted[min(lo, n - 1)]
    return (1.0 - frac) * a + frac * b if frac > 0 else a


def bootstrap_design(key, n: int, n_boot: int = 400):
    """Frozen uniforms for a moment row, shape ``(n_boot, n)``.

    Uniforms rather than indices: a run may use a different cloud size from the
    one the design was drawn against, and a JAX gather clamps an out-of-range
    index instead of failing.
    """
    return jax.random.uniform(key, (n_boot, n))


def bootstrap_row(x_sorted, u, fn):
    """``E*[fn]`` over the frozen design. ``fn(values)`` is the statistic."""
    _require_x64()
    x_sorted = jnp.asarray(x_sorted)
    idx = jnp.minimum((u * x_sorted.shape[0]).astype(jnp.int32),
                      x_sorted.shape[0] - 1)
    return jnp.mean(jax.vmap(fn)(x_sorted[idx]))


def _sample_sd(v):
    """The ``n-1`` sample standard deviation of one replicate."""
    return jnp.std(v, ddof=1)


def sd_row(x_sorted, u, log: bool = False):
    """A reported standard deviation, as ``E*[s]`` over the frozen design.

    No closed form: the correction depends on the shape of the pushforward, which
    is what ``phi`` controls, so it is neither distribution-free nor a fixed offset.
    """
    if not log:
        return bootstrap_row(x_sorted, u, _sample_sd)
    # Inside the expectation, not outside: the row printed a log, so eq:obs's
    # mean is E[log s] and log E[s] is larger by the Jensen gap.
    return bootstrap_row(x_sorted, u,
                         lambda v: jnp.log(jnp.clip(_sample_sd(v), 1e-30, None)))


def se_row(x_sorted, u, n: int, log: bool = False):
    """A reported standard error of a mean, as ``E*[s/sqrt(n)]``.

    The expectation of the estimator the source printed, not the sampling spread
    itself. The two differ at ``O(1/n)`` and only the first is ``E[printed]``.
    """
    rt = jnp.sqrt(float(n))
    if not log:
        return bootstrap_row(x_sorted, u, _sample_sd) / rt
    return bootstrap_row(x_sorted, u,
                         lambda v: jnp.log(jnp.clip(_sample_sd(v) / rt, 1e-30, None)))


def tau_row(spec: RowSpec, cloud_sorted, design=None, mass=None):
    """The model's prediction of the printed number: its expectation over ``n_c``.

    ``design`` is the frozen bootstrap design, needed only by the rows with no
    closed form. ``mass`` is a Beta weight vector from :func:`quantile_mass`,
    reused across rows that share ``(n_cloud, p, n, convention)``.
    """
    if spec.stat == "quantile":
        out = expected_quantile(cloud_sorted, spec.p, spec.n, spec.convention,
                                mass=mass)
    elif spec.stat in ("min", "max"):
        out = extreme_row(cloud_sorted, spec.n, spec.stat == "max")
    elif spec.stat == "mean":
        out = mean_row(cloud_sorted)
    elif spec.stat == "iqr":
        return iqr_row(cloud_sorted, spec.n, spec.convention, log=spec.log,
                       u=_need(design, spec) if spec.log else None)
    elif spec.stat == "sd":
        return sd_row(cloud_sorted, _need(design, spec), log=spec.log)
    elif spec.stat == "se":
        return se_row(cloud_sorted, _need(design, spec), spec.n, log=spec.log)
    else:  # pragma: no cover - row_specs rejects these
        raise ValueError(f"no evaluator for {spec.stat!r}")

    return jnp.log(jnp.clip(out, 1e-30, None)) if spec.log else out


def _need(design, spec: RowSpec):
    if design is None:
        raise ValueError(f"{spec.label}: a {spec.stat} row needs a bootstrap design")
    return design

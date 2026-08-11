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
on a frozen design instead, and so is any location row whose scale is not raw,
since ``E[g(T)] != g(E[T])`` unless ``T`` is an order statistic.

Closed forms exist for those, as a check rather than a replacement. The
second-order delta method gives ``E[g(T)] ~ g(mu_T) + g''(mu_T) var(T) / 2``, so
a mean row is ``g(m) + g''(m) sigma^2 / (2n)`` to ``O(1/n^2)`` and a logged
width row is ``log sigma - (kappa - 1 + 2/(n-1)) / (4n)``, with ``kappa`` the
cloud's kurtosis; at ``kappa = 3`` the latter is ``-1/(2(n-1))``, which is the
normal-theory answer, so it reproduces the exact case and carries the shape
dependence the fixed offset could not. It is not used because its error is the
term it drops: at ``kappa ~ 110``, which a lognormal pushforward at ``sigma = 1``
reaches, the ``n = 9`` correction is 3 log units and the expansion has stopped
meaning anything. The bootstrap assumes nothing, is one mechanism for all of
these rows, and costs 400 replicates against an emulator pass over the cloud.
The transforms above pull the pushforward toward normal, so the two should now
agree wherever the expansion is valid, and where they disagree is a heavy tail.

Scale rows are the schema's ``WIDTH_STATS | SAMPLING_WIDTH_STATS``. The draft's
location half is their complement, which is not the schema's ``LOCATION_STATS``:
that one excludes quantiles because it answers a different question.

Every functional takes an optional ``w_sorted``, eq:elig's per-patient weight
carrying that row's own sort. Without it the cloud is unweighted and the masses
are built once outside the gradient; with it each member's slice of [0, 1] is its
own share rather than 1/N, so the mass reads ``phi`` and is built per row. The
weights self-normalise, which is why no ``Z`` appears: it divides the numerator
and the denominator of one expectation. See ``predict``.

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
from jax.scipy.special import gammaln

from maple.core.calibration.shared_models import SAMPLING_WIDTH_STATS, WIDTH_STATS

__all__ = [
    # the corpus side
    "SCALE_STATS", "SUPPORTED_STATS", "NUMPY_QUANTILE_METHOD", "RowSpec",
    "SCALE_BY_KIND", "TRANSFORMS", "row_specs", "by_cohort", "hard_row",
    "hard_rows_fn",
    # the model side
    "QUANTILE_CONVENTIONS", "order_statistic_mass", "quantile_mass",
    "order_statistic_mass_w", "quantile_mass_w", "weighted_edges",
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

#: The scale eq:obs compares a row on.
TRANSFORMS = ("raw", "log", "asinh", "logit")

#: A location row's scale, from the target's declared ``quantity_kind``.
#:
#: eq:obs is a Gaussian on the row, so the scale is the claim about where the
#: quantity can be. Raw units assert it can be anywhere on the line, which is
#: false for every kind below and worst where the sampling spread approaches the
#: level: a fold change is a ratio, and the raw-scale bootstrap of its upper
#: quartile over a population cloud has no usable variance at all.
#:
#: Keyed on the KIND and not on the row, which is what stops this being a knob.
#: ``quantity_kind`` is a property of the quantity, is declared when the target
#: is written, and is the same field ``mechanism.build_Z`` reads for its ``kind:``
#: columns. Choosing per row, or by which rows fit badly, would be the thing this
#: is arranged to prevent.
#:
#: asinh rather than log for the unbounded kinds. It is log-like above
#: ``scale_ref`` and linear below it, so it needs no ``clip`` fiction for a
#: readout that is zero in some patients, and it is defined for negatives, which
#: ``time`` needs: a tumour doubling time is negative when the tumour shrinks.
#: The price is ``scale_ref``, the one number this introduces.
#:
#: fraction gets logit because log respects only its lower bound. Under log the
#: model may predict a fraction above 1 and pay nothing, and this corpus has a
#: cohort at 0.69/0.83/0.89 where that is not hypothetical.
SCALE_BY_KIND = {
    "foldchange": "asinh",
    "ratio": "asinh",
    "concentration": "asinh",
    "density": "asinh",
    "count": "asinh",
    "time": "asinh",
    "fraction": "logit",
    "percentage": "logit",
}

#: Placeholder for a kind with no entry, used only to finish building the spec
#: before :func:`row_specs` raises on it. Raw is the weakest claim rather than
#: the safest one -- it says the quantity may sit anywhere on the line -- so it
#: is not a fallback and nothing reaches a fit carrying it by default.
DEFAULT_SCALE = "raw"

#: Kinds whose h_r CHANNEL is a log-odds rather than a log, so eq:disc acts in a
#: coordinate the quantity's upper bound cannot be pushed through.
#:
#: SCALE_BY_KIND above puts the bound on the scale the RESIDUAL is measured on.
#: That is not where a fraction leaves the interval. eq:disc is
#: ``kappa (x - c) + c + gamma`` on the channel, and under a log channel kappa
#: widens a fraction multiplicatively with no ceiling: on the pdac corpus a
#: cloud spanning 0.68 to 0.918 about a pivot of 0.76 reaches 1.07 at
#: ``kappa = 1.85``, and the logit tangent then turns that 7% overshoot into a
#: 132-sigma residual on one row of stromal_fraction. Under a logit channel the
#: inverse is a sigmoid, so the same kappa lands inside (0, 1) for every real
#: argument and the tangent is unreachable.
#:
#: Same construction, and same argument, as the logit MARGIN that
#: ``predict.apply_margins`` gives a parameter bounded on (0, 1): a bound the
#: coordinate cannot violate has no wall to hit. This is that one level down, on
#: the readout instead of the parameter.
LOGIT_KINDS = frozenset(k for k, v in SCALE_BY_KIND.items() if v == "logit")

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

    # The scale eq:obs compares this row on, one of TRANSFORMS. A width row is
    # positive whatever its quantity is, and its sampling distribution is
    # right-skewed, so it takes log; asinh would sit in its linear regime for a
    # width, since scale_ref is the size of the LEVEL and not of the spread. A
    # location row takes its quantity's scale, from SCALE_BY_KIND.
    scale: str = "raw"
    # asinh's unit, in the row's native units. Set from the target's own printed
    # location values, so it is corpus-derived and frozen before any fit, and is
    # shared by every row of a target: V's off-diagonals mix rows of one readout,
    # so two of them on different scales would not be comparable.
    scale_ref: float = 1.0

    @property
    def is_scale(self) -> bool:
        return self.stat in SCALE_STATS

    @property
    def commutes(self) -> bool:
        """Whether the transform can be applied after the expectation.

        Order statistics are equivariant under a monotone map, so a quantile,
        min or max row transforms exactly and for free. A mean does not:
        ``E[g(mean)] != g(E[mean])``, so it needs the log-inside-the-expectation
        treatment the moment rows already use.
        """
        return self.stat in ("quantile", "min", "max")

    @property
    def label(self) -> str:
        return f"{self.target_id}/" + (
            f"q{self.p:g}" if self.stat == "quantile" else self.stat
        )


def _readout_attr(target: Mapping[str, Any], name: str) -> Optional[str]:
    """The same field ``mechanism.build_Z`` reads, duplicated to keep the import out.

    ``rows`` is the corpus side and pulls in nothing heavy; ``mechanism`` loads
    the emulator.
    """
    return ((target.get("observable") or {}).get("readout") or {}).get(name)


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
    unkinded: List[str] = []
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
        entries = [e for e in (od.get("statistics") or [])
                   if (tid, e.get("stat")) not in skip]
        for e in (od.get("statistics") or []):
            if (tid, e.get("stat")) in skip:
                unused.discard((tid, e.get("stat")))

        kind = _readout_attr(targets[tid], "quantity_kind")
        loc_scale = SCALE_BY_KIND.get(kind, DEFAULT_SCALE)
        if kind is not None and kind not in SCALE_BY_KIND:
            unkinded.append(f"{tid} ({kind})")
        elif kind is None:
            unkinded.append(f"{tid} (declares none)")

        # asinh's unit: the size of the thing, from what the source printed about
        # its LEVEL. Width rows are excluded because a width is not a level, and
        # a scale set from one would put every location row in asinh's linear
        # regime and undo the transform.
        levels = [abs(float(e["value"])) for e in entries
                  if e.get("stat") in SUPPORTED_STATS
                  and e.get("stat") not in SCALE_STATS]
        positive = [v for v in levels if v > 0]
        scale_ref = float(np.median(positive)) if positive else 1.0

        for entry in entries:
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
                scale=("log" if (log_scale_rows and stat in SCALE_STATS)
                       else "raw" if stat in SCALE_STATS
                       else loc_scale),
                scale_ref=scale_ref,
            ))

    if unsupported:
        raise ValueError(
            f"{len(unsupported)} printed statistics have no evaluator: "
            + ", ".join(sorted(unsupported))
        )
    if unkinded:
        # Raw is a claim, not an absence of one: it says the quantity may sit
        # anywhere on the line. A target that never declared its kind has not
        # made that claim, so falling back to it silently would assert on the
        # corpus's behalf.
        raise ValueError(
            f"{len(unkinded)} targets have no scale, because SCALE_BY_KIND has "
            "no entry for the quantity_kind they declare. Add the kind there, "
            "or declare one on the target: "
            + ", ".join(sorted(unkinded))
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


def to_scale(out, spec: RowSpec, xp):
    """eq:obs's coordinate for one row. ``xp`` is ``np`` or ``jnp``.

    One function for both sides on purpose. ``hard_row`` and ``tau_row`` have to
    land in the same coordinate or ``V`` describes a different quantity from the
    residual it standardises, and that mismatch is silent.
    """
    if spec.scale == "raw":
        return out
    if spec.scale == "log":
        return xp.log(xp.clip(out, 1e-30, None))
    if spec.scale == "asinh":
        return xp.arcsinh(out / spec.scale_ref)
    if spec.scale == "logit":
        return logit_link(out, xp)
    raise ValueError(f"{spec.label}: unknown scale {spec.scale!r}")


#: Where logit hands over to its tangent. Only the UPPER bound is guarded, and
#: the asymmetry is not an oversight: ``x >= 1`` means the model is asserting a
#: fraction above 100%, while small fractions are ordinary here. This corpus
#: prints them down to 5e-4, so any floor large enough to guard would sit above
#: real data and bend rows that are perfectly well posed. 1e-3 sits far above the
#: largest fraction the corpus prints away from the boundary (0.886).
#:
#: A non-positive argument therefore still gives a NaN. That exposure is
#: unchanged by ``logit_link`` also being the h_r channel for a LOGIT_KINDS
#: readout: ``to_scale`` already applied it to whatever the observable body
#: composed, so the channel adds no case that was not already reachable.
LOGIT_MARGIN = 1e-3


def logit_link(x, xp):
    """``log(x / (1-x))``, continued by its tangent at ``1 - LOGIT_MARGIN``.

    Not a clip. A clip invents a value AND flattens the gradient, and a flat
    gradient under a downstream ``sqrt`` or ``log`` is the ``0 * inf`` that puts
    NaN in a Jacobian while leaving the forward pass finite. The tangent is C1,
    monotone, and defined on all of R, so a fraction the model pushes past 1
    stays differentiable and reports itself as a large residual instead of
    killing the chain.

    :func:`natural_from_logit` is the inverse, and the pair is exact below
    ``1 - LOGIT_MARGIN``, which is where a declared fraction lives. Above it the
    two do NOT round-trip, on purpose: the inverse is a plain sigmoid so that
    eq:disc cannot put a fraction outside (0, 1) whatever ``kappa`` it applies.
    """
    hi = 1.0 - LOGIT_MARGIN
    slope = 1.0 / (hi * (1.0 - hi))
    at_hi = math.log(hi) - math.log1p(-hi)
    # minimum() keeps log1p inside its domain on BOTH branches. Evaluating the
    # unsafe branch and selecting afterwards is what forward-mode differentiates
    # through, so the guard has to be inside the expression, not around it.
    safe = xp.minimum(x, hi)
    inside = xp.log(safe) - xp.log1p(-safe)
    return xp.where(x < hi, inside, at_hi + (x - hi) * slope)


def natural_from_logit(x, xp):
    """A logit-channel readout back in the units the source printed, in (0, 1).

    Plain sigmoid, with no continuation to match :func:`logit_link`'s tangents.
    That asymmetry is the point of the pair: this is what eq:disc's output passes
    through, and a sigmoid is onto (0, 1) for every real argument, so no
    ``kappa`` and no ``gamma`` can produce a fraction outside its own bound.
    """
    # exp(-|x|) is bounded by 1, so neither branch can overflow and neither
    # carries an inf into the gradient. Writing it as 1/(1+exp(-x)) instead is
    # correct in value and gives inf/inf at large negative x under reverse mode.
    z = xp.exp(-xp.abs(x))
    return xp.where(x >= 0.0, 1.0 / (1.0 + z), z / (1.0 + z))


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

    return float(to_scale(out, spec, np))


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


def _beta_tail(kappa: int, n: int, x):
    """``I_x(kappa, n-kappa+1)``, as the binomial tail ``P(Bin(n, x) >= kappa)``.

    Exact rather than approximate: ``kappa`` and ``n - kappa + 1`` are always
    integers here, so the regularised incomplete beta is a sum of ``n - kappa + 1``
    binomial terms. ``betainc`` cannot know that and runs its general continued
    fraction, which is what it costs. A corpus's source cohorts are small -- this
    one has 17 distinct ``n`` between 6 and 215, median 10 -- so summing the tail
    is 27x cheaper through a value and gradient, and agrees to 1e-12.

    The endpoints are substituted before the log and restored after. ``x`` is a
    cumulative weight, so 0 and 1 are attained, and ``log 0`` there would put a
    ``0 * inf`` into both the value and the gradient.
    """
    j = jnp.arange(kappa, n + 1)
    safe = jnp.clip(x, jnp.finfo(jnp.asarray(x).dtype).tiny, 1.0 - 1e-16)
    log_c = gammaln(n + 1.0) - gammaln(j + 1.0) - gammaln(n - j + 1.0)
    log_term = (log_c[:, None] + j[:, None] * jnp.log(safe)[None, :]
                + (n - j)[:, None] * jnp.log1p(-safe)[None, :])
    tail = jnp.sum(jnp.exp(log_term), axis=0)
    return jnp.where(x <= 0.0, 0.0, jnp.where(x >= 1.0, 1.0, tail))


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
    return jnp.diff(_beta_tail(int(kappa), int(n), edges))


def weighted_edges(w_sorted):
    """Cumulative weight boundaries on [0, 1], for a cloud sorted by the row's own
    readout. ``w_sorted`` must carry that row's permutation.

    The outer two are literals, not ``0/total`` and ``total/total``. They are 0
    and 1 for any weights, and the tail is 0 and 1 there for any ``(kappa, n)``,
    so they say nothing about ``w`` -- but computed as divisions they carry a
    gradient into the tail's endpoint, which is where a ``0 * inf`` lives. That
    is guarded inside :func:`_beta_tail` as well; keeping the edges literal keeps
    the spurious dependence on ``w`` from being formed at all. The unweighted
    path never sees it because its edges are constants.
    """
    w = jnp.asarray(w_sorted)
    c = jnp.cumsum(w)
    return jnp.concatenate([jnp.zeros(1, dtype=c.dtype),
                            c[:-1] / c[-1],
                            jnp.ones(1, dtype=c.dtype)])


def order_statistic_mass_w(w_sorted, kappa: int, n: int):
    """eq:elig's :func:`order_statistic_mass`: the slice is the member's weight.

    Unweighted, member ``i`` spans ``[i/N, (i+1)/N]``. Weighted, it spans its own
    share of the total, so an ineligible member's slice closes to nothing and it
    leaves the row without being removed from the cloud. Self-normalising, which
    is why no ``Z`` appears: it divides the numerator and the denominator of the
    same expectation.

    Unlike the unweighted form this reads ``phi``, so it cannot be built once
    outside the gradient the way ``mass_table`` is.
    """
    _require_x64()
    return jnp.diff(_beta_tail(int(kappa), int(n), weighted_edges(w_sorted)))


def quantile_mass_w(w_sorted, p: float, n: int, convention: str = "type7"):
    """:func:`quantile_mass` on a weighted cloud. Same interpolation, same
    convention; only the slice widths change."""
    lo, frac = _position(p, n, convention)
    mass = (1.0 - frac) * order_statistic_mass_w(w_sorted, lo, n)
    if frac > 0:
        mass = mass + frac * order_statistic_mass_w(w_sorted, min(lo + 1, n), n)
    return mass


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
            u=None, w_sorted=None):
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
        if w_sorted is not None:
            return (x_sorted @ quantile_mass_w(w_sorted, 0.75, n, convention)
                    - x_sorted @ quantile_mass_w(w_sorted, 0.25, n, convention))
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

    return bootstrap_row(x_sorted, u, _log_width, w_sorted=w_sorted)


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


def bootstrap_row(x_sorted, u, fn, w_sorted=None):
    """``E*[fn]`` over the frozen design. ``fn(values)`` is the statistic.

    ``u`` is frozen, and stays frozen under eq:elig: the weights change which
    member a given ``u`` lands on, not which uniforms are drawn. Unweighted that
    map is ``floor(u N)``, the inverse CDF of a uniform draw over members;
    weighted it is the inverse CDF of the cumulative weight, which is the same
    statement about a population whose members carry unequal mass.

    A weighted resample cannot be written as a reindex of the unweighted one,
    which is why this takes the weights rather than a permutation.
    """
    _require_x64()
    x_sorted = jnp.asarray(x_sorted)
    if w_sorted is None:
        idx = jnp.minimum((u * x_sorted.shape[0]).astype(jnp.int32),
                          x_sorted.shape[0] - 1)
        return jnp.mean(jax.vmap(fn)(x_sorted[idx]))
    # Weighted, by interpolating the weighted quantile function rather than
    # looking a member up in it. An index lookup is a step function of the
    # weights: the replicate jumps when a member's cumulative weight crosses a
    # frozen u, so the row is piecewise constant in phi with zero gradient
    # between jumps, which is a potential HMC cannot integrate. Interpolating
    # moves the replicate continuously as the weights move.
    #
    # Member i owns (c_{i-1}, c_i], so its knot is that interval's midpoint. The
    # unweighted path keeps its own lookup above: at equal weights this agrees
    # with it only to the O(1/N) the discretisation is worth, and the
    # interpolated form is the better answer rather than the compatible one.
    w = jnp.asarray(w_sorted)
    c = jnp.cumsum(w)
    knots = (c - 0.5 * w) / c[-1]
    return jnp.mean(jax.vmap(fn)(jnp.interp(u, knots, x_sorted)))


def _sample_sd(v):
    """The ``n-1`` sample standard deviation of one replicate."""
    return jnp.std(v, ddof=1)


def sd_row(x_sorted, u, log: bool = False, w_sorted=None):
    """A reported standard deviation, as ``E*[s]`` over the frozen design.

    No closed form: the correction depends on the shape of the pushforward, which
    is what ``phi`` controls, so it is neither distribution-free nor a fixed offset.
    """
    if not log:
        return bootstrap_row(x_sorted, u, _sample_sd, w_sorted=w_sorted)
    # Inside the expectation, not outside: the row printed a log, so eq:obs's
    # mean is E[log s] and log E[s] is larger by the Jensen gap.
    return bootstrap_row(x_sorted, u,
                         lambda v: jnp.log(jnp.clip(_sample_sd(v), 1e-30, None)),
                         w_sorted=w_sorted)


def se_row(x_sorted, u, n: int, log: bool = False, w_sorted=None):
    """A reported standard error of a mean, as ``E*[s/sqrt(n)]``.

    The expectation of the estimator the source printed, not the sampling spread
    itself. The two differ at ``O(1/n)`` and only the first is ``E[printed]``.
    """
    rt = jnp.sqrt(float(n))
    if not log:
        return bootstrap_row(x_sorted, u, _sample_sd, w_sorted=w_sorted) / rt
    return bootstrap_row(x_sorted, u,
                         lambda v: jnp.log(jnp.clip(_sample_sd(v) / rt, 1e-30, None)),
                         w_sorted=w_sorted)


def tau_row(spec: RowSpec, cloud_sorted, design=None, mass=None, w_sorted=None):
    """The model's prediction of the printed number: its expectation over ``n_c``.

    ``design`` is the frozen bootstrap design, needed only by the rows with no
    closed form. ``mass`` is a Beta weight vector from :func:`quantile_mass`,
    reused across rows that share ``(n_cloud, p, n, convention)``.
    """
    log = spec.scale == "log"
    if spec.commutes:
        # Transform the cloud, THEN take the expectation. Every scale here is
        # monotone increasing, so it preserves the sort and carries the quantile
        # function of x to that of g(x): E[g(q_p)] = sum_i m_i g(x_(i)), the same
        # Beta kernel read on the transformed cloud. Applying g to the result
        # instead would give g(E[q_p]), short of this by the Jensen gap. The
        # equivariance of an order statistic is pathwise, not in expectation,
        # and only hard_row gets to use the pathwise form.
        x = to_scale(cloud_sorted, spec, jnp)
        if spec.stat == "quantile":
            if w_sorted is not None:
                mass = quantile_mass_w(w_sorted, spec.p, spec.n, spec.convention)
            return expected_quantile(x, spec.p, spec.n, spec.convention, mass=mass)
        if w_sorted is not None:
            kappa = spec.n if spec.stat == "max" else 1
            return x @ order_statistic_mass_w(w_sorted, kappa, spec.n)
        return extreme_row(x, spec.n, spec.stat == "max")

    if spec.stat == "mean":
        if spec.scale == "raw":
            return mean_row(cloud_sorted)
        else:
            # E[g(mean)], not g(E[mean]). The Beta kernel gives the order
            # statistics their expectation in closed form and a monotone g
            # passes straight through it, but a mean is not an order statistic
            # and g does not commute with it, so this row joins the moment rows
            # on the frozen design.
            return bootstrap_row(
                cloud_sorted, _need(design, spec),
                lambda v: to_scale(jnp.mean(v), spec, jnp), w_sorted=w_sorted)
    elif spec.stat == "iqr":
        return iqr_row(cloud_sorted, spec.n, spec.convention, log=log,
                       u=_need(design, spec) if log else None, w_sorted=w_sorted)
    elif spec.stat == "sd":
        return sd_row(cloud_sorted, _need(design, spec), log=log,
                      w_sorted=w_sorted)
    elif spec.stat == "se":
        return se_row(cloud_sorted, _need(design, spec), spec.n, log=log,
                      w_sorted=w_sorted)
    raise ValueError(f"no evaluator for {spec.stat!r}")  # row_specs rejects these


def _need(design, spec: RowSpec):
    if design is None:
        raise ValueError(f"{spec.label}: a {spec.stat} row needs a bootstrap design")
    return design

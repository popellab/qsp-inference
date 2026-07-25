"""Per-observable quantile-anchor summaries -- the maple ``ObservedDistribution``
contract for population (NLME) inference.

A reported distribution is a quantile function, and the hierarchical NPE should
condition on cohort quantile *values* at a set of probability levels (anchors),
with the observed conditioning vector being the reported ``Q(p)`` at those same
levels. median + IQR is only the ``{0.25, 0.5, 0.75}`` special case; a richer
grid (deciles) carries more of the reported shape when the source supports it.

Source priority per observable (highest first):

1. ``observed_distribution`` anchors, when the target declares them -- maple's
   general form (``ObservedDistribution._anchor_pairs()``: median, IQR edges,
   quartiles, deciles, or a dense empirical quantile function). The forward-
   looking source; the seam is here so targets flow automatically once they
   carry it.
2. across-patient population ``samples`` -> empirical ``Q(p)`` at the anchor
   grid. The honest inter-patient distribution; supports whatever grid the
   biological ``n`` can resolve (a per-``n`` predictive null keeps even tail
   anchors calibrated downstream).
3. no samples -> a ci95-derived expansion around the median, flagged
   ``feeds_spread=False`` because a measurement/center interval is not
   population variability.

This is the observed-data half of the maple target-data contract; the prior
half (population ``omega`` from ``n_biological`` / ``spread_source``) is the
sibling that a shared target resolver will factor against.

**Two small-``n`` choices are load-bearing and are made here, once, for both
sides of the contract** (the observed anchor and the training cohort summary):
the quantile estimator (:data:`QUANTILE_METHOD`) and the anchor-count budget
(:data:`MIN_PATIENTS_PER_ANCHOR`). See their docs for the measurements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

__all__ = [
    "ObservedAnchors",
    "anchors_from_sources",
    "cohort_quantiles",
    "QUANTILE_METHOD",
    "MIN_PATIENTS_PER_ANCHOR",
]

_Z_95 = 1.959963984540054  # Phi^-1(0.975)

QUANTILE_METHOD = "normal_unbiased"
"""The quantile estimator, for the observed anchor AND the cohort summary alike.

NumPy's default (``method='linear'``, Hyndman-Fan type 7) shrinks the sample IQR
badly at the ``n`` published QSP targets actually carry. Measured
``E[IQR_hat_n] / IQR_pop`` on a Gaussian population:

===  ========  ===============  =================
 n    linear   median_unbiased  normal_unbiased
===  ========  ===============  =================
  6     0.788            1.028              1.009
  9     0.848            1.026              1.015
 12     0.891            1.018              1.010
 16     0.917            1.012              1.006
 30     0.954            1.005              1.001
===  ========  ===============  =================

The same shrinkage was measured on a 2000-patient PDAC cloud at each target's
real published ``n`` (ratio 0.78 at n=6, 0.87 at n=9-12), i.e. it is a property
of the estimator and ``n``, not of any particular population shape.

**Why the default is not merely cosmetic.** Any estimator that compares an
observed sample quantile against a *population* quantile (the summary-likelihood
path of docs ch. 4b, whose mean model is ``Q_j(phi)``) inherits the shrinkage
directly as a **low** bias on ``sigma_u_hat`` of 15-25% at n=6-12 -- the failure
mode ch. 4's matched footing exists to prevent, arriving from a different source.
The amortized path (ch. 4) is only partly protected: the shrinkage cancels when
both sides are summarised by this module, but NOT for a target taking the
``observed_distribution`` branch, whose quantiles were computed by the reporting
paper's own software.

Do not set this per call site. Both sides must use the same estimator, and a
module constant is what enforces that.
"""

MIN_PATIENTS_PER_ANCHOR = 2.0
"""Patients required per retained anchor, beyond mere resolvability.

:func:`_resolvable_grid` drops anchors a sample of size ``n`` cannot *resolve*.
That is necessary and not sufficient: a grid can be individually resolvable and
still too dense for the **joint** sampling law of the anchor vector to be
Gaussian. Measured on the PDAC cloud, log scale, corrected estimator, as the
Mahalanobis ``D^2/k`` of the anchor vector against its asymptotic covariance
(1.00 is exact; below 1 means the asymptotic covariance overstates the true
joint spread):

=======  ==========  ============  ============  =========
   n      quartiles   (.2,.5,.8)    (.1,.5,.9)    deciles
=======  ==========  ============  ============  =========
   6-8        0.976         0.813            --      0.559
  9-12        0.949         0.963         0.975      0.751
 21-60        0.993         0.945         0.952      0.936
   >60        1.015         0.965         0.954      0.999
=======  ==========  ============  ============  =========

Nine anchors from <= 12 patients is not a Gaussian vector; three anchors are, at
every spacing tried. The failure direction is conservative rather than invalid,
but it is not harmless for ``omega``: an over-flat likelihood lets the
``log sigma_u`` prior dominate, so a population genuinely wider than the anchored
omega goes undetected. ``n >= 2k`` covers every cell measured (quartiles at n=6
is 2.0; deciles at n=21 is 2.3; deciles at n <= 16 is <= 1.8 and fails).
"""


@dataclass
class ObservedAnchors:
    """The observed quantile anchors for one observable.

    Attributes:
        p_levels: probability levels (ascending), e.g. ``(0.25, 0.5, 0.75)``.
        values: observed ``Q(p)`` at ``p_levels``, same order.
        feeds_spread: whether the spread anchors are genuine population
            variability (``samples`` / a population ``observed_distribution``)
            rather than a center/measurement interval (ci95).
        source: which branch produced these anchors
            (``observed_distribution`` / ``samples`` / ``ci95``).
    """

    p_levels: tuple
    values: np.ndarray
    feeds_spread: bool
    source: str

    def as_pairs(self) -> list:
        """``[(p, value), ...]`` -- the form
        :func:`qsp_inference.inference.predictive_checks.quantile_vpc` consumes."""
        return [(float(p), float(v)) for p, v in zip(self.p_levels, self.values)]


def _resolvable_grid(quantiles, n, *, min_per_anchor: float = MIN_PATIENTS_PER_ANCHOR) -> tuple:
    """Drop anchors a sample of size ``n`` cannot support. Two filters, in order.

    **Resolvability (per anchor).** Keep ``p`` in ``[1/(n+1), n/(n+1)]`` -- the
    range a rank statistic of ``n`` draws actually informs; a 6-donor study
    cannot resolve ``Q(0.95)``. The median always survives (for ``n >= 1``).

    **Density (the anchor vector as a whole).** Resolvable-one-at-a-time does not
    make the *joint* sampling law Gaussian: nine deciles from 11 patients all pass
    the first filter and are still not a Gaussian vector
    (:data:`MIN_PATIENTS_PER_ANCHOR` carries the measurement). So require
    ``n >= min_per_anchor * len(grid)``, and when that binds, fall back to a
    symmetric **3-anchor** subset (:func:`_coarse_triple`), then to the median
    below that. Three is where the fallback stops rather than some thinned
    intermediate count because ``k=3`` is what was measured calibrated, at every
    spacing from ``(0.25, 0.5, 0.75)`` out to ``(0.1, 0.5, 0.9)``; an
    interpolated anchor count would assert a calibration nobody checked.

    ``n=None`` leaves the grid untouched (a denoised full-pool quantile, not an
    ``n``-statistic, so neither filter applies).
    """
    if n is None:
        return tuple(quantiles)
    lo, hi = 1.0 / (n + 1.0), n / (n + 1.0)
    keep = tuple(p for p in quantiles if lo <= p <= hi)
    if not keep:
        return (0.5,)
    if len(keep) * min_per_anchor <= n:
        return keep
    coarse = _coarse_triple(keep)
    if len(coarse) * min_per_anchor <= n:
        return coarse
    return (min(keep, key=lambda p: (abs(p - 0.5), p)),)


def _coarse_triple(keep: tuple) -> tuple:
    """The 3-anchor symmetric subset of ``keep`` nearest ``(0.25, 0.5, 0.75)``.

    Ties break *outward*, so deciles reduce to ``(0.2, 0.5, 0.8)`` rather than
    ``(0.3, 0.5, 0.7)``: the wider pair carries more spread information at the
    same anchor count, and both were measured calibrated (see
    :data:`MIN_PATIENTS_PER_ANCHOR`).
    """
    lower = min(keep, key=lambda p: (abs(p - 0.25), p))
    mid = min(keep, key=lambda p: (abs(p - 0.5), p))
    upper = min(keep, key=lambda p: (abs(p - 0.75), -p))
    return tuple(sorted({lower, mid, upper}))


def _ci95_expand(median, lo, hi, p_levels) -> np.ndarray:
    """Expand a center and/or a 95% interval into ``Q(p)`` at ``p_levels``.

    Lognormal whenever the quantity is positive, which is the clinical default
    and the only branch a **log working scale** can consume (docs ch. 4b states
    the summary likelihood on the log scale). Linear Gaussian only for a
    genuinely signed quantity.

    Three inputs, any of which may be missing, so the branches are stated by what
    is actually available rather than by requiring all three:

    - **center and both bounds positive** -- two-sided log-sd, centered on the
      reported median. The ordinary case.
    - **no center, both bounds positive** -- a target that specifies a *range and
      no point estimate*. Common for qualitative mechanistic constraints ("tumour
      does not regress: fold change 1 to 5"). A 95% interval implies its own
      center under the same distribution used to expand it, so the geometric
      midpoint ``sqrt(lo*hi)`` is used. Before this branch existed these targets
      produced ``NaN + z*sigma`` and silently dropped out of any log-scale
      consumer.
    - **center positive, one bound usable** -- one-sided log-sd from whichever
      side is informative. A CI reported as touching zero is otherwise thrown
      onto the Gaussian branch, where it returns negative quantiles for a
      strictly positive quantity.

    A signed quantity (non-positive center) still gets the linear Gaussian, and
    its anchors can legitimately be negative. Those targets cannot feed a
    log-scale likelihood and the caller has to exclude them explicitly.
    """
    z = norm.ppf(p_levels)
    med_ok = bool(np.isfinite(median)) and median > 0
    lo_ok = bool(np.isfinite(lo)) and lo > 0
    hi_ok = bool(np.isfinite(hi)) and hi > 0

    center, sigma_log = None, None
    if lo_ok and hi_ok:
        center = median if med_ok else float(np.sqrt(lo * hi))
        sigma_log = (np.log(hi) - np.log(lo)) / (2 * _Z_95)
    elif med_ok and hi_ok:
        center, sigma_log = median, (np.log(hi) - np.log(median)) / _Z_95
    elif med_ok and lo_ok:
        center, sigma_log = median, (np.log(median) - np.log(lo)) / _Z_95
    if center is not None and sigma_log is not None and sigma_log > 0:
        return np.exp(np.log(center) + z * sigma_log)

    if np.isfinite([lo, hi]).all():
        sigma = (hi - lo) / (2 * _Z_95)
        return (median if np.isfinite(median) else 0.5 * (lo + hi)) + z * sigma
    return np.full(len(p_levels), median, dtype=np.float64)


def anchors_from_sources(
    obs_names: Sequence[str],
    samples: Sequence[Optional[np.ndarray]],
    medians: np.ndarray,
    ci95_lo: np.ndarray,
    ci95_hi: np.ndarray,
    *,
    observed_distributions: Optional[Sequence[Optional[object]]] = None,
    n: "Optional[Sequence[Optional[int]] | int]" = None,
    quantiles: Sequence[float] = (0.25, 0.5, 0.75),
    min_samples: int = 4,
    seed: int = 0,
) -> tuple[list, int]:
    """Assemble per-observable observed quantile anchors from resolved sources.

    Pure (no maple I/O): a caller resolves the maple targets into ``samples`` /
    ``observed_distributions`` / scalars and passes them here, so this is unit-
    testable in isolation and shared across projects.

    **Finite-sample handling.** A ``samples`` array is typically a large MC
    *population reconstruction* (e.g. 10k draws from a reported mean +/- SD), not
    the real ``n`` biological units, so its quantiles are denoised population
    values. To keep the observed anchor on the same finite-sample footing as the
    training summaries and the VPC null (which are ``n``-cohort statistics), when
    ``n`` is given the ``samples`` branch draws a single seeded ``n``-subsample
    from the pool and takes *its* quantiles, and the anchor grid is clipped to
    what ``n`` can resolve *and* support jointly (:func:`_resolvable_grid`).
    Quantiles use :data:`QUANTILE_METHOD`, not numpy's default.

    A reported ``observed_distribution`` is already an ``n``-statistic, so it is
    taken verbatim. **Two known holes in that branch**, both benign today because
    no target carries one yet, both real once they do: its quantiles were computed
    by the reporting paper's own software, so the estimator correction of
    :data:`QUANTILE_METHOD` does not apply to it and does not cancel against the
    training side; and its grid bypasses the density budget of
    :data:`MIN_PATIENTS_PER_ANCHOR`, so a paper reporting deciles from 9 patients
    would deliver a nine-anchor vector whose joint law is not Gaussian.

    Args:
        obs_names: observable ids.
        samples: per-observable population sample arrays (``None`` when the target
            declares no across-patient samples), aligned to ``obs_names``.
        medians, ci95_lo, ci95_hi: per-observable scalars for the ci95 fallback.
        observed_distributions: optional per-observable objects exposing
            ``_anchor_pairs()`` and ``feeds_population_spread`` (a maple
            ``ObservedDistribution``); ``None`` where absent. Takes precedence
            over ``samples`` when present.
        n: real biological sample size per observable (int broadcast, or a
            sequence, or ``None``). Drives the ``n``-subsample of the ``samples``
            pool and the grid clip. ``None`` keeps the legacy denoised
            full-pool quantile at the full grid.
        quantiles: default anchor grid for the ``samples`` / ci95 branches.
        min_samples: minimum finite samples to trust the empirical branch.
        seed: RNG seed for the per-observable ``n``-subsample (deterministic).

    Returns:
        ``(anchors, n_from_samples)`` -- ``anchors`` a list of
        :class:`ObservedAnchors` aligned to ``obs_names``; ``n_from_samples`` the
        count sourced from population samples.
    """
    grid = tuple(sorted(float(p) for p in quantiles))
    if not all(0.0 < p < 1.0 for p in grid):
        raise ValueError(f"quantiles must lie in (0, 1), got {quantiles}")
    n_obs = len(obs_names)
    if n is None or np.isscalar(n):
        n_per = [None if n is None else int(n)] * n_obs
    else:
        n_per = [None if v is None else int(v) for v in n]
        if len(n_per) != n_obs:
            raise ValueError(f"n has {len(n_per)} entries, need {n_obs}")

    rng = np.random.default_rng(seed)
    anchors: list = []
    n_from_samples = 0

    for i in range(n_obs):
        od = None if observed_distributions is None else observed_distributions[i]
        if od is not None:
            pairs = sorted(od._anchor_pairs())            # [(p, value), ...]
            p_levels = tuple(float(p) for p, _ in pairs)
            values = np.array([float(v) for _, v in pairs], dtype=np.float64)
            anchors.append(ObservedAnchors(
                p_levels, values, bool(od.feeds_population_spread),
                "observed_distribution",
            ))
            continue

        ni = n_per[i]
        gi = _resolvable_grid(grid, ni)
        s = samples[i]
        if s is not None:
            arr = np.asarray(s, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if arr.size >= min_samples:
                if ni is not None and ni >= 2 and arr.size > ni:
                    # Observed at real n: one seeded n-subsample of the pool, so
                    # the anchor carries the same finite-sample law as training.
                    draw = rng.choice(arr, size=int(ni), replace=False)
                    vals = np.quantile(draw, gi, method=QUANTILE_METHOD)
                else:
                    vals = np.quantile(arr, gi, method=QUANTILE_METHOD)  # n unknown / pool <= n
                anchors.append(ObservedAnchors(gi, vals, True, "samples"))
                n_from_samples += 1
                continue

        anchors.append(ObservedAnchors(
            gi, _ci95_expand(medians[i], ci95_lo[i], ci95_hi[i], gi),
            False, "ci95",
        ))

    return anchors, n_from_samples


def cohort_quantiles(
    x_flat: np.ndarray,
    n_cohorts: int,
    n_cohort: int,
    anchor_p_levels: Sequence[Sequence[float]],
    min_patients: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cohort empirical ``Q(p)`` at each observable's anchor levels.

    The training-summary generalization of the legacy ``[median, IQR]`` cohort
    reduction: instead of two moments per observable it emits the quantile
    *values* at that observable's anchor p-levels, concatenated across
    observables (ragged widths allowed).

    Uses :data:`QUANTILE_METHOD`, the same estimator
    :func:`anchors_from_sources` applies to the observed anchor. That pairing is
    the point: numpy's default shrinks the sample IQR by 15-25% at n=6-12, and a
    summary statistic computed one way on the observed side and another way here
    is a matched-footing failure that lands straight on ``sigma_u_hat``.

    Args:
        x_flat: ``(n_cohorts * n_cohort, n_obs)`` cohort-major patient matrix.
        anchor_p_levels: length ``n_obs``; the p-levels for each observable.
        min_patients: minimum finite patients for a cohort to be valid.

    Returns:
        ``(summary, valid)`` -- ``summary`` is ``(n_cohorts, sum_i len(anchor_i))``
        in observable-then-anchor order; ``valid`` is ``(n_cohorts,)``.
    """
    n_obs = x_flat.shape[1]
    if len(anchor_p_levels) != n_obs:
        raise ValueError(
            f"anchor_p_levels has {len(anchor_p_levels)} entries, "
            f"x_flat has {n_obs} observables"
        )
    x = x_flat.reshape(n_cohorts, n_cohort, n_obs).astype(np.float64)
    x[np.isinf(x)] = np.nan
    finite_per_patient = np.isfinite(x).all(axis=2)          # (C, K)
    valid = finite_per_patient.sum(axis=1) >= min_patients   # (C,)

    cols = []
    with np.errstate(all="ignore"):
        for j in range(n_obs):
            ps = np.asarray(anchor_p_levels[j], dtype=np.float64)
            # nanpercentile over the patient axis -> (len(ps), C); transpose.
            # Same estimator as the observed anchor (QUANTILE_METHOD) -- a
            # mismatch here is a matched-footing failure at the summary level.
            q = np.nanpercentile(
                x[:, :, j], ps * 100.0, axis=1, method=QUANTILE_METHOD
            )  # (len(ps), C)
            cols.append(np.atleast_2d(q).T)                       # (C, len(ps))
    summary = np.hstack(cols)                                     # (C, sum len)
    return summary, valid

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
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.stats import norm

__all__ = ["ObservedAnchors", "anchors_from_sources", "cohort_quantiles"]

_Z_95 = 1.959963984540054  # Phi^-1(0.975)


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


def _resolvable_grid(quantiles, n) -> tuple:
    """Drop anchors more extreme than a sample of size ``n`` can resolve.

    Keep ``p`` in ``[1/(n+1), n/(n+1)]`` -- the range a rank statistic of ``n``
    draws actually informs; a 6-donor study cannot resolve ``Q(0.95)``. The
    median always survives (for ``n >= 1``). ``n=None`` leaves the grid untouched.
    """
    if n is None:
        return tuple(quantiles)
    lo, hi = 1.0 / (n + 1.0), n / (n + 1.0)
    keep = tuple(p for p in quantiles if lo <= p <= hi)
    return keep if keep else (0.5,)


def _ci95_expand(median, lo, hi, p_levels) -> np.ndarray:
    """Expand a median + 95% interval into ``Q(p)`` at ``p_levels``.

    Lognormal when all three are positive (the clinical default), else a linear
    Gaussian around the median. Generalizes the legacy IQR-from-ci95 fallback to
    an arbitrary anchor grid.
    """
    z = norm.ppf(p_levels)
    if np.isfinite([median, lo, hi]).all() and median > 0 and lo > 0 and hi > 0:
        sigma_log = (np.log(hi) - np.log(lo)) / (2 * _Z_95)
        return np.exp(np.log(median) + z * sigma_log)
    if np.isfinite([lo, hi]).all():
        sigma = (hi - lo) / (2 * _Z_95)
        return median + z * sigma
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
    what ``n`` can resolve (:func:`_resolvable_grid`). A reported
    ``observed_distribution`` is already an ``n``-statistic, so it is taken
    verbatim.

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
                    vals = np.quantile(draw, gi)
                else:
                    vals = np.quantile(arr, gi)          # n unknown / pool <= n
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
            q = np.nanpercentile(x[:, :, j], ps * 100.0, axis=1)  # (len(ps), C)
            cols.append(np.atleast_2d(q).T)                       # (C, len(ps))
    summary = np.hstack(cols)                                     # (C, sum len)
    return summary, valid

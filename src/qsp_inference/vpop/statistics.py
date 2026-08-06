"""Reported statistics as functionals of the predicted cloud. eq:stat, eq:smoothq.

A cohort's printed statistic came from ``n_c`` patients, so ``tau`` predicts the
expectation of that statistic, not the population functional.

Order statistics get it in closed form. Sort ``n`` draws from any distribution and
the rank of the k-th in it is ``Beta(k, n-k+1)`` whatever the shape, so the
expectation is a Beta-weighted average of the cloud. That covers quantile rows,
and an interquartile range as the difference of two of them.

Moments do not: ``E[s]/sigma`` runs from 0.77 to 0.96 at ``n=8`` depending on
shape, and logging does not stabilise it. Those rows are bootstrapped at ``phi``
on a frozen design instead.

Requires ``jax_enable_x64``. The kernel differences a CDF across cloud members, so
at ``N`` in the hundreds of thousands each mass is order ``1e-5`` and float32
cumulative sums lose it.
"""

from __future__ import annotations

import math
from typing import Callable, Dict

import jax
import jax.numpy as jnp
from jax.scipy.special import betainc

__all__ = [
    "QUANTILE_CONVENTIONS",
    "order_statistic_mass",
    "quantile_mass",
    "expected_quantile",
    "extreme_row",
    "mean_row",
    "iqr_row",
    "bootstrap_design",
    "bootstrap_row",
    "sd_row",
    "se_row",
]

# Which order statistic an estimator selects for the p-quantile of n points.
# Named for the Hyndman-Fan types. No paper records which its software used.
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


def _require_x64() -> None:
    """Fail loudly rather than degrade. ``submodel.inference`` turns x64 off globally."""
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "vpop.statistics needs jax_enable_x64. The Beta kernel differences a "
            "CDF across cloud members, so each mass is order 1/N and float32 "
            "loses it. Set jax.config.update('jax_enable_x64', True)."
        )


def order_statistic_mass(w, kappa: int, n: int):
    """``Beta(kappa, n-kappa+1)`` mass on each cloud member's slice of [0, 1].

    ``w`` is the eligibility weight per member, in sorted-value order. Member ``i``
    spans cumulative weight ``[e[i-1], e[i]]``, so its mass is the CDF's increment
    across that span. Sums to one without normalising.
    """
    _require_x64()
    w = jnp.asarray(w)
    # Clipped because the edges are cumulative probabilities by definition but
    # not by arithmetic: cumsum and sum reduce in different orders, so the last
    # edge can land an ulp above 1, and betainc is nan just outside [0, 1] rather
    # than saturating. Whether it does depends on the platform's summation order,
    # which is how this passed on arm64 and returned nan on x86.
    edges = jnp.concatenate([jnp.zeros(1, w.dtype), jnp.cumsum(w) / jnp.sum(w)])
    edges = jnp.clip(edges, 0.0, 1.0)
    return jnp.diff(betainc(float(kappa), float(n - kappa + 1), edges))


def quantile_mass(w, p: float, n: int, convention: str = "type7"):
    """The Beta weights ``E[q_p]`` applies to the sorted cloud.

    A non-integer order statistic is what the estimator interpolates between, and
    the estimator is linear in the two, so the two Beta masses mix with the same
    weights. Split out from :func:`expected_quantile` because it depends only on
    ``(w, p, n, convention)``, so rows sharing those share the vector.
    """
    h = QUANTILE_CONVENTIONS[convention](p, n)
    h = min(max(h, 1.0), float(n))
    lo = int(h // 1)
    frac = h - lo

    mass = (1.0 - frac) * order_statistic_mass(w, lo, n)
    if frac > 0:
        mass = mass + frac * order_statistic_mass(w, min(lo + 1, n), n)
    return mass


def expected_quantile(x_sorted, w, p: float, n: int, convention: str = "type7",
                      mass=None):
    """``E[q_p]`` over an ``n``-sample from the weighted cloud. eq:smoothq.

    ``x_sorted`` is ascending. ``mass`` reuses a vector from :func:`quantile_mass`.
    """
    if mass is None:
        mass = quantile_mass(w, p, n, convention)
    return jnp.asarray(x_sorted) @ mass


def extreme_row(x_sorted, w, n: int, upper: bool):
    """A reported minimum or maximum: order statistic 1 or ``n``, exactly.

    A printed range is the pair, and each endpoint is an order statistic like any
    other, so no convention applies and nothing is interpolated. The expectation
    still reads the whole cloud, but the Beta mass concentrates on one tail, which
    is where the surrogate is least accurate.
    """
    return jnp.asarray(x_sorted) @ order_statistic_mass(w, n if upper else 1, n)


def mean_row(x_sorted, w):
    """A reported mean. ``E[sample mean] = population mean``, so no correction."""
    w = jnp.asarray(w)
    return jnp.sum(w * jnp.asarray(x_sorted)) / jnp.sum(w)


def iqr_row(x_sorted, w, n: int, convention: str = "type7", log: bool = False):
    """A reported interquartile range, exact by linearity of the expectation.

    ``log=True`` is not the expectation of the reported log: that needs the joint
    law of two order statistics. Use it only where the row was printed as a log.
    """
    hi = expected_quantile(x_sorted, w, 0.75, n, convention)
    lo = expected_quantile(x_sorted, w, 0.25, n, convention)
    width = hi - lo
    return jnp.log(jnp.clip(width, 1e-30, None)) if log else width


def bootstrap_design(key, n: int, n_boot: int = 400):
    """Frozen uniforms for a moment row, shape ``(n_boot, n)``.

    Uniforms rather than indices: the flat fit runs a smaller cloud than the
    population fit, and a JAX gather clamps an out-of-range index instead of
    failing, so indices drawn for one would be silently wrong in the other.
    """
    return jax.random.uniform(key, (n_boot, n))


def bootstrap_row(x_sorted, w, u, fn):
    """``E*[fn]`` over the frozen design. ``fn(values, weights)`` is the statistic.

    Eligibility rides as a weight on the replicates rather than as the sampling
    probability. Sampling proportional to ``w`` would make the design depend on
    ``phi`` and the row would stop being smooth in it.
    """
    _require_x64()
    x_sorted, w = jnp.asarray(x_sorted), jnp.asarray(w)
    idx = jnp.minimum((u * x_sorted.shape[0]).astype(jnp.int32),
                      x_sorted.shape[0] - 1)
    return jnp.mean(jax.vmap(fn)(x_sorted[idx], w[idx]))


def _weighted_sd(v, ww):
    """Unbiased weighted sample SD, reducing to the ``n-1`` form at equal weights."""
    sw = jnp.sum(ww)
    m = jnp.sum(ww * v) / sw
    denom = sw - jnp.sum(ww ** 2) / sw
    return jnp.sqrt(jnp.sum(ww * (v - m) ** 2) / denom)


def sd_row(x_sorted, w, u, log: bool = False):
    """A reported standard deviation, as ``E*[s]`` over the frozen design.

    No closed form: the correction depends on the shape of the pushforward, which
    is what ``phi`` controls, so it is neither distribution-free nor a fixed offset.
    """
    s = bootstrap_row(x_sorted, w, u, _weighted_sd)
    return jnp.log(jnp.clip(s, 1e-30, None)) if log else s


def se_row(x_sorted, w, u, n: int, log: bool = False):
    """A reported standard error of a mean, as ``E*[s/sqrt(n)]``.

    The expectation of the estimator the source printed, not the sampling spread
    itself. The two differ at ``O(1/n)`` and only the first is ``E[printed]``.
    """
    s = bootstrap_row(x_sorted, w, u, _weighted_sd) / jnp.sqrt(float(n))
    return jnp.log(jnp.clip(s, 1e-30, None)) if log else s
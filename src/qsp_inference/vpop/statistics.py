"""Reported statistics as functionals of the predicted cloud. eq:stat, eq:smoothq.

A cohort's printed quantile came from ``n_c`` patients, so ``tau`` predicts the
expectation of that statistic and not the population quantile. Sort ``n`` draws
from any distribution and the rank of the k-th in it is ``Beta(k, n-k+1)``,
whatever the shape, so that expectation is a Beta-weighted average of the cloud.

The draft writes the kernel as a density at each member's midpoint. Here it is
the Beta mass on each member's slice of [0, 1], which is the exact form and sums
to one without normalising.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np
from scipy.stats import beta as _beta

__all__ = [
    "QUANTILE_CONVENTIONS",
    "order_statistic_mass",
    "expected_quantile",
]

# Which order statistic an estimator selects for the p-quantile of n points.
# Named for the Hyndman-Fan types. No paper records which its software used.
QUANTILE_CONVENTIONS: Dict[str, Callable[[float, int], float]] = {
    "type7": lambda p, n: (n - 1) * p + 1,  # R, numpy, pandas (default)
    "type6": lambda p, n: (n + 1) * p,      # SPSS, Minitab
    "type4": lambda p, n: n * p,            # linear interpolation of the ecdf
}


def order_statistic_mass(w: np.ndarray, kappa: int, n: int) -> np.ndarray:
    """``Beta(kappa, n-kappa+1)`` mass on each cloud member's slice of [0, 1].

    ``w`` is the eligibility weight per member, in sorted-value order. Member ``i``
    spans cumulative weight ``[e[i-1], e[i]]``, so its mass is the Beta CDF's
    increment across that span.
    """
    w = np.asarray(w, dtype=float)
    edges = np.concatenate([[0.0], np.cumsum(w) / w.sum()])
    cdf = _beta.cdf(edges, kappa, n - kappa + 1)
    return np.diff(cdf)


def expected_quantile(
    x_sorted: np.ndarray,
    w: np.ndarray,
    p: float,
    n: int,
    convention: str = "type7",
) -> float:
    """``E[q_p]`` over an ``n``-sample from the weighted cloud. eq:smoothq.

    ``x_sorted`` is ascending. A non-integer order statistic is what the estimator
    interpolates between, and the estimator is linear in the two, so the two Beta
    masses mix with the same weights.
    """
    h = QUANTILE_CONVENTIONS[convention](p, n)
    h = min(max(h, 1.0), float(n))
    lo = int(np.floor(h))
    frac = h - lo

    mass = (1.0 - frac) * order_statistic_mass(w, lo, n)
    if frac > 0:
        mass = mass + frac * order_statistic_mass(w, min(lo + 1, n), n)
    return float(np.asarray(x_sorted, dtype=float) @ mass)
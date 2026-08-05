"""eq:ratio as a gate. Run before the fit.

``V_B`` takes no reported uncertainty anywhere, so every error bar rests on the
predicted cloud. If that cloud is too narrow every ``V_B`` is too small, the
likelihood overrides the prior, and nothing downstream reports it.

The check is the prior-predictive residual on the rows that carry width: what
``phi_0`` predicts for an ``se``, ``sd`` or ``iqr`` row against what the source
printed. It is the fit's own residual evaluated at ``phi_0`` rather than at
``phi``, so it costs one forward pass and no fit.

A cloud that is too narrow shows as a positive shortfall: the source reports more
spread than the model can produce. The response is to fix the model, not to widen
``V_B``, since widening hides the thing being measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["WidthRow", "WidthGate", "width_gate"]


@dataclass(frozen=True)
class WidthRow:
    """One scale row at ``phi_0``, against the number the source printed."""

    label: str
    cohort_id: str
    n: int
    observed: float
    predicted: float
    sd: float
    logged: bool

    @property
    def shortfall(self) -> float:
        """``obs - pred``. On a logged row this is the log widening required."""
        return self.observed - self.predicted

    @property
    def z(self) -> float:
        return self.shortfall / self.sd if self.sd > 0 else np.nan


@dataclass(frozen=True)
class WidthGate:
    """Every scale row, and whether the model can supply what they ask for."""

    rows: Tuple[WidthRow, ...]
    budget: float

    @property
    def shortfalls(self) -> np.ndarray:
        return np.array([r.shortfall for r in self.rows])

    @property
    def z(self) -> np.ndarray:
        return np.array([r.z for r in self.rows])

    @property
    def mean_shortfall(self) -> float:
        return float(np.mean(self.shortfalls)) if self.rows else 0.0

    @property
    def prior_sd(self) -> float:
        """The mean shortfall in units of the widening the model has available."""
        return abs(self.mean_shortfall) / self.budget

    @property
    def max_abs_z(self) -> float:
        return float(np.nanmax(np.abs(self.z))) if self.rows else 0.0

    @property
    def verdict(self) -> str:
        if not self.rows:
            return "no width rows in this problem"
        if self.prior_sd > 2.0:
            return ("unreachable: the model cannot widen this far without fighting "
                    "its own prior, which is a model problem and not one to fix by "
                    "widening V")
        if self.max_abs_z > 3.0:
            return ("reachable, but some rows sit far out relative to their error "
                    "bar and will dominate the fit for omega; check them first")
        return "reachable"


def width_gate(
    problem,
    cov,
    observed: Sequence,
    mu_0,
    omega_0,
    *,
    tau_s: float,
    sigma_b: float,
    log_R_0=None,
    beta_free=None,
) -> WidthGate:
    """Every scale row's prior-predictive residual at ``phi_0``, with its budget.

    ``budget`` is the prior sd of ``s + b_1``. Both enter a scale row additively
    and are the only two ways the model can widen, so a shortfall is read against
    their sum rather than against either alone.
    """
    import jax.numpy as jnp

    from qsp_inference.vpop.predict import tau_all

    zero = jnp.zeros(problem.mech.Z.shape[1])
    if beta_free is None:
        beta_free = jnp.zeros(problem.mech.beta_species.shape[0])
    pred = tau_all(mu_0, omega_0, zero, zero, beta_free, problem.plans,
                   problem.specs_by_cohort, problem.refs, problem.mech,
                   log_R=log_R_0, designs=problem.designs,
                   mass_table=problem.mass_table,
                   elig_fn=problem.elig_fn, elig_at=problem.elig_at)

    rows = []
    for plan, tau_B, T_B, V in zip(problem.plans, pred, observed, cov.V):
        sd = np.sqrt(np.diag(np.asarray(V)))
        tau_B, T_B = np.asarray(tau_B), np.asarray(T_B)
        at = 0
        for cohort_id in plan.cohort_ids:
            for spec in problem.specs_by_cohort[cohort_id]:
                if spec.is_scale:
                    rows.append(WidthRow(
                        label=spec.label, cohort_id=cohort_id, n=spec.n,
                        observed=float(T_B[at]), predicted=float(tau_B[at]),
                        sd=float(sd[at]), logged=spec.log,
                    ))
                at += 1

    return WidthGate(tuple(rows), float(np.hypot(tau_s, sigma_b)))

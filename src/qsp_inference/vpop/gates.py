"""eq:ratio as a gate. Run before the fit.

``V_B`` takes no reported uncertainty anywhere, so every error bar rests on the
predicted cloud. If that cloud is too narrow every ``V_B`` is too small, the
likelihood overrides the prior, and nothing downstream reports it.

The check is the prior-predictive residual on the rows that carry width, against
what the source printed. It is the fit's own residual evaluated at ``phi_0``
rather than at ``phi``, so it costs one forward pass and no fit.

A cloud that is too narrow shows as a positive shortfall: the source reports more
spread than the model can produce. The response is to fix the model, not to widen
``V_B``, since widening hides the thing being measured.

Which rows carry width is not the same question as which rows ``SCALE_STATS``
names. A source printing an ``iqr`` gets a scale row; a source printing ``q25``
and ``q75`` gets two location rows carrying the same width, and reading only the
first set makes the gate a sample of editorial style. So a target's widest
symmetric quantile pair is differenced into a width here
(:func:`paired_width_rows`) and enters the gate beside the ``se``, ``sd`` and
``iqr`` rows. The pair's own covariance is already in ``V_B``, since both its
rows belong to one cohort, so the derived row needs no new data and no new
assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["WidthRow", "WidthGate", "paired_width_rows", "width_gate"]


@dataclass(frozen=True)
class WidthRow:
    """One width at ``phi_0``, against the number the source printed.

    ``derived`` marks a row differenced from a quantile pair rather than printed
    as a width. It is the same quantity either way; only the corpus's typography
    differs, and the flag is carried so a listing can say which is which.
    """

    label: str
    cohort_id: str
    n: int
    observed: float
    predicted: float
    sd: float
    logged: bool
    derived: bool = False

    @property
    def shortfall(self) -> float:
        """``obs - pred``. On a logged row this is the log widening required."""
        return self.observed - self.predicted

    @property
    def z(self) -> float:
        return self.shortfall / self.sd if self.sd > 0 else np.nan


@dataclass(frozen=True)
class WidthGate:
    """Every width row, printed or derived, and whether the model can supply what
    they ask for."""

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


def paired_width_rows(specs, tau_B, T_B, V, *, offset: int = 0) -> list:
    """One cohort's quantile pairs, differenced into widths. ``offset`` is where
    its rows start in the block.

    A target contributes at most one row, from its widest symmetric pair, so a
    source printing ``q25``, ``q50`` and ``q75`` is not read as three overlapping
    widths of the same patients. A target that printed an ``iqr`` outright is
    skipped: it already has a scale row and the two would be the same number.

    The width is ``q_hi - q_lo`` on the raw scale, logged to match the printed
    scale rows, which is what makes the shortfall comparable to the budget. Its
    error bar is the contrast variance ``V[hi,hi] + V[lo,lo] - 2 V[lo,hi]``,
    already exact in ``V_B`` because both rows are one cohort's, carried onto the
    log scale by the delta method at the predicted width.
    """
    at_of, printed_iqr, meta = {}, set(), {}
    for j, spec in enumerate(specs):
        if spec.stat == "quantile":
            at_of[(spec.target_id, round(float(spec.p), 6))] = offset + j
        elif spec.stat == "iqr":
            printed_iqr.add(spec.target_id)
        meta[spec.target_id] = spec

    V = np.asarray(V)
    out = []
    for tid in sorted({t for t, _ in at_of} - printed_iqr):
        ps = sorted(p for t, p in at_of if t == tid)
        pair = next(((p, round(1.0 - p, 6)) for p in ps
                     if p < 0.5 and round(1.0 - p, 6) in ps), None)
        if pair is None:
            continue
        lo, hi = at_of[(tid, pair[0])], at_of[(tid, pair[1])]
        w_pred, w_obs = float(tau_B[hi] - tau_B[lo]), float(T_B[hi] - T_B[lo])
        if w_obs <= 0:
            raise ValueError(
                f"{tid}: printed q{pair[1]:g} <= q{pair[0]:g}, so its width is not "
                f"positive. That is a corpus error, not a narrow cloud."
            )
        if w_pred <= 0:
            raise ValueError(
                f"{tid}: predicted q{pair[1]:g} <= q{pair[0]:g}. Expected order "
                f"statistics of one sorted cloud are monotone in p, so this is a "
                f"bug in the prediction rather than a fact about the target."
            )
        var = float(V[hi, hi] + V[lo, lo] - 2.0 * V[lo, hi])
        spec = meta[tid]
        out.append(WidthRow(
            label=f"{tid}/q{pair[0]:g}-q{pair[1]:g}", cohort_id=spec.cohort_id,
            n=spec.n, observed=float(np.log(w_obs)),
            predicted=float(np.log(w_pred)),
            sd=float(np.sqrt(max(var, 0.0)) / w_pred), logged=True, derived=True,
        ))
    return out


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
    """Every width row's prior-predictive residual at ``phi_0``, with its budget.

    Printed ``se``/``sd``/``iqr`` rows and the quantile pairs of
    :func:`paired_width_rows`, which carry the same quantity in a form
    ``SCALE_STATS`` does not name.

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
        V = np.asarray(V)
        sd = np.sqrt(np.diag(V))
        tau_B, T_B = np.asarray(tau_B), np.asarray(T_B)
        at = 0
        for cohort_id in plan.cohort_ids:
            specs = problem.specs_by_cohort[cohort_id]
            for j, spec in enumerate(specs):
                if spec.is_scale:
                    rows.append(WidthRow(
                        label=spec.label, cohort_id=cohort_id, n=spec.n,
                        observed=float(T_B[at + j]), predicted=float(tau_B[at + j]),
                        sd=float(sd[at + j]), logged=spec.log,
                    ))
            rows.extend(paired_width_rows(specs, tau_B, T_B, V, offset=at))
            at += len(specs)

    return WidthGate(tuple(rows), float(np.hypot(tau_s, sigma_b)))

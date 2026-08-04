"""``tau_B(phi)``: the model's prediction of every reported number. eq:crn to eq:stat.

Ported from ``examples/toy_population_fit.py``, de-globalised. The chain is

    phi -> patients -> species -> readouts -> measurement map -> rows

and only the species step needs the emulator, which arrives as ``g_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp

from qsp_inference.vpop.rows import tau_row

__all__ = ["Mechanism", "patient_cloud", "readout_cloud", "apply_map",
           "reference_levels", "cohort_cloud", "block_weights", "tau_rows",
           "tau_block", "tau_all"]

#: ``(vartheta, scenario) -> (N, Q)`` raw positive species. The emulator predicts
#: log species, so a wrapper exponentiates: beta is multiplicative on species.
GFn = Callable[[jnp.ndarray, int], jnp.ndarray]

#: ``(S, N, Q), (A,) -> (S, N, M)`` log-scale readouts, in ``Mechanism.readouts``
#: order. The second argument is ``log R``, the declared assay conversions.
HFn = Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]

#: ``((N, M) mapped readouts, cohort) -> (N,)`` weights. eq:elig reads x-tilde, so
#: it moves with ``a``, ``b`` and ``beta`` and is recomputed at every ``phi``.
EligFn = Callable[[jnp.ndarray, str], jnp.ndarray]


@dataclass(frozen=True)
class Mechanism:
    """Everything ``tau`` needs that does not move with ``phi``."""

    L_R: jnp.ndarray            # (P, P), R = L_R L_R'
    z: jnp.ndarray              # (N, P) frozen common random numbers, eq:crn
    Z: jnp.ndarray              # (M, D) readout design, eq:disc
    readouts: Tuple[str, ...]   # M readout names, the row order of Z
    n_species: int              # Q
    n_scenarios: int            # S
    beta_species: jnp.ndarray   # indices of species carrying a free beta
    g_fn: GFn
    h_fn: HFn

    @property
    def n_readouts(self) -> int:
        return len(self.readouts)


def patient_cloud(mu, omega, mech: Mechanism) -> jnp.ndarray:
    """eq:crn. ``(N, P)`` log-parameters, smooth in ``(mu, omega)`` at frozen ``z``."""
    return mu[None, :] + mech.z @ (mech.L_R.T * omega[None, :])


def readout_cloud(mu, omega, beta_free, mech: Mechanism, log_R=None) -> jnp.ndarray:
    """``(S, N, M)`` readouts before the measurement map. eq:crn, eq:mech, eq:readout.

    Every scenario, since a fold change contrasts two timepoints of the same
    patient and ``h_r`` is given the whole set. ``beta`` multiplies species, so it
    enters upstream of ``h_r`` and propagates through the composition on its own.
    ``log R`` enters inside ``h_r``, being indexed by readout and species at once.
    """
    vartheta = patient_cloud(mu, omega, mech)
    y = jnp.stack([mech.g_fn(vartheta, s) for s in range(mech.n_scenarios)])
    beta = jnp.zeros(mech.n_species).at[mech.beta_species].set(beta_free)
    return mech.h_fn(y * jnp.exp(beta)[None, None, :], log_R)


def apply_map(x, a, b, c_row, Z) -> jnp.ndarray:
    """eq:disc for one cohort: ``kappa_r (x - c) + c + gamma_r``.

    ``c`` is that cohort's own level, so the offset applied is
    ``gamma_r + c_rc (1 - kappa_r)`` and varies by study at no parameter cost.
    Pivoting there is what keeps ``kappa`` a pure spread term: against a global
    pivot it moved location rows by up to 84 sd per unit of ``log kappa``.
    """
    return jnp.exp(Z @ b) * (x - c_row) + c_row + (Z @ a)


def _sorted(column, w):
    """Ascending values and their weights: every row functional wants both."""
    order = jnp.argsort(column)
    return column[order], w[order]


def _median(x_sorted, w):
    return x_sorted[jnp.searchsorted(jnp.cumsum(w) / jnp.sum(w), 0.5)]


def _enrolled_at(group, elig_at: Optional[Mapping[str, str]]) -> str:
    """The cohort a joint group's criterion was applied at. Never inferred."""
    if not group.is_joint:
        return group.cohort_ids[0]          # the only member
    source = (elig_at or {}).get(group.block_id)
    if source is None:
        raise ValueError(
            f"block {group.block_id}: {', '.join(group.cohort_ids)} share a draw and "
            f"so share one eligibility vector. Name the cohort the criterion "
            f"enrolled at in elig_at; member order is not it."
        )
    if source not in group.cohort_ids:
        raise ValueError(
            f"block {group.block_id}: elig_at names {source!r}, not a member"
        )
    return source


def block_weights(x_of: Mapping[str, jnp.ndarray], plan,
                  elig_fn: Optional[EligFn] = None,
                  elig_at: Optional[Mapping[str, str]] = None) -> Dict[str, jnp.ndarray]:
    """eq:elig per cohort, one evaluation per joint draw group.

    Eligibility is a property of the patient, fixed at enrolment, so a joint group
    takes one vector read off the cohort ``elig_at`` names. Cohorts joined only by
    a shared row are disjoint people and each reads its own.
    """
    n = next(iter(x_of.values())).shape[0]
    if elig_fn is None:
        return {c: jnp.ones(n) for c in x_of}

    out: Dict[str, jnp.ndarray] = {}
    for group in plan.groups:
        source = _enrolled_at(group, elig_at)
        w = elig_fn(x_of[source], source)
        for cohort_id in group.cohort_ids:
            out[cohort_id] = w
    return out


def reference_levels(mu_0, omega_0, mech: Mechanism, plans, scenario_of, *,
                     log_R_0=None,
                     elig_fn: Optional[EligFn] = None,
                     elig_at: Optional[Mapping[str, str]] = None,
                     ) -> Dict[str, jnp.ndarray]:
    """``c_rc``: each cohort's own median readout at the plug-in. ``(M,)`` per cohort.

    Fixed once and held there. eq:disc pivots on it, so letting it move with ``phi``
    would make ``kappa`` rescale about a moving point instead of about the study's
    own level. The plug-in is the no-discrepancy point, where the map is the
    identity and ``x-tilde = x``, so no reference is needed to build one. ``log R``
    is a declared conversion rather than discrepancy, so it sits at its prior
    centre here, not at zero.
    """
    x_all = readout_cloud(mu_0, omega_0, jnp.zeros(mech.beta_species.shape[0]),
                          mech, log_R_0)
    out: Dict[str, jnp.ndarray] = {}
    for plan in plans:
        x_of = {c: x_all[scenario_of[c]] for c in plan.cohort_ids}
        w_of = block_weights(x_of, plan, elig_fn, elig_at)
        for cohort_id in plan.cohort_ids:
            out[cohort_id] = jnp.stack([
                _median(*_sorted(x_of[cohort_id][:, m], w_of[cohort_id]))
                for m in range(mech.n_readouts)
            ])
    return out


def cohort_cloud(x_all, cohort_id: str, a, b, refs, mech: Mechanism, scenario_of):
    """One cohort's mapped readouts, ``(N, M)``: eq:disc at that cohort's scenario."""
    return apply_map(x_all[scenario_of[cohort_id]], a, b, refs[cohort_id], mech.Z)


def tau_rows(specs, x_cohort, w, mech: Mechanism, designs=None,
             uniform: bool = False) -> jnp.ndarray:
    """One cohort's rows, ``(K_c,)``, in the order the source printed them.

    ``uniform`` says every patient carries weight one, which is the case wherever
    no eligibility criterion is declared. Sorting permutes ``w`` differently per
    readout, so only then is the Beta mass the same vector for every row sharing
    ``(p, n, convention)``, and only then can it be computed once.
    """
    from qsp_inference.vpop.statistics import quantile_mass

    index_of = {r: i for i, r in enumerate(mech.readouts)}
    columns = sorted({index_of[spec.target_id] for spec in specs})
    order = jnp.argsort(x_cohort[:, jnp.asarray(columns)], axis=0)
    at = {c: k for k, c in enumerate(columns)}

    marginal: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {}
    for spec in specs:
        if spec.target_id not in marginal:
            col = index_of[spec.target_id]
            idx = order[:, at[col]]
            marginal[spec.target_id] = (x_cohort[idx, col], w[idx])

    masses: Dict[Tuple[float, int, str], jnp.ndarray] = {}
    out = []
    for spec in specs:
        x_sorted, w_sorted = marginal[spec.target_id]
        mass = None
        if uniform and spec.stat == "quantile":
            key = (spec.p, spec.n, spec.convention)
            if key not in masses:
                masses[key] = quantile_mass(w_sorted, spec.p, spec.n, spec.convention)
            mass = masses[key]
        out.append(tau_row(spec, x_sorted, w_sorted,
                           (designs or {}).get(spec.label), mass))
    return jnp.stack(out)


def tau_block(x_all, plan, specs_by_cohort, a, b, refs, mech: Mechanism, scenario_of,
              *, designs=None, elig_fn: Optional[EligFn] = None,
              elig_at: Optional[Mapping[str, str]] = None) -> jnp.ndarray:
    """``tau_B``. Cohorts concatenate in ``plan.cohort_ids`` order, matching ``V_B``."""
    x_of = {c: cohort_cloud(x_all, c, a, b, refs, mech, scenario_of)
            for c in plan.cohort_ids}
    w_of = block_weights(x_of, plan, elig_fn, elig_at)
    return jnp.concatenate([
        tau_rows(specs_by_cohort[c], x_of[c], w_of[c], mech, designs,
                 uniform=elig_fn is None)
        for c in plan.cohort_ids
    ])


def tau_all(mu, omega, a, b, beta_free, plans, specs_by_cohort, refs,
            mech: Mechanism, scenario_of, *, log_R=None, designs=None,
            elig_fn: Optional[EligFn] = None,
            elig_at: Optional[Mapping[str, str]] = None) -> Sequence[jnp.ndarray]:
    """Every block's prediction from one ``phi``, on one pass through the emulator."""
    x_all = readout_cloud(mu, omega, beta_free, mech, log_R)
    return [tau_block(x_all, plan, specs_by_cohort, a, b, refs, mech, scenario_of,
                      designs=designs, elig_fn=elig_fn, elig_at=elig_at)
            for plan in plans]

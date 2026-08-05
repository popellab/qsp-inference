"""``tau_B(phi)``: the model's prediction of every reported number. eq:crn to eq:stat.

Ported from ``examples/toy_population_fit.py``, de-globalised. The chain is

    phi -> patients -> species -> readouts -> measurement map -> rows

and only the species step needs the emulator, which arrives as ``g_fn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from qsp_inference.vpop.rows import tau_row

__all__ = ["Mechanism", "patient_cloud", "readout_cloud", "apply_map",
           "reference_levels", "cohort_cloud", "block_weights",
           "quantile_mass_table", "cohort_columns", "tau_rows",
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
    # Readouts the reduce already composed. They are not species and not every arm
    # emits one, so they travel beside g_fn's fixed-width block rather than in it.
    extra_fn: Optional[Callable] = None
    extra_at: Tuple[Tuple[int, str], ...] = ()

    @property
    def n_readouts(self) -> int:
        return len(self.readouts)

    def __post_init__(self):
        # Scaling column j of L_R' by omega_j commutes with the row sum, so
        # z (L_R' * omega) is (z L_R') * omega and the matmul is phi-free.
        # Held here because eq:crn is otherwise an (N,P)x(P,P) product per gradient.
        object.__setattr__(self, "zL",
                           jnp.asarray(self.z) @ jnp.asarray(self.L_R).T)


def patient_cloud(mu, omega, mech: Mechanism) -> jnp.ndarray:
    """eq:crn. ``(N, P)`` log-parameters, smooth in ``(mu, omega)`` at frozen ``z``."""
    return mu[None, :] + mech.zL * omega[None, :]


def readout_cloud(mu, omega, beta_free, mech: Mechanism, log_R=None) -> jnp.ndarray:
    """``(N, M)`` readouts before the measurement map. eq:crn, eq:mech, eq:readout.

    ``h_r`` is handed every scenario, since a fold change contrasts two timepoints
    of the same patient, but returns one column: a readout belongs to one cohort
    and so to one scenario. ``beta`` multiplies species, so it enters upstream of
    ``h_r`` and propagates through the composition on its own. ``log R`` enters
    inside ``h_r``, being indexed by readout and species at once.
    """
    vartheta = patient_cloud(mu, omega, mech)
    y = jnp.stack([mech.g_fn(vartheta, s) for s in range(mech.n_scenarios)])
    beta = jnp.zeros(mech.n_species).at[mech.beta_species].set(beta_free)
    # Not scaled by beta: a precomposed readout is already the readout, and the
    # ones that qualify are beta-invariant by construction.
    scaled = y * jnp.exp(beta)[None, None, :]
    if mech.extra_fn is None or not mech.extra_at:
        # Passed only when there is one, so an h_r with no precomposed readouts
        # keeps the two-argument signature it has always had.
        return mech.h_fn(scaled, log_R)
    return mech.h_fn(
        scaled, log_R,
        {(s, n): mech.extra_fn(vartheta, s, n) for s, n in mech.extra_at})


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


def reference_levels(mu_0, omega_0, mech: Mechanism, *, log_R_0=None,
                     elig_fn: Optional[EligFn] = None) -> jnp.ndarray:
    """``c_r``: the level each readout sits at, ``(M,)``, at the plug-in.

    One entry per readout, not per cohort: a readout belongs to one cohort, so the
    cohort index adds nothing. Fixed once and held there. eq:disc pivots on it, so
    letting it move with ``phi`` would make ``kappa`` rescale about a moving point
    instead of about the study's own level. The plug-in is the no-discrepancy
    point, where the map is the identity and ``x-tilde = x``. ``log R`` is a
    declared conversion rather than discrepancy, so it sits at its prior centre.
    """
    x = readout_cloud(mu_0, omega_0, jnp.zeros(mech.beta_species.shape[0]),
                      mech, log_R_0)
    w = jnp.ones(x.shape[0]) if elig_fn is None else elig_fn(x, None)
    return jnp.stack([_median(*_sorted(x[:, m], w)) for m in range(x.shape[1])])


def cohort_columns(specs_by_cohort, readouts) -> Dict[str, Tuple[int, ...]]:
    """The readout columns each cohort reports. Most of ``M`` is dead for any one."""
    index_of = {r: i for i, r in enumerate(readouts)}
    return {c: tuple(sorted({index_of[spec.target_id] for spec in specs}))
            for c, specs in specs_by_cohort.items()}


def cohort_cloud(x, cols, a, b, refs, mech: Mechanism):
    """One cohort's mapped readouts, ``(N, |cols|)``: eq:disc on its own columns."""
    idx = np.asarray(cols)
    return apply_map(x[:, idx], a, b, jnp.asarray(refs)[idx], mech.Z[idx])


def quantile_mass_table(specs_by_cohort, n_cloud: int) -> Dict[Tuple, jnp.ndarray]:
    """Beta masses for every ``(p, n, convention)`` a corpus uses, at uniform ``w``.

    At ``w`` uniform the edges are ``linspace(0, 1, N+1)``, so the mass reads none
    of ``phi``. Build it once outside the gradient rather than in every evaluation.
    """
    from qsp_inference.vpop.statistics import quantile_mass

    w = jnp.ones(n_cloud)
    keys = {(s.p, s.n, s.convention) for specs in specs_by_cohort.values()
            for s in specs if s.stat == "quantile"}
    return {k: quantile_mass(w, *k) for k in sorted(keys)}


def tau_rows(specs, x_cohort, w, mech: Mechanism, designs=None,
             uniform: bool = False, presorted: bool = False,
             mass_table=None, column_of=None) -> jnp.ndarray:
    """One cohort's rows, ``(K_c,)``, in the order the source printed them.

    ``uniform`` says every patient carries weight one, which holds wherever no
    eligibility criterion is declared. Sorting permutes ``w`` differently per
    readout, so only then does one Beta mass serve every row sharing
    ``(p, n, convention)``. ``presorted`` says the caller already sorted the
    patient axis, which eq:disc allows because it is monotone.
    """
    from qsp_inference.vpop.statistics import quantile_mass

    index_of = (column_of if column_of is not None
                else {r: i for i, r in enumerate(mech.readouts)})
    marginal: Dict[str, Tuple[jnp.ndarray, jnp.ndarray]] = {}

    if presorted:
        for spec in specs:
            marginal.setdefault(
                spec.target_id, (x_cohort[:, index_of[spec.target_id]], w))
    else:
        columns = sorted({index_of[spec.target_id] for spec in specs})
        order = jnp.argsort(x_cohort[:, jnp.asarray(columns)], axis=0)
        at = {c: k for k, c in enumerate(columns)}
        for spec in specs:
            if spec.target_id not in marginal:
                col = index_of[spec.target_id]
                idx = order[:, at[col]]
                marginal[spec.target_id] = (x_cohort[idx, col], w[idx])

    masses: Dict[Tuple[float, int, str], jnp.ndarray] = dict(mass_table or {})
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


def tau_block(x, plan, specs_by_cohort, a, b, refs, mech: Mechanism, cols_of,
              *, designs=None, elig_fn: Optional[EligFn] = None,
              elig_at: Optional[Mapping[str, str]] = None,
              presorted: bool = False, mass_table=None) -> jnp.ndarray:
    """``tau_B``. Cohorts concatenate in ``plan.cohort_ids`` order, matching ``V_B``."""
    x_of, column_of = {}, {}
    for c in plan.cohort_ids:
        cols = cols_of[c]
        x_of[c] = cohort_cloud(x, cols, a, b, refs, mech)
        column_of[c] = {mech.readouts[col]: k for k, col in enumerate(cols)}

    w_of = block_weights(x_of, plan, elig_fn, elig_at)
    return jnp.concatenate([
        tau_rows(specs_by_cohort[c], x_of[c], w_of[c], mech, designs,
                 uniform=elig_fn is None, presorted=presorted,
                 mass_table=mass_table, column_of=column_of[c])
        for c in plan.cohort_ids
    ])


def tau_all(mu, omega, a, b, beta_free, plans, specs_by_cohort, refs,
            mech: Mechanism, *, log_R=None, designs=None,
            elig_fn: Optional[EligFn] = None,
            elig_at: Optional[Mapping[str, str]] = None,
            mass_table=None) -> Sequence[jnp.ndarray]:
    """Every block's prediction from one ``phi``, on one pass through the emulator.

    With no eligibility rule the patient axis is sorted once, before eq:disc rather
    than after it per cohort: the map is monotone in ``x``, so ``sort(map(x))`` and
    ``map(sort(x))`` are the same array.
    """
    x = readout_cloud(mu, omega, beta_free, mech, log_R)
    presorted = elig_fn is None
    if presorted:
        x = jnp.sort(x, axis=0)

    cols_of = cohort_columns(specs_by_cohort, mech.readouts)
    return [tau_block(x, plan, specs_by_cohort, a, b, refs, mech, cols_of,
                      designs=designs, elig_fn=elig_fn, elig_at=elig_at,
                      presorted=presorted, mass_table=mass_table)
            for plan in plans]

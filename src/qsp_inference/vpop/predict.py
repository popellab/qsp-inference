"""``tau_B(phi)``: the model's prediction of every reported number. eq:crn to eq:stat.

The chain is

    phi -> patients -> species -> readouts -> measurement map -> rows

and only the species step needs the emulator, which arrives as ``g_fn``.

eq:elig is ``Mechanism.w_fn``, off unless a fit passes one. It weights each cloud
member by P(ok), the status head's estimate that the patient reaches diagnosis at
all, which is what the design conditioned on and eq:pop did not.

No ``Z``: the rows are expectations over the population, so the normaliser divides
the numerator and the denominator of the same average and cancels. It would only
survive if a source printed how many patients it screened to enrol its cohort,
and none does.

The criterion is one function of theta, not one per cohort: evolve_to_diagnosis
runs before the arms split, so the rejection codes are identical across arms. That
retires the worry that cohorts drawn together need a criterion they agree on.

Weighted, the patient axis is sorted by argsort rather than sort, because the
weights have to take each column's permutation with them. Unweighted the two are
the same array and the old path is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from qsp_inference.vpop.rows import quantile_mass, tau_row

__all__ = ["Mechanism", "apply_margins", "logit_median_coords", "patient_cloud",
           "readout_cloud", "apply_map", "reference_levels", "cohort_cloud",
           "quantile_mass_table", "cohort_columns", "tau_rows", "tau_block",
           "tau_from_readouts", "tau_all"]

#: ``(vartheta, scenario) -> (N, Q)`` raw positive species. The emulator predicts
#: log species, so a wrapper exponentiates: beta is multiplicative on species.
GFn = Callable[[jnp.ndarray, int], jnp.ndarray]

#: ``(S, N, Q), (A,) -> (S, N, M)`` log-scale readouts, in ``Mechanism.readouts``
#: order. The second argument is ``log R``, the declared assay conversions.
HFn = Callable[[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]


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
    # (P,) bool, or None for all-lognormal. True where the margin is bounded on
    # (0, 1) and omega is a log-odds sd. See patient_cloud.
    logit: Optional[jnp.ndarray] = None
    # eq:elig. ``w_fn(vartheta) -> (N,)`` is P(ok) from the status head; None
    # leaves the cloud unweighted, which is bit-for-bit the old path.
    w_fn: Optional[Callable] = None

    @property
    def n_readouts(self) -> int:
        return len(self.readouts)

    def __post_init__(self):
        # Scaling column j of L_R' by omega_j commutes with the row sum, so
        # z (L_R' * omega) is (z L_R') * omega and the matmul is phi-free.
        # Held here because eq:crn is otherwise an (N,P)x(P,P) product per gradient.
        object.__setattr__(self, "zL",
                           jnp.asarray(self.z) @ jnp.asarray(self.L_R).T)


def apply_margins(mu, omega, zL, logit=None) -> jnp.ndarray:
    """``(N, P)`` log theta from correlated standard normals. eq:crn's margin step.

    Split out of :func:`patient_cloud` so the emulator's training pool can be
    drawn through the identical expression. A pool drawn from a different margin
    than the fit evaluates the surrogate off the manifold it learned, with no
    symptom, because every number stays plausible.

    Returns log theta for every parameter whatever its margin, so nothing
    downstream has to know which is which.

    A parameter marked ``logit`` is bounded on (0, 1) and no lognormal margin
    respects that: a fractional maximum effect at median 0.8 puts a quarter of
    the population above complete inhibition at omega 0.35, and the only width
    that fixes it is one that denies the quantity varies. The correlation is a
    Gaussian copula -- ``zL`` is standard normal before any margin is applied --
    so the margin is separable and only this line changes.

    For a logit-margin parameter ``mu`` is the LOGIT of the population median,
    not its log, so ``sigmoid(mu)`` is in (0, 1) for every real ``mu`` and eq:muprior
    cannot propose a median outside the bound. ``mu`` stays on the log scale for
    every other parameter. :func:`logit_median_coords` is what moves a marginal
    stated on the log scale into this one, and both the pool and the fit call it.

    The alternative was to keep ``mu`` on the log scale throughout and reject
    ``exp(mu) >= 1`` with a ``-inf`` factor. That reads as free, since the factor
    is piecewise constant and its gradient is zero, but the gradient is never
    consulted: a leapfrog step into the excluded region gives infinite energy
    error and NUTS discards the whole trajectory as divergent. On pdac the four
    bounded parameters put 33.7% of the prior mass there jointly -- ``any`` over
    them, not the 15% the worst one carries alone -- and the chain diverged on
    275 of 300 draws. A bound the coordinate cannot violate has no wall to hit.
    """
    # ``mu`` is (P,) in the fit, where one mu is shared by the whole cloud, and
    # (N, P) when drawing the emulator's pool, where every row carries its own mu
    # draw and the two spreads compose. Same expression either way.
    mu = jnp.asarray(mu)
    mu = mu if mu.ndim == 2 else mu[None, :]
    log_theta = mu + zL * omega[None, :]
    if logit is None:
        return log_theta
    logit = jnp.asarray(logit)
    # Both branches evaluate, and this one is finite for every real mu, so the
    # discarded arm needs no masking to keep its gradient off a singularity.
    log_theta_b = jax.nn.log_sigmoid(mu + zL * omega[None, :])
    return jnp.where(logit[None, :], log_theta_b, log_theta)


def logit_median_coords(mu_0, sd_1, logit):
    """``(mu_0, sd_1)`` moved into the coordinate :func:`apply_margins` reads.

    A marginal states a median and a spread on the log scale. For a logit-margin
    parameter ``mu`` is the logit of the median instead, so both move. The median
    is preserved exactly; the spread is the delta method, ``d logit(m)/d log(m)``
    ``= 1/(1 - m)``, which is why a median near the bound widens most.

    Only the marginals move. The stage-1 copula is a correlation and a monotone
    per-parameter reparameterisation leaves it alone, so ``L_R`` is untouched.
    """
    mu_0 = np.array(mu_0, dtype=float, copy=True)
    sd_1 = np.array(sd_1, dtype=float, copy=True)
    if logit is None:
        return mu_0, sd_1
    mask = np.asarray(logit, dtype=bool)
    if not mask.any():
        return mu_0, sd_1
    m = np.exp(mu_0[mask])
    if np.any(m >= 1.0):
        bad = int(np.argmax(m >= 1.0))
        raise ValueError(
            f"a logit-margin parameter has median exp(mu_0) = {m[bad]:.4g}, which "
            f"is not inside (0, 1). The bound is a property of the parameter, so "
            f"this is the marginal being wrong rather than something to clamp."
        )
    mu_0[mask] = np.log(m) - np.log1p(-m)
    sd_1[mask] = sd_1[mask] / (1.0 - m)
    return mu_0, sd_1


def patient_cloud(mu, omega, mech: Mechanism) -> jnp.ndarray:
    """eq:crn. ``(N, P)`` log-parameters, smooth in ``(mu, omega)`` at frozen ``z``."""
    return apply_margins(mu, omega, mech.zL, mech.logit)


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


def reference_levels(mu_0, omega_0, mech: Mechanism, *, log_R_0=None) -> jnp.ndarray:
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
    # The lower of the two middle order statistics, not their average, which is
    # what the weighted form this replaced returned at uniform weights. A pivot
    # is fixed once and everything downstream is stated against it, so it is not
    # a place to change a definition while simplifying the expression.
    return jnp.sort(x, axis=0)[(x.shape[0] + 1) // 2 - 1]


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
    """Beta masses for every ``(p, n, convention)`` a corpus uses.

    The mass reads the cloud only through its size, so it is built once outside
    the gradient rather than in every evaluation.
    """
    keys = {(s.p, s.n, s.convention) for specs in specs_by_cohort.values()
            for s in specs if s.stat == "quantile"}
    return {k: quantile_mass(n_cloud, *k) for k in sorted(keys)}


def tau_rows(specs, x_cohort, mech: Mechanism, designs=None, mass_table=None,
             column_of=None, w_cohort=None) -> jnp.ndarray:
    """One cohort's rows, ``(K_c,)``, in the order the source printed them.

    ``x_cohort`` is already sorted up the patient axis, which eq:disc allows
    because the map is monotone.
    """
    index_of = (column_of if column_of is not None
                else {r: i for i, r in enumerate(mech.readouts)})
    masses: Dict[Tuple[float, int, str], jnp.ndarray] = dict(mass_table or {})
    n_cloud = x_cohort.shape[0]

    out = []
    for spec in specs:
        col = index_of[spec.target_id]
        # eq:elig weights are per readout: the cloud is sorted by each row's own
        # column, so the weight vector carries that column's permutation too. A
        # weight left in cloud order would pair the wrong probability with each
        # value, which is arithmetic that stays finite and is simply wrong.
        w_col = None if w_cohort is None else w_cohort[:, col]
        mass = None
        if spec.stat == "quantile":
            if w_col is None:
                key = (spec.p, spec.n, spec.convention)
                if key not in masses:
                    masses[key] = quantile_mass(n_cloud, *key)
                mass = masses[key]
            # weighted: the mass reads phi, so tau_row builds it per row rather
            # than looking it up. mass_table is for the unweighted path only.
        out.append(tau_row(spec, x_cohort[:, col],
                           (designs or {}).get(spec.label), mass,
                           w_sorted=w_col))
    return jnp.stack(out)


def tau_block(x, plan, specs_by_cohort, a, b, refs, mech: Mechanism, cols_of,
              *, designs=None, mass_table=None, w=None) -> jnp.ndarray:
    """``tau_B``. Cohorts concatenate in ``plan.cohort_ids`` order, matching ``V_B``."""
    out = []
    for c in plan.cohort_ids:
        cols = cols_of[c]
        # exp, because a row functional has to run in the units the source
        # printed. h_r returns logs and eq:disc acts there, which is what makes
        # gamma a multiplicative assay bias, but the reported number is a mean or
        # an IQR of cells/mm^2 and neither commutes with exp. Quantile rows do
        # commute, so only the moment and width rows depend on this being here
        # rather than applied to tau afterwards.
        x_c = jnp.exp(cohort_cloud(x, cols, a, b, refs, mech))
        w_c = None if w is None else w[:, jnp.asarray(cols)]
        column_of = {mech.readouts[col]: k for k, col in enumerate(cols)}
        out.append(tau_rows(specs_by_cohort[c], x_c, mech, designs,
                            mass_table=mass_table, column_of=column_of,
                            w_cohort=w_c))
    return jnp.concatenate(out)


def tau_from_readouts(x, a, b, plans, specs_by_cohort, refs, mech: Mechanism, *,
                      designs=None, mass_table=None, w=None) -> Sequence[jnp.ndarray]:
    """Every block's rows from an already-computed readout cloud, ``(N, M)``.

    Split out from :func:`tau_all` because ``E_B`` needs the same rows computed
    from a cloud the simulator produced, where there is no ``phi`` to push through
    the emulator at all.
    """
    if w is None:
        x = jnp.sort(x, axis=0)
        w_sorted = None
    else:
        # argsort rather than sort: each column has its own permutation and the
        # weights have to take it too. take_along_axis reproduces jnp.sort here.
        order = jnp.argsort(x, axis=0)
        x = jnp.take_along_axis(x, order, axis=0)
        w_sorted = jnp.asarray(w)[order]
    cols_of = cohort_columns(specs_by_cohort, mech.readouts)
    return [tau_block(x, plan, specs_by_cohort, a, b, refs, mech, cols_of,
                      designs=designs, mass_table=mass_table, w=w_sorted)
            for plan in plans]


def tau_all(mu, omega, a, b, beta_free, plans, specs_by_cohort, refs,
            mech: Mechanism, *, log_R=None, designs=None,
            mass_table=None, w=None) -> Sequence[jnp.ndarray]:
    """Every block's prediction from one ``phi``, on one pass through the emulator."""
    if w is None and mech.w_fn is not None:
        # The same vartheta the emulator reads, so the weight and the readouts
        # describe one patient each rather than two draws that happen to align.
        w = mech.w_fn(patient_cloud(mu, omega, mech))
    return tau_from_readouts(
        readout_cloud(mu, omega, beta_free, mech, log_R), a, b, plans,
        specs_by_cohort, refs, mech, designs=designs, mass_table=mass_table, w=w)

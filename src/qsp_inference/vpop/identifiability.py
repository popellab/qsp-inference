"""What the rows determine, before any fit. All of it from one whitened Jacobian.

Four questions, and none needs MCMC:

* :func:`conditioning_report` -- how anisotropic the posterior is at the plug-in,
  which is the diagnostic for "why is NUTS saturating its tree depth";
* :func:`z_cost` -- what it costs the mechanism to imitate a column of ``Z``,
  which is whether eq:disc is identified against eq:crn;
* :func:`projection_report` -- whether ``beta`` has a direction the measurement
  map cannot reach, and whether the row space is saturated at all;
* :func:`pivot_offsets` -- how much location lever ``kappa`` keeps after eq:disc
  centres on ``c_r``.

Everything is whitened by ``V^-1/2``, so a number is in units of the row's own
error bar, and read against a null: a residual of 0.25 is not evidence of
aliasing when a random direction already scores 0.40.

Every function returns lines rather than printing, so a caller decides what to do
with them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, List, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from qsp_inference.vpop.mass import _row_sizes, _whiten, laplace_blocks

__all__ = ["row_jacobians", "conditioning_report", "z_cost",
           "projection_report", "pivot_offsets"]


def row_jacobians(prior, problem, V_chol, *, at: Optional[Mapping[str, Any]] = None):
    """``d tau / d phi`` at the plug-in, whitened. One ``(A, dim)`` array per block.

    Keyed by the quantities of the draft (``mu``, ``omega``, ``a``, ``b``,
    ``beta``, ``log_R``) rather than by the model's sampling sites, because these
    are the objects the identification questions are asked about.

    The discrepancy sits at its prior centre, matching the point ``V`` was frozen
    at, so this is a statement about the model at the plug-in and not about any
    fit.
    """
    from qsp_inference.vpop.predict import tau_all

    n_beta = int(np.asarray(problem.mech.beta_species).shape[0])
    point = {
        "mu": jnp.asarray(prior.mu_0),
        "omega": jnp.asarray(prior.omega_0),
        "a": jnp.zeros(prior.dim_z),
        "b": jnp.zeros(prior.dim_z),
        "beta": jnp.zeros(n_beta),
        "log_R": (jnp.asarray(prior.log_R_0) if prior.n_aux else jnp.zeros(0)),
    }
    point.update({k: jnp.asarray(v) for k, v in (at or {}).items()})
    names = list(point)
    args = [point[k] for k in names]

    def tau_of(*values):
        d = dict(zip(names, values))
        taus = tau_all(d["mu"], d["omega"], d["a"], d["b"], d["beta"],
                       problem.plans, problem.specs_by_cohort, problem.refs,
                       problem.mech,
                       log_R=(d["log_R"] if prior.n_aux else None),
                       designs=problem.designs, mass_table=problem.mass_table,
                       elig_fn=problem.elig_fn, elig_at=problem.elig_at)
        return jnp.concatenate(taus)

    sizes = _row_sizes(problem)
    n_row = sum(sizes)
    out = {}
    for k, nm in enumerate(names):
        J = np.asarray(jax.jacfwd(tau_of, argnums=k)(*args)).reshape(n_row, -1)
        out[nm] = _whiten(J, V_chol, sizes)
    return out


# ---------------------------------------------------------------------------
def _orth(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Orthonormal basis for ``col(A)``, dropping numerically dependent directions."""
    if A.size == 0:
        return A.reshape(A.shape[0], 0)
    U, sv, _ = np.linalg.svd(A, full_matrices=False)
    if sv.size == 0:
        return U[:, :0]
    return U[:, sv > tol * max(float(sv[0]), 1e-30)]


def _residual_fraction(target: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Per-column fraction of ``target`` that ``basis`` cannot reproduce."""
    Qb = _orth(basis)
    res = target - Qb @ (Qb.T @ target)
    den = np.linalg.norm(target, axis=0)
    return np.linalg.norm(res, axis=0) / np.clip(den, 1e-30, None)


def _residual_spectrum(target: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Singular values of ``target`` after removing ``col(basis)``, scaled to ``sv[0]``.

    Reported beside the per-column residuals because columns can each be
    individually distinct while a combination of them is flat. That is exactly
    the case for a common ``beta``, so per-column numbers alone would miss it.
    """
    Qb = _orth(basis)
    res = target - Qb @ (Qb.T @ target)
    sv_full = np.linalg.svd(target, compute_uv=False)
    sv_res = np.linalg.svd(res, compute_uv=False)
    return sv_res / max(float(sv_full[0]), 1e-30)


def _null_residual(n_rows: int, basis: np.ndarray):
    """Residual a random direction would score against ``basis``, and its rank.

    Without this the numbers cannot be read. A competitor block of rank ``k`` in
    an ``A``-row space leaves only an ``(A-k)``-dimensional complement, so a
    direction drawn uniformly at random already scores ``sqrt((A-k)/A)``.
    """
    k = _orth(basis).shape[1]
    return float(np.sqrt(max(n_rows - k, 0) / max(n_rows, 1))), k


def _standardised(dims, prior_sd, blocks):
    """``J * prior_sd``: the chain rule for ``theta -> theta / prior_sd``.

    In those coordinates the prior precision is exactly ``I``, so an eigenvalue
    of ``G = I + J'J`` reads as "1 = this direction returns the prior".
    """
    return [B * (np.full(d, float(sd)) if np.ndim(sd) == 0
                 else np.asarray(sd, dtype=float))[None, :]
            for B, d, sd in zip(blocks, dims, prior_sd)]


def conditioning_report(prior, problem, V_chol, *, top: int = 8,
                        flat: bool = False, row_masks=None):
    """``(lines, names, standardised blocks)``: the posterior's anisotropy at the plug-in.

    Works in STANDARDISED sampling coordinates, the ones in which every prior is
    exactly ``N(0, 1)``. There the Gauss-Newton information is ``G = I + J'J``,
    so the eigenvalues are ``1 + sigma_k^2`` for the singular values of ``J``,
    the condition number is ``(1 + sigma_max^2) / (1 + sigma_min^2)``, and NUTS's
    trajectory length scales as its square root. With more coordinates than rows,
    at least ``dim - A`` eigenvalues are exactly 1 by construction: those
    directions return the prior and are perfectly conditioned, and the damage is
    all in ``sigma_max``.

    Reporting it this way separates the two candidate explanations, which need
    different fixes. Natural-scale spread would show up as many coordinates at
    moderate ``sigma``. A handful of sharply determined directions shows up as a
    large ``sigma_max`` with everything else at 1, and rescaling parameters would
    do nothing about that.

    The blocks come back standardised so :func:`z_cost` can reuse the Jacobian,
    which is a forward-mode pass over every coordinate and the slowest step here.
    """
    names, dims, prior_sd, blocks, labels = laplace_blocks(
        prior, problem, V_chol, flat=flat, row_masks=row_masks)
    blocks = _standardised(dims, prior_sd, blocks)
    Jw = np.hstack(blocks)
    dim = Jw.shape[1]
    sv = np.linalg.svd(Jw, compute_uv=False)
    eig = np.sort(np.concatenate([1.0 + sv ** 2,
                                  np.ones(max(dim - sv.size, 0))]))[::-1]
    cond = eig[0] / eig[-1]

    lines = [
        f"posterior conditioning at the plug-in ({dim} standardised coordinates, "
        f"{Jw.shape[0]} rows)",
        "  G = I + J'J, so an eigenvalue of 1 is a direction that returns the prior",
        f"  condition number {cond:,.0f}, so trajectories run about "
        f"{np.sqrt(cond):,.0f}x the shortest direction",
        f"  eigenvalues at the prior (= 1.000): "
        f"{int((eig < 1.0 + 1e-9).sum())} of {dim}",
        "  largest: " + ", ".join(f"{v:.3g}" for v in eig[:6]),
    ]

    _, _, Vt = np.linalg.svd(Jw, full_matrices=False)
    v = Vt[0]
    lines.append(f"  the sharpest direction (eigenvalue {eig[0]:,.0f}) loads on:")
    for t in np.argsort(-np.abs(v))[:top]:
        lines.append(f"    {labels[t]:<20}{v[t]:+.3f}")

    total = float((Jw ** 2).sum())
    lines.append(f"  {'block':<20}{'dim':>5}{'sigma_max':>12}{'n sigma>1':>11}"
                 f"{'share of tr(J^T J)':>20}")
    for nm, B in zip(names, blocks):
        s_b = np.linalg.svd(B, compute_uv=False)
        lines.append(f"  {nm:<20}{B.shape[1]:>5}{s_b[0]:>12.2f}"
                     f"{int((s_b > 1).sum()):>11}"
                     f"{float((B ** 2).sum()) / max(total, 1e-30):>20.3f}")
    return lines, names, blocks


def z_cost(names: Sequence[str], blocks: Sequence[np.ndarray],
           z_columns: Sequence[str], *, rcond: float = 1e-8) -> List[str]:
    """eq:zcost. What does it COST the mechanism to imitate a column of ``Z``?

    The older test asked whether the mechanism *can* reproduce a column's row
    effect, by projecting onto ``col(d tau / d(mu, omega))`` and reading the
    residual. At ``P >> K`` that is vacuous: the mechanism Jacobian already spans
    row space, so the residual is zero for every column whatever ``Z`` is, and a
    pass means nothing.

    Ask the price instead. Among all mechanism moves reproducing the effect
    exactly, take the smallest in prior units::

        delta* = argmin  delta' Sigma^-1 delta   s.t.   J_mech delta = W e_j

    In coordinates already standardised by the prior this is the minimum-norm
    solution ``pinv(J_mech) t``, whose norm reads directly in prior standard
    deviations. Large means only an implausible excursion imitates that column,
    so it is identified against the mechanism; small means the two are confounded
    and the priors decide the split.

    ``names`` and ``blocks`` are :func:`conditioning_report`'s second and third
    return values, already standardised.
    """
    by_name = dict(zip(names, blocks))
    mech = [n for n in ("mu_raw", "s", "u_raw", "log_omega_measured")
            if n in by_name]
    J_mech = np.hstack([by_name[n] for n in mech])

    targets = []
    for site, sym in (("a", "gamma"), ("b", "kappa")):
        if site not in by_name:
            continue
        B = by_name[site]
        for j in range(B.shape[1]):
            label = z_columns[j] if j < len(z_columns) else f"col{j}"
            targets.append((f"{sym}/{label}", B[:, j]))

    # rcond guards J_mech's near-null directions: without it the minimum-norm
    # solution loads on directions the mechanism barely moves, and the cost is
    # then dominated by numerical noise.
    pinv = np.linalg.pinv(J_mech, rcond=rcond)
    sv = np.linalg.svd(J_mech, compute_uv=False)
    keep = int((sv > rcond * sv[0]).sum())

    lines = [
        "Z cost test (eq:zcost): mechanism moves needed to imitate a column",
        f"  {J_mech.shape[1]} mechanism coordinates over {J_mech.shape[0]} rows, "
        f"effective rank {keep} (sigma {sv[0]:.3g} down to {sv[keep-1]:.3g})",
        f"  {'column':<28}{'cost (prior sd)':>17}{'unreachable':>13}",
    ]
    for label, t in targets:
        eta = pinv @ t
        nt = max(float(np.linalg.norm(t)), 1e-30)
        lines.append(f"  {label[:26]:<28}{float(np.linalg.norm(eta)):>17.2f}"
                     f"{float(np.linalg.norm(J_mech @ eta - t)) / nt:>13.3f}")
    lines += [
        "  cost >> 1: only an implausible mechanism move imitates it, so the "
        "column is identified",
        "  cost << 1: confounded, and the priors decide. unreachable > 0 means "
        "the mechanism cannot reproduce it at all, which is stronger still",
    ]
    return lines


def projection_report(J: Mapping[str, np.ndarray], *,
                      beta_names: Sequence[str] = ()) -> List[str]:
    """Aliasing between blocks, from :func:`row_jacobians`, against its null."""
    mech = np.hstack([J["mu"], J["omega"]])
    meas = np.hstack([J["a"], J["b"]])
    A = J["mu"].shape[0]

    lines = [
        f"projection tests, whitened by V^-1/2 ({A} rows)",
        "  residual = fraction of a direction the competing block cannot reproduce",
        "  read every residual against the random-direction null on its line: "
        "below the null means more aliased than chance, at or above means a "
        "direction of its own",
    ]

    n_beta = J["beta"].shape[1]
    if n_beta:
        null_meas, k_meas = _null_residual(A, meas)
        rbeta = _residual_fraction(J["beta"], meas)
        lines.append("")
        lines.append("  S test: can (gamma, kappa) absorb a species bias?")
        lines.append(f"    competitor rank {k_meas} of {A} rows, so the null is "
                     f"{null_meas:.3f}")
        lines.append(f"    {'species':<20}{'resid':>13}")
        for q in range(n_beta):
            nm = beta_names[q] if q < len(beta_names) else f"beta[{q}]"
            lines.append(f"    {nm[:18]:<20}{rbeta[q]:>13.3f}")

        sv = _residual_spectrum(J["beta"], meas)
        lines.append("    singular spectrum of the beta block after removing "
                     "(a, b): " + ", ".join(f"{v:.2e}" for v in sv))
        lines.append(f"    {int((sv > 1e-8).sum())} of {n_beta} directions survive")
        common = J["beta"] @ (np.ones(n_beta) / np.sqrt(n_beta))
        Qm = _orth(meas)
        r_common = (np.linalg.norm(common - Qm @ (Qm.T @ common))
                    / max(float(np.linalg.norm(common)), 1e-30))
        lines.append(f"    the common shift specifically: residual {r_common:.3f}")
        if r_common < 1e-6:
            lines.append("      exactly reproduced by gamma, which is why the "
                         "model centres beta")
    else:
        lines.append("")
        lines.append("  no free beta in this problem, so the S test does not apply")

    # The row budget. Everything above is a statement about directions in a space
    # of dimension A, so it is meaningful only while the competing blocks leave
    # some of that space over. Once (a, b, mu, omega) spans all A rows, every
    # residual against the full model is zero by arithmetic and no first-order
    # identification claim can be made about any block at all.
    full = np.hstack([J["a"], J["b"], J["mu"], J["omega"]])
    k_full = _orth(full).shape[1]
    n_par = sum(J[k].shape[1] for k in ("a", "b", "mu", "omega"))
    lines.append("")
    lines.append(f"  row budget: {n_par} parameters in (a, b, mu, omega), rank "
                 f"{k_full}, {A} rows")
    if k_full >= A:
        lines += [
            "    the row space is SATURATED. No block is identified at first "
            "order; the residuals above are relative aliasing between blocks, "
            "not evidence that any of them is estimable.",
            "    Only the priors separate them. More rows is the only fix, and "
            "this is a verdict on A, not on Z.",
        ]
    return lines


def pivot_offsets(prior, problem, V_chol, *, top: int = 5) -> List[str]:
    """How much location lever ``kappa`` keeps after eq:disc centres on ``c_r``.

    eq:disc is ``kappa_r (x - c_r) + c_r + gamma_r``, so a location row's
    sensitivity to ``log kappa`` is proportional to the gap between the cohort's
    own level and the frozen pivot. At the pivot the gap is zero and ``kappa``
    is a pure spread term; away from it, ``kappa`` moves location rows too, and
    holding the scale rows out of a flat fit no longer holds ``b`` out with them.

    Measured rather than derived: the whitened ``d tau / d b`` is the lever in
    units of the row's own error bar, exactly, for every statistic. The closed
    form only ever covered the median and the mean, and got the mean wrong by
    the skewness of the pushforward.
    """
    centred = row_jacobians(prior, problem, V_chol)["b"]
    # The same Jacobian with the pivot removed, which is the c_r = 0 version of
    # the same quantity and so says what centring bought.
    flat_refs = replace(problem, refs=jnp.zeros_like(jnp.asarray(problem.refs)))
    uncentred = row_jacobians(prior, flat_refs, V_chol)["b"]

    labels, is_loc = [], []
    for plan in problem.plans:
        for c in plan.cohort_ids:
            for spec in problem.specs_by_cohort[c]:
                labels.append(f"{c}/{spec.label}")
                is_loc.append(not spec.is_scale)
    loc = np.flatnonzero(np.asarray(is_loc))

    cen = np.linalg.norm(centred[loc], axis=1)
    unc = np.linalg.norm(uncentred[loc], axis=1)
    order = np.argsort(-cen)

    lines = [
        "kappa's residual lever on the location rows (eq:disc pivot)",
        "  sd of the row that one unit of log kappa moves it. Small means a "
        "flat fit's scale-row holdout really does hold b out.",
        f"  {'row':<40}{'centred':>10}{'uncentred':>11}",
    ]
    for k in order[:top]:
        lines.append(f"  {labels[loc[k]][:38]:<40}{cen[k]:>10.2f}{unc[k]:>11.2f}")
    lines.append(f"  {'max over ' + str(len(loc)) + ' location rows':<40}"
                 f"{cen.max():>10.2f}{unc.max():>11.2f}")
    lines.append(f"  {'mean':<40}{cen.mean():>10.2f}{unc.mean():>11.2f}")
    lines.append(f"  centring cut the mean lever by "
                 f"{unc.mean() / max(cen.mean(), 1e-12):.1f}x")
    return lines

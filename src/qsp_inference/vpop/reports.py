"""What the rows determine, and the geometry that follows from it. No fit needed.

All of it comes off one whitened Jacobian at the plug-in, ``V^-1/2 d tau/d phi``,
so a number is in units of the row's own error bar:

* the Laplace metric, ``G = prior precision + J'V^-1 J``, handed to NUTS as its
  mass matrix. Worth computing rather than adapting: warmup cannot estimate a
  few-hundred-square covariance from a few hundred draws, and a mass matrix does
  not touch the prior, so a bad one costs only speed;
* :func:`conditioning_report` -- how anisotropic the posterior is, which is the
  diagnostic for "why is NUTS saturating its tree depth" (eq:ginfo);
* :func:`z_cost` -- what it costs the mechanism to imitate a column of ``Z``,
  which is whether eq:disc is identified against eq:crn (eq:zcost);
* :func:`projection_report`, :func:`pivot_offsets` -- whether ``beta`` has a
  direction the measurement map cannot reach, and how much location lever
  ``kappa`` keeps after eq:disc centres on ``c_r``;
* :func:`width_gate` -- eq:ratio, whether the predicted cloud is wide enough for
  the error bars built from it to mean anything;
* :func:`dbar_absorption` -- eq:absorbfrac, whether the measurement map can hold
  the emulator's systematic offset or it lands on ``mu``;
* :func:`map_estimate` and the recovery summary, for the toy corpus.

Everything is read against a null: a residual of 0.25 is not evidence of aliasing
when a random direction already scores 0.40. Every report returns lines rather
than printing, so a caller decides what to do with them.

The metric differentiates :func:`~qsp_inference.vpop.fit.phi_from_sites`, the
same map the model samples, in the same site coordinates. A second copy of that
map would let a mass matrix be built for a model that is not the one being
sampled, and numpyro would accept it: the blocks would simply belong to the
wrong parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    # the metric
    "adaptation_window", "laplace_blocks", "laplace_inverse_mass",
    "should_fix_mass",
    # what the rows determine
    "row_jacobians", "conditioning_report", "z_cost", "projection_report",
    "pivot_offsets",
    # whether the width evidence is reachable
    "width_gate", "dbar_absorption",
    # recovery against a known phi*
    "map_estimate", "RecoveryRow", "summarise_recovery", "print_recovery",
]


# ------------------------------------------------------------------- the metric

def _whiten(J: np.ndarray, V_chol, sizes: Sequence[int]) -> np.ndarray:
    """``V^-1/2 J``, block by block. ``sizes`` is each block's row count."""
    from scipy.linalg import solve_triangular

    out, off = [], 0
    for L, k in zip(V_chol, sizes):
        out.append(solve_triangular(np.asarray(L), J[off:off + k], lower=True))
        off += k
    return np.concatenate(out) if out else J


def _row_sizes(problem) -> list[int]:
    return [sum(len(problem.specs_by_cohort[c]) for c in plan.cohort_ids)
            for plan in problem.plans]


def laplace_blocks(prior, problem, V_chol, *, at: Optional[Mapping[str, Any]] = None):
    """Whitened ``d tau / d site``, one array per site, in the model's coordinates.

    NOT standardised coordinates. The sites here are exactly the sites the model
    samples, with the shapes it samples them in, because the output feeds a mass
    matrix and a coordinate mismatch there is silent.

    ``at`` is the point to linearise about, as site values; the default is the
    prior centre, which is where ``V`` was frozen.

    Returns ``(names, dims, prior_sd, blocks, labels)``.
    """
    from qsp_inference.vpop.fit import phi_from_sites, site_spec
    from qsp_inference.vpop.predict import tau_all

    spec = site_spec(prior)
    names = [nm for nm, _, _ in spec]
    init = [jnp.asarray((at or {}).get(nm, v)) for nm, v, _ in spec]
    prior_sd = [sd for _, _, sd in spec]

    def tau_of(*values):
        sites = dict(zip(names, values))
        mu, omega, a, b, beta_free, log_R = phi_from_sites(sites, prior)
        taus = tau_all(mu, omega, a, b, beta_free, problem.plans,
                       problem.specs_by_cohort, problem.refs, problem.mech,
                       log_R=log_R, designs=problem.designs,
                       mass_table=problem.mass_table)
        return jnp.concatenate(taus)

    sizes = _row_sizes(problem)
    n_row = sum(sizes)
    blocks, labels, dims = [], [], []
    for k, nm in enumerate(names):
        J = np.asarray(jax.jacfwd(tau_of, argnums=k)(*init)).reshape(n_row, -1)
        Jw = _whiten(J, V_chol, sizes)
        blocks.append(Jw)
        dims.append(Jw.shape[1])
        labels += [nm if Jw.shape[1] == 1 else f"{nm}[{j}]"
                   for j in range(Jw.shape[1])]
    return names, dims, prior_sd, blocks, labels


def _precision(dims: Sequence[int], prior_sd: Sequence[Any]) -> np.ndarray:
    return np.concatenate([
        np.full(d, 1.0 / float(sd) ** 2) if np.isscalar(sd) or np.ndim(sd) == 0
        else 1.0 / np.asarray(sd, dtype=float) ** 2
        for d, sd in zip(dims, prior_sd)])


def laplace_inverse_mass(prior, problem, V_chol, **kw):
    """``{tuple(sorted(sites)): G^-1}``, for numpyro's ``inverse_mass_matrix``.

    numpyro packs a dense block in the order of the key tuple and, for
    ``dense_mass=True``, uses ``tuple(sorted(latent sites))`` as that key. Getting
    it wrong is the one way this fails quietly, so the blocks are permuted into
    sorted order here rather than assumed to already be in it.
    """
    names, dims, prior_sd, blocks, _ = laplace_blocks(prior, problem, V_chol, **kw)
    order = sorted(range(len(names)), key=lambda i: names[i])
    names = [names[i] for i in order]
    dims = [dims[i] for i in order]
    prior_sd = [prior_sd[i] for i in order]
    blocks = [blocks[i] for i in order]

    Jw = np.hstack(blocks)
    G = np.diag(_precision(dims, prior_sd)) + Jw.T @ Jw
    G_inv = np.linalg.inv(G)
    return {tuple(names): jnp.asarray(0.5 * (G_inv + G_inv.T))}


def adaptation_window(warmup: int) -> int:
    """Draws numpyro's largest mass-matrix estimation window actually gets.

    Not ``warmup``. The schedule spends an init buffer finding a step size and a
    terminal buffer re-tuning it, and estimates the metric only in the expanding
    windows between them, keeping the last estimate. At 1000 warmup draws the
    largest of those windows is 500; at 200 it is 50, against a total that looks
    four times larger.
    """
    from numpyro.infer.hmc_util import build_adaptation_schedule

    schedule = build_adaptation_schedule(warmup)
    return max((b - a + 1 for a, b in schedule[1:-1]), default=0)


def should_fix_mass(dim: int, warmup: int, factor: int = 3) -> bool:
    """Whether to freeze the computed metric or let warmup re-estimate it.

    numpyro's adaptation estimates the mass matrix from scratch, so leaving it on
    discards the Laplace metric and keeps only the benefit of having explored
    well while getting there. That is the right trade when the schedule has
    enough draws to estimate a ``dim x dim`` covariance and the wrong one when it
    does not, so the rule is the one that decides which estimator is better
    rather than a global switch: below about three draws per dimension, keep the
    computed metric.

    The comparison is against :func:`adaptation_window`, not against ``warmup``.
    Reading the total is what a caller would do by hand, and it overstates the
    draws behind the estimate several-fold.
    """
    return adaptation_window(warmup) < factor * dim

# ------------------------------------------------- what the rows determine

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
                       designs=problem.designs, mass_table=problem.mass_table)
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


def conditioning_report(prior, problem, V_chol, *, top: int = 8):
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
        prior, problem, V_chol)
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
    is a pure spread term; away from it, ``kappa`` moves location rows too, so
    ``b`` stops being a thing only the scale rows can see.

    Measured rather than derived: the whitened ``d tau / d b`` is the lever in
    units of the row's own error bar, exactly, for every statistic. The closed
    form only ever covered the median and the mean, and got the mean wrong by
    the skewness of the pushforward.
    """
    centred = row_jacobians(prior, problem, V_chol)["b"]
    # The same Jacobian with the pivot removed, which is the c_r = 0 version of
    # the same quantity and so says what centring bought.
    no_pivot = replace(problem, refs=jnp.zeros_like(jnp.asarray(problem.refs)))
    uncentred = row_jacobians(prior, no_pivot, V_chol)["b"]

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
        "  sd of the row that one unit of log kappa moves it. Small means the "
        "scale rows are where b is seen, and the location rows are not.",
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

# ------------------------------------- whether the width evidence is reachable

def _row_labels(problem):
    """``(labels, specs)`` in the order the blocks concatenate."""
    labels, specs = [], []
    for plan in problem.plans:
        for c in plan.cohort_ids:
            for spec in problem.specs_by_cohort[c]:
                labels.append(f"{c}/{spec.label}")
                specs.append(spec)
    return labels, specs


def _tau_at(prior, problem, *, omega=None):
    """``tau`` at the plug-in, with no discrepancy. One vector per block."""
    from qsp_inference.vpop.predict import tau_all

    n_beta = int(np.asarray(problem.mech.beta_species).shape[0])
    return tau_all(jnp.asarray(prior.mu_0),
                   jnp.asarray(prior.omega_0 if omega is None else omega),
                   jnp.zeros(prior.dim_z), jnp.zeros(prior.dim_z),
                   jnp.zeros(n_beta),
                   problem.plans, problem.specs_by_cohort, problem.refs,
                   problem.mech,
                   log_R=(jnp.asarray(prior.log_R_0) if prior.n_aux else None),
                   designs=problem.designs, mass_table=problem.mass_table)


def width_gate(observed: Sequence, prior, problem, V, *, top: int = 8) -> List[str]:
    """eq:ratio at the plug-in, as the gate it has to be. Run before the fit.

    The check is the prior predictive residual on the rows that carry width: an
    ``sd``, ``se`` or ``iqr`` row predicted at ``phi_0`` against the one that was
    reported. It is exactly the fit's own residual evaluated at ``phi_0`` instead
    of at ``phi``.

    A cloud that is too narrow shows up as a positive shortfall: the source
    reports more spread than the model can produce. The response is to fix the
    model, not to widen ``V_c``, since widening hides the thing being tested.
    """
    pred = _tau_at(prior, problem)
    labels, specs = _row_labels(problem)
    sd = np.concatenate([np.sqrt(np.diag(np.asarray(v))) for v in V])
    obs = np.concatenate([np.asarray(o, dtype=float) for o in observed])
    prd = np.concatenate([np.asarray(p, dtype=float) for p in pred])

    keep = [i for i, s in enumerate(specs) if s.is_scale]
    lines = ["width gate at phi_0 (eq:ratio), before any fit",
             "  every V_c rests on the predicted cloud, so this is the only "
             "check on it"]
    if not keep:
        lines.append("  no width rows in this problem")
        return lines

    d = obs[keep] - prd[keep]
    z = d / np.maximum(sd[keep], 1e-30)
    # A scale row is a log spread, so obs - pred IS the log widening the model
    # would have to supply. It can supply it two ways, s and b_1, which enter a
    # scale row additively, so the budget is the prior sd of their sum.
    budget = float(np.hypot(prior.tau_s, prior.sigma_b))

    lines.append(f"  {'row':<40}{'n_c':>6}{'shortfall':>11}{'z':>8}")
    for k in np.argsort(-np.abs(z))[:top]:
        i = keep[k]
        lines.append(f"  {labels[i][:38]:<40}{specs[i].n:>6}{d[k]:>11.3f}"
                     f"{z[k]:>8.2f}")
    lines.append(f"  {len(keep)} width rows, mean shortfall {d.mean():+.3f} log "
                 f"units, max |z| {np.abs(z).max():.2f}")
    lines.append(f"  the model supplies width through s and b_1, prior sd of "
                 f"their sum {budget:.2f}, so the mean shortfall is "
                 f"{abs(d.mean()) / budget:.2f} prior sd of the available widening")
    if abs(d.mean()) / budget > 2.0:
        lines.append("  ABOVE 2 PRIOR SD: the model cannot widen this far without "
                     "fighting its own prior. That is a model problem, not "
                     "something to fix by widening V_c.")
    elif np.abs(z).max() > 3.0:
        lines.append("  Reachable, but some rows are far out relative to their "
                     "error bar. Those rows will dominate the fit for omega; "
                     "check them before trusting the width.")
    return lines


def dbar_absorption(E_means: Sequence, V_chol, problem) -> tuple[float, List[str]]:
    """``(captured fraction, lines)``: can gamma and kappa take the emulator's offset?

    The absorbable subspace is not ``Z`` itself. A location row moves with
    ``gamma_r`` and a scale row does not -- a log IQR is shift invariant -- while
    a scale row moves with ``log kappa_r``. So it is ``Z`` on the location rows
    stacked beside ``Z`` on the scale rows.

    A high captured fraction means the offset is a measurement-map effect the
    model can represent, and the cost is a biased ``a`` rather than a biased
    mechanism. A low one means it lands on ``mu`` instead, and with ``a`` carrying
    most of the Fisher information that is the difference between an absorbable
    nuisance and a corrupted answer.
    """
    from scipy.linalg import solve_triangular

    Z = np.asarray(problem.mech.Z, dtype=float)
    at = {r: i for i, r in enumerate(problem.mech.readouts)}
    dim_z = Z.shape[1]

    rows_g, rows_k, dbar = [], [], []
    for i, plan in enumerate(problem.plans):
        G, K = [], []
        for c in plan.cohort_ids:
            for spec in problem.specs_by_cohort[c]:
                z_r = Z[at[spec.target_id]]
                G.append(np.zeros(dim_z) if spec.is_scale else z_r)
                K.append(z_r if spec.is_scale else np.zeros(dim_z))
        L = np.asarray(V_chol[i])
        rows_g.append(solve_triangular(L, np.array(G), lower=True))
        rows_k.append(solve_triangular(L, np.array(K), lower=True))
        dbar.append(solve_triangular(L, np.asarray(E_means[i], dtype=float),
                                     lower=True))

    D = np.hstack([np.concatenate(rows_g), np.concatenate(rows_k)])
    d = np.concatenate(dbar)
    Qd = _orth(D)
    proj = Qd @ (Qd.T @ d)
    captured = float(np.linalg.norm(proj) / max(np.linalg.norm(d), 1e-30))

    lines = [
        "can the measurement map absorb the emulator's offset? (eq:Ec dbar)",
        f"  |dbar| whitened                {np.linalg.norm(d):>8.2f}",
        f"  fraction in col(gamma, kappa)  {captured:>8.3f}",
        f"  residual, absorbed by nothing  "
        f"{np.sqrt(max(1 - captured ** 2, 0)):>8.3f}",
    ]
    if captured < 0.5:
        lines.append("  Most of the offset is OUTSIDE the measurement map, so "
                     "gamma cannot take it and it lands on mu instead. The "
                     "draft's claim that gamma_r absorbs the emulator's "
                     "systematic error does not hold here.")
    else:
        lines.append("  The map can represent most of the offset, so the cost "
                     "falls on a rather than on the mechanism -- but a is what "
                     "the data constrain.")
    return captured, lines

# ------------------------------------------------ recovery against a known phi*

def map_estimate(model: Callable, model_args: Sequence[Any], *,
                 steps: int = 600, lr: float = 5e-2,
                 init: Optional[Mapping[str, jnp.ndarray]] = None,
                 template: Optional[Mapping[str, jnp.ndarray]] = None,
                 seed: int = 0) -> Tuple[Dict[str, jnp.ndarray], float]:
    """``argmax p(phi | T)`` by Adam on the unconstrained parameters.

    Adam and not a quasi-Newton method: most directions are flat, and a
    curvature step along them is fit to rounding. ``template`` fixes the
    parameter shapes, and defaults to a prior draw. Starts at the prior mode,
    so recovery is not seeded with the answer.
    """
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import initialize_model

    # initialize_model, not a prior draw: it gives the latent sites alone, and a
    # potential in the unconstrained space with the transform Jacobians in it.
    info = initialize_model(jax.random.PRNGKey(seed), model,
                            model_args=tuple(model_args))
    if template is None:
        template = info.param_info.z
    template = {k: jnp.asarray(v) for k, v in template.items()}
    flat, unravel = ravel_pytree(template)

    neg_lp = info.potential_fn
    v = jnp.zeros(flat.size) if init is None \
        else ravel_pytree({k: jnp.asarray(init[k]) for k in template})[0]

    step = jax.jit(jax.value_and_grad(lambda w: neg_lp(unravel(w))))
    m = s = jnp.zeros_like(v)
    b1, b2, eps = 0.9, 0.999, 1e-8
    f = jnp.inf
    for t in range(1, steps + 1):
        f, g = step(v)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g ** 2
        v = v - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(s / (1 - b2 ** t)) + eps)
    # dynamic_args is off, so potential_fn and postprocess_fn close over the args.
    return info.postprocess_fn(unravel(v)), float(f)


#: Below this, the posterior is narrower than the prior by enough to call the
#: component identified. At 1.0 the corpus said nothing about it.
IDENTIFIED_SHRINK = 0.9


@dataclass(frozen=True)
class RecoveryRow:
    """One component of ``phi``, against the truth that generated the data."""

    block: str
    name: str
    truth: float
    mean: float
    sd: float
    prior_sd: float
    lo: float
    hi: float

    @property
    def pinned(self) -> bool:
        """The model held this at a value rather than sampling it.

        Scoring it against the truth as if it were an estimate gives an infinite
        z, which is not a huge error but the absence of one. A pinned component
        is either pinned AT the truth or pinned away from it, and that distance
        is the thing worth reporting.

        Tested against the prior sd rather than against zero: a numpyro
        deterministic site returns the same float every draw, and its sample sd
        comes back at rounding rather than at exactly 0.
        """
        return not (self.sd > 1e-8 * self.prior_sd)

    @property
    def z(self) -> float:
        """``(mean - truth)`` in posterior sd. Only meaningful where identified."""
        return np.nan if self.pinned else (self.mean - self.truth) / self.sd

    @property
    def bias_in_prior_sd(self) -> float:
        """``(mean - truth)`` in PRIOR sd. The reading a pinned component has."""
        return ((self.mean - self.truth) / self.prior_sd
                if self.prior_sd > 0 else np.nan)

    @property
    def shrink(self) -> float:
        """Posterior sd over prior sd. 1 is the prior back, 0 is a point mass."""
        return self.sd / self.prior_sd if self.prior_sd > 0 else np.nan

    @property
    def identified(self) -> bool:
        """Determined by the data. A pinned component is decided, not determined."""
        return bool(not self.pinned and self.shrink < IDENTIFIED_SHRINK)

    @property
    def covered(self) -> bool:
        return bool(self.lo <= self.truth <= self.hi)


def summarise_recovery(truth: Mapping[str, np.ndarray],
                       draws: Mapping[str, np.ndarray],
                       prior_sd: Mapping[str, np.ndarray],
                       *, names: Optional[Mapping[str, Sequence[str]]] = None,
                       level: float = 0.9) -> List[RecoveryRow]:
    """One :class:`RecoveryRow` per component of every block ``truth`` names.

    ``draws`` is the posterior with a leading sample axis. ``prior_sd`` is what
    separates the two ways a component can sit on its truth: a posterior that
    found it, and a posterior that never moved off a prior which happened to be
    centred near it. Only the first is evidence.
    """
    tail = (1.0 - level) / 2.0
    rows: List[RecoveryRow] = []
    for block in truth:
        if block not in draws:
            raise KeyError(f"no posterior draws for {block!r}")
        star = np.atleast_1d(np.asarray(truth[block], dtype=float)).ravel()
        d = np.asarray(draws[block], dtype=float).reshape(
            np.shape(draws[block])[0], -1)
        sd_0 = np.broadcast_to(
            np.atleast_1d(np.asarray(prior_sd[block], dtype=float)).ravel(),
            star.shape)
        if d.shape[1] != star.size:
            raise ValueError(
                f"{block}: truth has {star.size} components, draws have "
                f"{d.shape[1]}")
        label = (list(names[block]) if names and block in names
                 else [f"{block}[{i}]" for i in range(star.size)])
        lo, hi = np.quantile(d, [tail, 1.0 - tail], axis=0)
        rows.extend(
            RecoveryRow(block=block, name=label[i], truth=float(star[i]),
                        mean=float(d[:, i].mean()), sd=float(d[:, i].std(ddof=1)),
                        prior_sd=float(sd_0[i]), lo=float(lo[i]), hi=float(hi[i]))
            for i in range(star.size))
    return rows


def print_recovery(rows: Sequence[RecoveryRow], *, level: float = 0.9,
                   worst: int = 6) -> List[str]:
    """The recovery table as lines to print: per block, then the worst rows.

    Coverage is reported over the identified components and the unidentified ones
    separately. Pooling them hides the failure this is for: an unidentified
    component is covered at ``level`` by construction, so a corpus that determines
    nothing scores perfectly on the pooled number.
    """
    out = [f"recovery against phi*, {level:.0%} intervals",
           f"{'block':<10}{'n':>4}{'ident':>7}{'|z| med':>9}{'|z| max':>9}"
           f"{'cover id':>10}{'cover un':>10}{'shrink med':>12}"]
    pinned_blocks = []
    for block in dict.fromkeys(r.block for r in rows):
        got = [r for r in rows if r.block == block]
        if all(r.pinned for r in got):
            pinned_blocks.append((block, got))
            continue
        ident = [r for r in got if r.identified]
        unid = [r for r in got if not r.identified and not r.pinned]
        # Blank rather than nan where nothing is identified: |z| against a
        # posterior that is still the prior is not a number worth printing.
        z = np.abs([r.z for r in ident])
        med = f"{np.median(z):.2f}" if ident else "-"
        mx = f"{np.max(z):.2f}" if ident else "-"
        out.append(
            f"{block:<10}{len(got):>4}{len(ident):>7}{med:>9}{mx:>9}"
            f"{_frac(ident):>10}{_frac(unid):>10}"
            f"{np.median([r.shrink for r in got]):>12.2f}")

    # Pinned blocks are a claim the fit made, not an estimate it produced, so
    # they are reported by how far the pin sits from the truth in prior sd. Zero
    # is a pin on the answer; anything else is a bias with no error bar on it.
    for block, got in pinned_blocks:
        off = np.abs([r.bias_in_prior_sd for r in got])
        verdict = ("on the truth" if np.nanmax(off) < 1e-9
                   else f"off it by up to {np.nanmax(off):.2f} prior sd")
        out.append(f"{block:<10}{len(got):>4}   PINNED, {verdict}")

    bad = sorted((r for r in rows if r.identified and not r.covered),
                 key=lambda r: -abs(r.z))[:worst]
    if not bad:
        out.append("every identified component covers its truth")
        return out
    out.append(f"\nidentified components missing their truth ({len(bad)} shown):")
    # The block belongs in the label: one parameter appears in mu, in log_omega
    # and in u, and without it three different quantities read as duplicates of
    # one name with disagreeing truths.
    out.append(f"  {'component':<34}{'truth':>10}{'mean':>10}{'sd':>9}{'z':>8}"
               f"{'shrink':>9}")
    for r in bad:
        out.append(f"  {f'{r.block}/{r.name}'[:32]:<34}{r.truth:>10.3f}"
                   f"{r.mean:>10.3f}{r.sd:>9.3f}{r.z:>+8.2f}{r.shrink:>9.2f}")
    return out


def _frac(rows: Sequence[RecoveryRow]) -> str:
    if not rows:
        return "-"
    return f"{sum(r.covered for r in rows) / len(rows):.0%}"

"""Whether the corpus's width evidence is reachable, and who ends up carrying it.

``omega`` is what a virtual population is for, and three things can go wrong with
it that no posterior summary distinguishes:

* :func:`width_gate` -- the cohort reports more spread than the model can produce
  at any allowed ``phi``. Every ``V_c`` rests on the predicted cloud, so a cloud
  that is too narrow makes every error bar too small and nothing downstream
  catches it. Costs one forward pass and needs no fit;
* :func:`dbar_absorption` -- the emulator's systematic offset, and whether the
  measurement map can take it. ``E_c`` is a covariance and carries none of
  ``dbar``; the draft's answer is that ``gamma_r`` absorbs it, but ``gamma = Z a``
  is indexed by READOUT while ``dbar`` is per ROW;
* :func:`width_shortfall` -- the held-out scale rows against a population of
  spread ``omega_0``, which says how much wider the real cohort is before any
  spread parameter has been estimated.

All three return lines rather than printing.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

import jax.numpy as jnp
import numpy as np

__all__ = ["width_gate", "dbar_absorption", "width_shortfall"]


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
                   designs=problem.designs, mass_table=problem.mass_table,
                   elig_fn=problem.elig_fn, elig_at=problem.elig_at)


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

    from qsp_inference.vpop.identifiability import _orth

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


def width_shortfall(samples: Mapping[str, np.ndarray], observed: Sequence,
                    prior, problem, *, n_draws: int = 100,
                    s_star: Optional[float] = None) -> List[str]:
    """The held-out scale rows, predicted at ``omega_0``. sec:flat's width shortfall.

    The point of holding the scale rows out of a flat fit is that this comparison
    is then a prediction and not a residual. Each scale row is a log spread, so
    ``exp(observed - predicted)`` is the factor by which the real cohort is wider
    than a population of spread ``omega_0`` would be. Above one means ``omega_0``
    is too narrow at that readout, and it says so before any spread parameter has
    been estimated.

    Pooled over the cohorts reporting each readout, with a band from the fit's own
    draws so that uncertainty in ``mu`` carries through.
    """
    from qsp_inference.vpop.predict import tau_all

    mu_draws = np.asarray(samples["mu"], dtype=float)
    n = min(n_draws, len(mu_draws))
    idx = np.linspace(0, len(mu_draws) - 1, n).astype(int)
    n_beta = int(np.asarray(problem.mech.beta_species).shape[0])
    obs = np.concatenate([np.asarray(o, dtype=float) for o in observed])
    labels, specs = _row_labels(problem)

    per_readout: dict[str, list[float]] = {}
    for t in idx:
        # b is absent when the flat fit pinned it, which is the default; the
        # shortfall is then predicted at kappa = 1, exactly as that fit assumed.
        a = jnp.asarray(samples["a"][t]) if "a" in samples else jnp.zeros(prior.dim_z)
        b = jnp.asarray(samples["b"][t]) if "b" in samples else jnp.zeros(prior.dim_z)
        beta = (jnp.asarray(samples["beta_free"][t]) if "beta_free" in samples
                else jnp.zeros(n_beta))
        log_R = (jnp.asarray(samples["log_R"][t]) if "log_R" in samples
                 else (jnp.asarray(prior.log_R_0) if prior.n_aux else None))
        taus = tau_all(jnp.asarray(mu_draws[t]), jnp.asarray(prior.omega_0),
                       a, b, beta, problem.plans, problem.specs_by_cohort,
                       problem.refs, problem.mech, log_R=log_R,
                       designs=problem.designs, mass_table=problem.mass_table,
                       elig_fn=problem.elig_fn, elig_at=problem.elig_at)
        pred = np.concatenate([np.asarray(p, dtype=float) for p in taus])
        for i, spec in enumerate(specs):
            if spec.is_scale:
                per_readout.setdefault(spec.target_id, []).append(
                    float(obs[i] - pred[i]))

    lines = [
        "width shortfall on the scale rows, predicted at omega_0 (sec:flat)",
        "  ratio = exp(observed log spread - predicted at omega_0); above 1 "
        "means omega_0 is too narrow",
        f"  {'readout':<34}{'ratio':>8}{'5%':>8}{'95%':>8}{'rows':>7}",
    ]
    n_scale = 0
    for r in problem.mech.readouts:
        if r not in per_readout:
            continue
        d = np.array(per_readout[r])
        rows = sum(1 for s in specs if s.is_scale and s.target_id == r)
        n_scale += rows
        lines.append(f"  {r[:32]:<34}{np.exp(d.mean()):>8.2f}"
                     f"{np.exp(np.percentile(d, 5)):>8.2f}"
                     f"{np.exp(np.percentile(d, 95)):>8.2f}{rows:>7d}")
    lines.append(f"  {n_scale} scale rows of {len(specs)} total")
    if s_star is not None:
        lines.append(f"  phi* has s = {s_star:+.3f}, so the assumed parameters "
                     f"are truly {np.exp(s_star):.2f}x wider than omega_0. "
                     f"Readouts driven by those should show roughly that.")
    return lines

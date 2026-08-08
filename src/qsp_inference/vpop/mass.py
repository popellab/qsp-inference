"""The Laplace metric, as NUTS's mass matrix. eq:post's geometry, computed not adapted.

``G = diag(prior precision) + J^T V^-1 J`` is the Gauss-Newton information of the
posterior at a point: the Laplace metric. A mass matrix should equal the
posterior precision, so what NUTS wants is ``G``, and what its argument is called
is the inverse.

Worth computing rather than adapting. ``dense_mass=True`` tries to learn the same
matrix from the warmup draws, which at a few hundred coordinates and a few
hundred draws it cannot do, and the run then spends its whole trajectory budget
on a metric that is nearly arbitrary. Unlike a reparameterisation, a mass matrix
does not touch the prior: it is a choice of kinetic energy, so the stationary
distribution is the same whatever the matrix and a bad one costs only speed.

``G`` is positive definite by construction, since the prior precision is.

Everything here differentiates :func:`~qsp_inference.vpop.fit.phi_from_sites`,
the same map the model uses, in the same site coordinates the model samples. A
second copy of that map would let a mass matrix be built for a model that is not
the one being sampled, and numpyro would accept it: the blocks would simply
belong to the wrong parameters.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["laplace_blocks", "laplace_covariance", "laplace_inverse_mass",
           "should_fix_mass"]


def _whiten(J: np.ndarray, V_chol, sizes: Sequence[int]) -> np.ndarray:
    """``V^-1/2 J``, block by block. ``sizes`` is each block's row count."""
    from scipy.linalg import solve_triangular

    out, off = [], 0
    for L, k in zip(V_chol, sizes):
        out.append(solve_triangular(np.asarray(L), J[off:off + k], lower=True))
        off += k
    return np.concatenate(out) if out else J


def _row_sizes(problem, row_masks=None) -> list[int]:
    sizes = []
    for i, plan in enumerate(problem.plans):
        k = sum(len(problem.specs_by_cohort[c]) for c in plan.cohort_ids)
        sizes.append(k if row_masks is None
                     else int(np.asarray(row_masks[i]).sum()))
    return sizes


def laplace_blocks(prior, problem, V_chol, *, at: Optional[Mapping[str, Any]] = None,
                   flat: bool = False, row_masks=None):
    """Whitened ``d tau / d site``, one array per site, in the model's coordinates.

    NOT standardised coordinates. The sites here are exactly the sites the model
    samples, with the shapes it samples them in, because the output feeds a mass
    matrix and a coordinate mismatch there is silent.

    ``at`` is the point to linearise about, as site values; the default is the
    prior centre, which is where ``V`` was frozen. ``row_masks`` restricts to a
    subset of each block's rows, for a location-only likelihood.

    Returns ``(names, dims, prior_sd, blocks, labels)``.
    """
    from qsp_inference.vpop.fit import phi_from_sites, site_spec
    from qsp_inference.vpop.predict import tau_all

    spec = site_spec(prior, flat=flat)
    names = [nm for nm, _, _ in spec]
    init = [jnp.asarray((at or {}).get(nm, v)) for nm, v, _ in spec]
    prior_sd = [sd for _, _, sd in spec]

    def tau_of(*values):
        sites = dict(zip(names, values))
        mu, omega, a, b, beta_free, log_R = phi_from_sites(sites, prior, flat=flat)
        taus = tau_all(mu, omega, a, b, beta_free, problem.plans,
                       problem.specs_by_cohort, problem.refs, problem.mech,
                       log_R=log_R, designs=problem.designs,
                       mass_table=problem.mass_table,
                       elig_fn=problem.elig_fn, elig_at=problem.elig_at)
        if row_masks is not None:
            taus = [t[jnp.asarray(np.flatnonzero(np.asarray(m)))]
                    for t, m in zip(taus, row_masks)]
        return jnp.concatenate(taus)

    sizes = _row_sizes(problem, row_masks)
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


def laplace_covariance(prior, problem, V_chol, **kw):
    """``(site names, dims, G^-1)`` at a point. ``G`` is the Laplace metric."""
    names, dims, prior_sd, blocks, _ = laplace_blocks(prior, problem, V_chol, **kw)
    Jw = np.hstack(blocks)
    G = np.diag(_precision(dims, prior_sd)) + Jw.T @ Jw
    C = np.linalg.inv(G)
    return names, dims, 0.5 * (C + C.T)


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


def should_fix_mass(dim: int, warmup: int, factor: int = 3) -> bool:
    """Whether to freeze the computed metric or let warmup re-estimate it.

    numpyro's final adaptation window estimates the mass matrix from scratch, so
    leaving adaptation on discards the Laplace metric and keeps only the benefit
    of having explored well while getting there. That is the right trade when the
    warmup has enough draws to estimate a ``dim x dim`` covariance and the wrong
    one when it does not.

    So the rule is the one that decides which estimator is better, not a global
    switch: below about three draws per dimension, keep the computed metric.
    """
    return warmup < factor * dim

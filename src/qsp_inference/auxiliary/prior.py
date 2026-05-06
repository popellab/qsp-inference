"""Hierarchical prior over auxiliary parameters (option 4: implicit group mean).

The discovered auxiliary members from
:mod:`qsp_inference.auxiliary.discovery` carry one observation per
calibration target, but the *prior* lives at the group level: each group's
:class:`~qsp_inference.auxiliary.AuxiliaryGroupSpec` declares the structural
distribution all its members share.

For each group ``g`` with members ``[m_1, ..., m_n]``::

    log mu_g ~ Normal(base.mu, base.sigma)         # per-sim, NOT in theta
    delta_i ~ Normal(0, member_deviation_sigma)    # per-sim, per-member
    log theta_i = log mu_g + delta_i               # for lognormal base
    theta_i = mu_g + delta_i                       # for normal base

``mu_g`` is sampled per simulation but *not* added to the inferred theta
vector — only the per-member ``theta_i`` are. The marginal joint of the
``log theta`` vector (lognormal base) or ``theta`` vector (normal base)
across siblings is multivariate normal with::

    mean    = base.mu * 1
    cov     = base.sigma**2 * 1 1^T  +  member_deviation_sigma**2 * I

This is a rank-1 + diagonal MVN per group; groups are mutually independent,
giving a block-diagonal joint over the full member vector. The prior class
below evaluates and samples this joint directly — no bespoke sampler, no
explicit ``mu_g`` latent.

This module is the inference-time counterpart to
:func:`qsp_inference.auxiliary.discover_auxiliary_members`. Compose it with
the existing QSP prior (e.g.,
:class:`qsp_inference.priors.copula_prior.GaussianCopulaPrior`) by
concatenating their event dimensions; the joint log-prob is the sum of the
two pieces because QSP and auxiliary draws are independent at the prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.distributions import Distribution, MultivariateNormal

from qsp_inference.auxiliary.config import AuxiliaryGroupSpec
from qsp_inference.auxiliary.discovery import AuxiliaryRegistry


@dataclass(frozen=True)
class _GroupBlock:
    """Cached MVN block for one auxiliary group.

    ``transform`` is ``"exp"`` for lognormal (joint MVN is in log-space) or
    ``"identity"`` for normal (joint MVN is in linear space).
    """

    group_name: str
    member_names: tuple[str, ...]
    mvn: MultivariateNormal
    transform: str  # "exp" or "identity"

    @property
    def n_members(self) -> int:
        return len(self.member_names)


def _build_block(
    group_name: str,
    member_names: tuple[str, ...],
    spec: AuxiliaryGroupSpec,
    *,
    dtype: torch.dtype,
) -> _GroupBlock:
    n = len(member_names)
    mu_vec = torch.full((n,), float(spec.base_prior.mu), dtype=dtype)
    sigma_g = float(spec.base_prior.sigma)
    tau_g = float(spec.member_deviation_sigma)
    cov = (
        sigma_g**2 * torch.ones((n, n), dtype=dtype)
        + tau_g**2 * torch.eye(n, dtype=dtype)
    )
    # tau_g may be 0; the ones-matrix block is rank 1, so cov is at least
    # PSD with one positive eigenvalue. Add a tiny diagonal jitter only when
    # tau_g is exactly zero AND n > 1 (rank-deficient block) to keep
    # MultivariateNormal happy.
    if tau_g == 0.0 and n > 1:
        cov = cov + 1e-12 * torch.eye(n, dtype=dtype)
    mvn = MultivariateNormal(loc=mu_vec, covariance_matrix=cov)
    if spec.base_prior.distribution == "lognormal":
        transform = "exp"
    elif spec.base_prior.distribution == "normal":
        transform = "identity"
    else:  # pragma: no cover — schema validation already filters this
        raise ValueError(
            f"Auxiliary group '{group_name}': unsupported base_prior distribution "
            f"'{spec.base_prior.distribution}'"
        )
    return _GroupBlock(
        group_name=group_name,
        member_names=member_names,
        mvn=mvn,
        transform=transform,
    )


class HierarchicalAuxiliaryPrior(Distribution):
    """Joint prior over all auxiliary parameter members in a registry.

    Block-diagonal in the member-space dimension: each group contributes
    one MVN block (in log-space for lognormal base, linear-space for
    normal base). Groups are independent, so block log-probs sum.

    The flattened parameter vector follows :attr:`param_names` order, which
    is the registry's deterministic ``(group_name, member_name)`` lex
    ordering. Workflows that thread a numpy/pandas representation
    (e.g. ``priors_df``, ``theta_train``) should use :attr:`param_names`
    as the canonical column ordering.

    Args:
        registry: An :class:`~qsp_inference.auxiliary.AuxiliaryRegistry`
            built by :func:`~qsp_inference.auxiliary.discover_auxiliary_members`.
        dtype: Tensor dtype. Defaults to ``torch.float64`` for parity with
            :class:`qsp_inference.priors.copula_prior.GaussianCopulaPrior`.

    Notes:
        - Empty registry yields a zero-dimensional prior whose ``sample``
          returns an empty tensor and whose ``log_prob`` returns zeros
          (vacuous prior). This makes the prior safe to compose
          unconditionally even when a workflow has no auxiliary parameters.
        - The MVN cov for a group is ``sigma**2 * J + tau**2 * I`` where
          ``J`` is the all-ones matrix. For a single-member group this
          reduces to ``sigma**2 + tau**2`` (the σ vs τ split is degenerate
          but correctly inflates the marginal variance).
    """

    has_rsample = False

    def __init__(
        self,
        registry: AuxiliaryRegistry,
        *,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self._registry = registry
        self._dtype = dtype

        blocks: list[_GroupBlock] = []
        for group_name in registry.group_names:
            spec = registry.group_spec(group_name)
            members = tuple(m.name for m in registry.members_in(group_name))
            blocks.append(_build_block(group_name, members, spec, dtype=dtype))
        self._blocks: tuple[_GroupBlock, ...] = tuple(blocks)
        self._param_names: tuple[str, ...] = tuple(
            name for block in self._blocks for name in block.member_names
        )

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([len(self._param_names)]),
            validate_args=False,
        )

    @property
    def param_names(self) -> tuple[str, ...]:
        """Ordered member names, deterministic across runs.

        Same ordering as :attr:`AuxiliaryRegistry.member_names` — by
        group name (alpha) then by member name (alpha).
        """
        return self._param_names

    @property
    def event_dim(self) -> int:
        return len(self._param_names)

    @property
    def n_blocks(self) -> int:
        return len(self._blocks)

    def block_for(self, member_name: str) -> _GroupBlock:
        """Return the group block containing ``member_name``."""
        for block in self._blocks:
            if member_name in block.member_names:
                return block
        raise KeyError(f"Auxiliary member '{member_name}' not in this prior.")

    # ------------------------------------------------------------------
    # torch.distributions API
    # ------------------------------------------------------------------

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw ``sample_shape + (event_dim,)`` samples.

        The returned tensor's columns are ordered according to
        :attr:`param_names`. The lognormal blocks return values in the
        original (positive) parameter scale (``exp(MVN draw)``); normal
        blocks return raw MVN draws.
        """
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)

        if self.event_dim == 0:
            return torch.empty(sample_shape + (0,), dtype=self._dtype)

        outs: list[torch.Tensor] = []
        for block in self._blocks:
            x = block.mvn.sample(sample_shape)
            if block.transform == "exp":
                x = torch.exp(x)
            outs.append(x)
        return torch.cat(outs, dim=-1)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Joint log-density at ``theta`` (last dim = ``event_dim``)."""
        if theta.shape[-1] != self.event_dim:
            raise ValueError(
                f"theta last-dim {theta.shape[-1]} does not match prior event_dim "
                f"{self.event_dim}"
            )
        if self.event_dim == 0:
            return torch.zeros(theta.shape[:-1], dtype=self._dtype)

        theta_d = theta.to(self._dtype)
        total = torch.zeros(theta_d.shape[:-1], dtype=self._dtype)
        offset = 0
        for block in self._blocks:
            n = block.n_members
            x = theta_d[..., offset : offset + n]
            if block.transform == "exp":
                # change of variables: y = exp(x), so log p_y(y) = log p_x(log y) - sum log y
                log_x = torch.log(x)
                lp = block.mvn.log_prob(log_x) - log_x.sum(dim=-1)
            else:
                lp = block.mvn.log_prob(x)
            total = total + lp
            offset += n
        return total

    # ------------------------------------------------------------------
    # Convenience helpers for callers that want per-member dicts rather
    # than flat theta vectors.
    # ------------------------------------------------------------------

    def sample_as_records(
        self, num_samples: int, *, seed: int | None = None
    ) -> list[dict[str, float]]:
        """Draw ``num_samples`` and return one ``{member_name: value}`` dict per sim.

        Convenience for callers (e.g., observable-evaluation helpers) that
        merge auxiliary draws into ``observable.code`` constants per sim.
        Samples are drawn independently of any global RNG when ``seed``
        is given.
        """
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
            with torch.random.fork_rng():
                torch.manual_seed(int(seed))
                draws = self.sample((num_samples,))
        else:
            draws = self.sample((num_samples,))

        records: list[dict[str, float]] = []
        names = self.param_names
        for row in draws.detach().cpu().numpy():
            records.append({name: float(value) for name, value in zip(names, row)})
        return records


# ----------------------------------------------------------------------
# Stand-alone helpers
# ----------------------------------------------------------------------


def merge_into_constants(
    base_constants: dict,
    aux_record: dict,
    *,
    units_by_name: dict,
    ureg,
) -> dict:
    """Merge a per-sim auxiliary record into an observable's ``constants`` dict.

    The QSP simulator emits one trajectory per ``sample_index``; the
    observable-evaluation helper draws one auxiliary record per sample and
    folds it into the existing fixed-constants dict before invoking
    ``observable.code``. This helper keeps the unit-attachment logic in
    one place so callers don't accidentally pass raw floats into the Pint
    namespace.

    Args:
        base_constants: The fixed-constant mapping built from
            ``CalibrationTarget.observable.constants``. Each value is a
            Pint Quantity.
        aux_record: ``{member_name: float}`` from
            :meth:`HierarchicalAuxiliaryPrior.sample_as_records`.
        units_by_name: ``{member_name: pint-parseable units string}``
            from the registry (each
            :class:`AuxiliaryMember` carries its declared units).
        ureg: Pint UnitRegistry consumed by the calibration target.

    Returns:
        New dict with auxiliary entries attached as Pint Quantities.
        Does not mutate ``base_constants``.

    Raises:
        ValueError: an auxiliary name collides with a fixed-constant name.
    """
    out = dict(base_constants)
    for name, value in aux_record.items():
        if name in out:
            raise ValueError(
                f"Auxiliary parameter '{name}' collides with an existing "
                f"observable.constants entry. Pick a different name on the "
                f"AuxiliaryParameter."
            )
        units = units_by_name.get(name)
        if units is None:
            raise ValueError(
                f"Auxiliary parameter '{name}' has no entry in units_by_name; "
                f"the registry-derived unit map is incomplete."
            )
        out[name] = float(value) * ureg(units)
    return out


def build_units_by_name(registry: AuxiliaryRegistry) -> dict[str, str]:
    """Return ``{member_name: units string}`` for every member in ``registry``.

    Convenience for callers wiring :func:`merge_into_constants`.
    """
    return {name: registry.members[name].units for name in registry.member_names}


__all__ = [
    "HierarchicalAuxiliaryPrior",
    "build_units_by_name",
    "merge_into_constants",
]

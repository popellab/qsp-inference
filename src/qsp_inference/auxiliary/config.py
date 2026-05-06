"""Schema and loader for ``auxiliary_config.yaml``.

The file declares hierarchical groups for measurement-bridging auxiliary
parameters that are jointly inferred with the QSP parameters. Members are
*not* listed here — they are auto-discovered by walking calibration target
YAMLs (see :func:`qsp_inference.auxiliary.discover_auxiliary_members`).

Sampling pattern (option 4 — implicit group mean):

    log mu_g ~ Normal(base_prior.mu, base_prior.sigma)         # per-sim
    delta_i ~ Normal(0, member_deviation_sigma)                # per-sim, per-member
    log theta_i = log mu_g + delta_i                           # for lognormal base
    theta_i = mu_g + delta_i                                    # for normal base

``mu_g`` is sampled per-simulation but is *not* added to the inferred theta
vector — only the per-member ``theta_i`` are. The marginal joint prior over
sibling members is multivariate normal in log/linear space with
``base_prior.sigma**2`` shared off-diagonal covariance and
``base_prior.sigma**2 + member_deviation_sigma**2`` on the diagonal. The
prior module (:mod:`qsp_inference.auxiliary.prior`) consumes this closed
form directly; no bespoke sampler needed.

Example::

    groups:
      serum_to_tumor:
        description: >
          Serum:tumor concentration ratio for cytokines reported in human
          serum but consumed in the model as tumor-compartment concentrations.
        base_prior:
          distribution: lognormal
          mu: 0.0          # log-space; ratio centered on 1
          sigma: 0.7       # ~factor-of-two band
        member_deviation_sigma: 0.3
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AuxiliaryBasePrior(BaseModel):
    """Prior on the (per-sim) group mean.

    For ``lognormal``, ``mu`` and ``sigma`` are log-space parameters of the
    underlying Normal — i.e., the actual auxiliary value is
    ``exp(Normal(mu, sigma))`` before any per-member deviation.

    For ``normal``, ``mu`` and ``sigma`` are linear-space parameters and the
    auxiliary value is ``Normal(mu, sigma)`` plus the per-member deviation.
    """

    model_config = ConfigDict(extra="forbid")

    distribution: Literal["lognormal", "normal"] = Field(
        description="Distribution family for the group mean draw."
    )
    mu: float = Field(
        description=(
            "Location parameter (log-space mean for lognormal, linear-space "
            "mean for normal)."
        )
    )
    sigma: float = Field(
        gt=0.0,
        description="Scale parameter (log-space SD for lognormal, linear SD for normal).",
    )


class AuxiliaryGroupSpec(BaseModel):
    """Hierarchical group specification.

    Members are *not* declared here — they are auto-discovered by walking
    calibration targets. The group exists to share a common base prior and
    a fixed between-member deviation SD across all auxiliary parameters
    declared with this ``group`` value.

    ``member_deviation_sigma`` is the constant ``tau_g`` that controls how
    far each member can deviate from the per-sim group draw. Set to ``0.0``
    for groups whose members should track the group mean exactly (e.g.,
    a single-member group used as a YAML organizational namespace).
    """

    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        default="",
        description="Human-readable description of why these parameters share a prior.",
    )
    base_prior: AuxiliaryBasePrior = Field(
        description="Prior on the per-sim group mean.",
    )
    member_deviation_sigma: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Fixed between-member deviation SD (``tau_g``). Same units as the "
            "base prior's scale parameter (log-space for lognormal, linear "
            "for normal). Zero means all members in the group share the same "
            "per-sim draw."
        ),
    )


class AuxiliaryConfig(BaseModel):
    """Top-level config for ``auxiliary_config.yaml``."""

    model_config = ConfigDict(extra="forbid")

    groups: dict[str, AuxiliaryGroupSpec] = Field(
        default_factory=dict,
        description=(
            "Mapping of group name to its prior specification. Group names "
            "must match the ``group`` field of every ``AuxiliaryParameter`` "
            "that references them."
        ),
    )

    @model_validator(mode="after")
    def _reject_empty_group_name(self) -> "AuxiliaryConfig":
        for name in self.groups:
            if not name or not name.strip():
                raise ValueError("Auxiliary group names must be non-empty.")
        return self

    def get(self, group_name: str) -> AuxiliaryGroupSpec | None:
        """Return the spec for ``group_name`` or ``None`` if undeclared."""
        return self.groups.get(group_name)


def load_auxiliary_config(yaml_path: Path | str) -> AuxiliaryConfig:
    """Load and validate ``auxiliary_config.yaml``.

    Returns an empty :class:`AuxiliaryConfig` when the file does not exist
    or is empty — this keeps the downstream registry usable in workflows
    that don't (yet) declare any auxiliary parameters.

    Args:
        yaml_path: Path to the YAML file.

    Returns:
        Validated :class:`AuxiliaryConfig`.

    Raises:
        pydantic.ValidationError: schema validation failure.
    """
    import yaml

    path = Path(yaml_path)
    if not path.exists():
        return AuxiliaryConfig()

    with path.open() as f:
        data = yaml.safe_load(f)

    if data is None:
        return AuxiliaryConfig()

    return AuxiliaryConfig(**data)

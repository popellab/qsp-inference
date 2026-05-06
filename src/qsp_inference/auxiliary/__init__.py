"""Auxiliary (measurement-bridging) parameters for joint QSP inference.

Auxiliary parameters encode the gap between what the literature reports and
what the model species directly is — e.g., a serum:tumor compartment ratio,
a cross-species translation factor, an IHC-score-to-concentration conversion.

They are declared on a calibration target's ``Observable`` (maple side, via
``Observable.auxiliary_parameters``) and consumed inside ``observable.code``
via the ``constants`` dict at evaluation time. This package owns the inference
side: loading the structural prior from ``auxiliary_config.yaml``,
auto-discovering members by walking calibration target YAMLs, and (in
:mod:`qsp_inference.auxiliary.prior`) building the joint hierarchical prior
that gets sampled per-simulation alongside the QSP parameters.

Top-level entry points:

- :func:`load_auxiliary_config` — parse ``auxiliary_config.yaml`` into
  :class:`AuxiliaryConfig`.
- :func:`discover_auxiliary_members` — walk calibration targets, validate
  cross-target consistency, and return an :class:`AuxiliaryRegistry`.
"""
from qsp_inference.auxiliary.config import (
    AuxiliaryBasePrior,
    AuxiliaryConfig,
    AuxiliaryGroupSpec,
    load_auxiliary_config,
)
from qsp_inference.auxiliary.discovery import (
    AuxiliaryMember,
    AuxiliaryRegistry,
    discover_auxiliary_members,
)
from qsp_inference.auxiliary.prior import (
    HierarchicalAuxiliaryPrior,
    build_units_by_name,
    merge_into_constants,
)

__all__ = [
    "AuxiliaryBasePrior",
    "AuxiliaryConfig",
    "AuxiliaryGroupSpec",
    "AuxiliaryMember",
    "AuxiliaryRegistry",
    "HierarchicalAuxiliaryPrior",
    "build_units_by_name",
    "discover_auxiliary_members",
    "load_auxiliary_config",
    "merge_into_constants",
]

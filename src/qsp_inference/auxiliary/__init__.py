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

# The hierarchical prior depends on ``torch``, which is in the optional
# ``sbi`` extra (not part of the default install). Re-export it eagerly when
# torch is available; otherwise leave the names off ``__all__`` so a
# torch-less install can still use the config + discovery pieces.
try:
    from qsp_inference.auxiliary.prior import (  # noqa: F401
        HierarchicalAuxiliaryPrior,
        merge_into_constants,
    )
except ImportError:  # torch missing — prior module is opt-in.
    _PRIOR_EXPORTS: tuple[str, ...] = ()
else:
    _PRIOR_EXPORTS = (
        "HierarchicalAuxiliaryPrior",
        "merge_into_constants",
    )

__all__ = [
    "AuxiliaryBasePrior",
    "AuxiliaryConfig",
    "AuxiliaryGroupSpec",
    "AuxiliaryMember",
    "AuxiliaryRegistry",
    "discover_auxiliary_members",
    "load_auxiliary_config",
    *_PRIOR_EXPORTS,
]

"""Shared per-target resolver for the maple target-data contract.

Both halves of the contract read the same maple structure off a target -- the
observed-data half (quantile anchors, :mod:`qsp_inference.targets.anchors`) and
the prior half (population ``omega`` from ``n_biological`` / ``spread_source``,
:mod:`qsp_inference.targets.omega`). This module is the single seam that turns a
raw ``observed_distribution`` mapping (as it appears in a target YAML) into a
validated maple ``ObservedDistribution``, so every consumer reads the reported
distribution the same way rather than poking raw dict keys.
"""
from __future__ import annotations

from typing import Optional

__all__ = ["parse_observed_distribution", "population_n_biological"]


def parse_observed_distribution(od) -> Optional[object]:
    """Coerce an ``observed_distribution`` into a maple ``ObservedDistribution``.

    Accepts ``None`` (→ ``None``), an already-constructed model (returned as is),
    or a mapping (validated into the model). Returns ``None`` when the mapping
    cannot be validated, so callers can fall back cleanly.
    """
    if od is None:
        return None
    from maple.core.calibration.shared_models import ObservedDistribution

    if isinstance(od, ObservedDistribution):
        return od
    try:
        return ObservedDistribution.model_validate(od)
    except Exception:
        return None


def population_n_biological(observed_distributions) -> Optional[int]:
    """Largest ``n_biological`` among population-feeding distributions, or ``None``.

    ``observed_distributions`` is an iterable of raw mappings or parsed models.
    A distribution counts only when its ``spread_source`` feeds population spread
    (``feeds_population_spread``) and it declares an ``n_biological``. This is the
    ``n`` that weighs data against the class prior in the omega shrinkage.
    """
    ns = []
    for od in observed_distributions:
        model = parse_observed_distribution(od)
        if model is None:
            continue
        n = getattr(model, "n_biological", None)
        if n and getattr(model, "feeds_population_spread", False):
            ns.append(int(n))
    return max(ns) if ns else None

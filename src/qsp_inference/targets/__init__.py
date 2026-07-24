"""The maple target-data contract for population (NLME) inference.

Reads maple calibration/submodel target data structures (``ObservedDistribution``,
population ``samples``, ``spread_source`` / ``n_biological``) into the inputs
inference conditions on. Project code (paths, target dirs, role maps) injects the
resolved sources; the logic here is generic.

Currently houses the observed-data half (quantile anchors). The prior half
(population ``omega`` layered shrinkage) is a planned sibling; both build on a
shared per-target resolver.
"""

from qsp_inference.targets.anchors import (
    ObservedAnchors,
    anchors_from_sources,
    cohort_quantiles,
)

__all__ = ["ObservedAnchors", "anchors_from_sources", "cohort_quantiles"]

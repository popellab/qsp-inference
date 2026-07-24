"""Per-observable observed population samples, from maple calibration targets.

Matches each ``test_statistic_id`` to its calibration-target YAML and runs
maple's ``capture_bootstrap_samples`` to recover the observed sampling
distribution the target's ``distribution_code`` builds. The resulting
per-observable list is the empirical measurement-noise / population shape
(skew / tails / bounds) that downstream code trains and checks against, rather
than a parametric refit of the CI endpoints.

Generic over the target directories (passed in): the maple contract lives here;
a project injects its own calibration-target dirs.

Targets with no matching YAML, no ``distribution_code``, or a closed-form
analytic code (no internal sample array) yield ``None`` for that observable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import yaml

__all__ = ["load_population_samples"]


def load_population_samples(
    test_statistic_ids: Sequence[str],
    calibration_target_dirs: Sequence[str | Path],
    max_samples: int | None = None,
    verbose: bool = True,
) -> list[np.ndarray | None]:
    """Return a list (aligned to ``test_statistic_ids``) of observed population
    sample arrays, or ``None`` per entry.

    Args:
        test_statistic_ids: observable ids to resolve, in order.
        calibration_target_dirs: dirs of calibration-target YAMLs to search.
        max_samples: if set, deterministically thin any larger captured array.
        verbose: print a one-line coverage summary.
    """
    from maple.core.calibration import capture_bootstrap_samples

    path_by_id: dict[str, Path] = {}
    for d in calibration_target_dirs:
        for yp in Path(d).glob("*.yaml"):
            try:
                data = yaml.safe_load(yp.read_text()) or {}
            except yaml.YAMLError:
                continue
            tid = (
                data.get("calibration_target_id")
                or data.get("test_statistic_id")
                or yp.stem
            )
            path_by_id[tid] = yp

    out: list[np.ndarray | None] = []
    n_emp = 0
    missing: list[str] = []
    for tid in test_statistic_ids:
        yp = path_by_id.get(tid)
        samples = None
        if yp is not None:
            try:
                samples = capture_bootstrap_samples(str(yp), max_samples=max_samples)
            except Exception:
                samples = None
        if samples is None:
            missing.append(tid)
        else:
            n_emp += 1
        out.append(samples)

    if verbose:
        print(
            f"  [population-samples] {n_emp}/{len(test_statistic_ids)} observables "
            f"with empirical samples; {len(missing)} parametric fallback"
            + (f": {missing}" if missing else "")
        )
    return out

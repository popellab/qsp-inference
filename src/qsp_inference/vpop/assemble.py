"""The two bits of wiring the primitives do not already cover.

Everything else a run needs is built: ``row_specs`` (rows), ``block_draw_plan``
(resampling), ``quantile_mass_table`` (predict), ``build_h_fn`` (readouts).
Composing them is a handful of lines at the call site and belongs there.

What does not belong there is the scenario table, because the reference-vs-readout
ordering is where the fold-change bug lived, and ``L_R``'s provenance, because a
``Mechanism`` carrying a different correlation than the surrogate trained under
fails silently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from qsp_inference.vpop.readouts import declared_reference

__all__ = ["ScenarioTable", "scenario_table", "prior_cholesky"]


@dataclass(frozen=True)
class ScenarioTable:
    """Which ``(arm, time)`` each scenario index means, and what each readout reads.

    ``scenario_of[r]`` is ``(readout,)`` or ``(reference, readout)``, reference
    first, which is the order ``build_h_fn`` indexes: a body reaches its declared
    reference at ``x[0]`` and its own time last.
    """

    scenarios: Tuple[Tuple[str, float], ...]
    scenario_of: Dict[str, Tuple[int, ...]]

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    def arms(self) -> Tuple[str, ...]:
        seen: list = []
        for a, _ in self.scenarios:
            if a not in seen:
                seen.append(a)
        return tuple(seen)


def scenario_table(targets: Mapping[str, Mapping[str, Any]],
                   arm_of: Mapping[str, str],
                   readouts: Sequence[str]) -> ScenarioTable:
    """The ``(arm, time)`` scenarios ``readouts`` need, and each one's indices.

    Keyed by ``(arm, time)`` rather than by cohort: cohorts sharing an arm read
    the same trajectory, and what separates them is eq:disc and the weighting,
    not the mechanism. On the pdac corpus that is 7 scenarios rather than 30,
    which is what the campaign is sized on.
    """
    want: Dict[str, Tuple[Tuple[str, float], ...]] = {}
    for r in readouts:
        if r not in arm_of:
            raise KeyError(f"{r} has no arm; arm_of must cover every readout")
        arm = arm_of[r]
        t = float((targets[r].get("observable") or {}).get("readout_time") or 0.0)
        ref = declared_reference(targets[r])
        want[r] = (() if ref is None else ((arm, float(ref)),)) + ((arm, t),)

    keys = sorted({k for v in want.values() for k in v})
    index = {k: i for i, k in enumerate(keys)}
    return ScenarioTable(tuple(keys), {r: tuple(index[k] for k in v)
                                       for r, v in want.items()})


def prior_cholesky(prior_spec, param_names: Optional[Sequence[str]] = None
                   ) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """``L_R`` from the same ``PriorSpec`` the theta pool was drawn from.

    Derived rather than passed alongside. ``L_R`` sets the geometry of the patient
    cloud in eq:crn, and the surrogate was trained on draws from this correlation,
    so a ``Mechanism`` carrying a different one evaluates the surrogate off the
    manifold it learned -- with no symptom, because every number stays plausible.
    Taking both from one object removes the chance to disagree.
    """
    from qsp_inference.priors.inference_prior import build_prior_pair

    pair = build_prior_pair(prior_spec, verbose=False)
    names = tuple(pair.param_names)
    R = getattr(pair.prior, "_R", None)
    if R is None:
        raise TypeError(
            "the prior carries no correlation matrix, so there is no L_R to take "
            "from it. eq:crn needs the composite copula prior; build the spec "
            "with a submodel_priors_yaml."
        )
    if param_names is not None and tuple(param_names) != names:
        raise ValueError(
            "param order differs from the prior's. L_R is indexed by parameter, "
            "so a reordering silently permutes the patient cloud."
        )
    return np.linalg.cholesky(np.asarray(R, dtype=float)), names

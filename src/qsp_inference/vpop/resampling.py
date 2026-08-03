"""Who gets resampled with whom, from the cohort registry's blocks.

``V^boot_B`` (eq:V) resamples patients from the predicted cloud. Cohorts joined by
a shared row are disjoint people and draw independently; cohorts that share people
draw once from the block's patient set. ``covariance_blocks`` gives the partition
but not which relation applied, so the registry's ``blocks`` supply the rest: a
counted ``PatientBlock`` carries the strata a joint draw needs, an uncounted one
only says drawing its cohorts apart is wrong.

No covariance and no emulator here, which is what makes it testable against the
registry alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from maple.core.calibration.cohort import CohortRegistry, PatientBlock
from maple.core.calibration.registry_audit import covariance_blocks

__all__ = ["DrawGroup", "BlockPlan", "block_draw_plan", "draw_block_indices"]


@dataclass(frozen=True)
class DrawGroup:
    """One resample of ``n_draw`` patients, serving one cohort or several.

    ``member_positions`` indexes into that draw. A cohort's positions number its
    ``n_c``, and two cohorts intersect in exactly the patients their strata share.
    """

    cohort_ids: Tuple[str, ...]
    n_draw: int
    member_positions: Dict[str, Tuple[int, ...]]
    block_id: Optional[str] = None  # set iff the draw is joint

    @property
    def is_joint(self) -> bool:
        return self.block_id is not None


@dataclass(frozen=True)
class BlockPlan:
    """One covariance block. ``cohort_ids`` fixes the row order of ``V_B``.

    ``unhonoured`` names uncounted blocks spanning this component: real dependence
    the draw cannot reproduce, so ``V_B`` understates those correlations.
    """

    cohort_ids: Tuple[str, ...]
    groups: Tuple[DrawGroup, ...]
    unhonoured: Tuple[str, ...] = ()

    @property
    def is_exact(self) -> bool:
        return not self.unhonoured


def _joint_group(block: PatientBlock, n_c: Dict[str, int]) -> DrawGroup:
    """Lay the strata out as contiguous runs; a cohort takes the runs naming it."""
    positions: Dict[str, Tuple[int, ...]] = {}
    at = 0
    for s in block.strata:
        run = tuple(range(at, at + s.n))
        for cid in s.cohorts:
            positions[cid] = positions.get(cid, ()) + run
        at += s.n

    for cid, pos in positions.items():
        if len(pos) != n_c[cid]:
            raise ValueError(
                f"block {block.block_id}: strata place {len(pos)} patients in "
                f"{cid}, which declares n_c={n_c[cid]}"
            )
    return DrawGroup(
        cohort_ids=tuple(block.cohorts),
        n_draw=at,
        member_positions=positions,
        block_id=block.block_id,
    )


def _lone_group(cohort_id: str, n: int) -> DrawGroup:
    return DrawGroup((cohort_id,), n, {cohort_id: tuple(range(n))})


def block_draw_plan(
    registry: CohortRegistry, targets: Dict[str, Dict[str, Any]]
) -> List[BlockPlan]:
    """One :class:`BlockPlan` per covariance block, in ``covariance_blocks`` order.

    ``targets`` is the loaded target set, keyed by id, as ``covariance_blocks``
    takes it: it supplies the shared-row edges that the registry's blocks do not.
    """
    n_c = {c.cohort_id: c.n_c for c in registry.cohorts}
    plans = []
    for component in covariance_blocks(targets, registry):
        counted: Dict[str, PatientBlock] = {}
        unhonoured = set()
        for cid in component:
            block = registry.counted_block_for(cid)
            if block is not None:
                counted[block.block_id] = block
            unhonoured.update(b.block_id for b in registry.uncounted_blocks_for(cid))

        placed = {cid for b in counted.values() for cid in b.members}
        groups = [_joint_group(counted[bid], n_c) for bid in sorted(counted)]
        groups += [_lone_group(cid, n_c[cid]) for cid in sorted(component - placed)]

        plans.append(
            BlockPlan(
                cohort_ids=tuple(sorted(component)),
                groups=tuple(groups),
                unhonoured=tuple(sorted(unhonoured)),
            )
        )
    return plans


def _group_probs(
    group: DrawGroup, probs: Optional[Mapping[str, np.ndarray]]
) -> Optional[np.ndarray]:
    """One sampling weight for the group. Joint members must agree on eligibility.

    The toy applied eligibility per cohort, as the draw's probability. A joint
    group draws once for several cohorts, so members that filter differently have
    no common draw. Raise rather than pick one.
    """
    if probs is None:
        return None
    first = probs.get(group.cohort_ids[0])
    for cid in group.cohort_ids[1:]:
        other = probs.get(cid)
        same = (first is None and other is None) or (
            first is not None
            and other is not None
            and np.array_equal(np.asarray(first), np.asarray(other))
        )
        if not same:
            raise ValueError(
                f"block {group.block_id}: {group.cohort_ids[0]} and {cid} share a "
                f"draw but declare different eligibility weights"
            )
    if first is None:
        return None
    w = np.asarray(first, dtype=float)
    return w / w.sum()


def draw_block_indices(
    plan: BlockPlan,
    rng: np.random.Generator,
    n_boot: int,
    n_cloud: int,
    probs: Optional[Mapping[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Cloud indices per cohort, ``(n_boot, n_c)``. A joint group shares its draw.

    ``probs`` maps cohort to a cloud-length eligibility weight, unnormalised.
    """
    out: Dict[str, np.ndarray] = {}
    for group in plan.groups:
        idx = rng.choice(
            n_cloud,
            size=(n_boot, group.n_draw),
            replace=True,
            p=_group_probs(group, probs),
        )
        for cid in group.cohort_ids:
            out[cid] = idx[:, list(group.member_positions[cid])]
    return out

"""Who is drawn with whom, and the ``V_B`` that follows. eq:V, eq:Ec, eq:Vsplit.

Two halves of one question. The registry's blocks say which cohorts a resample
has to draw together: cohorts joined by a shared row are disjoint people and draw
independently, cohorts that share people draw once from the block's patient set.
``V^boot_B`` is then the covariance of that block's rows over those resamples,
``E_B`` is the surrogate's own error carried to the same statistics, and eq:Vsplit
adds them.

One matrix per covariance block, rows ordered by ``plan.cohort_ids``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from maple.core.calibration.cohort import CohortRegistry, PatientBlock
from maple.core.calibration.registry_audit import covariance_blocks

__all__ = [
    # who is drawn with whom
    "DrawGroup", "BlockPlan", "block_draw_plan", "restrict_plans",
    "draw_block_indices",
    # and what that makes V_B
    "BlockCovariance", "row_offsets", "bootstrap_V", "statistic_diffs",
    "emulator_E", "assemble_V",
]


# ------------------------------------------------------- who is drawn with whom

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


def restrict_plans(plans: List[BlockPlan], keep) -> List[BlockPlan]:
    """Drop cohorts reporting no row from each block's row order; drop empty blocks.

    ``covariance_blocks`` partitions every cohort the registry declares, and a
    corpus fits fewer than it declares. Only ``cohort_ids`` is filtered, never
    ``groups``: a dropped member of a counted block still occupies its strata, and
    removing it there would change the overlap its co-members draw against.
    """
    keep = set(keep)
    out = []
    for plan in plans:
        cohort_ids = tuple(c for c in plan.cohort_ids if c in keep)
        if not cohort_ids:
            continue
        out.append(plan if cohort_ids == plan.cohort_ids
                   else BlockPlan(cohort_ids, plan.groups, plan.unhonoured))
    return out


def draw_block_indices(
    plan: BlockPlan,
    rng: np.random.Generator,
    n_boot: int,
    n_cloud: int,
) -> Dict[str, np.ndarray]:
    """Cloud indices per cohort, ``(n_boot, n_c)``. A joint group shares its draw.

    Uniform over the cloud. Where a cohort declares an eligibility criterion the
    draw would have to be weighted by it, and a joint group would need its members
    to agree on that weight; no corpus declares one, so the branch is not carried.
    """
    out: Dict[str, np.ndarray] = {}
    for group in plan.groups:
        idx = rng.choice(n_cloud, size=(n_boot, group.n_draw), replace=True)
        for cid in group.cohort_ids:
            out[cid] = idx[:, list(group.member_positions[cid])]
    return out

# ------------------------------------------------------------------- V_B

RowsFn = Callable[[str, np.ndarray], np.ndarray]


def row_offsets(plan: BlockPlan, k_of: Mapping[str, int]) -> Dict[str, slice]:
    """Where each cohort's rows sit in ``V_B``. ``k_of`` is its ``K_c``."""
    out, at = {}, 0
    for cid in plan.cohort_ids:
        out[cid] = slice(at, at + k_of[cid])
        at += k_of[cid]
    return out


def bootstrap_V(
    plan: BlockPlan,
    rows_fn: RowsFn,
    rng: np.random.Generator,
    *,
    n_boot: int,
    n_cloud: int,
) -> np.ndarray:
    """``V^boot_B`` at a plug-in (eq:V): covariance of the block's rows over resamples.

    ``rows_fn(cohort_id, indices)`` returns that cohort's ``K_c`` statistics for the
    given cloud members. The caller owns the cloud, so an index means the same
    simulated patient in every cohort, which is what makes a joint draw shared.
    """
    idx = draw_block_indices(plan, rng, n_boot, n_cloud)
    reps = np.array([
        np.concatenate([
            np.atleast_1d(np.asarray(rows_fn(cid, idx[cid][t]), dtype=float))
            for cid in plan.cohort_ids
        ])
        for t in range(n_boot)
    ])
    k = reps.shape[1]
    return np.cov(reps, rowvar=False).reshape(k, k)


def statistic_diffs(
    x_emu,
    x_sim,
    plans: Sequence[BlockPlan],
    specs_by_cohort,
    refs,
    mech,
    *,
    n_cloud: int,
    n_draw: int,
    rng: np.random.Generator,
    designs=None,
    mass_table=None,
) -> np.ndarray:
    """``(n_draw, sum K_B)`` surrogate-minus-simulator differences, the input to eq:Ec.

    ``x_emu`` and ``x_sim`` are the readouts of the SAME pool of patients, one
    from the surrogate and one from the simulator. Each draw resamples a cloud
    and both sides read the same members, so the cloud's Monte Carlo fluctuation
    is common to the two and differences out. What survives is the surrogate's own
    error carried to the reported statistics, rather than measured on the readouts
    and expanded, which is the point: a per-readout error does not map to a
    per-statistic one, and a quantile row and a spread row do not feel it alike.

    eq:disc sits at its no-discrepancy point, matching where ``V`` is frozen.
    """
    import jax.numpy as jnp

    from qsp_inference.vpop.predict import tau_from_readouts

    x_emu, x_sim = jnp.asarray(x_emu), jnp.asarray(x_sim)
    if x_emu.shape != x_sim.shape:
        raise ValueError(
            f"the two evaluations must cover the same patients and readouts; "
            f"got {x_emu.shape} and {x_sim.shape}"
        )
    zero_a = jnp.zeros(mech.Z_a.shape[1])
    zero_b = jnp.zeros(mech.Z_b.shape[1])
    args = (plans, specs_by_cohort, refs, mech)
    kw = dict(designs=designs, mass_table=mass_table)

    out = []
    for _ in range(n_draw):
        idx = jnp.asarray(rng.integers(0, x_emu.shape[0], n_cloud))
        te = tau_from_readouts(x_emu[idx], zero_a, zero_b, *args, **kw)
        ts = tau_from_readouts(x_sim[idx], zero_a, zero_b, *args, **kw)
        out.append(np.concatenate([np.asarray(e) - np.asarray(s)
                                   for e, s in zip(te, ts)]))
    return np.array(out)


def emulator_E(
    plans: Sequence[BlockPlan],
    diffs: np.ndarray,
    k_of: Mapping[str, int],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """``E_B`` by eq:Ec, from surrogate-minus-simulator differences in the statistics.

    ``diffs`` is ``(L, sum K_B)`` in block then ``plan.cohort_ids`` order. The toy
    made it by evaluating one cloud through both the emulator and the true ODE. At
    QSP scale the second evaluation is the MATLAB simulator, so the rows must come
    from pool members the emulator did not train on.

    Returns the per-block covariances and the per-block mean difference.
    """
    diffs = np.asarray(diffs, dtype=float)
    blocks, means, at = [], [], 0
    for plan in plans:
        k = sum(k_of[cid] for cid in plan.cohort_ids)
        sl = diffs[:, at:at + k]
        blocks.append(np.cov(sl, rowvar=False).reshape(k, k))
        means.append(sl.mean(axis=0))
        at += k
    if at != diffs.shape[1]:
        raise ValueError(f"diffs has {diffs.shape[1]} columns, plans want {at}")
    return blocks, means


@dataclass(frozen=True)
class BlockCovariance:
    """``V_B`` and its Cholesky per block, with what the draw could not honour.

    ``unhonoured`` names uncounted blocks: cohorts the registry says share patients
    but whose overlap nobody counted, so those off-diagonals are zero and ``V``
    understates them. It travels with the matrices so a caller cannot take one
    without the other.
    """

    V: List[np.ndarray]
    chol: List[np.ndarray]
    unhonoured: Tuple[str, ...] = ()


def study_effect(
    plans: Sequence[BlockPlan],
    specs_by_cohort: Mapping[str, Sequence],
    study_of: Mapping[str, str],
    tau_eta: float,
) -> list:
    """eq:studymarg's ``tau_eta^2 sum_s 1_s 1_s'``, one matrix per block.

    eq:studyeff gives every study a shared level offset and marginalises exactly,
    since it is Gaussian and additive: what reaches the likelihood is a rank-one
    block per study rather than a sampled site. So this is the whole of it.

    Location rows only. A width row is untouched, because eta is a level effect
    and eq:disc already gives widths their own scale term.

    The index is the study, not the cohort: arms of one trial share a lab and a
    patient stream, so their levels move together. ``study_of`` maps a cohort id
    onto its study and is the caller's to supply, the corpus knowing which arms
    belong to which trial.

    ``tau_eta = 0`` returns zeros, which is eq:studyeff switched off and the old
    ``V`` bit for bit.
    """
    from qsp_inference.vpop.rows import SCALE_STATS

    out = []
    for plan in plans:
        studies, k = [], 0
        for cid in plan.cohort_ids:
            for spec in specs_by_cohort[cid]:
                studies.append(None if spec.stat in SCALE_STATS
                               else study_of.get(cid, cid))
                k += 1
        M = np.zeros((k, k))
        if tau_eta:
            for s in {x for x in studies if x is not None}:
                ind = np.array([1.0 if x == s else 0.0 for x in studies])
                M += (tau_eta ** 2) * np.outer(ind, ind)
        out.append(M)
    return out


def assemble_V(
    plans: Sequence[BlockPlan],
    v_boot: Sequence[np.ndarray],
    E: Optional[Sequence[np.ndarray]] = None,
    *,
    ridge: float = 1e-6,
    eta: Optional[Sequence[np.ndarray]] = None,
) -> BlockCovariance:
    """eq:Vsplit, ``V_B = V^boot_B + E_B``, plus eq:studymarg's ``eta`` when given.

    The ridge keeps the Cholesky well behaved when two of a block's readouts are
    nearly collinear, which a shared denominator makes likely.

    It is applied in CORRELATION form, so it is dimensionless: scale to unit
    diagonal, floor there, and carry the scale back through
    ``chol(D C D) = D chol(C)``. An absolute ridge cannot regularise a block
    whose rows are in different units, which every block here is: added to the
    raw matrix it is overwhelming for the smallest row and invisible to the
    largest, and the Cholesky then runs on a matrix whose diagonal spans the
    same range. Forward substitution compounds that down the block, which is how
    a 68-row block reached a condition number of 1e59. Read as a claim, the
    ridge now says no two rows of a block are correlated beyond ``1 - ridge``;
    the cost is inflating each row's sd by ``sqrt(1 + ridge)``, 5e-7 here.
    """
    Vs, chols = [], []
    for i, Vb in enumerate(v_boot):
        V = np.asarray(Vb, dtype=float).copy()
        if E is not None:
            V = V + np.asarray(E[i], dtype=float)
        if eta is not None:
            V = V + np.asarray(eta[i], dtype=float)
        d = np.sqrt(np.diag(V))
        if not np.all(d > 0):
            # A row with no sampling variance is the corpus claiming a number
            # measured to infinite precision, and no ridge makes that true. It
            # is also where a gradient dies, since sqrt has infinite slope at 0.
            bad = np.flatnonzero(d <= 0)
            raise ValueError(
                f"block {'+'.join(plans[i].cohort_ids)}: rows {bad.tolist()} "
                "have zero variance in V. Nothing is measured that precisely, "
                "so this is a degenerate readout rather than a tight one."
            )
        C = V / np.outer(d, d)
        C[np.diag_indices_from(C)] += ridge
        Vs.append(d[:, None] * C * d[None, :])
        chols.append(d[:, None] * np.linalg.cholesky(C))
    return BlockCovariance(
        V=Vs,
        chol=chols,
        unhonoured=tuple(sorted({b for p in plans for b in p.unhonoured})),
    )
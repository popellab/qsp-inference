"""``V_B`` per block: eq:V for the bootstrap, eq:Ec for the emulator, eq:Vsplit to add.

One matrix per covariance block, rows ordered by ``plan.cohort_ids``. Where the
toy had one block per cohort, a counted ``PatientBlock`` now puts the cohorts that
share patients in one matrix, and their shared draw is what fills the off-diagonal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from qsp_inference.vpop.resampling import BlockPlan, draw_block_indices

__all__ = [
    "BlockCovariance",
    "row_offsets",
    "bootstrap_V",
    "emulator_E",
    "assemble_V",
    "subset_V",
]

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
    probs: Optional[Mapping[str, np.ndarray]] = None,
) -> np.ndarray:
    """``V^boot_B`` at a plug-in (eq:V): covariance of the block's rows over resamples.

    ``rows_fn(cohort_id, indices)`` returns that cohort's ``K_c`` statistics for the
    given cloud members. The caller owns the cloud, so an index means the same
    simulated patient in every cohort, which is what makes a joint draw shared.
    """
    idx = draw_block_indices(plan, rng, n_boot, n_cloud, probs)
    reps = np.array([
        np.concatenate([
            np.atleast_1d(np.asarray(rows_fn(cid, idx[cid][t]), dtype=float))
            for cid in plan.cohort_ids
        ])
        for t in range(n_boot)
    ])
    k = reps.shape[1]
    return np.cov(reps, rowvar=False).reshape(k, k)


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


def assemble_V(
    plans: Sequence[BlockPlan],
    v_boot: Sequence[np.ndarray],
    E: Optional[Sequence[np.ndarray]] = None,
    *,
    ridge: float = 1e-10,
) -> BlockCovariance:
    """eq:Vsplit, ``V_B = V^boot_B + E_B``.

    The ridge keeps the Cholesky well behaved when two of a block's readouts are
    nearly collinear, which a shared denominator makes likely.
    """
    Vs, chols = [], []
    for i, Vb in enumerate(v_boot):
        V = np.asarray(Vb, dtype=float).copy()
        if E is not None:
            V = V + np.asarray(E[i], dtype=float)
        V = V + ridge * np.eye(V.shape[0])
        Vs.append(V)
        chols.append(np.linalg.cholesky(V))
    return BlockCovariance(
        V=Vs,
        chol=chols,
        unhonoured=tuple(sorted({b for p in plans for b in p.unhonoured})),
    )


def subset_V(cov: BlockCovariance, masks: Sequence[np.ndarray]) -> BlockCovariance:
    """Restrict each ``V_B`` to a subset of its rows and refactor.

    A Cholesky factor cannot be subset directly. Used to give the flat fit the
    location rows while leaving the scale rows out of the likelihood.
    """
    Vs, chols = [], []
    for V, mask in zip(cov.V, masks):
        idx = np.flatnonzero(np.asarray(mask))
        Vn = np.asarray(V)[np.ix_(idx, idx)]
        Vs.append(Vn)
        chols.append(np.linalg.cholesky(Vn))
    return BlockCovariance(V=Vs, chol=chols, unhonoured=cov.unhonoured)
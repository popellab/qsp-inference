"""``Z_r``: readout attributes with no mechanistic twin. eq:disc.

``gamma_r = Z_r' a`` and ``log kappa_r = Z_r' b``, so ``Z`` decides which readouts
pool their measurement correction. Two properties matter and neither needs a fit:

* every column wants at least two readouts, or its coefficient trades off freely
  against the intercept and only the prior separates them;
* ``rank(Z)`` wants to stay well below ``M``, since at ``rank(Z) = M`` gamma is a
  free per-readout intercept and absorbs beta's location signal entirely.

Two columns can also be the *same vector*, when a corpus happens to measure one
kind of quantity by one assay and nothing else. No data separates those, so
:func:`build_Z` merges them into one honestly-named column rather than leaving
two that look independent. Breaking such a confound needs a new target, not a
better encoding.

:func:`z_conditioning` reports all of it. Run it before any fit; it costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

__all__ = ["ZDesign", "build_Z", "z_conditioning"]

#: Separator in a merged column name. Reads as "aliased with".
ALIAS = "~"


@dataclass(frozen=True)
class ZDesign:
    """``Z`` and its labels. Rows are readouts in ``readouts`` order."""

    Z: np.ndarray
    columns: Tuple[str, ...]
    readouts: Tuple[str, ...]
    merged: Tuple[Tuple[str, ...], ...] = ()   # columns no data can separate
    dropped: Tuple[str, ...] = ()              # folded to the reference level

    @property
    def shape(self) -> Tuple[int, int]:
        return self.Z.shape


def _attr(target: Mapping[str, Any], name: str) -> Optional[str]:
    return ((target.get("observable") or {}).get("readout") or {}).get(name)


def build_Z(
    targets: Mapping[str, Dict[str, Any]],
    *,
    kind_reference: str = "density",
    modality_reference: str = "mihc",
    modality_groups: Optional[Mapping[str, str]] = None,
    intercept: bool = True,
    drop_singletons: bool = True,
    merge_aliases: bool = True,
) -> ZDesign:
    """Crossed indicators for quantity kind and assay modality, against references.

    ``modality_groups`` maps a modality onto the column it contributes to, which is
    how rare assays pool instead of each carrying a column of support one.

    ``drop_singletons`` folds a readout that is alone in its category to the
    reference level: with one member there is nothing to estimate, and the column
    would only trade against the intercept. ``merge_aliases`` collapses columns
    that are the same vector, which is a confound in the corpus rather than a
    choice of coding.
    """
    readouts = tuple(sorted(targets))
    kind_of, mod_of = {}, {}
    for r in readouts:
        kind_of[r] = _attr(targets[r], "quantity_kind")
        raw = _attr(targets[r], "assay_modality")
        mod_of[r] = (modality_groups or {}).get(raw, raw)

    kinds = sorted({k for k in kind_of.values() if k and k != kind_reference})
    mods = sorted({m for m in mod_of.values() if m and m != modality_reference})

    names = [f"kind:{k}" for k in kinds] + [f"assay:{m}" for m in mods]
    cols: Dict[str, np.ndarray] = {}
    for name in names:
        prefix, value = name.split(":", 1)
        source = kind_of if prefix == "kind" else mod_of
        cols[name] = np.array([1.0 if source[r] == value else 0.0 for r in readouts])

    dropped: List[str] = []
    if drop_singletons:
        for name in list(cols):
            if cols[name].sum() <= 1:
                dropped.append(name)
                del cols[name]

    merged: List[Tuple[str, ...]] = []
    if merge_aliases:
        names = list(cols)
        seen: List[str] = []
        for name in names:
            hit = next((s for s in seen if np.array_equal(cols[s], cols[name])), None)
            if hit is None:
                seen.append(name)
                continue
            new = f"{hit}{ALIAS}{name}"
            cols[new] = cols.pop(hit)
            del cols[name]
            seen[seen.index(hit)] = new
            merged.append((hit, name))

    ordered = (["intercept"] if intercept else []) + list(cols)
    Z = np.column_stack(
        ([np.ones(len(readouts))] if intercept else []) + [cols[c] for c in ordered
                                                           if c != "intercept"]
    ) if ordered else np.zeros((len(readouts), 0))

    return ZDesign(
        Z=Z,
        columns=tuple(ordered),
        readouts=readouts,
        merged=tuple(merged),
        dropped=tuple(sorted(dropped)),
    )


def z_conditioning(design: ZDesign) -> Dict[str, Any]:
    """How well the readouts pin down the columns of ``Z``. No fit required.

    ``support`` counts readouts carrying each column, ``rank`` against ``M`` says
    whether gamma can saturate readout space, and ``deficient`` is true when some
    column is still a combination of the others.
    """
    Z = np.asarray(design.Z, dtype=float)
    support = (np.abs(Z) > 0).sum(axis=0)
    sv = np.linalg.svd(Z, compute_uv=False) if Z.size else np.array([1.0])
    rank = int(np.linalg.matrix_rank(Z)) if Z.size else 0
    return {
        "support": dict(zip(design.columns, support.tolist())),
        "singletons": tuple(c for c, s in zip(design.columns, support) if s <= 1),
        "cond": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "rank": rank,
        "n_columns": Z.shape[1],
        "M": Z.shape[0],
        "deficient": rank < Z.shape[1],
        "saturates": rank >= Z.shape[0],
    }

"""Between-patient correlation from a factor model. eq:crn's ``R``.

This is not the correlation a stage-1 posterior carries. That one is epistemic --
"the same data constrained these jointly" -- and reading it as a between-patient
correlation asserts something quite different and unevidenced: that a patient high
in one parameter is high in the other. The two are separate objects and only this
one belongs in eq:crn.

Nothing measures between-patient correlation either, so it is elicited the same
way the widths are: as a claim about what drives a parameter. Each parameter loads
on at most a few latent patient axes, and the correlation is what those shared
loadings imply.

    z_j = sum_k lambda_jk f_k + psi_j e_j,   sum_k lambda_jk^2 + psi_j^2 = 1
    R    = Lambda Lambda' + diag(psi^2)

Positive definite by construction, so it never needs a nearest-correlation
projection -- which matters because such a projection rewrites entries far from
the violation it was called to fix, and what you reviewed stops being what you
shipped. Eliciting pairs directly cannot offer that: a pairwise matrix is
essentially never positive definite, and at P parameters there are P(P-1)/2 of
them to judge rather than P.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

__all__ = ["STRENGTH", "MAX_COMMUNALITY", "correlation_from_loadings"]

#: Loading magnitude per strength bin. The squares are the share of a patient's
#: variation in that parameter the axis explains: 64%, 25%, 9%. Bins rather than
#: free numbers because an elicitation cannot support three significant figures,
#: and printing them invites a reader to believe them.
STRENGTH = {"primary": 0.8, "secondary": 0.5, "weak": 0.3}

#: Ceiling on a parameter's communality. Leaving at least 19% of every
#: parameter's variance idiosyncratic keeps ``R`` strictly positive definite
#: rather than merely semidefinite, so a Cholesky of it cannot fail, and it
#: encodes the claim that no modelled parameter is fully determined by the axes.
MAX_COMMUNALITY = 0.81


def correlation_from_loadings(
    param_names: Sequence[str],
    loadings: Mapping[str, Sequence[Mapping]],
    axes: Sequence[str],
    *,
    strength: Mapping[str, float] = STRENGTH,
    max_communality: float = MAX_COMMUNALITY,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """``(R, Lambda, unknown_axes)`` for ``param_names``, in that order.

    Args:
        param_names: parameters in the order eq:crn expects them. A parameter with
            no entry in ``loadings`` is uncorrelated with everything, which is the
            default and not a failure.
        loadings: name -> list of ``{"axis", "strength", "sign"}``.
        axes: the declared axis names. A loading naming anything else is dropped
            and reported rather than silently creating an axis, since a typo would
            otherwise become a factor that nothing else loads on.
        strength: strength bin -> loading magnitude.
        max_communality: rescale a parameter's loadings if they exceed this.

    Returns:
        ``R`` (P, P) positive definite with unit diagonal, the loading matrix
        (P, K), and the sorted axis names that were dropped.
    """
    idx = {a: k for k, a in enumerate(axes)}
    lam = np.zeros((len(param_names), len(axes)), dtype=float)
    unknown: set[str] = set()

    for j, name in enumerate(param_names):
        for load in loadings.get(name) or ():
            axis = load["axis"]
            if axis not in idx:
                unknown.add(axis)
                continue
            if load["strength"] not in strength:
                raise ValueError(
                    f"{name}: unknown loading strength {load['strength']!r}. "
                    f"Known: {sorted(strength)}"
                )
            sign = 1.0 if load["sign"] == "+" else -1.0
            lam[j, idx[axis]] = sign * strength[load["strength"]]

    communality = (lam**2).sum(axis=1)
    over = communality > max_communality
    if over.any():
        lam[over] *= np.sqrt(max_communality / communality[over])[:, None]
        communality = (lam**2).sum(axis=1)

    R = lam @ lam.T + np.diag(1.0 - communality)
    # The diagonal is 1 by construction; set it so rounding cannot leave a
    # parameter correlated with itself at 0.9999 and fail a caller's check.
    np.fill_diagonal(R, 1.0)
    return R, lam, sorted(unknown)

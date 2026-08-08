"""Recovery from a known ``phi*``: does the fit return the truth it was given?

The synthetic cohorts must be drawn on THEIR OWN ``z``, not the fit's. Sharing it
would cancel ``tau``'s Monte Carlo displacement between the draw and the fit, and
a recovery that passes only because the two clouds are the same patients has not
tested the thing it looks like it tested. Data are cohort draws off a truth cloud
pushed through the simulator, so the emulator's error, the order-statistic
kernel's ``n``-correction and ``E_B`` all sit between ``phi*`` and the rows, which
is where they sit in the real fit.

What comes back tells you which directions the corpus determines and which return
to the prior. A direction whose posterior sd equals its prior sd is reported as
unidentified rather than as a failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["map_estimate", "RecoveryRow", "summarise_recovery", "print_recovery"]


def map_estimate(model: Callable, model_args: Sequence[Any], *,
                 steps: int = 600, lr: float = 5e-2,
                 init: Optional[Mapping[str, jnp.ndarray]] = None,
                 template: Optional[Mapping[str, jnp.ndarray]] = None,
                 seed: int = 0) -> Tuple[Dict[str, jnp.ndarray], float]:
    """``argmax p(phi | T)`` by Adam on the unconstrained parameters.

    Adam and not a quasi-Newton method: most directions are flat, and a
    curvature step along them is fit to rounding. ``template`` fixes the
    parameter shapes, and defaults to a prior draw. Starts at the prior mode,
    so recovery is not seeded with the answer.
    """
    from jax.flatten_util import ravel_pytree
    from numpyro.infer.util import initialize_model

    # initialize_model, not a prior draw: it gives the latent sites alone, and a
    # potential in the unconstrained space with the transform Jacobians in it.
    info = initialize_model(jax.random.PRNGKey(seed), model,
                            model_args=tuple(model_args))
    if template is None:
        template = info.param_info.z
    template = {k: jnp.asarray(v) for k, v in template.items()}
    flat, unravel = ravel_pytree(template)

    neg_lp = info.potential_fn
    v = jnp.zeros(flat.size) if init is None \
        else ravel_pytree({k: jnp.asarray(init[k]) for k in template})[0]

    step = jax.jit(jax.value_and_grad(lambda w: neg_lp(unravel(w))))
    m = s = jnp.zeros_like(v)
    b1, b2, eps = 0.9, 0.999, 1e-8
    f = jnp.inf
    for t in range(1, steps + 1):
        f, g = step(v)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g ** 2
        v = v - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(s / (1 - b2 ** t)) + eps)
    # dynamic_args is off, so potential_fn and postprocess_fn close over the args.
    return info.postprocess_fn(unravel(v)), float(f)


#: Below this, the posterior is narrower than the prior by enough to call the
#: component identified. At 1.0 the corpus said nothing about it.
IDENTIFIED_SHRINK = 0.9


@dataclass(frozen=True)
class RecoveryRow:
    """One component of ``phi``, against the truth that generated the data."""

    block: str
    name: str
    truth: float
    mean: float
    sd: float
    prior_sd: float
    lo: float
    hi: float

    @property
    def z(self) -> float:
        """``(mean - truth)`` in posterior sd. Only meaningful where identified."""
        return (self.mean - self.truth) / self.sd if self.sd > 0 else np.inf

    @property
    def shrink(self) -> float:
        """Posterior sd over prior sd. 1 is the prior back, 0 is a point mass."""
        return self.sd / self.prior_sd if self.prior_sd > 0 else np.nan

    @property
    def identified(self) -> bool:
        return bool(self.shrink < IDENTIFIED_SHRINK)

    @property
    def covered(self) -> bool:
        return bool(self.lo <= self.truth <= self.hi)


def summarise_recovery(truth: Mapping[str, np.ndarray],
                       draws: Mapping[str, np.ndarray],
                       prior_sd: Mapping[str, np.ndarray],
                       *, names: Optional[Mapping[str, Sequence[str]]] = None,
                       level: float = 0.9) -> List[RecoveryRow]:
    """One :class:`RecoveryRow` per component of every block ``truth`` names.

    ``draws`` is the posterior with a leading sample axis. ``prior_sd`` is what
    separates the two ways a component can sit on its truth: a posterior that
    found it, and a posterior that never moved off a prior which happened to be
    centred near it. Only the first is evidence.
    """
    tail = (1.0 - level) / 2.0
    rows: List[RecoveryRow] = []
    for block in truth:
        if block not in draws:
            raise KeyError(f"no posterior draws for {block!r}")
        star = np.atleast_1d(np.asarray(truth[block], dtype=float)).ravel()
        d = np.asarray(draws[block], dtype=float).reshape(
            np.shape(draws[block])[0], -1)
        sd_0 = np.broadcast_to(
            np.atleast_1d(np.asarray(prior_sd[block], dtype=float)).ravel(),
            star.shape)
        if d.shape[1] != star.size:
            raise ValueError(
                f"{block}: truth has {star.size} components, draws have "
                f"{d.shape[1]}")
        label = (list(names[block]) if names and block in names
                 else [f"{block}[{i}]" for i in range(star.size)])
        lo, hi = np.quantile(d, [tail, 1.0 - tail], axis=0)
        rows.extend(
            RecoveryRow(block=block, name=label[i], truth=float(star[i]),
                        mean=float(d[:, i].mean()), sd=float(d[:, i].std(ddof=1)),
                        prior_sd=float(sd_0[i]), lo=float(lo[i]), hi=float(hi[i]))
            for i in range(star.size))
    return rows


def print_recovery(rows: Sequence[RecoveryRow], *, level: float = 0.9,
                   worst: int = 6) -> List[str]:
    """The recovery table as lines to print: per block, then the worst rows.

    Coverage is reported over the identified components and the unidentified ones
    separately. Pooling them hides the failure this is for: an unidentified
    component is covered at ``level`` by construction, so a corpus that determines
    nothing scores perfectly on the pooled number.
    """
    out = [f"recovery against phi*, {level:.0%} intervals",
           f"{'block':<10}{'n':>4}{'ident':>7}{'|z| med':>9}{'|z| max':>9}"
           f"{'cover id':>10}{'cover un':>10}{'shrink med':>12}"]
    for block in dict.fromkeys(r.block for r in rows):
        got = [r for r in rows if r.block == block]
        ident = [r for r in got if r.identified]
        unid = [r for r in got if not r.identified]
        # Blank rather than nan where nothing is identified: |z| against a
        # posterior that is still the prior is not a number worth printing.
        z = np.abs([r.z for r in ident])
        med = f"{np.median(z):.2f}" if ident else "-"
        mx = f"{np.max(z):.2f}" if ident else "-"
        out.append(
            f"{block:<10}{len(got):>4}{len(ident):>7}{med:>9}{mx:>9}"
            f"{_frac(ident):>10}{_frac(unid):>10}"
            f"{np.median([r.shrink for r in got]):>12.2f}")

    bad = sorted((r for r in rows if r.identified and not r.covered),
                 key=lambda r: -abs(r.z))[:worst]
    if not bad:
        out.append("every identified component covers its truth")
        return out
    out.append(f"\nidentified components missing their truth ({len(bad)} shown):")
    out.append(f"  {'name':<28}{'truth':>10}{'mean':>10}{'sd':>9}{'z':>8}"
               f"{'shrink':>9}")
    for r in bad:
        out.append(f"  {r.name[:26]:<28}{r.truth:>10.3f}{r.mean:>10.3f}"
                   f"{r.sd:>9.3f}{r.z:>+8.2f}{r.shrink:>9.2f}")
    return out


def _frac(rows: Sequence[RecoveryRow]) -> str:
    if not rows:
        return "-"
    return f"{sum(r.covered for r in rows) / len(rows):.0%}"

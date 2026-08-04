"""MAP recovery from a known ``phi*``: the wiring test for eq:post.

Data drawn at ``phi*`` are recoverable in principle, because ``z`` is frozen and
shared between the draw and the fit, so ``tau``'s Monte Carlo displacement
cancels. What comes back tells you which directions the corpus determines and
which return to the prior.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["map_estimate", "recovery_table"]


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


def recovery_table(star: Mapping[str, jnp.ndarray],
                   hat: Mapping[str, jnp.ndarray]) -> Dict[str, Tuple[float, float, float]]:
    """Per block, ``(||hat - star||, ||star||, ||hat||)``.

    A block that returned to the prior mode reads ``||hat - star|| ~ ||star||``
    with ``||hat||`` near zero, which is the answer for an unidentified one and
    not a failure.
    """
    out: Dict[str, Tuple[float, float, float]] = {}
    for k in star:
        a = np.asarray(star[k], dtype=float).ravel()
        b = np.asarray(hat[k], dtype=float).ravel()
        out[k] = (float(np.linalg.norm(b - a)), float(np.linalg.norm(a)),
                  float(np.linalg.norm(b)))
    return out

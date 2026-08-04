"""The trained surrogate, as the differentiable ``g_fn`` eq:mech needs.

``train_emulator.py`` fits one net per arm in torch; this reads the checkpoint and
rebuilds the forward pass in JAX, because NUTS needs gradients of the whole
``phi -> cohort quantiles`` map and a torch module cannot supply them here.

The net is Linear/SiLU with dropout, and dropout is identity at evaluation, so
the port is the weight matrices and the two normalisations. Nothing is fitted
here and nothing may be: a discrepancy between this forward pass and the trained
one is silent, so :func:`check_against_torch` exists to close it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["load_arm", "arm_forward", "build_g_fn", "check_against_torch"]


def load_arm(path: str | Path) -> dict:
    """Read one ``emulator_<arm>.pt`` into plain numpy. Torch is import-only."""
    import torch

    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    layers, i = [], 0
    while f"{i}.weight" in sd:
        layers.append((np.asarray(sd[f"{i}.weight"], dtype=np.float64),
                       np.asarray(sd[f"{i}.bias"], dtype=np.float64)))
        # Linear, SiLU, Dropout -> stride 3; the last block has no dropout.
        i += 3 if f"{i + 3}.weight" in sd else 2
    if not layers:
        raise ValueError(f"{path}: no Linear layers found in the state dict")
    return {
        "layers": layers,
        "param_names": list(ck["param_names"]),
        "target_names": list(ck["target_names"]),
        **{k: np.asarray(ck[k], dtype=np.float64)
           for k in ("x_mu", "x_sd", "t_mu", "t_sd", "scale")},
    }


def arm_forward(arm: Mapping, log_theta: jnp.ndarray) -> jnp.ndarray:
    """``(N, P)`` log-parameters -> ``(N, K)`` species, in the trained units.

    Inverts the training transform: standardise, MLP, unstandardise, then
    ``x = scale * sinh(y)``.
    """
    h = (log_theta - jnp.asarray(arm["x_mu"])) / jnp.asarray(arm["x_sd"])
    last = len(arm["layers"]) - 1
    for k, (W, b) in enumerate(arm["layers"]):
        h = h @ jnp.asarray(W).T + jnp.asarray(b)
        if k != last:
            h = h * jax.nn.sigmoid(h)  # SiLU
    t = h * jnp.asarray(arm["t_sd"]) + jnp.asarray(arm["t_mu"])
    return jnp.asarray(arm["scale"]) * jnp.sinh(t)


def build_g_fn(
    arms: Mapping[str, Mapping],
    scenarios: Sequence[tuple[str, float]],
    species: Sequence[str],
) -> Callable[[jnp.ndarray, int], jnp.ndarray]:
    """``g_fn(vartheta, s) -> (N, Q)`` species for scenario ``s``.

    ``scenarios[s]`` is the ``(arm, readout_time)`` that scenario index means, and
    ``species`` fixes the column order the readout bodies expect. One trajectory
    per arm serves every time that arm carries, so several scenarios share a net
    and differ only in which of its columns they read.
    """
    picks = []
    for arm_name, t in scenarios:
        if arm_name not in arms:
            raise KeyError(f"scenario names arm {arm_name!r}, which has no emulator")
        arm = arms[arm_name]
        index = {n: i for i, n in enumerate(arm["target_names"])}
        cols = []
        for sp in species:
            key = f"{sp}@{t:g}"
            if key not in index:
                raise KeyError(
                    f"{arm_name} has no target {key!r}. The emulator was trained on "
                    f"a different (species, time) set than this problem asks for."
                )
            cols.append(index[key])
        picks.append((arm_name, np.asarray(cols, dtype=np.int64)))

    def g_fn(vartheta: jnp.ndarray, s: int) -> jnp.ndarray:
        arm_name, cols = picks[int(s)]
        return arm_forward(arms[arm_name], vartheta)[:, cols]

    return g_fn


def check_against_torch(path: str | Path, n: int = 64, seed: int = 0) -> float:
    """Max relative gap between this forward pass and the torch module it ports.

    Run it once per checkpoint. The failure it catches -- a layer stride read
    wrong, a normalisation applied in the wrong order -- produces plausible
    numbers, so nothing downstream would report it.
    """
    import torch

    arm = load_arm(path)
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    P = arm["x_mu"].shape[0]

    rng = np.random.default_rng(seed)
    x = arm["x_mu"] + arm["x_sd"] * rng.standard_normal((n, P))
    got = np.asarray(arm_forward(arm, jnp.asarray(x)), dtype=np.float64)

    net = _torch_net(ck)
    with torch.no_grad():
        t = net(torch.tensor((x - arm["x_mu"]) / arm["x_sd"], dtype=torch.float32)).numpy()
    want = arm["scale"] * np.sinh(t.astype(np.float64) * arm["t_sd"] + arm["t_mu"])
    return float(np.max(np.abs(got - want) / (np.abs(want) + 1e-30)))


def _torch_net(ck: Mapping):
    import torch.nn as nn

    sd = ck["state_dict"]
    hidden = list(ck["hidden"])
    P = sd["0.weight"].shape[1]
    K = sd[max(k for k in sd if k.endswith(".weight"))].shape[0]
    layers, prev = [], P
    for i, h in enumerate(hidden):
        layers += [nn.Linear(prev, h), nn.SiLU()]
        if i < len(hidden) - 1:
            layers += [nn.Dropout(0.0)]
        prev = h
    layers += [nn.Linear(prev, K)]
    net = nn.Sequential(*layers)
    net.load_state_dict(sd)
    net.eval()
    return net

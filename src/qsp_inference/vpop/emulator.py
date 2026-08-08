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

__all__ = ["load_arm", "arm_forward", "arm_status_logits",
           "arm_status_logprob", "build_g_fn", "build_extra_fn",
           "check_against_torch"]

def load_arm(path: str | Path) -> dict:
    """Read one ``emulator_<arm>.pt`` into plain numpy. Torch is import-only.

    The checkpoint is a shared trunk with two heads: ``species`` and ``status``.
    ``layers`` is trunk + species head, which is what :func:`arm_forward` walks;
    ``status_layers`` is trunk + status head.

    """
    import torch

    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    if "trunk.0.weight" not in sd:
        raise ValueError(
            f"{path}: no 'trunk.0.weight' in the state dict. This reader expects a "
            f"two-head checkpoint (species + status). Retrain with the current "
            f"train_emulator.py; a single-head checkpoint has no status head and "
            f"loading it here would silently drop the eligibility map."
        )

    def _lin(prefix: str):
        return (np.asarray(sd[f"{prefix}.weight"], dtype=np.float64),
                np.asarray(sd[f"{prefix}.bias"], dtype=np.float64))

    def _trunk(prefix: str):
        out, i = [], 0
        while f"{prefix}{i}.weight" in sd:
            out.append(_lin(f"{prefix}{i}"))
            # Linear, SiLU, Dropout -> stride 3; the last block has no dropout.
            i += 3 if f"{prefix}{i + 3}.weight" in sd else 2
        return out

    trunk = _trunk("trunk.")
    for head in ("species", "status"):
        if f"{head}.weight" not in sd:
            raise ValueError(f"{path}: state dict has no {head!r} head")
    # ``heads="separate"`` stores a second trunk for the status net. Absent it the
    # trunk is shared and both lists alias the same arrays, which is safe because
    # a forward pass only reads them.
    status_trunk = _trunk("status_trunk.") or trunk
    arm = {
        "layers": trunk + [_lin("species")],
        "status_layers": status_trunk + [_lin("status")],
        "heads": ck.get("heads", "shared"),
        "param_names": list(ck["param_names"]),
        "target_names": list(ck["target_names"]),
        "transform": ck.get("transform", "asinh"),
        "status_codes": list(ck.get("status_codes", [])),
        "status_labels": list(ck.get("status_labels", [])),
        **{k: np.asarray(ck[k], dtype=np.float64)
           for k in ("x_mu", "x_sd", "t_mu", "t_sd", "scale")},
    }
    return arm


def _mlp(layers, h: jnp.ndarray) -> jnp.ndarray:
    last = len(layers) - 1
    for k, (W, b) in enumerate(layers):
        h = h @ jnp.asarray(W).T + jnp.asarray(b)
        if k != last:
            h = h * jax.nn.sigmoid(h)  # SiLU
    return h


def arm_status_logits(arm: Mapping, log_theta: jnp.ndarray) -> jnp.ndarray:
    """``(N, P)`` log-parameters -> ``(N, C)`` status-class logits.

    The classes are ``arm["status_codes"]`` in order. No normalisation is undone
    here: the head's outputs are logits, not a transformed physical quantity.
    """
    h = (log_theta - jnp.asarray(arm["x_mu"])) / jnp.asarray(arm["x_sd"])
    return _mlp(arm["status_layers"], h)


def arm_status_logprob(arm: Mapping, log_theta: jnp.ndarray,
                       code: int = 0) -> jnp.ndarray:
    """``(N,)`` log-probability that theta lands in status ``code``.

    ``code=0`` is admissible, so ``exp`` of this is the smooth stand-in for the
    simulator's hard screen. Log rather than probability because it is a term in
    a log density and taking the log afterwards loses the tail.
    """
    codes = list(arm["status_codes"])
    if code not in codes:
        raise KeyError(
            f"status code {code} absent from this arm; it carries {codes}. No row "
            f"in the training set had that outcome, so the head cannot score it."
        )
    return jax.nn.log_softmax(arm_status_logits(arm, log_theta), axis=-1)[
        :, codes.index(code)]


def arm_forward(arm: Mapping, log_theta: jnp.ndarray) -> jnp.ndarray:
    """``(N, P)`` log-parameters -> ``(N, K)`` species, in the trained units.

    Inverts the training transform: standardise, MLP, unstandardise, then undo
    whichever transform the checkpoint records. ``log``/``exp`` is positive by
    construction, which is what ``h_r`` needs since it takes logs; ``asinh`` is
    read only for checkpoints trained before that was fixed, and ``sinh`` is
    unbounded below, so those emit negative cell counts.

    """
    h = (log_theta - jnp.asarray(arm["x_mu"])) / jnp.asarray(arm["x_sd"])
    t = _mlp(arm["layers"], h) * jnp.asarray(arm["t_sd"]) + jnp.asarray(arm["t_mu"])
    # Defaulting is not safe here: reading a checkpoint under the wrong inverse
    # returns plausible numbers and nothing downstream notices, so an unlabelled
    # one is assumed to predate the change rather than to match the current code.
    transform = arm.get("transform", "asinh")
    if transform == "log":
        return jnp.exp(t)
    if transform == "asinh":
        return jnp.asarray(arm["scale"]) * jnp.sinh(t)
    raise ValueError(f"unknown emulator transform {transform!r}")


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


def build_extra_fn(
    arms: Mapping[str, Mapping],
    scenarios: Sequence[tuple[str, float]],
    extra_at: Sequence[tuple[int, str]],
) -> Callable[[jnp.ndarray, int, str], jnp.ndarray]:
    """``extra_fn(vartheta, s, name) -> (N,)`` for a readout the reduce composed.

    Kept out of ``g_fn``'s block deliberately. A precomposed readout is not a
    species and not every arm emits one, so padding the block would either cost
    the arms that lack it their training rows or leave a column nothing may read.
    Here each ``(scenario, name)`` resolves against the arm that actually serves
    it, and one that does not is refused at build rather than at the first
    gradient.
    """
    picks = {}
    for s, name in extra_at:
        arm_name, t = scenarios[int(s)]
        if arm_name not in arms:
            raise KeyError(f"scenario names arm {arm_name!r}, which has no emulator")
        arm = arms[arm_name]
        index = {n: i for i, n in enumerate(arm["target_names"])}
        key = f"{name}@{t:g}"
        if key not in index:
            raise KeyError(
                f"{arm_name} emits no {key!r}, so it cannot serve a precomposed "
                f"readout. The reduce emits it only for the arms that carry one."
            )
        picks[(int(s), name)] = (arm_name, index[key])

    def extra_fn(vartheta: jnp.ndarray, s: int, name: str) -> jnp.ndarray:
        arm_name, col = picks[(int(s), name)]
        return arm_forward(arms[arm_name], vartheta)[:, col]

    return extra_fn


def check_against_torch(path: str | Path, n: int = 64,
                        seed: int = 0) -> tuple[float, float]:
    """``(species, status)`` max relative gap against the torch module ported here.

    Run it once per checkpoint. The failure it catches -- a layer stride read
    wrong, a normalisation applied in the wrong order -- produces plausible
    numbers, so nothing downstream would report it. Both heads are checked: they
    share a trunk, so a trunk bug shows in both, but a head wired to the wrong
    output block shows in only one.

    """
    import torch

    arm = load_arm(path)
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    P = arm["x_mu"].shape[0]

    rng = np.random.default_rng(seed)
    x = arm["x_mu"] + arm["x_sd"] * rng.standard_normal((n, P))
    xs = torch.tensor((x - arm["x_mu"]) / arm["x_sd"], dtype=torch.float32)

    got = np.asarray(arm_forward(arm, jnp.asarray(x)), dtype=np.float64)
    got_status = np.asarray(arm_status_logits(arm, jnp.asarray(x)), dtype=np.float64)

    net = _torch_net(ck)
    with torch.no_grad():
        t, s = net(xs)
    tt = t.numpy().astype(np.float64) * arm["t_sd"] + arm["t_mu"]
    want = (np.exp(tt) if arm.get("transform", "asinh") == "log"
            else arm["scale"] * np.sinh(tt))
    want_status = s.numpy().astype(np.float64)

    rel = lambda a, b: float(np.max(np.abs(a - b) / (np.abs(b) + 1e-30)))  # noqa: E731
    return rel(got, want), rel(got_status, want_status)


def _torch_net(ck: Mapping):
    import torch
    import torch.nn as nn

    sd = ck["state_dict"]
    hidden = list(ck["hidden"])
    P = sd["trunk.0.weight"].shape[1]
    K = sd["species.weight"].shape[0]
    C = sd["status.weight"].shape[0]
    separate = "status_trunk.0.weight" in sd

    def _stack():
        layers, prev = [], P
        for i, h in enumerate(hidden):
            layers += [nn.Linear(prev, h), nn.SiLU()]
            if i < len(hidden) - 1:
                # Dropout at p=0: identity here, and it keeps the module indices
                # aligned with the trained state dict's keys.
                layers += [nn.Dropout(0.0)]
            prev = h
        return nn.Sequential(*layers), prev

    class Emulator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk, prev = _stack()
            self.species = nn.Linear(prev, K)
            self.status = nn.Linear(prev, C)
            if separate:
                self.status_trunk, _ = _stack()

        def forward(self, x):
            return (self.species(self.trunk(x)),
                    self.status((self.status_trunk if separate else self.trunk)(x)))

    net = Emulator()
    net.load_state_dict(sd)
    net.eval()
    return net

"""The JAX port of the trained surrogate matches the torch module, both heads."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

from qsp_inference.vpop.emulator import (  # noqa: E402
    arm_status_logits,
    arm_status_logprob,
    check_against_torch,
    load_arm,
)

HIDDEN = (8, 8, 4)
P, K, C = 5, 3, 4
CODES = [0, 1, 4, 5]


def _checkpoint(tmp_path, seed=0, codes=CODES):
    """A checkpoint shaped exactly as train_emulator writes one."""
    import torch.nn as nn

    torch.manual_seed(seed)

    class Emulator(nn.Module):
        def __init__(self):
            super().__init__()
            layers, prev = [], P
            for i, h in enumerate(HIDDEN):
                layers += [nn.Linear(prev, h), nn.SiLU()]
                if i < len(HIDDEN) - 1:
                    layers += [nn.Dropout(0.05)]
                prev = h
            self.trunk = nn.Sequential(*layers)
            self.species = nn.Linear(prev, K)
            self.status = nn.Linear(prev, len(codes))

        def forward(self, x):
            z = self.trunk(x)
            return self.species(z), self.status(z)

    rng = np.random.default_rng(seed)
    path = tmp_path / "emulator_arm.pt"
    torch.save(
        {
            "state_dict": Emulator().state_dict(),
            "hidden": list(HIDDEN),
            "param_names": [f"p{i}" for i in range(P)],
            "target_names": [f"sp:S{i}@0" for i in range(K)],
            "transform": "log",
            "status_codes": list(codes),
            "status_labels": [f"class_{c}" for c in codes],
            "x_mu": rng.standard_normal(P),
            "x_sd": np.abs(rng.standard_normal(P)) + 0.5,
            "t_mu": rng.standard_normal(K),
            "t_sd": np.abs(rng.standard_normal(K)) + 0.5,
            "scale": np.abs(rng.standard_normal(K)) + 1.0,
        },
        path,
    )
    return path


def test_both_heads_port_exactly(tmp_path):
    """A trunk bug shows in both heads; a miswired head shows in only one."""
    species_gap, status_gap = check_against_torch(_checkpoint(tmp_path), n=32)
    assert species_gap < 1e-5
    assert status_gap < 1e-5


def test_trunk_is_shared_between_the_heads(tmp_path):
    arm = load_arm(_checkpoint(tmp_path))
    # Everything but the final head layer is the same object, not a copy.
    for (Wa, ba), (Wb, bb) in zip(arm["layers"][:-1], arm["status_layers"][:-1]):
        assert Wa is Wb and ba is bb
    assert arm["layers"][-1][0].shape[0] == K
    assert arm["status_layers"][-1][0].shape[0] == C


def test_status_logprob_is_the_log_softmax_column(tmp_path):
    arm = load_arm(_checkpoint(tmp_path))
    x = jnp.asarray(np.random.default_rng(1).standard_normal((16, P)))
    logits = np.asarray(arm_status_logits(arm, x))
    want = logits - np.log(np.exp(logits).sum(1, keepdims=True))
    for code in CODES:
        got = np.asarray(arm_status_logprob(arm, x, code=code))
        np.testing.assert_allclose(got, want[:, CODES.index(code)], atol=1e-5)


def test_status_logprob_normalises_across_the_classes(tmp_path):
    arm = load_arm(_checkpoint(tmp_path))
    x = jnp.asarray(np.random.default_rng(2).standard_normal((8, P)))
    total = sum(np.exp(np.asarray(arm_status_logprob(arm, x, code=c))) for c in CODES)
    np.testing.assert_allclose(total, np.ones(8), atol=1e-5)


def test_absent_status_code_raises(tmp_path):
    """An arm where nothing was rejected cannot score a rejection.

    Returning -inf would be worse than refusing: it reads as "certainly not
    rejected" when the truth is that the head was never shown one.
    """
    arm = load_arm(_checkpoint(tmp_path, codes=[0, 1]))
    x = jnp.zeros((2, P))
    with pytest.raises(KeyError, match="absent from this arm"):
        arm_status_logprob(arm, x, code=5)


def test_single_head_checkpoint_is_refused(tmp_path):
    """Silently loading a pre-two-head checkpoint would drop the status map."""
    import torch.nn as nn

    net = nn.Sequential(nn.Linear(P, 8), nn.SiLU(), nn.Linear(8, K))
    path = tmp_path / "old.pt"
    torch.save(
        {"state_dict": net.state_dict(), "hidden": [8],
         "param_names": [f"p{i}" for i in range(P)],
         "target_names": [f"sp:S{i}@0" for i in range(K)],
         "transform": "log",
         **{k: np.ones(P if k.startswith("x") else K)
            for k in ("x_mu", "x_sd", "t_mu", "t_sd", "scale")}},
        path,
    )
    with pytest.raises(ValueError, match="two-head checkpoint"):
        load_arm(path)

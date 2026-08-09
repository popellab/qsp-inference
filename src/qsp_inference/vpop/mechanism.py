"""The corpus and the surrogate, assembled into a ``Mechanism``. eq:mech, eq:readout.

Four things that all answer "what does the forward map consist of", and that a
run needs before any ``phi`` exists:

* the emulator checkpoints, as pure-jax ``g_fn`` closures over the arms;
* ``h_r``, compiled from each target's own observable body;
* ``Z``, the readout attributes eq:disc's measurement map is linear in;
* the scenario table and ``L_R``, whose provenance is the thing that has to
  agree with the pool the surrogate trained on.

``predict.Mechanism`` is what they are handed to.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    # the surrogate
    "load_arm", "arm_forward", "arm_status_logits", "arm_status_logprob",
    "build_g_fn", "build_extra_fn", "check_against_torch",
    # h_r
    "SHIM_NAMES", "NEEDS_TRAJECTORY", "shim_module", "compile_observable",
    "target_constants", "declared_reference", "build_h_fn",
    "UntraceableReadout", "NotPerPatient",
    # Z
    "ZDesign", "build_Z", "z_conditioning",
    # what the surrogate has to agree with
    "ScenarioTable", "scenario_table", "prior_cholesky",
]


# ---------------------------------------------------------------- the surrogate

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


#: How far above its own training range the surrogate may assert a species, in
#: standard deviations of that species' log over the training set.
#:
#: This bounds the SURROGATE, not the population. It says what the net is allowed
#: to claim, not which patients exist, so it moves no draw of eq:crn and changes
#: nothing for a patient the net was fitted anywhere near.
#:
#: Unbounded, an extrapolating net returns species that are individually finite
#: and jointly absurd, and the damage lands two steps later: the derived sums in
#: the observables overflow, ``inf - inf`` gives NaN, and a whole block's rows
#: lose their gradient while their VALUES stay finite and plausible. On the pdac
#: corpus one tail patient drew ``V_T.CD8_TLA`` at 1e188, which is 138 sd above
#: its training mean, and the five readouts sharing that denominator all read
#: exactly 1.0 with a NaN tangent.
#:
#: 20 is far outside anything trained on and still caps the species at about
#: 3e44, which no product downstream can overflow. The real remedies are
#: eq:elig, which would decline such a patient, and a pool that supports the
#: cloud; this only keeps the Jacobian finite until they land.
SPECIES_CEIL_SD = 20.0


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
        # Continued by its tangent above the ceiling, not clipped. A clip has a
        # zero derivative there, and a zero derivative under the ratios h_r takes
        # is the 0 * inf that puts NaN in a Jacobian while leaving the forward
        # pass finite. Same construction, and same reason, as _logit's upper
        # bound in vpop.rows.
        ceil = (jnp.asarray(arm["t_mu"])
                + SPECIES_CEIL_SD * jnp.asarray(arm["t_sd"]))
        safe = jnp.minimum(t, ceil)
        return jnp.exp(safe) * (1.0 + (t - safe))
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

# ------------------------------------------------------------------------ h_r

#: Readouts that are functionals of the whole trajectory, not of the species at a
#: readout time. They need a time vector and branch on the data, so they cannot
#: run under tracing and are refused rather than silently mis-evaluated.
NEEDS_TRAJECTORY = frozenset({"tumor_doubling_time_PDAC_deriv001"})

#: Every numpy name the 54 observable bodies use.
SHIM_NAMES = ("any", "asarray", "divide", "full_like", "log", "nan", "where",
              "zeros_like")


class UntraceableReadout(ValueError):
    """Observable code that cannot run under tracing."""


class NotPerPatient(ValueError):
    """Observable code whose value for one patient reads another's."""


def _divide(a, b, out=None, where=None, **_):
    """``np.divide``, including the ``out=``/``where=`` form the corpus uses.

    Substitutes the denominator before dividing. ``jnp.where`` evaluates both
    branches, so a zero the mask discards would still reach the reverse pass.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if where is None:
        return a / b
    fill = jnp.zeros_like(a) if out is None else jnp.asarray(out)
    return jnp.where(where, a / jnp.where(where, b, 1.0), fill)


def shim_module() -> types.SimpleNamespace:
    """A stand-in for ``numpy`` backed by ``jax.numpy``."""
    return types.SimpleNamespace(
        any=jnp.any,
        asarray=lambda x, dtype=None: jnp.asarray(x, dtype=dtype),
        divide=_divide,
        full_like=jnp.full_like,
        log=jnp.log,
        nan=jnp.nan,
        where=jnp.where,
        zeros_like=jnp.zeros_like,
    )


def _exec_globals() -> Dict[str, Any]:
    """Globals in which ``import numpy as np`` binds the shim."""
    shim = shim_module()
    builtins = dict(__builtins__) if isinstance(__builtins__, dict) \
        else dict(vars(__builtins__))
    real_import = builtins["__import__"]

    def _import(name, *args, **kwargs):
        return shim if name == "numpy" else real_import(name, *args, **kwargs)

    builtins["__import__"] = _import
    return {"__builtins__": builtins, "np": shim, "jnp": jnp}


def compile_observable(code: str, *, label: str = "<observable>") -> Callable:
    """Exec one target's ``code`` and return its ``compute_observable``."""
    namespace = _exec_globals()
    try:
        exec(compile(code, label, "exec"), namespace)
    except SyntaxError as exc:
        raise UntraceableReadout(f"{label}: {exc}") from exc
    fn = namespace.get("compute_observable")
    if fn is None:
        raise UntraceableReadout(f"{label}: defines no compute_observable")
    return fn


def target_constants(target: Mapping[str, Any],
                     aux: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``constants`` mapping one body expects.

    ``observable.constants`` are fixed and carry their value; ``auxiliary
    _parameters`` are inferred, so their value comes from ``aux``.
    """
    observable = target["observable"]
    out = {c["name"]: c["value"] for c in (observable.get("constants") or [])}
    for p in (observable.get("auxiliary_parameters") or []):
        name = p["name"]
        if name not in aux:
            raise UntraceableReadout(f"no value supplied for auxiliary {name!r}")
        out[name] = aux[name]
    return out


def declared_reference(target: Mapping[str, Any]) -> Optional[float]:
    """The timepoint a readout is taken against, or ``None`` if it stands alone.

    ``observable.readout.reference`` is the corpus's own declaration. Only
    ``kind: timepoint`` resolves to a scenario, so any other kind is refused
    rather than dropped.
    """
    readout = (target.get("observable") or {}).get("readout") or {}
    ref = readout.get("reference")
    if not ref:
        return None
    if ref.get("kind") != "timepoint":
        raise UntraceableReadout(
            f"reference kind {ref.get('kind')!r} does not resolve to a scenario"
        )
    return float(ref["timepoint"])


def _check_per_patient(h_fn, readouts, n_scenarios, n_species, n_aux,
                       n_patients: int = 8, needs_extra=()) -> None:
    """Refuse readouts whose value for one patient reads another's.

    eq:readout is a per-patient map, so perturbing patient ``j`` may move column
    ``j`` and nothing else. Indexing the patient axis and reducing over it both
    break that, and both otherwise return an array of the right shape holding the
    wrong numbers. Checked once at build, on a probe, not in the gradient.
    """
    rng = np.random.default_rng(0)
    shape = (n_scenarios, n_patients, n_species)
    y = jnp.asarray(rng.uniform(0.5, 2.0, shape))
    # Per species and per scenario, so a ratio and a fold change both move.
    bump = jnp.asarray(rng.uniform(1.2, 1.8, (n_scenarios, n_species)))
    log_R = jnp.zeros(n_aux) if n_aux else None
    # The precomposed channel is probed too: it is per-patient like everything
    # else, and a readout reading it must still move with its own patient.
    ex = {k: jnp.asarray(rng.uniform(0.5, 2.0, n_patients)) for k in needs_extra}
    bump_ex = rng.uniform(1.2, 1.8, len(needs_extra))

    base = np.asarray(h_fn(y, log_R, ex or None))
    leaks, deaf = set(), set(range(len(readouts)))
    for j in range(n_patients):
        # Bitwise, not within a tolerance. A per-patient map recomputes the other
        # columns from unchanged inputs, so they come back identical; a tolerance
        # would instead hide any readout whose scale the probe does not match.
        ex_j = {k: v.at[j].multiply(bump_ex[i]) for i, (k, v) in enumerate(ex.items())}
        got = np.asarray(h_fn(y.at[:, j, :].multiply(bump), log_R, ex_j or None))
        moved = (got != base) & ~(np.isnan(got) & np.isnan(base))
        own = tuple(range(moved.ndim - 2))       # patient axis already indexed out
        rest = tuple(range(moved.ndim - 1))
        deaf -= set(np.flatnonzero(moved[..., j, :].any(axis=own)).tolist())
        leaks |= set(np.flatnonzero(
            np.delete(moved, j, axis=-2).any(axis=rest)).tolist())

    if leaks:
        raise NotPerPatient(
            f"{', '.join(sorted(readouts[k] for k in leaks))} read patients other "
            f"than the one they report on. A body indexes its species as [i] to "
            f"reach the declared reference, never to reach a patient."
        )
    if deaf:
        raise NotPerPatient(
            f"{', '.join(sorted(readouts[k] for k in deaf))} do not vary with the "
            f"patient they report on. A fold change that declares no reference "
            f"looks like this: it divides by itself and comes back constant."
        )


def build_h_fn(
    targets: Mapping[str, Mapping[str, Any]],
    readouts: Sequence[str],
    states: Sequence[str],
    observables_fn: Callable,
    aux_order: Sequence[str] = (),
    scenario_of: Optional[Mapping[str, Sequence[int]]] = None,
    precomposed: Optional[Mapping[str, str]] = None,
) -> Callable:
    """``h_r`` for every readout at once: ``(S,N,Q), (A,) -> (N,M)`` on the log scale.

    eq:readout gives one value per patient per readout, not one per scenario: a
    target belongs to one cohort, and a cohort to one scenario. ``scenario_of``
    gives each readout the scenarios it needs, its declared reference first and
    its own readout time last, so each body runs once on ``(T,N)`` instead of
    once per scenario with the rest discarded. Without it every scenario is
    returned and readouts declaring a reference are refused, since there is then
    no way to say which slice that reference is.

    ``observables_fn`` is qsp-codegen's generated module, which turns raw species
    into the derived symbols the bodies name. ``aux_order`` fixes which entry of
    ``log R`` belongs to which auxiliary parameter.
    """
    given_pre = dict(precomposed or {})
    unknown = sorted(set(given_pre) - set(readouts))
    if unknown:
        raise UntraceableReadout(
            f"precomposed names {unknown} that are not among the readouts"
        )
    clash = sorted(n for n in given_pre.values() if n in states)
    if clash:
        raise UntraceableReadout(
            f"{clash} are precomposed readouts and also states. They travel in a "
            f"separate channel precisely because they are not species, so a name "
            f"in both is one of them shadowing the other."
        )

    # A trajectory functional is refused unless the surrogate emits it already.
    # tumor_doubling_time is the case: it reads each patient's first and last
    # timepoint, and the last is patient-dependent, so no fixed readout time
    # expresses it and the reduce computes it where the trajectory still exists.
    refused = sorted(set(readouts) & NEEDS_TRAJECTORY - set(given_pre))
    if refused:
        raise UntraceableReadout(
            f"{', '.join(refused)} read the whole trajectory, which h_r is not "
            f"given; exclude them, precompose them, or supply a separate evaluator"
        )

    given = {r: tuple(v) for r, v in (scenario_of or {}).items()}
    at = {r: given[r] for r in readouts if r in given}
    if at and len(at) != len(readouts):
        raise UntraceableReadout(
            f"scenario_of must cover every readout or none; it omits "
            f"{', '.join(sorted(set(readouts) - set(at)))}"
        )
    for r in readouts:
        want = 1 + (r not in given_pre and declared_reference(targets[r]) is not None)
        if not at:
            if want == 2:
                raise UntraceableReadout(
                    f"{r} is taken against a declared reference, so it needs the "
                    f"(reference, readout) scenario indices in scenario_of"
                )
        elif len(at[r]) != want:
            raise UntraceableReadout(
                f"{r}: scenario_of gives {len(at[r])} scenarios, expected {want} "
                f"(reference first, readout time last)"
            )

    compiled = {r: compile_observable(targets[r]["observable"]["code"], label=r)
                for r in readouts if r not in given_pre}
    wanted = {r: tuple(targets[r]["observable"]["species"])
              for r in readouts if r not in given_pre}

    times = {r: float(targets[r]["observable"].get("readout_time") or 0.0)
             for r in readouts}

    live = sorted({int(i) for s in at.values() for i in s})

    # Which (scenario, name) the caller has to supply. Declared so the caller
    # evaluates exactly these and no scenario is asked for a column its arm does
    # not emit.
    needs_extra = tuple(sorted(
        ((at[r][-1] if r in at else 0), name) for r, name in given_pre.items()))

    def h_fn(y, log_R=None, extra=None):
        y = jnp.asarray(y)
        aux = {} if log_R is None else {
            name: jnp.exp(jnp.asarray(log_R)[i])
            for i, name in enumerate(aux_order)
        }
        # Per scenario, on (N, Q). Reversing a gather into the (S, N, Q) stack
        # pads a cotangent back to full width once per readout, so evaluating on
        # the slice is what keeps the backward pass at the width it needs.
        def _at(ys):
            return observables_fn({n: ys[..., i] for i, n in enumerate(states)})

        derived = ({s: _at(y[s]) for s in live} if at else {None: _at(y)})

        columns = []
        for r in readouts:
            s = at.get(r)
            if r in given_pre:
                # Already the readout, so nothing to compose. It arrives outside
                # the species block and therefore untouched by beta, which is
                # right for a log-ratio of one species at two times: a per-species
                # bias cancels identically and beta could never have reached it.
                key = ((s[-1] if s is not None else 0), given_pre[r])
                if extra is None or key not in extra:
                    raise UntraceableReadout(
                        f"{r} is precomposed but no value was supplied for {key}"
                    )
                columns.append(jnp.log(extra[key]))
                continue
            if s is None:
                values = {sym: derived[None][sym] for sym in wanted[r]}
            else:
                # A requested symbol can be a bare model constant, which carries
                # no patient axis and so stands for every scenario at once.
                values = {sym: (derived[s[0]][sym] if jnp.ndim(derived[s[0]][sym]) < 1
                                else jnp.stack([derived[i][sym] for i in s]))
                          for sym in wanted[r]}
            out = compiled[r](jnp.asarray(times[r]), values,
                              target_constants(targets[r], aux))
            columns.append(jnp.log(out if s is None else out[-1]))
        return jnp.stack(columns, axis=-1)

    n_scenarios = 1 + max((int(i) for s in at.values() for i in s), default=1)
    _check_per_patient(h_fn, tuple(readouts), n_scenarios,
                       len(states), len(aux_order), needs_extra=needs_extra)
    h_fn.needs_extra = needs_extra
    return h_fn

# -------------------------------------------------------------------------- Z

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

# ------------------------------------------ the scenarios, and L_R's provenance


@dataclass(frozen=True)
class ScenarioTable:
    """Which ``(arm, time)`` each scenario index means, and what each readout reads.

    ``scenario_of[r]`` is ``(readout,)`` or ``(reference, readout)``, reference
    first, which is the order ``build_h_fn`` indexes: a body reaches its declared
    reference at ``x[0]`` and its own time last.
    """

    scenarios: Tuple[Tuple[str, float], ...]
    scenario_of: Dict[str, Tuple[int, ...]]

    @property
    def n_scenarios(self) -> int:
        return len(self.scenarios)

    def arms(self) -> Tuple[str, ...]:
        seen: list = []
        for a, _ in self.scenarios:
            if a not in seen:
                seen.append(a)
        return tuple(seen)


def scenario_table(targets: Mapping[str, Mapping[str, Any]],
                   arm_of: Mapping[str, str],
                   readouts: Sequence[str]) -> ScenarioTable:
    """The ``(arm, time)`` scenarios ``readouts`` need, and each one's indices.

    Keyed by ``(arm, time)`` rather than by cohort: cohorts sharing an arm read
    the same trajectory, and what separates them is eq:disc and the weighting,
    not the mechanism. On the pdac corpus that is 7 scenarios rather than 30,
    which is what the campaign is sized on.
    """
    want: Dict[str, Tuple[Tuple[str, float], ...]] = {}
    for r in readouts:
        if r not in arm_of:
            raise KeyError(f"{r} has no arm; arm_of must cover every readout")
        arm = arm_of[r]
        t = float((targets[r].get("observable") or {}).get("readout_time") or 0.0)
        ref = declared_reference(targets[r])
        want[r] = (() if ref is None else ((arm, float(ref)),)) + ((arm, t),)

    keys = sorted({k for v in want.values() for k in v})
    index = {k: i for i, k in enumerate(keys)}
    return ScenarioTable(tuple(keys), {r: tuple(index[k] for k in v)
                                       for r, v in want.items()})


def prior_cholesky(prior_spec, param_names: Optional[Sequence[str]] = None,
                   *, pair=None) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """``L_R`` from the same ``PriorSpec`` the theta pool was drawn from.

    Derived rather than passed alongside. ``L_R`` sets the geometry of the patient
    cloud in eq:crn, and the surrogate was trained on draws from this correlation,
    so a ``Mechanism`` carrying a different one evaluates the surrogate off the
    manifold it learned -- with no symptom, because every number stays plausible.
    Taking both from one object removes the chance to disagree.

    ``pair`` is an already-built prior pair. Building one reads three files and
    fits every marginal, and a driver needs the same object for ``mu_0`` and
    ``sd_1`` as well, so it is worth handing over rather than rebuilding.
    """
    from qsp_inference.priors.inference_prior import build_prior_pair

    pair = build_prior_pair(prior_spec, verbose=False) if pair is None else pair
    names = tuple(pair.param_names)
    R = getattr(pair.prior, "_R", None)
    if R is None:
        raise TypeError(
            "the prior carries no correlation matrix, so there is no L_R to take "
            "from it. eq:crn needs the composite copula prior; build the spec "
            "with a submodel_priors_yaml."
        )
    if param_names is not None and tuple(param_names) != names:
        raise ValueError(
            "param order differs from the prior's. L_R is indexed by parameter, "
            "so a reordering silently permutes the patient cloud."
        )
    return np.linalg.cholesky(np.asarray(R, dtype=float)), names

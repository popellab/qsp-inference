"""``h_r``: the corpus's own observable code, run on JAX arrays. eq:readout.

Each target carries an executable ``compute_observable(time, species_dict,
constants)``. Those bodies are shimmed onto ``jax.numpy`` at exec time rather
than rewritten here, so the corpus stays the one definition of what a readout is
and the trajectory evaluator keeps running the same code.
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import jax.numpy as jnp

__all__ = ["SHIM_NAMES", "NEEDS_TRAJECTORY", "shim_module", "compile_observable",
           "target_constants", "build_h_fn", "UntraceableReadout"]

#: Readouts that are functionals of the whole trajectory, not of the species at a
#: readout time. They need a time vector and branch on the data, so they cannot
#: run under tracing and are refused rather than silently mis-evaluated.
NEEDS_TRAJECTORY = frozenset({"tumor_doubling_time_PDAC_deriv001"})

#: Every numpy name the 54 observable bodies use.
SHIM_NAMES = ("any", "asarray", "divide", "full_like", "log", "nan", "where",
              "zeros_like")


class UntraceableReadout(ValueError):
    """Observable code that cannot run under tracing."""


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


def build_h_fn(
    targets: Mapping[str, Mapping[str, Any]],
    readouts: Sequence[str],
    states: Sequence[str],
    observables_fn: Callable,
    aux_order: Sequence[str] = (),
) -> Callable:
    """``h_r`` for every readout at once: ``(S,N,Q), (A,) -> (S,N,M)`` on the log scale.

    ``observables_fn`` is qsp-codegen's generated module, which turns raw species
    into the derived symbols the bodies name. ``aux_order`` fixes which entry of
    ``log R`` belongs to which auxiliary parameter.
    """
    refused = sorted(set(readouts) & NEEDS_TRAJECTORY)
    if refused:
        raise UntraceableReadout(
            f"{', '.join(refused)} read the whole trajectory, which h_r is not "
            f"given; exclude them or supply a separate evaluator"
        )

    compiled = {r: compile_observable(targets[r]["observable"]["code"], label=r)
                for r in readouts}
    wanted = {r: tuple(targets[r]["observable"]["species"]) for r in readouts}
    times = {r: float(targets[r]["observable"].get("readout_time") or 0.0)
             for r in readouts}

    def h_fn(y, log_R=None):
        species = {name: y[..., i] for i, name in enumerate(states)}
        derived = observables_fn(species)
        aux = {} if log_R is None else {
            name: jnp.exp(jnp.asarray(log_R)[i])
            for i, name in enumerate(aux_order)
        }
        columns = []
        for r in readouts:
            value = compiled[r](
                jnp.asarray(times[r]),
                {s: derived[s] for s in wanted[r]},
                target_constants(targets[r], aux),
            )
            columns.append(jnp.log(value))
        return jnp.stack(columns, axis=-1)

    return h_fn

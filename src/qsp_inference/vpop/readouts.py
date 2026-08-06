"""``h_r``: the corpus's own observable code, run on JAX arrays. eq:readout.

Each target carries an executable ``compute_observable(time, species_dict,
constants)``. Those bodies are shimmed onto ``jax.numpy`` at exec time rather
than rewritten here, so the corpus stays the one definition of what a readout is
and the trajectory evaluator keeps running the same code.

A body receives its species as ``(T, N)``: axis 0 is the readout's declared
reference followed by its own readout time, axis 1 is the patient. So ``x[0]``
is the reference in every caller, which is what the trajectory evaluator already
means by it, and ``T = 1`` where nothing is declared. Everything else must act
patient by patient, and ``build_h_fn`` checks that rather than assuming it.
"""

from __future__ import annotations

import types
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

__all__ = ["SHIM_NAMES", "NEEDS_TRAJECTORY", "shim_module", "compile_observable",
           "target_constants", "declared_reference", "build_h_fn",
           "UntraceableReadout", "NotPerPatient"]

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

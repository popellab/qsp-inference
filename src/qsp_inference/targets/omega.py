"""Layered population-spread (omega) prior -- the prior half of the maple
target-data contract.

omega is BETWEEN-PATIENT variability. It must not be read off a flat prior's
marginal width (that is epistemic uncertainty about a population CENTER, the
SEM-vs-SD conflation one level up), so it gets its own layered source (strongest
last):

  1. global default   -- ``global_default``, applied to anything more specific
  2. per-class        -- a ``role`` in the overrides CSV (e.g. packing limits /
                         material constants are tight: measurement-dominated,
                         not diverse)
  3. explicit         -- an ``omega`` value in that CSV, for params with a belief
  4. data-derived     -- a submodel ``population`` block, SHRUNK toward levels
                         1-3 by its own sample size (MBMA), not hard-substituted

Level 4 is a shrinkage combine on purpose: a sample SD from n=6 donors is itself
~30% uncertain, so it should PULL omega, not PIN it. ``sd(log s) ~ 1/sqrt(2(n-1))``
gives the data precision ``2(n-1)`` against a class-prior precision ``1/tau^2`` --
omega equals the prior when n is tiny and moves to the data as n grows.

Generic over the role map / defaults / CSV / target dirs: a project injects those;
the layering algorithm lives here. See notes on unit-panel spread design.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

__all__ = [
    "OmegaEntry",
    "shrink_toward_prior",
    "build_omega_center",
    "load_omega_overrides",
    "load_population_n",
    "write_provenance",
]


@dataclass(frozen=True)
class OmegaEntry:
    """One parameter's omega, and where it came from."""

    name: str
    omega: float
    level: str  # global_default | role | explicit | data_shrunk
    role: str = "default"
    prior_omega: float | None = None  # levels 1-3 value, before any data shrinkage
    data_omega: float | None = None  # raw population-block sigma, pre-shrinkage
    n_biological: int | None = None


def load_omega_overrides(
    path: Path | str, *, role_omega: Mapping[str, float]
) -> dict[str, tuple[str, float | None]]:
    """Read the role / explicit-omega overrides CSV. Missing file = no overrides.

    ``role_omega`` is the project's known-role map; a row naming a role outside it
    is an error (a typo would otherwise silently fall through to the default).
    """
    path = Path(path)
    if not path.exists():
        return {}
    out: dict[str, tuple[str, float | None]] = {}
    with path.open() as fh:
        rows = csv.DictReader(line for line in fh if not line.lstrip().startswith("#"))
        for row in rows:
            name = (row.get("name") or "").strip()
            if not name:
                continue
            role = (row.get("role") or "default").strip() or "default"
            if role not in role_omega:
                raise ValueError(
                    f"omega overrides: parameter '{name}' has unknown role '{role}'. "
                    f"Known roles: {sorted(role_omega)}"
                )
            raw = (row.get("omega") or "").strip()
            explicit = float(raw) if raw else None
            if explicit is not None and not (0.0 < explicit < 5.0):
                raise ValueError(
                    f"omega overrides: parameter '{name}' has implausible omega={explicit}. "
                    "omega is a log-sd; expected roughly (0, 2)."
                )
            out[name] = (role, explicit)
    return out


def load_population_n(targets_dir: Path | str) -> dict[str, int]:
    """Biological n behind each parameter's population-spread data.

    Walks the submodel targets under ``targets_dir`` and, for every parameter a
    target constrains, takes the largest ``n_biological`` among that target's
    population-feeding ``observed_distribution`` entries. That n weighs the data
    against the class prior in the level-4 shrinkage; a parameter with no such
    target simply does not appear and its prior stands.
    """
    import yaml

    pop_sources = {"across_patient", "biological_experimental"}
    out: dict[str, int] = {}
    for path in sorted(Path(targets_dir).glob("*.yaml")):
        if "submodel_config" in path.name:
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        cal = doc.get("calibration") or {}
        entries = cal.get("error_model") or []
        if isinstance(entries, dict):
            entries = [entries]
        ns = [
            int(od["n_biological"])
            for e in entries
            if isinstance(e, dict)
            for od in [e.get("observed_distribution") or {}]
            if od.get("spread_source") in pop_sources and od.get("n_biological")
        ]
        if not ns:
            continue
        n_max = max(ns)
        for spec in cal.get("parameters") or []:
            name = spec.get("name") if isinstance(spec, dict) else spec
            if not name or (isinstance(spec, dict) and spec.get("nuisance")):
                continue
            out[name] = max(out.get(name, 0), n_max)
    return out


def shrink_toward_prior(
    data_omega: float, n: Optional[int], prior_omega: float, *, tau: float
) -> float:
    """MBMA shrinkage of a sample spread toward the class prior, weighted by n.

    In log space: ``log(s)`` has sampling sd ``~ 1/sqrt(2(n-1))``, carrying
    precision ``2(n-1)`` against the prior's ``1/tau^2``. ``n <= 1`` carries no
    information and returns the prior untouched.
    """
    if n is None or n <= 1 or not np.isfinite(data_omega) or data_omega <= 0:
        return prior_omega
    prec_data = 2.0 * (n - 1)
    prec_prior = 1.0 / (tau**2)
    log_omega = (prec_data * np.log(data_omega) + prec_prior * np.log(prior_omega)) / (
        prec_data + prec_prior
    )
    return float(np.exp(log_omega))


def build_omega_center(
    param_names: list[str],
    *,
    role_omega: Mapping[str, float],
    global_default: float,
    overrides: Optional[Mapping[str, tuple[str, float | None]]] = None,
    overrides_path: Optional[Path | str] = None,
    population_sigma: Optional[Mapping[str, float]] = None,
    population_n: Optional[Mapping[str, int]] = None,
    tau: float,
) -> tuple[np.ndarray, list[OmegaEntry]]:
    """Build the per-parameter omega center by walking the four layers.

    Args:
        param_names: parameters in the order the hyperprior expects them.
        role_omega: the project's role -> omega map (level 2); must contain
            ``"default"``.
        global_default: the level-1 omega for anything more specific is unknown.
        overrides: a preloaded ``name -> (role, explicit)`` map (level 2/3); if
            ``None`` it is read from ``overrides_path`` via
            :func:`load_omega_overrides`.
        overrides_path: overrides CSV to read when ``overrides`` is None.
        population_sigma: name -> population-scale log-sd (level-4 data), shrunk
            toward the level-1/2/3 value, not substituted.
        population_n: name -> biological n behind that sigma. Absent / <=1 means
            the data cannot outweigh the prior and the prior stands.
        tau: how much to trust the class prior (log-sd).

    Returns:
        (omega array aligned to ``param_names``, per-parameter provenance).
    """
    if overrides is None:
        overrides = (
            load_omega_overrides(overrides_path, role_omega=role_omega)
            if overrides_path is not None
            else {}
        )
    population_sigma = population_sigma or {}
    population_n = population_n or {}

    omega = np.empty(len(param_names), dtype=np.float64)
    provenance: list[OmegaEntry] = []

    for j, name in enumerate(param_names):
        role, explicit = overrides.get(name, ("default", None))

        # Levels 1-3: the prior omega, before any data.
        if explicit is not None:
            prior_omega, level = explicit, "explicit"
        elif role != "default":
            prior_omega, level = role_omega[role], "role"
        else:
            prior_omega, level = global_default, "global_default"

        # Level 4: shrink the data toward it, weighted by n.
        data_omega = population_sigma.get(name)
        n_bio = population_n.get(name)
        if data_omega is not None:
            value = shrink_toward_prior(data_omega, n_bio, prior_omega, tau=tau)
            level = "data_shrunk"
        else:
            value = prior_omega

        omega[j] = value
        provenance.append(
            OmegaEntry(
                name=name,
                omega=float(value),
                level=level,
                role=role,
                prior_omega=float(prior_omega),
                data_omega=None if data_omega is None else float(data_omega),
                n_biological=n_bio,
            )
        )

    return omega, provenance


def write_provenance(provenance: list[OmegaEntry], path: Path | str) -> None:
    """Dump the per-parameter omega provenance for audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["name", "omega", "level", "role", "prior_omega", "data_omega", "n_biological"]
        )
        for e in provenance:
            w.writerow(
                [e.name, e.omega, e.level, e.role, e.prior_omega, e.data_omega, e.n_biological]
            )

"""Auto-discover auxiliary parameter members by walking calibration targets.

A calibration target declares its measurement-bridging auxiliary parameters
on ``observable.auxiliary_parameters`` (maple schema, PR #37). This module
walks a tree of calibration target YAMLs, collects every declaration, and
validates them against an :class:`~qsp_inference.auxiliary.AuxiliaryConfig`
to produce an :class:`AuxiliaryRegistry` keyed by parameter name.

Cross-target consistency rules (option C from the design discussion):

- Same ``name`` across multiple cal targets ⇒ same ``group``. Two cal
  targets that reference the same auxiliary name must declare it with the
  same group; otherwise the inference workflow can't decide which prior to
  apply.
- ``biological_basis`` is per-reference and is *not* enforced for
  consistency — a serum:tumor ratio used by TGFβ and IL-12 cal targets
  legitimately has cytokine-specific rationale text on each.
- ``units`` must match across references (otherwise the same name would
  carry incompatible Pint quantities).
- Every ``group`` referenced by any auxiliary parameter must be declared
  in :class:`~qsp_inference.auxiliary.AuxiliaryConfig`.

The walk reads YAML directly rather than going through ``maple`` validation
— it only needs ``observable.auxiliary_parameters`` and is decoupled from
the rest of the cal-target schema. Cal-target schema validation is the
authoring tooling's job (maple); discovery is just a name-collection pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml

from qsp_inference.auxiliary.config import AuxiliaryConfig, AuxiliaryGroupSpec


@dataclass(frozen=True)
class AuxiliaryMember:
    """A single auxiliary parameter member resolved from cal target walks.

    Attributes:
        name: Globally-unique parameter name (the dict key in the registry).
        group: Group name; must match an entry in ``AuxiliaryConfig.groups``.
        units: Pint-parseable units string (e.g., ``"dimensionless"``).
        references: Cal-target YAML paths that declared this member, in
            discovery order. Useful for diagnostics when a consistency
            check fails.
        biological_basis_per_reference: Per-reference rationale text,
            indexed by the same order as ``references``. Stored so the
            inference workflow can emit member-level provenance reports
            without re-walking the cal targets.
    """

    name: str
    group: str
    units: str
    references: tuple[Path, ...] = field(default_factory=tuple)
    biological_basis_per_reference: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AuxiliaryRegistry:
    """Resolved set of auxiliary members organized by group.

    Built by :func:`discover_auxiliary_members`. Member order within each
    group is deterministic — sorted by name — so downstream callers
    (e.g., the prior module) get a stable parameter-vector layout.
    """

    members: dict[str, AuxiliaryMember]
    config: AuxiliaryConfig

    def __post_init__(self) -> None:
        # Eagerly materialize the (group → members) index so callers can
        # iterate without re-scanning every time.
        self._by_group: dict[str, list[AuxiliaryMember]] = {}
        for member in self.members.values():
            self._by_group.setdefault(member.group, []).append(member)
        for group_name in self._by_group:
            self._by_group[group_name].sort(key=lambda m: m.name)

    @property
    def is_empty(self) -> bool:
        return not self.members

    @property
    def group_names(self) -> tuple[str, ...]:
        """Group names that have at least one discovered member.

        Sorted alphabetically for deterministic ordering. Groups declared
        in ``AuxiliaryConfig`` but never referenced by any cal target are
        excluded — the inference workflow doesn't need a prior block for
        them.
        """
        return tuple(sorted(self._by_group.keys()))

    def members_in(self, group_name: str) -> tuple[AuxiliaryMember, ...]:
        """Members in ``group_name``, sorted by name."""
        return tuple(self._by_group.get(group_name, ()))

    def group_spec(self, group_name: str) -> AuxiliaryGroupSpec:
        """Return the prior spec for ``group_name``.

        Raises:
            KeyError: ``group_name`` is not declared in the config.
        """
        spec = self.config.get(group_name)
        if spec is None:
            raise KeyError(
                f"Auxiliary group '{group_name}' is not declared in auxiliary_config.yaml"
            )
        return spec

    @property
    def member_names(self) -> tuple[str, ...]:
        """All member names in deterministic order (by group, then name)."""
        out: list[str] = []
        for group_name in self.group_names:
            out.extend(m.name for m in self.members_in(group_name))
        return tuple(out)


# =============================================================================
# Walk + validation
# =============================================================================


def _iter_yaml_files(roots: Iterable[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if root.suffix in {".yaml", ".yml"}:
                yield root
            continue
        for path in sorted(root.rglob("*.yaml")):
            yield path
        for path in sorted(root.rglob("*.yml")):
            yield path


def _extract_auxiliary_entries(yaml_path: Path) -> list[dict]:
    """Return ``observable.auxiliary_parameters`` for a cal-target YAML.

    Empty list if the file isn't a cal target, doesn't have an observable,
    or doesn't declare any auxiliary parameters. Loud about malformed YAML
    so silent extraction failures don't drop members from the registry.
    """
    with yaml_path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return []
    observable = data.get("observable")
    if not isinstance(observable, dict):
        return []
    aux = observable.get("auxiliary_parameters", [])
    if aux is None:
        return []
    if not isinstance(aux, list):
        raise ValueError(
            f"{yaml_path}: observable.auxiliary_parameters must be a list "
            f"(got {type(aux).__name__})"
        )
    out: list[dict] = []
    for entry in aux:
        if not isinstance(entry, dict):
            raise ValueError(
                f"{yaml_path}: each auxiliary_parameters entry must be a mapping "
                f"(got {type(entry).__name__})"
            )
        out.append(entry)
    return out


def discover_auxiliary_members(
    calibration_target_roots: Iterable[Path | str],
    config: AuxiliaryConfig,
) -> AuxiliaryRegistry:
    """Walk calibration targets and build a validated registry.

    Args:
        calibration_target_roots: Directories or files to search. Directory
            entries are walked recursively for ``*.yaml``/``*.yml`` files.
            Missing roots are silently skipped (so a workflow can pass a
            superset of expected locations without crashing).
        config: Loaded :class:`AuxiliaryConfig` declaring the structural
            prior for each referenced group.

    Returns:
        :class:`AuxiliaryRegistry` containing every auxiliary parameter
        referenced by at least one cal target, indexed by name.

    Raises:
        ValueError: cross-target inconsistency (group/units mismatch on
            same name), undeclared group reference, or missing required
            fields on an auxiliary entry.
    """
    roots = [Path(r) for r in calibration_target_roots]
    members: dict[str, AuxiliaryMember] = {}

    for yaml_path in _iter_yaml_files(roots):
        try:
            entries = _extract_auxiliary_entries(yaml_path)
        except yaml.YAMLError as e:
            raise ValueError(f"{yaml_path}: failed to parse YAML ({e})") from e
        for entry in entries:
            _ingest_entry(yaml_path, entry, members, config)

    _validate_groups_declared(members, config)

    return AuxiliaryRegistry(members=members, config=config)


def _ingest_entry(
    yaml_path: Path,
    entry: dict,
    members: dict[str, AuxiliaryMember],
    config: AuxiliaryConfig,
) -> None:
    """Merge one cal-target's auxiliary entry into the in-progress registry."""
    name = entry.get("name")
    group = entry.get("group")
    units = entry.get("units", "dimensionless")
    biological_basis = entry.get("biological_basis", "")

    if not isinstance(name, str) or not name:
        raise ValueError(
            f"{yaml_path}: auxiliary parameter is missing 'name' or it is not a string"
        )
    if not isinstance(group, str) or not group:
        raise ValueError(
            f"{yaml_path}: auxiliary parameter '{name}' is missing 'group' "
            f"or it is not a string"
        )
    if not isinstance(units, str):
        raise ValueError(
            f"{yaml_path}: auxiliary parameter '{name}' has non-string 'units'"
        )

    existing = members.get(name)
    if existing is None:
        members[name] = AuxiliaryMember(
            name=name,
            group=group,
            units=units,
            references=(yaml_path,),
            biological_basis_per_reference=(biological_basis,),
        )
        return

    # Cross-reference consistency (option C).
    if existing.group != group:
        raise ValueError(
            f"Auxiliary parameter '{name}' is declared with conflicting groups: "
            f"'{existing.group}' (in {existing.references[0]}) vs '{group}' "
            f"(in {yaml_path}). Same name must use the same group across all "
            f"calibration targets."
        )
    if existing.units != units:
        raise ValueError(
            f"Auxiliary parameter '{name}' is declared with conflicting units: "
            f"'{existing.units}' (in {existing.references[0]}) vs '{units}' "
            f"(in {yaml_path})."
        )

    members[name] = AuxiliaryMember(
        name=existing.name,
        group=existing.group,
        units=existing.units,
        references=existing.references + (yaml_path,),
        biological_basis_per_reference=(
            existing.biological_basis_per_reference + (biological_basis,)
        ),
    )


def _validate_groups_declared(
    members: dict[str, AuxiliaryMember],
    config: AuxiliaryConfig,
) -> None:
    referenced = {m.group for m in members.values()}
    undeclared = sorted(referenced - set(config.groups))
    if undeclared:
        # Surface one example reference for each missing group to make the
        # error message diagnostic without dumping the full registry.
        examples: list[str] = []
        for group_name in undeclared:
            for member in members.values():
                if member.group == group_name:
                    examples.append(
                        f"  - '{group_name}' (e.g., used by '{member.name}' "
                        f"in {member.references[0]})"
                    )
                    break
        joined = "\n".join(examples)
        raise ValueError(
            "Auxiliary groups referenced by calibration targets but not declared "
            f"in auxiliary_config.yaml:\n{joined}"
        )

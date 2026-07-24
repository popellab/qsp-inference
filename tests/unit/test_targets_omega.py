"""Unit tests for the omega layered-prior builder (qsp_inference.targets.omega).

The prior half of the maple target-data contract: the four-layer population-
spread center (global -> role -> explicit -> data-shrunk), generic over the role
map / defaults injected by a project.
"""
import numpy as np
import pytest

from qsp_inference.targets import (
    OmegaEntry,
    build_omega_center,
    load_omega_overrides,
    shrink_toward_prior,
    write_provenance,
)

ROLE = {"default": 0.35, "packing_limit": 0.15, "material_constant": 0.10}
GLOBAL = 0.35
TAU = 0.5


def _build(names, **kw):
    return build_omega_center(
        names, role_omega=ROLE, global_default=GLOBAL, tau=TAU, **kw
    )


def test_global_default_when_nothing_specific():
    omega, prov = _build(["a", "b"])
    assert np.allclose(omega, GLOBAL)
    assert all(e.level == "global_default" for e in prov)


def test_role_and_explicit_layers():
    overrides = {"a": ("packing_limit", None), "b": ("default", 0.9)}
    omega, prov = _build(["a", "b", "c"], overrides=overrides)
    assert omega[0] == pytest.approx(0.15)  # role
    assert omega[1] == pytest.approx(0.9)   # explicit
    assert omega[2] == pytest.approx(GLOBAL)  # untouched
    assert [e.level for e in prov] == ["role", "explicit", "global_default"]


def test_data_shrinks_toward_prior_by_n():
    """Large n pulls omega toward the data spread; tiny n leaves the prior."""
    names = ["hi_n", "lo_n"]
    overrides = {"hi_n": ("default", None), "lo_n": ("default", None)}
    omega, prov = _build(
        names, overrides=overrides,
        population_sigma={"hi_n": 1.2, "lo_n": 1.2},
        population_n={"hi_n": 500, "lo_n": 2},
    )
    # hi_n: n=500 -> nearly the data (1.2); lo_n: n=2 -> barely moved off 0.35.
    assert omega[0] > 1.0
    assert omega[1] < 0.6
    assert all(e.level == "data_shrunk" for e in prov)
    assert prov[0].data_omega == 1.2 and prov[0].n_biological == 500


def test_shrink_toward_prior_edges():
    assert shrink_toward_prior(1.2, None, 0.35, tau=0.5) == 0.35   # no n
    assert shrink_toward_prior(1.2, 1, 0.35, tau=0.5) == 0.35      # n<=1
    assert shrink_toward_prior(-1.0, 100, 0.35, tau=0.5) == 0.35   # bad data
    # monotone in n: more data -> closer to the data value.
    near = shrink_toward_prior(1.2, 1000, 0.35, tau=0.5)
    far = shrink_toward_prior(1.2, 5, 0.35, tau=0.5)
    assert abs(near - 1.2) < abs(far - 1.2)


def test_unknown_role_in_csv_raises(tmp_path):
    p = tmp_path / "omega.csv"
    p.write_text("name,role,omega\nfoo,bogus,\n")
    with pytest.raises(ValueError, match="unknown role 'bogus'"):
        load_omega_overrides(p, role_omega=ROLE)


def test_implausible_explicit_omega_raises(tmp_path):
    p = tmp_path / "omega.csv"
    p.write_text("name,role,omega\nfoo,default,9.9\n")
    with pytest.raises(ValueError, match="implausible omega"):
        load_omega_overrides(p, role_omega=ROLE)


def test_overrides_csv_roundtrip(tmp_path):
    p = tmp_path / "omega.csv"
    p.write_text("# comment\nname,role,omega\na,packing_limit,\nb,default,0.8\n")
    ov = load_omega_overrides(p, role_omega=ROLE)
    assert ov == {"a": ("packing_limit", None), "b": ("default", 0.8)}


def test_missing_overrides_file_is_empty(tmp_path):
    assert load_omega_overrides(tmp_path / "nope.csv", role_omega=ROLE) == {}


def test_write_provenance_roundtrips(tmp_path):
    prov = [OmegaEntry("a", 0.35, "global_default", "default", 0.35, None, None)]
    out = tmp_path / "prov.csv"
    write_provenance(prov, out)
    text = out.read_text()
    assert "name,omega,level,role,prior_omega,data_omega,n_biological" in text
    assert "a,0.35,global_default,default,0.35," in text

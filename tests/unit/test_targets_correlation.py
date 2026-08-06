"""Between-patient correlation from factor loadings. eq:crn's R."""

import numpy as np
import pytest

from qsp_inference.targets.correlation import (
    MAX_COMMUNALITY,
    STRENGTH,
    correlation_from_loadings,
)

NAMES = ["a", "b", "c", "d"]
AXES = ["fibrosis", "immune"]


def _load(axis, strength="primary", sign="+"):
    return [{"axis": axis, "strength": strength, "sign": sign}]


def test_no_loadings_is_the_identity():
    """The default is independence, and it has to be exactly independence."""
    R, lam, unknown = correlation_from_loadings(NAMES, {}, AXES)
    assert np.allclose(R, np.eye(4))
    assert not lam.any() and unknown == []


def test_a_shared_axis_correlates_and_an_unshared_one_does_not():
    R, _, _ = correlation_from_loadings(
        NAMES, {"a": _load("fibrosis"), "b": _load("fibrosis"), "c": _load("immune")}, AXES
    )
    assert R[0, 1] == pytest.approx(STRENGTH["primary"] ** 2)
    assert R[0, 2] == 0.0
    assert R[0, 3] == 0.0


def test_opposite_signs_give_a_negative_correlation():
    R, _, _ = correlation_from_loadings(
        NAMES, {"a": _load("fibrosis"), "b": _load("fibrosis", sign="-")}, AXES
    )
    assert R[0, 1] == pytest.approx(-STRENGTH["primary"] ** 2)


def test_it_is_positive_definite_however_it_is_loaded():
    """The whole reason for the factor form: no projection is ever needed."""
    rng = np.random.default_rng(0)
    names = [f"p{i}" for i in range(60)]
    axes = [f"ax{k}" for k in range(9)]
    loads = {
        n: [{"axis": rng.choice(axes), "strength": rng.choice(list(STRENGTH)),
             "sign": rng.choice(["+", "-"])} for _ in range(rng.integers(0, 3))]
        for n in names
    }
    R, _, _ = correlation_from_loadings(names, loads, axes)
    assert np.linalg.eigvalsh(R).min() > 0
    np.linalg.cholesky(R)  # must not raise
    assert np.allclose(np.diag(R), 1.0)
    assert np.allclose(R, R.T)


def test_two_maximal_loadings_are_capped_below_full_determination():
    """No parameter is entirely explained by the axes, so R stays PD not PSD."""
    two = [{"axis": "fibrosis", "strength": "primary", "sign": "+"},
           {"axis": "immune", "strength": "primary", "sign": "+"}]
    R, lam, _ = correlation_from_loadings(NAMES, {"a": two, "b": two}, AXES)
    assert (lam[0] ** 2).sum() == pytest.approx(MAX_COMMUNALITY)
    assert R[0, 1] == pytest.approx(MAX_COMMUNALITY)
    assert np.linalg.eigvalsh(R).min() > 0


def test_an_axis_nobody_declared_is_reported_not_invented():
    """A typo would otherwise become a factor that nothing else loads on."""
    R, _, unknown = correlation_from_loadings(
        NAMES, {"a": _load("fibrossis"), "b": _load("fibrosis")}, AXES
    )
    assert unknown == ["fibrossis"]
    assert R[0, 1] == 0.0


def test_an_unknown_strength_raises():
    with pytest.raises(ValueError, match="unknown loading strength"):
        correlation_from_loadings(NAMES, {"a": _load("fibrosis", strength="huge")}, AXES)


def test_parameter_order_is_the_callers():
    """R is indexed by parameter, so a reordering silently permutes the cloud."""
    loads = {"a": _load("fibrosis"), "b": _load("fibrosis")}
    R1, _, _ = correlation_from_loadings(["a", "b", "c"], loads, AXES)
    R2, _, _ = correlation_from_loadings(["c", "a", "b"], loads, AXES)
    assert R1[0, 1] == pytest.approx(R2[1, 2])
    assert R2[0, 1] == 0.0

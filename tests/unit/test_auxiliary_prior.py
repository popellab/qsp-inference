"""Tests for ``qsp_inference.auxiliary.prior``."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # sbi extra, not in default install

import yaml

from qsp_inference.auxiliary.config import (
    AuxiliaryBasePrior,
    AuxiliaryConfig,
    AuxiliaryGroupSpec,
)
from qsp_inference.auxiliary.discovery import discover_auxiliary_members
from qsp_inference.auxiliary.prior import (
    HierarchicalAuxiliaryPrior,
    merge_into_constants,
)


def _config(*, distribution: str = "lognormal", mu: float = 0.0,
            sigma: float = 0.5, tau: float = 0.2) -> AuxiliaryConfig:
    return AuxiliaryConfig(
        groups={
            "g": AuxiliaryGroupSpec(
                base_prior=AuxiliaryBasePrior(
                    distribution=distribution, mu=mu, sigma=sigma
                ),
                member_deviation_sigma=tau,
            )
        }
    )


def _write_target(path: Path, names: list[str], group: str = "g") -> None:
    aux = [
        {"name": n, "group": group, "biological_basis": "x" * 25, "units": "dimensionless"}
        for n in names
    ]
    path.write_text(
        yaml.safe_dump(
            {
                "observable": {
                    "code": "def compute_observable(t, s, c, u): pass",
                    "units": "nM",
                    "auxiliary_parameters": aux,
                }
            }
        )
    )


def _registry(tmp_path: Path, names: list[str], cfg: AuxiliaryConfig | None = None):
    if cfg is None:
        cfg = _config()
    _write_target(tmp_path / "t.yaml", names)
    return discover_auxiliary_members([tmp_path], cfg)


# --------------------------------------------------------------------------
# basic shape + parameter ordering
# --------------------------------------------------------------------------


def test_param_names_match_registry_order(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_b", "f_a", "f_c"])
    prior = HierarchicalAuxiliaryPrior(reg)
    assert prior.param_names == reg.member_names == ("f_a", "f_b", "f_c")
    assert prior.event_dim == 3


def test_empty_registry_returns_zero_dim_prior(tmp_path: Path) -> None:
    reg = discover_auxiliary_members([tmp_path], AuxiliaryConfig())
    prior = HierarchicalAuxiliaryPrior(reg)
    assert prior.event_dim == 0
    out = prior.sample((4,))
    assert out.shape == (4, 0)
    lp = prior.log_prob(out)
    assert lp.shape == (4,)
    assert torch.all(lp == 0)


def test_sample_shape(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"])
    prior = HierarchicalAuxiliaryPrior(reg)
    s = prior.sample((100,))
    assert s.shape == (100, 2)
    assert s.dtype == torch.float64


def test_sample_lognormal_is_positive(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"])  # default lognormal
    prior = HierarchicalAuxiliaryPrior(reg)
    s = prior.sample((1000,))
    assert torch.all(s > 0)


def test_sample_normal_can_be_negative(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"], cfg=_config(distribution="normal"))
    prior = HierarchicalAuxiliaryPrior(reg)
    torch.manual_seed(0)
    s = prior.sample((5000,))
    assert (s < 0).any()


# --------------------------------------------------------------------------
# marginal moments and sibling correlation (option-4 closed form)
# --------------------------------------------------------------------------


def test_lognormal_marginal_moments(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"], cfg=_config(mu=0.5, sigma=0.4, tau=0.3))
    prior = HierarchicalAuxiliaryPrior(reg)
    torch.manual_seed(42)
    s = prior.sample((50_000,))
    log_s = torch.log(s).numpy()
    expected_var = 0.4**2 + 0.3**2  # sigma**2 + tau**2
    np.testing.assert_allclose(log_s.mean(axis=0), [0.5, 0.5], atol=0.02)
    np.testing.assert_allclose(log_s.var(axis=0), [expected_var, expected_var], rtol=0.05)


def test_sibling_log_correlation(tmp_path: Path) -> None:
    """Within-group siblings should have log-space correlation
    sigma**2 / (sigma**2 + tau**2)."""
    reg = _registry(tmp_path, ["f_a", "f_b"], cfg=_config(sigma=0.6, tau=0.2))
    prior = HierarchicalAuxiliaryPrior(reg)
    torch.manual_seed(0)
    s = prior.sample((50_000,))
    log_s = torch.log(s).numpy()
    expected_rho = 0.6**2 / (0.6**2 + 0.2**2)  # ≈ 0.9
    rho = np.corrcoef(log_s.T)[0, 1]
    assert abs(rho - expected_rho) < 0.02


def test_groups_are_independent(tmp_path: Path) -> None:
    """Members of different groups are independent at the prior."""
    cfg = AuxiliaryConfig(
        groups={
            "g1": AuxiliaryGroupSpec(
                base_prior=AuxiliaryBasePrior(distribution="lognormal", mu=0, sigma=0.5),
                member_deviation_sigma=0.2,
            ),
            "g2": AuxiliaryGroupSpec(
                base_prior=AuxiliaryBasePrior(distribution="lognormal", mu=0, sigma=0.5),
                member_deviation_sigma=0.2,
            ),
        }
    )
    (tmp_path / "t.yaml").write_text(
        yaml.safe_dump(
            {
                "observable": {
                    "code": "x",
                    "units": "nM",
                    "auxiliary_parameters": [
                        {"name": "f1", "group": "g1", "biological_basis": "x" * 25,
                         "units": "dimensionless"},
                        {"name": "f2", "group": "g2", "biological_basis": "x" * 25,
                         "units": "dimensionless"},
                    ],
                }
            }
        )
    )
    reg = discover_auxiliary_members([tmp_path], cfg)
    prior = HierarchicalAuxiliaryPrior(reg)
    torch.manual_seed(0)
    s = prior.sample((50_000,))
    log_s = torch.log(s).numpy()
    rho = np.corrcoef(log_s.T)[0, 1]
    assert abs(rho) < 0.02


# --------------------------------------------------------------------------
# log_prob
# --------------------------------------------------------------------------


def test_log_prob_shape(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"])
    prior = HierarchicalAuxiliaryPrior(reg)
    s = prior.sample((7,))
    lp = prior.log_prob(s)
    assert lp.shape == (7,)
    assert torch.all(torch.isfinite(lp))


def test_log_prob_dim_mismatch_raises(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"])
    prior = HierarchicalAuxiliaryPrior(reg)
    bad = torch.zeros((3, 4), dtype=torch.float64)
    with pytest.raises(ValueError, match="event_dim"):
        prior.log_prob(bad)


def test_lognormal_log_prob_matches_change_of_variables(tmp_path: Path) -> None:
    """Single-member lognormal group: log_prob(theta) should match
    Normal(mu, sigma_total).log_prob(log(theta)) - log(theta)."""
    reg = _registry(tmp_path, ["f_a"], cfg=_config(mu=0.5, sigma=0.4, tau=0.3))
    prior = HierarchicalAuxiliaryPrior(reg)
    theta = torch.tensor([[0.7], [1.2], [3.0]], dtype=torch.float64)
    lp = prior.log_prob(theta)

    sigma_tot2 = 0.4**2 + 0.3**2  # marginal log-space var for n=1
    log_theta = torch.log(theta).squeeze(-1)
    expected = (
        -0.5 * torch.log(torch.tensor(2 * np.pi * sigma_tot2))
        - 0.5 * (log_theta - 0.5) ** 2 / sigma_tot2
        - log_theta
    )
    torch.testing.assert_close(lp, expected, rtol=1e-6, atol=1e-8)


def test_normal_log_prob_matches_mvn(tmp_path: Path) -> None:
    reg = _registry(
        tmp_path, ["f_a", "f_b"], cfg=_config(distribution="normal", mu=1.0, sigma=0.5, tau=0.3)
    )
    prior = HierarchicalAuxiliaryPrior(reg)
    theta = torch.tensor([[1.1, 0.9], [1.5, 1.6]], dtype=torch.float64)
    lp = prior.log_prob(theta)

    cov = 0.5**2 * torch.ones((2, 2), dtype=torch.float64) + 0.3**2 * torch.eye(
        2, dtype=torch.float64
    )
    mean = torch.tensor([1.0, 1.0], dtype=torch.float64)
    expected = torch.distributions.MultivariateNormal(mean, cov).log_prob(theta)
    torch.testing.assert_close(lp, expected, rtol=1e-6, atol=1e-8)


# --------------------------------------------------------------------------
# Solo group (n=1) and tau=0 edge cases
# --------------------------------------------------------------------------


def test_solo_group_with_tau_zero(tmp_path: Path) -> None:
    """A solo group with tau=0 reduces to plain lognormal sampling."""
    reg = _registry(
        tmp_path, ["f_a"], cfg=_config(mu=0.0, sigma=0.4, tau=0.0)
    )
    prior = HierarchicalAuxiliaryPrior(reg)
    torch.manual_seed(0)
    s = prior.sample((50_000,)).squeeze(-1)
    log_s = torch.log(s).numpy()
    np.testing.assert_allclose(log_s.mean(), 0.0, atol=0.02)
    np.testing.assert_allclose(log_s.var(), 0.4**2, rtol=0.05)


def test_two_member_group_tau_zero_perfectly_correlated(tmp_path: Path) -> None:
    """With tau=0 but n>1, sibling members share the same draw."""
    reg = _registry(
        tmp_path, ["f_a", "f_b"], cfg=_config(mu=0.0, sigma=0.5, tau=0.0)
    )
    prior = HierarchicalAuxiliaryPrior(reg)
    torch.manual_seed(0)
    s = prior.sample((1000,))
    # log f_a == log f_b within numerical jitter
    log_s = torch.log(s).numpy()
    diff = log_s[:, 0] - log_s[:, 1]
    assert np.abs(diff).max() < 1e-3


# --------------------------------------------------------------------------
# Helpers: sample_as_records, merge_into_constants
# --------------------------------------------------------------------------


def test_sample_as_records_yields_one_dict_per_sim(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"])
    prior = HierarchicalAuxiliaryPrior(reg)
    records = prior.sample_as_records(5, seed=123)
    assert len(records) == 5
    for rec in records:
        assert set(rec) == {"f_a", "f_b"}
        assert all(isinstance(v, float) for v in rec.values())


def test_sample_as_records_seed_is_reproducible(tmp_path: Path) -> None:
    reg = _registry(tmp_path, ["f_a", "f_b"])
    prior = HierarchicalAuxiliaryPrior(reg)
    a = prior.sample_as_records(3, seed=7)
    b = prior.sample_as_records(3, seed=7)
    assert a == b


def test_merge_into_constants_attaches_aux_as_floats() -> None:
    base = {"area_per_cell": 80.0}
    aux = {"f_serum_to_tumor_TGFb": 1.5}
    merged = merge_into_constants(base, aux)
    assert merged["area_per_cell"] == 80.0
    assert merged["f_serum_to_tumor_TGFb"] == pytest.approx(1.5)
    assert isinstance(merged["f_serum_to_tumor_TGFb"], float)
    # Original dict unchanged
    assert "f_serum_to_tumor_TGFb" not in base


def test_merge_into_constants_rejects_collision() -> None:
    base = {"shared_name": 2.0}
    aux = {"shared_name": 1.5}
    with pytest.raises(ValueError, match="collides"):
        merge_into_constants(base, aux)

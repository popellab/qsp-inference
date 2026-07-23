"""Tests for derived-parameter priors (power-law children of anchored parents).

Covers ``apply_derived_priors`` (the closed-form linear-Gaussian injection),
``load_derived_specs`` (the YAML policy loader), and ``_derived_topo_order``.
A derived param is ``value_d = coeff * prod_i parent_i ** power_i`` with
``coeff ~ LogNormal(log_coeff, sigma_coeff)``; in log-space it is a
linear-Gaussian child, so the augmented prior stays a ``GaussianCopulaPrior``.
"""

import math

import numpy as np
import pytest

pytest.importorskip("torch")  # copula_prior pulls in torch (sbi extra)
from scipy import stats

from qsp_inference.priors.copula_prior import (
    GaussianCopulaPrior,
    apply_derived_priors,
    load_derived_specs,
    _derived_topo_order,
)


def _prior(names, mus, sigmas, R=None):
    marg = [stats.norm(loc=mu, scale=s) for mu, s in zip(mus, sigmas)]
    return GaussianCopulaPrior(marginals=marg, correlation=R, param_names=names)


def _log_moments(prior, n=200_000, seed=0):
    """Empirical (mean, std) of each log-column and the full correlation matrix."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    X = prior.sample((n,)).double().numpy()  # log-space values
    return X.mean(0), X.std(0), np.corrcoef(X, rowvar=False)


# ---------------------------------------------------------------------------
# The closed form matches sampled ground truth.
# ---------------------------------------------------------------------------

def test_single_parent_marginal_and_correlation():
    # parent 'k' (anchored), 'd' placeholder to be overwritten as d = 0.6 * k.
    mu_k, sig_k = math.log(0.08), 0.53
    log_coeff, sig_c = math.log(0.6), 0.5
    prior = _prior(["k", "d"], [mu_k, 0.0], [sig_k, 1.5])

    specs = {"d": {"parents": {"k": 1.0}, "log_coeff": log_coeff,
                   "sigma_coeff": sig_c, "provenance": ""}}
    out = apply_derived_priors(prior, specs)

    # Closed-form expectations.
    mu_d = log_coeff + mu_k
    sig_d = math.sqrt(sig_k**2 + sig_c**2)
    corr_dk = sig_k / sig_d

    emp_mu, emp_sig, emp_R = _log_moments(out)
    # index 1 == 'd', 0 == 'k'
    assert emp_mu[1] == pytest.approx(mu_d, abs=0.02)
    assert emp_sig[1] == pytest.approx(sig_d, rel=0.03)
    assert emp_R[0, 1] == pytest.approx(corr_dk, abs=0.03)
    # parent untouched
    assert emp_mu[0] == pytest.approx(mu_k, abs=0.02)
    assert emp_sig[0] == pytest.approx(sig_k, rel=0.03)


def test_derived_inherits_narrow_width_from_parent():
    # A wide placeholder (sigma 1.5) becomes narrow once derived from a narrow parent.
    prior = _prior(["k", "d"], [math.log(0.08), math.log(0.5)], [0.5, 1.5])
    specs = {"d": {"parents": {"k": 1.0}, "log_coeff": math.log(0.6),
                   "sigma_coeff": 0.4, "provenance": ""}}
    out = apply_derived_priors(prior, specs)
    _, emp_sig, emp_R = _log_moments(out)
    assert emp_sig[1] < 0.8  # was 1.5, now ~sqrt(.5^2+.4^2)=0.64
    assert emp_R[0, 1] > 0.6  # strongly correlated with the parent


def test_product_of_two_parents_with_powers():
    # d = c * a^1 * b^0.5, parents independent.
    mu_a, mu_b = math.log(2.0), math.log(9.0)
    sa, sb = 0.3, 0.4
    log_coeff, sc = math.log(1.5), 0.2
    prior = _prior(["a", "b", "d"], [mu_a, mu_b, 0.0], [sa, sb, 1.0])
    specs = {"d": {"parents": {"a": 1.0, "b": 0.5}, "log_coeff": log_coeff,
                   "sigma_coeff": sc, "provenance": ""}}
    out = apply_derived_priors(prior, specs)

    mu_d = log_coeff + 1.0 * mu_a + 0.5 * mu_b
    sig_d = math.sqrt((1.0 * sa) ** 2 + (0.5 * sb) ** 2 + sc**2)
    emp_mu, emp_sig, emp_R = _log_moments(out)
    assert emp_mu[2] == pytest.approx(mu_d, abs=0.02)
    assert emp_sig[2] == pytest.approx(sig_d, rel=0.04)
    # corr to a: (1*sa)/sig_d ; to b: (0.5*sb)/sig_d
    assert emp_R[0, 2] == pytest.approx(sa / sig_d, abs=0.03)
    assert emp_R[1, 2] == pytest.approx(0.5 * sb / sig_d, abs=0.03)


def test_correlated_parents_covariance_propagates():
    # Parents a,b correlated; derived variance must use the full covariance.
    rho = 0.6
    R = np.array([[1.0, rho, 0.0], [rho, 1.0, 0.0], [0.0, 0.0, 1.0]])
    sa, sb = 0.4, 0.5
    prior = _prior(["a", "b", "d"], [0.0, 0.0, 0.0], [sa, sb, 1.0], R=R)
    specs = {"d": {"parents": {"a": 1.0, "b": 1.0}, "log_coeff": 0.0,
                   "sigma_coeff": 0.0, "provenance": ""}}
    out = apply_derived_priors(prior, specs)
    # Var(log d) = sa^2 + sb^2 + 2*rho*sa*sb
    sig_d = math.sqrt(sa**2 + sb**2 + 2 * rho * sa * sb)
    _, emp_sig, _ = _log_moments(out)
    assert emp_sig[2] == pytest.approx(sig_d, rel=0.04)


def test_output_is_gaussian_copula_and_logprob_finite():
    prior = _prior(["k", "d"], [0.0, 0.0], [0.5, 1.0])
    specs = {"d": {"parents": {"k": 1.0}, "log_coeff": 0.1,
                   "sigma_coeff": 0.3, "provenance": ""}}
    out = apply_derived_priors(prior, specs)
    assert isinstance(out, GaussianCopulaPrior)
    x = out.sample((16,))
    lp = out.log_prob(x)
    assert lp.shape[0] == 16
    assert bool(np.isfinite(lp.detach().numpy()).all())


# ---------------------------------------------------------------------------
# Topological ordering: a derived param may depend on another derived param.
# ---------------------------------------------------------------------------

def test_derived_on_derived_chain():
    # e = 2 * d ; d = 3 * k. So e should track k at product 6.
    prior = _prior(["k", "d", "e"], [math.log(1.0), 0.0, 0.0], [0.4, 1.0, 1.0])
    specs = {
        "e": {"parents": {"d": 1.0}, "log_coeff": math.log(2.0), "sigma_coeff": 0.0,
              "provenance": ""},
        "d": {"parents": {"k": 1.0}, "log_coeff": math.log(3.0), "sigma_coeff": 0.0,
              "provenance": ""},
    }
    order = _derived_topo_order(specs)
    assert order.index("d") < order.index("e")
    out = apply_derived_priors(prior, specs)
    emp_mu, _, emp_R = _log_moments(out)
    assert emp_mu[1] == pytest.approx(math.log(3.0), abs=0.02)  # d = 3*1
    assert emp_mu[2] == pytest.approx(math.log(6.0), abs=0.02)  # e = 2*d = 6
    assert emp_R[0, 2] > 0.9  # e strongly tracks k through d


def test_topo_order_raises_on_cycle():
    specs = {
        "a": {"parents": {"b": 1.0}, "log_coeff": 0.0, "sigma_coeff": 0.0, "provenance": ""},
        "b": {"parents": {"a": 1.0}, "log_coeff": 0.0, "sigma_coeff": 0.0, "provenance": ""},
    }
    with pytest.raises(ValueError, match="cycle"):
        _derived_topo_order(specs)


# ---------------------------------------------------------------------------
# Errors and edge cases.
# ---------------------------------------------------------------------------

def test_missing_derived_column_raises():
    prior = _prior(["k"], [0.0], [0.5])
    specs = {"d": {"parents": {"k": 1.0}, "log_coeff": 0.0, "sigma_coeff": 0.1,
                   "provenance": ""}}
    with pytest.raises(ValueError, match="not present as prior columns"):
        apply_derived_priors(prior, specs)


def test_unknown_parent_raises():
    prior = _prior(["k", "d"], [0.0, 0.0], [0.5, 1.0])
    specs = {"d": {"parents": {"nope": 1.0}, "log_coeff": 0.0, "sigma_coeff": 0.1,
                   "provenance": ""}}
    with pytest.raises(ValueError, match="unknown parent"):
        apply_derived_priors(prior, specs)


def test_empty_specs_is_identity():
    prior = _prior(["k", "d"], [0.0, 0.1], [0.5, 1.0])
    out = apply_derived_priors(prior, {})
    assert out is prior


# ---------------------------------------------------------------------------
# load_derived_specs parsing.
# ---------------------------------------------------------------------------

def test_load_derived_specs(tmp_path):
    p = tmp_path / "derived.yaml"
    p.write_text(
        "parameters:\n"
        "  k_CD8_T_vac_pro:\n"
        "    parents: {k_CD8_T_pro: 1.0}\n"
        "    log_coeff: -0.51\n"
        "    sigma_coeff: 0.5\n"
        "    provenance: fraction of constitutive\n"
    )
    specs = load_derived_specs(p)
    assert set(specs) == {"k_CD8_T_vac_pro"}
    s = specs["k_CD8_T_vac_pro"]
    assert s["parents"] == {"k_CD8_T_pro": 1.0}
    assert s["log_coeff"] == pytest.approx(-0.51)
    assert s["sigma_coeff"] == pytest.approx(0.5)
    assert "fraction" in s["provenance"]


def test_load_derived_specs_rejects_parentless(tmp_path):
    p = tmp_path / "derived.yaml"
    p.write_text("parameters:\n  x:\n    log_coeff: 0.0\n")
    with pytest.raises(ValueError, match="no parents"):
        load_derived_specs(p)

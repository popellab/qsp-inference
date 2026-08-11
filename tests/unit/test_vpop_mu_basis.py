"""Unit tests for eq:muprior restricted to a subspace.

``mu_basis`` is fix_mu's answer to a corpus that constrains directions rather
than coordinates. The claims here are that the centre moves only inside the
span, that ``Sigma_1``'s correlation still applies (which is the whole reason
this exists rather than fix_mu), and that a basis which is not orthonormal or
not built against this parameter vector is refused rather than used.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.fit import PopulationPrior, phi_from_sites, site_spec


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _prior(**kw):
    # A deliberately CORRELATED Sigma_1: the off-diagonal is what fix_mu refuses
    # and what a basis has to carry through untouched.
    L = jnp.array([[0.4, 0.0, 0.0],
                   [0.2, 0.3, 0.0],
                   [0.1, 0.1, 0.5]])
    base = dict(
        mu_0=jnp.array([1.5, 1.0, 0.2]),
        L_sigma_1=L,
        omega_0=jnp.array([0.5, 0.3, 0.4]),
        tau_s=0.3, tau_u=0.085,
        sigma_a=0.5, sigma_b=0.5,
        tau_beta=0.15, n_beta=0, dim_a=3, dim_b=2,
        log_R_0=jnp.zeros(0), sigma_R=jnp.zeros(0),
        pin_discrepancy=False, pin_aux=False, pin_b_columns=(),
    )
    base.update(kw)
    return PopulationPrior(**base)


def _basis(cols):
    """Orthonormal (3, k) from the given columns."""
    Q, _ = np.linalg.qr(np.asarray(cols, dtype=float))
    return Q


class TestValidation:
    def test_non_orthonormal_is_refused(self):
        B = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # second column not unit
        with pytest.raises(ValueError, match="not orthonormal"):
            _prior(mu_basis=B)

    def test_wrong_number_of_rows_is_refused(self):
        with pytest.raises(ValueError, match=r"expected \(3, k\)"):
            _prior(mu_basis=np.eye(4)[:, :2])

    def test_one_dimensional_array_is_refused(self):
        with pytest.raises(ValueError, match=r"expected \(3, k\)"):
            _prior(mu_basis=np.ones(3))

    def test_with_fix_mu_is_refused(self):
        # Both restrict where the centre may go and they disagree about how.
        with pytest.raises(ValueError, match="Pass one"):
            _prior(mu_basis=_basis([[1.0], [0.0], [0.0]]), fix_mu=(1,))

    def test_a_full_rank_basis_is_allowed(self):
        # k = P is the unrestricted model written in another basis, not an error.
        p = _prior(mu_basis=_basis(np.eye(3)))
        assert p.mu_basis.shape == (3, 3)


class TestTheCentreMovesOnlyInTheSpan:
    def test_site_is_named_and_sized_for_the_basis(self):
        p = _prior(mu_basis=_basis([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        names = [n for n, _, _ in site_spec(p)]
        assert "mu_c" in names
        assert "mu_raw" not in names and "mu_free" not in names
        shape = dict((n, v.shape) for n, v, _ in site_spec(p))["mu_c"]
        assert shape == (2,)

    def test_mu_raw_lands_in_the_span(self):
        B = _basis([[1.0], [0.0], [0.0]])
        p = _prior(mu_basis=B)
        mu, *_ = phi_from_sites({"mu_c": jnp.array([1.3]),
                                 "u_raw": jnp.zeros(3), "s": jnp.zeros(()),
                                 "a": jnp.zeros(3), "b": jnp.zeros(2)}, p)
        mu_raw = np.linalg.solve(np.asarray(p.L_sigma_1),
                                 np.asarray(mu) - np.asarray(p.mu_0))
        # The component orthogonal to the span is zero, which is the claim.
        resid = mu_raw - B @ (B.T @ mu_raw)
        assert np.abs(resid).max() == pytest.approx(0.0, abs=1e-12)

    def test_zero_coefficients_leave_the_centre_at_the_prior(self):
        p = _prior(mu_basis=_basis([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
        mu, *_ = phi_from_sites({"mu_c": jnp.zeros(2), "u_raw": jnp.zeros(3),
                                 "s": jnp.zeros(()), "a": jnp.zeros(3),
                                 "b": jnp.zeros(2)}, p)
        assert np.asarray(mu) == pytest.approx(np.asarray(p.mu_0), rel=1e-12)

    def test_correlation_survives_the_restriction(self):
        # The point of a basis over fix_mu. Moving one coefficient moves a
        # parameter that is NOT in the basis vector, through Sigma_1.
        B = _basis([[1.0], [0.0], [0.0]])
        p = _prior(mu_basis=B)
        mu, *_ = phi_from_sites({"mu_c": jnp.array([1.0]),
                                 "u_raw": jnp.zeros(3), "s": jnp.zeros(()),
                                 "a": jnp.zeros(3), "b": jnp.zeros(2)}, p)
        d = np.asarray(mu) - np.asarray(p.mu_0)
        assert abs(d[0]) > 1e-9
        # L_sigma_1 rows 1 and 2 load on column 0, so they move too. fix_mu
        # would have refused this prior outright rather than carry it.
        assert abs(d[1]) > 1e-9
        assert abs(d[2]) > 1e-9
        assert d == pytest.approx(np.asarray(p.L_sigma_1)[:, 0], rel=1e-12)

    def test_full_rank_basis_reproduces_the_unrestricted_map(self):
        Q = _basis(np.eye(3))
        p_b, p_full = _prior(mu_basis=Q), _prior()
        c = jnp.array([0.7, -1.1, 0.4])
        common = dict(u_raw=jnp.zeros(3), s=jnp.zeros(()),
                      a=jnp.zeros(3), b=jnp.zeros(2))
        mu_b, *_ = phi_from_sites({"mu_c": c, **common}, p_b)
        mu_f, *_ = phi_from_sites({"mu_raw": Q @ c, **common}, p_full)
        assert np.asarray(mu_b) == pytest.approx(np.asarray(mu_f), rel=1e-12)

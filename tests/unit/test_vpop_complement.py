"""Unit tests for restoring the subspace's complement when reporting a centre.

The fit conditions on the complement being zero and moves its variance into V.
Reporting the fitted ``mu`` alone would then print a zero-width interval along
every dropped direction, which is the overconfidence the truncation term exists
to prevent, arriving one step later.
"""
import numpy as np
import pytest

from qsp_inference.vpop.fit import PopulationPrior, centres_with_complement


def _prior(B=None, P=4):
    rng = np.random.default_rng(0)
    L = np.tril(rng.standard_normal((P, P))) + np.eye(P) * 2.0
    import jax.numpy as jnp
    return PopulationPrior(
        mu_0=jnp.asarray(np.arange(P, dtype=float)),
        L_sigma_1=jnp.asarray(L),
        omega_0=jnp.asarray(np.full(P, 0.3)),
        tau_s=0.3, tau_u=0.085, sigma_a=0.5, sigma_b=0.5,
        tau_beta=0.15, n_beta=0, dim_a=2, dim_b=2,
        log_R_0=jnp.zeros(0), sigma_R=jnp.zeros(0),
        pin_discrepancy=False, pin_aux=False, pin_b_columns=(),
        mu_basis=B,
    )


def _basis(P, k, seed=1):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((P, k)))
    return Q[:, :k]


class TestItRestoresWhatTheFitHeldAtZero:
    def test_refused_without_a_basis(self):
        with pytest.raises(ValueError, match="no complement"):
            centres_with_complement(np.zeros((3, 2)), _prior(),
                                    np.random.default_rng(0))

    def test_wrong_coefficient_count_is_refused(self):
        B = _basis(4, 2)
        with pytest.raises(ValueError, match="against the basis"):
            centres_with_complement(np.zeros((3, 3)), _prior(B),
                                    np.random.default_rng(0))

    def test_the_in_span_part_is_untouched(self):
        # Projecting the restored centre back onto the span has to give exactly
        # what the fit sampled; the complement is added, not mixed in.
        B = _basis(4, 2)
        p = _prior(B)
        c = np.random.default_rng(2).standard_normal((50, 2))
        mu = centres_with_complement(c, p, np.random.default_rng(3))
        raw = np.linalg.solve(np.asarray(p.L_sigma_1),
                              (mu - np.asarray(p.mu_0)).T).T
        assert raw @ B == pytest.approx(c, rel=1e-10)

    def test_the_complement_is_standard_normal(self):
        # It was never learned, so what comes back is the prior: unit variance in
        # every dropped direction.
        B = _basis(6, 2)
        p = _prior(B, P=6)
        c = np.zeros((40000, 2))
        mu = centres_with_complement(c, p, np.random.default_rng(4))
        raw = np.linalg.solve(np.asarray(p.L_sigma_1),
                              (mu - np.asarray(p.mu_0)).T).T
        perp = raw - (raw @ B) @ B.T
        C = np.cov(perp, rowvar=False)
        assert np.diag(C @ (np.eye(6) - B @ B.T)).sum() == \
            pytest.approx(4.0, rel=0.05)          # trace = P - k

    def test_it_widens_rather_than_shifts(self):
        # A coherent population-level shift, mean-zero: the centre's location is
        # not moved on average, only made honestly uncertain.
        B = _basis(5, 2)
        p = _prior(B, P=5)
        c = np.tile(np.array([[0.4, -0.9]]), (40000, 1))
        mu = centres_with_complement(c, p, np.random.default_rng(5))
        held = np.asarray(p.mu_0) + (c[:1] @ B.T) @ np.asarray(p.L_sigma_1).T
        assert mu.mean(axis=0) == pytest.approx(held[0], abs=0.05)
        assert (mu.std(axis=0) > 0).all()

    def test_a_full_rank_basis_restores_nothing(self):
        B = _basis(4, 4)
        p = _prior(B)
        c = np.random.default_rng(6).standard_normal((100, 4))
        a = centres_with_complement(c, p, np.random.default_rng(7))
        b = centres_with_complement(c, p, np.random.default_rng(8))
        assert a == pytest.approx(b, rel=1e-10)   # no complement, so no randomness

    def test_draws_are_independent_across_rows(self):
        # A single xi reused for every draw would be a slice, not a posterior.
        B = _basis(5, 1)
        p = _prior(B, P=5)
        mu = centres_with_complement(np.zeros((2000, 1)), p,
                                     np.random.default_rng(9))
        assert mu.std(axis=0).min() > 0
        assert np.abs(np.corrcoef(mu[:-1, 0], mu[1:, 0])[0, 1]) < 0.1

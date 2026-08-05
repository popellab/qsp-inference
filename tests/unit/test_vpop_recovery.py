"""MAP recovery, and the degeneracies PopulationPrior refuses.

The recovery checks use a small numpyro model rather than the population one, so
what is under test is the optimiser and the report, not the mechanism.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.fit import PopulationPrior
from qsp_inference.vpop.recovery import map_estimate, recovery_table

numpyro = pytest.importorskip("numpyro")
import numpyro.distributions as dist


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _model(sigma, observed=None):
    """theta ~ N(0, 1), y | theta ~ N(theta, sigma). Conjugate, so the MAP is known."""
    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
    numpyro.sample("y", dist.Normal(theta, sigma).to_event(1), obs=observed)


class TestMapEstimate:
    def test_it_finds_the_conjugate_posterior_mode(self):
        y, sigma = jnp.array([2.0, -1.0]), 0.5
        # posterior mean = y / (1 + sigma^2) for unit prior variance
        want = np.asarray(y) / (1.0 + sigma ** 2)
        hat, _ = map_estimate(_model, (sigma, y), steps=800, lr=0.05)
        assert np.allclose(np.asarray(hat["theta"]), want, atol=1e-3)

    def test_a_tight_likelihood_pulls_further_than_a_loose_one(self):
        y = jnp.array([2.0, -1.0])
        tight, _ = map_estimate(_model, (0.1, y), steps=800, lr=0.05)
        loose, _ = map_estimate(_model, (2.0, y), steps=800, lr=0.05)
        assert np.abs(tight["theta"][0]) > np.abs(loose["theta"][0])

    def test_an_uninformative_likelihood_returns_the_prior_mode(self):
        hat, _ = map_estimate(_model, (1e4, jnp.array([2.0, -1.0])),
                              steps=800, lr=0.05)
        assert np.allclose(np.asarray(hat["theta"]), 0.0, atol=1e-3)

    def test_it_starts_at_the_prior_mode_not_at_the_answer(self):
        """One step from zero must still be near zero, or the init is seeded."""
        hat, _ = map_estimate(_model, (0.5, jnp.array([2.0, -1.0])), steps=1, lr=0.05)
        assert np.all(np.abs(np.asarray(hat["theta"])) < 0.1)

    def test_init_overrides_the_start(self):
        far = {"theta": jnp.array([9.0, 9.0])}
        hat, _ = map_estimate(_model, (0.5, jnp.array([2.0, -1.0])), steps=1,
                              lr=0.05, init=far)
        assert np.all(np.asarray(hat["theta"]) > 8.0)

    def test_it_returns_the_negative_log_density(self):
        _, f = map_estimate(_model, (0.5, jnp.array([2.0, -1.0])), steps=200)
        assert np.isfinite(f) and f > 0


class TestRecoveryTable:
    def test_perfect_recovery_reads_zero(self):
        star = {"a": jnp.array([1.0, 2.0])}
        t = recovery_table(star, star)
        assert t["a"][0] == pytest.approx(0.0)
        assert t["a"][1] == pytest.approx(t["a"][2])

    def test_a_block_at_the_prior_mode_reads_as_unidentified(self):
        star = {"a": jnp.array([3.0, 4.0])}
        t = recovery_table(star, {"a": jnp.zeros(2)})
        assert t["a"][0] == pytest.approx(5.0)   # ||hat - star|| == ||star||
        assert t["a"][2] == pytest.approx(0.0)


class TestDegeneraciesRefused:
    def _kw(self, **over):
        # PopulationPrior has no defaults on purpose, so a test fixture states
        # the whole claim set the same way a project does.
        kw = dict(
            mu_0=jnp.zeros(3), L_sigma_1=jnp.eye(3), omega_0=jnp.full(3, 0.4),
            measured=(), tau_s=0.3, tau_u=0.3,
            sigma_a=0.5, sigma_b=0.5, tau_beta=0.15, n_beta=0, dim_z=1,
            log_R_0=jnp.zeros(0), sigma_R=jnp.zeros(0),
            pin_discrepancy=False, pin_u=False, pin_aux=False,
        )
        kw.update(over)
        return kw

    def test_a_single_free_beta_is_refused(self):
        """eq:betaprior centres, so |S| = 1 gives beta = 0 identically."""
        with pytest.raises(ValueError, match="beta = 0"):
            PopulationPrior(**self._kw(n_beta=1))

    def test_zero_or_two_free_betas_are_fine(self):
        for n in (0, 2):
            assert PopulationPrior(**self._kw(n_beta=n)).n_beta == n

    def test_every_width_measured_is_refused(self):
        with pytest.raises(ValueError, match="nothing to do"):
            PopulationPrior(**self._kw(measured=(0, 1, 2),
                                       tau_omega_measured=0.2))

    def test_a_measured_width_needs_its_own_prior_width(self):
        """eq:omegameas' sigma is a claim, so it cannot ride on a default."""
        with pytest.raises(ValueError, match="tau_omega_measured"):
            PopulationPrior(**self._kw(measured=(0,)))
        assert PopulationPrior(
            **self._kw(measured=(0,), tau_omega_measured=0.2)
        ).tau_omega_measured == 0.2

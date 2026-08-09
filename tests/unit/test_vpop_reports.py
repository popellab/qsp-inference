"""MAP recovery, and the degeneracies PopulationPrior refuses.

The recovery checks use a small numpyro model rather than the population one, so
what is under test is the optimiser and the report, not the mechanism.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.fit import PopulationPrior, site_spec
from qsp_inference.vpop.reports import (
    map_estimate,
    print_recovery,
    summarise_recovery,
)

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


class TestSummariseRecovery:
    def _draws(self, mean, sd, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        return mean + sd * rng.standard_normal((n, np.size(mean)))

    def test_a_tight_posterior_on_the_truth_is_identified_and_covered(self):
        rows = summarise_recovery({"a": np.array([1.0, 2.0])},
                                  {"a": self._draws(np.array([1.0, 2.0]), 0.05)},
                                  {"a": np.ones(2)})
        assert all(r.identified and r.covered for r in rows)
        assert max(abs(r.z) for r in rows) < 0.5

    def test_a_posterior_that_is_the_prior_reads_unidentified(self):
        """The truth may still be inside it; that is not evidence and says so."""
        rows = summarise_recovery({"a": np.array([0.0])},
                                  {"a": self._draws(np.array([0.0]), 1.0)},
                                  {"a": np.ones(1)})
        assert not rows[0].identified
        assert rows[0].shrink == pytest.approx(1.0, abs=0.05)

    def test_a_tight_posterior_off_the_truth_misses_it(self):
        rows = summarise_recovery({"a": np.array([1.0])},
                                  {"a": self._draws(np.array([3.0]), 0.05)},
                                  {"a": np.ones(1)})
        assert rows[0].identified and not rows[0].covered
        assert rows[0].z > 10

    def test_the_two_coverage_columns_are_reported_apart(self):
        """Pooling them lets a corpus that determines nothing score 100%."""
        rows = summarise_recovery(
            {"a": np.array([0.0, 5.0])},
            {"a": np.column_stack([self._draws(np.array([0.0]), 1.0),
                                   self._draws(np.array([9.0]), 0.05, seed=1)])},
            {"a": np.ones(2)})
        lines = print_recovery(rows)
        header = next(line for line in lines if "cover id" in line)
        row = next(line for line in lines if line.startswith("a "))
        assert header.index("cover id") < header.index("cover un")
        assert "0%" in row and "100%" in row

    def test_a_component_count_mismatch_raises(self):
        with pytest.raises(ValueError, match="components"):
            summarise_recovery({"a": np.zeros(3)}, {"a": np.zeros((10, 2))},
                               {"a": np.ones(3)})


class TestDegeneraciesRefused:
    def _kw(self, **over):
        # PopulationPrior has no defaults on purpose, so a test fixture states
        # the whole claim set the same way a project does.
        kw = dict(
            mu_0=jnp.zeros(3), L_sigma_1=jnp.eye(3), omega_0=jnp.full(3, 0.4),
            tau_s=0.3, tau_u=0.3,
            sigma_a=0.5, sigma_b=0.5, tau_beta=0.15, n_beta=0, dim_z=1,
            log_R_0=jnp.zeros(0), sigma_R=jnp.zeros(0),
            pin_discrepancy=False, pin_aux=False, pin_b_columns=(),
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

    def test_every_width_moves_together_or_by_pattern(self):
        """eq:omegameas is gone, so u covers every parameter and s is the level.

        There is no measured-width set to carve out, which is what makes the
        s / b_1 alias unconditional: a measured width was the only thing that
        would have given a readout a loading on s different from its loading
        on b_1.
        """
        prior = PopulationPrior(**self._kw())
        assert dict((n, jnp.shape(v)) for n, v, _ in site_spec(prior))["u_raw"] \
            == (prior.n_params,)


class TestPinnedComponents:
    """A pinned site is a claim the fit made, not an estimate it produced."""

    def test_a_pinned_block_is_named_rather_than_scored(self):
        rows = summarise_recovery({"a": np.zeros(2)},
                                  {"a": np.zeros((100, 2))},
                                  {"a": np.full(2, 0.5)})
        assert all(r.pinned and not r.identified for r in rows)
        assert np.isnan(rows[0].z)
        line = next(x for x in print_recovery(rows) if x.startswith("a "))
        assert "PINNED, on the truth" in line

    def test_a_pin_away_from_the_truth_reports_the_distance(self):
        rows = summarise_recovery({"x": np.array([1.0])},
                                  {"x": np.full((100, 1), 0.5)},
                                  {"x": np.array([0.25])})
        line = next(x for x in print_recovery(rows) if x.startswith("x "))
        assert "off it by up to 2.00 prior sd" in line

    def test_rounding_in_a_deterministic_site_still_reads_as_pinned(self):
        """numpyro returns the same float every draw; its sample sd is not 0."""
        draws = np.full((500, 1), 0.6931471805599453)
        rows = summarise_recovery({"x": np.array([0.9])}, {"x": draws},
                                  {"x": np.array([0.4])})
        assert rows[0].pinned
        assert np.isfinite(rows[0].bias_in_prior_sd)

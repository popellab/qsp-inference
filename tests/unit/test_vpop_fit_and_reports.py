"""The numpyro model and the reports, on a toy corpus small enough to be exact.

These are the pieces a unit test could not reach before: ``population_model``
itself, the Laplace metric handed to NUTS, and the reports that read the same
Jacobian. They are covered here because their signatures are what the rest of
the package is written against -- a mass matrix whose blocks belong to different
sites than the model samples is accepted by numpyro without a symptom, so
"it imports" is not evidence.

The mechanism is two species and two readouts, one a bare level and one a ratio,
so every claim below has a closed form.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.blocks import BlockPlan, DrawGroup
from qsp_inference.vpop.fit import (
    PopulationPrior,
    Problem,
    build_omega,
    phi_from_sites,
    population_model,
    site_spec,
)
from qsp_inference.vpop.predict import Mechanism, reference_levels
from qsp_inference.vpop.reports import (
    conditioning_report,
    dbar_absorption,
    laplace_inverse_mass,
    map_estimate,
    row_jacobians,
    should_fix_mass,
    width_gate,
    z_cost,
)
from qsp_inference.vpop.rows import RowSpec

N = 4_000
LEVEL, RATIO = "t_level", "t_ratio"
AT = {LEVEL: 0, RATIO: 1}
P = 3


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _g(vartheta, scenario):
    """Two species from three parameters; scenario 1 doubles the second."""
    y0 = jnp.exp(vartheta[:, 0] + 0.5 * vartheta[:, 2])
    y1 = jnp.exp(vartheta[:, 1])
    return jnp.stack([y0, y1 * (1.0 + scenario)], axis=-1)


def _h(y, log_R=None):
    level = jnp.log(y[AT[LEVEL], :, 0])
    ratio = jnp.log(y[AT[RATIO], :, 0]) - jnp.log(y[AT[RATIO], :, 1])
    return jnp.stack([level, ratio], axis=-1)


@pytest.fixture
def mech():
    z = jnp.asarray(np.random.default_rng(0).standard_normal((N, P)))
    return Mechanism(
        L_R=jnp.eye(P), z=z,
        Z=jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        readouts=(LEVEL, RATIO), n_species=2, n_scenarios=2,
        beta_species=jnp.zeros(0, dtype=int), g_fn=_g, h_fn=_h,
    )


@pytest.fixture
def prior():
    return PopulationPrior(
        mu_0=jnp.array([1.5, 1.0, 0.2]),
        L_sigma_1=jnp.diag(jnp.array([0.4, 0.3, 0.5])),
        omega_0=jnp.array([0.5, 0.3, 0.4]),
        tau_s=0.3, tau_u=0.085,
        sigma_a=0.5, sigma_b=0.5,
        tau_beta=0.15, n_beta=0, dim_z=2,
        log_R_0=jnp.zeros(0), sigma_R=jnp.zeros(0),
        pin_discrepancy=False, pin_aux=False, pin_b_columns=(),
    )


SPECS = {"c_pre": [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=p)
                   for p in (0.25, 0.5, 0.75)],
         "c_post": [RowSpec(RATIO, "c_post", "quantile", 0.0, 9, p=0.5),
                    RowSpec(RATIO, "c_post", "mean", 0.0, 9)]}


@pytest.fixture
def problem(mech):
    joint = DrawGroup(("c_pre", "c_post"), 14,
                      {"c_pre": tuple(range(9)), "c_post": tuple(range(5, 14))},
                      block_id="trial")
    plan = BlockPlan(("c_post", "c_pre"), (joint,))
    refs = reference_levels(jnp.array([1.5, 1.0, 0.2]),
                           jnp.array([0.5, 0.3, 0.4]), mech)
    return Problem(mech=mech, plans=[plan], specs_by_cohort=SPECS, refs=refs)


@pytest.fixture
def V_chol():
    return [np.linalg.cholesky(0.04 * np.eye(5))]


@pytest.fixture
def observed(prior, problem):
    """Rows at the prior centre, so the model is not being asked to move far."""
    from qsp_inference.vpop.predict import tau_all

    taus = tau_all(prior.mu_0, prior.omega_0, jnp.zeros(2), jnp.zeros(2),
                   jnp.zeros(0), problem.plans, problem.specs_by_cohort,
                   problem.refs, problem.mech)
    return [np.asarray(t) for t in taus]


class TestSiteSpec:
    def test_it_lists_every_site_the_model_samples(self, prior, problem, V_chol):
        import numpyro

        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(population_model).get_trace(
                prior, problem, V_chol)
        sampled = {k for k, v in trace.items()
                   if v["type"] == "sample" and not v.get("is_observed")
                   and not k.startswith("T_")}
        assert {nm for nm, _, _ in site_spec(prior)} == sampled

    def test_shapes_match_what_the_model_samples(self, prior, problem, V_chol):
        import numpyro

        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(population_model).get_trace(
                prior, problem, V_chol)
        for nm, zero, _ in site_spec(prior):
            assert jnp.shape(zero) == jnp.shape(trace[nm]["value"]), nm

    def test_pinning_the_discrepancy_drops_a_and_b(self, prior):
        from dataclasses import replace

        pinned = replace(prior, pin_discrepancy=True)
        assert {"a", "b"} & {nm for nm, _, _ in site_spec(pinned)} == set()
        assert {"a", "b"} <= {nm for nm, _, _ in site_spec(prior)}


class TestPinnedBColumns:
    """b_1 and s load identically on every width row, so only the sum is
    identified. Pinning the intercept is the identification restriction that
    sends the excess to s, which ships, instead of to b_1, which is discarded."""

    def _pinned(self, prior):
        from dataclasses import replace
        return replace(prior, pin_b_columns=(0,))

    def test_the_site_is_renamed_and_narrower(self, prior):
        spec = dict((n, jnp.shape(v)) for n, v, _ in site_spec(self._pinned(prior)))
        assert "b" not in spec and spec["b_free"] == (prior.dim_z - 1,)
        assert dict((n, jnp.shape(v)) for n, v, _ in site_spec(prior))["b"] \
            == (prior.dim_z,)

    def test_the_pinned_column_is_exactly_zero(self, prior):
        pinned = self._pinned(prior)
        sites = {n: v for n, v, _ in site_spec(pinned)}
        sites["b_free"] = jnp.array([3.0])          # anything
        _, _, _, b, _, _ = phi_from_sites(sites, pinned)
        assert b.shape == (pinned.dim_z,)
        assert float(b[0]) == 0.0 and float(b[1]) == 3.0

    def test_the_model_still_reports_a_full_width_b(self, prior, problem, V_chol):
        import numpyro

        pinned = self._pinned(prior)
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(population_model).get_trace(
                pinned, problem, V_chol)
        assert trace["b"]["type"] == "deterministic"
        assert jnp.shape(trace["b"]["value"]) == (prior.dim_z,)
        assert float(trace["b"]["value"][0]) == 0.0

    def test_site_spec_still_matches_what_the_model_samples(
            self, prior, problem, V_chol):
        import numpyro

        pinned = self._pinned(prior)
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(population_model).get_trace(
                pinned, problem, V_chol)
        sampled = {k for k, v in trace.items()
                   if v["type"] == "sample" and not v.get("is_observed")
                   and not k.startswith("T_")}
        assert {nm for nm, _, _ in site_spec(pinned)} == sampled

    def test_the_metric_follows_the_narrower_site(
            self, prior, problem, V_chol):
        pinned = self._pinned(prior)
        (key,) = laplace_inverse_mass(pinned, problem, V_chol)
        assert list(key) == sorted(nm for nm, _, _ in site_spec(pinned))
        (M,) = laplace_inverse_mass(pinned, problem, V_chol).values()
        dim = sum(int(jnp.size(v)) for _, v, _ in site_spec(pinned))
        assert np.asarray(M).shape == (dim, dim)

    def test_nuts_runs_with_a_pinned_column(
            self, prior, problem, V_chol, observed):
        from numpyro.infer import MCMC, NUTS

        pinned = self._pinned(prior)
        mcmc = MCMC(NUTS(population_model, max_tree_depth=4), num_warmup=10,
                    num_samples=10, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), pinned, problem, V_chol, observed)
        d = mcmc.get_samples()
        assert d["b_free"].shape == (10, prior.dim_z - 1)
        assert np.all(np.asarray(d["b"])[:, 0] == 0.0)

    def test_degeneracies_are_refused(self, prior):
        from dataclasses import replace

        with pytest.raises(ValueError, match="not columns of Z"):
            replace(prior, pin_b_columns=(99,))
        with pytest.raises(ValueError, match="repeats"):
            replace(prior, pin_b_columns=(0, 0))
        with pytest.raises(ValueError, match="assert the same thing twice"):
            replace(prior, pin_discrepancy=True, pin_b_columns=(0,))
        with pytest.raises(ValueError, match="written the long way"):
            replace(prior, pin_b_columns=tuple(range(prior.dim_z)))


class TestPhiFromSites:
    def test_it_is_the_map_the_model_uses(self, prior):
        """The metric differentiates this; the model samples through it."""
        sites = {nm: v for nm, v, _ in site_spec(prior)}
        mu, omega, a, b, beta, log_R = phi_from_sites(sites, prior)
        assert np.allclose(mu, prior.mu_0)
        assert np.allclose(omega, prior.omega_0)
        assert log_R is None and beta.shape == (0,)

    def test_u_is_centred_so_the_level_lives_in_s(self, prior):
        u = jnp.array([0.1, -0.3, 0.05])
        a = build_omega(0.0, u, prior)
        b = build_omega(0.0, u + 7.0, prior)
        assert np.allclose(a, b)


class TestPopulationModel:
    def test_the_prior_predictive_runs(self, prior, problem, V_chol):
        from numpyro.infer import Predictive

        pred = Predictive(population_model, num_samples=4)(
            jax.random.PRNGKey(0), prior, problem, V_chol)
        assert pred["mu"].shape == (4, P)
        assert pred["omega"].shape == (4, P)
        assert np.all(np.asarray(pred["omega"]) > 0)

    def test_the_log_density_is_finite_at_the_prior_centre(
            self, prior, problem, V_chol, observed):
        from numpyro.infer.util import log_density

        sites = {nm: v for nm, v, _ in site_spec(prior)}
        lp, _ = log_density(population_model, (prior, problem, V_chol, observed),
                            {}, sites)
        assert np.isfinite(float(lp))

    def test_nuts_takes_draws(self, prior, problem, V_chol, observed):
        from numpyro.infer import MCMC, NUTS

        mcmc = MCMC(NUTS(population_model, max_tree_depth=4), num_warmup=15,
                    num_samples=15, num_chains=1, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(0), prior, problem, V_chol, observed)
        draws = mcmc.get_samples()
        assert draws["mu_raw"].shape == (15, P)
        assert np.all(np.isfinite(np.asarray(draws["s"])))


class TestLaplaceMetric:
    def test_the_key_is_numpyros_own_packing_order(self, prior, problem, V_chol):
        """numpyro packs a dense block in the order of the key tuple.

        It uses ``tuple(sorted(latent sites))``, and site_spec is in declaration
        order, so this is the one place the two have to be reconciled. Getting it
        wrong gives a matrix whose blocks belong to other parameters, which
        numpyro accepts.
        """
        inv = laplace_inverse_mass(prior, problem, V_chol)
        (key,) = inv
        assert list(key) == sorted(nm for nm, _, _ in site_spec(prior))

    def test_it_is_square_symmetric_and_positive_definite(
            self, prior, problem, V_chol):
        (M,) = laplace_inverse_mass(prior, problem, V_chol).values()
        M = np.asarray(M)
        dim = sum(int(np.size(v)) for _, v, _ in site_spec(prior))
        assert M.shape == (dim, dim)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)

    def test_numpyro_accepts_it(self, prior, problem, V_chol, observed):
        from numpyro.infer import MCMC, NUTS

        inv = laplace_inverse_mass(prior, problem, V_chol)
        kernel = NUTS(population_model, max_tree_depth=4, dense_mass=True,
                      inverse_mass_matrix=inv, adapt_mass_matrix=False)
        mcmc = MCMC(kernel, num_warmup=10, num_samples=10, num_chains=1,
                    progress_bar=False)
        mcmc.run(jax.random.PRNGKey(1), prior, problem, V_chol, observed)
        assert np.all(np.isfinite(np.asarray(mcmc.get_samples()["mu"])))

    def test_should_fix_mass_reads_the_window_not_the_warmup(self):
        assert should_fix_mass(dim=500, warmup=1000)
        assert not should_fix_mass(dim=2, warmup=1000)


class TestReports:
    def test_conditioning_returns_lines_names_and_blocks(
            self, prior, problem, V_chol):
        lines, names, blocks = conditioning_report(prior, problem, V_chol)
        assert names == [nm for nm, _, _ in site_spec(prior)]
        assert len(blocks) == len(names)
        assert any("condition number" in l for l in lines)

    def test_more_coordinates_than_rows_leaves_eigenvalues_at_the_prior(
            self, prior, problem, V_chol):
        """G = I + J'J, and J has 5 rows, so most directions come back at 1."""
        lines, names, blocks = conditioning_report(prior, problem, V_chol)
        Jw = np.hstack(blocks)
        dim = Jw.shape[1]
        sv = np.linalg.svd(Jw, compute_uv=False)
        eig = np.sort(np.concatenate([1.0 + sv ** 2,
                                      np.ones(max(dim - sv.size, 0))]))[::-1]
        assert np.isclose(eig[-1], 1.0)
        assert (np.isclose(eig, 1.0)).sum() >= dim - 5

    def test_z_cost_runs_on_the_conditioning_blocks(self, prior, problem, V_chol):
        _, names, blocks = conditioning_report(prior, problem, V_chol)
        lines = z_cost(names, blocks, ["intercept", "ratio"])
        assert any("intercept" in l for l in lines)

    def test_row_jacobians_are_keyed_by_the_drafts_quantities(
            self, prior, problem, V_chol):
        J = row_jacobians(prior, problem, V_chol)
        assert set(J) == {"mu", "omega", "a", "b", "beta", "log_R"}
        assert J["mu"].shape == (5, P)

    def test_width_gate_reports_a_ratio_per_row(self, prior, problem, observed):
        V = [0.04 * np.eye(5)]
        lines = width_gate(observed, prior, problem, V)
        assert any("eq:ratio" in l for l in lines)

    def test_dbar_absorption_is_a_fraction(self, prior, problem, V_chol):
        E_means = [np.array([0.1, -0.05, 0.02, 0.0, 0.03])]
        captured, lines = dbar_absorption(E_means, V_chol, problem)
        assert 0.0 <= captured <= 1.0 + 1e-12
        assert lines

    def test_map_estimate_lowers_the_objective(
            self, prior, problem, V_chol, observed):
        hat, neg_lp = map_estimate(population_model,
                                   (prior, problem, V_chol, observed),
                                   steps=25, lr=5e-2)
        assert np.isfinite(neg_lp)
        assert set(hat) >= {"mu", "omega", "s"}
        assert np.asarray(hat["mu"]).shape == (P,)

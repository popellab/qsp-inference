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
import dataclasses

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
        # deliberately different widths: eq:disc's offset and its spread carry
        # their own designs, and a test that shares one cannot see them swap
        Z_a=jnp.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]),
        Z_b=jnp.array([[1.0, 0.0], [1.0, 1.0]]),
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
        tau_beta=0.15, n_beta=0, dim_a=3, dim_b=2,
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

    taus = tau_all(prior.mu_0, prior.omega_0,
                   jnp.zeros(prior.dim_a), jnp.zeros(prior.dim_b),
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
        assert "b" not in spec and spec["b_free"] == (prior.dim_b - 1,)
        assert dict((n, jnp.shape(v)) for n, v, _ in site_spec(prior))["b"] \
            == (prior.dim_b,)

    def test_the_pinned_column_is_exactly_zero(self, prior):
        pinned = self._pinned(prior)
        sites = {n: v for n, v, _ in site_spec(pinned)}
        sites["b_free"] = jnp.array([3.0])          # anything
        _, _, _, b, _, _ = phi_from_sites(sites, pinned)
        assert b.shape == (pinned.dim_b,)
        assert float(b[0]) == 0.0 and float(b[1]) == 3.0

    def test_the_model_still_reports_a_full_width_b(self, prior, problem, V_chol):
        import numpyro

        pinned = self._pinned(prior)
        with numpyro.handlers.seed(rng_seed=0):
            trace = numpyro.handlers.trace(population_model).get_trace(
                pinned, problem, V_chol)
        assert trace["b"]["type"] == "deterministic"
        assert jnp.shape(trace["b"]["value"]) == (prior.dim_b,)
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
        assert d["b_free"].shape == (10, prior.dim_b - 1)
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
            replace(prior, pin_b_columns=tuple(range(prior.dim_b)))


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


class TestHeldCentres:
    """``fix_mu``: a gate is an input to the population, not a thing inferred."""

    def _held(self, prior, j=1):
        return dataclasses.replace(prior, fix_mu=(j,))

    def test_site_is_renamed_and_shorter(self, prior):
        held = self._held(prior)
        names = dict((n, np.shape(v)) for n, v, _ in site_spec(held))
        assert "mu_raw" not in names
        assert names["mu_free"] == (P - 1,)

    def test_held_centre_comes_back_at_mu_0(self, prior):
        held = self._held(prior, j=1)
        rng = np.random.default_rng(0)
        sites = {n: jnp.asarray(rng.standard_normal(np.shape(v)))
                 for n, v, _ in site_spec(held)}
        mu = phi_from_sites(sites, held)[0]
        assert float(mu[1]) == pytest.approx(float(held.mu_0[1]), abs=1e-12)
        # and the free ones still move
        assert not np.allclose(np.asarray(mu)[[0, 2]],
                               np.asarray(held.mu_0)[[0, 2]])

    def test_free_mu_complements_fix_mu(self, prior):
        assert self._held(prior, j=1).free_mu == (0, 2)

    def test_rejects_an_index_that_is_not_a_parameter(self, prior):
        with pytest.raises(ValueError, match="not parameters"):
            dataclasses.replace(prior, fix_mu=(P,))

    def test_rejects_holding_every_centre(self, prior):
        with pytest.raises(ValueError, match="every centre"):
            dataclasses.replace(prior, fix_mu=tuple(range(P)))

    def test_rejects_a_held_centre_that_sigma_1_correlates(self, prior):
        # Dropping mu_raw[j] equals conditioning only when the row is diagonal.
        L = np.asarray(prior.L_sigma_1).copy()
        L[2, 1] = 0.5 * L[2, 2]
        with pytest.raises(ValueError, match="correlates it with others"):
            dataclasses.replace(prior, L_sigma_1=jnp.asarray(L), fix_mu=(1,))

    def test_model_samples_the_shorter_site(self, prior, problem, V_chol):
        import numpyro
        from numpyro.infer.util import initialize_model

        held = self._held(prior)
        init = initialize_model(jax.random.PRNGKey(0), population_model,
                                model_args=(held, problem, V_chol, None),
                                init_strategy=numpyro.infer.init_to_median)
        assert "mu_free" in init[0].z
        assert "mu_raw" not in init[0].z
        assert np.asarray(init[0].z["mu_free"]).shape == (P - 1,)


class TestTheRestrictedModelIsTheFullModelOnItsSpan:
    """Compose the basis with the model, rather than checking each in isolation.

    Every other mu_basis test is a property of one piece: that B is orthonormal,
    that the centre lands in the span, that truncation_V is the projected
    quadratic form. All of them can hold while the composition is wrong, which
    is how a subspace fit could report a posterior that belongs to a different
    model. The claim here is the one that ties them together: restricting the
    centre to a span must not change the posterior anywhere on that span.
    """

    def _sites(self, prior, k, seed=0):
        rng = np.random.default_rng(seed)
        return {"s": jnp.asarray(rng.standard_normal()),
                "u_raw": jnp.asarray(rng.standard_normal(3)),
                "a": jnp.asarray(rng.standard_normal(prior.dim_a)),
                "b": jnp.asarray(rng.standard_normal(prior.dim_b))}

    def _basis(self, P, k, seed=1):
        Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((P, P)))
        return Q[:, :k]

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_the_potential_agrees_with_the_full_model_on_the_span(
            self, prior, problem, V_chol, observed, k):
        from qsp_inference.vpop.reports import potential_fn

        B = self._basis(3, k)
        sub = dataclasses.replace(prior, mu_basis=B)
        args_f = (prior, problem, V_chol, observed)
        args_s = (sub, problem, V_chol, observed)
        pot_f = potential_fn(population_model, args_f)
        pot_s = potential_fn(population_model, args_s)

        rng = np.random.default_rng(7)
        for _ in range(5):
            c = rng.standard_normal(k)
            rest = self._sites(prior, k, seed=int(rng.integers(1 << 30)))
            here = float(pot_s({**rest, "mu_c": jnp.asarray(c)}))
            there = float(pot_f({**rest, "mu_raw": jnp.asarray(B @ c)}))
            # B is orthonormal, so |B c| = |c| and the quadratic part matches.
            # What is left is the normaliser of the 3 - k standard normal
            # dimensions the restricted model no longer has, which is a constant
            # in every site and cannot tilt a posterior. Asserted at its
            # predicted value rather than subtracted, so a real discrepancy
            # cannot hide inside it.
            gap = 0.5 * (3 - k) * np.log(2 * np.pi)
            assert there - here == pytest.approx(gap, rel=1e-9, abs=1e-9)

    def test_a_full_rank_rotation_is_only_a_change_of_coordinates(
            self, prior, problem, V_chol, observed):
        from qsp_inference.vpop.reports import potential_fn

        B = self._basis(3, 3, seed=4)
        sub = dataclasses.replace(prior, mu_basis=B)
        pot_f = potential_fn(population_model, (prior, problem, V_chol, observed))
        pot_s = potential_fn(population_model, (sub, problem, V_chol, observed))
        rng = np.random.default_rng(11)
        x = rng.standard_normal(3)
        rest = self._sites(prior, 3, seed=5)
        assert float(pot_s({**rest, "mu_c": jnp.asarray(B.T @ x)})) == \
            pytest.approx(float(pot_f({**rest, "mu_raw": jnp.asarray(x)})),
                          rel=1e-9, abs=1e-9)

    @pytest.mark.parametrize("k", [1, 2])
    def test_the_metric_sees_the_basis(self, prior, problem, V_chol, k):
        # The mass matrix is built from these blocks, so if the basis reaches the
        # model but not the metric the sampler gets a metric for another model.
        from qsp_inference.vpop.reports import laplace_blocks

        B = self._basis(3, k)
        sub = dataclasses.replace(prior, mu_basis=B)
        nf, _, _, bf, _ = laplace_blocks(prior, problem, V_chol)
        ns, _, _, bs, _ = laplace_blocks(sub, problem, V_chol)
        A = np.asarray(bf[nf.index("mu_raw")])
        Ab = np.asarray(bs[ns.index("mu_c")])
        assert Ab == pytest.approx(A @ B, rel=1e-7, abs=1e-9)

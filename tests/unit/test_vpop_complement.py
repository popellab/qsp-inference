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


class TestAuxiliarySupport:
    """eq:auxprior is truncated where the observation operator bounds it."""

    def _p(self, low):
        import jax.numpy as jnp
        return _prior(P=4).__class__(
            **{**{f.name: getattr(_prior(P=4), f.name)
                  for f in __import__("dataclasses").fields(_prior(P=4))},
               "log_R_0": jnp.asarray([2.3026]),
               "sigma_R": jnp.asarray([1.2]),
               "log_R_low": low})

    def test_unbounded_is_a_plain_normal(self):
        from qsp_inference.vpop.fit import aux_distribution
        import numpyro.distributions as dist
        d = aux_distribution(self._p(None))
        assert isinstance(d.base_dist, dist.Normal)

    def test_bounded_declares_the_support_and_never_leaves_it(self):
        import jax, jax.numpy as jnp
        import numpy as np
        from numpyro.distributions.transforms import biject_to
        from qsp_inference.vpop.fit import aux_distribution
        d = aux_distribution(self._p(jnp.zeros(1)))
        # The bound binds through the bijector NUTS samples in, not through a
        # -inf density: numpyro does not mask outside the support unless
        # validation is on, and a wall is what apply_margins argues against.
        assert np.asarray(d.support.base_constraint.lower_bound).ravel()[0] == 0.0
        x = d.sample(jax.random.PRNGKey(0), (4000,))
        assert float(np.asarray(x).min()) >= 0.0
        # and every real number maps to a point inside it
        t = biject_to(d.support)
        for u in (-40.0, -3.0, 0.0, 3.0, 40.0):
            assert float(np.asarray(t(jnp.asarray([u])))[0]) >= 0.0

    def test_site_spec_starts_inside_the_support(self):
        import jax.numpy as jnp
        import numpy as np
        from qsp_inference.vpop.fit import site_spec
        # a centre below the bound would put the initial point outside it
        p = self._p(jnp.asarray([3.0]))
        start = dict((n, v) for n, v, _ in site_spec(p))["log_R"]
        assert float(np.asarray(start)[0]) >= 3.0

    def test_the_bound_is_reached_by_softplus_not_exp(self):
        import jax.numpy as jnp
        import numpy as np
        from numpyro.distributions.transforms import biject_to
        from qsp_inference.vpop.fit import aux_distribution
        # log_R is already a log, so an exp bijector would make R = exp(exp(u)).
        # At u = 10 that is 2.2e4 rather than 10, which is the whole point.
        t = biject_to(aux_distribution(self._p(jnp.zeros(1))).support)
        for u in (10.0, 20.0):
            assert float(np.asarray(t(jnp.asarray([u])))[0]) == \
                pytest.approx(u, abs=1e-2)
        assert float(np.asarray(t(jnp.asarray([5.0])))[0]) == \
            pytest.approx(5.0, abs=1e-2)
        assert float(np.asarray(t(jnp.asarray([-30.0])))[0]) >= 0.0

    def test_softplus_leaves_the_density_alone(self):
        import jax, jax.numpy as jnp
        import numpy as np
        import numpyro.distributions as dist
        from qsp_inference.vpop.fit import aux_distribution
        # Only the coordinate changes. The prior is the same distribution, so a
        # fit is comparable across the change and eq:auxprior still reads true.
        d = aux_distribution(self._p(jnp.zeros(1)))
        ref = dist.TruncatedNormal(jnp.asarray([2.3026]), jnp.asarray([1.2]),
                                   low=jnp.zeros(1)).to_event(1)
        xs = jnp.asarray([[0.05], [1.0], [2.3026], [6.0]])
        assert np.asarray(jax.vmap(d.log_prob)(xs)) == \
            pytest.approx(np.asarray(jax.vmap(ref.log_prob)(xs)), rel=1e-6)


class TestConstrainedAndUnconstrainedDoNotGetSwapped:
    """A bounded site makes the two spaces differ, which no normal site does.

    Every latent in the vpop model is normal, so the spaces coincide and code
    that confuses them is correct by accident. These pin the contract against a
    model that has one bounded site, which is the only way the class of bug is
    visible at all.
    """

    def _model(self):
        import jax.numpy as jnp
        import numpyro
        import numpyro.distributions as dist
        from qsp_inference.vpop.fit import aux_distribution
        p = TestAuxiliarySupport()._p(jnp.zeros(1))

        def model():
            numpyro.sample("log_R", aux_distribution(p))
            numpyro.sample("z", dist.Normal(0.0, 1.0).expand([2]).to_event(1))
        return model

    def test_init_is_read_as_a_value_not_as_a_coordinate(self):
        import jax.numpy as jnp
        import numpy as np
        import jax
        from numpyro.infer.util import initialize_model
        from qsp_inference.vpop.reports import map_estimate
        model = self._model()
        init = {"log_R": jnp.asarray([2.3026]), "z": jnp.zeros(2)}
        # lr 0 freezes the point, so the objective returned is the potential at
        # exactly the init. It must agree with the potential at the transformed
        # init, not at the init read straight off as a coordinate.
        _, f = map_estimate(model, (), steps=1, lr=0.0, init=init)
        info = initialize_model(jax.random.PRNGKey(0), model, model_args=())
        raw = float(info.potential_fn({k: jnp.asarray(v)
                                       for k, v in init.items()}))
        assert f != pytest.approx(raw, abs=1e-3)      # the bug this replaces
        from numpyro.infer.util import unconstrain_fn
        z = unconstrain_fn(model, (), {}, init)
        assert f == pytest.approx(float(info.potential_fn(z)), abs=1e-3)

    def test_unconstrained_output_round_trips_to_the_constrained_one(self):
        import jax.numpy as jnp
        import numpy as np
        import jax
        from numpyro.infer.util import initialize_model
        from qsp_inference.vpop.reports import map_estimate
        model = self._model()
        con, f = map_estimate(model, (), steps=40, lr=0.05)
        unc, g = map_estimate(model, (), steps=40, lr=0.05, unconstrained=True)
        assert f == pytest.approx(g)
        info = initialize_model(jax.random.PRNGKey(0), model, model_args=())
        # the unconstrained return is what potential_fn accepts, and pushing it
        # forward gives the constrained return
        assert float(info.potential_fn(unc)) == pytest.approx(f, abs=1e-4)
        fwd = info.postprocess_fn(unc)
        assert np.asarray(fwd["log_R"]) == \
            pytest.approx(np.asarray(con["log_R"]), rel=1e-5)
        assert float(np.asarray(unc["log_R"])[0]) != \
            pytest.approx(float(np.asarray(con["log_R"])[0]), abs=1e-3)

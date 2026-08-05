"""Unit tests for tau_B (qsp_inference.vpop.predict).

The toy mechanism is built so each claim is exact rather than statistical: one
readout is a bare species and the other a ratio, so beta's propagation through
h_r has a closed form to check against.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.predict import (
    Mechanism,
    apply_map,
    block_weights,
    cohort_cloud,
    cohort_columns,
    patient_cloud,
    quantile_mass_table,
    readout_cloud,
    reference_levels,
    tau_all,
    tau_rows,
)
from qsp_inference.vpop.resampling import BlockPlan, DrawGroup
from qsp_inference.vpop.rows import RowSpec, tau_row
from qsp_inference.vpop.statistics import bootstrap_design

N = 40_000
LEVEL, RATIO = "t_level", "t_ratio"

#: Which scenario each readout is reported at. eq:readout gives one value per
#: patient per readout, so h_r selects a slice rather than returning every one.
AT = {LEVEL: 0, RATIO: 1}


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _g(vartheta, scenario):
    """Two positive species; scenario 1 doubles the second."""
    return jnp.exp(vartheta) * jnp.array([1.0, 1.0 + scenario])


def _h(y, log_R=None):
    """``(N, 2)``: a level in species 0, then a ratio of the two.

    ``log_R`` scales the level, standing in for a declared assay conversion.
    """
    level = jnp.log(y[AT[LEVEL], :, 0])
    ratio = jnp.log(y[AT[RATIO], :, 0]) - jnp.log(y[AT[RATIO], :, 1])
    if log_R is not None:
        level = level + jnp.asarray(log_R)[0]
    return jnp.stack([level, ratio], axis=-1)


@pytest.fixture
def mech():
    z = jnp.asarray(np.random.default_rng(0).standard_normal((N, 2)))
    return Mechanism(
        L_R=jnp.eye(2), z=z,
        Z=jnp.array([[1.0, 0.0], [1.0, 1.0]]),
        readouts=(LEVEL, RATIO), n_species=2, n_scenarios=2,
        beta_species=jnp.array([0, 1]), g_fn=_g, h_fn=_h,
    )


#: The level sits well away from zero, so pivoting at c_r and pivoting at zero
#: give different answers and the kappa tests can tell them apart.
MU = jnp.array([1.5, 1.0])
OMEGA = jnp.array([0.5, 0.3])
ZERO2 = jnp.zeros(2)

SPECS = {"c_pre": [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=p)
                   for p in (0.25, 0.5, 0.75)],
         "c_post": [RowSpec(RATIO, "c_post", "quantile", 0.0, 9, p=0.5),
                    RowSpec(RATIO, "c_post", "mean", 0.0, 9)]}


@pytest.fixture
def plan():
    joint = DrawGroup(("c_pre", "c_post"), 14,
                      {"c_pre": tuple(range(9)), "c_post": tuple(range(5, 14))},
                      block_id="trial")
    return BlockPlan(("c_post", "c_pre"), (joint,))


class TestPatientCloud:
    def test_recovers_mu_and_omega(self, mech):
        v = patient_cloud(MU, OMEGA, mech)
        assert np.allclose(v.mean(0), MU, atol=0.02)
        assert np.allclose(v.std(0), OMEGA, rtol=0.02)

    def test_correlation_comes_from_L_R(self):
        rho = 0.6
        L = jnp.linalg.cholesky(jnp.array([[1.0, rho], [rho, 1.0]]))
        z = jnp.asarray(np.random.default_rng(1).standard_normal((N, 2)))
        m = Mechanism(L_R=L, z=z, Z=jnp.eye(2), readouts=(LEVEL, RATIO),
                      n_species=2, n_scenarios=2, beta_species=jnp.array([0]),
                      g_fn=_g, h_fn=_h)
        v = patient_cloud(MU, jnp.ones(2), m)
        assert np.corrcoef(np.asarray(v).T)[0, 1] == pytest.approx(rho, abs=0.02)

    def test_frozen_z_makes_it_a_deterministic_function_of_phi(self, mech):
        assert jnp.array_equal(patient_cloud(MU, OMEGA, mech),
                               patient_cloud(MU, OMEGA, mech))


class TestReadoutCloud:
    def test_one_column_per_readout_not_per_scenario(self, mech):
        """eq:readout is one value per patient per readout."""
        assert readout_cloud(MU, OMEGA, ZERO2, mech).shape == (N, 2)

    def test_beta_propagates_through_h_r(self, mech):
        """A level inherits beta_0; a ratio inherits beta_0 - beta_1."""
        base = readout_cloud(MU, OMEGA, ZERO2, mech)
        moved = readout_cloud(MU, OMEGA, jnp.array([0.4, -0.1]), mech)
        shift = np.asarray((moved - base).mean(axis=0))
        assert shift[0] == pytest.approx(0.4)
        assert shift[1] == pytest.approx(0.5)

    def test_each_readout_reads_its_own_scenario(self, mech):
        """The ratio sits at scenario 1, where species 1 is doubled."""
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        v = patient_cloud(MU, OMEGA, mech)
        assert float(x[:, 1].mean()) == pytest.approx(
            float((v[:, 0] - v[:, 1]).mean()) - np.log(2), abs=1e-9)

    def test_log_R_moves_only_the_readout_that_declares_it(self, mech):
        base = readout_cloud(MU, OMEGA, ZERO2, mech)
        moved = readout_cloud(MU, OMEGA, ZERO2, mech, jnp.array([np.log(3.0)]))
        assert np.allclose(moved[:, 0] - base[:, 0], np.log(3.0))
        assert np.allclose(moved[:, 1], base[:, 1])


class TestApplyMap:
    Z = jnp.array([[1.0, 0.0], [1.0, 1.0]])

    def test_no_discrepancy_is_the_identity(self):
        x = jnp.asarray(np.random.default_rng(2).standard_normal((50, 2)))
        for c in (jnp.zeros(2), jnp.array([3.0, -7.0])):
            assert np.allclose(apply_map(x, ZERO2, ZERO2, c, self.Z), x)

    def test_kappa_leaves_the_pivot_fixed(self):
        c = jnp.array([2.0, -1.0])
        x = jnp.broadcast_to(c, (5, 2))
        assert np.allclose(apply_map(x, ZERO2, jnp.array([0.3, 0.2]), c, self.Z), x)

    def test_kappa_scales_deviations_from_the_pivot(self):
        c = jnp.array([2.0, -1.0])
        out = apply_map(c + jnp.array([[1.0, 1.0]]), ZERO2,
                        jnp.array([np.log(2.0), 0.0]), c, self.Z)
        assert np.allclose(out - c, [[2.0, 2.0]])

    def test_gamma_is_Z_a_and_shifts(self):
        out = apply_map(jnp.zeros((3, 2)), jnp.array([0.5, 0.25]), ZERO2,
                        jnp.zeros(2), self.Z)
        assert np.allclose(out, [[0.5, 0.75]] * 3)

    def test_the_map_preserves_order(self):
        """Why the sort may be hoisted above eq:disc: kappa > 0."""
        x = jnp.asarray(np.random.default_rng(3).standard_normal((200, 2)))
        a, b, c = jnp.array([0.3, -0.2]), jnp.array([0.4, 0.1]), jnp.array([1.0, -0.5])
        assert jnp.array_equal(jnp.sort(apply_map(x, a, b, c, self.Z), axis=0),
                               apply_map(jnp.sort(x, axis=0), a, b, c, self.Z))


class TestReferenceLevels:
    def test_one_level_per_readout(self, mech):
        assert reference_levels(MU, OMEGA, mech).shape == (2,)

    def test_the_pivot_is_the_population_median_at_the_plug_in(self, mech):
        refs = reference_levels(MU, OMEGA, mech)
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        assert np.allclose(refs, np.median(np.asarray(x), axis=0), atol=1e-3)

    def test_log_R_enters_the_pivot_at_its_prior_centre(self, mech):
        """R is a declared conversion, so the plug-in is R_0 and not one."""
        base = reference_levels(MU, OMEGA, mech)
        moved = reference_levels(MU, OMEGA, mech, log_R_0=jnp.array([np.log(4.0)]))
        assert float(moved[0] - base[0]) == pytest.approx(np.log(4.0), abs=1e-9)


class TestCohortColumns:
    def test_each_cohort_keeps_only_what_it_reports(self, mech):
        assert cohort_columns(SPECS, mech.readouts) == {"c_pre": (0,), "c_post": (1,)}

    def test_a_readout_no_cohort_reports_is_absent(self):
        cols = cohort_columns(SPECS, ("dead_a", LEVEL, "dead_b", RATIO))
        assert cols == {"c_pre": (1,), "c_post": (3,)}

    def test_the_cut_cloud_matches_the_full_map_on_those_columns(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        refs = reference_levels(MU, OMEGA, mech)
        a, b = jnp.array([0.2, -0.1]), jnp.array([0.3, 0.15])
        full = apply_map(x, a, b, refs, mech.Z)
        cut = cohort_cloud(x, (1,), a, b, refs, mech)
        assert np.allclose(cut[:, 0], full[:, 1], rtol=1e-12)


class TestBlockWeights:
    def _x_of(self, plan):
        return {c: jnp.asarray(np.random.default_rng(i).standard_normal((100, 2)))
                for i, c in enumerate(plan.cohort_ids)}

    def _elig(self, seen):
        def f(x, cohort):
            seen.append(cohort)
            return jax.nn.sigmoid(x[:, 0])
        return f

    def test_no_criterion_gives_unit_weights(self, plan):
        assert all(np.allclose(v, 1.0)
                   for v in block_weights(self._x_of(plan), plan).values())

    def test_a_joint_group_without_a_nomination_raises(self, plan):
        with pytest.raises(ValueError, match="share a draw"):
            block_weights(self._x_of(plan), plan, self._elig([]))

    def test_a_nomination_outside_the_group_raises(self, plan):
        with pytest.raises(ValueError, match="not a member"):
            block_weights(self._x_of(plan), plan, self._elig([]),
                          {"trial": "somewhere_else"})

    def test_a_joint_group_shares_one_vector_read_off_the_nominee(self, plan):
        seen = []
        w = block_weights(self._x_of(plan), plan, self._elig(seen), {"trial": "c_pre"})
        assert seen == ["c_pre"] and w["c_pre"] is w["c_post"]

    def test_a_lone_cohort_reads_itself(self):
        lone = BlockPlan(("solo",),
                         (DrawGroup(("solo",), 7, {"solo": tuple(range(7))}),))
        seen = []
        w = block_weights(self._x_of(lone), lone, self._elig(seen))
        assert seen == ["solo"] and w["solo"].shape == (100,)


class TestTauRows:
    ROWS = [RowSpec(LEVEL, "c", "quantile", 0.0, 9, p=0.5),
            RowSpec(LEVEL, "c", "quantile", 0.0, 9, p=0.25),
            RowSpec(LEVEL, "c", "mean", 0.0, 9)]

    def _column(self, mech):
        return readout_cloud(MU, OMEGA, ZERO2, mech)[:, :1]

    def test_rows_come_back_in_the_order_the_source_printed_them(self, mech):
        col = self._column(mech)
        out = tau_rows(self.ROWS, col, jnp.ones(N), mech, column_of={LEVEL: 0})
        assert out.shape == (3,)
        assert float(out[2]) == pytest.approx(float(col[:, 0].mean()), abs=1e-6)
        assert float(out[0]) > float(out[1])          # median above lower quartile

    def test_a_moment_row_needs_its_design(self, mech):
        spec = RowSpec(LEVEL, "c", "sd", 0.0, 9, log=True)
        with pytest.raises(ValueError, match="bootstrap design"):
            tau_rows([spec], self._column(mech), jnp.ones(N), mech,
                     column_of={LEVEL: 0})
        out = tau_rows([spec], self._column(mech), jnp.ones(N), mech,
                       {spec.label: bootstrap_design(jax.random.PRNGKey(0), 9, 200)},
                       column_of={LEVEL: 0})
        assert float(out[0]) < float(jnp.log(OMEGA[0]))   # E[log s] below log sigma

    def test_the_batched_sort_matches_a_per_readout_sort(self, mech):
        got = tau_rows(self.ROWS, self._column(mech), jnp.ones(N), mech,
                       column_of={LEVEL: 0})
        col = jnp.sort(self._column(mech)[:, 0])
        for k, spec in enumerate(self.ROWS):
            assert float(got[k]) == pytest.approx(
                float(tau_row(spec, col, jnp.ones(N))), rel=1e-12)


class TestSharedQuantileMass:
    """Rows sharing (p, n, convention) share one Beta mass when w is uniform."""

    ROWS = [RowSpec(LEVEL, "c", "quantile", 0.0, 9, p=p) for p in (0.25, 0.5, 0.75)]
    ROWS += [RowSpec(LEVEL, "c", "mean", 0.0, 9)]

    def test_sharing_the_mass_changes_no_number(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[:, :1]
        args = (self.ROWS, x, jnp.ones(N), mech)
        assert np.array_equal(tau_rows(*args, uniform=True, column_of={LEVEL: 0}),
                              tau_rows(*args, uniform=False, column_of={LEVEL: 0}))

    def test_a_non_uniform_weight_is_not_shared(self, mech):
        """Two readouts sort w into different orders, so the mass genuinely differs.

        One readout could not show this: with a single column there is only one
        ordering and the two paths agree whatever ``w`` is.
        """
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        rows = [RowSpec(r, "c", "quantile", 0.0, 9, p=0.5) for r in (LEVEL, RATIO)]
        w = jax.nn.sigmoid(x[:, 0] - float(MU[0]))
        args = (rows, x, w, mech)
        cols = {LEVEL: 0, RATIO: 1}
        assert not np.allclose(tau_rows(*args, uniform=True, column_of=cols),
                               tau_rows(*args, uniform=False, column_of=cols))

    def test_the_table_covers_only_quantile_rows(self):
        assert set(quantile_mass_table(SPECS, 1000)) == {
            (0.25, 9, "type7"), (0.5, 9, "type7"), (0.75, 9, "type7")}


class TestTauAll:
    def test_cohorts_concatenate_in_plan_order(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech)
        out = tau_all(MU, OMEGA, ZERO2, ZERO2, ZERO2, [plan], SPECS, refs, mech)[0]
        assert plan.cohort_ids == ("c_post", "c_pre")
        assert out.shape == (5,)      # c_post's 2 rows, then c_pre's 3
        # exp, because a row functional runs in the units the source printed and
        # h_r returns logs. A mean does not commute with it, so this row is the
        # mean of the exponentiated cloud and not the exponentiated mean.
        x = jnp.sort(readout_cloud(MU, OMEGA, ZERO2, mech), axis=0)
        assert float(out[1]) == pytest.approx(float(jnp.exp(x[:, 1]).mean()),
                                              rel=1e-9)

    def test_gamma_scales_a_location_row_by_exp_Z_a(self, mech, plan):
        args = ([plan], SPECS, reference_levels(MU, OMEGA, mech), mech)
        base = tau_all(MU, OMEGA, ZERO2, ZERO2, ZERO2, *args)[0]
        moved = tau_all(MU, OMEGA, jnp.array([0.3, 0.2]), ZERO2, ZERO2, *args)[0]
        # Z rows are [1,0] for the level and [1,1] for the ratio. gamma is
        # additive on the log readout, so it is a constant factor on the row,
        # which is what makes it a multiplicative assay bias.
        assert np.allclose(moved / base, np.exp([0.5, 0.5, 0.3, 0.3, 0.3]),
                           rtol=1e-9)

    @pytest.mark.parametrize("n, tol", [(9, 0.05), (201, 3e-3), (4001, 5e-4)])
    def test_kappa_leaves_a_median_row_at_the_pivot(self, mech, plan, n, tol):
        """Exact in the population, and approached as ``n`` grows.

        eq:disc is affine on the log readout, so it moves the population median
        by ``kappa (med - c) + c``, which is ``c`` exactly. A reported median is
        the expectation over ``n`` draws, and the row is computed in the source's
        units, where the map is a power rather than affine -- so the expectation
        and the map stop commuting and Jensen moves the row up. The gap is the
        median's own sampling spread, and it closes at ``sqrt(n)``.
        """
        specs = {"c_pre": [RowSpec(LEVEL, "c_pre", "quantile", 0.0, n, p=0.5)],
                 "c_post": [RowSpec(RATIO, "c_post", "quantile", 0.0, n, p=0.5)]}
        args = ([plan], specs, reference_levels(MU, OMEGA, mech), mech)
        base = tau_all(MU, OMEGA, ZERO2, ZERO2, ZERO2, *args)[0]
        moved = tau_all(MU, OMEGA, ZERO2, jnp.array([0.5, 0.0]), ZERO2, *args)[0]
        assert float(moved[1]) == pytest.approx(float(base[1]), rel=tol)

    def test_a_precomputed_mass_table_changes_no_number(self, mech, plan):
        args = (MU, OMEGA, ZERO2, ZERO2, ZERO2, [plan], SPECS,
                reference_levels(MU, OMEGA, mech), mech)
        assert np.array_equal(tau_all(*args)[0],
                              tau_all(*args,
                                      mass_table=quantile_mass_table(SPECS, N))[0])

    def test_eligibility_takes_the_unsorted_path_to_the_same_answer(self, mech, plan):
        args = (MU, OMEGA, ZERO2, ZERO2, ZERO2, [plan], SPECS,
                reference_levels(MU, OMEGA, mech), mech)
        assert np.allclose(
            tau_all(*args)[0],
            tau_all(*args, elig_fn=lambda x, c: jnp.ones(x.shape[0]),
                    elig_at={"trial": "c_pre"})[0], rtol=1e-10)

    def test_differentiable_in_every_parameter(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech)
        specs = {k: list(v) for k, v in SPECS.items()}
        sd = RowSpec(LEVEL, "c_pre", "sd", 0.0, 9, log=True)
        specs["c_pre"].append(sd)
        designs = {sd.label: bootstrap_design(jax.random.PRNGKey(0), 9, 100)}

        def total(mu, omega, a, b, beta, log_R):
            return jnp.sum(tau_all(mu, omega, a, b, beta, [plan], specs, refs, mech,
                                   log_R=log_R, designs=designs)[0])

        grads = jax.grad(total, argnums=(0, 1, 2, 3, 4, 5))(
            MU, OMEGA, ZERO2, ZERO2, ZERO2, jnp.array([0.0]))
        assert all(np.all(np.isfinite(g)) for g in grads)
        assert all(np.any(np.abs(np.asarray(g)) > 1e-6) for g in grads)

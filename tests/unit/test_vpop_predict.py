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
    patient_cloud,
    readout_cloud,
    reference_levels,
    tau_all,
    tau_block,
    tau_rows,
    quantile_mass_table,
    scenario_columns,
)
from qsp_inference.vpop.resampling import BlockPlan, DrawGroup
from qsp_inference.vpop.rows import RowSpec, tau_row
from qsp_inference.vpop.statistics import bootstrap_design

N = 40_000
LEVEL, RATIO = "t_level", "t_ratio"


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


def _g(vartheta, scenario):
    """Two positive species; scenario 1 doubles the second."""
    return jnp.exp(vartheta) * jnp.array([1.0, 1.0 + scenario])


def _h(y, log_R=None):
    """Readouts in sorted order: a level in species 0, then a ratio of the two.

    ``log_R`` scales the level, standing in for a declared assay conversion.
    """
    level = jnp.log(y[:, :, 0])
    ratio = jnp.log(y[:, :, 0]) - jnp.log(y[:, :, 1])
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


#: The level sits well away from zero, so pivoting at c_rc and pivoting at zero
#: give different answers and the kappa tests can tell them apart.
MU = jnp.array([1.5, 1.0])
OMEGA = jnp.array([0.5, 0.3])
ZERO2 = jnp.zeros(2)
SCENARIO_OF = {"c_pre": 0, "c_post": 1}


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
        a = patient_cloud(MU, OMEGA, mech)
        b = patient_cloud(MU, OMEGA, mech)
        assert jnp.array_equal(a, b)


class TestReadoutCloud:
    def test_beta_propagates_through_h_r(self, mech):
        """A level inherits beta_0; a ratio inherits beta_0 - beta_1."""
        base = readout_cloud(MU, OMEGA, ZERO2, mech)
        beta = jnp.array([0.4, -0.1])
        moved = readout_cloud(MU, OMEGA, beta, mech)
        shift = np.asarray((moved - base).mean(axis=1))
        assert np.allclose(shift[:, 0], 0.4)
        assert np.allclose(shift[:, 1], 0.5)

    def test_every_scenario_is_returned(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        assert x.shape == (2, N, 2)
        # scenario 1 doubles species 1, so only the ratio moves, by log 2.
        assert float((x[0, :, 0] - x[1, :, 0]).mean()) == pytest.approx(0.0)
        assert float((x[1, :, 1] - x[0, :, 1]).mean()) == pytest.approx(-np.log(2))


class TestApplyMap:
    Z = jnp.array([[1.0, 0.0], [1.0, 1.0]])

    def test_no_discrepancy_is_the_identity(self):
        x = jnp.asarray(np.random.default_rng(2).standard_normal((50, 2)))
        for c in (jnp.zeros(2), jnp.array([3.0, -7.0])):
            assert np.allclose(apply_map(x, ZERO2, ZERO2, c, self.Z), x)

    def test_kappa_leaves_the_pivot_fixed(self):
        c = jnp.array([2.0, -1.0])
        x = jnp.broadcast_to(c, (5, 2))
        out = apply_map(x, ZERO2, jnp.array([0.3, 0.2]), c, self.Z)
        assert np.allclose(out, x)

    def test_kappa_scales_deviations_from_the_pivot(self):
        c = jnp.array([2.0, -1.0])
        x = c + jnp.array([[1.0, 1.0]])
        b = jnp.array([np.log(2.0), 0.0])
        out = apply_map(x, ZERO2, b, c, self.Z)
        assert np.allclose(out - c, [[2.0, 2.0]])

    def test_gamma_is_Z_a_and_shifts(self):
        x = jnp.zeros((3, 2))
        out = apply_map(x, jnp.array([0.5, 0.25]), ZERO2, jnp.zeros(2), self.Z)
        assert np.allclose(out, [[0.5, 0.75]] * 3)


class TestReferenceLevels:
    def test_pivot_is_the_cohort_median_at_the_plug_in(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        for cohort, s in SCENARIO_OF.items():
            want = np.median(np.asarray(x[s]), axis=0)
            assert np.allclose(refs[cohort], want, atol=1e-3)

    def test_cohorts_at_different_scenarios_get_different_pivots(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        assert float(refs["c_pre"][1] - refs["c_post"][1]) == pytest.approx(
            np.log(2), abs=1e-3)

    def test_every_cohort_of_every_plan_gets_one(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        assert set(refs) == set(plan.cohort_ids)
        assert all(v.shape == (2,) for v in refs.values())


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
        w = block_weights(self._x_of(plan), plan)
        assert all(np.allclose(v, 1.0) for v in w.values())

    def test_a_joint_group_without_a_nomination_raises(self, plan):
        with pytest.raises(ValueError, match="share a draw"):
            block_weights(self._x_of(plan), plan, self._elig([]))

    def test_a_nomination_outside_the_group_raises(self, plan):
        with pytest.raises(ValueError, match="not a member"):
            block_weights(self._x_of(plan), plan, self._elig([]),
                          {"trial": "somewhere_else"})

    def test_a_joint_group_shares_one_vector_read_off_the_nominee(self, plan):
        seen = []
        w = block_weights(self._x_of(plan), plan, self._elig(seen),
                          {"trial": "c_pre"})
        assert seen == ["c_pre"]
        assert w["c_pre"] is w["c_post"]

    def test_a_lone_cohort_reads_itself(self):
        lone = BlockPlan(("solo",), (DrawGroup(("solo",), 7, {"solo": tuple(range(7))}),))
        seen = []
        w = block_weights(self._x_of(lone), lone, self._elig(seen))
        assert seen == ["solo"] and w["solo"].shape == (100,)


class TestTauRows:
    def _specs(self):
        return [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=0.5),
                RowSpec(RATIO, "c_pre", "quantile", 0.0, 9, p=0.25),
                RowSpec(LEVEL, "c_pre", "mean", 0.0, 9)]

    def test_rows_come_back_in_the_order_the_source_printed_them(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        w = jnp.ones(N)
        out = tau_rows(self._specs(), x, w, mech)
        assert out.shape == (3,)
        # row 2 is the mean of the level column, exactly.
        assert float(out[2]) == pytest.approx(float(x[:, 0].mean()), abs=1e-6)

    def test_each_row_reads_its_own_readout_column(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        out = tau_rows(self._specs(), x, jnp.ones(N), mech)
        # the level median sits near mu_0; the ratio quartile does not.
        assert float(out[0]) == pytest.approx(float(MU[0]), abs=0.02)
        assert abs(float(out[1]) - float(MU[0])) > 0.5

    def test_a_moment_row_needs_its_design(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        spec = RowSpec(LEVEL, "c_pre", "sd", 0.0, 9, log=True)
        with pytest.raises(ValueError, match="bootstrap design"):
            tau_rows([spec], x, jnp.ones(N), mech)
        designs = {spec.label: bootstrap_design(jax.random.PRNGKey(0), 9, 200)}
        out = tau_rows([spec], x, jnp.ones(N), mech, designs)
        # E[log s] at n=9 sits below log sigma, and sigma here is omega_0.
        assert float(out[0]) < float(jnp.log(OMEGA[0]))


class TestTauBlock:
    SPECS = {
        "c_pre": [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=0.5),
                  RowSpec(RATIO, "c_pre", "mean", 0.0, 9)],
        "c_post": [RowSpec(LEVEL, "c_post", "mean", 0.0, 9)],
    }

    def test_cohorts_concatenate_in_plan_order(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        out = tau_block(x, plan, self.SPECS, ZERO2, ZERO2, refs, mech, SCENARIO_OF)
        assert plan.cohort_ids == ("c_post", "c_pre")
        assert out.shape == (3,)
        want = float(cohort_cloud(x, "c_post", ZERO2, ZERO2, refs, mech,
                                  SCENARIO_OF)[:, 0].mean())
        assert float(out[0]) == pytest.approx(want, abs=1e-6)

    def test_gamma_shifts_a_location_row_by_Z_a(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        base = tau_block(x, plan, self.SPECS, ZERO2, ZERO2, refs, mech, SCENARIO_OF)
        moved = tau_block(x, plan, self.SPECS, jnp.array([0.3, 0.2]), ZERO2, refs,
                          mech, SCENARIO_OF)
        # Z rows are [1,0] for the level and [1,1] for the ratio.
        assert np.allclose(moved - base, [0.3, 0.3, 0.5], atol=1e-6)

    def test_kappa_leaves_a_median_row_at_the_pivot(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        base = tau_block(x, plan, self.SPECS, ZERO2, ZERO2, refs, mech, SCENARIO_OF)
        moved = tau_block(x, plan, self.SPECS, ZERO2, jnp.array([0.5, 0.0]), refs,
                          mech, SCENARIO_OF)
        assert float(moved[1]) == pytest.approx(float(base[1]), abs=2e-3)


class TestTauAll:
    SPECS = TestTauBlock.SPECS

    def test_one_vector_per_plan(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        out = tau_all(MU, OMEGA, ZERO2, ZERO2, ZERO2, [plan], self.SPECS, refs,
                      mech, SCENARIO_OF)
        assert len(out) == 1 and out[0].shape == (3,)

    def test_differentiable_in_every_parameter(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        specs = dict(self.SPECS)
        sd = RowSpec(LEVEL, "c_pre", "sd", 0.0, 9, log=True)
        specs["c_pre"] = specs["c_pre"] + [sd]
        designs = {sd.label: bootstrap_design(jax.random.PRNGKey(0), 9, 100)}

        def total(mu, omega, a, b, beta):
            return jnp.sum(tau_all(mu, omega, a, b, beta, [plan], specs, refs, mech,
                                   SCENARIO_OF, designs=designs)[0])

        grads = jax.grad(total, argnums=(0, 1, 2, 3, 4))(MU, OMEGA, ZERO2, ZERO2, ZERO2)
        assert all(np.all(np.isfinite(g)) for g in grads)
        # omega and b move the sd row; mu, a and beta move the location rows.
        assert all(np.any(np.abs(np.asarray(g)) > 1e-6) for g in grads)


class TestSharedQuantileMass:
    """Rows sharing (p, n, convention) share one Beta mass when w is uniform."""

    SPECS = [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=p) for p in (0.25, 0.5, 0.75)]
    SPECS += [RowSpec(RATIO, "c_pre", "quantile", 0.0, 9, p=p) for p in (0.25, 0.75)]
    SPECS += [RowSpec(LEVEL, "c_pre", "mean", 0.0, 9)]

    def test_sharing_the_mass_changes_no_number(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        w = jnp.ones(N)
        shared = tau_rows(self.SPECS, x, w, mech, uniform=True)
        alone = tau_rows(self.SPECS, x, w, mech, uniform=False)
        assert np.allclose(shared, alone, rtol=0, atol=0)

    def test_a_non_uniform_weight_is_not_shared(self, mech):
        """Sorting permutes w per readout, so the mass genuinely differs."""
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        w = jax.nn.sigmoid(x[:, 0] - float(MU[0]))
        assert not np.allclose(tau_rows(self.SPECS, x, w, mech, uniform=True),
                               tau_rows(self.SPECS, x, w, mech, uniform=False))

    def test_the_batched_sort_matches_a_per_readout_sort(self, mech):
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        got = tau_rows(self.SPECS, x, jnp.ones(N), mech, uniform=False)
        for k, spec in enumerate(self.SPECS):
            col = jnp.sort(x[:, mech.readouts.index(spec.target_id)])
            want = tau_row(spec, col, jnp.ones(N))
            assert float(got[k]) == pytest.approx(float(want), rel=1e-12)


class TestPresortedAndMassTable:
    """Two exact speedups: sorting once per scenario, and a phi-free Beta mass."""

    SPECS = {"c_pre": [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=p)
                       for p in (0.25, 0.5, 0.75)],
             "c_post": [RowSpec(RATIO, "c_post", "quantile", 0.0, 9, p=0.5),
                        RowSpec(LEVEL, "c_post", "mean", 0.0, 9)]}

    def test_the_measurement_map_preserves_order(self, mech):
        """Why sorting may be hoisted above eq:disc: kappa > 0."""
        x = readout_cloud(MU, OMEGA, ZERO2, mech)[0]
        a, b = jnp.array([0.3, -0.2]), jnp.array([0.4, 0.1])
        c = jnp.array([1.0, -0.5])
        lhs = jnp.sort(apply_map(x, a, b, c, mech.Z), axis=0)
        rhs = apply_map(jnp.sort(x, axis=0), a, b, c, mech.Z)
        assert jnp.array_equal(lhs, rhs)

    def test_presorting_gives_the_same_tau(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        a, b = jnp.array([0.2, -0.1]), jnp.array([0.3, 0.15])
        x = readout_cloud(MU, OMEGA, ZERO2, mech)
        loose = tau_block(x, plan, self.SPECS, a, b, refs, mech, SCENARIO_OF)
        tight = tau_block(jnp.sort(x, axis=1), plan, self.SPECS, a, b, refs, mech,
                          SCENARIO_OF, presorted=True)
        assert np.allclose(loose, tight, rtol=1e-12, atol=0)

    def test_tau_all_presorts_only_without_eligibility(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        a, b = jnp.array([0.2, -0.1]), jnp.array([0.3, 0.15])
        args = ([plan], self.SPECS, refs, mech, SCENARIO_OF)
        plain = tau_all(MU, OMEGA, a, b, ZERO2, *args)
        elig = tau_all(MU, OMEGA, a, b, ZERO2, *args,
                       elig_fn=lambda x, c: jnp.ones(x.shape[0]),
                       elig_at={"trial": "c_pre"})
        assert np.allclose(plain[0], elig[0], rtol=1e-10)

    def test_a_precomputed_mass_table_changes_no_number(self, mech, plan):
        refs = reference_levels(MU, OMEGA, mech, [plan], SCENARIO_OF)
        a, b = jnp.array([0.2, -0.1]), jnp.array([0.3, 0.15])
        args = ([plan], self.SPECS, refs, mech, SCENARIO_OF)
        table = quantile_mass_table(self.SPECS, N)
        assert set(table) == {(0.25, 9, "type7"), (0.5, 9, "type7"),
                              (0.75, 9, "type7")}
        assert np.allclose(tau_all(MU, OMEGA, a, b, ZERO2, *args)[0],
                           tau_all(MU, OMEGA, a, b, ZERO2, *args,
                                   mass_table=table)[0], rtol=0, atol=0)

    def test_the_table_covers_only_quantile_rows(self):
        table = quantile_mass_table(self.SPECS, 1000)
        assert all(len(k) == 3 for k in table)
        assert len(table) == 3          # the mean row contributes no key


class TestScenarioColumns:
    """Most readouts are dead at most scenarios: a target belongs to one cohort."""

    SPECS = {"c_pre": [RowSpec(LEVEL, "c_pre", "quantile", 0.0, 9, p=0.5)],
             "c_post": [RowSpec(RATIO, "c_post", "mean", 0.0, 9)]}

    def test_each_scenario_keeps_only_what_it_reports(self, mech):
        cols = scenario_columns(self.SPECS, SCENARIO_OF, mech.readouts)
        assert cols == {0: (mech.readouts.index(LEVEL),),
                        1: (mech.readouts.index(RATIO),)}

    def test_a_readout_no_cohort_reports_is_dropped(self):
        readouts = ("dead_a", LEVEL, "dead_b", RATIO)
        cols = scenario_columns(self.SPECS, SCENARIO_OF, readouts)
        assert cols == {0: (1,), 1: (3,)}

    def test_restricting_columns_changes_no_number(self, plan):
        """A mechanism carrying two readouts nobody reports must give the same tau."""
        z = jnp.asarray(np.random.default_rng(0).standard_normal((N, 2)))

        def h4(y, log_R=None):
            level = jnp.log(y[:, :, 0])
            ratio = level - jnp.log(y[:, :, 1])
            dead = level * 3.0 + 11.0
            return jnp.stack([dead, level, dead - 1.0, ratio], axis=-1)

        wide = Mechanism(L_R=jnp.eye(2), z=z,
                         Z=jnp.array([[1., 1.], [1., 0.], [1., 1.], [1., 1.]]),
                         readouts=("dead_a", LEVEL, "dead_b", RATIO),
                         n_species=2, n_scenarios=2, beta_species=jnp.array([0, 1]),
                         g_fn=_g, h_fn=h4)
        narrow = Mechanism(L_R=jnp.eye(2), z=z,
                           Z=jnp.array([[1., 0.], [1., 1.]]),
                           readouts=(LEVEL, RATIO), n_species=2, n_scenarios=2,
                           beta_species=jnp.array([0, 1]), g_fn=_g, h_fn=_h)
        a, b = jnp.array([0.2, -0.1]), jnp.array([0.3, 0.15])
        got, want = [], []
        for m in (wide, narrow):
            refs = reference_levels(MU, OMEGA, m, [plan], SCENARIO_OF)
            (got if m is wide else want).append(
                tau_all(MU, OMEGA, a, b, ZERO2, [plan], self.SPECS, refs, m,
                        SCENARIO_OF)[0])
        assert np.allclose(got[0], want[0], rtol=1e-12)

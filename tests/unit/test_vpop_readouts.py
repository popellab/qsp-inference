"""Unit tests for h_r (qsp_inference.vpop.readouts).

The corpus bodies are written against numpy and run here on JAX arrays, so the
checks are that the shim reproduces numpy's answer, that a masked division has a
finite gradient rather than the NaN plain ``jnp.where`` would give, and that the
bodies that cannot trace are refused instead of quietly mis-evaluated.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from qsp_inference.vpop.readouts import (
    NEEDS_TRAJECTORY,
    NotPerPatient,
    UntraceableReadout,
    build_h_fn,
    compile_observable,
    shim_module,
    target_constants,
)


@pytest.fixture(autouse=True)
def _x64():
    jax.config.update("jax_enable_x64", True)


RATIO = """
def compute_observable(time, species_dict, constants):
    import numpy as np
    a = species_dict['num']
    b = species_dict['den']
    return np.divide(a, b, out=np.zeros_like(a), where=b > 0.0)
"""

SCALED = """
def compute_observable(time, species_dict, constants):
    x = species_dict['x']
    return constants['R'] * x
"""

#: Axis 0 is the declared reference, exactly as the corpus's fold changes write it.
FOLD = """
def compute_observable(time, species_dict, constants):
    x = species_dict['x']
    return x / x[0]
"""

#: Divides by patient 0 rather than by the reference: the bug this guards.
LEAKY = """
def compute_observable(time, species_dict, constants):
    x = species_dict['x']
    return x / x[0][0]
"""

DEAF = """
def compute_observable(time, species_dict, constants):
    x = species_dict['x']
    return (x / x) * 3.0
"""


class TestShim:
    def test_import_numpy_inside_the_body_binds_the_shim(self):
        fn = compile_observable(
            "def compute_observable(time, species_dict, constants):\n"
            "    import numpy as np\n"
            "    return np.asarray(species_dict['x'])\n")
        out = fn(0.0, {"x": jnp.array([1.0, 2.0])}, {})
        assert isinstance(out, jax.Array)

    def test_other_imports_still_work(self):
        fn = compile_observable(
            "def compute_observable(time, species_dict, constants):\n"
            "    import math\n"
            "    return math.pi\n")
        assert fn(0.0, {}, {}) == pytest.approx(np.pi)

    def test_divide_matches_numpy_including_the_masked_form(self):
        a, b = np.array([2.0, 1.0, 6.0]), np.array([8.0, 0.0, 3.0])
        want = np.divide(a, b, out=np.zeros_like(a), where=b > 0.0)
        got = shim_module().divide(jnp.asarray(a), jnp.asarray(b),
                                   out=jnp.zeros_like(jnp.asarray(a)),
                                   where=jnp.asarray(b) > 0.0)
        assert np.allclose(got, want)

    def test_a_masked_division_has_a_finite_gradient(self):
        """The whole reason the corpus uses divide(where=) and not where()."""
        fn = compile_observable(RATIO)

        def total(den):
            return jnp.sum(fn(0.0, {"num": jnp.array([2.0, 1.0]),
                                    "den": jnp.array([8.0, den])}, {}))

        assert float(total(0.0)) == pytest.approx(0.25)
        assert np.isfinite(jax.grad(total)(0.0))
        assert jax.grad(total)(4.0) == pytest.approx(-1.0 / 16.0)

    def test_the_plain_where_form_is_what_this_avoids(self):
        bad = lambda d: jnp.sum(jnp.where(jnp.array([8.0, d]) > 0,
                                          jnp.array([2.0, 1.0]) / jnp.array([8.0, d]),
                                          0.0))
        assert np.isnan(jax.grad(bad)(0.0))


class TestCompile:
    def test_a_body_with_no_compute_observable_raises(self):
        with pytest.raises(UntraceableReadout, match="no compute_observable"):
            compile_observable("x = 1\n", label="t")

    def test_a_syntax_error_names_the_target(self):
        with pytest.raises(UntraceableReadout, match="t:"):
            compile_observable("def broken(:\n", label="t")


def _target(code, species, constants=(), aux=(), readout_time=0.0, reference=None):
    obs = {"code": code, "species": list(species), "constants": list(constants),
           "auxiliary_parameters": list(aux), "readout_time": readout_time}
    if reference is not None:
        obs["readout"] = {
            "quantity_kind": "foldchange",
            "reference": {"kind": "timepoint", "timepoint": reference,
                          "timepoint_unit": "day"},
        }
    return {"observable": obs}


class TestTargetConstants:
    def test_fixed_constants_carry_their_value(self):
        t = _target(SCALED, ["x"], constants=[{"name": "R", "value": 3.0}])
        assert target_constants(t, {}) == {"R": 3.0}

    def test_auxiliary_values_come_from_the_caller(self):
        t = _target(SCALED, ["x"], aux=[{"name": "R"}])
        assert target_constants(t, {"R": 7.0}) == {"R": 7.0}

    def test_a_missing_auxiliary_raises(self):
        t = _target(SCALED, ["x"], aux=[{"name": "R"}])
        with pytest.raises(UntraceableReadout, match="auxiliary 'R'"):
            target_constants(t, {})


STATES = ("den", "num", "x")


def _obs(species):
    """Stands in for the generated module: passes raw states through."""
    return dict(species)


class TestBuildHFn:
    TARGETS = {
        "r_ratio": _target(RATIO, ["num", "den"]),
        "r_scaled": _target(SCALED, ["x"], aux=[{"name": "R"}]),
    }

    def _y(self, S=2, N=4):
        return jnp.stack([jnp.full((S, N), v) for v in (4.0, 1.0, 2.0)], axis=-1)

    def test_shape_and_readout_order(self):
        h = build_h_fn(self.TARGETS, ["r_ratio", "r_scaled"], STATES, _obs, ["R"])
        x = h(self._y(), jnp.array([np.log(3.0)]))
        assert x.shape == (2, 4, 2)
        assert np.allclose(x[..., 0], np.log(0.25))     # num/den = 1/4
        assert np.allclose(x[..., 1], np.log(3.0 * 2.0))

    def test_readouts_are_returned_on_the_log_scale(self):
        h = build_h_fn({"r_scaled": self.TARGETS["r_scaled"]}, ["r_scaled"],
                       STATES, _obs, ["R"])
        assert np.allclose(h(self._y(), jnp.array([0.0])), np.log(2.0))

    def test_log_R_moves_only_the_readout_that_declares_it(self):
        h = build_h_fn(self.TARGETS, ["r_ratio", "r_scaled"], STATES, _obs, ["R"])
        base = h(self._y(), jnp.array([0.0]))
        moved = h(self._y(), jnp.array([np.log(5.0)]))
        assert np.allclose(moved[..., 0], base[..., 0])
        assert np.allclose(moved[..., 1] - base[..., 1], np.log(5.0))

    def test_the_gradient_in_log_R_counts_the_elements_it_touches(self):
        h = build_h_fn(self.TARGETS, ["r_ratio", "r_scaled"], STATES, _obs, ["R"])
        g = jax.grad(lambda r: jnp.sum(h(self._y(S=3, N=5), r)))(jnp.array([0.0]))
        assert float(g[0]) == pytest.approx(3 * 5)   # one readout, S*N entries

    def test_a_trajectory_readout_is_refused(self):
        name = sorted(NEEDS_TRAJECTORY)[0]
        targets = dict(self.TARGETS, **{name: _target(RATIO, ["num", "den"])})
        with pytest.raises(UntraceableReadout, match="whole trajectory"):
            build_h_fn(targets, ["r_ratio", name], STATES, _obs)

    def test_it_jits(self):
        h = build_h_fn(self.TARGETS, ["r_ratio", "r_scaled"], STATES, _obs, ["R"])
        out = jax.jit(h)(self._y(), jnp.array([0.0]))
        assert out.shape == (2, 4, 2)


class TestScenarioSelection:
    """eq:readout is one value per patient per readout, not one per scenario."""

    TARGETS = {"r_ratio": _target(RATIO, ["num", "den"]),
               "r_scaled": _target(SCALED, ["x"], aux=[{"name": "R"}])}
    NAMES = ["r_ratio", "r_scaled"]

    def _y(self, S=3, N=4):
        """Scenario s scales every state by (s + 1)."""
        base = jnp.stack([jnp.full((N,), v) for v in (4.0, 1.0, 2.0)], axis=-1)
        return jnp.stack([base * (s + 1.0) for s in range(S)])

    def test_without_scenario_of_every_scenario_is_returned(self):
        h = build_h_fn(self.TARGETS, self.NAMES, STATES, _obs, ["R"])
        assert h(self._y(), jnp.array([0.0])).shape == (3, 4, 2)

    def test_with_scenario_of_one_column_per_readout(self):
        h = build_h_fn(self.TARGETS, self.NAMES, STATES, _obs, ["R"],
                       scenario_of={"r_ratio": (0,), "r_scaled": (2,)})
        assert h(self._y(), jnp.array([0.0])).shape == (4, 2)

    def test_each_readout_reads_the_scenario_it_was_given(self):
        h = build_h_fn(self.TARGETS, self.NAMES, STATES, _obs, ["R"],
                       scenario_of={"r_ratio": (0,), "r_scaled": (2,)})
        x = h(self._y(), jnp.array([0.0]))
        # the ratio is scale-free, so scenario does not move it
        assert np.allclose(x[:, 0], np.log(0.25))
        # r_scaled reads x at scenario 2, where it is 3x its base
        assert np.allclose(x[:, 1], np.log(2.0 * 3.0))

    def test_it_agrees_with_selecting_from_the_wide_form(self):
        at = {"r_ratio": (0,), "r_scaled": (2,)}
        wide = build_h_fn(self.TARGETS, self.NAMES, STATES, _obs, ["R"])
        narrow = build_h_fn(self.TARGETS, self.NAMES, STATES, _obs, ["R"],
                            scenario_of=at)
        y, r = self._y(), jnp.array([0.3])
        full, cut = wide(y, r), narrow(y, r)
        for k, name in enumerate(self.NAMES):
            assert np.allclose(cut[:, k], full[at[name][0], :, k])

    def test_partial_coverage_is_refused(self):
        with pytest.raises(UntraceableReadout, match="omits r_scaled"):
            build_h_fn(self.TARGETS, self.NAMES, STATES, _obs, ["R"],
                       scenario_of={"r_ratio": (0,)})

    def test_a_map_covering_more_than_this_build_is_fine(self):
        h = build_h_fn(self.TARGETS, ["r_ratio"], STATES, _obs, ["R"],
                       scenario_of={"r_ratio": (0,), "elsewhere": (7,)})
        assert h(self._y(), None).shape == (4, 1)


class TestDeclaredReference:
    """A fold change is taken against axis 0, which the target names."""

    def _targets(self, reference=0.0):
        return {"r_fold": _target(FOLD, ["x"], readout_time=21.0,
                                  reference=reference),
                "r_scaled": _target(SCALED, ["x"], constants=[{"name": "R", "value": 1.0}])}

    def _y(self, S=3, N=4):
        """State x is 2.0 at scenario 0 and rises by 1.0 per scenario."""
        cols = [jnp.full((S, N), v) for v in (1.0, 1.0, 1.0)]
        cols[2] = jnp.arange(2.0, 2.0 + S)[:, None] * jnp.ones((S, N))
        return jnp.stack(cols, axis=-1)

    def test_axis_zero_is_the_reference_not_a_patient(self):
        h = build_h_fn(self._targets(), ["r_fold"], STATES, _obs,
                       scenario_of={"r_fold": (0, 2)})
        # x is 2.0 at scenario 0 and 4.0 at scenario 2
        assert np.allclose(h(self._y(), None)[:, 0], np.log(4.0 / 2.0))

    def test_the_reference_scenario_is_the_one_named_first(self):
        h = build_h_fn(self._targets(), ["r_fold"], STATES, _obs,
                       scenario_of={"r_fold": (1, 2)})
        assert np.allclose(h(self._y(), None)[:, 0], np.log(4.0 / 3.0))

    def test_a_declared_reference_needs_two_scenarios(self):
        with pytest.raises(UntraceableReadout, match="expected 2"):
            build_h_fn(self._targets(), ["r_fold"], STATES, _obs,
                       scenario_of={"r_fold": (2,)})

    def test_one_without_a_reference_may_not_be_given_two(self):
        with pytest.raises(UntraceableReadout, match="expected 1"):
            build_h_fn(self._targets(), ["r_scaled"], STATES, _obs,
                       scenario_of={"r_scaled": (0, 2)})

    def test_the_wide_form_cannot_honour_a_reference_and_says_so(self):
        with pytest.raises(UntraceableReadout, match="declared reference"):
            build_h_fn(self._targets(), ["r_fold"], STATES, _obs)

    def test_an_unresolvable_reference_kind_is_refused(self):
        targets = {"r_fold": _target(FOLD, ["x"])}
        targets["r_fold"]["observable"]["readout"] = {
            "reference": {"kind": "arm", "arm": "control"}}
        with pytest.raises(UntraceableReadout, match="does not resolve"):
            build_h_fn(targets, ["r_fold"], STATES, _obs,
                       scenario_of={"r_fold": (0, 1)})


class TestPerPatient:
    """eq:readout is a per-patient map, and build refuses code that is not."""

    def test_reaching_another_patient_is_refused(self):
        targets = {"r_leaky": _target(LEAKY, ["x"])}
        with pytest.raises(NotPerPatient, match="read patients other than"):
            build_h_fn(targets, ["r_leaky"], STATES, _obs,
                       scenario_of={"r_leaky": (0,)})

    def test_the_offending_readout_is_named(self):
        targets = {"r_ok": _target(SCALED, ["x"], constants=[{"name": "R", "value": 2.0}]),
                   "r_leaky": _target(LEAKY, ["x"])}
        with pytest.raises(NotPerPatient, match=r"^r_leaky\b"):
            build_h_fn(targets, ["r_ok", "r_leaky"], STATES, _obs,
                       scenario_of={"r_ok": (0,), "r_leaky": (0,)})

    def test_a_readout_that_ignores_its_patient_is_refused(self):
        targets = {"r_deaf": _target(DEAF, ["x"])}
        with pytest.raises(NotPerPatient, match="do not vary with the patient"):
            build_h_fn(targets, ["r_deaf"], STATES, _obs,
                       scenario_of={"r_deaf": (0,)})

    def test_an_undeclared_fold_change_is_caught(self):
        """The corpus is not uniform: some fold changes declare no reference.

        With nothing declared the body gets ``T = 1``, so ``x / x[0]`` broadcasts
        to all ones and stops depending on the patient at all.
        """
        targets = {"r_fold": _target(FOLD, ["x"])}
        with pytest.raises(NotPerPatient, match="do not vary"):
            build_h_fn(targets, ["r_fold"], STATES, _obs,
                       scenario_of={"r_fold": (0,)})

    def test_a_scale_invariant_fraction_passes(self):
        """The probe perturbs each species separately, so a ratio still moves."""
        h = build_h_fn({"r_ratio": _target(RATIO, ["num", "den"])}, ["r_ratio"],
                       STATES, _obs, scenario_of={"r_ratio": (0,)})
        assert h(jnp.ones((1, 4, 3)), None).shape == (4, 1)

    def test_a_correct_fold_change_passes(self):
        targets = {"r_fold": _target(FOLD, ["x"], reference=0.0)}
        h = build_h_fn(targets, ["r_fold"], STATES, _obs,
                       scenario_of={"r_fold": (0, 1)})
        assert h(jnp.ones((2, 4, 3)), None).shape == (4, 1)

    def test_the_wide_form_is_checked_too(self):
        targets = {"r_leaky": _target(LEAKY, ["x"])}
        with pytest.raises(NotPerPatient):
            build_h_fn(targets, ["r_leaky"], STATES, _obs)

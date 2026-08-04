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


def _target(code, species, constants=(), aux=(), readout_time=0.0):
    return {"observable": {"code": code, "species": list(species),
                           "constants": list(constants),
                           "auxiliary_parameters": list(aux),
                           "readout_time": readout_time}}


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

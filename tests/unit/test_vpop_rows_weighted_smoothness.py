"""The weighted moment path is C1 (qsp_inference.vpop.rows).

``bootstrap_row``'s weighted branch reads the cloud through an interpolated
weighted quantile function. Linear interpolation made it C0 and not C1: the
slope jumps when a frozen ``u`` crosses a knot, and eq:elig's knots are
cumulative weights that move with ``phi``, so the jumps sweep the design as the
chain steps. These check the monotone cubic that replaced it.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from qsp_inference.vpop.rows import (  # noqa: E402
    _pchip_interp,
    bootstrap_design,
    mean_row,
)


def _cloud(n=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.lognormal(0.0, 0.7, n))


def _knots(w):
    c = np.cumsum(w)
    return (c - 0.5 * w) / c[-1]


def test_interpolates_the_knots_it_is_given():
    x = _cloud()
    k = _knots(np.full(x.size, 1.0 / x.size))
    got = _pchip_interp(jnp.asarray(k), jnp.asarray(k), jnp.asarray(x))
    np.testing.assert_allclose(np.asarray(got), x, rtol=1e-12, atol=1e-12)


def test_monotone_and_within_the_cloud():
    """A quantile function cannot run backwards, and cannot leave the cloud.

    This is why the interpolant is monotone cubic and not a natural spline: a
    spline overshoots, which here would print a member the cloud does not have.
    """
    rng = np.random.default_rng(3)
    x = _cloud()
    w = rng.random(x.size) ** 3          # very uneven, as eq:elig's weights are
    u = np.linspace(0.0, 1.0, 4001)
    got = np.asarray(_pchip_interp(jnp.asarray(u), jnp.asarray(_knots(w)),
                                   jnp.asarray(x)))
    assert np.all(np.diff(got) >= -1e-12)
    assert got.min() >= x.min() - 1e-12
    assert got.max() <= x.max() + 1e-12


def test_constant_extrapolation_matches_the_linear_form():
    """Outside the knots the estimand is unchanged, so both forms are flat."""
    x = _cloud()
    k = _knots(np.full(x.size, 1.0 / x.size))
    u = jnp.asarray([0.0, k[0] * 0.5, 1.0])
    got = np.asarray(_pchip_interp(u, jnp.asarray(k), jnp.asarray(x)))
    ref = np.asarray(jnp.interp(u, jnp.asarray(k), jnp.asarray(x)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_equal_weights_agree_with_the_lookup_to_the_discretisation():
    """At equal weights the interpolated form and the unweighted lookup answer
    the same question, and differ only by the O(1/N) the lookup discards."""
    x = _cloud(n=256)
    u = np.linspace(1e-6, 1 - 1e-6, 20_000)
    w = np.full(x.size, 1.0 / x.size)
    got = np.asarray(_pchip_interp(jnp.asarray(u), jnp.asarray(_knots(w)),
                                   jnp.asarray(x)))
    idx = np.minimum((u * x.size).astype(int), x.size - 1)
    assert np.abs(got.mean() - x[idx].mean()) < 5.0 / x.size


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_gradient_is_continuous_across_a_knot_crossing(seed):
    """The claim the change exists for.

    Sweep the weights along a line so the knots slide past a frozen ``u``, and
    watch the derivative of the row. Linear interpolation steps here; monotone
    cubic does not. The test is scale free: the largest jump in the derivative
    is compared against its own median over the sweep.
    """
    rng = np.random.default_rng(seed)
    x = jnp.asarray(_cloud(n=48, seed=seed))
    u = jnp.asarray(rng.random((16, 12)))
    w0 = jnp.asarray(rng.random(48) ** 2 + 1e-3)
    step = jnp.asarray(rng.standard_normal(48))

    def row(t):
        w = w0 * jnp.exp(0.3 * t * step)
        c = jnp.cumsum(w)
        return jnp.mean(_pchip_interp(u, (c - 0.5 * w) / c[-1], x))

    ts = jnp.linspace(-1.0, 1.0, 3001)
    g = np.asarray(jax.vmap(jax.grad(row))(ts))
    dg = np.abs(np.diff(g))
    assert np.max(dg) < 60.0 * np.median(dg)


def test_mean_row_reads_the_weights():
    """``tau_row`` threads ``w_sorted`` into every other branch; a raw-scale mean
    silently returned the unweighted cloud mean before this."""
    x = _cloud(n=32)
    rng = np.random.default_rng(7)
    w = rng.random(x.size) ** 2
    got = float(mean_row(jnp.asarray(x), w_sorted=jnp.asarray(w)))
    assert got == pytest.approx(float(np.sum(w * x) / np.sum(w)), rel=1e-12)
    assert float(mean_row(jnp.asarray(x))) == pytest.approx(float(x.mean()),
                                                            rel=1e-12)


def test_zero_weights_stay_finite_in_value_and_gradient():
    """An ineligible member's slice closes to nothing, so knots land on top of
    each other. Neither the row nor its gradient may go non-finite there."""
    x = jnp.asarray(_cloud(n=32))
    u = bootstrap_design(jax.random.PRNGKey(0), n=10, n_boot=8)
    base = jnp.asarray(np.r_[np.zeros(16), np.ones(16)])

    def row(t):
        w = jnp.clip(base + t, 0.0, None) + 1e-30
        c = jnp.cumsum(w)
        return jnp.mean(_pchip_interp(u, (c - 0.5 * w) / c[-1], x))

    assert np.isfinite(float(row(0.0)))
    assert np.isfinite(float(jax.grad(row)(0.0)))

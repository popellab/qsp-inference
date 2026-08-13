"""The scaled mean row reads no ordering (qsp_inference.vpop.rows).

``E[g(sample mean)]`` used to be taken over the frozen design, which reads the
cloud through its quantile function and so needs it sorted. Under eq:elig the
weights travel with that order, and a crossing steps the row's derivative. The
second-order expansion needs only the cloud's weighted moments, and those are
sums.

The claims: it agrees with what the design was estimating, it does not depend on
the order of the cloud, and it is smooth where the design was not.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from qsp_inference.vpop.rows import (  # noqa: E402
    RowSpec,
    bootstrap_design,
    bootstrap_row,
    mean_row,
    mean_row_scaled,
    to_scale,
    weighted_moments,
)


def _spec(scale, n, ref=1.0):
    return RowSpec(target_id="t", cohort_id="c", stat="mean", value=0.0, n=n,
                   p=None, convention="type7", convention_recorded=False,
                   scale=scale, scale_ref=ref)


def _cloud(n=256, seed=0, lo=0.5, hi=3.0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(lo, hi, n))


@pytest.mark.parametrize("scale", ["raw", "log", "asinh", "logit"])
def test_order_of_the_cloud_does_not_matter(scale):
    """The property the change exists for: the row is a function of sums."""
    x = _cloud(lo=0.05, hi=0.9) if scale == "logit" else _cloud()
    rng = np.random.default_rng(1)
    w = rng.random(x.size) ** 3 + 1e-6
    s = _spec(scale, 12)
    a = float(mean_row_scaled(jnp.asarray(x), s, w_sorted=jnp.asarray(w)))
    perm = rng.permutation(x.size)
    b = float(mean_row_scaled(jnp.asarray(x[perm]), s,
                              w_sorted=jnp.asarray(w[perm])))
    assert a == pytest.approx(b, rel=1e-12)


def test_raw_scale_is_the_population_mean_exactly():
    """g'' = 0, so no correction survives and this is mean_row."""
    x = jnp.asarray(_cloud())
    w = jnp.asarray(np.random.default_rng(2).random(int(x.size)) ** 2 + 1e-6)
    got = float(mean_row_scaled(x, _spec("raw", 9), w_sorted=w))
    assert got == pytest.approx(float(mean_row(x, w_sorted=w)), rel=1e-12)


@pytest.mark.parametrize("scale,n", [("log", 6), ("log", 30), ("log", 215),
                                     ("asinh", 10), ("logit", 10)])
def test_agrees_with_a_high_accuracy_bootstrap(scale, n):
    """Against the quantity the design was estimating, not against the design.

    n_boot is far above the 400 a run uses, so the reference carries much less
    Monte Carlo error than the path being replaced. The expansion drops at
    O(n^-2), so the tolerance loosens as the cohort shrinks.
    """
    x = _cloud(lo=0.05, hi=0.9) if scale == "logit" else _cloud()
    xj = jnp.asarray(x)
    w = jnp.asarray(np.random.default_rng(3).random(x.size) ** 2 + 1e-6)
    s = _spec(scale, n)
    u = bootstrap_design(jax.random.PRNGKey(0), n=n, n_boot=60_000)
    ref = float(bootstrap_row(xj, u, lambda v: to_scale(jnp.mean(v), s, jnp),
                              w_sorted=w))
    got = float(mean_row_scaled(xj, s, w_sorted=w))
    spread = float(jnp.sqrt(weighted_moments(xj, w)[1] / n))
    # measured against the sd of the sample mean, which is what the row's own V
    # is built out of: a bias well under that cannot move a residual
    assert abs(got - ref) < 0.05 * spread


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_crossing_step_in_the_derivative(seed):
    """Sweep the cloud so members cross, and watch the row's derivative.

    The design-based form steps at every crossing; this one cannot, since a
    permutation leaves its sums alone. Compared against that form on the same
    sweep so the test states a difference rather than a threshold.
    """
    rng = np.random.default_rng(seed)
    N, n = 48, 12
    a = jnp.asarray(rng.uniform(0.5, 3.0, N))
    b = jnp.asarray(rng.normal(0, 1, N))
    w = jnp.asarray(rng.random(N) ** 2 + 1e-3)
    s = _spec("log", n)
    u = bootstrap_design(jax.random.PRNGKey(seed), n=n, n_boot=64)

    def closed(t):
        x = a + b * t
        p = jnp.argsort(x)
        return mean_row_scaled(x[p], s, w_sorted=w[p])

    def design(t):
        x = a + b * t
        p = jnp.argsort(x)
        return bootstrap_row(x[p], u, lambda v: to_scale(jnp.mean(v), s, jnp),
                             w_sorted=w[p])

    ts = jnp.linspace(-0.3, 0.3, 4001)
    gc = np.abs(np.diff(np.asarray(jax.vmap(jax.grad(closed))(ts))))
    gd = np.abs(np.diff(np.asarray(jax.vmap(jax.grad(design))(ts))))
    # the design's worst step against its own typical one, and the closed form's
    assert np.max(gc) / np.median(gc) < 20.0
    assert np.max(gc) / np.median(gc) < np.max(gd) / np.median(gd)


def test_weighted_moments_match_numpy():
    x = _cloud(n=64)
    w = np.random.default_rng(5).random(x.size) + 1e-6
    mu, var = weighted_moments(jnp.asarray(x), jnp.asarray(w))
    m = np.sum(w * x) / np.sum(w)
    assert float(mu) == pytest.approx(m, rel=1e-12)
    assert float(var) == pytest.approx(np.sum(w * (x - m) ** 2) / np.sum(w),
                                       rel=1e-12)

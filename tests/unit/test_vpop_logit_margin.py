"""The logit margin in eq:crn: a bounded parameter must stay inside its bound."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    from qsp_inference.vpop.predict import Mechanism, patient_cloud
except ImportError as exc:  # pragma: no cover
    # vpop.rows wants WIDTH_STATS / SAMPLING_WIDTH_STATS from maple's
    # shared_models and the installed maple exports neither, so vpop.predict
    # does not import at all here. Narrow on purpose: any other ImportError is
    # a real regression and must still fail the suite.
    if "WIDTH_STATS" not in str(exc):
        raise
    pytest.skip(f"vpop.predict unimportable: {exc}", allow_module_level=True)

P = 3
LOGIT = jnp.array([False, True, False])
# median 0.80 is Emax_Cy_Treg's: a fractional maximum effect, so >1 is more than
# complete inhibition. index 2 is the same value left on the log scale, as the
# control that shows the breach is the margin's doing and not the width's.
MU = jnp.log(jnp.array([5.0, 0.80, 0.80]))
OMEGA = jnp.full(P, 0.35)


def _mech(z, logit=LOGIT):
    return Mechanism(
        L_R=jnp.eye(P), z=jnp.asarray(z), Z=jnp.zeros((1, 1)), readouts=("r",),
        n_species=1, n_scenarios=1, beta_species=jnp.array([], int),
        g_fn=lambda v, s: v, h_fn=lambda *a: a, logit=logit,
    )


@pytest.fixture(scope="module")
def cloud():
    z = np.random.default_rng(0).standard_normal((200_000, P))
    return np.asarray(jnp.exp(patient_cloud(MU, OMEGA, _mech(z))))


def test_a_logit_margin_cannot_leave_the_unit_interval(cloud):
    assert cloud[:, 1].max() < 1.0
    assert cloud[:, 1].min() > 0.0


def test_the_log_margin_at_the_same_median_does_leave_it(cloud):
    """Guards the reason this exists: no width fixes it, only the margin does."""
    assert (cloud[:, 2] > 1.0).mean() > 0.2


def test_the_median_is_untouched(cloud):
    for j in range(P):
        assert np.median(cloud[:, j]) == pytest.approx(float(jnp.exp(MU[j])), rel=0.01)


def test_mu_keeps_its_log_scale_meaning():
    """At z = 0 the cloud is exp(mu) on every margin, so eq:muprior is unchanged."""
    got = patient_cloud(MU, OMEGA, _mech(np.zeros((1, P))))[0]
    assert np.allclose(np.asarray(got), np.asarray(MU))


def test_gradients_stay_finite_through_the_discarded_branch():
    """The double-where guard: a median above 1 must not put a nan in d/dmu."""
    m = _mech(np.random.default_rng(1).standard_normal((64, P)))
    g_mu = jax.grad(lambda x: patient_cloud(x, OMEGA, m).sum())(MU)
    g_om = jax.grad(lambda x: patient_cloud(MU, x, m).sum())(OMEGA)
    assert bool(jnp.all(jnp.isfinite(g_mu)))
    assert bool(jnp.all(jnp.isfinite(g_om)))


@pytest.mark.parametrize("logit", [None, jnp.zeros(P, bool)])
def test_no_logit_parameters_is_the_old_behaviour_exactly(logit):
    z = np.random.default_rng(2).standard_normal((128, P))
    m = _mech(z, logit)
    expected = MU[None, :] + m.zL * OMEGA[None, :]
    assert np.allclose(np.asarray(patient_cloud(MU, OMEGA, m)), np.asarray(expected))


def test_a_per_row_mu_composes_the_two_spreads():
    """The pool draws one mu per row; the fit shares one across the cloud. The
    same expression has to serve both, or the emulator trains on a margin the fit
    does not use."""
    n = 4096
    rng = np.random.default_rng(3)
    z = rng.standard_normal((n, P))
    mech = _mech(z)
    # mu must stay inside the bound on the logit column: exp(mu) is that
    # patient-set's median, and a median outside (0, 1) is not a width question.
    # See the note on eq:muprior for bounded parameters.
    jitter = 0.2 * rng.standard_normal((n, P))
    jitter[:, 1] = -np.abs(jitter[:, 1])
    mu_rows = jnp.asarray(MU[None, :] + jitter)

    per_row = patient_cloud(mu_rows, OMEGA, mech)
    assert per_row.shape == (n, P)

    # row i of the (N, P) call equals the shared-mu call restricted to that row
    one = patient_cloud(mu_rows[7], OMEGA, _mech(z[7:8]))
    assert np.allclose(np.asarray(per_row[7]), np.asarray(one[0]), atol=1e-12)

    # and the bounded margin still holds when mu varies per row
    theta = np.asarray(jnp.exp(per_row))
    assert theta[:, 1].max() < 1.0


def test_an_out_of_bound_mu_is_finite_in_the_margin_and_flagged_for_rejection():
    """Two halves of one mechanism. The margin must stay finite so the leapfrog
    gradient survives; the flag is what actually rejects, via -inf on the density.
    A margin returning -inf would give theta = 0, which the emulator would
    cheerfully simulate."""
    from qsp_inference.vpop.predict import apply_margins, mu_out_of_bound

    bad = jnp.asarray([jnp.log(5.0), 0.30, jnp.log(0.8)])   # index 1 above its bound
    ok = jnp.asarray([jnp.log(5.0), -0.30, jnp.log(0.8)])

    assert bool(mu_out_of_bound(bad, LOGIT))
    assert not bool(mu_out_of_bound(ok, LOGIT))
    assert not bool(mu_out_of_bound(bad, None))

    zL = jnp.asarray(np.random.default_rng(4).standard_normal((32, P)))
    got = apply_margins(bad, OMEGA, zL, LOGIT)
    assert bool(jnp.all(jnp.isfinite(got))), "a rejected draw must not poison the gradient"
    g = jax.grad(lambda m: apply_margins(m, OMEGA, zL, LOGIT).sum())(bad)
    assert bool(jnp.all(jnp.isfinite(g)))

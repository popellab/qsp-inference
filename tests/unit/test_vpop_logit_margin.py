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

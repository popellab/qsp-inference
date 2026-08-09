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
MEDIAN = jnp.array([5.0, 0.80, 0.80])
# mu is the LOGIT of the median on a logit margin and its log elsewhere, which is
# what logit_median_coords produces from a marginal stated on the log scale.
MU = jnp.array([jnp.log(5.0),
                jnp.log(0.80) - jnp.log1p(-0.80),
                jnp.log(0.80)])
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
        assert np.median(cloud[:, j]) == pytest.approx(float(MEDIAN[j]), rel=0.01)


def test_mu_is_the_median_in_its_own_coordinate():
    """At z = 0 the cloud is the median on every margin: log for a log margin,
    logit for a logit one. That is the coordinate eq:muprior is Gaussian in."""
    got = np.asarray(jnp.exp(patient_cloud(MU, OMEGA, _mech(np.zeros((1, P))))[0]))
    assert np.allclose(got, np.asarray(MEDIAN))


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
    # No sign constraint on the logit column any more: mu is the logit of that
    # patient-set's median there, so every real value names a median in (0, 1).
    jitter = 0.2 * rng.standard_normal((n, P))
    mu_rows = jnp.asarray(MU[None, :] + jitter)

    per_row = patient_cloud(mu_rows, OMEGA, mech)
    assert per_row.shape == (n, P)

    # row i of the (N, P) call equals the shared-mu call restricted to that row
    one = patient_cloud(mu_rows[7], OMEGA, _mech(z[7:8]))
    assert np.allclose(np.asarray(per_row[7]), np.asarray(one[0]), atol=1e-12)

    # and the bounded margin still holds when mu varies per row
    theta = np.asarray(jnp.exp(per_row))
    assert theta[:, 1].max() < 1.0


def test_no_mu_can_put_a_bounded_parameter_outside_its_bound():
    """The bound is a property of the coordinate, so there is nothing to reject.

    mu for a logit-margin parameter is the logit of the median, so every real mu
    gives a median in (0, 1). This is what replaced a -inf factor on exp(mu) >= 1:
    the factor's gradient is zero, but a leapfrog step into the excluded region
    gives infinite energy error and NUTS discards the trajectory as divergent.
    """
    from qsp_inference.vpop.predict import apply_margins

    zL = jnp.asarray(np.random.default_rng(4).standard_normal((32, P)))
    for mu_1 in (-8.0, -0.3, 0.0, 0.30, 8.0):
        mu = jnp.asarray([jnp.log(5.0), mu_1, jnp.log(0.8)])
        got = apply_margins(mu, OMEGA, zL, LOGIT)
        theta = np.asarray(jnp.exp(got))
        assert np.all(np.isfinite(theta))
        assert theta[:, 1].max() < 1.0, f"bounded parameter left (0,1) at mu={mu_1}"
        assert theta[:, 1].min() > 0.0
        g = jax.grad(lambda m: apply_margins(m, OMEGA, zL, LOGIT).sum())(mu)
        assert bool(jnp.all(jnp.isfinite(g)))


def test_logit_median_coords_preserves_the_median_and_widens_near_the_bound():
    from qsp_inference.vpop.predict import logit_median_coords

    mu_0 = np.array([np.log(5.0), np.log(0.8), np.log(0.5)])
    sd_1 = np.array([0.4, 0.2, 0.2])
    mu_new, sd_new = logit_median_coords(mu_0, sd_1, np.asarray(LOGIT))

    # index 1 is the only logit-margin entry: its median survives the move
    assert 1.0 / (1.0 + np.exp(-mu_new[1])) == pytest.approx(0.8)
    assert mu_new[0] == mu_0[0] and mu_new[2] == mu_0[2]
    assert sd_new[0] == sd_1[0] and sd_new[2] == sd_1[2]
    # d logit(m)/d log(m) = 1/(1 - m), so a median at 0.8 widens fivefold
    assert sd_new[1] == pytest.approx(sd_1[1] / (1.0 - 0.8))


def test_a_median_outside_the_bound_is_the_marginal_being_wrong():
    from qsp_inference.vpop.predict import logit_median_coords

    with pytest.raises(ValueError, match="not inside"):
        logit_median_coords(np.array([0.0, np.log(1.5), 0.0]),
                            np.ones(3), np.asarray(LOGIT))

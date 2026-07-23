"""Tests for temper_prior — building a training proposal from the anchored prior."""

import numpy as np
import pytest
from scipy import stats

torch = pytest.importorskip("torch")  # copula_prior pulls in torch (sbi extra)

from qsp_inference.priors.copula_prior import GaussianCopulaPrior, temper_prior


def _prior(locs, scales, correlation=None, names=None):
    marginals = [stats.norm(loc=m, scale=s) for m, s in zip(locs, scales)]
    names = names or [f"p{i}" for i in range(len(locs))]
    return GaussianCopulaPrior(
        marginals=marginals, correlation=correlation, param_names=names
    )


def _within_clamp(prior, theta, eps=1e-8, margin=10.0):
    """Mask of rows whose marginal CDFs stay clear of log_prob's u clamp.

    ``GaussianCopulaPrior.log_prob`` clamps u to ``[1e-8, 1-1e-8]`` (about 5.6σ)
    before inverting to normal scores, which distorts the *copula* term beyond
    that point. The marginal term is exact everywhere.
    """
    x = theta.double().numpy()
    u = np.column_stack([m.cdf(x[:, j]) for j, m in enumerate(prior._marginals)])
    return np.all((u > eps * margin) & (u < 1 - eps * margin), axis=1)


class TestIsActuallyTempering:
    """The defining property: log π̃(θ) = (1/T)·log π(θ) + const, for all θ."""

    @pytest.mark.parametrize("T", [0.5, 2.0, 5.0])
    def test_log_density_is_scaled_log_prior_plus_constant(self, T):
        R = np.array([[1.0, 0.6, -0.3], [0.6, 1.0, 0.2], [-0.3, 0.2, 1.0]])
        prior = _prior([0.0, 1.0, -2.0], [1.0, 0.5, 2.0], correlation=R)
        tempered = temper_prior(prior, T)

        torch.manual_seed(0)
        theta = prior.sample((2000,))
        lp = prior.log_prob(theta).double().numpy()
        lp_t = tempered.log_prob(theta).double().numpy()
        residual = lp_t - lp / T

        # Restricted to where neither prior's u saturates the clamp. Outside that
        # region the deviation is the clamp's, not tempering's — see
        # TestUClampBoundary.
        keep = _within_clamp(prior, theta) & _within_clamp(tempered, theta)
        assert keep.sum() > 1500, "clamp mask threw away too much of the sample"
        assert np.std(residual[keep]) < 1e-4, (
            f"not a temper: residual std {np.std(residual[keep])}"
        )

    @pytest.mark.parametrize("T", [0.5, 2.0, 5.0, 50.0])
    def test_exact_everywhere_without_a_copula(self, T):
        """With R = I there is no copula term, so the clamp cannot bite."""
        prior = _prior([0.0, 1.0, -2.0], [1.0, 0.5, 2.0])  # independent
        tempered = temper_prior(prior, T)

        torch.manual_seed(0)
        theta = prior.sample((20_000,))
        residual = (
            tempered.log_prob(theta).double().numpy()
            - prior.log_prob(theta).double().numpy() / T
        )
        assert np.std(residual) < 1e-4

    def test_scales_sigma_by_sqrt_T(self):
        prior = _prior([0.0, 3.0], [1.0, 4.0])
        tempered = temper_prior(prior, 9.0)
        stds = [float(m.std()) for m in tempered._marginals]
        assert stds == pytest.approx([3.0, 12.0])

    def test_leaves_location_alone(self):
        prior = _prior([1.5, -2.5], [1.0, 1.0])
        tempered = temper_prior(prior, 4.0)
        means = [float(m.mean()) for m in tempered._marginals]
        assert means == pytest.approx([1.5, -2.5])

    def test_correlation_is_preserved(self):
        R = np.array([[1.0, 0.75], [0.75, 1.0]])
        prior = _prior([0.0, 0.0], [1.0, 2.0], correlation=R)
        tempered = temper_prior(prior, 3.0)
        assert np.allclose(tempered._R, R)

    def test_sampled_covariance_scales_by_T(self):
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        prior = _prior([0.0, 0.0], [1.0, 2.0], correlation=R)
        tempered = temper_prior(prior, 4.0)

        torch.manual_seed(0)
        c0 = np.cov(prior.sample((200_000,)).numpy().T)
        torch.manual_seed(0)
        c1 = np.cov(tempered.sample((200_000,)).numpy().T)

        assert np.allclose(c1, 4.0 * c0, rtol=0.05)


class TestInertness:
    def test_T_one_returns_same_object(self):
        prior = _prior([0.0], [1.0])
        assert temper_prior(prior, 1.0) is prior

    def test_T_one_is_exactly_inert_in_density(self):
        prior = _prior([0.0, 1.0], [1.0, 2.0])
        theta = prior.sample((100,))
        assert torch.allclose(
            prior.log_prob(theta), temper_prior(prior, 1.0).log_prob(theta)
        )


class TestPreservation:
    def test_param_names_carry_over_in_order(self):
        names = ["k_a", "k_b", "k_c"]
        prior = _prior([0.0] * 3, [1.0] * 3, names=names)
        assert temper_prior(prior, 2.0).param_names == names

    def test_pinned_parameter_stays_pinned(self):
        """sqrt(T)*0 == 0: which params vary is a prior claim, not a proposal one."""
        prior = _prior([2.0, 0.0], [1e-9, 1.0])
        tempered = temper_prior(prior, 100.0)
        assert float(tempered._marginals[0].std()) < 1e-6
        assert float(tempered._marginals[1].std()) == pytest.approx(10.0)

    def test_result_is_usable_as_a_prior(self):
        prior = _prior([0.0, 0.0], [1.0, 1.0])
        tempered = temper_prior(prior, 2.0)
        s = tempered.sample((50,))
        assert s.shape == (50, 2)
        assert tempered.log_prob(s).shape == (50,)


class TestReweightRoundTrip:
    """Tempering is the proposal side of the decoupling; the two must compose."""

    def test_reweighting_a_tempered_proposal_recovers_the_prior(self):
        from qsp_inference.inference.importance import reweight_to_prior

        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        prior = _prior([0.0, 1.0], [1.0, 1.0], correlation=R)
        proposal = temper_prior(prior, 4.0)

        theta = proposal.sample((200_000,))
        res = reweight_to_prior(theta, prior, proposal)

        x = theta.numpy()
        mean = (res.weights[:, None] * x).sum(axis=0)
        cov = np.cov(x.T, aweights=res.weights)

        assert mean == pytest.approx([0.0, 1.0], abs=0.05)
        assert cov[0, 0] == pytest.approx(1.0, rel=0.08)
        assert cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]) == pytest.approx(0.5, abs=0.05)

    def test_ess_falls_as_temperature_rises(self):
        from qsp_inference.inference.importance import reweight_to_prior

        prior = _prior([0.0] * 3, [1.0] * 3)
        fracs = []
        for T in (1.0, 1.5, 3.0, 8.0):
            proposal = temper_prior(prior, T)
            theta = proposal.sample((40_000,))
            res = reweight_to_prior(theta, prior, proposal, warn_below=None)
            fracs.append(res.ess_fraction)

        assert fracs[0] == pytest.approx(1.0, rel=1e-6)
        assert fracs == sorted(fracs, reverse=True)


class TestUClampBoundary:
    """Documents where the exact identity stops holding, and why it is benign.

    ``log_prob`` clamps the copula's u to [1e-8, 1-1e-8]. A draw further than
    about 5.6σ into a prior's tail therefore gets a distorted copula term under
    that prior. When reweighting a wide proposal onto a narrow prior this happens
    on the proposal's tail draws — which carry negligible weight, since they sit
    where the target density is vanishing.
    """

    def test_deep_tail_deviates(self):
        R = np.array([[1.0, 0.6], [0.6, 1.0]])
        prior = _prior([0.0, 0.0], [1.0, 1.0], correlation=R)
        tempered = temper_prior(prior, 0.25)  # sigma halved -> tails reached sooner

        # ~7 sigma under the tempered prior: past the clamp.
        theta = torch.tensor([[3.5, 0.0]])
        residual = float(
            tempered.log_prob(theta).double() - prior.log_prob(theta).double() / 0.25
        )
        bulk = torch.zeros(1, 2)
        residual_bulk = float(
            tempered.log_prob(bulk).double() - prior.log_prob(bulk).double() / 0.25
        )
        assert abs(residual - residual_bulk) > 1e-2

    def test_tail_draws_carry_negligible_weight(self):
        """The practical consequence: clamp-affected draws barely enter a summary."""
        from qsp_inference.inference.importance import reweight_to_prior

        R = np.array([[1.0, 0.6], [0.6, 1.0]])
        prior = _prior([0.0, 0.0], [1.0, 1.0], correlation=R)
        proposal = temper_prior(prior, 9.0)

        torch.manual_seed(0)
        theta = proposal.sample((50_000,))
        res = reweight_to_prior(theta, prior, proposal, warn_below=None)

        affected = ~_within_clamp(prior, theta)
        assert affected.sum() > 0, "expected some draws past the clamp"
        assert res.weights[affected].sum() < 1e-6


class TestValidation:
    @pytest.mark.parametrize("T", [0.0, -1.0])
    def test_nonpositive_temperature_raises(self, T):
        prior = _prior([0.0], [1.0])
        with pytest.raises(ValueError, match="must be > 0"):
            temper_prior(prior, T)

    def test_non_normal_marginal_raises(self):
        prior = GaussianCopulaPrior(
            marginals=[stats.lognorm(s=0.5, scale=1.0)], param_names=["k_a"]
        )
        with pytest.raises(ValueError, match="normal marginals"):
            temper_prior(prior, 2.0)

    def test_error_names_the_offending_parameter(self):
        prior = GaussianCopulaPrior(
            marginals=[stats.norm(0, 1), stats.gamma(a=2.0)],
            param_names=["fine", "bad_one"],
        )
        with pytest.raises(ValueError, match="bad_one"):
            temper_prior(prior, 2.0)

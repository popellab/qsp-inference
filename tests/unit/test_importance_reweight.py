"""Tests for proposal -> prior importance reweighting."""

import numpy as np
import pytest
import torch
from scipy import stats

from qsp_inference.inference.importance import (
    ReweightResult,
    effective_sample_size,
    log_importance_weights,
    reweight_to_prior,
    weighted_quantile,
)
from qsp_inference.priors.copula_prior import GaussianCopulaPrior


def _prior(scales, names=None, correlation=None):
    """Independent zero-mean normal marginals with the given scales."""
    marginals = [stats.norm(0.0, s) for s in scales]
    names = names or [f"p{i}" for i in range(len(scales))]
    return GaussianCopulaPrior(
        marginals=marginals, correlation=correlation, param_names=names
    )


class TestIdentity:
    def test_same_object_gives_uniform_weights(self):
        prior = _prior([1.0, 2.0])
        theta = prior.sample((2000,))

        res = reweight_to_prior(theta, prior, prior)

        assert np.allclose(res.weights, 1.0 / 2000)
        assert res.ess == pytest.approx(2000.0, rel=1e-9)
        assert res.ess_fraction == pytest.approx(1.0, rel=1e-9)
        assert not res.is_degenerate

    def test_log_weights_are_zero(self):
        prior = _prior([1.0, 2.0, 0.5])
        theta = prior.sample((500,))
        log_w = log_importance_weights(theta, prior, prior)
        assert np.allclose(log_w, 0.0, atol=1e-4)


class TestCorrectness:
    def test_recovers_target_moments(self):
        """Reweighting a wide proposal onto a narrow target recovers its moments."""
        rng = torch.manual_seed(0)  # noqa: F841
        proposal = _prior([3.0])
        target = _prior([1.0])

        theta = proposal.sample((200_000,))
        res = reweight_to_prior(theta, target, proposal)

        x = theta.numpy().ravel()
        mean = float(np.sum(res.weights * x))
        var = float(np.sum(res.weights * (x - mean) ** 2))

        assert mean == pytest.approx(0.0, abs=0.05)
        assert var == pytest.approx(1.0, rel=0.05)

    def test_recovers_shifted_target(self):
        """A target displaced from the proposal is recovered in the mean."""
        proposal = GaussianCopulaPrior([stats.norm(0.0, 2.0)], param_names=["p0"])
        target = GaussianCopulaPrior([stats.norm(1.0, 1.0)], param_names=["p0"])

        theta = proposal.sample((200_000,))
        res = reweight_to_prior(theta, target, proposal)

        x = theta.numpy().ravel()
        mean = float(np.sum(res.weights * x))
        assert mean == pytest.approx(1.0, abs=0.05)

    def test_weighted_quantile_matches_target(self):
        proposal = _prior([3.0])
        target = _prior([1.0])
        theta = proposal.sample((200_000,))
        res = reweight_to_prior(theta, target, proposal)

        med = weighted_quantile(theta.numpy().ravel(), res.weights, 0.5)
        assert float(med) == pytest.approx(0.0, abs=0.05)

    def test_correlated_target(self):
        """The copula term carries through the weights."""
        R = np.array([[1.0, 0.8], [0.8, 1.0]])
        proposal = _prior([2.0, 2.0])
        target = _prior([1.0, 1.0], correlation=R)

        theta = proposal.sample((200_000,))
        res = reweight_to_prior(theta, target, proposal)

        x = theta.numpy()
        mean = (res.weights[:, None] * x).sum(axis=0)
        cov = np.cov(x.T, aweights=res.weights)
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

        assert mean == pytest.approx([0.0, 0.0], abs=0.06)
        assert corr == pytest.approx(0.8, abs=0.05)


class TestTruncationNormalizer:
    """A constant factor in the proposal density must cancel.

    This is the claim the module relies on to justify evaluating log_prob on the
    untruncated base object when the proposal has been TSNPE-truncated.
    """

    def test_constant_offset_leaves_weights_unchanged(self):
        class _Offset:
            """Proposal whose log-density is shifted by a constant (a normalizer)."""

            def __init__(self, base, log_z):
                self._base, self._log_z = base, log_z
                self.param_names = base.param_names

            def log_prob(self, v):
                return self._base.log_prob(v) - self._log_z

        proposal = _prior([2.0, 2.0])
        target = _prior([1.0, 1.5])
        theta = proposal.sample((5000,))

        plain = reweight_to_prior(theta, target, proposal)
        shifted = reweight_to_prior(theta, target, _Offset(proposal, log_z=3.7))

        # Exact in real arithmetic. The floor here is float32: GaussianCopulaPrior
        # computes log_prob in float64 and casts to float32 on return, so shifting
        # a log-density by a constant perturbs its rounding by ~1 ulp (~1e-7 rel).
        assert np.allclose(plain.weights, shifted.weights, rtol=1e-5, atol=0.0)
        assert plain.ess == pytest.approx(shifted.ess, rel=1e-5)
        # The unnormalized log weights DO differ, by exactly the constant.
        assert np.allclose(shifted.log_weights - plain.log_weights, 3.7, atol=1e-4)


class TestAlignment:
    def test_name_mismatch_raises(self):
        a = _prior([1.0, 1.0], names=["k_a", "k_b"])
        b = _prior([1.0, 1.0], names=["k_a", "k_c"])
        theta = b.sample((10,))
        with pytest.raises(ValueError, match="same parameters in the same order"):
            reweight_to_prior(theta, a, b)

    def test_reordering_raises(self):
        a = _prior([1.0, 1.0], names=["k_a", "k_b"])
        b = _prior([1.0, 1.0], names=["k_b", "k_a"])
        theta = b.sample((10,))
        with pytest.raises(ValueError, match="different order"):
            reweight_to_prior(theta, a, b)

    def test_alignment_check_can_be_disabled(self):
        a = _prior([1.0, 1.0], names=["k_a", "k_b"])
        b = _prior([1.0, 1.0], names=["k_b", "k_a"])
        theta = b.sample((10,))
        res = reweight_to_prior(theta, a, b, check_alignment=False)
        assert res.weights.size == 10

    def test_wrong_dimension_raises(self):
        a = _prior([1.0, 1.0, 1.0])
        b = _prior([1.0, 1.0, 1.0])
        theta = torch.randn(10, 2)
        with pytest.raises(ValueError, match="columns"):
            reweight_to_prior(theta, a, b)


class TestSupport:
    def test_sample_outside_proposal_support_raises(self):
        """theta not drawn from the proposal is caught, not silently weighted."""
        proposal = GaussianCopulaPrior([stats.uniform(0.0, 1.0)], param_names=["p0"])
        target = GaussianCopulaPrior([stats.uniform(0.0, 2.0)], param_names=["p0"])
        theta = torch.tensor([[0.5], [5.0]])  # 5.0 is outside the proposal
        with pytest.raises(ValueError, match="non-finite log-density under the proposal"):
            reweight_to_prior(theta, target, proposal)

    def test_zero_weight_outside_target_support(self):
        """Outside the *target* support is legitimate: that draw just gets weight 0."""
        proposal = GaussianCopulaPrior([stats.uniform(0.0, 2.0)], param_names=["p0"])
        target = GaussianCopulaPrior([stats.uniform(0.0, 1.0)], param_names=["p0"])
        theta = torch.tensor([[0.5], [1.5]])
        res = reweight_to_prior(theta, target, proposal, warn_below=None)
        assert res.weights[0] == pytest.approx(1.0)
        assert res.weights[1] == pytest.approx(0.0)

    def test_disjoint_support_raises(self):
        proposal = GaussianCopulaPrior([stats.uniform(5.0, 1.0)], param_names=["p0"])
        target = GaussianCopulaPrior([stats.uniform(0.0, 1.0)], param_names=["p0"])
        theta = proposal.sample((20,))
        with pytest.raises(ValueError, match="disjoint support"):
            reweight_to_prior(theta, target, proposal, warn_below=None)


class TestESS:
    def test_ess_falls_as_priors_diverge(self):
        proposal = _prior([2.0])
        theta = proposal.sample((20_000,))

        fractions = []
        for scale in (2.0, 1.0, 0.5, 0.25):
            target = _prior([scale])
            res = reweight_to_prior(theta, target, proposal, warn_below=None)
            fractions.append(res.ess_fraction)

        assert fractions == sorted(fractions, reverse=True)
        assert fractions[0] == pytest.approx(1.0, rel=1e-6)

    def test_ess_helper_matches_result(self):
        proposal = _prior([2.0])
        target = _prior([1.0])
        theta = proposal.sample((1000,))
        res = reweight_to_prior(theta, target, proposal, warn_below=None)
        assert effective_sample_size(res.weights) == pytest.approx(res.ess)

    def test_degenerate_warns(self):
        proposal = _prior([5.0])
        target = GaussianCopulaPrior([stats.norm(12.0, 0.05)], param_names=["p0"])
        theta = proposal.sample((2000,))
        with pytest.warns(UserWarning, match="ESS"):
            reweight_to_prior(theta, target, proposal, warn_below=0.05)


class TestWeightedQuantile:
    def test_uniform_weights_match_numpy(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=5000)
        w = np.full(5000, 1 / 5000)
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert weighted_quantile(x, w, q) == pytest.approx(
                np.quantile(x, q), abs=0.02
            )

    def test_vector_of_quantiles(self):
        x = np.arange(100.0)
        w = np.full(100, 0.01)
        out = weighted_quantile(x, w, np.array([0.25, 0.5, 0.75]))
        assert out.shape == (3,)
        assert out[0] < out[1] < out[2]

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="entries"):
            weighted_quantile(np.arange(5.0), np.ones(4), 0.5)


class TestResult:
    def test_summary_mentions_ess(self):
        prior = _prior([1.0])
        theta = prior.sample((100,))
        res = reweight_to_prior(theta, prior, prior)
        assert "ESS" in res.summary()
        assert isinstance(res, ReweightResult)
        assert res.n == 100

    def test_accepts_numpy_input(self):
        prior = _prior([1.0, 1.0])
        theta = prior.sample((50,)).numpy()
        res = reweight_to_prior(theta, prior, prior)
        assert res.weights.size == 50

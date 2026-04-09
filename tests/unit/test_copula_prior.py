"""Tests for qsp_inference.priors.copula_prior."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from qsp_inference.priors.copula_prior import GaussianCopulaPrior, load_copula_prior


def _lognormal_marginal(mu=0.0, sigma=0.5):
    from scipy import stats
    return stats.lognorm(s=sigma, scale=np.exp(mu))


class TestGaussianCopulaPrior:
    def test_sample_shape(self):
        marginals = [_lognormal_marginal(0.0, 0.5), _lognormal_marginal(-1.0, 0.3)]
        prior = GaussianCopulaPrior(marginals=marginals)

        samples = prior.sample((100,))
        assert samples.shape == (100, 2)

    def test_samples_positive_for_lognormal(self):
        marginals = [_lognormal_marginal(0.0, 0.5)]
        prior = GaussianCopulaPrior(marginals=marginals)
        samples = prior.sample((1000,))
        assert (samples > 0).all()

    def test_log_prob_shape(self):
        marginals = [_lognormal_marginal(0.0, 0.5), _lognormal_marginal(-1.0, 0.3)]
        prior = GaussianCopulaPrior(marginals=marginals)

        samples = prior.sample((50,))
        lp = prior.log_prob(samples)
        assert lp.shape == (50,)
        assert torch.isfinite(lp).all()

    def test_log_prob_single_sample(self):
        marginals = [_lognormal_marginal(0.0, 0.5)]
        prior = GaussianCopulaPrior(marginals=marginals)

        sample = prior.sample((1,)).squeeze(0)
        lp = prior.log_prob(sample)
        assert lp.shape == ()

    def test_independent_recovers_marginal_stats(self):
        """Without copula correlation, samples should match marginal statistics."""
        from scipy import stats

        mu, sigma = 0.0, 0.5
        marginals = [_lognormal_marginal(mu, sigma)]
        prior = GaussianCopulaPrior(marginals=marginals)

        samples = prior.sample((20000,))
        expected_median = np.exp(mu)
        assert samples.median().item() == pytest.approx(expected_median, rel=0.1)

    def test_copula_induces_correlation(self):
        """Correlated copula should produce correlated samples."""
        R = np.array([[1.0, 0.8], [0.8, 1.0]])
        marginals = [_lognormal_marginal(0.0, 0.5), _lognormal_marginal(0.0, 0.5)]
        prior = GaussianCopulaPrior(marginals=marginals, correlation=R)

        samples = prior.sample((10000,))
        corr = np.corrcoef(samples[:, 0].numpy(), samples[:, 1].numpy())[0, 1]
        # Rank correlation should be strongly positive
        assert corr > 0.5

    def test_independent_no_correlation(self):
        """Identity copula should produce uncorrelated samples."""
        marginals = [_lognormal_marginal(0.0, 0.5), _lognormal_marginal(1.0, 0.3)]
        prior = GaussianCopulaPrior(marginals=marginals, correlation=np.eye(2))

        samples = prior.sample((10000,))
        corr = np.corrcoef(samples[:, 0].numpy(), samples[:, 1].numpy())[0, 1]
        assert abs(corr) < 0.1

    def test_log_prob_higher_at_mode(self):
        """Log prob should be higher near the marginal modes than in the tails."""
        marginals = [_lognormal_marginal(0.0, 0.3)]
        prior = GaussianCopulaPrior(marginals=marginals)

        near_mode = torch.tensor([[1.0]])  # mode of lognorm(0, 0.3)
        in_tail = torch.tensor([[10.0]])
        assert prior.log_prob(near_mode) > prior.log_prob(in_tail)


class TestLoadCopulaPrior:
    def _write_yaml(self, tmpdir, data):
        from ruamel.yaml import YAML

        yaml = YAML()
        path = Path(tmpdir) / "submodel_priors.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        return path

    def test_load_independent(self):
        data = {
            "metadata": {"n_parameters": 2, "n_samples": 1000},
            "parameters": [
                {"name": "k1", "marginal": {"distribution": "lognormal", "mu": 0.0, "sigma": 0.5, "median": 1.0, "cv": 0.5}},
                {"name": "k2", "marginal": {"distribution": "lognormal", "mu": -1.0, "sigma": 0.3, "median": 0.368, "cv": 0.3}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, data)
            prior, names = load_copula_prior(path)

            assert names == ["k1", "k2"]
            samples = prior.sample((100,))
            assert samples.shape == (100, 2)
            assert (samples > 0).all()

    def test_load_with_copula(self):
        data = {
            "metadata": {"n_parameters": 3, "n_samples": 1000},
            "parameters": [
                {"name": "a", "marginal": {"distribution": "lognormal", "mu": 0.0, "sigma": 0.5, "median": 1.0, "cv": 0.5}},
                {"name": "b", "marginal": {"distribution": "lognormal", "mu": 0.0, "sigma": 0.5, "median": 1.0, "cv": 0.5}},
                {"name": "c", "marginal": {"distribution": "gamma", "shape": 2.0, "scale": 1.0, "median": 1.7, "cv": 0.7}},
            ],
            "copula": {
                "type": "gaussian",
                "parameters": ["a", "b"],
                "correlation": [[1.0, 0.7], [0.7, 1.0]],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, data)
            prior, names = load_copula_prior(path)

            assert names == ["a", "b", "c"]
            samples = prior.sample((10000,))
            assert samples.shape == (10000, 3)

            # a-b should be correlated, a-c should not
            corr_ab = np.corrcoef(samples[:, 0].numpy(), samples[:, 1].numpy())[0, 1]
            corr_ac = np.corrcoef(samples[:, 0].numpy(), samples[:, 2].numpy())[0, 1]
            assert corr_ab > 0.4
            assert abs(corr_ac) < 0.1

    def test_roundtrip_with_parameterizer(self):
        """Write with parameterizer, read back with load_copula_prior."""
        from qsp_inference.submodel.parameterizer import write_priors_yaml

        rng = np.random.default_rng(42)
        # Correlated lognormal
        mean = [0.0, -0.5]
        cov = [[1.0, 0.6], [0.6, 0.5]]
        z = rng.multivariate_normal(mean, cov, 5000)
        s1, s2 = np.exp(z[:, 0]), np.exp(z[:, 1])

        from qsp_inference.submodel.parameterizer import (
            _build_marginal_cdf,
            fit_gaussian_copula,
            fit_marginals,
            threshold_copula,
        )

        samples = {"k1": s1, "k2": s2}
        marginals = fit_marginals(samples)
        param_names = ["k1", "k2"]
        matrix = np.column_stack([s1, s2])
        cdfs = [_build_marginal_cdf(marginals[n]) for n in param_names]
        R = fit_gaussian_copula(matrix, cdfs)
        R_thresh, copula_params = threshold_copula(R, param_names)

        result = {
            "metadata": {"n_parameters": 2, "n_samples": 5000},
            "parameters": [
                {
                    "name": n,
                    "marginal": {
                        "distribution": marginals[n].name,
                        **marginals[n].params,
                        "median": float(marginals[n].median),
                        "cv": float(marginals[n].cv),
                    },
                }
                for n in param_names
            ],
        }
        if copula_params:
            indices = [param_names.index(p) for p in copula_params]
            R_sub = R_thresh[np.ix_(indices, indices)]
            result["copula"] = {
                "type": "gaussian",
                "parameters": copula_params,
                "correlation": R_sub.tolist(),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "priors.yaml"
            write_priors_yaml(result, path)

            prior, names = load_copula_prior(path)
            assert names == ["k1", "k2"]

            new_samples = prior.sample((5000,))
            assert new_samples.shape == (5000, 2)

            # Correlation should be preserved
            corr = np.corrcoef(new_samples[:, 0].numpy(), new_samples[:, 1].numpy())[0, 1]
            assert corr > 0.3


class TestLoadCopulaPriorLog:
    def _write_yaml(self, tmpdir, data):
        from ruamel.yaml import YAML

        yaml = YAML()
        path = Path(tmpdir) / "submodel_priors.yaml"
        with open(path, "w") as f:
            yaml.dump(data, f)
        return path

    def test_log_space_samples_centered_on_mu(self):
        from qsp_inference.priors.copula_prior import load_copula_prior_log

        mu, sigma = 1.5, 0.4
        data = {
            "metadata": {"n_parameters": 1, "n_samples": 1000},
            "parameters": [
                {"name": "k", "marginal": {"distribution": "lognormal", "mu": mu, "sigma": sigma, "median": float(np.exp(mu)), "cv": sigma}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, data)
            prior, names = load_copula_prior_log(path)

            samples = prior.sample((10000,))
            # In log-space, samples should be Normal(mu, sigma)
            assert samples.mean().item() == pytest.approx(mu, abs=0.1)
            assert samples.std().item() == pytest.approx(sigma, abs=0.1)

    def test_log_space_preserves_copula_correlation(self):
        from qsp_inference.priors.copula_prior import load_copula_prior_log

        data = {
            "metadata": {"n_parameters": 2, "n_samples": 1000},
            "parameters": [
                {"name": "a", "marginal": {"distribution": "lognormal", "mu": 0.0, "sigma": 0.5, "median": 1.0, "cv": 0.5}},
                {"name": "b", "marginal": {"distribution": "lognormal", "mu": -1.0, "sigma": 0.3, "median": 0.368, "cv": 0.3}},
            ],
            "copula": {
                "type": "gaussian",
                "parameters": ["a", "b"],
                "correlation": [[1.0, 0.7], [0.7, 1.0]],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, data)
            prior, names = load_copula_prior_log(path)

            samples = prior.sample((10000,))
            corr = np.corrcoef(samples[:, 0].numpy(), samples[:, 1].numpy())[0, 1]
            assert corr > 0.5

    def test_log_prob_finite(self):
        from qsp_inference.priors.copula_prior import load_copula_prior_log

        data = {
            "metadata": {"n_parameters": 2, "n_samples": 1000},
            "parameters": [
                {"name": "a", "marginal": {"distribution": "lognormal", "mu": 0.0, "sigma": 0.5, "median": 1.0, "cv": 0.5}},
                {"name": "b", "marginal": {"distribution": "lognormal", "mu": -1.0, "sigma": 0.3, "median": 0.368, "cv": 0.3}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, data)
            prior, _ = load_copula_prior_log(path)

            samples = prior.sample((100,))
            lp = prior.log_prob(samples)
            assert lp.shape == (100,)
            assert torch.isfinite(lp).all()

    def test_rejects_non_lognormal(self):
        from qsp_inference.priors.copula_prior import load_copula_prior_log

        data = {
            "metadata": {"n_parameters": 1, "n_samples": 1000},
            "parameters": [
                {"name": "k", "marginal": {"distribution": "gamma", "shape": 2.0, "scale": 1.0, "median": 1.7, "cv": 0.7}},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = self._write_yaml(tmpdir, data)
            with pytest.raises(ValueError, match="not supported"):
                load_copula_prior_log(path)

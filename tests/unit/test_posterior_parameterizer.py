"""Tests for qsp_inference.submodel.parameterizer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from qsp_inference.submodel.parameterizer import (
    _build_marginal_cdf,
    fit_gaussian_copula,
    fit_marginals,
    parameterize_posteriors,
    threshold_copula,
    write_priors_yaml,
)


# =============================================================================
# Helpers
# =============================================================================


def _make_lognormal_samples(mu, sigma, n=5000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.lognormal(mu, sigma, n)


def _make_correlated_lognormal_samples(n=5000, seed=42):
    """Generate two correlated lognormal parameter samples."""
    rng = np.random.default_rng(seed)
    # Correlated normal in log-space
    mean = [0.0, -0.5]
    cov = [[1.0, 0.6], [0.6, 0.5]]
    z = rng.multivariate_normal(mean, cov, n)
    return np.exp(z[:, 0]), np.exp(z[:, 1])


# =============================================================================
# Tests: Marginal fitting
# =============================================================================


class TestFitMarginals:
    def test_lognormal_samples(self):
        samples = {"k": _make_lognormal_samples(mu=0.0, sigma=0.5)}
        marginals = fit_marginals(samples)

        assert "k" in marginals
        fit = marginals["k"]
        assert fit.name == "lognormal"
        assert fit.params["mu"] == pytest.approx(0.0, abs=0.1)
        assert fit.params["sigma"] == pytest.approx(0.5, abs=0.1)

    def test_multiple_parameters(self):
        samples = {
            "k1": _make_lognormal_samples(mu=0.0, sigma=0.5, seed=1),
            "k2": _make_lognormal_samples(mu=1.0, sigma=0.3, seed=2),
        }
        marginals = fit_marginals(samples)
        assert len(marginals) == 2
        assert marginals["k1"].median == pytest.approx(1.0, rel=0.15)
        assert marginals["k2"].median == pytest.approx(np.exp(1.0), rel=0.15)


# =============================================================================
# Tests: Gaussian copula
# =============================================================================


class TestFitGaussianCopula:
    def test_correlated_samples(self):
        s1, s2 = _make_correlated_lognormal_samples(n=10000)
        samples = {"a": s1, "b": s2}
        marginals = fit_marginals(samples)

        matrix = np.column_stack([s1, s2])
        cdfs = [_build_marginal_cdf(marginals[n]) for n in ["a", "b"]]
        R = fit_gaussian_copula(matrix, cdfs)

        assert R.shape == (2, 2)
        # Diagonal should be 1
        assert R[0, 0] == pytest.approx(1.0, abs=0.01)
        assert R[1, 1] == pytest.approx(1.0, abs=0.01)
        # Should recover positive correlation
        assert R[0, 1] > 0.3
        assert R[0, 1] == pytest.approx(R[1, 0], abs=0.01)

    def test_independent_samples(self):
        rng = np.random.default_rng(42)
        s1 = rng.lognormal(0, 0.5, 10000)
        s2 = rng.lognormal(1, 0.3, 10000)
        samples = {"a": s1, "b": s2}
        marginals = fit_marginals(samples)

        matrix = np.column_stack([s1, s2])
        cdfs = [_build_marginal_cdf(marginals[n]) for n in ["a", "b"]]
        R = fit_gaussian_copula(matrix, cdfs)

        # Off-diagonal should be near zero
        assert abs(R[0, 1]) < 0.05


class TestThresholdCopula:
    def test_small_correlations_zeroed(self):
        R = np.array(
            [
                [1.0, 0.03, 0.5],
                [0.03, 1.0, -0.02],
                [0.5, -0.02, 1.0],
            ]
        )
        R_thresh, participants = threshold_copula(R, ["a", "b", "c"], threshold=0.05)

        # a-b and b-c should be zeroed out
        assert R_thresh[0, 1] == 0.0
        assert R_thresh[1, 0] == 0.0
        assert R_thresh[1, 2] == 0.0
        # a-c should be kept
        assert R_thresh[0, 2] == 0.5
        # Only a and c participate
        assert set(participants) == {"a", "c"}

    def test_all_independent(self):
        R = np.eye(3)
        R[0, 1] = R[1, 0] = 0.01
        R_thresh, participants = threshold_copula(R, ["a", "b", "c"], threshold=0.05)
        assert participants == []


# =============================================================================
# Tests: Full parameterization
# =============================================================================


class TestParameterizePosteriors:
    def test_single_parameter(self):
        samples = {"k": _make_lognormal_samples(mu=0.0, sigma=0.5)}
        result = parameterize_posteriors(samples, targets=[])

        assert "metadata" in result
        assert "parameters" in result
        assert len(result["parameters"]) == 1
        assert result["parameters"][0]["name"] == "k"
        assert result["parameters"][0]["marginal"]["distribution"] == "lognormal"

    def test_two_correlated_parameters(self):
        s1, s2 = _make_correlated_lognormal_samples(n=10000)
        samples = {"a": s1, "b": s2}
        result = parameterize_posteriors(samples, targets=[], copula_threshold=0.05)

        assert "copula" in result
        assert result["copula"]["type"] == "gaussian"
        assert len(result["copula"]["parameters"]) == 2
        # Correlation matrix should be 2x2
        assert len(result["copula"]["correlation"]) == 2

    def test_metadata_fields(self):
        samples = {"k": _make_lognormal_samples(mu=0.0, sigma=0.5)}
        result = parameterize_posteriors(
            samples,
            targets=[],
            mcmc_config={"num_chains": 4, "num_samples": 5000},
        )

        meta = result["metadata"]
        assert meta["n_parameters"] == 1
        assert meta["num_chains"] == 4
        assert "timestamp" in meta


# =============================================================================
# Tests: YAML roundtrip
# =============================================================================


class TestWritePriorsYaml:
    def test_write_and_read(self):
        samples = {"k": _make_lognormal_samples(mu=0.0, sigma=0.5)}
        result = parameterize_posteriors(samples, targets=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_priors.yaml"
            write_priors_yaml(result, path)

            assert path.exists()

            # Read back
            from ruamel.yaml import YAML

            yaml = YAML()
            loaded = yaml.load(path)

            assert "metadata" in loaded
            assert "parameters" in loaded
            assert loaded["parameters"][0]["name"] == "k"

    def test_write_with_copula(self):
        s1, s2 = _make_correlated_lognormal_samples(n=10000)
        samples = {"a": s1, "b": s2}
        result = parameterize_posteriors(samples, targets=[])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_priors.yaml"
            write_priors_yaml(result, path)

            from ruamel.yaml import YAML

            yaml = YAML()
            loaded = yaml.load(path)

            assert "copula" in loaded
            assert loaded["copula"]["type"] == "gaussian"


# =============================================================================
# Tests: Audit report submodel priors export
# =============================================================================


class TestWriteSubmodelPriors:
    def test_writes_yaml_with_marginals_and_copula(self):
        from qsp_inference.audit.report import _write_submodel_priors

        s1, s2 = _make_correlated_lognormal_samples(n=5000)
        joint_samples = {
            "k_kill": s1.tolist(),
            "k_growth": s2.tolist(),
        }
        targets = {
            "k_kill": ["target_immune"],
            "k_growth": ["target_tumor", "target_immune"],
        }
        groups = {"k_kill": "killing_rates"}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submodel_priors.yaml"
            _write_submodel_priors(joint_samples, targets, groups, path)

            assert path.exists()

            from ruamel.yaml import YAML

            yaml = YAML()
            loaded = yaml.load(path)

            assert loaded["metadata"]["n_parameters"] == 2
            assert len(loaded["parameters"]) == 2

            # Check provenance fields
            by_name = {p["name"]: p for p in loaded["parameters"]}
            assert by_name["k_kill"]["source_targets"] == ["target_immune"]
            assert by_name["k_kill"]["group"] == "killing_rates"
            assert "group" not in by_name["k_growth"]

            # Should have copula for correlated params
            assert "copula" in loaded

    def test_filters_nuisance_and_hyperparams(self):
        from qsp_inference.audit.report import _write_submodel_priors

        joint_samples = {
            "k_real": _make_lognormal_samples(mu=0.0, sigma=0.5).tolist(),
            "k_nuisance": _make_lognormal_samples(mu=0.0, sigma=0.5, seed=2).tolist(),
            "grp__base": _make_lognormal_samples(mu=0.0, sigma=0.3, seed=3).tolist(),
            "grp__tau": _make_lognormal_samples(mu=0.0, sigma=0.2, seed=4).tolist(),
        }
        # Only k_real is in the targets dict (non-nuisance)
        targets = {"k_real": ["target_1"]}
        groups = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submodel_priors.yaml"
            _write_submodel_priors(joint_samples, targets, groups, path)

            from ruamel.yaml import YAML

            yaml = YAML()
            loaded = yaml.load(path)

            assert len(loaded["parameters"]) == 1
            assert loaded["parameters"][0]["name"] == "k_real"

    def test_skips_when_no_samples(self):
        from qsp_inference.audit.report import _write_submodel_priors

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submodel_priors.yaml"
            _write_submodel_priors({}, {"k": ["t1"]}, {}, path)
            assert not path.exists()

    def test_differing_sample_sizes(self):
        """Parameters with different sample counts (from different components) should not crash."""
        import yaml

        from qsp_inference.audit.report import _write_submodel_priors

        rng = np.random.default_rng(99)
        # Simulate two components with different sample counts
        joint_samples = {
            "k_A": rng.lognormal(0, 0.5, 8000).tolist(),  # NUTS 2-chain
            "k_B": rng.lognormal(-1, 0.7, 8000).tolist(),  # same component as A
            "k_C": rng.lognormal(1, 0.3, 4000).tolist(),  # NPE component
        }
        targets = {"k_A": ["t1"], "k_B": ["t1"], "k_C": ["t2"]}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "submodel_priors.yaml"
            _write_submodel_priors(joint_samples, targets, {}, path)
            assert path.exists()

            with open(path) as f:
                loaded = yaml.safe_load(f)

            assert len(loaded["parameters"]) == 3

            # k_A and k_B are from the same component (8000 samples) —
            # they may have nonzero copula correlation
            # k_C is from a different component — cross-component
            # correlations should be ~0 (below threshold)
            param_names_out = [p["name"] for p in loaded["parameters"]]
            assert "k_A" in param_names_out
            assert "k_B" in param_names_out
            assert "k_C" in param_names_out

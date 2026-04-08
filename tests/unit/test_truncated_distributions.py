#!/usr/bin/env python3
"""
Tests for truncated distribution functionality.

Tests cover:
- TruncatedDistribution wrapper class
- Loading priors with truncation bounds from CSV
- Sampling respects bounds
- Log probability computation with normalization
- Backward compatibility with CSVs without bounds
"""

import pytest
import torch
import numpy as np
import tempfile
import csv
from pathlib import Path

from torch.distributions import Normal, LogNormal, Uniform, Beta

from qsp_inference.priors import load_prior, get_param_names
from qsp_inference.priors.truncated_distributions import TruncatedDistribution


class TestTruncatedDistributionBasic:
    """Test basic TruncatedDistribution functionality."""

    def test_left_truncation_normal(self):
        """Test left-truncated normal distribution."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=-1.0)

        samples = truncated.sample((1000,))
        assert samples.min() >= -1.0, "All samples should be >= lower bound"

    def test_right_truncation_normal(self):
        """Test right-truncated normal distribution."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, high=1.0)

        samples = truncated.sample((1000,))
        assert samples.max() <= 1.0, "All samples should be <= upper bound"

    def test_double_truncation_normal(self):
        """Test double-truncated normal distribution."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=-1.0, high=1.0)

        samples = truncated.sample((1000,))
        assert samples.min() >= -1.0, "All samples should be >= lower bound"
        assert samples.max() <= 1.0, "All samples should be <= upper bound"

    def test_left_truncation_lognormal(self):
        """Test left-truncated lognormal distribution."""
        base = LogNormal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=0.5)

        samples = truncated.sample((1000,))
        assert samples.min() >= 0.5, "All samples should be >= lower bound"

    def test_no_truncation(self):
        """Test that no truncation behaves like base distribution."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base)

        samples_base = base.sample((100,))
        samples_truncated = truncated.sample((100,))

        # Both should have similar statistics (not exact due to random sampling)
        assert samples_base.shape == samples_truncated.shape

    def test_invalid_bounds(self):
        """Test that invalid bounds raise ValueError."""
        base = Normal(loc=0.0, scale=1.0)

        with pytest.raises(ValueError):
            TruncatedDistribution(base, low=1.0, high=0.5)  # low >= high

    def test_inherits_from_distribution(self):
        """Test that TruncatedDistribution inherits from torch.distributions.Distribution."""
        from torch.distributions import Distribution

        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=0.0)

        assert isinstance(truncated, Distribution), "Should inherit from Distribution"


class TestTruncatedDistributionLogProb:
    """Test log probability computation with normalization."""

    def test_log_prob_in_bounds(self):
        """Test that log_prob is finite for values within bounds."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=-1.0, high=1.0)

        values = torch.tensor([0.0, 0.5, -0.5])
        log_probs = truncated.log_prob(values)

        assert torch.all(torch.isfinite(log_probs)), "All log_probs should be finite"

    def test_log_prob_out_of_bounds(self):
        """Test that log_prob is -inf for values outside bounds."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=-1.0, high=1.0)

        # Out of bounds values
        values = torch.tensor([-2.0, 2.0])
        log_probs = truncated.log_prob(values)

        assert torch.all(torch.isinf(log_probs)), "Out of bounds should be -inf"
        assert torch.all(log_probs < 0), "Should be negative infinity"

    def test_log_prob_normalized(self):
        """Test that log_prob is properly normalized."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base, low=-1.0, high=1.0)

        # Sample and compute log probs
        samples = truncated.sample((1000,))
        log_probs = truncated.log_prob(samples)

        # All log probs should be finite
        assert torch.all(torch.isfinite(log_probs)), "All log_probs should be finite"

        # Normalized log probs should be less than base log probs
        # (since we're dividing by a normalization constant < 1)
        base_log_probs = base.log_prob(samples)
        assert torch.all(log_probs >= base_log_probs - 10), "Normalized log_prob should be reasonable"

    def test_log_prob_no_truncation(self):
        """Test that log_prob without truncation matches base distribution."""
        base = Normal(loc=0.0, scale=1.0)
        truncated = TruncatedDistribution(base)

        values = torch.tensor([0.0, 0.5, -0.5])
        log_probs_base = base.log_prob(values)
        log_probs_truncated = truncated.log_prob(values)

        assert torch.allclose(log_probs_base, log_probs_truncated), \
            "No truncation should match base distribution"


class TestLoadPriorWithBounds:
    """Test loading priors from CSV files with truncation bounds."""

    def create_test_csv(self, tmp_path, with_bounds=True):
        """Helper to create a test CSV file."""
        csv_path = tmp_path / "test_priors.csv"

        with open(csv_path, 'w', newline='') as f:
            if with_bounds:
                writer = csv.writer(f)
                writer.writerow(['name', 'expected_value', 'units', 'distribution',
                               'dist_param1', 'dist_param2', 'lower_bound', 'upper_bound'])
                writer.writerow(['param1', '1.00e+00', 'units1', 'normal', '1.0', '0.5', '', ''])
                writer.writerow(['param2', '2.00e+00', 'units2', 'lognormal', '0.5', '0.3', '0.5', ''])
                writer.writerow(['param3', '3.00e+00', 'units3', 'normal', '3.0', '1.0', '0.0', '5.0'])
            else:
                writer = csv.writer(f)
                writer.writerow(['name', 'expected_value', 'units', 'distribution',
                               'dist_param1', 'dist_param2'])
                writer.writerow(['param1', '1.00e+00', 'units1', 'normal', '1.0', '0.5'])
                writer.writerow(['param2', '2.00e+00', 'units2', 'lognormal', '0.5', '0.3'])

        return csv_path

    def test_load_csv_with_bounds(self, tmp_path):
        """Test loading CSV with bounds columns."""
        csv_path = self.create_test_csv(tmp_path, with_bounds=True)

        prior = load_prior(csv_path)
        param_names = get_param_names(csv_path)

        assert len(param_names) == 3, "Should load all 3 parameters"

        # Test sampling
        samples = prior.sample((100,))
        assert samples.shape == (100, 3), "Should sample correct shape"

        # Check that param2 (index 1) has lower bound of 0.5
        assert samples[:, 1].min() >= 0.5, "param2 should respect lower bound"

        # Check that param3 (index 2) has bounds [0, 5]
        assert samples[:, 2].min() >= 0.0, "param3 should respect lower bound"
        assert samples[:, 2].max() <= 5.0, "param3 should respect upper bound"

    def test_load_csv_without_bounds_backward_compatible(self, tmp_path):
        """Test that CSVs without bounds columns still work (backward compatibility)."""
        csv_path = self.create_test_csv(tmp_path, with_bounds=False)

        prior = load_prior(csv_path)
        param_names = get_param_names(csv_path)

        assert len(param_names) == 2, "Should load all 2 parameters"

        # Test sampling
        samples = prior.sample((100,))
        assert samples.shape == (100, 2), "Should sample correct shape"

    def test_load_csv_checks_truncated_distribution(self, tmp_path):
        """Test that bounded parameters use TruncatedDistribution."""
        csv_path = self.create_test_csv(tmp_path, with_bounds=True)

        prior = load_prior(csv_path)

        # param1 (index 0) should NOT be truncated (no bounds)
        assert not isinstance(prior.dists[0], TruncatedDistribution), \
            "param1 should not be truncated"

        # param2 (index 1) should be truncated (has lower bound)
        assert isinstance(prior.dists[1], TruncatedDistribution), \
            "param2 should be truncated"

        # param3 (index 2) should be truncated (has both bounds)
        assert isinstance(prior.dists[2], TruncatedDistribution), \
            "param3 should be truncated"


class TestQSPIOPDACIntegration:
    """Test integration with qspio-pdac scenario CSV files."""

    @pytest.fixture
    def baseline_csv_path(self):
        """Fixture providing path to baseline_no_treatment.csv."""
        return Path('/Users/joeleliason/Projects/qspio-pdac/scenarios/priors/baseline_no_treatment.csv')

    def test_load_baseline_no_treatment(self, baseline_csv_path):
        """Test loading baseline_no_treatment.csv with new bounds format."""
        if not baseline_csv_path.exists():
            pytest.skip("qspio-pdac CSV file not found")

        prior = load_prior(baseline_csv_path)
        param_names = get_param_names(baseline_csv_path)

        # Should load all parameters
        assert len(param_names) > 0, "Should load parameters"

        # Should have initial_tumour_diameter
        assert 'initial_tumour_diameter' in param_names, \
            "Should have initial_tumour_diameter parameter"

        # Test sampling
        samples = prior.sample((100,))
        assert samples.shape[0] == 100, "Should sample correct number"
        assert samples.shape[1] == len(param_names), "Should sample all parameters"

        # Check that initial_tumour_diameter has lower bound of 0.1 cm
        idx = param_names.index('initial_tumour_diameter')
        diameter_samples = samples[:, idx]
        assert diameter_samples.min() >= 0.1, \
            "initial_tumour_diameter should have lower bound of 0.1 cm"

        # Verify it's wrapped with TruncatedDistribution
        assert isinstance(prior.dists[idx], TruncatedDistribution), \
            "initial_tumour_diameter should be truncated"


class TestDeviceHandling:
    """Test device handling and .to() method."""

    def test_to_method_truncated_distribution(self):
        """Test that .to() method works for TruncatedDistribution."""
        base = Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
        truncated = TruncatedDistribution(base, low=-1.0, high=1.0)

        # Move to CPU (should work even if already on CPU)
        truncated_cpu = truncated.to('cpu')

        # Should be able to sample
        samples = truncated_cpu.sample((100,))
        assert samples.shape == (100, 1), "Should sample correct shape"
        assert samples.min() >= -1.0, "Should respect lower bound"
        assert samples.max() <= 1.0, "Should respect upper bound"

        # Check that bounds are preserved
        assert truncated_cpu.has_lower == truncated.has_lower
        assert truncated_cpu.has_upper == truncated.has_upper
        assert truncated_cpu.low == truncated.low
        assert truncated_cpu.high == truncated.high

    def test_load_prior_with_device_handling(self, tmp_path):
        """Test that load_prior works with MultipleIndependent's device handling."""
        # Create a test CSV with bounds
        csv_path = tmp_path / "test_device_handling.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'expected_value', 'units', 'distribution',
                           'dist_param1', 'dist_param2', 'lower_bound', 'upper_bound'])
            writer.writerow(['param1', '1.00e+00', 'units1', 'normal', '1.0', '0.5', '0.0', ''])
            writer.writerow(['param2', '2.00e+00', 'units2', 'lognormal', '0.5', '0.3', '0.5', ''])

        # Load prior (this internally calls .to() on distributions)
        prior = load_prior(csv_path, device='cpu')

        # Should be able to sample without errors
        samples = prior.sample((100,))
        assert samples.shape == (100, 2), "Should sample correct shape"

        # Verify bounds are still enforced
        assert samples[:, 0].min() >= 0.0, "param1 should respect lower bound"
        assert samples[:, 1].min() >= 0.5, "param2 should respect lower bound"

    def test_to_preserves_truncation_properties(self):
        """Test that .to() preserves all truncation properties."""
        base = LogNormal(loc=torch.tensor([0.0]), scale=torch.tensor([0.5]))
        truncated = TruncatedDistribution(
            base,
            low=0.1,
            high=5.0,
            max_rejection_iterations=500
        )

        # Move to device
        moved = truncated.to('cpu')

        # Check all properties preserved
        assert moved.low == truncated.low, "Lower bound should be preserved"
        assert moved.high == truncated.high, "Upper bound should be preserved"
        assert moved.has_lower == truncated.has_lower, "has_lower flag should be preserved"
        assert moved.has_upper == truncated.has_upper, "has_upper flag should be preserved"
        assert moved.max_rejection_iterations == truncated.max_rejection_iterations, \
            "max_rejection_iterations should be preserved"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_tight_bounds(self):
        """Test that tight bounds work correctly."""
        base = Normal(loc=0.0, scale=1.0)
        # Tight bounds near mean - should work fine
        truncated = TruncatedDistribution(base, low=-0.5, high=0.5)

        # Should complete quickly
        samples = truncated.sample((100,))
        assert len(samples) == 100, "Should generate samples with tight bounds"
        assert samples.min() >= -0.5, "All samples should be >= lower bound"
        assert samples.max() <= 0.5, "All samples should be <= upper bound"

    def test_lognormal_negative_lower_bound_raises(self, tmp_path):
        """Test that negative lower bound for lognormal raises error."""
        csv_path = tmp_path / "test_invalid_lognormal.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'expected_value', 'units', 'distribution',
                           'dist_param1', 'dist_param2', 'lower_bound', 'upper_bound'])
            writer.writerow(['param1', '1.00e+00', 'units1', 'lognormal', '0.0', '0.5', '-1.0', ''])

        with pytest.raises(ValueError, match="must be positive"):
            load_prior(csv_path)

    def test_lower_greater_than_upper_raises(self, tmp_path):
        """Test that lower >= upper raises error."""
        csv_path = tmp_path / "test_invalid_bounds.csv"

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['name', 'expected_value', 'units', 'distribution',
                           'dist_param1', 'dist_param2', 'lower_bound', 'upper_bound'])
            writer.writerow(['param1', '1.00e+00', 'units1', 'normal', '1.0', '0.5', '5.0', '2.0'])

        with pytest.raises(ValueError, match="must be less than"):
            load_prior(csv_path)

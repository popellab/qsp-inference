#!/usr/bin/env python3
"""
Truncated Distribution Wrapper for PyTorch Distributions

Provides a wrapper class that adds truncation bounds to any PyTorch distribution.
Supports left-truncation (minimum bound), right-truncation (maximum bound), or both.

Usage:
    from torch.distributions import Normal
    from qsp_inference.priors.truncated_distributions import TruncatedDistribution

    # Left-truncated normal (minimum value of 0)
    base_dist = Normal(loc=1.0, scale=0.5)
    truncated_dist = TruncatedDistribution(base_dist, low=0.0)

    # Double-truncated normal (bounded to [0, 2])
    truncated_dist = TruncatedDistribution(base_dist, low=0.0, high=2.0)

    # Sample from truncated distribution
    samples = truncated_dist.sample((1000,))
"""

import torch
from torch.distributions import Distribution
from torch.distributions.constraints import interval, real
from typing import Optional, Tuple


class TruncatedDistribution(Distribution):
    """
    Wrapper that adds truncation bounds to any PyTorch distribution.

    Uses rejection sampling to generate samples within the specified bounds.
    For efficiency, batches rejection sampling when many samples are needed.

    Args:
        base_distribution: The underlying PyTorch distribution to truncate
        low: Lower bound (inclusive). If None, no lower bound.
        high: Upper bound (inclusive). If None, no upper bound.
        max_rejection_iterations: Maximum rejection sampling iterations (default: 1000)

    Raises:
        ValueError: If low >= high when both are specified
        RuntimeError: If rejection sampling fails to find valid samples
    """

    def __init__(
        self,
        base_distribution,
        low: Optional[float] = None,
        high: Optional[float] = None,
        max_rejection_iterations: int = 1000
    ):
        self.base_distribution = base_distribution
        self.low = low if low is not None else -float('inf')
        self.high = high if high is not None else float('inf')
        self.max_rejection_iterations = max_rejection_iterations

        # Initialize parent Distribution class
        super().__init__(
            batch_shape=base_distribution.batch_shape,
            event_shape=base_distribution.event_shape,
            validate_args=False
        )

        # Validate bounds
        if self.low >= self.high:
            raise ValueError(
                f"Lower bound ({self.low}) must be less than upper bound ({self.high})"
            )

        # Store whether we actually have bounds (for efficiency)
        self.has_lower = low is not None
        self.has_upper = high is not None
        self.has_bounds = self.has_lower or self.has_upper

    @property
    def support(self):
        """
        Return the support constraint for this truncated distribution.

        This is required by SBI for building proper MCMC transforms.
        Returns an interval constraint based on the truncation bounds.
        """
        if self.has_lower and self.has_upper:
            return interval(self.low, self.high)
        elif self.has_lower:
            # Only lower bound - use interval with inf upper
            return interval(self.low, float('inf'))
        elif self.has_upper:
            # Only upper bound - use interval with -inf lower
            return interval(float('-inf'), self.high)
        else:
            # No bounds - return real (unbounded)
            return real

    @property
    def mean(self):
        """
        Mean of the truncated distribution.

        For truncated Normal distributions, uses the analytical formula.
        For other distributions, falls back to base distribution mean.
        """
        from torch.distributions import Normal

        # Check if base is Normal - use analytical formula
        if isinstance(self.base_distribution, Normal):
            return self._truncated_normal_mean()

        # Fallback: use base distribution mean (approximate)
        if hasattr(self.base_distribution, 'mean'):
            return self.base_distribution.mean
        else:
            # Last resort: sample-based estimate
            samples = self.sample((1000,))
            return samples.mean(dim=0)

    @property
    def stddev(self):
        """
        Standard deviation of the truncated distribution.

        For truncated Normal distributions, uses the analytical formula.
        For other distributions, falls back to base distribution stddev.
        """
        from torch.distributions import Normal

        # Check if base is Normal - use analytical formula
        if isinstance(self.base_distribution, Normal):
            return self._truncated_normal_stddev()

        # Fallback: use base distribution stddev (approximate)
        if hasattr(self.base_distribution, 'stddev'):
            return self.base_distribution.stddev
        else:
            # Last resort: sample-based estimate
            samples = self.sample((1000,))
            return samples.std(dim=0)

    def _truncated_normal_mean(self):
        """
        Analytical mean for truncated Normal distribution.

        For X ~ TruncatedNormal(μ, σ, a, b):
        E[X] = μ + σ * (φ(α) - φ(β)) / (Φ(β) - Φ(α))

        where α = (a - μ) / σ, β = (b - μ) / σ
        φ is standard normal PDF, Φ is standard normal CDF
        """
        mu = self.base_distribution.loc
        sigma = self.base_distribution.scale

        # Standardize bounds
        alpha = (self.low - mu) / sigma if self.has_lower else torch.tensor(-10.0)
        beta = (self.high - mu) / sigma if self.has_upper else torch.tensor(10.0)

        # Standard normal for PDF and CDF calculations
        std_normal = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

        # φ(α) and φ(β) - standard normal PDF
        phi_alpha = torch.exp(std_normal.log_prob(alpha))
        phi_beta = torch.exp(std_normal.log_prob(beta))

        # Φ(α) and Φ(β) - standard normal CDF
        Phi_alpha = std_normal.cdf(alpha)
        Phi_beta = std_normal.cdf(beta)

        # Normalization constant Z = Φ(β) - Φ(α)
        Z = Phi_beta - Phi_alpha
        Z = torch.clamp(Z, min=1e-10)  # Numerical stability

        # Mean formula
        mean = mu + sigma * (phi_alpha - phi_beta) / Z
        return mean

    def _truncated_normal_stddev(self):
        """
        Analytical standard deviation for truncated Normal distribution.

        For X ~ TruncatedNormal(μ, σ, a, b):
        Var[X] = σ² * [1 + (α*φ(α) - β*φ(β))/(Φ(β) - Φ(α)) - ((φ(α) - φ(β))/(Φ(β) - Φ(α)))²]

        where α = (a - μ) / σ, β = (b - μ) / σ
        """
        mu = self.base_distribution.loc
        sigma = self.base_distribution.scale

        # Standardize bounds
        alpha = (self.low - mu) / sigma if self.has_lower else torch.tensor(-10.0)
        beta = (self.high - mu) / sigma if self.has_upper else torch.tensor(10.0)

        # Standard normal for PDF and CDF calculations
        std_normal = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

        # φ(α) and φ(β) - standard normal PDF
        phi_alpha = torch.exp(std_normal.log_prob(alpha))
        phi_beta = torch.exp(std_normal.log_prob(beta))

        # Φ(α) and Φ(β) - standard normal CDF
        Phi_alpha = std_normal.cdf(alpha)
        Phi_beta = std_normal.cdf(beta)

        # Normalization constant Z = Φ(β) - Φ(α)
        Z = Phi_beta - Phi_alpha
        Z = torch.clamp(Z, min=1e-10)  # Numerical stability

        # Variance formula components
        term1 = (alpha * phi_alpha - beta * phi_beta) / Z
        term2 = ((phi_alpha - phi_beta) / Z) ** 2

        variance = sigma**2 * (1 + term1 - term2)
        variance = torch.clamp(variance, min=1e-10)  # Ensure non-negative

        return torch.sqrt(variance)

    def sample(self, sample_shape: Tuple = torch.Size()) -> torch.Tensor:
        """
        Sample from the truncated distribution using rejection sampling.

        Args:
            sample_shape: Shape of samples to generate

        Returns:
            Samples within the truncation bounds

        Raises:
            RuntimeError: If unable to generate valid samples within max iterations
        """
        if not self.has_bounds:
            # No truncation, just sample from base distribution
            return self.base_distribution.sample(sample_shape)

        # Convert sample_shape to tuple if it's a torch.Size
        if isinstance(sample_shape, torch.Size):
            shape_tuple = tuple(sample_shape)
        else:
            shape_tuple = sample_shape if isinstance(sample_shape, tuple) else (sample_shape,)

        # Calculate total number of samples needed
        if len(shape_tuple) == 0:
            total_samples = 1
        else:
            total_samples = int(torch.prod(torch.tensor(shape_tuple)).item())

        # Rejection sampling with batching for efficiency
        valid_samples = []
        iterations = 0

        while len(valid_samples) < total_samples and iterations < self.max_rejection_iterations:
            # Generate a batch of samples (oversample to reduce iterations)
            batch_size = max(total_samples * 2, 1000)
            candidate_samples = self.base_distribution.sample((batch_size,))

            # Apply bounds check
            if self.has_lower and self.has_upper:
                mask = (candidate_samples >= self.low) & (candidate_samples <= self.high)
            elif self.has_lower:
                mask = candidate_samples >= self.low
            else:  # has_upper only
                mask = candidate_samples <= self.high

            # Keep valid samples
            valid_batch = candidate_samples[mask]
            valid_samples.append(valid_batch)

            iterations += 1

        # Concatenate all valid samples
        if len(valid_samples) == 0:
            raise RuntimeError(
                f"Rejection sampling failed after {iterations} iterations. "
                f"Bounds [{self.low}, {self.high}] may be too restrictive for the "
                f"base distribution."
            )

        all_valid = torch.cat(valid_samples, dim=0)

        # Take exactly the number we need
        if len(all_valid) < total_samples:
            raise RuntimeError(
                f"Only generated {len(all_valid)}/{total_samples} valid samples "
                f"after {iterations} iterations. Bounds [{self.low}, {self.high}] "
                f"may be too restrictive."
            )

        result = all_valid[:total_samples]

        # Reshape to match the shape that base distribution would have returned
        # Base distribution returns shape: sample_shape + batch_shape + event_shape
        expected_shape = self.base_distribution.sample(sample_shape).shape
        result = result.reshape(expected_shape)

        return result

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """
        Compute normalized log probability for truncated distribution.

        The truncated distribution has PDF:
            f_truncated(x) = f_base(x) / Z
        where Z = CDF_base(high) - CDF_base(low) is the normalization constant.

        Therefore:
            log p_truncated(x) = log p_base(x) - log(Z)

        Args:
            value: Values to compute log probability for

        Returns:
            Normalized log probabilities (-inf for out-of-bounds values)
        """
        if not self.has_bounds:
            # No truncation, just return base log prob
            return self.base_distribution.log_prob(value)

        # Get device from value
        device = value.device if isinstance(value, torch.Tensor) else self._get_device()

        # Check bounds
        if self.has_lower and self.has_upper:
            in_bounds = (value >= self.low) & (value <= self.high)
        elif self.has_lower:
            in_bounds = value >= self.low
        elif self.has_upper:
            in_bounds = value <= self.high
        else:
            in_bounds = torch.ones_like(value, dtype=torch.bool)

        # Compute base log prob
        base_log_prob = self.base_distribution.log_prob(value)

        # Compute normalization constant: log(CDF(high) - CDF(low))
        # Convert bounds to tensors on the correct device
        if self.has_lower:
            low_tensor = torch.tensor(self.low, device=device)
            cdf_low = self.base_distribution.cdf(low_tensor)
        else:
            cdf_low = torch.tensor(0.0, device=device)

        if self.has_upper:
            high_tensor = torch.tensor(self.high, device=device)
            cdf_high = self.base_distribution.cdf(high_tensor)
        else:
            cdf_high = torch.tensor(1.0, device=device)

        # Normalization constant
        # Add small epsilon for numerical stability
        normalizer = cdf_high - cdf_low
        eps = torch.tensor(1e-10, device=device)
        normalizer = torch.maximum(normalizer, eps)
        log_normalizer = torch.log(normalizer)

        # Normalized log prob
        normalized_log_prob = base_log_prob - log_normalizer

        # Set out-of-bounds to -inf
        result = torch.where(
            in_bounds,
            normalized_log_prob,
            torch.tensor(-float('inf'), device=device)
        )

        return result

    def _get_device(self) -> torch.device:
        """
        Infer device from base distribution parameters.

        Returns:
            torch.device object
        """
        # Try common parameter names
        for param_name in ['loc', 'scale', 'low', 'high', 'concentration1', 'concentration0']:
            if hasattr(self.base_distribution, param_name):
                param = getattr(self.base_distribution, param_name)
                if isinstance(param, torch.Tensor):
                    return param.device

        # Default to CPU
        return torch.device('cpu')

    def to(self, device):
        """
        Move the truncated distribution to a different device.

        This is required for compatibility with sbi's MultipleIndependent,
        which calls .to(device) to move distributions to the correct device.

        Args:
            device: Target device (cpu or cuda)

        Returns:
            New TruncatedDistribution instance on the target device
        """
        # PyTorch distributions have .to() method, so we can just use it
        # For custom distributions without .to(), this will fail gracefully
        try:
            new_base = self.base_distribution.to(device)
        except AttributeError:
            # If base distribution doesn't support .to(), return self
            # (assume it's already device-agnostic)
            return self

        # Create new TruncatedDistribution with moved base distribution
        return TruncatedDistribution(
            base_distribution=new_base,
            low=self.low if self.has_lower else None,
            high=self.high if self.has_upper else None,
            max_rejection_iterations=self.max_rejection_iterations
        )

    def __repr__(self) -> str:
        """String representation of the truncated distribution."""
        bounds_str = f"[{self.low}, {self.high}]"
        return f"TruncatedDistribution({self.base_distribution}, bounds={bounds_str})"

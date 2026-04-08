
#!/usr/bin/env python3
"""
Data Processing Pipeline for SBI Workflows

Functions for processing simulation data for neural posterior estimation,
including data splitting, filtering, and transformations.

Usage:
    from qsp_inference.inference import processed_simulator

    # Generate and process simulations
    theta_train, x_train, theta_test, x_test, obs_quantiles = processed_simulator(
        qsp_simulator=qsp_sim,
        num_simulations=5000,
        test_fraction=0.1,
        split_seed=2027
    )
"""

import numpy as np
import torch
from pathlib import Path
from typing import Dict, Union
from qsp_inference.inference.gaussian_copula_transform import (
    compute_quantiles_from_array,
    transform_to_normal_from_array
)


def get_observed_data(
    test_stats_csv: Union[str, Path],
    value_column: str = 'mean'
) -> Dict[str, np.ndarray]:
    """
    Extract observed data from test statistics CSV as dictionary.

    Args:
        test_stats_csv: Path to test statistics CSV file
        value_column: Column name to use for observed values (default: 'mean')

    Returns:
        Dictionary with observable names as keys and 2D numpy arrays as values.
        Each array has shape (1, 1) for compatibility with BayesFlow workflow.

    Example:
        obs = get_observed_data('projects/pdac_2025/cache/test_stats.csv')
        # Returns: {'tumor_growth_rate': array([[0.0059]])}
    """
    import pandas as pd

    test_stats_csv = Path(test_stats_csv)
    if not test_stats_csv.exists():
        raise FileNotFoundError(f"Test statistics CSV not found: {test_stats_csv}")

    # Read CSV
    df = pd.read_csv(test_stats_csv)

    if value_column not in df.columns:
        raise ValueError(
            f"Column '{value_column}' not found in test statistics CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )

    if 'test_statistic_id' not in df.columns:
        raise ValueError(
            f"Column 'test_statistic_id' not found in test statistics CSV. "
            f"Available columns: {', '.join(df.columns)}"
        )

    # Extract observable names and values
    observable_names = df['test_statistic_id'].tolist()
    observed_values = df[value_column].values

    # Build dictionary with 2D arrays (1, 1) for each observable
    obs_dict = {}
    for i, obs_name in enumerate(observable_names):
        obs_dict[obs_name] = observed_values[i:i+1].reshape(1, 1)

    return obs_dict


def add_observation_noise(
    x: np.ndarray,
    ci95_lower: np.ndarray,
    ci95_upper: np.ndarray,
    medians: np.ndarray,
    seed: int = 42,
    sample_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Add observation noise to simulator outputs based on calibration target uncertainty.

    For each observable, determines whether the CI95 is better described by
    lognormal or Gaussian noise (based on CI asymmetry around the median),
    then draws noise accordingly. This teaches the NPE that observations are
    noisy measurements, so targets with wide CIs constrain less.

    When sample_sizes are provided, the noise scale is divided by sqrt(n) to
    reflect the standard error of the population summary statistic (median)
    rather than the full population spread. The CI95 in calibration targets
    typically represents inter-patient variability, but x_obs is the *median*
    of n patients — its uncertainty is sqrt(n) times smaller.

    Lognormal noise (multiplicative): x_noisy = x * exp(N(0, log_sigma))
        Used when CI95 is approximately symmetric in log-space.
    Gaussian noise (additive): x_noisy = x + N(0, sigma)
        Used when CI95 is approximately symmetric in linear space.

    Args:
        x: Simulator outputs, shape (n_samples, n_observables).
        ci95_lower: Lower CI95 bound per observable, shape (n_observables,).
        ci95_upper: Upper CI95 bound per observable, shape (n_observables,).
        medians: Observed medians per observable, shape (n_observables,).
        seed: Random seed for reproducibility.
        sample_sizes: Sample size per observable, shape (n_observables,).
            When provided, noise scale is divided by sqrt(n) to use
            standard error instead of population spread.

    Returns:
        x_noisy: Noisy simulator outputs, same shape as x.
    """
    n_obs = x.shape[1]
    rng = np.random.default_rng(seed)
    x_noisy = x.copy()

    n_logn = 0
    n_gauss = 0
    n_skip = 0

    for i in range(n_obs):
        lo, hi, med = ci95_lower[i], ci95_upper[i], medians[i]

        # Skip if CI95 is missing or degenerate
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo and lo > 0 and med > 0):
            n_skip += 1
            continue

        # SEM correction: divide noise scale by sqrt(n)
        sqrt_n = np.sqrt(sample_sizes[i]) if sample_sizes is not None else 1.0

        # Determine noise type by CI asymmetry
        log_upper = np.log(hi / med)
        log_lower = np.log(med / lo)
        add_upper = hi - med
        add_lower = med - lo

        log_asymmetry = abs(log_upper - log_lower) / max(log_upper + log_lower, 1e-10)
        add_asymmetry = abs(add_upper - add_lower) / max(add_upper + add_lower, 1e-10)

        if log_asymmetry <= add_asymmetry:
            # Lognormal: CI95 is more symmetric in log-space
            log_sigma = (np.log(hi) - np.log(lo)) / (2 * 1.96) / sqrt_n
            noise = rng.normal(0, log_sigma, size=x.shape[0])
            x_noisy[:, i] = x[:, i] * np.exp(noise)
            n_logn += 1
        else:
            # Gaussian: CI95 is more symmetric in linear space
            sigma = (hi - lo) / (2 * 1.96) / sqrt_n
            noise = rng.normal(0, sigma, size=x.shape[0])
            x_noisy[:, i] = x[:, i] + noise
            n_gauss += 1

    sem_str = " (SEM-scaled)" if sample_sizes is not None else ""
    print(f"  Observation noise{sem_str}: {n_logn} lognormal, {n_gauss} Gaussian, {n_skip} skipped"
          f" (of {n_obs} observables)")

    return x_noisy


def _filter_nans(theta, x, verbose=False):
    """Filter out rows with NaNs in either theta or x."""
    has_nan = np.isnan(theta).any(axis=1) | np.isnan(x).any(axis=1)
    n_nans = has_nan.sum()
    n_before = len(theta)

    if n_nans > 0:
        print(f"  ⚠️  Filtering {n_nans}/{n_before} rows with NaNs ({100*n_nans/n_before:.1f}%)")
        theta = theta[~has_nan]
        x = x[~has_nan]
        print(f"  ✓ {len(theta)} clean samples remaining after NaN filtering")
    else:
        print(f"  ✓ No NaNs found - all {n_before} samples are clean")

    return theta, x


def _split_data(theta, x, test_fraction, seed):
    """Split data into train/test sets."""
    np.random.seed(seed)
    n_samples = theta.shape[0]
    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test

    print(f"  Train/test split: {n_train} train ({100*(1-test_fraction):.0f}%), {n_test} test ({100*test_fraction:.0f}%)")

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return theta[train_idx], theta[test_idx], x[train_idx], x[test_idx]


def _transform_data(theta_train, x_train, theta_test, x_test):
    """Apply Gaussian copula and log transformations, convert to tensors."""
    # Compute quantiles from training data only (prevent leakage)
    obs_quantiles = compute_quantiles_from_array(x_train, n_quantiles=1000)

    # Transform observables to standard normal
    x_train = transform_to_normal_from_array(x_train, obs_quantiles)
    x_test = transform_to_normal_from_array(x_test, obs_quantiles)

    # Log-transform parameters
    theta_train = np.log(theta_train)
    theta_test = np.log(theta_test)

    # Convert to float32 torch tensors
    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    theta_test = torch.tensor(theta_test, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)

    return theta_train, x_train, theta_test, x_test, obs_quantiles


def processed_simulator(qsp_simulator, num_simulations, test_fraction=0.1, split_seed=2027, verbose=False):
    """
    Generate simulations, split data, and apply all transformations for SBI.

    This function:
    1. Generates parameters and runs QSP simulations (returns theta, x as 2D arrays)
    2. Filters out rows with NaNs
    3. Splits data into train/test sets
    4. Computes Gaussian copula quantiles from training observables only
    5. Transforms observables to standard normal (prevents data leakage)
    6. Log-transforms parameters
    7. Converts to float32 torch tensors

    Args:
        qsp_simulator: Callable QSP simulator function
        num_simulations: Total number of simulations to generate
        test_fraction: Fraction of data to use for testing (default: 0.1)
        split_seed: Random seed for reproducible splits (default: 2027)
        verbose: If True, print detailed progress information (default: False)

    Returns:
        tuple: (theta_train, x_train, theta_test, x_test, obs_quantiles)
            - theta_train: Training parameters (log-transformed, torch.Tensor float32)
            - x_train: Training observables (Gaussian copula transformed, torch.Tensor float32)
            - theta_test: Test parameters (log-transformed, torch.Tensor float32)
            - x_test: Test observables (Gaussian copula transformed, torch.Tensor float32)
            - obs_quantiles: Quantiles computed from training data (numpy array, for transforming new observations)
    """
    # Generate simulations
    print(f"\n{'='*80}")
    print(f"GENERATING AND PROCESSING {num_simulations} SIMULATIONS")
    print(f"{'='*80}")
    theta, x = qsp_simulator(num_simulations)
    print(f"  ✓ Received {theta.shape[0]} parameter sets and {x.shape[0]} observable sets from simulator")

    # Filter NaNs
    print(f"\nChecking for NaNs...")
    theta, x = _filter_nans(theta, x, verbose=verbose)

    # Split train/test
    print(f"\nSplitting into train/test (test_fraction={test_fraction}, seed={split_seed})...")
    theta_train, theta_test, x_train, x_test = _split_data(theta, x, test_fraction, split_seed)

    # Transform data (copula, log, tensors)
    print(f"\nApplying transformations (Gaussian copula, log transform)...")
    theta_train, x_train, theta_test, x_test, obs_quantiles = _transform_data(
        theta_train, x_train, theta_test, x_test
    )

    print(f"\n{'='*80}")
    print(f"FINAL PROCESSED DATA")
    print(f"{'='*80}")
    print(f"  Train: {theta_train.shape[0]} samples")
    print(f"  Test:  {theta_test.shape[0]} samples")
    print(f"  Total: {theta_train.shape[0] + theta_test.shape[0]} samples")
    print(f"{'='*80}\n")

    return theta_train, x_train, theta_test, x_test, obs_quantiles


def processed_multi_scenario_simulator(
    simulators: dict,
    scenarios: list,
    num_simulations: int,
    test_fraction: float = 0.1,
    split_seed: int = 2027,
    verbose: bool = False
):
    """
    Generate simulations for multiple scenarios with joint NaN filtering.

    For multi-scenario SBI, we need the SAME parameter sets evaluated under each
    scenario. This function:
    1. Generates simulations for each scenario independently
    2. Finds the intersection of valid (non-NaN) samples across ALL scenarios
    3. Applies joint NaN filtering to keep only samples valid in all scenarios
    4. Splits data into train/test sets (same split across all scenarios)
    5. Applies per-scenario transformations (Gaussian copula, log transform)

    Args:
        simulators: Dictionary mapping scenario names to QSPSimulator instances
        scenarios: List of scenario names (determines concatenation order)
        num_simulations: Total number of simulations to generate per scenario
        test_fraction: Fraction of data to use for testing (default: 0.1)
        split_seed: Random seed for reproducible splits (default: 2027)
        verbose: If True, print detailed progress information (default: False)

    Returns:
        dict: Dictionary with keys:
            - 'theta_train': Training parameters (log-transformed, torch.Tensor)
            - 'theta_test': Test parameters (log-transformed, torch.Tensor)
            - 'scenario_data': Dict mapping scenario -> {
                'x_train': Training observables for this scenario
                'x_test': Test observables for this scenario
                'obs_quantiles': Quantiles for this scenario
              }
            - 'x_train_multi': Concatenated training observables (all scenarios)
            - 'x_test_multi': Concatenated test observables (all scenarios)
    """
    print(f"\n{'='*80}")
    print(f"GENERATING MULTI-SCENARIO DATA WITH JOINT NAN FILTERING")
    print(f"{'='*80}")
    print(f"Scenarios: {scenarios}")
    print(f"Simulations per scenario: {num_simulations}")

    # Step 1: Generate raw simulations for each scenario
    print(f"\n--- Step 1: Generating simulations for each scenario ---")
    raw_data = {}
    for scenario in scenarios:
        print(f"\n  📊 Scenario: {scenario}")
        theta, x = simulators[scenario](num_simulations)
        raw_data[scenario] = {'theta': theta, 'x': x}
        print(f"     Received {theta.shape[0]} samples, {x.shape[1]} observables")

    # Step 2: Find joint valid mask (non-NaN in ALL scenarios)
    print(f"\n--- Step 2: Joint NaN filtering across all scenarios ---")

    # Start with all True mask
    n_samples = raw_data[scenarios[0]]['theta'].shape[0]
    joint_valid_mask = np.ones(n_samples, dtype=bool)

    # Check each scenario and update mask
    for scenario in scenarios:
        theta = raw_data[scenario]['theta']
        x = raw_data[scenario]['x']

        # Find NaNs in this scenario
        has_nan_theta = np.isnan(theta).any(axis=1)
        has_nan_x = np.isnan(x).any(axis=1)
        has_nan = has_nan_theta | has_nan_x

        n_nan_scenario = has_nan.sum()
        print(f"  {scenario}: {n_nan_scenario}/{n_samples} samples have NaNs ({100*n_nan_scenario/n_samples:.1f}%)")

        # Update joint mask
        joint_valid_mask = joint_valid_mask & ~has_nan

    n_valid = joint_valid_mask.sum()
    n_filtered = n_samples - n_valid
    print(f"\n  ✓ Joint filtering: {n_filtered} samples removed ({100*n_filtered/n_samples:.1f}%)")
    print(f"  ✓ {n_valid} samples valid across ALL scenarios")

    if n_valid == 0:
        raise ValueError("No samples are valid across all scenarios! Check for systematic simulation failures.")

    # Step 3: Apply joint mask to all scenarios
    print(f"\n--- Step 3: Applying joint filter to all scenarios ---")
    filtered_data = {}
    for scenario in scenarios:
        theta = raw_data[scenario]['theta'][joint_valid_mask]
        x = raw_data[scenario]['x'][joint_valid_mask]
        filtered_data[scenario] = {'theta': theta, 'x': x}
        print(f"  {scenario}: {theta.shape[0]} samples, {x.shape[1]} observables")

    # Verify all thetas are identical (same parameter draws)
    reference_theta = filtered_data[scenarios[0]]['theta']
    for scenario in scenarios[1:]:
        if not np.allclose(reference_theta, filtered_data[scenario]['theta'], rtol=1e-10):
            print(f"  ⚠️  Warning: theta values differ between scenarios (may indicate different simulation seeds)")

    # Step 4: Split into train/test (same indices for all scenarios)
    print(f"\n--- Step 4: Train/test split (shared across scenarios) ---")
    np.random.seed(split_seed)
    n_samples = reference_theta.shape[0]
    indices = np.random.permutation(n_samples)

    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    print(f"  Train: {n_train} samples ({100*(1-test_fraction):.0f}%)")
    print(f"  Test:  {n_test} samples ({100*test_fraction:.0f}%)")

    # Step 5: Transform each scenario's observables independently
    print(f"\n--- Step 5: Per-scenario transformations ---")
    scenario_data = {}

    for scenario in scenarios:
        theta = filtered_data[scenario]['theta']
        x = filtered_data[scenario]['x']

        # Split
        theta_train = theta[train_idx]
        theta_test = theta[test_idx]
        x_train = x[train_idx]
        x_test = x[test_idx]

        # Compute quantiles from training data only
        obs_quantiles = compute_quantiles_from_array(x_train, n_quantiles=1000)

        # Transform observables to standard normal
        x_train_transformed = transform_to_normal_from_array(x_train, obs_quantiles)
        x_test_transformed = transform_to_normal_from_array(x_test, obs_quantiles)

        # Convert to tensors
        x_train_tensor = torch.tensor(x_train_transformed, dtype=torch.float32)
        x_test_tensor = torch.tensor(x_test_transformed, dtype=torch.float32)

        scenario_data[scenario] = {
            'x_train': x_train_tensor,
            'x_test': x_test_tensor,
            'x_train_raw': x_train,  # Untransformed for diagnostics
            'x_test_raw': x_test,    # Untransformed for diagnostics
            'obs_quantiles': obs_quantiles
        }
        print(f"  {scenario}: x_train={x_train_tensor.shape}, x_test={x_test_tensor.shape}")

    # Step 6: Log-transform and convert theta (shared across scenarios)
    theta_train = np.log(reference_theta[train_idx])
    theta_test = np.log(reference_theta[test_idx])
    theta_train = torch.tensor(theta_train, dtype=torch.float32)
    theta_test = torch.tensor(theta_test, dtype=torch.float32)

    # Step 7: Concatenate observables across scenarios
    print(f"\n--- Step 6: Concatenating observables across scenarios ---")
    x_train_multi = torch.cat([scenario_data[s]['x_train'] for s in scenarios], dim=1)
    x_test_multi = torch.cat([scenario_data[s]['x_test'] for s in scenarios], dim=1)
    print(f"  x_train_multi: {x_train_multi.shape}")
    print(f"  x_test_multi: {x_test_multi.shape}")

    print(f"\n{'='*80}")
    print(f"MULTI-SCENARIO DATA READY")
    print(f"{'='*80}")
    print(f"  theta_train: {theta_train.shape} (shared parameters)")
    print(f"  theta_test: {theta_test.shape}")
    print(f"  x_train_multi: {x_train_multi.shape} (concatenated observables)")
    print(f"  x_test_multi: {x_test_multi.shape}")
    print(f"{'='*80}\n")

    return {
        'theta_train': theta_train,
        'theta_test': theta_test,
        'scenario_data': scenario_data,
        'x_train_multi': x_train_multi,
        'x_test_multi': x_test_multi
    }


def prepare_observed_data(test_stats_csv, observable_names, obs_quantiles):
    """
    Load observed data and transform for posterior sampling.

    Args:
        test_stats_csv: Path to test statistics CSV with observed data
        observable_names: List of observable names (in correct order)
        obs_quantiles: Precomputed quantiles from training data

    Returns:
        tuple: (obs_original, obs_transformed_tensor)
            - obs_original: Original observed data (dict format, for PPCs)
            - obs_transformed_tensor: Transformed data (torch tensor, for posterior sampling)
    """
    # Get observed data in original space
    obs_original = get_observed_data(test_stats_csv=test_stats_csv)

    # Extract observable values as numpy array
    if isinstance(obs_original, dict):
        obs_values = np.array([np.squeeze(obs_original[name]) for name in observable_names])
    else:
        # Assume DataFrame or similar
        obs_values = obs_original[observable_names].values.flatten()

    # Transform to copula space for posterior sampling
    obs_transformed = transform_to_normal_from_array(obs_values[np.newaxis, :], obs_quantiles)[0]
    obs_transformed_tensor = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)

    return obs_original, obs_transformed_tensor


def convert_posterior_samples_to_original_space(posterior_samples, param_names):
    """
    Convert posterior samples from log space to original parameter space.

    Args:
        posterior_samples: Posterior samples in log space (torch tensor)
        param_names: List of parameter names

    Returns:
        dict: Posterior samples in original space {param_name: samples_array}
    """
    # Exponentiate to get original space
    samples_exp = torch.exp(posterior_samples).detach().cpu().numpy()

    # Convert to dict format
    samples_dict = {param_names[i]: samples_exp[:, i] for i in range(len(param_names))}

    return samples_dict

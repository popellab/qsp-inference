#!/usr/bin/env python3
"""
Posterior and Prior Predictive Checks and Simulations for SBI

This module provides utilities for generating prior predictive checks (PPCs from prior)
and posterior predictive samples and simulations from trained SBI models.

Prior Predictive Checks (PPCs from prior):
    - QSP simulator samples parameters from prior p(θ) internally
    - Automatically reuses existing simulations from shared pool when available
    - Computes test statistics from simulations
    - Compare distribution of predictive test statistics to observed test statistics
    - Used to validate model setup BEFORE training

Posterior Predictive Checks (PPCs):
    - Sample parameters from posterior p(θ|x_obs)
    - Run QSP simulations for these parameters
    - Compute test statistics from simulations
    - Compare distribution of predictive test statistics to observed test statistics

Posterior Predictive Simulations:
    - Sample parameters from posterior p(θ|x_obs)
    - Run full QSP simulations to get complete time series
    - Visualize full trajectory distributions (spaghetti plots)

Simulation Pooling:
    Both PPCs and full simulations share the same simulation pool (identified by
    observed data hash). This means simulations can be reused between both functions:
    - Run PPCs first → generates simulations in pool
    - Run full sims later → reuses those simulations from pool
    - Or vice versa!

Usage:
    from qsp_inference.inference import (
        generate_prior_predictive_checks,
        generate_posterior_predictive_checks,
        generate_posterior_predictive_simulations,
        plot_ppc_histograms,
        plot_posterior_predictive_spaghetti,
        transform_to_normal
    )
    from qsp_hpc.simulation import get_observed_data

    # Get observed data in both spaces
    obs_original = get_observed_data(test_stats_csv)
    obs_transformed = transform_to_normal(obs_original.copy(), observable_names, quantiles)

    # Run prior predictive checks (BEFORE training)
    # Note: QSP simulator samples from priors internally
    prior_ppc_results = generate_prior_predictive_checks(
        observed_data=obs_original,
        qsp_simulator=qsp_sim,
        n_samples=200
    )

    # Visualize prior PPCs
    plot_ppc_histograms(prior_ppc_results, title='Prior Predictive Checks')

    # Sample from posterior using TRANSFORMED data (AFTER training)
    posterior_samples = posterior.sample((100,), x=obs_transformed)

    # Run posterior predictive checks using ORIGINAL data for comparison
    ppc_results = generate_posterior_predictive_checks(
        posterior_samples=posterior_samples,
        observed_data=obs_original,  # Original space!
        qsp_simulator=qsp_sim
    )

    # Visualize posterior PPCs
    plot_ppc_histograms(ppc_results, title='Posterior Predictive Checks')

    # Run posterior predictive simulations
    pp_sims = generate_posterior_predictive_simulations(
        posterior_samples=posterior_samples,
        observed_data=obs_original,  # For cache hash only
        qsp_simulator=qsp_sim,
        species_of_interest=['V_T_C1', 'V_T_CD8']
    )

    # Visualize as spaghetti plots
    plot_posterior_predictive_spaghetti(pp_sims, figsize=(16, 8))
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import tempfile
import csv


def generate_prior_predictive_checks(
    observed_data: Dict[str, np.ndarray],
    qsp_simulator,
    n_samples: int = 200
) -> Dict:
    """
    Generate prior predictive checks (PPCs from prior).

    Run simulations with parameters sampled from the prior and compare to observed data.
    This validates that:
    1. The prior can generate reasonable simulations
    2. The observed data falls within the range of prior predictions
    3. The model and simulation infrastructure are working correctly

    The QSP simulator samples parameters from the priors configured in its priors_csv,
    and automatically reuses any existing simulations from the shared simulation pool.

    Args:
        observed_data: Dictionary of observed test statistics IN ORIGINAL SPACE
                      (from get_observed_data, NOT transformed)
        qsp_simulator: QSPSimulator instance for running simulations
        n_samples: Number of samples to draw from prior (default: 200)

    Returns:
        Dictionary containing:
        - 'prior_samples': Parameter samples from prior (n_samples, n_params)
        - 'predictive_test_stats': Test statistics from prior predictive sims (n_samples, n_test_stats)
        - 'observed_test_stats': Observed test statistics (1, n_test_stats)
        - 'test_stat_names': List of test statistic names
        - 'param_names': List of parameter names

    Example:
        # Load observed data
        obs_original = get_observed_data('test_stats.csv')

        # Run prior predictive checks
        prior_ppc_results = generate_prior_predictive_checks(
            observed_data=obs_original,
            qsp_simulator=qsp_sim,
            n_samples=200
        )
    """

    print(f"\n🔮 Generating Prior Predictive Checks")
    print(f"   Observed data: {len(observed_data)} test statistics")
    print(f"   Prior samples: {n_samples}")

    # Run QSP simulations (samples from prior internally and reuses existing simulations)
    print(f"\n1️⃣  Running {n_samples} QSP simulations from prior...")
    print(f"   → Simulator will sample from prior and reuse existing simulations if available")

    theta_prior, observables_prior = qsp_simulator(n_samples)

    print(f"   ✓ Completed {observables_prior.shape[0]} prior predictive simulations")
    print(f"   ✓ Prior samples shape: {theta_prior.shape}")
    print(f"   → Sample 0 parameter values:")
    for i, param_name in enumerate(qsp_simulator.param_names[:5]):
        print(f"      {param_name}: {theta_prior[0, i]:.2e}")
    if len(qsp_simulator.param_names) > 5:
        print(f"      ... ({len(qsp_simulator.param_names) - 5} more parameters)")

    # Convert observed data to array format
    print(f"\n2️⃣  Formatting results...")
    test_stat_names = list(observed_data.keys())
    observed_array = np.array([[observed_data[key].flatten()[0] for key in test_stat_names]])

    print(f"   ✓ Test statistics: {len(test_stat_names)}")
    print(f"   ✓ Predictive shape: {observables_prior.shape}")
    print(f"   ✓ Observed shape: {observed_array.shape}")

    return {
        'prior_samples': theta_prior,
        'predictive_test_stats': observables_prior,
        'observed_test_stats': observed_array,
        'test_stat_names': test_stat_names,
        'param_names': qsp_simulator.param_names
    }


def generate_posterior_predictive_checks(
    posterior_samples: Dict[str, np.ndarray],
    observed_data: Dict[str, np.ndarray],
    qsp_simulator
) -> Dict:
    """
    Generate posterior predictive checks (PPCs).

    IMPORTANT: Pass observed_data in ORIGINAL space (not transformed).
    The posterior_samples should already be computed from the workflow using
    TRANSFORMED observed data for proper conditioning.

    This function:
    1. Extracts parameters from pre-computed posterior samples
    2. Runs QSP simulations for these posterior parameter samples
    3. Computes test statistics from the simulations (in original space)
    4. Compares to observed data (in original space)

    Args:
        posterior_samples: Pre-computed posterior samples from workflow.sample()
                          (dictionary with parameter arrays or 'inference_variables')
        observed_data: Dictionary of observed test statistics IN ORIGINAL SPACE
                      (from get_observed_data, NOT transformed)
        qsp_simulator: QSPSimulator instance for running simulations

    Returns:
        Dictionary containing:
        - 'posterior_samples': Parameter samples from posterior (n_samples, n_params)
        - 'predictive_test_stats': Test statistics from posterior predictive sims (n_samples, n_test_stats)
        - 'observed_test_stats': Observed test statistics (1, n_test_stats)
        - 'test_stat_names': List of test statistic names
        - 'param_names': List of parameter names

    Example:
        # Sample from posterior using TRANSFORMED data
        obs_transformed = transform_to_normal(obs_original, observable_names, quantiles)
        posterior_samples = workflow.sample(conditions=obs_transformed, num_samples=100)

        # Run PPCs using ORIGINAL data for comparison
        ppc_results = generate_posterior_predictive_checks(
            posterior_samples=posterior_samples,
            observed_data=obs_original,  # Use original, not transformed!
            qsp_simulator=qsp_sim
        )
    """

    print(f"\n🔮 Generating Posterior Predictive Checks")
    print(f"   Observed data: {len(observed_data)} test statistics")

    # Compute hash of observed data to ensure cache is specific to this dataset
    import hashlib
    obs_values = []
    for key in sorted(observed_data.keys()):  # Sort for deterministic hash
        val = observed_data[key]
        if isinstance(val, np.ndarray):
            val = val.flatten()[0]
        obs_values.append(float(val))
    obs_array = np.array(obs_values)
    obs_hash = hashlib.sha256(obs_array.tobytes()).hexdigest()[:8]

    print(f"   Observed data hash: {obs_hash} (ensures cache is dataset-specific)")

    # Extract parameter matrix (unstandardized)
    print(f"\n   🔍 Extracting parameters from posterior samples...")
    print(f"   → Posterior samples keys: {list(posterior_samples.keys())[:5]}...")

    if 'inference_variables' in posterior_samples:
        # Combined format: (num_samples, num_params)
        theta_posterior = posterior_samples['inference_variables']
        print(f"   → Using 'inference_variables' array: {theta_posterior.shape}")
        print(f"   → WARNING: Cannot verify parameter order with combined array!")
        print(f"   → Assuming order matches qsp_simulator.param_names")
    else:
        # Individual parameter format - need to stack
        # Get param names from qsp_simulator
        param_names_from_sim = qsp_simulator.param_names
        print(f"   → Stacking individual parameters...")
        print(f"   → Simulator expects {len(param_names_from_sim)} parameters")

        param_arrays = []
        missing_params = []
        for name in param_names_from_sim:
            if name in posterior_samples:
                arr = posterior_samples[name]
                if arr.ndim > 1:
                    arr = arr.flatten()
                param_arrays.append(arr)
            else:
                missing_params.append(name)

        if missing_params:
            print(f"   ⚠️  WARNING: {len(missing_params)} parameters missing from posterior!")
            print(f"   → Missing: {missing_params[:5]}")
            raise ValueError(f"Posterior samples missing parameters: {missing_params}")

        theta_posterior = np.column_stack(param_arrays)
        print(f"   ✓ Successfully extracted all {len(param_names_from_sim)} parameters")

    # Get sample count from extracted matrix
    n_samples = theta_posterior.shape[0]
    print(f"   ✓ Posterior shape: ({n_samples} samples, {theta_posterior.shape[1]} parameters)")

    # CRITICAL: Verify parameter order
    print(f"\n   🔍 Verifying parameter order...")
    print(f"   → Simulator expects parameters in this order:")
    print(f"      {qsp_simulator.param_names[:5]}... ({len(qsp_simulator.param_names)} total)")

    # Show sample values for first few parameters
    print(f"\n   → Sample 0 parameter values:")
    for i, param_name in enumerate(qsp_simulator.param_names[:5]):
        print(f"      {param_name}: {theta_posterior[0, i]:.2e}")
    if len(qsp_simulator.param_names) > 5:
        print(f"      ... ({len(qsp_simulator.param_names) - 5} more parameters)")

    # CRITICAL: Check if samples are in log space (from adapter transformation)
    # BayesFlow workflow.sample() should return samples in ORIGINAL space (with inverse transforms applied)
    # We only need exp() if samples have NEGATIVE values (clear sign of log space)
    # Note: Many biological parameters are < 1.0 naturally (rates, probabilities), so median < 1.0 is NOT a sign of log space!

    if np.any(theta_posterior < 0):
        # Negative values = definitely in log space
        print(f"\n   🔍 DETECTED NEGATIVE VALUES - samples are in LOG space")
        print(f"   → Back-transforming with exp() to original parameter space")
        theta_posterior_original = np.exp(theta_posterior)
        print(f"   → Median before exp(): {np.median(theta_posterior):.2e}")
        print(f"   → Median after exp(): {np.median(theta_posterior_original):.2e}")
    else:
        # No negative values - samples should already be in original space
        print(f"\n   ✓ Samples appear to be in ORIGINAL space (no negative values)")
        print(f"   → Median: {np.median(theta_posterior):.2e}")
        print(f"   → Range: [{theta_posterior.min():.2e}, {theta_posterior.max():.2e}]")
        theta_posterior_original = theta_posterior

    # Use the (possibly) back-transformed samples for simulations
    theta_posterior = theta_posterior_original

    # 2. Run QSP simulations for posterior samples (with caching!)
    print(f"\n2️⃣  Running {n_samples} QSP simulations for posterior samples...")
    print(f"   → Using cached simulation pool (will reuse if available)")

    # Use the new simulate_with_parameters method which checks cache first
    # Include observed data hash in pool suffix to ensure cache is dataset-specific
    # NOTE: This pool is shared with full posterior predictive sims for efficiency!
    pool_suffix = f'ppc_{obs_hash}'  # e.g., 'ppc_a1b2c3d4'
    observables_posterior = qsp_simulator.simulate_with_parameters(
        theta=theta_posterior,
        pool_suffix=pool_suffix
    )

    print(f"   ✓ Completed {observables_posterior.shape[0]} posterior predictive simulations")
    print(f"   ✓ Test statistics: {observables_posterior.shape[1]} observables")

    # Check for NaN issues
    n_nan_rows = np.sum(np.any(np.isnan(observables_posterior), axis=1))
    n_nan_total = np.sum(np.isnan(observables_posterior))
    if n_nan_rows > 0:
        print(f"\n   ⚠️  WARNING: {n_nan_rows}/{observables_posterior.shape[0]} simulations have NaN test statistics")
        print(f"   ⚠️  Total NaN values: {n_nan_total}/{observables_posterior.size} ({100*n_nan_total/observables_posterior.size:.1f}%)")

        # Check which test statistics have the most NaNs
        nan_counts = np.sum(np.isnan(observables_posterior), axis=0)
        test_stats_df = pd.read_csv(qsp_simulator.test_stats_csv)
        test_stat_names_local = test_stats_df['test_statistic_id'].tolist()

        print(f"\n   Test statistics with NaNs:")
        for i, count in enumerate(nan_counts):
            if count > 0:
                print(f"     {test_stat_names_local[i]}: {count}/{observables_posterior.shape[0]} ({100*count/observables_posterior.shape[0]:.1f}%)")

        # Check parameter ranges
        print(f"\n   Posterior parameter ranges:")
        for i, param_name in enumerate(qsp_simulator.param_names[:5]):  # Show first 5
            param_vals = theta_posterior[:, i]
            print(f"     {param_name}: [{param_vals.min():.2e}, {param_vals.max():.2e}]")
        if len(qsp_simulator.param_names) > 5:
            print(f"     ... ({len(qsp_simulator.param_names) - 5} more parameters)")

        print(f"\n   💡 Possible causes:")
        print(f"      1. Posterior samples in regions that cause simulation failures")
        print(f"      2. Numerical issues in QSP model with these parameter values")
        print(f"      3. Test statistic computation errors (e.g., log of negative values)")
        print(f"      4. Check posterior_samples to see if values are reasonable")

    # 3. Extract observed test statistics
    print(f"\n3️⃣  Extracting observed test statistics...")

    # Get test statistic names from qsp_simulator
    test_stats_df = pd.read_csv(qsp_simulator.test_stats_csv)
    test_stat_names = test_stats_df['test_statistic_id'].tolist()

    # Extract observed values from observed_data dict
    observed_values = []
    for name in test_stat_names:
        if name in observed_data:
            obs_val = observed_data[name]
            # Handle 2D arrays (1, 1)
            if isinstance(obs_val, np.ndarray):
                obs_val = obs_val.flatten()[0]
            observed_values.append(obs_val)
        else:
            print(f"   ⚠️  Warning: Test statistic '{name}' not found in observed_data")
            observed_values.append(np.nan)

    observed_test_stats = np.array(observed_values).reshape(1, -1)

    print(f"   ✓ Observed test statistics: {observed_test_stats.shape}")

    # 4. Return results
    results = {
        'posterior_samples': theta_posterior,
        'predictive_test_stats': observables_posterior,
        'observed_test_stats': observed_test_stats,
        'test_stat_names': test_stat_names,
        'param_names': qsp_simulator.param_names
    }

    print(f"\n✅ Posterior predictive checks complete!")
    print(f"   Posterior samples: {theta_posterior.shape}")
    print(f"   Predictive test stats: {observables_posterior.shape}")
    print(f"   Observed test stats: {observed_test_stats.shape}")

    return results


def generate_posterior_predictive_simulations(
    posterior_samples: Dict[str, np.ndarray],
    observed_data: Dict[str, np.ndarray],
    qsp_simulator,
    species_of_interest: Optional[List[str]] = None
) -> Dict:
    """
    Generate full posterior predictive simulations (complete time series).

    IMPORTANT: Pass observed_data in ORIGINAL space (not transformed).
    The posterior_samples should already be computed from the workflow using
    TRANSFORMED observed data for proper conditioning.

    This function:
    1. Extracts parameters from pre-computed posterior samples
    2. Runs full QSP simulations for these parameters (saves to HPC)
    3. Downloads full time series data for all species
    4. Returns complete trajectory data for visualization

    Args:
        posterior_samples: Pre-computed posterior samples from workflow.sample()
                          (dictionary with parameter arrays or 'inference_variables')
        observed_data: Dictionary of observed test statistics IN ORIGINAL SPACE
                      (used only for cache hash identification)
        qsp_simulator: QSPSimulator instance for running simulations
        species_of_interest: Optional list of species names to extract (default: all)

    Returns:
        Dictionary containing:
        - 'posterior_samples': Parameter samples from posterior (n_samples, n_params)
        - 'time': Time vector (n_timepoints,)
        - 'simulations': Dict mapping species name to array (n_samples, n_timepoints)
        - 'param_names': List of parameter names
        - 'species_names': List of species names included

    Example:
        # Sample from posterior using TRANSFORMED data
        obs_transformed = transform_to_normal(obs_original, observable_names, quantiles)
        posterior_samples = workflow.sample(conditions=obs_transformed, num_samples=50)

        # Run full simulations
        pp_sims = generate_posterior_predictive_simulations(
            posterior_samples=posterior_samples,
            observed_data=obs_original,  # For cache hash only
            qsp_simulator=qsp_sim,
            species_of_interest=['V_T_C1', 'V_T_CD8']
        )
    """
    print(f"\n🎬 Generating Full Posterior Predictive Simulations")

    # Compute hash of observed data to ensure cache is specific to this dataset
    import hashlib
    obs_values = []
    for key in sorted(observed_data.keys()):  # Sort for deterministic hash
        val = observed_data[key]
        if isinstance(val, np.ndarray):
            val = val.flatten()[0]
        obs_values.append(float(val))
    obs_array = np.array(obs_values)
    obs_hash = hashlib.sha256(obs_array.tobytes()).hexdigest()[:8]

    print(f"   Observed data hash: {obs_hash} (ensures cache is dataset-specific)")

    # Extract parameter matrix
    print(f"\n   🔍 Extracting parameters from posterior samples...")
    if 'inference_variables' in posterior_samples:
        theta_posterior = posterior_samples['inference_variables']
        print(f"   → Using 'inference_variables' array: {theta_posterior.shape}")
    else:
        param_names = qsp_simulator.param_names
        print(f"   → Stacking {len(param_names)} individual parameters...")
        param_arrays = []
        for name in param_names:
            arr = posterior_samples[name]
            if arr.ndim > 1:
                arr = arr.flatten()
            param_arrays.append(arr)
        theta_posterior = np.column_stack(param_arrays)
        print(f"   ✓ Successfully extracted all {len(param_names)} parameters")

    # Get sample count from extracted matrix
    n_samples = theta_posterior.shape[0]
    print(f"   ✓ Posterior shape: ({n_samples} samples, {theta_posterior.shape[1]} parameters)")

    # CRITICAL: Check if samples are in log space (from adapter transformation)
    # BayesFlow workflow.sample() should return samples in ORIGINAL space
    # Only apply exp() if there are NEGATIVE values (clear sign of log space)
    if np.any(theta_posterior < 0):
        print(f"\n   🔍 DETECTED NEGATIVE VALUES - samples are in LOG space")
        print(f"   → Back-transforming with exp() to original parameter space")
        theta_posterior_original = np.exp(theta_posterior)
        print(f"   → Median before exp(): {np.median(theta_posterior):.2e}")
        print(f"   → Median after exp(): {np.median(theta_posterior_original):.2e}")
    else:
        print(f"\n   ✓ Samples appear to be in ORIGINAL space (no negative values)")
        print(f"   → Median: {np.median(theta_posterior):.2e}")
        print(f"   → Range: [{theta_posterior.min():.2e}, {theta_posterior.max():.2e}]")
        theta_posterior_original = theta_posterior

    # Use the (possibly) back-transformed samples for simulations
    theta_posterior = theta_posterior_original

    # Debug: Check actual parameter values
    print(f"\n   🔍 Sample parameter values (first sample):")
    for i in range(min(5, len(qsp_simulator.param_names))):
        print(f"      {qsp_simulator.param_names[i]}: {theta_posterior[0, i]:.6e}")
    print(f"   → Parameter range: [{theta_posterior.min():.6e}, {theta_posterior.max():.6e}]")
    print(f"   → Parameter median: {np.median(theta_posterior):.6e}")

    # 2. Run full simulations with HPC storage enabled (with caching!)
    print(f"\n2️⃣  Running {n_samples} full QSP simulations (with time series saved)...")
    print(f"   → Using cached simulation pool (will reuse if available)")
    print(f"   → Pool suffix: ppc_{obs_hash}")

    # Use the cached simulation method
    # Include observed data hash to ensure cache is dataset-specific
    # NOTE: Using same pool_suffix as PPCs to enable reuse of simulations!
    pool_suffix = f'ppc_{obs_hash}'  # Same as PPCs - enables simulation reuse
    _ = qsp_simulator.simulate_with_parameters(
        theta=theta_posterior,
        pool_suffix=pool_suffix
    )

    print(f"   ✓ Simulations complete - Parquet files in HPC pool")

    # 3. Download full time series from HPC
    print(f"\n3️⃣  Downloading full time series from HPC simulation pool...")

    # Import HPC job manager for downloads
    from qsp_hpc.batch.hpc_job_manager import HPCJobManager
    job_manager = HPCJobManager()

    # Determine pool path (matching the pool_suffix used above)
    priors_hash = qsp_simulator._compute_priors_hash()
    pool_id = f"{qsp_simulator.model_version}_{priors_hash[:8]}_{pool_suffix}"
    pool_path = f"{job_manager.config.simulation_pool_path}/{pool_id}"

    print(f"   → Pool path: {pool_path}")

    # Create persistent local cache directory (same location as PPC summary stats)
    local_cache = Path('projects/pdac_2025/cache/posterior_predictive') / pool_id
    local_cache.mkdir(parents=True, exist_ok=True)

    # Check if we already have the data locally
    existing_parquet_files = list(local_cache.glob('*.parquet'))

    if existing_parquet_files:
        print(f"   ✓ Found {len(existing_parquet_files)} cached Parquet file(s) locally")
        downloaded_files = existing_parquet_files
    else:
        # Download all Parquet batches (to get all n_samples)
        # Calculate how many batches we need (assuming ~5 sims per batch, adjust if needed)
        n_files_needed = max(1, int(np.ceil(n_samples / 5)))
        print(f"   📥 Downloading up to {n_files_needed} most recent Parquet batch(es) from HPC...")
        downloaded_files = job_manager.download_latest_parquet_batch(
            pool_path=pool_path,
            local_dest=local_cache,
            n_files=n_files_needed  # Download enough batches to cover all samples
        )

    if not downloaded_files:
        raise RuntimeError("Failed to download Parquet files from HPC")

    # Parse all Parquet files and combine
    all_simulations = []
    time = None
    species_names = None
    total_parsed = 0

    for parquet_file in downloaded_files:
        parquet_results = job_manager.parse_parquet_simulations(
            parquet_file=parquet_file,
            species_of_interest=species_of_interest,
            max_simulations=n_samples - total_parsed  # Only parse what we still need
        )

        # Store time and species names from first file
        if time is None:
            time = parquet_results['time']
            species_names = parquet_results['species_names']

        sims = parquet_results['simulations']

        # If simulations is a dict (species -> array), convert to 3D array
        if isinstance(sims, dict):
            # Stack species: (n_sims, n_timepoints, n_species)
            species_arrays = [sims[sp] for sp in species_names]
            sims_array = np.stack(species_arrays, axis=2)
            all_simulations.append(sims_array)
        else:
            all_simulations.append(sims)

        print(f"   📊 Parsed {parquet_results['n_simulations']} simulations from {Path(parquet_file).name}")

        total_parsed += parquet_results['n_simulations']

        # Stop if we have enough
        if total_parsed >= n_samples:
            break

    # Concatenate all simulations
    if len(all_simulations) > 1:
        combined_simulations = np.concatenate(all_simulations, axis=0)
    else:
        combined_simulations = all_simulations[0]

    # Trim to exact n_samples if we got more
    combined_simulations = combined_simulations[:n_samples]

    print(f"\n   ✅ Downloaded and parsed {combined_simulations.shape[0]} simulations")
    print(f"   ✅ Time series: {len(time)} time points")
    print(f"   ✅ Species: {len(species_names)} selected")

    # Convert 3D array back to dict format for plotting function
    # combined_simulations shape: (n_sims, n_time, n_species)
    simulations_dict = {}
    for i, species_name in enumerate(species_names):
        simulations_dict[species_name] = combined_simulations[:, :, i]  # (n_sims, n_time)

    # Build results dictionary
    results = {
        'posterior_samples': theta_posterior,
        'time': time,
        'simulations': simulations_dict,  # Dict format: {species_name: (n_sims, n_time)}
        'param_names': qsp_simulator.param_names,
        'species_names': species_names
    }

    return results


def _is_log_normal(data: np.ndarray) -> bool:
    """
    Detect if data appears to be log-normally distributed.

    Uses coefficient of variation (CV = std/mean) as a heuristic:
    - For lognormal: CV is typically > 0.5
    - For normal: CV is typically < 0.5
    - Also checks that data spans multiple orders of magnitude

    Args:
        data: Array of values to test

    Returns:
        True if data appears log-normally distributed
    """
    # Filter positive values
    positive_data = data[data > 0]

    if len(positive_data) < 10:  # Need enough data
        return False

    # Check coefficient of variation
    mean_val = np.mean(positive_data)
    std_val = np.std(positive_data)
    cv = std_val / mean_val if mean_val > 0 else 0

    # Check if data spans multiple orders of magnitude
    if np.max(positive_data) / np.min(positive_data) < 10:
        return False

    # Lognormal typically has CV > 0.5
    return cv > 0.5


def plot_ppc_histograms(
    ppc_results: Dict,
    figsize: Tuple[int, int] = (16, 12),
    max_cols: int = 3,
    bins: int = 30,
    alpha: float = 0.6,
    log_transform: Union[bool, str] = 'auto',
    clip_outliers: bool = True,
    outlier_percentiles: Tuple[float, float] = (1, 99),
    title: str = 'Posterior Predictive Checks',
    show: bool = True,
    save_path: Optional[str] = None,
    test_stat_indices: Optional[list] = None
):
    """
    Plot posterior predictive check histograms.

    For each test statistic:
    - Shows histogram of posterior predictive distribution
    - Overlays vertical line for observed value
    - Computes p-value (fraction of posterior predictive ≥ observed)

    Args:
        ppc_results: Output from generate_posterior_predictive_checks()
        figsize: Figure size in inches (width, height)
        max_cols: Maximum number of columns in subplot grid
        bins: Number of histogram bins
        alpha: Histogram transparency
        log_transform: If True, apply log10 to all statistics. If 'auto', automatically
                      detect log-normally distributed statistics and transform them.
                      If False, no transformation (default: 'auto')
        clip_outliers: If True, clip x-axis to percentile range to avoid
                      extreme outliers stretching the plot (default: True)
        outlier_percentiles: Tuple of (lower, upper) percentiles for clipping
                            x-axis limits (default: (1, 99))
        title: Main title for the figure (default: 'Posterior Predictive Checks')
        show: If True, display the plot
        save_path: Optional path to save figure
        test_stat_indices: Optional list/array of test statistic indices to plot (subset)

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    predictive = ppc_results['predictive_test_stats']
    observed = ppc_results['observed_test_stats']
    test_stat_names = ppc_results['test_stat_names']

    # Apply test statistic subsetting if specified
    if test_stat_indices is not None:
        test_stat_indices = list(test_stat_indices)
        predictive = predictive[:, test_stat_indices]
        observed = observed[:, test_stat_indices]
        test_stat_names = [test_stat_names[i] for i in test_stat_indices]

    n_test_stats = len(test_stat_names)
    n_cols = min(max_cols, n_test_stats)
    n_rows = int(np.ceil(n_test_stats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, test_stat_name in enumerate(test_stat_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get predictive and observed values
        y_pred = predictive[:, idx]
        y_obs = observed[0, idx]

        # Skip if all NaN
        if np.all(np.isnan(y_pred)) or np.isnan(y_obs):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(test_stat_name, fontsize=18, fontweight='bold')
            continue

        # Remove NaN values for plotting
        y_pred_clean = y_pred[~np.isnan(y_pred)]

        if len(y_pred_clean) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(test_stat_name, fontsize=18, fontweight='bold')
            continue

        # Determine if log transform should be applied
        apply_log = False
        if log_transform == 'auto':
            apply_log = _is_log_normal(y_pred_clean)
        elif log_transform is True:
            apply_log = True

        # Apply log transformation if needed
        label_prefix = ""
        if apply_log:
            # Filter non-positive values
            y_pred_clean = y_pred_clean[y_pred_clean > 0]
            if y_obs <= 0 or len(y_pred_clean) == 0:
                ax.text(0.5, 0.5, 'No positive values\nfor log transform',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(test_stat_name, fontsize=18, fontweight='bold')
                continue

            # Apply log10
            y_pred_clean = np.log10(y_pred_clean)
            y_obs = np.log10(y_obs)
            label_prefix = "log₁₀"

        # Determine plotting range (for outlier clipping)
        if clip_outliers:
            # Calculate percentile-based limits
            lower_percentile, upper_percentile = outlier_percentiles
            xlim_lower = np.percentile(y_pred_clean, lower_percentile)
            xlim_upper = np.percentile(y_pred_clean, upper_percentile)

            # Extend limits to include observed value if needed
            xlim_lower = min(xlim_lower, y_obs)
            xlim_upper = max(xlim_upper, y_obs)

            # Add small margin (5% of range)
            range_margin = (xlim_upper - xlim_lower) * 0.05
            xlim_lower -= range_margin
            xlim_upper += range_margin

            # Create bins for the clipped range
            bin_edges = np.linspace(xlim_lower, xlim_upper, bins + 1)
        else:
            # Use full data range
            bin_edges = bins
            xlim_lower = None
            xlim_upper = None

        # Plot histogram of posterior predictive distribution
        ax.hist(y_pred_clean, bins=bin_edges, density=True, alpha=alpha,
               color='steelblue', edgecolor='black', linewidth=0.5,
               label='Posterior predictive')

        # Add KDE for smoother visualization
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(y_pred_clean)
            x_range = np.linspace(y_pred_clean.min(), y_pred_clean.max(), 200)
            ax.plot(x_range, kde(x_range), 'b-', linewidth=2, alpha=0.8)
        except Exception:
            pass  # Skip KDE if it fails

        # Overlay observed value
        ax.axvline(y_obs, color='red', linestyle='--', linewidth=3,
                  label=f'Observed ({y_obs:.2e})')

        # Compute p-value (two-tailed: fraction of samples more extreme than observed)
        # More extreme = further from median
        median_pred = np.median(y_pred_clean)
        obs_distance = np.abs(y_obs - median_pred)
        pred_distances = np.abs(y_pred_clean - median_pred)
        p_value = np.mean(pred_distances >= obs_distance)

        # Set x-axis limits and add clipping note if applicable
        clipped_note = ""
        if clip_outliers and xlim_lower is not None and xlim_upper is not None:
            # Apply limits
            ax.set_xlim(xlim_lower, xlim_upper)

            # Count how many data points are outside the visible range
            # (before we added margins)
            lower_percentile, upper_percentile = outlier_percentiles
            orig_lower = np.percentile(y_pred_clean, lower_percentile)
            orig_upper = np.percentile(y_pred_clean, upper_percentile)
            n_clipped = np.sum((y_pred_clean < orig_lower) | (y_pred_clean > orig_upper))
            if n_clipped > 0:
                pct_clipped = 100 * n_clipped / len(y_pred_clean)
                clipped_note = f" ({pct_clipped:.1f}% clipped)"

        # Add p-value to title (with clipping note if applicable)
        subplot_title = f'p = {p_value:.3f}{clipped_note}'
        ax.set_title(subplot_title, fontsize=10, fontweight='bold')

        # Set x-axis label with log prefix if transformed
        # Break long names across multiple lines for readability
        xlabel = f'{label_prefix}({test_stat_name})' if label_prefix else test_stat_name
        if len(xlabel) > 30:  # Break long labels
            # Try to break at underscore or after 30 chars
            if '_' in xlabel[15:]:
                break_idx = xlabel.index('_', 15)
                xlabel = xlabel[:break_idx+1] + '\n' + xlabel[break_idx+1:]
        ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_test_stats, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved PPC plot to {save_path}")

    if show:
        plt.show()

    return fig, axes


def plot_posterior_predictive_spaghetti(
    pp_sim_results: Dict,
    species_to_plot: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (16, 10),
    max_cols: int = 2,
    alpha: float = 0.2,
    show_median: bool = True,
    show_credible: bool = True,
    credible_level: float = 0.95,
    ylim_percentiles: Optional[Tuple[float, float]] = (5, 95),
    show: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot posterior predictive simulations as spaghetti plots.

    For each species of interest:
    - Shows individual trajectories as semi-transparent lines
    - Optionally overlays median trajectory
    - Optionally shows credible interval band

    Args:
        pp_sim_results: Output from generate_posterior_predictive_simulations()
        species_to_plot: Optional list of species names to plot (default: all)
        figsize: Figure size in inches (width, height)
        max_cols: Maximum number of columns in subplot grid
        alpha: Line transparency for individual trajectories
        show_median: If True, overlay median trajectory
        show_credible: If True, show credible interval band
        credible_level: Credible interval level (default: 0.95)
        ylim_percentiles: Tuple of (lower, upper) percentiles for y-axis limits
                         to avoid extreme outliers. Default (5, 95) clips to 5th-95th percentile.
                         Set to None to use full data range.
        show: If True, display the plot
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    time = pp_sim_results['time']
    simulations = pp_sim_results['simulations']
    species_names = pp_sim_results.get('species_names', list(simulations.keys()))

    # Determine which species to plot
    if species_to_plot is None:
        species_to_plot = species_names

    n_species = len(species_to_plot)
    n_cols = min(max_cols, n_species)
    n_rows = int(np.ceil(n_species / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    # Ensure axes is 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, species_name in enumerate(species_to_plot):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        if species_name not in simulations:
            ax.text(0.5, 0.5, f'{species_name}\nNot available',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        trajectories = simulations[species_name]  # (n_samples, n_timepoints)
        n_samples = trajectories.shape[0]

        # Plot individual trajectories
        for i in range(n_samples):
            ax.plot(time, trajectories[i, :], color='steelblue', alpha=alpha, linewidth=1)

        # Plot median trajectory
        if show_median:
            median_traj = np.median(trajectories, axis=0)
            ax.plot(time, median_traj, color='darkblue', linewidth=3,
                   label='Median', zorder=100)

        # Plot credible interval band
        if show_credible:
            lower_percentile = (1 - credible_level) / 2 * 100
            upper_percentile = (1 - (1 - credible_level) / 2) * 100

            lower_bound = np.percentile(trajectories, lower_percentile, axis=0)
            upper_bound = np.percentile(trajectories, upper_percentile, axis=0)

            ax.fill_between(time, lower_bound, upper_bound, color='steelblue',
                           alpha=0.3, label=f'{int(credible_level*100)}% CI')

        ax.set_xlabel('Time (days)', fontsize=10)
        ax.set_ylabel(species_name, fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

        if show_median or show_credible:
            ax.legend(fontsize=8, loc='best')

        # Use log scale if values span many orders of magnitude
        if trajectories.max() / max(trajectories.min(), 1e-10) > 1000:
            ax.set_yscale('log')

        # Set y-axis limits based on percentiles to avoid extreme outliers
        if ylim_percentiles is not None:
            all_values = trajectories.flatten()
            y_min = np.percentile(all_values, ylim_percentiles[0])
            y_max = np.percentile(all_values, ylim_percentiles[1])

            # Add some padding
            if ax.get_yscale() == 'log':
                # For log scale, expand range by multiplicative factor
                y_min = y_min / 2
                y_max = y_max * 2
            else:
                # For linear scale, add 10% padding
                y_range = y_max - y_min
                y_min = y_min - 0.1 * y_range
                y_max = y_max + 0.1 * y_range

            ax.set_ylim(y_min, y_max)

    # Hide unused subplots
    for idx in range(n_species, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.suptitle(f'Posterior Predictive Simulations ({n_samples} samples)',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved spaghetti plot to {save_path}")

    if show:
        plt.show()

    return fig, axes


def plot_prior_vs_posterior_predictive(
    prior_ppc_results: Dict,
    posterior_ppc_results: Dict,
    figsize: Tuple[int, int] = (16, 12),
    max_cols: int = 4,
    bins: int = 50,
    log_transform: Union[bool, str] = 'auto',
    test_stat_indices: Optional[list] = None,
    show: bool = True,
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot prior vs posterior predictive distributions overlaid with observed data.

    Similar to plot_posterior_vs_prior_marginals but for test statistics/observables.
    Shows how much the posterior predictive has concentrated relative to the prior predictive.

    Args:
        prior_ppc_results: Output from generate_prior_predictive_checks()
        posterior_ppc_results: Output from generate_posterior_predictive_checks()
        figsize: Figure size in inches (width, height)
        max_cols: Maximum number of columns in subplot grid
        bins: Number of bins for histograms
        log_transform: If True, apply log10 to all statistics. If 'auto', automatically
                      detect log-normally distributed statistics. If False, no transformation.
        test_stat_indices: Optional list/array of test statistic indices to plot (subset)
        show: If True, display the plot
        save_path: Optional path to save figure

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    prior_predictive = prior_ppc_results['predictive_test_stats']
    post_predictive = posterior_ppc_results['predictive_test_stats']
    observed = posterior_ppc_results['observed_test_stats']  # Same for both
    test_stat_names = posterior_ppc_results['test_stat_names']

    # Apply test statistic subsetting if specified
    if test_stat_indices is not None:
        test_stat_indices = list(test_stat_indices)
        prior_predictive = prior_predictive[:, test_stat_indices]
        post_predictive = post_predictive[:, test_stat_indices]
        observed = observed[:, test_stat_indices]
        test_stat_names = [test_stat_names[i] for i in test_stat_indices]

    n_test_stats = len(test_stat_names)
    n_cols = min(max_cols, n_test_stats)
    n_rows = int(np.ceil(n_test_stats / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1) if n_test_stats > 1 else [axes]

    for idx, test_stat_name in enumerate(test_stat_names):
        ax = axes[idx]

        # Get values
        prior_vals = prior_predictive[:, idx]
        post_vals = post_predictive[:, idx]
        obs_val = observed[0, idx]

        # Skip if all NaN
        if np.all(np.isnan(prior_vals)) or np.all(np.isnan(post_vals)) or np.isnan(obs_val):
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(test_stat_name, fontsize=18, fontweight='bold')
            continue

        # Remove NaN values
        prior_vals = prior_vals[~np.isnan(prior_vals)]
        post_vals = post_vals[~np.isnan(post_vals)]

        if len(prior_vals) == 0 or len(post_vals) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_xlabel(test_stat_name, fontsize=18, fontweight='bold')
            continue

        # Determine if log transform should be applied
        apply_log = False
        if log_transform == 'auto':
            # Check if distribution looks log-normal
            apply_log = _is_log_normal(post_vals)
        elif log_transform is True:
            apply_log = True

        label_prefix = ""
        if apply_log:
            # Filter non-positive values
            prior_vals = prior_vals[prior_vals > 0]
            post_vals = post_vals[post_vals > 0]

            if obs_val <= 0 or len(prior_vals) == 0 or len(post_vals) == 0:
                ax.text(0.5, 0.5, 'No positive values\nfor log transform',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(test_stat_name, fontsize=18, fontweight='bold')
                continue

            # Apply log10
            prior_vals = np.log10(prior_vals)
            post_vals = np.log10(post_vals)
            obs_val = np.log10(obs_val)
            label_prefix = "log₁₀"

        # Determine x-axis range that includes all values
        all_vals = np.concatenate([prior_vals, post_vals, [obs_val]])
        x_min, x_max = all_vals.min(), all_vals.max()

        # Add some padding
        x_range_pad = (x_max - x_min) * 0.1
        x_min -= x_range_pad
        x_max += x_range_pad

        # Create bins spanning both distributions
        bin_edges = np.linspace(x_min, x_max, bins + 1)

        # Plot PRIOR predictive first (background, lighter color)
        ax.hist(prior_vals, bins=bin_edges, density=True, alpha=0.3,
                color='gray', edgecolor='darkgray', linewidth=0.5, label='Prior predictive')

        # Add prior KDE
        from scipy.stats import gaussian_kde
        try:
            prior_kde = gaussian_kde(prior_vals)
            x_plot = np.linspace(x_min, x_max, 200)
            ax.plot(x_plot, prior_kde(x_plot), '--', color='gray',
                   linewidth=2, alpha=0.7)
        except Exception:
            pass

        # Plot POSTERIOR predictive (foreground, darker color)
        ax.hist(post_vals, bins=bin_edges, density=True, alpha=0.6,
                color='steelblue', edgecolor='black', linewidth=0.5, label='Posterior predictive')

        # Add posterior KDE
        try:
            post_kde = gaussian_kde(post_vals)
            ax.plot(x_plot, post_kde(x_plot), '-', color='darkblue',
                   linewidth=2)
        except Exception:
            pass

        # Add observed value (red vertical line)
        ax.axvline(obs_val, color='red', linestyle='-', linewidth=2.5,
                  label='Observed', alpha=0.8)

        # Set x-axis label with log prefix if transformed
        # Break long names across multiple lines for readability
        xlabel = f'{label_prefix}({test_stat_name})' if label_prefix else test_stat_name
        if len(xlabel) > 30:  # Break long labels
            if '_' in xlabel[15:]:
                break_idx = xlabel.index('_', 15)
                xlabel = xlabel[:break_idx+1] + '\n' + xlabel[break_idx+1:]
        ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        ax.set_xlim(x_min, x_max)

    # Hide unused subplots
    for idx in range(n_test_stats, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle('Prior vs Posterior Predictive Checks', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   💾 Saved prior vs posterior predictive plot to {save_path}")

    if show:
        plt.show()

    return fig, axes

#!/usr/bin/env python3
"""
Gaussian Copula Transform for BayesFlow Observables

Pre-transforms observables to standard normal using Gaussian copula.
This is much faster than transforming during each training batch.

Usage:
    >>> from qsp_inference.inference import (
    ...     compute_quantiles,
    ...     transform_to_normal
    ... )
    >>>
    >>> # 1. Compute quantiles from training data
    >>> quantiles = compute_quantiles(training_data, observable_names)
    >>>
    >>> # 2. Transform all datasets upfront (once, not per batch!)
    >>> training_data = transform_to_normal(training_data, observable_names, quantiles)
    >>> validation_data = transform_to_normal(validation_data, observable_names, quantiles)
    >>> test_data = transform_to_normal(test_data, observable_names, quantiles)
    >>>
    >>> # 3. Before posterior sampling, transform observed data
    >>> obs = get_observed_data(...)
    >>> obs = transform_to_normal(obs, observable_names, quantiles)
    >>> posterior = workflow.sample(conditions=obs, num_samples=1000)
"""

import numpy as np


def compute_quantiles(data, variable_names=None):
    """
    Compute empirical quantiles from training data for Gaussian copula transform.

    Args:
        data: Dictionary with variable names as keys and arrays as values,
              OR numpy array of shape (n_samples, n_features)
        variable_names: List of variable names to extract (if data is dict).
                       If None and data is dict, uses all keys.
                       If data is array, this parameter is ignored.

    Returns:
        Numpy array of shape (n_samples, n_features) with sorted values for each feature
    """
    # Extract data matrix
    if isinstance(data, dict):
        if variable_names is None:
            variable_names = list(data.keys())

        # Stack arrays
        arrays = []
        for name in variable_names:
            arr = data[name]
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            arrays.append(arr)
        data_matrix = np.hstack(arrays)
    else:
        data_matrix = np.asarray(data)

    # Ensure 2D
    if data_matrix.ndim == 1:
        data_matrix = data_matrix.reshape(-1, 1)

    n_samples, n_features = data_matrix.shape

    # Compute empirical quantiles (sorted values for each feature)
    quantiles = np.zeros((n_samples, n_features), dtype=np.float32)
    for j in range(n_features):
        quantiles[:, j] = np.sort(data_matrix[:, j])

    return quantiles


def transform_to_normal(data, variable_names, quantiles):
    """
    Transform data to standard normal using Gaussian copula (pre-transform approach).

    This modifies data IN-PLACE for efficiency. Use scipy for speed.

    Args:
        data: Dictionary with variable names as keys and arrays as values
        variable_names: List of variable names to transform
        quantiles: Array of shape (n_samples, n_features) with sorted training values

    Returns:
        Modified data dictionary (same object, modified in-place)
    """
    from scipy.stats import norm

    n_quantiles = len(quantiles)

    for i, var_name in enumerate(variable_names):
        # Get data for this variable
        var_data = data[var_name]
        original_shape = var_data.shape
        var_data_flat = var_data.flatten()

        # Empirical CDF: find rank in quantiles
        ranks = np.searchsorted(quantiles[:, i], var_data_flat)

        # Convert to uniform [0, 1]
        uniform = (ranks + 0.5) / (n_quantiles + 1)
        uniform = np.clip(uniform, 1e-7, 1 - 1e-7)

        # Transform to standard normal (scipy is fast!)
        normal = norm.ppf(uniform)

        # Update in-place
        data[var_name] = normal.reshape(original_shape).astype(np.float32)

    return data


def compute_quantiles_from_array(data, n_quantiles=1000):
    """
    Compute quantiles for Gaussian copula transformation from 2D array.

    This is a simplified array-based version for use with direct numpy arrays
    (as opposed to dictionaries). Uses evenly-spaced quantiles rather than
    empirical CDF.

    Args:
        data: 2D numpy array of shape (n_samples, n_features)
        n_quantiles: Number of quantiles to compute (default: 1000)

    Returns:
        quantiles: 2D array of shape (n_quantiles, n_features)
    """
    quantile_values = np.linspace(0, 1, n_quantiles)
    quantiles = np.quantile(data, quantile_values, axis=0)
    return quantiles


def transform_to_normal_from_array(data, quantiles, debug=False):
    """
    Transform data to standard normal using Gaussian copula with precomputed quantiles.

    This is a simplified array-based version for use with direct numpy arrays.
    Does not modify data in-place (returns transformed copy).

    Args:
        data: 2D numpy array of shape (n_samples, n_features)
        quantiles: 2D array of shape (n_quantiles, n_features)
        debug: If True, print debug information

    Returns:
        transformed_data: 2D array with standard normal marginals
    """
    from scipy import stats

    n_samples, n_features = data.shape
    n_quantiles = quantiles.shape[0]

    transformed = np.zeros_like(data)

    for i in range(n_features):
        # Compute empirical CDF values using precomputed quantiles
        # For each data point, find where it falls in the quantile array
        u = np.searchsorted(quantiles[:, i], data[:, i]) / n_quantiles

        if debug and i == 0:  # Debug first feature only
            print(f"     DEBUG feature 0:")
            print(f"       data range: [{data[:, i].min():.4f}, {data[:, i].max():.4f}]")
            print(f"       quantiles range: [{quantiles[:, i].min():.4f}, {quantiles[:, i].max():.4f}]")
            print(f"       u range before clip: [{u.min():.6f}, {u.max():.6f}]")
            print(f"       u unique values: {len(np.unique(u))}")

        # Clip to avoid infinities at boundaries
        u = np.clip(u, 1e-10, 1 - 1e-10)

        # Transform to standard normal using inverse CDF
        transformed[:, i] = stats.norm.ppf(u)

    return transformed

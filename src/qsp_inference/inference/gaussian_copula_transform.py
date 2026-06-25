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

    Tail handling: the map is built as a piecewise-linear function from the
    empirical quantile grid (x) to its normal scores (z), with **linear
    extrapolation in z** beyond the training support using the slope of the
    outer ~5% of the grid. The previous implementation instead bucketed values
    with ``searchsorted`` and clipped the resulting uniform level to
    ``[1e-10, 1-1e-10]`` — so every value above the training max collapsed to the
    SAME ``norm.ppf(1-1e-10) ≈ +6.36`` (and every value below the min to -6.36),
    destroying tail ordering and, critically, placing an out-of-support observed
    statistic ~2x beyond the [-3.1, +3.1] range the NPE ever trained on. An NPE
    conditioned on that saturated input extrapolates wildly and reverts toward the
    prior. Linear z-extrapolation keeps the map monotone and continuous so an
    out-of-support value lands just past the boundary the flow actually saw,
    degrading gracefully instead of saturating. In-support values are unchanged up
    to the switch from bucketed-rank to linear interpolation (a strict smoothing).

    Args:
        data: 2D numpy array of shape (n_samples, n_features)
        quantiles: 2D array of shape (n_quantiles, n_features), e.g. from
                   ``compute_quantiles_from_array`` (np.quantile at linspace(0,1,n))
        debug: If True, print debug information

    Returns:
        transformed_data: 2D array with standard normal marginals
    """
    from scipy import stats

    data = np.asarray(data, dtype=float)
    n_samples, n_features = data.shape
    n_quantiles = quantiles.shape[0]

    # Normal scores of the quantile grid. quantiles[:, i] are the
    # linspace(0, 1, n_quantiles) empirical quantiles, so grid point k sits at
    # uniform level k/(n_quantiles-1); offset the 0/1 ends by half a step to keep
    # ppf finite (matches the spirit of the old 1e-10 clip but only at the grid).
    p = np.linspace(0.0, 1.0, n_quantiles)
    eps = 0.5 / n_quantiles
    z_grid = stats.norm.ppf(np.clip(p, eps, 1.0 - eps))

    transformed = np.zeros_like(data)

    for i in range(n_features):
        xq = quantiles[:, i]
        x = data[:, i]

        # Deduplicate flat regions so the interpolation x-grid is strictly
        # increasing; ties collapse to their first normal score (a flat
        # observable segment maps to a single z).
        xu, idx = np.unique(xq, return_index=True)
        if xu.size < 2:
            # Degenerate feature: all training values identical -> map to 0.
            transformed[:, i] = 0.0
            continue
        zu = z_grid[idx]

        # In-support: piecewise-linear interpolation (np.interp clamps to the
        # ends outside [xu[0], xu[-1]]; we overwrite those below).
        z = np.interp(x, xu, zu)

        # Out-of-support: linear extrapolation in z using the slope across the
        # outer ~5% of the grid (smoother than a single extreme-quantile gap).
        k = max(1, xu.size // 20)
        dx_lo = xu[k] - xu[0]
        dx_hi = xu[-1] - xu[-1 - k]
        slope_lo = (zu[k] - zu[0]) / dx_lo if dx_lo > 0 else 0.0
        slope_hi = (zu[-1] - zu[-1 - k]) / dx_hi if dx_hi > 0 else 0.0
        below = x < xu[0]
        above = x > xu[-1]
        z[below] = zu[0] + (x[below] - xu[0]) * slope_lo
        z[above] = zu[-1] + (x[above] - xu[-1]) * slope_hi

        if debug and i == 0:
            print(f"     DEBUG feature 0:")
            print(f"       data range:      [{x.min():.4f}, {x.max():.4f}]")
            print(f"       support range:   [{xu[0]:.4f}, {xu[-1]:.4f}]")
            print(f"       z range:         [{z.min():.4f}, {z.max():.4f}]  (grid ends ±{abs(zu[-1]):.2f})")
            print(f"       n below/above support: {int(below.sum())}/{int(above.sum())}")

        transformed[:, i] = z

    return transformed.astype(data.dtype, copy=False)

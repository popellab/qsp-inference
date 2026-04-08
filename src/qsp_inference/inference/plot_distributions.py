#!/usr/bin/env python3
"""
Distribution Plotting Functions for SBI

Provides plotting utilities for visualizing parameter distributions
(marginal and pairwise) from posterior samples.

Usage:
    from qsp_inference.inference import (
        plot_posterior_marginals,
        plot_posterior_pairs,
        plot_posterior_vs_prior_marginals
    )

    # Plot marginal distributions
    fig, axes = plot_posterior_marginals(
        post_samples,
        param_names=param_names,
        priors_csv='priors.csv',
        credible_intervals=[0.95]
    )

    # Plot pairwise relationships (corner plot)
    fig, axes = plot_posterior_pairs(
        post_samples,
        param_names_subset=['param1', 'param2', 'param3'],
        priors_csv='priors.csv',
        contour_levels=[0.68, 0.95]
    )

    # Compare posterior to prior
    fig, axes = plot_posterior_vs_prior_marginals(
        post_samples,
        priors_csv='priors.csv',
        param_names=param_names,
        n_prior_samples=5000
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def plot_posterior_marginals(
    samples_dict,
    param_names=None,
    priors_csv=None,
    true_values=None,
    figsize=(16, 12),
    max_cols=4,
    bins=50,
    log_transform=False,
    log_scale=False,
    credible_intervals=None,
    show=True,
    param_indices=None
):
    """
    Plot marginal distributions for each parameter from posterior samples.

    Args:
        samples_dict: Dictionary from workflow.sample() containing posterior samples
        param_names: Optional list of parameter names (auto-extracted if None)
        priors_csv: Optional path to priors CSV for extracting units
        true_values: Optional dict or array of true parameter values to overlay
        figsize: Figure size in inches (width, height)
        max_cols: Maximum number of columns in subplot grid
        bins: Number of bins for histograms
        log_transform: If True, apply log10 transformation to samples before plotting
        log_scale: If True, use log scale for x-axis (default: False)
        credible_intervals: List of credible interval levels to show as shaded regions
                           (e.g., [0.5, 0.95] for 50% and 95% CIs). None for no intervals.
        show: If True, display the plot
        param_indices: Optional list/array of parameter indices to plot (subset)

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Load units from priors CSV if provided
    units_dict = {}
    if priors_csv is not None:
        priors_df = pd.read_csv(priors_csv)
        if 'units' in priors_df.columns:
            units_dict = dict(zip(priors_df['name'], priors_df['units']))
    # Extract parameter samples (unstandardized)
    if 'inference_variables' in samples_dict:
        # Combined format: (num_samples, num_params)
        param_samples = samples_dict['inference_variables']
        # If param_names not provided, use generic names
        if param_names is None:
            param_names = [f'param_{i}' for i in range(param_samples.shape[1])]
    else:
        # Individual parameter format - extract param names from keys
        if param_names is None:
            # Get all keys that look like parameter names (not special keys)
            special_keys = {'inference_variables', 'inference_conditions', 'summary_variables'}
            param_names = [k for k in samples_dict.keys() if k not in special_keys]

        # Stack parameter arrays and ensure 2D shape (num_samples, num_params)
        param_arrays = []
        for name in param_names:
            arr = samples_dict[name]
            # Flatten to 1D if needed (removes extra dimensions)
            if arr.ndim > 1:
                arr = arr.flatten()
            param_arrays.append(arr)
        param_samples = np.column_stack(param_arrays)

    # Apply parameter subsetting if specified
    if param_indices is not None:
        param_indices = list(param_indices)
        param_samples = param_samples[:, param_indices]
        param_names = [param_names[i] for i in param_indices]

    n_params = len(param_names)
    n_cols = min(max_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, param_name in enumerate(param_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get samples for this parameter
        param_vals = param_samples[:, idx]

        # Apply log transformation if requested
        if log_transform:
            # Filter out non-positive values before log transform
            param_vals = param_vals[param_vals > 0]

            if len(param_vals) == 0:
                print(f"Warning: Skipping {param_name} - no positive values for log transform")
                ax.text(0.5, 0.5, f'No positive values\nfor {param_name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_visible(False)
                continue

            # Apply log10 transformation
            param_vals = np.log10(param_vals)

        # Plot credible intervals as shaded regions (behind histogram)
        if credible_intervals is not None:
            # Sort intervals from widest to narrowest for proper layering
            sorted_cis = sorted(credible_intervals, reverse=True)
            for ci_level in sorted_cis:
                # Compute equal-tailed credible interval
                lower_percentile = (1 - ci_level) / 2 * 100
                upper_percentile = (1 - (1 - ci_level) / 2) * 100
                ci_lower = np.percentile(param_vals, lower_percentile)
                ci_upper = np.percentile(param_vals, upper_percentile)

                # Shade the region with opacity based on CI level
                # Wider intervals get lighter shading
                alpha = 0.15 + (ci_level - min(sorted_cis)) / (max(sorted_cis) - min(sorted_cis) + 0.01) * 0.15
                ax.axvspan(ci_lower, ci_upper, alpha=alpha, color='gray')

        # Create appropriate bins based on scale
        if log_scale:
            # Use logarithmically-spaced bins for log scale
            # Avoid issues with zero or negative values
            param_vals_pos = param_vals[param_vals > 0]
            if len(param_vals_pos) > 0:
                bin_edges = np.logspace(np.log10(param_vals_pos.min()),
                                       np.log10(param_vals_pos.max()),
                                       bins + 1)
            else:
                bin_edges = bins
        else:
            # Use linear bins for linear scale
            bin_edges = bins

        # Plot histogram
        ax.hist(param_vals, bins=bin_edges, density=True, alpha=0.6,
                color='steelblue', edgecolor='black', linewidth=0.5)

        # Add KDE
        kde = gaussian_kde(param_vals)
        x_range = np.linspace(param_vals.min(), param_vals.max(), 200)
        ax.plot(x_range, kde(x_range), 'r-', linewidth=2)

        # Add mean and median lines with values in legend
        mean_val = np.mean(param_vals)
        median_val = np.median(param_vals)
        ax.axvline(mean_val, color='orange', linestyle='-', linewidth=2,
                  label=f'Mean ({mean_val:.1e})')
        ax.axvline(median_val, color='purple', linestyle='--', linewidth=2,
                  label=f'Median ({median_val:.1e})')

        # Add true value if provided
        if true_values is not None:
            if isinstance(true_values, dict) and param_name in true_values:
                true_val = true_values[param_name]
            elif isinstance(true_values, (np.ndarray, list)) and idx < len(true_values):
                true_val = true_values[idx]
            else:
                true_val = None

            if true_val is not None:
                ax.axvline(true_val, color='green', linestyle='--',
                          linewidth=2, label=f'True ({true_val:.1e})')

        # Formatting with units (split onto multiple lines to prevent overlap)
        units = units_dict.get(param_name, '')
        if log_transform:
            xlabel = f'log₁₀({param_name})\n({units})' if units else f'log₁₀({param_name})'
        else:
            xlabel = f'{param_name}\n({units})' if units else param_name
        ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
        ax.set_yticks([])

        # Apply log scale if requested
        if log_scale:
            ax.set_xscale('log')
        else:
            # Format x-axis ticks in scientific notation with 2 decimal places
            from matplotlib.ticker import ScalarFormatter
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))  # Use scientific notation outside this range
            ax.xaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)

            # Reduce number of x-axis ticks to prevent crowding
            ax.locator_params(axis='x', nbins=5)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_posterior_pairs(
    samples_dict,
    param_names_subset,
    all_param_names=None,
    priors_csv=None,
    true_values=None,
    bins=30,
    log_transform=False,
    log_scale=False,
    credible_intervals=None,
    contour_levels=[0.68, 0.95],
    figsize=None,
    show=True
):
    """
    Create a pairs plot (corner plot) for posterior samples.

    Args:
        samples_dict: Dictionary from workflow.sample() containing posterior samples
        param_names_subset: List of parameter names to include in pairs plot
        all_param_names: Full list of parameter names (needed if using combined format).
                        If None, assumes individual parameter format.
        priors_csv: Optional path to priors CSV for extracting units
        true_values: Optional dict or array of true parameter values to overlay
        bins: Number of bins for histograms (default: 30)
        log_transform: If True, apply log10 transformation to samples before plotting
        log_scale: If True, use log scale for axes (default: False)
        credible_intervals: List of credible interval levels for diagonal plots
        contour_levels: List of probability levels for contours (default: [0.68, 0.95])
        figsize: Figure size in inches. If None, auto-sized based on number of params
        show: If True, display the plot

    Returns:
        fig, axes: Matplotlib figure and axes objects
    """
    # Load units from priors CSV if provided
    units_dict = {}
    if priors_csv is not None:
        priors_df = pd.read_csv(priors_csv)
        if 'units' in priors_df.columns:
            units_dict = dict(zip(priors_df['name'], priors_df['units']))

    # Extract parameter samples
    if 'inference_variables' in samples_dict:
        # Combined format - need to match param names
        if all_param_names is None:
            raise ValueError("all_param_names required when using combined format")
        indices = [all_param_names.index(name) for name in param_names_subset]
        param_samples = samples_dict['inference_variables'][:, indices]
    else:
        # Individual parameter format
        param_arrays = []
        for name in param_names_subset:
            arr = samples_dict[name]
            if arr.ndim > 1:
                arr = arr.flatten()
            param_arrays.append(arr)
        param_samples = np.column_stack(param_arrays)

    # Apply log transformation if requested
    if log_transform:
        # Filter out non-positive values
        mask = np.all(param_samples > 0, axis=1)
        if np.sum(mask) == 0:
            raise ValueError("No samples with all positive values for log transform")
        param_samples = param_samples[mask, :]

        # Apply log10 transformation
        param_samples = np.log10(param_samples)

        if np.sum(~mask) > 0:
            print(f"Warning: Filtered out {np.sum(~mask)} samples with non-positive values")

    n_params = len(param_names_subset)

    # Auto-size figure if not provided
    if figsize is None:
        size = max(10, n_params * 2.5)
        figsize = (size, size)

    # Create figure with subplots
    fig, axes = plt.subplots(n_params, n_params, figsize=figsize)

    # Ensure axes is 2D
    if n_params == 1:
        axes = np.array([[axes]])

    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]

            if i == j:
                # Diagonal: marginal distribution
                param_vals = param_samples[:, i]

                # Plot credible intervals
                if credible_intervals is not None:
                    sorted_cis = sorted(credible_intervals, reverse=True)
                    for ci_level in sorted_cis:
                        lower_percentile = (1 - ci_level) / 2 * 100
                        upper_percentile = (1 - (1 - ci_level) / 2) * 100
                        ci_lower = np.percentile(param_vals, lower_percentile)
                        ci_upper = np.percentile(param_vals, upper_percentile)
                        alpha = 0.15 + (ci_level - min(sorted_cis)) / (max(sorted_cis) - min(sorted_cis) + 0.01) * 0.15
                        ax.axvspan(ci_lower, ci_upper, alpha=alpha, color='gray')

                # Create bins
                if log_scale:
                    param_vals_pos = param_vals[param_vals > 0]
                    if len(param_vals_pos) > 0:
                        bin_edges = np.logspace(np.log10(param_vals_pos.min()),
                                               np.log10(param_vals_pos.max()),
                                               bins + 1)
                    else:
                        bin_edges = bins
                else:
                    bin_edges = bins

                # Histogram
                ax.hist(param_vals, bins=bin_edges, density=True, alpha=0.6,
                       color='steelblue', edgecolor='black', linewidth=0.5)

                # KDE
                kde = gaussian_kde(param_vals)
                x_range = np.linspace(param_vals.min(), param_vals.max(), 200)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2)

                # Mean and median
                mean_val = np.mean(param_vals)
                median_val = np.median(param_vals)
                ax.axvline(mean_val, color='orange', linestyle='-', linewidth=2)
                ax.axvline(median_val, color='purple', linestyle='--', linewidth=2)

                ax.set_yticks([])

            elif i > j:
                # Lower triangle: scatter only
                x_vals = param_samples[:, j]
                y_vals = param_samples[:, i]

                # Scatter plot (subset of points for clarity, darker)
                n_scatter = min(500, len(x_vals))
                indices = np.random.choice(len(x_vals), n_scatter, replace=False)
                ax.scatter(x_vals[indices], y_vals[indices], alpha=0.5, s=10, color='steelblue')

            else:
                # Upper triangle: contours only
                x_vals = param_samples[:, j]
                y_vals = param_samples[:, i]

                # 2D KDE contours
                try:
                    xy = np.vstack([x_vals, y_vals])
                    kde_2d = gaussian_kde(xy)

                    # Create grid for contour
                    x_min, x_max = x_vals.min(), x_vals.max()
                    y_min, y_max = y_vals.min(), y_vals.max()

                    # Use log-spaced grid if log_scale
                    if log_scale:
                        x_min_pos = max(x_min, x_vals[x_vals > 0].min() * 0.9) if np.any(x_vals > 0) else x_min
                        x_max_pos = x_max * 1.1 if x_max > 0 else x_max
                        y_min_pos = max(y_min, y_vals[y_vals > 0].min() * 0.9) if np.any(y_vals > 0) else y_min
                        y_max_pos = y_max * 1.1 if y_max > 0 else y_max
                        xx = np.logspace(np.log10(x_min_pos), np.log10(x_max_pos), 100)
                        yy = np.logspace(np.log10(y_min_pos), np.log10(y_max_pos), 100)
                        xx, yy = np.meshgrid(xx, yy)
                    else:
                        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]

                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    zz = kde_2d(positions).reshape(xx.shape)

                    # Compute levels for specified probabilities
                    sorted_z = np.sort(zz.ravel())[::-1]
                    cumsum = np.cumsum(sorted_z)
                    cumsum = cumsum / cumsum[-1]
                    levels = [sorted_z[np.searchsorted(cumsum, level)] for level in contour_levels]

                    # Ensure levels are unique and sorted in ascending order
                    levels = sorted(set(levels))
                    if len(levels) > 1:  # Only plot if we have valid levels
                        # Create filled contours with different opacities
                        # Innermost (highest probability) gets darkest shading
                        n_levels = len(levels)
                        for idx in range(n_levels):
                            level_pair = [levels[idx], levels[idx+1]] if idx < n_levels - 1 else [levels[idx], zz.max()]
                            # Higher probability (inner) = darker alpha
                            alpha = 0.15 + (idx / max(n_levels - 1, 1)) * 0.2
                            ax.contourf(xx, yy, zz, levels=level_pair, colors='red', alpha=alpha)

                        # Add contour lines for clarity
                        ax.contour(xx, yy, zz, levels=levels, colors='darkred', linewidths=1, alpha=0.6)
                except Exception as e:
                    print(f"Warning: Contour failed for {param_names_subset[i]} vs {param_names_subset[j]}: {e}")
                    pass

            # Apply log scale if requested
            if log_scale:
                ax.set_xscale('log')
                if i != j:  # Set y-scale for all off-diagonal
                    ax.set_yscale('log')

            # Format axis ticks to prevent crowding
            if not log_scale and i != j:
                from matplotlib.ticker import MaxNLocator
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))

            # Use scientific notation for tick labels
            from matplotlib.ticker import ScalarFormatter
            if i == n_params - 1 or (i == j):  # Bottom row or diagonal
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                ax.xaxis.set_major_formatter(formatter)
                ax.tick_params(axis='x', labelsize=7)

            if j == 0 and i != j:  # Left column (not diagonal)
                formatter = ScalarFormatter(useMathText=True)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-2, 2))
                ax.yaxis.set_major_formatter(formatter)
                ax.tick_params(axis='y', labelsize=7)

            # Labels
            if i == n_params - 1:
                # Bottom row: x-labels
                units = units_dict.get(param_names_subset[j], '')
                if log_transform:
                    xlabel = f'log₁₀({param_names_subset[j]})\n({units})' if units else f'log₁₀({param_names_subset[j]})'
                else:
                    xlabel = f'{param_names_subset[j]}\n({units})' if units else param_names_subset[j]
                ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
            else:
                ax.set_xticks([])

            if j == 0 and i > 0:
                # Left column: y-labels
                units = units_dict.get(param_names_subset[i], '')
                if log_transform:
                    ylabel = f'log₁₀({param_names_subset[i]})\n({units})' if units else f'log₁₀({param_names_subset[i]})'
                else:
                    ylabel = f'{param_names_subset[i]}\n({units})' if units else param_names_subset[i]
                ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
            else:
                if i != j:  # Don't hide y-ticks on diagonal
                    ax.set_yticks([])

            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_posterior_vs_prior_marginals(
    samples_dict,
    priors_csv,
    param_names=None,
    n_prior_samples=5000,
    true_values=None,
    figsize=(16, 12),
    max_cols=4,
    bins=50,
    log_transform=False,
    log_scale=False,
    credible_intervals=None,
    show=True,
    param_indices=None
):
    """
    Plot posterior marginals overlaid with prior distributions.

    Useful for visualizing how much the posterior has shifted from the prior
    after observing data. The prior is shown as a semi-transparent histogram
    behind the posterior distribution.

    Args:
        samples_dict: Dictionary from workflow.sample() containing posterior samples
        priors_csv: Path to priors CSV file (REQUIRED)
        param_names: Optional list of parameter names (auto-extracted if None)
        n_prior_samples: Number of samples to draw from prior for visualization
        true_values: Optional dict or array of true parameter values to overlay
        figsize: Figure size in inches (width, height)
        max_cols: Maximum number of columns in subplot grid
        bins: Number of bins for histograms
        log_transform: If True, apply log10 transformation to both prior and posterior
                       values before plotting (useful for log-distributed parameters)
        log_scale: If True, use log scale for x-axis (default: False)
        credible_intervals: List of credible interval levels for posterior
                           (e.g., [0.5, 0.95]). None for no intervals.
        show: If True, display the plot
        param_indices: Optional list/array of parameter indices to plot (subset)

    Returns:
        fig, axes: Matplotlib figure and axes objects

    Example:
        >>> fig, axes = plot_posterior_vs_prior_marginals(
        ...     post_samples,
        ...     priors_csv='priors.csv',
        ...     param_names=['k_C1_growth', 'k_C1_death'],
        ...     log_transform=True
        ... )
    """
    if priors_csv is None:
        raise ValueError("priors_csv is required for prior vs posterior comparison")

    # Load priors and sample from them
    from qsp_inference.priors import load_prior, get_param_names as load_param_names

    # Load parameter names from CSV if not provided
    if param_names is None:
        param_names = load_param_names(priors_csv)

    # Load prior and sample
    prior = load_prior(priors_csv)
    prior_samples_all = prior.sample((n_prior_samples,)).detach().numpy()

    # Ensure prior_samples is 2D
    if prior_samples_all.ndim == 1:
        prior_samples_all = prior_samples_all.reshape(-1, 1)

    # Load units and full parameter name list from priors CSV
    units_dict = {}
    priors_df = pd.read_csv(priors_csv)
    if 'units' in priors_df.columns:
        units_dict = dict(zip(priors_df['name'], priors_df['units']))
    all_csv_names = priors_df['name'].tolist()

    # Extract posterior samples
    if 'inference_variables' in samples_dict:
        # Combined format: (num_samples, num_params)
        post_samples = samples_dict['inference_variables']
        if param_names is None:
            param_names = [f'param_{i}' for i in range(post_samples.shape[1])]
    else:
        # Individual parameter format
        if param_names is None:
            special_keys = {'inference_variables', 'inference_conditions', 'summary_variables'}
            param_names = [k for k in samples_dict.keys() if k not in special_keys]

        param_arrays = []
        for name in param_names:
            arr = samples_dict[name]
            if arr.ndim > 1:
                arr = arr.flatten()
            param_arrays.append(arr)
        post_samples = np.column_stack(param_arrays)

    # Map param_names to correct columns in the full prior (param_names may be
    # a subset of the CSV, e.g. after PRCC tier filtering)
    csv_name_to_idx = {name: i for i, name in enumerate(all_csv_names)}
    prior_col_indices = [csv_name_to_idx[name] for name in param_names
                         if name in csv_name_to_idx]
    prior_samples = prior_samples_all[:, prior_col_indices]

    # Apply parameter subsetting if specified
    if param_indices is not None:
        param_indices = list(param_indices)
        post_samples = post_samples[:, param_indices]
        prior_samples = prior_samples[:, param_indices]
        param_names = [param_names[i] for i in param_indices]

    n_params = len(param_names)
    n_cols = min(max_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for idx, param_name in enumerate(param_names):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # Get samples for this parameter
        post_vals = post_samples[:, idx]
        prior_vals = prior_samples[:, idx]

        # Apply log transformation if requested
        if log_transform:
            # Filter out non-positive values before log transform
            post_vals = post_vals[post_vals > 0]
            prior_vals = prior_vals[prior_vals > 0]

            if len(post_vals) == 0 or len(prior_vals) == 0:
                print(f"Warning: Skipping {param_name} - no positive values for log transform")
                ax.text(0.5, 0.5, f'No positive values\nfor {param_name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_visible(False)
                continue

            # Apply log10 transformation
            post_vals = np.log10(post_vals)
            prior_vals = np.log10(prior_vals)

        # Determine x-axis range that includes both prior and posterior
        all_vals = np.concatenate([prior_vals, post_vals])
        x_min, x_max = all_vals.min(), all_vals.max()

        # Create appropriate bins based on scale
        if log_scale:
            # Use logarithmically-spaced bins for log scale
            vals_pos = all_vals[all_vals > 0]
            if len(vals_pos) > 0:
                bin_edges = np.logspace(np.log10(vals_pos.min()),
                                       np.log10(vals_pos.max()),
                                       bins + 1)
            else:
                bin_edges = bins
        else:
            # Use linear bins spanning both distributions
            bin_edges = np.linspace(x_min, x_max, bins + 1)

        # Plot PRIOR first (background, lighter color)
        ax.hist(prior_vals, bins=bin_edges, density=True, alpha=0.3,
                color='gray', edgecolor='darkgray', linewidth=0.5, label='Prior')

        # Add prior KDE
        prior_kde = gaussian_kde(prior_vals)
        x_range = np.linspace(x_min, x_max, 200)
        ax.plot(x_range, prior_kde(x_range), '--', color='gray',
                linewidth=2, alpha=0.7)

        # Plot POSTERIOR (foreground, darker color)
        ax.hist(post_vals, bins=bin_edges, density=True, alpha=0.6,
                color='steelblue', edgecolor='black', linewidth=0.5, label='Posterior')

        # Add posterior KDE
        post_kde = gaussian_kde(post_vals)
        ax.plot(x_range, post_kde(x_range), '-', color='darkblue',
                linewidth=2)

        # Add true value if provided
        if true_values is not None:
            if isinstance(true_values, dict) and param_name in true_values:
                true_val = true_values[param_name]
            elif isinstance(true_values, (np.ndarray, list)) and idx < len(true_values):
                true_val = true_values[idx]
            else:
                true_val = None

            if true_val is not None:
                ax.axvline(true_val, color='green', linestyle='--',
                          linewidth=2, label=f'True ({true_val:.1e})')

        # Formatting with units (split onto multiple lines to prevent overlap)
        units = units_dict.get(param_name, '')
        if log_transform:
            # Show log10(param) in label if transformed
            xlabel = f'log₁₀({param_name})\n({units})' if units else f'log₁₀({param_name})'
        else:
            xlabel = f'{param_name}\n({units})' if units else param_name
        ax.set_xlabel(xlabel, fontsize=18, fontweight='bold')
        ax.set_ylabel('Density', fontsize=10)

        # Apply log scale if requested
        if log_scale:
            ax.set_xscale('log')
        else:
            # Format x-axis ticks in scientific notation
            from matplotlib.ticker import ScalarFormatter
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-2, 2))
            ax.xaxis.set_major_formatter(formatter)
            ax.ticklabel_format(style='sci', axis='x', scilimits=(-2, 2), useMathText=True)
            ax.locator_params(axis='x', nbins=5)

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')

    # Hide unused subplots
    for idx in range(n_params, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes

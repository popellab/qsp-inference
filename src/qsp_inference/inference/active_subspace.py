#!/usr/bin/env python3
"""
Active Subspace Analysis for SBI

This module provides visualization utilities for active subspace analysis using
SBI's ActiveSubspace class. Active subspaces reveal which parameter combinations
(directions) most affect the posterior density, enabling:

1. Dimensionality reduction for high-dimensional parameter spaces
2. Sensitivity analysis (which parameters matter most)
3. Identifiability analysis (which parameter combinations are identifiable)
4. Visualization of posterior structure in low dimensions

Active Subspace Method (from SBI):
    The ActiveSubspace class computes active directions by analyzing the Hessian
    of the log posterior density. For a posterior p(θ|x), the active subspace is
    found by:

    1. Compute expected Hessian: H = E[∇²log p(θ|x)]
    2. Eigendecomposition: H = QΛQ⁻¹
       - Eigenvectors (Q) are the active directions
       - Eigenvalues (Λ) indicate importance of each direction
    3. Active variables: u = Qᵀθ
       - Large eigenvalues → active directions (posterior varies strongly)
       - Small eigenvalues → inactive directions (posterior varies weakly)

Usage:
    from sbi.analysis import ActiveSubspace
    from qsp_inference.inference import (
        plot_eigenvalue_decay,
        plot_active_weights,
        plot_1d_active_subspace_density,
        plot_2d_active_subspace_density,
        plot_active_inactive_comparison,
        plot_observables_vs_active_subspace
    )

    # Create ActiveSubspace from posterior
    posterior_with_default = posterior.set_default_x(obs_transformed)
    active_subspace = ActiveSubspace(posterior_with_default)

    # Find active directions
    eigenvalues, eigenvectors = active_subspace.find_directions(
        posterior_log_prob_as_property=True
    )

    # Visualize eigenvalue decay
    plot_eigenvalue_decay(eigenvalues, param_names)

    # Show parameter importance
    plot_active_weights(eigenvectors, eigenvalues, param_names, n_active=2)

    # Sample and project posterior
    samples = posterior.sample((1000,), x=obs_transformed)
    projected_1d = active_subspace.project(samples, num_dimensions=1)

    # Plot 1D active subspace density
    plot_1d_active_subspace_density(projected_1d, samples, param_names)

    # For 2D active subspace
    projected_2d = active_subspace.project(samples, num_dimensions=2)
    plot_2d_active_subspace_density(projected_2d, samples, param_names)

References:
    Constantine, P. G. (2015). Active Subspaces: Emerging Ideas for Dimension
    Reduction in Parameter Studies. SIAM.

    SBI documentation: https://sbi.readthedocs.io/en/latest/reference/_autosummary/sbi.analysis.ActiveSubspace.html
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import torch


def plot_eigenvalue_decay(
    eigenvalues: torch.Tensor,
    param_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    gap_threshold: float = 10.0,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot eigenvalue decay to identify active subspace dimension.

    A steep drop in eigenvalues indicates a low-dimensional active subspace.
    The gap_threshold parameter identifies significant drops.

    Args:
        eigenvalues: Eigenvalues from ActiveSubspace.find_directions()
        param_names: Optional list of parameter names (for axis labeling)
        figsize: Figure size
        gap_threshold: Ratio threshold for detecting significant eigenvalue gaps
        show: If True, display the plot

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Convert to numpy
    if isinstance(eigenvalues, torch.Tensor):
        eigenvalues = eigenvalues.detach().cpu().numpy()

    # Take absolute value (SBI may return signed eigenvalues)
    eigenvalues = np.abs(eigenvalues)

    # Sort in descending order (largest first)
    eigenvalues = np.sort(eigenvalues)[::-1]

    fig, ax = plt.subplots(figsize=figsize)

    n_eigs = len(eigenvalues)
    indices = np.arange(1, n_eigs + 1)

    # Plot eigenvalues
    ax.semilogy(indices, eigenvalues, 'o-', linewidth=2, markersize=8, color='steelblue',
                label='Eigenvalues')

    # Add gap detection (where eigenvalue drops significantly)
    ratios = eigenvalues[:-1] / (eigenvalues[1:] + 1e-10)
    if np.any(ratios > gap_threshold):
        gap_idx = np.argmax(ratios > gap_threshold)
        ax.axvline(gap_idx + 1.5, color='red', linestyle='--', linewidth=2,
                   label=f'Suggested cutoff\n(gap at dim {gap_idx+1})')

        # Annotate the gap
        gap_ratio = ratios[gap_idx]
        ax.text(gap_idx + 1.5, eigenvalues[gap_idx],
                f'  {gap_ratio:.1f}× drop',
                verticalalignment='center', fontsize=10, color='red',
                fontweight='bold')

    ax.set_xlabel('Eigenvalue Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigenvalue (|λ|)', fontsize=12, fontweight='bold')
    ax.set_title('Active Subspace Eigenvalue Decay', fontsize=14, fontweight='bold')
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add note about dimension
    n_params = len(eigenvalues)
    ax.text(0.98, 0.02, f'Total dimensions: {n_params}',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_active_weights(
    eigenvectors: torch.Tensor,
    eigenvalues: torch.Tensor,
    param_names: List[str],
    n_active: int = 2,
    max_params: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot parameter weights for the first n_active eigenvectors.

    Shows which parameters contribute most to each active direction.
    Large magnitude weights indicate important parameters.

    Interpreting Weight Signs:
        The weights define linear combinations of parameters:
            Active Variable u = w₁·θ₁ + w₂·θ₂ + ... + wₙ·θₙ

        - Positive weight (+, blue): Increasing parameter increases active variable
        - Negative weight (-, red): Increasing parameter decreases active variable

        Parameters with opposite signs indicate trade-offs or compensatory relationships:
        - Same sign (+/+ or -/-): Parameters correlated, move together
        - Opposite sign (+/-): Parameters anti-correlated, compensate each other

        For log-transformed parameters (θ in log space):
            u = w₁·log(θ₁) + w₂·log(θ₂) = log(θ₁^w₁ · θ₂^w₂)

        If w₁ = +0.8 and w₂ = -0.8:
            u = 0.8·log(θ₁/θ₂)
        → Posterior depends on the RATIO θ₁/θ₂, not individual values!

        Identifiability Implications:
        - Large magnitude with opposite signs → Strong non-identifiability
        - Only the parameter combination (ratio/difference) is identifiable
        - Individual parameter values cannot be uniquely determined

        Example:
            k_tumor_growth:  +0.65 (blue)
            k_tumor_death:   -0.32 (red)
        → Posterior sensitive to NET growth rate (growth - death)
        → Cannot identify growth and death rates independently

    Args:
        eigenvectors: Eigenvectors from ActiveSubspace.find_directions() (n_params, n_params)
        eigenvalues: Eigenvalues corresponding to eigenvectors
        param_names: List of parameter names
        n_active: Number of active directions to plot
        max_params: Maximum number of parameters to show (top N by absolute weight). None = show all
        figsize: Figure size (auto-computed if None)
        show: If True, display the plot

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Convert to numpy
    if isinstance(eigenvectors, torch.Tensor):
        eigenvectors = eigenvectors.detach().cpu().numpy()
    if isinstance(eigenvalues, torch.Tensor):
        eigenvalues = eigenvalues.detach().cpu().numpy()

    # Take absolute value of eigenvalues
    eigenvalues = np.abs(eigenvalues)

    # Sort eigenvalues in descending order and reorder eigenvectors to match
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # Auto-compute figsize based on number of parameters to display
    if figsize is None:
        n_params_to_show = min(max_params, len(param_names)) if max_params is not None else len(param_names)
        figsize = (6 * n_active, max(6, n_params_to_show * 0.3))

    fig, axes = plt.subplots(1, n_active, figsize=figsize)
    if n_active == 1:
        axes = [axes]

    for i in range(n_active):
        ax = axes[i]
        weights = eigenvectors[:, i]

        # Sort by absolute weight (descending)
        sorted_idx = np.argsort(np.abs(weights))[::-1]

        # Limit to top N parameters if specified
        if max_params is not None:
            sorted_idx = sorted_idx[:max_params]

        sorted_weights = weights[sorted_idx]
        sorted_names = [param_names[j] for j in sorted_idx]

        # Reverse order so largest absolute values are at the TOP of the plot
        sorted_weights = sorted_weights[::-1]
        sorted_names = sorted_names[::-1]

        # Bar plot with colors for positive/negative
        colors = ['red' if w < 0 else 'steelblue' for w in sorted_weights]
        ax.barh(range(len(sorted_names)), sorted_weights, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=14, fontweight='bold')
        ax.set_xlabel('Weight', fontsize=11, fontweight='bold')
        ax.set_title(f'Active Direction {i+1}\n(|λ| = {eigenvalues[i]:.2e})',
                     fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_1d_active_subspace_density(
    projected_samples: torch.Tensor,
    original_samples: torch.Tensor,
    param_names: List[str],
    n_bins: int = 50,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot density of samples in 1D active subspace.

    Shows both the marginal density of the active variable and how
    the original parameters distribute across the active direction.

    Args:
        projected_samples: Projected samples from ActiveSubspace.project() (n_samples, 1)
        original_samples: Original parameter samples (n_samples, n_params)
        param_names: List of parameter names
        n_bins: Number of histogram bins
        figsize: Figure size
        show: If True, display the plot

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Convert to numpy
    if isinstance(projected_samples, torch.Tensor):
        projected_samples = projected_samples.detach().cpu().numpy().flatten()
    if isinstance(original_samples, torch.Tensor):
        original_samples = original_samples.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: Histogram of active variable
    axes[0].hist(projected_samples, bins=n_bins, density=True, alpha=0.7,
                color='steelblue', edgecolor='black', linewidth=0.5)

    # Add KDE
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(projected_samples)
        x_range = np.linspace(projected_samples.min(), projected_samples.max(), 200)
        axes[0].plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        axes[0].legend()
    except Exception:
        pass

    axes[0].set_xlabel('Active Variable u₁ᵀθ', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0].set_title('1D Active Subspace Density', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Right: Scatter of top 3 parameters vs active variable
    n_show = min(3, original_samples.shape[1])
    for i in range(n_show):
        axes[1].scatter(projected_samples, original_samples[:, i],
                       alpha=0.3, s=10, label=param_names[i])

    axes[1].set_xlabel('Active Variable u₁ᵀθ', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Parameter Value (log space)', fontsize=12, fontweight='bold')
    axes[1].set_title('Top Parameters vs Active Variable', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_2d_active_subspace_density(
    projected_samples: torch.Tensor,
    figsize: Tuple[int, int] = (10, 8),
    n_bins: int = 50,
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 2D density of samples in active subspace.

    Shows the joint distribution of the first two active variables.

    Args:
        projected_samples: Projected samples from ActiveSubspace.project() (n_samples, 2)
        figsize: Figure size
        n_bins: Number of histogram bins per dimension
        show: If True, display the plot

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Convert to numpy
    if isinstance(projected_samples, torch.Tensor):
        projected_samples = projected_samples.detach().cpu().numpy()

    if projected_samples.shape[1] != 2:
        raise ValueError(f"Expected 2D projected samples, got shape {projected_samples.shape}")

    fig, ax = plt.subplots(figsize=figsize)

    # 2D histogram
    h, xedges, yedges = np.histogram2d(
        projected_samples[:, 0],
        projected_samples[:, 1],
        bins=n_bins,
        density=True
    )

    # Plot as image
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, origin='lower', extent=extent, aspect='auto',
                   cmap='viridis', interpolation='bilinear')

    # Add contours
    from scipy.ndimage import gaussian_filter
    h_smooth = gaussian_filter(h, sigma=1.0)
    levels = np.percentile(h_smooth[h_smooth > 0], [10, 30, 50, 70, 90])
    ax.contour(h_smooth.T, levels=levels, extent=extent,
              colors='white', linewidths=1.5, alpha=0.5)

    # Scatter overlay (subsample for visibility)
    n_show = min(1000, len(projected_samples))
    idx = np.random.choice(len(projected_samples), n_show, replace=False)
    ax.scatter(projected_samples[idx, 0], projected_samples[idx, 1],
              s=5, alpha=0.3, color='red', edgecolors='none')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Density', fontsize=11, fontweight='bold')

    ax.set_xlabel('Active Variable u₁ᵀθ', fontsize=12, fontweight='bold')
    ax.set_ylabel('Active Variable u₂ᵀθ', fontsize=12, fontweight='bold')
    ax.set_title('2D Active Subspace Density', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_active_inactive_comparison(
    active_subspace,
    samples: torch.Tensor,
    n_active: int = 1,
    figsize: Tuple[int, int] = (12, 5),
    show: bool = True
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Compare samples in active vs inactive subspace projections.

    Active subspace should show concentrated density.
    Inactive subspace should show more diffuse/uniform distribution.

    Args:
        active_subspace: SBI ActiveSubspace object (already fitted)
        samples: Parameter samples to project (n_samples, n_params)
        n_active: Dimension of active subspace
        figsize: Figure size
        show: If True, display the plot

    Returns:
        fig, axes: Matplotlib figure and axes
    """
    # Project onto active and inactive subspaces
    n_params = samples.shape[1]
    projected_active = active_subspace.project(samples, num_dimensions=n_active)
    projected_inactive = active_subspace.project(samples, num_dimensions=n_params - n_active,
                                                 inactive=True)

    # Convert to numpy
    if isinstance(projected_active, torch.Tensor):
        projected_active = projected_active.detach().cpu().numpy()
    if isinstance(projected_inactive, torch.Tensor):
        projected_inactive = projected_inactive.detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Active subspace: should show concentrated density
    if n_active == 1:
        axes[0].hist(projected_active.flatten(), bins=50, density=True,
                    alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('Active Variable u₁ᵀθ', fontsize=11, fontweight='bold')
    else:
        # Use norm of projection for multi-D
        active_norm = np.linalg.norm(projected_active, axis=1)
        axes[0].hist(active_norm, bins=50, density=True,
                    alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        axes[0].set_xlabel('||Active Projection||', fontsize=11, fontweight='bold')

    axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
    axes[0].set_title('Active Subspace\n(should be concentrated)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Inactive subspace: should show diffuse distribution
    if projected_inactive.shape[1] > 0:
        inactive_norm = np.linalg.norm(projected_inactive, axis=1)
        axes[1].hist(inactive_norm, bins=50, density=True,
                    alpha=0.7, color='gray', edgecolor='black', linewidth=0.5)
        axes[1].set_xlabel('||Inactive Projection||', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
        axes[1].set_title('Inactive Subspace\n(should be diffuse)', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No inactive subspace\n(all dimensions active)',
                     ha='center', va='center', transform=axes[1].transAxes,
                     fontsize=12, fontweight='bold')
        axes[1].set_xticks([])
        axes[1].set_yticks([])

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_parameter_projection_matrix(
    eigenvectors: torch.Tensor,
    param_names: List[str],
    n_active: int = 3,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot parameter projection matrix as a heatmap.

    Shows the full weight matrix for the first n_active directions.
    Useful for understanding parameter combinations.

    Args:
        eigenvectors: Eigenvectors from ActiveSubspace.find_directions() (n_params, n_params)
        param_names: List of parameter names
        n_active: Number of active directions to show
        figsize: Figure size
        show: If True, display the plot

    Returns:
        fig, ax: Matplotlib figure and axes
    """
    # Convert to numpy
    if isinstance(eigenvectors, torch.Tensor):
        eigenvectors = eigenvectors.detach().cpu().numpy()

    # Note: Eigenvectors should already be sorted by eigenvalue magnitude
    # by the caller (or by plot_active_weights if using same source)
    # We assume columns are ordered: largest eigenvalue → smallest eigenvalue

    # Take first n_active eigenvectors (corresponding to largest eigenvalues)
    weight_matrix = eigenvectors[:, :n_active]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    im = ax.imshow(weight_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-np.abs(weight_matrix).max(),
                   vmax=np.abs(weight_matrix).max())

    # Set ticks
    ax.set_xticks(np.arange(n_active))
    ax.set_xticklabels([f'u{i+1}' for i in range(n_active)], fontsize=11, fontweight='bold')
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_yticklabels(param_names, fontsize=9)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Weight', fontsize=11, fontweight='bold')

    # Add grid
    ax.set_xticks(np.arange(n_active + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(param_names) + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)

    ax.set_xlabel('Active Direction', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parameter', fontsize=12, fontweight='bold')
    ax.set_title('Parameter-to-Active Direction Projection Matrix',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_observables_vs_active_subspace(
    active_subspace,
    samples: torch.Tensor,
    observables: torch.Tensor,
    observable_names: List[str],
    n_active: int = 2,
    normalize: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    show: bool = True,
    robust_bounds: float = 3.0
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot how observables vary along active subspace directions.

    Shows scatter plots of observables vs active variables to understand
    which active directions affect which observables. This helps connect
    parameter sensitivity (active directions) to observable predictions.

    Use cases:
        - Identify which observables are sensitive to which active directions
        - Understand if all observables depend on the same active direction
        - Diagnose if some observables are insensitive to the active subspace

    Args:
        active_subspace: SBI ActiveSubspace object (already fitted)
        samples: Parameter samples (n_samples, n_params)
        observables: Observable values (n_samples, n_observables)
        observable_names: List of observable names
        n_active: Number of active directions to plot (1 or 2)
        normalize: If True, standardize each observable to z-scores (mean=0, std=1)
                   and clip to [-robust_bounds, +robust_bounds] for visualization.
                   This prevents outliers from dominating the color scale.
        figsize: Figure size (auto-computed if None)
        show: If True, display the plot
        robust_bounds: Number of standard deviations for color scale bounds (default: 3.0).
                       Values beyond ±robust_bounds are clipped for visualization only.

    Returns:
        fig, axes: Matplotlib figure and axes

    Example:
        # After training and sampling from posterior
        samples = posterior.sample((1000,), x=obs_transformed)

        # Get observables for these samples (transformed back to original space)
        observables = qsp_simulator.simulate_with_parameters(
            samples.detach().numpy()
        )

        # Plot how observables vary with active directions
        plot_observables_vs_active_subspace(
            active_subspace, samples, observables, observable_names, n_active=2
        )
    """
    # Project samples onto active subspace
    projected = active_subspace.project(samples, num_dimensions=n_active)

    # Convert to numpy
    if isinstance(projected, torch.Tensor):
        projected = projected.detach().cpu().numpy()
    if isinstance(observables, torch.Tensor):
        observables = observables.detach().cpu().numpy()

    n_observables = len(observable_names)

    if n_active == 1:
        # 1D active subspace: one column per observable
        if figsize is None:
            n_cols = min(3, n_observables)
            n_rows = int(np.ceil(n_observables / n_cols))
            figsize = (5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten() if n_observables > 1 else [axes]

        for i, obs_name in enumerate(observable_names):
            ax = axes[i]

            # Remove NaNs
            valid = ~np.isnan(observables[:, i])
            x_valid = projected[valid, 0]
            y_valid = observables[valid, i]

            # Standardize observable to z-scores if requested
            if normalize and len(y_valid) > 0:
                y_mean = np.mean(y_valid)
                y_std = np.std(y_valid)
                if y_std > 0:
                    y_valid_normalized = (y_valid - y_mean) / y_std
                    # Clip to robust bounds to prevent outliers from dominating scale
                    y_valid_normalized = np.clip(y_valid_normalized, -robust_bounds, robust_bounds)
                else:
                    y_valid_normalized = np.zeros_like(y_valid)
            else:
                y_valid_normalized = y_valid

            # Scatter plot
            ax.scatter(x_valid, y_valid_normalized, alpha=0.3, s=10, color='steelblue')

            # Add trend line
            try:
                from scipy.signal import savgol_filter
                sorted_idx = np.argsort(x_valid)
                x_sorted = x_valid[sorted_idx]
                y_sorted = y_valid_normalized[sorted_idx]

                if len(y_sorted) > 50:
                    window = min(51, len(y_sorted) // 4)
                    if window % 2 == 0:
                        window += 1
                    y_smooth = savgol_filter(y_sorted, window, 3)
                    ax.plot(x_sorted, y_smooth, 'r-', linewidth=2, alpha=0.8)
            except Exception:
                pass

            ax.set_xlabel('Active Variable u₁ᵀθ', fontsize=10, fontweight='bold')
            ylabel = f'{obs_name} (z-score)' if normalize else obs_name
            ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
            ax.set_title(obs_name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)

            # Set y-axis limits to robust bounds if normalized
            if normalize:
                ax.set_ylim(-robust_bounds * 1.05, robust_bounds * 1.05)

        # Hide unused subplots
        for idx in range(n_observables, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Observables vs Active Subspace (1D)',
                     fontsize=14, fontweight='bold', y=0.995)

    elif n_active == 2:
        # 2D active subspace: one subplot per observable
        if figsize is None:
            n_cols = min(3, n_observables)
            n_rows = int(np.ceil(n_observables / n_cols))
            figsize = (5 * n_cols, 4 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.array(axes).flatten() if n_observables > 1 else [axes]

        for i, obs_name in enumerate(observable_names):
            ax = axes[i]

            # Set black background
            ax.set_facecolor('black')

            # Remove NaNs
            valid = ~np.isnan(observables[:, i])
            x_valid = projected[valid, 0]
            y_valid = projected[valid, 1]
            z_valid = observables[valid, i]

            # Standardize observable to z-scores if requested
            if normalize and len(z_valid) > 0:
                z_mean = np.mean(z_valid)
                z_std = np.std(z_valid)
                if z_std > 0:
                    z_valid_normalized = (z_valid - z_mean) / z_std
                    # Clip to robust bounds to prevent outliers from dominating color scale
                    z_valid_normalized = np.clip(z_valid_normalized, -robust_bounds, robust_bounds)
                else:
                    z_valid_normalized = np.zeros_like(z_valid)
            else:
                z_valid_normalized = z_valid

            # Scatter plot colored by observable value (higher alpha for visibility on dark background)
            scatter = ax.scatter(x_valid, y_valid, c=z_valid_normalized,
                               s=30, alpha=0.8, cmap='coolwarm',
                               vmin=-robust_bounds if normalize else None,
                               vmax=robust_bounds if normalize else None)

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar_label = f'{obs_name} (z-score)' if normalize else obs_name
            cbar.set_label(cbar_label, fontsize=9)

            ax.set_xlabel('Active Variable u₁ᵀθ', fontsize=10, fontweight='bold', color='white')
            ax.set_ylabel('Active Variable u₂ᵀθ', fontsize=10, fontweight='bold', color='white')
            ax.set_title(obs_name, fontsize=11, fontweight='bold', color='white')
            ax.grid(True, alpha=0.2, color='white')

            # Make tick labels white
            ax.tick_params(colors='white', which='both')

            # Make spine (border) white
            for spine in ax.spines.values():
                spine.set_edgecolor('white')

        # Hide unused subplots
        for idx in range(n_observables, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle('Observables vs Active Subspace (2D)',
                     fontsize=14, fontweight='bold', y=0.995)

    else:
        raise ValueError(f"n_active must be 1 or 2, got {n_active}")

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes

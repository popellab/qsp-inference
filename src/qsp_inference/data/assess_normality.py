#!/usr/bin/env python3
"""
Normality Assessment for SBI Workflows

Provides visualization tools to assess whether processed data (parameters/observables)
are approximately normally distributed after transformations.

Usage:
    from qsp_inference.data import plot_processed_data_normality

    # After data processing in SBI workflow
    plot_processed_data_normality(theta_train, x_train, param_names, observable_names)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path


def plot_processed_data_normality(theta, x, param_names, observable_names, max_params=12, save_dir='projects/pdac_2025/figures', param_indices=None, obs_indices=None):
    """
    Plot distributions of processed data to check normality.

    Specifically designed for SBI workflows where:
    - Parameters are log-transformed (should look normal)
    - Observables are Gaussian copula transformed (should be standard normal)

    Args:
        theta: Parameters tensor (log-transformed)
        x: Observables tensor (Gaussian copula transformed)
        param_names: List of parameter names
        observable_names: List of observable names
        max_params: Maximum number of parameters to plot (default: 12)
        save_dir: Directory to save figures (default: 'projects/pdac_2025/figures')
        param_indices: Optional list/array of parameter indices to plot (subset)
        obs_indices: Optional list/array of observable indices to plot (subset)
    """
    import torch

    # Create save directory if needed
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Convert to numpy if tensors
    if torch.is_tensor(theta):
        theta = theta.detach().cpu().numpy()
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()

    # Apply parameter subsetting if specified
    if param_indices is not None:
        param_indices = list(param_indices)
        theta = theta[:, param_indices]
        param_names = [param_names[i] for i in param_indices]

    # Apply observable subsetting if specified
    if obs_indices is not None:
        obs_indices = list(obs_indices)
        x = x[:, obs_indices]
        observable_names = [observable_names[i] for i in obs_indices]

    n_params_to_plot = min(max_params, len(param_names))
    n_obs = len(observable_names)

    # Plot parameters (log-transformed)
    fig = plt.figure(figsize=(20, 4 + 3*n_params_to_plot//4))
    print(f"\n📊 Plotting distributions for {n_params_to_plot} parameters (log-transformed)...")

    for i in range(n_params_to_plot):
        ax = plt.subplot(n_params_to_plot//4 + 1, 4, i + 1)

        # Histogram with normal overlay
        data = theta[:, i]
        ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal distribution
        mu, sigma = data.mean(), data.std()
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', lw=2, label='Normal fit')

        # Break long parameter names for readability
        title = param_names[i]
        if len(title) > 20 and '_' in title[10:]:
            break_idx = title.index('_', 10)
            title = f'{title[:break_idx+1]}\n{title[break_idx+1:]}'
        ax.set_title(f'{title}\n(log scale)', fontsize=18, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'processed_params_normality.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.show()

    # Plot observables (Gaussian copula transformed - should be standard normal)
    fig = plt.figure(figsize=(16, 3*n_obs//3))
    print(f"\n📊 Plotting distributions for {n_obs} observables (Gaussian copula transformed)...")

    for i in range(n_obs):
        ax = plt.subplot((n_obs + 2)//3, 3, i + 1)

        # Histogram with standard normal overlay
        data = x[:, i]
        ax.hist(data, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Overlay standard normal distribution
        x_range = np.linspace(data.min(), data.max(), 100)
        ax.plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r-', lw=2, label='N(0,1)')

        # Add mean/std text
        mu, sigma = data.mean(), data.std()
        ax.text(0.02, 0.98, f'μ={mu:.2f}\nσ={sigma:.2f}',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Break long observable names for readability
        title = observable_names[i]
        if len(title) > 25 and '_' in title[12:]:
            break_idx = title.index('_', 12)
            title = f'{title[:break_idx+1]}\n{title[break_idx+1:]}'
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'processed_observables_normality.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to {save_path}")
    plt.show()

#!/usr/bin/env python3
"""
SBI Diagnostic Plots and Numerical Summaries

Diagnostic functions for evaluating simulation-based inference (SBI) results.
Adapted from BayesFlow diagnostics for use with SBI/sbi package.

All diagnostic functions return numerical results alongside figures so that
results can be saved to CSV/JSON without duplicating computation.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt


def sbi_recovery(samples, theta_test, param_names, figsize=(16, 12), max_cols=4, param_indices=None, plot=True):
    """
    Recovery plot: median of posterior samples vs true parameters.

    Args:
        samples: Posterior samples, shape (n_samples, n_test, n_params)
        theta_test: True test parameters, shape (n_test, n_params)
        param_names: List of parameter names (full list or subset)
        figsize: Figure size
        max_cols: Maximum columns in subplot grid
        param_indices: Optional list/array of parameter indices to plot (subset)
        plot: If False, skip figure creation and return (None, None, r2_values)

    Returns:
        fig, axes, r2_values: dict mapping parameter name to R² score
    """
    from sklearn.metrics import r2_score

    # Convert to numpy if needed
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if torch.is_tensor(theta_test):
        theta_test = theta_test.detach().cpu().numpy()

    # Apply parameter subsetting if specified
    if param_indices is not None:
        param_indices = list(param_indices)
        samples = samples[:, :, param_indices]
        theta_test = theta_test[:, param_indices]
        if len(param_names) != len(param_indices):
            # Assume param_names is full list, subset it
            param_names = [param_names[i] for i in param_indices]

    n_params = theta_test.shape[1]

    # Compute median across posterior samples (axis 0)
    median_posterior = np.median(samples, axis=0)  # Shape: (n_test, n_params)

    # Compute R² for each parameter
    r2_values = {}
    for i in range(n_params):
        r2_values[param_names[i]] = float(r2_score(theta_test[:, i], median_posterior[:, i]))

    if not plot:
        return None, None, r2_values

    # Create subplots
    n_cols = min(max_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_params):
        ax = axes[i]

        # Scatter plot: true vs predicted
        ax.scatter(theta_test[:, i], median_posterior[:, i], alpha=0.5, s=20)

        # Diagonal line (perfect recovery)
        min_val = min(theta_test[:, i].min(), median_posterior[:, i].min())
        max_val = max(theta_test[:, i].max(), median_posterior[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

        r2 = r2_values[param_names[i]]

        # Break long parameter names for readability
        title = f'{param_names[i]} (R² = {r2:.3f})'
        if len(param_names[i]) > 20:
            # Break at underscore for long names
            if '_' in param_names[i][10:]:
                break_idx = param_names[i].index('_', 10)
                title = f'{param_names[i][:break_idx+1]}\n{param_names[i][break_idx+1:]} (R² = {r2:.3f})'
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig, axes, r2_values


def sbi_z_score_contraction(samples, theta_test, prior, param_names, figsize=(16, 12), max_cols=4, param_indices=None, plot=True, prior_var=None):
    """
    Z-score vs contraction plot: checks global model sensitivity.

    Posterior z-score: (posterior_mean - true_parameters) / posterior_std
    - Should center around 0 and spread roughly in [-3, 3]

    Posterior contraction: 1 - (posterior_variance / prior_variance)
    - Measures reduction in uncertainty from prior to posterior
    - Should tend toward 1 (high contraction = good learning)
    - Near 0 indicates posterior variance ≈ prior variance (weak learning)

    Args:
        samples: Posterior samples, shape (n_samples, n_test, n_params)
        theta_test: True test parameters, shape (n_test, n_params)
        prior: Prior distribution (to compute prior variance). Ignored if prior_var is provided.
        param_names: List of parameter names (full list or subset)
        figsize: Figure size
        max_cols: Maximum columns in subplot grid
        param_indices: Optional list/array of parameter indices to plot (subset)
        plot: If False, skip figure creation and return (None, None, z_scores, contractions)
        prior_var: Pre-computed prior variance array, shape (n_params,). If None, estimated
            by sampling from prior (slow for truncated distributions).

    Returns:
        fig, axes, z_scores, contractions
    """
    # Convert to numpy if needed
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if torch.is_tensor(theta_test):
        theta_test = theta_test.detach().cpu().numpy()

    # Compute prior variance for ALL parameters first (before subsetting)
    if prior_var is None:
        prior_samples = prior.sample((2000,))
        if torch.is_tensor(prior_samples):
            prior_samples = prior_samples.detach().cpu().numpy()
        prior_var = np.var(prior_samples, axis=0)  # Shape: (n_params,)
    else:
        if torch.is_tensor(prior_var):
            prior_var = prior_var.detach().cpu().numpy()
        prior_var = np.asarray(prior_var)

    # Apply parameter subsetting if specified
    if param_indices is not None:
        param_indices = list(param_indices)
        samples = samples[:, :, param_indices]
        theta_test = theta_test[:, param_indices]
        prior_var = prior_var[param_indices]
        if len(param_names) != len(param_indices):
            # Assume param_names is full list, subset it
            param_names = [param_names[i] for i in param_indices]

    n_samples, n_test, n_params = samples.shape

    # Compute posterior mean and variance across samples (axis 0)
    mean_posterior = np.mean(samples, axis=0)  # Shape: (n_test, n_params)
    var_posterior = np.var(samples, axis=0)    # Shape: (n_test, n_params)

    # Compute z-scores: (posterior_mean - true_parameters) / posterior_std
    z_scores = (mean_posterior - theta_test) / (np.sqrt(var_posterior) + 1e-10)  # Shape: (n_test, n_params)

    # Compute posterior contraction: 1 - (posterior_var / prior_var)
    # Shape: (n_test, n_params) - contraction for each test observation and parameter
    contractions = 1 - (var_posterior / prior_var[np.newaxis, :])

    if not plot:
        return None, None, z_scores, contractions

    # Create subplots
    n_cols = min(max_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_params):
        ax = axes[i]

        # Scatter plot: z-score vs contraction for this parameter
        ax.scatter(contractions[:, i], z_scores[:, i], alpha=0.5, s=20)

        # Reference lines
        ax.axhline(0, color='r', linestyle='-', lw=1.5, alpha=0.8)
        ax.axhline(-3, color='gray', linestyle='--', lw=0.8, alpha=0.5)
        ax.axhline(3, color='gray', linestyle='--', lw=0.8, alpha=0.5)

        # Shaded region for acceptable z-scores
        ax.axhspan(-3, 3, alpha=0.1, color='green')

        # Break long parameter names for readability
        title = param_names[i]
        if len(title) > 20 and '_' in title[10:]:
            break_idx = title.index('_', 10)
            title = f'{title[:break_idx+1]}\n{title[break_idx+1:]}'
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_ylim(-5, 5)
        ax.set_xlim(-0.1, 1.1)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig, axes, z_scores, contractions


def compute_per_param_calibration(samples, theta_test, z_scores=None):
    """Per-parameter calibration metrics reduced across test points.

    Complements ``sbi_z_score_contraction`` (which returns per-test-point arrays)
    by producing a single number per parameter for tabulation/sweeps.

    Args:
        samples: Posterior samples, shape (n_samples, n_test, n_params). Assumed to
            live in whatever space ``theta_test`` lives in (typically log-space when
            priors are log-transformed — callers are responsible for consistency).
        theta_test: True parameter values, shape (n_test, n_params).
        z_scores: Optional pre-computed z-scores of shape (n_test, n_params) from
            ``sbi_z_score_contraction``. If None, recomputed from ``samples``.

    Returns:
        dict with keys (each a ``(n_params,)`` array):
            - ``zscore_mean``: mean across test points (should be ~0; bias detector)
            - ``zscore_std``: std across test points (should be ~1; under/overconfidence)
            - ``coverage95``: fraction of test points where ``theta_test`` falls within
              the central 95% percentile interval of ``samples`` (target 0.95)
            - ``post_sd``: median across test points of posterior std per test point
              (absolute width, in whatever space ``samples`` is in)
    """
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if torch.is_tensor(theta_test):
        theta_test = theta_test.detach().cpu().numpy()

    if z_scores is None:
        mean_post = np.mean(samples, axis=0)
        std_post = np.std(samples, axis=0)
        z_scores = (mean_post - theta_test) / (std_post + 1e-10)
    elif torch.is_tensor(z_scores):
        z_scores = z_scores.detach().cpu().numpy()

    q_lo, q_hi = np.percentile(samples, [2.5, 97.5], axis=0)
    coverage = ((q_lo <= theta_test) & (theta_test <= q_hi)).mean(axis=0)
    post_sd = np.median(np.std(samples, axis=0), axis=0)

    return {
        "zscore_mean": np.mean(z_scores, axis=0),
        "zscore_std": np.std(z_scores, axis=0),
        "coverage95": coverage,
        "post_sd": post_sd,
    }


def sbi_calibration_ecdf(samples, theta_test, param_names, figsize=(16, 12), max_cols=4, difference=True, param_indices=None, plot=True):
    """
    Calibration check using empirical CDF (rank statistics).

    For each test observation, computes the percentile of the true value
    in the posterior samples. If well-calibrated, these percentiles should
    be uniformly distributed.

    Args:
        samples: Posterior samples, shape (n_samples, n_test, n_params)
        theta_test: True test parameters, shape (n_test, n_params)
        param_names: List of parameter names (full list or subset)
        figsize: Figure size
        max_cols: Maximum columns in subplot grid
        difference: If True, plot difference from uniform CDF
        param_indices: Optional list/array of parameter indices to plot (subset)
        plot: If False, skip figure creation and return (None, None, ranks, ks_stats)

    Returns:
        fig, axes, ranks, ks_stats: dict mapping parameter name to
            {'ks_statistic': float, 'ks_p_value': float}
    """
    from scipy.stats import kstest

    # Convert to numpy if needed
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()
    if torch.is_tensor(theta_test):
        theta_test = theta_test.detach().cpu().numpy()

    # Apply parameter subsetting if specified
    if param_indices is not None:
        param_indices = list(param_indices)
        samples = samples[:, :, param_indices]
        theta_test = theta_test[:, param_indices]
        if len(param_names) != len(param_indices):
            # Assume param_names is full list, subset it
            param_names = [param_names[i] for i in param_indices]

    n_samples, n_test, n_params = samples.shape

    # Compute rank statistics (percentile of true value in posterior)
    ranks = np.zeros((n_test, n_params))

    for i in range(n_test):
        for j in range(n_params):
            # Count how many samples are <= true value
            rank = (samples[:, i, j] <= theta_test[i, j]).sum() / n_samples
            ranks[i, j] = rank

    # Compute KS statistics per parameter
    ks_stats = {}
    for j in range(n_params):
        stat, p = kstest(ranks[:, j], 'uniform')
        ks_stats[param_names[j]] = {
            'ks_statistic': float(stat),
            'ks_p_value': float(p),
        }

    if not plot:
        return None, None, ranks, ks_stats

    # Create subplots
    n_cols = min(max_cols, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for j in range(n_params):
        ax = axes[j]

        # Get ranks for this parameter
        ranks_param = ranks[:, j]
        sorted_ranks = np.sort(ranks_param)
        ecdf = np.arange(1, len(sorted_ranks) + 1) / len(sorted_ranks)

        if difference:
            # Plot empirical CDF - uniform CDF
            diff = ecdf - sorted_ranks

            ax.plot(sorted_ranks, diff, 'b-', lw=2)
            ax.axhline(0, color='r', linestyle='--', lw=1.5)
            ax.fill_between([0, 1], -1.36/np.sqrt(len(ranks_param)), 1.36/np.sqrt(len(ranks_param)),
                            color='gray', alpha=0.3)
        else:
            # Plot empirical CDF vs uniform
            ax.plot(sorted_ranks, ecdf, 'b-', lw=2)
            ax.plot([0, 1], [0, 1], 'r--', lw=1.5)

        # Break long parameter names for readability
        title = param_names[j]
        if len(title) > 20 and '_' in title[10:]:
            break_idx = title.index('_', 10)
            title = f'{title[:break_idx+1]}\n{title[break_idx+1:]}'
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.set_xlim(0, 1)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig, axes, ranks, ks_stats


def sbi_mmd_misspecification(obs_transformed_tensor, x_train, inference, mode="x_space"):
    """
    Run MMD misspecification test from sbi.

    Wraps ``sbi.diagnostics.misspecification.calc_misspecification_mmd``
    and returns a simple dict of results.

    Args:
        obs_transformed_tensor: Observed data in transformed space, shape (1, n_obs)
        x_train: Training observables in transformed space, shape (n_train, n_obs)
        inference: Trained sbi inference object (e.g. NPE instance after .train())
        mode: "x_space" or "theta_space"

    Returns:
        dict with keys: p_value, mmd_statistic, misspecified (bool at alpha=0.05)
    """
    from sbi.diagnostics.misspecification import calc_misspecification_mmd

    p_val, (mmds_baseline, mmd_observed) = calc_misspecification_mmd(
        x_obs=obs_transformed_tensor,
        x=x_train,
        inference=inference,
        mode=mode,
    )

    return {
        'p_value': float(p_val),
        'mmd_observed': float(mmd_observed),
        'misspecified': bool(p_val < 0.05),
    }


def sbi_coverage_check(x_train_raw, obs_values, observable_names):
    """
    Check whether observed values fall within the training simulation range.

    Args:
        x_train_raw: Raw (untransformed) training observables, shape (n_train, n_obs)
        obs_values: Observed values, shape (n_obs,) or dict {name: value}
        observable_names: List of observable names

    Returns:
        coverage_df: DataFrame with columns [observable, obs_value, train_min,
            train_max, train_mean, train_std, z_score, in_range]
    """
    if isinstance(obs_values, dict):
        obs_arr = np.array([float(np.squeeze(obs_values[n])) for n in observable_names])
    else:
        obs_arr = np.asarray(obs_values).flatten()

    records = []
    for i, name in enumerate(observable_names):
        obs_val = float(obs_arr[i])
        col = x_train_raw[:, i]
        train_min = float(np.nanmin(col))
        train_max = float(np.nanmax(col))
        train_mean = float(np.nanmean(col))
        train_std = float(np.nanstd(col))
        z = (obs_val - train_mean) / train_std if train_std > 0 else 0.0
        in_range = bool(train_min <= obs_val <= train_max)
        records.append({
            'observable': name,
            'obs_value': obs_val,
            'train_min': train_min,
            'train_max': train_max,
            'train_mean': train_mean,
            'train_std': train_std,
            'z_score': z,
            'in_range': in_range,
        })

    return pd.DataFrame(records)


def sbi_boundary_piling(post_samples_dict, priors_df, threshold=0.2, tail_fraction=0.05):
    """
    Detect posterior boundary piling against prior bounds.

    Parameters where >threshold of posterior mass concentrates in the outer
    tail_fraction of the prior range (in log-space) indicate potential prior
    misspecification.

    Args:
        post_samples_dict: Dict mapping parameter name to 1D array of posterior samples
            (original space, not log-transformed)
        priors_df: DataFrame with columns [name, dist_param1, dist_param2] and
            optionally [lower_bound, upper_bound]
        threshold: Fraction of posterior mass that triggers a warning (default 0.2)
        tail_fraction: Width of the boundary region as fraction of log-range (default 0.05)

    Returns:
        piling_df: DataFrame with columns [parameter, side, fraction, lower_bound,
            upper_bound, prior_median] for ALL parameters, sorted by fraction descending.
            Parameters with fraction > threshold are flagged.
    """
    records = []
    for _, row in priors_df.iterrows():
        name = row['name']
        if name not in post_samples_dict:
            continue

        samples_i = np.asarray(post_samples_dict[name])
        mu = row['dist_param1']
        sigma = row['dist_param2']

        lb = row.get('lower_bound', np.nan)
        ub = row.get('upper_bound', np.nan)
        if pd.isna(lb) or lb == 0:
            lb = float(np.exp(mu - 3 * sigma))
        else:
            lb = float(lb)
        if pd.isna(ub):
            ub = float(np.exp(mu + 3 * sigma))
        else:
            ub = float(ub)

        log_range = np.log(ub) - np.log(lb)
        if log_range <= 0:
            continue

        log_samples = np.log(samples_i)
        frac_lower = float((log_samples < np.log(lb) + tail_fraction * log_range).mean())
        frac_upper = float((log_samples > np.log(ub) - tail_fraction * log_range).mean())

        side = 'lower' if frac_lower > frac_upper else 'upper'
        frac = max(frac_lower, frac_upper)

        records.append({
            'parameter': name,
            'side': side,
            'fraction': frac,
            'frac_near_lower': frac_lower,
            'frac_near_upper': frac_upper,
            'lower_bound': lb,
            'upper_bound': ub,
            'prior_median': float(np.exp(mu)),
            'flagged': frac > threshold,
        })

    return pd.DataFrame(records).sort_values('fraction', ascending=False).reset_index(drop=True)


def sbi_prior_predictive_pvalues(x_train_raw, obs_values, observable_names):
    """
    Per-observable prior predictive p-values.

    For each observable, computes the two-sided p-value: the probability that
    a simulation from the prior produces a value at least as extreme as the
    observed value. Small p-values indicate the observation is unlikely under
    the model prior.

    Args:
        x_train_raw: Prior predictive simulations (untransformed), shape (n_sims, n_obs)
        obs_values: Observed values, shape (n_obs,) or dict {name: value}
        observable_names: List of observable names

    Returns:
        pval_df: DataFrame sorted by p-value with columns
            [observable, p_value, percentile, tail]
        fig: Horizontal bar plot of p-values
    """
    if isinstance(obs_values, dict):
        obs_arr = np.array([float(np.squeeze(obs_values[n])) for n in observable_names])
    else:
        obs_arr = np.asarray(obs_values).flatten()

    records = []
    for i, name in enumerate(observable_names):
        obs_val = float(obs_arr[i])
        sims = x_train_raw[:, i]
        valid = sims[~np.isnan(sims)]
        if len(valid) == 0:
            records.append({
                'observable': name, 'p_value': np.nan,
                'percentile': np.nan, 'tail': 'nan',
            })
            continue

        percentile = float((valid <= obs_val).mean())
        p_value = float(2 * min(percentile, 1 - percentile))
        p_value = min(p_value, 1.0)
        tail = 'lower' if percentile < 0.5 else 'upper'

        records.append({
            'observable': name,
            'p_value': p_value,
            'percentile': percentile,
            'tail': tail,
        })

    df = pd.DataFrame(records).sort_values('p_value').reset_index(drop=True)

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(observable_names) * 0.35)))
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'steelblue'
              for p in df['p_value']]
    ax.barh(range(len(df)), df['p_value'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['observable'], fontsize=9)
    ax.axvline(0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
    ax.axvline(0.1, color='orange', linestyle='--', alpha=0.7, label='α = 0.1')
    ax.set_xlabel('Prior Predictive p-value')
    ax.set_title('Prior Predictive p-values (two-sided)')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()

    return df, fig


def sbi_self_reference_null(x_samples, obs_values, observable_names, loo_bias_correction=True):
    """
    Compute a self-reference null distribution for Mahalanobis D² and per-observable
    LOO influence, using the predictive samples themselves as the null.

    Each predictive sample x_i is treated as a hypothetical "observation" and its D²
    is computed against the predictive distribution. Under correct specification,
    the actual observation should look like a typical predictive sample, so the
    empirical p-value = fraction of D²_i >= D²_obs.

    This avoids distributional assumptions (no chi-square reference needed) and
    naturally handles over-parameterization: if excess parameters allow overfitting,
    both D²_obs and the null shift together, keeping the p-value calibrated.

    Args:
        x_samples: Predictive samples (prior or posterior), shape (n_samples, n_obs).
        obs_values: Observed values, shape (n_obs,) or dict {name: value}.
        observable_names: List of observable names.
        loo_bias_correction: If True, compute each D²_i using leave-one-out
            mean and covariance (removes self-influence bias). If False, use the
            full sample mean/covariance (conservative / biased low).

    Returns:
        result: dict with keys:
            'd2_obs': float, D² of observed data against predictive distribution
            'd2_null': ndarray (n_samples,), D² of each predictive sample
            'p_value': float, empirical p-value (fraction of d2_null >= d2_obs)
            'influence_obs': DataFrame, per-observable LOO influence for obs_values
            'influence_null': DataFrame, per-observable LOO influence percentiles
                from the null distribution (columns: observable, median, q95, q99)
            'influence_p_values': DataFrame, per-observable empirical p-values
                (fraction of null influence >= observed influence)
        fig: Figure with two panels:
            Left: histogram of null D² with observed D² marked
            Right: per-observable influence with null 95th percentile bands
    """
    x = np.asarray(x_samples)
    n_samples, n_obs = x.shape

    if isinstance(obs_values, dict):
        obs = np.array([float(np.squeeze(obs_values[n])) for n in observable_names])
    else:
        obs = np.asarray(obs_values).flatten()

    # --- Full-sample statistics for D²_obs ---
    mean_x = np.mean(x, axis=0)
    cov_x = np.cov(x, rowvar=False)
    reg = 1e-6 * np.trace(cov_x) / n_obs
    cov_x += np.eye(n_obs) * reg
    cov_inv = np.linalg.inv(cov_x)

    diff_obs = obs - mean_x
    d2_obs = float(diff_obs @ cov_inv @ diff_obs)

    # --- LOO influence for observed data ---
    all_idx = list(range(n_obs))
    influence_obs = np.zeros(n_obs)
    for j in range(n_obs):
        keep = [i for i in all_idx if i != j]
        sub_mean = mean_x[keep]
        sub_cov = cov_x[np.ix_(keep, keep)]
        sub_diff = obs[keep] - sub_mean
        try:
            d_loo = float(sub_diff @ np.linalg.inv(sub_cov) @ sub_diff)
        except np.linalg.LinAlgError:
            d_loo = d2_obs  # no influence if inversion fails
        influence_obs[j] = d2_obs - d_loo

    # --- Null distribution: D² for each predictive sample ---
    d2_null = np.zeros(n_samples)
    influence_null = np.zeros((n_samples, n_obs))

    if loo_bias_correction:
        # Precompute sum and outer product for efficient LOO stats
        x_sum = np.sum(x, axis=0)
        x_outer_sum = x.T @ x
        for i in range(n_samples):
            # LOO mean and covariance (exclude sample i)
            loo_mean = (x_sum - x[i]) / (n_samples - 1)
            loo_outer = (x_outer_sum - np.outer(x[i], x[i])) / (n_samples - 2) \
                - np.outer(loo_mean, loo_mean) * (n_samples - 1) / (n_samples - 2)
            loo_cov = loo_outer + np.eye(n_obs) * reg
            try:
                loo_cov_inv = np.linalg.inv(loo_cov)
            except np.linalg.LinAlgError:
                d2_null[i] = np.nan
                continue

            diff_i = x[i] - loo_mean
            d2_null[i] = float(diff_i @ loo_cov_inv @ diff_i)

            # Per-observable LOO influence for this null sample
            for j in range(n_obs):
                keep = [k for k in all_idx if k != j]
                sub_mean = loo_mean[keep]
                sub_cov = loo_cov[np.ix_(keep, keep)]
                sub_diff = x[i, keep] - sub_mean
                try:
                    d_loo_j = float(sub_diff @ np.linalg.inv(sub_cov) @ sub_diff)
                except np.linalg.LinAlgError:
                    d_loo_j = d2_null[i]
                influence_null[i, j] = d2_null[i] - d_loo_j
    else:
        # Use full-sample mean/cov (conservative, biased low)
        for i in range(n_samples):
            diff_i = x[i] - mean_x
            d2_null[i] = float(diff_i @ cov_inv @ diff_i)

            for j in range(n_obs):
                keep = [k for k in all_idx if k != j]
                sub_mean = mean_x[keep]
                sub_cov = cov_x[np.ix_(keep, keep)]
                sub_diff = x[i, keep] - sub_mean
                try:
                    d_loo_j = float(sub_diff @ np.linalg.inv(sub_cov) @ sub_diff)
                except np.linalg.LinAlgError:
                    d_loo_j = d2_null[i]
                influence_null[i, j] = d2_null[i] - d_loo_j

    # --- Empirical p-values ---
    valid_null = d2_null[np.isfinite(d2_null)]
    p_value = float(np.mean(valid_null >= d2_obs)) if len(valid_null) > 0 else np.nan

    # Per-observable influence p-values and percentiles
    influence_records = []
    influence_p_records = []
    for j, name in enumerate(observable_names):
        null_j = influence_null[:, j]
        valid_j = null_j[np.isfinite(null_j)]
        obs_inf = influence_obs[j]
        p_j = float(np.mean(valid_j >= obs_inf)) if len(valid_j) > 0 else np.nan
        influence_records.append({
            'observable': name,
            'influence_obs': obs_inf,
            'null_median': float(np.median(valid_j)) if len(valid_j) > 0 else np.nan,
            'null_q95': float(np.percentile(valid_j, 95)) if len(valid_j) > 0 else np.nan,
            'null_q99': float(np.percentile(valid_j, 99)) if len(valid_j) > 0 else np.nan,
        })
        influence_p_records.append({
            'observable': name,
            'influence_obs': obs_inf,
            'p_value': p_j,
        })

    influence_null_df = pd.DataFrame(influence_records).sort_values(
        'influence_obs', ascending=False
    ).reset_index(drop=True)
    influence_p_df = pd.DataFrame(influence_p_records).sort_values(
        'influence_obs', ascending=False
    ).reset_index(drop=True)

    # --- Influence for obs (same format as sbi_loo_predictive_check) ---
    influence_obs_df = pd.DataFrame({
        'observable': observable_names,
        'd_full': d2_obs,
        'influence': influence_obs,
    }).sort_values('influence', ascending=False).reset_index(drop=True)

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, n_obs * 0.3)))

    # Left panel: D² null histogram
    ax1.hist(valid_null, bins=min(50, max(10, len(valid_null) // 5)),
             color='steelblue', alpha=0.7, edgecolor='white', density=True)
    ax1.axvline(d2_obs, color='red', linewidth=2, linestyle='-',
                label=f'D²_obs = {d2_obs:.1f} (p = {p_value:.3f})')
    ax1.set_xlabel('Mahalanobis D²')
    ax1.set_ylabel('Density')
    ax1.set_title('Self-Reference Null Distribution')
    ax1.legend(fontsize=9)

    # Right panel: per-observable influence with null bands
    inf_sorted = influence_null_df
    y_pos = range(len(inf_sorted))
    ax2.barh(y_pos, inf_sorted['influence_obs'], color='steelblue', alpha=0.8,
             label='Observed influence')
    ax2.scatter(inf_sorted['null_q95'], y_pos, color='orange', marker='|', s=200,
                linewidths=2, label='Null 95th pctl', zorder=5)
    # Highlight observables where influence exceeds null 95th percentile
    for idx, (_, row) in enumerate(inf_sorted.iterrows()):
        if row['influence_obs'] > row['null_q95']:
            ax2.barh(idx, row['influence_obs'], color='red', alpha=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(inf_sorted['observable'], fontsize=9)
    ax2.set_xlabel('LOO Influence (ΔD²)')
    ax2.set_title('Per-Observable Influence vs Null')
    ax2.legend(fontsize=9)
    ax2.invert_yaxis()

    plt.tight_layout()

    result = {
        'd2_obs': d2_obs,
        'd2_null': valid_null,
        'p_value': p_value,
        'influence_obs': influence_obs_df,
        'influence_null': influence_null_df,
        'influence_p_values': influence_p_df,
    }

    return result, fig


def sbi_loo_predictive_check(x_train, obs_values, observable_names):
    """
    Leave-one-out predictive check: identify which observables drive misspecification.

    Computes the squared Mahalanobis distance of the observation from the
    training distribution, then repeats with each observable excluded.
    The "influence" of observable j is how much the distance decreases when
    j is removed. Large positive influence = j contributes to joint misspecification.

    Args:
        x_train: Training simulations (raw or transformed), shape (n_train, n_obs)
        obs_values: Observed values (same space as x_train), shape (n_obs,) or dict
        observable_names: List of observable names

    Returns:
        loo_df: DataFrame with columns [observable, d_full, d_loo, influence],
            sorted by influence descending
        fig: Horizontal bar plot of per-observable influences
    """
    x = np.asarray(x_train)
    if isinstance(obs_values, dict):
        obs = np.array([float(np.squeeze(obs_values[n])) for n in observable_names])
    else:
        obs = np.asarray(obs_values).flatten()

    n_obs = len(observable_names)
    mean_x = np.mean(x, axis=0)
    cov_x = np.cov(x, rowvar=False)

    # Regularize covariance
    reg = 1e-6 * np.trace(cov_x) / n_obs
    cov_x += np.eye(n_obs) * reg

    # Full Mahalanobis distance
    diff = obs - mean_x
    cov_inv = np.linalg.inv(cov_x)
    d_full = float(diff @ cov_inv @ diff)

    # LOO for each observable
    records = []
    all_idx = list(range(n_obs))
    for j, name in enumerate(observable_names):
        keep = [i for i in all_idx if i != j]
        sub_mean = mean_x[keep]
        sub_cov = cov_x[np.ix_(keep, keep)]
        sub_diff = obs[keep] - sub_mean
        try:
            sub_cov_inv = np.linalg.inv(sub_cov)
            d_loo = float(sub_diff @ sub_cov_inv @ sub_diff)
        except np.linalg.LinAlgError:
            d_loo = np.nan

        records.append({
            'observable': name,
            'd_full': d_full,
            'd_loo': d_loo,
            'influence': d_full - d_loo if not np.isnan(d_loo) else np.nan,
        })

    df = pd.DataFrame(records).sort_values('influence', ascending=False).reset_index(drop=True)

    # Bar plot
    avg_influence = d_full / n_obs
    fig, ax = plt.subplots(figsize=(10, max(6, len(observable_names) * 0.35)))
    colors = ['red' if inf > avg_influence else 'steelblue' for inf in df['influence']]
    ax.barh(range(len(df)), df['influence'], color=colors)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['observable'], fontsize=9)
    ax.axvline(avg_influence, color='gray', linestyle='--', alpha=0.7,
               label=f'Avg influence ({avg_influence:.1f})')
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Mahalanobis Influence (ΔD²)')
    ax.set_title(f'LOO Predictive Check (D²_full = {d_full:.1f})')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()

    return df, fig


def sbi_posterior_predictive_check(
    theta_train, x_train_raw, posterior_samples, obs_values,
    observable_names, k=10,
):
    """
    Approximate posterior predictive check using nearest-neighbor proxy.

    For each posterior sample, finds its k nearest neighbors in theta_train
    and uses their simulation outputs as approximate posterior predictive
    samples. No new simulations are required.

    Args:
        theta_train: Training parameters, shape (n_train, n_params)
        x_train_raw: Training observables (raw/untransformed), shape (n_train, n_obs)
        posterior_samples: Posterior samples for a single observation,
            shape (n_post, n_params)
        obs_values: Observed values, shape (n_obs,) or dict {name: value}
        observable_names: List of observable names
        k: Number of nearest neighbors per posterior sample

    Returns:
        ppc_df: DataFrame with per-observable PPC statistics
            [observable, obs_value, ppc_mean, ppc_median, ppc_std,
             ppc_q05, ppc_q95, ppc_p_value, obs_in_90ci]
        fig: Histograms of approximate PPC distributions vs observed values
    """
    from sklearn.neighbors import NearestNeighbors

    if isinstance(obs_values, dict):
        obs_arr = np.array([float(np.squeeze(obs_values[n])) for n in observable_names])
    else:
        obs_arr = np.asarray(obs_values).flatten()

    if torch.is_tensor(theta_train):
        theta_train = theta_train.detach().cpu().numpy()
    if torch.is_tensor(posterior_samples):
        posterior_samples = posterior_samples.detach().cpu().numpy()
    x_raw = np.asarray(x_train_raw)

    # Fit nearest neighbors on training parameters
    nn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn.fit(theta_train)

    # Find neighbors for each posterior sample
    _, indices = nn.kneighbors(posterior_samples)  # (n_post, k)

    # Gather approximate posterior predictive samples
    ppc_samples = x_raw[indices.flatten()]  # (n_post * k, n_obs)

    n_obs = len(observable_names)
    records = []
    for i, name in enumerate(observable_names):
        ppc_col = ppc_samples[:, i]
        valid = ppc_col[~np.isnan(ppc_col)]
        if len(valid) == 0:
            records.append({
                'observable': name, 'obs_value': float(obs_arr[i]),
                'ppc_mean': np.nan, 'ppc_median': np.nan, 'ppc_std': np.nan,
                'ppc_q05': np.nan, 'ppc_q95': np.nan,
                'ppc_p_value': np.nan, 'obs_in_90ci': False,
            })
            continue

        ppc_mean = float(np.mean(valid))
        ppc_std = float(np.std(valid))
        ppc_median = float(np.median(valid))
        ppc_q05 = float(np.percentile(valid, 5))
        ppc_q95 = float(np.percentile(valid, 95))

        percentile = float((valid <= obs_arr[i]).mean())
        p_value = float(2 * min(percentile, 1 - percentile))
        p_value = min(p_value, 1.0)

        records.append({
            'observable': name,
            'obs_value': float(obs_arr[i]),
            'ppc_mean': ppc_mean,
            'ppc_median': ppc_median,
            'ppc_std': ppc_std,
            'ppc_q05': ppc_q05,
            'ppc_q95': ppc_q95,
            'ppc_p_value': p_value,
            'obs_in_90ci': bool(ppc_q05 <= obs_arr[i] <= ppc_q95),
        })

    df = pd.DataFrame(records)

    # Histogram grid
    n_cols = min(4, n_obs)
    n_rows = int(np.ceil(n_obs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(observable_names):
        ax = axes[i]
        ppc_col = ppc_samples[:, i]
        valid = ppc_col[~np.isnan(ppc_col)]
        if len(valid) > 0:
            ax.hist(valid, bins=30, density=True, alpha=0.7, color='steelblue')
        ax.axvline(obs_arr[i], color='red', linewidth=2, label='Observed')
        ax.set_title(name, fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_obs, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Approximate Posterior Predictive Check (NN proxy)', fontsize=12)
    plt.tight_layout()

    return df, fig


def sbi_posterior_predictive_coverage(x_posterior_pred, obs_values, observable_names,
                                      log_scale=False):
    """
    Per-observable coverage check using actual posterior predictive simulations.

    Unlike sbi_posterior_predictive_check (which uses kNN proxy), this function
    takes actual posterior predictive simulation outputs and checks whether
    observed values fall within the posterior predictive distribution.

    Args:
        x_posterior_pred: Posterior predictive simulations (raw space),
            shape (n_sims, n_obs). These are actual QSP simulation outputs
            from posterior parameter samples.
        obs_values: Observed values, shape (n_obs,) or dict {name: value}
        observable_names: List of observable names
        log_scale: If True, plot histograms on log x-axis (useful for
            lognormal-distributed observables with long right tails).
            Observables with non-positive values fall back to linear scale.

    Returns:
        coverage_df: DataFrame with columns [observable, obs_value, pp_mean,
            pp_median, pp_q05, pp_q95, p_value, in_90ci]
        fig: Histogram grid of posterior predictive distributions vs observed
    """
    if isinstance(obs_values, dict):
        obs_arr = np.array([float(np.squeeze(obs_values[n])) for n in observable_names])
    else:
        obs_arr = np.asarray(obs_values).flatten()

    x_pp = np.asarray(x_posterior_pred)
    n_obs = len(observable_names)

    records = []
    for i, name in enumerate(observable_names):
        col = x_pp[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            records.append({
                'observable': name, 'obs_value': float(obs_arr[i]),
                'pp_mean': np.nan, 'pp_median': np.nan,
                'pp_q05': np.nan, 'pp_q95': np.nan,
                'p_value': np.nan, 'in_90ci': False,
            })
            continue

        pp_mean = float(np.mean(valid))
        pp_median = float(np.median(valid))
        pp_q05 = float(np.percentile(valid, 5))
        pp_q95 = float(np.percentile(valid, 95))

        percentile = float((valid <= obs_arr[i]).mean())
        p_value = float(2 * min(percentile, 1 - percentile))
        p_value = min(p_value, 1.0)

        records.append({
            'observable': name,
            'obs_value': float(obs_arr[i]),
            'pp_mean': pp_mean,
            'pp_median': pp_median,
            'pp_q05': pp_q05,
            'pp_q95': pp_q95,
            'p_value': p_value,
            'in_90ci': bool(pp_q05 <= obs_arr[i] <= pp_q95),
        })

    coverage_df = pd.DataFrame(records)

    # Histogram grid
    n_cols = min(4, n_obs)
    n_rows = int(np.ceil(n_obs / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    for i, name in enumerate(observable_names):
        ax = axes[i]
        col = x_pp[:, i]
        valid = col[~np.isnan(col)]

        # Decide whether to use log scale for this observable
        use_log = log_scale and len(valid) > 0 and np.all(valid > 0) and obs_arr[i] > 0

        if len(valid) > 0:
            if use_log:
                log_valid = np.log10(valid)
                ax.hist(log_valid, bins=30, density=True, alpha=0.7, color='darkorange')
                ax.axvline(np.log10(obs_arr[i]), color='red', linewidth=2, label='Observed')
                # Relabel x-ticks as powers of 10
                from matplotlib.ticker import FixedLocator, FixedFormatter
                ticks = ax.get_xticks()
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FixedFormatter(
                    [f"$10^{{{t:.0f}}}$" if t == int(t)
                     else f"$10^{{{t:.1f}}}$" for t in ticks]
                ))
                ax.tick_params(axis='x', labelsize=6)
            else:
                ax.hist(valid, bins=30, density=True, alpha=0.7, color='darkorange')
                ax.axvline(obs_arr[i], color='red', linewidth=2, label='Observed')
        else:
            ax.axvline(obs_arr[i], color='red', linewidth=2, label='Observed')

        row = coverage_df.loc[coverage_df['observable'] == name]
        if len(row) > 0:
            p_str = f"p={float(row['p_value'].iloc[0]):.3f}"
            in_ci = "in 90CI" if bool(row['in_90ci'].iloc[0]) else "outside 90CI"
            ax.set_title(f"{name}\n{p_str}, {in_ci}", fontsize=7)
        else:
            ax.set_title(name, fontsize=8)
        if i == 0:
            ax.legend(fontsize=7)

    for i in range(n_obs, len(axes)):
        axes[i].axis('off')

    fig.suptitle('Posterior Predictive Coverage (Actual Simulations)', fontsize=12)
    plt.tight_layout()

    return coverage_df, fig


def sbi_posterior_correlations(posterior_samples, param_names, threshold=0.5, figsize=None):
    """
    Pairwise posterior correlation matrix.

    Computes the Pearson correlation matrix of posterior samples and
    identifies strongly correlated parameter pairs.

    Args:
        posterior_samples: Posterior samples, shape (n_samples, n_params).
            Can be a 2D array or a dict {param_name: 1D array}.
        param_names: List of parameter names
        threshold: Flag pairs with |correlation| above this value
        figsize: Figure size (auto-scaled if None)

    Returns:
        corr_df: Full correlation matrix as DataFrame (param x param)
        strong_pairs: DataFrame of pairs with |corr| > threshold, columns
            [param_1, param_2, correlation, abs_correlation]
        fig: Heatmap of correlation matrix
    """
    if isinstance(posterior_samples, dict):
        posterior_samples = np.column_stack([
            posterior_samples[n] for n in param_names
        ])
    if torch.is_tensor(posterior_samples):
        posterior_samples = posterior_samples.detach().cpu().numpy()

    n_params = len(param_names)
    corr_matrix = np.corrcoef(posterior_samples, rowvar=False)

    corr_df = pd.DataFrame(corr_matrix, index=param_names, columns=param_names)

    # Extract strong pairs
    pairs = []
    for i in range(n_params):
        for j in range(i + 1, n_params):
            r = float(corr_matrix[i, j])
            if abs(r) > threshold:
                pairs.append({
                    'param_1': param_names[i],
                    'param_2': param_names[j],
                    'correlation': r,
                    'abs_correlation': abs(r),
                })
    strong_pairs = (
        pd.DataFrame(pairs).sort_values('abs_correlation', ascending=False).reset_index(drop=True)
        if pairs else pd.DataFrame(columns=['param_1', 'param_2', 'correlation', 'abs_correlation'])
    )

    # Heatmap
    if figsize is None:
        size = max(10, n_params * 0.25)
        figsize = (size, size)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8)

    if n_params <= 50:
        ax.set_xticks(range(n_params))
        ax.set_xticklabels(param_names, rotation=90, fontsize=6)
        ax.set_yticks(range(n_params))
        ax.set_yticklabels(param_names, fontsize=6)

    ax.set_title(
        f'Posterior Correlations ({len(strong_pairs)} pairs with |r| > {threshold})'
    )
    plt.tight_layout()

    return corr_df, strong_pairs, fig


def sbi_learning_curve(
    theta_train, x_train, theta_test,
    train_and_sample_fn,
    param_names,
    fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    seed=42,
):
    """
    Sample-size scaling: retrain at various fractions of training data.

    Evaluates how posterior quality (R²) improves with more training data.
    A saturating curve suggests more simulations won't help; a rising curve
    suggests the network needs more data.

    Args:
        theta_train: Full training parameters (log-transformed), shape (n_train, n_params)
        x_train: Full training observables (transformed), shape (n_train, n_obs)
        theta_test: Test parameters (log-transformed), shape (n_test, n_params)
        train_and_sample_fn: Callable(theta_sub, x_sub, seed) -> samples
            where samples has shape (n_post, n_test, n_params). The function
            should train an NPE on the provided subset and sample posterior
            for each test observation.
        param_names: List of parameter names
        fractions: Fractions of training data to use
        seed: Random seed for subset selection

    Returns:
        curve_df: DataFrame with columns [fraction, n_train, r2_median,
            r2_mean, r2_q25, r2_q75, n_r2_positive, n_r2_above_0.5]
        fig: Learning curve plot (R² vs training set size)
    """
    from sklearn.metrics import r2_score

    if torch.is_tensor(theta_train):
        theta_train_np = theta_train.detach().cpu().numpy()
    else:
        theta_train_np = np.asarray(theta_train)
    if torch.is_tensor(theta_test):
        theta_test_np = theta_test.detach().cpu().numpy()
    else:
        theta_test_np = np.asarray(theta_test)

    n_train_full = len(theta_train_np)
    rng = np.random.RandomState(seed)

    records = []
    for frac in fractions:
        n_sub = max(100, int(frac * n_train_full))
        n_sub = min(n_sub, n_train_full)

        # Random subset
        idx = rng.choice(n_train_full, size=n_sub, replace=False)
        if torch.is_tensor(theta_train):
            theta_sub = theta_train[idx]
            x_sub = x_train[idx]
        else:
            theta_sub = theta_train_np[idx]
            x_sub = np.asarray(x_train)[idx]

        # Train and sample
        samples = train_and_sample_fn(theta_sub, x_sub, seed)
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()

        # Compute R² per parameter
        median_post = np.median(samples, axis=0)  # (n_test, n_params)
        r2_vals = []
        for j in range(len(param_names)):
            r2 = float(r2_score(theta_test_np[:, j], median_post[:, j]))
            r2_vals.append(r2)

        records.append({
            'fraction': float(frac),
            'n_train': n_sub,
            'r2_median': float(np.median(r2_vals)),
            'r2_mean': float(np.mean(r2_vals)),
            'r2_q25': float(np.percentile(r2_vals, 25)),
            'r2_q75': float(np.percentile(r2_vals, 75)),
            'n_r2_positive': int(sum(r > 0 for r in r2_vals)),
            'n_r2_above_0.5': int(sum(r > 0.5 for r in r2_vals)),
        })

    curve_df = pd.DataFrame(records)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(curve_df['n_train'], curve_df['r2_median'], 'o-', label='Median R²')
    ax.fill_between(
        curve_df['n_train'], curve_df['r2_q25'], curve_df['r2_q75'], alpha=0.3
    )
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('R²')
    ax.set_title('Learning Curve: Recovery R²')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(curve_df['n_train'], curve_df['n_r2_above_0.5'],
            'o-', color='green', label='R² > 0.5')
    ax.plot(curve_df['n_train'], curve_df['n_r2_positive'],
            's-', color='orange', label='R² > 0')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Learning Curve: Well-Recovered Parameters')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return curve_df, fig


def sbi_seed_stability(
    theta_train, x_train, theta_test,
    train_and_sample_fn,
    param_names,
    seeds=(42, 123, 456, 789, 1234),
):
    """
    Seed stability: retrain with different random seeds and compare posteriors.

    High variance across seeds indicates the posterior estimate is noisy and
    may benefit from ensembling or more training data.

    Args:
        theta_train: Training parameters (log-transformed), shape (n_train, n_params)
        x_train: Training observables (transformed), shape (n_train, n_obs)
        theta_test: Test parameters (log-transformed), shape (n_test, n_params)
        train_and_sample_fn: Callable(theta, x, seed) -> samples
            where samples has shape (n_post, n_test, n_params)
        param_names: List of parameter names
        seeds: Random seeds to test

    Returns:
        stability_df: Long-format DataFrame [seed, parameter, r2]
        summary_df: Per-parameter summary [parameter, r2_mean, r2_std,
            r2_min, r2_max, r2_range], sorted by r2_mean descending
        fig: Stability plots (R² distributions across seeds, top-20 bar chart)
    """
    from sklearn.metrics import r2_score

    if torch.is_tensor(theta_test):
        theta_test_np = theta_test.detach().cpu().numpy()
    else:
        theta_test_np = np.asarray(theta_test)

    all_r2 = {}  # seed -> {param: r2}

    for s in seeds:
        samples = train_and_sample_fn(theta_train, x_train, s)
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()

        median_post = np.median(samples, axis=0)
        r2_vals = {}
        for j, name in enumerate(param_names):
            r2_vals[name] = float(r2_score(theta_test_np[:, j], median_post[:, j]))
        all_r2[s] = r2_vals

    # Long-format DataFrame
    stability_rows = []
    for s, r2_vals in all_r2.items():
        for name, r2 in r2_vals.items():
            stability_rows.append({'seed': s, 'parameter': name, 'r2': r2})
    stability_df = pd.DataFrame(stability_rows)

    # Per-parameter summary
    summary_rows = []
    for name in param_names:
        vals = [all_r2[s][name] for s in seeds]
        summary_rows.append({
            'parameter': name,
            'r2_mean': float(np.mean(vals)),
            'r2_std': float(np.std(vals)),
            'r2_min': float(np.min(vals)),
            'r2_max': float(np.max(vals)),
            'r2_range': float(np.max(vals) - np.min(vals)),
        })
    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values('r2_mean', ascending=False)
        .reset_index(drop=True)
    )

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² distribution per seed
    ax = axes[0]
    for s in seeds:
        r2_vals = list(all_r2[s].values())
        ax.hist(r2_vals, bins=20, alpha=0.3, label=f'seed={s}')
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title('R² Distribution Across Seeds')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Top-20 parameters by mean R² with error bars
    ax = axes[1]
    top = summary_df.head(20)
    x_pos = range(len(top))
    ax.bar(x_pos, top['r2_mean'], yerr=top['r2_std'], capsize=3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top['parameter'], rotation=90, fontsize=7)
    ax.set_ylabel('R² (mean ± std)')
    ax.set_title('Seed Stability: Top 20 Parameters')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return stability_df, summary_df, fig


def sbi_dimensionality_sweep(
    theta_train, x_train, theta_test,
    train_and_sample_fn,
    param_names,
    param_ranking,
    subsets=(10, 20, 30, 50),
    seed=42,
):
    """
    Dimensionality sweep: retrain NPE with top-N parameters and measure recovery.

    Tests whether reducing parameter dimensionality improves identifiability.
    For each subset size, selects the top-N parameters by sensitivity ranking,
    retrains the NPE on the subsetted theta (keeping all observables), and
    computes R² recovery metrics.

    Args:
        theta_train: Training parameters (log-transformed), shape (n_train, n_params)
        x_train: Training observables (transformed), shape (n_train, n_obs)
        theta_test: Test parameters (log-transformed), shape (n_test, n_params)
        train_and_sample_fn: Callable(theta_sub, x_sub, seed) -> samples
            where samples has shape (n_post, n_test, n_params_sub).
            The callable must construct an NPE prior matching theta_sub's
            dimensionality (inferred from theta_sub.shape[1]).
        param_names: List of all parameter names (matching theta columns)
        param_ranking: List of parameter names in priority order (best first).
            Names not in param_names are silently ignored.
        subsets: Tuple of top-N values to test. The full parameter set is
            appended automatically.
        seed: Random seed for training

    Returns:
        sweep_df: DataFrame with columns [n_params, r2_median, r2_mean,
            r2_q25, r2_q75, n_r2_positive, n_r2_above_0.5]
        per_param_df: Long-format DataFrame [n_params, parameter, r2]
        fig: 2-panel plot (R² vs D, well-recovered count vs D)
    """
    from sklearn.metrics import r2_score

    if torch.is_tensor(theta_train):
        theta_train_np = theta_train.detach().cpu().numpy()
    else:
        theta_train_np = np.asarray(theta_train)
    if torch.is_tensor(theta_test):
        theta_test_np = theta_test.detach().cpu().numpy()
    else:
        theta_test_np = np.asarray(theta_test)

    n_full = len(param_names)

    # Build ordered list of param indices from ranking, filtering to active params
    ranked_indices = []
    for name in param_ranking:
        if name in param_names:
            ranked_indices.append(param_names.index(name))
    # Append any params not in ranking (preserves all params for full set)
    for i in range(n_full):
        if i not in ranked_indices:
            ranked_indices.append(i)

    # Ensure full set is included, deduplicate subset sizes
    subset_sizes = sorted(set(s for s in subsets if s < n_full))
    subset_sizes.append(n_full)

    records = []
    per_param_rows = []

    for n_sub in subset_sizes:
        top_idx = ranked_indices[:n_sub]
        sub_names = [param_names[i] for i in top_idx]

        # Subset theta columns
        if torch.is_tensor(theta_train):
            theta_sub_train = theta_train[:, top_idx]
            theta_sub_test = theta_test[:, top_idx]
        else:
            theta_sub_train = theta_train_np[:, top_idx]
            theta_sub_test = theta_test_np[:, top_idx]

        # Train and sample (x stays full-dimensional)
        samples = train_and_sample_fn(theta_sub_train, x_train, seed)
        if torch.is_tensor(samples):
            samples = samples.detach().cpu().numpy()

        # Subset test theta for R² computation
        theta_sub_test_np = theta_test_np[:, top_idx]

        # Compute R² per parameter
        median_post = np.median(samples, axis=0)  # (n_test, n_sub)
        r2_vals = []
        for j in range(n_sub):
            r2 = float(r2_score(theta_sub_test_np[:, j], median_post[:, j]))
            r2_vals.append(r2)
            per_param_rows.append({
                'n_params': n_sub,
                'parameter': sub_names[j],
                'r2': r2,
            })

        records.append({
            'n_params': n_sub,
            'r2_median': float(np.median(r2_vals)),
            'r2_mean': float(np.mean(r2_vals)),
            'r2_q25': float(np.percentile(r2_vals, 25)),
            'r2_q75': float(np.percentile(r2_vals, 75)),
            'n_r2_positive': int(sum(r > 0 for r in r2_vals)),
            'n_r2_above_0.5': int(sum(r > 0.5 for r in r2_vals)),
        })

    sweep_df = pd.DataFrame(records)
    per_param_df = pd.DataFrame(per_param_rows)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(sweep_df['n_params'], sweep_df['r2_median'], 'o-', label='Median R²')
    ax.fill_between(
        sweep_df['n_params'], sweep_df['r2_q25'], sweep_df['r2_q75'], alpha=0.3
    )
    ax.set_xlabel('Number of Parameters (D)')
    ax.set_ylabel('R²')
    ax.set_title('Dimensionality Sweep: Recovery R²')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(sweep_df['n_params'], sweep_df['n_r2_above_0.5'],
            'o-', color='green', label='R² > 0.5')
    ax.plot(sweep_df['n_params'], sweep_df['n_r2_positive'],
            's-', color='orange', label='R² > 0')
    ax.set_xlabel('Number of Parameters (D)')
    ax.set_ylabel('Number of Parameters')
    ax.set_title('Dimensionality Sweep: Well-Recovered Parameters')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return sweep_df, per_param_df, fig


def save_diagnostics(
    output_dir,
    *,
    r2_values=None,
    z_scores=None,
    contractions=None,
    param_names=None,
    ks_stats=None,
    coverage_df=None,
    piling_df=None,
    mmd_p_value=None,
    prior_predictive_pval_df=None,
    loo_df=None,
    ppc_df=None,
    corr_strong_pairs=None,
    learning_curve_df=None,
    seed_stability_summary_df=None,
    dimensionality_sweep_df=None,
    training_metadata=None,
):
    """
    Save all diagnostic numerical results to a directory.

    Writes CSVs for tabular data and a summary JSON for scalar metrics.
    All arguments are optional — only provided diagnostics are saved.

    Args:
        output_dir: Directory to write files into (created if needed)
        r2_values: Dict {param_name: R²} from sbi_recovery
        z_scores: Array (n_test, n_params) from sbi_z_score_contraction
        contractions: Array (n_test, n_params) from sbi_z_score_contraction
        param_names: List of parameter names (required if z_scores/contractions provided)
        ks_stats: Dict {param_name: {ks_statistic, ks_p_value}} from sbi_calibration_ecdf
        coverage_df: DataFrame from sbi_coverage_check
        piling_df: DataFrame from sbi_boundary_piling
        mmd_p_value: Float from calc_misspecification_mmd
        prior_predictive_pval_df: DataFrame from sbi_prior_predictive_pvalues
        loo_df: DataFrame from sbi_loo_predictive_check
        ppc_df: DataFrame from sbi_posterior_predictive_check
        corr_strong_pairs: DataFrame from sbi_posterior_correlations (strong pairs)
        learning_curve_df: DataFrame from sbi_learning_curve
        seed_stability_summary_df: DataFrame from sbi_seed_stability (summary)
        dimensionality_sweep_df: DataFrame from sbi_dimensionality_sweep (sweep summary)
        training_metadata: Optional dict of training info (n_train, n_test, n_params, etc.)

    Returns:
        Dict of file paths written
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    # Recovery R²
    if r2_values is not None:
        df = pd.DataFrame([
            {'parameter': k, 'r2': v} for k, v in r2_values.items()
        ]).sort_values('r2', ascending=False)
        path = output_dir / 'recovery_r2.csv'
        df.to_csv(path, index=False)
        written['recovery_r2'] = str(path)

    # Z-score / contraction
    if z_scores is not None and contractions is not None and param_names is not None:
        z_np = z_scores.detach().cpu().numpy() if torch.is_tensor(z_scores) else np.asarray(z_scores)
        c_np = contractions.detach().cpu().numpy() if torch.is_tensor(contractions) else np.asarray(contractions)
        records = []
        for i, name in enumerate(param_names):
            z_col = z_np[:, i]
            c_col = c_np[:, i]
            records.append({
                'parameter': name,
                'z_score_mean': float(np.mean(z_col)),
                'z_score_std': float(np.std(z_col)),
                'z_score_median': float(np.median(z_col)),
                'z_score_abs_mean': float(np.mean(np.abs(z_col))),
                'contraction_mean': float(np.mean(c_col)),
                'contraction_std': float(np.std(c_col)),
                'contraction_median': float(np.median(c_col)),
            })
        path = output_dir / 'z_score_contraction.csv'
        pd.DataFrame(records).to_csv(path, index=False)
        written['z_score_contraction'] = str(path)

    # Calibration ECDF / KS stats
    if ks_stats is not None:
        records = [
            {'parameter': k, **v} for k, v in ks_stats.items()
        ]
        path = output_dir / 'calibration_ks.csv'
        pd.DataFrame(records).to_csv(path, index=False)
        written['calibration_ks'] = str(path)

    # Coverage
    if coverage_df is not None:
        path = output_dir / 'coverage.csv'
        coverage_df.to_csv(path, index=False)
        written['coverage'] = str(path)

    # Boundary piling
    if piling_df is not None:
        path = output_dir / 'boundary_piling.csv'
        piling_df.to_csv(path, index=False)
        written['boundary_piling'] = str(path)

    # Prior predictive p-values
    if prior_predictive_pval_df is not None:
        path = output_dir / 'prior_predictive_pvalues.csv'
        prior_predictive_pval_df.to_csv(path, index=False)
        written['prior_predictive_pvalues'] = str(path)

    # LOO predictive check
    if loo_df is not None:
        path = output_dir / 'loo_predictive.csv'
        loo_df.to_csv(path, index=False)
        written['loo_predictive'] = str(path)

    # Posterior predictive check
    if ppc_df is not None:
        path = output_dir / 'posterior_predictive.csv'
        ppc_df.to_csv(path, index=False)
        written['posterior_predictive'] = str(path)

    # Posterior correlations (strong pairs)
    if corr_strong_pairs is not None:
        path = output_dir / 'posterior_correlations.csv'
        corr_strong_pairs.to_csv(path, index=False)
        written['posterior_correlations'] = str(path)

    # Learning curve
    if learning_curve_df is not None:
        path = output_dir / 'learning_curve.csv'
        learning_curve_df.to_csv(path, index=False)
        written['learning_curve'] = str(path)

    # Seed stability
    if seed_stability_summary_df is not None:
        path = output_dir / 'seed_stability.csv'
        seed_stability_summary_df.to_csv(path, index=False)
        written['seed_stability'] = str(path)

    # Dimensionality sweep
    if dimensionality_sweep_df is not None:
        path = output_dir / 'dimensionality_sweep.csv'
        dimensionality_sweep_df.to_csv(path, index=False)
        written['dimensionality_sweep'] = str(path)

    # Summary JSON (scalar metrics + metadata)
    summary = {}
    if mmd_p_value is not None:
        summary['mmd_p_value'] = float(mmd_p_value)
    if r2_values is not None:
        vals = list(r2_values.values())
        summary['r2_median'] = float(np.median(vals))
        summary['r2_mean'] = float(np.mean(vals))
        summary['r2_min'] = float(np.min(vals))
        summary['n_params_r2_above_0.5'] = int(sum(v > 0.5 for v in vals))
    if ks_stats is not None:
        ks_ps = [v['ks_p_value'] for v in ks_stats.values()]
        summary['n_params_ks_reject_0.05'] = int(sum(p < 0.05 for p in ks_ps))
    if piling_df is not None:
        summary['n_params_boundary_piling'] = int(piling_df['flagged'].sum())
    if coverage_df is not None:
        summary['n_observables_out_of_range'] = int((~coverage_df['in_range']).sum())
    if prior_predictive_pval_df is not None:
        pvals = prior_predictive_pval_df['p_value'].dropna()
        summary['n_obs_prior_pval_below_0.05'] = int((pvals < 0.05).sum())
        summary['n_obs_prior_pval_below_0.1'] = int((pvals < 0.1).sum())
    if ppc_df is not None:
        ci_col = 'obs_in_90ci' if 'obs_in_90ci' in ppc_df.columns else 'in_90ci'
        summary['n_obs_ppc_in_90ci'] = int(ppc_df[ci_col].sum())
        summary['n_obs_ppc_total'] = int(len(ppc_df))
    if corr_strong_pairs is not None:
        summary['n_strong_correlations'] = int(len(corr_strong_pairs))
    if learning_curve_df is not None:
        summary['learning_curve_final_r2_median'] = float(
            learning_curve_df['r2_median'].iloc[-1]
        )
    if seed_stability_summary_df is not None:
        summary['seed_r2_std_median'] = float(
            seed_stability_summary_df['r2_std'].median()
        )
    if dimensionality_sweep_df is not None:
        best_row = dimensionality_sweep_df.loc[
            dimensionality_sweep_df['r2_median'].idxmax()
        ]
        summary['dimensionality_sweep_best_r2_median'] = float(best_row['r2_median'])
        summary['dimensionality_sweep_best_n_params'] = int(best_row['n_params'])
    if training_metadata is not None:
        summary['training'] = training_metadata

    if summary:
        path = output_dir / 'summary.json'
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        written['summary'] = str(path)

    return written

"""
Optimal Bayesian Experimental Design (OBED) utilities for SBI workflows.

Identifies which observables or parameter priors would most reduce uncertainty
in decision-relevant predictive outputs (e.g., ORR, treatment response).

Components:
- Response classification (MPR, RECIST) from tumor volume trajectories
- Mutual information estimation (KSG for continuous, binned for binary)
- MI sweep over feature matrices
- LOO retraining harness for observable value decomposition
"""

import numpy as np
import pandas as pd
from scipy.special import digamma
from sklearn.neighbors import KDTree


# =============================================================================
# RESPONSE CLASSIFICATION
# =============================================================================


def classify_mpr(
    tumor_volume_trajectory: np.ndarray,
    time_points: np.ndarray | None = None,
    viable_threshold: float = 0.35,
    baseline_index: int = 0,
) -> np.ndarray:
    """Classify virtual patients by Major Pathologic Response (MPR).

    MPR is defined as < viable_threshold fraction of baseline tumor volume
    remaining at resection (final timepoint).

    Args:
        tumor_volume_trajectory: (n_patients, n_timepoints) tumor volumes
        time_points: (n_timepoints,) time values — unused, kept for API consistency
        viable_threshold: Fraction of baseline volume below which MPR is declared
        baseline_index: Index of baseline timepoint (default 0)

    Returns:
        Boolean array (n_patients,) — True = responder (MPR)
    """
    baseline_vol = tumor_volume_trajectory[:, baseline_index]
    final_vol = tumor_volume_trajectory[:, -1]

    with np.errstate(divide="ignore", invalid="ignore"):
        viable_fraction = np.where(
            baseline_vol > 0,
            final_vol / baseline_vol,
            1.0,
        )

    return viable_fraction < viable_threshold


def classify_recist(
    tumor_volume_trajectory: np.ndarray,
    time_points: np.ndarray | None = None,
    baseline_index: int = 0,
) -> np.ndarray:
    """Classify virtual patients by RECIST 1.1 criteria from volume trajectories.

    Volume-based thresholds (derived from diameter criteria):
    - CR: volume < 1% of baseline (effectively gone)
    - PR: volume decreased >= 65.7% from baseline (30% diameter decrease)
    - PD: volume increased >= 72.8% from nadir (20% diameter increase)
    - SD: everything else

    Args:
        tumor_volume_trajectory: (n_patients, n_timepoints) tumor volumes
        time_points: (n_timepoints,) time values — unused, kept for API consistency
        baseline_index: Index of baseline timepoint

    Returns:
        Array of strings (n_patients,) with values 'CR', 'PR', 'SD', 'PD'
    """
    n_patients = tumor_volume_trajectory.shape[0]
    baseline_vol = tumor_volume_trajectory[:, baseline_index]
    nadir_vol = np.min(tumor_volume_trajectory[:, baseline_index:], axis=1)
    final_vol = tumor_volume_trajectory[:, -1]

    categories = np.full(n_patients, "SD", dtype="U2")

    with np.errstate(divide="ignore", invalid="ignore"):
        cr_mask = (baseline_vol > 0) & (final_vol / baseline_vol < 0.01)
        categories[cr_mask] = "CR"

        pr_mask = ~cr_mask & (baseline_vol > 0) & (
            (baseline_vol - final_vol) / baseline_vol >= 0.657
        )
        categories[pr_mask] = "PR"

        pd_mask = (
            ~cr_mask
            & ~pr_mask
            & (nadir_vol > 0)
            & ((final_vol - nadir_vol) / nadir_vol >= 0.728)
        )
        categories[pd_mask] = "PD"

    return categories


def compute_orr(recist_categories: np.ndarray) -> float:
    """Compute ORR (objective response rate) from RECIST categories.

    ORR = fraction of patients with CR or PR.
    """
    responders = np.isin(recist_categories, ["CR", "PR"])
    return float(np.mean(responders))


# =============================================================================
# MUTUAL INFORMATION ESTIMATION
# =============================================================================


def mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Estimate MI using KSG estimator (Kraskov et al. 2004, Algorithm 1).

    For two continuous variables. Uses Chebyshev (max-norm) distance.

    Args:
        x: (n_samples,) or (n_samples, d_x)
        y: (n_samples,) or (n_samples, d_y)
        k: Number of nearest neighbors

    Returns:
        Estimated MI in nats (non-negative)
    """
    n = len(x)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]

    xy = np.hstack([x, y])

    tree_xy = KDTree(xy, metric="chebyshev")
    dists, _ = tree_xy.query(xy, k=k + 1)
    eps = dists[:, -1]

    tree_x = KDTree(x, metric="chebyshev")
    tree_y = KDTree(y, metric="chebyshev")

    n_x = np.array(
        [
            tree_x.query_radius(x[i : i + 1], r=eps[i] - 1e-15, count_only=True)[0]
            - 1
            for i in range(n)
        ]
    )
    n_y = np.array(
        [
            tree_y.query_radius(y[i : i + 1], r=eps[i] - 1e-15, count_only=True)[0]
            - 1
            for i in range(n)
        ]
    )

    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return max(mi, 0.0)


def mi_continuous_binary(x: np.ndarray, y_binary: np.ndarray) -> float:
    """Estimate MI between a continuous variable and a binary label.

    Uses binned conditional entropy: MI = H(Y) - H(Y|X), where X is
    discretized into equal-frequency bins.

    Args:
        x: (n_samples,) continuous variable
        y_binary: (n_samples,) binary labels (bool or 0/1)

    Returns:
        Estimated MI in nats (non-negative)
    """
    n = len(x)
    y = np.asarray(y_binary, dtype=bool)

    p1 = np.mean(y)
    if p1 == 0 or p1 == 1:
        return 0.0
    h_y = -p1 * np.log(p1) - (1 - p1) * np.log(1 - p1)

    n_bins = max(5, min(50, int(np.sqrt(n))))
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x, percentiles)
    bin_edges[0] -= 1e-10
    bin_edges[-1] += 1e-10
    bin_indices = np.digitize(x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    h_y_given_x = 0.0
    for b in range(n_bins):
        mask = bin_indices == b
        n_b = np.sum(mask)
        if n_b == 0:
            continue
        p1_b = np.mean(y[mask])
        if p1_b == 0 or p1_b == 1:
            continue
        h_b = -p1_b * np.log(p1_b) - (1 - p1_b) * np.log(1 - p1_b)
        h_y_given_x += (n_b / n) * h_b

    return max(h_y - h_y_given_x, 0.0)


def mi_sweep_binary(
    features: np.ndarray,
    response: np.ndarray,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compute MI between each feature column and a binary response.

    Args:
        features: (n_samples, n_features) continuous feature matrix
        response: (n_samples,) binary labels
        feature_names: Optional names for each feature column

    Returns:
        DataFrame with columns ['feature', 'mi'] sorted by MI descending
    """
    n_features = features.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    mi_values = []
    for j in range(n_features):
        col = features[:, j]
        if np.all(np.isnan(col)) or np.nanstd(col) < 1e-15:
            mi_values.append(0.0)
        else:
            valid = ~np.isnan(col)
            if valid.sum() < 20:
                mi_values.append(0.0)
            else:
                mi_values.append(mi_continuous_binary(col[valid], response[valid]))

    df = pd.DataFrame({"feature": feature_names, "mi": mi_values})
    return df.sort_values("mi", ascending=False).reset_index(drop=True)


def mi_sweep_continuous(
    features: np.ndarray,
    outcome: np.ndarray,
    feature_names: list[str] | None = None,
    k: int = 5,
) -> pd.DataFrame:
    """Compute MI between each feature column and a continuous outcome.

    Uses the KSG estimator (continuous-continuous MI) for each column.

    Args:
        features: (n_samples, n_features) continuous feature matrix
        outcome: (n_samples,) continuous outcome variable
        feature_names: Optional names for each feature column
        k: Number of neighbors for KSG estimator

    Returns:
        DataFrame with columns ['feature', 'mi'] sorted by MI descending
    """
    n_features = features.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    import time as _time

    mi_values = []
    t0 = _time.time()
    log_interval = max(1, n_features // 20)  # ~20 progress updates
    for j in range(n_features):
        col = features[:, j]
        valid = ~np.isnan(col) & ~np.isnan(outcome)
        if valid.sum() < max(20, k + 1) or np.nanstd(col[valid]) < 1e-15:
            mi_values.append(0.0)
        else:
            mi_val = mi_ksg(col[valid], outcome[valid], k=k)
            mi_values.append(max(0.0, mi_val))  # KSG can be slightly negative

        if (j + 1) % log_interval == 0 or j == n_features - 1:
            elapsed = _time.time() - t0
            rate = (j + 1) / elapsed if elapsed > 0 else 0
            eta = (n_features - j - 1) / rate if rate > 0 else 0
            print(
                f"    MI sweep: {j + 1}/{n_features} "
                f"({100 * (j + 1) / n_features:.0f}%) "
                f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                flush=True,
            )

    df = pd.DataFrame({"feature": feature_names, "mi": mi_values})
    return df.sort_values("mi", ascending=False).reset_index(drop=True)


# =============================================================================
# LOO RETRAINING HARNESS
# =============================================================================


def loo_retrain_posterior_width(
    theta_train,
    x_train,
    obs_transformed: np.ndarray,
    prior_normal,
    observable_names: list[str],
    param_names: list[str],
    npe_config: dict | None = None,
    n_posterior_samples: int = 1000,
    prcc_sensitivity: np.ndarray | None = None,
) -> pd.DataFrame:
    """LOO retraining: measure posterior width change per dropped observable.

    For each observable j:
    1. Remove column j from x_train and obs_transformed
    2. Retrain NPE
    3. Sample posterior, compute marginal variances
    4. Optionally weight by PRCC sensitivity to decision output

    This is the "cheap" OBED approach (no posterior predictive simulations).
    For the direct MSE decomposition, use with small posterior predictive runs.

    Args:
        theta_train: (n_train, n_params) parameter tensor (log-space)
        x_train: (n_train, n_obs) observable tensor (copula-transformed)
        obs_transformed: (n_obs,) observed data in copula space
        prior_normal: Prior distribution in normal space (sbi-compatible)
        observable_names: Names of observables
        param_names: Names of parameters
        npe_config: Dict with 'hidden_features', 'num_transforms', 'num_bins',
                     'batch_size', 'max_epochs'. Uses defaults if None.
        n_posterior_samples: Number of posterior samples for variance estimation
        prcc_sensitivity: Optional (n_params,) array of |PRCC| for decision output.
                         If provided, computes sensitivity-weighted variance reduction.

    Returns:
        DataFrame with per-observable, per-parameter variance decomposition
    """
    import torch
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    cfg = npe_config or {}
    hidden = cfg.get("hidden_features", 32)
    transforms = cfg.get("num_transforms", 3)
    bins = cfg.get("num_bins", 10)
    batch_size = cfg.get("batch_size", 128)
    max_epochs = cfg.get("max_epochs", 500)

    n_obs = x_train.shape[1]
    n_params = theta_train.shape[1]

    # --- Full model baseline ---
    de_full = posterior_nn(
        model="nsf",
        hidden_features=hidden,
        num_transforms=transforms,
        num_bins=bins,
    )
    inf_full = NPE(prior=prior_normal, density_estimator=de_full)
    inf_full = inf_full.append_simulations(theta_train, x_train)
    inf_full.train(
        training_batch_size=batch_size,
        max_num_epochs=max_epochs,
        show_train_summary=False,
    )
    posterior_full = inf_full.build_posterior()

    obs_tensor_full = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)
    samples_full = posterior_full.sample(
        (n_posterior_samples,), x=obs_tensor_full
    ).numpy()
    var_full = np.var(samples_full, axis=0)

    # --- LOO for each observable ---
    results = []
    for j in range(n_obs):
        obs_name = observable_names[j]
        print(f"  LOO [{j + 1}/{n_obs}]: dropping {obs_name}")

        keep = [i for i in range(n_obs) if i != j]
        x_loo = x_train[:, keep]
        obs_loo = obs_transformed[keep]

        de_loo = posterior_nn(
            model="nsf",
            hidden_features=hidden,
            num_transforms=transforms,
            num_bins=bins,
        )
        inf_loo = NPE(prior=prior_normal, density_estimator=de_loo)
        inf_loo = inf_loo.append_simulations(theta_train, x_loo)
        inf_loo.train(
            training_batch_size=batch_size,
            max_num_epochs=max_epochs,
            show_train_summary=False,
        )
        posterior_loo = inf_loo.build_posterior()

        obs_tensor_loo = torch.tensor(obs_loo, dtype=torch.float32).unsqueeze(0)
        samples_loo = posterior_loo.sample(
            (n_posterior_samples,), x=obs_tensor_loo
        ).numpy()
        var_loo = np.var(samples_loo, axis=0)

        for k in range(n_params):
            row = {
                "observable": obs_name,
                "param": param_names[k],
                "var_full": var_full[k],
                "var_loo": var_loo[k],
                "delta_var": var_loo[k] - var_full[k],
            }
            if prcc_sensitivity is not None:
                row["prcc_weight"] = abs(prcc_sensitivity[k])
                row["weighted_delta_var"] = row["delta_var"] * abs(prcc_sensitivity[k])
            results.append(row)

    return pd.DataFrame(results)


def summarize_loo_by_observable(
    loo_df: pd.DataFrame,
    weighted: bool = True,
) -> pd.DataFrame:
    """Summarize LOO results: total variance reduction per observable.

    Args:
        loo_df: Output of loo_retrain_posterior_width
        weighted: If True and 'weighted_delta_var' column exists, use it

    Returns:
        DataFrame sorted by total variance reduction (most valuable first)
    """
    col = (
        "weighted_delta_var"
        if (weighted and "weighted_delta_var" in loo_df.columns)
        else "delta_var"
    )

    summary = (
        loo_df.groupby("observable")
        .agg(
            total_delta_var=(col, "sum"),
            max_delta_var=(col, "max"),
            n_params_helped=("delta_var", lambda s: (s > 0).sum()),
        )
        .sort_values("total_delta_var", ascending=False)
        .reset_index()
    )

    return summary


# =============================================================================
# PRIOR TIGHTENING ANALYSIS
# =============================================================================


def generate_tightened_theta_sets(
    posterior_samples: np.ndarray,
    param_names: list[str],
    candidate_params: list[str],
    tightening_factor: float = 0.5,
    method: str = "clamp",
) -> dict[str, np.ndarray]:
    """Generate modified posterior sample sets with individual parameters tightened.

    For each candidate parameter, creates a copy of the posterior samples where
    that parameter's spread is reduced (simulating the effect of having a tighter
    prior or better calibration data for that parameter). These can then be run
    through forward simulations to measure how much ORR variance decreases.

    Args:
        posterior_samples: Array of shape (n_samples, n_params) from NPE posterior
        param_names: List of parameter names matching columns of posterior_samples
        candidate_params: Subset of param_names to test tightening
        tightening_factor: How much to reduce spread. 0.5 = halve the range
            around the median. Smaller = tighter constraint.
        method: How to tighten. Options:
            "clamp": Clamp values to median ± factor * (original spread).
                Preserves sample count but distorts tail correlations.
            "filter": Keep only samples within the tightened range.
                Preserves correlations but reduces sample count.

    Returns:
        Dict mapping parameter name → modified theta array.
        "baseline" key contains the unmodified posterior samples.
        Each array has shape (n_samples, n_params) for "clamp",
        or (n_kept, n_params) for "filter" (n_kept varies per parameter).
    """
    if method not in ("clamp", "filter"):
        raise ValueError(f"method must be 'clamp' or 'filter', got '{method}'")

    param_index = {name: i for i, name in enumerate(param_names)}
    for p in candidate_params:
        if p not in param_index:
            raise ValueError(f"Parameter '{p}' not in param_names")

    result = {"baseline": posterior_samples.copy()}

    for p in candidate_params:
        k = param_index[p]
        col = posterior_samples[:, k]
        median = np.median(col)
        half_range = tightening_factor * (np.percentile(col, 84) - np.percentile(col, 16)) / 2

        if method == "clamp":
            modified = posterior_samples.copy()
            modified[:, k] = np.clip(col, median - half_range, median + half_range)
            result[p] = modified
        else:  # filter
            mask = (col >= median - half_range) & (col <= median + half_range)
            result[p] = posterior_samples[mask]

    return result
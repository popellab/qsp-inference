
#!/usr/bin/env python3
"""Per-observable processing behind amortized inference on QSP summaries.

Two things live here, one for each half of the measurement model:

- :func:`add_observation_noise` convolves the simulator's clean predictive with
  the observed measurement noise (empirical bootstrap residuals by default, a
  CI-derived parametric fallback otherwise), so the NPE learns that data are
  noisy.
- :class:`ScenarioTransform` / :func:`fit_scenario_transform` fit one scenario's
  Gaussian-copula quantiles on the *noisy training* observables and carry every
  other array the NPE must compare against (held-out test, the observed vector,
  later-round or posterior-predictive draws) into that same latent space.

Plus two array/index utilities the multi-scenario assembly needs around those:
:func:`joint_finite_mask` (a draw is usable only if it produced finite
observables under *every* scenario) and :func:`train_test_split_indices` (a
reproducible split with no global-RNG side effect). Cohort random effects and
cross-scenario alignment stay with the caller: they are modelling choices, not
generic data plumbing.

Usage:
    from qsp_inference.inference import fit_scenario_transform

    x_train_t, xf = fit_scenario_transform(
        x_train_raw,
        ci95_lower=lo, ci95_upper=hi, medians=med,
        bootstrap_samples=boots, sample_sizes=n, noise_seed=7,
    )
    x_test_t = xf.transform(x_test_raw)      # same quantiles, no leakage
    obs_t = xf.transform_vector(obs_values)  # observed vector into the same space
"""

import warnings
from dataclasses import dataclass

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

try:
    import torch
except ImportError:  # torch is only needed by the tensor-producing NPE helpers;
    torch = None     # add_observation_noise et al. are pure numpy
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
    *,
    bootstrap_samples: "list | None",
    seed: int = 42,
    sample_sizes: np.ndarray | None = None,
) -> np.ndarray:
    """
    Add observation noise to simulator outputs based on calibration target uncertainty.

    Two noise models are supported per observable:

    1. **Empirical** (preferred, used when ``bootstrap_samples[i]`` is available):
       convolve the predictive with the observed bootstrap's OWN residual shape.
       The bootstrap (the sampling distribution of the observed statistic produced
       by the calibration target's ``distribution_code``) is the real measurement
       noise — including skew, tails and bounds — so no parametric family or
       CI-derived sigma is assumed. Multiplicative (relative) residuals
       ``boot_j / median(boot)`` are used when the predictive column and the
       bootstrap are strictly positive (natural for densities/concentrations and
       respects positivity); additive residuals ``boot_j - median(boot)``
       otherwise. The bootstrap already represents the *statistic's* sampling
       uncertainty, so ``sample_sizes`` (the sqrt(n) SEM rescale) is intentionally
       NOT applied to empirical observables.

    2. **Parametric** (fallback, when no bootstrap is provided for an observable):
       determine whether the CI95 is better described by lognormal or Gaussian
       noise (by CI asymmetry around the median) and draw accordingly. NOTE: the
       parametric refit only reproduces the observed CI when the median equals the
       geometric mean of (lo, hi) — i.e. when the bootstrap is (log-)symmetric —
       so it mis-scales skewed bootstraps. Prefer the empirical path when samples
       are available.

    Either way the NPE learns that observations are noisy measurements, so targets
    with wider uncertainty constrain the posterior less.

    Lognormal noise (multiplicative): x_noisy = x * exp(N(0, log_sigma))
    Gaussian noise (additive): x_noisy = x + N(0, sigma)

    When ``sample_sizes`` are provided (parametric path only), the noise scale is
    divided by sqrt(n) to reflect the standard error of the population summary
    statistic (median) rather than the full population spread.

    Args:
        x: Simulator outputs, shape (n_samples, n_observables).
        ci95_lower: Lower CI95 bound per observable, shape (n_observables,).
        ci95_upper: Upper CI95 bound per observable, shape (n_observables,).
        medians: Observed medians per observable, shape (n_observables,).
        seed: Random seed for reproducibility.
        sample_sizes: Sample size per observable, shape (n_observables,).
            When provided, the parametric noise scale is divided by sqrt(n) to use
            standard error instead of population spread. Ignored for observables
            handled by the empirical path.
        bootstrap_samples: REQUIRED (keyword-only). List (length n_observables) of
            1-D arrays of observed bootstrap samples per observable, or None per
            entry. When a usable array (>= 5 finite samples) is given for
            observable i, the empirical path is used for that observable; otherwise
            it falls back to the parametric path. Empirical noise is the expected
            default, so this argument is mandatory — pass the bootstrap list (with
            None entries where a target has no samples) for the standard path, or
            pass ``bootstrap_samples=None`` to deliberately run fully parametric
            (legacy behavior; emits a warning).

    Returns:
        x_noisy: Noisy simulator outputs, same shape as x.
    """
    n_obs = x.shape[1]
    rng = np.random.default_rng(seed)
    x_noisy = x.copy()

    # Empirical bootstrap-residual noise is the default/expected path: it carries
    # the observed noise's real skew/tails/bounds. The parametric CI refit only
    # reproduces the observation when the bootstrap is log-symmetric, so running
    # fully parametric (no bootstrap_samples at all) is a fallback worth flagging.
    if bootstrap_samples is None:
        warnings.warn(
            "add_observation_noise: no bootstrap_samples provided — falling back to "
            "the parametric lognormal/Gaussian CI refit for ALL observables. Empirical "
            "bootstrap-residual noise is the default/expected path; pass per-observable "
            "bootstrap samples (from each target's distribution_code) where available.",
            UserWarning,
            stacklevel=2,
        )

    n_emp = 0
    n_logn = 0
    n_gauss = 0
    n_skip = 0

    for i in range(n_obs):
        col = x[:, i]

        # --- Empirical path: convolve with the observed bootstrap's residual shape.
        if bootstrap_samples is not None and bootstrap_samples[i] is not None:
            boot = np.asarray(bootstrap_samples[i], dtype=float)
            boot = boot[np.isfinite(boot)]
            if boot.size >= 5:
                med_b = float(np.median(boot))
                idx = rng.integers(0, boot.size, size=x.shape[0])
                if med_b > 0 and np.all(boot > 0) and np.all(col > 0):
                    x_noisy[:, i] = col * (boot[idx] / med_b)   # multiplicative
                else:
                    x_noisy[:, i] = col + (boot[idx] - med_b)   # additive
                n_emp += 1
                continue
            # too few bootstrap samples -> fall through to the parametric path

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
    emp_str = f"{n_emp} empirical, " if bootstrap_samples is not None else ""
    print(f"  Observation noise{sem_str}: {emp_str}{n_logn} lognormal, {n_gauss} Gaussian, "
          f"{n_skip} skipped (of {n_obs} observables)")

    return x_noisy


@dataclass(frozen=True)
class ScenarioTransform:
    """One scenario's Gaussian-copula quantiles plus the map into its latent space.

    Fit by :func:`fit_scenario_transform` on the *noisy training* observables,
    then reused to carry every other array the NPE must compare against into the
    same standard-normal space: the held-out test set, the observed vector, and
    any later-round or posterior-predictive draws. Sharing one set of quantiles
    is what makes those tensors comparable. Re-fitting per array would place each
    in a different marginal gauge, and fitting on anything but the training set
    would leak the held-out or observed data into the transform.
    """

    quantiles: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Carry a ``(n, d)`` batch of raw observables into the fitted space."""
        return transform_to_normal_from_array(np.asarray(x), self.quantiles)

    def transform_vector(self, x: np.ndarray) -> np.ndarray:
        """Carry a single ``(d,)`` raw observable vector into the fitted space."""
        return self.transform(np.asarray(x)[np.newaxis, :])[0]


def fit_scenario_transform(
    x_train_raw: np.ndarray,
    *,
    ci95_lower: np.ndarray,
    ci95_upper: np.ndarray,
    medians: np.ndarray,
    bootstrap_samples: Optional[List[Optional[np.ndarray]]],
    sample_sizes: Optional[np.ndarray] = None,
    noise_seed: int = 42,
    n_quantiles: int = 1000,
) -> Tuple[np.ndarray, ScenarioTransform]:
    """Fit one scenario's copula transform on its noisy training observables.

    The single per-scenario processing step behind amortized inference on QSP
    summaries: inject observation noise onto the training observables (so the NPE
    learns the data are noisy measurements), fit Gaussian-copula quantiles on the
    *noisy* training set, and return both the transformed training tensor and the
    :class:`ScenarioTransform` that carries any other array (test, observed,
    later rounds) into the same space.

    Quantiles are fit on the noisy training data alone. Fitting on the test or
    observed arrays would leak them into the transform; fitting on the noise-free
    training data would put the NPE's inputs in a slightly different gauge than
    the one the noise model just declared the data live in.

    Args:
        x_train_raw: ``(n_train, d)`` clean simulator observables for training.
        ci95_lower, ci95_upper, medians: per-observable ``(d,)`` calibration
            summaries, forwarded to :func:`add_observation_noise` for the
            parametric fallback.
        bootstrap_samples: per-observable observed bootstrap arrays (or ``None``
            entries), selecting the empirical noise path where available. Pass
            ``None`` to run fully parametric (emits a warning there).
        sample_sizes: per-observable ``(d,)`` sizes for the SEM rescale of the
            parametric path; ignored for empirical observables.
        noise_seed: seed for the noise draw.
        n_quantiles: resolution of the empirical copula CDF.

    Returns:
        ``(x_train_transformed, transform)``. ``transform.transform`` /
        ``transform.transform_vector`` map further arrays into the same space.
    """
    x_noisy = add_observation_noise(
        x_train_raw,
        ci95_lower,
        ci95_upper,
        medians,
        bootstrap_samples=bootstrap_samples,
        seed=noise_seed,
        sample_sizes=sample_sizes,
    )
    quantiles = compute_quantiles_from_array(x_noisy, n_quantiles=n_quantiles)
    transform = ScenarioTransform(quantiles=quantiles)
    return transform.transform(x_noisy), transform


def joint_finite_mask(x_arrays: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean mask of the rows finite across *every* array.

    Multi-scenario amortized inference evaluates the SAME parameter draws under
    each scenario, and a draw is usable only if it produced finite observables in
    all of them: one solver failure anywhere disqualifies that row everywhere, or
    the scenarios' observables would no longer come from a common set of draws.
    So the mask is the logical AND of the per-array all-finite-row masks.

    Args:
        x_arrays: one ``(n, d_s)`` observable array per scenario, all sharing the
            same row count ``n`` (row i is the same draw in each). Must be
            non-empty.

    Returns:
        ``(n,)`` boolean mask, ``True`` where the row is finite in every array.
    """
    arrays = list(x_arrays)
    if not arrays:
        raise ValueError("joint_finite_mask requires at least one array")
    n = arrays[0].shape[0]
    mask = np.ones(n, dtype=bool)
    for i, x in enumerate(arrays):
        x = np.asarray(x)
        if x.shape[0] != n:
            raise ValueError(
                f"row-count mismatch: array 0 has {n} rows, array {i} has "
                f"{x.shape[0]}; the arrays must be aligned draw-for-draw"
            )
        mask &= np.all(np.isfinite(x), axis=1)
    return mask


def train_test_split_indices(
    n: int,
    *,
    test_fraction: float = 0.1,
    seed: int,
    subsample: Optional[int] = None,
    subsample_seed_offset: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproducible ``(train_idx, test_idx)`` over ``n`` rows.

    The permutation is drawn from a *local* ``np.random.RandomState(seed)``, which
    yields indices identical to the legacy ``np.random.seed(seed);
    np.random.permutation(n)`` idiom while leaving the global NumPy RNG state
    untouched. The first ``(1 - test_fraction)`` fraction of the permutation is
    train, the remainder test.

    Args:
        n: number of rows to split.
        test_fraction: fraction assigned to the test set (floored to an int).
        seed: seed for the split permutation.
        subsample: if given and smaller than the train set, keep this many train
            rows, drawn without replacement from a ``default_rng(seed +
            subsample_seed_offset)``. The test set is never subsampled.
        subsample_seed_offset: offset added to ``seed`` for the subsample RNG, so
            the subsample is independent of the split permutation.

    Returns:
        ``(train_idx, test_idx)`` integer arrays.
    """
    perm = np.random.RandomState(seed).permutation(n)
    n_test = int(n * test_fraction)
    n_train = n - n_test
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if subsample is not None and subsample < n_train:
        sub_rng = np.random.default_rng(seed + subsample_seed_offset)
        keep = sub_rng.choice(n_train, size=subsample, replace=False)
        train_idx = train_idx[keep]
    return train_idx, test_idx


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

"""Stage 2 NPE pipeline: from Stage 1 copula prior to posterior + diagnostics.

A linear, single-scenario walk-through of the Stage 2 workflow. Real projects
add multi-scenario stacking, simulation caching, and shell-local calibration;
this script keeps just the spine so the qsp-inference pieces (prior loading,
copula transform, diagnostics, PPC, save_diagnostics) are easy to follow.

Simulator: ``qsp_hpc.simulation.QSPSimulator`` from qsp-hpc-tools
(https://github.com/popellab/qsp-hpc-tools), wired here through its
``simulate_with_parameters(theta)`` method. The C++ counterpart
``qsp_hpc.simulation.CppSimulator`` exposes the same interface.

Inputs (replace these with project paths):
    submodel_priors.yaml      from ``qsp-audit ... report``
    parameters/priors.csv     the CSV path the audit was run against
    QSPSimulator(...)         configured for the project's QSP model
    obs_values                observed test statistics, dict {name: value}

Outputs (under SAVE_DIR):
    posterior_samples.csv
    diagnostics CSVs (recovery, contraction, calibration, coverage, ...)
    figures (recovery, ECDF, PPC histograms, ...)

The CSV outputs match what the qsp-audit Stage 2 sections expect under
``AuditConfig.sbi_run_path``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from qsp_hpc.simulation import QSPSimulator
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from qsp_inference.inference import (
    add_observation_noise,
    compute_per_param_calibration,
    compute_quantiles_from_array,
    convert_posterior_samples_to_original_space,
    get_observed_data,
    sample_restricted,
    save_diagnostics,
    sbi_calibration_ecdf,
    sbi_loo_predictive_check,
    sbi_posterior_predictive_coverage,
    sbi_recovery,
    sbi_self_reference_null,
    sbi_z_score_contraction,
    train_restriction_classifier,
    transform_to_normal_from_array,
)
from qsp_inference.priors.copula_prior import load_composite_prior_log


# ---------------------------------------------------------------------------
# Configuration. In a real project these come from a RunConfig / argparse.
# ---------------------------------------------------------------------------
SUBMODEL_PRIORS = Path("submodel_priors.yaml")
PRIORS_CSV = Path("parameters/priors.csv")
SAVE_DIR = Path("runs/stage2_example")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Configure for the project's QSP model. See qsp-hpc-tools docs for the full
# QSPSimulator API (model_version, observable selection, pool config, etc.).
sim = QSPSimulator(...)  # noqa: F821 — placeholder, project-specific config

def qsp_simulator(theta: np.ndarray) -> np.ndarray:
    """Theta -> test statistics, shape (n, n_obs).

    ``QSPSimulator.simulate_with_parameters`` uses a theta-hashed suffix pool,
    so repeated calls with identical parameter matrices hit the cache instead
    of resimulating.
    """
    return sim.simulate_with_parameters(theta)

NUM_TRAIN = 5_000
NUM_TEST = 500
NUM_POSTERIOR_SAMPLES = 10_000
NOISE_FRACTION = 0.05
SEED = 2027

NPE_HIDDEN_FEATURES = 64
NPE_NUM_TRANSFORMS = 5
NPE_NUM_BINS = 8
NPE_TRAINING_BATCH_SIZE = 256
NPE_MAX_EPOCHS = 200


# ---------------------------------------------------------------------------
# 1. Stage 1 posterior -> Stage 2 prior.
# ---------------------------------------------------------------------------
# load_composite_prior_log returns a torch.distributions object over
# log(theta). Submodel-covered params come from the Gaussian copula; the rest
# fall through to the priors CSV (lognormal/normal/uniform/Beta).
prior_log, param_names = load_composite_prior_log(SUBMODEL_PRIORS, PRIORS_CSV)
n_params = len(param_names)
print(f"Loaded composite prior over {n_params} params")


# ---------------------------------------------------------------------------
# 2. (Optional) train a RestrictionClassifier from a pilot pool.
# ---------------------------------------------------------------------------
# Run a small pilot pool, mark sims that returned finite outputs as valid, and
# fit the classifier. P(valid | theta) then lets sample_restricted reject
# low-survival draws before the main training pool is generated.
pilot_theta = prior_log.sample((2_000,)).exp().numpy()
pilot_x = qsp_simulator(pilot_theta)
pilot_valid = np.all(np.isfinite(pilot_x), axis=1)

clf = train_restriction_classifier(
    theta=pilot_theta,
    valid_mask=pilot_valid,
    feature_order=param_names,
    cv_folds=5,
)
clf.save(SAVE_DIR / "restriction_classifier")
print(f"  classifier cv_auc={clf.cv_auc:.3f} baseline_survival={clf.baseline_survival:.3f}")
print(f"  threshold curve: {clf.threshold_curve}")


# ---------------------------------------------------------------------------
# 3. Generate the main training pool against the restricted prior.
# ---------------------------------------------------------------------------
def restricted_prior_sample(n: int) -> np.ndarray:
    return prior_log.sample((n,)).exp().numpy()


theta_train_orig, total_draws = sample_restricted(
    classifier=clf,
    prior_sample_fn=restricted_prior_sample,
    n_accepted=NUM_TRAIN,
    threshold=0.5,
    max_draws=NUM_TRAIN * 20,
)
theta_test_orig, _ = sample_restricted(
    classifier=clf,
    prior_sample_fn=restricted_prior_sample,
    n_accepted=NUM_TEST,
    threshold=0.5,
    max_draws=NUM_TEST * 20,
)
print(f"  drew {total_draws} prior samples to get {NUM_TRAIN} accepted training thetas")

x_train_raw = qsp_simulator(theta_train_orig)
x_test_raw = qsp_simulator(theta_test_orig)

# Filter joint NaNs (a sim that fails on any observable is excluded).
train_ok = np.all(np.isfinite(x_train_raw), axis=1)
test_ok = np.all(np.isfinite(x_test_raw), axis=1)
theta_train_orig, x_train_raw = theta_train_orig[train_ok], x_train_raw[train_ok]
theta_test_orig, x_test_raw = theta_test_orig[test_ok], x_test_raw[test_ok]

# Optional measurement-noise injection for ablations.
x_train_noisy = add_observation_noise(x_train_raw, fraction=NOISE_FRACTION, seed=SEED)


# ---------------------------------------------------------------------------
# 4. Gaussian copula transform on observables (compute once, reuse).
# ---------------------------------------------------------------------------
quantiles = compute_quantiles_from_array(x_train_noisy, n_quantiles=1000)
x_train_t = transform_to_normal_from_array(x_train_noisy, quantiles)
x_test_t = transform_to_normal_from_array(x_test_raw, quantiles)

obs_values = get_observed_data(SAVE_DIR / "observed_test_stats.csv")  # dict
observable_names = list(obs_values.keys())
obs_raw = np.array([obs_values[n] for n in observable_names], dtype=float)
obs_t = transform_to_normal_from_array(obs_raw[None, :], quantiles)[0]


# ---------------------------------------------------------------------------
# 5. Prior predictive sanity gate (fail loud here, before NPE).
# ---------------------------------------------------------------------------
loo_prior_df, _ = sbi_loo_predictive_check(x_train_raw, obs_raw, observable_names)
prior_pred_df, _ = sbi_posterior_predictive_coverage(
    x_train_raw, obs_raw, observable_names, log_scale=True
)
sr_prior, _ = sbi_self_reference_null(x_train_raw, obs_raw, observable_names)
print(f"  prior-predictive D² p-value = {sr_prior['p_value']:.4f}")


# ---------------------------------------------------------------------------
# 6. Train NPE.
# ---------------------------------------------------------------------------
theta_train_log = torch.from_numpy(np.log(theta_train_orig)).float()
theta_test_log = torch.from_numpy(np.log(theta_test_orig)).float()
x_train_tt = torch.from_numpy(x_train_t).float()
x_test_tt = torch.from_numpy(x_test_t).float()

torch.manual_seed(SEED)
density_estimator = posterior_nn(
    model="nsf",
    hidden_features=NPE_HIDDEN_FEATURES,
    num_transforms=NPE_NUM_TRANSFORMS,
    num_bins=NPE_NUM_BINS,
)
inference = NPE(prior=prior_log, density_estimator=density_estimator)
inference.append_simulations(theta_train_log, x_train_tt)
inference.train(
    training_batch_size=NPE_TRAINING_BATCH_SIZE,
    max_num_epochs=NPE_MAX_EPOCHS,
    show_train_summary=True,
)
posterior = inference.build_posterior()


# ---------------------------------------------------------------------------
# 7. Diagnostics. Each call returns numerical results + a figure; pass
#    plot=False when you only want the numbers (e.g. for save_diagnostics).
# ---------------------------------------------------------------------------
# Stack per-test-point posterior samples: shape (n_post, n_test, n_params).
samples_per_test = []
for x_t in x_test_tt:
    s = posterior.sample((NUM_POSTERIOR_SAMPLES,), x=x_t, show_progress_bars=False)
    samples_per_test.append(s.numpy())
samples = np.stack(samples_per_test, axis=1)

prior_var = torch.var(prior_log.sample((2_000,)), dim=0).numpy()

fig_recovery, _, r2_values = sbi_recovery(samples, theta_test_log.numpy(), param_names)
fig_recovery.savefig(SAVE_DIR / "recovery.png", dpi=150, bbox_inches="tight")

_, _, z_scores, contractions = sbi_z_score_contraction(
    samples, theta_test_log.numpy(), prior_log, param_names, prior_var=prior_var
)

fig_ecdf, _, ranks_global, ks_global = sbi_calibration_ecdf(
    samples, theta_test_log.numpy(), param_names
)
fig_ecdf.savefig(SAVE_DIR / "calibration_ecdf.png", dpi=150, bbox_inches="tight")

calib_df = compute_per_param_calibration(samples, theta_test_log.numpy(), z_scores=z_scores)


# ---------------------------------------------------------------------------
# 8. Posterior predictive on the actual observation.
# ---------------------------------------------------------------------------
posterior_log_obs = posterior.sample(
    (NUM_POSTERIOR_SAMPLES,), x=torch.from_numpy(obs_t).float(), show_progress_bars=False
)
post_samples = posterior_log_obs.numpy()
theta_posterior_orig = np.exp(post_samples)

x_posterior_pred = qsp_simulator(theta_posterior_orig)
ok = np.all(np.isfinite(x_posterior_pred), axis=1)
x_posterior_pred = x_posterior_pred[ok]

coverage_df, fig_coverage = sbi_posterior_predictive_coverage(
    x_posterior_pred, obs_raw, observable_names, log_scale=True
)
fig_coverage.savefig(SAVE_DIR / "posterior_predictive_coverage.png", dpi=150, bbox_inches="tight")

sr_post, _ = sbi_self_reference_null(x_posterior_pred, obs_raw, observable_names)
print(f"  posterior-predictive D² p-value = {sr_post['p_value']:.4f}")


# ---------------------------------------------------------------------------
# 9. Save artifacts in the layout the audit's Stage 2 sections read.
# ---------------------------------------------------------------------------
# Posterior samples in original (un-logged) parameter space.
posterior_samples_df = convert_posterior_samples_to_original_space(post_samples, param_names)
posterior_samples_df.to_csv(SAVE_DIR / "posterior_samples.csv", index=False)

# Diagnostics CSVs (recovery, contraction, calibration KS, coverage, ...)
save_diagnostics(
    save_dir=SAVE_DIR,
    param_names=param_names,
    observable_names=observable_names,
    r2_values=r2_values,
    z_scores=z_scores,
    contractions=contractions,
    ranks=ranks_global,
    ks_results=ks_global,
    coverage_df=coverage_df,
    prior_pred_df=prior_pred_df,
    loo_prior_df=loo_prior_df,
    self_ref_prior=sr_prior,
    self_ref_posterior=sr_post,
)

# `local_calibration.csv` (shell-local KS p-values + contraction) is what the
# audit's Stage 2 NPE shifts section reads. Produce it by repeating
# sbi_calibration_ecdf and sbi_z_score_contraction on a shell-local
# (param-tier) subset and saving the resulting metrics.
print(f"\nDone. See {SAVE_DIR}/")

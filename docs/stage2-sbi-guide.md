# Stage 2 SBI Guide

A guide to the Stage 2 simulation-based inference (SBI) tooling in qsp-inference. Stage 2 picks up where the [submodel inference guide](submodel-inference-guide.md) leaves off: the joint Stage 1 posterior (marginals + Gaussian copula) becomes the prior for a neural posterior estimator (NPE) trained on full QSP simulations and conditioned on aggregate observables.

This guide is for modelers who already have a Stage 1 `submodel_priors.yaml` and want to (a) run Stage 2 NPE, (b) diagnose it, and (c) feed the results back into the parameter audit.

## How Stage 2 fits in

```
Stage 1 (submodel inference)        Stage 2 (full-model NPE)
 - per-component MCMC                 - draw theta from copula prior
 - marginals + copula posterior  -->  - run full QSP sims
 - submodel_priors.yaml                - train NPE on (theta, x)
                                       - condition on observed x_obs
                                       - posterior + diagnostics
                                                  |
                                                  v
                                       audit report rereads
                                       Stage 2 outputs and emits
                                       posterior-shift + clinical
                                       predictive uncertainty sections
```

Stage 1 turns scattered literature into a tractable joint prior. Stage 2 turns that prior plus full-model behavior plus aggregate clinical/preclinical observables into the parameter posterior you actually use for predictions.

### Reference implementation

For an end-to-end Stage 2 pipeline that wires together everything in this guide — copula-prior loading, restriction classifier training, NPE training, the full diagnostics suite, PPC, and the CSV outputs the audit expects — see the **sbi runner in pdac-build**. It is the canonical consumer of qsp-inference's Stage 2 surface and is the shortest path to a working setup. Treat the rest of this document as the reference for the individual building blocks; the pdac-build sbi runner shows how they compose.

The trajectory plotting in that runner uses `qsp_inference.inference.evaluate_calibration_target_over_trajectory` to evaluate calibration-target observables over long-form trajectory frames produced upstream by `qsp_hpc.cpp.evolve_trajectory.assemble_evolve_trajectory_long`.

## Loading the Stage 1 posterior as a Stage 2 prior

The Stage 1 audit writes `submodel_priors.yaml` with marginals (lognormal/normal/uniform/Beta/gamma/invgamma) and a Gaussian copula correlation matrix. Two PyTorch-distribution loaders consume it:

```python
from qsp_inference.priors import load_copula_prior_log, load_composite_prior_log

# Just the parameters covered by the copula (log space)
prior, names = load_copula_prior_log("submodel_priors.yaml")

# Copula for submodel params + independent CSV priors for everything else
prior, names = load_composite_prior_log(
    "submodel_priors.yaml",
    "parameters/pdac_priors.csv",
)
samples = prior.sample((10_000,))   # log-space theta
log_p   = prior.log_prob(samples)
```

`load_composite_prior_log` is the usual choice when the full QSP model has parameters the submodels don't touch (e.g. PK params, ODE-only constants). Those parameters fall through to the CSV path, which now supports lognormal, normal, uniform, and Beta marginals (see the [submodel guide](submodel-inference-guide.md#prerequisites)).

Notes on numerical robustness:

- Sample correlation matrices that aren't strictly PSD are projected onto the PSD cone before Cholesky factorization, so corner cases from finite posterior samples don't crash the loader.
- YAML marginals that don't admit a clean log-space normal fit (e.g. heavy-tailed gamma) fall back to truncating sampled marginals to the log domain or to the CSV's lognormal row.
- A `comonotonic-coupling artifact` in composite priors (perfectly correlated marginals when the copula is degenerate) is detected and corrected.

## Prior restriction with RestrictionClassifier

QSP simulators frequently fail for some fraction of prior draws — solver blow-ups, non-physical regimes, infeasible parameter combinations. Submitting all prior draws to the simulator wastes HPC; submitting none of them is the prior. `RestrictionClassifier` is the middle path: train a classifier that predicts `P(valid | theta)` from a pilot pool, then reject low-probability draws before they hit the simulator.

This is the simplest form of the truncated sequential NPE idea (Deistler et al. 2022): sklearn boosted trees on log-theta, no integration with `sbi`'s prior abstractions, no proper `log_prob` on the restricted prior — just a scorer over candidate thetas. The trade-off is that it works with *any* prior whose samples you can produce as a numpy array, so composite copula priors and hierarchical priors all plug in identically.

### Training

```python
import numpy as np
from qsp_inference.inference import train_restriction_classifier

# theta: (n, p) in original (un-logged) scale, valid_mask: (n,) bool
theta      = np.load("pilot_theta.npy")
valid_mask = np.load("pilot_valid.npy")

clf = train_restriction_classifier(
    theta=theta,
    valid_mask=valid_mask,
    feature_order=param_names,    # column names matching theta
    input_transform="log",        # default; pass "identity" for already-log priors
    cv_folds=5,                   # 0 disables CV, faster
    default_threshold=0.5,
)
print(clf.cv_auc, clf.in_sample_auc, clf.baseline_survival)
print(clf.threshold_curve)        # (accept_frac, survival, yield_per_draw) per tau
clf.save("classifiers/v1/")
```

`threshold_curve` is the diagnostic that tells you what tau to pick: each row reports the fraction of draws accepted, the survival rate among accepted draws, and the product (`yield_per_draw`). Pick the tau that maximizes yield, or the one that hits a survival target you care about — survival of 0.9 at accept_frac of 0.4 means you need ~2.5x as many prior draws as final sims, but each accepted draw is very likely to run.

### Scoring

```python
from qsp_inference.inference import RestrictionClassifier

clf = RestrictionClassifier.load("classifiers/v1/")
keep = clf.accept(theta_new)             # boolean mask, default threshold
p    = clf.score(theta_new)              # P(valid | theta) per row
```

`theta_new` must be in the original scale and have its columns in `clf.feature_order`. The classifier applies the log transform internally.

### Surviving prior drift

A classifier trained on a pilot pool ages: parameters get retired, new ones are added, priors are tightened. Strict shape-checked `score`/`accept` reject any caller-side theta that doesn't match `feature_order` exactly. The projection helpers handle drift without retraining:

```python
score = clf.score_named(
    theta=theta_live,                       # caller-side theta
    theta_feature_names=live_param_names,   # caller-side column order
    fills={"IFNg_50": 1000.0},              # value for retired classifier features
)
keep = clf.accept_named(theta_live, live_param_names, fills={"IFNg_50": 1000.0})
```

`project()` (called internally by `*_named`) drops caller-side columns the classifier doesn't know, fills classifier-side features missing from the caller using the `fills` dict, and reorders columns to match `feature_order`. Use this when the live prior is "close enough" to the pilot prior that the classifier still has predictive value but isn't shape-compatible. Retrain the classifier when accept rates collapse or `score_named` calibration drifts.

### Rejection sampling against a prior

```python
from qsp_inference.inference import sample_restricted

def prior_sample(n):
    # Must return (n, p) in the original scale, columns in clf.feature_order
    return prior.sample((n,)).exp().cpu().numpy()

theta_acc, total_draws = sample_restricted(
    classifier=clf,
    prior_sample_fn=prior_sample,
    n_accepted=20_000,
    threshold=0.5,
    batch_size=10_000,
    max_draws=1_000_000,    # safety cap for low-survival regimes
)
```

`max_draws` is the safeguard against infinite loops when survival is very low — set it to something sane (e.g. 10x what `threshold_curve` predicts you'll need) so a misconfigured classifier fails fast instead of running forever.

### Persistence

`save(out_dir)` writes `classifier.pkl` (the sklearn model) and `metadata.json` (`feature_order`, `input_transform`, `default_threshold`, AUCs, threshold curve, training size, model class). Both round-trip through `RestrictionClassifier.load(out_dir)`. Ship these alongside the simulation jobs so remote workers can score locally.

## Data prep for NPE

The training pool is `(theta, x)` where `x` is the vector of observables. Two utilities handle the bookkeeping:

```python
from qsp_inference.inference import (
    processed_simulator,                # single-scenario
    processed_multi_scenario_simulator, # stacks scenarios into one feature vector
    get_observed_data,
    prepare_observed_data,
    add_observation_noise,
)
```

`processed_simulator` calls a user-supplied QSP simulator, splits into train/test, filters NaN rows from solver failures, and applies a Gaussian-copula transform that maps each observable to standard normal:

```python
theta_train, x_train, theta_test, x_test, obs_quantiles = processed_simulator(
    qsp_simulator=qsp_sim,
    num_simulations=5000,
    test_fraction=0.1,
    split_seed=2027,
)
```

The copula transform is precomputed once (`compute_quantiles_from_array`) and applied to every batch (`transform_to_normal_from_array`) so training doesn't pay the transform cost per epoch. When you eventually condition on the real observation, run it through `prepare_observed_data` with the same `obs_quantiles` so it lives in the same standard-normal space the network was trained on. `add_observation_noise` is a convenience for ablation runs that want to see how diagnostics degrade with measurement noise.

## Diagnostics

`qsp_inference.inference.diagnostics` exposes ~16 functions, each returning numerical results alongside figures so output can be saved to CSV/JSON without re-computation. Pick from them by what you're trying to learn:

| Question | Diagnostic | What it gives you |
|---|---|---|
| Did the network learn the parameters? | `sbi_recovery` | R² of posterior medians vs. true theta on the test set |
| Is the posterior tighter than the prior? Is the bias acceptable? | `sbi_z_score_contraction` | Per-param contraction `1 - var_post/var_prior` and z-score `(mean_post - theta_true)/std_post` |
| Per-parameter calibration in numerical form | `compute_per_param_calibration` | Mean |z|, contraction, KS pass/fail |
| Are credible intervals correctly calibrated? | `sbi_calibration_ecdf` | Rank-statistic ECDF (uniform if calibrated; deviation = miscalibration) |
| Is the observed x in the training x-distribution? | `sbi_coverage_check` | Per-observable z-score and in-range flag against `x_train_raw` |
| Per-observable prior-predictive surprise | `sbi_prior_predictive_pvalues` | Two-sided p-values per observable; small = obs is extreme under the prior |
| Joint prior-predictive misspecification | `sbi_self_reference_null` | Mahalanobis D² of obs vs. predictive samples, with self-reference null (no chi-square assumption) |
| Which observables drive misspecification? | `sbi_loo_predictive_check` | Per-observable Mahalanobis influence (joint D² when each is dropped) |
| Posterior bunching at prior bounds | `sbi_boundary_piling` | Fraction of posterior mass within `tail_fraction` of bounds, per param/side |
| Is x_obs reproducible from posterior? | `sbi_posterior_predictive_check`, `sbi_posterior_predictive_coverage` | PPC distribution per observable + 95% CI coverage of obs |
| Strong residual correlations? | `sbi_posterior_correlations` | Per-pair correlation matrix above a threshold |
| Train/network mismatch indicator | `sbi_mmd_misspecification` | MMD between observed and training x in the chosen space |
| Are we training-data-limited? | `sbi_learning_curve` | R² vs. training-set fraction, retrains at each fraction |
| Is the trained NPE seed-stable? | `sbi_seed_stability` | Posterior variance across seeds at fixed theta_test |
| Does dimensionality reduction help recovery? | `sbi_dimensionality_sweep` | Recovery R² as you keep top-N params by sensitivity |
| Persist diagnostics to disk | `save_diagnostics` | Writes CSVs/JSON next to figures so the audit can reread them |

`sbi_learning_curve`, `sbi_seed_stability`, and `sbi_dimensionality_sweep` all take a `train_and_sample_fn(theta, x, seed) -> samples` callable, so you wire them up once for a training recipe and they handle the retraining sweep.

For NPE-specific calibration (the framework qsp-inference targets via the `sbi` package), the workflow is usually:

1. `sbi_recovery` and `sbi_z_score_contraction` for a high-level "did it work" read.
2. `sbi_calibration_ecdf` + `compute_per_param_calibration` for per-param calibration.
3. `sbi_boundary_piling` to flag prior-bound issues.
4. The PPC suite (next section) to confirm posterior predictive consistency with the real data.
5. `sbi_learning_curve` and `sbi_seed_stability` only when an earlier diagnostic shows a problem.

## Posterior predictive checks

`qsp_inference.inference.posterior_predictive` separates "checks" from "simulations":

- `generate_prior_predictive_checks(...)` — runs the QSP simulator on prior draws, computes test statistics, returns `x_pred` for the prior-predictive distribution. Good for sanity-checking the model setup *before* training.
- `generate_posterior_predictive_checks(...)` — same, but draws from `p(theta | x_obs)`. The diagnostic question is "can the posterior reproduce the observed test statistics?".
- `generate_posterior_predictive_simulations(...)` — runs full QSP simulations rather than only test statistics, so you can plot trajectory distributions.

A simulation pool keyed on the observed-data hash is shared across PPC and full-sim functions, so running PPC first and full-sims later reuses the cached simulations. Plot helpers `plot_ppc_histograms` (per-observable distributions with the observed value) and `plot_posterior_predictive_spaghetti` (trajectory bundles) are the standard outputs.

PPC results from a clinical scenario can be saved as `posterior_predictive_clinical.csv`; the audit report's [Clinical predictive uncertainty section](submodel-inference-guide.md#what-the-report-tells-you) reads that file when present.

## Optimal Bayesian experimental design (OBED)

`qsp_inference.inference.obed` answers "what should we measure next to most reduce decision-relevant predictive uncertainty?". The relevant decision is usually a clinical endpoint — major pathologic response (MPR) or RECIST classification — derived from posterior-predictive tumor trajectories.

The pipeline:

1. **Classify responses** from posterior-predictive tumor-volume trajectories: `classify_mpr`, `classify_recist`, `compute_orr` (objective response rate).
2. **Estimate mutual information** between candidate observables and the response: `mi_ksg` (KSG estimator for continuous-continuous), `mi_continuous_binary`, `mi_sweep_binary`/`mi_sweep_continuous` over a feature matrix.
3. **LOO retraining** for observable-value decomposition: `loo_retrain_posterior_width`, `summarize_loo_by_observable` — retrain the NPE leaving each observable out, measure how posterior width grows. Observables whose removal balloons posterior width are the ones that were most informative.
4. **Tightened theta sets** (`generate_tightened_theta_sets`): generate counterfactual posteriors with subsets of priors tightened, to probe which prior would matter most if better-constrained.

The audit report's "Clinical predictive uncertainty" section is upstream of OBED proper: it identifies *which parameters* drive remaining clinical predictive uncertainty (via Spearman correlation between posterior samples and endpoints, restricted to PRCC-significant parameters when PRCC is available). The OBED utilities then ask *which observables* would shrink that uncertainty most.

## Plumbing Stage 2 outputs back into the audit

The audit reports two Stage 2 sections when `AuditConfig.sbi_run_path` (or `--sbi-run-path` on the CLI) is set:

- **Stage 2 NPE posterior shifts** — per-parameter contraction and shift relative to both the pre-stage-1 prior and the post-stage-1 prior, plus global SBC contraction and shell-local contraction from `local_calibration.csv`. KS p-values flag local miscalibration; mean |z| flags SBC bias.
- **Clinical predictive uncertainty** — read from `posterior_predictive_clinical.csv`, written by your Stage 2 PPC step.

The minimum file set the audit looks for under the Stage 2 run path:

| File | Source | Used for |
|---|---|---|
| `posterior_samples.csv` (or equivalent in the loader) | NPE posterior samples on the held-out test set | Stage 2 shifts, predictive correlations |
| `local_calibration.csv` | shell-local calibration KS p-values + contraction | Stage 2 shifts (local miscal flag) |
| `aggregate_parameter_ranking.csv` | PRCC + n_significant_prcc | Driver attribution restriction, audit priority weighting |
| `posterior_predictive_clinical.csv` | PPC of clinical endpoints | Clinical predictive uncertainty section |

`save_diagnostics` is the canonical writer for the diagnostic CSVs.

## Practical tips

**Run prior-predictive sanity first.** `sbi_coverage_check` and `sbi_prior_predictive_pvalues` are cheap and catch the most common failure: x_obs simply isn't in the support of what the model can produce. Fix the model or the priors before training NPE.

**Train a RestrictionClassifier early.** Pilot ~5–10k sims, fit the classifier, look at `threshold_curve`. If yield is acceptable at a useful threshold, use the restricted prior for the main training pool — you'll save HPC and the network won't waste capacity on regions that always fail.

**Keep the test set in original space.** Most diagnostics expect transformed `x_train` but `x_train_raw` for coverage / prior-predictive p-values. The two arrays should round-trip through `obs_quantiles` cleanly; if they don't, the copula transform was learned on different data than you're checking against.

**Wire diagnostics through `save_diagnostics`.** The audit's Stage 2 sections want CSVs, not in-memory results. Saving once at the end of each diagnostic pass makes the audit cheap to re-run.

**OBED is downstream of decent calibration.** If `sbi_calibration_ecdf` shows obvious miscalibration, OBED's MI estimates and LOO widths are unreliable. Fix calibration first (more sims, prior tightening, observable selection) before chasing experimental designs.

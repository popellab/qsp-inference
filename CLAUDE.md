# CLAUDE.md

Developer guide for Claude Code when working with this repository.

---

## Overview

**qsp-inference** provides Bayesian inference tools for quantitative systems pharmacology (QSP) models. It consolidates inference code from three sources:

- **Submodel MCMC** (from maple): Joint NumPyro NUTS inference across SubmodelTarget YAMLs
- **SBI/NPE diagnostics** (from qsp-sbi): Neural posterior estimation, copula transforms, posterior predictive checks
- **Parameter audit** (from pdac-build): Coverage reporting, priority scoring, DAG visualization

**Dependency graph:**
```
pdac-build → qsp-inference → maple (schemas only)
```

qsp-inference imports `SubmodelTarget`, `SourceRelevanceAssessment`, and other Pydantic models from `maple.core.calibration`. It does NOT import inference code from maple — all inference lives here.

## Read this first

**[`docs/model-draft.tex`](docs/model-draft.tex)** is the model spec, and the only one. Cite its `eq:` labels when referring to the model. In short: one generative model (informative prior → QSP simulator → measurement noise) and two inference targets on it.

- **Flat inference** — the single-unit posterior `p(θ | x_obs)`; fixed-effects, the "median patient".
- **Virtual population (VPop)** — the random-effects version: `θᵢ ~ F(θ | μ, ω)`, data are cohort (median, IQR), a virtual patient is a draw from the fitted `F`.

Two constraints drive most design decisions here, and are worth keeping in mind before proposing changes:

1. **The likelihood is unavailable** — the mean map is an ODE solve, and the *default* observation noise (`inference/data_processing.py:add_observation_noise`) is an empirical bootstrap resampler, not a parametric density. Noise enters as augmentation on training pairs; nothing evaluates a likelihood. Do not describe the observation model as Gaussian — the parametric lognormal/Gaussian path is a per-observable *fallback* for targets with no bootstrap.
2. **Only a handful of parameters are identified.** Along the rest the posterior equals the prior, so the prior carries the answer and must be built from external data (submodel targets, derived priors). Widening a prior is not a safe default — it over-disperses the population and masks mis-centering.

`docs/` holds `model-draft.tex` and its slides. The six-chapter guide set that used to sit beside it described a package layout that no longer exists and is deleted; do not reinstate it from git history as a reference.

## Installation

```bash
uv pip install -e .

# With submodel MCMC support (NumPyro/JAX)
uv pip install -e ".[submodel]"

# With SBI support (PyTorch/sbi)
uv pip install -e ".[sbi]"

# With audit visualization (graphviz)
uv pip install -e ".[audit]"
```

Requires `maple` to be installed (for schema imports).

## Package Structure

```
src/qsp_inference/
├── submodel/                    # Submodel-based Bayesian inference
│   ├── inference.py             # Joint NumPyro NUTS MCMC
│   ├── comparison.py            # Component-wise NPE, single vs joint comparison
│   ├── parameterizer.py         # Posterior → marginal fits + Gaussian copula
│   ├── prior.py                 # Translation sigma rubric, distribution fitting
│   ├── parameter_groups.py      # Hierarchical parameter groups + cascade cuts
│   ├── freshness.py             # Content fingerprints / stale-posterior detection
│   ├── ppc_audit.py             # Per-observable fit evidence out of the compare
│   │                            #   cache: the datum, the CSV prior pushed through
│   │                            #   the forward model, and the posterior. Reports,
│   │                            #   deliberately does not judge.
│   ├── refit_check.py           # Does an edit to a target improve its own fit?
│   │                            #   Fits with and without it over an identical
│   │                            #   isolated target set. See below.
│   └── utils.py                 # ODE/algebraic forward model evaluation
├── inference/                   # SBI diagnostics and data processing
│   ├── sbc.py                   # Weighted SBC — the end-to-end calibration gate.
│   │                            #   Ranks θ* ~ π inside importance-weighted draws,
│   │                            #   so it checks train-on-π̃-then-reweight-to-π.
│   │                            #   Distinct from diagnostics.sbi_calibration_ecdf,
│   │                            #   which ranks θ_test ~ π̃ in *unweighted* draws
│   │                            #   (an estimator check that can pass while the
│   │                            #   reported posterior is wrong).
│   ├── importance.py            # π/π̃ reweighting: log_importance_weights,
│   │                            #   reweight_to_prior, ESS, weighted_quantile
│   ├── diagnostics.py           # Recovery, calibration, coverage; misspecification:
│   │                            #   sbi_self_reference_null (Mahalanobis D² + self-ref
│   │                            #   null), sbi_loo_predictive_check (per-obs LOO influence)
│   ├── data_processing.py       # NaN filtering, add_observation_noise (empirical
│   │                            #   bootstrap default / parametric fallback), wrappers
│   ├── restriction.py           # RestrictionClassifier — implausible-θ rejection
│   ├── gaussian_copula_transform.py  # Quantile-based normalization
│   ├── plot_distributions.py    # Posterior visualization (marginals, pairs)
│   ├── posterior_predictive.py  # Prior/posterior predictive checks
│   ├── trajectory_eval.py       # Trajectory-level scoring
│   └── obed.py                  # Optimal Bayesian experimental design (+ LOO retraining)
├── vpop/                        # Population inference. THE LIVE FIT is fit.py +
│   │                            #   predict.py, driven by pdac-build's
│   │                            #   workflows/vpop_fit.py. Nothing on that path is
│   │                            #   re-exported from vpop/__init__ (it needs
│   │                            #   jax/numpyro); import the modules directly.
│   ├── fit.py                   # eq:pop to eq:post as a numpyro model.
│   │                            #   site_spec is the single statement of what the
│   │                            #   latent space is; the mass matrix and every
│   │                            #   diagnostic read it rather than re-deriving it
│   ├── predict.py               # tau_B(phi): phi -> patients -> species ->
│   │                            #   readouts -> measurement map -> rows. Carries
│   │                            #   apply_margins and logit_median_coords, both
│   │                            #   shared with the pool so the two cannot drift
│   ├── mass.py                  # the Laplace metric as NUTS's mass matrix, computed
│   │                            #   rather than adapted. Adapting a 543x543 metric
│   │                            #   from a few hundred warmup draws cannot work
│   ├── rows.py, statistics.py   # a row functional (median, IQR, sd, fraction) over
│   │                            #   the cloud, with the frozen bootstrap designs
│   ├── resampling.py            # block plans: which cohorts share a draw
│   ├── emulator.py              # the surrogate, and the guards that refuse one
│   │                            #   trained on a different parameter set
│   ├── covariance.py, design.py, assemble.py, readouts.py   # V, Z, L_R, readouts
│   ├── recovery.py              # MAP, and recovery against a known phi*
│   ├── identifiability.py       # conditioning, row budget, z cost, pivot offsets
│   ├── width.py                 # the omega-versus-discrepancy reports
│   ├── weighting.py             # fixed-cloud route: prevalence weighting (Allen 2016)
│   ├── diagnostics.py           # fixed-cloud route: joint-reachability scoring
│   └── eigenbasis.py, proposal.py  # prior-metric eigenbasis, hierarchical NPE
├── auxiliary/                   # Auxiliary-parameter discovery and priors
├── data/                        # Data aggregation
│   ├── test_stat_functions.py   # Test statistics from QSP outputs
│   ├── aggregate_test_statistics.py
│   ├── aggregate_quick_estimates.py
│   ├── assess_normality.py
│   └── combine_test_stats_chunks.py
├── priors/                      # Prior loading and transformation
│   ├── copula_prior.py          # GaussianCopulaPrior: sample/log_prob/subset,
│   │                            #   composite + overlay loaders, derived priors
│   ├── load_sbi_priors.py       # Load priors from CSV
│   ├── generate_sbi_priors.py   # Generate SBI-compatible priors
│   └── truncated_distributions.py  # PyTorch truncation wrapper
└── audit/                       # Parameter audit reporting
    ├── report.py                # Coverage audit with AuditConfig
    └── plots.py                 # DAG, marginals, PPC visualizations
```

## Key Modules

### `submodel.inference` — Joint MCMC

Builds a joint NumPyro model from SubmodelTarget YAMLs + priors CSV:
- Independent priors from CSV (non-grouped params)
- Hierarchical priors for grouped params (base + tau + deltas)
- Forward models: structured algebraic, exec'd code, analytical ODE, or diffrax ODE
- Likelihoods with translation sigma in observation noise
- NaN guard: solver failures → -inf log-prob → NUTS rejects

```python
from qsp_inference.submodel.prior import process_targets

result = process_targets(
    priors_csv=Path("pdac_priors.csv"),
    yaml_paths=[Path("target1.yaml"), Path("target2.yaml")],
    output_dir=Path("priors/"),
)
```

### `submodel.prior` — Translation Sigma

Computes per-target translation sigma from `SourceRelevanceAssessment` (8 axes, added in quadrature, floor of 0.15). Applied inside the likelihood so MCMC naturally upweights more relevant sources.

### `submodel.parameter_groups` — Hierarchical Groups

Declares groups of related parameters that share a latent base rate:
`k_base ~ LogNormal(mu, sigma)`, `tau ~ HalfNormal(sigma_tau)`, `k_i = k_base * exp(delta_i)`.
Partial pooling: members with data get pulled by observations; members without data shrink toward the group mean.

Also manages cascade cuts for staged inference DAGs (upstream components' posteriors become downstream priors).

### `submodel.ppc_audit` + `submodel.refit_check` — Auditing a target's fit

Two halves of "is this submodel target actually right?". Used to find targets
whose posterior predictive misses its own observables, which usually means a
unit error, a forward model whose asymptotes are assigned to the wrong ends of
the curve, or a badly mis-centred CSV prior.

`ppc_audit` reads the `.compare_cache` and reports, per observable, the datum
with its own interval, the CSV prior pushed through the forward model, and the
posterior. Two derived columns carry most of the signal: `sens` (prior
predictive width over CSV prior width, both in decades) near zero means the
fitted parameters do not move that observable at all, so whatever it sits at is
asserted by the forward model rather than fitted; `z` places the datum against
the prior predictive spread. It attaches no verdict on purpose. Thresholds for
"badly fitting" did not survive testing, so the evidence is what gets reported
and the refit is what decides.

```python
from qsp_inference.submodel.ppc_audit import load_components, rank_by_miss, format_component

comps = load_components(cache_dir, priors_csv)
worst = rank_by_miss([c for c in comps if c.coverage < 1.0])
print(format_component(worst[0]))
```

`refit_check` decides whether a proposed edit helps, by fitting the target set
with and without it and comparing. Both arms run identical code over an
identical target set, so the edit is the only difference.

```python
from qsp_inference.submodel.refit_check import compare_edit

result = compare_edit(
    target_dir=submodel_dir,
    filenames=["IL1_50_IL6_PDAC_deriv001.yaml"],
    edits={"IL1_50_IL6_PDAC_deriv001.yaml": candidate_path},
    priors_csv=priors_csv,
    config_path=submodel_config,
    params={"IL1_50", "n_IL1"},
)
result.improved, result.before.coverage, result.after.coverage
```

**Gotcha this encodes, do not re-derive it:** `_build_stage_dag` walks the
*entire* cascade cut list and raises on any upstream target it cannot place,
whether or not that cut's parameter is in the run. So an isolated target
directory must carry every cut's upstream, not just the ones its own parameters
trigger. `resolve_target_set` does that closure, transitively.

Caveats when reading a comparison: the per-component RNG seed is derived from a
hash of component content, so editing a target changes the trajectory and small
before/after movements are partly seed noise. `improved` also accepts a coverage
tie with a smaller worst miss, so improved is not the same as fixed.

Two callers in pdac-build, both project-side; the prompt and paths live there
and the measurement lives here:

- `scripts/staged_extraction.py` stage 3d calls `check_targets` on every newly
  promoted submodel target. Schema and snippet validation say a target is well
  formed and faithful to its paper, neither says the forward model reproduces
  the numbers it carries. Advisory, nothing is un-promoted on a miss.
- `scripts/repair_submodel_targets.py` runs an LLM review over badly fitting
  components and gates each proposed edit on `compare_edit`.

### `audit.report` — Parameter Audit

Project-agnostic audit engine. Configure via `AuditConfig`:

```python
from qsp_inference.audit.report import AuditConfig, run_audit

config = AuditConfig(project_root=Path("/path/to/pdac-build"))
report = run_audit(config, output=Path("audit_report.md"))
```

Iteration script (re-MCMC components touched by an edited parameter, skip PPC + report):
```bash
python examples/regen_submodel_priors.py --project-root /path/to/project --invalidate k_CD8_kill
```

## Testing

```bash
pytest                           # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests (requires NumPyro)
```

## Development

### Import Conventions

- **maple schema imports** (OK): `from maple.core.calibration.submodel_target import SubmodelTarget`
- **Internal imports**: `from qsp_inference.submodel.inference import run_joint_inference`
- **Never** import inference code from maple — it's all here now

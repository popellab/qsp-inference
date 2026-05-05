# Submodel Inference Guide

A practical guide to calibrating QSP model parameters using the submodel inference pipeline. This guide is written for modelers familiar with typical QSP calibration workflows (e.g., SimBiology parameter estimation, Latin hypercube sampling) who want to move to a principled Bayesian approach.

## Why submodel inference?

### The posterior is the goal

The central object of Bayesian inference is the **posterior distribution** — the probability distribution over parameter values given all available data. The posterior tells you not just *what* parameter values are plausible, but *how confident* you should be in them, and *how they relate to each other*.

When you have a posterior, everything else follows naturally:

- **Point estimates** are just summaries of the posterior (median, MAP)
- **Uncertainty quantification** comes from the posterior's spread — no need to choose arbitrary ranges
- **Sensitivity analysis** can weight parameter samples by their posterior probability
- **Virtual populations** can be drawn directly from the joint posterior
- **Model predictions** carry proper uncertainty by propagating the posterior forward

There's nothing wrong with fitting some parameters to time-course data, setting others from literature values, or using different estimation methods for different parameters — all of that is fine in a Bayesian setting. The problem with ad hoc calibration is when the probability model is never made explicit: what distributions are you assuming for each parameter? How does each data source connect to which parameters? What uncertainty propagates from where? Without a clear probabilistic structure (priors, likelihoods, and the DAG connecting parameters to data), the "calibrated" parameter set has no coherent interpretation as a posterior, and uncertainty estimates become artifacts of arbitrary choices rather than consequences of the data.

### How it works in practice

The submodel inference pipeline makes the probabilistic structure explicit: each parameter has a prior, each data source connects to parameters through a likelihood (the forward model + error model), and the DAG connecting them is defined by the SubmodelTarget YAMLs.

Once this structure is defined, the pipeline automatically identifies which parameters share data (connected components in the DAG) and runs Bayesian inference on each component. In practice, most QSP parameter DAGs are sparse — a T cell trafficking rate and a drug clearance rate don't share any data — so the components are small (typically 1-10 parameters each), making exact MCMC tractable without approximation.

The per-component posteriors are then recombined into a joint posterior using marginal distributions + a Gaussian copula to capture within-component correlations. Parameters in different components are independent by construction (they share no data), so the copula only needs to represent correlations within components.

### What submodel inference gives you that ad hoc calibration doesn't

- **Principled priors**: Every parameter starts with an explicit prior distribution (from the priors CSV). The posterior updates this prior based on data — parameters with informative data get tight posteriors, parameters without data keep their priors.
- **Data weighting by relevance**: Not all literature values are equally trustworthy. Translation sigma quantifies how well each data source translates to your model (species match, indication match, measurement quality) and inflates the likelihood accordingly.
- **Correlations**: Parameters estimated from the same data source are naturally correlated in the posterior. The copula preserves this, so downstream sampling reflects joint uncertainty rather than treating parameters as independent.
- **Diagnostics**: Every inference result comes with contraction (did the data help?), convergence diagnostics (did the sampler work?), and posterior predictive checks (can the posterior reproduce the data?).

## The workflow at a glance

```
Literature data          SubmodelTarget YAMLs         Prior CSV
     |                         |                         |
     v                         v                         v
  maple extraction    -->  Calibration spec   +    Parameter priors
                                  |
                                  v
                        Component-wise inference
                       (NUTS MCMC or NPE per component)
                                  |
                                  v
                          Audit report (markdown)
                        + submodel_priors.yaml
                        + diagnostic figures
```

1. **Extract data** from the literature into SubmodelTarget YAMLs using maple
2. **Run the audit** (`qsp_inference.audit.report.run_audit`) to perform inference and generate a report
3. **Review the report** — check for conflicts, weak targets, and MCMC diagnostics
4. **Iterate** — add more targets, fix issues, re-run

## Key concepts

### What is a submodel target?

A submodel target is a YAML file that connects a piece of literature data to a small mathematical model involving a few QSP parameters. It has three main parts:

**Data** (`inputs`): The values you extracted from the paper — measurements, unit conversions, reference values. Each input has full provenance (source, location in paper, exact text snippet).

**Forward model** (`calibration.forward_model`): A simple mathematical relationship that predicts what the data *should* look like given parameter values. This is a "submodel" — a small piece of your full QSP model that you can evaluate cheaply.

**Error model** (`calibration.error_model`): How to compare the forward model prediction to the data, accounting for measurement uncertainty. This uses parametric bootstrapping from the paper's reported statistics.

### Forward model types

You don't need to write ODE solvers. Most submodel targets use one of the built-in structured types:

| Type | What it captures | Example |
|------|-----------------|---------|
| `steady_state_density` | Trafficking rate → cell density | "There are 50 CD8+ T cells per mm^2 in PDAC tumors" |
| `steady_state_concentration` | Secretion → cytokine level | "IL-6 concentration in PDAC is 15 pg/mL" |
| `steady_state_ratio` | Relative rates → population ratio | "M2:M1 macrophage ratio is 3:1" |
| `steady_state_fraction` | Rate balance → cell fraction | "30% of T cells are CD4+" |
| `steady_state_proliferation_index` | Proliferation rate → Ki-67+ fraction | "Ki-67+ fraction of tumor cells is 40%" |
| `algebraic` | Custom formula | Half-life to rate conversion, EC50 extraction |
| `first_order_decay` | Exponential decay ODE | Drug clearance time course |
| `custom_ode` | Arbitrary ODE system | Complex multi-compartment dynamics |

In practice, most targets end up using `algebraic` (custom formula relating parameters to an observable) or `custom_ode` (when dynamics matter). The structured steady-state types are convenient shortcuts when the relationship fits one of the standard patterns, but `algebraic` with a short `compute()` function handles anything.

### Translation sigma: weighting data by relevance

Not all data is equally informative. A measurement of T cell density in human PDAC tumors is more relevant to your PDAC model than mouse data from a different cancer type. Translation sigma captures this.

Each data source gets a relevance assessment with axes like:
- **Indication match**: Is this data from PDAC (exact), another GI cancer (related), or melanoma (proxy)?
- **Species**: Human data vs. mouse?
- **Measurement directness**: Did they measure the parameter directly, or is it inferred from other observables?
- **Source quality**: Clinical trial vs. cell line experiment vs. review article?

These scores combine (added in quadrature) into a single `translation_sigma` that inflates the likelihood variance. High translation sigma means "this data is informative but noisy" — the inference will still learn from it, but won't overfit to it.

You fill in the relevance assessment when creating the SubmodelTarget YAML in maple. The inference pipeline handles the math.

### Nuisance parameters

Some submodels need parameters that are part of the *experimental* description but not of the full QSP model — a proliferation rate that shapes a time course, a convertible-subpopulation fraction that bounds an asymptote, an APC contact area that converts pMHC counts to densities. Mark these with `nuisance: true` on the parameter and attach an `InlinePrior` directly (they are not in the priors CSV because the full model doesn't know about them):

```yaml
- name: F_inf
  units: dimensionless
  nuisance: true
  prior:
    distribution: uniform
    lower: 0.5
    upper: 1.0
```

Nuisance parameters are sampled alongside the QSP parameters during MCMC but are excluded from the output marginals and copula — they do their job in stage 1 and then disappear, leaving only the QSP parameter posteriors for downstream SBI. `InlinePrior` supports `lognormal` (use `mu`/`sigma`), `normal` (`mu`/`sigma`), and `uniform` (`lower`/`upper`). Pick whichever matches the physical constraint — bounded fractions get `uniform`, log-normally distributed rates get `lognormal`, approximately symmetric measurements get `normal`.

### Parameter groups

Some parameters are biologically related — for example, all the killing rates of different immune cell types. Parameter groups declare this relationship:

```
k_base ~ LogNormal(mu, sigma)      # shared group mean
k_CD8_kill = k_base * exp(delta_1)  # member-specific deviation
k_NK_kill  = k_base * exp(delta_2)
```

This is partial pooling: members with data get pulled by their observations, members without data shrink toward the group mean. It's a principled way to estimate parameters you don't have direct data for, based on related parameters you do.

### Cascade cuts

Sometimes you want staged inference: estimate upstream parameters first, then use those posteriors as priors for downstream parameters. Cascade cuts define where to split:

```
Stage 0: Immune cell trafficking rates (from density data)
    ↓ posteriors become priors
Stage 1: Cytokine concentrations (depend on cell densities)
    ↓ posteriors become priors  
Stage 2: Functional responses (depend on cytokines)
```

Without cascade cuts, parameters that appear across multiple targets pull those targets into one large joint component. When those targets involve different approximate forward models (submodels, not the full QSP model), their data can conflict — and fitting them jointly with shared parameters can produce biased posteriors as the sampler tries to compromise between incompatible constraints.

Cascade cuts break the joint into stages, so each stage fits its own data with its own forward models, and passes the resulting posterior forward as a prior. This is a bias/variance tradeoff: in a fully joint model, data from all stages can inform all shared parameters — cutting removes that cross-stage information flow, so uncertainty isn't propagated equally across the full DAG. But it also prevents conflicting approximate submodels from biasing each other's estimates. Whether the tradeoff is worth it depends on the specific DAG — cascade cuts are a tool to try when joint fitting produces suspicious results or conflicting posteriors.

## Running the audit

### Prerequisites

```bash
# Install qsp-inference with submodel support
uv pip install -e ".[submodel]"
```

Your project needs:
- A priors CSV (`parameters/pdac_priors.csv`) with prior distributions for all parameters. The loader accepts `distribution` values of `lognormal` (uses `dist_param1=mu`, `dist_param2=sigma`), `normal` (`mu`, `sigma`), `uniform` (`lower`, `upper`), and `beta` (`a`, `b`). Beta is useful for bounded fractions like simplex shares from a stick-breaking parameterization.
- SubmodelTarget YAMLs in `calibration_targets/submodel_targets/`
- Optionally, a `submodel_config.yaml` in the same directory for parameter groups and cascade cuts

### How to run it

The audit and supporting helpers are exposed as importable Python functions. The full report flow is one call:

```python
from qsp_inference.audit.report import AuditConfig, run_audit

run_audit(
    AuditConfig(project_root="/path/to/project"),
    output="audit_report.md",
    invalidate_params=["k_CD8_kill", "k_Treg_act"],  # optional
)
```

For the iterative debugging loop (re-MCMC only the components touched by a changed parameter, skip the slow PPC + report steps), use [`examples/regen_submodel_priors.py`](../examples/regen_submodel_priors.py):

```bash
python examples/regen_submodel_priors.py --project-root /path/to/project --invalidate k_CD8_kill
python examples/regen_submodel_priors.py --project-root /path/to/project --rebuild-all
```

The DAG visualization is its own helper:

```python
from qsp_inference.audit.plots import plot_inference_dag
from qsp_inference.audit.report import AuditConfig

config = AuditConfig(project_root="/path/to/project")
plot_inference_dag(config.submodel_dir, "/path/to/project/figures")
```

The first run takes a while (MCMC for each component). Subsequent runs are fast: results are cached per component and only re-run when invalidated.

### What the report tells you

The audit report has four main sections:

**1. "What should I extract next?"**
Parameters ranked by priority — high PRCC sensitivity (matters to model output) and high uncertainty (not yet constrained by data). This tells you where to focus your literature search.

**2. "Are my existing targets working?"**
Diagnostic checks on the inference results:
- **Contraction**: Did the posterior get tighter than the prior? If not, the data isn't informative for that parameter.
- **Conflicts**: Do different targets disagree about a parameter value? This might indicate a modeling issue.
- **MCMC health**: Effective sample size and R-hat — standard Bayesian diagnostics.
- **Posterior predictive checks**: Can the posterior reproduce the observed data?

**3. "What's left before stage 2?"**
Coverage summary — how many of your PRCC-significant parameters have posterior estimates.

**4. Component diagnostics**
Per-component inference details — which components used NUTS vs. NPE, divergence counts, SBC calibration for NPE components.

**5. Stage 2 NPE posterior shifts** (only when a Stage 2 SBI run is configured)
Per-parameter contraction and shift relative to both the pre-stage-1 prior and the post-stage-1 prior, plus global SBC contraction and shell-local contraction from `local_calibration.csv`. KS p-values flag local miscalibration; mean |z| flags SBC bias. Wired through `AuditConfig.sbi_run_path` (or `--sbi-run-path` on the CLI).

**6. Clinical predictive uncertainty** (only when `posterior_predictive_clinical.csv` is present)
A 95% CI table per (scenario, endpoint) with driver attribution via Spearman correlation between posterior samples and endpoint values. When PRCC results are available, drivers are restricted to PRCC-significant parameters to keep the table readable. Side artifacts `ppc_endpoint_ci.csv` and `ppc_endpoint_correlations.csv` are written next to the report for downstream analysis. This section is upstream of OBED proper: it identifies which parameters drive remaining clinical predictive uncertainty.

For details on producing the Stage 2 inputs the audit reads here (`posterior_samples`, `local_calibration.csv`, `posterior_predictive_clinical.csv`), see the [Stage 2 SBI guide](stage2-sbi-guide.md).

### Outputs

The audit produces:
- **Markdown report** with tables and diagnostics
- **`submodel_priors.yaml`**: Posterior marginal distributions + Gaussian copula correlation matrix — ready for downstream SBI calibration
- **Figures**: Prior vs. posterior marginals, PPC histograms, inference DAG

### Cache freshness

Each per-component cache (`comp_*.json`) carries a freshness manifest: hashes of the target YAMLs that fed the component, the prior CSV rows for the parameters it touched, the relevant slice of `submodel_config.yaml`, `reference_values.yaml`, and the qsp-inference version. The full set of fingerprints (including upstream cascade components) is mirrored into the `metadata.freshness` block of `submodel_priors.yaml`, so a downstream consumer can decide whether the file is stale relative to the current tree without re-running inference.

On a cache hit the audit warns when content has drifted but does not auto-invalidate. To force re-inference for specific parameters, pass them via `invalidate_params=[...]` to `run_audit`, or use the `--invalidate` flag on `examples/regen_submodel_priors.py`.

## Comparison with typical QSP calibration

| | Ad hoc calibration | Submodel inference |
|---|---|---|
| **Central object** | Point estimates + arbitrary ranges | Joint posterior distribution |
| **Priors** | Implicit (chosen ranges) or absent | Explicit, updatable distributions |
| **Uncertainty** | Artifact of chosen sampling range | Principled: posterior reflects data + prior |
| **Data weighting** | Equal weight (or manual) | Translation sigma weights by source relevance |
| **Correlations** | Not captured | Gaussian copula preserves joint structure |
| **Data sources** | Needs simulated-vs-observed trajectories | Works with any quantitative claim (steady-state values, ratios, fractions) |
| **Iteration** | Re-fit everything | Cache per component, only re-run what changed |
| **Diagnostics** | Visual fit quality | Contraction, convergence, posterior predictive checks |

The approaches aren't mutually exclusive. Full-model calibration against time-course data is the right thing to do when you have it — and the submodel posterior can serve as a well-informed prior for that calibration, rather than starting from arbitrary ranges.

## Creating targets with maple

Maple handles the literature extraction and YAML generation. There are two workflows (see the [maple README](https://github.com/popellab/maple) for details):

1. **Staged batch pipeline** (`examples/staged_extraction.py`): Automates multi-step extraction across many targets — literature search, PDF collection via Zotero, paper assessment, extraction, and validation. Each stage caches results per-target, so you can rerun any subset without redoing work. Best for extracting many parameters at once.

2. **MCP server** (`maple.mcp_server`): Exposes extraction and validation as tools that Claude Code (or any MCP client) can call interactively. Best for one-off extractions or iterating on a specific target.

Both workflows produce SubmodelTarget YAMLs that go in your project's `calibration_targets/submodel_targets/` directory. Maple chooses the forward model type based on the data — `algebraic` for most cases, `custom_ode` when dynamics matter.

The key thing to review after maple generates a target:

- **Are the extracted values correct?** Check the `value_snippet` against the actual paper.
- **Is the forward model appropriate?** Does the math connecting your QSP parameter to the observable make sense?
- **Is the source relevance assessment honest?** Especially `indication_match` and `measurement_directness` — these have the largest impact on translation sigma.
- **Does the observation code make sense?** The bootstrap should reflect how the paper's reported statistics (mean +/- SEM, median + IQR, etc.) translate to uncertainty in the observable.

## Annotated example: a real SubmodelTarget

Here's a real target from a PDAC model, annotated. This one estimates the homeostatic tumor cDC2 density (`APC0_cDC2_T`) from immunohistochemistry data.

```yaml
target_id: APC0_cDC2_T

# --- Scientific context ---
# These fields document WHY this data is relevant to the model.
# maple generates these from the paper; review them for accuracy.

study_interpretation: >
  Plesca2022 used multiplex immunohistochemistry on FFPE sections from
  40 resected human PDAC specimens to quantify cDC2s as absolute densities
  in the whole tumor area. The reported mean whole-tumor-area cDC2 density
  of 2.98 cells/mm^2 reflects a quasi-steady-state balance between cDC2
  recruitment, maturation, and death in established PDAC.

key_assumptions:
  - Resected PDAC tumors represent a quasi-steady state for DC infiltration
  - Whole tumor area density is representative of the average over the tumor volume
  - Thin FFPE sections have uniform effective thickness (~4 um midpoint of 3-5 um)
  - DC pore-size exclusion is minimal at baseline in PDAC

experimental_context:
  species: human
  system: clinical_in_vivo
  indication: pancreatic ductal adenocarcinoma

# --- Data source with relevance assessment ---
primary_data_source:
  doi: 10.3390/cancers14051216
  title: "Clinical Significance of Tumor-Infiltrating DCs in PDAC"
  authors: [Plesca, Benesova, Beer, ...]
  year: 2022
  source_tag: Plesca2022

  # This is where translation sigma comes from.
  # Each axis adds uncertainty (in quadrature) to the likelihood.
  source_relevance:
    indication_match: exact              # Human PDAC data for a PDAC model → 0.0
    indication_match_justification: >
      Study analyzes DC subsets directly in human PDAC surgical specimens.
    species_source: human                # Same species → 0.0
    species_target: human
    source_quality: primary_human_clinical  # Clinical tissue → 0.0
    perturbation_type: pathological_state   # PDAC tumors → 0.0
    perturbation_relevance: >
      Patients had established PDAC at resection; cDC2 densities measured
      in untreated tumor tissue blocks.
    tme_compatibility: high              # PDAC TME matches model → 0.0
    tme_compatibility_notes: >
      Source tumors are human PDAC with characteristic desmoplastic stroma.
    measurement_directness: direct       # Counted cells per mm^2 → 0.0
    temporal_resolution: snapshot_or_equilibrium  # Single timepoint → 0.2
    experimental_system: clinical_in_vivo         # Clinical tissue → 0.0

    # Total translation sigma for this source:
    # sqrt(0^2 + 0^2 + 0^2 + 0^2 + 0^2 + 0^2 + 0.2^2 + 0^2) = 0.2
    # But floored at 0.15, so sigma_translation = 0.2

# --- Extracted data ---
# Each input has full provenance: where in the paper, exact text snippet.
inputs:
  - name: cDC2_WTA_mean_mm2
    value: 2.98
    units: cell / millimeter ** 2
    input_type: direct_measurement       # Read directly from the paper
    source_ref: Plesca2022
    source_location: "Results section 3.1, Figure 2B"
    value_snippet: "cDC2s (2.98 +/- 0.597 cDC2s/mm^2)"

  - name: cDC2_WTA_sem_mm2
    value: 0.597
    units: cell / millimeter ** 2
    input_type: direct_measurement
    source_ref: Plesca2022
    source_location: "Results section 3.1, Figure 2B"
    value_snippet: "cDC2s (2.98 +/- 0.597 cDC2s/mm^2)"

  - name: section_thickness_um_mean
    value: 4.0
    units: micrometer
    input_type: derived_arithmetic       # Computed from other inputs
    source_ref: Plesca2022
    source_location: "Methods 2.2"
    source_inputs: [section_thickness_min_um, section_thickness_max_um]
    formula: (section_thickness_min_um + section_thickness_max_um) / 2.0

  - name: n_patients_cDC
    value: 40.0
    units: dimensionless
    input_type: direct_measurement
    source_ref: Plesca2022
    source_location: "Figure 2 caption"

# --- Calibration specification ---
calibration:
  # The QSP parameters this target constrains
  parameters:
    - name: APC0_cDC2_T
      units: cell / milliliter

  # How to predict the observable from parameter values.
  # This algebraic model converts volumetric density to surface density
  # using the section thickness (2D-to-3D geometric projection).
  forward_model:
    type: algebraic
    formula: >
      density_mm2 = APC0_cDC2_T * section_thickness_cm / 100
    code: |
      def compute(params, inputs):
          rho_cdc2 = params['APC0_cDC2_T']           # cells/mL = cells/cm^3
          t_um = inputs['section_thickness_um_mean']   # micrometers
          t_cm = t_um * 1e-4                           # -> cm
          density_mm2 = rho_cdc2 * t_cm / 100.0        # -> cells/mm^2
          return density_mm2
    data_rationale: >
      Plesca2022 reports mean cDC2 densities as cells/mm^2 from thin sections.
    submodel_rationale: >
      At steady state, tumor cDC2 density equals APC0_cDC2_T projected into 2D.

  # How to generate bootstrap samples of what the data "should" look like.
  # This captures measurement uncertainty from the paper's reported statistics.
  error_model:
    - name: cDC2_WTA_density_Plesca2022
      units: cell / millimeter ** 2
      uses_inputs: [cDC2_WTA_mean_mm2, cDC2_WTA_sem_mm2, n_patients_cDC]
      sample_size_input: n_patients_cDC
      observation_code: |
        def derive_observation(inputs, sample_size, rng, n_bootstrap):
            import numpy as np
            mean = inputs['cDC2_WTA_mean_mm2']
            sem = inputs['cDC2_WTA_sem_mm2']
            n = int(inputs['n_patients_cDC'])
            sd = sem * np.sqrt(n)              # SEM -> SD
            cv = sd / mean
            sigma_ln = np.sqrt(np.log(1.0 + cv**2))
            mu_ln = np.log(mean) - 0.5 * sigma_ln**2
            return rng.lognormal(mu_ln, sigma_ln / np.sqrt(sample_size), n_bootstrap)
      n_bootstrap: 10000

  identifiability_notes: >
    Single parameter, directly observed via geometric projection.
    Identifiable up to section thickness uncertainty and inter-patient variability.
```

**What happens when you run the audit with this target:**

1. The bootstrap runs `derive_observation` 10,000 times to get a distribution of what the mean cDC2 density *should* look like given the paper's reported uncertainty
2. A distribution (lognormal, gamma, etc.) is fit to the bootstrap samples by AIC
3. The forward model `compute()` is called for each proposed parameter value during MCMC
4. The likelihood compares the forward model output to the fitted data distribution, with the variance inflated by the translation sigma (0.2 in this case — almost entirely from `snapshot_or_equilibrium`)
5. The posterior for `APC0_cDC2_T` emerges as a distribution that balances the prior (from the CSV) against this data

## Interpreting the audit report diagnostics

### Contraction

Contraction measures how much the posterior shrank compared to the prior:

```
contraction = 1 - (sigma_posterior / sigma_prior)
```

| Contraction | Interpretation |
|---|---|
| > 50% | Strong — the data substantially constrained this parameter |
| 20-50% | Moderate — the data is informative but there's still substantial uncertainty |
| < 20% | Weak — the data barely moved the posterior from the prior |
| ~0% or negative | The target isn't informative for this parameter. Check whether the forward model actually depends on it, or if the translation sigma is too large. |

Low contraction isn't necessarily bad — it might mean your prior was already well-calibrated. But if you expected a target to be informative and contraction is low, investigate.

### Z-score

Z-score measures how far the posterior median shifted from the prior median, in units of prior uncertainty:

```
z_score = (log(posterior_median) - log(prior_median)) / sigma_prior
```

| |z-score| | Interpretation |
|---|---|
| < 1 | Posterior is consistent with the prior — the data confirms your initial estimate |
| 1-2 | Moderate shift — the data is pulling the parameter in a specific direction |
| > 2 | Large shift — the data strongly disagrees with the prior. Check whether the prior was poorly chosen or the data source has issues. |

A large z-score combined with high contraction usually means the data is genuinely informative and is updating a prior that was initially off. A large z-score with *low* contraction is a red flag — something may be wrong with the target specification.

### Conflicts

The report flags parameters where different targets produce substantially different posterior estimates. This can mean:

- **Modeling issue**: The submodels make inconsistent assumptions (e.g., one assumes steady state, another doesn't)
- **Data inconsistency**: The papers genuinely disagree, which is common when sources span different experimental systems
- **Forward model error**: The mathematical relationship between the parameter and the observable is wrong in one of the targets

When you see conflicts, look at the targets involved and check whether their assumptions are compatible. Translation sigma should absorb *some* disagreement (that's what it's for), but large conflicts need investigation.

### MCMC diagnostics

**R-hat** (should be < 1.05): Measures whether the MCMC chains converged to the same distribution. Values above 1.05 mean the chains disagree — the sampler hasn't explored the posterior well enough. This usually means the component needs more warmup iterations or the model has identifiability issues.

**Effective sample size (n_eff)** (should be > 100): The number of effectively independent posterior samples after accounting for autocorrelation. Low n_eff means the chains are highly correlated — the sampler is taking small steps. This can indicate a difficult posterior geometry (strong parameter correlations, multimodality).

**Divergences**: NUTS MCMC warns when it encounters regions of high curvature in the posterior that it can't navigate accurately. A few divergences are usually fine; many (>5% of samples) suggest the model needs reparameterization or the data is in tension with the prior.

### Posterior predictive checks (PPC)

PPC asks: "If the parameter posterior is correct, can it reproduce the observed data?" For each observable, the report shows:

- **Prior predictive**: What the model predicts with the prior (before seeing data)
- **Observed**: The data (bootstrap median)
- **Posterior predictive**: What the model predicts with the posterior (after seeing data)

The observed value should fall within the 95% CI of the posterior predictive. If it doesn't, either:
- The forward model can't reproduce the data for any parameter value (model misspecification)
- The observation uncertainty is underestimated
- The translation sigma is doing its job — the model fits the data loosely because the source is low-relevance

### Translation sigma reference table

For reference, here are the sigma contributions for each relevance axis. These add in quadrature with a floor of 0.15:

| Axis | Level | Sigma |
|---|---|---|
| **Indication** | exact | 0.0 |
| | related | 0.2 |
| | proxy | 0.5 |
| | unrelated | 1.0 |
| **Species** | same | 0.0 |
| | different | 0.3 |
| **Source quality** | primary_human_clinical | 0.0 |
| | primary_human_in_vitro | 0.1 |
| | review_article | 0.2 |
| | primary_animal_in_vivo | 0.3 |
| | textbook | 0.3 |
| | primary_animal_in_vitro | 0.4 |
| | non_peer_reviewed | 0.5 |
| **Perturbation** | pathological_state | 0.0 |
| | physiological_baseline | 0.1 |
| | pharmacological | 0.25 |
| | genetic_perturbation | 0.4 |
| **TME compatibility** | high | 0.0 |
| | moderate | 0.15 |
| | low | 0.5 |
| **Measurement directness** | direct | 0.0 |
| | single_inversion | 0.15 |
| | steady_state_inversion | 0.3 |
| | proxy_observable | 0.5 |
| **Temporal resolution** | timecourse | 0.0 |
| | endpoint_pair | 0.1 |
| | snapshot_or_equilibrium | 0.2 |
| **Experimental system** | clinical_in_vivo | 0.0 |
| | ex_vivo | 0.1 |
| | animal_in_vivo | 0.15 |
| | in_vitro_coculture | 0.15 |
| | in_vitro_primary | 0.2 |
| | in_vitro_cell_line | 0.3 |

**Example**: Mouse in vivo data (species: 0.3) from a related cancer type (indication: 0.2) measuring a proxy observable (directness: 0.5) at a single timepoint (temporal: 0.2):

```
sigma = sqrt(0.3^2 + 0.2^2 + 0.5^2 + 0.2^2) = sqrt(0.42) = 0.65
```

This means the likelihood variance is inflated by `exp(2 * 0.65) ~ 3.7x` — the data is still used, but weighted much less than a direct human clinical measurement.

## Practical tips

**Start small.** Pick 3-5 parameters with clear literature data and create targets for them. Run the audit, look at the report, and iterate.

**Trust the diagnostics.** If contraction is low, the data isn't informative — don't force it. If targets conflict, investigate the biology before adding more data.

**Check the DAG.** The inference DAG visualization shows you how your targets decompose into components. If one component has too many parameters, consider whether some can be cascade-cut.

**Cache is your friend.** The `.compare_cache/` directory stores per-component results. Adding a new target only triggers inference for its component, everything else stays cached. Pass `invalidate_params=[...]` to `run_audit` (or `--invalidate` to `examples/regen_submodel_priors.py`) when you change a target's specification.

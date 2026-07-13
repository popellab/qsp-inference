---
marp: true
paginate: true
math: mathjax
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  :root {
    --color-bg: #fdfdfd;
    --color-fg: #1e1e2e;
    --color-accent: #2563eb;
    --color-muted: #64748b;
    --color-highlight: #dc2626;
    --color-surface: #f1f5f9;
  }
  section {
    background: var(--color-bg);
    color: var(--color-fg);
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 28px;
    padding: 50px 70px;
    line-height: 1.5;
  }
  h1 {
    font-weight: 700;
    font-size: 44px;
    color: var(--color-fg);
    margin-bottom: 0.3em;
    letter-spacing: -0.02em;
  }
  h2 {
    font-weight: 600;
    font-size: 32px;
    color: var(--color-muted);
    margin-top: -0.2em;
  }
  h3 {
    font-weight: 600;
    font-size: 26px;
    color: var(--color-accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5em;
  }
  strong {
    font-weight: 600;
    color: var(--color-fg);
  }
  em {
    font-style: normal;
    color: var(--color-highlight);
    font-weight: 600;
  }
  ul, ol {
    padding-left: 1.2em;
  }
  li {
    margin-bottom: 0.4em;
  }
  li::marker {
    color: var(--color-accent);
  }
  blockquote {
    background: var(--color-surface);
    border-left: 4px solid var(--color-accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8em 1.2em;
    margin: 1em 0;
    font-size: 26px;
    color: var(--color-muted);
  }
  blockquote strong {
    color: var(--color-fg);
  }
  table {
    font-size: 22px;
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
  }
  th {
    background: var(--color-accent);
    color: white;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
  }
  td {
    padding: 10px 16px;
    border-bottom: 1px solid #e2e8f0;
  }
  tr:nth-child(even) td {
    background: var(--color-surface);
  }
  code {
    background: var(--color-surface);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
  }
  pre {
    background: var(--color-surface);
    padding: 1em;
    border-radius: 8px;
    font-size: 20px;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2em;
    align-items: start;
  }
  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    padding: 60px 100px;
  }
  section.lead h1 {
    font-size: 52px;
    line-height: 1.2;
  }
  section.lead h2 {
    font-weight: 300;
    font-size: 28px;
    color: var(--color-muted);
    margin-top: 0.5em;
  }
  section.lead h3 {
    text-transform: none;
    font-size: 22px;
    color: var(--color-muted);
    font-weight: 400;
    letter-spacing: 0;
  }
  section.divider {
    display: flex;
    flex-direction: column;
    justify-content: center;
    background: var(--color-accent);
    color: white;
    padding: 60px 100px;
  }
  section.divider h1 {
    color: white;
    font-size: 52px;
  }
  section.divider h2 {
    color: rgba(255,255,255,0.7);
    font-weight: 300;
  }
  .dim {
    color: var(--color-muted);
  }
  .small {
    font-size: 20px;
    color: var(--color-muted);
  }
---

<!-- _class: lead -->
<!-- _paginate: false -->

# Using qsp-inference

## A hands-on walkthrough: from a project directory to a calibrated posterior

### qsp-inference &nbsp;|&nbsp; Lab meeting

---

# Why qsp-inference Exists

Most QSP parameters can't be measured in your clinical context — you borrow from mouse melanoma, not human PDAC.

The usual move: pick point values, pad with arbitrary ranges.

The real problem — the *probabilistic structure is never made explicit*:

- What distribution per parameter?
- Which data source informs which parameters?
- What uncertainty propagates from where?

> Without it, the "calibrated" set has no coherent meaning — uncertainty is an *artifact of arbitrary choices*.

---

# The Posterior Is the Goal

The central object is the **posterior** $p(\theta \mid \text{data})$.

Once you have it, everything else is just a *read-off*:

- **Point estimates** — posterior median or MAP
- **Uncertainty** — posterior spread, no arbitrary ranges
- **Virtual populations** — sample the joint posterior directly
- **Predictions** — propagate the posterior through the model

> The posterior is what makes calibration *reproducible* and *composable*.

---

# What qsp-inference Adds

<div class="cols">
<div>

### Explicit priors
Every parameter starts with a stated prior; data updates it

### Data weighting by relevance
Translation sigma downweights mouse / in vitro / off-indication sources

</div>
<div>

### Correlations preserved
Gaussian copula keeps joint structure across parameters

### Diagnostics built in
Contraction, convergence, posterior predictive checks

</div>
</div>

> Two stages: literature → joint posterior (Stage 1), then + full simulator + clinical data → final posterior (Stage 2).

---

# What This Talk Is

The *how*, not the *why*: what files you need, what to call, what to look at.

By the end you can:

- Set up a project directory the pipeline reads
- Run Stage 1, interpret the audit report
- Load that posterior into Stage 2, train an NPE
- Know which diagnostics to trust

<span class="small">Reference docs: <code>docs/submodel-inference-guide.md</code>, <code>docs/stage2-sbi-guide.md</code></span>

---

# The Two Stages, Concretely

```
STAGE 1 — submodel inference            STAGE 2 — full-model NPE
┌──────────────────────────────┐        ┌──────────────────────────────┐
│ inputs:                      │        │ inputs:                      │
│  • priors CSV                │        │  • submodel_priors.yaml      │
│  • SubmodelTarget YAMLs      │        │  • full QSP simulator        │
│  • submodel_config.yaml (opt)│        │  • observed test statistics  │
│                              │        │                              │
│ run: run_audit(...)          │ ─────> │ run: stage2_pipeline.py      │
│                              │        │                              │
│ output:                      │        │ output:                      │
│  • audit_report.md           │        │  • posterior_samples.csv     │
│  • submodel_priors.yaml      │        │  • diagnostics CSVs + figs   │
└──────────────────────────────┘        └──────────────────────────────┘
```

Stage 1: exact MCMC on cheap forward models. Stage 2: NPE on the full simulator. You can stop after Stage 1.

---

# Install

```bash
# core
uv pip install -e .

# Stage 1 — joint MCMC (NumPyro / JAX)
uv pip install -e ".[submodel]"

# Stage 2 — NPE (PyTorch / sbi)
uv pip install -e ".[sbi]"

# audit visualizations (graphviz)
uv pip install -e ".[audit]"
```

Requires `maple` installed (schema imports only — `SubmodelTarget`, `CalibrationTarget`)

Stage 2 also needs `qsp-hpc-tools` for the QSP simulator wrapper

---

<!-- _class: divider -->

# Stage 1
## Literature directory → joint posterior

---

# What Stage 1 Needs On Disk

The audit reads a **project directory**:

```
my-project/
├── parameters/
│   └── priors.csv                  # one prior per QSP param
├── calibration_targets/
│   └── submodel_targets/
│       ├── APC0_cDC2_T.yaml         # one target per extraction (from maple)
│       ├── k_CD8_kill.yaml
│       └── ...
│       └── submodel_config.yaml     # optional: groups + cascade cuts
```

CSV `distribution` column: `lognormal` / `normal` / `uniform` / `beta`

---

# Anatomy of a SubmodelTarget

One YAML = one paper's measurement connected to QSP parameters. Three parts:

<div class="cols">
<div>

### 1. Data — `inputs`
Extracted values with full provenance (source, location, exact snippet)

### 2. Forward model
`compute(params, inputs) -> observable` — the submodel math

</div>
<div>

### 3. Error model
`derive_observation(...)` — parametric bootstrap from reported stats

### + Source relevance
8-axis assessment → translation sigma

</div>
</div>

> maple generates these. **Your job is to review them** — extracted values, forward-model math, and an honest relevance assessment.

---

# Targets and Parameters Form a Graph

Each target touches a few parameters. Parameters shared across targets link them up.

```
target A ── k_CD8_kill ── target B          target D ── k_drug_clear
                │                                          │
            target C                                   target E
   └──────── component 1 ────────┘          └──── component 2 ────┘
```

- The graph splits into **independent components** — clusters sharing no parameters
- The audit fits each component **separately** — most are small (1–10 params), so MCMC stays exact and cheap
- A **mega-component** — too many targets in one cluster, usually via one over-shared parameter

---

# `submodel_config.yaml` — Parameter Groups

Optional file, two knobs. First: **hierarchical partial pooling** for biologically related params.

```yaml
groups:
  - group_id: kill_rates
    between_member_sd:
      distribution: half_normal
      sigma: 0.4
    members:
      - name: k_CD8_kill
      - name: k_NK_kill
```

The group shares one latent base rate; each member deviates from it. Members **with** data pull the base rate; members **without** data shrink toward it — a real estimate for params you have no direct data for.

---

# `submodel_config.yaml` — Cascade Cuts

Second knob: **staged inference**. Fit a shared param in one component, inject its posterior as the prior for the downstream component.

```yaml
cascade_cuts:
  - parameter: APC0_cDC2_T
    upstream: [cDC2_density_target]
```

- Without it, a shared param merges every target that touches it into one giant component
- The cut keeps the param in both components but **breaks the merge** — upstream runs first, downstream inherits its posterior
- Reach for it when conflicting targets are fighting inside one mega-component

---

# What `run_audit` Does

```python
from qsp_inference.audit.report import AuditConfig, run_audit

run_audit(
    AuditConfig(project_root="/path/to/my-project"),
    output="audit_report.md",
    invalidate_params=["k_CD8_kill"],   # optional: force re-MCMC
)
```

Per connected component of the parameter–target DAG:

1. Bootstrap each target's observable, fit a distribution by AIC
2. Run NUTS MCMC (or NPE if the component has slow custom ODEs)
3. Recombine components → marginals + Gaussian copula
4. Posterior predictive checks + write the markdown report

First run is slow. Results cache per component in `.compare_cache/` — only invalidated components re-run.

---

# Reading the Audit Report

The markdown report has four sections, each answering one question:

| Section | Question | What you do with it |
|---|---|---|
| **1. Extract next** | Where's the missing data? | Params ranked by PRCC × uncertainty — your literature to-do list |
| **2. Targets working?** | Did inference succeed? | Contraction, conflicts, MCMC health, PPC |
| **3. Left before Stage 2** | Coverage progress | How many sensitive params have posteriors |
| **4. Component diagnostics** | Per-component detail | NUTS vs NPE, divergences, SBC |

Start at section 2 — if targets aren't working, nothing downstream matters.

---

# Diagnostics Cheat Sheet

<div class="cols">
<div>

### Contraction
$1 - \sigma_{post}/\sigma_{prior}$

| Value | Meaning |
|---|---|
| > 50% | Strongly constrained |
| 20–50% | Moderately informative |
| < 20% | Data barely moved prior |

### MCMC health
R-hat < 1.05, n_eff > 100, divergences < 5%

</div>
<div>

### Z-score
prior → posterior shift, in prior-$\sigma$ units

| Value | Meaning |
|---|---|
| < 1 | Consistent with prior |
| 1–2 | Moderate shift |
| > 2 | Data disagrees with prior |

### The red flag
High z-score **+ low contraction** → investigate the target

</div>
</div>

---

# Fast Iteration Loop

`run_audit` is the full thing — PPC sims, plots, report. When you're just debugging a target, that's too slow.

```bash
# re-MCMC only the component(s) touching this param, rewrite priors yaml
python examples/regen_submodel_priors.py \
    --project-root /path/to/my-project --invalidate k_CD8_kill

# nuke the cache, rebuild everything
python examples/regen_submodel_priors.py \
    --project-root /path/to/my-project --rebuild-all
```

Skips PPC, plots, and the markdown report — just refreshes `submodel_priors.yaml`

**Loop:** edit a target YAML → `--invalidate` it → check contraction → repeat. Run the full `run_audit` when you want the report.

---

# The Stage 1 Output

`submodel_priors.yaml` — the joint posterior, ready to load:

```yaml
parameters:
  - name: k_apsc_prolif
    marginal: { distribution: lognormal, mu: -0.31, sigma: 0.85 }
  - name: k_apsc_death
    marginal: { distribution: gamma, shape: 2.1, scale: 0.05 }
copula:
  type: gaussian
  parameters: [k_apsc_prolif, k_apsc_death]
  correlation: [[1.00, -0.42], [-0.42, 1.00]]
metadata:
  freshness: { ... }   # hashes of inputs — detect staleness without re-running
```

Marginals (best-AIC fit per parameter) + a Gaussian copula for within-component correlation. This *is* the Stage 2 prior.

---

<!-- _class: divider -->

# Stage 2
## submodel_priors.yaml + full simulator → final posterior

---

# Stage 2: The Spine

`examples/stage2_pipeline.py` is the reference — a linear, single-scenario walkthrough:

<div class="cols">
<div>

1. Load Stage 1 posterior as prior
2. Train a RestrictionClassifier
3. Generate the training pool
4. Copula-transform observables
5. Prior-predictive sanity gate

</div>
<div>

6. Train the NPE
7. **Misspecification audit**
8. Diagnostics (recovery, calibration, ...)
9. Posterior predictive on real data
10. Save artifacts the audit re-reads

</div>
</div>

Real projects add multi-scenario stacking and sim caching — the script keeps just the spine.

---

# Step 1: Stage 1 Posterior → Stage 2 Prior

```python
from qsp_inference.priors import load_composite_prior_log

prior_log, param_names = load_composite_prior_log(
    "submodel_priors.yaml",
    "parameters/priors.csv",
)
samples = prior_log.sample((10_000,))   # log-space theta
```

- A `torch.distributions` object over **log(theta)**
- Stage-1 params → from the Gaussian copula; everything else → from the CSV
- Off-by-a-little correlation matrices are auto-repaired

`load_composite_prior_log` covers *all* params; `load_copula_prior_log` only the Stage-1 subset.

---

# Step 2: RestrictionClassifier

A fraction of prior draws are unusable — mostly tumors that never reach detectable diameter in the simulated time window. Don't waste HPC on them.

```python
from qsp_inference.inference import train_restriction_classifier

clf = train_restriction_classifier(
    theta=pilot_theta,           # (n, p) original scale
    valid_mask=pilot_valid,      # bool: did the sim return finite output?
    feature_order=param_names,
    cv_folds=5,
)
print(clf.threshold_curve)       # (accept_frac, survival, yield) per cutoff
clf.save("classifiers/v1/")
```

sklearn boosted trees on log-theta predicting $P(\text{valid} \mid \theta)$. You then pick a **cutoff** — draws scoring below it get rejected.

---

# Step 2: Picking the Cutoff

A high cutoff gives better yield, but it also distorts the prior.

The `restrict_threshold_geometry.py` workflow <span class="dim">(landing in qsp-inference soon)</span> shows how each cutoff reshapes the prior:

- **Per-param KS distance** — how far the accepted marginal moved
- **Median shift** — did the accepted draws recenter?
- **Width ratio** — did the spread narrow?

<div class="cols">
<div>

### Loose cutoff
Filters via *joint* constraints — marginals stay intact

</div>
<div>

### Tight cutoff
Starts squeezing biologically meaningful prior mass

</div>
</div>

---

# Step 3: Generate the Training Pool

```python
from qsp_inference.inference import sample_restricted

theta_train, total_draws = sample_restricted(
    classifier=clf,
    prior_sample_fn=lambda n: prior_log.sample((n,)).exp().numpy(),
    n_accepted=5_000,
    threshold=0.5,
    max_draws=100_000,        # safety cap for low-survival regimes
)
x_train_raw = qsp_simulator(theta_train)   # your QSPSimulator wrapper

# drop any sim that failed on any observable
ok = np.all(np.isfinite(x_train_raw), axis=1)
theta_train, x_train_raw = theta_train[ok], x_train_raw[ok]
```

`qsp_simulator` wraps `QSPSimulator.simulate_with_parameters` from qsp-hpc-tools — theta-hashed caching, so repeated calls hit the cache.

---

# Step 4: Copula Transform on Observables

NPE trains better when observables are standard-normal. Compute the transform **once**, reuse it.

```python
from qsp_inference.inference import (
    compute_quantiles_from_array, transform_to_normal_from_array,
)

quantiles = compute_quantiles_from_array(x_train_raw, n_quantiles=1000)
x_train_t = transform_to_normal_from_array(x_train_raw, quantiles)
x_test_t  = transform_to_normal_from_array(x_test_raw, quantiles)

# the real observation must live in the SAME space
obs_t = transform_to_normal_from_array(obs_raw[None, :], quantiles)[0]
```

The one rule: condition the trained network on `obs_t`, not `obs_raw`.

---

# Step 5: Prior-Predictive Sanity Gate

Cheap check, run it *before* training: **is the observed data even in the model's reach?**

```python
from qsp_inference.inference import sbi_self_reference_null

sr_prior, _ = sbi_self_reference_null(x_train_raw, obs_raw, observable_names)
print(f"prior-predictive D² p-value = {sr_prior['p_value']:.4f}")
```

If `x_obs` is outside what the model can produce, **no amount of NPE training fixes that** — fix the model or priors first. The most common Stage 2 failure; fail loud here.

---

# Step 6: Train the NPE

```python
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

density_estimator = posterior_nn(
    model="nsf", hidden_features=64, num_transforms=5, num_bins=8,
)
inference = NPE(prior=prior_log, density_estimator=density_estimator)
inference.append_simulations(theta_train_log, x_train_t)
inference.train(training_batch_size=256, max_num_epochs=200)
posterior = inference.build_posterior()
```

Standard `sbi` — a normalizing-flow density estimator on (log-theta, transformed-x) pairs. The Stage 1 prior goes in as `prior=` so `sbi` knows the support.

---

# Step 7: Misspecification Audit — Why

Every diagnostic that follows **assumes the simulator can produce the observed data.**

If it can't, recovery and calibration measure the wrong thing — the network inverts a model that doesn't match reality, and its posterior still looks confident.

> Misspecification first. A well-calibrated posterior of a wrong model is still wrong.

Step 5's prior-predictive gate is the cheap front-end; Step 7 is the full version.

---

# Step 7: Misspecification Audit — What

`misspec_audit.py` <span class="dim">(landing in qsp-inference soon)</span> runs joint and per-observable checks, prior vs. posterior:

- Is `x_obs` inside the predictive distribution, jointly — not just marginally?
- Which observables are in tension with the rest?
- Does the posterior predictive actually close the gap the prior left?

If the audit flags misspecification, the fix is upstream — the model, the priors, or which observables you condition on — not more NPE training.

---

# Step 8: Diagnostics — Pick by Question

`qsp_inference.inference.diagnostics` — ~16 functions, each returns numbers **and** a figure.

| Question | Function |
|---|---|
| Did the network learn the params? | `sbi_recovery` (R² vs true theta) |
| Posterior tighter than prior? Biased? | `sbi_z_score_contraction` |
| Are credible intervals calibrated? | `sbi_calibration_ecdf` |
| Per-param calibration in numbers | `compute_per_param_calibration` |
| Posterior piling at prior bounds? | `sbi_boundary_piling` |
| Persist everything for the audit | `save_diagnostics` |

Usual order: recovery + contraction → calibration ECDF → boundary piling → PPC.

---

# Step 9: Posterior Predictive on Real Data

```python
# sample the posterior at the real observation
post = posterior.sample((10_000,), x=torch.from_numpy(obs_t).float())
theta_post = np.exp(post.numpy())

# push back through the simulator
x_post_pred = qsp_simulator(theta_post)

coverage_df, fig = sbi_posterior_predictive_coverage(
    x_post_pred, obs_raw, observable_names, log_scale=True,
)
sr_post, _ = sbi_self_reference_null(x_post_pred, obs_raw, observable_names)
```

The question: *can the posterior reproduce the observed test statistics?* Observed value should land in the 95% PPC band per observable.

---

# Step 10: Save Artifacts the Audit Re-Reads

```python
posterior_samples_df = convert_posterior_samples_to_original_space(
    post_samples, param_names)
posterior_samples_df.to_csv(SAVE_DIR / "posterior_samples.csv")

save_diagnostics(save_dir=SAVE_DIR, param_names=param_names, ...)
```

Point the audit at the Stage 2 run and it adds two report sections:

```python
run_audit(AuditConfig(project_root="...", sbi_run_path="runs/stage2_example"),
          output="audit_report.md")
```

- **NPE posterior shifts** — contraction/shift vs the Stage 1 prior
- **Clinical predictive uncertainty** — which params drive remaining endpoint uncertainty

---

# Then: OBED

Once calibration is decent, `qsp_inference.inference.obed` answers *what to measure next*:

- **Classify responses** from posterior-predictive tumor trajectories — `classify_mpr`, `classify_recist`
- **Mutual information** between candidate observables and the response — `mi_ksg`, `mi_sweep_*`
- **LOO retraining** — drop each observable, see how posterior width grows

> **OBED is downstream of decent calibration.** If `sbi_calibration_ecdf` shows miscalibration, MI estimates are unreliable. Fix calibration first.

---

# End-to-End Checklist

<div class="cols">
<div>

### Stage 1
1. Project dir: priors CSV + target YAMLs
2. `run_audit(...)`
3. Read report §2 — contraction, conflicts, PPC
4. `--invalidate` + iterate
5. Ship `submodel_priors.yaml`

</div>
<div>

### Stage 2
1. `load_composite_prior_log(...)`
2. RestrictionClassifier on a pilot pool
3. Training pool + copula transform
4. **Prior-predictive sanity gate**
5. Train NPE → diagnostics → PPC
6. `save_diagnostics` → audit re-reads

</div>
</div>

<br>

The cache makes both stages incremental — change one target, only its component re-runs.

---

# Gotchas

- **Cache staleness** — audit *warns* on drift, doesn't auto-invalidate. Use `--invalidate`.
- **Stage 2 is log-space** — prior, NPE theta, posterior samples. Convert back for outputs.
- **Same transform for `obs`** — condition on `obs_t`, not `obs_raw`.
- **NaN rows** — solver failures; filter before training, every time.
- **Loose PPC?** — check the relevance assessment before blaming the model.

---

<!-- _class: divider -->

# Discussion
## What's the first parameter set you'd run this on? Where do you expect it to break?

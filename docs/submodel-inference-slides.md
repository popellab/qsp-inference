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

# Submodel Inference
## From literature data to a joint posterior for QSP calibration

### qsp-inference &nbsp;|&nbsp; April 2025

---

# The Posterior Is the Goal

The central object of Bayesian inference is the **posterior distribution** $p(\theta \mid \text{data})$

When you have a posterior, everything else follows:

- **Point estimates** — summaries of the posterior (median, MAP)
- **Uncertainty** — posterior spread, no arbitrary ranges needed
- **Virtual populations** — draw directly from the joint posterior
- **Predictions** — propagate the posterior forward through the model

The posterior is not just a nice-to-have — it's what makes calibration *reproducible* and *composable*

---

# What Goes Wrong Without One

There's nothing wrong with using different methods for different parameters — fitting to time-course data, setting values from literature, estimating from in vitro assays

The problem is when the *probabilistic structure* is never made explicit:

- What distribution are you assuming for each parameter?
- How does each data source connect to which parameters?
- What uncertainty propagates from where?

Without a clear DAG of priors $\rightarrow$ likelihoods $\rightarrow$ posteriors, the "calibrated" parameter set has no coherent interpretation, and uncertainty estimates become *artifacts of arbitrary choices*

---

<!-- _class: divider -->

# Discussion
## How do you currently choose parameter values? What do you do when two sources disagree?

---

# Two-Stage Calibration

```
  Stage 1 — submodel priors              Stage 2 — full model calibration
  ┌────────────────────────────┐          ┌──────────────────────────┐
  │                            │          │                          │
  │  Literature data           │          │  Informative priors      │
  │  (in vitro, preclinical)   │          │  (from Stage 1)          │
  │  + self-contained forward  │          │  + clinical data         │
  │    models                  │ ──────>  │  + full QSP simulator    │
  │           ↓                │          │           ↓              │
  │  Joint MCMC (NumPyro)      │          │  SBI (SNPE-C)            │
  │           ↓                │          │           ↓              │
  │  Marginals + copula        │          │  Final posterior         │
  └────────────────────────────┘          └──────────────────────────┘
```

- **Stage 1** uses **SubmodelTargets** — self-contained forward models, differentiable, NUTS MCMC
- **Stage 2** uses **CalibrationTargets** — clinical observables requiring the full simulator, simulation-based inference

This talk covers Stage 1.

---

# What Is a SubmodelTarget?

A structured extraction from one paper connecting a literature measurement to model parameters

<div class="cols">
<div>

### Components

- **Inputs** — extracted values with full source provenance
- **Forward model** — self-contained math: params $\rightarrow$ observable
- **Error model** — bootstrap from reported statistics
- **Source relevance** — how well the context translates

</div>
<div>

### Key property

Each target is a *standalone inference problem* — no full QSP simulation needed

The forward model is a submodel: a small piece of the QSP model that can be evaluated cheaply and differentiated for NUTS

</div>
</div>

---

# Example: Tumor cDC2 Density

Plesca 2022 — multiplex IHC on 40 resected human PDAC specimens

### Forward model (algebraic)

$$\text{density}_{mm^2} = \text{APC0\_cDC2\_T} \times t_{section} / 100$$

Volumetric density (cells/mL) $\rightarrow$ surface density (cells/mm$^2$) via section thickness

### Error model

Mean $= 2.98 \pm 0.597$ cells/mm$^2$ (SEM, n=40)

Bootstrap: SEM $\rightarrow$ SD, fit lognormal, sample mean of n patients

Source: human PDAC clinical tissue, direct measurement $\rightarrow$ $\sigma_{trans} = 0.2$

---

# How the Pipeline Processes a Target

1. Run `derive_observation` 10,000 times $\rightarrow$ bootstrap distribution of the observable
2. Fit a parametric distribution (lognormal, gamma, ...) to the bootstrap by AIC
3. During MCMC, call the forward model `compute()` for each proposed $\theta$
4. Likelihood compares forward model output to fitted data distribution, variance inflated by $\sigma_{trans}$
5. Posterior for each parameter balances the prior against the data

Parameters sharing data across targets are estimated jointly — shared parameters are sampled once and reused across likelihoods

---

<!-- _class: divider -->

# Discussion
## What makes a good forward model? When is the literature measurement too far from the model parameterization to be useful?

---

# Translation Sigma

Literature data rarely comes from the exact context of the model

Each target has a **source relevance assessment** — 8 axes that quantify the gap between the data context and the model context

$$\sigma_{trans} = \max\left(\sqrt{\sum_i \sigma_i^2},\; 0.15\right)$$

Applied *inside* the likelihood during MCMC, not post-hoc. MCMC naturally upweights more relevant sources.

---

# Translation Sigma Rubric

<div class="cols">
<div>

| Axis | Levels |
|------|--------|
| **Indication** | exact (0), related (0.2), proxy (0.5), unrelated (1.0) |
| **Species** | same (0), different (0.3) |
| **Source quality** | clinical (0) $\rightarrow$ non-peer (0.5) |
| **TME compatibility** | high (0), moderate (0.15), low (0.5) |

</div>
<div>

| Axis | Levels |
|------|--------|
| **Perturbation** | pathological (0) $\rightarrow$ genetic (0.4) |
| **Directness** | direct (0) $\rightarrow$ proxy (0.5) |
| **Temporal** | timecourse (0) $\rightarrow$ snapshot (0.2) |
| **System** | clinical (0) $\rightarrow$ cell line (0.3) |

</div>
</div>

<br>

**Example:** mouse in vivo, related indication, proxy observable, snapshot

$\sigma_{trans} = \sqrt{0.3^2 + 0.2^2 + 0.5^2 + 0.2^2} = 0.65$

---

# Component Decomposition

The DAG of parameters $\leftrightarrow$ targets is usually *sparse*

- A T cell trafficking rate and a drug clearance rate don't share any data
- The pipeline automatically finds connected components and runs MCMC on each independently
- Components are small (1-10 parameters) $\rightarrow$ exact MCMC, no approximations

Per-component posteriors are recombined into a joint posterior:
- **Marginals** — best-AIC fit per parameter (lognormal, gamma, ...)
- **Gaussian copula** — captures within-component correlations
- Cross-component correlations are zero by construction

---

# Parameter Groups

Some parameters are biologically related — all killing rates, all trafficking rates

**Hierarchical partial pooling:**

$$k_{base} \sim \text{LogNormal}(\mu, \sigma)$$
$$k_i = k_{base} \cdot \exp(\delta_i), \quad \delta_i \sim \text{Normal}(0, \tau)$$

- Members *with* data get pulled by their observations
- Members *without* data shrink toward the group mean
- $\tau$ controls how much variation is allowed within the group

A principled way to estimate parameters you don't have direct data for

---

# Cascade Cuts

When shared parameters pull many targets into one large joint component, fitting can struggle — the forward models are *approximations*, and conflicting data can bias the posterior

**Cascade cuts** split the joint into stages:

```
Stage 0: Immune cell densities (from IHC data)
    ↓ posteriors become priors
Stage 1: Cytokine concentrations (depend on cell densities)
```

This is a **bias/variance tradeoff**: cutting removes cross-stage information flow (data can't "talk back" to upstream parameters), but prevents conflicting submodels from biasing each other

Whether the tradeoff is worth it depends on the DAG — try it when joint fitting produces suspicious results

---

<!-- _class: divider -->

# Discussion
## When would you split the DAG vs. keep it joint? What signals would make you try a cascade cut?

---

# The Audit Report

Run the full pipeline with one call:

```python
from qsp_inference.audit.report import AuditConfig, run_audit
run_audit(AuditConfig(project_root="/path/to/project"),
          output="audit_report.md")
```

The report tells you:

1. **What to extract next** — parameters ranked by PRCC sensitivity $\times$ uncertainty
2. **Are targets working?** — contraction, conflicts, MCMC health, PPC
3. **Coverage progress** — how many sensitive parameters have posteriors
4. **Component diagnostics** — per-component details (NUTS vs NPE, divergences)

Results are cached per component — adding a target only re-runs its component

---

# Key Diagnostics

<div class="cols">
<div>

### Contraction

$1 - \sigma_{post} / \sigma_{prior}$

| Value | Meaning |
|-------|---------|
| > 50% | Data strongly constrains |
| 20-50% | Moderately informative |
| < 20% | Data barely moved the prior |

</div>
<div>

### Z-score

$(\log \tilde\theta_{post} - \log \tilde\theta_{prior}) / \sigma_{prior}$

| Value | Meaning |
|-------|---------|
| < 1 | Consistent with prior |
| 1-2 | Moderate shift |
| > 2 | Data disagrees with prior |

</div>
</div>

<br>

High z-score + high contraction = informative data updating a wrong prior

High z-score + *low* contraction = red flag — investigate the target

---

# Posterior Predictive Checks

"If the posterior is correct, can it reproduce the data?"

For each observable:

- **Prior predictive** — model predictions under the prior
- **Observed** — bootstrap median from the paper
- **Posterior predictive** — model predictions under the posterior

Observed should fall within the 95% CI of the posterior predictive

If not: forward model misspecification, underestimated uncertainty, or translation sigma doing its job (loose fit for low-relevance sources)

---

# Output: `submodel_priors.yaml`

The audit produces a structured YAML with the joint posterior:

```yaml
parameters:
  - name: k_apsc_prolif
    marginal: { distribution: lognormal, mu: -0.31, sigma: 0.85 }
  - name: k_apsc_death
    marginal: { distribution: gamma, shape: 2.1, scale: 0.05 }

copula:
  type: gaussian
  parameters: [k_apsc_prolif, k_apsc_death]
  correlation:
    - [1.00, -0.42]
    - [-0.42, 1.00]
```

This is loaded directly as a PyTorch prior for Stage 2 SBI calibration

---

# Comparison

| | Ad hoc calibration | Submodel inference |
|---|---|---|
| **Central object** | Point estimates + arbitrary ranges | Joint posterior |
| **Priors** | Implicit or absent | Explicit, updatable |
| **Uncertainty** | Artifact of chosen range | Consequence of data |
| **Data weighting** | Equal or manual | Translation sigma |
| **Correlations** | Not captured | Gaussian copula |
| **Diagnostics** | Visual fit | Contraction, PPC, convergence |

The approaches are complementary — the submodel posterior is a well-informed *starting point* for full-model calibration, not a replacement for it

---

# Getting Started

1. **Pick 3-5 parameters** with clear literature data
2. **Create SubmodelTargets** using maple (batch pipeline or MCP server)
3. **Run `run_audit`** — review contraction, conflicts, PPC
4. **Iterate** — add targets, adjust cascade cuts, re-run

```bash
uv pip install -e ".[submodel]"
```
```python
from qsp_inference.audit.report import AuditConfig, run_audit
run_audit(AuditConfig(project_root="."), output="audit_report.md")
# fast iteration: python examples/regen_submodel_priors.py --invalidate <param>
```

Cache is your friend — only invalidated components re-run

---

<!-- _class: divider -->

# Discussion
## What parameters in your model would you want to try this on first? What data sources would you use?

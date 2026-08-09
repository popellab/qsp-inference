# qsp-inference

Bayesian calibration and virtual-population generation for quantitative systems pharmacology (QSP) models.

Most QSP parameters can't be measured directly in the clinical context being modeled, and most calibration workflows pick literature point values and pad them with arbitrary ranges for sensitivity analysis. `qsp-inference` replaces that with a two-stage Bayesian calibration: turn each literature measurement into a forward-model likelihood with automatic downweighting for context-mismatched sources, combine them into a joint posterior over the QSP parameters, then use that posterior as an informative prior for full-simulator inference against clinical data.

Project page with full write-up and figures: [joeleliason.com/projects/qsp-inference](https://joeleliason.com/projects/qsp-inference/).

## The model

Underneath everything here is one generative model — parameters drawn from an informative prior, pushed through the QSP simulator, observed with measurement noise — and two questions asked of it.

**Flat calibration** is the single-unit posterior `p(θ | x_obs)`: one parameter set, uncertainty and all, consistent with one observed summary. A **virtual population** is the random-effects version of the same model: each patient carries their own `θ` drawn from a population distribution `F(θ | μ, ω)`, the data are cohort summaries (a median and an IQR per target), and a virtual patient is a draw from the fitted distribution. Same simulator, same prior — only the question changes. In mixed-effects terms these are the typical-value fit and the random-effects distribution.

Two facts shape every method here. The likelihood cannot be evaluated — the mean map is an ODE solve with no analytic form, and the default noise model is an empirical resampler rather than a density — so inference is simulation-based throughout. And only a handful of the parameters are identified by the clinical data; along the rest the posterior equals the prior. That is why the prior is *constructed* as a measurement model rather than asserted, and why a wide prior is not a safe default.

**[`docs/model-draft.tex`](docs/model-draft.tex)** states all of this properly and is the model spec. It is the page to read first, and its `eq:` labels are what the code's docstrings cite.

## How it works

### Stage 1: literature → joint posterior

The Stage 1 input is a set of *SubmodelTargets*: structured extractions from individual papers, produced by [maple](https://github.com/popellab/maple), each pairing a measurement with a small forward model that predicts that measurement from QSP parameters. Each target also carries a source relevance assessment that produces a *translation sigma* — extra likelihood noise that downweights context-mismatched sources (e.g. mouse data on a related cancer downweighting relative to direct human clinical data).

Stage 1 partitions the parameter–target graph into independent inference chunks (typically 1–10 parameters each) and fits each chunk separately: NumPyro NUTS for chunks with JAX-jittable forward models, component-wise neural posterior estimation for chunks whose ODE solves are too slow for NUTS' many-step trajectories. A `submodel_config.yaml` adds two optional knobs on top: parameter groups for hierarchical partial pooling, and cascade cuts that force a parameter to be inferred upstream and pass its posterior forward as a prior for downstream chunks.

The joint posterior is parameterized as marginals plus a Gaussian copula and stamped with per-component content fingerprints in `submodel_priors.yaml` so consumers can detect when the posterior is stale.

`qsp_inference.audit.report.run_audit()` runs the full Stage 1 pipeline plus a markdown diagnostic report: contraction, conflicts, MCMC health, and an extraction-priority ranking (which parameters most need more data).

### Stage 2: clinical data → final posterior

Stage 2 inputs are *CalibrationTargets*: clinical observables (baseline immune cell densities, tumor volume trajectories, biomarker time courses, etc.) that need the full QSP simulator to evaluate. The Stage 1 posterior loads as a `torch.distributions` object and serves as the prior for neural posterior estimation via [`sbi`](https://sbi-dev.github.io/sbi/) — simulate many `(θ, x)` pairs, train a normalizing-flow conditional density estimator, and condition on the observed `x` to get the Stage 2 posterior.

A `RestrictionClassifier` (sklearn boosted trees on log-θ) rejection-samples the prior to filter out biologically implausible parameter combinations before the simulator gets called, and survives prior changes (parameters added or retired) via projection helpers. Diagnostics cover recovery, calibration ECDF, posterior predictive coverage, Mahalanobis self-reference null and LOO predictive influence for misspecification, and clinical predictive uncertainty for optimal Bayesian experimental design (OBED).

## Related projects

`qsp-inference` is one piece of a four-repo QSP modeling stack:

- **[maple](https://github.com/popellab/maple)** — schema-validated LLM extraction of QSP calibration targets from literature; produces the SubmodelTargets and CalibrationTargets that feed Stage 1 / Stage 2 here.
- **[qsp-codegen](https://github.com/popellab/qsp-codegen)** — SBML to C++ CVODE code generator that emits the `qsp_sim` simulator.
- **[qsp-hpc-tools](https://github.com/popellab/qsp-hpc-tools)** — SLURM-aware orchestration and three-tier caching for the simulation campaigns Stage 2 needs.

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

Requires [maple](https://github.com/popellab/maple) for SubmodelTarget and CalibrationTarget schemas.

## Quick start

### Run the parameter audit

```python
from qsp_inference.audit.report import AuditConfig, run_audit

run_audit(
    AuditConfig(project_root="/path/to/project"),
    output="audit_report.md",
)
```

This runs component-wise Bayesian inference on all SubmodelTarget YAMLs, generates a diagnostic report, and writes `submodel_priors.yaml` with the joint posterior.

For the iterative debugging loop (re-MCMC only the components touched by an edited parameter or YAML, skip the slow PPC + report steps), use [`examples/regen_submodel_priors.py`](examples/regen_submodel_priors.py).

### Audit whether a submodel target is right

When a component's posterior predictive misses its own observables, the cause is
usually a unit error, a forward model with its asymptotes on the wrong ends of
the curve, or a badly mis-centred CSV prior. `ppc_audit` lays out the evidence
and `refit_check` decides whether a proposed fix actually helps.

```python
from qsp_inference.submodel.ppc_audit import load_components, rank_by_miss, format_component
from qsp_inference.submodel.refit_check import compare_edit

comps = load_components(cache_dir, priors_csv)
print(format_component(rank_by_miss([c for c in comps if c.coverage < 1.0])[0]))

result = compare_edit(
    target_dir=submodel_dir,
    filenames=["IL1_50_IL6_PDAC_deriv001.yaml"],
    edits={"IL1_50_IL6_PDAC_deriv001.yaml": candidate_path},
    priors_csv=priors_csv,
    config_path=submodel_config,
    params={"IL1_50", "n_IL1"},
)
result.improved  # coverage rose, or held while the worst miss shrank
```

`compare_edit` fits with and without the edit over an identical isolated target
set, so the edit is the only difference. `ppc_audit` reports evidence and
attaches no verdict, on purpose: thresholds for "badly fitting" did not hold up
under testing, so the refit is what decides.

### Load the posterior as a prior for SBI

```python
from qsp_inference.priors import load_composite_prior_log

# Copula prior for submodel params + independent fallback for the rest
prior, param_names = load_composite_prior_log(
    "submodel_priors.yaml",
    "parameters/priors.csv",
)

samples = prior.sample((10000,))  # log-space samples
log_p = prior.log_prob(samples)   # evaluates joint density
```

## Documentation

- **[`docs/model-draft.tex`](docs/model-draft.tex)** — the model spec, and the only one. Its `eq:` labels are what the docstrings cite.
- **[`docs/slides/`](docs/slides)** — the slides built from it.

The guide set that used to sit here (statistical-model, submodel-inference-guide,
stage2-sbi-guide, population-inference-guide, population-inference-tractable)
described a package layout that no longer exists and has been deleted.

## Testing

```bash
pytest                           # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests (requires NumPyro)
```

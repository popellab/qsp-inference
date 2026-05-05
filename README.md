# qsp-inference

Bayesian inference tools for quantitative systems pharmacology (QSP) models. Decomposes large parameter spaces into tractable submodels, runs MCMC or neural posterior estimation on each component, and produces a joint posterior (marginals + Gaussian copula) for downstream calibration.

## What it does

- **Submodel inference**: Joint NumPyro NUTS MCMC across SubmodelTarget and CalibrationTarget YAMLs, with translation sigma weighting data sources by relevance. Lognormal, normal, uniform, and Beta priors are supported in the priors CSV.
- **Parameter audit**: Coverage reporting, priority scoring, diagnostics, and DAG visualization via `qsp_inference.audit.report.run_audit()`. When a Stage 2 SBI run is provided, the report adds Stage 2 NPE posterior shifts and a clinical predictive uncertainty section.
- **Copula priors**: Posterior parameterization as marginal distributions (lognormal, normal, uniform, Beta, gamma, invgamma) + Gaussian copula, loadable as PyTorch distributions for SBI workflows. Composite priors fall back to the CSV for parameters not covered by the copula.
- **Prior restriction for SBI**: Classifier-based rejection sampling (`RestrictionClassifier`) with projection helpers so the classifier survives prior changes (parameters added or retired).
- **SBI diagnostics**: Recovery, calibration, posterior predictive checks, optimal Bayesian experimental design (OBED), and learning curves for neural posterior estimation.
- **Cache freshness**: Per-component content fingerprints (target YAMLs, prior CSV rows, config slices, package version) are stamped into `submodel_priors.yaml` so consumers can detect when posteriors are stale relative to the current tree.

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

### Load the posterior as a prior for SBI

```python
from qsp_inference.priors import load_composite_prior_log

# Copula prior for submodel params + independent fallback for the rest
prior, param_names = load_composite_prior_log(
    "submodel_priors.yaml",
    "parameters/pdac_priors.csv",
)

samples = prior.sample((10000,))  # log-space samples
log_p = prior.log_prob(samples)   # evaluates joint density
```

## Documentation

- **[Submodel Inference Guide](docs/submodel-inference-guide.md)** — Stage 1: practical guide covering the Bayesian framework, SubmodelTarget YAML anatomy, maple workflows, the audit API, and diagnostics interpretation
- **[Stage 2 SBI Guide](docs/stage2-sbi-guide.md)** — Stage 2: loading the Stage 1 posterior as an SBI prior, prior restriction with `RestrictionClassifier`, NPE data prep, the diagnostics suite, posterior predictive checks, OBED, and how Stage 2 outputs feed back into the audit report

## Testing

```bash
pytest                           # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests (requires NumPyro)
```

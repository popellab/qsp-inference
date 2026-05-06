# qsp-inference

Two-stage Bayesian calibration for quantitative systems pharmacology (QSP) models.

Most QSP parameters can't be measured directly in the clinical context being modeled, and most calibration workflows pick literature point values and pad them with arbitrary ranges for sensitivity analysis. `qsp-inference` replaces that with a two-stage Bayesian calibration: turn each literature measurement into a forward-model likelihood with automatic downweighting for context-mismatched sources, combine them into a joint posterior over the QSP parameters, then use that posterior as an informative prior for full-simulator inference against clinical data.

Project page with full write-up and figures: [joeleliason.com/projects/qsp-inference](https://joeleliason.com/projects/qsp-inference/).

## How it works

### Stage 1: literature → joint posterior

The Stage 1 input is a set of *SubmodelTargets*: structured extractions from individual papers, produced by [maple](https://github.com/popellab/maple), each pairing a measurement with a small forward model that predicts that measurement from QSP parameters. Each target also carries a source relevance assessment that produces a *translation sigma* — extra likelihood noise that downweights context-mismatched sources (e.g. mouse data on a related cancer downweighting relative to direct human clinical data).

Stage 1 partitions the parameter–target graph into independent inference chunks (typically 1–10 parameters each) and fits each chunk separately: NumPyro NUTS for chunks with JAX-jittable forward models, component-wise neural posterior estimation for chunks whose ODE solves are too slow for NUTS' many-step trajectories. A `submodel_config.yaml` adds two optional knobs on top: parameter groups for hierarchical partial pooling, and cascade cuts that force a parameter to be inferred upstream and pass its posterior forward as a prior for downstream chunks.

The joint posterior is parameterized as marginals plus a Gaussian copula and stamped with per-component content fingerprints in `submodel_priors.yaml` so consumers can detect when the posterior is stale.

`qsp_inference.audit.report.run_audit()` runs the full Stage 1 pipeline plus a markdown diagnostic report: contraction, conflicts, MCMC health, and an extraction-priority ranking (which parameters most need more data). See the [Submodel Inference Guide](docs/submodel-inference-guide.md) for the full Stage 1 walkthrough.

### Stage 2: clinical data → final posterior

Stage 2 inputs are *CalibrationTargets*: clinical observables (baseline immune cell densities, tumor volume trajectories, biomarker time courses, etc.) that need the full QSP simulator to evaluate. The Stage 1 posterior loads as a `torch.distributions` object and serves as the prior for neural posterior estimation via [`sbi`](https://sbi-dev.github.io/sbi/) — simulate many `(θ, x)` pairs, train a normalizing-flow conditional density estimator, and condition on the observed `x` to get the Stage 2 posterior.

A `RestrictionClassifier` (sklearn boosted trees on log-θ) rejection-samples the prior to filter out biologically implausible parameter combinations before the simulator gets called, and survives prior changes (parameters added or retired) via projection helpers. Diagnostics cover recovery, calibration ECDF, posterior predictive coverage, Mahalanobis self-reference null and LOO predictive influence for misspecification, and clinical predictive uncertainty for optimal Bayesian experimental design (OBED). See the [Stage 2 SBI Guide](docs/stage2-sbi-guide.md) for the full walkthrough.

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

- **[Submodel Inference Guide](docs/submodel-inference-guide.md)** — Stage 1: practical guide covering the Bayesian framework, SubmodelTarget YAML anatomy, maple workflows, the audit API, and diagnostics interpretation
- **[Stage 2 SBI Guide](docs/stage2-sbi-guide.md)** — Stage 2: loading the Stage 1 posterior as an SBI prior, prior restriction with `RestrictionClassifier`, NPE data prep, the diagnostics suite, posterior predictive checks, OBED, and how Stage 2 outputs feed back into the audit report

## Testing

```bash
pytest                           # All tests
pytest tests/unit/               # Unit tests only
pytest tests/integration/        # Integration tests (requires NumPyro)
```

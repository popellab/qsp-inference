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
│   └── utils.py                 # ODE/algebraic forward model evaluation
├── inference/                   # SBI diagnostics and data processing
│   ├── diagnostics.py           # Recovery, calibration, coverage, MMD, learning curves
│   ├── data_processing.py       # NaN filtering, noise injection, simulator wrappers
│   ├── gaussian_copula_transform.py  # Quantile-based normalization
│   ├── plot_distributions.py    # Posterior visualization (marginals, pairs)
│   ├── posterior_predictive.py  # Prior/posterior predictive checks
│   ├── active_subspace.py       # Active subspace analysis
│   └── obed.py                  # Optimal Bayesian experimental design
├── data/                        # Data aggregation
│   ├── test_stat_functions.py   # Test statistics from QSP outputs
│   ├── aggregate_test_statistics.py
│   ├── aggregate_quick_estimates.py
│   ├── assess_normality.py
│   └── combine_test_stats_chunks.py
├── priors/                      # Prior loading and transformation
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

### `audit.report` — Parameter Audit

Project-agnostic audit engine. Configure via `AuditConfig`:

```python
from qsp_inference.audit.report import AuditConfig, run_audit

config = AuditConfig(project_root=Path("/path/to/pdac-build"))
report = run_audit(config, output=Path("audit_report.md"))
```

CLI:
```bash
python -m qsp_inference.audit.report --project-root /path/to/project --output report.md
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

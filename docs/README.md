# qsp-inference documentation

`qsp-inference` is **one generative model with two inference targets** — a
single-unit (flat) posterior and a random-effects population (VPop). Start with the
model, then read the chapter you need.

## Start here

- **[The statistical model](statistical-model.md)** — the generative model, the two
  inference targets (fixed-effects vs. random-effects), the identifiability picture,
  and a dictionary bridging our terms to classical Bayes, pharmacometrics/NLME, and
  simulation-based inference. **Read this first.** Also typeset as a short chapter:
  [`statistical-model.tex`](statistical-model.tex) → [`statistical-model.pdf`](statistical-model.pdf).

## The chapters

The package is organized as the parts of that model:

| # | Chapter | What it covers | Guide | Package |
|---|---|---|---|---|
| 1 | **The statistical model** | generative spec, two targets, vocabulary | [statistical-model.md](statistical-model.md) | — |
| 2 | **Priors as a measurement model** | literature → informative joint prior $\pi$; translation sigma; copula; derived children | [submodel-inference-guide.md](submodel-inference-guide.md) | `priors/`, `submodel/` |
| 3 | **Flat inference** | intractable-likelihood posterior $p(\theta\mid x_{\text{obs}})$ via NPE; restriction; TSNPE; proposal↔prior reweight | [stage2-sbi-guide.md](stage2-sbi-guide.md) | `inference/` |
| 4 | **Population inference: the model** | hierarchical $(\mu,\omega)$; virtual patients as random-effects draws; two covariances ($\Gamma_\pi$ epistemic vs $\Gamma_\omega$ biological); eigenbasis = identified subspace; matched footing; discrepancy layer; variance budget. Also the full amortized estimator ($\tilde\pi$/TSNPE, reweight at the **hyper** level $h/\tilde h$), which is the **escalation path**, not the default | [population-inference-guide.md](population-inference-guide.md) | `vpop/`, `targets/` |
| 4b | **Population inference: diagnose first, then fit** | **start here.** What a low prevalence-weighting ESS actually means and how to give it a null; localizing structural misspecification (LOO-by-observable, dual variables, per-observable $\tau^2$); then the tractable fit: an explicit asymptotic summary likelihood over the anchor grid, sampled with NUTS, no proposal and no reweight | [population-inference-tractable.md](population-inference-tractable.md) | `vpop/`, `targets/` |
| 5 | **Model checking & calibration** | the Bayesian-workflow suite: SBC gate → prior-data conflict → reachability → joint discrepancy → LOO-PIT | [stage2-sbi-guide.md §Diagnostics](stage2-sbi-guide.md#diagnostics) | `inference/sbc.py`, `inference/diagnostics.py`, `vpop/diagnostics.py`, `audit/` |
| 6 | **Experimental design** | OBED — which measurement would identify a soft parameter | [stage2-sbi-guide.md §OBED](stage2-sbi-guide.md#optimal-bayesian-experimental-design-obed) | `inference/obed.py` |

## Slides & talks

- [Submodel inference slides](submodel-inference-slides.md)
- [Usage slides](qsp-inference-usage-slides.md)
- [ACOP 2026 abstract](acop-2026-abstract.md)

## Status of the docs spine

The model front-door (Ch. 1) and this index are the statistician-facing entry point.
Every chapter now has a guide.

**Population inference is split in two, and 4b is the route to take.** Ch. 4 states
the hierarchical design in full: the two covariances, the eigenbasis, the provenance
split, matched footing, the discrepancy layer, the variance budget, and the amortized
$\tilde\pi$/$\tilde h$ estimator with its hyper-level reweight and full-path SBC gate.
Ch. 4b keeps that model, leads with the misspecification diagnosis the fixed-cloud
fit's low ESS was asking for, and then fits it with an explicit summary likelihood
instead. The split is deliberate: the *model* is shared and is the part that matters;
the amortized *estimator* is heavier than the data can exercise and is kept as the
escalation path.

Both run ahead of the code on purpose. Ch. 4's *Open work* table is the ledger of that
gap, ordered by which items change $\hat\varphi$ rather than only weaken a guarantee;
Ch. 4b's *Build order* is the sequence to actually work through. Today `vpop/` holds
the eigenbasis, the proposal, and the prevalence-weighting fallback, and `targets/`
holds the omega prior and the observed anchors. `vpop/diagnostics.py` already carries
most of 4b's Part I (the prior-predictive ESS null at each target's real $n$, the
calibrated $D^2$ misspecification ratio, ESS scaling, the collinear-constraint check,
and greedy-core/conflict-ranking localization); the fitted-weight null, per-observable
$\tau^2$, and the dual-variable localizer are not. The summary likelihood, the
hyperprior, and the cohort construction are not built at all. The package namespace
converges on these chapter names as that machinery migrates in (Stage 3).

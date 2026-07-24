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
| 4 | **Population inference (VPops)** | hierarchical $(\mu,\omega)$; virtual patients as random-effects draws; eigenbasis = identified subspace; $\tilde\pi$/TSNPE/reweight loop; prevalence-weighting fallback | [population-inference-guide.md](population-inference-guide.md) | `vpop/` |
| 5 | **Model checking & calibration** | the Bayesian-workflow suite: SBC gate → prior-data conflict → reachability → joint discrepancy → LOO-PIT | [stage2-sbi-guide.md §Diagnostics](stage2-sbi-guide.md#diagnostics) | `inference/sbc.py`, `inference/diagnostics.py`, `vpop/diagnostics.py`, `audit/` |
| 6 | **Experimental design** | OBED — which measurement would identify a soft parameter | [stage2-sbi-guide.md §OBED](stage2-sbi-guide.md#optimal-bayesian-experimental-design-obed) | `inference/obed.py` |

## Slides & talks

- [Submodel inference slides](submodel-inference-slides.md)
- [Usage slides](qsp-inference-usage-slides.md)
- [ACOP 2026 abstract](acop-2026-abstract.md)

## Status of the docs spine

The model front-door (Ch. 1) and this index are the statistician-facing entry point.
All six chapters now have guides. **Chapter 4 (population inference)** states the
hierarchical $(\mu,\omega)$ design — the $\pi$/$\tilde\pi$/$(\mu,\omega)$ loop, the
eigenbasis, reweight-to-$\pi$, and the full-path SBC gate. Today `vpop/` holds the
eigenbasis and the prevalence-weighting fallback; the round loop itself still lives in
a downstream project's runner, and the package namespace converges on these chapter
names as that machinery migrates in (Stage 3).

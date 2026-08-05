ok, we are trying to model the joint distribution of the parameters \theta for a population of PDAC patients. A draw from \theta corresponds to a patient from this population.

we are going to assume that log \theta ~ N(\mu, \omega) - that is, that \theta is log-normally distributed, that \mu is a vector of length p (number of parameters), and \omega is a covariance matrix pxp. \mu and \omega also have priors:

log \mu ~ N(\mu0,\Sigma). mu0 and Sigma are derived from the submodel priors overlay - that is, \mu0 is given by the submodel priors yaml (stage 1 fitting) and converting that to a gaussian copula. \Sigma, same. wait I am not sure how mu0 and Sigma fall out of the copula. i know we fit the stage 1 posterior and then we basically convert that to a copula, somehow - right? like find the center and derive the correlation matrix. so if pi0 is the posterior draws from stage 1 (and this is the flat case, mind you, where we assume we are fitting to the average patient) - then mu0 = E(pi0) and Sigma = Cov(pi0) (estimated from the actual pi0 draws). is that right? and note that mu has its own dispersion Sigma that is separate from the omega dispersion. just fyi. is it correct to have those separated?

omega ~ .... I actually don't know we parameterize it. like the parametric family we are using for it. what is a common parametric faimly for covariance matrices? either way, probably a two parameter family, and we will likely do the same thing where we run the sumbodel priors overlay on some defaults from omega_priors.csv. where we are moving the submodel population spreads forward to form the basis of the population spreads when fitting against the clinical data. is this scientifically coherent? it is what we are doing in the flat, mu-only case (when we fit only mu against the median clinical data, and assume our only dispersion is in Sigma).

then, moving forward in the generative process: we simulate from the

---

# Reference: what the code implements

Everything below is read off the code, not the docstrings. Edit the prose above
against it. File references are `pdac-build/workflows/hierarchical_runner.py`
unless stated otherwise.

```

patient      log θᵢ = μ + W(σ_u ⊙ zᵢ)         z ~ N(0, I_P), FIXED across the fit
                                               (common random numbers)
                                               W fixed, from Γ; W = I and σ_u = ω
                                               in the non-reparameterized path

center       μⱼ ~ N(μ0ⱼ, τ_μ)                 τ_μ = 0.5, independent over j
                                               μ0 = E_π[log θ], composite prior

spread       σ_u,k ~ LogNormal(log ω_k, τ_ω)  τ_ω = 0.5
                                               ω from the 4-layer omega prior

observation  Q(φ) = quantiles of the patient cloud pushed through the emulator
             y_obs ~ N(Q(φ) + η + δ, Σ)        Σ FROZEN, not a function of φ
```

```
μ0           mean of 20k log draws from π      π = pdac_priors.csv (lognormals, 271)
                                                 + submodel_priors.yaml (65, + copula)
                                                 + derived_priors.yaml (last)
                                                 [+ vary_policy: rest -> σ=1e-3]
τ_μ          0.5, flat                          NOT center_sigma; that is computed
                                                 from the same draws and used only
                                                 for omega_rescale

ω            4 layers, strongest last:
  1  global_default  0.35                       263 params
  2  role  0.15 packing / 0.10 material         8 params, omega_priors.csv
  3  explicit                                   0 params
  4  submodel population block, shrunk by n     ≤9 params; crossover at n≈3
                                                 gated on n_biological from a
                                                 target observed_distribution;
                                                 only 8 of 159 targets declare one
                                                 (block covers 178, but 145 of them
                                                 just re-emit the center sigma)
                                                 passed by hierarchical_runner,
                                                 NOT by nn_emulator/emulator

σ_u          prior centered on 1, not on ω      W Wᵀ = Γ, so σ_u=1 reproduces Γ
             ω enters only through Γ:            omega_rescale = ω / center_sigma
             Γ = Cov(π draws · omega_rescale)    diagonal rescale => Corr(Γ) = Corr(π)
                                                 exactly: correlations are stage-1
                                                 epistemic, ω never touches them
```

`φ = (μ, log σ_u, η_raw, δ_raw, log τ_η, log τ_δ)`. The offsets are non-centered:
`η = τ_η · η_raw` (between-study), `δ = τ_δ · δ_raw` (per-observable structural
discrepancy).


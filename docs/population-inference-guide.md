# Population inference (VPops) — the hierarchical $(\mu,\omega)$ path

> **Chapter 4 of the [docs spine](README.md).** Read [the statistical
> model](statistical-model.md) first — this guide assumes its vocabulary: the
> anchored prior $\pi$, the proposal $\tilde\pi$, the population
> $F(\theta\mid\varphi)$ with $\varphi=(\mu,\omega)$, and the self-normalized
> reweight $w\propto\pi/\tilde\pi$.

Flat inference (Ch. 3) gives the single-unit posterior $p(\theta\mid x_{\text{obs}})$
— the typical patient. A **virtual population** is the random-effects version:
patients are draws $\theta_i \sim F(\theta\mid\varphi)$, the data are *cohort*
summaries (median + spread), and the answer is the fitted distribution
$F(\theta\mid\hat\varphi)$, not a point. This chapter is the machinery that infers
$\varphi$ and generates the population — and it is the same three-object design as
flat inference, with $(\mu,\omega)$ inferred on top.

## The three objects

Everything here is one generative model observed at the cohort level, and it
separates a **scientific** distribution from a **computational** one:

- **$\pi$ — the anchored prior / population base.** Built from submodel targets,
  derived priors, and declared assumptions (Ch. 2). It sets the population
  geometry $\Gamma_{\text{pop}} = \operatorname{Cov}_\pi[\log\theta]$ and the
  center $\mu$. **Fixed; never touched inside the loop.** This is the only surface
  where hand-work is legitimate.
- **$\tilde\pi$ — the proposal.** Wide on the *identified* directions only
  (widening a degenerate direction buys nothing but ESS cost), then **truncated
  toward the reachable set across rounds** (TSNPE). Reference simulations and
  cohort patients are drawn from $\tilde\pi$. It is a computational device and
  drops out of the reported answer.
- **$(\mu,\omega)$ — the population, inferred not set.** The width is inferred
  where the data carry signal and falls back to $\pi$'s anchored $\sigma$ where it
  does not. Inferred in the **identified subspace** (the eigenbasis below), not
  per-parameter — only a handful of combination directions are identifiable, so
  inferring two hundred spreads would fit noise.

They reinforce each other. Decoupling $\tilde\pi$ from $\pi$ *alone* is ESS-costly
— reweighting a wide $\tilde\pi$ to a narrow $\pi$ blows up on degenerate
directions. **TSNPE truncation is what makes the decoupling cheap**: after a few
rounds $\tilde\pi$ has shrunk to $\approx$ the reachable set, so $w=\pi/\tilde\pi$
is evaluated over a concentrated region and the ESS is fine. The weight is a
one-liner — `log_importance_weights` in log-$\theta$ space — and self-normalized,
so a truncated proposal's normalizer cancels exactly (`importance.py`).

## Why an eigenbasis for $\omega$

Structural non-identifiability means only $\approx$ a dozen *combination*
directions of the population spread recover from cohort summaries; the rest are
sloppy and must ride the prior. So $\omega$ is inferred as $\sigma_u$ along the top
$K$ directions of a basis, not as $P$ per-parameter spreads:

$$\log\theta \sim \mathcal N\!\big(\mu,\; V \operatorname{diag}(\sigma_u^2) V^\top\big),
\qquad \theta = \exp\!\big(\mu + V(\sigma_u \odot z)\big),\ z\sim\mathcal N(0,I).$$

The basis is the **active subspace under $\pi$** (`vpop/eigenbasis.py`): the data
Fisher $G = \sum_i J_i^\top \Sigma_{\text{obs}}^{-1} J_i$ (pool-derived,
prior-independent) whitened by the population metric $\Gamma_{\text{pop}}$, so
directions are ranked by *data information per unit of population variance*. With
$\sigma_u=1$ on every direction the draw reproduces $\Gamma_{\text{pop}}$ exactly:
$\pi$ sets the geometry, and the inferred top-$K$ deviate only where the data pull.
The sloppy complement is marginalized over its prior spread rather than pinned, so
cohorts keep full correlated spread. **$\Gamma_{\text{pop}}$ is always computed
from $\pi$, never from the wide $\tilde\pi$** — the proposal must not set the
population geometry.

## Center vs spread: which targets feed $\omega$

Not every target constrains the population spread, and the split matters for both
$\omega$ and the reachability envelope. It follows the **source** of a target's
observed distribution:

- **`across_patient` targets** carry a real between-patient distribution (population
  `samples`). They feed **both** $\mu$ (their median) and $\omega$ (their spread
  anchors), and they are the only targets that define the reachability envelope.
- **`center_only` targets** report a median and a 95% CI but no across-patient
  samples. The CI is *center* (measurement) uncertainty that shrinks with $\sqrt n$,
  **not** between-patient spread — so a `center_only` target feeds **$\mu$ only**
  (its median). Its spread anchors are dropped and it is absent from the
  reachability envelope. Letting a CI masquerade as population spread silently
  mis-disperses $\omega$; excluding it is the same call the fixed-cloud VPop
  (`vpop/weighting.py`) already makes.

The `ObservedAnchors.feeds_spread` flag (`targets/anchors.py`) is the seam: the
spread half of the conditioning vector and the reachability envelope are built only
from anchors where it is set; the median (center) half is built from all of them.

## The loop

One amortized NPE over cohorts, wrapped in TSNPE rounds that **re-simulate** into
the shrinking reachable set — a single wide reference batch would spread the
pool-bound emulator too thin to gate on, so each round re-sims where the patients
are. Everything is fit inline from the current round's reference batch — no
precomputed artifacts.

For round $r = 0,\dots,R$:

1. **Reference simulations.** Draw $\theta$ from $\tilde\pi_r$ (restriction-filtered,
   so non-viable $\theta$ never burn a sim), simulate through the real forward
   model. The sim cache serves repeats.
2. **Fit from the batch, inline.** A $\theta\to\text{obs}$ **emulator** (with a
   per-observable approximation-error budget), the Fisher $G$ (its local
   Jacobian), and the basis $V = \texttt{prior\_metric\_eigenbasis}(G,
   \Gamma_{\text{pop}})$.
3. **Cohorts.** Draw from the hyper-proposal (wide $\sigma_u$ on the identified
   directions), **emulate** them — free, so cohort *count* is not a cost — and
   summarize each target at its real published sample size. Train the amortized
   $(\mu,\log\sigma_u)$ NPE on the cohort summaries.
4. **Truncate.** Estimate the reachable set and restrict $\tilde\pi_{r+1}$ to it.
5. **Stop** when the truncation stabilizes (the reachable set stops shrinking, and
   the ESS of $w=\pi/\tilde\pi$ recovers — a *compute* symptom, logged not gated).

The expensive step is the real forward model in (1); the emulator exists so that
per-cohort resimulation in (3) is affordable. It is **pool-bound**, so its validity
domain has to track $\tilde\pi$ — which is why it is refit each round rather than
built once. Its approximation error flows into the VPC null (Ch. 5), so a spread
miss must exceed the surrogate error to be labeled.

### Reporting under $\pi$

The reported population is **under $\pi$**, not $\tilde\pi$. Draw virtual-patient
candidates from the fitted posterior at the $\tilde\pi_R$ geometry, weight
$w = \exp\!\big(\pi.\text{log\_prob}(\theta) - \tilde\pi.\text{log\_prob}(\theta)\big)$
(`reweight_to_prior`, self-normalized), and resample. The ESS is logged as a
compute symptom — under prior–data conflict it collapses, but that is a *downstream
symptom of a conflict already detected in observable space* (Ch. 5), never a
diagnostic read here.

## Validating the shipped path

The gate is **SBC on synthetic truth over the full path** — not "train and infer".
Draw $\varphi^\star$ from the hyperprior (the $\pi$ reporting geometry), synthesize
an observed cohort from it, run the **whole loop** (train on $\tilde\pi$, truncate,
reweight to $\pi$), and rank $\varphi^\star$ in the reported posterior. This uses
the **real** importance weights at the calibration step (weighted SBC + joint TARP,
`inference/sbc.py`). Ranking $\varphi^\star\sim\pi$ inside importance-weighted draws
is the only check that blesses train-on-$\tilde\pi$-report-under-$\pi$; ranking in
*unweighted* draws would bless a pipeline we do not run.

Acceptance:
1. **Calibrated** — flat rank ECDFs for the inferred direction spreads (weighted
   SBC gate passes; TARP coverage near diagonal).
2. **Recovery** of the top-$K$ direction spreads matches the offline
   identifiability ceiling; a large gap is a wiring bug.
3. **Sloppy directions ride the prior** — their posterior $\sigma_u \approx$ prior,
   no spurious contraction.
4. **VP correlation structure** shows the expected mechanistic pairings; random
   pairs $\approx 0$.

## Where the code lives

The generic machinery is this package; a downstream project supplies only its
prior, scenarios, and forward model as a thin caller.

| Piece | Home |
|---|---|
| eigenbasis (Fisher $G$, prior-metric whitening, draw/project matrices) | `vpop/eigenbasis.py` |
| proposal $\tilde\pi$ (log-normal population, widen-on-identified, reachability accept-fn) | `vpop/proposal.py` |
| importance reweight ($w=\pi/\tilde\pi$, ESS, weighted quantiles) | `inference/importance.py` |
| TSNPE truncation substrate (density thresholder → restricted proposal) | `inference/` |
| weighted SBC + TARP gate | `inference/sbc.py` |
| VPC null with emulator-error inflation | `inference/predictive_checks.py` |
| prevalence-weighting fallback (fixed-cloud VPop, Allen 2016) | `vpop/weighting.py` |

The prevalence-weighting construction in `vpop/weighting.py` is the older
fixed-cloud alternative: reweight an existing simulation cloud to the observed
marginals instead of inferring the width. Its ESS ceiling is the known wall; it is
kept as a fallback for when the hierarchical NPE underperforms, not as the primary
loop.

## Reachable-set truncation

This is the step-4 truncation, and it is **load-bearing, not a diagnostic** — it
decides where the next round's real simulations go, hence the region the emulator is
trained to be accurate on. (Distinct from the *labeler's* reachability in Ch. 5,
"could any $\theta$ reach $x_{\text{obs}}$?", which is a pure diagnostic and does not
touch the fit.)

"Reachable set" for a VPop has no single $x_{\text{obs}}$ to threshold against — the
target is a distribution — so the truncation is defined in **observable space
against the data**: keep $\theta$ whose per-patient (emulated) observables land
inside the observed envelope. Concretely, the same `RestrictedPrior` substrate as
flat TSNPE, but with the density thresholder swapped for an accept-fn that asks
*"are this $\theta$'s emulated observables inside the observed envelope?"*
(`vpop/proposal.py:reachable_accept_fn`) — free, since the round's emulator is
already in hand.

This is chosen over the alternative — truncating $\tilde\pi$ to the high-density
region of the *fitted* $F(\theta\mid\hat\varphi)$ — for three reasons, all sharper
because the inferred quantity here is a **spread**:

1. **No feedback into the estimate.** Truncating on the fitted population couples
   the proposal to the very $\sigma_u$ we infer: an under-estimate in round $r$
   narrows $\tilde\pi$, tightens round $r{+}1$'s cohorts, and self-fulfils. A
   data-anchored envelope has no such loop.
2. **Right-sized emulator domain.** Cohort patients are drawn from the *wide*
   hyper-proposal, broader than the fitted $F$; a fit-truncated $\tilde\pi$ would
   leave the emulator extrapolating on exactly the tail patients that carry the
   spread signal.
3. **Matches the reachability definition.** The labeler defines reachability in
   observable space ("could a physical $\theta$ produce $x_{\text{obs}}$?"), not by
   a parameter-space density.

The envelope comes from the `across_patient` targets' observed `samples` — the
*outer band* of that between-patient distribution (a generous quantile band, not the
median±IQR core, and never the CI95, which is center uncertainty). `center_only`
targets have no observed population distribution, so they contribute no envelope
(the center-vs-spread split above). The one requirement: keep the band **generous**,
or the population's legitimate tail patients get cut and $\omega$ is reported too
narrow — the failure this whole path exists to avoid.

Honest caveat: the "support" of a finite sample is not a clean object — the observed
min/max grows with $n$, and a low-$n$ target ($n=6$) barely defines a band. A fixed
generous quantile band is the pragmatic proxy; the truncation it drives is
correspondingly soft, which is why it only *shrinks* $\tilde\pi$ toward the data and
never sets the population geometry (that is $\pi$'s job, always).

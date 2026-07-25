# Population inference: diagnose first, then fit

> **Chapter 4b of the [docs spine](README.md).** The sibling of
> [Chapter 4](population-inference-guide.md), which specifies the population model in
> full and the amortized estimator that goes with it. This chapter keeps Ch. 4's
> *model* and replaces its *estimator*, and it puts a diagnostic stage in front of
> both. Read Ch. 4 for the model: the two covariances $\Gamma_\pi$ and
> $\Gamma_\omega$, the provenance split, matched footing, the discrepancy layer, and
> the variance budget. Those are unchanged here and are the part that matters. Read
> this chapter for what to do first, and for the smaller machine that fits the model
> once you know what you are fitting.

## Why this document exists

The fixed-cloud VPop (`vpop/weighting.py`) came back with a very low effective
sample size. That observation is what motivated the hierarchical path in Ch. 4: the
ESS was read as the fixed-cloud method's known wall, and the response was to build an
estimator that does not have that wall.

That reading skipped a step. A low ESS in prevalence weighting is not a property of
the method, it is a **measurement of how hard the tilt had to work to reconcile the
model's simulated cloud with the observed data**, and there are at least five
distinct reasons it can be small. Only one of them is structural misspecification,
and if that is the one, then **no estimator fixes it.** A more sophisticated fit on a
misspecified mechanism produces a more confident wrong answer, not a better one.

So the order is: find out which of the five it was, fix that, and only then decide
how much estimator the problem deserves. That is the whole argument of this chapter,
and Part II is the observation that once the diagnosis is done, the fit the data can
support is much smaller than Ch. 4.

### Two things called ESS, and they mean different things

This is the confusion that turned a symptom into a research program, so it is worth
separating up front.

| | Ch. 4's ESS | the fixed-cloud ESS |
|---|---|---|
| what is being weighted | $\varphi$-draws, by $h/\tilde h$ | cloud patients, by the fitted prevalence tilt |
| what the weight is doing | changing measure from a computational proposal to the reporting prior | moving the model's population onto the observed marginals |
| does the data enter the weight? | no | yes, the weight *is* the fit |
| what low means | the proposal and the prior disagree about where mass sits, a statement about compute | the tilt had to work hard, which is partly compute and partly **fit** |

Ch. 4 is right that *its* ESS is not a conflict diagnostic. That argument does not
transfer. The prevalence weight is fitted to the data, so its degeneracy carries
information about the data. Reading across from one to the other is what licensed
treating a low fixed-cloud ESS as a nuisance to engineer around.

## Part I. What a low ESS actually means

### The five causes, in triage order

Check them in this order, because the cheap ones are also the most likely and they
mask the interesting one.

**1. `n_obs` was not supplied.** Without it `fit_prevalence_weights` matches the
observed bin proportions *exactly*, which treats a 6-patient proportion as truth and
spends effective sample size reproducing its sampling noise. The module's own
docstring gives the magnitude: with 46 observables at $n=11$, exact matching burns
about 97% of the cloud. If the low-ESS run did not pass `n_obs`, the ESS is fully
explained and there is nothing else to diagnose. This is a one-line check and it goes
first.

**2. The constraint count.** With $C = n_{\text{obs}} \times n_{\text{bins}}$ dual
variables, exponential tilting loses effective sample size even under a perfectly
specified model, because each constraint is estimated with noise and the tilt chases
all of them. 46 observables at 5 bins is 230 constraints. There is no threshold that
makes this interpretable in the abstract, which is the argument for the null
distribution below: it prices in $C$ automatically.

**3. Support deficiency.** The cloud does not reach the data. `VPopResult` already
reports this (`support_deficient`, any bin with fewer than `MIN_SUPPORT` cloud
members). Reweighting cannot conjure a patient the cloud does not contain, so the
weights strain toward a corner. This is a *reachability* failure, and it splits
further: a too-narrow cloud generator (fixable) versus a mechanism that cannot
produce the observation at any admissible $\theta$ (not fixable by widening).
Ch. 5's reachability labeler is what separates those.

**4. Contradictory constraints from tied bin edges.** The continuity correction in
`build_quantile_constraints` documents this failure mode precisely: two observables
the model treats as the same variable receive incompatible targets at the same
threshold, the dual has no solution, weights blow up, and the ESS collapses **while
the TV distance still looks perfect**. It reads exactly like a joint conflict and is
not one. The correction is in the code; confirm the low-ESS run had it.

**5. Genuine structural misspecification.** What is left after the first four. This
is the interesting case and the rest of Part I is about characterizing it rather than
just detecting it.

### The constructive reframe: ESS is a joint-compatibility statistic

Here is the reason the ESS is worth this much attention rather than being engineered
around.

Prevalence weighting constrains *marginals*, one set of quantile bins per observable.
But it does so with a **single weight vector over patients**. One vector has to
satisfy every observable's marginal simultaneously, and that is only possible if the
cloud's joint structure, its copula, is compatible with the whole set of observed
marginals at once. When it is not, there is no weighting that satisfies everything,
the ridge keeps the fit finite, and the ESS collapses.

So the fixed-cloud ESS is the population-level analogue of Ch. 5's joint discrepancy
check: *every target individually reachable, but no single $\theta$ reconciles all of
them at once*, one level up, as *every marginal individually matchable, but no single
reweighting matches all of them at once*. That is a genuinely informative statistic,
and it points where a per-observable check cannot: at the **coupling** between
observables, which is exactly where mechanism errors live.

It is also the statistic the hierarchical path is worst at surfacing. A parametric
$F(\theta\mid\varphi)$ fitted by NPE will happily return a $\hat\varphi$ that
reconciles nothing well, with no degeneracy signal at all, because the estimator
always returns something. Ch. 4 has to reintroduce that information deliberately, as
$\hat\tau_\delta$ and the $x$-space check. The fixed cloud gives it away for free.

### Give the ESS a null

The reason the low ESS could not be acted on is that it is a raw number with no
reference. Fix that first; it is cheap and it is the same idiom the rest of the
package already uses (`sbi_self_reference_null`, the VPC null, the prior-predictive
tail probability), which is that a diagnostic earns its place by carrying a null
computed from the model's own simulations.

**Recipe.** For $b = 1,\dots,B$:

1. Pick a reference population on the cloud (see below) and draw a synthetic cohort
   of size $n_j$ for each observable $j$ from it, sampling cloud members and reading
   off their simulated observables.
2. Run `build_quantile_constraints` and `fit_prevalence_weights` on that synthetic
   cohort with the **same** `n_bins`, `ridge`, and `n_obs` as the real fit.
3. Record $\text{ESS}_b$ and $\tau^2_b$.

Then $p = \#\{b : \text{ESS}_b \le \text{ESS}_{\text{obs}}\}/B$, and the same for
$\tau^2$ in the other tail. Now "our ESS was super low" becomes "our ESS sat at the
0.2nd percentile of the well-specified null", which is a statement with a
false-positive rate.

**Two reference populations, two questions.**

- **Unweighted cloud** (a prior-predictive null). Asks: is the observed data
  consistent with the population the cloud generator already implies? A small $p$
  here means prior-data conflict at the population level, which may be a mis-centered
  prior rather than a broken mechanism.
- **Fitted-weight cloud** (a posterior-predictive null). Asks: after the best
  population the cloud can express, is the *residual* explainable by sampling noise?
  This reuses the data, so it is conservative and $p$ is biased large. That
  conservatism is what makes it useful: **a small $p$ from the fitted-weight null is
  strong evidence of structural misspecification**, because the null had every
  opportunity to absorb the misfit and could not.

Run both. Their disagreement is itself informative: prior-predictive small with
posterior-predictive large means re-anchor, both small means mechanism.

This null costs one convex solve per replicate on a cloud you have already simulated.
It should have existed before Ch. 4 did.

### Localize it

A scalar $p$-value says the misfit is real. It does not say where. Three localizers,
all cheap, all reading off objects the fit already produces.

**Leave-one-observable-out.** Drop observable $j$'s bins, refit, record
$\text{ESS}_{-j}$. Rank by recovery. This is the population analogue of
`sbi_loo_predictive_check`.

There is a trap here worth naming: removing any observable removes $n_{\text{bins}}$
constraints and therefore raises the ESS mechanically, so a raw $\text{ESS}_{-j}$ is
not interpretable. Compare $\text{ESS}_{-j}$ to the **median over $j$** of
$\text{ESS}_{-j}$, or to the null of removing a random block of the same size. A
single observable whose removal recovers the ESS far beyond that reference is a
localized misfit and you go read that readout's trace. If no single removal helps,
the failure is diffuse and joint, which is the case §*The constructive reframe*
predicts and the one that actually indicts the coupling.

If singles do not explain it, run leave-pair-out on the top handful. Pairs are where
coupling errors show up, and $\binom{10}{2}$ convex solves is seconds.

**The dual variables.** $\lambda$ is already fitted and the constraint columns are
indicators, so $|\lambda|$ is comparable across bins. Report the top bins by
$|\lambda|$: those are the constraints doing the work, i.e. where the tilt is
straining. This is free and it is more specific than a per-observable TV distance,
because it names the *bin*, so it distinguishes "the model's upper tail is wrong"
from "the model's center is wrong".

**Per-observable $\tau^2$.** The current `tau2` is a single method-of-moments number
over all residuals. Compute it per observable from that observable's own bins and
rank. Aggregate $\tau^2$ says how much misfit there is; per-observable $\tau^2$ says
whose.

### Name the direction of the misfit

Detection and localization still do not say what to change. This last step does, and
it is the one that maps onto an action. For each flagged observable, compare the
**unweighted** cloud's simulated distribution to the observed:

| what is off | reading | response |
|---|---|---|
| center shifted, spread comparable | the prior is aimed wrong, mechanism plausibly fine | re-anchor $\mu^\pi$ (Ch. 2), or absorb as $\eta_s$/$\delta_j$ if it is study-specific |
| center fine, cloud too narrow | the omega layer is too tight for that parameter class | revisit `targets/omega.py` levels 1 to 3 for the parameters driving that readout |
| center fine, cloud too wide | the omega layer is too loose, and the report is over-dispersed | same, other direction; this is the failure a generous cloud generates by construction |
| shape wrong in a way no shift or scale fixes | the mechanism or the population family is wrong | mechanistic trace; no population parameter will fix it |
| observed outside cloud support entirely | reachability | Ch. 5's labeler: is it re-anchoring or is it mechanism? |

Only the last two rows are structural misspecification in the sense that motivated
Ch. 4. The middle three are prior work, which is where Ch. 1 says the human belongs.

### The triage, end to end

1. Confirm `n_obs` was passed and the tie correction is in. If not, rerun; you may be
   done.
2. Check `support_deficient`. If any, that is reachability, not conflict, and it goes
   to Ch. 5's labeler before anything else.
3. Compute the ESS and $\tau^2$ null, both references. If $p$ is unremarkable, the
   low ESS was the constraint count and the population fit is fine.
4. If $p$ is small: localize (LOO, $|\lambda|$, per-observable $\tau^2$).
5. For each flagged readout, name the direction and take the corresponding action.
6. Refit. The ESS after step 5 is the measure of whether the diagnosis was right.

Steps 1 to 5 cost a day and reuse simulations you already have. That is the
comparison to keep in mind against building `vpop/hyperprior.py` and
`vpop/cohorts.py`.

## Part II. Then fit

Once you know what is being fitted, the estimator the data can support is much
smaller than Ch. 4's. The reason is a counting argument: with $K$ bounded above by
the number of spread-feeding observables, the inference target is on the order of a
few spread numbers, a comparable number of center components, and two discrepancy
scales, from perhaps fifteen to twenty-five noisy summary statistics.

### The summary likelihood is available in closed form

The observed data are quantile anchors $\hat Q_j(p)$ at each target's real $n_j$
(`targets/anchors.py`). The asymptotic joint sampling law of sample quantiles is
standard. State it in its **general** form, because the cross-target case below is
the same formula and an earlier draft of this chapter wrote down only its diagonal.
For observables $j,l$ summarized over the **same** $n$ patients,

$$\operatorname{Cov}\!\big(\hat Q_j(p_a),\ \hat Q_l(p_b)\big) \;=\;
\frac{C_{jl}(p_a,\,p_b)\;-\;p_a\,p_b}
     {n\ f_j\!\big(Q_j(p_a)\big)\ f_l\!\big(Q_l(p_b)\big)},$$

where $C_{jl}$ is the **copula** of $(X_j, X_l)$ under the population predictive.
Setting $j=l$ gives $C_{jj}(p_a,p_b)=\min(p_a,p_b)$ and recovers the familiar
within-observable form $[\Sigma_Q]_{ab} = p_a(1-p_b)/(f f)$ for $p_a\le p_b$. So one
formula covers the whole conditioning vector, and the only new object it needs is the
cloud's rank-dependence structure, which the cloud already is a joint sample of.

Two things to notice about the diagonal case. It handles the whole anchor grid
**jointly**, including the median/IQR covariance, which is exactly the
$\mu$-versus-$\sigma_u$ confounding Ch. 4 has to diagnose separately at small $n$.
And it reproduces both constants that chapter already relies on:
$\operatorname{SE}(\hat Q_{.5}) = 1.2533\,\sigma/\sqrt n$ and
$\operatorname{SE}(\widehat{\text{IQR}}) = 1.5730\,\sigma/\sqrt n$ under a Gaussian.
The whitening in `whiten_sensitivity_rows` and this likelihood are the same formula.
Ch. 4 uses it to rank directions and then declines to use it to fit, which is an
inconsistent position on how much approximation is acceptable.

#### It was checked, and it holds only in a particular configuration

"Asymptotic" is the whole question when $n$ is 6. Measured on a 2000-patient PDAC
cloud at each target's real published $n$: draw cohorts, summarize them, and compare
the true sampling spread of the anchor vector against the covariance above, as the
Mahalanobis $D^2/k$ (1.00 is exact) and the coverage of the implied ellipsoids.

| $n$ | quartiles | $(.2,.5,.8)$ | deciles |
|---|---|---|---|
| 6-8 | 0.976 | 0.813 | 0.559 |
| 9-12 | 0.949 | 0.963 | 0.751 |
| 21-60 | 0.993 | 0.945 | 0.936 |
| >60 | 1.015 | 0.965 | 0.999 |

The law holds down to $n=6$, but only **in log space** (on the raw scale the same
check is off by factors of hundreds at small $n$, since these are positive skewed
quantities) and only at **three anchors** for the small-$n$ targets. Nine deciles from
eleven patients is not a Gaussian vector, at any bandwidth. That is why
`_resolvable_grid` now enforces $n \ge 2k$ on top of per-anchor resolvability, falling
back to a symmetric triple rather than to the median, so a small-$n$ target keeps its
spread anchors. It also settles §*When to escalate* case 1 in this chapter's favour.

**A defaults bug the check turned up, which is not specific to this chapter.** NumPy's
default quantile estimator (`method='linear'`) shrinks the sample IQR by 21% at $n=6$
and 15% at $n=9$. This chapter's mean model is the *population* quantile $Q_j(\varphi)$,
so it inherits that shrinkage directly as a **low** bias on $\hat\sigma_u$ of 15 to 25%
at $n=6$ to $12$, which is the failure Ch. 4's matched footing exists to prevent,
arriving through a different door. `targets/anchors.py:QUANTILE_METHOD` pins
`normal_unbiased` for the observed anchor and the cohort summary alike (residual bias
about 1%). Ch. 4 is only partly protected: the shrinkage cancels when both sides are
summarized by that module, but not for a target on the `observed_distribution` branch,
whose quantiles came from the reporting paper's own software.

**The density plug-in is the weakest part, and weaker than the table above suggests.**
$f_j$ is estimated from the simulated population predictive by quantile spacing, and
the bandwidth matters: $D^2/k$ for $n\le20$ ran from 0.95 at $h=0.02$ to 0.76 at
$h=0.15$. The table was measured against a smoothed population (the pool's
piecewise-linear quantile function), which flatters the estimator. Measured against the
raw discrete cloud instead, the same $h=0.035$ makes the asserted *variance* about 18%
too large on two heavy-tailed fold-change marginals (`cd8_fc`, `cd8gzmb_fc`) while
staying within 2% on the well-behaved ones. So the bandwidth is not one global number
to declare once; it wants a per-observable rule and a per-observable check. And $f_j$ is
re-estimated at every $\varphi$, so it is a moving plug-in whose smoothness in $\varphi$
the sampler depends on.

Note which way this cuts. An error in $f_j$ scales the whole covariance for that
observable, so it acts as a *weight* on how much that target counts, not as a bias on
$\hat\varphi$ through the mean. It is a precision-misallocation problem, not a
correctness one, which is why it ranks below the mean-model issues above.

#### Targets that share patients share sampling noise

This is the correction the cross-target block exists for, and it is not optional in
this dataset: **33 of 49 PDAC observables sit in groups sharing a scenario and an $n$**,
the largest being ten observables at $n=10$ from one GVAX+nivo arm and six at $n=9$
from the GVAX arm. Those are one assay panel on one set of biopsies.

Their sampling errors are correlated: a cohort that happened to draw high-CD8 patients
reads high across every CD8 readout at once. A likelihood that factorizes over targets
counts correlated readouts as independent observations, which is the overconfident
direction.

**Measured, and it is smaller and more concentrated than that argument suggests.**
Drawing cohorts as *rows* of the cloud (one set of $n$ patients, every one of a study's
observables read off them) and comparing the measured study covariance against the
formula:

| study | $J$ | $n$ | median $\lvert\rho\rvert$ | $D^2/k$ copula | $D^2/k$ independent | cov90 independent |
|---|---|---|---|---|---|---|
| GVAX+nivo | 10 | 10 | 0.21 | 0.871 | 0.945 | 0.860 |
| GVAX | 6 | 9 | 0.25 | 0.894 | 0.937 | 0.878 |
| baseline | 7 | 113 | 0.08 | 1.010 | 1.005 | 0.873 |

The formula is right: across-observable covariance entries reproduce to within a few
percent (median measured/formula ratio 1.0 over all eight multi-target studies). But
the *penalty for ignoring it* is modest, because **the model's own cross-observable
dependence is weak**: median $\lvert\rho\rvert = 0.21$ across the ten GVAX+nivo
readouts, giving an anchor-vector correlation of only 0.14 against 0.63 within an
observable. Independence costs about four points of coverage, not an order of
magnitude.

The dependence is **concentrated in a few pairs** rather than spread across the cohort:
in that same group $\max\lvert\rho\rvert = 0.992$, and the two-target studies
`(cd8_pct, cd8gzmb_pct)` at $\rho = 0.990$ and `(cd8_fc, cd8gzmb_fc)` at $\rho = 0.994$
carry anchor correlations of 0.63 and 0.67.

**Those correlations are not a model degeneracy, and they are not a fixed number.** The
tempting reading, which `vpop/diagnostics.py:duplicate_observables` invites at its 0.99
threshold, is that the model cannot tell the two readouts apart. Checked, it can:
GZMB+ CD8 cells are a subset of CD8 cells, and the model's per-patient GZMB *fraction*
varies with log-sd 0.31 to 0.41. It is simply small next to the level, which varies with
log-sd 2.4 to 2.6. Writing $\log X_{\text{gzmb}} = \log X_{\text{cd8}} + \log r$ with the
two roughly independent gives

$$\rho \;=\; \big(1 + (\sigma_r/\sigma_L)^2\big)^{-1/2},$$

which is $0.99$ at $\sigma_L = 2.55$ and reproduces the measurement. So $\rho$ is
manufactured by the width of the population it is evaluated on, and **the cloud is a
deliberately generous proposal, not the fitted population**: its log-sd for
`cd8_pct_baseline` is 2.55 against 1.03 in the published patient values. At the observed
width the same decomposition gives $\rho \approx 0.93$. That is the right way round for
the implementation, since $C_{jl}$ is evaluated at the current $\varphi$ and so is
computed on the fitted population rather than on the proposal, but it means the numbers
in the table above are an upper bound on how much the block will matter in the fit.

The screen is still worth running, and it does find one real duplicate:
`treg_fraction_cd4` and `treg_fraction_cd4_hiraoka2006` are byte-identical columns in
the cloud, the same simulated quantity emitted under two target names. That is the case
the diagnostic is for. The lesson for the diagnostic itself is that a Spearman screen on
*levels* conflates "the same variable" with "the same scale, different fraction"; a
subset-versus-total pair fires it without any model defect.

**A provenance limit worth knowing before relying on any of this.** The per-patient
pairing is not recoverable from the targets: all 21 rows of
`vpop_marginal_targets.csv` carrying a `values` list have it stored **sorted**, so those
are marginal samples with the patient correspondence discarded. The covariance block is
unaffected, since it needs only the *model's* copula. What is not answerable from the
current target data is the validation question, whether the model's cross-observable
dependence is right. Recovering that needs the paired per-patient records from the
source, not the digitized marginals.

**$\eta_s$ does not cover this, and pretending it does corrupts a misspecification
statistic.** The study offset is a shared systematic *shift* with an inferred scale;
this is shared *sampling noise*. Omit the block and $\hat\tau_\eta$ inflates to absorb
the correlation, which then reads as between-study heterogeneity when it is finite-$n$
noise on a shared cohort. Ch. 4 gets this free (one cohort per `cohort_id`, all its
targets summarized together, so the correlation is in the training data by
construction); here it has to be written down. It is the second place where departing
from matched cohort summarization is the liability, the IQR shrinkage above being the
first.

Writing it down is cheap. $C_{jl}$ is estimated from the same simulated population
predictive that supplies $Q_j$ and $f_j$, since each simulated patient carries a full
observable vector. The empirical copula is a step function of $\varphi$, so for a
covariance plug-in prefer the Gaussian-copula approximation
$C_{jl}(p_a,p_b) = \Phi_2\!\big(z_{p_a}, z_{p_b};\, \rho_{jl}\big)$ with $\rho_{jl}$ the
normal-score correlation of the cloud, which is smooth, cheap, and reuses
`inference/gaussian_copula_transform.py`. It is an approximation to a nuisance object
in the covariance, not to the mean, so its error does not move $\hat\varphi$ the way an
error in $Q_j(\varphi)$ would.

#### The model

Stack per study rather than per target. For each distinct `cohort_id` $s$, let
$\hat Q_s$ concatenate the anchors of every target measured on that cohort:

$$\hat Q_s \ \sim\ \mathcal N\!\left(\,Q_s(\varphi) + \delta_s + \eta_s\mathbf 1,\ \
\tfrac{1}{n_s}\Sigma_{Q,s}(\varphi) \;+\; \operatorname{diag}(\sigma_{c}^2)
\;+\; \operatorname{diag}(\varepsilon^2) \,\right),$$

with $\Sigma_{Q,s}$ the full block from the copula formula above, $Q_s(\varphi)$ the
population predictive quantiles obtained by pushing patients from
$F_{\mathcal V}(\theta\mid\varphi)$ through the emulator (with per-patient assay noise
convolved in, matched-footing term 3), $\sigma_{c,j}$ the target's center uncertainty
(term 1), $\varepsilon_j$ the emulator's held-out residual SD (term 2), and
$\delta_j,\eta_s$ the discrepancy terms. Studies are independent of each other, so the
log-density is a sum over $s$. A target with no cohort mate is the $1\times1$ case and
reduces to the per-target form this section used to state.

Note which terms stay diagonal and why. $\sigma_c$ is per-target center uncertainty
from that target's own reported CI, and $\varepsilon$ is the emulator's per-observable
residual: neither is a property of the patient sample, so neither picks up the shared
cohort. Only $\Sigma_{Q,s}$, which *is* finite-sample noise on shared patients, has
off-diagonal structure.

This is an ordinary hierarchical model with an explicit log-density. Sample
$\varphi$ from the posterior under $h$ directly with NUTS. It is structurally the
same object as the joint hierarchical fit `submodel/inference.py` already runs one
level down, and the honest name for it is **model-based meta-analysis on a
mechanistic forward model**: aggregate published summaries, per-study $n$,
between-study heterogeneity, a relevance discount. That literature is large and
boring, which is a feature.

**Provenance this needs that the targets do not yet carry.** `cohort_id` is the seam
(Ch. 4 §*Center vs spread* already requires it, for $\eta_s$ and for per-study cohort
construction), and `calibration_targets/vpop_marginal_targets.csv` has no such column
today. Scenario plus $n$ is the working proxy and it is a good one here, but it is a
proxy: two targets from the same paper at the same $n$ measured on *different* patient
subsets would be wrongly coupled. The default when provenance is unknown should stay
*distinct*, matching Ch. 4's rule, since coupling is the assumption that manufactures
correlation.

### Common random numbers make it differentiable

Draw $z \sim \mathcal N(0,I)$ once and hold it fixed. Then

$$\theta(\varphi, z) \;=\; \exp\!\big(\mu + W(\sigma_u \odot z)\big)$$

is a smooth deterministic function of $\varphi$, the emulator is differentiable, and
the sorted quantile map is piecewise linear and differentiable almost everywhere. So
$\varphi \mapsto Q_j(\varphi)$ is smooth, NUTS works, and the likelihood carries no
Monte Carlo noise across MCMC steps.

**Differentiability is a build requirement, not a property to hope for.** It is the
hidden coupling in this chapter: §*$K$ is an artifact of the estimator* infers all $P$
components of $\mu$ and $\log\sigma_u$, which is a few hundred correlated dimensions
with a nearly flat likelihood in most of them. NUTS handles that geometry given
gradients. A gradient-free sampler does not, at that dimension, and the fallback would
be to select $K$ after all, which is the thing this chapter switched estimators to
avoid. So the emulator has to be differentiable and vectorized, not merely fast.
**Decided: a neural-network emulator**, which supplies both and makes the whole
$\varphi \mapsto \hat Q_s(\varphi)$ map one autodiff graph.

The one remaining discontinuity is the viability filter. $Z(\varphi)$ under common
random numbers is the fraction of the fixed $z$-draws that pass, which is a step
function of $\varphi$ per draw, and the quantile map is then taken over a
surviving set whose *membership* changes with $\varphi$. Sorting a variable-length set
is where the non-smoothness actually bites, not the normalizer. So use a smooth
viability weight $w_{\mathcal V}(\theta)\in(0,1)$ with weighted quantiles
(`inference/importance.py:weighted_quantile`) rather than a hard filter plus a penalty:
$Z(\varphi)$ becomes a differentiable Monte Carlo estimate, the patient set is fixed,
and the population being fitted stays the population being reported, which is the
obligation Ch. 4 §*Patients are filtered the same way sims are* imposes.

### Surrogate error, done exactly

This is where the explicit likelihood earns its keep, and it is worth isolating
because it is Ch. 4's ugliest correction.

Ch. 4 has to inject $\varepsilon$-draws into the observed per-patient values, average
the posterior over $M$ noise copies to kill the seed dependence, and then caveat that
$\varepsilon(\theta)$ is a smooth field rather than iid noise so the correction is
approximate in an unsigned direction. All of that is forced by noise augmentation
being the only way to communicate with a trained density estimator.

With a likelihood you write it down. The emulator's error is *uncertainty about the
model's prediction*, not spread in the population, so it belongs in the covariance as
$\varepsilon_j^2$ and nowhere else. No injection, no $M$-draw averaging, no seed, no
unsigned approximation. Matched-footing terms 1 and 3 stay exactly as Ch. 4 states
them, because those are genuine population-level and center-level noise.

### $K$ is an artifact of the estimator

The identifiability wall is real: only a handful of directions of the population
spread recover from cohort summaries. **Selecting** $K$ of them and pinning the rest
is not a consequence of that wall, it is a consequence of a neural estimator needing
a finite label vector.

With an explicit likelihood and a proper prior you infer all $P$ components of $\mu$
and $\log\sigma_u$ and let the prior return the answer where the data are silent,
which is the correct posterior and what Ch. 1 says should happen. The eigenbasis
stays, in two reduced roles that are both better than selection:

- **Parameterization.** $\theta = \exp(\mu + W(\sigma_u \odot z))$ is the non-centered,
  prior-whitened form, which is exactly the geometry NUTS needs to sample a few
  hundred correlated dimensions with a nearly flat likelihood in most of them.
- **Reporting.** Which directions moved, and by how much, is the variance budget of
  Ch. 4's §*Reporting*. It is a readout, not a gate.

Three things fall out. $K$ and $K_\mu$ stop being tuning decisions, so the two-Gram
apparatus of Ch. 4 is no longer needed to decide what to infer (it remains useful for
reporting which directions carry information). Acceptance criterion 3 becomes a real
test rather than a tautology: sloppy directions genuinely can contract spuriously
now, so checking that they do not is meaningful. And the rank bound
$K \le \#\{\text{spread-feeding observables}\}$ turns from a constraint you must
enforce into a prediction you can check, since that is how many directions should
show contraction.

### What this drops, and why each piece was estimator-specific

| Ch. 4 machinery | why it existed | status here |
|---|---|---|
| $\tilde h$, the bracketing requirement, the edge check | the NPE trains on a proposal and is calibrated only over it | gone; sample $h$ directly |
| $w = h/\tilde h$, the ESS gate, the defensive mixture | recovering the prior from the proposal | gone; there is no proposal |
| $K$, $K_\mu$ selection, the second Gram as a gate | a finite label vector | gone as a gate, kept as reporting |
| surrogate noise injection, $M$-draw averaging | you cannot tell a density estimator "this is prediction error" | gone; a variance term |
| TSNPE rounds with a predictive envelope | the emulator's domain must track a proposal that is itself being inferred | reduced to: fit the emulator, fit $\varphi$, refit the emulator on the fitted population's support, refit $\varphi$, stop when $\hat\varphi$ stops moving |
| SBC over the amortized path | validating a trained estimator | still required, now ordinary SBC on a hierarchical fit |

That is roughly a third of Ch. 4, and none of it is model.

### What it keeps, unchanged

These are correctness, not elaboration, and they apply to the fixed-cloud fit too:

- $\Gamma_\omega$ from the layered omega prior, distinct from $\Gamma_\pi$.
- The `spread_source` provenance split, including on the Gram and on the envelope.
- Per-target real $n$.
- **Per-study grouping by `cohort_id`.** Ch. 4 needs it to build cohorts; this chapter
  needs it for the covariance block and for $\eta_s$. Same provenance, two uses. The
  earlier draft of this list omitted it, which is what let the likelihood factorize
  over targets that share patients.
- Matched footing (terms 1 and 3 as stated; term 2 as above; viability filtering).
- The discrepancy layer, $\eta_s$ and $\delta_j$.
- The variance budget, and reporting $F_{\mathcal V}(\theta\mid\hat\varphi)$ separately
  from the posterior-predictive marginal, with $\varphi$ drawn once per simulated
  trial.
- SBC as the gate.

## When to escalate to Chapter 4

Three cases, and the point of starting here is that you find out cheaply which one
you are in. Two of the three are now closed by measurement rather than by argument.

1. ~~**The asymptotic quantile law fails at small $n$.**~~ **Checked, and it does not**,
   at every $n$ in the PDAC target set, in the configuration §*It was checked* pins
   (log scale, three anchors below $n=20$, `normal_unbiased`). The reserve rung stands
   if a future target set moves outside that envelope: a **simulated** summary
   likelihood (synthetic likelihood) for the offending targets and the asymptotic form
   for the rest, which keeps everything else in this chapter intact. SBC remains the
   backstop.
2. **The anchor grid gets dense enough that the joint sampling law is not something
   you want to write down**, or the summary vector stops being quantiles. The density
   half of this is now handled by construction rather than by judgement:
   `_resolvable_grid` enforces $n \ge 2k$ and falls back to a symmetric triple. What
   remains open is the second clause, a summary vector that is not quantiles at all.
3. **The emulator is not usable** and you must go from real simulations, in which case
   a likelihood-free estimator with a truncation loop is the right tool and Ch. 4 is
   its specification. Note this case now carries a second trigger: the emulator must be
   **differentiable**, not merely available (§*Common random numbers*). A
   non-differentiable surrogate is an unusable one for this chapter.

## Build order

1. The ESS and $\tau^2$ null, both reference populations, on the existing cloud.
2. LOO-by-observable, top-$|\lambda|$ bins, per-observable $\tau^2$.
3. The direction table, and whatever prior work it calls for.
4. The `spread_source` split ported into `fit_prevalence_weights`, and $\Gamma_\omega$
   used for the cloud generator.
5. Refit the fixed cloud. Report the ESS against its null. This is the reference
   result.
6. `cohort_id` onto the targets, and the per-study covariance block validated the same
   way the diagonal was (draw cohorts as *rows* of the cloud, build one study's full
   anchor vector, check the block against the copula formula). This gates step 7: the
   likelihood is misspecified without it on 33 of 49 observables.
7. The summary likelihood and the NUTS fit, gated by SBC.
8. Report both, with the variance budget. If they agree, ship the simple one.

Steps 1 to 3 are the ones that were skipped, and they are the ones that answer the
question the low ESS was asking.

Much of step 1 and 2 already exists: `vpop/diagnostics.py` ships `perfect_model_null`
(the prior-predictive ESS null at each target's real $n$), `misspecification_ratio`
(the calibrated $D^2$ ratio, which is a better statistic than a raw percentile because
it is comparable across observable sets), `ess_scaling`, `duplicate_observables` (cause
4 above), and `greedy_core` + `conflict_ranking`, the last of which is strictly better
than leave-one-observable-out because it separates `unreachable` from `coupling` from
`costly` and knows that an unreachable observable costs *no* ESS. What is genuinely
missing is the fitted-weight (posterior-predictive) reference, per-observable
$\tau^2$, and the top-$|\lambda|$ dual localizer.

## Where the code lives

| Piece | Home |
|---|---|
| prevalence weighting, `tau2`, support deficiency | `vpop/weighting.py` |
| ESS null, calibrated $D^2$ ratio, ESS scaling, greedy core, conflict ranking | `vpop/diagnostics.py` (built) |
| fitted-weight null, per-observable $\tau^2$, top-$\lvert\lambda\rvert$ localizer | `vpop/diagnostics.py` (to build) |
| reachability labeler, joint discrepancy, LOO influence | `inference/diagnostics.py` (Ch. 5) |
| observed quantile anchors + provenance; `QUANTILE_METHOD`, `MIN_PATIENTS_PER_ANCHOR` | `targets/anchors.py` |
| population omega prior feeding $\Gamma_\omega$ | `targets/omega.py` |
| normal-score correlations for the copula block $C_{jl}$ | `inference/gaussian_copula_transform.py` |
| smooth viability weight + weighted quantiles | `inference/importance.py:weighted_quantile` |
| summary likelihood + NUTS population fit | `vpop/summary_likelihood.py` (to build) |
| SBC gate | `inference/sbc.py` |
| the full amortized path | [Chapter 4](population-inference-guide.md) |

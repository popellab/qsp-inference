# The statistical model

*The page to read first. It sets down the probability model that the rest of the
package implements, and the two questions we ask of it. No code, no `Stage 1 / Stage
2`, no QSP jargon that is not defined here. Terms are given three names where they
have three — one from Bayesian statistics, one from pharmacometrics, one from the
simulation-based-inference literature — so that whichever field you came from, you
can see that the object is one you already know.*

---

A mechanistic model of this kind carries a few hundred parameters — rate constants,
half-saturation concentrations, homeostatic cell densities — and almost none of them
can be measured in the clinical setting the model is meant to describe. The usual
practice is to take a literature point value for each and attach a range wide enough
to look cautious. That yields parameterizations that run, but it answers no question
about how much the data actually constrain, and it cannot tell you when a calibration
target is simply out of the model's reach.

We take the other route: write the model down as a probability model, and infer. One
question runs underneath everything that follows, and it is worth stating plainly
before the notation arrives.

> If we trust the mechanism, and we trust the priors, then the calibration should fit
> the data on its own — whether the target is a single median patient or an entire
> population. Where it does, there is nothing left to hand-tune. Where it does not,
> the failure is telling us something, and the job is to say precisely *what kind* of
> failure it is.

Most of the machinery here exists to turn that question from a slogan into something
you can actually answer.

## The generative model

There is one model, and it has three pieces: a parameter vector, a prior over it, and
a rule for how parameters produce data.

$$
\theta \sim \pi(\theta), \qquad x \mid \theta \sim p(x \mid \theta), \qquad \mathbb{E}[x \mid \theta] = g(\theta).
$$

The parameters $\theta \in \mathbb{R}^d$ live on a log scale wherever they are
positive rates, so a prior is lognormal and the natural notion of distance between two
values is a ratio, not a difference. The prior $\pi(\theta)$ is informative and is
built from external data; the next section is about how. The map $g(\theta)$ is the
mechanism itself — integrate the ODE system forward and reduce each trajectory to the
quantity that was measured — and it is deterministic: a given $\theta$ always produces
the same $g(\theta)$.

The randomness is the measurement, and it is deliberately *not* assumed Gaussian. For
each observable the noise is taken from the observed statistic's own bootstrap sampling
distribution — its skew, its tails, its bounds — and convolved onto the prediction,
multiplicatively where the quantity is positive so that positivity is preserved. Only
when a target supplies no bootstrap does a parametric fallback step in, and even then it
is a lognormal or a Gaussian chosen *per observable* by the asymmetry of the reported
confidence interval, not a blanket Normal. In the default path there is no parametric
noise family at all: the observation layer is itself available only as a sampler.

The one feature that shapes every method downstream is that the likelihood
$p(x \mid \theta)$ can be neither written down nor evaluated, for two compounding
reasons — the mean map $g$ is an ODE solve with no analytic form, and the noise layer,
in its default, is an empirical resampler rather than a density. We can *draw* pairs
$(\theta, x)$ as fast as we can simulate, but we cannot *evaluate* the density of one.
Inference that needs only sampling — not evaluation — is called **simulation-based**
(equivalently, *likelihood-free*): we learn the posterior with a neural density
estimator trained on simulations, and the observation noise enters as augmentation on
those training pairs, not as a likelihood we compute.

## Two questions, one model

The distinction between calibrating to a median patient and generating a virtual
population is not a distinction between two workflows. It is two questions asked of the
single model above, and it is the same distinction a mixed-effects modeler draws
between a typical-value fit and a random-effects distribution.

The first question fixes one observed summary $x_{\text{obs}}$ — the target medians —
and asks for the posterior $p(\theta \mid x_{\text{obs}})$. This is calibration in the
ordinary sense: one set of parameters, uncertainty and all, consistent with one
observation. Call it the **flat**, or single-unit, or fixed-effects problem.

The second question does not treat the parameters as fixed across patients. It supposes
each patient $i$ has their own $\theta_i$, drawn from a population distribution
$F(\theta \mid \varphi)$ with hyperparameters $\varphi = (\mu, \omega)$ — a location and
a spread — and the data are now summaries *across a cohort*: a median and an
inter-quartile range for each target, over some real number of patients $n_j$. What we
infer is $\varphi$, and the **virtual population is the fitted distribution
$F(\theta \mid \hat\varphi)$**. A virtual patient is a draw from it. That is all a
virtual patient is — a sample from an estimated random-effects distribution — and
saying so is the whole point of writing the model this way.

| | flat inference | virtual population |
|---|---|---|
| in Bayesian terms | single-unit posterior $p(\theta \mid x_{\text{obs}})$ | hierarchical model; infer $\varphi=(\mu,\omega)$ |
| in NLME terms | typical-value fit | the random-effects distribution ($\Theta$, $\Omega$) |
| the answer is | a posterior over $\theta$ | a distribution $F(\theta \mid \hat\varphi)$ over patients |
| a "patient" is | a posterior draw | a draw $\theta \sim F(\theta \mid \hat\varphi)$ |
| the data are | target medians | target (median, IQR) across $n_j$ patients |

Nothing about the mechanism $g$, the prior $\pi$, or the observation model changes
between the two. Only the question does. The package is arranged around that fact:
the same simulator and the same prior serve both, and the population problem is the
flat problem with a level added on top.

## The prior is a measurement model, not a guess

For a model this over-parameterized the prior is not a nuisance to be set wide and
forgotten. As the next section will make precise, the data identify only a handful of
directions in parameter space; along the rest, *the prior is the answer the calibration
returns*. A prior that carries the answer for two hundred parameters has to be built
with the care one would give to a likelihood.

So it is built, not asserted. Each quantitative claim in the literature is paired with
a small forward model that predicts *that measurement* from a few parameters, and the
pair is inverted — an ordinary measurement-error model. Three features are worth
naming because each corresponds to a standard statistical device:

A source measured in the wrong context should count for less. Each source carries a
*translation sigma* that inflates its likelihood variance according to how far its
setting — species, indication, assay — sits from the one being modeled. This is the
between-study heterogeneity of a **random-effects meta-analysis**, the $\tau^2$ that
lets mouse data inform a human parameter without pretending to be human data.

Parameters informed by the same measurement are correlated, and throwing that away
would overstate what we know. The per-parameter marginals are therefore joined by a
**Gaussian copula** — Sklar's construction, marginals for the uncertainty and a
correlation matrix for the dependence — giving a genuine joint distribution that can be
both sampled and evaluated.

A parameter with no data of its own, but a mechanistic tie to one that does, enters as
a log-linear-Gaussian **derived** child of its anchor: it inherits the parent's width
and correlation instead of receiving an invented target of its own.

## Inference without a likelihood

Since the likelihood can only be sampled, we estimate the posterior directly. Simulate
many pairs $(\theta, x)$ from the model, fit a normalizing-flow conditional density
$q(\theta \mid x)$ to them, and read it at $x = x_{\text{obs}}$. This is **neural
posterior estimation** (NPE), the amortized member of the simulation-based-inference
family: the cost is paid once, at training, after which any observation can be
conditioned on cheaply.

Three refinements matter, and each is a named technique rather than a tuning knob.
The distribution we *draw training samples from* need not be the prior we *report
under*. We draw from a wider **proposal** $\tilde\pi$ — wide in the directions the data
can actually inform, where coverage is worth paying for — and recover the posterior
under the anchored prior $\pi$ by self-normalized importance weighting, $w \propto
\pi/\tilde\pi$, watching the effective sample size. Separating the two is what lets the
proposal be a computational choice and the prior a scientific one. Left to itself the
proposal is still an ad hoc width; **truncated sequential NPE** (Deistler et al. 2022)
removes that by narrowing it, round by round, toward the region the posterior occupies,
which also keeps the importance weights well-behaved. The result reported is the
posterior under $\pi$; the proposal has done its work and drops out.

## Populations: the level on top

The population problem adds the hierarchy the flat problem omits. Patient $i$ draws
$\theta_i \sim F(\theta \mid \varphi)$, the hyperparameter $\varphi = (\mu, \omega)$ has
its own hyperprior, and the observations are cohort summaries. We infer
$p(\varphi \mid \text{summaries})$ by amortized inference over simulated cohorts, and
emit the virtual population as draws at $\hat\varphi$.

Two choices here are statistical rather than incidental. The first: you cannot estimate
two hundred population spreads when only a few directions carry signal, so $\log\omega$
is inferred along the leading eigen-directions of $\pi$'s covariance — the directions
the data speak to — and left at its anchored prior value everywhere else. The
identifiability limit that governs the flat posterior governs the population spread in
exactly the same way, and the eigenbasis is where that limit is imposed. The second:
the amount a target says about the spread should match how many patients it was measured
in. A target reported over six patients constrains $\omega$ weakly; one over nine
hundred constrains it tightly. Cohort summaries are drawn at each target's real
published $n$, so the evidence is weighted honestly.

An older construction — generate a cloud of plausible patients and reweight it to match
the observed marginals (Allen et al. 2016) — remains available as a fixed-cloud
fallback. The hierarchical model is preferred because it *infers* the spread from data
instead of inheriting it from whatever width the proposal happened to have.

## The hard part: most parameters are not identified

Everything above would be routine if the data determined the parameters. They do not,
and it is better to confront this directly than to let it surface as mysterious
posterior widths.

Only a handful of the model's parameters are identified by the clinical data. Along
every other direction the likelihood is nearly flat: many different $\theta$ produce
almost the same $x$, so the data cannot choose among them. This is *practical* non-identifiability — not a defect of the sampler, and
not something more simulations would cure. It is the geometry of the model, the
**sloppy** structure (Gutenkunst et al. 2007) that mechanistic models of this size
almost always have: a few stiff directions hold all the constraint and the rest are
soft.

The consequence is worth stating without softening. For an unidentified parameter the
posterior is equal to the prior, and the population spread is equal to the prior spread.
The data have added nothing, and the honest report says so — no shrinkage on a soft
direction is the correct answer, not a failure to converge. Reporting a confidently
narrow posterior there would be reporting the proposal, not the data.

This is also why the prior work earns the name *science* rather than *tuning*. Along the
identified directions the data speak and the prior yields to them. Along the soft
directions the prior is the answer, and it can only come from outside the calibration —
from the measurement-model priors and their derived children. No sampler, however
elaborate, invents information the data do not contain. That external half of the model
is irreducible, and it is the one part of this pipeline where a human is meant to be in
the loop.

## Checking the fit

A calibration is not to be believed until it has been checked, and each check earns its
place by carrying a null distribution computed from the model's own simulations — so
that "this target is a poor fit" is a statement with a false-positive rate, not an
impression. Taken together they are the diagnostics of an ordinary Bayesian workflow
(Gelman et al. 2020), run in a fixed order because each step's verdict is only
trustworthy once the previous one has passed.

The gate is **simulation-based calibration** (Talts et al. 2018): draw a parameter from
the prior, simulate, infer, and check that the true value lands uniformly within its
own posterior. It never sees the real observation, so it validates the inference
procedure rather than the model — but if it fails, no later verdict can be trusted,
because the machinery itself is miscalibrated. With that passed, a **prior-predictive
check** (Box 1980; Evans & Moshonov 2006) asks whether the model, run under the prior,
produces the observed value at all, or whether $x_{\text{obs}}$ sits in the tail where
the anchored model rarely goes. When it does sit in the tail, one more question
separates the two things that could be wrong: point the sampler at $x_{\text{obs}}$ and
see whether *any* admissible $\theta$ reaches it. If some does, the mechanism is fine
and the prior is merely aimed elsewhere — a matter of re-anchoring. If none does — if
the inverse map's own best guesses, simulated forward, still miss — then the mechanism
cannot produce the observation, and no prior will rescue it. A final **joint check**
(a Mahalanobis discrepancy against a self-reference null, with per-observable
leave-one-out influence) catches the case where every target is individually reachable
but no single $\theta$ reconciles all of them at once, which points not at any one
observable but at the coupling between them.

It is worth being clear about what this suite does and does not do, because the failure
mode it is meant to prevent is a seductive one. It **localizes** a miss and it **gates**
the next step; it does not prescribe the fix. No scalar summary can tell you which
mechanism to add or which parameter value is right — that reading always comes from a
mechanistic trace and knowledge of the biology. A diagnostic that hands you a parameter
to turn is not saving you work; it is guessing, and a calibration tuned to a guessing
diagnostic will confirm whatever it was pointed at. The suite's job is to tell you that
a miss is real, where it lives, and whether re-anchoring or a mechanism change is called
for — and then to hand a short, ranked list to the trace, where the actual diagnosis is
made.

## A dictionary

Every term this package uses has an older name, usually two. The forward map from one
vocabulary to the others:

| here | Bayesian statistics | pharmacometrics / NLME | reference |
|---|---|---|---|
| SubmodelTarget → Stage-1 posterior | measurement-error / meta-analytic prior | literature-informed prior | Box & Tiao 1973 |
| translation sigma | between-study heterogeneity $\tau^2$ | inter-study variability | DerSimonian & Laird 1986 |
| copula prior | marginals joined by a copula | correlated random effects | Sklar 1959; Nelsen 2006 |
| derived prior | log-linear-Gaussian child | structural/covariate relation | — |
| flat inference | single-unit posterior | typical-value fit | — |
| virtual population / patient | random-effects distribution / a draw from it | population distribution / subject | Gelman & Hill 2007 |
| $(\mu, \omega)$ | population hyperparameters | $\Theta$, $\Omega$ | — |
| prevalence weighting | importance reweighting to fixed marginals | VPop selection | Allen et al. 2016 |
| NPE / SBI | amortized, likelihood-free posterior | — | Cranmer et al. 2020; Greenberg et al. 2019 |
| TSNPE | truncated sequential NPE | — | Deistler et al. 2022 |
| proposal → prior reweight | self-normalized importance sampling | — | Kong et al. 1994 |
| identifiability wall | practical non-identifiability | — | Gutenkunst et al. 2007 |
| eigenbasis reparameterization | identified subspace / stiff directions | — | Gutenkunst et al. 2007 |
| SBC | simulation-based calibration | — | Talts et al. 2018 |
| prior-data conflict | prior-predictive check | — | Box 1980; Evans & Moshonov 2006 |
| self-reference $D^2$ / LOO influence | predictive discrepancy + leave-one-out | — | Vehtari et al. 2017 |
| LOO-PIT *(planned)* | leave-one-out probability integral transform | — | Gelfand et al. 1992 |
| misspecification detector | model criticism for amortized inference | — | Schmitt et al. 2023; Ward et al. 2022 |
| OBED | Bayesian experimental design | optimal design | Chaloner & Verdinelli 1995 |

## Where each part lives

The package is laid out as the parts of this model; the [docs index](README.md) is the
full map.

| part of the model | package surface | guide |
|---|---|---|
| building the prior $\pi$ | `priors/`, `submodel/` | [Submodel Inference Guide](submodel-inference-guide.md) |
| the flat posterior | `inference/` | [Stage 2 SBI Guide](stage2-sbi-guide.md) |
| the population $F(\theta\mid\hat\varphi)$ | `vpop/` | *(guide lands with the hierarchical path)* |
| the workflow checks | `inference/diagnostics.py`, `vpop/diagnostics.py`, `audit/` | [Stage 2 SBI Guide](stage2-sbi-guide.md#diagnostics) |
| experimental design | `inference/obed.py` | [Stage 2 SBI Guide](stage2-sbi-guide.md#optimal-bayesian-experimental-design-obed) |

---

### References

Allen, Rieger & Musante (2016), *Efficient generation and selection of virtual
populations in QSP models*, CPT:PSP. —
Box (1980), *Sampling and Bayes' inference in scientific modelling and robustness*,
JRSS-A. —
Box & Tiao (1973), *Bayesian Inference in Statistical Analysis*. —
Chaloner & Verdinelli (1995), *Bayesian experimental design: a review*, Statist. Sci. —
Cranmer, Brehmer & Louppe (2020), *The frontier of simulation-based inference*, PNAS. —
Deistler, Gonçalves & Macke (2022), *Truncated proposals for scalable and hassle-free
simulation-based inference*, NeurIPS. —
DerSimonian & Laird (1986), *Meta-analysis in clinical trials*, Control. Clin. Trials. —
Evans & Moshonov (2006), *Checking for prior-data conflict*, Bayesian Analysis. —
Gelfand, Dey & Chang (1992), *Model determination using predictive distributions*,
in Bayesian Statistics 4. —
Gelman & Hill (2007), *Data Analysis Using Regression and Multilevel/Hierarchical
Models*. —
Gelman et al. (2020), *Bayesian workflow*, arXiv:2011.01808. —
Greenberg, Nonnenmacher & Macke (2019), *Automatic posterior transformation for
likelihood-free inference*, ICML. —
Gutenkunst et al. (2007), *Universally sloppy parameter sensitivities in systems biology
models*, PLoS Comput. Biol. —
Kong, Liu & Wong (1994), *Sequential imputation and Bayesian missing data problems*,
JASA. —
Nelsen (2006), *An Introduction to Copulas*. —
Schmitt et al. (2023), *Detecting model misspecification in amortized Bayesian
inference with neural networks*, GCPR. —
Sklar (1959), *Fonctions de répartition à n dimensions et leurs marges*. —
Talts, Betancourt, Simpson, Vehtari & Gelman (2018), *Validating Bayesian inference
algorithms with simulation-based calibration*, arXiv:1804.06788. —
Vehtari, Gelman & Gabry (2017), *Practical Bayesian model evaluation using leave-one-out
cross-validation and WAIC*, Stat. Comput. —
Ward, Cannon, Beaumont, Fasiolo & Schmon (2022), *Robust neural posterior estimation and
statistical model criticism*, NeurIPS.

# Population inference (VPops): the hierarchical $(\mu,\omega)$ path

> **Chapter 4 of the [docs spine](README.md).** Read [the statistical
> model](statistical-model.md) first: this guide assumes its vocabulary, the
> anchored prior $\pi$, the proposal $\tilde\pi$, the population
> $F(\theta\mid\varphi)$, and self-normalized importance reweighting. Three things
> change on arrival here. The inference target is $\varphi$ rather than $\theta$, so
> the reweight that recovers the reported answer is at the **hyper** level,
> $w\propto h/\tilde h$, not the flat chapter's $w\propto\pi/\tilde\pi$. The prior
> contributes **two** covariances rather than one, because epistemic uncertainty
> about the typical patient and between-patient variability are different objects.
> And the model carries an explicit **discrepancy** layer, because a population fit
> with nowhere to put structural misfit charges it to $\hat\omega$. §*The four
> distributions* sets up the first two, §*Discrepancy* the third, and §*What this
> represents* states plainly which parts of the reported population are inference
> and which are assumption. §*Open work* tracks what is documented here versus what
> is built.
>
> **This chapter is the full specification and the escalation path, not the default
> route.** Its *model* (the two covariances, the provenance split, matched footing,
> the discrepancy layer, the variance budget) is the part that matters and is shared.
> Its *estimator* (amortized NPE trained on a hyper-proposal, recovered by an
> importance reweight) is heavier than the data can exercise: roughly a third of what
> follows exists only because the estimator trains on a proposal rather than on the
> prior. [Chapter 4b](population-inference-tractable.md) keeps the model, replaces
> the estimator with an explicit summary likelihood fitted by NUTS, and puts a
> misspecification diagnosis in front of both. **Start there.** Come here when 4b's
> gate fails, for the reasons 4b's §*When to escalate* lists.

Flat inference (Ch. 3) gives the single-unit posterior $p(\theta\mid x_{\text{obs}})$,
the typical patient. A **virtual population** is the random-effects version:
patients are draws $\theta_i \sim F(\theta\mid\varphi)$, the data are *cohort*
summaries (a quantile grid over $n$ patients), and the answer is the fitted
distribution $F(\theta\mid\hat\varphi)$, not a point. This chapter is the machinery
that infers $\varphi$ and generates the population, and it is the same three-object
design as flat inference with $(\mu,\omega)$ inferred on top.

## The four distributions, and which level each lives on

Everything here is one generative model observed at the cohort level. Four
distributions appear, and the single most important thing to keep straight is that
they do **not** all live on the same level. Two are over patients ($\theta$) and two
are over hyperparameters ($\varphi$). Conflating the levels is one of the two errors
this chapter used to make, and §*Reporting under the hyperprior* is where it bites.
The other is conflating two covariances at the patient level, immediately below.

**Patient level ($\theta$):**

- **$\pi$, the anchored prior, and *two* covariances from it.** Built from submodel
  targets, derived priors, and declared assumptions (Ch. 2). It supplies the anchored
  center $\mu^\pi$ and two distinct second-moment objects:

  $$\Gamma_\pi \;=\; \operatorname{Cov}_\pi[\log\theta],
  \qquad
  \Gamma_\omega \;=\; D\,R\,D, \quad D=\operatorname{diag}(\omega),\ \ R=\operatorname{corr}_\pi[\log\theta].$$

  $\Gamma_\pi$ is **epistemic**: how well we know the typical patient's parameters.
  It sets the center's prior slack, and nothing else. $\Gamma_\omega$ is
  **biological**: how much patients differ. It is the population geometry, the thing
  $\sigma_u=1$ reproduces. Its marginals $\omega$ come from the layered
  population-spread prior (`targets/omega.py`: a global default, then a per-role
  class value, then an explicit belief, then a submodel `population` block shrunk
  toward those by its own $n$), **not** from $\pi$'s marginal widths. That module
  states the rule this chapter has to obey:

  > omega is BETWEEN-PATIENT variability. It must not be read off a flat prior's
  > marginal width (that is epistemic uncertainty about a population CENTER, the
  > SEM-vs-SD conflation one level up).

  Reading the population off $\Gamma_\pi$ reports a wide population wherever we
  happen to be ignorant and a narrow one wherever the literature is dense. That is
  not a biological claim, and because it governs the whole sloppy complement it is
  most of the reported spread.

  Only the *correlations* $R$ are taken from $\pi$, and those are epistemic too:
  parameters measured by the same source are correlated in their estimation error,
  which is not the same as covarying across patients. That is a declared assumption,
  not a measurement, and §*What this represents* says where it bites.

  **Both are fixed; neither is touched inside the loop.** This is the only surface
  where hand-work is legitimate. Note what $\pi$ is *not*: it is not the reported
  population. The reported population is $F(\theta\mid\hat\varphi)$, and if the data
  say the cohort is wider than $\Gamma_\omega$, the report must be wider.

- **$\tilde\pi$, the simulation proposal.** Where the *real forward model* is
  evaluated. Wide on the identified directions only (widening a degenerate direction
  buys nothing but wasted sims), then truncated toward the reachable set across
  rounds (TSNPE). Its job is to decide where the expensive simulations go, and
  therefore the region the emulator is trained to be accurate on. It does not enter
  the reported density. It is **not** inert, though: through the reference batch it
  sets the local Jacobian's operating point, hence $G$, hence the basis, hence
  *which* directions of $\omega$ are inferred at all. That is why the stop rule
  carries a basis-stability term rather than treating $\tilde\pi$ as a pure
  computational detail.

**Hyper level ($\varphi = (\mu,\ \log\sigma_u,\ \eta,\ \delta)$):**

- **$h(\varphi)$, the hyperprior.** The reporting geometry, written out in full in
  §*The hyperprior*: $\mu\sim\mathcal N(\mu^\pi,\tau_\mu^2\Gamma_\pi)$ on the
  inferred center components, $\log\sigma_{u,k}\sim\mathcal N(0,s^2)$ on the
  inferred spread directions, and the discrepancy laws (§*Discrepancy*). This is
  what the reported posterior must be conditioned on.
- **$\tilde h(\varphi)$, the hyper-proposal.** The distribution the training
  cohorts' $\varphi$ are drawn from: the same $\mu$ law and the same discrepancy
  laws, with $\log\sigma_{u,k}\sim\mathcal N(0,(fs)^2)$, $f>1$, so the amortized NPE
  sees cohorts both far wider **and far narrower** than the anchor. $\tilde h$ is the
  **support over which the amortized posterior is calibrated** and the denominator of
  the only legitimate importance weight in this chapter, so it is a first-class
  object with an explicit definition, not an implementation detail of step 3.

$(\mu,\omega)$ itself is **inferred, not set**: the width is inferred where the data
carry signal and falls back to the anchored $\sigma_u\approx1$ where it does not.
Inferred in the **identified subspace** (the eigenbasis below), not per-parameter,
since only a handful of combination directions are identifiable and inferring two
hundred spreads would fit noise.

### Which reweight is legitimate

Exactly one, and it is at the hyper level. The amortized NPE is trained on cohorts
whose $\varphi$ came from $\tilde h$, so what it learns is
$q(\varphi\mid x)\propto L(x\mid\varphi)\,\tilde h(\varphi)$: the posterior with the
hyper-proposal standing in for the hyperprior. Recovering the posterior under $h$ is
one self-normalized weight on $\varphi$-draws,

$$w(\varphi)\;\propto\;\frac{h(\varphi)}{\tilde h(\varphi)}.$$

There is **no patient-level $\pi/\tilde\pi$ weight anywhere in the report.** Applying
one would be a category error with a concrete consequence: $\pi$ and $\tilde\pi$ are
the same `EigenbasisPopulation` differing only in $\sigma_u$, so self-normalized IS
from $\tilde\pi$ to $\pi$ converges to *the anchored population with
$\sigma_u\approx1$*. It would drag $\hat\omega$ back to the anchor from whichever
side the data moved it, undoing the inference it sits downstream of. $\tilde\pi$
drops out of the reported density not by being reweighted away but by never having
been a population in the first place: it only decides where sims are spent.

The weight machinery is the same one-liner as flat inference
(`log_importance_weights`, self-normalized, so a truncated proposal's normalizer
cancels exactly), just evaluated on $\varphi$ rather than $\log\theta$
(`importance.py`). Two facts about $w$ are easy to miss and both are load-bearing.

**$w$ collapses to the $\sigma_u$ ratio.** $h$ and $\tilde h$ share the $\mu$ law and
the discrepancy laws by construction, so those factors cancel and

$$w(\varphi)\;\propto\;\frac{h_\sigma(\log\sigma_u)}{\tilde h_\sigma(\log\sigma_u)}
\;=\;\exp\!\left[-\tfrac{1}{2}\left(\tfrac{1}{s^2}-\tfrac{1}{f^2 s^2}\right)\sum_{k\le K}(\log\sigma_{u,k})^2\right].$$

Everything the reweight does to the reported answer is a shrinkage of $\hat\omega$
toward the anchor, governed by one number, $s$. That is legitimate (it is a genuine
prior, and applying it is what makes the answer a posterior under $h$ rather than
under a computational convenience), but it means $s$ has to be declared and defended
like any other prior, and the per-direction prior-to-posterior contraction has to be
reported alongside $\hat\sigma_u$. §*The hyperprior* fixes $s$; §*Reporting* reports
the contraction.

**The weight is bounded, on purpose.** Self-normalized IS needs $h\ll\tilde h$ with a
ratio that does not blow up in the tails, and nothing enforces that automatically.
The Gaussian-on-$\log\sigma_u$ pair above does: with $f>1$ the exponent is
negative-definite, so $w\le1$ everywhere and no single draw can run away. A
$\tilde h$ built by *scaling* $\sigma_u$ (a box, or a point-widened range) would
instead have hard edges, and $h$ would then have to be truncated to them, which also
constrains what the SBC gate is allowed to draw. Prefer the lognormal pair.

## Why an eigenbasis for $\omega$

Structural non-identifiability means only a handful of *combination* directions of
the population spread recover from cohort summaries; the rest are sloppy and must
ride the prior. So $\omega$ is inferred as $\sigma_u$ along the top $K$ directions of
a basis, not as $P$ per-parameter spreads:

$$\log\theta \sim \mathcal N\!\big(\mu,\; W \operatorname{diag}(\sigma_u^2) W^\top\big),
\qquad \theta = \exp\!\big(\mu + W(\sigma_u \odot z)\big),\ z\sim\mathcal N(0,I).$$

The basis is the **active subspace in the population metric** (`vpop/eigenbasis.py`):
the data Fisher $G$ whitened by $\Gamma_\omega$, so directions are ranked by *spread
information per unit of between-patient variance*. With $\sigma_u=1$ on every
direction the draw reproduces $\Gamma_\omega$ exactly ($WW^\top=\Gamma_\omega$): the
omega layer sets the geometry, and the inferred top-$K$ deviate only where the data
pull. The sloppy complement is marginalized over its prior spread rather than pinned,
so cohorts keep full correlated spread. **$\Gamma_\omega$ comes from the anchored
omega layer, never from the wide $\tilde\pi$ and never from $\pi$'s marginal
widths**: neither the proposal nor our ignorance may set the population geometry.

**$G$ is built from spread-feeding observables only, and that caps $K$.** The
provenance split (§*Center vs spread*) restricts the *conditioning vector*'s spread
half to `across_patient` and `biological_experimental`. It has to restrict the Gram
too. A `technical` or `center_only` row admitted into $G$ contributes a direction
informed by no spread data at all, whose $\sigma_u$ then simply returns the
$\tilde h$ prior while displacing a genuinely informed direction out of the top-$K$.
Building $G$ from spread rows only also supplies the ceiling nothing else states:

$$K \;\le\; \operatorname{rank}(G) \;\le\; \#\{\text{spread-feeding observables}\}.$$

Five spread-feeding targets means at most five inferable spread directions, however
promising the spectrum looks. The eigenvalue gap will not tell you this: past the
rank of $G$ the whitened spectrum is floating-point noise, and `eigh` still returns
it sorted, so a naive top-$K$ will select it happily. Check the rank, not the gap.

Three properties of this construction are easy to over-read, and each has a
consequence downstream.

**$G$ is not prior-independent here, and it moves every round.** The
`vpop/eigenbasis.py` module docstring describes $G$ as computed once, offline, from a
fixed pool, independent of the prior. That is true of the flat, offline use. It is
**false in this loop**: step 2 refits $G$ inline from the current round's reference
batch, which was drawn from $\tilde\pi_r$, which has been truncated toward the data.
So $G_r$, and therefore $W_r$, depend on the round. Consequences to carry:

- $\sigma_{u,k}$ names a *different physical direction* in round $r$ and round
  $r{+}1$, so comparing $\hat\omega$ across rounds, and the "truncation stabilizes"
  stop rule, are not measuring what they appear to measure. The stop rule therefore
  includes an explicit **basis-stability** term: the principal angles between the
  top-$K$ subspaces of $W_r$ and $W_{r+1}$ must be small, not just the envelope
  shrinkage. A loop that converges in envelope while the basis is still rotating has
  not converged.
- The SBC gate must **re-derive the basis inside each synthetic replicate**. Holding
  $W$ fixed across replicates validates a pipeline we do not run, and it hides
  exactly the basis instability above. This is a real cost multiplier on the gate;
  §*Validating* says what it costs and what gets relaxed.

**$J$ is a local linear fit, so the "identifiability ceiling" is partly
self-referential.** `fit_local_jacobian` returns a kernel-weighted linear secant.
Acceptance criterion 2 compares recovery against a ceiling derived from that same
linearization, so a linearization error shows up as *agreement* rather than as a gap.
The criterion is therefore a wiring check, not evidence that the top-$K$ directions
are genuinely recoverable under the nonlinear map. Nonlinearity is checked separately,
by the end-to-end SBC gate, which never linearizes.

**$W$'s columns are not orthogonal in log-$\theta$ space.** $WW^\top=\Gamma_\omega$
makes the $u$-coordinates uncorrelated and unit-variance, but $W = L\tilde V$ with $L$
the Cholesky factor of $\Gamma_\omega$, so the columns are orthonormal only in the
*whitened* metric. Widening $\sigma_u$ on the top-$K$ adds a rank-$K$ term
$\sum_{k\le K}(f^2-1)\,w_k w_k^\top$ whose effect on parameter-space marginals spills
onto sloppy parameters too. So `proposal.py`'s "unchanged (anchored) on the sloppy
complement" is a statement about $u$-coordinates only; it is **not** a guarantee that
$\tilde\pi=\pi$ marginally along the sloppy directions.

### The hyperprior: two metrics, one basis

We infer $\varphi$ against a hyperprior. Center and spread share the *basis*, so both
halves of $\varphi$ are expressed in one coordinate system and a single hyper-level
reweight puts the whole of $\varphi$ back under $h$. They do **not** share a metric,
and an earlier version of this chapter argued that they should:

$$\mu \sim \mathcal N\!\big(\mu^\pi,\ \tau_\mu^2\,\Gamma_\pi\big)
\;\Longleftrightarrow\; \mu = \mu^\pi + \tau_\mu L_\pi z,\qquad
\log\sigma_{u,k}\sim\mathcal N(0,s^2)\ \ (k\le K),\qquad
\sigma_{u,k}=1\ \ (k>K),$$

with $L_\pi$ the Cholesky factor of $\Gamma_\pi$, and $\Gamma_\omega$ (not
$\Gamma_\pi$) the covariance that $\sigma_u=1$ reproduces. The elegant version,
$\mu\sim\mathcal N(\mu^\pi,\tau_\mu^2\Gamma_\omega)$, welds the center's prior slack
to the between-patient spread, which is the SEM-vs-SD conflation `targets/omega.py`
exists to prevent, running the other way. A parameter can be known precisely on
average and vary a lot across patients (a well-measured receptor density with real
inter-individual range), or be barely known on average and be nearly constant across
patients (a physical constant nobody has looked up). Those two profiles are not
proportional, and forcing them to share a metric asserts that they are.

The cost of splitting is small and worth naming: in the $u$-coordinates that diagonalize
the spread, $\operatorname{Cov}[u_\mu]=\tau_\mu^2 W^{-1}\Gamma_\pi W^{-\top}$ is no
longer diagonal. It is still an ordinary Gaussian and still analytic. What we give up is
a slogan, not tractability.

$\tau_\mu$ keeps its interpretation: one dimensionless knob, how many *prior* widths
the center may move, so the center moves more where $\pi$ is genuinely uncertain and
less where the literature is tight. A flat $\mu\sim\mathcal N(\mu^\pi,\tau_\mu I)$ in
raw log-units would instead let the center wander equally far in a tightly-known rate
constant and a barely-known one, incoherent with a $\pi$ that knows those two to very
different precision.

**Fixing $s$.** $s$ is the prior log-sd of the population spread *relative to the
anchored omega*, so $s=0.4$ says a factor of $e^{0.8}\approx2.2$ either way is a
two-sigma surprise. It is the single number deciding how hard the report is pulled
back to the omega layer (§*Which reweight is legitimate*), so pick it from what the
omega layer's own provenance supports rather than for convenience: a level-1 or
level-2 omega (global default, role class) is a class guess and deserves a generous
$s$; a level-4 omega already shrunk against $n$ donors deserves a tight one.
Reporting $\hat\sigma_u$ without reporting $s$ and the per-direction contraction is
reporting the prior and calling it an inference.

### Which components of $\mu$ are inferred

The metric argument above is about $\Gamma$. The *ranking* is a separate question and
comes from the Gram, and the version of this argument that appeared in earlier drafts
does not survive checking.

That draft proposed a second Gram whitened by the median's SE
($\approx1.253\,\sigma/\sqrt{n_i}$) instead of the IQR's
($\approx1.573\,\sigma/\sqrt{n_i}$), on the grounds that these are "a different
$n$-scaling and a different per-observable weighting". They are neither.
`whiten_sensitivity_rows` divides row $i$ by
$\sqrt{(c\,\omega_0\lVert J_i\rVert/\sqrt{n_i})^2+\text{floor}_i^2}$, so with a
negligible floor the whitened row is

$$\frac{J_i}{c\,\omega_0\lVert J_i\rVert/\sqrt{n_i}}\;=\;\frac{\sqrt{n_i}}{c\,\omega_0}\,\hat J_i .$$

The sensitivity *magnitude* cancels, which is correct and worth knowing on its own:
Fisher information about a scale parameter is scale-free, so an observable ten times
more sensitive has ten times the IQR and ten times the SE of that IQR, and learns the
same amount about $\sigma_u$. Swapping $1.573$ for $1.253$ multiplies **every** row by
the same constant, so $G_\mu=(1.573/1.253)^2\,G$ and $W_\mu=W$ exactly. A second Gram
built that way is a no-op.

What actually separates the center's ranking from the spread's is that center noise is
**not** proportional to $\lVert J_i\rVert$, and that the center is informed by targets
the spread never sees:

$$\Sigma_{\text{med},i} \;=\; \left(\frac{1.253\,\omega_0\lVert J_i\rVert}{\sqrt{n_i}}\right)^{2}
\;+\;\sigma_{c,i}^{2}\;+\;\tau_{\text{trans},i}^{2}\;+\;\text{floor}_i^{2},$$

summed over **all six** `spread_source` values, against a $G$ summed over the two
population-spread sources only. $\sigma_{c,i}$ is the target's reported center
uncertainty (its CI, in log space) and $\tau_{\text{trans},i}$ its translation sigma
(Ch. 2). Both are per-target constants with no $\lVert J_i\rVert$ in them, and the
four $\mu$-only sources contribute to $G_\mu$ and to nothing else. That is what makes
$W_\mu\neq W$ and $K_\mu$ its own number, with $\mu$ inferred on the top $K_\mu$
directions of $W_\mu$ and pinned at $\mu^\pi$ on the rest. Build $G_\mu$ that way or
do not build it.

**At small $n$ the two are genuinely confounded, and we report it.** At $n=6$ the
sample median's own noise is $\approx1.253\,\sigma/\sqrt6\approx0.51\,\sigma$,
comparable to the between-patient spread being read off the IQR, and in overlapping
directions. So a target can carry real information about $(\mu,\sigma_u)$ jointly
while identifying neither separately. The diagnostic is cheap and mandatory: compute
the prior-predictive correlation between $\hat\mu$ and $\hat\sigma_u$ per direction at
each target's **real** $n$, and where it is high, say the direction is not separately
identified rather than reporting two numbers that only exist as a ridge.

### The hyper-proposal $\tilde h$, explicitly

$\tilde h$ shares $h$'s $\mu$ law and discrepancy laws, and inflates the spread
prior's *log-scale* on the top-$K$ directions:

$$\log\sigma_{u,k}\ \sim\ \mathcal N\!\big(0,\ (f\,s)^2\big),\qquad f>1,\ \ k\le K,$$

leaving the sloppy complement anchored at $\sigma_u=1$. Note what this is not.
`widen_on_identified` (`vpop/proposal.py`) *multiplies* $\sigma_u$ by a factor $>1$,
which is the right shape for $\tilde\pi$ and the wrong one for $\tilde h$: it makes
the anchored value a **floor**, so every training cohort is at least as wide as the
anchor, a truth narrower than the anchor falls outside the training support entirely,
and $\hat\omega$ is biased high by construction. $\tilde h$ must bracket on **both**
sides, which is what centering the law on $\log\sigma_u=0$ and widening its scale
does.

Bracketing is a validity requirement, not a tuning preference: the amortized NPE is
calibrated only over the $\varphi$ it trained on, so a truth outside $\tilde h$'s bulk
lands the observed cohort outside the training support and the posterior extrapolates
silently. Hence an explicit **edge check**: if posterior mass on $\log\sigma_{u,k}$
piles up in the outer decile of $\tilde h$'s range on either side, raise $f$ and
refit. Do not report a population from an edge-pinned posterior. The check is
per-direction, and it is normal for one direction to force the expansion for all.

**A defensive floor on the weights.** If the ESS of $w=h/\tilde h$ binds
(§*Reporting*), the fix is not to narrow $\tilde h$, which trades a compute problem
for a validity one. Draw $\varphi$ from a defensive mixture
$\tilde h = \alpha\,h + (1-\alpha)\,\tilde h_{\text{wide}}$ instead (Hesterberg 1995).
Then $w\le1/\alpha$ pointwise, the ESS has a floor by construction, and the wide
component still brackets. $\alpha=0.5$ costs almost nothing and removes the tension
entirely.

## Center vs spread: which targets feed $\omega$

The split is a **provenance** decision made per target, not a factorization of the
likelihood: inside a spread-feeding target the NPE conditions on the *joint* anchor
vector, so its median and its spread anchors inform $\mu$ and $\sigma_u$ together.
What the split decides is narrower and more defensible: is this target's reported
dispersion genuine between-patient variability, or is it center/measurement
uncertainty that must not be read as population spread?

The seam is `ObservedDistribution.feeds_population_spread`, true iff the target's
`spread_source` is in maple's `POPULATION_SPREAD_SOURCES` $=\{$`across_patient`,
`biological_experimental`$\}$; it flows through to `ObservedAnchors.feeds_spread`
(`targets/anchors.py`). All six `spread_source` values map cleanly:

| `spread_source` | feeds $\omega$? | informs | anchor branch | spread scale |
|---|---|---|---|---|
| `across_patient` | yes | $\mu$ and $\omega$ | population `samples` / `observed_distribution` | `n_biological` |
| `biological_experimental` | yes | $\mu$ and $\omega$ | same | `n_biological` |
| `technical` | no | $\mu$ only | ci95 expand | (center noise) |
| `translation` | no | $\mu$ only | ci95 expand | (center noise) |
| `assumed` | no | $\mu$ only | ci95 expand | (center noise) |
| `center_only` | no | $\mu$ only | ci95 expand | (center noise) |

The spread half of the conditioning vector, the spread Gram $G$, and the reachability
envelope are built only from the two population-spread sources; the center (median)
half and $G_\mu$ are built from all six. Reading a `technical` replicate SD or a
`center_only` CI as population spread would silently over-disperse $\omega$, the same
call the fixed-cloud VPop (`vpop/weighting.py`) already makes when it drops them.

A $\mu$-only target still carries information the model must neither discard nor
over-trust: the **center uncertainty** on its median, which its CI measures. That
enters as matched noise on the center anchor (see *Matched footing* below), not as
spread. The `n_biological` column is the magnitude anchor for the two spread sources:
a spread reported over six donors constrains $\omega$ weakly, one over thirty
strongly, and the training cohorts are summarized at that same $n$ so they carry the
identical finite-sample law.

**The conditioning vector is a quantile grid, not median + IQR.**
`targets/anchors.py` builds $Q(p)$ at whatever anchor grid the source supports
(deciles where a target declares them, quartiles otherwise), with `_resolvable_grid`
dropping levels more extreme than a sample of size $n$ can resolve. Median + IQR is
the $\{0.25,0.5,0.75\}$ special case. Two consequences. Extra anchors carry *shape*,
not additional scale information, so they do not relax the rank bound on $K$ above.
And `whiten_sensitivity_rows` hard-codes `se_iqr_c` $=1.573$, the Gaussian SE of the
IQR; on a decile grid the corresponding quantile SEs are different constants, and
using the IQR's throughout mis-weights those targets relative to quartile ones.

Three further provenance facts have to travel with each target, because the cohort
construction in step 3 depends on them.

**Per-patient assay noise (`cv_technical`).** A spread-feeding target's observed
per-patient values carry assay noise *on top of* biological variability, so its
reported IQR estimates biological $\oplus$ technical. This is a distinct quantity from
`spread_source` (which asks what the dispersion *is*) and from the CI (which measures
center uncertainty). It is the third matched-footing term, and without it $\hat\omega$
is biased high. See *Matched footing*, term 3.

**Study membership (`cohort_id`).** Targets measured on the same patients share a
`cohort_id`; targets from different studies do not. Step 3 draws one virtual cohort
per distinct `cohort_id`, not one shared cohort subsampled per target. The default
when provenance is unknown is *distinct*, since shared-patient coupling is the
assumption that manufactures correlation (see *Cohorts are per study*). `cohort_id` is
also what the study-offset term $\eta_s$ attaches to (§*Discrepancy*).

**Cohort size for a $\mu$-only target.** A spread-feeding target's training cohort is
drawn at its real `n_biological`. A $\mu$-only target has no such $n$, and the choice
is not free: if its training summary is a median over $n$ emulated patients, that
median carries $\approx1.253\,\sigma_{\text{pop}}/\sqrt n$ of *population*-driven
noise, while the observed value carries only its measurement CI. The NPE reads the
difference as information about $\sigma_u$, from a target the table above declares
must not feed $\omega$. So: **$\mu$-only training summaries are population medians at
large $n$**, where that sampling term is negligible, plus the fresh center noise of
matched-footing term 1. The converse trap is just as real: match the study's own $n$
instead and part of the reported CI is already present in the training summary, so
adding the full CI-derived $\sigma_c$ on top double-counts. Take the large-$n$
convention and apply it everywhere.

**On subsampling.** Where a target reports over more patients than the anchor uses,
prefer using the full reported sample and matching that $n$ in training over drawing a
seeded $n$-subsample. A single seeded subsample makes the observed conditioning vector
depend on the seed and discards information; if one is used anyway, the seed dependence
is a reported quantity and the estimate is averaged over seeds.

## The loop

One amortized NPE over cohorts, wrapped in TSNPE rounds that **re-simulate** into the
shrinking reachable set: a single wide reference batch would spread the pool-bound
emulator too thin to gate on, so each round re-sims where the patients are. Everything
is fit inline from the current round's reference batch, with no precomputed artifacts.

For round $r = 0,\dots,R$:

1. **Reference simulations.** Draw $\theta$ from $\tilde\pi_r$ (restriction-filtered,
   so non-viable $\theta$ never burn a sim), simulate through the real forward model.
   The sim cache serves repeats.
2. **Fit from the batch, inline.** A $\theta\to\text{obs}$ **emulator** (with a
   per-observable approximation-error budget); the spread Fisher $G$ over
   spread-feeding rows and the center Fisher $G_\mu$ over all six sources, from the
   same local Jacobian and whitened as in §*Which components of $\mu$ are inferred*;
   and the two bases $W = \texttt{prior\_metric\_eigenbasis}(G,\Gamma_\omega)$,
   $W_\mu = \texttt{prior\_metric\_eigenbasis}(G_\mu,\Gamma_\pi)$.
3. **Cohorts.** Draw $\varphi\sim\tilde h$; for each distinct `cohort_id`, draw an
   independent patient cohort from $F(\theta\mid\varphi)$ at that study's real $n$
   **through the same viability filter the reference sims used**, **emulate** it (free,
   so cohort *count* is not a cost), add per-patient assay noise and per-patient
   surrogate noise, and summarize at that target's anchor grid. Train the amortized NPE
   on the concatenated cohort summaries, with $\varphi=(\mu,\log\sigma_u,\eta,\delta)$
   as the label.
4. **Truncate.** Estimate the reachable set (the *predictive* envelope, not the
   observed sample range) and restrict $\tilde\pi_{r+1}$ to it. Log the fraction of
   step-3 cohort patients falling outside the round's simulation support; above
   threshold, widen rather than proceed.
5. **Stop** when *both* the truncation and the basis stabilize: the reachable set stops
   shrinking **and** the principal angles between the top-$K$ subspaces of $W_r$ and
   $W_{r+1}$ are small. The ESS of $w=h/\tilde h$ is logged alongside as a compute
   symptom (see *Reporting*), not as a convergence criterion.

The expensive step is the real forward model in (1); the emulator exists so that
per-cohort resimulation in (3) is affordable. It is **pool-bound**, so its validity
domain has to track $\tilde\pi$, which is why it is refit each round rather than built
once. Its approximation error is handled twice, on purpose: it inflates the VPC null
so a spread *miss* must exceed the surrogate error to be labeled (Ch. 5, detection),
and it enters the matched-footing correction so the *point estimate* of $\omega$ stays
unbiased (*Matched footing*, term 2).

### Cohorts are per study, not one shared cohort

Step 3 draws an **independent** cohort per `cohort_id`. The tempting shortcut is to
draw one virtual cohort and summarize every target from it at that target's $n$, but
that puts the same virtual patients in every target's summary, and real targets come
from different studies with disjoint patients. The training conditioning vector would
then carry cross-target correlation induced by shared patients that the observed
vector does not have, and the NPE would read the observed vector's *absence* of that
correlation as signal about $\varphi$.

It also contaminates acceptance criterion 4 below: "VP correlation structure shows the
expected mechanistic pairings" is only a statement about mechanism if the training
construction did not manufacture the pairings itself. Targets that genuinely were
measured on the same patients keep a shared cohort, which is what `cohort_id` records,
and that shared structure is then real and worth conditioning on.

### Patients are filtered the same way sims are

Step 1 rejects non-viable $\theta$ before spending a simulation; step 3 must reject
them before putting a patient in a cohort. Skipping that is not a bookkeeping nicety,
it is a fourth matched-footing failure, and unlike the other three it is
$\varphi$-dependent. Unfiltered, the training cohorts contain patients the emulator was
never trained on and that no real cohort could have contained. Filtered, the population
being fitted is

$$F_{\mathcal V}(\theta\mid\varphi)\;=\;F(\theta\mid\varphi)\,\mathbf 1[\theta\in\mathcal V]\,/\,Z(\varphi),$$

and $Z$ shrinks as $\sigma_u$ grows, so the map from $\sigma_u$ to cohort spread
saturates and $\sigma_u$ goes weakly identified at the top of its range, exactly where
the edge check lives.

Filtered is the right choice, and it is the one the fixed-cloud VPop
(`vpop/weighting.py`) has always made: its cloud is the plausible cloud, its weights
live on that cloud, and its virtual patients are draws from it. It comes with two
obligations. The **reported** virtual patients must be draws from the same
$F_{\mathcal V}$ that was fitted, not from the unrestricted $F$. And $Z(\hat\varphi)$
must be logged: a small $Z(\hat\varphi)$ means most of the nominal fitted population is
non-viable, and $\hat\omega$ is then describing an object nobody would sample from.

### Reporting under the hyperprior

The reported posterior is **under $h$**, not $\tilde h$, and the reweight is at the
hyper level (§*Which reweight is legitimate*):

1. Draw $\varphi$ from the amortized posterior $q(\varphi\mid x_{\text{obs}})$, mixed
   over the $M$ surrogate-noise copies of the observed vector (matched footing,
   term 2).
2. Weight $w=\exp\!\big(h.\text{log\_prob}(\varphi)-\tilde h.\text{log\_prob}(\varphi)\big)$
   (`reweight_to_prior`, self-normalized) and resample.
3. Emit the population, as **two objects that are not the same size**, below.

No patient-level $\pi/\tilde\pi$ weight is applied at any point. Doing so would
resample the virtual patients onto $\pi$ itself, i.e. onto $\sigma_u\approx1$,
discarding $\hat\omega$.

**The population, and our uncertainty about it, reported apart.** Ch. 1 says the
virtual population is the fitted $F_{\mathcal V}(\theta\mid\hat\varphi)$. An earlier
version of this section instead emitted
$\int F_{\mathcal V}(\theta\mid\varphi)\,p(\varphi\mid x)\,d\varphi$, drawing $\varphi$
and then $\theta\mid\varphi$ per patient. That marginal is **wider**: it convolves
between-patient variability with hyperparameter uncertainty, which at $n=6$ per target
is not small. A chapter organized around not over-dispersing the population should not
quietly ship the over-dispersed object. Both are legitimate and they answer different
questions, so report them separately:

- **The population.** $F_{\mathcal V}(\theta\mid\hat\varphi)$ at the weighted posterior
  median $\hat\varphi$, with a posterior band on each reported population quantile
  (recompute the quantile at each posterior draw). "Patients vary this much, and we
  know that to within this much."
- **A simulated trial.** Draw $\varphi$ **once per trial**, then that trial's $n$
  patients from $F_{\mathcal V}(\theta\mid\varphi)$. Drawing $\varphi$ per patient
  destroys within-cohort coherence, so every simulated trial comes out looking like the
  average one and between-trial variability vanishes, which is the opposite of what a
  trial simulation is for.

**The variance budget.** Along the $P-K$ sloppy directions the reported spread is the
omega prior, not an inference. That is correct Bayesian behaviour (the data are silent
there, so the prior is the answer) and it is also the thing a reader of a VPop report
will most reliably over-read, so it ships as numbers rather than as a caveat: the
fraction of $\operatorname{tr}\Gamma_\omega$ that is

- **inferred**, in the top-$K$ where the data moved $\sigma_u$, with the per-direction
  prior-to-posterior contraction (§*Which reweight is legitimate*);
- **asserted**, the sloppy complement riding the omega layer, broken down by which
  omega level supplied it (global default / role class / explicit / data-shrunk);
- **hyperparameter uncertainty**, the difference between the two population objects
  above.

One table, and it is the difference between "we inferred the population" and "we
inferred $K$ numbers".

**ESS is gated, as a numerical check, not as a conflict diagnostic.** These are two
separate claims and only the second is about misspecification. Under prior-data
conflict the ESS does collapse, but that conflict is detected in observable space
(Ch. 5) and the ESS is a downstream symptom, never the diagnostic. Independently of any
misspecification, an ESS of a handful of draws means the reported population is a Monte
Carlo artifact of those few draws, and shipping it is a numerical error. So the ESS
fraction carries a hard threshold with its own failure message ("the reported population
is not resolved by the available draws"), explicitly worded so it is not read as
evidence of conflict. `importance.py` already warns below 5%; here that warning is a
stop, not a note.

The remedy has a precedence order, because the two obvious ones pull opposite ways.
Raise the posterior sample count first. Do **not** narrow $\tilde h$: bracketing is a
validity requirement and the ESS is a compute one, so that trade buys a well-resolved
report of the wrong thing. If more draws are not enough, use the defensive mixture
(§*The hyper-proposal*), which floors the ESS at $\alpha$ by construction and keeps the
bracket.

**What "under $h$" means given truncation.** Self-normalization cancels the constant
normalizer of a truncated proposal exactly, which is why evaluating `log_prob` on the
untruncated base object is correct. The consequence worth stating: draws that all live
inside the truncated region $S$ reweight to the target *restricted to* $S$, not to the
unrestricted target. Combined with the envelope discussion below, this means the
generosity of the reachability envelope directly sets the support of the reported
population. That is defensible (we do not want to report patients the mechanism cannot
reach) but it is a modelling choice, not a free consequence of the arithmetic, and a
too-tight envelope shows up here as a population reported too narrow. Report the
fraction of the population's mass the envelope truncated, per direction; it is cheap
and it is the only thing that makes the choice visible.

## Matched footing

The amortized NPE is calibrated only if the observed conditioning vector and the
training-cohort summaries carry the **same** noise model. A noise term present on one
side but not the other is read as signal and biases $\varphi$. **Three** such terms
matter at the summary level, and they are not all missing on the same side, so the
corrections run in different directions and do not cancel. A fourth, viability
filtering, lives at the patient level and is covered above in §*Patients are filtered
the same way sims are*.

1. **Measurement error on the center** (the $\mu$-only targets). The observed median
   already contains its own measurement noise, which is exactly what the target's CI
   reports, while the emulated training cohorts produce a clean median at large $n$
   (§*Cohort size for a $\mu$-only target*). So the correction adds fresh
   $\mathcal N(0,\sigma_c^2)$ to the *training* median summary, with
   $\sigma_c=(\log\text{hi}-\log\text{lo})/(2\,z_{.975})$ from the CI (the log-space
   center SD already formed inside `_ci95_expand`). Omit it and $\mu$ is overconfident
   in whatever direction that target constrains. The two population-spread sources need
   no such term *for the center*: their center footing is already matched by the
   finite-$n$ median law that the observed anchor and the training summaries share.
   That argument is about the median only, and it does not extend to the spread, which
   is term 3.
2. **Surrogate error** (every observable). The emulated training cohorts carry the
   emulator's approximation error $\varepsilon(\theta)$; the observed summary is a real
   measurement and carries none. So the correction adds the per-observable surrogate
   budget (the emulator's held-out residual SD, the same budget `inflate_cloud` folds
   into the VPC null, Ch. 5) to the **observed** side. Omit it and the NPE charges
   $\varepsilon$'s extra training spread to $\sigma_u$, biasing $\hat\omega$ **low**,
   the precise failure this whole path exists to prevent.

   It has to be added **at the patient level**, for exactly the reason term 3 gives.
   $\varepsilon$ enters each emulated patient before the cohort is summarized, so it
   inflates the training spread anchors, and the IQR of a noisy cohort is not the IQR of
   a clean cohort plus anything. Adding a budget to the observed *summary* cannot
   reproduce that law. For the two spread-feeding sources, add $\varepsilon$-draws to
   the observed per-patient values before summarizing (those targets have per-patient
   values, which is what makes them spread-feeding); for $\mu$-only targets, where only
   the median is used, adding at the summary level is exact.

   Two riders. First, what gets added is a **draw**, so a single one makes $\hat\varphi$
   seed-dependent, which is the objection §*On subsampling* raises against a seeded
   subsample. Draw $M$ perturbed copies of the observed vector, run the amortized
   posterior on each, and mix the $M$ posteriors before the reweight. Second,
   $\varepsilon(\theta)$ is a smooth deterministic field, not iid noise; using its
   marginal residual SD as iid patient-level noise is an approximation whose sign for a
   spread anchor is not obvious, since a field correlated with the biological gradient
   can inflate or deflate it. It is the best available correction, not an exact one, and
   it is a reason to keep the emulator's error budget small rather than to lean on the
   correction.
3. **Per-patient assay noise on the spread** (the two population-spread sources). Each
   observed per-patient measurement carries assay noise on top of that patient's
   biological value, so the observed IQR estimates $\omega\oplus\text{technical}$, while
   the emulated training patients are clean. So the correction draws per-patient noise
   at the target's `cv_technical` and adds it to each **emulated training patient**
   before the cohort is summarized, not to the summary afterward. Omit it and the NPE
   charges the assay's extra observed spread to $\sigma_u$, biasing $\hat\omega$
   **high**.

   The alternative, deconvolving the technical component out of the observed IQR, is
   rejected: it can go negative at small $n$, and it is the wrong direction under the
   rule below. Note also that this bias runs *opposite* to term 2, on a different set of
   targets and with an unrelated magnitude, so the two do not offset and neither excuses
   the other.

These are the same rule seen from three sides: **every summary-noise term appears on
both the observed vector and the training summaries; wherever one side already carries
it, add the complement to the other, at the level the noise actually enters.** (1) adds
to training because the observed already has the noise; (2) adds to the observed, at the
patient level, because training already has it; (3) adds to training, at the patient
level, because the observed patients already had it. The budgets are not free parameters
($\sigma_c$ is the reported CI, the surrogate budget is the emulator's held-out residual
SD, `cv_technical` is the target's reported assay precision), so matched footing costs no
tuning, only bookkeeping.

A target with no reported assay precision is the one place this needs a judgement call.
Treat it as a provenance gap, not a zero: either carry a conservative platform-level
`cv_technical` for that assay class, or demote the target to $\mu$-only. Silently
assuming clean per-patient measurement is the option that biases $\hat\omega$ high
without leaving a trace.

## Discrepancy: where structural misfit goes

Matched footing makes the *noise* symmetric. It says nothing about the mechanism being
wrong, and a hierarchical fit with nowhere to put structural misfit does not report it,
it absorbs it. The residual has to go somewhere, and the only places available are
$\hat\mu$ (a compromise center that fits no target well) and $\hat\omega$ (spread
inflated to cover targets one center cannot reconcile). Neither is labelled as misfit in
the output.

The fixed-cloud fallback already carries the correction the primary path lacks:
`vpop/weighting.py` estimates a model-discrepancy variance $\tau^2$ by method of moments,
as the part of the model-versus-data gap that sampling noise alone cannot explain, and
reports it on every fit. The hierarchical path needs the same object, inferred rather
than moment-matched, and it needs it at two levels because two different things go
wrong:

$$\text{study offset}\quad \eta_s \sim \mathcal N\!\big(0,\ \tau_\eta^2\,\Gamma_\pi\big),
\qquad
\text{target discrepancy}\quad \delta_j \sim \mathcal N\!\big(0,\ \tau_\delta^2\big).$$

$\eta_s$ is shared by every target carrying the same `cohort_id`. Studies differ in
inclusion criteria, line of therapy, and assay platform, and their medians will not be
reconcilable under one $\mu$. This is the between-study heterogeneity that Ch. 2's
translation sigma handles at the prior level, applied one level up, and its absence is
why a package that takes $\tau^2$ seriously for the prior currently takes nothing
seriously for the cohort. $\delta_j$ is per-observable and shifts one readout: the
mechanism is wrong *there*, in a way no population parameter fixes.

The two are separately identified whenever studies carry several targets each ($\eta_s$
moves a study's targets together, $\delta_j$ moves one target across studies), and both
are separated from $\sigma_u$ by the fact that they shift centers while $\sigma_u$ moves
spread anchors. Both enter the NPE as extra labels, drawn from the same law under $h$
and $\tilde h$ so they cancel from the reweight, and both are reported:
$\hat\tau_\delta$ is a misspecification statistic in the same currency as the fallback's
$\tau^2$, and the per-target $\hat\delta_j$ are the ranked list Ch. 5's diagnostics hand
to the mechanistic trace.

The cost of adding them is that $\hat\omega$ comes out *smaller* and its interval comes
out *wider*, because spread that was silently covering structural misfit now has
somewhere honest to go. That is the correction working, not a regression.

## Validating the shipped path

The gate is **SBC on synthetic truth over the full path**, not "train and infer". Draw
$\varphi^\star$ from the **hyperprior** $h$, synthesize an observed cohort from it
(through the same per-study cohort construction, the same viability filter, and the same
matched-footing noise terms the real observed vector carries), run the loop (simulate
from $\tilde\pi$, refit the bases, train on $\tilde h$, truncate, reweight to $h$), and
rank $\varphi^\star$ in the reported $\varphi$-posterior. This uses the **real**
importance weights at the calibration step (weighted SBC + joint TARP,
`inference/sbc.py`). Ranking $\varphi^\star\sim h$ inside importance-weighted draws is
the only check that blesses train-on-$\tilde h$-report-under-$h$; ranking in *unweighted*
draws would bless a pipeline we do not run. The bounded-ratio construction of §*Which
reweight is legitimate* matters here too: an unbounded $h/\tilde h$ makes the weighted
rank statistic itself ill-behaved.

Two properties of the gate follow from the fixes above and are easy to lose:

- The ranked object is $\varphi$, matching the reported posterior. An earlier
  formulation ranked against a resampled $\theta$-cloud, which is a different object and
  does not test the estimator being shipped.
- Each replicate **re-derives $W$, $W_\mu$ and the emulator from its own reference
  batch**. Freezing them across replicates would hide basis instability and the
  emulator's round-to-round domain shift, both of which are live failure modes here.

**What the gate costs, and what gets relaxed.** Taken literally, "run the whole loop" at
the ~100 replicates a rank ECDF needs is ~100 full analyses of real ODE simulations,
which is not a budget anyone has. Rather than describe a gate that will not run, state
the split:

- **Emulator-conditional SBC** (the gate that runs). One shared real-simulation pool
  across replicates; each replicate re-derives $W$, $W_\mu$, the emulator, the
  truncation, and the cohort construction from that pool. Everything the two properties
  above require still varies per replicate. The single approximation is that the pool
  does not chase each replicate's truncation, which understates round-to-round domain
  shift and nothing else.
- **Emulator fidelity**, validated separately on held-out real simulations inside the
  final round's $\tilde\pi$, since that is precisely what a shared pool cannot test.

Report which was run. A gate described but not executed is worth less than a smaller
gate that is.

Acceptance:

1. **Calibrated.** Flat rank ECDFs for the inferred direction spreads, for the inferred
   center components, and for the discrepancy scales (weighted SBC gate passes; TARP
   coverage near diagonal on the joint $\varphi$).
2. **Recovery** of the top-$K$ direction spreads matches the offline identifiability
   ceiling; a large gap is a wiring bug. Read this as a wiring check only: the ceiling
   comes from the same linearized $G$, so it cannot detect nonlinearity (criterion 1 is
   what does).
3. **$K$ is not over-reaching.** Run the gate at $K'=K+m$, inferring $m$ directions
   beyond the ones claimed, and require *those* to come back at their prior with no
   spurious contraction. The earlier form of this criterion ("sloppy directions ride the
   prior") could not fail: the sloppy $\sigma_u$ are fixed at 1 in both $h$ and
   $\tilde h$ and are not NPE labels, so their posterior is a point mass at the prior by
   construction. The $K'$ form is a real test, and it is the only thing in the suite that
   validates the choice of $K$.
4. **VP correlation structure** shows the expected mechanistic pairings; random pairs
   $\approx0$. Two contaminations to keep in view: it is a statement about mechanism only
   once cohorts are drawn per `cohort_id`, and on the sloppy complement it is largely
   reproducing $R$, the *epistemic* correlations inherited from $\pi$ (§*The four
   distributions*). Read it as a check on the inferred subspace, not on the whole
   population.
5. **No silent extrapolation**, on all three surfaces. In $\varphi$-space: the edge check
   on $\tilde h$ never fires unresolved. In $\theta$-space: the fraction of cohort
   patients outside the round's emulator support stays under its threshold. In
   $x$-space, which nothing else covers: $x_{\text{obs}}$ sits in the bulk of the cohort
   summaries $\tilde h$ generated. The edge check is on the *posterior*, so it cannot
   catch a conditioning vector far from anything the network trained on that nonetheless
   yields a comfortable interior posterior, which is the documented amortized failure
   mode (Schmitt et al. 2023; Ward et al. 2022). The check is a Mahalanobis discrepancy
   of $x_{\text{obs}}$ against the training $x$ cloud, the same machinery as
   `sbi_self_reference_null` one level up, and it costs nothing because the training
   summaries are already in hand.

## What this represents, and what it does not

Two claims a VPop report can make, and they are very different sizes: "patients vary
this much", and "we inferred that patients vary this much". This section is the
accounting that keeps them apart, and the comparison with the fixed-cloud construction
the package still ships.

**Represented well.** Data-side uncertainty, and better than is usual in this
literature. Finite $n$ is priced: training cohorts are summarized at each target's real
published $n$, so a 6-donor IQR constrains $\omega$ weakly and a 900-patient one
constrains it tightly, and both sides of the conditioning vector carry the same
finite-sample law. Provenance is enforced at the schema level rather than by convention,
so a `technical` replicate SD cannot silently become population spread. Noise footing is
symmetric in all four terms. And in the top-$K$ directions you get a genuine posterior
over $\varphi$, propagated rather than collapsed, checked by a calibration gate.

**Not represented.** Three things, and they belong in the report, not only here.

*Most of the reported spread is assertion.* Along the $P-K$ sloppy directions the
population is exactly the omega layer. That is the correct posterior, and the variance
budget (§*Reporting*) is what makes it visible instead of implied.

*Everything is conditional on plug-ins.* $W$, $W_\mu$, the emulator, $K$, $K_\mu$ and
the truncation set are each estimated or chosen and then conditioned on, so the reported
object is $p(\varphi\mid x, W, \hat g_{\text{emu}}, K, S)$. This is a stronger argument
for the per-replicate gate than the one given above: re-deriving the bases and the
emulator inside each SBC replicate is precisely what prices those plug-ins into the
calibration. If the gate is relaxed to hold them fixed, the intervals go back to being
conditional and must be labelled that way.

*The tails are assumption.* $F$ is log-Gaussian and the data are location and scale
anchors. Nothing in the acceptance criteria touches shape. VPops get used for tail
questions ("what fraction of patients respond?"), which is the one thing the data never
constrained. The fixed-cloud construction is better here, which is the next point.

### Against the fixed-cloud VPop (Allen et al. 2016)

`vpop/weighting.py` is not simply the weaker option. The two constructions fail in
opposite directions, and the honest summary is that they regularize the same problem at
different times.

The hierarchical path's decisive advantage is that prevalence weighting returns a
**point estimate**. One weighting, no posterior, no band on the population. Generating
several VPops samples the non-uniqueness of the solution, not posterior uncertainty, and
the maximum-entropy tie-break collapses that non-uniqueness by fiat anyway. If the
question is "what fraction of patients respond, and how sure are we", the fixed-cloud
construction structurally cannot answer it. That is the reason to pay for this
machinery.

Three places the fallback is ahead:

1. **It is nonparametric in the population.** The tilted cloud can be bimodal or skewed,
   and the quantile-bin constraints match shape by design. A log-Gaussian in an
   eigenbasis cannot. Where real per-patient samples carry structure, the fallback
   preserves it and this path erases it.
2. **The plausibility filter is coherent by construction.** The cloud is the filtered
   cloud, the weights live on it, and virtual patients are draws from it. That is the
   discipline §*Patients are filtered the same way sims are* has to import.
3. **It carries a discrepancy statistic.** `tau2`, method of moments, on every fit.
   §*Discrepancy* is this path catching up.

And the sharp criticism of the fallback, which is the same criticism this chapter makes
of a patient-level reweight. Maximum-entropy weighting is the I-projection of the **cloud
generator** onto the constraint set, and that generator is a deliberately generous
population prior. So in every direction not pinned by a bin constraint, the reported
population's width is the proposal's width. The answer inherits the proposal,
structurally, and toward over-dispersion rather than toward the anchor. It cannot be
removed from that construction, only made explicit.

Seen that way, the ESS wall and the identifiability wall are one phenomenon with two
symptoms. Prevalence weighting fits an infinite-dimensional object (a distribution on the
cloud) against finitely many bin constraints and regularizes *after*, with maximum
entropy picking a point in the null space. The eigenbasis restricts the unknown to $K$
numbers *before* fitting. Restricting before is more honest about what the data
identified; it also bakes the restriction into the answer, and $K$ carries no uncertainty
of its own, which is what acceptance criterion 3 now tests.

Two things follow. The provenance split is orthogonal to the fitter and should be ported
down: `fit_prevalence_weights` takes any per-patient array, with nothing enforcing that it
is genuine across-patient variability. And there is a hybrid worth building: use the
fitted $F_{\mathcal V}(\theta\mid\hat\varphi)$ as the reference measure for a light,
strongly-ridged prevalence tilt. The reference is then a fitted population rather than a
generous prior, which removes the fallback's worst property, while the tilt recovers the
shape the log-Gaussian cannot express. The ESS of that tilt is a direct measure of how
much shape the parametric family missed.

## Where the code lives

The generic machinery is this package; a downstream project supplies only its prior,
scenarios, and forward model as a thin caller.

| Piece | Home |
|---|---|
| population omega prior (layered, MBMA-shrunk) feeding $\Gamma_\omega$ | `targets/omega.py` |
| observed quantile anchors + provenance (`feeds_spread`) | `targets/anchors.py` |
| eigenbasis (Fisher $G$ and $G_\mu$, prior-metric whitening, draw/project matrices) | `vpop/eigenbasis.py` |
| population/proposal $\tilde\pi$ (log-normal population, widen-on-identified, reachability accept-fn) | `vpop/proposal.py` |
| hyperprior $h$ / hyper-proposal $\tilde h$ over $\varphi$, edge check, discrepancy layer | `vpop/hyperprior.py` (to build) |
| per-study cohort construction, viability filter, the three matched-footing noise terms | `vpop/cohorts.py` (to build) |
| importance reweight ($w=h/\tilde h$, ESS, weighted quantiles) | `inference/importance.py` |
| TSNPE truncation substrate (density thresholder → restricted proposal) | `inference/` |
| weighted SBC + TARP gate | `inference/sbc.py` |
| VPC null with emulator-error inflation | `inference/predictive_checks.py` |
| prevalence-weighting fallback (fixed-cloud VPop, Allen 2016) | `vpop/weighting.py` |

The prevalence-weighting construction in `vpop/weighting.py` is the older fixed-cloud
alternative: reweight an existing simulation cloud to the observed marginals instead of
inferring the width. §*Against the fixed-cloud VPop* is the actual comparison; the short
version is that it is a point estimate with no uncertainty about the population, and it is
ahead of this path on population shape, on plausibility bookkeeping, and (until
§*Discrepancy* lands) on model discrepancy. It is also ahead on one thing this chapter
cannot easily replace: because a single weight vector has to satisfy every observable's
marginal at once, its ESS is a **joint-compatibility statistic**, and its degeneracy
carries information about misspecification that a parametric fit does not surface.
[Chapter 4b](population-inference-tractable.md) is built around that observation.

## Open work

This chapter describes the path as it should be, which is ahead of what is built. The
table tracks the gap. "Changes the answer" marks items where the current description or
code produces a materially different $\hat\varphi$, not just a weaker guarantee; those are
the ones to land first.

| # | Fix | Where | Changes the answer | Status |
|---|---|---|---|---|
| 1 | $\Gamma_\omega$ (layered omega) as the population geometry, distinct from $\Gamma_\pi$ (epistemic) for the center | `vpop/eigenbasis.py` callers, `targets/omega.py` | **yes** (it sets the whole sloppy complement's spread) | documented, not built |
| 2 | Reweight at the hyper level, $w=h/\tilde h$ on $\varphi$; no patient-level weight in the report | `vpop/hyperprior.py`, report path | **yes** (a $\theta$-level reweight discards $\hat\omega$) | documented, not built |
| 3 | $h$'s $\sigma_u$ law written down ($s$ declared); $\tilde h$ two-sided in $\log\sigma_u$; defensive mixture | `vpop/hyperprior.py` | **yes** (a one-sided $\tilde h$ biases $\hat\omega$ high) | documented, not built |
| 4 | Discrepancy layer: study offset $\eta_s$, target discrepancy $\delta_j$ | `vpop/hyperprior.py`, `vpop/cohorts.py` | **yes** ($\hat\omega$ currently absorbs structural misfit) | documented, not built |
| 5 | $G$ from spread-feeding rows only; $K\le\operatorname{rank}(G)$ enforced, not read off the spectrum | `vpop/eigenbasis.py` | **yes** | documented, not built |
| 6 | Viability filter on cohort patients; $Z(\hat\varphi)$ logged; report drawn from the same $F_{\mathcal V}$ | `vpop/cohorts.py` | **yes** | documented, not built |
| 7 | Matched footing term 3: per-patient assay noise on emulated patients before summarizing | `vpop/cohorts.py`, target schema `cv_technical` | **yes** (biases $\hat\omega$ high) | documented, not built |
| 8 | Matched footing term 2 moved to the patient level, averaged over $M$ noise draws | `vpop/cohorts.py`, report path | **yes** (biases $\hat\omega$ low; opposite sign to 7, so land them together) | documented, not built |
| 9 | Independent cohort per `cohort_id` instead of one shared cohort | `vpop/cohorts.py`, target schema `cohort_id` | yes, and it contaminates acceptance criterion 4 | documented, not built |
| 10 | $\mu$-only training summaries at large $n$, so no spread signal leaks from a $\mu$-only target | `vpop/cohorts.py` | yes | documented, not built |
| 11 | Predictive reachability envelope replacing the observed-sample range | `vpop/proposal.py` | yes (an observed-range envelope reports $\omega$ narrow) | documented, not built |
| 12 | `min_fraction` default below 1, or joint Mahalanobis acceptance instead of a marginal-box conjunction | `vpop/proposal.py:reachable_accept_fn` | yes at large $n_{\text{obs}}$ | documented, not built |
| 13 | $G_\mu$ built with center noise ($\sigma_c$, $\tau_{\text{trans}}$) over all six sources; its own $K_\mu$ | `vpop/eigenbasis.py` | yes (the SE-constant version was a no-op: $G_\mu\propto G$) | documented, not built |
| 14 | Two population objects reported apart; $\varphi$ drawn once per simulated trial; variance budget | report path | no, but it is what stops the report being over-read | documented, not built |
| 15 | Basis stability (principal angles) in the stop rule; per-round extrapolation fraction logged | loop runner | no, but it is how 11 and the round-to-round drift get caught | documented, not built |
| 16 | ESS gated as a numerical-validity stop, with the precedence order on the remedy | `inference/importance.py` caller | no (guards against shipping an unresolved report) | warning exists, gate does not |
| 17 | SBC ranks $\varphi$, emulator-conditional per replicate; $x$-space extrapolation check | `inference/sbc.py` caller | no, but the gate is not valid without it | documented, not built |
| 18 | Acceptance criterion 3 replaced by the $K'=K+m$ over-reach test | gate | no (the old form could not fail) | documented, not built |
| 19 | Docstring corrections: $G$ is neither offline nor prior-independent in this loop; "anchored on the sloppy complement" holds in $u$-coordinates only; `widen_on_identified` is for $\tilde\pi$, not $\tilde h$ | `vpop/eigenbasis.py`, `vpop/proposal.py` | no | documented, not applied |
| 20 | Per-anchor quantile SE constants instead of `se_iqr_c` everywhere | `vpop/eigenbasis.py` | marginal | documented, not built |
| 21 | Prefer the full reported sample over a seeded $n$-subsample; if subsampling, average over seeds | anchor construction | marginal | documented, not built |
| 22 | Port the `spread_source` provenance split into the fixed-cloud fallback | `vpop/weighting.py` | yes, for the fallback | documented, not built |

Four items are worth flagging as still open *questions* rather than pending work. The
predictive envelope (11) trades the observed range's finite-$n$ pathology for a dependence
on $\tilde h$'s bracketing, which is an assumption, not a measurement; if $f$ is chosen
badly the envelope inherits it. Terms 2 and 3 of matched footing bias $\hat\omega$ in
opposite directions on overlapping sets of targets, so a path implementing only one may
look *better* calibrated than one implementing neither while being no more correct. The
correlation structure $R$ in $\Gamma_\omega$ is epistemic and used as biological, and
nothing in the package currently offers an alternative; it is a declared assumption whose
only honest treatment today is a sensitivity run against a diagonal $R$. And the
$\hat\omega$-versus-$\hat\tau_\delta$ trade in §*Discrepancy* is a real identifiability
question at small numbers of studies, not just an implementation detail: with one study
and one target per observable, $\delta_j$ and a mis-centered $\mu$ are the same thing.

## Reachable-set truncation

This is the step-4 truncation, and it is **load-bearing, not a diagnostic**: it decides
where the next round's real simulations go, hence the region the emulator is trained to be
accurate on. (Distinct from the *labeler's* reachability in Ch. 5, "could any $\theta$
reach $x_{\text{obs}}$?", which is a pure diagnostic and does not touch the fit.)

"Reachable set" for a VPop has no single $x_{\text{obs}}$ to threshold against, since the
target is a distribution, so the truncation is defined in **observable space against the
data**: keep $\theta$ whose per-patient (emulated) observables land inside the observed
envelope. Concretely, the same `RestrictedPrior` substrate as flat TSNPE, but with the
density thresholder swapped for an accept-fn that asks *"are this $\theta$'s emulated
observables inside the observed envelope?"* (`vpop/proposal.py:reachable_accept_fn`),
which is free since the round's emulator is already in hand.

This is chosen over the alternative, truncating $\tilde\pi$ to the high-density region of
the *fitted* $F(\theta\mid\hat\varphi)$, for three reasons, all sharper because the
inferred quantity here is a **spread**:

1. **No feedback into the estimate.** Truncating on the fitted population couples the
   proposal to the very $\sigma_u$ we infer: an under-estimate in round $r$ narrows
   $\tilde\pi$, tightens round $r{+}1$'s cohorts, and self-fulfils. A data-anchored
   envelope has no such loop.
2. **Right-sized emulator domain.** Cohort patients are drawn from the *wide*
   hyper-proposal, broader than the fitted $F$; a fit-truncated $\tilde\pi$ would leave
   the emulator extrapolating on exactly the tail patients that carry the spread signal.
3. **Matches the reachability definition.** The labeler defines reachability in
   observable space ("could a physical $\theta$ produce $x_{\text{obs}}$?"), not by a
   parameter-space density.

### The envelope must be predictive, not the observed range

Reason 2 above is the sharp one, and taken seriously it also indicts the *naive* data
envelope. If the envelope is the observed `samples`' own range, it is a **finite-sample**
band from $n=6$ to $n=30$ patients. A population whose spread genuinely matches the data
has patients well outside that band, and those are precisely the tail patients reason 2
identifies as carrying the spread signal. After truncation the emulator is refit only
inside the band, so it extrapolates on them anyway, and $\hat\omega$ is read partly off
extrapolated output. The observed-range envelope avoids the *feedback* pathology of reason
1 while reintroducing the *coverage* pathology of reason 2.

So the envelope is defined **predictively**: the range a cohort of size $n$ could produce
under the widest $\varphi$ in $\tilde h$'s bulk, computed from the round's emulator and
$\tilde h$. That is the region the emulator actually has to be accurate on, which is the
only thing the truncation is for. It is anchored to the data through $\mu^\pi$ and
$\Gamma_\omega$ and through $\tilde h$'s bracketing of the observed spread, so it inherits
none of the fitted-$F$ feedback loop of reason 1: it depends on the *support* of the
hyper-proposal, not on the current estimate.

Sources still follow the center-vs-spread split: `center_only` targets have no observed
population distribution and contribute no spread envelope. The one requirement is
unchanged and now easier to satisfy honestly: keep the band **generous**, or the
population's legitimate tail patients get cut and $\omega$ is reported too narrow, the
failure this whole path exists to avoid.

**Instrumentation, every round.** Log the fraction of cohort patients whose $\theta$ falls
outside the current round's simulation support. Above threshold, $\hat\omega$ is being
read off extrapolated emulator output and does not ship; the response is to widen
$\tilde\pi_{r+1}$ or lower the emulator's error budget in that region, not to proceed.
This is criterion 5 of the gate.

**Conjunctive acceptance is tighter than it looks.** `reachable_accept_fn`'s
`min_fraction=1.0` requires *every* observable inside its box, so acceptance decays roughly
geometrically in the number of observables and the effective truncation is far tighter than
any per-observable width suggests. Prefer a joint distance (Mahalanobis in the emulator's
output space) over a conjunction of marginal boxes; if the box form is kept, `min_fraction`
defaults below 1 and its value is reported alongside the acceptance rate.

Honest caveat, retained: the "support" of a finite sample is not a clean object, the
observed min/max grows with $n$, and a low-$n$ target ($n=6$) barely defines a band. The
predictive envelope replaces the sample range with something better posed, but it inherits
$\tilde h$'s bracketing assumption in its place. The truncation it drives stays
correspondingly soft, which is why it only *shrinks* $\tilde\pi$ toward the region the data
could have come from and never sets the population geometry (that is the omega layer's job,
always).

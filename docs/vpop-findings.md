# VPop fit: what is measured

Results as of 2026-08-11, against `docs/model-draft.tex`. Configurations are
named by their directory under the vpop scratch. Every convergence number is
split R-hat and n_eff over four chains at n_cloud 250, 300 warmup and 300
samples, `--max-tree-depth 8`.

## The best configuration

`fixs_08e/c1`: discrepancy free, `b_1` pinned, `Z_b = same`, `--fix-s`,
`initial_tumour_diameter` pinned in both `mu` and `omega`, `tau_eta 0`, Laplace
metric at the prior with adaptation off.

| site | R-hat median / max | n_eff |
| --- | --- | --- |
| `omega` | 1.010 / 1.212 | 290 |
| `u_free` | 1.010 / 1.199 | 283 |
| `mu` | 1.046 / 2.420 | 91 |
| `b_free` | 1.147 / 1.283 | 17 |

`omega` has no coordinate past R-hat 1.3. `mu` has sixteen.

## eq:omegaassumed's global level does nothing

Linearised at the prior with `u` and `b` free, the 17 width rows prefer
`s = -0.003` at a one-prior-sd step and `-0.014` unpenalised, while `u` reaches
1.04 and `b` 0.95. `s` alone takes the width residual from 4.921 to 4.858 and
raises the count past 2 sd from 12 to 13, because the rows want opposite signs.

A one-dimensional profile with `u` held at zero puts `s` at +0.5 with curvature
5516. That identification is an artifact of pinning what `s` is redundant with.

Pinning `s` takes `omega`'s worst coordinate from 2.748 to 1.212 and its n_eff
from 130 to 290, `mu`'s n_eff from 29 to 91, and `u_free`'s from 228 to 283.
The cost is that `omega`'s global level is asserted by `omega_0` rather than
fitted.

## eq:studyeff helps only while `s` is free

At matched settings with `s` free, `tau_eta 0.5` takes `mu`'s worst coordinate
from 3.498 to 1.931 and its n_eff from 18 to 44. With `s` pinned it reverses:
`fixs_08e/c3` against `c1` is n_eff 46 against 91 on `mu` and 114 against 290 on
`omega`.

A per-study *width* offset is not supported by the corpus. Thirteen studies
print width rows, nine of them exactly one, so 13 offsets would explain 17 rows
and leave four residual degrees of freedom. `eta` is well determined by
comparison: li2022 alone prints 58 location rows.

## The corpus against the prior predictive

Every row's printed value against the model's expectation of it, at
`(mu_0, omega_0)`, ungated: location rows median `log r` -0.234 with 33 of 121
past 1 and 8 past 2; width rows median -0.255 with 5 of 17 past 1.

The corpus carries 138 rows over 29 cohorts and 22 studies, of which 121 are
location rows. `omega` rests on the remaining 17.

Between-source disagreement is real. Converted to a common footing, the implied
population sd of baseline CD8 density is 26.3 (liu2015), 50.9 (golesworthy2022)
and 288 (jansen2021, from `se` rows). No term in the model is indexed to absorb
it: `Z_b` is kind by assay, and all four rows share their columns.

## Eligibility

36.5% of pool draws are status 5 (too_slow) and 12.3% status 4, so roughly half
the ungated population could not have been diagnosed.

With `gates/gate_08d_full.npz` (AUC 0.9704), `tumor_doubling_time/mean` moves
from `log r` -6.060 to -1.813, `/sd` from -7.858 to -2.999, and location rows
past `|log r| > 1` from 33 to 30. Width rows past 1 go the other way, 5 to 8.

The gate inside the log density costs 60x to 90x in step size: `fit_08e_gate`
ran at 2.5e-04 to 4.9e-04 against `fit_08e_off`'s 8.3e-03 to 2.3e-02. Freezing
the weights at a reference does not recover it. Paired between the prior and a
fitted point the weights correlate 0.46, and `|dtau|/sd` under frozen weights is
worse than under no weighting at all, median 0.361 against 0.273.

`gate_08d_best.npz` is the best of a hyperparameter sweep at AUC 0.8627, not the
best gate.

## The surrogate

`tumor_doubling_time@0` is the emulator's worst column on the raw scale,
`r2_min -5.29`. On the log scale, which is what `h_fn` emits and the fit
compares on, it is R2 0.707 with residual sd 0.635, against a median of 0.911
and a minimum of 0.660 over the other 53 readouts. The surrogate does not
inflate the tail: 9.1% of its values exceed 976.8 days against the simulator's
10.5%.

`pool_readouts` built the precomposed channel once from `arm_forward` and handed
it to both sides, so those rows differenced to exactly zero and carried no `E`.
The pool has always carried `sp:tumor_doubling_time@0`; it was never read
because `sp:` columns are selected by species name. Fixed in pdac-build
`31a8046d`.

## Growth rate

Among eligible pool patients the simulator's tumour volume doubling time has
median 182.6 days and mean 1006, against Ahn 2016's 132.3 +/- 132.1 over n=100
with a range of 20.0 to 976.8. 10.5% of eligible patients grow slower than any
of Ahn's 100.

## A fraction travels a log-odds channel

`h_r` returned a log for every readout and eq:disc acts there, so `kappa` widened
a bounded quantity with no ceiling. A `stromal_fraction` cloud spanning 0.68 to
0.918 about a pivot of 0.76 reaches 1.105 at `kappa` 1.85, and `to_scale`'s logit
tangent, slope 1001 above `1 - 1e-3`, turns that 7% overshoot into a 132-sigma
residual on one row. 27 of 54 readouts are a bounded kind.

Carried as a log-odds and inverted through a sigmoid, no `kappa` up to `e^4`
leaves (0, 1) on any of the 27. The plug-in is untouched, since
`sigmoid(logit(f)) = exp(log(f))`: the corpus audit at the prior reproduces all
four summary statistics to the digit.

`logit_08e/c1` against `fixs_08e/c1`. `omega` improves sharply and `mu` degrades:

| site | pass R-hat <= 1.01 | n_eff median |
| --- | --- | --- |
| `omega`, baseline | 137 / 271 | 290 |
| `omega`, logit | 238 / 271 | 1848 |
| `mu`, baseline | 19 / 271 | 91 |
| `mu`, logit | 25 / 271 | 28 |

`stromal_fraction`'s location miss falls from 131.70 sigma to 1.10.

## The corpus states about fifteen things

`S[i,j]`, how many of row `i`'s sd the prediction moves per prior sd of `mu_j`,
is 138 by 271 and its spectrum collapses: 12 directions carry 90% of the
variance, 16 carry 95%, 28 carry 99%, participation ratio 5.6. Measured at
`logit_08e/c1` it is tighter still, 9 / 14 / 27 and 2.9.

No parameter is inert. The floor is `k_Cy_clear` at 0.041 row-sd per prior-sd,
0 of 271 fall below 0.01, and at the fitted point none falls below 0.1. The
coordinates that will not converge are among the most influential rather than
the least: `f_iCAF_of_non_apCAF` ranks 2 of 271, `k_myCAF_to_iCAF` 6,
`f_stroma_max` 9. High leverage and rank deficiency are compatible when the
leverage is degenerate, and the rank deficiency is the aliasing.

Column-pivoted QR selects 30 parameters spanning 96.8% of `||S||_F^2`, 20 for
91.7%. Holding those coordinates is refused wherever `Sigma_1` correlates them,
which is most of them, so the coordinate reading is a record rather than a
configuration. The subspace reading is `mu_basis`.

## eq:disc is load-bearing for the sampler

`--pin`, a = b = 0 and `log R` held, is worse than the discrepancy in every
column:

| run | `mu` <= 1.01 | `mu` <= 1.05 | `mu` n_eff | `omega` <= 1.01 | `omega` n_eff |
| --- | --- | --- | --- | --- | --- |
| `logit_08e/c1` | 25 | 121 | 28 | 238 | 1848 |
| `pin_08e/fix_s` | 11 | 67 | 9 | 201 | 920 |
| `pin_08e/free_s` | 3 | 33 | 7 | 1 | 4 |

So eq:disc is not merely absorbing residuals: removing it costs `mu` half its
converged coordinates and two thirds of its effective samples. A discrepancy
competing with the mechanism would have predicted the opposite.

`s` stays pinned with `b` at zero. The argument that pinning every column of `b`
removes `s`'s only competitor and identifies it predicts `pin_08e/free_s`; that
arm has 1 of 271 `omega` coordinates below R-hat 1.01 and an n_eff median of 4,
the worst fit measured. `s`'s inertness was never about `b`.

Fit quality is not readable from either `--pin` arm. At n_eff 4 to 9 the
posterior mean is not a point the chains agree on. What eq:disc does to the rows
is measured at a converged point instead, below.

## eq:disc's spread carries the damage, not its offset

Split per row at `fixs_08e/c1`, `a` alone against `b` alone in row sd:

| row | z bare | z fit | a only | b only |
| --- | --- | --- | --- | --- |
| `cd8_fc` gvax_nivo | +0.54 | -7.27 | -0.91 | +9.24 |
| `cd8_fc` gvax_nivo | +0.40 | -6.50 | -0.46 | +7.57 |
| `cd8_fc` gvax | +4.24 | -3.23 | -0.46 | +8.18 |
| `cd8gzmb_fc` gvax | +0.11 | -4.33 | -0.49 | +4.95 |

`b` rescales the cloud before the statistic, so on a tail quantile it moves the
location. Every fold-change row is a tail quantile on n=6 or n=10, and `Z_b` is
the coarser design by construction. Corpus-wide `b` has p90 5.19 against `a`'s
3.05.

Net, eq:disc still helps: median |z| 1.52 to 1.27, 82 rows improved against 56.
The damage is narrow, six rows the mechanism fits at |z| < 2 pushed past 4.

Under the logit channel the two swap, `a` reaching p90 5.10 and `b` falling to
2.85. The channel did not remove the discrepancy's leverage, it moved it.

## Where the corpus fits worst

CD8, by a factor of two. 21 of 60 targets are CD8-related and carry median |z|
4.02 against 2.09 for the other 35; 11 of the 18 worst are CD8. The sharpest
signal is baseline CD8 density, missed in the same direction by three
independent studies with eq:disc absorbing +10 to +21 sd: jansen2021 dog1neg
(z -13.03, disc +20.68), golesworthy2022 (-7.46), jansen2021 dog1pos (-7.06).
These are asinh density rows, untouched by the channel change.

Direction 1 of `S`, sigma 36.4 against 24.6 for the next, is
`k_CD8_T_pro -0.80`. The corpus's loudest statement is about the compartment it
fits worst.

## Open

Sixteen `mu` coordinates past R-hat 1.3 at `fixs_08e/c1`, twenty-one at
`logit_08e/c1`. Four explanations have been tested: inert parameters do not
exist, the fraction channel helps `omega` and hurts `mu`, removing eq:disc is
worse, and `s` is inert either way. Rank deficiency is the one that survives.

`initial_tumour_diameter` is varied and conditioned on, not varied and inferred,
and is declared with the inferred parameters. Held out of the basis it costs
nothing measurable: 0.9744 of the sensitivity variance against 0.9738 free, same
leading directions. Taking it out properly means removing surrogate input 89 of
271 and retraining the arms; the pool does not need redrawing.

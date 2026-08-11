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

## Open

Sixteen `mu` coordinates past R-hat 1.3, led by `f_iCAF_of_non_apCAF` (2.420),
`f_stroma_max` (2.413), `CXCL13_50_TLA` (2.039) and `k_myCAF_to_iCAF` (1.655).
These are not unidentified: the worst twelve have moved a median of 0.79 from
`mu_0` against 0.20 corpus-wide. The same compartment holds the corpus's worst
location rows, `stromal_fraction/max` at `log r` +3.009 and
`stromal_fraction/q0.5` at z -7.67.

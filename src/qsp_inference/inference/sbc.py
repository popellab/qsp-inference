"""Simulation-based calibration for a posterior reported under a prior it was
not trained on.

Simulation-based calibration (Talts, Betancourt, Simpson, Vehtari & Gelman 2018)
rests on a single identity. If

    theta* ~ pi,    x ~ p(. | theta*),    theta_1..theta_L ~ p(. | x),

then the rank of ``theta*`` among the posterior draws is uniform on
``{0, ..., L}``. Uniformity is a property of the *pair* (prior, posterior
sampler): it is the statement that the sampler inverts the same generative model
the ground truth was drawn from. Break the pair and the identity goes with it.

That is exactly what the proposal/prior decoupling does. The estimator is trained
on pairs with ``theta ~ pi_tilde``, so it learns
``q(theta | x) ~ p(x | theta) pi_tilde(theta)``: a posterior under the *proposal*.
Reporting under the anchored prior ``pi`` is a reweighting,
``w(theta) = pi(theta) / pi_tilde(theta)`` (see
:mod:`qsp_inference.inference.importance`). Two checks live on this path and they
answer different questions:

1. Ranks of ``theta_test`` (drawn from ``pi_tilde``) in the *unweighted* draws.
   This asks whether the network fits its own training joint. It is a check on
   the estimator, it is what :func:`~qsp_inference.inference.diagnostics.
   sbi_calibration_ecdf` computes, and it can pass while the reported posterior
   is wrong.

2. Ranks of ``theta*`` drawn from ``pi`` in the *weighted* draws. This asks
   whether the composition (train on ``pi_tilde``, reweight to ``pi``) is a
   calibrated posterior under ``pi``. This is the gate, and it is what this
   module computes.

The distinction is the whole reason the module exists. Check 1 is blind to a
mis-specified reweighting because the weights never enter it, and it is blind to
regions where ``pi`` puts mass and ``pi_tilde`` does not, because it never draws
a ``theta*`` there. Only check 2 exercises the path that produces reported
numbers.

**The statistic.** With weights the integer rank is not available, so the
statistic is the posterior CDF evaluated at the truth, the probability integral
transform

    u = F_q(theta*) = sum_l w_l 1[theta_l < theta*],

which is uniform on ``(0, 1)`` in the limit of infinitely many draws. At finite
``L`` the unweighted version is the discrete rank, and the two are reconciled by
the mid-rank continuity correction that :func:`weighted_pit` applies; with
uniform weights it returns exactly ``(R + 1/2) / (L + 1)``, the standard
convention. See that function for the weighted generalization.

**Resolution.** The correction replaces ``L`` by the Kish effective sample size
of the weights, because that is the number of independent draws the weighted
sample is actually worth. This has a practical consequence worth stating plainly:
the resolution of the gate is set by the per-replicate ESS, not by how many
posterior draws were requested. Asking for 4000 draws from a proposal that
carries an ESS of 30 gives a rank statistic resolved to about 1 part in 30, and
a test built on it cannot see calibration defects finer than that. This is the
one place where ESS is a legitimate diagnostic quantity rather than a compute
number: it bounds what the gate can detect.

**Reading a failure.** A non-uniform PIT localizes: it names the parameter and
the direction (see :attr:`ParameterCalibration.shape`). It does not name the
cause. Draws piled at the edges mean the reported posterior is narrower than the
truth warrants, which is consistent with an under-trained flow, a proposal that
misses the region, a reweighting whose ESS is too low to resolve the tails, and
several other things. Deciding among them is a matter of looking at the model,
not of computing another scalar.

**What uniformity does not certify.** SBC averages over the prior predictive. It
can pass while the posterior at the one ``x_obs`` you care about is badly wrong,
because that ``x_obs`` may be nowhere near the bulk of ``p(x)`` (Modrak, Moon,
Kim et al. 2023 catalogue this and other limits). SBC is necessary, not
sufficient. Prior-data conflict at ``x_obs`` is a separate question, answered in
observable space by :func:`~qsp_inference.inference.diagnostics.
sbi_self_reference_null` and its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "weighted_pit",
    "weighted_sbc",
    "ParameterCalibration",
    "SBCResult",
    "SBCGate",
    "uniform_band_halfwidth",
    "plot_sbc_ecdf",
]

# Var(U) and Var(s^2) for U ~ Uniform(0, 1): the null moments the shape test uses.
_UNIF_VAR = 1.0 / 12.0
_UNIF_S2_VAR_COEF = 1.0 / 180.0  # Var(s^2) = 1 / (180 n)


def _as_array(a) -> np.ndarray:
    """Accept numpy arrays and torch tensors without importing torch."""
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float64)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------
def weighted_pit(
    posterior_samples,
    theta_star,
    weights=None,
) -> np.ndarray:
    """Weighted probability integral transform of the truth, one replicate.

    Args:
        posterior_samples: Draws from the trained posterior at this replicate's
            ``x``, shape ``(L, d)``. These are the *unweighted* draws the
            estimator produced; the reweighting enters through ``weights``.
        theta_star: The ground truth that generated ``x``, shape ``(d,)``. Must
            have been drawn from the prior being reported under.
        weights: Self-normalized importance weights ``pi / pi_tilde`` per draw,
            shape ``(L,)``. ``None`` means uniform, which reduces this to the
            ordinary SBC mid-rank.

    Returns:
        ``(d,)`` array of PIT values, strictly inside ``(0, 1)``.

    The correction. With uniform weights the raw statistic
    ``R / L = #{theta_l < theta*} / L`` takes ``L + 1`` values and is uniform on
    them, so a continuous-uniform test applied to it is biased: its mean is
    ``1/2 + 1/(2L)``. The standard fix is the mid-rank ``(R + 1/2) / (L + 1)``,
    whose mean is exactly ``1/2``. The ``+1`` in the denominator is the truth
    itself taking a slot among the draws, and the ``+1/2`` splits the interval
    the truth landed in.

    Extending this to weights means asking how many draws the weighted sample is
    worth. That is the Kish effective sample size ``m = 1 / sum w^2``, and the
    correction becomes

        u = (m F + 1/2) / (m + 1),   F = sum_l w_l 1[theta_l < theta*],

    which recovers ``(R + 1/2) / (L + 1)`` exactly when the weights are uniform,
    since then ``m = L`` and ``F = R / L``. Ties (probability zero for a
    continuous posterior, but not for a degenerate one) split their mass evenly.
    """
    s = _as_array(posterior_samples)
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2:
        raise ValueError(f"posterior_samples must be (L, d); got {s.shape}")

    t = _as_array(theta_star).ravel()
    if t.size != s.shape[1]:
        raise ValueError(
            f"theta_star has {t.size} entries but posterior_samples has " f"{s.shape[1]} columns"
        )

    n_draws = s.shape[0]
    if weights is None:
        w = np.full(n_draws, 1.0 / n_draws)
        m = float(n_draws)
    else:
        w = _as_array(weights).ravel()
        if w.size != n_draws:
            raise ValueError(
                f"weights has {w.size} entries but posterior_samples has " f"{n_draws} rows"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        total = float(w.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("weights must sum to a positive finite value")
        w = w / total
        s2 = float(np.sum(w**2))
        m = 1.0 / s2 if s2 > 0 else float(n_draws)

    below = (s < t[None, :]).astype(np.float64)
    tied = (s == t[None, :]).astype(np.float64)
    f = w @ below + 0.5 * (w @ tied)

    return (m * f + 0.5) / (m + 1.0)


# ---------------------------------------------------------------------------
# Assembling replicates into a verdict
# ---------------------------------------------------------------------------
@dataclass
class ParameterCalibration:
    """Per-parameter reading of the PIT sample across replicates.

    Three tests are run against ``Uniform(0, 1)``: a Kolmogorov-Smirnov test,
    which is omnibus, and two directional tests on the first two moments. The
    directional pair earns its place. The characteristic SBC failures are a shift
    of the mean and an inflation or deflation of the variance, and a test aimed
    at those has far more power against them than an omnibus one. A proposal
    tempered to ``T = 4`` against a likelihood that leaves the prior largely
    intact produces a PIT variance of about ``0.053`` against a null of
    ``0.083``: eight standard errors at 400 replicates, and a KS statistic that
    sits right on its critical value.

    The three are combined by taking the smallest p-value and multiplying by
    three. Bonferroni within a parameter is conservative and, unlike anything
    smarter, needs no assumption about how the KS statistic and the sample
    moments covary.

    Attributes:
        name: Parameter name.
        ks_statistic: Kolmogorov-Smirnov distance from ``Uniform(0, 1)``.
        ks_pvalue: Two-sided KS p-value.
        pvalue_mean: Two-sided p-value for ``mean == 1/2``, from ``z_mean``.
        pvalue_variance: Two-sided p-value for ``variance == 1/12``, from
            ``z_variance``. Asymptotic; read it with care below ~50 replicates.
        pvalue: The three combined, Bonferroni within this parameter.
        pvalue_adjusted: :attr:`pvalue` after Benjamini-Hochberg across the
            informative parameters. ``nan`` for parameters excluded as
            uninformative.
        mean: Sample mean of the PIT values. ``1/2`` under the null.
        variance: Sample variance. ``1/12`` under the null.
        z_mean: Standardized departure of ``mean`` from ``1/2``.
        z_variance: Standardized departure of ``variance`` from ``1/12``.
        informative: False when ``theta*`` barely varies across replicates, so
            there is nothing for SBC to check. Excluded from the multiplicity
            correction and from the gate.
    """

    name: str
    ks_statistic: float
    ks_pvalue: float
    pvalue_mean: float
    pvalue_variance: float
    pvalue: float
    pvalue_adjusted: float
    mean: float
    variance: float
    z_mean: float
    z_variance: float
    informative: bool = True

    @property
    def shape(self) -> str:
        """Direction of the departure, in the vocabulary of SBC rank plots.

        ``"uniform"`` when neither moment departs by more than two standard
        errors. Otherwise the dominant departure, by ``|z|``:

        - ``"narrow"``: PIT mass at both edges (the classic U). The reported
          posterior is more concentrated than the truth warrants; intervals
          undercover.
        - ``"wide"``: PIT mass in the middle (inverted U). The posterior is
          more diffuse than it needs to be; intervals overcover.
        - ``"biased low"`` / ``"biased high"``: PIT mass shifted. ``mean > 1/2``
          means the truth sits above the posterior draws more often than it
          should, so the posterior is centred below the truth.

        This is a descriptive label at a fixed two-standard-error threshold, not
        a test: under the null it reads something other than ``"uniform"`` for
        roughly one parameter in eleven. The gate decides on
        :attr:`pvalue_adjusted`, which is corrected for both the three tests and
        the number of parameters; the label is there to say *which way* a
        parameter the gate has already flagged is wrong.

        And it localizes a defect rather than diagnosing one: see the module
        docstring.
        """
        if abs(self.z_mean) < 2.0 and abs(self.z_variance) < 2.0:
            return "uniform"
        if abs(self.z_mean) >= abs(self.z_variance):
            return "biased low" if self.mean > 0.5 else "biased high"
        return "narrow" if self.variance > _UNIF_VAR else "wide"


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH step-up adjusted p-values (monotone, clipped at 1)."""
    p = np.asarray(pvalues, dtype=np.float64)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=np.float64)
    out[order] = np.minimum(ranked, 1.0)
    return out


@dataclass
class SBCGate:
    """Three-valued verdict on the calibration check.

    A gate that only says pass or fail lies when it has too little evidence to
    say either, so ``inconclusive`` is a separate state. It is not a pass.
    """

    passed: bool
    inconclusive: bool
    reasons: list[str] = field(default_factory=list)
    failing: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def summary(self) -> str:
        if self.inconclusive:
            head = "SBC gate INCONCLUSIVE"
        elif self.passed:
            head = "SBC gate PASSED"
        else:
            head = f"SBC gate FAILED ({len(self.failing)} parameter(s))"
        return head + "".join(f"\n  - {r}" for r in self.reasons)


@dataclass
class SBCResult:
    """PIT values across replicates, plus the per-parameter reading.

    Attributes:
        pit: ``(n_replicates, n_params)`` PIT values.
        ess: ``(n_replicates,)`` per-replicate effective sample size of the
            importance weights, or ``None`` when the draws were unweighted.
        param_names: Parameter names, length ``n_params``.
        parameters: One :class:`ParameterCalibration` per parameter, same order.
    """

    pit: np.ndarray
    ess: Optional[np.ndarray]
    param_names: list[str]
    parameters: list[ParameterCalibration]

    @property
    def n_replicates(self) -> int:
        return int(self.pit.shape[0])

    @property
    def n_params(self) -> int:
        return int(self.pit.shape[1])

    @property
    def median_ess(self) -> Optional[float]:
        return None if self.ess is None else float(np.median(self.ess))

    def by_name(self, name: str) -> ParameterCalibration:
        return self.parameters[self.param_names.index(name)]

    def table(self):
        """Per-parameter summary as a DataFrame, worst first."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "parameter": [p.name for p in self.parameters],
                "informative": [p.informative for p in self.parameters],
                "ks_statistic": [p.ks_statistic for p in self.parameters],
                "ks_pvalue": [p.ks_pvalue for p in self.parameters],
                "pvalue_mean": [p.pvalue_mean for p in self.parameters],
                "pvalue_variance": [p.pvalue_variance for p in self.parameters],
                "pvalue": [p.pvalue for p in self.parameters],
                "pvalue_adjusted": [p.pvalue_adjusted for p in self.parameters],
                "pit_mean": [p.mean for p in self.parameters],
                "pit_variance": [p.variance for p in self.parameters],
                "z_mean": [p.z_mean for p in self.parameters],
                "z_variance": [p.z_variance for p in self.parameters],
                "shape": [p.shape for p in self.parameters],
            }
        )
        return df.sort_values("pvalue_adjusted", na_position="last").reset_index(drop=True)

    def gate(
        self,
        *,
        alpha: float = 0.05,
        min_replicates: int = 100,
        min_ess: float = 20.0,
    ) -> SBCGate:
        """Pass, fail, or decline to call it.

        Args:
            alpha: Benjamini-Hochberg level across informative parameters,
                applied to :attr:`ParameterCalibration.pvalue` (already
                Bonferroni-corrected over the three per-parameter tests).
            min_replicates: Below this the KS test has too little power for a
                pass to mean anything, so the verdict is inconclusive. 100 is
                the usual SBC working number; it resolves departures of roughly
                0.14 in CDF distance.
            min_ess: Floor on the median per-replicate effective sample size.
                Below it the PIT is resolved too coarsely for the test to be
                read (see the module docstring on resolution).

        Returns:
            An :class:`SBCGate`. ``inconclusive`` implies ``not passed``.
        """
        reasons: list[str] = []
        inconclusive = False

        if self.n_replicates < min_replicates:
            inconclusive = True
            reasons.append(
                f"{self.n_replicates} replicates is below the floor of "
                f"{min_replicates}; the test has too little power to certify "
                "calibration."
            )

        med = self.median_ess
        if med is not None and med < min_ess:
            inconclusive = True
            reasons.append(
                f"median per-replicate ESS is {med:.1f}, below the floor of "
                f"{min_ess:g}. The PIT is resolved to about 1 part in {med:.0f}, "
                "which is coarser than the departures this test is meant to "
                "catch. Lower the proposal temperature or draw more."
            )

        informative = [p for p in self.parameters if p.informative]
        n_skipped = self.n_params - len(informative)
        if n_skipped:
            reasons.append(
                f"{n_skipped} of {self.n_params} parameters excluded as "
                "uninformative (theta* does not vary across replicates)."
            )
        if not informative:
            inconclusive = True
            reasons.append("no informative parameters left to test.")

        failing = [p for p in informative if p.pvalue_adjusted < alpha]
        if failing:
            worst = sorted(failing, key=lambda p: p.pvalue_adjusted)[:8]
            reasons.append(
                f"{len(failing)} of {len(informative)} informative parameters "
                f"reject uniformity at BH alpha={alpha:g}: "
                + ", ".join(f"{p.name} ({p.shape})" for p in worst)
                + (", ..." if len(failing) > len(worst) else "")
            )

        return SBCGate(
            passed=(not inconclusive) and (not failing),
            inconclusive=inconclusive,
            reasons=reasons,
            failing=[p.name for p in failing],
        )

    def summary(self) -> str:
        lines = [f"weighted SBC: {self.n_replicates} replicates x {self.n_params} parameters"]
        if self.ess is not None:
            lines.append(
                f"  per-replicate ESS: median {np.median(self.ess):.0f}, "
                f"min {np.min(self.ess):.0f}, max {np.max(self.ess):.0f}"
            )
        bad = sorted(
            (p for p in self.parameters if p.informative),
            key=lambda p: p.pvalue_adjusted,
        )[:5]
        for p in bad:
            lines.append(
                f"  {p.name}: p_adj={p.pvalue_adjusted:.3g} [{p.shape}] "
                f"KS={p.ks_statistic:.3f} mean={p.mean:.3f} var={p.variance:.4f}"
            )
        return "\n".join(lines)


def weighted_sbc(
    pit,
    *,
    param_names: Optional[Sequence[str]] = None,
    ess=None,
    theta_star=None,
    informative_cv: float = 1e-6,
) -> SBCResult:
    """Assemble per-replicate PIT values into a per-parameter reading.

    Args:
        pit: ``(n_replicates, n_params)`` PIT values, as returned per replicate
            by :func:`weighted_pit`.
        param_names: Names, length ``n_params``. Defaults to ``p0, p1, ...``.
        ess: ``(n_replicates,)`` per-replicate effective sample size of the
            importance weights. Optional, but without it the gate cannot check
            that the PIT is resolved finely enough to be read.
        theta_star: ``(n_replicates, n_params)`` ground truths, used only to mark
            parameters whose truth does not vary as uninformative. A prior that
            pins most of its parameters would otherwise contribute hundreds of
            vacuous tests to the multiplicity correction and dilute the real
            ones.
        informative_cv: A parameter counts as varying when the standard
            deviation of its ``theta*`` exceeds this times ``1 + |mean|``.

    Returns:
        An :class:`SBCResult`.
    """
    u = _as_array(pit)
    if u.ndim != 2:
        raise ValueError(f"pit must be (n_replicates, n_params); got {u.shape}")
    n_rep, n_par = u.shape
    if n_rep < 2:
        raise ValueError("need at least 2 replicates")
    if np.any(u < 0) or np.any(u > 1):
        raise ValueError("pit values must lie in [0, 1]")

    names = [f"p{j}" for j in range(n_par)] if param_names is None else list(param_names)
    if len(names) != n_par:
        raise ValueError(f"param_names has {len(names)} entries, pit has {n_par}")

    ess_arr = None if ess is None else _as_array(ess).ravel()
    if ess_arr is not None and ess_arr.size != n_rep:
        raise ValueError(f"ess has {ess_arr.size} entries but pit has {n_rep} replicates")

    informative = np.ones(n_par, dtype=bool)
    if theta_star is not None:
        ts = _as_array(theta_star)
        if ts.shape != u.shape:
            raise ValueError(f"theta_star has shape {ts.shape}, expected {u.shape}")
        scale = 1.0 + np.abs(np.mean(ts, axis=0))
        informative = np.std(ts, axis=0) > informative_cv * scale

    ks_stat = np.empty(n_par)
    ks_p = np.empty(n_par)
    for j in range(n_par):
        stat, p = stats.kstest(u[:, j], "uniform")
        ks_stat[j], ks_p[j] = float(stat), float(p)

    mean = u.mean(axis=0)
    var = u.var(axis=0, ddof=1)
    z_mean = (mean - 0.5) / np.sqrt(_UNIF_VAR / n_rep)
    z_var = (var - _UNIF_VAR) / np.sqrt(_UNIF_S2_VAR_COEF / n_rep)
    p_mean = 2.0 * stats.norm.sf(np.abs(z_mean))
    p_var = 2.0 * stats.norm.sf(np.abs(z_var))

    # Bonferroni over the three tests within each parameter, then BH across
    # parameters. See ParameterCalibration for why the moment tests are here.
    p_combined = np.minimum(3.0 * np.minimum(ks_p, np.minimum(p_mean, p_var)), 1.0)

    p_adj = np.full(n_par, np.nan)
    idx = np.flatnonzero(informative)
    if idx.size:
        p_adj[idx] = _benjamini_hochberg(p_combined[idx])

    params = [
        ParameterCalibration(
            name=names[j],
            ks_statistic=float(ks_stat[j]),
            ks_pvalue=float(ks_p[j]),
            pvalue_mean=float(p_mean[j]),
            pvalue_variance=float(p_var[j]),
            pvalue=float(p_combined[j]),
            pvalue_adjusted=float(p_adj[j]),
            mean=float(mean[j]),
            variance=float(var[j]),
            z_mean=float(z_mean[j]),
            z_variance=float(z_var[j]),
            informative=bool(informative[j]),
        )
        for j in range(n_par)
    ]

    return SBCResult(pit=u, ess=ess_arr, param_names=names, parameters=params)


# ---------------------------------------------------------------------------
# Reading it by eye
# ---------------------------------------------------------------------------
def uniform_band_halfwidth(n: int, alpha: float = 0.05) -> float:
    """Half-width of a simultaneous band for the ECDF of ``n`` uniform draws.

    Uses the asymptotic Kolmogorov quantile, ``k_alpha / sqrt(n)``. The band is
    simultaneous over the whole curve, which is what makes it the right thing to
    draw on an ECDF-difference plot: a pointwise band is crossed by a well
    calibrated curve far more often than ``alpha`` of the time. It is
    conservative relative to the tighter construction of Sailynoja, Burkner &
    Vehtari (2022), which trades the closed form for a simulation.
    """
    if n < 1:
        raise ValueError("n must be positive")
    return float(stats.kstwobign.ppf(1.0 - alpha) / np.sqrt(n))


def plot_sbc_ecdf(
    result: SBCResult,
    *,
    param_indices=None,
    alpha: float = 0.05,
    max_cols: int = 4,
    figsize=(16, 12),
):
    """ECDF-difference plot of the PIT values, one panel per parameter.

    The curve is ``ECDF(u) - u``, so a calibrated parameter is a flat line at
    zero inside the shaded simultaneous band. Excursions above the band on the
    left and below on the right are the U shape (posterior too narrow); the
    mirror image is the inverted U; a one-sided excursion is a location bias.

    Returns:
        ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt

    idx = list(range(result.n_params)) if param_indices is None else [int(i) for i in param_indices]
    n = len(idx)
    n_cols = min(max_cols, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
    flat = axes.flatten()

    half = uniform_band_halfwidth(result.n_replicates, alpha)

    for k, j in enumerate(idx):
        ax = flat[k]
        u = np.sort(result.pit[:, j])
        ecdf = np.arange(1, u.size + 1) / u.size
        ax.fill_between([0, 1], -half, half, color="gray", alpha=0.3)
        ax.axhline(0.0, color="r", ls="--", lw=1.0)
        ax.step(u, ecdf - u, where="post", lw=1.8)
        p = result.parameters[j]
        tag = "" if p.informative else " [uninformative]"
        ax.set_title(f"{p.name}{tag}\n{p.shape}", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)

    for k in range(n, flat.size):
        flat[k].axis("off")

    fig.tight_layout()
    return fig, axes

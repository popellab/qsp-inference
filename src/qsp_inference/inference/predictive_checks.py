"""Simulation-calibrated predictive checks for the population (NLME) model.

The QSP calibration is a nonlinear mixed-effects (NLME) problem: a population
model with **fixed effects** (the typical parameter values), **random effects**
(between-subject variability), a nonlinear **structural model** (the QSP ODE),
and **residual error** (the measurement model). A predictive check asks whether
an observed summary is consistent with the distribution this population model
predicts, using the predictive distribution *itself* as the null, so there is no
chi-square or normality assumption to violate. This is the mixed-effects reading
of the redesign's prior-data-conflict step: the tests are VPC / prediction
discrepancy, with a simulation null.

Two forms, one per inference target:

- :func:`prediction_discrepancy` -- the **fixed-effects / median-patient** form.
  The rank of an observed value within the model's predictive sample for that
  observable (the NLME *prediction discrepancy*, the univariate NPDE before
  decorrelation). A proper predictive includes residual error, so the caller
  passes a predictive that has already been through the measurement model
  (:func:`~qsp_inference.inference.data_processing.add_observation_noise`).

- :func:`population_vpc` -- the **mixed-effects** form, a visual predictive
  check. It confronts the observed cohort *summary* -- median (center) **and**
  inter-quartile spread -- with the distribution of that summary under cohorts
  simulated from the population model. The spread arm is the only one that can
  see a random-effects (omega) miss; the fixed-effects form is structurally
  blind to it, exactly as flat SBI is blind to population spread.

Both accept importance weights, so the null is the reporting prior ``pi`` rather
than a tempered training proposal ``pi_tilde``. At temperature 1 the weights are
uniform and the checks are unweighted. A discrepancy is only interpretable once
the calibration method itself is validated (the SBC gate); a small p here on an
ungated pipeline can be an estimator artifact rather than a model conflict.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "prediction_discrepancy",
    "population_vpc",
    "quantile_vpc",
    "inflate_cloud",
    "pit_calibration",
    "loo_pit",
    "label_marginal_conflict",
    "iqr",
]

ArrayLike = Union[np.ndarray, Sequence[float]]


def iqr(a: np.ndarray, axis=None) -> np.ndarray:
    """Inter-quartile range (75th minus 25th percentile), the spread statistic.

    Used as the random-effects summary in :func:`population_vpc`: the observed IQR
    carries the between-subject variability that a mixed-effects model must
    reproduce, the way the median carries the fixed-effect center.
    """
    q75, q25 = np.percentile(a, [75, 25], axis=axis)
    return q75 - q25


def _normalized_weights(n: int, weights: Optional[np.ndarray]) -> np.ndarray:
    """Weights summing to 1, or uniform ``1/n`` when ``weights`` is None."""
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must sum to a positive, finite value")
    return w / total


def _two_sided_pd(pctl: float) -> float:
    """Two-sided tail probability from a lower-tail rank ``pctl`` in [0, 1]."""
    return float(min(2.0 * min(pctl, 1.0 - pctl), 1.0))


def inflate_cloud(
    cloud: np.ndarray,
    noise_sd: Union[float, ArrayLike],
    *,
    asinh_scale: Optional[ArrayLike] = None,
    seed: int = 0,
):
    """Fold a per-observable additive-error budget into a predictive cloud.

    When the cloud comes from a **surrogate** forward model rather than the real
    one -- e.g. a GBM emulator standing in for the QSP ODE -- its approximation
    error is epistemic uncertainty about the *true* model's predictive. A
    predictive check that scores an observation against the raw surrogate null
    would charge that surrogate error as model misspecification: a deterministic
    emulator produces an artificially tight, biased null, and a real conflict
    cannot be told apart from a surrogate slip. Folding the error budget into the
    null fixes that: it convolves each observable's predictive with independent
    ``N(0, noise_sd_j)`` noise, widening the null so a discrepancy must *exceed*
    the surrogate error to be flagged. The test becomes conservative with respect
    to the budget -- the honest direction, because an error *magnitude* (which is
    all a held-out residual SD gives you) never licenses *narrowing* a null.

    The widening is applied in the space where the error was measured. GBM
    surrogate error is measured on ``asinh(obs / scale)`` (a signed-log transform,
    roughly homoscedastic across the observable's dynamic range), so pass
    ``asinh_scale`` to inject the noise there and map back to raw space; the
    injected raw-space spread then tracks the observable's local magnitude the way
    the residual does. Omit ``asinh_scale`` to add the noise in the cloud's own
    space.

    Args:
        cloud: ``(n_sim, n_obs)`` predictive samples (raw space). Non-finite
            entries stay non-finite.
        noise_sd: per-observable error SD -- a scalar (shared) or ``(n_obs,)``.
            In the ``asinh`` space when ``asinh_scale`` is given, else raw space.
            Non-finite or non-positive entries leave that observable untouched.
        asinh_scale: optional ``(n_obs,)`` per-observable asinh scale. When given,
            noise is additive on ``asinh(cloud / scale)`` and mapped back via
            ``scale * sinh(.)``; entries must be positive.
        seed: RNG seed (deterministic output).

    Returns:
        A new ``(n_sim, n_obs)`` array; ``cloud`` is not modified.
    """
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2:
        raise ValueError(f"cloud must be 2-D (n_sim, n_obs), got {cloud.shape}")
    n_sim, n_obs = cloud.shape

    sd = np.asarray(noise_sd, dtype=float)
    if sd.ndim == 0:
        sd = np.full(n_obs, float(sd))
    elif sd.shape != (n_obs,):
        raise ValueError(f"noise_sd must be scalar or ({n_obs},), got {sd.shape}")
    # A non-finite / non-positive budget means "no correction for this observable".
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 0.0)

    if asinh_scale is not None:
        scale = np.asarray(asinh_scale, dtype=float)
        if scale.shape != (n_obs,):
            raise ValueError(f"asinh_scale must be ({n_obs},), got {scale.shape}")
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("asinh_scale entries must be finite and positive")

    if not sd.any():
        return cloud.copy()

    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, 1.0, size=(n_sim, n_obs)) * sd[None, :]

    if asinh_scale is None:
        return cloud + eps
    a = np.arcsinh(cloud / scale[None, :]) + eps
    return scale[None, :] * np.sinh(a)


def prediction_discrepancy(
    predictive: np.ndarray,
    observed: ArrayLike,
    *,
    weights: Optional[np.ndarray] = None,
    observable_names: Optional[Sequence[str]] = None,
):
    """Per-observable prediction discrepancy against the predictive distribution.

    The fixed-effects / median-patient form. For each observable it computes the
    (weighted) fraction of the predictive at or below the observed value -- the
    prediction discrepancy ``pd`` -- and the two-sided tail probability
    ``p = 2*min(pd, 1-pd)``. A small ``p`` means the population model, drawn under
    the reporting prior, rarely produces the observed value: a real conflict, not
    a fit.

    Args:
        predictive: ``(n_sim, n_obs)`` predictive samples, already through the
            measurement model (residual error included). NaN rows per observable
            are dropped from that observable's null.
        observed: ``(n_obs,)`` observed values, or a ``{name: value}`` mapping
            when ``observable_names`` is given.
        weights: optional ``(n_sim,)`` importance weights carrying the predictive
            onto the reporting prior ``pi``; uniform if None (correct at T=1).
        observable_names: optional names, used to order a dict ``observed`` and to
            label the output.

    Returns:
        A ``pandas.DataFrame`` with one row per observable: ``observable``, ``pd``
        (lower-tail rank), ``p_value`` (two-sided), ``tail`` (``lower``/``upper``),
        sorted by ``p_value`` ascending (most conflicting first).
    """
    import pandas as pd

    predictive = np.asarray(predictive, dtype=float)
    if predictive.ndim != 2:
        raise ValueError(f"predictive must be 2-D (n_sim, n_obs), got {predictive.shape}")
    n_sim, n_obs = predictive.shape

    if observable_names is None:
        observable_names = [f"obs_{i}" for i in range(n_obs)]
    else:
        observable_names = list(observable_names)
        if len(observable_names) != n_obs:
            raise ValueError(
                f"observable_names has {len(observable_names)} entries, "
                f"predictive has {n_obs} observables"
            )

    if isinstance(observed, dict):
        obs_arr = np.array([float(np.squeeze(observed[n])) for n in observable_names])
    else:
        obs_arr = np.asarray(observed, dtype=float).reshape(-1)
        if obs_arr.shape[0] != n_obs:
            raise ValueError(
                f"observed has {obs_arr.shape[0]} values, predictive has {n_obs} observables"
            )

    w_full = _normalized_weights(n_sim, weights)

    records = []
    for i, name in enumerate(observable_names):
        col = predictive[:, i]
        finite = np.isfinite(col)
        if not finite.any():
            records.append({"observable": name, "pd": np.nan, "p_value": np.nan, "tail": "nan"})
            continue
        w = w_full[finite]
        w = w / w.sum()
        c = col[finite]
        obs_val = obs_arr[i]
        # Weighted lower-tail rank of the observation within the predictive.
        pd_val = float(w[c <= obs_val].sum())
        records.append(
            {
                "observable": name,
                "pd": pd_val,
                "p_value": _two_sided_pd(pd_val),
                "tail": "lower" if pd_val < 0.5 else "upper",
            }
        )

    return pd.DataFrame(records).sort_values("p_value").reset_index(drop=True)


def population_vpc(
    cloud: np.ndarray,
    observed_median: ArrayLike,
    observed_iqr: ArrayLike,
    *,
    cohort_size: Union[int, Sequence[int]],
    weights: Optional[np.ndarray] = None,
    n_replicates: int = 2000,
    seed: int = 0,
    observable_names: Optional[Sequence[str]] = None,
):
    """Visual predictive check on cohort summaries -- center and spread.

    The mixed-effects form. It builds the predictive distribution of a cohort
    summary by resampling cohorts of ``cohort_size`` individuals from the
    population-predictive ``cloud`` (weighted, so the cohorts come from the
    reporting prior ``pi``), computing each cohort's median and IQR per
    observable, and locating the *observed* median and IQR as prediction
    discrepancies against those predictive distributions.

    The two arms separate the two ways a population model fails:

    - **median pd** small -> a **center** conflict (fixed-effect / prior miss);
    - **IQR pd** small -> a **spread** conflict (random-effect / omega miss) --
      the population's between-subject variability does not match the data's. This
      is the axis the fixed-effects :func:`prediction_discrepancy` cannot see.

    The cohort size matters: predictive summary bands widen for small ``n``, so a
    low-``n`` target loosely constrains the spread and a high-``n`` one tightly,
    which is the honest per-target weighting. Pass a per-observable sequence when
    targets differ in ``n``.

    Args:
        cloud: ``(n_sim, n_obs)`` population-predictive samples of *individuals*
            (each row one draw of theta ~ pi pushed through the structural model
            and the measurement model).
        observed_median, observed_iqr: ``(n_obs,)`` observed cohort summaries.
        cohort_size: individuals per simulated cohort; an int (shared) or a
            per-observable sequence (the real published ``n`` of each target).
        weights: optional ``(n_sim,)`` importance weights onto ``pi``; uniform if
            None.
        n_replicates: simulated cohorts forming each predictive summary null.
        seed: RNG seed for the cohort resampling (deterministic output).
        observable_names: optional labels.

    Returns:
        A ``pandas.DataFrame`` with one row per observable: ``observable``,
        ``median_pd``, ``median_p`` (two-sided), ``iqr_pd``, ``iqr_p``, and
        ``spread_verdict`` (``over``/``under``/``ok`` at the 0.05 two-sided
        level: ``under`` = observed IQR below the predictive, i.e. the model is
        over-dispersed relative to the data). Sorted by the smaller of the two
        p-values ascending.
    """
    import pandas as pd

    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2:
        raise ValueError(f"cloud must be 2-D (n_sim, n_obs), got {cloud.shape}")
    n_sim, n_obs = cloud.shape

    obs_med = np.asarray(observed_median, dtype=float).reshape(-1)
    obs_iqr = np.asarray(observed_iqr, dtype=float).reshape(-1)
    if obs_med.shape[0] != n_obs or obs_iqr.shape[0] != n_obs:
        raise ValueError("observed_median / observed_iqr length must match cloud n_obs")

    if np.isscalar(cohort_size):
        sizes = [int(cohort_size)] * n_obs
    else:
        sizes = [int(s) for s in cohort_size]
        if len(sizes) != n_obs:
            raise ValueError(f"cohort_size sequence has {len(sizes)} entries, need {n_obs}")
    if any(s < 2 for s in sizes):
        raise ValueError("cohort_size must be >= 2 for an IQR to be defined")

    if observable_names is None:
        observable_names = [f"obs_{i}" for i in range(n_obs)]
    else:
        observable_names = list(observable_names)
        if len(observable_names) != n_obs:
            raise ValueError("observable_names length must match cloud n_obs")

    w_full = _normalized_weights(n_sim, weights)
    rng = np.random.default_rng(seed)

    records = []
    for i, name in enumerate(observable_names):
        col = cloud[:, i]
        finite = np.isfinite(col)
        if finite.sum() < sizes[i]:
            records.append(
                {
                    "observable": name,
                    "median_pd": np.nan,
                    "median_p": np.nan,
                    "iqr_pd": np.nan,
                    "iqr_p": np.nan,
                    "spread_verdict": "nan",
                }
            )
            continue
        c = col[finite]
        w = w_full[finite]
        w = w / w.sum()
        # Resample n_replicates cohorts of `n` individuals (weighted -> under pi),
        # and summarize each cohort. The spread of these summaries IS the
        # predictive null for the observed summary.
        idx = rng.choice(c.shape[0], size=(n_replicates, sizes[i]), replace=True, p=w)
        cohorts = c[idx]  # (n_replicates, n)
        med_null = np.median(cohorts, axis=1)
        iqr_null = iqr(cohorts, axis=1)

        med_pd = float(np.mean(med_null <= obs_med[i]))
        iqr_pd = float(np.mean(iqr_null <= obs_iqr[i]))
        iqr_p = _two_sided_pd(iqr_pd)
        if iqr_p < 0.05:
            spread = "under" if iqr_pd < 0.5 else "over"
        else:
            spread = "ok"
        records.append(
            {
                "observable": name,
                "median_pd": med_pd,
                "median_p": _two_sided_pd(med_pd),
                "iqr_pd": iqr_pd,
                "iqr_p": iqr_p,
                "spread_verdict": spread,
            }
        )

    df = pd.DataFrame(records)
    df["_min_p"] = df[["median_p", "iqr_p"]].min(axis=1)
    df = df.sort_values("_min_p").drop(columns="_min_p").reset_index(drop=True)
    return df


def quantile_vpc(
    cloud: np.ndarray,
    anchors: Sequence[Sequence[tuple]],
    *,
    cohort_size: Union[int, Sequence[int]],
    weights: Optional[np.ndarray] = None,
    n_replicates: int = 2000,
    seed: int = 0,
    observable_names: Optional[Sequence[str]] = None,
    center_band: float = 0.15,
):
    """Population VPC on the general quantile-anchor summary -- the maple
    ``ObservedDistribution`` contract, of which median/IQR is the special case.

    A reported distribution is a quantile function: the source gives the value at
    a set of probability levels (the median at ``p=0.5``, IQR edges at 0.25/0.75,
    or quartiles/deciles/a dense empirical function). :func:`population_vpc` is
    this check pinned to the three anchors ``{0.25, 0.5, 0.75}``; here the anchors
    are whatever maple reports per observable, so the check confronts the *shape*
    the data actually carries rather than a fixed two-moment reduction.

    For each observable it resamples weighted cohorts of ``cohort_size`` from the
    population-predictive ``cloud`` and locates **each** observed anchor
    ``Q(p)`` as a prediction discrepancy against the predictive distribution of
    that cohort quantile. An anchor near ``p=0.5`` probes the **center** (fixed
    effect); an anchor in the tails probes the **spread** (random effect /
    ``omega``) -- the axis a fixed-effects check cannot see. A per-observable
    spread verdict comes from the widest anchor pair's inter-quantile width, the
    direct generalization of the IQR arm.

    Args:
        cloud: ``(n_sim, n_obs)`` population-predictive samples of individuals
            (each row one ``theta ~ pi`` pushed through the structural and
            measurement models).
        anchors: length ``n_obs``; ``anchors[i]`` is a sequence of
            ``(p, observed_value)`` pairs for observable ``i`` (``p`` in
            ``(0, 1)``). Ragged across observables is allowed; an empty list
            skips that observable.
        cohort_size: individuals per simulated cohort; an int (shared) or a
            per-observable sequence (each target's real published ``n``).
        weights: optional ``(n_sim,)`` importance weights onto ``pi``; uniform if
            None.
        n_replicates: simulated cohorts forming each predictive summary null.
        seed: RNG seed for the cohort resampling (deterministic output).
        observable_names: optional labels, length ``n_obs``.
        center_band: an anchor with ``|p - 0.5| <= center_band`` is a ``center``
            anchor, otherwise ``spread``. The default 0.15 makes 0.5 the center
            and 0.25/0.75 spread, matching :func:`population_vpc`.

    Returns:
        A ``pandas.DataFrame`` with one row per ``(observable, p)`` anchor:
        ``observable``, ``p``, ``kind`` (``center``/``spread``), ``observed``,
        ``pd`` (lower-tail rank), ``p_value`` (two-sided), and ``spread_verdict``
        (per observable, repeated on its rows: ``over``/``under``/``ok`` at the
        0.05 level from the widest anchor pair -- ``under`` = observed spread
        below the predictive, i.e. the model is over-dispersed; ``center_only``
        when fewer than two anchors span a width). Sorted by each observable's
        smallest anchor ``p_value`` ascending.
    """
    import pandas as pd

    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2:
        raise ValueError(f"cloud must be 2-D (n_sim, n_obs), got {cloud.shape}")
    n_sim, n_obs = cloud.shape
    if len(anchors) != n_obs:
        raise ValueError(f"anchors has {len(anchors)} entries, cloud has {n_obs} observables")

    if np.isscalar(cohort_size):
        sizes = [int(cohort_size)] * n_obs
    else:
        sizes = [int(s) for s in cohort_size]
        if len(sizes) != n_obs:
            raise ValueError(f"cohort_size sequence has {len(sizes)} entries, need {n_obs}")
    if any(s < 2 for s in sizes):
        raise ValueError("cohort_size must be >= 2 for a spread anchor to be defined")

    if observable_names is None:
        observable_names = [f"obs_{i}" for i in range(n_obs)]
    else:
        observable_names = list(observable_names)
        if len(observable_names) != n_obs:
            raise ValueError("observable_names length must match cloud n_obs")

    w_full = _normalized_weights(n_sim, weights)
    rng = np.random.default_rng(seed)

    records = []
    for i, name in enumerate(observable_names):
        anc = [(float(p), float(v)) for p, v in anchors[i]]
        if not anc:
            continue
        for p, _ in anc:
            if not (0.0 < p < 1.0):
                raise ValueError(f"anchor p must be in (0, 1) for {name}, got {p}")
        col = cloud[:, i]
        finite = np.isfinite(col)
        if finite.sum() < sizes[i]:
            for p, val in anc:
                records.append({
                    "observable": name, "p": p,
                    "kind": "center" if abs(p - 0.5) <= center_band else "spread",
                    "observed": val, "pd": np.nan, "p_value": np.nan,
                    "spread_verdict": "nan",
                })
            continue

        c = col[finite]
        w = w_full[finite]
        w = w / w.sum()
        idx = rng.choice(c.shape[0], size=(n_replicates, sizes[i]), replace=True, p=w)
        cohorts = c[idx]  # (n_replicates, n)

        ps = np.array([p for p, _ in anc])
        q_null = np.quantile(cohorts, ps, axis=1)  # (n_anchor, n_replicates)

        # Per-observable spread verdict from the widest anchor pair's width.
        if len(anc) >= 2:
            lo, hi = int(np.argmin(ps)), int(np.argmax(ps))
            width_null = q_null[hi] - q_null[lo]
            width_obs = anc[hi][1] - anc[lo][1]
            w_pd = float(np.mean(width_null <= width_obs))
            w_p = _two_sided_pd(w_pd)
            verdict = "ok" if w_p >= 0.05 else ("under" if w_pd < 0.5 else "over")
        else:
            verdict = "center_only"

        for j, (p, val) in enumerate(anc):
            pd_j = float(np.mean(q_null[j] <= val))
            records.append({
                "observable": name, "p": p,
                "kind": "center" if abs(p - 0.5) <= center_band else "spread",
                "observed": val, "pd": pd_j, "p_value": _two_sided_pd(pd_j),
                "spread_verdict": verdict,
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    order = df.groupby("observable")["p_value"].transform("min")
    df = df.assign(_min_p=order).sort_values(
        ["_min_p", "observable", "p"]
    ).drop(columns="_min_p").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# PIT calibration (LOO-PIT)
# ---------------------------------------------------------------------------
def pit_calibration(pit_values: ArrayLike, *, n_bins: int = 10) -> dict:
    """Marginal-calibration summary of a set of PIT values.

    The probability-integral-transform (PIT) of an observation is its rank within
    its own predictive distribution. Under a correctly specified model those PITs
    are Uniform(0, 1); systematic departure is the classic NLME calibration
    signal, and its *shape* names the failure:

    - mean far from 1/2        -> a center bias (predictions systematically off);
    - variance below 1/12      -> an **over-dispersed** predictive (PITs bunch at
      1/2 because the model's spread is wider than the data's);
    - variance above 1/12      -> an **under-dispersed** predictive (PITs pile at
      0 and 1).

    Uniform(0, 1) has mean 1/2 and variance 1/12, so those are the reference
    values. The Kolmogorov-Smirnov test against Uniform(0, 1) is the omnibus
    check; the moments say which way it fails.

    Args:
        pit_values: one PIT per observation/observable (e.g. the ``pd`` column of
            :func:`prediction_discrepancy`). Non-finite entries are dropped.
        n_bins: histogram resolution returned for a PIT plot.

    Returns:
        A dict with ``ks_stat``, ``ks_p`` (vs Uniform(0, 1)), ``mean``, ``var``,
        ``dispersion`` (``over`` / ``under`` / ``ok`` at var vs 1/12, 20% band),
        ``n``, and ``hist`` / ``bin_edges``.
    """
    from scipy import stats

    pit = np.asarray(pit_values, dtype=float)
    pit = pit[np.isfinite(pit)]
    n = pit.size
    if n == 0:
        return {
            "ks_stat": np.nan, "ks_p": np.nan, "mean": np.nan, "var": np.nan,
            "dispersion": "nan", "n": 0, "hist": np.array([]), "bin_edges": np.array([]),
        }
    ks = stats.kstest(pit, "uniform")
    var = float(pit.var())
    uniform_var = 1.0 / 12.0
    if var < uniform_var * 0.8:
        dispersion = "over"     # predictive too wide -> PITs bunch near 1/2
    elif var > uniform_var * 1.2:
        dispersion = "under"    # predictive too narrow -> PITs pile at the ends
    else:
        dispersion = "ok"
    hist, bin_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return {
        "ks_stat": float(ks.statistic),
        "ks_p": float(ks.pvalue),
        "mean": float(pit.mean()),
        "var": var,
        "dispersion": dispersion,
        "n": int(n),
        "hist": hist,
        "bin_edges": bin_edges,
    }


def loo_pit(
    predictive: np.ndarray,
    observed: ArrayLike,
    *,
    weights: Optional[np.ndarray] = None,
    observable_names: Optional[Sequence[str]] = None,
    n_bins: int = 10,
):
    """Leave-one-out PIT calibration across observables.

    For each observable the PIT is the (weighted) rank of the observed value in
    its predictive -- the ``pd`` of :func:`prediction_discrepancy` -- and the set
    of PITs is checked for uniformity by :func:`pit_calibration`. Uniform PITs are
    a calibrated fit; a systematic shape is a marginal miss whose direction the
    calibration summary names (center vs over/under-dispersion).

    **What makes it leave-one-out is the** ``predictive`` **the caller passes.**
    A proper LOO-PIT predicts observable *j* from the *other* observables (obs *j*
    held out of the conditioning), so obs *j* never informs its own predictive and
    the check is not optimistic. That needs a masking-capable / set-based
    conditioning the fixed-dimensional embedding does not support yet (the
    redesign stages it as a planned addition). Until it lands, pass the posterior
    predictive of each observable and read the result as **in-sample** marginal
    calibration -- honest, but optimistic, because each observable helped fit the
    posterior it is then scored against. The statistic is identical; only the
    conditioning differs, which is why it lives in one function.

    Args:
        predictive: ``(n_sim, n_obs)`` predictive samples, already through the
            measurement model. Leave-one-out if produced with obs *j* masked,
            posterior-predictive otherwise.
        observed: ``(n_obs,)`` observed values or a ``{name: value}`` mapping.
        weights: optional importance weights onto the reporting prior.
        observable_names: optional labels.
        n_bins: PIT-histogram resolution.

    Returns:
        ``(pit_df, calibration)`` where ``pit_df`` is the per-observable
        :func:`prediction_discrepancy` frame (its ``pd`` column is the PIT) and
        ``calibration`` is the :func:`pit_calibration` summary over those PITs.
    """
    pit_df = prediction_discrepancy(
        predictive, observed, weights=weights, observable_names=observable_names
    )
    calibration = pit_calibration(pit_df["pd"].to_numpy(), n_bins=n_bins)
    return pit_df, calibration


# ---------------------------------------------------------------------------
# The marginal misfit labeler (redesign steps 2 -> 3)
# ---------------------------------------------------------------------------
def label_marginal_conflict(
    prior_predictive: np.ndarray,
    observed: ArrayLike,
    *,
    posterior_resim: Optional[np.ndarray] = None,
    weights_prior: Optional[np.ndarray] = None,
    weights_resim: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    observable_names: Optional[Sequence[str]] = None,
):
    """Per-observable ``consistent`` / ``prior_limited`` / ``structural`` label.

    The marginal arm of the principled labeler, combining the two tests that have
    a simulation null:

    - **Step 2, prior-data conflict.** :func:`prediction_discrepancy` of the
      observed value in the *prior* predictive (the model under the anchored prior
      pi, with residual error). ``p >= alpha`` -> ``consistent``: the anchor
      already predicts it, it is a fit.
    - **Step 3, reachability.** For a conflicting observable, the same discrepancy
      in the *posterior-at-x_obs resimulated* predictive -- the observables you get
      by drawing theta from the NPE posterior at ``x_obs``, keeping the admissible
      ones, and simulating them forward (a TSNPE round pointed at ``x_obs``). If
      the resim now reaches the observation (``p >= alpha``) a plausible theta does
      produce it and the anchor simply did not put mass there -> ``prior_limited``
      (re-anchor). If the model's own best attempt still misses -> ``structural``
      (the mechanism cannot produce it).

    Structural is only asserted when ``posterior_resim`` is supplied: the doc's OOD
    caveat is that "cannot fit" is untrustworthy unless the sampler actually tried,
    so a conflicting observable with no resim is labeled ``conflict_unresolved``,
    not structural. The resimulation itself needs the simulator and is the caller's
    (this function owns only the decision from the two predictive samples).

    Args:
        prior_predictive: ``(n_sim, n_obs)`` prior-predictive samples (with
            residual error).
        observed: ``(n_obs,)`` observed values or a ``{name: value}`` mapping.
        posterior_resim: ``(n_resim, n_obs)`` observables resimulated from the
            posterior at ``x_obs``. ``None`` leaves conflicts unresolved.
        weights_prior, weights_resim: optional importance weights onto pi for each
            predictive.
        alpha: two-sided significance for both the conflict and reachability calls.
        observable_names: optional labels (also orders a dict ``observed``).

    Returns:
        A ``pandas.DataFrame`` with one row per observable: ``observable``,
        ``prior_p`` (step 2), ``resim_p`` (step 3, NaN if no resim), and ``label``
        in {``consistent``, ``prior_limited``, ``structural``,
        ``conflict_unresolved``}. Sorted with the unresolved/structural misses
        first.
    """
    import pandas as pd

    prior_dd = prediction_discrepancy(
        prior_predictive, observed, weights=weights_prior, observable_names=observable_names
    ).set_index("observable")

    names = list(prior_dd.index)
    resim_p_by_name = {}
    if posterior_resim is not None:
        resim_dd = prediction_discrepancy(
            posterior_resim, observed, weights=weights_resim, observable_names=observable_names
        ).set_index("observable")
        resim_p_by_name = resim_dd["p_value"].to_dict()

    records = []
    for name in names:
        prior_p = float(prior_dd.loc[name, "p_value"])
        resim_p = float(resim_p_by_name.get(name, np.nan))
        if not np.isfinite(prior_p):
            label = "nan"
        elif prior_p >= alpha:
            label = "consistent"
        elif posterior_resim is None or not np.isfinite(resim_p):
            label = "conflict_unresolved"
        elif resim_p >= alpha:
            label = "prior_limited"
        else:
            label = "structural"
        records.append(
            {"observable": name, "prior_p": prior_p, "resim_p": resim_p, "label": label}
        )

    order = {"structural": 0, "conflict_unresolved": 1, "prior_limited": 2, "consistent": 3, "nan": 4}
    df = pd.DataFrame(records)
    df["_o"] = df["label"].map(order).fillna(5)
    df = df.sort_values(["_o", "prior_p"]).drop(columns="_o").reset_index(drop=True)
    return df

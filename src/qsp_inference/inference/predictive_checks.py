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

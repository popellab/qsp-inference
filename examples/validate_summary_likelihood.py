"""Validate the Ch. 4b summary likelihood against a simulation cloud.

The summary likelihood (``docs/population-inference-tractable.md``) asserts that a
study's observed quantile anchors are Gaussian around the population quantiles with a
covariance given by the copula formula

    Cov( Qhat_j(p_a), Qhat_l(p_b) ) = [ C_jl(p_a, p_b) - p_a p_b ]
                                      / ( n f_j(Q_j(p_a)) f_l(Q_l(p_b)) )

where C_jl is the copula of (X_j, X_l) under the population predictive and j = l gives
back the familiar within-observable form. "Asymptotic" is the whole question when n is
6, so this script measures it rather than assuming it, and it is the harness behind
``targets/anchors.py:QUANTILE_METHOD`` and ``MIN_PATIENTS_PER_ANCHOR``.

RERUN THIS whenever the simulation pool changes. The conclusions are pool-dependent in
one direction that matters: the anchor-density budget and the estimator choice are
distribution-free (they reproduce a pure-Gaussian reference), but the density plug-in
bandwidth and the cross-target block sizes are properties of the cloud in hand.

Two checks:

  diagonal   per observable, at that target's real n: is the asymptotic law a good
             description of the finite-n sampling spread? Reported as the Mahalanobis
             D^2/k of the anchor vector against the asserted covariance (1.00 exact)
             and the coverage of the implied ellipsoids. Also reports the finite-n IQR
             bias, which lands directly on sigma_u_hat.

  block      per study (observables sharing a cohort), with cohorts drawn as ROWS so
             one set of n patients feeds every observable: does the cross-target block
             hold, and what does assuming independence cost?

Usage
-----
    python examples/validate_summary_likelihood.py \\
        --cloud path/to/cloud.npz --targets path/to/marginal_targets.csv

``--cloud`` is an ``.npz`` with a patient-by-observable matrix and matching observable
names; ``--targets`` a CSV with ``scenario``, ``observable`` and the real published n.
Column names are options so this is not bound to one project's schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, rankdata

from qsp_inference.targets.anchors import QUANTILE_METHOD, _resolvable_grid

QUARTILES = (0.25, 0.5, 0.75)


def _sparsity(x: np.ndarray, p: np.ndarray, h: float) -> np.ndarray:
    """f(Q(p)) by quantile spacing: f = 2h / (Q(p+h) - Q(p-h)).

    The bandwidth is a real sensitivity, not a detail. Measured on one PDAC cloud,
    D^2/k for n<=20 ran from 0.95 at h=0.02 to 0.76 at h=0.15, so declare it, keep it
    fixed, and re-check it per observable on a new pool.
    """
    lo, hi = np.clip(p - h, 1e-6, 1 - 1e-6), np.clip(p + h, 1e-6, 1 - 1e-6)
    dq = np.quantile(x, hi) - np.quantile(x, lo)
    return np.where(dq > 0, (hi - lo) / np.maximum(dq, 1e-300), np.nan)


def _copula_cov(pool, ps, f, n, *, gaussian=False, independent=False):
    """Stacked sampling covariance of a study's anchor vector, from the copula form."""
    J = len(ps)
    offs = np.cumsum([0] + [len(p) for p in ps])
    S = np.zeros((offs[-1], offs[-1]))
    if gaussian:
        z = norm.ppf((np.apply_along_axis(rankdata, 0, pool) - 0.5) / len(pool))
        R = np.corrcoef(z.T)
        from scipy.stats import multivariate_normal as mvn
    q = [np.quantile(pool[:, j], ps[j], method=QUANTILE_METHOD) for j in range(J)]
    for j in range(J):
        for ell in range(J):
            if independent and j != ell:
                continue
            for a, pa in enumerate(ps[j]):
                for b, pb in enumerate(ps[ell]):
                    if j == ell:
                        C = min(pa, pb)
                    elif gaussian:
                        C = mvn.cdf([norm.ppf(pa), norm.ppf(pb)], mean=[0, 0],
                                    cov=[[1.0, R[j, ell]], [R[j, ell], 1.0]])
                    else:
                        C = float(((pool[:, j] <= q[j][a])
                                   & (pool[:, ell] <= q[ell][b])).mean())
                    S[offs[j] + a, offs[ell] + b] = (
                        (C - pa * pb) / (n * f[j][a] * f[ell][b])
                    )
    return 0.5 * (S + S.T)


def _d2k(Qhat, mu, S):
    k = S.shape[0]
    try:
        L = np.linalg.cholesky(S + 1e-12 * np.eye(k) * np.trace(S) / k)
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    d2 = (np.linalg.solve(L, (Qhat - mu).T).T ** 2).sum(axis=1)
    return float(d2.mean() / k), float((d2 <= chi2.ppf(0.90, k)).mean())


def _log_pool(X, cols, min_pool):
    """Rows where every column of the study is finite and positive, in log space."""
    ok = np.isfinite(X[:, cols]).all(axis=1) & (X[:, cols] > 0).all(axis=1)
    pool = np.log(X[np.ix_(ok, cols)])
    return pool if len(pool) >= min_pool else None


def check_diagonal(X, names, n_by, args, rng) -> pd.DataFrame:
    """Per-observable calibration of the asymptotic law at each target's real n."""
    rows = []
    for j, nm in enumerate(names):
        n = n_by.get(nm)
        if n is None:
            continue
        pool = _log_pool(X, [j], args.min_pool)
        if pool is None:
            continue
        x = pool[:, 0]
        p = np.asarray(_resolvable_grid(QUARTILES, int(n)), float)
        if len(p) < 2:
            continue
        f = _sparsity(x, p, args.bandwidth)
        if not np.all(np.isfinite(f)) or np.any(f <= 0):
            continue
        Q = np.quantile(x, p, method=QUANTILE_METHOD)
        S = _copula_cov(pool, [p], [f], int(n))

        # true finite-n law: iid draws from the pool's piecewise-linear quantile
        # function (continuous, so no tie artefacts), summarised with the same
        # estimator the package uses on both sides of the contract.
        xs = np.sort(x)
        gu = np.arange(xs.size) / (xs.size - 1.0)
        b = args.reps if n <= 60 else max(args.reps // 4, 5000)
        Qh = np.quantile(np.interp(rng.random((b, int(n))), gu, xs), p,
                         axis=1, method=QUANTILE_METHOD).T
        d2k, cov90 = _d2k(Qh, Q, S)
        rec = {"observable": nm.split("/")[-1], "n": int(n), "k": len(p),
               "d2_over_k": d2k, "cov90": cov90}
        if {0.25, 0.75} <= set(np.round(p, 6)):
            lo = int(np.where(np.isclose(p, 0.25))[0][0])
            hi = int(np.where(np.isclose(p, 0.75))[0][0])
            rec["iqr_ratio"] = float((Qh[:, hi] - Qh[:, lo]).mean() / (Q[hi] - Q[lo]))
        rows.append(rec)
    return pd.DataFrame(rows)


def check_blocks(X, names, n_by, args, rng) -> pd.DataFrame:
    """Cross-target block: cohorts drawn as rows, so a study's targets share patients."""
    groups = {}
    for nm, n in n_by.items():
        if nm in names:
            groups.setdefault((nm.split("/")[0], int(n)), []).append(names.index(nm))

    rows = []
    for (scen, n), cols in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(cols) < 2:
            continue
        pool = _log_pool(X, cols, args.min_pool)
        if pool is None:
            continue
        N, J = pool.shape
        p = np.asarray(_resolvable_grid(QUARTILES, n), float)
        ps = [p] * J
        f = [_sparsity(pool[:, j], p, args.bandwidth) for j in range(J)]
        if any(not np.all(np.isfinite(fi)) or np.any(fi <= 0) for fi in f):
            continue
        mu = np.concatenate(
            [np.quantile(pool[:, j], p, method=QUANTILE_METHOD) for j in range(J)]
        )
        b = args.reps if n <= 60 else max(args.reps // 4, 5000)
        coh = pool[rng.integers(0, N, size=(b, n))]          # rows: shared patients
        Qh = np.concatenate(
            [np.quantile(coh[:, :, j], p, axis=1, method=QUANTILE_METHOD).T
             for j in range(J)], axis=1
        )
        zz = norm.ppf((np.apply_along_axis(rankdata, 0, pool) - 0.5) / N)
        R = np.corrcoef(zz.T)
        rec = {"study": scen[:28], "J": J, "n": n,
               "median_abs_rho": float(np.median(np.abs(R[np.triu_indices(J, 1)])))}
        for tag, kw in (("copula", {}), ("gaussian", {"gaussian": True}),
                        ("independent", {"independent": True})):
            d2k, cov90 = _d2k(Qh, mu, _copula_cov(pool, ps, f, n, **kw))
            rec[f"d2k_{tag}"], rec[f"cov90_{tag}"] = d2k, cov90
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cloud", required=True, type=Path,
                    help=".npz holding the patient-by-observable matrix.")
    ap.add_argument("--targets", required=True, type=Path,
                    help="CSV with scenario / observable / published n.")
    ap.add_argument("--cloud-key", default="after", help="npz key for the matrix.")
    ap.add_argument("--names-key", default="obs_names", help="npz key for the names.")
    ap.add_argument("--n-column", default="n_published", help="CSV column for real n.")
    ap.add_argument("--bandwidth", type=float, default=0.035,
                    help="Sparsity bandwidth h for f(Q(p)). Declare it; do not tune "
                         "it during fitting (default 0.035).")
    ap.add_argument("--reps", type=int, default=20000,
                    help="Synthetic cohorts per cell (default 20000).")
    ap.add_argument("--min-pool", type=int, default=400,
                    help="Minimum usable patients to treat a cloud column as a "
                         "population (default 400).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional directory for the per-cell CSVs.")
    args = ap.parse_args()

    d = np.load(args.cloud, allow_pickle=True)
    X = d[args.cloud_key]
    names = [str(s) for s in d[args.names_key]]
    tg = pd.read_csv(args.targets)
    n_by = dict(zip(tg["scenario"] + "/" + tg["observable"], tg[args.n_column].astype(int)))
    rng = np.random.default_rng(0)

    pd.set_option("display.width", 200)
    diag = check_diagonal(X, names, n_by, args, rng)
    print(f"=== diagonal: {len(diag)} observables, h={args.bandwidth}, "
          f"estimator={QUANTILE_METHOD!r}")
    print("    d2_over_k 1.00 = the asserted covariance is exact; >1 = overconfident.")
    print("    iqr_ratio 1.00 = the summary is unbiased for the population IQR;")
    print("    below 1 biases sigma_u_hat LOW by the same factor.\n")
    if len(diag):
        diag["n_bin"] = pd.cut(diag.n, [0, 8, 12, 20, 60, 10_000],
                               labels=["6-8", "9-12", "13-20", "21-60", ">60"])
        print(diag.groupby("n_bin", observed=True).agg(
            cells=("n", "size"), d2_over_k=("d2_over_k", "median"),
            cov90=("cov90", "median"), iqr_ratio=("iqr_ratio", "median"),
        ).round(3).to_string())

    blk = check_blocks(X, names, n_by, args, rng)
    print(f"\n=== blocks: {len(blk)} multi-target studies (cohorts drawn as rows)")
    print("    compare d2k_independent against d2k_copula: the gap is what")
    print("    factorising over targets that share patients costs.\n")
    if len(blk):
        print(blk.round(3).to_string(index=False))

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        diag.to_csv(args.out / "summary_likelihood_diagonal.csv", index=False)
        blk.to_csv(args.out / "summary_likelihood_blocks.csv", index=False)
        print(f"\nwrote {args.out}/summary_likelihood_{{diagonal,blocks}}.csv")


if __name__ == "__main__":
    main()

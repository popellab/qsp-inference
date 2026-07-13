"""Joint-reachability diagnostics for a virtual population.

Every other calibration metric in a QSP workflow is **marginal** -- posterior-predictive
coverage asks, per observable, whether the cloud covers the datum; Wasserstein distance is
per target. None of them ever asks whether a *single patient* can satisfy all of them at
once. These functions score the joint, which is where a VPop lives or dies.

They also exist because the joint is easy to get wrong. Three traps, each of which produced
a confident and false conclusion before being caught:

1. **Raw ESS collapse is not evidence of misspecification.** Matching many marginals by
   importance weighting costs effective sample size *multiplicatively* even for a perfect
   model. The discriminator is how ESS scales with cloud size -- :func:`ess_scaling`.
2. **Observable ORDER changes the answer by ~100x.** An arbitrary subset of 24 observables
   gave ESS 12; the greedily-chosen 23 gave ESS 1,291. Always report the set --
   :func:`greedy_core`.
3. **ESS collapse with a perfect fit means the SOLVER is broken, not the model.** Two
   observables the model cannot distinguish, handed slightly different targets, make the
   dual unsolvable; the optimiser then exploits floating-point noise and the weights
   explode while the fit still looks perfect. :func:`duplicate_observables` finds the
   collinear rows that cause this. Run it FIRST.

And one measure deliberately NOT provided: counting cloud members inside the observed IQR of
every observable at once. It goes to zero for *any* model, perfect or not, because a real
patient sits in any one observable's IQR only ~50% of the time and 0.5^k vanishes. It looks
devastating and means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from qsp_inference.vpop.weighting import fit_prevalence_weights

__all__ = [
    "CoreResult",
    "duplicate_observables",
    "self_target_control",
    "ess_scaling",
    "greedy_core",
    "conflict_ranking",
    "paired_effect_sizes",
]


def _ess(sim_obs, observed, obs_names, idx, **kw) -> float:
    names = [obs_names[i] for i in idx]
    res = fit_prevalence_weights(
        sim_obs[:, idx], {n: observed[n] for n in names}, names, **kw
    )
    return float(res.ess)


def duplicate_observables(
    sim_obs: np.ndarray, obs_names: list[str], *, threshold: float = 0.99
) -> pd.DataFrame:
    """Observable pairs the MODEL treats as the same variable (rank correlation >= threshold).

    **Run this before trusting any joint result.** Near-duplicate columns produce collinear
    constraint rows. If their observed targets disagree even slightly, the max-entropy dual
    becomes unsolvable, the optimiser exploits numerical noise between the columns, and ESS
    collapses while the fit still looks perfect -- indistinguishable from a real joint
    conflict unless you look.

    A duplicate pair is also a *finding* in its own right: if a readout in two treatment
    arms is near-perfectly correlated across the cloud, the model cannot tell the arms
    apart, and no calibration can satisfy two different observed distributions for them.
    """
    ranks = np.apply_along_axis(rankdata, 0, sim_obs)
    R = np.corrcoef(ranks.T)
    rows = []
    for i in range(len(obs_names)):
        for j in range(i + 1, len(obs_names)):
            if abs(R[i, j]) >= threshold:
                rows.append(
                    {"observable_a": obs_names[i], "observable_b": obs_names[j],
                     "spearman": float(R[i, j])}
                )
    return pd.DataFrame(rows).sort_values("spearman", ascending=False, ignore_index=True)


def self_target_control(
    sim_obs: np.ndarray, obs_names: list[str], *, n_draw: int = 400, seed: int = 0, **kw
) -> float:
    """Null control: feed the cloud its OWN marginals and refit.

    A sound weighting returns near-uniform weights (ESS fraction near 1) because no tilting
    is needed. A low value here means the machinery is broken and every other number in this
    module is meaningless. Check it before believing a collapse.
    """
    rng = np.random.default_rng(seed)
    observed = {
        n: sim_obs[rng.choice(len(sim_obs), size=n_draw, replace=False), k]
        for k, n in enumerate(obs_names)
    }
    res = fit_prevalence_weights(sim_obs, observed, obs_names, **kw)
    return float(res.ess_fraction)


def ess_scaling(
    sim_obs: np.ndarray,
    observed: dict[str, np.ndarray],
    obs_names: list[str],
    *,
    sizes: list[int] | None = None,
    seed: int = 0,
    **kw,
) -> pd.DataFrame:
    """THE misspecification discriminator: how does ESS scale with cloud size?

    - **ESS fraction roughly constant** -> ESS grows with N -> the cost is ordinary
      importance-sampling overhead, which a bigger cloud buys you out of. The model may be
      perfectly fine.
    - **ESS absolutely flat** (fraction falling as 1/N) -> the weights are locked onto a
      fixed handful of tail draws and more simulation finds no more of them -> the region is
      genuinely unreachable.

    Without this, a large ESS loss is uninterpretable.
    """
    n = len(sim_obs)
    sizes = sizes or [max(1000, n // 8), n // 4, n // 2, n]
    rng = np.random.default_rng(seed)
    idx = list(range(len(obs_names)))
    rows = []
    for N in sizes:
        take = rng.choice(n, size=min(N, n), replace=False) if N < n else np.arange(n)
        e = _ess(sim_obs[take], observed, obs_names, idx, **kw)
        rows.append({"n_cloud": int(len(take)), "ess": e, "ess_fraction": e / len(take)})
    return pd.DataFrame(rows)


@dataclass
class CoreResult:
    """The largest set of observables the model can jointly reach, and what it cannot."""

    core: list[str]
    blockers: list[str]
    core_ess: float
    trajectory: pd.DataFrame
    """ESS after each greedy addition -- shows where the cohort thins out."""


def _fit(sim_obs, observed, obs_names, idx, **kw):
    names = [obs_names[i] for i in idx]
    return fit_prevalence_weights(
        sim_obs[:, idx], {n: observed[n] for n in names}, names, **kw
    )


def greedy_core(
    sim_obs: np.ndarray,
    observed: dict[str, np.ndarray],
    obs_names: list[str],
    *,
    min_ess: float = 1000.0,
    max_tv: float = 0.05,
    **kw,
) -> CoreResult:
    """Greedily grow the largest observable set the model can jointly satisfy.

    At each step add whichever remaining observable costs the least ESS; stop when no
    remaining observable can be added while keeping ESS above ``min_ess`` **and** every
    observable fitted to within ``max_tv`` total-variation distance.

    **Both gates are load-bearing.** Selecting on ESS alone is wrong: an observable the model
    cannot reach *at all* costs no ESS, because the weights correctly stay uniform rather
    than straining toward data they cannot represent. Judged on ESS it looks free, and greedy
    would happily admit it to the "reachable" core with a fit of TV ~1. The ``max_tv`` gate is
    what keeps unreachable observables out.

    Greedy, not exhaustive -- the exact maximal set is combinatorial. But order matters by
    ~100x, so a greedy order is vastly more informative than an arbitrary one, and the
    trajectory shows where the model runs out of joint capacity.
    """
    chosen: list[int] = []
    rest = list(range(len(obs_names)))
    traj = []
    while rest:
        scored = []
        for i in rest:
            res = _fit(sim_obs, observed, obs_names, chosen + [i], **kw)
            fits = float(res.per_observable["tv_distance"].max()) <= max_tv
            scored.append((fits, float(res.ess), i))
        # prefer candidates that actually fit; among those, the cheapest in ESS
        fits_ok = [c for c in scored if c[0] and c[1] >= min_ess]
        if not fits_ok:
            break
        _, e, best = max(fits_ok, key=lambda c: c[1])
        chosen.append(best)
        rest.remove(best)
        traj.append({"n_observables": len(chosen), "added": obs_names[best], "ess": e})
    return CoreResult(
        core=[obs_names[i] for i in chosen],
        blockers=[obs_names[i] for i in rest],
        core_ess=_ess(sim_obs, observed, obs_names, chosen, **kw) if chosen else float("nan"),
        trajectory=pd.DataFrame(traj),
    )


def conflict_ranking(
    sim_obs: np.ndarray,
    observed: dict[str, np.ndarray],
    obs_names: list[str],
    core: CoreResult,
    **kw,
) -> pd.DataFrame:
    """Rank blockers, separating COUPLING defects from plain unreachability.

    ``conflict = ESS(core + i) / (ESS_core * ESS_i / N)`` -- the joint cost relative to what
    independent tilting would predict from the blocker's own difficulty.

    Three kinds, and telling them apart is the whole point (without it every blocker looks
    alike and you chase the wrong ones):

    - ``unreachable`` -- the model cannot reproduce this observable even on its OWN
      (``solo_tv`` above ``max_tv``). A mechanism or parameter-range problem. Note this is
      detected by FIT, never by ESS: an unreachable observable costs *no* ESS, because the
      weights correctly stay uniform rather than straining toward data they cannot represent.
      Its ``solo_ess_fraction`` is near 1.0, which looks deceptively healthy.
    - ``coupling`` -- fits easily alone (high ``solo_ess_fraction``, low ``solo_tv``) yet
      cannot coexist with the core (``conflict`` well below 1). The model's *wiring* is
      wrong, not its level. These are the interesting ones.
    - ``costly`` -- fits alone, but only by heavy tilting (low ``solo_ess_fraction``). The
      model can reach it, but only in a thin corner of the cloud.
    """
    n = len(sim_obs)
    max_tv = kw.pop("max_tv", 0.05)
    idx = {nm: k for k, nm in enumerate(obs_names)}
    core_idx = [idx[nm] for nm in core.core]
    rows = []
    for nm in core.blockers:
        i = idx[nm]
        solo_res = _fit(sim_obs, observed, obs_names, [i], **kw)
        solo = float(solo_res.ess)
        solo_tv = float(solo_res.per_observable["tv_distance"].max())
        joint = _ess(sim_obs, observed, obs_names, core_idx + [i], **kw)
        expected = core.core_ess * solo / n
        conflict = joint / expected if expected > 0 else np.nan

        if solo_tv > max_tv:
            kind = "unreachable"
        elif conflict < 0.75 and solo / n > 0.5:
            kind = "coupling"
        elif solo / n < 0.25:
            kind = "costly"
        else:
            kind = "mixed"

        rows.append(
            {
                "observable": nm,
                "conflict": conflict,
                "solo_ess_fraction": solo / n,
                "solo_tv": solo_tv,
                "joint_ess": joint,
                "kind": kind,
            }
        )
    return pd.DataFrame(rows).sort_values("conflict", ignore_index=True)


def paired_effect_sizes(
    sim_obs: np.ndarray,
    observed: dict[str, np.ndarray],
    obs_names: list[str],
    pairs: list[tuple[str, str, str]],
) -> pd.DataFrame:
    """Compare the model's TREATMENT EFFECT to the observed one, per paired readout.

    ``pairs`` is ``(label, control_observable, treated_observable)``. Returns the model's
    median per-patient treated/control ratio next to the observed ratio of medians.

    No weighting is involved, so this is immune to every solver trap above. A model ratio of
    ~1.000 where the data show a real effect means the intervention is **inert** for that
    readout -- and if that holds across a whole panel, the two arms are the same simulation,
    their targets are near-duplicates, and no calibration can satisfy both.
    """
    idx = {nm: k for k, nm in enumerate(obs_names)}
    rows = []
    for label, ctrl, treat in pairs:
        if ctrl not in idx or treat not in idx:
            continue
        c, t = sim_obs[:, idx[ctrl]], sim_obs[:, idx[treat]]
        oc = np.asarray(observed[ctrl], float)
        ot = np.asarray(observed[treat], float)
        rows.append(
            {
                "readout": label,
                "model_ratio": float(np.median(t / c)),
                "observed_ratio": float(np.median(ot) / np.median(oc)),
                "model_arm_correlation": float(np.corrcoef(c, t)[0, 1]),
            }
        )
    df = pd.DataFrame(rows)
    df["inert_in_model"] = (df["model_ratio"] - 1.0).abs() < 0.02
    return df

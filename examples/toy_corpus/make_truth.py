#!/usr/bin/env python
"""Pick a truth ``phi*`` and write the cohort statistics it implies.

The data are what the *simulator* produced for ``n_c`` patients drawn off a truth
cloud, put through the corpus's own observable bodies and then through the fit's
own ``hard_row``. Two things follow from doing it that way rather than writing
plausible numbers by hand. The statistic the target reports is the estimator the
fit models it as, in the units the fit reads it in, so a log-versus-native slip
cannot hide. And the sampling noise in the numbers is real ``n_c`` noise, so
``V^boot`` is estimating something that is actually there.

``phi*`` is a draw from the fit's own prior, not a hand-set point: ``mu*`` from
``N(mu_0, Sigma_1)``, ``omega*`` from ``omega_0 exp(s* + u*)``, ``log R*`` from
its group prior. The discrepancy is zero, so a fit that invents an ``a`` or a
``b`` is inventing it.

Patients the init hook refuses are kept. Their trajectories are fine -- the hook
is a labelling rule, not a solver failure -- and the fit reads its emulator at
every patient in the cloud with no status gate, so screening them here would make
the truth cloud a different population from the one the fit models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

import model as M
from build_root import COHORTS, TARGETS

#: The truth is a draw, so it needs a seed. Separate from every seed the fit
#: uses: sharing one would let a fluctuation cancel between truth and fit.
PHI_SEED = 71237
CLOUD_SEED = 71238

#: How far the level of the spreads sits from the rubric. Inside the tau_s = 0.3
#: prior, and large enough that a fit which simply returned omega_0 would be
#: visibly wrong.
S_STAR = 0.25


def _rowspecs(target_id: str, cohort_id: str, n: int):
    """The rows this target declares, as the fit's own :class:`RowSpec`.

    Taken from the corpus declaration rather than from the YAML, because the
    YAML's statistics are what this script is about to write.

    ``log=False`` on every one of them, including the width rows. A target
    records what the source printed, in the units it printed it in, and
    ``build_problem`` is what takes the log of a scale row; writing ``hard_row``'s
    logged output into the YAML would have it logged twice.
    """
    from qsp_inference.vpop.rows import RowSpec

    t = next(t for t in TARGETS if t["id"] == target_id)
    return [RowSpec(target_id=target_id, cohort_id=cohort_id, stat=s["stat"],
                    value=float("nan"), n=n, p=s.get("p"), convention="type7",
                    convention_recorded=True, log=False)
            for s in t["stats"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--phi-seed", type=int, default=PHI_SEED)
    ap.add_argument("--cloud-seed", type=int, default=CLOUD_SEED)
    ap.add_argument("--s-star", type=float, default=S_STAR)
    ap.add_argument("--pdac-build", type=Path, required=True,
                    help="checkout holding workflows/vpop_fit.py, the driver "
                         "this corpus is built to be read by")
    args = ap.parse_args()

    sys.path.insert(0, str(args.pdac_build / "workflows"))
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import vpop_fit as V
    from omega_prior import build_omega_center, build_omega_corr

    from qsp_inference.priors.inference_prior import PriorSpec, build_prior_pair
    from qsp_inference.vpop.assemble import prior_cholesky, scenario_table
    from qsp_inference.vpop.predict import apply_margins
    from qsp_inference.vpop.readouts import build_h_fn
    from qsp_inference.vpop.rows import hard_row

    root = args.root
    inputs = V.resolve_inputs(root)
    cfg = V.load_config(root)
    K = cfg["constants"]

    spec = PriorSpec(priors_csv=str(inputs.priors_csv),
                     submodel_priors_yaml=str(inputs.submodel_priors),
                     derived_yaml=str(inputs.derived_priors),
                     proposal_temperature=1.0)
    L_sig, names = prior_cholesky(spec)
    names = list(names)
    pair = build_prior_pair(spec, verbose=False)
    mu_0 = np.array([m.ppf(0.5) for m in pair.prior._marginals])
    sd_1 = np.array([m.std() for m in pair.prior._marginals])
    omega_0, prov = build_omega_center(names, overrides_path=inputs.omega_roles)
    logit_mask = np.array([e.scale == "logit" for e in prov])
    R_bp, _ = build_omega_corr(names, axes_path=inputs.omega_axes,
                               loadings_path=inputs.omega_loadings)
    L_bp = np.linalg.cholesky(R_bp)

    targets, arm_of = V.load_targets(root, cfg["arms"])
    readouts = sorted(targets)
    aux_order, log_R_0, sigma_R = V.auxiliary_prior(root, targets, readouts)
    obs = V.load_observables(root)
    table = scenario_table(targets, arm_of, readouts)
    h_fn = build_h_fn(targets, readouts, list(obs.STATES), obs.observables,
                      aux_order=aux_order, scenario_of=table.scenario_of,
                      precomposed={})

    # -- phi*, a draw from the prior the fit will be given --
    rng = np.random.default_rng(args.phi_seed)
    mu_star = mu_0 + sd_1 * (np.asarray(L_sig) @ rng.standard_normal(len(names)))
    u_star = K["TAU_U"] * rng.standard_normal(len(names))
    u_star -= u_star.mean()
    omega_star = omega_0 * np.exp(args.s_star + u_star)
    log_R_star = log_R_0 + sigma_R * rng.standard_normal(len(log_R_0))

    print(f"phi* seed {args.phi_seed}")
    print(f"  mu* - mu_0, in prior sd:  median "
          f"{np.median(np.abs((mu_star - mu_0) / sd_1)):.2f}  "
          f"max {np.abs((mu_star - mu_0) / sd_1).max():.2f}")
    print(f"  omega*/omega_0: s* {args.s_star:+.2f} so x"
          f"{np.exp(args.s_star):.2f} on the level, u* up to x"
          f"{np.exp(np.abs(u_star).max()):.2f} on one parameter")
    print(f"  log R*: {dict(zip(aux_order, np.round(log_R_star, 3)))}")
    print("  a* = b* = 0: the corpus has no measurement discrepancy, so a fit "
          "that finds one is finding noise")

    # -- one patient cloud per cohort, honouring the declared blocks --
    n_of = {c["cohort_id"]: c["n_c"] for c in COHORTS}
    registry = yaml.safe_load(inputs.cohorts.read_text())
    # A counted block's members are the same people, so they draw once and the
    # smaller cohort takes a prefix. Drawing them apart would put a zero
    # off-diagonal in the truth that the fit's V says is nonzero.
    draw_group, size = {}, {}
    for b in registry.get("blocks") or []:
        members = sorted(b["cohorts"], key=lambda c: -n_of[c])
        for m in members:
            draw_group[m] = members[0]
        size[members[0]] = max(n_of[m] for m in members)
    for c in n_of:
        draw_group.setdefault(c, c)
        size.setdefault(c, n_of[c])

    cloud_rng = np.random.default_rng(args.cloud_seed)
    theta_of = {}
    for g in sorted(set(draw_group.values())):
        z = cloud_rng.standard_normal((size[g], len(names))) @ L_bp.T
        theta_of[g] = np.asarray(np.exp(apply_margins(
            np.broadcast_to(mu_star, (size[g], len(names))),
            omega_star, z, jnp.asarray(logit_mask))), dtype=float)

    # -- the simulator, on every scenario, for every drawn patient --
    stats_of: dict[str, list] = {}
    for cohort_id in sorted(n_of):
        theta = theta_of[draw_group[cohort_id]][:n_of[cohort_id]]
        block = []
        failed = 0
        for arm, t in table.scenarios:
            y, status = M.solve_many(theta, arm, M.READOUT_TIMES)
            failed += int((status == M.STATUS_FAILED).sum())
            block.append(y[:, M.READOUT_TIMES.index(t), :])
        if failed:
            raise SystemExit(
                f"{cohort_id}: the solver failed on {failed} patient-scenarios. "
                f"A truth cohort with a hole in it is not a cohort; redraw with "
                f"another --cloud-seed or narrow omega*.")
        x = np.exp(np.asarray(h_fn(jnp.asarray(np.stack(block)),
                                   jnp.asarray(log_R_star)), dtype=float))
        for tid in readouts:
            if targets[tid].get("cohort_id") != cohort_id:
                continue
            col = x[:, readouts.index(tid)]
            stats_of[tid] = [
                {"stat": s.stat, **({"p": s.p} if s.p is not None else {}),
                 "value": float(hard_row(s, col))}
                for s in _rowspecs(tid, cohort_id, n_of[cohort_id])]

    # -- write them back into the targets --
    for tid, entries in stats_of.items():
        path = next(p for p in (root / "calibration_targets").rglob(f"{tid}.yaml"))
        doc = yaml.safe_load(path.read_text())
        doc["empirical_data"]["observed_distribution"]["statistics"] = entries
        path.write_text(yaml.safe_dump(doc, sort_keys=False))

    (root / "truth_phi.json").write_text(json.dumps({
        "phi_seed": args.phi_seed, "cloud_seed": args.cloud_seed,
        "s_star": args.s_star,
        "param_names": names, "aux_order": list(aux_order),
        "mu_star": mu_star.tolist(), "omega_star": omega_star.tolist(),
        "u_star": u_star.tolist(), "log_R_star": log_R_star.tolist(),
        "mu_0": mu_0.tolist(), "sd_1": sd_1.tolist(),
        "omega_0": omega_0.tolist(),
        "a_star": [0.0] * 0, "b_star": [0.0] * 0,
        "note": "a and b are zero at every length; the fit's Z decides how many "
                "columns that is, and the corpus asserts no discrepancy on any "
                "of them",
    }, indent=2))

    n_rows = sum(len(v) for v in stats_of.values())
    print(f"\n{n_rows} statistics over {len(stats_of)} targets -> "
          f"{root/'calibration_targets'}")
    print(f"truth phi* -> {root/'truth_phi.json'}")


if __name__ == "__main__":
    main()

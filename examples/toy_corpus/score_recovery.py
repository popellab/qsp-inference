#!/usr/bin/env python
"""Score a posterior against the ``phi*`` that generated its data.

Reads ``truth_phi.json`` from ``make_truth`` and the ``.npz`` the fit's ``nuts``
stage wrote, and reports per block whether the truth is inside the posterior and
whether the posterior moved off the prior at all. Those are separate questions
and the second decides how to read the first.

``omega`` is scored on the log scale, where its prior is the Gaussian
``s + u`` and a shrinkage factor means what it says.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from qsp_inference.vpop.recovery import print_recovery, summarise_recovery


def _centre(u, axis=-1):
    """``u - mean(u)``, which is the only part of it eq:omegaassumed uses."""
    return u - np.mean(u, axis=axis, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--draws", type=Path, required=True)
    ap.add_argument("--level", type=float, default=0.9)
    args = ap.parse_args()

    star = json.loads(args.truth.read_text())
    d = np.load(args.draws, allow_pickle=True)
    names = [str(n) for n in d["param_names"]]
    cols = [str(c) for c in d["z_columns"]]
    aux = [str(a) for a in d["aux_order"]]
    assumed = np.asarray(d["assumed"], dtype=int)

    if names != list(star["param_names"]):
        raise SystemExit(
            "the posterior and the truth are indexed by different parameters, "
            "so every comparison below would be against the wrong name")

    tau_s = float(d["prior_tau_s"])
    tau_u = float(d["prior_tau_u"])
    omega_star = np.asarray(star["omega_star"], dtype=float)

    truth = {
        "mu": np.asarray(star["mu_star"], dtype=float),
        # The width, on the scale its prior is Gaussian on.
        "log_omega": np.log(omega_star),
        "s": np.array([star["s_star"]], dtype=float),
        # Centred, as build_omega centres it. Only the centred vector reaches
        # omega, so the raw one carries a free location that no data can see and
        # scoring it would report a shift the model does not have.
        "u": _centre(np.asarray(star["u_star"], dtype=float)[assumed]),
        # No discrepancy was applied, so both are exactly zero.
        "a": np.zeros(len(cols)), "b": np.zeros(len(cols)),
        "log_R": np.asarray(star["log_R_star"], dtype=float),
    }
    draws = {
        "mu": d["mu"], "log_omega": np.log(d["omega"]),
        "s": np.asarray(d["s"]).reshape(-1, 1),
        "u": _centre(np.asarray(d["u_raw"], dtype=float), axis=1),
        "a": d["a"], "b": d["b"], "log_R": d["log_R"],
    }
    # The prior sd each component is measured against. omega's is the two spread
    # terms in quadrature: eq:omegaassumed's level and its pattern both move it.
    prior_sd = {
        "mu": np.asarray(d["prior_sd_1"], dtype=float),
        "log_omega": np.full(len(names), np.hypot(tau_s, tau_u)),
        "s": np.array([tau_s]), "u": np.full(len(assumed), tau_u),
        "a": np.full(len(cols), float(d["prior_sigma_a"])),
        "b": np.full(len(cols), float(d["prior_sigma_b"])),
        "log_R": np.asarray(d["prior_sigma_R"], dtype=float),
    }
    labels = {
        "mu": names, "log_omega": names, "s": ["s"],
        "u": [names[i] for i in assumed],
        "a": cols, "b": cols, "log_R": aux,
    }

    rows = summarise_recovery(truth, draws, prior_sd, names=labels,
                              level=args.level)
    for line in print_recovery(rows, level=args.level):
        print(line)

    # The one comparison a per-component table cannot make: omega's level against
    # the truth's, which is what a virtual population is reported in.
    om = np.asarray(d["omega"], dtype=float)
    ratio = np.median(om / omega_star[None, :], axis=1)
    print(f"\nomega / omega*, median over parameters: posterior mean "
          f"x{ratio.mean():.2f}, {args.level:.0%} interval "
          f"[{np.quantile(ratio, (1-args.level)/2):.2f}, "
          f"{np.quantile(ratio, 1-(1-args.level)/2):.2f}]")
    om0 = np.asarray(d["prior_omega_0"], dtype=float)
    print(f"  for reference, omega_0 / omega* is x"
          f"{np.median(om0 / omega_star):.2f}: that is what the fit returns if "
          f"the corpus says nothing about the width")


if __name__ == "__main__":
    main()

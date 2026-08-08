#!/usr/bin/env python
"""The whole recovery study, end to end, in one command.

    python examples/toy_corpus/run.py --work <dir> --pdac-build <checkout>

Six steps, each of which is a script that can be run on its own:

    build_root      write the project root
    make_truth      pick phi*, generate the cohort data it implies
    make_pool       the emulator design, from campaign/make_pool.py unchanged
    simulate        solve it
    train_emulator  train the surrogate, from campaign/train_emulator.py unchanged
    vpop_fit        the population fit, from workflows/vpop_fit.py unchanged
    score_recovery  the posterior against phi*

The three that come from ``--pdac-build`` are the point. They are the production
driver and its campaign scripts, reading a corpus they have never seen, with no
flag saying which corpus it is: everything that differs travels in the root.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(label: str, cmd: list[str], *, log: Path) -> None:
    print(f"\n=== {label}")
    t0 = time.time()
    with log.open("w") as fh:
        proc = subprocess.run([str(c) for c in cmd], stdout=fh,
                              stderr=subprocess.STDOUT, cwd=HERE)
    tail = log.read_text().splitlines()[-25:]
    print("\n".join(tail))
    if proc.returncode:
        raise SystemExit(f"{label} failed ({proc.returncode}); see {log}")
    print(f"--- {label}: {time.time() - t0:.0f}s, full log at {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True,
                    help="everything this run writes goes under here")
    ap.add_argument("--pdac-build", type=Path, required=True,
                    help="checkout holding workflows/vpop_fit.py and "
                         "workflows/campaign/")
    ap.add_argument("--n-total", type=int, default=24576,
                    help="emulator design rows")
    ap.add_argument("--n-score", type=int, default=4096,
                    help="rows held out of training, which E_B is measured on")
    ap.add_argument("--jobs", type=int, default=1, help="simulator processes")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--hidden", default="256,256,128")
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--max-tree-depth", type=int, default=8)
    ap.add_argument("--from-step", default="build_root",
                    choices=("build_root", "make_truth", "make_pool", "simulate",
                             "train", "fit", "score"),
                    help="resume; every earlier step's output is reused as is")
    args = ap.parse_args()

    work = args.work.resolve()
    root, pool, emu = work / "root", work / "pool", work / "emulator"
    logs = work / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    wf = args.pdac_build.resolve() / "workflows"
    if not (wf / "vpop_fit.py").exists():
        raise SystemExit(f"{wf}/vpop_fit.py not found; --pdac-build is wrong")
    py = sys.executable
    draws = work / "posterior_draws.npz"

    order = ("build_root", "make_truth", "make_pool", "simulate", "train",
             "fit", "score")
    start = order.index(args.from_step)

    steps = {
        "build_root": ("build root", [py, HERE / "build_root.py", "--root", root]),
        "make_truth": ("truth and data", [py, HERE / "make_truth.py",
                                          "--root", root,
                                          "--pdac-build", args.pdac_build]),
        "make_pool": ("emulator design", [py, wf / "campaign/make_pool.py",
                                          "--root", root, "--out", pool,
                                          "--n-total", args.n_total,
                                          "--n-score", args.n_score]),
        "simulate": ("simulate the design", [py, HERE / "simulate.py",
                                             "--pool-dir", pool,
                                             "--jobs", args.jobs]),
        "train": ("train the emulator", [py, wf / "campaign/train_emulator.py",
                                         "--train-dir", pool, "--out", emu,
                                         "--hidden", args.hidden,
                                         "--max-epochs", args.epochs]),
        "fit": ("population fit", [py, wf / "vpop_fit.py", "--root", root,
                                   "--emulator-dir", emu, "--pool-dir", pool,
                                   "--stage", "nuts", "--warmup", args.warmup,
                                   "--samples", args.samples,
                                   "--chains", args.chains,
                                   "--max-tree-depth", args.max_tree_depth,
                                   "--out", draws]),
        "score": ("recovery", [py, HERE / "score_recovery.py",
                               "--truth", root / "truth_phi.json",
                               "--draws", draws]),
    }
    for name in order[start:]:
        label, cmd = steps[name]
        run(label, cmd, log=logs / f"{name}.log")


if __name__ == "__main__":
    main()

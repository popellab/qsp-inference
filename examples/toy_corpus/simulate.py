#!/usr/bin/env python
"""Solve the emulator pool. Stands in for the cluster campaign.

Reads the ``theta.csv`` ``campaign/make_pool.py`` wrote and emits one
``train_<arm>.parquet`` per arm in the format ``train_emulator`` and
``vpop_fit.pool_readouts`` both read: ``sample_index``, one ``param:`` column per
parameter, one ``sp:<species>@<time>`` column per species and readout time, and
the solver ``status``.

Refused patients keep their species. The hook is a labelling rule here, so the
trajectory exists; ``train_emulator`` drops them from the species head by status
and keeps them for the status head, exactly as it does on the real campaign.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import model as M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-dir", type=Path, required=True,
                    help="make_pool's output; the parquets are written beside "
                         "theta.csv so pool_meta.json is where both readers "
                         "expect it")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="solve only the first N rows, for a smoke run")
    args = ap.parse_args()

    meta = json.loads((args.pool_dir / "pool_meta.json").read_text())
    df = pd.read_csv(args.pool_dir / "theta.csv")
    if args.limit:
        df = df.iloc[:args.limit]
    names = [c for c in df.columns if c != "sample_index"]
    if names != list(M.PARAM_NAMES):
        raise SystemExit(
            f"theta.csv carries {len(names)} parameters in an order this model "
            f"does not use; the pool and the model disagree about what a column "
            f"means. First difference: "
            f"{next(a for a, b in zip(names, M.PARAM_NAMES) if a != b)}")
    theta = df[names].to_numpy(dtype=float)
    print(f"{len(theta)} rows x {len(M.ARMS)} arms, {meta['n_design']} of them "
          f"the design")

    for arm in M.ARMS:
        y, status = M.solve_many(theta, arm, M.READOUT_TIMES, jobs=args.jobs)
        out = pd.DataFrame({"sample_index": df["sample_index"].to_numpy()})
        for j, name in enumerate(names):
            out[f"param:{name}"] = theta[:, j]
        for k, t in enumerate(M.READOUT_TIMES):
            for q, sp in enumerate(M.SPECIES):
                out[f"sp:{sp}@{t:g}"] = y[:, k, q]
        out["status"] = status
        path = args.pool_dir / f"train_{arm}.parquet"
        out.to_parquet(path, index=False)
        counts = {int(c): int((status == c).sum()) for c in np.unique(status)}
        print(f"  {arm:<10} -> {path.name}  status {counts}")


if __name__ == "__main__":
    main()

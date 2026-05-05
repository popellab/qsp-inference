#!/usr/bin/env python3
"""Re-run submodel inference and regenerate ``submodel_priors.yaml``.

Fast iteration loop for Stage 1: re-MCMC the components affected by an edit
to a SubmodelTarget YAML or a prior CSV row, reuse cached posteriors for
everything else, and rewrite ``submodel_priors.yaml`` so Stage 2 can
reload it. Skips the slow parts of ``qsp-audit report`` (PPC simulations,
markdown report, plots).

Use ``qsp-audit report`` for the full audit when you want the report.
Use this script when you just want fresh priors.

What this DOES:
    1. Re-runs MCMC for components containing any ``--invalidate``-d
       parameter (others reuse their ``.compare_cache/`` entries). When
       the cache is empty, ``--rebuild-all`` forces inference on every
       component.
    2. Re-aggregates the joint posterior cache (marginals + per-component
       Gaussian copula) and rewrites ``submodel_priors.yaml`` at the
       output path, including freshness fingerprints per component.

What this DOES NOT do:
    - Posterior predictive checks
    - Markdown audit report
    - Plots
    - Stage 2 cross-checks

Usage:
    python examples/regen_submodel_priors.py
        # Re-aggregate the existing cache. Errors out if the cache is empty.

    python examples/regen_submodel_priors.py --invalidate APC0_cDC2_T
        # Re-run MCMC for the component(s) containing APC0_cDC2_T,
        # then re-aggregate. Other components reuse their cached posteriors.

    python examples/regen_submodel_priors.py --invalidate q_CD8_T_in k_Mac_rec
        # Multi-param invalidation.

    python examples/regen_submodel_priors.py --rebuild-all
        # Wipe the cache and run inference for every component (slow).

    python examples/regen_submodel_priors.py --output some/path.yaml
        # Override the default output path.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Internal helpers from the audit machinery. ``_run_inference`` and
# ``_write_submodel_priors`` are currently private to qsp_inference.audit.report;
# they're used here because the full ``qsp-audit report`` flow bundles them
# with PPC + report generation, which are the slow steps this script skips.
from qsp_inference.audit.report import (
    AuditConfig,
    _run_inference,
    _write_submodel_priors,
    load_freshness_by_component,
    load_joint_samples,
    load_joint_samples_by_component,
    load_parameter_groups,
    load_submodel_targets,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run submodel inference and regenerate submodel_priors.yaml."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd).",
    )
    parser.add_argument(
        "--invalidate",
        nargs="+",
        metavar="PARAM",
        default=None,
        help=(
            "Re-run MCMC for components containing these parameters. "
            "Other components reuse their cached posteriors. Pass nothing "
            "to skip re-inference and just re-aggregate the existing cache."
        ),
    )
    parser.add_argument(
        "--rebuild-all",
        action="store_true",
        help=(
            "Force a full inference pass: wipe the cache and re-run MCMC "
            "for every component. Use this on a clean slate or when you "
            "want to guarantee no stale cache hits. Slow."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output path for submodel_priors.yaml "
            "(default: <project-root>/submodel_priors.yaml)."
        ),
    )
    args = parser.parse_args()

    config = AuditConfig(project_root=args.project_root)
    output = args.output or args.project_root / "submodel_priors.yaml"

    cache_empty = not config.compare_cache.exists() or not any(
        config.compare_cache.glob("comp_*.json")
    )

    if args.rebuild_all:
        if config.compare_cache.exists():
            shutil.rmtree(config.compare_cache)
            print(f"Wiped cache: {config.compare_cache}")
        _run_inference(config, invalidate_params=None)
    elif args.invalidate:
        # Cached components are skipped; only those containing the named
        # parameters get re-MCMCed. Same machinery the full audit uses.
        _run_inference(config, invalidate_params=args.invalidate)
    elif cache_empty:
        print(
            f"Cache at {config.compare_cache} is empty; running full "
            "inference to populate it (equivalent to --rebuild-all)."
        )
        _run_inference(config, invalidate_params=None)

    joint_samples = load_joint_samples(config.compare_cache)
    if not joint_samples:
        print(f"No cached joint samples in {config.compare_cache}.")
        print("Inference appears to have failed; check upstream logs.")
        return 1

    # _write_submodel_priors expects samples grouped by component_id so
    # the copula is built block-diagonally per inference component.
    joint_samples_by_component = load_joint_samples_by_component(config.compare_cache)
    if not joint_samples_by_component:
        print(f"No per-component samples in {config.compare_cache}.")
        return 1

    targets = load_submodel_targets(config.submodel_dir)
    groups = load_parameter_groups(config.param_groups)
    freshness_by_component = load_freshness_by_component(config.compare_cache)

    print(f"Cache: {config.compare_cache}")
    print(
        f"Loaded joint samples for {len(joint_samples)} parameters "
        f"across {len(joint_samples_by_component)} components."
    )
    if freshness_by_component:
        print(
            f"Stamping content fingerprints for "
            f"{len(freshness_by_component)} components into metadata.freshness."
        )
    else:
        print(
            "WARNING: no freshness manifests found in cache "
            "(comp_*.json predates fingerprinting). Re-run with "
            "--rebuild-all to add them."
        )
    print(f"Writing -> {output}")
    _write_submodel_priors(
        joint_samples_by_component,
        targets,
        groups,
        output,
        freshness_by_component=freshness_by_component,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

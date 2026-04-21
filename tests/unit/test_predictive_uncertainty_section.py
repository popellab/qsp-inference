"""Tests for the ``Clinical predictive uncertainty`` audit section.

Covers the synthesis logic: CI95 stats per (scenario, endpoint), Spearman
attribution, zero-variance filtering, undefined-median handling, and CSV
export. Uses synthetic posteriors + PPC DataFrames so no simulator is
required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qsp_inference.audit.report import _section_predictive_uncertainty


def _synthetic_sbi_run(
    n_draws: int = 40,
    scenarios: tuple[str, ...] = ("arm1", "arm2"),
    seed: int = 0,
) -> dict:
    """Build a minimal ``sbi_run`` dict: param_a drives tumor; param_flat
    has zero variance; PPC values encode a clear Spearman signal against
    param_a so the driver ranking is testable."""
    rng = np.random.default_rng(seed)
    post_df = pd.DataFrame(
        {
            "param_a": rng.lognormal(0, 0.5, n_draws),
            "param_b": rng.lognormal(0, 0.3, n_draws),
            "param_flat": np.ones(n_draws),
        }
    )

    rows = []
    for si in range(n_draws):
        a = float(post_df.iloc[si]["param_a"])
        for scen in scenarios:
            rows.append((si, 0, scen, "os_at_12mo", 1.0 if a > 1.0 else 0.0))
            rows.append((si, 0, scen, "tumor_diam_365", 2.0 * a + rng.normal(0, 0.05)))
            rows.append((si, 0, scen, "recist_cr_at_day_90", 0.0))  # zero-median
    ppc_df = pd.DataFrame(
        rows, columns=["sample_index", "status", "scenario", "endpoint", "value"]
    )

    return {
        "run_path": Path("/tmp/fake_sbi_run"),
        "posterior_samples": post_df,
        "ppc_clinical": ppc_df,
        "observed": {},
        "z_score_contraction": None,
        "local_calibration": None,
        "metadata": {},
    }


class TestSectionPredictiveUncertainty:
    def test_returns_empty_when_sbi_run_none(self):
        assert _section_predictive_uncertainty(None, {}, None, None) == []

    def test_returns_empty_when_ppc_absent(self):
        sbi_run = _synthetic_sbi_run()
        sbi_run["ppc_clinical"] = None
        assert _section_predictive_uncertainty(sbi_run, {}, None, None) == []

    def test_emits_heading_and_table(self, tmp_path: Path):
        sbi_run = _synthetic_sbi_run()
        lines = _section_predictive_uncertainty(sbi_run, {}, None, tmp_path)
        text = "\n".join(lines)
        assert "## Clinical predictive uncertainty" in text
        # CI CSV carries one row per (scenario, endpoint) deterministically;
        # markdown row counting collides with the driver-attribution table
        # which shares the ``| `armN`...`` prefix.
        ci_df = pd.read_csv(tmp_path / "ppc_endpoint_ci.csv")
        assert len(ci_df) == 6  # 2 scenarios × 3 endpoints

    def test_zero_median_endpoint_is_undefined_frac(self, tmp_path: Path):
        """recist_cr_at_day_90 has zero values in both scenarios → undefined
        CI95/|median|. Whether os_at_12mo lands undefined is seed-sensitive
        (depends on whether majority-alive or majority-dead), so we just
        check the lower bound."""
        sbi_run = _synthetic_sbi_run()
        lines = _section_predictive_uncertainty(sbi_run, {}, None, tmp_path)
        headline = next(ln for ln in lines if "undefined" in ln)
        import re
        m = re.search(r"\*\*(\d+)\*\* are undefined", headline)
        assert m is not None
        n_undefined = int(m.group(1))
        assert n_undefined >= 2

    def test_driver_attribution_identifies_param_a(self, tmp_path: Path):
        sbi_run = _synthetic_sbi_run(n_draws=80)
        lines = _section_predictive_uncertainty(sbi_run, {}, None, tmp_path)
        text = "\n".join(lines)
        # param_a engineered to drive tumor_diam_365 with ρ ≈ +1.0.
        tumor_lines = [ln for ln in lines if "tumor_diam_365" in ln and "|" in ln]
        tumor_drivers = [ln for ln in tumor_lines if "param_a" in ln]
        assert tumor_drivers
        # param_flat has zero variance → must be filtered out everywhere.
        assert "param_flat" not in text

    def test_prcc_restriction_filters_drivers(self, tmp_path: Path):
        sbi_run = _synthetic_sbi_run(n_draws=80)
        # Only param_b is "significant" — param_a should drop out of the
        # driver table even though it has the stronger correlation.
        prcc = {"param_b": {"significant": True}, "param_a": {"significant": False}}
        lines = _section_predictive_uncertainty(sbi_run, {}, prcc, tmp_path)
        driver_rows = [ln for ln in lines if "|" in ln and "tumor_diam_365" in ln and "`param_" in ln]
        # At least one driver row; none should mention param_a (filtered out)
        assert driver_rows
        combined = "\n".join(driver_rows)
        assert "param_a" not in combined
        assert "param_b" in combined

    def test_csv_artifacts_are_written(self, tmp_path: Path):
        # output_dir's sibling "calibration" doesn't exist → function falls
        # back to writing into output_dir itself.
        sbi_run = _synthetic_sbi_run()
        out_dir = tmp_path / "report_out"
        out_dir.mkdir()
        _section_predictive_uncertainty(sbi_run, {}, None, out_dir)
        ci = out_dir / "ppc_endpoint_ci.csv"
        corr = out_dir / "ppc_endpoint_correlations.csv"
        assert ci.exists()
        assert corr.exists()
        ci_df = pd.read_csv(ci)
        assert {"scenario", "endpoint", "p05", "median", "p95"} <= set(ci_df.columns)
        corr_df = pd.read_csv(corr)
        assert {"scenario", "endpoint", "parameter", "spearman_rho"} <= set(corr_df.columns)

    def test_prefers_sibling_calibration_dir_when_present(self, tmp_path: Path):
        """pdac-build layout: report lives in notes/calibration/ but the
        function is given notes/calibration-ish paths; CSV artifacts should
        co-locate with the report."""
        calibration_dir = tmp_path / "calibration"
        calibration_dir.mkdir()
        sbi_run = _synthetic_sbi_run()
        _section_predictive_uncertainty(sbi_run, {}, None, calibration_dir)
        assert (calibration_dir / "ppc_endpoint_ci.csv").exists()

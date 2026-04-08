#!/usr/bin/env python3
"""
Python Test Statistic Functions for QSP Simulations

This module contains Python implementations of test statistic functions
that were originally written in MATLAB. These functions compute observables
from QSP simulation outputs (time series of species concentrations).

Each function takes:
  - time: np.ndarray - time vector (days)
  - species_1, species_2, ...: np.ndarray - species concentrations over time

Returns:
  - float: computed test statistic value

The functions are designed to work with data extracted from full simulation
outputs stored in Parquet format on HPC.
"""

import numpy as np
from typing import Dict, Callable


def tumor_growth_rate_0_60d(time: np.ndarray, V_T_C1: np.ndarray) -> float:
    """
    Compute tumor log growth rate over days 0-60.

    Estimates exponential growth rate (1/day) from tumor cancer cell counts
    by linear regression of ln(V_T.C1) vs time.

    Args:
        time: Time vector (days)
        V_T_C1: Tumor cancer cell counts (cells) corresponding to time

    Returns:
        Estimated growth rate r_hat (1/day)
    """
    t0, t1 = 0, 60

    if len(time) == 0 or len(V_T_C1) == 0:
        return np.nan

    # Create daily evaluation grid
    t_eval = np.arange(t0, t1 + 1, 1, dtype=float)

    # Interpolate V_T.C1 on the grid
    c1_interp = np.interp(t_eval, time, V_T_C1)

    # Protect against non-positive values
    c1_interp = np.maximum(c1_interp, np.finfo(float).eps)

    # Log-transform
    y = np.log(c1_interp)

    # Compute OLS slope
    slope = np.polyfit(t_eval, y, 1)[0]  # [slope, intercept]

    return float(slope)


def cd8_treg_ratio_baseline(time: np.ndarray, V_T_CD8: np.ndarray, V_T_Treg: np.ndarray) -> float:
    """
    Compute baseline intratumoral CD8/Treg ratio at day 0.

    Args:
        time: Time vector (days)
        V_T_CD8: CD8+ T cell counts in tumor (cells)
        V_T_Treg: Regulatory T cell counts in tumor (cells)

    Returns:
        CD8/Treg ratio at baseline (dimensionless)
    """
    t_query = 0.0  # days

    # Interpolate species at baseline
    cd8_0 = np.interp(t_query, time, V_T_CD8)
    treg_0 = np.interp(t_query, time, V_T_Treg)

    # Numerical guard to avoid division by zero
    eps_denom = 1.0e-12

    return float(cd8_0 / max(treg_0, eps_denom))


def m1_m2_ratio_baseline(time: np.ndarray, V_T_Mac_M1: np.ndarray, V_T_Mac_M2: np.ndarray) -> float:
    """
    Compute baseline tumor M1/M2 macrophage ratio at day 0.

    Args:
        time: Time vector (days)
        V_T_Mac_M1: M1 macrophage counts in tumor (cells)
        V_T_Mac_M2: M2 macrophage counts in tumor (cells)

    Returns:
        M1/M2 ratio at baseline (dimensionless)
    """
    t_eval = 0.0  # days

    # Interpolate species at day 0
    M1_0 = np.interp(t_eval, time, V_T_Mac_M1)
    M2_0 = np.interp(t_eval, time, V_T_Mac_M2)

    # Validate values
    if not np.isfinite(M1_0) or not np.isfinite(M2_0):
        return np.nan

    # Guard against non-physical values
    if M1_0 < 0 or M2_0 < 0:
        return np.nan

    if M2_0 == 0:
        return np.nan  # undefined ratio if denominator is zero

    return float(M1_0 / M2_0)


def mdsc_cd8_ratio_baseline(time: np.ndarray, V_T_MDSC: np.ndarray, V_T_CD8: np.ndarray) -> float:
    """
    Compute baseline MDSC:CD8 ratio in tumor at day 0.

    Args:
        time: Time vector (days)
        V_T_MDSC: MDSC counts in tumor (cells)
        V_T_CD8: CD8 T cell counts in tumor (cells)

    Returns:
        MDSC/CD8 ratio at baseline (dimensionless)
    """
    t_assess = 0  # days

    # Interpolate species at baseline
    M0 = np.interp(t_assess, time, V_T_MDSC)
    E0 = np.interp(t_assess, time, V_T_CD8)

    # Numerical safety for division
    eps_den = 1e-12

    return float(M0 / max(E0, eps_den))


def apc_maturation_ratio_baseline(time: np.ndarray, V_T_mAPC: np.ndarray, V_T_APC: np.ndarray) -> float:
    """
    Compute baseline fraction of tumor APC that are mature at day 0.

    Args:
        time: Time vector (days)
        V_T_mAPC: Mature APC counts in tumor (cells)
        V_T_APC: Immature APC counts in tumor (cells)

    Returns:
        Fraction of mature APC at baseline (dimensionless, [0,1])
    """
    t_eval = 0  # days

    # Interpolate species at day 0
    mAPC0 = np.interp(t_eval, time, V_T_mAPC)
    APC0 = np.interp(t_eval, time, V_T_APC)

    denom = APC0 + mAPC0

    if not np.isfinite(denom) or denom <= 0:
        return np.nan  # undefined if no APC present

    return float(mAPC0 / denom)


def exhausted_cd8_fraction_baseline(time: np.ndarray, V_T_CD8_exh: np.ndarray, V_T_CD8: np.ndarray) -> float:
    """
    Compute baseline fraction of exhausted CD8 T cells at day 0.

    Args:
        time: Time vector (days)
        V_T_CD8_exh: Exhausted CD8 T cell counts in tumor (cells)
        V_T_CD8: Active CD8 T cell counts in tumor (cells)

    Returns:
        Fraction of exhausted CD8 at baseline (dimensionless, [0,1])
    """
    t_assess = 0  # days

    # Interpolate species at day 0
    CD8_exh_0 = np.interp(t_assess, time, V_T_CD8_exh)
    CD8_0 = np.interp(t_assess, time, V_T_CD8)

    denom = CD8_exh_0 + CD8_0

    if np.isnan(denom) or denom <= 0:
        return np.nan  # undefined if no CD8 TILs

    frac = CD8_exh_0 / denom

    # Numerical safety: clamp to [0,1]
    frac = max(0.0, min(1.0, frac))

    return float(frac)


def til_per_cancercell_baseline(
    time: np.ndarray,
    V_T_CD8: np.ndarray,
    V_T_Th: np.ndarray,
    V_T_Treg: np.ndarray,
    V_T_C1: np.ndarray
) -> float:
    """
    Compute baseline TILs per cancer cell at day 0.

    Args:
        time: Time vector (days)
        V_T_CD8: CD8+ T cell counts in tumor (cells)
        V_T_Th: CD4+ helper T cell counts in tumor (cells)
        V_T_Treg: Regulatory T cell counts in tumor (cells)
        V_T_C1: Cancer cell counts in tumor (cells)

    Returns:
        TILs per cancer cell at baseline (cells per cancer cell, dimensionless)
    """
    t0 = 0  # days

    # Interpolate baseline values
    CD8_0 = np.interp(t0, time, V_T_CD8)
    Th_0 = np.interp(t0, time, V_T_Th)
    Treg_0 = np.interp(t0, time, V_T_Treg)
    C1_0 = np.interp(t0, time, V_T_C1)

    # Guard against invalid denominator
    if np.isnan(C1_0) or C1_0 <= 0:
        return np.nan

    # Compute baseline TILs per cancer cell
    return float((CD8_0 + Th_0 + Treg_0) / C1_0)


def tgfb_concentration_baseline(time: np.ndarray, V_T_TGFb: np.ndarray) -> float:
    """
    Compute baseline tumor TGF-β concentration at day 0.

    Args:
        time: Time vector (days)
        V_T_TGFb: Tumor TGF-β concentration (nM)

    Returns:
        Baseline TGF-β concentration (pg/mL)
    """
    t0 = 0.0  # days

    # TGFβ: MW = 25 kDa
    # Convert nM to pg/mL
    TGFb_pgmL = V_T_TGFb * 25000.0

    # Interpolate to day 0
    baseline_TGFb = np.interp(t0, time, TGFb_pgmL)

    return float(baseline_TGFb)


def ccl2_concentration_baseline(time: np.ndarray, V_T_CCL2: np.ndarray) -> float:
    """
    Compute baseline tumor CCL2 concentration at day 0.

    Args:
        time: Time vector (days)
        V_T_CCL2: Tumor CCL2 concentration (nM)

    Returns:
        Baseline CCL2 concentration (pg/mL)
    """
    # Sanity checks
    if len(time) == 0 or len(V_T_CCL2) == 0 or len(time) != len(V_T_CCL2):
        return np.nan

    # CCL2: MW = 13 kDa
    # Convert nM to pg/mL
    CCL2_pgmL = V_T_CCL2 * 13000.0

    # Interpolate to day 0 (baseline)
    test_statistic = np.interp(0.0, time, CCL2_pgmL)

    # Ensure non-negative (biological constraint)
    if not np.isfinite(test_statistic) or test_statistic < 0:
        test_statistic = max(0.0, test_statistic)

    return float(test_statistic)


# ============================================================================
# TEMPORAL TEST STATISTICS (measuring dynamics after equilibration)
# ============================================================================

def cd8_treg_ratio_7d(time: np.ndarray, V_T_CD8: np.ndarray, V_T_Treg: np.ndarray) -> float:
    """
    Compute CD8/Treg ratio at day 7 after equilibration.

    Measures early immune infiltration dynamics.

    Args:
        time: Time vector (days, starts at 0 after equilibration)
        V_T_CD8: CD8+ T cell counts in tumor (cells)
        V_T_Treg: Regulatory T cell counts in tumor (cells)

    Returns:
        CD8/Treg ratio at day 7 (dimensionless)
    """
    t_assess = 7.0  # days

    # Interpolate species at day 7
    cd8_7 = np.interp(t_assess, time, V_T_CD8)
    treg_7 = np.interp(t_assess, time, V_T_Treg)

    # Numerical guard
    eps_denom = 1.0e-12

    return float(cd8_7 / max(treg_7, eps_denom))


def cd8_treg_ratio_14d(time: np.ndarray, V_T_CD8: np.ndarray, V_T_Treg: np.ndarray) -> float:
    """
    Compute CD8/Treg ratio at day 14 after equilibration.

    Measures intermediate immune infiltration dynamics.

    Args:
        time: Time vector (days)
        V_T_CD8: CD8+ T cell counts in tumor (cells)
        V_T_Treg: Regulatory T cell counts in tumor (cells)

    Returns:
        CD8/Treg ratio at day 14 (dimensionless)
    """
    t_assess = 14.0  # days

    cd8_14 = np.interp(t_assess, time, V_T_CD8)
    treg_14 = np.interp(t_assess, time, V_T_Treg)

    eps_denom = 1.0e-12
    return float(cd8_14 / max(treg_14, eps_denom))


def tumor_fold_change_0_14d(time: np.ndarray, V_T_C1: np.ndarray) -> float:
    """
    Compute tumor size fold-change from day 0 to day 14.

    Measures tumor growth dynamics after equilibration.

    Args:
        time: Time vector (days)
        V_T_C1: Tumor cancer cell counts (cells)

    Returns:
        Fold change in tumor size (dimensionless)
    """
    # Get tumor size at day 0 and day 14
    c1_0 = np.interp(0.0, time, V_T_C1)
    c1_14 = np.interp(14.0, time, V_T_C1)

    if np.isnan(c1_0) or np.isnan(c1_14) or c1_0 <= 0:
        return np.nan

    return float(c1_14 / c1_0)


def cd8_infiltration_rate_0_7d(time: np.ndarray, V_T_CD8: np.ndarray) -> float:
    """
    Compute CD8 infiltration rate from day 0 to day 7.

    Measures how quickly CD8 cells infiltrate tumor.

    Args:
        time: Time vector (days)
        V_T_CD8: CD8 T cell counts in tumor (cells)

    Returns:
        Infiltration rate (cells/day)
    """
    cd8_0 = np.interp(0.0, time, V_T_CD8)
    cd8_7 = np.interp(7.0, time, V_T_CD8)

    if np.isnan(cd8_0) or np.isnan(cd8_7):
        return np.nan

    return float((cd8_7 - cd8_0) / 7.0)


def cd8_fold_change_0_14d(time: np.ndarray, V_T_CD8: np.ndarray) -> float:
    """
    Compute CD8 fold-change from day 0 to day 14.

    Measures CD8 infiltration magnitude.

    Args:
        time: Time vector (days)
        V_T_CD8: CD8 T cell counts in tumor (cells)

    Returns:
        Fold change in CD8 (dimensionless)
    """
    cd8_0 = np.interp(0.0, time, V_T_CD8)
    cd8_14 = np.interp(14.0, time, V_T_CD8)

    if np.isnan(cd8_0) or np.isnan(cd8_14) or cd8_0 <= 0:
        return np.nan

    return float(cd8_14 / cd8_0)


def mdsc_cd8_ratio_7d(time: np.ndarray, V_T_MDSC: np.ndarray, V_T_CD8: np.ndarray) -> float:
    """
    Compute MDSC:CD8 ratio at day 7.

    Measures immunosuppressive environment dynamics.

    Args:
        time: Time vector (days)
        V_T_MDSC: MDSC counts in tumor (cells)
        V_T_CD8: CD8 T cell counts in tumor (cells)

    Returns:
        MDSC/CD8 ratio at day 7 (dimensionless)
    """
    t_assess = 7.0

    mdsc_7 = np.interp(t_assess, time, V_T_MDSC)
    cd8_7 = np.interp(t_assess, time, V_T_CD8)

    eps_den = 1e-12
    return float(mdsc_7 / max(cd8_7, eps_den))


def exhausted_cd8_fraction_7d(time: np.ndarray, V_T_CD8_exh: np.ndarray, V_T_CD8: np.ndarray) -> float:
    """
    Compute fraction of exhausted CD8 T cells at day 7.

    Measures CD8 exhaustion dynamics.

    Args:
        time: Time vector (days)
        V_T_CD8_exh: Exhausted CD8 T cell counts (cells)
        V_T_CD8: Active CD8 T cell counts (cells)

    Returns:
        Fraction of exhausted CD8 at day 7 (dimensionless, [0,1])
    """
    t_assess = 7.0

    cd8_exh_7 = np.interp(t_assess, time, V_T_CD8_exh)
    cd8_7 = np.interp(t_assess, time, V_T_CD8)

    denom = cd8_exh_7 + cd8_7

    if np.isnan(denom) or denom <= 0:
        return np.nan

    frac = cd8_exh_7 / denom
    frac = max(0.0, min(1.0, frac))  # Clamp to [0,1]

    return float(frac)


# Registry mapping test_statistic_id -> function
# This registry is used by the derivation worker to look up functions
TEST_STAT_REGISTRY: Dict[str, Callable] = {
    # Baseline observables (t=0 after equilibration)
    'tumor_log_growth_rate_0_60d': tumor_growth_rate_0_60d,
    'cd8_treg_ratio_baseline': cd8_treg_ratio_baseline,
    'm1_m2_ratio_baseline': m1_m2_ratio_baseline,
    'mdsc_cd8_ratio_baseline': mdsc_cd8_ratio_baseline,
    'apc_maturation_ratio_baseline': apc_maturation_ratio_baseline,
    'exhausted_cd8_fraction_baseline': exhausted_cd8_fraction_baseline,
    'til_per_cancercell_baseline': til_per_cancercell_baseline,
    'tgfb_concentration_baseline': tgfb_concentration_baseline,
    'ccl2_concentration_baseline': ccl2_concentration_baseline,

    # Temporal observables (t=7d, t=14d after equilibration)
    'cd8_treg_ratio_7d': cd8_treg_ratio_7d,
    'cd8_treg_ratio_14d': cd8_treg_ratio_14d,
    'tumor_fold_change_0_14d': tumor_fold_change_0_14d,
    'cd8_infiltration_rate_0_7d': cd8_infiltration_rate_0_7d,
    'cd8_fold_change_0_14d': cd8_fold_change_0_14d,
    'mdsc_cd8_ratio_7d': mdsc_cd8_ratio_7d,
    'exhausted_cd8_fraction_7d': exhausted_cd8_fraction_7d,
}


def get_test_stat_function(test_statistic_id: str) -> Callable:
    """
    Get test statistic function by ID.

    Args:
        test_statistic_id: Test statistic identifier (e.g., 'tumor_log_growth_rate_0_60d')

    Returns:
        Test statistic function

    Raises:
        KeyError: If test_statistic_id not found in registry
    """
    if test_statistic_id not in TEST_STAT_REGISTRY:
        raise KeyError(
            f"Test statistic '{test_statistic_id}' not found in registry. "
            f"Available: {', '.join(TEST_STAT_REGISTRY.keys())}"
        )

    return TEST_STAT_REGISTRY[test_statistic_id]

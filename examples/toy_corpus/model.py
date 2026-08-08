"""A small tumour-immune ODE, standing in for the QSP simulator.

Twenty parameters, six species, two arms. Small enough to solve a whole emulator
pool on a laptop and nonlinear enough that a surrogate has work to do.

The init hook mirrors the real one: a patient whose tumour does not at least
double over the window, or which saturates its own carrying capacity, is refused
rather than returned, so the status head has two classes to learn and the pool
has the attrition the fit's E_B is measured through.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

#: Solver status codes, matching ``qsp_hpc.cpp.batch_runner``: 0 solved, 1 the
#: solver failed, 4/5 the init hook refused theta.
STATUS_OK = 0
STATUS_FAILED = 1
STATUS_REJECTED_FAST = 4
STATUS_REJECTED_SLOW = 5

#: Species, in the order ``observables`` unpacks them. Sorted, because the
#: generated module's ``STATES`` is and the emulator's columns follow it.
SPECIES = ("C", "E", "M", "R", "T", "X")

#: The arms, and what each does to the mechanism.
ARMS = ("baseline", "vaccine")

#: Where species are recorded. Every arm emits both; a scenario reads one.
READOUT_TIMES = (14.0, 28.0)

T_END = 28.0
T0 = 1.0          # tumour at t = 0, in the model's own units
GROWTH_FLOOR = 2.0    # the window must at least double it
SATURATION = 0.95     # ... and must not park it at the carrying capacity

#: name, median, prior log-sd, omega role, margin, what it is.
#:
#: The log-sd is stage-1 epistemic uncertainty about the population centre. The
#: role sets the between-patient width and is a separate claim; the two are not
#: derived from each other anywhere in this corpus, which is the point.
PARAMETERS = (
    ("r_T",     0.08,  0.30, "default",           "log", "tumour intrinsic growth rate, 1/day"),
    ("K_T",   100.0,   0.50, "patient_trait",     "log", "tumour carrying capacity"),
    ("k_kill",  0.50,  0.40, "lumped",            "log", "CD8 killing rate constant"),
    ("h_kill",  5.00,  0.50, "lumped",            "log", "tumour half-saturation of killing"),
    ("i_R",     2.00,  0.50, "lumped",            "log", "Treg suppression constant"),
    ("i_M",     5.00,  0.50, "lumped",            "log", "macrophage suppression constant"),
    ("s_E",     0.05,  0.40, "default",           "log", "constitutive effector influx"),
    ("k_rec_E", 0.40,  0.35, "patient_trait",     "log", "cytokine-driven effector recruitment"),
    ("h_rec",  10.0,   0.50, "lumped",            "log", "tumour half-saturation of recruitment"),
    ("d_E",     0.15,  0.30, "default",           "log", "effector death rate, 1/day"),
    ("k_exh",   0.05,  0.40, "default",           "log", "effector exhaustion rate"),
    ("s_R",     0.02,  0.40, "default",           "log", "constitutive Treg influx"),
    ("k_rec_R", 0.10,  0.40, "patient_trait",     "log", "cytokine-driven Treg recruitment"),
    ("d_R",     0.10,  0.30, "default",           "log", "Treg death rate, 1/day"),
    ("s_M",     0.05,  0.40, "default",           "log", "constitutive macrophage influx"),
    ("k_rec_M", 0.50,  0.35, "patient_trait",     "log", "tumour-driven macrophage recruitment"),
    ("d_M",     0.08,  0.30, "default",           "log", "macrophage death rate, 1/day"),
    ("k_sec_C", 0.20,  0.40, "default",           "log", "cytokine secretion per cell"),
    ("d_C",     1.00,  0.25, "material_constant", "log", "cytokine clearance rate, 1/day"),
    ("f_vac",   3.00,  0.50, "patient_trait",     "log", "vaccine fold-increase on recruitment"),
)

PARAM_NAMES = tuple(p[0] for p in PARAMETERS)
AT = {n: i for i, n in enumerate(PARAM_NAMES)}


def _rhs(t, y, p, vac):
    C, E, M, R, T, X = y
    supp = 1.0 / (1.0 + R / p[AT["i_R"]] + M / p[AT["i_M"]])
    occ_kill = T / (T + p[AT["h_kill"]])
    occ_rec = T / (T + p[AT["h_rec"]])
    kill = p[AT["k_kill"]] * E * occ_kill * supp
    exh = p[AT["k_exh"]] * E * occ_kill
    rec_E = p[AT["k_rec_E"]] * vac * C * occ_rec * supp
    return np.array([
        p[AT["k_sec_C"]] * (T + M) - p[AT["d_C"]] * C,
        p[AT["s_E"]] + rec_E - p[AT["d_E"]] * E - exh,
        p[AT["s_M"]] + p[AT["k_rec_M"]] * occ_rec - p[AT["d_M"]] * M,
        p[AT["s_R"]] + p[AT["k_rec_R"]] * C - p[AT["d_R"]] * R,
        p[AT["r_T"]] * T * (1.0 - T / p[AT["K_T"]]) - kill,
        exh - p[AT["d_E"]] * X,
    ])


def _y0() -> np.ndarray:
    y = np.zeros(len(SPECIES))
    y[SPECIES.index("T")] = T0
    return y


def solve_one(theta: np.ndarray, arm: str, times: Sequence[float] = READOUT_TIMES):
    """One patient in one arm: ``(species at each time, status)``.

    Species are returned even for a refused patient, so a caller can look at what
    the hook threw away; ``status`` is what says whether they count.
    """
    from scipy.integrate import solve_ivp

    vac = theta[AT["f_vac"]] if arm == "vaccine" else 1.0
    sol = solve_ivp(_rhs, (0.0, float(max(times))), _y0(), args=(theta, vac),
                    method="LSODA", t_eval=list(times), rtol=1e-7, atol=1e-10)
    if not sol.success or not np.isfinite(sol.y).all() or (sol.y < -1e-6).any():
        return np.full((len(times), len(SPECIES)), np.nan), STATUS_FAILED

    y = np.clip(sol.y.T, 0.0, None)
    final_T = y[-1, SPECIES.index("T")]
    if final_T >= SATURATION * theta[AT["K_T"]]:
        return y, STATUS_REJECTED_FAST
    if final_T < GROWTH_FLOOR * T0:
        return y, STATUS_REJECTED_SLOW
    return y, STATUS_OK


def _one(args):
    theta, arm, times = args
    return solve_one(np.asarray(theta, dtype=float), arm, times)


def solve_many(theta: np.ndarray, arm: str, times: Sequence[float] = READOUT_TIMES,
               jobs: int = 1):
    """``(N, len(times), len(SPECIES))`` species and ``(N,)`` status."""
    rows = [(theta[i], arm, tuple(times)) for i in range(len(theta))]
    if jobs > 1:
        import multiprocessing as mp

        with mp.Pool(jobs) as pool:
            out = pool.map(_one, rows, chunksize=256)
    else:
        out = [_one(r) for r in rows]
    return (np.stack([o[0] for o in out]),
            np.array([o[1] for o in out], dtype=np.int64))

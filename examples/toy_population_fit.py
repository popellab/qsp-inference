"""Toy end-to-end validation of the population model in ``docs/model-draft.tex``.

Everything here is synthetic and small enough to run on a laptop, so the
statistical machinery can be checked against a known ``phi*`` before it meets the
real simulator. The parts below match the parts of the note:

    PART 1   the inputs        -- the toy ODE, the emulator, the readouts, the
                                  cohorts, the population inputs (R, omega_0, z),
                                  the priors, the ground truth, the toy data,
                                  and V_c
    PART 2   the model         -- eq:pop through eq:post as one NumPyro model
    PART 3   the fit           -- NUTS, then recovery against phi*

There really is an emulator
---------------------------
The fit reads a trained neural surrogate, not the ODE. That is the arrangement
the note assumes, and it makes two things real rather than notional:

  * ``E_c`` is measured by eq:Ec, from held-out clouds evaluated both ways, and
    it is not zero.
  * The data are generated with the *true* ODE while the fit reads the
    *emulator*, so emulator error is a live misspecification that ``E_c`` has to
    account for. If ``E_c`` is dropped the fit should get overconfident, which is
    a switch this script exposes (``--no-emulator-error``).

The emulator learns ``g``, meaning log species, and not the readouts. eq:disc
puts ``beta`` between ``g`` and ``h_r``, so a surrogate trained on readouts could
not express a species-level bias at all.

What is toy and what is real
----------------------------
Toy: the ODE (3 species), the readout compositions, the cohort table, and the
emulator architecture. Real: the population layer, the discrepancy terms, the
smooth weighted quantiles, the frozen common random numbers, the cohort-block
bootstrap for ``V_c``, eq:Ec, and the priors. The second list is what has to be
right; the first only has to be cheap.

The data are generated the honest way. Rather than adding ``N(0, V_c)`` noise to
the true statistics, PART 1 draws ``n_c`` patients from the true cloud and
computes the statistics from those ``n_c`` with the ordinary hard quantile a
paper would have used. So the run tests whether the frozen plug-in ``V_c`` of
eq:V is a good enough description of real sampling noise, not merely whether NUTS
can invert a Gaussian.

Run::

    python examples/toy_population_fit.py
    python examples/toy_population_fit.py --quick               # smoke test
    python examples/toy_population_fit.py --no-emulator-error   # drop E_c
"""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import jax

# x64 has to be set before jax.numpy is imported, so the rest of the imports
# follow rather than lead. Quantile differences on the log scale lose too much
# in float32 for the width rows to be trustworthy.
jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from jax import random, vmap  # noqa: E402
from jax.tree_util import tree_map  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_median  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402
from scipy.stats import norm, qmc  # noqa: E402

# XLA host devices. These are virtual -- they all share the physical CPU -- but
# they are the ONLY axis JAX parallelises well on CPU, so they do double duty:
# numpyro puts one chain on each, and the true-ODE solves are pmapped across them
# (see make_true_cloud). A vmapped diffrax solve is not automatically threaded by
# the XLA CPU backend; measured on a 24-core node, the emulator build ran at about
# 4 cores of 24 until the solves were pmapped. Set TOY_HOST_DEVICES to the core
# count on a batch node.
_N_HOST_DEVICES = int(os.environ.get("TOY_HOST_DEVICES", "4"))
numpyro.set_host_device_count(_N_HOST_DEVICES)


def make_true_cloud(g_one):
    """``(vartheta_cloud, scenario) -> species``, spread across host devices.

    Only ever wraps the TRUE ODE. The emulator's cloud must not be pmapped: it is
    called inside the NUTS gradient, where the batch is one cloud and the pmap
    overhead and the nested tracing would both cost more than they save.

    Falls back to a plain vmap when there is one device or the batch is too small
    to divide usefully, so the local single-device path is unchanged.
    """
    vcloud = vmap(g_one, in_axes=(0, None))
    compiled = {}

    def cloud(vartheta, scenario):
        nd = jax.local_device_count()
        n = vartheta.shape[0]
        if nd <= 1 or n < 2 * nd:
            return vcloud(vartheta, scenario)
        if scenario not in compiled:
            compiled[scenario] = jax.pmap(lambda v: vcloud(v, scenario))
        per = -(-n // nd)  # ceil, so the reshape is exact
        pad = per * nd - n
        v = vartheta if not pad else jnp.concatenate(
            [vartheta, jnp.repeat(vartheta[:1], pad, axis=0)]
        )
        out = compiled[scenario](v.reshape(nd, per, v.shape[-1]))
        return out.reshape(nd * per, -1)[:n]

    return cloud

_T_START = time.time()


def stamp(msg):
    """A timestamped, flushed phase marker.

    Interactively the progress bar tells you the run is alive. Under a batch
    scheduler stdout is a file, the bar goes to stderr, and without these the log
    is silent for the twenty minutes of setup before NUTS starts -- which is
    indistinguishable from a hung job. Elapsed minutes are included so each phase
    can be costed from the log alone.
    """
    el = (time.time() - _T_START) / 60.0
    print(f"[{time.strftime('%H:%M:%S')} +{el:6.1f}m] {msg}", flush=True)

# =============================================================================
# PART 1 -- THE INPUTS
# =============================================================================

# -----------------------------------------------------------------------------
# 1a. The problem interface. Names follow the note's index table.
# -----------------------------------------------------------------------------
#
# Everything the size of the problem touches lives in a Problem and is installed
# into the module namespace once, before anything runs. The statistics, the model,
# eq:V, eq:Ec and the projection tests are then written once against the installed
# names and do not know which problem they are running.
#
# Two problems are supplied:
#
#   small   P=8, Q=3, M=7, S=2, A=42.   The debugging vehicle. Runs in about
#           13 minutes and the row space is not saturated, so recovery against
#           phi* is a meaningful test and the diagnostics have something to say.
#
#   full    P=271, Q=179, M=30, S=5.    The size of the real PDAC model, taken
#           from cpp/qsp/ode/QSP_enum.h and parameters/pdac_priors.csv. Here the
#           row space IS saturated and recovery against phi* is NOT the question;
#           what the run measures is what survives when 550 parameters meet about
#           100 rows, and whether the emulator holds up in 271 input dimensions.
#
#   medium  A reduced version of full, same code path, for iterating.

QUANTILE_BANDWIDTH = 0.025  # kernel width in cumulative-probability space
ELIGIBILITY_BANDWIDTH = 0.15  # logistic width for the smooth eligibility weight


@dataclass(frozen=True)
class Problem:
    """Every input whose shape depends on the size of the model.

    ``g_one`` and ``h_all`` are the mechanism and the readouts. ``h_all`` takes
    ALL scenarios at once, shape ``(S, N, Q) -> (S, N, M)``, because a fold-change
    readout is a contrast between two scenarios in the same patient and the common
    random numbers are what make that meaningful. A per-scenario signature could
    not express it, and the real target table has three such observables.
    """

    name: str
    P: int
    Q: int
    S: int
    param_names: list
    species_names: list
    readout_names: list
    g_one: object  # (vartheta, scenario) -> species, shape (Q,)
    h_all: object  # (S, N, Q) -> (S, N, M)
    Z: jnp.ndarray
    z_col_names: list
    beta_species: jnp.ndarray
    cohorts: tuple
    R_corr: jnp.ndarray
    omega_0: jnp.ndarray
    measured_idx: jnp.ndarray
    n_j: jnp.ndarray
    mu_true: jnp.ndarray
    mu_0: jnp.ndarray
    sigma_1: jnp.ndarray
    truth: dict  # s, u_raw, log_omega_meas offset, a, b, beta
    # Run sizes. These are part of the problem because they scale with it.
    n_cloud: int
    n_cloud_flat: int
    n_truth: int
    b_boot: int
    l_emu: int
    emu_pool: int
    emu_hidden: tuple
    emu_steps: int


def install_problem(prob):
    """Bind a Problem into the module namespace, with everything derived from it.

    A single mutable configuration installed once before any work happens, in the
    same spirit as the ``jax_enable_x64`` call at the top of this file. The
    alternative -- threading a problem object through sixty call sites -- buys
    nothing here, because there is never more than one problem live in a process.
    """
    g = globals()
    g["PROB"] = prob
    g["P"], g["Q"], g["S"] = prob.P, prob.Q, prob.S
    g["PARAM_NAMES"] = prob.param_names
    g["SPECIES_NAMES"] = prob.species_names
    g["READOUT_NAMES"] = prob.readout_names
    g["M"] = len(prob.readout_names)
    g["g_true_one"] = prob.g_one
    g["g_true_cloud"] = make_true_cloud(prob.g_one)
    g["h_all"] = prob.h_all
    g["Z"] = prob.Z
    g["Z_COL_NAMES"] = prob.z_col_names
    g["DIM_Z"] = int(prob.Z.shape[1])
    g["BETA_SPECIES"] = prob.beta_species
    g["N_BETA"] = int(prob.beta_species.shape[0])

    g["COHORTS"] = prob.cohorts
    g["C"] = len(prob.cohorts)
    g["A_ROWS"] = sum(c.K for c in prob.cohorts)
    g["ROW_LABELS"] = [lab for c in prob.cohorts for lab in c.row_labels]
    g["LOC_IDX"] = [jnp.array(np.flatnonzero(c.location_mask)) for c in prob.cohorts]
    g["A_ROWS_LOC"] = sum(int(c.location_mask.sum()) for c in prob.cohorts)

    g["R_CORR"] = prob.R_corr
    g["L_R"] = jnp.linalg.cholesky(prob.R_corr)
    g["OMEGA_0"] = prob.omega_0
    g["MEASURED_IDX"] = prob.measured_idx
    meas = set(int(j) for j in np.asarray(prob.measured_idx))
    g["ASSUMED_IDX"] = jnp.array([j for j in range(prob.P) if j not in meas],
                                 dtype=jnp.int32)
    g["N_MEASURED"] = int(prob.measured_idx.shape[0])
    g["N_ASSUMED"] = prob.P - g["N_MEASURED"]
    g["N_J"] = prob.n_j
    g["TAU_OMEGA_MEASURED"] = (
        1.0 / jnp.sqrt(2.0 * (prob.n_j - 1.0) + TAU_CLASS**-2)
        if g["N_MEASURED"] else jnp.zeros(0)
    )

    g["MU_TRUE"], g["MU_0"], g["SIGMA_1"] = prob.mu_true, prob.mu_0, prob.sigma_1
    g["L_SIGMA_1"] = jnp.linalg.cholesky(prob.sigma_1)

    g["N_CLOUD"], g["N_CLOUD_FLAT"] = prob.n_cloud, prob.n_cloud_flat
    g["N_TRUTH"], g["B_BOOT"], g["L_EMU"] = prob.n_truth, prob.b_boot, prob.l_emu
    g["EMU_POOL"], g["EMU_HIDDEN"], g["EMU_STEPS"] = (
        prob.emu_pool, prob.emu_hidden, prob.emu_steps
    )

    # Fold-change readouts are identically zero in the baseline scenario, so a
    # cohort asking for one there would report log IQR of zero. Catch it here
    # rather than as a clipped row a hundred lines into the fit.
    fc = {r for r, nm in enumerate(prob.readout_names) if nm.endswith("_fc")}
    for c in prob.cohorts:
        if c.scenario == 0 and fc.intersection(c.readouts):
            raise ValueError(
                f"cohort {c.name} asks for a fold-change readout in the baseline "
                f"scenario, where it is identically zero"
            )

# -----------------------------------------------------------------------------
# 1b. The toy mechanism, g. Three species, two scenarios.
# -----------------------------------------------------------------------------
#
# A tumour-effector-stroma system. Three features keep it well behaved across the
# whole population, which matters because a solver excursion anywhere in the
# cloud would poison a quantile:
#
#   EPS_T   a small tumour source, so the tumour has a positive floor instead of
#           going extinct and pinning log(x_T) at a clip
#   K_E     a cap on effector expansion, so the killing term cannot run away
#   clamp   the state is clamped inside the right-hand side, which bounds every
#           derivative and makes the fixed-step solve unconditionally stable
#
# Fixed-step Tsit5, so reverse-mode AD is cheap and the step count does not
# depend on theta. A theta-dependent step count would make tau(phi) jump.

T_OBS = 30.0  # readout time
# Fixed solver steps. This is the single most important number in the toy. At
# dt = 0.5 the solver overshoots into negative tumour in the stiff killing
# regime, the clip turns that into a hard floor at log(1e-9), and the emulator
# is then asked to fit a discontinuity: held-out RMSE on log tumour goes from
# 0.10 at 200 steps to 1.71 at 60. The ODE is off the NUTS path -- only the
# design, the data and eq:Ec touch it -- so a fine solve is nearly free.
N_STEPS = 200
GK = 0.5  # Michaelis constant, killing
HE = 1.0  # Michaelis constant, effector recruitment
K_E = 2.0  # effector carrying capacity
EPS_T = 0.02  # tumour source term
Y_MAX = 50.0  # state clamp inside the RHS
Y0 = jnp.array([0.20, 0.10, 0.50])  # initial species

# Scenario s scales the killing rate: 0 = untreated, 1 = treated.
SCENARIO_KILL_MULTIPLIER = jnp.array([1.0, 1.8])


def _rhs(t, y, args):
    """Right-hand side of the toy ODE. ``args`` is (theta, kill_multiplier)."""
    theta, kill_mult = args
    r_T, K_T, k_kill, s_E, p_E, d_E, r_S, K_S = theta
    yc = jnp.clip(y, 1e-9, Y_MAX)
    x_T, x_E, x_S = yc[0], yc[1], yc[2]

    growth = r_T * x_T * (1.0 - x_T / K_T)
    killing = kill_mult * k_kill * x_E * x_T / (GK + x_T)
    recruit = p_E * x_T * x_E / (HE + x_T) * (1.0 - x_E / K_E)

    return jnp.stack(
        [
            EPS_T + growth - killing,
            s_E + recruit - d_E * x_E,
            r_S * x_S * (1.0 - x_S / K_S),
        ]
    )


def _small_g_one(vartheta, scenario):
    """eq:mech -- one patient, one scenario, species at the measured time.

    Args:
        vartheta: log parameters, shape ``(P,)``.
        scenario: integer scenario index.

    Returns:
        Species vector at ``T_OBS``, shape ``(Q,)``, strictly positive.
    """
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(_rhs),
        diffrax.Tsit5(),
        t0=0.0,
        t1=T_OBS,
        dt0=T_OBS / N_STEPS,
        y0=Y0,
        args=(jnp.exp(vartheta), SCENARIO_KILL_MULTIPLIER[scenario]),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=N_STEPS + 8,
        adjoint=diffrax.DirectAdjoint(),
    )
    return jnp.clip(sol.ys[-1], 1e-9, None)


# -----------------------------------------------------------------------------
# 1c. The emulator. An MLP standing in for g.
# -----------------------------------------------------------------------------
#
# Hand-rolled so the surrogate is fully visible and the script needs no training
# framework. It maps (standardised vartheta, one-hot scenario) to LOG species.
# Logs for two reasons: positivity is free, and the readouts are log-scale
# compositions, so the emulator is learning in the units the fit reads.

# The design size, the net and the step count are per problem; the rest is not.
EMU_BATCH = 512
EMU_LR = 3e-3
EMU_POOL_INFLATE = 1.20  # cover the flanks, not only the centre
EMU_HOLDOUT = 4_000  # iid draws from the pool law, for an honest score


def init_mlp(key, sizes):
    """Xavier-ish init for a tanh MLP. Params are a list of (W, b)."""
    params = []
    for k, (n_in, n_out) in zip(random.split(key, len(sizes) - 1), zip(sizes, sizes[1:])):
        W = random.normal(k, (n_in, n_out)) * np.sqrt(1.0 / n_in)
        params.append((W, jnp.zeros(n_out)))
    return params


def mlp_apply(params, x):
    """Forward pass. tanh hidden layers, linear output."""
    for W, b in params[:-1]:
        x = jnp.tanh(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def adam_step(params, grads, state, lr):
    """One Adam update. Returns (params, state)."""
    m, v, t = state
    t = t + 1
    m = tree_map(lambda a, g: 0.9 * a + 0.1 * g, m, grads)
    v = tree_map(lambda a, g: 0.999 * a + 0.001 * g**2, v, grads)
    mh = tree_map(lambda a: a / (1 - 0.9**t), m)
    vh = tree_map(lambda a: a / (1 - 0.999**t), v)
    params = tree_map(lambda p, a, b: p - lr * a / (jnp.sqrt(b) + 1e-8), params, mh, vh)
    return params, (m, v, t)


@dataclass
class Emulator:
    """A trained surrogate for ``g``, plus the standardisation it was fit under."""

    params: list
    x_mean: jnp.ndarray
    x_sd: jnp.ndarray
    y_mean: jnp.ndarray
    y_sd: jnp.ndarray
    holdout_rmse: np.ndarray  # per species, on log scale

    def one(self, vartheta, scenario):
        onehot = jnp.zeros(S).at[scenario].set(1.0)
        z = jnp.concatenate([(vartheta - self.x_mean) / self.x_sd, onehot])
        log_y = mlp_apply(self.params, z) * self.y_sd + self.y_mean
        return jnp.exp(jnp.clip(log_y, -25.0, 25.0))

    def cloud(self, vartheta, scenario):
        onehot = jnp.zeros(S).at[scenario].set(1.0)
        zt = (vartheta - self.x_mean[None, :]) / self.x_sd[None, :]
        z = jnp.concatenate([zt, jnp.broadcast_to(onehot, (zt.shape[0], S))], axis=1)
        log_y = mlp_apply(self.params, z) * self.y_sd[None, :] + self.y_mean[None, :]
        return jnp.exp(jnp.clip(log_y, -25.0, 25.0))


def sobol_normal(n, seed):
    """``n`` standard-normal design points via a scrambled Sobol sequence.

    Space filling in the standardised coordinates, which is where the emulator
    needs coverage. On this problem the design was worth about 3% of held-out
    RMSE against iid Gaussian draws, so it is a free improvement rather than the
    lever; the solver step count and the pool width were the levers.
    """
    u = qmc.Sobol(d=P, scramble=True, seed=seed).random(n)
    return norm.ppf(np.clip(u, 1e-6, 1.0 - 1e-6))


def _cache_key(*parts):
    """A content hash over the inputs that determine a cached artefact.

    Arrays are hashed by their exact bytes, not by shape or a summary, so a cache
    hit means the inputs are identical rather than merely similar. Getting this
    wrong is worse than having no cache: a stale emulator would be silently wrong
    everywhere downstream and nothing in the diagnostics would point at it.
    """
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, (jnp.ndarray, np.ndarray)):
            a = np.ascontiguousarray(np.asarray(p, dtype=np.float64))
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:16]


def _cache_path(cache_dir, kind, key):
    return None if cache_dir is None else Path(cache_dir) / f"{kind}-{key}.npz"


def species_loss_weights(key, mu_0, omega_0, n=64, eps=1e-3, floor=0.05,
                         verbose=True):
    """How much the readouts depend on each species, for weighting the emulator.

    A uniform loss over log species spends the surrogate's capacity in proportion
    to nothing in particular. At Q = 179 a large minority of species never enter a
    readout: they matter to the mechanism, because they regulate other species
    inside the ODE, but the emulator is a map from vartheta to species and does not
    propagate anything, so its error on a species the fit never reads is free.
    Capacity spent there is capacity taken from the species that carry the rows.

    The weight is measured, not declared: perturb species q by ``eps`` in log
    across a plug-in cloud and take the root-mean-square change in the readouts.
    That is exactly ``|d h / d log y_q|`` aggregated over readouts, scenarios and
    patients, so it counts a species once for every readout that uses it and
    weights by how strongly.

    A floor keeps every species at some weight. Driving one to zero would let the
    surrogate diverge arbitrarily there, and ``h_all`` involves sums over classes
    where a species that is individually unimportant can still matter to a total.

    Returns weights of mean one, shape ``(Q,)``.
    """
    z = draw_cloud_z(key, n)
    vartheta = mu_0[None, :] + z @ (L_R.T * omega_0[None, :])
    y_all = jnp.stack([g_true_cloud(vartheta, s) for s in range(S)])  # (S, n, Q)
    base = h_all(y_all)

    def sens(q):
        bumped = y_all * jnp.exp(eps * jax.nn.one_hot(q, Q))[None, None, :]
        return jnp.sqrt(jnp.mean(((h_all(bumped) - base) / eps) ** 2))

    s_q = np.asarray(jax.lax.map(sens, jnp.arange(Q)))
    s_q = np.maximum(s_q, floor * s_q.max())
    w = s_q / s_q.mean()
    if verbose:
        n_floor = int((s_q <= floor * s_q.max() * (1 + 1e-9)).sum())
        print(f"emulator loss weights: {n_floor} of {Q} species are at the floor "
              f"(no readout reads them),")
        print(f"  weight range {w.min():.3f} to {w.max():.3f}")
    return jnp.array(w)


def train_emulator(key, mu_0, sigma_1, omega_pool, hidden=None,
                   n_pool=None, n_steps=None, seed=0, verbose=True,
                   solve_chunk=2048, species_weights=None, cache_dir=None):
    """Build the design, run the true ODE on it, and fit the surrogate.

    The design covers where the *fit* will read the emulator: a cloud of width
    ``omega_pool`` centred anywhere the stage-1 prior allows. So the pool
    covariance is ``Sigma_1 + Omega``, inflated, and every parameter varies. A
    pool that pinned a parameter would leave the likelihood undefined along it
    rather than merely flat.

    The held-out set is an independent iid draw from the same pool law, not a
    split of the Sobol points. Splitting a low-discrepancy sequence leaves the
    two halves interleaved, which flatters the score.
    """
    hidden = EMU_HIDDEN if hidden is None else hidden
    n_pool = EMU_POOL if n_pool is None else n_pool
    n_steps = EMU_STEPS if n_steps is None else n_steps

    # The emulator is a pure function of its design and its seed, and at the QSP
    # size it is ~102,000 ODE solves plus 10,000 Adam steps -- ten minutes or so
    # before anything else can start. Nothing about it depends on the data, so
    # re-deriving it on every run is pure waste.
    ck = _cache_key("emu", PROB.name, P, Q, S, seed, n_pool, tuple(hidden),
                    n_steps, EMU_BATCH, EMU_LR, EMU_POOL_INFLATE, EMU_HOLDOUT,
                    N_STEPS, T_OBS, mu_0, sigma_1, omega_pool,
                    jnp.ones(Q) if species_weights is None else species_weights)
    cpath = _cache_path(cache_dir, "emulator", ck)
    if cpath is not None and cpath.exists():
        d = np.load(cpath)
        n_layers = int(d["n_layers"])
        params = [(jnp.array(d[f"W{i}"]), jnp.array(d[f"b{i}"]))
                  for i in range(n_layers)]
        if verbose:
            print(f"emulator: loaded from cache {cpath.name}")
            print(f"  held-out RMSE on log species: median "
                  f"{np.median(d['rmse']):.4f}, max {d['rmse'].max():.4f}")
        return Emulator(params, jnp.array(d["x_mean"]), jnp.array(d["x_sd"]),
                        jnp.array(d["y_mean"]), jnp.array(d["y_sd"]), d["rmse"])

    k_init, k_batch, k_hold = random.split(key, 3)

    omega_mat = jnp.diag(omega_pool) @ R_CORR @ jnp.diag(omega_pool)
    pool_cov = (sigma_1 + omega_mat) * EMU_POOL_INFLATE**2
    L_pool = jnp.linalg.cholesky(pool_cov)

    vartheta = mu_0[None, :] + jnp.array(sobol_normal(n_pool, seed)) @ L_pool.T
    vartheta_ho = mu_0[None, :] + random.normal(k_hold, (EMU_HOLDOUT, P)) @ L_pool.T

    def solve_chunked(v, s):
        """The true ODE over a design, in chunks. At Q=179 and 16k design points a
        single vmap holds 16384 x 179 states per stage and the memory is the
        binding constraint, not the arithmetic."""
        return jnp.concatenate(
            [g_true_cloud(v[i:i + solve_chunk], s)
             for i in range(0, v.shape[0], solve_chunk)]
        )

    def build(v):
        xs, ys = [], []
        for s in range(S):
            onehot = jnp.broadcast_to(jnp.zeros(S).at[s].set(1.0), (v.shape[0], S))
            xs.append(jnp.concatenate([v, onehot], axis=1))
            ys.append(jnp.log(solve_chunked(v, s)))
        return jnp.concatenate(xs), jnp.concatenate(ys)

    X, Y = build(vartheta)
    Xh, Yh = build(vartheta_ho)

    # Standardise on the parameter block only; the one-hot stays as is.
    x_mean, x_sd = vartheta.mean(0), vartheta.std(0)
    y_mean, y_sd = Y.mean(0), Y.std(0)

    def std_x(A):
        return A.at[:, :P].set((A[:, :P] - x_mean[None, :]) / x_sd[None, :])

    Xs, Ys = std_x(X), (Y - y_mean[None, :]) / y_sd[None, :]
    Xhs, Yhs = std_x(Xh), (Yh - y_mean[None, :]) / y_sd[None, :]

    sizes = (P + S, *hidden, Q)
    params = init_mlp(k_init, sizes)
    state = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params), 0)

    # Weighted on the STANDARDISED residual, which is the right place: y_sd has
    # already removed each species' own dynamic range, so the weight expresses
    # readout relevance and nothing else.
    w_species = jnp.ones(Q) if species_weights is None else species_weights

    def loss(pr, xb, yb):
        return jnp.mean(w_species[None, :] * (mlp_apply(pr, xb) - yb) ** 2)

    @jax.jit
    def step(pr, st, xb, yb, lr):
        loss_val, grads = jax.value_and_grad(loss)(pr, xb, yb)
        pr, st = adam_step(pr, grads, st, lr)
        return pr, st, loss_val

    t0 = time.time()
    n_tr = Xs.shape[0]
    for i in range(n_steps):
        k_batch, kb = random.split(k_batch)
        idx = random.randint(kb, (EMU_BATCH,), 0, n_tr)
        # Cosine decay: the tail matters, because the fit reads the flanks.
        lr = EMU_LR * 0.5 * (1 + np.cos(np.pi * i / n_steps))
        params, state, _ = step(params, state, Xs[idx], Ys[idx], lr)

    resid = np.asarray((mlp_apply(params, Xhs) - Yhs) * y_sd[None, :])
    rmse = np.sqrt(np.mean(resid**2, axis=0))
    if verbose:
        net = "x".join(str(h) for h in hidden)
        print(f"emulator: sobol design {n_pool:,} x {S} scenarios, net {net}, "
              f"{n_steps} steps, {time.time() - t0:.1f}s")
        w = np.asarray(w_species)
        if Q <= 12:
            print("  held-out RMSE on log species: "
                  + ", ".join(f"{nm} {r:.4f}" for nm, r in zip(SPECIES_NAMES, rmse)))
        else:
            # The unweighted mean is the wrong headline at this size: it is
            # dominated by species no readout reads. Report the weighted one
            # beside it, and name the worst among the species that matter.
            print(f"  held-out RMSE on log species over {Q} species: "
                  f"median {np.median(rmse):.4f}, "
                  f"90th pct {np.percentile(rmse, 90):.4f}, max {rmse.max():.4f}")
            print(f"  readout-weighted mean RMSE {np.average(rmse, weights=w):.4f} "
                  f"(unweighted {rmse.mean():.4f})")
            order = np.argsort(-(rmse * w))[:5]
            print("  worst readout-relevant species: "
                  + ", ".join(f"{SPECIES_NAMES[j]} {rmse[j]:.3f} (w {w[j]:.2f})"
                              for j in order))
    if cpath is not None:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        blob = {"n_layers": len(params), "x_mean": np.asarray(x_mean),
                "x_sd": np.asarray(x_sd), "y_mean": np.asarray(y_mean),
                "y_sd": np.asarray(y_sd), "rmse": rmse}
        for i, (W, b) in enumerate(params):
            blob[f"W{i}"] = np.asarray(W)
            blob[f"b{i}"] = np.asarray(b)
        np.savez_compressed(cpath, **blob)
        if verbose:
            print(f"  cached to {cpath}")
    return Emulator(params, x_mean, x_sd, y_mean, y_sd, rmse)


# -----------------------------------------------------------------------------
# 1d. The readouts, h_r. eq:readout.
# -----------------------------------------------------------------------------
#
# Readouts 2, 3 and 4 share the denominator (the nucleated total) and readouts 5
# and 6 share x_T, which is what makes their sampling errors correlated inside a
# cohort and therefore what exercises the off-diagonal blocks of V_c. The ratios
# inherit a difference of two species biases under beta.
#
# Readouts 2 and 3 are both the effector fraction, by flow and by histology. That
# pair is what identifies the modality column of Z from a within-quantity-type
# contrast rather than leaving modality confounded with quantity type.
#
# They must not be the *same function* of the species. Flow gates on lymphoid plus
# tumour and histology counts all nucleated cells, so the two are strongly but not
# perfectly correlated. Making them identical instead gives cohort A a singular
# V_c: two rows with the same functional of the same patients have perfectly
# correlated sampling noise, the ridge in ``assemble_V`` turns that into a
# 1e10 direction in V^-1, and NUTS stalls at a 5e-4 step size.


def _small_h_all(y):
    """All M readouts from species, on the log scale.

    Args:
        y: species, shape ``(S, N, Q)``.

    Returns:
        Readouts, shape ``(S, N, M)``. The small problem has no fold-change
        readout, so the scenario axis passes straight through; it is present
        because the QSP-scale problem does have them and the two must share a
        signature.
    """
    x_T, x_E, x_S = y[..., 0], y[..., 1], y[..., 2]
    total = x_T + x_E + x_S
    return jnp.stack(
        [
            jnp.log(x_T),                  # 0  density,  imaging
            jnp.log(x_S),                  # 1  density,  histology
            jnp.log(x_E / (x_E + x_T)),    # 2  fraction, flow
            jnp.log(x_E / total),          # 3  fraction, histology
            jnp.log(x_T / total),          # 4  fraction, histology
            jnp.log(x_E / x_T),            # 5  ratio,    flow
            jnp.log(x_S / x_T),            # 6  ratio,    histology
        ],
        axis=-1,
    )


# Species allowed a mechanism discrepancy (the note's set S). All Q of them, but
# constrained to sum to zero, and the constraint is not a convenience.
#
# A common bias beta_q = c on every species is reproduced *exactly* by the
# measurement map, at a = (c, -c, -c, 0) for the Z below, because e^c cancels from
# every fraction and every ratio and shifts only the two densities. That is an
# exactly flat direction of the posterior at every phi, not a first-order alias,
# so with a free beta in R^Q the prior alone would decide it. Centring beta
# removes it and leaves the Q-1 directions the readouts can actually separate.
#
# This is also why a horseshoe over a free beta in R^Q is the wrong instrument. A
# sparsity prior chooses among identified-but-weak coefficients; on an exactly
# aliased direction its heavy tail simply wins against the Gaussian on a and
# claims the shared component. Constrain first, then shrink.
_SMALL_BETA_SPECIES = jnp.array([0, 1, 2])

# Z_r: readout attributes with no mechanistic twin, as two crossed factors.
# Columns are [intercept, is_fraction, is_ratio, is_flow]: quantity type against
# the density reference level, then assay modality.
#
# Readouts 0 and 1 share a row, and so do 3 and 4, so their gamma and kappa are
# pooled rather than free. rank(Z) = 4 < M = 7, which is the property that keeps
# gamma from saturating readout space; a Z of rank M would make gamma equivalent
# to a free per-readout intercept and absorb beta's location signal entirely.
#
# The intercept was dropped once, on the argument that a common shift on every
# readout is what mu can reproduce and that it scored the lowest residual of any
# column on the Z test. That bought cond(Z) 4.66 -> 2.25, freed the common beta
# shift from the measurement map, and removed simplification 4 outright, but the
# sampler ran three times slower: pinning gamma = 0 and kappa = 1 on the two
# density readouts does not remove their misfit, it transfers it into omega and s
# where the curvature is worse. Reverted. The row budget printed by
# ``print_projection_report`` is the reason to stop tuning Z at all: at 22
# parameters against 19 rows the row space is saturated whatever coding is used.
_SMALL_Z = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],  # 0  log tumour       density   imaging
        [1.0, 0.0, 0.0, 0.0],  # 1  log stroma       density   histology
        [1.0, 1.0, 0.0, 1.0],  # 2  effector frac    fraction  flow
        [1.0, 1.0, 0.0, 0.0],  # 3  effector frac    fraction  histology
        [1.0, 1.0, 0.0, 0.0],  # 4  tumour frac      fraction  histology
        [1.0, 0.0, 1.0, 1.0],  # 5  E:T ratio        ratio     flow
        [1.0, 0.0, 1.0, 0.0],  # 6  S:T ratio        ratio     histology
    ]
)
_SMALL_Z_COLS = ["intercept", "is_fraction", "is_ratio", "is_flow"]


def z_conditioning():
    """How well the readouts pin down the columns of Z. No fit required.

    A column carried by only one readout gives a coefficient that is nearly free
    against the intercept: the two can trade off along that single row and only
    the priors separate them. Every column here has at least two members, which
    is what the crossed modality factor buys over a single categorical partition.

    Reports the rank against M as well. At rank(Z) = M the measurement map spans
    readout space and beta keeps no location signal of its own.

    Worth computing for the real Z before any fit. It costs nothing and it says
    which measurement-map coefficients can be estimated at all. It is the cheap
    half of the story; ``projection_report`` is the half that uses the data.
    """
    Zn = np.asarray(Z)
    support = (np.abs(Zn) > 0).sum(axis=0)
    sv = np.linalg.svd(Zn, compute_uv=False)
    return support, float(sv[0] / sv[-1]), int(np.linalg.matrix_rank(Zn))


# -----------------------------------------------------------------------------
# 1e. The cohort table.
# -----------------------------------------------------------------------------


# The statistics a cohort can report for a readout. The note's K_rc is the number
# of these it lists. "mean" beside "med" is the pair sec:flat singles out: the gap
# between them is the skewness of the pushforward, which at omega = omega_0 the
# model predicts rather than absorbs, and which a point-mass cloud could not
# predict at all. Both are location rows, so they are also the only way to give the
# flat fit more rows without inventing more cohorts.
LOCATION_STATS = ("med", "mean")
SCALE_STATS = ("iqr",)


@dataclass(frozen=True)
class Cohort:
    """One study cohort. ``readouts`` is the note's set R_c."""

    name: str
    scenario: int
    n: int  # n_c, the real cohort size
    readouts: tuple[int, ...]
    # Per readout, the statistics reported for it. This is the note's K_rc.
    stats: tuple[tuple[str, ...], ...]
    # Eligibility on a readout, as (readout index, low, high). None means w = 1.
    eligibility: tuple[int, float, float] | None = None
    # Whether this cohort's statistics were reconstructed rather than reported.
    # 28 of the 49 rows of vpop_marginal_targets.csv are kind = synthesized, so
    # more than half the real row budget is of this sort. It is a property of the
    # TARGET, not of the readout, so it cannot enter Z -- eq:disc indexes gamma and
    # kappa by readout only. Its home is the reported uncertainty U of eq:Vsplit,
    # which inflates the scales of those rows while the correlations stay from the
    # bootstrap. See ``build_V``.
    synthesized: bool = False

    @property
    def K(self) -> int:
        """K_c, the number of rows this cohort contributes."""
        return sum(len(s) for s in self.stats)

    @property
    def row_labels(self) -> list[str]:
        return [
            f"{self.name}/{READOUT_NAMES[r]}/{st}"
            for r, sts in zip(self.readouts, self.stats)
            for st in sts
        ]

    @property
    def location_mask(self) -> np.ndarray:
        """True on the location rows, False on the scale rows. eq:phiflat's holdout.

        The flat fit of sec:flat fits the location rows and holds the scale rows
        out, so that predicting the held-out rows at omega_0 is an honest statement
        about the width shortfall rather than a residual from a fit that saw them.
        """
        return np.array(
            [st in LOCATION_STATS for sts in self.stats for st in sts]
        )

    @property
    def scale_readouts(self) -> list[tuple[int, int]]:
        """``(row index, readout)`` for each scale row this cohort contributes."""
        out, k = [], 0
        for r, sts in zip(self.readouts, self.stats):
            for st in sts:
                if st in SCALE_STATS:
                    out.append((k, r))
                k += 1
        return out


_MI = ("med", "iqr")
_MMI = ("med", "mean", "iqr")

_SMALL_COHORTS: tuple[Cohort, ...] = (
    # The modality cohort: the effector fraction by flow and by histology, in the
    # same patients, beside the tumour density. The flow-against-histology
    # contrast within one quantity type is what identifies the is_flow column, and
    # having it inside one cohort puts the contrast's sampling error in the
    # off-diagonal block where it belongs rather than across independent blocks.
    Cohort("A", scenario=0, n=20, readouts=(0, 2, 3), stats=(_MI, _MI, _MI)),
    # A small cohort reporting a centre only.
    Cohort("B", scenario=0, n=6, readouts=(5,), stats=(("med",),)),
    # Readout 0 repeats here, under the other scenario, and reports a mean beside
    # its median.
    Cohort("C", scenario=1, n=50, readouts=(0, 5), stats=(_MMI, _MI)),
    # A large cohort with an eligibility criterion on tumour burden. Its two
    # readouts share the nucleated total.
    Cohort(
        "D",
        scenario=1,
        n=200,
        readouts=(3, 4),
        stats=(_MMI, _MMI),
        eligibility=(0, -3.0, 1.5),
    ),
    # The stromal arm: a density and a ratio sharing x_T, both histology. Without
    # it the is_ratio column rests on one flow readout and log_stroma is unmeasured.
    Cohort("E", scenario=0, n=35, readouts=(1, 6), stats=(_MI, _MI)),
    # From here down the cohorts exist to buy rows. At five cohorts the row space
    # was saturated: 22 parameters in (a, b, mu, omega) spanned all 19 rows, so no
    # block was identified at first order and only the priors separated them. The
    # flat fit was worse still, 20 parameters against 10 location rows. Nothing
    # about Z or the discrepancy could be tested until A exceeded that.
    Cohort("F", scenario=1, n=80, readouts=(1, 2, 6), stats=(_MI, _MI, ("med",))),
    Cohort("G", scenario=0, n=45, readouts=(4, 5), stats=(_MMI, _MI)),
    Cohort("H", scenario=1, n=25, readouts=(0, 3), stats=(_MI, _MI)),
    # Means on three readouts at n = 120, which is where the mean-median gap is
    # least noisy and so where sec:flat's skewness check has the most to say.
    Cohort("I", scenario=0, n=120, readouts=(0, 1, 4),
           stats=(("med", "mean"), _MI, _MI)),
)


# -----------------------------------------------------------------------------
# 1f. Population inputs: R, omega_0, the measured set M, and the frozen cloud.
# -----------------------------------------------------------------------------

# R, the population correlation matrix. eq:pop treats it as an input. In the small
# problem it is a mild block structure: the two tumour-growth parameters move
# together, as do the two effector parameters and the two stroma parameters.
_SMALL_R = np.eye(8)
_SMALL_R[0, 1] = _SMALL_R[1, 0] = 0.4  # r_T with K_T
_SMALL_R[3, 4] = _SMALL_R[4, 3] = 0.3  # s_E with p_E
_SMALL_R[6, 7] = _SMALL_R[7, 6] = 0.5  # r_S with K_S

# omega_0, the prior centre of the widths. Layers L1 to L3 of the note's build:
# a 0.20 default, with three parameters treated as tighter "material constants".
_SMALL_OMEGA_0 = jnp.array([0.20, 0.12, 0.20, 0.20, 0.20, 0.20, 0.12, 0.12])

# The class-prior width tau from the L4 shrinkage, which with the source sample
# sizes n_j gives the prior width of eq:omegameas. Not per problem.
TAU_CLASS = 0.5


def draw_cloud_z(key, n):
    """The latent cloud z_{1:N}. eq:crn draws it once and keeps it."""
    return random.normal(key, (n, P))


# -----------------------------------------------------------------------------
# 1g. Priors. mu_0 and Sigma_1 stand in for the stage-1 composite.
# -----------------------------------------------------------------------------

# True parameter values, on the natural scale. The stage-1 prior is centred near
# but not on these, so the fit has something to do.
_SMALL_THETA_TRUE = jnp.array([0.90, 1.60, 1.10, 0.05, 0.55, 0.30, 0.35, 0.80])

# mu_0: stage-1 centre, displaced from the truth by a fixed offset per parameter.
_SMALL_MU_OFFSET = jnp.array([0.25, -0.20, 0.30, 0.15, -0.25, 0.20, -0.15, 0.10])

# Sigma_1: stage-1 covariance on the log scale. Correlated, and wide enough that
# the offset above sits comfortably inside it. Not wider: Sigma_1 sets how far
# the emulator design has to reach, and a stage-1 prior of sd 0.45 pushed the
# pool into the regime where the tumour goes extinct and the surrogate fails.
_SMALL_SD1 = np.array([0.30, 0.20, 0.32, 0.28, 0.30, 0.28, 0.22, 0.20])

# How much wider a reconstructed statistic's uncertainty is than the bootstrap's.
# A guess, and it should be a per-source number in the real analysis; the point of
# having it here is that the synthesized rows stop being treated as though they
# were reported.
SYNTH_INFLATE = 1.6

TAU_S = 0.5  # prior width on the global width multiplier s
TAU_U = 0.5  # prior width on the width pattern u
SIGMA_A = 0.5  # prior width on a (location half of the measurement map)
SIGMA_B = 0.5  # prior width on b (scale half)

# tau_s and sigma_b agree by design: equal widths state that we have no
# preference between a narrow population and a narrow simulator before the data
# arrive. See the width-ridge note in the model.

# tau_beta is FIXED, which is a departure from eq:betaprior and needs saying.
# The note puts a hyperprior on it. On these rows that hyperprior was the largest
# single source of divergences: beta is centred, so Q-1 = 2 free directions hang
# off one scale, the funnel is deep, and log_tau_beta was consistently the
# worst-mixing coordinate in both fits (n_eff 161 in the flat fit against 24
# divergences). Two directions buy no pooling, so the hyperprior costs geometry
# and returns nothing. Fixing it at the prior centre keeps beta's role -- a
# tightly shrunk species bias that competes with mu -- without the funnel.
TAU_BETA = 0.15


# -----------------------------------------------------------------------------
# Smooth statistics. The note requires these to be differentiable in phi.
# -----------------------------------------------------------------------------


def smooth_weighted_quantiles(x, w, ps, bandwidth=QUANTILE_BANDWIDTH):
    """Weighted quantiles with a Gaussian kernel in cumulative-weight space.

    A hard quantile selects one order statistic, so its gradient is carried by a
    single patient and jumps when two patients swap rank. Weighting the order
    statistics by a kernel in their own cumulative weight removes both problems
    at the cost of an O(bandwidth^2) bias.

    Takes several probability levels at once because the sort dominates the cost
    and a median beside an interquartile range needs three levels off one sort.

    Args:
        x: values, shape ``(N,)``.
        w: non-negative weights, shape ``(N,)``.
        ps: probability levels in (0, 1).
        bandwidth: kernel width, in cumulative-probability units.

    Returns:
        One scalar per level, in the order given.
    """
    order = jnp.argsort(x)
    xs, ws = x[order], w[order]
    cw = (jnp.cumsum(ws) - 0.5 * ws) / jnp.sum(ws)
    out = []
    for p in ps:
        k = jnp.exp(-0.5 * ((cw - p) / bandwidth) ** 2) * ws
        out.append(jnp.sum(k * xs) / jnp.sum(k))
    return out


def cohort_rows(x_cloud, w, cohort):
    """The rows one cohort contributes, given its patients' readouts.

    Args:
        x_cloud: corrected readouts, shape ``(N, M)``.
        w: eligibility weights, shape ``(N,)``.
        cohort: the :class:`Cohort`.

    Returns:
        Stacked rows, shape ``(K_c,)``, ordered by readout then by statistic.
    """
    rows = []
    for r, sts in zip(cohort.readouts, cohort.stats):
        xr = x_cloud[:, r]
        levels = [0.50] + ([0.25, 0.75] if "iqr" in sts else [])
        q = smooth_weighted_quantiles(xr, w, levels)
        for st in sts:
            if st == "med":
                rows.append(q[0])
            elif st == "mean":
                rows.append(jnp.sum(w * xr) / jnp.sum(w))
            elif st == "iqr":
                rows.append(jnp.log(jnp.clip(q[2] - q[1], 1e-6, None)))
            else:
                raise ValueError(f"unknown statistic {st!r}")
    return jnp.stack(rows)


def eligibility_weights(x_cloud, cohort):
    """Smooth eligibility weight per patient. eq:elig, with a logistic boundary."""
    if cohort.eligibility is None:
        return jnp.ones(x_cloud.shape[0])
    r, lo, hi = cohort.eligibility
    xr = x_cloud[:, r]
    h = ELIGIBILITY_BANDWIDTH
    return jax.nn.sigmoid((xr - lo) / h) * jax.nn.sigmoid((hi - xr) / h)


# -----------------------------------------------------------------------------
# The forward map: phi -> tau. eq:crn through eq:stat.
# -----------------------------------------------------------------------------


def build_omega(s, u_raw, log_omega_measured):
    """Assemble omega from its measured and assumed halves.

    eq:omegameas for j in M, eq:omegaassumed for the rest, with u centred so the
    global level lives in s alone.
    """
    u = u_raw - jnp.mean(u_raw)
    log_omega = jnp.log(OMEGA_0)
    log_omega = log_omega.at[ASSUMED_IDX].add(s + u)
    log_omega = log_omega.at[MEASURED_IDX].set(log_omega_measured)
    return jnp.exp(log_omega)


def reference_levels(mu_0, omega_0, key, g_fn, n=None):
    """c_r, the readout level kappa pivots about. A fixed input, like Z.

    eq:disc as written is ``kappa_r h_r + gamma_r``, which pivots about h_r = 0.
    Zero is not a neutral point for a log-scale readout: log tumour sits near -1.6
    and log fraction near -2, so a 10% change in kappa moves the median of that
    readout by 0.16 to 0.2 in log units. That is a location effect, and it is why
    holding the scale rows out of the flat fit does not hold b out: the location
    rows alone shrink b to about a third of its prior width.

    Pivoting about c_r instead removes that lever to first order. It is an exact
    reparameterisation of eq:disc, since

        kappa (h - c) + c + gamma_new  =  kappa h + gamma_new + c (1 - kappa),

    so gamma_old = gamma_new + c(1 - kappa) and the model family is unchanged.
    What does change is the prior: iid a is now iid on the intercept *at the
    reference level* rather than at zero. That is the intended change, and it is
    the same kind of change the QR attempt made accidentally, so it is worth being
    explicit that this one is deliberate.

    One pivot per cohort and readout, shape ``(C, M)``: each study's own median at
    the plug-in, under its own eligibility rule. See :func:`apply_map` for why the
    study index belongs here and what it costs (no parameters).

    Two weaker choices were tried first and are worth recording, because both look
    reasonable and both fail in the same place. A single median over the whole
    cloud mixes the scenarios in proportions no cohort uses, and left a lever of
    5.90 sd. Taking the median over the cohorts that report each readout improved
    the average but made the worst row worse, 7.55 sd, because a readout measured
    under several scenarios with a real treatment effect has no single level to
    sit at. Only indexing by study removes it.
    """
    n = N_CLOUD if n is None else n
    z = draw_cloud_z(key, n)
    vartheta = mu_0[None, :] + z @ (L_R.T * omega_0[None, :])
    y_all = jnp.stack([g_fn(vartheta, s) for s in range(S)])
    x_all = h_all(y_all)  # (S, N, M)

    # A readout a cohort does not report never enters a row, so its pivot there is
    # irrelevant; the cloud median keeps the array rectangular.
    cloud_med = jnp.median(x_all.reshape(-1, x_all.shape[-1]), axis=0)
    rows = []
    for cohort in COHORTS:
        x_cloud = x_all[cohort.scenario]
        w = eligibility_weights(x_cloud, cohort)
        c = cloud_med
        for r in cohort.readouts:
            med = smooth_weighted_quantiles(x_cloud[:, r], w, [0.50])[0]
            c = c.at[r].set(med)
        rows.append(c)
    return jnp.stack(rows)


def raw_readouts(mu, omega, z, beta_free, g_fn):
    """The cloud's readouts before the measurement map. eq:crn, mech, readout.

    Args:
        g_fn: ``(vartheta_cloud, scenario) -> species``. The true ODE when
            generating data or measuring E_c, the emulator when fitting.

    Returns:
        Shape ``(S, N, M)``. Every scenario, because a fold-change readout is a
        contrast between two of them in the same patient.
    """
    vartheta = mu[None, :] + z @ (L_R.T * omega[None, :])
    y_all = jnp.stack([g_fn(vartheta, s) for s in range(S)])  # (S, N, Q)
    beta = jnp.zeros(Q).at[BETA_SPECIES].set(beta_free)
    return h_all(y_all * jnp.exp(beta)[None, None, :])  # beta is upstream of h_r


def apply_map(x, a, b, c_row):
    """eq:disc for one cohort: ``kappa_r (x - c) + c + gamma_r``.

    ``c`` is that cohort's own level, so the effective offset the map applies is
    ``gamma_r + c_{r,c} (1 - kappa_r)`` and therefore varies by study. That IS a
    different model from a single global pivot, not a reparameterisation of it,
    and the reason to prefer it is that it is the more believable one: two studies
    measuring the same quantity by the same assay do not share a bias, they share
    a calibration. It costs no parameters -- the study dependence is pinned to how
    far each study's level sits from the reference, not free.

    What it buys is that kappa is a pure spread term within a cohort. Under a
    single global pivot kappa moved the location rows by up to 84 sd per unit of
    log kappa, which is why holding the scale rows out of the flat fit did not
    hold b out with them; a global centring cut that to about 7 sd but could not
    remove it, because one pivot cannot sit at the level of a readout measured in
    several scenarios at once.
    """
    gamma = Z @ a  # (M,)
    kappa = jnp.exp(Z @ b)  # (M,)
    return kappa * (x - c_row) + c_row + gamma


def tau_all(mu, omega, z, a, b, beta_free, g_fn, c_ref):
    """Stack tau_c over every cohort.

    The mechanism and the readouts are computed once per scenario and shared, as
    they must be for the cost to be tolerable; only the affine map is per cohort,
    and it is three arithmetic operations.
    """
    x_raw = raw_readouts(mu, omega, z, beta_free, g_fn)
    out = []
    for i, cohort in enumerate(COHORTS):
        x_cloud = apply_map(x_raw[cohort.scenario], a, b, c_ref[i])
        w = eligibility_weights(x_cloud, cohort)
        out.append(cohort_rows(x_cloud, w, cohort))
    return out


# -----------------------------------------------------------------------------
# 1h. Ground truth and the toy data.
# -----------------------------------------------------------------------------


@dataclass
class GroundTruth:
    mu: jnp.ndarray
    omega: jnp.ndarray
    s: float
    u_raw: jnp.ndarray
    log_omega_measured: jnp.ndarray
    a: jnp.ndarray
    b: jnp.ndarray
    beta_free: jnp.ndarray


def make_ground_truth():
    """Pick phi*, from the installed problem. Deliberately not at the priors."""
    t = PROB.truth
    s_true = float(t["s"])
    u_raw_true = jnp.array(t["u_raw"])
    log_omega_meas_true = (
        jnp.log(OMEGA_0[MEASURED_IDX]) + jnp.array(t["log_omega_meas_offset"])
        if N_MEASURED else jnp.zeros(0)
    )
    a_true = jnp.array(t["a"])
    b_true = jnp.array(t["b"])
    # Centred, because the model can only see the centred part. An uncentred
    # beta* would put truth on the flat direction and the recovery table would be
    # reporting the prior.
    beta_true = jnp.array(t["beta"])
    beta_true = beta_true - jnp.mean(beta_true)

    omega_true = build_omega(s_true, u_raw_true, log_omega_meas_true)
    return GroundTruth(
        mu=MU_TRUE,
        omega=omega_true,
        s=s_true,
        u_raw=u_raw_true,
        log_omega_measured=log_omega_meas_true,
        a=a_true,
        b=b_true,
        beta_free=beta_true,
    )


# -----------------------------------------------------------------------------
# 1h2. The two problems.
# -----------------------------------------------------------------------------


def small_problem():
    """P=8, Q=3, M=7, S=2, A=42. The debugging vehicle."""
    return Problem(
        name="small",
        P=8, Q=3, S=2,
        param_names=["r_T", "K_T", "k_kill", "s_E", "p_E", "d_E", "r_S", "K_S"],
        species_names=["tumour", "effector", "stroma"],
        readout_names=[
            "log_tumour",      # density      imaging
            "log_stroma",      # density      histology
            "log_fracE_flow",  # fraction     flow
            "log_fracE_ihc",   # fraction     histology
            "log_fracT_ihc",   # fraction     histology
            "log_ratioET",     # ratio        flow
            "log_ratioST",     # ratio        histology
        ],
        g_one=_small_g_one,
        h_all=_small_h_all,
        Z=_SMALL_Z,
        z_col_names=_SMALL_Z_COLS,
        beta_species=_SMALL_BETA_SPECIES,
        cohorts=_SMALL_COHORTS,
        R_corr=jnp.array(_SMALL_R),
        omega_0=_SMALL_OMEGA_0,
        measured_idx=jnp.array([0, 3]),
        n_j=jnp.array([12.0, 5.0]),
        mu_true=jnp.log(_SMALL_THETA_TRUE),
        mu_0=jnp.log(_SMALL_THETA_TRUE) + _SMALL_MU_OFFSET,
        sigma_1=jnp.array(np.outer(_SMALL_SD1, _SMALL_SD1) * _SMALL_R),
        truth=dict(
            s=0.30,  # the population is 35% wider than omega_0 assumes
            u_raw=[0.35, -0.20, 0.10, -0.30, 0.25, -0.20],
            log_omega_meas_offset=[0.15, -0.25],
            a=[0.10, -0.18, 0.22, 0.12],  # intercept, fraction, ratio, flow
            b=[0.08, 0.12, -0.10, 0.09],
            beta=[0.12, -0.16, 0.07],
        ),
        n_cloud=800,
        # sec:flat's cost note: the flat fit's N only has to resolve the location
        # statistics and a rough width, not the flanks at a precision that supports
        # inference on omega, so it sits well below the population fit's N.
        n_cloud_flat=400,
        n_truth=40_000,
        b_boot=400,
        l_emu=60,  # held-out clouds for E_c; must exceed max_c K_c
        emu_pool=8_192,  # a power of two, for Sobol balance
        emu_hidden=(96, 96),
        emu_steps=4_000,
    )


# The QSP-scale cohort table, laid out against
# pdac-build/calibration_targets/vpop_marginal_targets.csv: 5 scenarios, cohort
# sizes from 6 to 702, a bit over half the rows synthesized rather than reported,
# and the same mix of quantity types. Readout indices follow
# ``qsp_scale_mechanism.build_readouts``:
#
#   0-7    densities    tumour cd8 cd4 treg m2tam mdsc caf collagen
#   8-15   percentages  cd8 cd4 treg cd8exh m1tam m2tam dc nk
#   16-19  fractions    treg/cd4  icaf/caf  mycaf/caf  texh/cd8
#   20-22  ratios       m1:m2  cd8:treg  tumour:caf
#   23-25  fold change  cd8  treg  m2tam        (treated arms only)
#   26-29  concentr.    il6 il10 vegf cxcl9
#
# The fold-change cohorts sit in scenarios 3 and 4 only. In the gvax arm the
# effect is small enough that the between-patient sd of a fold change is 0.005,
# which would make that row's IQR numerical noise and its V_c block singular.
_QSP_COHORTS: tuple[Cohort, ...] = (
    Cohort("li2022", 0, 16, (8, 11), (_MI, _MI)),
    Cohort("dens_pooled", 0, 113, (1, 2, 3), (_MI, _MI, _MI)),
    Cohort("myeloid", 0, 113, (4, 5, 12), (_MI, _MI, ("med",)), synthesized=True),
    # icaf_fraction and mycaf_fraction are log p and log (1 - p) of the same
    # quantity, so two rows of one cohort would be a deterministic function of each
    # other, their bootstrap noise would be perfectly correlated, and that block of
    # V_c would be singular: max|corr| = 1.00, a 1e10 direction in V^-1 through the
    # ridge, and a stalled sampler. The real corpus reports both, measured
    # separately with independent error; this toy computes both from the same
    # species, so they have to go in different cohorts.
    Cohort("stroma", 0, 215, (6, 7, 17), (_MI, _MI, _MI), synthesized=True),
    Cohort("ratios", 0, 50, (20, 21, 18), (_MMI, _MI, _MI)),
    Cohort("cytokine", 0, 10, (26, 27, 28), (_MI, _MI, ("med",)), synthesized=True),
    Cohort("fractions", 0, 120, (16, 19, 15), (_MMI, _MI, _MI)),
    # A burden-selected cohort, as the real trials are.
    Cohort("burden", 0, 40, (0, 22), (_MMI, _MI), eligibility=(0, -2.0, 2.5)),
    Cohort("progression", 1, 6, (0,), (("med",),)),
    Cohort("gvax_d21", 2, 9, (8, 9, 10, 12, 13), (_MI, _MI, _MI, _MI, _MI)),
    Cohort("nivo_d21", 3, 11, (8, 11, 10, 12, 13), (_MI, _MI, _MI, _MI, _MI)),
    Cohort("nivo_fc", 3, 11, (23, 24), (_MI, _MI)),
    Cohort("nivo_deep", 3, 113, (14, 29, 22), (_MMI, _MI, _MI), synthesized=True),
    Cohort("urelumab", 4, 10, (8, 23), (_MI, _MI)),
    Cohort("urelumab_big", 4, 702, (25, 10, 27), (_MMI, _MI, _MI), synthesized=True),
)

_QSP_SIZES = {
    # Q, P, n_cloud, n_cloud_flat, n_truth, b_boot, l_emu, pool, hidden, steps
    "medium": (60, 90, 300, 200, 4_000, 300, 80, 4_096, (128, 128), 4_000),
    "full": (179, 271, 500, 250, 8_000, 400, 90, 16_384, (256, 256, 256), 10_000),
}


def qsp_problem(size="full", seed=0):
    """The real model's size: Q=179, P=271, S=5, M=30, A=88.

    Recovery against phi* is not the question here. 271 mu directions and 271
    omega directions against 88 rows means the mechanism block spans the whole row
    space, so gamma and kappa are exactly aliased with mu rather than nearly, the
    Z test returns zero for every column by arithmetic, and the posterior along
    almost every coordinate is the prior. What the run measures is whether that is
    what actually happens, what the emulator does in 271 input dimensions, and
    which handful of directions survive.

    ``|M| = 0`` by construction, because that is the real state of
    parameters/omega_priors.csv: every entry there is a pin, not a measurement.
    """
    from qsp_scale_mechanism import (
        build_network, build_readouts, make_g, population_inputs,
    )

    Q_, P_, ncl, nclf, ntr, nboot, lemu, pool, hid, steps = _QSP_SIZES[size]
    S_ = 5
    net = build_network(Q_, P_, S_, seed=seed)
    names, kinds, modality, h_fn, Z_, zcols = build_readouts(net)
    R, omega_0, mu_true, mu_0, sigma_1 = population_inputs(net, S_, seed=seed)

    # The discrepancy set S: one representative species per functional class. All
    # Q of them would put 179 free coefficients against 88 rows on their own, and
    # the centring constraint that makes beta identifiable at Q = 3 does nothing
    # about that. One per class is the coarsest split the readouts could in
    # principle separate, which is already optimistic.
    beta_species = jnp.array([lo for (lo, hi) in net.classes.values()])

    rng = np.random.default_rng(seed + 4242)
    dim_z = int(Z_.shape[1])
    return Problem(
        name=f"qsp-{size}",
        P=P_, Q=Q_, S=S_,
        param_names=net.param_names,
        species_names=net.species_names,
        readout_names=names,
        g_one=make_g(net),
        h_all=h_fn,
        Z=Z_,
        z_col_names=zcols,
        beta_species=beta_species,
        cohorts=_QSP_COHORTS,
        R_corr=R,
        omega_0=omega_0,
        measured_idx=jnp.zeros(0, dtype=jnp.int32),  # |M| = 0, as in the real prior
        n_j=jnp.zeros(0),
        mu_true=mu_true,
        mu_0=mu_0,
        sigma_1=sigma_1,
        truth=dict(
            s=0.30,
            u_raw=list(rng.normal(0.0, 0.22, P_)),
            log_omega_meas_offset=[],
            a=list(rng.normal(0.0, 0.16, dim_z)),
            b=list(rng.normal(0.0, 0.10, dim_z)),
            beta=list(rng.normal(0.0, 0.12, int(beta_species.shape[0]))),
        ),
        n_cloud=ncl, n_cloud_flat=nclf, n_truth=ntr, b_boot=nboot, l_emu=lemu,
        emu_pool=pool, emu_hidden=hid, emu_steps=steps,
    )


# The small problem is installed at import so the module-level names exist for
# anything that reads them before main runs. ``--size`` re-installs.
install_problem(small_problem())


def _hard_rows(sample, cohort):
    """The statistics a paper would print, from n_c patients. Numpy, not JAX."""
    rows = []
    for r, sts in zip(cohort.readouts, cohort.stats):
        xr = sample[:, r]
        for st in sts:
            if st == "med":
                rows.append(np.median(xr))
            elif st == "mean":
                rows.append(np.mean(xr))
            elif st == "iqr":
                q25, q75 = np.percentile(xr, [25.0, 75.0])
                rows.append(np.log(max(q75 - q25, 1e-6)))
            else:
                raise ValueError(f"unknown statistic {st!r}")
    return rows


def generate_data(truth, key, c_ref):
    """Draw the observed statistics the honest way.

    Not ``tau + N(0, V_c)``. A big true cloud stands in for the population, each
    cohort draws its own ``n_c`` patients from it, and the statistics come from
    those ``n_c`` with the ordinary hard quantile. So the observed rows carry
    real finite-``n`` sampling noise with the real cross-readout dependence, and
    eq:V has something to be right or wrong about.
    """
    key_cloud, key_draw = random.split(key)
    z_truth = draw_cloud_z(key_cloud, N_TRUTH)

    x_raw = raw_readouts(truth.mu, truth.omega, z_truth, truth.beta_free,
                         g_true_cloud)

    observed = []
    for i, (cohort, k) in enumerate(zip(COHORTS, random.split(key_draw, len(COHORTS)))):
        x_cloud = apply_map(x_raw[cohort.scenario], truth.a, truth.b, c_ref[i])
        w = np.asarray(eligibility_weights(x_cloud, cohort))
        prob = jnp.array(w / w.sum())
        idx = np.asarray(
            random.choice(k, N_TRUTH, shape=(cohort.n,), replace=True, p=prob)
        )
        observed.append(jnp.array(_hard_rows(np.asarray(x_cloud)[idx], cohort)))
    return observed


# -----------------------------------------------------------------------------
# 1i. V_c. eq:V for the bootstrap, eq:Ec for the emulator, eq:Vsplit to assemble.
# -----------------------------------------------------------------------------


def bootstrap_V(phi_0_mu, phi_0_omega, key, emu, c_ref, n_boot=None):
    """V^boot_c at a plug-in, one block per cohort. eq:V.

    Resamples *patients*, so each replicate carries the whole readout vector and
    the dependence between readouts of one cohort lands in the off-diagonal
    blocks. The discrepancy terms sit at their prior centres, which is what makes
    this a plug-in rather than a function of phi.
    """
    n_boot = B_BOOT if n_boot is None else n_boot
    key_cloud, key_boot = random.split(key)
    z0 = draw_cloud_z(key_cloud, N_CLOUD)
    a0, b0, beta0 = jnp.zeros(DIM_Z), jnp.zeros(DIM_Z), jnp.zeros(N_BETA)

    x_raw = raw_readouts(phi_0_mu, phi_0_omega, z0, beta0, emu.cloud)

    blocks = []
    for i, (cohort, k) in enumerate(zip(COHORTS, random.split(key_boot, len(COHORTS)))):
        x_cloud = apply_map(x_raw[cohort.scenario], a0, b0, c_ref[i])
        x_np = np.asarray(x_cloud)
        w = np.asarray(eligibility_weights(x_cloud, cohort))
        prob = jnp.array(w / w.sum())
        idx = np.asarray(
            random.choice(k, N_CLOUD, shape=(n_boot, cohort.n), replace=True, p=prob)
        )
        reps = np.array([_hard_rows(x_np[idx[t]], cohort) for t in range(n_boot)])
        blocks.append(np.cov(reps, rowvar=False).reshape(cohort.K, cohort.K))
    return blocks


def emulator_E(phi_0_mu, phi_0_omega, key, emu, c_ref, n_clouds=None):
    """E_c by eq:Ec -- held-out clouds, evaluated both ways, covariance of d.

    Both evaluations use the same ``z``, so the Monte Carlo fluctuation of the
    cloud is common to the two and differences out. What is left is the
    emulator's own error, propagated all the way to the statistics rather than
    measured on the readouts and expanded.
    """
    n_clouds = L_EMU if n_clouds is None else n_clouds
    a0, b0, beta0 = jnp.zeros(DIM_Z), jnp.zeros(DIM_Z), jnp.zeros(N_BETA)
    diffs = []
    for k in random.split(key, n_clouds):
        z = draw_cloud_z(k, N_CLOUD)
        emu_tau = tau_all(phi_0_mu, phi_0_omega, z, a0, b0, beta0, emu.cloud, c_ref)
        sim_tau = tau_all(phi_0_mu, phi_0_omega, z, a0, b0, beta0, g_true_cloud, c_ref)
        diffs.append(np.concatenate([np.asarray(e - s) for e, s in zip(emu_tau, sim_tau)]))
    D = np.array(diffs)  # (L, A_ROWS)

    blocks, means, off = [], [], 0
    for cohort in COHORTS:
        sl = D[:, off:off + cohort.K]
        blocks.append(np.cov(sl, rowvar=False).reshape(cohort.K, cohort.K))
        means.append(sl.mean(axis=0))
        off += cohort.K
    return blocks, means


def assemble_V(v_boot_blocks, reported_U=None, E_blocks=None):
    """eq:Vsplit -- V_c = D_c rho_c D_c + E_c.

    Correlations always come from the bootstrap, because no source reports a
    correlation between readouts. Scales come from a reported uncertainty where
    one exists and from the bootstrap otherwise.

    Args:
        v_boot_blocks: list of ``(K_c, K_c)`` bootstrap covariances.
        reported_U: optional list of ``(K_c,)`` arrays; ``nan`` where no source
            reports an uncertainty for that row.
        E_blocks: optional list of ``(K_c, K_c)`` emulator covariances.

    Returns:
        ``(V blocks, Cholesky factors)``.
    """
    out, chols = [], []
    for i, Vb in enumerate(v_boot_blocks):
        sd_boot = np.sqrt(np.diag(Vb))
        rho = Vb / np.outer(sd_boot, sd_boot)
        np.fill_diagonal(rho, 1.0)

        sd = sd_boot.copy()
        if reported_U is not None and reported_U[i] is not None:
            U = np.asarray(reported_U[i], dtype=float)
            sd = np.where(np.isnan(U), sd_boot, U)

        V = np.outer(sd, sd) * rho
        if E_blocks is not None:
            V = V + np.asarray(E_blocks[i])
        # A small ridge keeps the Cholesky well behaved when two readouts of one
        # cohort are nearly collinear, which a shared denominator makes likely.
        V = V + 1e-10 * np.eye(V.shape[0])
        out.append(jnp.array(V))
        chols.append(jnp.array(np.linalg.cholesky(V)))
    return out, chols


def subset_V(V_blocks, masks):
    """Restrict each ``V_c`` to a subset of its rows, and refactor.

    A Cholesky factor cannot be subset directly, so the submatrix is taken from
    ``V_c`` itself and refactored. Used to give the flat fit the location-row
    blocks while leaving the scale rows out of the likelihood entirely.
    """
    out, chols = [], []
    for Vb, msk in zip(V_blocks, masks):
        idx = np.flatnonzero(np.asarray(msk))
        Vn = np.asarray(Vb)[np.ix_(idx, idx)]
        out.append(jnp.array(Vn))
        chols.append(jnp.array(np.linalg.cholesky(Vn)))
    return out, chols


# -----------------------------------------------------------------------------
# 1j. The projection tests. What eq:disc asserts, checked against the forward map.
# -----------------------------------------------------------------------------
#
# Every term in the model moves the A stacked rows in some direction. Nudge a
# parameter, see how the A predicted numbers shift: that is one column of
# d tau / d phi. Two terms are aliased exactly when their directions overlap.
#
# So both tests below are one operation. Take the direction under test, project
# out everything the competing block can reproduce, and report what is left as a
# fraction of what you started with. Near zero means aliased and the prior decides
# the split; near one means the data can separate it.
#
#   Z test  competitor is the mechanism (mu, omega). eq:disc claims the columns of
#           Z have no mechanistic twin. A near-zero residual falsifies that claim
#           for that column. This is the note's own test in "Z is asserted, not
#           tested", which it says needs the emulator.
#
#   S test  competitor is the measurement map (a, b). A species whose bias gamma
#           and kappa can absorb does not belong in S.
#
# Both are whitened by V^{-1/2} first. A direction is not identified because it is
# geometrically orthogonal to another, it has to be orthogonal in the metric the
# data impose, and V_c is that metric.


def _tau_flat(mu, omega, a, b, beta_free, z, g_fn, c_ref):
    """All A rows as one vector, so it can be differentiated in one go."""
    return jnp.concatenate(tau_all(mu, omega, z, a, b, beta_free, g_fn, c_ref))


def row_jacobians(phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref):
    """d(tau)/d(block) at phi_0, whitened by V^{-1/2}. One (A, dim) array per block.

    The discrepancy sits at its prior centre, matching :func:`bootstrap_V`, so this
    is a statement about the model at the plug-in and not about any fit.
    """
    a0, b0, beta0 = jnp.zeros(DIM_Z), jnp.zeros(DIM_Z), jnp.zeros(N_BETA)

    def f(mu, omega, a, b, beta):
        return _tau_flat(mu, omega, a, b, beta, z, emu.cloud, c_ref)

    args = (phi_0_mu, phi_0_omega, a0, b0, beta0)
    names = ("mu", "omega", "a", "b", "beta")
    out = {}
    for k, nm in enumerate(names):
        J = np.asarray(jax.jacfwd(f, argnums=k)(*args))
        # Blockwise V_c^{-1/2}, by triangular solve against the Cholesky factor.
        rows, off = [], 0
        for i, cohort in enumerate(COHORTS):
            L = np.asarray(V_chol[i])
            rows.append(solve_triangular(L, J[off:off + cohort.K], lower=True))
            off += cohort.K
        out[nm] = np.concatenate(rows)
    return out


def _orth(A, tol=1e-10):
    """Orthonormal basis for col(A), dropping numerically dependent directions."""
    U, sv, _ = np.linalg.svd(A, full_matrices=False)
    if sv.size == 0:
        return U[:, :0]
    return U[:, sv > tol * max(float(sv[0]), 1e-30)]


def _residual_fraction(target, basis):
    """Per-column fraction of ``target`` that ``basis`` cannot reproduce."""
    Qb = _orth(basis)
    res = target - Qb @ (Qb.T @ target)
    den = np.linalg.norm(target, axis=0)
    return np.linalg.norm(res, axis=0) / np.clip(den, 1e-30, None)


def _residual_spectrum(target, basis):
    """Singular values of ``target`` after removing col(basis), scaled to sv[0].

    Reported beside the per-column residuals because columns can each be
    individually distinct while a combination of them is flat. That is exactly the
    case for a common beta, so per-column numbers alone would miss it.
    """
    Qb = _orth(basis)
    res = target - Qb @ (Qb.T @ target)
    sv_full = np.linalg.svd(target, compute_uv=False)
    sv_res = np.linalg.svd(res, compute_uv=False)
    scale = max(float(sv_full[0]), 1e-30)
    return sv_res / scale


def print_pivot_offsets(phi_0_mu, phi_0_omega, z, V, emu, c_ref, top=5):
    """How much location lever kappa keeps after centring. No fit required.

    The location row for readout r in cohort c moves with log kappa_r as

        d median_c(x~_r) / d log kappa_r = kappa_r (median_c(h_r) - c_r),

    which at the plug-in is just the gap between the cohort's own median and the
    frozen pivot. So that gap IS the lever, in the readout's own units, and
    dividing by the row's sd from V_c says how many sd of that row a unit change
    in log kappa buys. This is the number that decides whether holding the scale
    rows out of the flat fit holds b out with them.

    Reported beside the uncentred lever, which is the same quantity at c_r = 0 and
    is therefore just the median itself. The ratio of the two columns is what
    centring bought.
    """
    a0, b0, beta0 = jnp.zeros(DIM_Z), jnp.zeros(DIM_Z), jnp.zeros(N_BETA)
    x_raw = raw_readouts(phi_0_mu, phi_0_omega, z, beta0, emu.cloud)

    rows = []
    for i, cohort in enumerate(COHORTS):
        x_cloud = apply_map(x_raw[cohort.scenario], a0, b0, c_ref[i])
        w = eligibility_weights(x_cloud, cohort)
        sd = np.sqrt(np.diag(np.asarray(V[i])))
        k = 0
        for r, sts in zip(cohort.readouts, cohort.stats):
            xr = x_cloud[:, r]
            med = float(smooth_weighted_quantiles(xr, w, [0.50])[0])
            mean = float(jnp.sum(w * xr) / jnp.sum(w))
            for st in sts:
                if st in LOCATION_STATS:
                    # The lever is |statistic - c|, and the statistic is not
                    # always the median. c_ref is the median, so a median row is
                    # centred exactly and a MEAN row is left with the skewness of
                    # the pushforward: mean(kappa(x-c)+c+gamma) = kappa(mean-c)+..,
                    # so the leftover lever is mean - median. One pivot cannot zero
                    # both, and pivoting on the mean instead would simply move the
                    # leak to the median rows. Using the median for both here
                    # under-reported the mean rows and hid the residual channel by
                    # which the flat fit still learns about b.
                    stat = med if st == "med" else mean
                    rows.append((f"{cohort.name}/{READOUT_NAMES[r]}/{st}",
                                 abs(stat - float(c_ref[i, r])) / sd[k],
                                 abs(stat) / sd[k]))
                k += 1

    rows.sort(key=lambda t: -t[1])
    cent = np.array([t[1] for t in rows])
    unc = np.array([t[2] for t in rows])
    print("\nkappa's residual lever on the location rows (eq:disc pivot)")
    print("  sd of the row that one unit of log kappa moves it. Small means the")
    print("  flat fit's scale-row holdout really does hold b out.")
    print(f"  {'row':<26}{'centred':>10}{'uncentred':>11}")
    for lab, c, u in rows[:top]:
        print(f"  {lab:<26}{c:>10.2f}{u:>11.2f}")
    print(f"  {'max over ' + str(len(rows)) + ' location rows':<26}"
          f"{cent.max():>10.2f}{unc.max():>11.2f}")
    print(f"  {'mean':<26}{cent.mean():>10.2f}{unc.mean():>11.2f}")
    print(f"  centring cut the mean lever by {unc.mean() / max(cent.mean(), 1e-12):.1f}x")


def laplace_blocks(phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref, flat=False,
                   rows=None, at=None):
    """Whitened ``d tau / d theta`` per latent site, in NUMPYRO's coordinates.

    Not the standardised ones. The sites here are exactly the sites the model
    samples, with the shapes it samples them in, because the output feeds a mass
    matrix and a coordinate mismatch there would be silent: numpyro would happily
    accept a matrix whose blocks correspond to the wrong parameters.

    ``rows`` restricts to a subset of each cohort's rows, for the flat fit's
    location-only likelihood. ``at`` is the point to linearise about, as a dict of
    site values; the default is the prior centre, which is where V_c was built.

    Returns ``(names, dims, prior_sd, whitened blocks, labels)``.
    """
    zero_meas = jnp.zeros(0)
    log_om_meas0 = jnp.log(OMEGA_0[MEASURED_IDX]) if N_MEASURED else zero_meas

    def tau_of(*vals):
        d = dict(zip(names, vals))
        mu = MU_0 + L_SIGMA_1 @ d["mu_raw"]
        if flat:
            omega = OMEGA_0
        else:
            omega = build_omega(
                d["s"], d["u_raw"],
                d.get("log_omega_measured", log_om_meas0),
            )
        beta_free = TAU_BETA * (d["beta_raw"] - jnp.mean(d["beta_raw"]))
        taus = tau_all(mu, omega, z, d["a"], d["b"], beta_free, emu.cloud, c_ref)
        if rows is not None:
            taus = [t[jnp.array(np.flatnonzero(m))] for t, m in zip(taus, rows)]
        return jnp.concatenate(taus)

    # The order here is the order the mass matrix will be packed in.
    names, init, prior_sd = ["mu_raw"], [jnp.zeros(P)], [1.0]
    if not flat:
        names += ["s", "u_raw"]
        init += [0.0, jnp.zeros(N_ASSUMED)]
        prior_sd += [TAU_S, TAU_U]
        if N_MEASURED:
            names += ["log_omega_measured"]
            init += [log_om_meas0]
            prior_sd += [np.asarray(TAU_OMEGA_MEASURED)]
    names += ["a", "b", "beta_raw"]
    init += [jnp.zeros(DIM_Z), jnp.zeros(DIM_Z), jnp.zeros(N_BETA)]
    prior_sd += [SIGMA_A, SIGMA_B, 1.0]
    if at is not None:
        init = [at.get(nm, v) for nm, v in zip(names, init)]

    n_row = A_ROWS if rows is None else sum(int(np.asarray(m).sum()) for m in rows)
    blocks, labels, dims = [], [], []
    for k, nm in enumerate(names):
        J = np.asarray(jax.jacfwd(tau_of, argnums=k)(*init)).reshape(n_row, -1)
        out, off = [], 0
        for i, cohort in enumerate(COHORTS):
            k_c = cohort.K if rows is None else int(np.asarray(rows[i]).sum())
            L = np.asarray(V_chol[i])
            out.append(solve_triangular(L, J[off:off + k_c], lower=True))
            off += k_c
        Jw = np.concatenate(out)
        blocks.append(Jw)
        dims.append(Jw.shape[1])
        labels += [nm if Jw.shape[1] == 1 else f"{nm}[{j}]"
                   for j in range(Jw.shape[1])]
    return names, dims, prior_sd, blocks, labels


def should_fix_mass(dim, warmup, factor=3):
    """Whether to freeze the computed metric or let warmup re-estimate it.

    numpyro's final adaptation window estimates the mass matrix from scratch, so
    leaving adaptation on discards the Laplace metric and keeps only the benefit of
    having explored well while getting there. That is the right trade when the
    warmup has enough draws to estimate a dim x dim covariance, and the wrong one
    when it does not: at 572 coordinates and 400 draws the re-estimate is worse
    than the matrix it replaces.

    So the rule is the one that decides which estimator is better, not a global
    switch. Below about three draws per dimension, keep the computed metric.
    """
    return warmup < factor * dim


def laplace_covariance(phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref,
                       flat=False, rows=None, at=None):
    """``(site names, dims, G^-1)`` at a point. G is the Laplace metric."""
    names, dims, prior_sd, blocks, _ = laplace_blocks(
        phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref, flat=flat, rows=rows, at=at
    )
    Jw = np.hstack(blocks)
    prec = np.concatenate([
        np.full(d, 1.0 / float(sd) ** 2) if np.isscalar(sd)
        else 1.0 / np.asarray(sd, dtype=float) ** 2
        for d, sd in zip(dims, prior_sd)
    ])
    G = np.diag(prec) + Jw.T @ Jw
    C = np.linalg.inv(G)
    return names, dims, 0.5 * (C + C.T)


def laplace_inverse_mass(phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref,
                         flat=False, rows=None):
    """``G^-1`` as numpyro's ``inverse_mass_matrix``, keyed by the site tuple.

    ``G = diag(prior precision) + J^T V^-1 J`` is the Gauss-Newton information of
    the posterior at the plug-in: the Laplace METRIC. A mass matrix should equal
    the posterior precision, so what NUTS wants is ``G``, and what its argument is
    called is the inverse, hence ``G^-1`` here.

    This is worth computing rather than adapting. ``dense_mass`` tries to learn the
    same matrix from the warmup draws, which at 210 coordinates and 400 draws it
    cannot do; the run this replaces was spending 1023 gradients an iteration
    because the metric it had was nearly arbitrary. And unlike the QR
    reparameterisation tried earlier in this file, a mass matrix does not touch the
    prior: it is a choice of kinetic energy, so the stationary distribution is
    unchanged whatever the matrix, and a bad one costs only speed.

    G is positive definite by construction, since the prior precision is.
    """
    names, dims, prior_sd, blocks, _ = laplace_blocks(
        phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref, flat=flat, rows=rows
    )

    # numpyro packs a dense mass matrix over sites in SORTED name order and keys
    # it by tuple(sorted(sites)). Getting that wrong is the one way this can fail
    # quietly -- the matrix would be accepted with its blocks attached to the wrong
    # parameters -- so the blocks are permuted here rather than assumed to match.
    order = sorted(range(len(names)), key=lambda i: names[i])
    blocks = [blocks[i] for i in order]
    dims = [dims[i] for i in order]
    prior_sd = [prior_sd[i] for i in order]
    names = [names[i] for i in order]

    Jw = np.hstack(blocks)
    prec = np.concatenate([
        np.full(d, 1.0 / float(sd) ** 2) if np.isscalar(sd)
        else 1.0 / np.asarray(sd, dtype=float) ** 2
        for d, sd in zip(dims, prior_sd)
    ])
    G = np.diag(prec) + Jw.T @ Jw
    G_inv = np.linalg.inv(G)
    G_inv = 0.5 * (G_inv + G_inv.T)  # symmetrise against round-off
    return {tuple(names): jnp.array(G_inv)}


def conditioning_report(phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref, top=8,
                        flat=False, rows=None):
    """The posterior's anisotropy at the plug-in, before any MCMC is run.

    This is the diagnostic for "why is NUTS saturating its tree depth". Work in
    STANDARDISED sampling coordinates, meaning the ones in which every prior is
    exactly N(0, 1): mu_raw and beta_raw already are, and s, u_raw, a, b are
    divided by their prior sds. In those coordinates the Gauss-Newton information
    of the posterior is

        G = I + J^T J,

    with J the whitened d tau / d (standardised coordinate). So the eigenvalues of
    G are ``1 + sigma_k^2`` for the singular values of J, the condition number is
    ``(1 + sigma_max^2) / (1 + sigma_min^2)``, and NUTS's trajectory length scales
    as its square root. With A rows and more parameters than that, at least
    (dim - A) eigenvalues are exactly 1 by construction: those directions return
    the prior and are perfectly conditioned. The damage is all in sigma_max.

    Reporting it this way separates the two candidate explanations, which need
    different fixes. If the natural-scale spread of the parameters were the
    problem, it would show up as many coordinates with moderate sigma. If instead
    a handful of directions are very sharply determined, sigma_max is large and
    everything else is at 1 -- that is a likelihood-anisotropy problem, and
    rescaling parameters would do nothing about it.
    """
    names, dims, prior_sd, blocks, labels = laplace_blocks(
        phi_0_mu, phi_0_omega, z, V_chol, emu, c_ref, flat=flat, rows=rows
    )

    # Standardise: J_std = J * prior_sd, which is the chain rule for the change of
    # coordinate theta -> theta / prior_sd. In those coordinates the prior
    # precision is exactly I and the eigenvalues read as "1 = returns the prior".
    sd_all = np.concatenate([np.full(d, sd) if np.isscalar(sd) else np.asarray(sd)
                             for d, sd in zip(dims, prior_sd)])
    Jw = np.hstack(blocks) * sd_all[None, :]
    dim = Jw.shape[1]
    sv = np.linalg.svd(Jw, compute_uv=False)
    eig = np.concatenate([1.0 + sv**2, np.ones(max(dim - sv.size, 0))])
    eig = np.sort(eig)[::-1]
    cond = eig[0] / eig[-1]
    blocks = [B * np.asarray(sd if not np.isscalar(sd) else np.full(d, sd))[None, :]
              for B, d, sd in zip(blocks, dims, prior_sd)]

    n_row = Jw.shape[0]
    print(f"\nposterior conditioning at the plug-in "
          f"({dim} standardised coordinates, {n_row} rows)")
    print(f"  G = I + J^T J, so an eigenvalue of 1 is a direction that returns "
          f"the prior")
    print(f"  condition number {cond:,.0f}, so trajectories run about "
          f"{np.sqrt(cond):,.0f}x the")
    print(f"  shortest direction. NUTS needs that many steps before it can "
          f"U-turn.")
    print(f"  eigenvalues at the prior (= 1.000): {(eig < 1.0 + 1e-9).sum()} "
          f"of {dim}")
    print("  largest: " + ", ".join(f"{v:.3g}" for v in eig[:6]))

    # Where does the sharp direction live? The leading right singular vector of
    # Jw is the most-determined direction; its loadings name the coordinates.
    _, _, Vt = np.linalg.svd(Jw, full_matrices=False)
    v = Vt[0]
    print(f"  the sharpest direction (eigenvalue {eig[0]:,.0f}) loads on:")
    for t in np.argsort(-np.abs(v))[:top]:
        print(f"    {labels[t]:<16}{v[t]:+.3f}")

    print(f"  {'block':<10}{'dim':>5}{'sigma_max':>12}{'n sigma>1':>11}"
          f"{'share of tr(J^T J)':>20}")
    total = float((Jw**2).sum())
    off = 0
    for nm, B in zip(names, blocks):
        s_b = np.linalg.svd(B, compute_uv=False)
        print(f"  {nm:<10}{B.shape[1]:>5}{s_b[0]:>12.2f}{int((s_b > 1).sum()):>11}"
              f"{float((B**2).sum()) / max(total, 1e-30):>20.3f}")
        off += B.shape[1]
    return cond, eig, labels


def _null_residual(n_rows, basis):
    """Residual a random direction would score against ``basis``.

    Without this the numbers cannot be read. A competitor block of rank k in an
    A-row space leaves only an (A-k)-dimensional complement, so a direction drawn
    uniformly at random already scores sqrt((A-k)/A). Reporting a residual of 0.25
    as evidence of aliasing is meaningless when chance alone gives 0.40.
    """
    k = _orth(basis).shape[1]
    return float(np.sqrt(max(n_rows - k, 0) / max(n_rows, 1))), k


def print_projection_report(J):
    """The two tests, from the whitened Jacobians of :func:`row_jacobians`."""
    mech = np.hstack([J["mu"], J["omega"]])
    meas = np.hstack([J["a"], J["b"]])
    A = J["mu"].shape[0]

    print(f"\nprojection tests, whitened by V^-1/2 ({A} rows)")
    print("  residual = fraction of a direction the competing block cannot reproduce")
    print("  read every residual against the random-direction null on its line: "
          "below the null")
    print("  means more aliased than chance, at or above means a direction of its own")

    null_mech, k_mech = _null_residual(A, mech)
    ra = _residual_fraction(J["a"], mech)
    rb = _residual_fraction(J["b"], mech)
    print("\n  Z test: does each column of Z have a mechanistic twin in (mu, omega)?")
    print(f"    competitor rank {k_mech} of {A} rows, so the null is {null_mech:.3f}")
    print(f"    {'column':<14}{'gamma resid':>13}{'kappa resid':>13}")
    for k, nm in enumerate(Z_COL_NAMES[:DIM_Z]):
        print(f"    {nm:<14}{ra[k]:>13.3f}{rb[k]:>13.3f}")
    if max(float(ra.max()), float(rb.max())) < null_mech:
        print(f"    every column is below the null. The rows cannot separate the "
              f"measurement map")
        print(f"    from the mechanism at all here: {k_mech} mechanism directions "
              f"in {A} rows leaves")
        print("    too little room. This is a verdict on the row budget, not on Z.")

    null_meas, k_meas = _null_residual(A, meas)
    rbeta = _residual_fraction(J["beta"], meas)
    print("\n  S test: can (gamma, kappa) absorb a species bias?")
    print(f"    competitor rank {k_meas} of {A} rows, so the null is {null_meas:.3f}")
    print(f"    {'species':<14}{'resid':>13}")
    for q in range(N_BETA):
        nm = SPECIES_NAMES[int(BETA_SPECIES[q])]
        print(f"    {nm:<14}{rbeta[q]:>13.3f}")

    sv = _residual_spectrum(J["beta"], meas)
    kept = int((sv > 1e-8).sum())
    print("    singular spectrum of the beta block after removing (a, b): "
          + ", ".join(f"{v:.2e}" for v in sv))
    print(f"    {kept} of {N_BETA} directions survive.")
    common = J["beta"] @ (np.ones(N_BETA) / np.sqrt(N_BETA))
    Qm = _orth(meas)
    r_common = (np.linalg.norm(common - Qm @ (Qm.T @ common))
                / max(float(np.linalg.norm(common)), 1e-30))
    print(f"    the common shift specifically: residual {r_common:.3f}")
    if r_common < 1e-6:
        print("      exactly reproduced by gamma, which is why the model centres "
              "beta")
    else:
        print("      free of the measurement map. With an intercept in Z it was "
              "exactly 0,")
        print("      so this is what dropping the intercept bought. beta stays "
              "centred anyway:")
        print("      see the row budget below for why the constraint is still the "
              "safe choice.")

    # The row budget. Everything above is a statement about directions in a space
    # of dimension A, so it is only meaningful while the competing blocks leave
    # some of that space over. Once (a, b, mu, omega) spans all A rows, every
    # residual against the full model is zero by arithmetic and no first-order
    # identification claim can be made about any block at all.
    full = np.hstack([J["a"], J["b"], J["mu"], J["omega"]])
    k_full = _orth(full).shape[1]
    n_par = sum(J[k].shape[1] for k in ("a", "b", "mu", "omega"))
    print(f"\n  row budget: {n_par} parameters in (a, b, mu, omega), rank {k_full}, "
          f"{A} rows")
    if k_full >= A:
        print("    the row space is SATURATED. No block is identified at first "
              "order; the")
        print("    residuals above are relative aliasing between blocks, not "
              "evidence that")
        print("    any of them is estimable. Only the priors separate them. More "
              "rows is the")
        print("    only fix, and this is a verdict on A, not on Z.")


# =============================================================================
# PART 2 -- THE MODEL
# =============================================================================


def population_model(z, V_chol, emu, c_ref, observed=None, flat=False,
                     location_only=False):
    """eq:pop through eq:post, as one NumPyro model.

    Args:
        z: the frozen latent cloud, shape ``(N, P)``. eq:crn draws it once.
        V_chol: list of Cholesky factors of the frozen ``V_c``. Shapes must match
            whichever row set ``location_only`` selects.
        emu: the :class:`Emulator`. The fit never touches the true ODE.
        observed: list of observed statistic vectors, or ``None`` to draw from the
            prior predictive.
        flat: eq:phiflat. Pin ``omega = omega_0`` and drop the three spread terms
            from the parameter vector. Nothing else changes, exactly as sec:flat
            says: eq:crn through eq:post run as written with omega a constant.
        location_only: keep only the location rows in the likelihood, so the scale
            rows are genuinely held out.
    """
    # --- the centre. eq:mu, with the stage-1 covariance rather than a diagonal.
    mu_raw = numpyro.sample("mu_raw", dist.Normal(0.0, 1.0).expand([P]).to_event(1))
    mu = numpyro.deterministic("mu", MU_0 + L_SIGMA_1 @ mu_raw)

    # --- the spread. eq:omegameas for the measured set, eq:omegaassumed for the
    # rest. s is the global level, u the pattern, and u is centred so the two do
    # not overlap.
    #
    # s and b_1 are aliased in principle (known simplification 4), and sampling
    # the rotated pair (s+b_1, s-b_1) was tried here to make that an axis. It made
    # things worse: lambda_m varies across readouts and the two marginal sds are
    # unequal, so a 45-degree rotation is not the principal axis, and it cost six
    # times the divergences. Dropping the intercept from Z removes the alias
    # outright but costs a factor of three in sampling speed; see the note above Z.
    # Report the split; do not reparameterise around it.
    if flat:
        # eq:phiflat. omega_0 and not zero: a point mass would make eq:V return
        # zero, turn w^(c) into a switch on the whole cohort, and collapse every
        # location functional onto one number. See sec:flat.
        omega = numpyro.deterministic("omega", OMEGA_0)
    else:
        s = numpyro.sample("s", dist.Normal(0.0, TAU_S))
        u_raw = numpyro.sample(
            "u_raw", dist.Normal(0.0, TAU_U).expand([N_ASSUMED]).to_event(1)
        )
        # The real system has no measured widths at all: every row of
        # parameters/omega_priors.csv is a pin (material constant or packing
        # limit) with an empty omega, so eq:omegameas never applies and every
        # parameter's width comes from eq:omegaassumed. That makes lambda_m
        # identically zero and turns known simplification 4 from an approximation
        # into an exact alias -- s and b_1 then enter every scale row only through
        # their sum. The site is skipped rather than given a zero-length shape.
        if N_MEASURED:
            log_omega_measured = numpyro.sample(
                "log_omega_measured",
                dist.Normal(jnp.log(OMEGA_0[MEASURED_IDX]),
                            TAU_OMEGA_MEASURED).to_event(1),
            )
        else:
            log_omega_measured = jnp.zeros(0)
        omega = numpyro.deterministic(
            "omega", build_omega(s, u_raw, log_omega_measured)
        )

    # --- the measurement discrepancy. eq:abprior, in the coefficient basis the
    # note writes it in: a ~ N(0, sigma_a^2 I) on the columns of Z themselves.
    #
    # Sampling an orthonormalised basis a_q with Z = Q_Z R_Z was tried here to
    # improve the geometry. It is not a reparameterisation: iid on a_q is a
    # different prior from iid on a, and on this Z it halved the prior width on
    # the fraction readouts' gamma and installed corr(a_0, a_1) = -0.82, which
    # showed up as two coefficients of b missing their truth by more than 2 sd.
    # Geometry is handled by the metric in ``laplace_metric`` instead, which does
    # not touch the prior.
    a = numpyro.sample("a", dist.Normal(0.0, SIGMA_A).expand([DIM_Z]).to_event(1))
    b = numpyro.sample("b", dist.Normal(0.0, SIGMA_B).expand([DIM_Z]).to_event(1))

    # --- the mechanism discrepancy. eq:betaprior with tau_beta fixed rather than
    # sampled; see the TAU_BETA note. Tightly shrunk, because it competes with mu,
    # and centred across species, because the common shift is exactly aliased with
    # a (see BETA_SPECIES).
    beta_raw = numpyro.sample(
        "beta_raw", dist.Normal(0.0, 1.0).expand([N_BETA]).to_event(1)
    )
    beta_free = numpyro.deterministic(
        "beta_free", TAU_BETA * (beta_raw - jnp.mean(beta_raw))
    )

    # --- the forward map, then one Gaussian block per cohort. eq:obs.
    taus = tau_all(mu, omega, z, a, b, beta_free, emu.cloud, c_ref)
    for i, (cohort, tau_c) in enumerate(zip(COHORTS, taus)):
        if location_only:
            tau_c = tau_c[LOC_IDX[i]]
        numpyro.sample(
            f"T_{cohort.name}",
            dist.MultivariateNormal(tau_c, scale_tril=V_chol[i]),
            obs=None if observed is None else observed[i],
        )


def flat_map_fit(key, z, V_chol, emu, c_ref, obs_loc, loc_masks,
                 n_steps=800, lr=0.03, n_draws=200, verbose=True, omega=None,
                 fit_b=False):
    """eq:phiflat by optimisation plus a Laplace covariance, not by MCMC.

    sec:flat asks the flat fit for two things: a plug-in ``mu_hat_flat`` for
    building V_c, and a prediction of the held-out scale rows. A plug-in is a
    point estimate, and running a sampler to take the mean of its draws is using
    an expensive tool for a cheap job. The width shortfall does need uncertainty
    in mu carried through, but a Gauss-Newton covariance at the optimum supplies
    that -- it is the same matrix already computed for the mass matrix.

    Doing it this way also removes the component that was not converging. With
    NUTS the flat fit disagreed with itself across mass-matrix settings (19/19
    coverage with 13 divergences frozen, 17/19 with b_1 at z = -3.0 adapted),
    while the population fit was stable at 29/30 either way. That instability was
    buying nothing, because only the posterior mean was ever used.

    Returns a samples-like dict, so :func:`print_width_shortfall` and
    :func:`summarise_recovery` read it unchanged.
    """
    loc_idx = [jnp.array(np.flatnonzero(m)) for m in loc_masks]
    # omega is a knob only so the pinning can be tested against a known truth;
    # eq:phiflat pins it at omega_0 and that stays the default.
    omega_fixed = OMEGA_0 if omega is None else omega

    def neg_log_post(p):
        mu = MU_0 + L_SIGMA_1 @ p["mu_raw"]
        beta_free = TAU_BETA * (p["beta_raw"] - jnp.mean(p["beta_raw"]))
        taus = tau_all(mu, omega_fixed, z, p["a"], p.get("b", b_fixed), beta_free,
                       emu.cloud, c_ref)
        nll = 0.0
        for i in range(len(COHORTS)):
            r = obs_loc[i] - taus[i][loc_idx[i]]
            w = jax.scipy.linalg.solve_triangular(V_chol[i], r, lower=True)
            nll = nll + 0.5 * jnp.sum(w**2)
        return (nll
                + 0.5 * jnp.sum(p["mu_raw"] ** 2)
                + 0.5 * jnp.sum(p["a"] ** 2) / SIGMA_A**2
                + 0.5 * jnp.sum(p.get("b", b_fixed) ** 2) / SIGMA_B**2
                + 0.5 * jnp.sum(p["beta_raw"] ** 2))

    # b is pinned at zero by default, and that is a considered choice rather than
    # a simplification. On noiseless data the flat fit recovers b exactly, so the
    # parameterisation is right; with real rows it lands 3 sd from truth, because
    # 19 parameters against 25 location rows leaves six spare degrees of freedom
    # and mu, a and b trade off inside them. The only channel carrying b here is
    # the mean-median gap, which is skewness -- dropping the mean rows takes b's
    # posterior from 0.26 of its prior width to 0.80. So the flat fit cannot
    # estimate b, and a badly estimated b is not free: it is a nuisance that
    # mu_hat_flat has to trade against, and mu_hat_flat is the thing sec:flat
    # actually wants. Pinning it returns those degrees of freedom to mu.
    params = {"mu_raw": jnp.zeros(P), "a": jnp.zeros(DIM_Z),
              "beta_raw": jnp.zeros(N_BETA)}
    if fit_b:
        params["b"] = jnp.zeros(DIM_Z)
    b_fixed = jnp.zeros(DIM_Z)
    state = (tree_map(jnp.zeros_like, params), tree_map(jnp.zeros_like, params), 0)

    @jax.jit
    def step(p, st, lr_t):
        val, g = jax.value_and_grad(neg_log_post)(p)
        p, st = adam_step(p, g, st, lr_t)
        return p, st, val

    t0 = time.time()
    for i in range(n_steps):
        params, state, val = step(params, state,
                                  lr * 0.5 * (1 + np.cos(np.pi * i / n_steps)))
    if verbose:
        print(f"flat MAP in {time.time() - t0:.1f}s, "
              f"{n_steps} Adam steps, final -log post {float(val):.2f}")

    # b is in `full` whether or not it was fitted, because the Laplace block set is
    # keyed on the model's sites, not on which of them this fit chose to move.
    full = dict(params, b=params.get("b", b_fixed))
    names, dims, C = laplace_covariance(
        MU_0, OMEGA_0, z, V_chol, emu, c_ref, flat=True, rows=loc_masks, at=full
    )
    # Draw from the Laplace normal so the width shortfall carries mu's uncertainty.
    mean = np.concatenate([np.atleast_1d(np.asarray(full[nm])) for nm in names])
    L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
    draws = mean[None, :] + np.asarray(random.normal(key, (n_draws, C.shape[0]))) @ L.T

    out, off = {}, 0
    for nm, d in zip(names, dims):
        out[nm] = draws[:, off:off + d]
        off += d
    mu_raw = jnp.array(out["mu_raw"])
    out["mu"] = np.asarray(MU_0[None, :] + mu_raw @ np.asarray(L_SIGMA_1).T)
    br = out["beta_raw"]
    out["beta_free"] = TAU_BETA * (br - br.mean(axis=1, keepdims=True))
    # b is deliberately absent from the returned dict when it was pinned. Putting
    # zeros there would make the recovery table score four rows as MISS at z = nan,
    # which reads as four failed estimates rather than one modelling choice.
    if not fit_b:
        out.pop("b", None)
    out["_map"] = {nm: params[nm] for nm in params}
    return out


def print_width_shortfall(samples, observed, z, emu, truth, c_ref, n_draws=100):
    """The held-out scale rows, predicted at omega_0. sec:flat's width shortfall.

    The whole point of holding the scale rows out of the flat fit is that this
    comparison is then a prediction and not a residual. Each scale row is a log
    IQR, so ``exp(observed - predicted)`` is the factor by which the real cohort is
    wider than a population of spread omega_0 would be. Above one means omega_0 is
    too narrow at that readout, and it says so before any spread parameter has been
    estimated.

    Reported per readout, pooled over the cohorts that report it, with a posterior
    band from the flat fit so that uncertainty in mu carries through.
    """
    n = min(n_draws, len(np.asarray(samples["mu"])))
    idx = np.linspace(0, len(np.asarray(samples["mu"])) - 1, n).astype(int)

    per_readout = {}
    for t in idx:
        mu = jnp.array(np.asarray(samples["mu"])[t])
        a = jnp.array(np.asarray(samples["a"])[t])
        b = jnp.array(np.asarray(samples["b"])[t])
        beta = jnp.array(np.asarray(samples["beta_free"])[t])
        taus = tau_all(mu, OMEGA_0, z, a, b, beta, emu.cloud, c_ref)
        for i, cohort in enumerate(COHORTS):
            pred = np.asarray(taus[i])
            obs = np.asarray(observed[i])
            for k, r in cohort.scale_readouts:
                per_readout.setdefault(r, []).append(float(obs[k] - pred[k]))

    print("\nwidth shortfall on the held-out scale rows (sec:flat)")
    print("  ratio = exp(observed log IQR - predicted at omega_0); "
          "above 1 means omega_0 too narrow")
    print(f"  {'readout':<16}{'ratio':>8}{'5%':>8}{'95%':>8}{'n rows':>8}")
    n_scale = 0
    for r in range(M):
        if r not in per_readout:
            continue
        d = np.array(per_readout[r])
        rows = sum(1 for c in COHORTS for _, rr in c.scale_readouts if rr == r)
        n_scale += rows
        print(f"  {READOUT_NAMES[r]:<16}{np.exp(d.mean()):>8.2f}"
              f"{np.exp(np.percentile(d, 5)):>8.2f}"
              f"{np.exp(np.percentile(d, 95)):>8.2f}{rows:>8d}")
    print(f"  {n_scale} scale rows held out of the flat fit, "
          f"{A_ROWS_LOC} location rows fitted")
    true_ratio = float(np.exp(truth.s))
    print(f"  phi* has s = {truth.s:.3f}, so the assumed parameters are truly "
          f"{true_ratio:.2f}x wider")
    print("  than omega_0. Readouts driven by those parameters should show "
          "roughly that.")


# =============================================================================
# PART 3 -- THE FIT
# =============================================================================


def summarise_recovery(samples, truth):
    """Compare the posterior against phi*, in posterior-sd units."""
    rows = []

    def add(name, post, true_val):
        post = np.asarray(post).ravel()
        m, sd = post.mean(), post.std()
        lo, hi = np.percentile(post, [2.5, 97.5])
        rows.append(
            {
                "term": name,
                "truth": float(true_val),
                "post_mean": float(m),
                "post_sd": float(sd),
                "z": float((m - true_val) / sd) if sd > 0 else np.nan,
                "covered": bool(lo <= true_val <= hi),
            }
        )

    for j, nm in enumerate(PARAM_NAMES):
        add(f"mu[{nm}]", samples["mu"][:, j], truth.mu[j])
    # The spread terms are absent from eq:phiflat, so the same table serves both
    # fits and the flat one simply has fewer rows.
    if "s" in samples:
        add("s", samples["s"], truth.s)
    if "log_omega_measured" in samples:
        for j in range(N_MEASURED):
            add(f"log_omega_meas[{j}]", samples["log_omega_measured"][:, j],
                truth.log_omega_measured[j])
    for j in range(DIM_Z):
        add(f"a[{j}]", samples["a"][:, j], truth.a[j])
    # The flat fit pins b, so it simply has no rows for it.
    if "b" in samples:
        for j in range(DIM_Z):
            add(f"b[{j}]", samples["b"][:, j], truth.b[j])
    for j in range(N_BETA):
        add(f"beta[{SPECIES_NAMES[int(BETA_SPECIES[j])]}]",
            samples["beta_free"][:, j], truth.beta_free[j])
    if "s" in samples:
        for j in range(P):
            add(f"omega[{PARAM_NAMES[j]}]", samples["omega"][:, j], truth.omega[j])
    return rows


def lambda_per_readout(mu, omega, emu):
    """lambda_m: the share of readout m's across-patient variance coming from M.

    The note makes this the whole question for the width split. If lambda_m is
    near zero for every readout then s and b_1 enter every scale row only through
    their sum, and the prior decides the split rather than the data. To first
    order Var[x_m] = sum_jk J_mj J_mk R_jk omega_j omega_k, so lambda_m is that
    sum restricted to j, k in M over the full sum.
    """
    def readouts_of(vt):
        y_all = jnp.stack([emu.one(vt, s) for s in range(S)])[:, None, :]
        return h_all(y_all)[0, 0]  # (M,) in the baseline scenario

    J = jax.jacobian(readouts_of)(mu)  # (M, P)
    Sigma = jnp.diag(omega) @ R_CORR @ jnp.diag(omega)
    full = jnp.einsum("mj,jk,mk->m", J, Sigma, J)
    mask = jnp.zeros(P).at[MEASURED_IDX].set(1.0)
    Sm = Sigma * mask[:, None] * mask[None, :]
    meas = jnp.einsum("mj,jk,mk->m", J, Sm, J)
    return np.asarray(meas / jnp.clip(full, 1e-12, None))


def print_width_ridge(samples, truth, mu_hat, omega_hat, emu):
    """The joint posterior of (s, b_1), which is what the note asks to report.

    Reported as shrinkage against the prior rather than against each other. Each
    rotated coordinate has prior sd ``TAU_S``, so ``post sd / prior sd`` says
    directly whether the data moved that direction or the prior returned it.
    That is the variance-budget question, one row per direction.
    """
    root2 = np.sqrt(2.0)
    s = np.asarray(samples["s"]).ravel()
    b1 = np.asarray(samples["b"])[:, 0]
    xi, eta = (s + b1) / root2, (s - b1) / root2
    xi_t = float(truth.s + truth.b[0]) / root2
    eta_t = float(truth.s - truth.b[0]) / root2

    print("\nthe width ridge (known simplification 4)")
    print(f"  corr(s, b_1) in the posterior   {np.corrcoef(s, b1)[0, 1]:+.2f}")
    print(f"  {'direction':<22}{'truth':>8}{'post mean':>11}{'post sd':>9}"
          f"{'z':>7}{'sd/prior sd':>13}")
    for lab, post, tv, prior_sd in [
        ("s", s, float(truth.s), TAU_S),
        ("b_1", b1, float(truth.b[0]), SIGMA_B),
        ("xi = (s+b_1)/sqrt2", xi, xi_t, TAU_S),
        ("eta = (s-b_1)/sqrt2", eta, eta_t, TAU_S),
    ]:
        print(f"  {lab:<22}{tv:>8.3f}{post.mean():>11.3f}{post.std():>9.3f}"
              f"{(post.mean() - tv) / post.std():>7.2f}{post.std() / prior_sd:>13.2f}")
    print("  xi is the direction the data see; eta is the one the prior decides.")

    lam = lambda_per_readout(mu_hat, omega_hat, emu)
    print("\n  lambda_m, the share of each readout's variance from M (scenario 0):")
    print("    " + "  ".join(f"{nm}={v:.2f}" for nm, v in zip(READOUT_NAMES, lam)))
    print(f"    spread {lam.min():.2f} to {lam.max():.2f}. A scale row loads on s "
          f"as (1-lambda_m) and on b_1 as 1,")
    print("    so a near-constant lambda means the prior sets the split, not the data.")


def print_posterior_correlations(samples, top=12):
    """The largest posterior correlations, in the coordinates NUTS samples.

    This is the diagnosis for a slow sampler. A diagonal mass matrix rescales the
    axes but cannot rotate, so a correlation near 1 leaves a ridge running
    diagonally: the step size collapses to what the narrow direction allows while
    the long direction still has to be traversed, and the tree depth saturates.
    The condition number below is what ``dense_mass`` is buying, and the smallest
    eigenvector names the direction that was setting the step size.

    Worth reading before assuming which pair is responsible. On this toy the
    obvious suspect (s against b_1) turned out not to be it.
    """
    sites = ["mu_raw", "s", "u_raw", "log_omega_measured", "a", "b", "beta_raw"]
    cols, labels = [], []
    for nm in sites:
        if nm not in samples:
            continue
        arr = np.asarray(samples[nm])
        arr = arr[:, None] if arr.ndim == 1 else arr
        for j in range(arr.shape[1]):
            cols.append(arr[:, j])
            labels.append(nm if arr.shape[1] == 1 else f"{nm}[{j}]")
    X = np.column_stack(cols)
    Cm = np.corrcoef(X, rowvar=False)

    iu = np.triu_indices_from(Cm, k=1)
    order = np.argsort(-np.abs(Cm[iu]))
    print(f"\ntop posterior correlations ({X.shape[1]} sampling coordinates)")
    for t in order[:top]:
        i, j = iu[0][t], iu[1][t]
        print(f"  {labels[i]:<22}{labels[j]:<22}{Cm[i, j]:+.3f}")

    w, Vv = np.linalg.eigh(Cm)
    print(f"\n  condition number {w[-1] / w[0]:.1f}  "
          f"(eigenvalues {w[0]:.3f} to {w[-1]:.3f})")
    print("  narrowest direction, the one that sets the step size:")
    v = Vv[:, 0]
    for t in np.argsort(-np.abs(v))[:6]:
        print(f"    {labels[t]:<22}{v[t]:+.3f}")


def print_readout_effects(samples, truth):
    """gamma_r and kappa_r per readout, which is what the model actually applies.

    The columns of ``a`` and ``b`` are contrasts between readout attributes and
    trade against each other and against ``s``. The per-readout values they
    combine to are a different question, and the one that matters for the fit.

    ``gamma_r`` here is the intercept at the pivot ``c_r``, not at zero. The shift
    the map applies at a readout level ``x`` is ``gamma_r + c_r (1 - kappa_r)``;
    see :func:`reference_levels`.
    """
    a = np.asarray(samples["a"])
    b = np.asarray(samples.get("b", np.zeros_like(a)))
    Zn = np.asarray(Z)
    gam, kap = a @ Zn.T, np.exp(b @ Zn.T)
    gam_t = np.asarray(Zn @ np.asarray(truth.a))
    kap_t = np.exp(np.asarray(Zn @ np.asarray(truth.b)))

    print("\nper-readout discrepancy (what eq:disc applies)")
    print(f"  {'readout':<16}{'gamma truth':>12}{'gamma post':>12}{'z':>7}"
          f"{'kappa truth':>13}{'kappa post':>12}{'z':>7}")
    for r, nm in enumerate(READOUT_NAMES):
        g, kp = gam[:, r], kap[:, r]
        print(f"  {nm:<16}{gam_t[r]:>12.3f}{g.mean():>12.3f}"
              f"{(g.mean() - gam_t[r]) / g.std():>7.2f}"
              f"{kap_t[r]:>13.3f}{kp.mean():>12.3f}"
              f"{(kp.mean() - kap_t[r]) / kp.std():>7.2f}")


def phi0_sensitivity(key, build_V_fn, z, emu, c_ref, observed, n_probe=3):
    """Does the answer depend on where V_c was built? The draft asks; nothing did.

    eq:V freezes V_c at a plug-in, which is only defensible if the posterior is
    insensitive to that choice. This rebuilds V_c at ``n_probe`` points drawn from
    the stage-1 prior and reports how far the Laplace posterior moves, in units of
    its own sd. Below about 0.1 the plug-in does not matter and the flat fit is not
    earning its cost; above about 0.5 the frozen-V_c approximation is doing real
    work and the plug-in has to be chosen carefully.

    The Laplace posterior stands in for the full fit here, so the comparison costs
    one V_c build and one Jacobian per probe rather than one MCMC run per probe.
    It is the right proxy because it is exactly the quadratic the sampler explores.
    """
    print(f"\nphi_0 sensitivity: rebuilding V_c at {n_probe} draws from Sigma_1")
    k_probe, k_base = random.split(key)

    _, V_chol_base = build_V_fn(MU_0, OMEGA_0, k_base, "mu_0", quiet=True)
    names, dims, C0 = laplace_covariance(MU_0, OMEGA_0, z, V_chol_base, emu, c_ref)
    sd0 = np.sqrt(np.diag(C0))
    labels = [nm if d == 1 else f"{nm}[{j}]"
              for nm, d in zip(names, dims) for j in range(d)]

    print(f"  {'probe':<8}{'|dmu_0| max':>13}{'max shift/sd':>15}"
          f"{'mean shift/sd':>15}{'worst coordinate':>22}")
    for t, kk in enumerate(random.split(k_probe, n_probe)):
        k_draw, k_build = random.split(kk)
        mu_probe = MU_0 + L_SIGMA_1 @ random.normal(k_draw, (P,))
        _, V_chol_p = build_V_fn(mu_probe, OMEGA_0, k_build, f"probe{t}", quiet=True)
        _, _, Cp = laplace_covariance(MU_0, OMEGA_0, z, V_chol_p, emu, c_ref)
        # The mean of a Gauss-Newton posterior at a fixed linearisation point moves
        # with V_c only through the weighting, so compare the sds: a change in the
        # posterior width IS a change in what the data are taken to say.
        shift = np.abs(np.sqrt(np.diag(Cp)) - sd0) / np.clip(sd0, 1e-30, None)
        j = int(np.argmax(shift))
        print(f"  {t:<8}{float(np.abs(mu_probe - MU_0).max()):>13.3f}"
              f"{shift.max():>15.3f}{shift.mean():>15.3f}{labels[j]:>22}")
    print("  shift is the relative change in each coordinate's posterior sd.")
    print("  If these are small the plug-in does not matter and sec:flat's "
          "refinement is optional.")


def print_table(rows):
    hdr = f"{'term':<26}{'truth':>9}{'post mean':>11}{'post sd':>9}{'z':>7}  cover"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['term']:<26}{r['truth']:>9.3f}{r['post_mean']:>11.3f}"
            f"{r['post_sd']:>9.3f}{r['z']:>7.2f}  {' ok ' if r['covered'] else 'MISS'}"
        )
    n_cov = sum(r["covered"] for r in rows)
    zs = np.array([r["z"] for r in rows])
    print("-" * len(hdr))
    print(f"95% coverage: {n_cov}/{len(rows)} = {n_cov / len(rows):.0%}")
    print(f"|z| mean {np.abs(zs).mean():.2f}, max {np.abs(zs).max():.2f}")


def main():
    ap = argparse.ArgumentParser(description="Toy validation of the population model.")
    ap.add_argument("--size", choices=("small", "medium", "full"), default="small",
                    help="small: P=8 Q=3 A=42. full: the real model's size, "
                         "P=271 Q=179 S=5 A=88. medium: full's code path, reduced.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--chains", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="short run, for smoke tests")
    ap.add_argument("--no-emulator-error", action="store_true",
                    help="drop E_c from V_c, to see what it was worth")
    ap.add_argument("--big-emulator", action="store_true",
                    help="128x128x128 on 16k design points; slower, more accurate")
    ap.add_argument("--diag-mass", action="store_true",
                    help="diagonal mass matrix, to see what the rotation alone buys")
    # At the QSP sizes the trees saturate: NUTS spends 1023 gradients an iteration
    # and never finds a U-turn, because the posterior's longest direction is about
    # a hundred times its shortest even after mass adaptation. That is what a row
    # space of 88 against 550 parameters looks like -- most directions return the
    # prior at unit scale, a few are data-constrained and narrow. Capping the depth
    # trades per-iteration mixing for eight times the iterations, which is the
    # right trade when the trees are saturating rather than terminating.
    ap.add_argument("--max-tree-depth", type=int, default=10,
                    help="NUTS max tree depth; 7 caps at 127 gradients per "
                         "iteration instead of 1023")
    # The Laplace metric is computed, not adapted, so mass adaptation is turned
    # off with it: the point is that at these dimensions we can evaluate the
    # matrix more accurately than warmup can estimate it.
    ap.add_argument("--laplace-mass", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="seed NUTS with G^-1 from the Gauss-Newton information "
                         "at the plug-in and disable mass adaptation")
    ap.add_argument("--cache-dir", default=".emulator-cache", metavar="DIR",
                    help="where to cache the emulator and V_c; both are pure "
                         "functions of the design and the seed")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the cache and rebuild from scratch")
    ap.add_argument("--no-progress", action="store_true",
                    help="silence the NUTS progress bar; tqdm's carriage returns "
                         "make a batch-scheduler log unreadable")
    ap.add_argument("--phi0-sweep", type=int, default=0, metavar="N",
                    help="rebuild V_c at N draws from Sigma_1 and report how far "
                         "the posterior moves; 0 skips. This is the test of "
                         "whether the plug-in matters at all.")
    ap.add_argument("--skip-flat", action="store_true",
                    help="skip the flat fit and take phi_0 = (mu_0, omega_0)")
    args = ap.parse_args()
    if args.quick:
        args.warmup, args.samples, args.chains = 100, 100, 1
    if args.size != "small":
        install_problem(qsp_problem(args.size, seed=args.seed))

    key = random.PRNGKey(args.seed)
    (k_emu, k_data, k_boot, k_cloud, k_mcmc,
     k_flat, k_flat_cloud, k_boot2, k_ref) = random.split(key, 9)

    print("=" * 72)
    print(f"PART 1  inputs  [problem: {PROB.name}]")
    print("=" * 72)
    print(f"P={P} parameters, Q={Q} species, M={M} readouts, "
          f"S={S} scenarios, C={C} cohorts")
    print(f"A={A_ROWS} rows ({A_ROWS_LOC} location, {A_ROWS - A_ROWS_LOC} scale), "
          f"N={N_CLOUD} simulated patients, |M_set|={N_MEASURED} measured widths")
    print(f"cohort sizes n_c = {[c.n for c in COHORTS]}")
    n_synth = sum(c.K for c in COHORTS if c.synthesized)
    if n_synth:
        print(f"{n_synth} of {A_ROWS} rows are synthesized rather than reported, "
              f"and carry an inflated U in eq:Vsplit")
    if N_MEASURED == 0:
        print("no parameter has a measured width, so lambda_m = 0 identically and "
              "s and b_1")
        print("  enter every scale row only through their sum "
              "(known simplification 4, exactly)")

    stamp(f"problem {PROB.name} installed: P={P} Q={Q} M={M} S={S} A={A_ROWS}")
    support, z_cond, z_rank = z_conditioning()
    print(f"Z design: readouts per column {list(support)}, cond(Z) = {z_cond:.1f}, "
          f"rank {z_rank} of M={M}")
    thin = [j for j, n in enumerate(support) if n < 2]
    if thin:
        print(f"  columns {thin} are carried by one readout each, so their "
              f"coefficients are near-aliased with the intercept")
    if z_rank >= M:
        print("  rank(Z) = M, so gamma spans readout space and beta keeps no "
              "location signal of its own")
    print()

    # One prior sd of s, not two: the design has to cover a plausibly wide
    # population, and every extra factor of e^{tau_s} costs emulator accuracy
    # where the fit actually reads.
    omega_pool = OMEGA_0 * jnp.exp(TAU_S)
    stamp("measuring readout sensitivity for the emulator loss weights")
    cache_dir = None if args.no_cache else Path(args.cache_dir)
    w_species = species_loss_weights(k_ref, MU_0, OMEGA_0)
    stamp(f"training the emulator ({EMU_POOL:,} design points x {S} scenarios)")
    emu = train_emulator(
        k_emu, MU_0, SIGMA_1, omega_pool, species_weights=w_species,
        cache_dir=cache_dir,
        hidden=(128, 128, 128) if args.big_emulator else EMU_HIDDEN,
        n_pool=16_384 if args.big_emulator else EMU_POOL,
        n_steps=8_000 if args.big_emulator else EMU_STEPS,
        seed=args.seed,
    )

    # The pivot of eq:disc's affine map. A fixed input: computed once from the
    # emulator at the plug-in and then frozen, so it is available in the real
    # setting where the simulator is too expensive to call inside the fit.
    c_ref = reference_levels(MU_0, OMEGA_0, k_ref, emu.cloud)
    cr = np.asarray(c_ref)
    reported = np.zeros((C, M), dtype=bool)
    for i, cohort in enumerate(COHORTS):
        reported[i, list(cohort.readouts)] = True
    print(f"\nkappa pivot c_{{r,c}} at the plug-in, {C} cohorts x {M} readouts")
    print(f"  {'readout':<18}{'cohorts':>9}{'lo':>8}{'hi':>8}{'spread':>8}")
    for r in range(M):
        if not reported[:, r].any():
            continue
        v = cr[reported[:, r], r]
        print(f"  {READOUT_NAMES[r]:<18}{reported[:, r].sum():>9d}"
              f"{v.min():>8.2f}{v.max():>8.2f}{v.max() - v.min():>8.2f}")
    print("  spread is the range one global pivot would have had to straddle")

    # V_c depends on the emulator, so its cache key has to include one. Hash the
    # trained weights themselves rather than the config that produced them: that
    # way a cached V_c can never be paired with a different surrogate, however the
    # emulator came to exist.
    emu_key = _cache_key("emu-weights",
                         *[np.asarray(w) for layer in emu.params for w in layer])

    stamp("emulator trained; generating the toy data from the true ODE")
    truth = make_ground_truth()
    observed = generate_data(truth, k_data, c_ref)
    stamp(f"data generated ({A_ROWS} rows); building V_c at the prior centre")
    print(f"data: {A_ROWS} rows drawn from a true-ODE cloud of {N_TRUTH:,}")

    def build_V(phi_mu, phi_omega, key, label, quiet=False):
        """eq:V, eq:Ec and eq:Vsplit at one plug-in, with the eq:Ec report.

        Cached on the plug-in and the emulator. eq:Ec is the dominant cost of the
        whole script at QSP scale -- L_EMU clouds evaluated through BOTH the
        emulator and the true ODE, which is 225,000 solves per build, done at
        least twice per run. Like the emulator it depends on nothing that varies
        between runs at a fixed seed.
        """
        t0 = time.time()
        kb, kec = random.split(key)
        vk = _cache_key("V", PROB.name, args.seed, emu_key, B_BOOT, L_EMU,
                        N_CLOUD, N_TRUTH, phi_mu, phi_omega, c_ref,
                        tuple(c.name for c in COHORTS))
        vpath = _cache_path(cache_dir, "vc", vk)
        if vpath is not None and vpath.exists():
            d = np.load(vpath)
            n = int(d["n_blocks"])
            v_boot = [d[f"boot{i}"] for i in range(n)]
            E_blocks = [d[f"E{i}"] for i in range(n)]
            E_means = [d[f"Em{i}"] for i in range(n)]
            if not quiet:
                print(f"V_c at phi_0 = ({label}, omega_0): loaded from cache "
                      f"{vpath.name}")
        else:
            v_boot = bootstrap_V(phi_mu, phi_omega, kb, emu, c_ref)
            E_blocks, E_means = emulator_E(phi_mu, phi_omega, kec, emu, c_ref)
            if vpath is not None:
                vpath.parent.mkdir(parents=True, exist_ok=True)
                blob = {"n_blocks": len(v_boot)}
                for i in range(len(v_boot)):
                    blob[f"boot{i}"] = np.asarray(v_boot[i])
                    blob[f"E{i}"] = np.asarray(E_blocks[i])
                    blob[f"Em{i}"] = np.asarray(E_means[i])
                np.savez_compressed(vpath, **blob)
        # eq:Vsplit's other half. A synthesized row's statistic was reconstructed
        # from a figure or a summary rather than reported, so its uncertainty is
        # wider than the sampling uncertainty a bootstrap at n_c would give. That
        # is exactly a reported U: it overrides the scale while the correlations
        # stay from the bootstrap, which is the split eq:Vsplit is written for and
        # which nothing in this script exercised until the real target table
        # turned out to be more than half synthesized.
        reported_U = [
            SYNTH_INFLATE * np.sqrt(np.diag(np.asarray(vb))) if c.synthesized
            else None
            for c, vb in zip(COHORTS, v_boot)
        ]
        V, V_chol = assemble_V(
            v_boot, reported_U=reported_U,
            E_blocks=None if args.no_emulator_error else E_blocks
        )
        if quiet:
            return V, V_chol
        print(f"V_c built at phi_0 = ({label}, omega_0) in {time.time() - t0:.1f}s "
              f"({B_BOOT} bootstrap replicates, {L_EMU} clouds for E_c)")
        if args.no_emulator_error:
            print("  E_c DROPPED (--no-emulator-error)")

        # eq:Ec says to report d-bar beside E_c. E_c is a covariance, so it is
        # blind to the emulator's systematic offset; |d-bar| / sd(V_c) is the part
        # of the emulator error that no entry of V_c is carrying, and that gamma_r
        # is supposed to absorb. Well above 1 means the fit will be biased however
        # careful eq:Vsplit was.
        print(f"  {'cohort':>8}{'K_c':>5}{'n_c':>6}{'max|corr|':>11}"
              f"{'tr(E)/tr(Vb)':>14}{'max|dbar|/sd':>14}")
        for i, cohort in enumerate(COHORTS):
            Vn = np.asarray(V[i])
            sd = np.sqrt(np.diag(Vn))
            rho = Vn / np.outer(sd, sd)
            off = rho[~np.eye(cohort.K, dtype=bool)]
            peak = abs(off[np.argmax(np.abs(off))]) if off.size else 0.0
            share = np.trace(np.asarray(E_blocks[i])) / np.trace(np.asarray(v_boot[i]))
            bias = np.max(np.abs(np.asarray(E_means[i])) / sd)
            print(f"  {cohort.name:>8}{cohort.K:>5}{cohort.n:>6}{peak:>11.2f}"
                  f"{share:>14.3f}{bias:>14.2f}")
            # Two rows of one cohort that are deterministic functions of each other
            # give a singular block. The ridge in assemble_V keeps the Cholesky
            # from failing, which is worse than failing: it turns the direction into
            # a ~1e10 eigenvalue of V^-1 and the sampler stalls at a 1e-4 step size
            # a long way from here, looking like a geometry problem.
            if peak > 0.99:
                print(f"      WARNING: {cohort.name} is effectively singular "
                      f"(max|corr| = {peak:.4f}).")
                print("      Two of its readouts are the same functional of the "
                      "same patients. Split")
                print("      them across cohorts; do not rely on the ridge.")
            if bias > 1.0:
                print(f"      NOTE: |dbar| reaches {bias:.2f} sd here, so the "
                      f"emulator's systematic")
                print("      offset exceeds this cohort's sampling noise and no "
                      "entry of V_c carries it.")
        return V, V_chol

    V, V_chol = build_V(MU_0, OMEGA_0, k_boot, "mu_0")

    z = draw_cloud_z(k_cloud, N_CLOUD)

    stamp("V_c built; running the projection tests")
    t0 = time.time()
    J_rows = row_jacobians(MU_0, OMEGA_0, z, V_chol, emu, c_ref)
    print_projection_report(J_rows)
    print(f"  ({time.time() - t0:.1f}s)")
    print_pivot_offsets(MU_0, OMEGA_0, z, V, emu, c_ref)
    stamp("conditioning report: Jacobian over every sampling coordinate")
    conditioning_report(MU_0, OMEGA_0, z, V_chol, emu, c_ref)
    stamp("conditioning report done")

    if args.phi0_sweep:
        phi0_sensitivity(k_boot2, build_V, z, emu, c_ref, observed,
                         n_probe=args.phi0_sweep)

    # -------------------------------------------------------------------------
    # PART 2b -- the flat fit. sec:flat.
    # -------------------------------------------------------------------------
    if not args.skip_flat:
        print()
        print("=" * 72)
        print("PART 2b  the flat fit (eq:phiflat)")
        print("=" * 72)
        print(f"omega pinned at omega_0, {A_ROWS_LOC} location rows fitted, "
              f"{A_ROWS - A_ROWS_LOC} scale rows held out")
        print(f"N = {N_CLOUD_FLAT} rather than {N_CLOUD}: only the location "
              f"statistics have to resolve")

        z_flat = draw_cloud_z(k_flat_cloud, N_CLOUD_FLAT)
        loc_masks = [c.location_mask for c in COHORTS]
        _, V_loc_chol = subset_V(V, loc_masks)
        obs_loc = [
            o[jnp.array(np.flatnonzero(m))] for o, m in zip(observed, loc_masks)
        ]

        stamp("flat fit: MAP by Adam, then a Gauss-Newton covariance")
        flat_samples = flat_map_fit(k_flat, z_flat, V_loc_chol, emu, c_ref,
                                    obs_loc, loc_masks)

        print("\nflat recovery against phi* (mu and the discrepancy only)")
        print("  Laplace, not MCMC: the interval is the Gauss-Newton curvature at "
              "the MAP,")
        print("  so it is a quadratic approximation and not a posterior sample.")
        print_table(summarise_recovery(flat_samples, truth))
        print_width_shortfall(flat_samples, observed, z_flat, emu, truth, c_ref)

        # sec:flat: "It supplies the plug-in. Take phi_0 = (mu_hat_flat, omega_0)."
        # The MAP itself, not the mean of the Laplace draws: the draws exist only
        # to carry mu's uncertainty into the width shortfall, and their mean is the
        # MAP plus Monte Carlo noise.
        mu_hat_flat = MU_0 + L_SIGMA_1 @ flat_samples["_map"]["mu_raw"]
        drift = np.abs(np.asarray(mu_hat_flat - MU_0))
        print(f"\nplug-in moves from mu_0 by up to {drift.max():.3f} in log units "
              f"(mean {drift.mean():.3f})")
        stamp("rebuilding V_c at the flat plug-in")
        print("rebuilding V_c at the flat plug-in, which is what sec:flat asks for")
        V, V_chol = build_V(mu_hat_flat, OMEGA_0, k_boot2, "mu_hat_flat")

    print()
    print("=" * 72)
    print("PART 3  fit")
    print("=" * 72)
    # dense_mass is not optional here. s and b_1 are aliased (the note's known
    # simplification 4), so the posterior has a long narrow ridge. With a
    # diagonal mass matrix NUTS saturates at 1023 leapfrog steps per iteration
    # and a step size near 3e-3; a dense mass matrix learns the ridge and the
    # trajectories collapse. The sampler cost here is a direct readout of the
    # aliasing, which is worth knowing before the real fit is attempted.
    stamp("computing the Laplace metric for the population fit")
    pop_mass, pop_adapt = None, True
    if args.laplace_mass and not args.diag_mass:
        pop_mass = laplace_inverse_mass(MU_0, OMEGA_0, z, V_chol, emu, c_ref)
        dim_pop = int(next(iter(pop_mass.values())).shape[0])
        pop_adapt = not should_fix_mass(dim_pop, args.warmup)
        print(f"NUTS seeded with the Laplace metric: {dim_pop} coordinates, "
              f"{args.warmup} warmup draws -> mass adaptation "
              f"{'ON (seeded)' if pop_adapt else 'OFF (metric frozen)'}")
    kernel = NUTS(population_model, init_strategy=init_to_median,
                  target_accept_prob=0.85, dense_mass=not args.diag_mass,
                  max_tree_depth=args.max_tree_depth,
                  inverse_mass_matrix=pop_mass,
                  adapt_mass_matrix=pop_adapt)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, progress_bar=not args.no_progress)
    stamp(f"starting NUTS: {args.warmup} warmup + {args.samples} samples "
          f"x {args.chains} chains")
    t0 = time.time()
    mcmc.run(k_mcmc, z=z, V_chol=V_chol, emu=emu, c_ref=c_ref, observed=observed)
    stamp("NUTS finished")
    print(f"\nNUTS done in {time.time() - t0:.1f}s")
    mcmc.print_summary(exclude_deterministic=True)

    samples = mcmc.get_samples()
    print()
    print("=" * 72)
    print("recovery against phi*")
    print("=" * 72)
    print_table(summarise_recovery(samples, truth))
    print_readout_effects(samples, truth)
    print_posterior_correlations(samples)
    print_width_ridge(
        samples, truth,
        jnp.array(np.asarray(samples["mu"]).mean(0)),
        jnp.array(np.asarray(samples["omega"]).mean(0)),
        emu,
    )


if __name__ == "__main__":
    main()

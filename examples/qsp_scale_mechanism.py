"""A QSP-scale synthetic mechanism for ``toy_population_fit.py``.

The three-species toy proved the statistics; it cannot say anything about what
happens when 271 parameters meet 100 rows. This module builds a mechanism at the
size of the real PDAC model so that question can be asked:

    species           179    cpp/qsp/ode/QSP_enum.h, QSPSpeciesEnum
    parameters        271    parameters/pdac_priors.csv (268 lognormal, 3 beta)
    scenarios           5    baseline, progression, gvax, +nivo, +urelumab
    target rows      ~100    calibration_targets/vpop_marginal_targets.csv,
                             49 observables carrying a median and an IQR
    cohort sizes    6..702   n_published in the same file

It is not the PDAC model. It is a synthetic sparse reaction network with the
same *shape*: a slow logistic tumour that everything else hangs off, a cascade of
fast cell populations, a still faster soluble layer, parameters shared across
several species rather than one per species, and regulation through saturating
edges. What has to be right is the size, the sparsity, the timescale spread, and
the readout kinds, because those are what set the geometry the sampler sees.

Why a generated network and not a linearisation
-----------------------------------------------
A linear surrogate at this size would be identified in closed form and would say
nothing about the emulator, which is the part that has to survive 271 input
dimensions. The network keeps the nonlinearity and the sparsity so the emulator
has a real job.

Design of the state equations
-----------------------------
Every species but the tumour relaxes toward a gain times its parent, modulated by
its regulators::

    dx_j/dt = k_j (g_j x_{p(j)} M_j - x_j),      M_j = prod over edges into j

with ``M_j`` in (0, 1]. Writing it as a gain rather than a production rate is
what keeps the cascade well scaled: the steady state is ``g_j M_j`` times the
parent, so with a gain median near ``1 / E[M]`` every species sits at order one
however deep in the cascade it is, and the fixed-step solve stays inside the
clamp. A production-rate parameterisation makes the deep species vanish and the
emulator then has to fit a floor.

The tumour is logistic and slow, with its growth modulated by the same ``M``
machinery, so inhibitory edges from the effector species are tumour control and
the treatment scenarios have somewhere to act.
"""
from __future__ import annotations

from dataclasses import dataclass

import diffrax
import jax
import jax.numpy as jnp
import numpy as np

# Fixed-step Tsit5, as in the toy: a theta-dependent step count would make the
# statistics jump. dt = T_OBS / N_STEPS = 0.15 against a fastest rate near 3
# leaves k dt = 0.45, comfortably inside the stability region.
T_OBS = 30.0
N_STEPS = 200
Y_MAX = 50.0
Y_FLOOR = 1e-9

# Rate medians are log-uniform over this band, which is the timescale spread. The
# top of the band equilibrates in under a time unit and the bottom is still
# moving at T_OBS, so the readouts see both a relaxed layer and a transient one.
RATE_LO, RATE_HI = 0.03, 1.20
TUMOUR_RATE = 0.055  # slow on purpose: the tumour must not be at equilibrium
TUMOUR_CAP = 8.0
GAIN_MEDIAN = 2.0  # about 1 / E[M], so the cascade neither grows nor dies out
EDGE_MULT = 1.8  # edges per species
M_FLOOR = 5e-3  # smooth floor on the regulator product; see make_rhs


@dataclass(frozen=True)
class Network:
    """The generated structure. Everything here is fixed, not inferred."""

    Q: int
    P: int
    parent: np.ndarray  # (Q,) parent species, -1 for the tumour root
    gain_idx: np.ndarray  # (Q,) which gain parameter each species uses
    dec_idx: np.ndarray  # (Q,) which rate parameter each species uses
    src: np.ndarray  # (E,) regulator species
    tgt: np.ndarray  # (E,) regulated species
    ec_idx: np.ndarray  # (E,) which EC50 parameter each edge uses
    sign: np.ndarray  # (E,) +1 activating, -1 inhibiting
    n_gain: int
    n_dec: int
    n_ec: int
    classes: dict  # name -> (lo, hi) species index range
    theta_median: np.ndarray  # (P,) natural-scale medians
    scenario_mult: np.ndarray  # (S, P) treatment as a parameter perturbation
    param_names: list
    species_names: list


def _class_layout(Q):
    """Species index ranges, in the proportions the PDAC model has.

    Roughly: a small tumour compartment, a large immune compartment split into
    the subsets the targets actually report, a stromal compartment, and a soluble
    layer that is about 45% of the state vector.
    """
    frac = [
        ("tumour", 0.034),
        ("cd8", 0.045),
        ("cd4", 0.045),
        ("treg", 0.034),
        ("texh", 0.034),
        ("nk", 0.022),
        ("dc", 0.045),
        ("m1", 0.034),
        ("m2", 0.034),
        ("mdsc", 0.034),
        ("bcell", 0.034),
        ("icaf", 0.045),
        ("mycaf", 0.045),
        ("ecm", 0.056),
        ("soluble", 0.489),
    ]
    out, lo = {}, 0
    for i, (nm, f) in enumerate(frac):
        hi = Q if i == len(frac) - 1 else min(Q, lo + max(1, int(round(f * Q))))
        out[nm] = (lo, hi)
        lo = hi
    return out


CELL_CLASSES = ("tumour", "cd8", "cd4", "treg", "texh", "nk", "dc", "m1", "m2",
                "mdsc", "bcell", "icaf", "mycaf", "ecm")
EFFECTOR_CLASSES = ("cd8", "nk")
SUPPRESSOR_CLASSES = ("treg", "m2", "mdsc")


def build_network(Q, P, S, seed=0):
    """Generate the structure once, from a fixed seed. Reproducible, not fitted.

    The parameter budget is split three ways: gains, rates, and EC50s. Parameters
    are *shared* across species -- 179 species draw their rate from a pool of
    about 90 -- which is the property that makes the real model's parameter count
    smaller than its species count and which a one-parameter-per-species toy
    would not reproduce.
    """
    rng = np.random.default_rng(seed)
    classes = _class_layout(Q)
    cell_hi = classes["ecm"][1]  # everything below this index is a cell

    n_gain = int(round(P * 0.33))
    n_dec = int(round(P * 0.33))
    n_ec = P - n_gain - n_dec

    gain_idx = rng.integers(0, n_gain, size=Q)
    dec_idx = rng.integers(0, n_dec, size=Q) + n_gain

    # Parents: a cascade rooted at the tumour. A soluble species is produced by a
    # cell, never by another soluble, which is what keeps the soluble layer a leaf
    # set rather than a second cascade.
    parent = np.full(Q, -1, dtype=int)
    for j in range(1, Q):
        hi = cell_hi if j >= cell_hi else j
        parent[j] = rng.integers(0, min(hi, j) if j < cell_hi else hi)

    # Regulatory edges. Mostly feed-forward from lower indices, with a fifth
    # running backwards so the network has feedback loops rather than being a DAG.
    n_edge = int(round(EDGE_MULT * Q))
    src = rng.integers(0, Q, size=n_edge)
    tgt = rng.integers(1, Q, size=n_edge)
    forward = src < tgt
    flip = (~forward) & (rng.random(n_edge) > 0.20)
    src[flip], tgt[flip] = tgt[flip], src[flip]
    ec_idx = rng.integers(0, n_ec, size=n_edge) + n_gain + n_dec
    sign = np.where(rng.random(n_edge) < 0.55, 1, -1)

    # The canonical immune cascade, wired explicitly rather than left to the random
    # edges. Without it the arms do not separate: gvax raises the DC gains, the
    # random network happens to give DC no path to the tumour, and the gvax arm
    # returns the baseline tumour density to three decimals. Treatment has to
    # propagate for a fold-change readout to carry any information at all.
    cascade = [
        ("dc", "cd8", +1), ("dc", "cd4", +1), ("cd4", "cd8", +1),
        ("treg", "cd8", -1), ("m2", "cd8", -1), ("mdsc", "cd8", -1),
        ("cd8", "texh", +1), ("tumour", "treg", +1), ("tumour", "m2", +1),
        ("tumour", "mdsc", +1), ("tumour", "icaf", +1), ("icaf", "ecm", +1),
        ("mycaf", "ecm", +1), ("tumour", "mycaf", +1),
        # Tumour control: the effectors suppress the tumour's capacity, the
        # suppressors relieve it.
        ("cd8", "tumour", -1), ("nk", "tumour", -1),
        ("treg", "tumour", +1), ("m2", "tumour", +1), ("mdsc", "tumour", +1),
    ]
    extra_s, extra_t, extra_e, extra_sg = [], [], [], []
    for a, b, sg in cascade:
        alo, ahi = classes[a]
        blo, bhi = classes[b]
        n_pair = min(3, ahi - alo, bhi - blo)
        for u, v in zip(rng.choice(np.arange(alo, ahi), n_pair, replace=False),
                        rng.choice(np.arange(blo, bhi), n_pair, replace=False)):
            extra_s.append(int(u))
            extra_t.append(int(v))
            extra_e.append(int(rng.integers(0, n_ec) + n_gain + n_dec))
            extra_sg.append(sg)
    src = np.concatenate([src, np.array(extra_s, dtype=int)])
    tgt = np.concatenate([tgt, np.array(extra_t, dtype=int)])
    ec_idx = np.concatenate([ec_idx, np.array(extra_e, dtype=int)])
    sign = np.concatenate([sign, np.array(extra_sg, dtype=int)])

    # Parameter medians. Gains around 2, rates log-uniform across the band, EC50s
    # around the order-one level the species sit at.
    theta = np.empty(P)
    theta[:n_gain] = GAIN_MEDIAN * np.exp(rng.normal(0.0, 0.25, n_gain))
    theta[n_gain:n_gain + n_dec] = np.exp(
        rng.uniform(np.log(RATE_LO), np.log(RATE_HI), n_dec)
    )
    theta[n_gain + n_dec:] = np.exp(rng.normal(0.0, 0.5, n_ec))

    # The tumour's own rate is pinned slow, whatever the pool drew, so that it is
    # still moving at T_OBS while the cell and soluble layers have relaxed. Its
    # gain is the capacity scale, pinned at 1 so the capacity is TUMOUR_CAP times
    # the regulator product. Both are shared parameters, as every other parameter
    # here is, so pinning the median does not pin them in the fit.
    theta[gain_idx[0]] = 1.0
    theta[dec_idx[0]] = TUMOUR_RATE

    # Scenarios as parameter perturbations, which is how a treatment enters a QSP
    # model: it moves a rate or an EC50, not a state. Cumulative, so the arms
    # nest the way the trial arms do.
    scenario_mult = np.ones((S, P))
    dc_lo, dc_hi = classes["dc"]
    cd8_lo, cd8_hi = classes["cd8"]
    texh_lo, texh_hi = classes["texh"]
    if S > 1:  # clinical progression: a faster tumour and a weaker effector arm
        scenario_mult[1, dec_idx[0]] *= 1.45
        scenario_mult[1, np.unique(gain_idx[cd8_lo:cd8_hi])] *= 0.75
    if S > 2:  # gvax: priming, so the DC gains go up
        scenario_mult[2, np.unique(gain_idx[dc_lo:dc_hi])] *= 1.6
    if S > 3:  # +nivo: checkpoint release, so exhaustion falls and CD8 rises
        scenario_mult[3] = scenario_mult[2]
        scenario_mult[3, np.unique(gain_idx[texh_lo:texh_hi])] *= 0.55
        scenario_mult[3, np.unique(gain_idx[cd8_lo:cd8_hi])] *= 1.35
    if S > 4:  # +urelumab: CD137 agonism on top
        scenario_mult[4] = scenario_mult[3]
        scenario_mult[4, np.unique(dec_idx[cd8_lo:cd8_hi])] *= 0.70

    param_names = (
        [f"gain{i:03d}" for i in range(n_gain)]
        + [f"rate{i:03d}" for i in range(n_dec)]
        + [f"ec50_{i:03d}" for i in range(n_ec)]
    )
    species_names = []
    for nm, (lo, hi) in classes.items():
        species_names += [f"{nm}{i:03d}" for i in range(hi - lo)]

    return Network(
        Q=Q, P=P, parent=parent, gain_idx=gain_idx, dec_idx=dec_idx,
        src=src, tgt=tgt, ec_idx=ec_idx, sign=sign,
        n_gain=n_gain, n_dec=n_dec, n_ec=n_ec, classes=classes,
        theta_median=theta, scenario_mult=scenario_mult,
        param_names=param_names, species_names=species_names,
    )


def make_g(net):
    """Return ``g_one(vartheta, scenario) -> species at T_OBS``, shape ``(Q,)``.

    The right-hand side is a segment sum over edges rather than a loop over
    species, so the cost is set by the edge count and the whole thing stays inside
    one XLA kernel. The state is clamped inside the RHS, which bounds every
    derivative and makes the fixed-step solve unconditionally stable, exactly as
    in the small toy.
    """
    Q = net.Q
    parent = jnp.array(net.parent)
    has_parent = jnp.array(net.parent >= 0)
    parent_safe = jnp.array(np.clip(net.parent, 0, None))
    gain_idx = jnp.array(net.gain_idx)
    dec_idx = jnp.array(net.dec_idx)
    src = jnp.array(net.src)
    tgt = jnp.array(net.tgt)
    ec_idx = jnp.array(net.ec_idx)
    activating = jnp.array(net.sign > 0)
    scenario_mult = jnp.array(net.scenario_mult)
    y0 = jnp.full(Q, 0.5).at[0].set(0.35)

    def rhs(t, y, args):
        theta = args
        yc = jnp.clip(y, Y_FLOOR, Y_MAX)
        gain = theta[gain_idx]
        dec = theta[dec_idx]
        ec = theta[ec_idx]

        xs = yc[src]
        m = jnp.where(activating, xs / (ec + xs), ec / (ec + xs))
        # Products over incoming edges, done as a sum of logs so the reduction is
        # a segment_sum. m is in (0, 1) by construction, so the log is safe.
        logM = jax.ops.segment_sum(
            jnp.log(jnp.clip(m, 1e-6, 1.0)), tgt, num_segments=Q
        )
        # A SMOOTH floor on the regulator product, not a clip. A species with
        # several inhibitory edges has M near 1e-9, its steady state collapses onto
        # the state clamp, and the emulator is then asked to fit a species that is
        # at a hard floor for some theta and not for others. That discontinuity is
        # the documented failure mode in the small toy, where it took held-out RMSE
        # on log tumour from 0.10 to 1.71. Nine of 179 species did it here before
        # this line. An affine floor keeps M in [M_FLOOR, 1] and is differentiable
        # everywhere, so nothing collapses and no kink is introduced in its place.
        M = M_FLOOR + (1.0 - M_FLOOR) * jnp.exp(logM)

        inflow = jnp.where(has_parent, yc[parent_safe], 1.0)
        dx = dec * (gain * inflow * M - yc)

        # The tumour is logistic and slow, and M[0] scales its CAPACITY, not its
        # growth rate. On the rate it cancels: the tumour is near equilibrium at
        # T_OBS whatever the immune compartment does, so every treated arm returns
        # the same tumour density to three decimals and the fold-change readouts
        # are identically zero. Immune pressure has to move the fixed point.
        x0 = yc[0]
        cap = gain[0] * TUMOUR_CAP * M[0]
        dx0 = dec[0] * x0 * (1.0 - x0 / cap)
        return dx.at[0].set(dx0)

    def g_one(vartheta, scenario):
        theta = jnp.exp(vartheta) * scenario_mult[scenario]
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(rhs),
            diffrax.Tsit5(),
            t0=0.0,
            t1=T_OBS,
            dt0=T_OBS / N_STEPS,
            y0=y0,
            args=theta,
            stepsize_controller=diffrax.ConstantStepSize(),
            max_steps=N_STEPS + 8,
            adjoint=diffrax.DirectAdjoint(),
        )
        return jnp.clip(sol.ys[-1], Y_FLOOR, None)

    return g_one


# -----------------------------------------------------------------------------
# Readouts, in the kinds the real target table reports.
# -----------------------------------------------------------------------------
#
# Reading vpop_marginal_targets.csv, the 49 observables fall into six kinds:
#
#   density        cd8_density_baseline, tam_density, mdsc_density, ...
#   percentage     cd8_pct_baseline_li2022, treg_pct_nonLA_gvax_d21, ...
#   subset frac    treg_fraction_cd4, icaf_fraction_of_caf, mycaf_fraction_of_caf
#   ratio          m1_m2_ratio, apc_maturation_ratio
#   fold change    cd8_fc_nonLA_gvax_d21, cd8gzmb_fc_nonLA_gvax_nivo_d21
#   concentration  PDAC_IL10_tumor_tissue_Herremans2023, IL6, VEGF
#
# The fold change is the one the small toy has no analogue of, and it is the most
# useful row in the table for eq:disc. Under the centred map,
#
#   x~[s] - x~[0] = kappa_r (h_r[s] - h_r[0]),
#
# so gamma cancels exactly and kappa survives as a pure scale. It is the only
# readout kind that loads on b with no gamma contamination at all, which makes it
# the cleanest evidence about the scale half of the measurement map that the real
# corpus contains.

# (name, kind, modality, spec) where spec depends on the kind.
READOUT_KINDS = ("density", "pct", "fraction", "ratio", "foldchange", "conc")


def build_readouts(net):
    """Readout definitions plus the ``Z`` they imply.

    Returns ``(names, kinds, modality, h_all, Z, z_col_names)``.
    """
    cls = net.classes

    def head(name, k=0):
        return cls[name][0] + k

    def rng_of(name):
        return cls[name]

    dens = [("tumour_density", "tumour"), ("cd8_density", "cd8"),
            ("cd4_density", "cd4"), ("treg_density", "treg"),
            ("m2tam_density", "m2"), ("mdsc_density", "mdsc"),
            ("caf_density", "icaf"), ("collagen_level", "ecm")]
    pcts = [("cd8_pct", "cd8"), ("cd4_pct", "cd4"), ("treg_pct", "treg"),
            ("cd8exh_pct", "texh"), ("m1tam_pct", "m1"), ("m2tam_pct", "m2"),
            ("dc_pct", "dc"), ("nk_pct", "nk")]
    fracs = [("treg_fraction_cd4", "treg", "cd4"),
             ("icaf_fraction_of_caf", "icaf", "icaf"),
             ("mycaf_fraction_of_caf", "mycaf", "icaf"),
             ("texh_fraction_cd8", "texh", "cd8")]
    ratios = [("m1_m2_ratio", "m1", "m2"), ("cd8_treg_ratio", "cd8", "treg"),
              ("tumour_caf_ratio", "tumour", "icaf")]
    fcs = [("cd8_fc", "cd8"), ("treg_fc", "treg"), ("m2tam_fc", "m2")]
    concs = [("il6_conc", 0), ("il10_conc", 1), ("vegf_conc", 2),
             ("cxcl9_conc", 3)]

    names, kinds, modality = [], [], []
    # Modality follows the assay a source of that kind would have used: densities
    # and percentages are split between flow and histology so the modality column
    # is identified from a within-quantity-type contrast, exactly as in the small
    # toy. Concentrations are all ELISA, so they carry no modality contrast and
    # is_flow is 0 there by construction rather than by choice.
    for i, (nm, _) in enumerate(dens):
        names.append(nm)
        kinds.append("density")
        modality.append(1.0 if i % 2 == 0 else 0.0)
    for i, (nm, _) in enumerate(pcts):
        names.append(nm)
        kinds.append("pct")
        modality.append(1.0 if i % 2 == 1 else 0.0)
    for nm, _, _ in fracs:
        names.append(nm)
        kinds.append("fraction")
        modality.append(1.0)
    for nm, _, _ in ratios:
        names.append(nm)
        kinds.append("ratio")
        modality.append(0.0)
    for nm, _ in fcs:
        names.append(nm)
        kinds.append("foldchange")
        modality.append(1.0)
    for nm, _ in concs:
        names.append(nm)
        kinds.append("conc")
        modality.append(0.0)

    cell_hi = cls["ecm"][1]
    sol_lo = cls["soluble"][0]

    dens_lo = jnp.array([rng_of(c)[0] for _, c in dens])
    dens_hi = jnp.array([rng_of(c)[1] for _, c in dens])
    pct_lo = jnp.array([rng_of(c)[0] for _, c in pcts])
    pct_hi = jnp.array([rng_of(c)[1] for _, c in pcts])
    fr_nlo = jnp.array([rng_of(a)[0] for _, a, _ in fracs])
    fr_nhi = jnp.array([rng_of(a)[1] for _, a, _ in fracs])
    fr_dlo = jnp.array([rng_of(b)[0] for _, _, b in fracs])
    fr_dhi = jnp.array([rng_of(b)[1] for _, _, b in fracs])
    # icaf/mycaf fractions are of the CAF total, which is both classes together.
    fr_dhi = fr_dhi.at[1].set(cls["mycaf"][1]).at[2].set(cls["mycaf"][1])
    ra_nlo = jnp.array([rng_of(a)[0] for _, a, _ in ratios])
    ra_nhi = jnp.array([rng_of(a)[1] for _, a, _ in ratios])
    ra_dlo = jnp.array([rng_of(b)[0] for _, _, b in ratios])
    ra_dhi = jnp.array([rng_of(b)[1] for _, _, b in ratios])
    fc_lo = jnp.array([rng_of(c)[0] for _, c in fcs])
    fc_hi = jnp.array([rng_of(c)[1] for _, c in fcs])
    conc_idx = jnp.array([sol_lo + k for _, k in concs])

    def _seg(y_all, lo, hi):
        """Sum over a species range, for each (scenario, patient). Masked, so it
        stays a fixed-shape reduction that jit and vmap can both handle."""
        q = y_all.shape[-1]
        ar = jnp.arange(q)
        mask = (ar[None, :] >= lo[:, None]) & (ar[None, :] < hi[:, None])
        return jnp.einsum("snq,kq->snk", y_all, mask.astype(y_all.dtype))

    def h_all(y_all):
        """All readouts from species. ``(S, N, Q) -> (S, N, M)``.

        Takes every scenario at once because a fold change is a contrast between
        two scenarios *in the same patient*, which the common random numbers make
        meaningful and which a per-scenario signature could not express.
        """
        eps = 1e-12
        cell_total = jnp.sum(y_all[..., :cell_hi], axis=-1, keepdims=True)

        d = _seg(y_all, dens_lo, dens_hi)
        p = _seg(y_all, pct_lo, pct_hi) / (cell_total + eps)
        fn = _seg(y_all, fr_nlo, fr_nhi)
        fd = _seg(y_all, fr_dlo, fr_dhi)
        rn = _seg(y_all, ra_nlo, ra_nhi)
        rd = _seg(y_all, ra_dlo, ra_dhi)
        fc = _seg(y_all, fc_lo, fc_hi)
        c = y_all[..., conc_idx]

        log = lambda v: jnp.log(jnp.clip(v, 1e-30, None))
        return jnp.concatenate(
            [
                log(d),
                log(p),
                log(fn) - log(fd),
                log(rn) - log(rd),
                log(fc) - log(fc[0:1]),  # against the baseline scenario
                log(c),
            ],
            axis=-1,
        )

    # Z: quantity type against the density reference level, crossed with modality.
    z_col_names = ["intercept", "is_pct", "is_fraction", "is_ratio",
                   "is_foldchange", "is_conc", "is_flow"]
    rows = []
    for k, mo in zip(kinds, modality):
        rows.append([
            1.0,
            1.0 if k == "pct" else 0.0,
            1.0 if k == "fraction" else 0.0,
            1.0 if k == "ratio" else 0.0,
            1.0 if k == "foldchange" else 0.0,
            1.0 if k == "conc" else 0.0,
            mo,
        ])
    Z = jnp.array(rows)
    return names, kinds, modality, h_all, Z, z_col_names


def population_inputs(net, S, seed=0):
    """``R``, ``omega_0``, ``mu*``, ``mu_0`` and ``Sigma_1`` at this size.

    ``R`` is built as low rank plus diagonal so it is positive definite by
    construction at P = 271; a hand-written correlation matrix at that size would
    not be. The eight factors stand in for the parameter classes that move
    together in the real prior.
    """
    rng = np.random.default_rng(seed + 991)
    P = net.P

    F = rng.normal(0.0, 1.0, (P, 8)) * 0.35
    Cov = F @ F.T + np.eye(P)
    d = np.sqrt(np.diag(Cov))
    R = Cov / np.outer(d, d)

    # omega_0: the note's L1 default of 0.20, with a fifth of the parameters
    # treated as tighter material constants at 0.12.
    omega_0 = np.full(P, 0.20)
    omega_0[rng.random(P) < 0.20] = 0.12

    mu_true = np.log(net.theta_median)
    # mu_0 is displaced from the truth, parameter by parameter, by about half a
    # stage-1 sd. The fit has to move, and the emulator design has to cover it.
    sd1 = rng.uniform(0.18, 0.34, P)
    mu_0 = mu_true + rng.normal(0.0, 1.0, P) * sd1 * 0.5
    Sigma_1 = np.outer(sd1, sd1) * R

    return (jnp.array(R), jnp.array(omega_0), jnp.array(mu_true),
            jnp.array(mu_0), jnp.array(Sigma_1))

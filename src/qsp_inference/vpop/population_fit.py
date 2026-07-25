"""Sampling the population hyperparameters against the closed-form summary
likelihood (docs ch. 4b, Part II).

:mod:`qsp_inference.vpop.summary_likelihood` supplies
``log p(observed anchors | phi)``. This module supplies the prior, the packing
into one unconstrained vector, and NUTS. Together they are an ordinary
hierarchical model with an explicit log-density -- the honest name is
*model-based meta-analysis on a mechanistic forward model*: aggregate published
summaries, per-study ``n``, between-study heterogeneity, a relevance discount.

The parameter vector
--------------------
::

    phi = [ mu (P) | log_sigma_u (P) | eta_raw (S) | delta_raw (D)
            | log_tau_eta | log_tau_delta ]

All ``P`` components of ``mu`` and ``log_sigma_u`` are inferred. Selecting ``K``
identified directions and pinning the rest was never a consequence of the
identifiability wall; it was a consequence of a neural estimator needing a finite
label vector. With an explicit likelihood and a proper prior you infer everything
and let the prior return the answer where the data are silent, which is the
correct posterior. Which directions actually moved is then a *readout* -- the
variance budget of ch. 4 §*Reporting* -- rather than a gate.

**The hard qualifier is a fact about the pool, not about the estimator.** If the
simulations pinned a parameter, the emulator never saw it move, so along that
direction the likelihood is not merely flat, it is undefined, and pushing a
``theta`` that varies it through the emulator is extrapolation rather than
inference. Read ``omega_supported_frac`` before quoting any per-parameter
posterior: on the current PDAC pool it is 0.17, so 83% of the population's
nominal log-variance sits on parameters that cannot be represented at all. That
share is *not representable*, which is a different row of the variance budget
from *asserted* (the data were silent) -- see ch. 4 §*Reporting*.

``eta`` and ``delta`` are non-centered (``eta = tau_eta * eta_raw``), which is the
parameterisation NUTS needs when the study count is small and ``tau`` can
approach zero.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from qsp_inference.vpop.summary_likelihood import SummaryLikelihood

__all__ = [
    "PopulationPrior",
    "PopulationPosterior",
    "PopulationFit",
    "run_nuts",
]

_DTYPE = torch.float64
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _normal_logpdf(x: Tensor, loc: Tensor, scale: Tensor) -> Tensor:
    z = (x - loc) / scale
    return (-0.5 * z * z - torch.log(scale) - _LOG_SQRT_2PI).sum()


@dataclass
class PopulationPrior:
    """The hyperprior ``h(phi)``, all pieces independent and log-normal or normal.

    Args:
        mu_loc, mu_scale: ``(P,)`` prior on the population center in log-theta
            space. This is ``Gamma_pi``, the *epistemic* covariance -- how well
            the median patient's parameter is known -- and it is a different
            object from ``Gamma_omega`` below. Conflating them is the SEM-vs-SD
            error one level up and it over-disperses the reported population.
            Taken diagonal here; supply a whitened basis through
            ``SummaryLikelihood.basis`` if the prior has meaningful correlations.
        omega: ``(P,)`` the population spread center, from the layered omega
            prior (:mod:`qsp_inference.targets.omega`). This is ``Gamma_omega``,
            between-patient variability, and it must not be read off the width of
            a flat prior marginal.
        omega_scale: ``(P,)`` or scalar log-sd on ``log sigma_u`` around
            ``log omega``. This is how much the data are allowed to move the
            spread. Widening it is not a safe default: along an unidentified
            direction the posterior equals the prior, so a loose ``omega_scale``
            reports an over-dispersed population that nothing measured.
        tau_eta_loc, tau_eta_scale: log-normal prior on the between-study offset
            scale (matched to the translation-sigma rubric's floor of ~0.15 on
            the log scale by default).
        tau_delta_loc, tau_delta_scale: same for the per-observable structural
            discrepancy scale.
    """

    mu_loc: np.ndarray
    mu_scale: np.ndarray
    omega: np.ndarray
    omega_scale: float | np.ndarray = 0.5
    tau_eta_loc: float = math.log(0.15)
    tau_eta_scale: float = 1.0
    tau_delta_loc: float = math.log(0.15)
    tau_delta_scale: float = 1.0

    def __post_init__(self) -> None:
        self.mu_loc = torch.as_tensor(np.asarray(self.mu_loc, dtype=np.float64), dtype=_DTYPE)
        self.mu_scale = torch.as_tensor(np.asarray(self.mu_scale, dtype=np.float64), dtype=_DTYPE)
        self.omega = torch.as_tensor(np.asarray(self.omega, dtype=np.float64), dtype=_DTYPE)
        self.omega_scale = torch.as_tensor(
            np.broadcast_to(np.asarray(self.omega_scale, dtype=np.float64), self.omega.shape).copy(),
            dtype=_DTYPE,
        )
        if not (self.mu_loc.shape == self.mu_scale.shape == self.omega.shape):
            raise ValueError("mu_loc, mu_scale and omega must all be (P,) and agree")
        if torch.any(self.mu_scale <= 0) or torch.any(self.omega <= 0):
            raise ValueError("mu_scale and omega must be strictly positive")

    @property
    def n_param(self) -> int:
        return int(self.omega.numel())

    def log_prob(
        self,
        mu: Tensor,
        log_sigma_u: Tensor,
        eta_raw: Tensor,
        delta_raw: Tensor,
        log_tau_eta: Tensor,
        log_tau_delta: Tensor,
    ) -> Tensor:
        one = torch.ones((), dtype=mu.dtype, device=mu.device)
        return (
            _normal_logpdf(mu, self.mu_loc, self.mu_scale)
            + _normal_logpdf(log_sigma_u, torch.log(self.omega), self.omega_scale)
            + _normal_logpdf(eta_raw, 0.0 * one, one)
            + _normal_logpdf(delta_raw, 0.0 * one, one)
            + _normal_logpdf(log_tau_eta, one * self.tau_eta_loc, one * self.tau_eta_scale)
            + _normal_logpdf(log_tau_delta, one * self.tau_delta_loc, one * self.tau_delta_scale)
        )


@dataclass
class PopulationPosterior:
    """The unnormalised log-posterior, packed onto one unconstrained vector.

    This is the object NUTS differentiates. It is deliberately flat rather than a
    probabilistic-programming model: the whole point of ch. 4b is that the
    density is available in closed form, so the only machinery needed is a
    potential function and its gradient.
    """

    likelihood: SummaryLikelihood
    prior: PopulationPrior
    n_studies: int
    n_delta: int

    def __post_init__(self) -> None:
        if self.prior.n_param != self.likelihood.n_param:
            raise ValueError(
                f"prior is over {self.prior.n_param} parameters but the likelihood's "
                f"z is ({self.likelihood.n_patients}, {self.likelihood.n_param})"
            )

    @property
    def size(self) -> int:
        return 2 * self.prior.n_param + self.n_studies + self.n_delta + 2

    def unpack(self, phi: Tensor) -> dict:
        """Split the flat vector and apply the non-centering."""
        p, s, d = self.prior.n_param, self.n_studies, self.n_delta
        i = 0
        mu, i = phi[i : i + p], i + p
        log_sigma_u, i = phi[i : i + p], i + p
        eta_raw, i = phi[i : i + s], i + s
        delta_raw, i = phi[i : i + d], i + d
        log_tau_eta, log_tau_delta = phi[i], phi[i + 1]
        tau_eta, tau_delta = torch.exp(log_tau_eta), torch.exp(log_tau_delta)
        return {
            "mu": mu,
            "log_sigma_u": log_sigma_u,
            "sigma_u": torch.exp(log_sigma_u),
            "eta_raw": eta_raw,
            "delta_raw": delta_raw,
            "log_tau_eta": log_tau_eta,
            "log_tau_delta": log_tau_delta,
            "tau_eta": tau_eta,
            "tau_delta": tau_delta,
            "eta": tau_eta * eta_raw,
            "delta": tau_delta * delta_raw,
        }

    def pack(
        self,
        *,
        mu: Optional[np.ndarray] = None,
        sigma_u: Optional[np.ndarray] = None,
        tau_eta: float = 0.15,
        tau_delta: float = 0.15,
    ) -> Tensor:
        """A starting vector. Defaults sit at the prior center with zero offsets,
        which is the right place to start: it is the population the priors alone
        imply, so the first gradient step is the data's first word."""
        p = self.prior.n_param
        mu_t = (
            self.prior.mu_loc.clone()
            if mu is None
            else torch.as_tensor(np.asarray(mu, dtype=np.float64), dtype=_DTYPE)
        )
        ls = (
            torch.log(self.prior.omega)
            if sigma_u is None
            else torch.log(torch.as_tensor(np.asarray(sigma_u, dtype=np.float64), dtype=_DTYPE))
        )
        if mu_t.numel() != p or ls.numel() != p:
            raise ValueError(f"mu and sigma_u must have {p} entries")
        return torch.cat(
            [
                mu_t,
                ls,
                torch.zeros(self.n_studies, dtype=_DTYPE),
                torch.zeros(self.n_delta, dtype=_DTYPE),
                torch.tensor([math.log(tau_eta), math.log(tau_delta)], dtype=_DTYPE),
            ]
        )

    def log_prob(self, phi: Tensor) -> Tensor:
        v = self.unpack(phi)
        lp = self.prior.log_prob(
            v["mu"], v["log_sigma_u"], v["eta_raw"], v["delta_raw"],
            v["log_tau_eta"], v["log_tau_delta"],
        )
        ll = self.likelihood.log_prob(v["mu"], v["sigma_u"], v["eta"], v["delta"])
        return lp + ll

    def potential(self, params: dict) -> Tensor:
        """``-log p(phi, data)``, the form Pyro's NUTS consumes.

        A non-finite value is returned rather than raised: an emulator asked for a
        ``theta`` far outside the pool can produce one, and the sampler's job is
        to reject that trajectory, not to crash.
        """
        value = -self.log_prob(params["phi"])
        if not bool(torch.isfinite(value.detach())):
            return torch.full_like(value, float("inf"))
        return value


@dataclass
class PopulationFit:
    """Posterior draws, with the labels needed to read them.

    Attributes:
        mu, sigma_u: ``(n_samples, P)``. ``sigma_u`` is in the eigenbasis when
            ``SummaryLikelihood.basis`` was set.
        eta: ``(n_samples, S)`` study offsets, aligned to ``cohort_ids``.
        delta: ``(n_samples, D)`` per-observable discrepancies, aligned to
            ``delta_names``.
        tau_eta, tau_delta: ``(n_samples,)``.
        param_names, cohort_ids, delta_names: the labels for those axes.
        diagnostics: whatever the sampler reported (``r_hat``, ``n_eff``,
            divergences).
    """

    mu: np.ndarray
    sigma_u: np.ndarray
    eta: np.ndarray
    delta: np.ndarray
    tau_eta: np.ndarray
    tau_delta: np.ndarray
    param_names: list
    cohort_ids: list
    delta_names: list
    diagnostics: dict

    @property
    def n_samples(self) -> int:
        return int(self.mu.shape[0])

    def spread_movement(self, prior: PopulationPrior) -> np.ndarray:
        """Per-direction ``log(sigma_u_posterior_median / omega_prior)``.

        The reporting readout that replaces ``K`` selection: which directions the
        data actually moved, and by how much. A direction near zero here is one
        where the posterior equals the prior, which is the correct answer when the
        data are silent and a warning when you expected otherwise. Cross-check it
        against ``omega_supported_frac``: a direction the pool never varied cannot
        move, and its zero means *not representable* rather than *asserted*.
        """
        omega = prior.omega.detach().cpu().numpy()
        return np.log(np.median(self.sigma_u, axis=0) / omega)


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def run_nuts(
    posterior: PopulationPosterior,
    *,
    num_samples: int = 500,
    warmup_steps: int = 500,
    init: Optional[Tensor] = None,
    param_names: Optional[Sequence[str]] = None,
    cohort_ids: Optional[Sequence[str]] = None,
    delta_names: Optional[Sequence[str]] = None,
    max_tree_depth: int = 8,
    target_accept_prob: float = 0.8,
    jit_compile: bool = False,
    seed: int = 0,
    progress_bar: bool = True,
) -> PopulationFit:
    """Sample ``phi`` with NUTS (Pyro), against the closed-form summary likelihood.

    ``max_tree_depth`` defaults below Pyro's 10 because every leapfrog step is a
    full emulator forward over the whole virtual cohort, so trajectory length is
    the dominant cost. Raise it if the sampler saturates the depth.

    Freeze the covariance first
    (:meth:`~qsp_inference.vpop.summary_likelihood.SummaryLikelihood.freeze_covariance_at`).
    Sampling against a live covariance biases the population spread low by
    ``O(1/n)``; that method carries the measurement. It also makes every leapfrog
    step cheaper, since the Cholesky is cached instead of refactorised.

    Requires ``pyro-ppl``. The sampler is PyTorch-side rather than the NumPyro one
    ``submodel/inference.py`` uses, because the emulator is a torch module and the
    whole ``phi -> anchors`` map has to be one autodiff graph. Moving to NumPyro
    would mean rewriting this function only; the likelihood is sampler-agnostic.
    """
    try:
        from pyro.infer import MCMC, NUTS
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "run_nuts needs pyro-ppl (pip install pyro-ppl). The likelihood itself "
            "(vpop.summary_likelihood) needs only torch, so a gradient-free or "
            "optimiser-based driver can be written against it instead."
        ) from exc

    if not posterior.likelihood.covariance_is_frozen:
        warnings.warn(
            "sampling against a LIVE anchor covariance. Sigma scales with the "
            "population width, so the log-determinant term biases sigma_u low by "
            "O(1/n) -- 12% at n=20. Call likelihood.freeze_covariance_at(mu0, "
            "sigma_u0) first.",
            RuntimeWarning,
            stacklevel=2,
        )

    torch.manual_seed(seed)
    phi0 = posterior.pack() if init is None else torch.as_tensor(init, dtype=_DTYPE)
    if phi0.numel() != posterior.size:
        raise ValueError(f"init has {phi0.numel()} entries, expected {posterior.size}")
    if not torch.isfinite(posterior.log_prob(phi0)):
        raise ValueError(
            "the initial phi has non-finite log-posterior. Usually the emulator "
            "returned non-finite observables for the prior-center population; check "
            "the working-scale wrapper and any broken observable columns."
        )

    kernel = NUTS(
        potential_fn=posterior.potential,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
        jit_compile=jit_compile,
    )
    mcmc = MCMC(
        kernel,
        num_samples=num_samples,
        warmup_steps=warmup_steps,
        initial_params={"phi": phi0},
        disable_progbar=not progress_bar,
    )
    mcmc.run()

    draws = mcmc.get_samples()["phi"]
    unpacked = [posterior.unpack(d) for d in draws]
    stack = lambda key: np.stack([_to_numpy(u[key]) for u in unpacked])  # noqa: E731

    p, s, d = posterior.prior.n_param, posterior.n_studies, posterior.n_delta
    diagnostics = {"num_samples": int(draws.shape[0])}
    try:
        diagnostics.update(mcmc.diagnostics())
    except Exception:  # pragma: no cover - diagnostics are best-effort
        pass

    return PopulationFit(
        mu=stack("mu"),
        sigma_u=stack("sigma_u"),
        eta=stack("eta"),
        delta=stack("delta"),
        tau_eta=stack("tau_eta"),
        tau_delta=stack("tau_delta"),
        param_names=list(param_names) if param_names is not None else [f"p{i}" for i in range(p)],
        cohort_ids=list(cohort_ids) if cohort_ids is not None else [f"study{i}" for i in range(s)],
        delta_names=list(delta_names) if delta_names is not None else [f"obs{i}" for i in range(d)],
        diagnostics=diagnostics,
    )

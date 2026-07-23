"""Gaussian copula prior from submodel_priors.yaml.

Reads the marginal fits and copula correlation matrix produced by the audit
pipeline and constructs a PyTorch Distribution that can be used directly
with SBI workflows.

Usage::

    from qsp_inference.priors.copula_prior import load_copula_prior

    prior, param_names = load_copula_prior("submodel_priors.yaml")
    samples = prior.sample((10000,))
    log_p = prior.log_prob(samples)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import torch
import numpy as np
from scipy import stats
from torch.distributions import Distribution

_LOG_TWO_PI = float(np.log(2.0 * np.pi))


def _build_scipy_marginal(marginal: dict):
    """Build a scipy frozen distribution from a marginal spec dict.

    Supports lognormal/normal/gamma/invgamma (used by submodel-prior YAML
    fits) plus uniform/beta (used by CSV priors). Beta parameterization
    follows qsp-inference's convention: ``alpha``/``beta`` keys map to
    scipy's ``a``/``b`` (concentration1, concentration0).
    """
    dist_name = marginal["distribution"]
    if dist_name == "lognormal":
        return stats.lognorm(s=marginal["sigma"], scale=np.exp(marginal["mu"]))
    elif dist_name == "gamma":
        return stats.gamma(marginal["shape"], scale=marginal["scale"])
    elif dist_name == "invgamma":
        return stats.invgamma(marginal["shape"], scale=marginal["scale"])
    elif dist_name == "normal":
        return stats.norm(loc=marginal["mu"], scale=marginal["sigma"])
    elif dist_name == "uniform":
        return stats.uniform(loc=marginal["low"], scale=marginal["high"] - marginal["low"])
    elif dist_name == "beta":
        return stats.beta(marginal["alpha"], marginal["beta"])
    else:
        raise ValueError(f"Unknown marginal distribution: {dist_name}")


class GaussianCopulaPrior(Distribution):
    """Joint prior using marginal distributions coupled by a Gaussian copula.

    Sampling:
        1. Draw z ~ N(0, R) using Cholesky decomposition
        2. Transform to uniform: u = Phi(z)
        3. Transform to parameter space: x_i = F_i^{-1}(u_i)

    Log-prob:
        log p(x) = sum_i log f_i(x_i)
                  + log c(u_1, ..., u_n)

        where c is the Gaussian copula density:
        log c(u) = -0.5 * (z^T (R^{-1} - I) z)  - 0.5 * log|R|
        and z_i = Phi^{-1}(u_i)

    Args:
        marginals: List of scipy frozen distributions (one per parameter).
        correlation: (n, n) correlation matrix for the Gaussian copula.
            If None, parameters are treated as independent.
        param_names: Optional list of parameter names for reference.
    """

    has_rsample = False

    def __init__(
        self,
        marginals: list,
        correlation: np.ndarray | None = None,
        param_names: list[str] | None = None,
    ):
        self._marginals = marginals
        self._param_names = param_names or [f"p{i}" for i in range(len(marginals))]
        n = len(marginals)

        if correlation is not None:
            R = np.array(correlation, dtype=np.float64)
        else:
            R = np.eye(n)

        # Sample correlation matrices from MCMC posteriors can have tiny
        # negative eigenvalues from numerical noise, breaking Cholesky.
        # Project onto PSD cone via eigenvalue clipping if needed, then
        # renormalize to a correlation matrix (unit diagonal).
        try:
            np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh((R + R.T) / 2.0)
            w_clipped = np.clip(w, a_min=1e-8, a_max=None)
            R = V @ np.diag(w_clipped) @ V.T
            d = np.sqrt(np.diag(R))
            R = R / np.outer(d, d)

        self._R = R
        self._L = torch.tensor(
            np.linalg.cholesky(R), dtype=torch.float64
        )
        self._R_inv = torch.tensor(
            np.linalg.inv(R), dtype=torch.float64
        )
        self._log_det_R = float(np.linalg.slogdet(R)[1])
        self._has_copula = not np.allclose(R, np.eye(n))

        # Exact path for all-normal marginals (every marginal the *_log loaders
        # produce). For a normal, Phi^{-1}(Phi((x-loc)/scale)) == (x-loc)/scale,
        # so the cdf->ppf round trip in sample/log_prob is mathematically the
        # identity — but numerically it underflows in the tails, which is why
        # those paths clamp u to [1e-8, 1-1e-8]. That clamp caps |z| at ~5.61,
        # truncating samples and distorting the copula term beyond ~5.6 sigma.
        # Standardizing directly avoids the round trip, so it is exact at any
        # distance, and skips two scipy calls per parameter per evaluation.
        locs, scales, all_normal = [], [], True
        for m in marginals:
            if getattr(getattr(m, "dist", None), "name", None) != "norm":
                all_normal = False
                break
            locs.append(float(m.mean()))
            scales.append(float(m.std()))
        if all_normal and not all(np.isfinite(s) and s > 0 for s in scales):
            all_normal = False  # a degenerate/pinned scale would divide by zero
        self._all_normal = all_normal
        if all_normal:
            self._locs = torch.tensor(locs, dtype=torch.float64)
            self._scales = torch.tensor(scales, dtype=torch.float64)
            self._log_scales_sum = float(np.sum(np.log(scales)))

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([n]),
            validate_args=False,
        )

    # The copula couples log-space marginals (normal scores), so the joint
    # support is unbounded R^n. Declaring it explicitly makes this a fully
    # sbi-compatible Distribution: sbi only auto-assigns
    # constraints.independent(constraints.real, 1) to *non*-Distribution custom
    # priors, so a Distribution subclass that omits `support` raises
    # NotImplementedError inside DirectPosterior.sample / RestrictedPrior /
    # get_density_thresholder (the TSNPE path). This matches that default.
    @property
    def support(self):
        from torch.distributions import constraints
        return constraints.independent(constraints.real, 1)

    @property
    def param_names(self) -> list[str]:
        return self._param_names

    def subset(self, indices: list[int]) -> "GaussianCopulaPrior":
        """Return a new GaussianCopulaPrior over a subset of parameters.

        Extracts the corresponding marginals and correlation submatrix.

        Args:
            indices: Parameter indices to keep.

        Returns:
            A new GaussianCopulaPrior with only the selected parameters.
        """
        marginals = [self._marginals[i] for i in indices]
        R_sub = self._R[np.ix_(indices, indices)]
        names = [self._param_names[i] for i in indices]
        return GaussianCopulaPrior(
            marginals=marginals,
            correlation=R_sub,
            param_names=names,
        )

    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:
        n = len(self._marginals)
        shape = torch.Size(sample_shape) if not isinstance(sample_shape, torch.Size) else sample_shape
        n_samples = int(shape.numel())

        # z ~ N(0, R) via Cholesky
        z_indep = torch.randn(n_samples, n, dtype=torch.float64)
        z = z_indep @ self._L.T  # (n_samples, n)

        if self._all_normal:
            # x = loc + scale*z, exact. The general path below would round-trip
            # z through Phi and Phi^{-1} and clamp u, which silently truncates
            # draws at ~5.61 sigma.
            x = self._locs + self._scales * z
            return x.reshape(*shape, n).float()

        # z -> uniform via Phi
        u = torch.tensor(
            stats.norm.cdf(z.numpy()), dtype=torch.float64
        )
        u = torch.clamp(u, 1e-8, 1 - 1e-8)

        # uniform -> parameter space via marginal inverse CDF
        x = torch.empty_like(u)
        for j, marg in enumerate(self._marginals):
            x[:, j] = torch.tensor(
                marg.ppf(u[:, j].numpy()), dtype=torch.float64
            )

        return x.reshape(*shape, n).float()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.double()
        if value.dim() == 1:
            value = value.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        n_samples, n = value.shape

        if self._all_normal:
            # Normal scores directly: z = (x - loc)/scale is exactly
            # Phi^{-1}(F(x)) here, with no cdf/ppf round trip to underflow and
            # no u clamp, so this stays exact arbitrarily far into the tails.
            z = (value - self._locs) / self._scales
            marginal_lp = (
                -0.5 * (z * z).sum(dim=1)
                - self._log_scales_sum
                - 0.5 * n * _LOG_TWO_PI
            )
        else:
            # Marginal log-probs: sum_i log f_i(x_i)
            marginal_lp = torch.zeros(n_samples, dtype=torch.float64)
            u = torch.empty(n_samples, n, dtype=torch.float64)
            for j, marg in enumerate(self._marginals):
                xj = value[:, j].numpy()
                marginal_lp += torch.tensor(marg.logpdf(xj), dtype=torch.float64)
                u[:, j] = torch.tensor(marg.cdf(xj), dtype=torch.float64)

            u = torch.clamp(u, 1e-8, 1 - 1e-8)
            z = torch.tensor(stats.norm.ppf(u.numpy()), dtype=torch.float64)

        if self._has_copula:
            # copula log-density: -0.5 * (z^T (R^{-1} - I) z) - 0.5 * log|R|
            # Expand: z^T R^{-1} z - z^T z
            z_Rinv_z = (z @ self._R_inv * z).sum(dim=1)
            z_z = (z * z).sum(dim=1)
            copula_lp = -0.5 * (z_Rinv_z - z_z + self._log_det_R)
        else:
            copula_lp = torch.zeros(n_samples, dtype=torch.float64)

        result = (marginal_lp + copula_lp).float()
        return result.squeeze(0) if squeeze else result


def _csv_log_marginal(csv_row: dict):
    """Build a log-space normal marginal from a priors CSV row.

    Lognormal(mu, sigma) → Normal(mu, sigma) in log-space, exact.
    Other distributions → empirical fit: sample, log-transform, fit Normal.
    """
    dist = csv_row["distribution"]
    p1 = csv_row["p1"]
    p2 = csv_row["p2"]
    if dist == "lognormal":
        return stats.norm(loc=p1, scale=p2)
    if dist == "normal":
        marginal_spec = {"distribution": "normal", "mu": p1, "sigma": p2}
    elif dist == "uniform":
        marginal_spec = {"distribution": "uniform", "low": p1, "high": p2}
    elif dist == "beta":
        marginal_spec = {"distribution": "beta", "alpha": p1, "beta": p2}
    else:
        raise ValueError(
            f"_csv_log_marginal: unsupported CSV distribution '{dist}' "
            f"for parameter '{csv_row.get('name', '?')}'"
        )
    orig = _build_scipy_marginal(marginal_spec)
    samples = orig.rvs(size=200_000, random_state=42)
    samples = samples[np.isfinite(samples) & (samples > 0)]
    if samples.size < 100:
        raise ValueError(
            f"_csv_log_marginal: CSV distribution '{dist}' for "
            f"'{csv_row.get('name', '?')}' yielded only {samples.size} "
            "positive samples out of 200k; cannot fit log-space normal."
        )
    log_samples = np.log(samples)
    mu, sigma = float(np.mean(log_samples)), float(np.std(log_samples))
    return stats.norm(loc=mu, scale=sigma)


def _log_transform_marginal(marginal_spec: dict):
    """Convert a marginal spec to its log-space equivalent.

    Lognormal(mu, sigma) -> Normal(mu, sigma) exactly.
    Gamma/InvGamma -> fit Normal to log-samples (empirical approximation).
    """
    dist_name = marginal_spec["distribution"]
    if dist_name == "lognormal":
        return stats.norm(loc=marginal_spec["mu"], scale=marginal_spec["sigma"])
    else:
        # Empirical: sample, log-transform, fit normal. The MCMC posterior
        # parameterizer can fit Normal/Gamma/InvGamma marginals whose support
        # crosses or touches zero; rejection-truncate to log's domain before
        # fitting. Oversample so the truncation doesn't starve the fit.
        orig = _build_scipy_marginal(marginal_spec)
        samples = orig.rvs(size=200_000, random_state=42)
        samples = samples[np.isfinite(samples) & (samples > 0)]
        if samples.size < 100:
            raise ValueError(
                f"_log_transform_marginal: marginal '{dist_name}' yielded "
                f"only {samples.size} positive samples out of 200k; cannot "
                "fit log-space normal."
            )
        log_samples = np.log(samples)
        mu, sigma = float(np.mean(log_samples)), float(np.std(log_samples))
        return stats.norm(loc=mu, scale=sigma)


def load_composite_prior_log(
    yaml_path: str | Path,
    csv_path: str | Path,
    derived_yaml: str | Path | None = None,
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load a composite log-space prior: copula for submodel params, independent for the rest.

    Parameters in submodel_priors.yaml get their fitted marginals and copula
    correlations. Parameters only in the CSV get independent normal priors
    (from lognormal mu/sigma). The returned prior covers all CSV parameters
    in CSV order.

    Args:
        yaml_path: Path to submodel_priors.yaml from the audit pipeline.
        csv_path: Path to the full priors CSV (e.g. pdac_priors.csv).
        derived_yaml: Optional derived-parameter policy (see
            :func:`apply_derived_priors`). When given, the listed params are
            replaced by power-law children of their parents.

    Returns:
        (prior, param_names) where prior operates in log-space and covers
        all parameters from the CSV.
    """
    import csv as csv_mod
    from ruamel.yaml import YAML

    # Load YAML params
    yaml = YAML()
    with open(yaml_path) as f:
        yaml_data = yaml.load(f)

    yaml_entries = {p["name"]: p for p in yaml_data["parameters"]}

    # Load CSV params (preserves ordering). Capture the distribution field
    # so non-lognormal CSV priors (normal, uniform, beta) can be honored
    # for the params not in submodel_priors.yaml. dist_param1/2 are kept
    # under their original names; the marginal spec dict downstream maps
    # them to the right scipy/stats arguments per distribution.
    csv_params = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            csv_params.append({
                "name": row["name"],
                "distribution": row.get("distribution", "lognormal").strip(),
                "p1": float(row["dist_param1"]),
                "p2": float(row["dist_param2"]),
            })

    param_names = [p["name"] for p in csv_params]
    n = len(param_names)

    # Build marginals: YAML entries override CSV. Fall back to the CSV's
    # declared distribution if the YAML marginal is degenerate (e.g.
    # parameterizer fit a gamma with scale=0 from a posterior pinned at
    # zero). The CSV path is log-space — for non-lognormal distributions
    # we sample from the named distribution and fit a normal to log-samples,
    # matching the empirical-fit pattern used for gamma/invgamma marginals
    # from YAML.
    marginals = []
    fallback_params: list[str] = []
    for p in csv_params:
        if p["name"] in yaml_entries:
            try:
                marginals.append(_log_transform_marginal(yaml_entries[p["name"]]["marginal"]))
                continue
            except (ValueError, ZeroDivisionError, FloatingPointError):
                fallback_params.append(p["name"])
        marginals.append(_csv_log_marginal(p))
    if fallback_params:
        import warnings
        warnings.warn(
            f"load_composite_prior_log: {len(fallback_params)} YAML marginal(s) "
            f"failed log-domain fit; falling back to CSV lognormal: {fallback_params}",
            stacklevel=2,
        )

    # Build correlation matrix: identity everywhere, copula block for YAML params
    R = np.eye(n)
    copula = yaml_data.get("copula")
    if copula and copula.get("correlation"):
        copula_params = copula["parameters"]
        R_sub = np.array(copula["correlation"])
        for i, pi in enumerate(copula_params):
            for j, pj in enumerate(copula_params):
                if pi in param_names and pj in param_names:
                    fi = param_names.index(pi)
                    fj = param_names.index(pj)
                    R[fi, fj] = R_sub[i, j]

    prior = GaussianCopulaPrior(
        marginals=marginals,
        correlation=R,
        param_names=param_names,
    )

    if derived_yaml is not None:
        prior = apply_derived_priors(prior, load_derived_specs(derived_yaml))

    return prior, param_names


def _select_prior_block(data: dict, block: str | None) -> dict:
    """Return the ``{parameters, copula}`` node for the requested block.

    ``block=None`` → the top-level center-scale prior (the flat-SBI prior).
    ``block="population"`` → the parallel population (ω) block written by the
    Option-A population pass. The population block mirrors the top-level shape
    (``parameters`` + optional ``copula``) but covers only the params that
    actually have a population ``observed_distribution``.

    Raises ``KeyError`` if the requested block is absent — callers that treat
    the population block as optional should catch it.
    """
    if block is None:
        return data
    node = data.get(block)
    if node is None:
        raise KeyError(
            f"submodel_priors.yaml has no '{block}' block "
            "(was it generated with the population pass, "
            "run_comparison(run_population_pass=True)?)"
        )
    return node


def load_copula_prior(
    yaml_path: str | Path,
    device: str = "cpu",
    block: str | None = None,
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load a Gaussian copula prior from submodel_priors.yaml.

    Args:
        yaml_path: Path to the YAML file produced by the audit pipeline.
        device: Torch device (currently only 'cpu' supported for scipy ops).
        block: ``None`` for the center-scale prior (default), or
            ``"population"`` for the parallel ω block (Option A).

    Returns:
        (prior, param_names) tuple. The prior is a PyTorch Distribution with
        sample() and log_prob() methods compatible with SBI.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(yaml_path) as f:
        data = yaml.load(f)
    data = _select_prior_block(data, block)

    param_entries = data["parameters"]
    param_names = [p["name"] for p in param_entries]
    marginals = [_build_scipy_marginal(p["marginal"]) for p in param_entries]

    # Build full correlation matrix
    n = len(param_names)
    R = np.eye(n)

    copula = data.get("copula")
    if copula and copula.get("correlation"):
        copula_params = copula["parameters"]
        R_sub = np.array(copula["correlation"])
        # Map copula participant indices to full matrix
        for i, pi in enumerate(copula_params):
            for j, pj in enumerate(copula_params):
                fi = param_names.index(pi)
                fj = param_names.index(pj)
                R[fi, fj] = R_sub[i, j]

    prior = GaussianCopulaPrior(
        marginals=marginals,
        correlation=R,
        param_names=param_names,
    )

    return prior, param_names


def _log_marginal_loc_scale(marginal) -> tuple[float, float]:
    """Recover (mu, sigma) from a log-space marginal.

    The log-space loaders (``_log_transform_marginal`` / ``_csv_log_marginal``)
    always return a ``scipy.stats.norm`` frozen distribution, so ``mean``/``std``
    are exactly the normal's ``loc``/``scale``. Guarded so a non-log prior
    (lognormal marginals) can't silently be misread as if it were log-space.
    """
    if getattr(marginal, "dist", None) is None or marginal.dist.name != "norm":
        raise ValueError(
            "sigma-overlay requires a log-space prior (normal marginals); got "
            f"marginal '{getattr(getattr(marginal, 'dist', None), 'name', '?')}'. "
            "Pass the output of load_composite_prior_log / load_copula_prior_log."
        )
    return float(marginal.mean()), float(marginal.std())


def compose_overlay_prior(
    center: GaussianCopulaPrior,
    *,
    population_sigma: Mapping[str, float] | None = None,
    pin_params: Iterable[str] | None = None,
    pin_sigma: float = 1e-3,
) -> GaussianCopulaPrior:
    """Compose the cloud-generation prior: center mu, overlaid sigma.

    Builds a new log-space :class:`GaussianCopulaPrior` from a *center* prior by
    the per-parameter rule:

    - **mu = center mu, always** — the center is never moved.
    - **sigma = population sigma** where ``population_sigma`` supplies one,
      **pinned (``pin_sigma`` ~ 0)** for ``pin_params``, **center sigma** otherwise.
    - **Pins win over population** — a param in both is pinned.
    - Copula rows/cols are **zeroed for pinned params** (they become independent);
      the result stays PSD because it is ``blockdiag(1, principal-submatrix)`` of a
      correlation matrix. Correlations among the remaining params are preserved.

    This is the single hook shared by (a) a hand-curated vary/pin policy — pass its
    pinned set as ``pin_params`` — and (b) the data-driven population block — pass
    its per-param sigma as ``population_sigma`` (see :func:`load_overlay_prior_log`).
    As ``observed_distribution`` coverage grows, params flip from pinned/center to
    data-sigma through this same overlay with no new plumbing.

    Args:
        center: A log-space center prior (normal marginals), e.g. from
            :func:`load_composite_prior_log`.
        population_sigma: Optional ``{param_name: sigma}`` overrides (log-space).
        pin_params: Optional param names to pin (sigma -> ``pin_sigma``, decoupled).
        pin_sigma: Log-space sigma used for pinned params. Small but nonzero to keep
            the marginal a proper (samplable, finite-log-prob) distribution.

    Returns:
        ``(prior, param_names)`` — a new prior over the same params in the same order.
    """
    population_sigma = dict(population_sigma or {})
    pin = set(pin_params or [])
    names = list(center.param_names)

    mus: list[float] = []
    sigmas: list[float] = []
    for m in center._marginals:
        mu, sigma = _log_marginal_loc_scale(m)
        mus.append(mu)
        sigmas.append(sigma)

    R = np.array(center._R, dtype=np.float64).copy()

    # Population overrides first; pins applied second so they always win.
    for j, nm in enumerate(names):
        if nm not in pin and nm in population_sigma:
            sigmas[j] = float(population_sigma[nm])
    for j, nm in enumerate(names):
        if nm in pin:
            sigmas[j] = float(pin_sigma)
            R[j, :] = 0.0
            R[:, j] = 0.0
            R[j, j] = 1.0

    marginals = [stats.norm(loc=mus[j], scale=sigmas[j]) for j in range(len(names))]
    prior = GaussianCopulaPrior(marginals=marginals, correlation=R, param_names=names)
    return prior, names


def load_overlay_prior_log(
    yaml_path: str | Path,
    csv_path: str | Path,
    *,
    pin_params: Iterable[str] | None = None,
    vary_params: Iterable[str] | None = None,
    use_population_block: bool = True,
    pin_sigma: float = 1e-3,
    derived_yaml: str | Path | None = None,
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load the cloud-generation prior: composite center with a sigma-overlay.

    Convenience wrapper that loads the center composite prior
    (:func:`load_composite_prior_log`), pulls the population-block sigma for any
    param that declares an ``observed_distribution`` (if the block is present),
    and applies :func:`compose_overlay_prior`.

    Args:
        yaml_path: ``submodel_priors.yaml`` from the audit pipeline (optionally
            carrying a ``population`` block from the population pass).
        csv_path: Full priors CSV (defines param order and the fallback center sigma).
        pin_params: Param names to pin (sigma -> ``pin_sigma``).
        vary_params: Allowlist form of the pin policy — when given, every param
            NOT listed is pinned. Merged with any explicit ``pin_params`` (union).
            This is the shape a vary/pin policy file carries (a ``vary:`` list);
            passing it here avoids the caller having to invert it against the
            full param set.
        use_population_block: If True and a ``population`` block exists, override
            sigma with the population sigma for its params. Silently ignored when
            the block is absent.
        pin_sigma: Log-space sigma for pinned params.
        derived_yaml: Optional derived-parameter policy. Applied LAST (after the
            overlay), so derived params track their post-overlay parents and are
            themselves neither pinned nor population-widened.

    Returns:
        ``(prior, param_names)`` covering all CSV params in CSV order.
    """
    center, names = load_composite_prior_log(yaml_path, csv_path)

    # Allowlist → pin everything outside it, unioned with any explicit pins.
    if vary_params is not None:
        vary = set(vary_params)
        derived_pins = [n for n in names if n not in vary]
        pin_params = sorted(set(derived_pins) | set(pin_params or []))

    population_sigma: dict[str, float] = {}
    if use_population_block:
        try:
            pop_prior, pop_names = load_copula_prior_log(yaml_path, block="population")
        except KeyError:
            pop_prior = None
        if pop_prior is not None:
            population_sigma = {
                nm: _log_marginal_loc_scale(pop_prior._marginals[i])[1]
                for i, nm in enumerate(pop_names)
            }

    overlaid = compose_overlay_prior(
        center,
        population_sigma=population_sigma,
        pin_params=pin_params,
        pin_sigma=pin_sigma,
    )
    prior, names = overlaid
    if derived_yaml is not None:
        # A derived param must NOT be pinned by the vary-allowlist (it has no
        # free center to pin) — it is defined by its parents. Applied here, after
        # the overlay, it overwrites whatever pin/population sigma it was given.
        prior = apply_derived_priors(prior, load_derived_specs(derived_yaml))
    return prior, names


def load_derived_specs(path: str | Path) -> dict[str, dict]:
    """Parse a derived-parameter policy YAML into a spec dict.

    Schema (see the class docstring on :func:`apply_derived_priors`)::

        parameters:
          <derived_name>:
            parents: {<parent_name>: <power>, ...}   # power on each parent
            log_coeff: <float>                        # ln of the multiplicative constant
            sigma_coeff: <float>                      # log-space scatter of the constant
            provenance: <str>                         # optional, human note

    Returns ``{name: {parents, log_coeff, sigma_coeff, provenance}}``.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(path) as f:
        data = yaml.load(f)
    out: dict[str, dict] = {}
    for name, spec in (data.get("parameters") or {}).items():
        parents = {str(k): float(v) for k, v in (spec.get("parents") or {}).items()}
        if not parents:
            raise ValueError(f"derived param '{name}' has no parents.")
        out[str(name)] = {
            "parents": parents,
            "log_coeff": float(spec.get("log_coeff", 0.0)),
            "sigma_coeff": float(spec.get("sigma_coeff", 0.0)),
            "provenance": spec.get("provenance", ""),
        }
    return out


def _derived_topo_order(specs: dict[str, dict]) -> list[str]:
    """Order derived params so each is processed after its derived parents.

    A derived param may reference another derived param; that parent's marginal
    must be finalized first. Non-derived (base) parents impose no ordering.
    Raises on a cycle.
    """
    derived_names = set(specs)
    ordered: list[str] = []
    seen: set[str] = set()
    temp: set[str] = set()

    def visit(nm: str):
        if nm in seen:
            return
        if nm in temp:
            raise ValueError(f"cycle in derived-parameter dependencies at '{nm}'.")
        temp.add(nm)
        for parent in specs[nm]["parents"]:
            if parent in derived_names:
                visit(parent)
        temp.discard(nm)
        seen.add(nm)
        ordered.append(nm)

    for nm in specs:
        visit(nm)
    return ordered


def apply_derived_priors(
    prior: GaussianCopulaPrior,
    derived_specs: dict[str, dict],
) -> GaussianCopulaPrior:
    """Inject derived parameters into a log-space composite prior.

    A derived parameter is a multiplicative (power-law) child of its parents::

        value_d = coeff * prod_i parent_i ** power_i,   coeff ~ LogNormal(log_coeff, sigma_coeff)

    In log-space this is a linear-Gaussian child:
    ``log v_d = log_coeff + sum_i power_i * log parent_i + eps``, ``eps ~ N(0, sigma_coeff^2)``.
    Because the composite prior is a Gaussian copula over log-normal marginals,
    the child is exactly representable as one more copula dimension — so this
    returns a plain ``GaussianCopulaPrior`` (``sample``/``log_prob``/``subset``
    all keep working). The child's marginal and its correlation to every base
    param are derived closed-form from the parents' marginals and the base
    correlation ``R`` (with ``Sigma_ij = sigma_i sigma_j R_ij``)::

        mu_d    = log_coeff + sum_i power_i * mu_i
        sigma_d = sqrt(a^T Sigma a + sigma_coeff^2)          # a = power vector
        R_dk    = (sum_i power_i sigma_i R_ik) / sigma_d

    The derived param must already be a column of ``prior`` (from the CSV); its
    CSV marginal is OVERWRITTEN here — the CSV row is only a fallback for when no
    derived spec is supplied. Exact when parent marginals are normal-in-log
    (lognormal CSV rows); a moment-matched approximation for fitted non-normal
    marginals, consistent with how the copula already treats them.

    Args:
        prior: A log-space ``GaussianCopulaPrior`` (from ``load_composite_prior_log``
            / ``load_overlay_prior_log``).
        derived_specs: Output of :func:`load_derived_specs`.

    Returns:
        A new ``GaussianCopulaPrior`` with the derived columns replaced.
    """
    if not derived_specs:
        return prior

    names = list(prior._param_names)
    idx = {nm: i for i, nm in enumerate(names)}
    marginals = list(prior._marginals)
    R = np.array(prior._R, dtype=np.float64).copy()
    n = len(names)

    missing = [nm for nm in derived_specs if nm not in idx]
    if missing:
        raise ValueError(
            f"derived params not present as prior columns (add them to the CSV "
            f"first): {missing}"
        )

    for nm in _derived_topo_order(derived_specs):
        spec = derived_specs[nm]
        d = idx[nm]
        bad_parents = [p for p in spec["parents"] if p not in idx]
        if bad_parents:
            raise ValueError(f"derived '{nm}' references unknown parent(s): {bad_parents}")

        # Current (μ, σ) of every column — parents already-processed derived
        # columns carry their finalized values via topo order.
        mu = np.array([_log_marginal_loc_scale(m)[0] for m in marginals])
        sig = np.array([_log_marginal_loc_scale(m)[1] for m in marginals])
        Sigma = np.outer(sig, sig) * R

        a = np.zeros(n)
        for p, power in spec["parents"].items():
            a[idx[p]] = power

        mu_d = spec["log_coeff"] + float(a @ mu)
        var_d = float(a @ Sigma @ a) + spec["sigma_coeff"] ** 2
        sigma_d = float(np.sqrt(max(var_d, 1e-12)))

        # Correlation of the child to every base column k: (Σa)_k / (σ_d σ_k).
        Sig_a = Sigma @ a
        with np.errstate(divide="ignore", invalid="ignore"):
            R_d = Sig_a / (sigma_d * sig)
        R_d = np.nan_to_num(R_d, nan=0.0, posinf=0.0, neginf=0.0)
        R_d = np.clip(R_d, -0.999, 0.999)

        marginals[d] = stats.norm(loc=mu_d, scale=sigma_d)
        R[d, :] = R_d
        R[:, d] = R_d
        R[d, d] = 1.0

    return GaussianCopulaPrior(marginals=marginals, correlation=R, param_names=names)


def temper_prior(
    prior: GaussianCopulaPrior, temperature: float
) -> GaussianCopulaPrior:
    """Return ``prior^(1/T)`` — the prior flattened by a temperature.

    Used to build a *training proposal* from the anchored prior. Drawing training
    θ from a wider distribution than the one we report under is what lets the
    proposal be a computational choice and the prior a scientific one; the
    posterior is recovered under the original prior by importance reweighting
    (:mod:`qsp_inference.inference.importance`).

    **This is exact, not an approximation.** Every log-space marginal produced by
    the ``*_log`` loaders is a normal (``_csv_log_marginal`` and
    ``_log_transform_marginal`` both return ``stats.norm``), so a log-space
    :class:`GaussianCopulaPrior` *is* a multivariate normal with covariance
    ``Σ = D R D``. Tempering a Gaussian gives ``N(μ, T·Σ)``, and
    ``T·Σ = (√T·D) R (√T·D)`` — so scaling every marginal's σ by ``√T`` and
    leaving the correlation matrix alone is precisely ``π^(1/T)``.

    Deriving the proposal *from* the prior rather than specifying it alongside
    has a practical payoff: it cannot disagree with the prior about anything but
    width. Anchors, pins, derived children and correlations all carry over, and
    the parameter ordering is identical by construction, so the reweight cannot
    be silently misaligned.

    A note on what this deliberately cannot do: a parameter pinned to σ ≈ 0 by a
    σ-overlay stays pinned, because ``√T · 0 = 0``. That is the correct behaviour
    here — which parameters vary is a claim about the population and belongs to
    the prior, not to a sampling device.

    **Where the identity stops being exact.** ``GaussianCopulaPrior.log_prob``
    clamps the copula's ``u`` to ``[1e-8, 1-1e-8]`` (roughly ±5.6σ) before
    inverting to normal scores. Past that the *copula* term is distorted; the
    marginal term is exact everywhere, and with an uncorrelated prior there is no
    copula term at all, so the identity then holds exactly however far out you
    go. In practice this bites on the tail draws of a wide proposal evaluated
    under a narrow prior — precisely the draws whose importance weight is
    vanishing, so weighted summaries are unaffected. It is a property of the
    density guard, not of tempering.

    Args:
        prior: A log-space prior (normal marginals). Passing a prior in natural
            units raises, since tempering lognormal marginals this way is not
            the same operation.
        temperature: ``T > 0``. ``T > 1`` widens (the usual case, σ×√T);
            ``T < 1`` sharpens. ``T == 1`` returns ``prior`` unchanged, so the
            decoupled path is exactly inert when switched off.

    Returns:
        A new :class:`GaussianCopulaPrior`, or ``prior`` itself when ``T == 1``.

    Raises:
        ValueError: if ``temperature <= 0``, or if any marginal is not normal.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0; got {temperature}")
    if temperature == 1.0:
        return prior

    scale = float(np.sqrt(temperature))
    tempered = []
    for name, marg in zip(prior.param_names, prior._marginals):
        dist_name = getattr(getattr(marg, "dist", None), "name", None)
        if dist_name != "norm":
            raise ValueError(
                f"temper_prior needs normal marginals (log-space), but parameter "
                f"'{name}' has a '{dist_name}' marginal. Load the prior with one "
                "of the *_log loaders (load_composite_prior_log / "
                "load_overlay_prior_log / load_copula_prior_log)."
            )
        # For a normal, mean() and std() are exactly loc and scale, and are
        # robust to whether the frozen dist was built positionally or by keyword.
        tempered.append(
            stats.norm(loc=float(marg.mean()), scale=scale * float(marg.std()))
        )

    return GaussianCopulaPrior(
        marginals=tempered,
        correlation=prior._R,
        param_names=list(prior.param_names),
    )


def load_copula_prior_log(
    yaml_path: str | Path,
    block: str | None = None,
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load a Gaussian copula prior in log-space for SBI workflows.

    Transforms lognormal marginals to normal marginals in log-space,
    preserving the copula correlation structure. This is the drop-in
    replacement for ``load_prior`` + ``transform_lognormal_prior_to_normal``.

    Args:
        yaml_path: Path to the YAML file produced by the audit pipeline.
        block: ``None`` for the center-scale prior (default), or
            ``"population"`` for the parallel ω block (Option A). The
            population block covers only params with a population
            observed_distribution; ``KeyError`` if it is absent.

    Returns:
        (prior, param_names) where prior operates in log-space.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(yaml_path) as f:
        data = yaml.load(f)
    data = _select_prior_block(data, block)

    param_entries = data["parameters"]
    param_names = [p["name"] for p in param_entries]
    marginals = [_log_transform_marginal(p["marginal"]) for p in param_entries]

    n = len(param_names)
    R = np.eye(n)

    copula = data.get("copula")
    if copula and copula.get("correlation"):
        copula_params = copula["parameters"]
        R_sub = np.array(copula["correlation"])
        for i, pi in enumerate(copula_params):
            for j, pj in enumerate(copula_params):
                fi = param_names.index(pi)
                fj = param_names.index(pj)
                R[fi, fj] = R_sub[i, j]

    prior = GaussianCopulaPrior(
        marginals=marginals,
        correlation=R,
        param_names=param_names,
    )

    return prior, param_names

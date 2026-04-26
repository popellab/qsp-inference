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

from pathlib import Path

import torch
import numpy as np
from scipy import stats
from torch.distributions import Distribution


def _build_scipy_marginal(marginal: dict):
    """Build a scipy frozen distribution from a marginal spec dict."""
    dist_name = marginal["distribution"]
    if dist_name == "lognormal":
        return stats.lognorm(s=marginal["sigma"], scale=np.exp(marginal["mu"]))
    elif dist_name == "gamma":
        return stats.gamma(marginal["shape"], scale=marginal["scale"])
    elif dist_name == "invgamma":
        return stats.invgamma(marginal["shape"], scale=marginal["scale"])
    elif dist_name == "normal":
        return stats.norm(loc=marginal["mu"], scale=marginal["sigma"])
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

        super().__init__(
            batch_shape=torch.Size(),
            event_shape=torch.Size([n]),
            validate_args=False,
        )

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

        # Marginal log-probs: sum_i log f_i(x_i)
        marginal_lp = torch.zeros(n_samples, dtype=torch.float64)
        u = torch.empty(n_samples, n, dtype=torch.float64)
        for j, marg in enumerate(self._marginals):
            xj = value[:, j].numpy()
            marginal_lp += torch.tensor(marg.logpdf(xj), dtype=torch.float64)
            u[:, j] = torch.tensor(marg.cdf(xj), dtype=torch.float64)

        u = torch.clamp(u, 1e-8, 1 - 1e-8)

        if self._has_copula:
            # z = Phi^{-1}(u)
            z = torch.tensor(
                stats.norm.ppf(u.numpy()), dtype=torch.float64
            )
            # copula log-density: -0.5 * (z^T (R^{-1} - I) z) - 0.5 * log|R|
            # Expand: z^T R^{-1} z - z^T z
            z_Rinv_z = (z @ self._R_inv * z).sum(dim=1)
            z_z = (z * z).sum(dim=1)
            copula_lp = -0.5 * (z_Rinv_z - z_z + self._log_det_R)
        else:
            copula_lp = torch.zeros(n_samples, dtype=torch.float64)

        result = (marginal_lp + copula_lp).float()
        return result.squeeze(0) if squeeze else result


def _log_transform_marginal(marginal_spec: dict):
    """Convert a marginal spec to its log-space equivalent.

    Lognormal(mu, sigma) -> Normal(mu, sigma) exactly.
    Gamma/InvGamma -> fit Normal to log-samples (empirical approximation).
    """
    dist_name = marginal_spec["distribution"]
    if dist_name == "lognormal":
        return stats.norm(loc=marginal_spec["mu"], scale=marginal_spec["sigma"])
    else:
        # Empirical: sample, log-transform, fit normal
        orig = _build_scipy_marginal(marginal_spec)
        log_samples = np.log(orig.rvs(size=50_000, random_state=42))
        mu, sigma = float(np.mean(log_samples)), float(np.std(log_samples))
        return stats.norm(loc=mu, scale=sigma)


def load_composite_prior_log(
    yaml_path: str | Path,
    csv_path: str | Path,
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load a composite log-space prior: copula for submodel params, independent for the rest.

    Parameters in submodel_priors.yaml get their fitted marginals and copula
    correlations. Parameters only in the CSV get independent normal priors
    (from lognormal mu/sigma). The returned prior covers all CSV parameters
    in CSV order.

    Args:
        yaml_path: Path to submodel_priors.yaml from the audit pipeline.
        csv_path: Path to the full priors CSV (e.g. pdac_priors.csv).

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

    # Load CSV params (preserves ordering)
    csv_params = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            csv_params.append({
                "name": row["name"],
                "mu": float(row["dist_param1"]),
                "sigma": float(row["dist_param2"]),
            })

    param_names = [p["name"] for p in csv_params]
    n = len(param_names)

    # Build marginals: YAML entries override CSV
    marginals = []
    for p in csv_params:
        if p["name"] in yaml_entries:
            marginals.append(_log_transform_marginal(yaml_entries[p["name"]]["marginal"]))
        else:
            # CSV lognormal -> log-space normal
            marginals.append(stats.norm(loc=p["mu"], scale=p["sigma"]))

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

    return prior, param_names


def load_copula_prior(
    yaml_path: str | Path,
    device: str = "cpu",
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load a Gaussian copula prior from submodel_priors.yaml.

    Args:
        yaml_path: Path to the YAML file produced by the audit pipeline.
        device: Torch device (currently only 'cpu' supported for scipy ops).

    Returns:
        (prior, param_names) tuple. The prior is a PyTorch Distribution with
        sample() and log_prob() methods compatible with SBI.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(yaml_path) as f:
        data = yaml.load(f)

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


def load_copula_prior_log(
    yaml_path: str | Path,
) -> tuple[GaussianCopulaPrior, list[str]]:
    """Load a Gaussian copula prior in log-space for SBI workflows.

    Transforms lognormal marginals to normal marginals in log-space,
    preserving the copula correlation structure. This is the drop-in
    replacement for ``load_prior`` + ``transform_lognormal_prior_to_normal``.

    Args:
        yaml_path: Path to the YAML file produced by the audit pipeline.

    Returns:
        (prior, param_names) where prior operates in log-space.
    """
    from ruamel.yaml import YAML

    yaml = YAML()
    with open(yaml_path) as f:
        data = yaml.load(f)

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

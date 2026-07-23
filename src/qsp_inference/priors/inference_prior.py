"""One owner for the pair of distributions an inference run needs.

An amortized-inference run uses two distributions over the same parameters, and
they do different jobs:

    pi        the reporting prior. What the literature says, what a posterior is
              conditioned on, what a virtual patient is a draw from. Science.
    pi_tilde  the training proposal. What simulated theta are drawn from, chosen
              to cover the region the posterior might occupy and keep the
              density estimator trained there. Computation.

Conflating them is what forces a modeller to widen a prior for coverage and then
discover the population is over-dispersed. Keeping them apart costs one line of
arithmetic at report time (:mod:`qsp_inference.inference.importance`) and one
object here.

**Why this module exists rather than a helper in each caller.** The pair has to
be constructed identically everywhere it appears, because the training theta and
the density the importance weights are computed from must come from the same
distribution. Before this module there were two independent implementations, one
building the cloud and one building the density, in different repositories, and
they had drifted:

- the cloud composed the sigma-overlay through :func:`load_overlay_prior_log`
  while the density reimplemented it inline, reading population sigma via
  ``.std()`` and so skipping the guard that the marginal is log-space;
- on the CSV-only path they disagreed outright. The cloud sampled each row's
  declared family; the density treated every row as ``exp(N(p1, p2))``, so a
  parameter declared ``Beta(2, 18)`` (a fraction, confined to the unit interval)
  became a lognormal spanning tens of orders of magnitude.

Neither could raise. A wrong-but-finite density produces wrong-but-finite
weights. So construction and identity live in one place, and callers receive an
object rather than a recipe.

**Identity.** :meth:`PriorSpec.fingerprint` hashes the *content* of every input
file plus the temperature. It is the token a cache keys on. This matters more
than it looks: a distribution change that does not change a cache key is served
from the cache silently and answers with data from the wrong distribution.
Nothing downstream can detect that, so the identity belongs with the
construction.

**Tempering.** ``pi_tilde = pi^(1/T)``. Every log-space marginal of the
composite prior is normal, so this is exactly ``sigma -> sqrt(T) * sigma`` with
the copula correlation untouched: one knob, derived *from* pi rather than
hand-set beside it. ``T = 1`` returns the prior object itself, so
``pair.prior is pair.proposal`` and a run behaves exactly as it did before any
of this existed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "PriorSpec",
    "PriorPair",
    "CsvIndependentPrior",
    "build_prior_pair",
]

# Below this the importance weights pi/pi_tilde have infinite variance for
# Gaussian marginals: the integrand goes as exp(-z^2 (1 - 1/(2T))), which is
# integrable only for T > 1/2. Estimates built on such weights do not converge,
# so this is a hard floor rather than a warning.
_MIN_TEMPERATURE = 0.5


# ---------------------------------------------------------------------------
# What distribution
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PriorSpec:
    """Everything that determines pi and pi_tilde, and nothing else.

    Deliberately excludes the draw count, the seed and the restriction
    classifier: those identify a *pool* of samples, not the distribution it came
    from. See :class:`~qsp_inference.priors.theta_pool.ThetaPoolSpec`, which
    composes this with them.

    Attributes:
        priors_csv: Full priors CSV. Defines parameter order, and supplies the
            fallback marginal for parameters the submodel YAML does not cover.
        submodel_priors_yaml: Composite prior from the audit pipeline. ``None``
            selects the CSV-only path, which has no density and therefore
            supports neither tempering nor reweighting.
        vary_policy: Vary/pin policy YAML (a top-level ``vary:`` list). Every
            parameter outside the list is pinned to its centre.
        derived_yaml: Derived-parameter policy. Applied after the overlay, so a
            derived child tracks its post-overlay parents.
        proposal_temperature: ``T`` in ``pi_tilde = pi^(1/T)``. ``1.0`` means
            the proposal *is* the prior.
    """

    priors_csv: str
    submodel_priors_yaml: Optional[str] = None
    vary_policy: Optional[str] = None
    derived_yaml: Optional[str] = None
    proposal_temperature: float = 1.0

    def __post_init__(self) -> None:
        T = float(self.proposal_temperature)
        object.__setattr__(self, "proposal_temperature", T)
        if T <= 0:
            raise ValueError(f"proposal_temperature must be > 0, got {T}")
        if T != 1.0 and T <= _MIN_TEMPERATURE:
            raise ValueError(
                f"proposal_temperature={T} is at or below 1/2, where the "
                "importance weights pi/pi_tilde have infinite variance for "
                "Gaussian marginals. Reported summaries would not converge."
            )
        if self.submodel_priors_yaml is None:
            for field_name in ("vary_policy", "derived_yaml"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"{field_name} requires a submodel_priors_yaml: it "
                        "modifies the composite copula prior, which the "
                        "CSV-only path does not build."
                    )
            if T != 1.0:
                raise ValueError(
                    "proposal_temperature requires a submodel_priors_yaml. "
                    "Tempering is defined on the log-space copula prior (scale "
                    "every marginal sigma by sqrt(T), leave the correlation "
                    "alone); the CSV-only path samples families including "
                    "uniform and beta, for which that has no meaning."
                )

    @property
    def is_composite(self) -> bool:
        return self.submodel_priors_yaml is not None

    @property
    def is_tempered(self) -> bool:
        return self.proposal_temperature != 1.0

    def at_temperature(self, temperature: float) -> "PriorSpec":
        """Same distribution family, different proposal temperature."""
        return replace(self, proposal_temperature=temperature)

    def hash_bytes(self) -> bytes:
        """Content bytes identifying this spec, for hashing into a cache key.

        Files are hashed by *content*, not by path, so moving or renaming an
        input does not invalidate a pool while editing one does. A missing
        optional file contributes nothing, matching the loaders, which ignore
        it. The temperature contributes nothing at ``1.0``, so every cache
        entry created before the proposal/prior decoupling existed stays valid.
        """
        buf = Path(self.priors_csv).read_bytes()
        for tag, path in (
            (b"|submodel|", self.submodel_priors_yaml),
            (b"|vary_policy|", self.vary_policy),
            (b"|derived_yaml|", self.derived_yaml),
        ):
            if path is not None and Path(path).exists():
                buf += tag + Path(path).read_bytes()
        if self.is_tempered:
            buf += f"|T={self.proposal_temperature:.12g}".encode("utf-8")
        return buf

    def fingerprint(self, length: int = 16) -> str:
        """Short hex digest of :meth:`hash_bytes`."""
        return hashlib.sha256(self.hash_bytes()).hexdigest()[:length]

    def label(self) -> str:
        """Human-readable suffix naming the non-default choices in this spec."""
        bits = []
        if not self.is_composite:
            bits.append("csvonly")
        if self.vary_policy is not None:
            bits.append("overlay")
        if self.derived_yaml is not None:
            bits.append("derived")
        if self.is_tempered:
            bits.append(f"T{self.proposal_temperature:g}")
        return "_".join(bits)


# ---------------------------------------------------------------------------
# The CSV-only fallback, which has a sampler but no density
# ---------------------------------------------------------------------------
class CsvIndependentPrior:
    """Independent per-parameter prior read straight from the priors CSV.

    Each row is sampled from its declared ``distribution`` in *original* space.
    This is the fallback for runs with no composite prior, and it is
    deliberately sampler-only: the families are heterogeneous (lognormal,
    normal, uniform, beta), there is no copula, and nothing that consumes it
    needs a density. Asking for one raises rather than returning a plausible
    wrong number.

    The single implementation is the point. The previous arrangement had the
    cloud honour the ``distribution`` column while the embedding prior assumed
    lognormal for every row, which silently turned three Beta-distributed
    fractions into lognormals many orders of magnitude wide.
    """

    _SUPPORTED = ("lognormal", "normal", "uniform", "beta")

    def __init__(self, csv_path: str | Path):
        import pandas as pd

        self._df = pd.read_csv(csv_path)
        missing = {"name", "distribution", "dist_param1", "dist_param2"} - set(self._df.columns)
        if missing:
            raise ValueError(f"priors CSV is missing column(s): {sorted(missing)}")
        bad = sorted(set(self._df["distribution"]) - set(self._SUPPORTED))
        if bad:
            raise ValueError(
                f"unsupported distribution(s) in {csv_path}: {bad}. "
                f"Supported: {list(self._SUPPORTED)}"
            )
        self.param_names = self._df["name"].tolist()

    def __len__(self) -> int:
        return len(self.param_names)

    def sample_original(self, n: int, seed: int) -> np.ndarray:
        """``(n, d)`` draws in original parameter space."""
        rng = np.random.default_rng(seed)
        theta = np.zeros((n, len(self.param_names)))
        for i, row in enumerate(self._df.itertuples(index=False)):
            p1, p2 = float(row.dist_param1), float(row.dist_param2)
            if row.distribution == "lognormal":
                theta[:, i] = rng.lognormal(mean=p1, sigma=p2, size=n)
            elif row.distribution == "normal":
                theta[:, i] = rng.normal(loc=p1, scale=p2, size=n)
            elif row.distribution == "uniform":
                theta[:, i] = rng.uniform(low=p1, high=p2, size=n)
            else:  # beta; validated in __init__
                theta[:, i] = rng.beta(a=p1, b=p2, size=n)
        return theta

    def log_prob(self, value):  # pragma: no cover - the raise is the behaviour
        raise NotImplementedError(
            "the CSV-only prior is a sampler, not a density. Importance "
            "reweighting and tempering need the composite prior; build the spec "
            "with a submodel_priors_yaml."
        )


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------
@dataclass
class PriorPair:
    """The reporting prior and the training proposal, built together.

    ``prior is proposal`` exactly when the spec is untempered. That identity is
    the check that a run is behaving as it did before the decoupling existed,
    and it is cheaper to assert than to compare two distributions.
    """

    prior: object
    proposal: object
    param_names: list
    spec: PriorSpec
    population_prior: Optional[object] = None
    population_param_names: Optional[list] = None

    @property
    def is_tempered(self) -> bool:
        return self.prior is not self.proposal

    @property
    def has_density(self) -> bool:
        """False for the CSV-only path, which cannot be reweighted."""
        return not isinstance(self.prior, CsvIndependentPrior)

    @property
    def fingerprint(self) -> str:
        return self.spec.fingerprint()

    @property
    def n_params(self) -> int:
        return len(self.param_names)

    def sample_original(self, n: int, seed: int) -> np.ndarray:
        """``(n, d)`` draws from the *proposal*, in original parameter space.

        This is what a simulator consumes. It samples the proposal, not the
        prior, because that is the distribution the training cloud is meant to
        come from; reporting is corrected afterwards by :meth:`reweight`.
        """
        if isinstance(self.proposal, CsvIndependentPrior):
            return self.proposal.sample_original(n, seed)
        import torch

        torch.manual_seed(int(seed))
        with torch.no_grad():
            return np.exp(self.proposal.sample((n,)).numpy())

    def sample_log(self, n: int, seed: int) -> np.ndarray:
        """``(n, d)`` draws from the proposal, in log space."""
        if isinstance(self.proposal, CsvIndependentPrior):
            return np.log(self.proposal.sample_original(n, seed))
        import torch

        torch.manual_seed(int(seed))
        with torch.no_grad():
            return self.proposal.sample((n,)).numpy()

    def reweight(self, theta_log, **kwargs):
        """Importance weights carrying proposal draws onto the prior.

        Args:
            theta_log: ``(n, d)`` samples in log space, drawn from the proposal.
            **kwargs: forwarded to
                :func:`qsp_inference.inference.importance.reweight_to_prior`.

        Returns:
            A ``ReweightResult``. Untempered specs return uniform weights, which
            is correct and costs one evaluation; callers do not need to branch.
        """
        if not self.has_density:
            raise NotImplementedError(
                "the CSV-only prior has no density, so it cannot be reweighted."
            )
        from qsp_inference.inference.importance import reweight_to_prior

        return reweight_to_prior(theta_log, self.prior, self.proposal, **kwargs)

    def subset(self, indices: Sequence[int]) -> "PriorPair":
        """Restrict both distributions to a subset of parameters, in order."""
        if not self.has_density:
            raise NotImplementedError("the CSV-only prior does not support subset()")
        idx = list(indices)
        sub_prior = self.prior.subset(idx)
        sub_proposal = sub_prior if not self.is_tempered else self.proposal.subset(idx)
        return PriorPair(
            prior=sub_prior,
            proposal=sub_proposal,
            param_names=[self.param_names[i] for i in idx],
            spec=self.spec,
            population_prior=self.population_prior,
            population_param_names=self.population_param_names,
        )

    def summary(self) -> str:
        lines = [
            f"prior: {self.n_params} params, fingerprint {self.fingerprint}"
            + ("" if self.has_density else "  [CSV-only, no density]")
        ]
        if self.is_tempered:
            T = self.spec.proposal_temperature
            lines.append(
                f"proposal: pi^(1/{T:g})  (sigma -> {T ** 0.5:.3g}*sigma, "
                f"{self.n_varying()} varying params, correlation unchanged)"
            )
            lines.append("training draws from the proposal; reports reweight to the prior")
        else:
            lines.append("proposal: is the prior (T = 1)")
        return "\n".join(lines)

    def n_varying(self, pin_sigma: float = 1e-3) -> int:
        """Parameters the prior actually varies.

        Pinned parameters are not set to exactly zero sigma (a degenerate
        marginal breaks the copula); :func:`~qsp_inference.priors.copula_prior.
        compose_overlay_prior` gives them ``pin_sigma``, 1e-3 by default. So the
        count is of marginals wider than that, and a genuinely free parameter
        known to better than 0.1% in log space would be miscounted. This is a
        display number, not a decision input.
        """
        if not self.has_density:
            return self.n_params
        return sum(1 for m in self.prior._marginals if float(m.std()) > pin_sigma)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
def _load_vary_list(vary_policy: str | Path) -> list[str]:
    """Read the top-level ``vary:`` allowlist from a policy YAML."""
    import yaml

    with open(vary_policy) as f:
        data = yaml.safe_load(f)
    vary = list((data or {}).get("vary") or [])
    if not vary:
        raise ValueError(f"vary policy {vary_policy} has no non-empty `vary:` list")
    return vary


def build_prior_pair(
    spec: PriorSpec,
    *,
    load_population: bool = False,
    verbose: bool = False,
) -> PriorPair:
    """Build ``(pi, pi_tilde)`` from a spec.

    Composition order is fixed and load-bearing: composite centre, then the
    sigma-overlay, then derived parameters, then tempering. Derived children
    must see post-overlay parents, and the proposal must be a pure widening of
    the fully composed prior rather than of some intermediate.

    Args:
        spec: What distribution to build.
        load_population: Also load the ``population`` block from the submodel
            YAML, when present. Absent is not an error.
        verbose: Print the composition, as the workflow scripts expect.

    Returns:
        A :class:`PriorPair`. ``pair.prior is pair.proposal`` when the spec is
        untempered.
    """
    if not spec.is_composite:
        prior = CsvIndependentPrior(spec.priors_csv)
        if verbose:
            print(f"  Prior: {len(prior)} params (CSV-only, no composite)")
        return PriorPair(
            prior=prior,
            proposal=prior,
            param_names=list(prior.param_names),
            spec=spec,
        )

    from qsp_inference.priors.copula_prior import (
        load_composite_prior_log,
        load_overlay_prior_log,
    )

    derived = spec.derived_yaml
    if spec.vary_policy is not None:
        prior, names = load_overlay_prior_log(
            spec.submodel_priors_yaml,
            spec.priors_csv,
            vary_params=_load_vary_list(spec.vary_policy),
            derived_yaml=derived,
        )
    else:
        prior, names = load_composite_prior_log(
            spec.submodel_priors_yaml, spec.priors_csv, derived_yaml=derived
        )

    population_prior = None
    population_names = None
    if load_population:
        from qsp_inference.priors.copula_prior import load_copula_prior_log

        try:
            population_prior, population_names = load_copula_prior_log(
                spec.submodel_priors_yaml, block="population"
            )
        except KeyError:
            population_prior, population_names = None, None

    if spec.is_tempered:
        from qsp_inference.priors.copula_prior import temper_prior

        proposal = temper_prior(prior, spec.proposal_temperature)
        _assert_aligned(proposal, names)
    else:
        proposal = prior

    pair = PriorPair(
        prior=prior,
        proposal=proposal,
        param_names=list(names),
        spec=spec,
        population_prior=population_prior,
        population_param_names=(list(population_names) if population_names is not None else None),
    )
    if verbose:
        print("  " + pair.summary().replace("\n", "\n  "))
        if load_population:
            if population_names is None:
                print(
                    "  Population omega block: absent (submodel_priors.yaml built "
                    "without the population pass)"
                )
            else:
                print(f"  Population omega block: {len(population_names)} param(s)")
    return pair


def _assert_aligned(proposal, names: Iterable[str]) -> None:
    """Fail loudly if the proposal is not over the same parameters, in order.

    The importance weight is a difference of two log-densities evaluated on one
    theta matrix. Misaligned columns give finite, plausible weights that are
    wrong for every sample, and nothing downstream can detect it.
    """
    got = list(getattr(proposal, "param_names", []) or [])
    want = list(names)
    if got == want:
        return
    if len(got) != len(want):
        detail = f"{len(got)} params vs {len(want)}"
    else:
        j = next(i for i, (a, b) in enumerate(zip(got, want)) if a != b)
        detail = f"first mismatch at index {j}: {got[j]!r} vs {want[j]!r}"
    raise RuntimeError(
        f"tempered proposal is not aligned with the prior ({detail}). Importance "
        "weights over misaligned priors are finite, plausible, and wrong for "
        "every sample."
    )

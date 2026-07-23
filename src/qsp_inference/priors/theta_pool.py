"""The deterministic, indexable pool of theta a run simulates.

A pool is ``n_total`` rows drawn once from the training proposal and reused by
every scenario, so that row ``i`` means the same parameter vector everywhere.
Joint multi-scenario inference depends on that: alignment across scenarios is an
integer-set intersection on ``sample_index`` rather than a positional join, and
drift in the pool destroys it.

**Why the pool lives next to the priors.** A pool is a distribution plus a
sampling plan. Splitting those across repositories is what produced the
divergence documented in :mod:`qsp_inference.priors.inference_prior`: the code
that drew the cloud and the code that evaluated its density composed the same
prior by different routes. Here :class:`ThetaPoolSpec` composes a
:class:`~qsp_inference.priors.inference_prior.PriorSpec` with the three things
that identify a *sample* rather than a distribution (seed, size, restriction),
and one object answers both "what were these drawn from" and "which pool is
this".

**Identity is the load-bearing part.** A pool is cached on disk under a hash of
its spec. If a change to the distribution fails to change that hash, the cache
answers with rows drawn from something else, and nothing downstream can tell:
the matrix has the right shape, the values are plausible, and every diagnostic
computed from it is quietly wrong. So the fingerprint covers file *content*, not
paths, and every knob that reaches the sampler is in it.

**Restriction.** With a restriction classifier the pool is built by rejection
sampling against it, so simulator time is not spent on draws that always fail.
The retry schedule is deterministic in the seed, which keeps the cache key a
function of the inputs alone.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np

from qsp_inference.priors.inference_prior import PriorSpec, build_prior_pair

__all__ = [
    "ThetaPoolSpec",
    "get_theta_pool",
    "theta_for_indices",
]

PathLike = Union[str, Path]


@dataclass(frozen=True)
class ThetaPoolSpec:
    """A distribution plus the sampling plan that turns it into a pool.

    Attributes:
        prior: What the rows are drawn from. Note this samples the *proposal*,
            not the reporting prior: see
            :meth:`~qsp_inference.priors.inference_prior.PriorPair.sample_original`.
        seed: Base RNG seed. Restriction retries derive their seeds from it, so
            the pool stays a deterministic function of the spec.
        n_total: Number of rows.
        restriction_classifier_dir: Directory holding a serialized
            :class:`~qsp_inference.inference.restriction.RestrictionClassifier`.
            ``None`` disables rejection sampling.
        restriction_threshold: Accept draws scoring at or above this.
        restriction_oversample_factor: First batch is this multiple of
            ``n_total``.
        restriction_max_oversample: Give up once the factor reaches this
            multiple of the starting factor, rather than looping forever on a
            threshold nothing can satisfy.
        classifier_feature_fills: Values for features the classifier expects but
            the live prior no longer carries, used when the two have drifted.
    """

    prior: PriorSpec
    seed: int
    n_total: int
    restriction_classifier_dir: Optional[str] = None
    restriction_threshold: float = 0.5
    restriction_oversample_factor: float = 2.5
    restriction_max_oversample: int = 8
    classifier_feature_fills: Optional[Mapping[str, float]] = field(default=None)

    def __post_init__(self) -> None:
        if self.n_total <= 0:
            raise ValueError(f"n_total must be positive, got {self.n_total}")

    @property
    def is_restricted(self) -> bool:
        return self.restriction_classifier_dir is not None

    def _classifier_bytes(self) -> bytes:
        """Content bytes for the classifier, threshold and fills.

        The classifier's own file contents are hashed, not its path: retraining
        in place must invalidate the pool, since the accepted region moves.
        """
        if self.restriction_classifier_dir is None:
            return b""
        d = Path(self.restriction_classifier_dir)
        buf = b"|classifier|"
        for fname in ("classifier.pkl", "metadata.json"):
            f = d / fname
            if f.exists():
                buf += f.read_bytes()
        buf += f"|tau={self.restriction_threshold:.6f}".encode("utf-8")
        if self.classifier_feature_fills:
            # Sorted so the hash does not depend on dict ordering.
            fills = ",".join(
                f"{k}={float(v):.12g}" for k, v in sorted(self.classifier_feature_fills.items())
            )
            buf += f"|fills={fills}".encode("utf-8")
        return buf

    def fingerprint(self, length: int = 16) -> str:
        """Short hex digest identifying this pool.

        Note what is *not* here: the oversample factor and retry ceiling. They
        change how long the search takes, not which rows come out, because the
        retry seeds are derived from ``seed`` and acceptance is a deterministic
        function of the draw. Including them would fragment the cache for no
        gain.
        """
        h = hashlib.sha256()
        h.update(self.prior.hash_bytes())
        h.update(f"|seed={int(self.seed)}".encode("utf-8"))
        h.update(f"|n={int(self.n_total)}".encode("utf-8"))
        h.update(self._classifier_bytes())
        return h.hexdigest()[:length]

    def cache_path(self, cache_dir: PathLike = "cache/theta_pools") -> Path:
        label = self.prior.label()
        suffix = ("_restricted" if self.is_restricted else "") + (f"_{label}" if label else "")
        return Path(cache_dir) / f"theta_pool_{self.fingerprint()}_n{self.n_total}{suffix}.npy"

    def legacy_cache_path(self, cache_dir: PathLike = "cache/theta_pools") -> Path:
        """Path this pool had before construction moved into this package.

        Read-only compatibility. Pools are expensive (each row is a simulation
        downstream), so a spec whose new path is missing but whose legacy path
        exists loads the legacy file rather than regenerating. The byte sequence
        below is a transcription of the old
        ``qsp_hpc.simulation.theta_pool.theta_pool_cache_path`` and must not be
        "cleaned up": its only job is to reproduce hashes computed by code that
        no longer exists.
        """
        p = self.prior
        h = hashlib.sha256()
        h.update(Path(p.priors_csv).read_text().encode("utf-8"))
        if p.submodel_priors_yaml is not None and Path(p.submodel_priors_yaml).exists():
            h.update(Path(p.submodel_priors_yaml).read_text().encode("utf-8"))
        h.update(str(int(self.seed)).encode("utf-8"))
        h.update(str(int(self.n_total)).encode("utf-8"))
        h.update(self._classifier_bytes())
        overlay_suffix = ""
        if p.vary_policy is not None:
            if Path(p.vary_policy).exists():
                h.update(b"|vary_policy|")
                h.update(Path(p.vary_policy).read_text().encode("utf-8"))
            overlay_suffix = "_overlay"
        derived_suffix = ""
        if p.derived_yaml is not None:
            if Path(p.derived_yaml).exists():
                h.update(b"|derived_yaml|")
                h.update(Path(p.derived_yaml).read_text().encode("utf-8"))
            derived_suffix = "_derived"
        proposal_suffix = ""
        if p.is_tempered:
            h.update(f"|proposal_temperature={p.proposal_temperature:.12g}".encode("utf-8"))
            proposal_suffix = f"_T{p.proposal_temperature:g}"
        suffix = (
            ("_restricted" if self.is_restricted else "")
            + overlay_suffix
            + derived_suffix
            + proposal_suffix
        )
        return Path(cache_dir) / f"theta_pool_{h.hexdigest()[:16]}_n{self.n_total}{suffix}.npy"


def _draw(spec: ThetaPoolSpec, n: int, seed: int) -> tuple[np.ndarray, list]:
    pair = build_prior_pair(spec.prior)
    return pair.sample_original(n, seed), list(pair.param_names)


def get_theta_pool(
    spec: ThetaPoolSpec,
    cache_dir: PathLike = "cache/theta_pools",
    *,
    allow_legacy_cache: bool = True,
) -> np.ndarray:
    """Return the ``(n_total, n_params)`` pool, generating it if needed.

    Args:
        spec: Which pool.
        cache_dir: Where pools live.
        allow_legacy_cache: Fall back to reading a pool cached under the
            pre-consolidation path. Set False to force regeneration, which is
            the honest thing to do if you suspect a legacy pool is not what its
            name claims.

    Returns:
        Rows in *original* parameter space.
    """
    path = spec.cache_path(cache_dir)
    if path.exists():
        return np.load(path)
    if allow_legacy_cache:
        legacy = spec.legacy_cache_path(cache_dir)
        if legacy.exists():
            return np.load(legacy)

    if not spec.is_restricted:
        theta, _ = _draw(spec, spec.n_total, spec.seed)
    else:
        from qsp_inference.inference.restriction import RestrictionClassifier

        clf = RestrictionClassifier.load(spec.restriction_classifier_dir)
        accepted: list[np.ndarray] = []
        n_accepted = 0
        factor = float(spec.restriction_oversample_factor)
        ceiling = spec.restriction_oversample_factor * spec.restriction_max_oversample
        attempt = 0
        while n_accepted < spec.n_total:
            attempt += 1
            if attempt > 1 and factor >= ceiling:
                raise RuntimeError(
                    f"restricted pool: could not reach {spec.n_total} accepted at "
                    f"threshold {spec.restriction_threshold}; got {n_accepted} "
                    f"after oversample factor {factor:.1f}"
                )
            # Seed offset per attempt, so the retry schedule is deterministic
            # and the cache key stays a function of the spec alone.
            batch, names = _draw(spec, int(factor * spec.n_total), spec.seed + attempt - 1)
            if list(names) == list(clf.feature_order):
                keep = clf.accept(batch, threshold=spec.restriction_threshold)
            else:
                # The live prior has drifted relative to the classifier (params
                # added or retired); project onto the classifier's feature order.
                keep = clf.accept_named(
                    batch,
                    names,
                    fills=spec.classifier_feature_fills,
                    threshold=spec.restriction_threshold,
                )
            accepted.append(batch[keep])
            n_accepted += int(keep.sum())
            factor *= 2.0
        theta = np.concatenate(accepted, axis=0)[: spec.n_total]

    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, theta)
    return theta


def theta_for_indices(
    indices,
    spec: ThetaPoolSpec,
    cache_dir: PathLike = "cache/theta_pools",
    **kwargs,
) -> np.ndarray:
    """Slice the pool by integer ``sample_index``.

    ``indices`` may be unordered or have gaps; rows come back in the order asked
    for. Out-of-range indices raise rather than wrapping, because a silent wrap
    would pair a simulation's outputs with the wrong parameters.
    """
    pool = get_theta_pool(spec, cache_dir, **kwargs)
    idx = np.asarray(indices, dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= spec.n_total):
        raise IndexError(
            f"sample_index out of range: min={idx.min()} max={idx.max()} " f"n_total={spec.n_total}"
        )
    return pool[idx]

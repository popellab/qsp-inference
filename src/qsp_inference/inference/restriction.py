"""Classifier-based prior restriction for simulation-based inference.

Given a pool of ``(theta, valid_mask)`` pairs — where ``valid_mask[i]`` is
``True`` iff theta ``i`` produced a usable simulation output — fit a simple
classifier that predicts validity from theta. Use it to reject low-p(valid)
theta draws from the prior *before* submitting them to the simulator,
raising the fraction of usable sims per HPC dollar.

This is the "truncated sequential neural posterior estimation" idea from
Deistler et al. 2022, stripped down to its simplest form: sklearn boosted
trees on log-theta, no sbi-prior integration required. The downside vs
``sbi.utils.RestrictionEstimator`` is that this doesn't hand you a
proper RestrictedPrior object with a log-prob; it just scores candidate
thetas. The upside is that it works with any prior whose samples you
can produce as a numpy array — so composite copula priors, hierarchical
priors, etc. all work identically.

Trained classifiers and metadata are designed to round-trip through disk
(pickle + JSON) so they can travel to remote workers alongside sim jobs.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np


@dataclass
class RestrictionClassifier:
    """Trained classifier + feature metadata for prior restriction.

    ``feature_order`` is the parameter-name ordering the classifier expects.
    ``input_transform`` is ``"log"`` (default) or ``"identity"``; priors are
    usually log-normal so log-theta is the natural feature space. Callers
    scoring new thetas MUST present them in this order and in the original
    (un-logged) scale — the scorer applies the transform.
    """

    model: Any
    feature_order: list[str]
    input_transform: str = "log"
    default_threshold: float = 0.5
    # Populated when training; callers can also compute them fresh.
    cv_auc: Optional[float] = None
    in_sample_auc: Optional[float] = None
    baseline_survival: Optional[float] = None
    threshold_curve: Optional[list[dict]] = None
    n_train: Optional[int] = None

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _transform(self, theta: np.ndarray) -> np.ndarray:
        if self.input_transform == "log":
            return np.log(theta)
        if self.input_transform == "identity":
            return theta
        raise ValueError(f"unknown input_transform: {self.input_transform}")

    def score(self, theta: np.ndarray) -> np.ndarray:
        """Return ``P(valid | theta)`` for each row of ``theta``.

        ``theta`` shape: ``(n, n_features)`` in the original scale (not
        log-transformed). Column order must match ``feature_order``.
        """
        if theta.shape[1] != len(self.feature_order):
            raise ValueError(
                f"theta has {theta.shape[1]} columns; expected "
                f"{len(self.feature_order)} matching feature_order"
            )
        X = self._transform(theta)
        return self.model.predict_proba(X)[:, 1]

    def accept(self, theta: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Boolean mask: rows with ``P(valid | theta) >= threshold``."""
        tau = self.default_threshold if threshold is None else threshold
        return self.score(theta) >= tau

    def project(
        self,
        theta: np.ndarray,
        theta_feature_names: Sequence[str],
        fills: Optional[Mapping[str, float]] = None,
    ) -> np.ndarray:
        """Project ``theta`` (in caller-side feature order) onto ``feature_order``.

        Drops caller-side columns that the classifier doesn't know, fills
        classifier-side features missing from the caller using ``fills``.
        Use this when the live prior has drifted relative to the classifier
        (added/retired params) but is still close enough that the classifier
        retains predictive value.

        Args:
            theta: ``(n, n_caller_features)`` in caller-side column order.
            theta_feature_names: parameter names matching ``theta`` columns.
            fills: ``{name: value}`` for classifier features missing from
                ``theta_feature_names``. Required if any classifier features
                aren't in ``theta_feature_names``.

        Returns:
            ``(n, n_features)`` ndarray in ``feature_order`` column layout,
            ready for ``score`` or ``accept``.
        """
        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 2:
            raise ValueError(f"theta must be 2D, got shape {theta.shape}")
        if theta.shape[1] != len(theta_feature_names):
            raise ValueError(
                f"theta has {theta.shape[1]} columns; theta_feature_names has "
                f"{len(theta_feature_names)} entries"
            )
        name_to_col = {n: i for i, n in enumerate(theta_feature_names)}
        fills = dict(fills) if fills is not None else {}
        out = np.empty((theta.shape[0], len(self.feature_order)), dtype=np.float64)
        missing: list[str] = []
        for j, feat in enumerate(self.feature_order):
            col = name_to_col.get(feat)
            if col is not None:
                out[:, j] = theta[:, col]
            elif feat in fills:
                out[:, j] = float(fills[feat])
            else:
                missing.append(feat)
        if missing:
            raise ValueError(
                "classifier feature(s) absent from theta_feature_names and not in "
                f"fills: {missing!r}"
            )
        return out

    def score_named(
        self,
        theta: np.ndarray,
        theta_feature_names: Sequence[str],
        fills: Optional[Mapping[str, float]] = None,
    ) -> np.ndarray:
        """``score`` with automatic projection. See :meth:`project`."""
        return self.score(self.project(theta, theta_feature_names, fills))

    def accept_named(
        self,
        theta: np.ndarray,
        theta_feature_names: Sequence[str],
        fills: Optional[Mapping[str, float]] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """``accept`` with automatic projection. See :meth:`project`."""
        return self.accept(
            self.project(theta, theta_feature_names, fills), threshold=threshold
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, out_dir: Path | str) -> None:
        """Serialize to ``out_dir/classifier.pkl`` + ``out_dir/metadata.json``."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classifier.pkl").write_bytes(pickle.dumps(self.model))
        meta = {
            "feature_order": self.feature_order,
            "n_features": len(self.feature_order),
            "input_transform": self.input_transform,
            "default_threshold": self.default_threshold,
            "cv_auc": self.cv_auc,
            "in_sample_auc": self.in_sample_auc,
            "baseline_survival": self.baseline_survival,
            "threshold_curve": self.threshold_curve,
            "n_train": self.n_train,
            "model_class": type(self.model).__module__ + "." + type(self.model).__name__,
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, out_dir: Path | str) -> "RestrictionClassifier":
        out_dir = Path(out_dir)
        model = pickle.loads((out_dir / "classifier.pkl").read_bytes())
        meta = json.loads((out_dir / "metadata.json").read_text())
        return cls(
            model=model,
            feature_order=list(meta["feature_order"]),
            input_transform=meta.get("input_transform", "log"),
            default_threshold=float(meta.get("default_threshold", 0.5)),
            cv_auc=meta.get("cv_auc"),
            in_sample_auc=meta.get("in_sample_auc"),
            baseline_survival=meta.get("baseline_survival"),
            threshold_curve=meta.get("threshold_curve"),
            n_train=meta.get("n_train"),
        )


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
def train_restriction_classifier(
    theta: np.ndarray,
    valid_mask: np.ndarray,
    feature_order: Sequence[str],
    input_transform: str = "log",
    default_threshold: float = 0.5,
    cv_folds: int = 5,
    thresholds_for_curve: Sequence[float] = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
    model: Optional[Any] = None,
    random_state: int = 0,
) -> RestrictionClassifier:
    """Fit a ``HistGradientBoostingClassifier`` on ``(theta, valid_mask)``.

    Args:
        theta: ``(n, n_features)`` in the original scale (not yet log'd).
        valid_mask: ``(n,)`` bool/int, ``True`` where the simulator produced
            a usable output for the corresponding theta.
        feature_order: parameter names, one per column of ``theta``. Saved
            into the classifier so remote scorers can self-check.
        input_transform: ``"log"`` (default; applied to theta before fitting)
            or ``"identity"``.
        default_threshold: stored on the classifier; callers can override at
            score time.
        cv_folds: 0 disables CV (faster; in-sample AUC only).
        thresholds_for_curve: populate ``threshold_curve`` in metadata at
            these τ values (accept_frac, survival, yield_per_draw).
        model: optional pre-configured sklearn classifier. Defaults to
            ``HistGradientBoostingClassifier(max_iter=400, max_depth=5,
            learning_rate=0.05, random_state=random_state)``.

    Returns:
        Trained ``RestrictionClassifier`` wrapping ``model``.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score

    feature_order = list(feature_order)
    if theta.shape[1] != len(feature_order):
        raise ValueError(
            f"theta has {theta.shape[1]} columns; feature_order has "
            f"{len(feature_order)} entries"
        )
    y = np.asarray(valid_mask).astype(np.int32)

    if input_transform == "log":
        X = np.log(theta)
    elif input_transform == "identity":
        X = theta
    else:
        raise ValueError(f"unknown input_transform: {input_transform}")

    if model is None:
        model = HistGradientBoostingClassifier(
            max_iter=400, max_depth=5, learning_rate=0.05, random_state=random_state
        )

    cv_auc = None
    if cv_folds and cv_folds > 1:
        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring="roc_auc", n_jobs=-1)
        cv_auc = float(cv_scores.mean())

    model.fit(X, y)
    in_sample_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))

    # Threshold → (accept_frac, survival, yield) curve for docs/telemetry.
    p = model.predict_proba(X)[:, 1]
    curve = []
    for tau in thresholds_for_curve:
        acc = p >= tau
        af = float(acc.mean())
        surv = float(y[acc].mean()) if acc.any() else 0.0
        curve.append({"threshold": float(tau), "accept_frac": af,
                      "survival": surv, "yield_per_draw": af * surv})

    return RestrictionClassifier(
        model=model,
        feature_order=feature_order,
        input_transform=input_transform,
        default_threshold=default_threshold,
        cv_auc=cv_auc,
        in_sample_auc=in_sample_auc,
        baseline_survival=float(y.mean()),
        threshold_curve=curve,
        n_train=int(len(y)),
    )


# ----------------------------------------------------------------------
# Rejection sampling against a prior
# ----------------------------------------------------------------------
def sample_restricted(
    classifier: RestrictionClassifier,
    prior_sample_fn,
    n_accepted: int,
    threshold: Optional[float] = None,
    batch_size: int = 10_000,
    max_draws: Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """Draw from a prior until ``n_accepted`` samples pass the classifier.

    Args:
        classifier: trained ``RestrictionClassifier``.
        prior_sample_fn: callable ``n -> np.ndarray`` of shape ``(n, n_features)``.
            Columns in the same order as ``classifier.feature_order``.
        n_accepted: target number of accepted samples.
        threshold: τ for ``p(valid) >= τ``. Defaults to the classifier's
            ``default_threshold``.
        batch_size: prior draws per rejection batch.
        max_draws: safety cap on total prior draws (``None`` = unbounded).
            Useful when survival rates are low enough to risk infinite loops.

    Returns:
        ``(theta_accepted, total_draws)`` — theta has shape
        ``(n_accepted, n_features)``; ``total_draws`` is the number of
        prior samples it took to reach ``n_accepted``.
    """
    accepted: list[np.ndarray] = []
    n_so_far = 0
    total = 0
    while n_so_far < n_accepted:
        if max_draws is not None and total >= max_draws:
            raise RuntimeError(
                f"sample_restricted: exceeded max_draws ({max_draws}); "
                f"got only {n_so_far}/{n_accepted} accepted"
            )
        batch = prior_sample_fn(batch_size)
        total += batch.shape[0]
        keep = classifier.accept(batch, threshold=threshold)
        if keep.any():
            accepted.append(batch[keep])
            n_so_far += int(keep.sum())
    theta = np.concatenate(accepted, axis=0)[:n_accepted]
    return theta, total

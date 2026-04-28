"""Unit tests for the classifier-based prior restriction module."""
import numpy as np
import pytest

from qsp_inference.inference import (
    RestrictionClassifier,
    train_restriction_classifier,
    sample_restricted,
)


def _synthetic_pool(n, n_features=3, rng=None, seed=0):
    """Valid iff the first feature is > e^0 = 1 (i.e. log-feature[0] > 0).

    Features are drawn lognormal so the classifier has to learn the log-space
    threshold, matching the real use case.
    """
    rng = rng or np.random.default_rng(seed)
    theta = np.exp(rng.normal(size=(n, n_features)))
    valid = theta[:, 0] > 1.0
    return theta, valid


def test_train_and_score():
    theta, valid = _synthetic_pool(4000, seed=0)
    names = [f"p{i}" for i in range(theta.shape[1])]
    clf = train_restriction_classifier(theta, valid, names, cv_folds=3)
    assert clf.feature_order == names
    assert clf.n_train == 4000
    assert clf.cv_auc is not None and clf.cv_auc > 0.9
    assert clf.baseline_survival == pytest.approx(valid.mean(), rel=1e-6)
    # Threshold curve populated + monotone survival (higher τ → higher survival)
    assert clf.threshold_curve is not None
    survivals = [row["survival"] for row in clf.threshold_curve]
    assert all(b >= a - 1e-6 for a, b in zip(survivals, survivals[1:]))

    p = clf.score(theta)
    assert p.shape == (4000,)
    assert p.min() >= 0.0 and p.max() <= 1.0
    # Points clearly in the valid region score high; clearly in invalid region score low.
    theta_clear_valid = np.tile(np.array([10.0, 1.0, 1.0]), (10, 1))
    theta_clear_invalid = np.tile(np.array([0.1, 1.0, 1.0]), (10, 1))
    assert clf.score(theta_clear_valid).mean() > 0.7
    assert clf.score(theta_clear_invalid).mean() < 0.3


def test_accept_uses_default_threshold_and_override():
    theta, valid = _synthetic_pool(2000, seed=1)
    names = [f"p{i}" for i in range(theta.shape[1])]
    clf = train_restriction_classifier(theta, valid, names, cv_folds=0,
                                       default_threshold=0.5)
    assert clf.accept(theta).sum() == (clf.score(theta) >= 0.5).sum()
    assert clf.accept(theta, threshold=0.9).sum() == (clf.score(theta) >= 0.9).sum()


def test_round_trip_save_load(tmp_path):
    theta, valid = _synthetic_pool(1500, seed=2)
    names = ["a", "b", "c"]
    clf = train_restriction_classifier(theta, valid, names, cv_folds=0)
    clf.save(tmp_path / "r")
    loaded = RestrictionClassifier.load(tmp_path / "r")
    assert loaded.feature_order == clf.feature_order
    assert loaded.input_transform == clf.input_transform
    assert loaded.default_threshold == clf.default_threshold
    np.testing.assert_allclose(loaded.score(theta), clf.score(theta))


def test_sample_restricted_hits_target_count():
    theta, valid = _synthetic_pool(3000, seed=3)
    names = ["a", "b", "c"]
    clf = train_restriction_classifier(theta, valid, names, cv_folds=0)

    rng = np.random.default_rng(42)
    def prior_sample_fn(n):
        return np.exp(rng.normal(size=(n, 3)))

    theta_acc, total = sample_restricted(clf, prior_sample_fn, n_accepted=500,
                                         threshold=0.5, batch_size=500)
    assert theta_acc.shape == (500, 3)
    # Acceptance rate ≈ baseline survival (~0.5); so ~1000 draws is plausible.
    assert total >= 500
    # Every returned row must actually score >= 0.5.
    assert (clf.score(theta_acc) >= 0.5).all()


def test_sample_restricted_raises_on_max_draws():
    # A classifier with an impossibly-high threshold against a prior that
    # almost never scores high — should raise RuntimeError under max_draws.
    theta, valid = _synthetic_pool(2000, seed=4)
    names = ["a", "b", "c"]
    clf = train_restriction_classifier(theta, valid, names, cv_folds=0)

    def prior_sample_fn(n):
        # Draw from a region that's mostly < 0 in log-space → invalid.
        return np.exp(np.random.default_rng(0).normal(loc=-3.0, size=(n, 3)))

    with pytest.raises(RuntimeError, match="max_draws"):
        sample_restricted(clf, prior_sample_fn, n_accepted=100,
                          threshold=0.99, batch_size=200, max_draws=1000)


def test_identity_transform():
    rng = np.random.default_rng(5)
    theta = rng.normal(size=(2000, 2))  # not positive — log would fail
    valid = theta[:, 0] > 0
    clf = train_restriction_classifier(theta, valid, ["x", "y"],
                                       input_transform="identity", cv_folds=0)
    assert clf.input_transform == "identity"
    # Score a positive vs negative first coord; positive should dominate.
    pos = clf.score(np.array([[2.0, 0.0]]))
    neg = clf.score(np.array([[-2.0, 0.0]]))
    assert pos[0] > neg[0]


def test_project_drops_extra_columns():
    theta, valid = _synthetic_pool(1000, seed=10)
    clf = train_restriction_classifier(theta, valid, ["a", "b", "c"], cv_folds=0)
    # Caller has 4 columns: a, b, c, d_extra. Projection should drop d_extra.
    rng = np.random.default_rng(11)
    extra_col = rng.lognormal(size=(theta.shape[0], 1))
    theta_extra = np.concatenate([theta, extra_col], axis=1)
    proj = clf.project(theta_extra, ["a", "b", "c", "d_extra"])
    np.testing.assert_array_equal(proj, theta)


def test_project_fills_missing_columns():
    theta, valid = _synthetic_pool(1000, seed=12)
    clf = train_restriction_classifier(theta, valid, ["a", "b", "c"], cv_folds=0)
    # Caller is missing column "b"; supply via fills.
    theta_drop = theta[:, [0, 2]]
    proj = clf.project(theta_drop, ["a", "c"], fills={"b": 7.5})
    np.testing.assert_array_equal(proj[:, 0], theta[:, 0])
    np.testing.assert_array_equal(proj[:, 2], theta[:, 2])
    assert (proj[:, 1] == 7.5).all()


def test_project_reorders_columns():
    theta, valid = _synthetic_pool(1000, seed=13)
    clf = train_restriction_classifier(theta, valid, ["a", "b", "c"], cv_folds=0)
    # Caller hands theta in scrambled order [c, a, b].
    theta_scrambled = theta[:, [2, 0, 1]]
    proj = clf.project(theta_scrambled, ["c", "a", "b"])
    np.testing.assert_array_equal(proj, theta)


def test_project_raises_on_unfilled_missing():
    theta, valid = _synthetic_pool(500, seed=14)
    clf = train_restriction_classifier(theta, valid, ["a", "b", "c"], cv_folds=0)
    with pytest.raises(ValueError, match="absent from theta_feature_names"):
        clf.project(theta[:, :2], ["a", "b"])  # "c" missing, no fill provided


def test_accept_named_matches_accept_after_project():
    theta, valid = _synthetic_pool(2000, seed=15)
    clf = train_restriction_classifier(theta, valid, ["a", "b", "c"], cv_folds=0)
    rng = np.random.default_rng(16)
    extra = rng.lognormal(size=(theta.shape[0], 1))
    theta_extra = np.concatenate([theta, extra], axis=1)
    direct = clf.accept(theta)
    via_named = clf.accept_named(theta_extra, ["a", "b", "c", "d_extra"])
    np.testing.assert_array_equal(direct, via_named)

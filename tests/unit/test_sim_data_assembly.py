"""Unit tests for joint_finite_mask and train_test_split_indices.

These are the multi-scenario assembly utilities around the per-scenario
transform: which draws survived in every scenario, and a reproducible split that
does not perturb the global NumPy RNG.
"""
import numpy as np
import pytest

from qsp_inference.inference.data_processing import (
    joint_finite_mask,
    train_test_split_indices,
)


# --- joint_finite_mask ------------------------------------------------------

def test_mask_is_and_across_scenarios():
    a = np.array([[1.0, 2.0], [np.nan, 1.0], [3.0, 4.0], [5.0, 6.0]])
    b = np.array([[1.0, 1.0], [1.0, 1.0], [np.inf, 1.0], [2.0, 2.0]])
    mask = joint_finite_mask([a, b])
    # row 1 fails in a (nan), row 2 fails in b (inf); rows 0 and 3 survive both.
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_single_array_is_its_own_finite_mask():
    a = np.array([[1.0], [np.nan], [2.0]])
    np.testing.assert_array_equal(joint_finite_mask([a]), [True, False, True])


def test_all_finite_all_true():
    a = np.ones((5, 3))
    assert joint_finite_mask([a, a, a]).all()


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one array"):
        joint_finite_mask([])


def test_row_count_mismatch_raises():
    a = np.ones((4, 2))
    b = np.ones((3, 2))
    with pytest.raises(ValueError, match="row-count mismatch"):
        joint_finite_mask([a, b])


# --- train_test_split_indices ----------------------------------------------

def test_split_matches_legacy_global_rng_idiom():
    """The whole point: indices are byte-identical to the retired
    ``np.random.seed(seed); np.random.permutation(n)`` split, so existing
    train/test partitions (and cached bundles) are unchanged."""
    n, seed = 1000, 2027
    np.random.seed(seed)
    perm = np.random.permutation(n)
    n_test = int(n * 0.1)
    exp_train, exp_test = perm[: n - n_test], perm[n - n_test:]

    train_idx, test_idx = train_test_split_indices(n, seed=seed)
    np.testing.assert_array_equal(train_idx, exp_train)
    np.testing.assert_array_equal(test_idx, exp_test)


def test_split_does_not_touch_global_rng():
    """A pure primitive: calling it must not advance the global NumPy RNG, or it
    would silently shift any bare np.random draw that follows it."""
    np.random.seed(123)
    before = np.random.get_state()
    train_test_split_indices(500, seed=999)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2] == after[2]


def test_split_partitions_all_rows_without_overlap():
    n = 777
    train_idx, test_idx = train_test_split_indices(n, seed=5)
    assert len(train_idx) + len(test_idx) == n
    assert set(train_idx).isdisjoint(test_idx)
    assert set(train_idx) | set(test_idx) == set(range(n))
    assert len(test_idx) == int(n * 0.1)


def test_test_fraction_respected():
    n = 1000
    _, test_idx = train_test_split_indices(n, seed=1, test_fraction=0.25)
    assert len(test_idx) == 250


def test_subsample_keeps_requested_train_count_and_is_seeded():
    n = 2000
    tr1, te1 = train_test_split_indices(n, seed=7, subsample=300)
    tr2, te2 = train_test_split_indices(n, seed=7, subsample=300)
    assert len(tr1) == 300
    np.testing.assert_array_equal(tr1, tr2)       # deterministic
    np.testing.assert_array_equal(te1, te2)
    # subsampled train is a subset of the full train partition
    full_train, _ = train_test_split_indices(n, seed=7)
    assert set(tr1).issubset(full_train)
    # test set is never subsampled
    assert len(te1) == int(n * 0.1)


def test_subsample_larger_than_train_is_noop():
    n = 100
    full_train, _ = train_test_split_indices(n, seed=3)
    tr, _ = train_test_split_indices(n, seed=3, subsample=10_000)
    np.testing.assert_array_equal(tr, full_train)

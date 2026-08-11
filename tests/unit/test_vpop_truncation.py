"""Unit tests for the variance a truncated basis would otherwise discard.

Holding a direction at ``mu_0`` keeps the population spread ``omega`` describes
and throws away eq:muprior's epistemic uncertainty about where the centre sits.
``truncation_V`` is what puts the second back, and the claims here are that it
carries exactly the complement, that an asserted coordinate is excluded from it,
and that it vanishes when the basis spans everything.
"""
import numpy as np
import pytest

from qsp_inference.vpop.blocks import truncation_V


def _basis(P, k, seed=0):
    Q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((P, k)))
    return Q[:, :k]


class TestItCarriesTheComplement:
    def test_full_rank_basis_carries_nothing(self):
        # Nothing is dropped, so nothing is owed.
        A = np.random.default_rng(1).standard_normal((6, 4))
        out = truncation_V(A, _basis(4, 4), [6])
        assert np.abs(out[0]).max() == pytest.approx(0.0, abs=1e-20)

    def test_empty_basis_carries_the_whole_prior_predictive(self):
        A = np.random.default_rng(2).standard_normal((6, 4))
        out = truncation_V(A, np.zeros((4, 0)), [6])
        assert out[0] == pytest.approx(A @ A.T, rel=1e-12)

    def test_it_equals_the_projected_quadratic_form(self):
        A = np.random.default_rng(3).standard_normal((5, 7))
        B = _basis(7, 3, seed=4)
        P = np.eye(7) - B @ B.T
        assert truncation_V(A, B, [5])[0] == pytest.approx(A @ P @ A.T, rel=1e-10)

    def test_it_is_positive_semidefinite(self):
        A = np.random.default_rng(5).standard_normal((8, 6))
        out = truncation_V(A, _basis(6, 2, seed=6), [8])
        assert np.linalg.eigvalsh(out[0]).min() > -1e-12

    def test_blocks_split_by_row_and_stay_aligned(self):
        A = np.random.default_rng(7).standard_normal((7, 5))
        B = _basis(5, 2, seed=8)
        whole = truncation_V(A, B, [7])[0]
        parts = truncation_V(A, B, [3, 4])
        assert [p.shape for p in parts] == [(3, 3), (4, 4)]
        assert parts[0] == pytest.approx(whole[:3, :3], rel=1e-12)
        assert parts[1] == pytest.approx(whole[3:, 3:], rel=1e-12)


class TestAnAssertedCoordinateIsExcluded:
    def test_held_column_contributes_nothing(self):
        # The distinction the term exists to make: a dropped direction is one
        # nobody measured, a held one is a number the model states.
        A = np.random.default_rng(9).standard_normal((6, 5))
        B = _basis(5, 2, seed=10)
        B[3] = 0.0                      # as --hold builds it
        B, _ = np.linalg.qr(B)
        B = B[:, :2]
        B[3] = 0.0
        with_held = truncation_V(A, B, [6], held=[3])
        A2 = A.copy()
        A2[:, 3] = 0.0
        assert with_held[0] == pytest.approx(
            truncation_V(A2, B, [6])[0], rel=1e-10)

    def test_holding_shrinks_the_term(self):
        A = np.random.default_rng(11).standard_normal((6, 5))
        B = _basis(5, 2, seed=12)
        B[3] = 0.0
        free = np.trace(truncation_V(A, B, [6])[0])
        held = np.trace(truncation_V(A, B, [6], held=[3])[0])
        assert held < free

    def test_a_held_coordinate_inside_the_span_is_refused(self):
        # In the span and asserted outside it at once. The basis has to be built
        # with --hold, and this is what catches a mismatched pair.
        A = np.random.default_rng(13).standard_normal((6, 5))
        with pytest.raises(ValueError, match="inside the span"):
            truncation_V(A, _basis(5, 2, seed=14), [6], held=[3])

    def test_out_of_range_held_index_is_refused(self):
        A = np.random.default_rng(15).standard_normal((6, 5))
        with pytest.raises(ValueError, match="out of range"):
            truncation_V(A, np.zeros((5, 0)), [6], held=[9])


class TestShapeGuards:
    def test_basis_must_span_the_jacobian_columns(self):
        with pytest.raises(ValueError, match="span the"):
            truncation_V(np.zeros((6, 5)), np.zeros((4, 2)), [6])

    def test_block_sizes_must_total_the_rows(self):
        with pytest.raises(ValueError, match="blocks total"):
            truncation_V(np.zeros((6, 5)), np.zeros((5, 2)), [3, 2])

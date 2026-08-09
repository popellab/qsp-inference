"""Unit tests for V_B per block (qsp_inference.vpop.blocks).

The point of a joint draw is the off-diagonal: cohorts sharing patients must come
out correlated, and by more when they share more.
"""
import numpy as np
import pytest

from maple.core.calibration.cohort import Cohort, CohortRegistry, PatientBlock, Stratum

from qsp_inference.vpop.blocks import (
    assemble_V,
    bootstrap_V,
    emulator_E,
    row_offsets,
)
from qsp_inference.vpop.blocks import block_draw_plan

N_CLOUD = 4_000
N_BOOT = 3_000


def _cohort(cid, n):
    return Cohort(cohort_id=cid, description=f"{cid} desc", scenarios=["s"],
                  n_c=n, source_tag=f"tag_{cid}")


def _li_registry():
    return CohortRegistry(
        cohorts=[_cohort("base", 16), _cohort("arm_a", 9), _cohort("arm_b", 10)],
        blocks=[PatientBlock(
            block_id="li", description="counted",
            cohorts=["base", "arm_a", "arm_b"],
            strata=[Stratum(cohorts=["base", "arm_a"], n=6),
                    Stratum(cohorts=["arm_a"], n=3),
                    Stratum(cohorts=["base", "arm_b"], n=10)],
        )],
    )


@pytest.fixture
def cloud():
    return np.random.default_rng(7).normal(size=N_CLOUD)


def _median_rows(cloud):
    """One row per cohort: the median of its drawn patients."""
    return lambda cid, idx: np.median(cloud[idx])


def _corr(V):
    d = np.sqrt(np.diag(V))
    return V / np.outer(d, d)


class TestBootstrapV:
    def test_shared_patients_correlate_by_how_much_they_share(self, cloud):
        (plan,) = block_draw_plan(_li_registry(), {})
        V = bootstrap_V(plan, _median_rows(cloud), np.random.default_rng(0),
                        n_boot=N_BOOT, n_cloud=N_CLOUD)
        assert V.shape == (3, 3)
        i = {c: k for k, c in enumerate(plan.cohort_ids)}
        R = _corr(V)
        # base shares 10 of its 16 with arm_b and 6 with arm_a; the arms share none
        assert R[i["base"], i["arm_b"]] > R[i["base"], i["arm_a"]] > 0.15
        assert abs(R[i["arm_a"], i["arm_b"]]) < 0.06

    def test_independent_draws_leave_the_off_diagonal_empty(self, cloud):
        """A shared-row block joins disjoint people, so V_B is near diagonal."""
        reg = CohortRegistry(cohorts=[_cohort("a", 20), _cohort("b", 20)])
        target = {"observable": {"inputs": [{"cohort_id": "a"}, {"cohort_id": "b"}]}}
        (plan,) = block_draw_plan(reg, {"fc": target})
        V = bootstrap_V(plan, _median_rows(cloud), np.random.default_rng(1),
                        n_boot=N_BOOT, n_cloud=N_CLOUD)
        assert abs(_corr(V)[0, 1]) < 0.06

    def test_variance_falls_with_cohort_size(self, cloud):
        reg = CohortRegistry(cohorts=[_cohort("small", 10), _cohort("big", 250)])
        var = {
            p.cohort_ids[0]: bootstrap_V(p, _median_rows(cloud),
                                         np.random.default_rng(2),
                                         n_boot=N_BOOT, n_cloud=N_CLOUD)[0, 0]
            for p in block_draw_plan(reg, {})
        }
        assert var["big"] < var["small"] / 10

    def test_multi_row_cohorts_concatenate_in_plan_order(self, cloud):
        (plan,) = block_draw_plan(_li_registry(), {})
        k_of = {"base": 2, "arm_a": 1, "arm_b": 3}
        rows = lambda cid, idx: np.full(k_of[cid], np.median(cloud[idx]))
        V = bootstrap_V(plan, rows, np.random.default_rng(3),
                        n_boot=200, n_cloud=N_CLOUD)
        assert V.shape == (6, 6)
        assert row_offsets(plan, k_of) == {
            "arm_a": slice(0, 1), "arm_b": slice(1, 4), "base": slice(4, 6)
        }


class TestAssemble:
    def test_adds_E_and_factors(self):
        plans = block_draw_plan(CohortRegistry(cohorts=[_cohort("a", 5)]), {})
        boot = [np.array([[4.0, 1.0], [1.0, 4.0]])]
        E = [np.array([[1.0, 0.0], [0.0, 1.0]])]
        cov = assemble_V(plans, boot, E)
        assert np.allclose(cov.V[0], [[5.0, 1.0], [1.0, 5.0]], atol=1e-8)
        assert np.allclose(cov.chol[0] @ cov.chol[0].T, cov.V[0])

    def test_ridge_rescues_a_singular_block(self):
        plans = block_draw_plan(CohortRegistry(cohorts=[_cohort("a", 5)]), {})
        singular = [np.ones((3, 3))]  # rank 1, as two collinear readouts give
        cov = assemble_V(plans, singular)
        assert np.all(np.isfinite(cov.chol[0]))

    def test_unhonoured_travels_with_the_matrices(self):
        reg = CohortRegistry(
            cohorts=[_cohort("a", 5), _cohort("b", 7)],
            blocks=[PatientBlock(block_id="trial", description="same trial",
                                 cohorts=["a", "b"])],
        )
        plans = block_draw_plan(reg, {})
        cov = assemble_V(plans, [np.eye(2)])
        assert cov.unhonoured == ("trial",)


class TestEmulatorE:
    def test_slices_by_block(self):
        reg = CohortRegistry(cohorts=[_cohort("a", 5), _cohort("b", 7)])
        plans = block_draw_plan(reg, {})
        k_of = {"a": 2, "b": 1}
        diffs = np.random.default_rng(4).normal(size=(500, 3))
        blocks, means = emulator_E(plans, diffs, k_of)
        assert [b.shape for b in blocks] == [(2, 2), (1, 1)]
        assert np.allclose(means[0], diffs[:, :2].mean(0))

    def test_column_count_must_match_the_plans(self):
        plans = block_draw_plan(CohortRegistry(cohorts=[_cohort("a", 5)]), {})
        with pytest.raises(ValueError, match="plans want"):
            emulator_E(plans, np.zeros((10, 5)), {"a": 2})
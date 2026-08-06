"""Unit tests for the block resampling plan (qsp_inference.vpop.resampling).

Cohorts joined by a shared row draw independently; cohorts that share people draw
once from the block's patient set. These check that the plan splits a component
the right way and that the drawn indices carry the strata's overlaps exactly.
"""
import numpy as np
import pytest

from maple.core.calibration.cohort import Cohort, CohortRegistry, PatientBlock, Stratum

from qsp_inference.vpop.resampling import (block_draw_plan, draw_block_indices,
                                           restrict_plans)


def _cohort(cid, n, scenarios=("s",)):
    return Cohort(
        cohort_id=cid, description=f"{cid} desc", scenarios=list(scenarios),
        n_c=n, source_tag=f"tag_{cid}",
    )


def _target(*cohort_ids):
    """A scalar target names one cohort; a cross-scenario one names its arms."""
    if len(cohort_ids) == 1:
        return {"cohort_id": cohort_ids[0]}
    return {"observable": {"inputs": [{"cohort_id": c} for c in cohort_ids]}}


# The Li 2022 shape: a baseline biopsy set and two randomised arms, where 6 of
# arm A and all 10 of arm B have a baseline, and the arms share nobody.
LI_STRATA = [
    Stratum(cohorts=["base", "arm_a"], n=6),
    Stratum(cohorts=["arm_a"], n=3),
    Stratum(cohorts=["base", "arm_b"], n=10),
]


def _li_registry(**block_kw):
    return CohortRegistry(
        cohorts=[_cohort("base", 16), _cohort("arm_a", 9), _cohort("arm_b", 10)],
        blocks=[
            PatientBlock(
                block_id="li", description="counted",
                cohorts=["base", "arm_a", "arm_b"], strata=LI_STRATA, **block_kw,
            )
        ],
    )


class TestRestrict:
    def test_a_block_with_no_reporting_cohort_goes(self):
        reg = CohortRegistry(cohorts=[_cohort("a", 5), _cohort("b", 7)])
        plans = block_draw_plan(reg, {"t1": _target("a"), "t2": _target("b")})
        assert [p.cohort_ids for p in restrict_plans(plans, {"a"})] == [("a",)]

    def test_a_dropped_member_keeps_its_place_in_the_joint_draw(self):
        """Row order loses the cohort; the strata do not, or the overlap changes."""
        (plan,) = block_draw_plan(_li_registry(), {"t": _target("base")})
        (kept,) = restrict_plans([plan], {"base", "arm_a"})
        assert kept.cohort_ids == ("arm_a", "base")
        assert kept.groups == plan.groups
        idx = draw_block_indices(kept, np.random.default_rng(0), 4, 50)
        assert set(idx) == {"base", "arm_a", "arm_b"}

    def test_an_untouched_plan_is_returned_unchanged(self):
        reg = CohortRegistry(cohorts=[_cohort("a", 5)])
        plans = block_draw_plan(reg, {"t": _target("a")})
        assert restrict_plans(plans, {"a", "absent"})[0] is plans[0]


class TestPlanShape:
    def test_lone_cohort_is_its_own_block(self):
        reg = CohortRegistry(cohorts=[_cohort("a", 5), _cohort("b", 7)])
        plans = block_draw_plan(reg, {"t1": _target("a"), "t2": _target("b")})
        assert [p.cohort_ids for p in plans] == [("a",), ("b",)]
        assert all(p.is_exact for p in plans)
        assert all(len(p.groups) == 1 and not p.groups[0].is_joint for p in plans)

    def test_shared_row_joins_but_draws_apart(self):
        """A contrast across arms joins disjoint people: one block, two draws."""
        reg = CohortRegistry(cohorts=[_cohort("a", 5), _cohort("b", 7)])
        (plan,) = block_draw_plan(reg, {"fc": _target("a", "b")})
        assert plan.cohort_ids == ("a", "b")
        assert len(plan.groups) == 2
        assert not any(g.is_joint for g in plan.groups)
        assert plan.is_exact

    def test_counted_block_draws_jointly(self):
        (plan,) = block_draw_plan(_li_registry(), {"t": _target("base")})
        assert plan.cohort_ids == ("arm_a", "arm_b", "base")
        (group,) = plan.groups
        assert group.is_joint and group.block_id == "li"
        assert group.n_draw == 19
        assert {c: len(p) for c, p in group.member_positions.items()} == {
            "base": 16, "arm_a": 9, "arm_b": 10
        }

    def test_strata_positions_reproduce_declared_overlaps(self):
        (plan,) = block_draw_plan(_li_registry(), {"t": _target("base")})
        pos = {c: set(p) for c, p in plan.groups[0].member_positions.items()}
        assert len(pos["base"] & pos["arm_a"]) == 6
        assert len(pos["base"] & pos["arm_b"]) == 10
        assert len(pos["arm_a"] & pos["arm_b"]) == 0

    def test_uncounted_block_is_reported_not_honoured(self):
        reg = CohortRegistry(
            cohorts=[_cohort("a", 5), _cohort("b", 7)],
            blocks=[PatientBlock(block_id="trial", description="same trial",
                                 cohorts=["a", "b"])],
        )
        (plan,) = block_draw_plan(reg, {})
        assert plan.cohort_ids == ("a", "b")
        assert plan.unhonoured == ("trial",)
        assert not plan.is_exact
        assert not any(g.is_joint for g in plan.groups)

    def test_counted_and_uncounted_overlay(self):
        """Li's three draw jointly inside a wider uncounted trial block."""
        reg = _li_registry()
        reg.cohorts.append(_cohort("other", 8))
        reg.blocks.append(
            PatientBlock(block_id="trial", description="same trial",
                         cohorts=["arm_a", "other"])
        )
        (plan,) = block_draw_plan(reg, {})
        assert plan.cohort_ids == ("arm_a", "arm_b", "base", "other")
        joint = [g for g in plan.groups if g.is_joint]
        lone = [g for g in plan.groups if not g.is_joint]
        assert [g.block_id for g in joint] == ["li"]
        assert [g.cohort_ids for g in lone] == [("other",)]
        assert plan.unhonoured == ("trial",)

    def test_strata_disagreeing_with_n_c_raises(self):
        reg = _li_registry()
        reg.cohorts[0] = _cohort("base", 15)  # strata place 16
        with pytest.raises(ValueError, match=r"strata place 16 .* n_c=15"):
            block_draw_plan(reg, {})


class TestDraw:
    def test_joint_members_share_their_patients(self):
        (plan,) = block_draw_plan(_li_registry(), {"t": _target("base")})
        idx = draw_block_indices(plan, np.random.default_rng(0), 200, 1_000)
        pos = plan.groups[0].member_positions
        # base and arm_a agree exactly on the patients their strata share
        shared = sorted(set(pos["base"]) & set(pos["arm_a"]))
        cols = lambda c: [pos[c].index(p) for p in shared]
        assert np.array_equal(idx["base"][:, cols("base")], idx["arm_a"][:, cols("arm_a")])

    def test_shapes_match_n_c(self):
        (plan,) = block_draw_plan(_li_registry(), {"t": _target("base")})
        idx = draw_block_indices(plan, np.random.default_rng(0), 50, 1_000)
        assert {c: v.shape for c, v in idx.items()} == {
            "base": (50, 16), "arm_a": (50, 9), "arm_b": (50, 10)
        }

    def test_independent_groups_do_not_covary(self):
        reg = CohortRegistry(cohorts=[_cohort("a", 40), _cohort("b", 40)])
        (plan,) = block_draw_plan(reg, {"fc": _target("a", "b")})
        idx = draw_block_indices(plan, np.random.default_rng(1), 4_000, 500)
        r = np.corrcoef(idx["a"].mean(1), idx["b"].mean(1))[0, 1]
        assert abs(r) < 0.06

    def test_eligibility_weights_bias_the_draw(self):
        reg = CohortRegistry(cohorts=[_cohort("a", 30)])
        (plan,) = block_draw_plan(reg, {"t": _target("a")})
        w = np.zeros(100)
        w[:10] = 1.0
        idx = draw_block_indices(plan, np.random.default_rng(2), 100, 100, {"a": w})
        assert idx["a"].max() < 10

    def test_joint_members_must_agree_on_eligibility(self):
        (plan,) = block_draw_plan(_li_registry(), {"t": _target("base")})
        probs = {"base": np.ones(100), "arm_a": np.arange(100.0), "arm_b": np.ones(100)}
        with pytest.raises(ValueError, match="share a draw but declare different"):
            draw_block_indices(plan, np.random.default_rng(3), 10, 100, probs)
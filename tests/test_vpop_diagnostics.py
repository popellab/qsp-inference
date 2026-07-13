"""Joint-reachability diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from qsp_inference.vpop import (
    conflict_ranking,
    duplicate_observables,
    ess_scaling,
    greedy_core,
    paired_effect_sizes,
    self_target_control,
)


def test_duplicate_observables_finds_a_cloned_column():
    """The guard that catches collinear constraints before they fake a joint conflict."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=2000)
    sim = np.column_stack([a, a + 1e-6 * rng.normal(size=2000), rng.normal(size=2000)])
    dup = duplicate_observables(sim, ["a", "a_clone", "unrelated"], threshold=0.99)
    assert len(dup) == 1
    assert set(dup.iloc[0][["observable_a", "observable_b"]]) == {"a", "a_clone"}
    assert dup.iloc[0]["spearman"] > 0.99


def test_self_target_control_is_high_for_a_sound_solver():
    rng = np.random.default_rng(1)
    sim = rng.normal(size=(5000, 3))
    assert self_target_control(sim, ["a", "b", "c"], seed=2) > 0.5


def test_ess_scaling_shows_constant_fraction_for_a_reachable_target():
    """Reachable-with-tilting: the cost is plain importance-sampling overhead.

    The ESS *fraction* holds roughly steady as the cloud grows, so ESS itself grows with N --
    which is the signal that a bigger cloud buys you out of the problem. (The contrasting
    regime, ESS absolutely flat in N, needs several constraints intersecting and is not
    reproducible in a one-observable unit test; it is documented from the real cloud.)
    """
    rng = np.random.default_rng(2)
    sim = rng.normal(0.0, 1.0, size=(8000, 1))
    ok = ess_scaling(sim, {"y": rng.normal(0.3, 1.0, size=200)}, ["y"],
                     sizes=[1000, 2000, 4000, 8000], seed=3)
    fracs = ok["ess_fraction"].to_numpy()
    assert fracs.min() > 0.5 * fracs.max()      # fraction roughly constant
    assert ok["ess"].iloc[-1] > 3 * ok["ess"].iloc[0]  # so ESS grows with N


def test_an_unreachable_target_costs_no_ess_which_is_why_max_tv_exists():
    """The trap that ESS-only selection falls into.

    When the data sit outside the cloud the weights correctly stay ~uniform -- there is
    nothing to tilt toward -- so ESS stays HIGH while the fit is garbage. Anything selecting
    on ESS alone would call this observable free.
    """
    rng = np.random.default_rng(9)
    sim = rng.normal(0.0, 1.0, size=(4000, 1))
    far = ess_scaling(sim, {"y": rng.normal(8.0, 0.2, size=200)}, ["y"],
                      sizes=[1000, 4000], seed=3)
    assert far["ess_fraction"].min() > 0.9   # no tilt happened at all


def test_greedy_core_splits_reachable_from_blocking():
    rng = np.random.default_rng(3)
    n = 6000
    easy1 = rng.normal(0.0, 1.0, size=n)
    easy2 = rng.normal(0.0, 1.0, size=n)
    hard = rng.normal(0.0, 1.0, size=n)
    sim = np.column_stack([easy1, easy2, hard])
    observed = {
        "easy1": rng.normal(0.0, 1.0, size=200),   # already matched
        "easy2": rng.normal(0.0, 1.0, size=200),   # already matched
        "hard": rng.normal(6.0, 0.2, size=200),    # way off the cloud
    }
    res = greedy_core(sim, observed, ["easy1", "easy2", "hard"], min_ess=500)
    assert set(res.core) == {"easy1", "easy2"}
    assert res.blockers == ["hard"]
    assert res.core_ess > 500
    assert list(res.trajectory["n_observables"]) == [1, 2]


def test_conflict_ranking_labels_unreachable_vs_coupling():
    rng = np.random.default_rng(4)
    n = 6000
    a = rng.normal(0.0, 1.0, size=n)
    sim = np.column_stack([a, rng.normal(0.0, 1.0, size=n)])
    observed = {"core_obs": rng.normal(0.0, 1.0, size=200),
                "far": rng.normal(7.0, 0.2, size=200)}
    core = greedy_core(sim, observed, ["core_obs", "far"], min_ess=500)
    rank = conflict_ranking(sim, observed, ["core_obs", "far"], core)
    assert list(rank["observable"]) == ["far"]
    # 'far' blocks because the model cannot produce it at all, NOT because of coupling.
    assert rank.iloc[0]["kind"] == "unreachable"
    assert rank.iloc[0]["solo_tv"] > 0.5           # it does not fit even alone
    # ...and note the trap: its ESS looks perfectly healthy, because nothing tilted.
    assert rank.iloc[0]["solo_ess_fraction"] > 0.9


def test_paired_effect_sizes_flags_an_inert_intervention():
    """The nivolumab case: model says the arms are identical, data say they differ."""
    rng = np.random.default_rng(5)
    ctrl = rng.lognormal(0.0, 0.4, size=4000)
    sim = np.column_stack([ctrl, ctrl * 1.0001])   # treatment does nothing in the model
    observed = {
        "ctrl": rng.lognormal(0.0, 0.4, size=100),
        "treated": rng.lognormal(np.log(2.5), 0.4, size=100),  # but 2.5x in the data
    }
    df = paired_effect_sizes(sim, observed, ["ctrl", "treated"],
                             [("readout", "ctrl", "treated")])
    row = df.iloc[0]
    assert row["model_ratio"] == pytest.approx(1.0, abs=0.01)
    assert row["observed_ratio"] > 2.0
    assert row["model_arm_correlation"] > 0.999
    assert bool(row["inert_in_model"]) is True

"""The scenario table, and taking L_R from the prior rather than alongside it."""
import numpy as np
import pytest

from qsp_inference.vpop.mechanism import ScenarioTable, scenario_table

pytest.importorskip("jax")


def _t(time, ref=None):
    obs = {"readout_time": time, "code": "def compute_observable(): pass"}
    if ref is not None:
        obs["readout"] = {"reference": {"kind": "timepoint", "timepoint": ref}}
    return {"observable": obs}


class TestScenarioTable:
    def test_one_scenario_per_arm_time_not_per_readout(self):
        """Two readouts on the same arm and time share a scenario."""
        targets = {"a": _t(0.0), "b": _t(0.0), "c": _t(21.0)}
        arm = {"a": "base", "b": "base", "c": "base"}
        st = scenario_table(targets, arm, ["a", "b", "c"])
        assert st.n_scenarios == 2
        assert st.scenario_of["a"] == st.scenario_of["b"]
        assert st.scenario_of["c"] != st.scenario_of["a"]

    def test_cohorts_sharing_an_arm_collapse(self):
        """The saving the campaign is sized on: same arm and time, one scenario."""
        targets = {f"r{i}": _t(0.0) for i in range(20)}
        arm = {f"r{i}": "base" for i in range(20)}
        assert scenario_table(targets, arm, list(targets)).n_scenarios == 1

    def test_a_declared_reference_adds_its_own_scenario_first(self):
        targets = {"fold": _t(21.0, ref=0.0)}
        st = scenario_table(targets, {"fold": "gvax"}, ["fold"])
        assert st.n_scenarios == 2
        i_ref, i_out = st.scenario_of["fold"]
        assert st.scenarios[i_ref] == ("gvax", 0.0)
        assert st.scenarios[i_out] == ("gvax", 21.0)

    def test_reference_first_readout_last(self):
        """build_h_fn indexes x[0] as the reference; the order is load-bearing."""
        targets = {"fold": _t(21.0, ref=0.0)}
        st = scenario_table(targets, {"fold": "gvax"}, ["fold"])
        idx = st.scenario_of["fold"]
        assert len(idx) == 2
        assert st.scenarios[idx[0]][1] < st.scenarios[idx[-1]][1]

    def test_a_plain_readout_gets_one_index(self):
        st = scenario_table({"x": _t(0.0)}, {"x": "base"}, ["x"])
        assert len(st.scenario_of["x"]) == 1

    def test_the_same_time_on_different_arms_is_two_scenarios(self):
        targets = {"a": _t(0.0), "b": _t(0.0)}
        st = scenario_table(targets, {"a": "base", "b": "gvax"}, ["a", "b"])
        assert st.n_scenarios == 2

    def test_a_readout_with_no_arm_is_refused(self):
        with pytest.raises(KeyError, match="no arm"):
            scenario_table({"a": _t(0.0)}, {}, ["a"])

    def test_arms_are_reported_in_first_use_order(self):
        targets = {"a": _t(0.0), "b": _t(21.0)}
        st = scenario_table(targets, {"a": "base", "b": "gvax"}, ["a", "b"])
        assert set(st.arms()) == {"base", "gvax"}

    def test_missing_readout_time_reads_as_zero(self):
        st = scenario_table({"a": {"observable": {}}}, {"a": "base"}, ["a"])
        assert st.scenarios == (("base", 0.0),)

    def test_it_matches_the_pdac_shape(self):
        """5 arms, 7 (arm, time) points: the number the campaign is sized on."""
        spec = {"baseline_no_treatment": [0.0], "clinical_progression": [0.0],
                "gvax_neoadjuvant": [0.0, 21.0], "gvax_nivo_neoadjuvant": [0.0, 21.0],
                "gvax_nivo_urelumab_neoadjuvant": [21.0]}
        targets, arm = {}, {}
        for a, ts in spec.items():
            for t in ts:
                k = f"{a}@{t}"
                targets[k], arm[k] = _t(t), a
        st = scenario_table(targets, arm, list(targets))
        assert st.n_scenarios == 7
        assert len(st.arms()) == 5


class TestPriorCholesky:
    def test_it_refuses_a_prior_with_no_correlation(self):
        from qsp_inference.vpop.mechanism import prior_cholesky

        class NoR:
            pass

        class Pair:
            prior = NoR()
            param_names = ["a"]

        import qsp_inference.priors.inference_prior as ip

        orig = ip.build_prior_pair
        ip.build_prior_pair = lambda *a, **k: Pair()
        try:
            with pytest.raises(TypeError, match="no correlation matrix"):
                prior_cholesky(object())
        finally:
            ip.build_prior_pair = orig

    def test_it_refuses_a_reordered_param_list(self):
        """L_R is indexed by parameter; a permutation silently rotates the cloud."""
        from qsp_inference.vpop.mechanism import prior_cholesky

        class Pair:
            prior = type("P", (), {"_R": np.eye(3)})()
            param_names = ["a", "b", "c"]

        import qsp_inference.priors.inference_prior as ip

        orig = ip.build_prior_pair
        ip.build_prior_pair = lambda *a, **k: Pair()
        try:
            with pytest.raises(ValueError, match="silently permutes"):
                prior_cholesky(object(), param_names=["c", "b", "a"])
            L, names = prior_cholesky(object(), param_names=["a", "b", "c"])
            assert L.shape == (3, 3) and names == ("a", "b", "c")
        finally:
            ip.build_prior_pair = orig

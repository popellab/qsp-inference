"""Tests for the single owner of (pi, pi_tilde).

The load-bearing tests are the equivalence ones. This module replaces two
independent implementations that had to agree and did not, so it is not enough
that the new builder is self-consistent: it has to produce exactly what the
composition it replaces produced, or every cached pool and every trained
estimator built before it becomes incomparable.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # copula_prior pulls in torch (sbi extra)
import torch  # noqa: E402

from qsp_inference.priors.inference_prior import (  # noqa: E402
    CsvIndependentPrior,
    PriorPair,
    PriorSpec,
    build_prior_pair,
)

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
CSV_HEADER = "name,expected_value,units,distribution,dist_param1,dist_param2\n"


def _write_csv(tmp_path, rows, name="priors.csv"):
    path = Path(tmp_path) / name
    with open(path, "w") as f:
        f.write(CSV_HEADER)
        for r in rows:
            f.write(
                f"{r['name']},{r.get('ev', 1.0)},{r.get('units', '1/day')},"
                f"{r['dist']},{r['p1']},{r['p2']}\n"
            )
    return str(path)


def _write_yaml(tmp_path, data, name="submodel_priors.yaml"):
    from ruamel.yaml import YAML

    path = Path(tmp_path) / name
    with open(path, "w") as f:
        YAML().dump(data, f)
    return str(path)


LOGNORMAL_ROWS = [
    {"name": "k1", "dist": "lognormal", "p1": -0.5, "p2": 0.8},
    {"name": "k2", "dist": "lognormal", "p1": -1.0, "p2": 0.3},
    {"name": "k3", "dist": "lognormal", "p1": 0.5, "p2": 0.4},
]

YAML_DATA = {
    "metadata": {"n_parameters": 1, "n_samples": 1000},
    "parameters": [
        {
            "name": "k1",
            "marginal": {
                "distribution": "lognormal",
                "mu": 0.0,
                "sigma": 0.5,
                "median": 1.0,
                "cv": 0.5,
            },
        }
    ],
}


@pytest.fixture
def composite(tmp_path):
    """A minimal composite spec: 3 params, one covered by the submodel YAML."""
    return PriorSpec(
        priors_csv=_write_csv(tmp_path, LOGNORMAL_ROWS),
        submodel_priors_yaml=_write_yaml(tmp_path, YAML_DATA),
    )


# ---------------------------------------------------------------------------
# equivalence with what this replaces
# ---------------------------------------------------------------------------
class TestMatchesTheCompositionItReplaces:
    """The builder must reproduce the route the theta pool used.

    The pool's route was canonical because it is the one that actually generated
    the training clouds: load_composite_prior_log, or load_overlay_prior_log
    when a vary policy is set, with derived_yaml passed into the loader.
    """

    def test_plain_composite_matches_the_loader(self, composite):
        from qsp_inference.priors.copula_prior import load_composite_prior_log

        want, want_names = load_composite_prior_log(
            composite.submodel_priors_yaml, composite.priors_csv
        )
        pair = build_prior_pair(composite)

        assert pair.param_names == list(want_names)
        theta = want.sample((256,))
        assert torch.allclose(pair.prior.log_prob(theta), want.log_prob(theta))

    def test_overlay_matches_load_overlay_prior_log(self, tmp_path):
        import yaml as pyyaml

        from qsp_inference.priors.copula_prior import load_overlay_prior_log

        csv = _write_csv(tmp_path, LOGNORMAL_ROWS)
        ypath = _write_yaml(tmp_path, YAML_DATA)
        policy = Path(tmp_path) / "vary.yaml"
        policy.write_text(pyyaml.safe_dump({"vary": ["k1", "k3"]}))

        want, want_names = load_overlay_prior_log(
            ypath, csv, vary_params=["k1", "k3"], derived_yaml=None
        )
        pair = build_prior_pair(
            PriorSpec(priors_csv=csv, submodel_priors_yaml=ypath, vary_policy=str(policy))
        )

        assert pair.param_names == list(want_names)
        theta = want.sample((256,))
        assert torch.allclose(pair.prior.log_prob(theta), want.log_prob(theta))
        # k2 is outside the allowlist, so it must be pinned.
        assert float(pair.prior._marginals[1].std()) < 1e-2

    def test_csv_only_honours_the_declared_family(self, tmp_path):
        """The divergence that motivated this module.

        The old embedding prior read every row as exp(N(p1, p2)), so a Beta row
        became a lognormal orders of magnitude wide. Draws must respect the
        support the CSV declares.
        """
        csv = _write_csv(
            tmp_path,
            [
                {"name": "k1", "dist": "lognormal", "p1": 0.0, "p2": 0.5},
                {"name": "f_frac", "dist": "beta", "p1": 2.0, "p2": 18.0},
                {"name": "u_par", "dist": "uniform", "p1": 3.0, "p2": 4.0},
            ],
        )
        pair = build_prior_pair(PriorSpec(priors_csv=csv))
        theta = pair.sample_original(4000, seed=0)

        assert np.all(theta[:, 0] > 0)
        assert np.all((theta[:, 1] > 0) & (theta[:, 1] < 1)), "beta left [0,1]"
        assert theta[:, 1].mean() == pytest.approx(2 / 20, abs=0.02)
        assert np.all((theta[:, 2] >= 3.0) & (theta[:, 2] <= 4.0))


# ---------------------------------------------------------------------------
# the pair
# ---------------------------------------------------------------------------
class TestUntemperedIsInert:
    def test_prior_and_proposal_are_the_same_object(self, composite):
        pair = build_prior_pair(composite)
        assert pair.prior is pair.proposal
        assert not pair.is_tempered

    def test_reweighting_an_untempered_pair_is_uniform(self, composite):
        pair = build_prior_pair(composite)
        theta = pair.sample_log(500, seed=0)
        res = pair.reweight(theta)
        assert res.ess_fraction == pytest.approx(1.0, rel=1e-6)

    def test_sampling_is_deterministic_in_the_seed(self, composite):
        pair = build_prior_pair(composite)
        a = pair.sample_original(64, seed=7)
        b = pair.sample_original(64, seed=7)
        c = pair.sample_original(64, seed=8)
        assert np.array_equal(a, b)
        assert not np.array_equal(a, c)


class TestTempered:
    def test_proposal_is_a_pure_widening(self, composite):
        pair = build_prior_pair(composite.at_temperature(4.0))
        assert pair.is_tempered
        for m_pi, m_q in zip(pair.prior._marginals, pair.proposal._marginals):
            assert float(m_q.mean()) == pytest.approx(float(m_pi.mean()))
            assert float(m_q.std()) == pytest.approx(2.0 * float(m_pi.std()))
        assert np.allclose(pair.proposal._R, pair.prior._R)

    def test_reweight_recovers_the_prior_moments(self, composite):
        pair = build_prior_pair(composite.at_temperature(4.0))
        theta = pair.sample_log(200_000, seed=0)
        res = pair.reweight(theta, warn_below=None)

        mean = (res.weights[:, None] * theta).sum(axis=0)
        want = np.array([float(m.mean()) for m in pair.prior._marginals])
        assert mean == pytest.approx(want, abs=0.05)

    def test_ess_falls_as_temperature_rises(self, composite):
        fracs = []
        for T in (1.0, 1.5, 4.0):
            pair = build_prior_pair(composite.at_temperature(T))
            theta = pair.sample_log(20_000, seed=0)
            fracs.append(pair.reweight(theta, warn_below=None).ess_fraction)
        assert fracs == sorted(fracs, reverse=True)

    def test_tempering_is_applied_after_the_overlay(self, tmp_path):
        """A pinned param must stay pinned: sqrt(T)*0 == 0.

        Which params vary is a claim about the prior, not about the proposal.
        Tempering before the overlay would let the overlay overwrite it; after,
        a pin survives.
        """
        import yaml as pyyaml

        csv = _write_csv(tmp_path, LOGNORMAL_ROWS)
        ypath = _write_yaml(tmp_path, YAML_DATA)
        policy = Path(tmp_path) / "vary.yaml"
        policy.write_text(pyyaml.safe_dump({"vary": ["k1"]}))

        pair = build_prior_pair(
            PriorSpec(
                priors_csv=csv,
                submodel_priors_yaml=ypath,
                vary_policy=str(policy),
                proposal_temperature=9.0,
            )
        )
        assert float(pair.proposal._marginals[1].std()) < 1e-2
        assert float(pair.proposal._marginals[0].std()) == pytest.approx(
            3.0 * float(pair.prior._marginals[0].std())
        )

    def test_n_varying_counts_unpinned_params(self, tmp_path):
        import yaml as pyyaml

        csv = _write_csv(tmp_path, LOGNORMAL_ROWS)
        ypath = _write_yaml(tmp_path, YAML_DATA)
        policy = Path(tmp_path) / "vary.yaml"
        policy.write_text(pyyaml.safe_dump({"vary": ["k1", "k3"]}))
        pair = build_prior_pair(
            PriorSpec(priors_csv=csv, submodel_priors_yaml=ypath, vary_policy=str(policy))
        )
        # Pinned params carry compose_overlay_prior's pin_sigma (1e-3), not 0,
        # so the count is of marginals wider than that.
        assert pair.n_varying() == 2
        assert pair.n_varying(pin_sigma=1e-9) == 3


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
class TestFingerprint:
    def test_temperature_is_inert_at_one(self, composite):
        assert composite.fingerprint() == composite.at_temperature(1.0).fingerprint()

    def test_temperature_changes_the_fingerprint(self, composite):
        assert composite.fingerprint() != composite.at_temperature(4.0).fingerprint()

    def test_distinct_temperatures_are_distinct(self, composite):
        seen = {composite.at_temperature(T).fingerprint() for T in (1.0, 2.0, 4.0, 9.0)}
        assert len(seen) == 4

    def test_editing_the_csv_changes_it(self, tmp_path, composite):
        before = composite.fingerprint()
        rows = list(LOGNORMAL_ROWS)
        rows[0] = {**rows[0], "p2": 0.9}
        edited = PriorSpec(
            priors_csv=_write_csv(tmp_path, rows, name="priors2.csv"),
            submodel_priors_yaml=composite.submodel_priors_yaml,
        )
        assert edited.fingerprint() != before

    def test_renaming_a_file_does_not_change_it(self, tmp_path, composite):
        """Hash content, not paths: moving inputs must not invalidate a pool."""
        copy = _write_csv(tmp_path, LOGNORMAL_ROWS, name="elsewhere.csv")
        moved = PriorSpec(priors_csv=copy, submodel_priors_yaml=composite.submodel_priors_yaml)
        assert moved.fingerprint() == composite.fingerprint()

    def test_adding_a_vary_policy_changes_it(self, tmp_path, composite):
        import yaml as pyyaml

        policy = Path(tmp_path) / "vary.yaml"
        policy.write_text(pyyaml.safe_dump({"vary": ["k1"]}))
        with_policy = PriorSpec(
            priors_csv=composite.priors_csv,
            submodel_priors_yaml=composite.submodel_priors_yaml,
            vary_policy=str(policy),
        )
        assert with_policy.fingerprint() != composite.fingerprint()

    def test_label_names_the_non_default_choices(self, tmp_path, composite):
        assert composite.label() == ""
        assert composite.at_temperature(4.0).label() == "T4"
        assert PriorSpec(priors_csv=composite.priors_csv).label() == "csvonly"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class TestValidation:
    @pytest.mark.parametrize("T", [0.0, -1.0])
    def test_nonpositive_temperature_rejected(self, composite, T):
        with pytest.raises(ValueError, match="must be > 0"):
            composite.at_temperature(T)

    @pytest.mark.parametrize("T", [0.5, 0.25])
    def test_infinite_variance_temperature_rejected(self, composite, T):
        with pytest.raises(ValueError, match="infinite variance"):
            composite.at_temperature(T)

    def test_temperature_just_above_the_floor_is_allowed(self, composite):
        assert composite.at_temperature(0.51).proposal_temperature == 0.51

    def test_csv_only_cannot_be_tempered(self, tmp_path):
        csv = _write_csv(tmp_path, LOGNORMAL_ROWS)
        with pytest.raises(ValueError, match="requires a submodel_priors_yaml"):
            PriorSpec(priors_csv=csv, proposal_temperature=4.0)

    def test_csv_only_rejects_overlay_and_derived(self, tmp_path):
        csv = _write_csv(tmp_path, LOGNORMAL_ROWS)
        with pytest.raises(ValueError, match="vary_policy requires"):
            PriorSpec(priors_csv=csv, vary_policy="x.yaml")
        with pytest.raises(ValueError, match="derived_yaml requires"):
            PriorSpec(priors_csv=csv, derived_yaml="x.yaml")

    def test_csv_only_has_no_density(self, tmp_path):
        pair = build_prior_pair(PriorSpec(priors_csv=_write_csv(tmp_path, LOGNORMAL_ROWS)))
        assert not pair.has_density
        with pytest.raises(NotImplementedError, match="no density"):
            pair.reweight(np.zeros((4, 3)))
        with pytest.raises(NotImplementedError, match="subset"):
            pair.subset([0])

    def test_unsupported_family_is_named(self, tmp_path):
        csv = _write_csv(tmp_path, [{"name": "k1", "dist": "cauchy", "p1": 0.0, "p2": 1.0}])
        with pytest.raises(ValueError, match="cauchy"):
            CsvIndependentPrior(csv)

    def test_empty_vary_policy_is_rejected(self, tmp_path, composite):
        import yaml as pyyaml

        policy = Path(tmp_path) / "vary.yaml"
        policy.write_text(pyyaml.safe_dump({"vary": []}))
        spec = PriorSpec(
            priors_csv=composite.priors_csv,
            submodel_priors_yaml=composite.submodel_priors_yaml,
            vary_policy=str(policy),
        )
        with pytest.raises(ValueError, match="no non-empty"):
            build_prior_pair(spec)

    def test_misaligned_proposal_is_caught(self):
        from qsp_inference.priors.inference_prior import _assert_aligned

        class Fake:
            param_names = ["a", "c"]

        with pytest.raises(RuntimeError, match="first mismatch at index 1"):
            _assert_aligned(Fake(), ["a", "b"])

        class Short:
            param_names = ["a"]

        with pytest.raises(RuntimeError, match="1 params vs 2"):
            _assert_aligned(Short(), ["a", "b"])


class TestSubset:
    def test_subset_keeps_the_pair_aligned(self, composite):
        pair = build_prior_pair(composite.at_temperature(4.0)).subset([0, 2])
        assert pair.param_names == ["k1", "k3"]
        assert pair.is_tempered
        theta = pair.sample_log(2000, seed=1)
        assert theta.shape == (2000, 2)
        assert pair.reweight(theta, warn_below=None).ess_fraction < 1.0

    def test_untempered_subset_stays_identical(self, composite):
        pair = build_prior_pair(composite).subset([1, 2])
        assert pair.prior is pair.proposal


class TestSummary:
    def test_summary_mentions_the_temperature(self, composite):
        assert "T = 1" in build_prior_pair(composite).summary()
        assert "pi^(1/4)" in build_prior_pair(composite.at_temperature(4.0)).summary()

    def test_verbose_build_prints(self, composite, capsys):
        build_prior_pair(composite, verbose=True)
        assert "prior:" in capsys.readouterr().out

    def test_population_block_absent_is_not_an_error(self, composite):
        pair = build_prior_pair(composite, load_population=True)
        assert pair.population_prior is None
        assert isinstance(pair, PriorPair)


class TestImportsWithoutTorch:
    """A simulation host asks which theta to run; it should not need torch.

    The spec and pool objects are pure numpy, and the CSV-only sampler is too.
    Only the composite copula prior needs torch, and it says so itself. This is
    what lets qsp-hpc-tools depend on this package without carrying a
    deep-learning stack onto every simulation node.
    """

    def test_csv_only_pool_works_with_torch_blocked(self, tmp_path, monkeypatch):
        import subprocess
        import sys
        import textwrap

        csv = tmp_path / "p.csv"
        csv.write_text(CSV_HEADER + "k1,1.0,1/day,lognormal,0.0,0.5\n" + "f,0.1,-,beta,2.0,18.0\n")
        script = textwrap.dedent(f"""
            import sys
            class Blocker:
                def find_module(self, name, path=None):
                    return self if name == "torch" or name.startswith("torch.") else None
                def load_module(self, name):
                    raise ImportError("No module named 'torch'")
            sys.meta_path.insert(0, Blocker())

            from qsp_inference.priors import PriorSpec, ThetaPoolSpec, get_theta_pool

            spec = ThetaPoolSpec(
                prior=PriorSpec(priors_csv={str(csv)!r}), seed=1, n_total=64
            )
            pool = get_theta_pool(spec, {str(tmp_path / "pools")!r})
            assert pool.shape == (64, 2)
            assert (pool[:, 1] > 0).all() and (pool[:, 1] < 1).all()
            print("OK")
            """)
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert "OK" in out.stdout

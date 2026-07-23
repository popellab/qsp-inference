"""Tests for pool construction and pool identity, now in one place.

The identity tests matter more than the construction ones. A pool that is
generated slightly wrong will usually show up somewhere downstream; a pool that
is *cached* under a key that does not distinguish it from a different
distribution will not, because the matrix has the right shape and plausible
values and every diagnostic computed from it looks fine.
"""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")  # copula_prior pulls in torch (sbi extra)

from qsp_inference.priors.inference_prior import PriorSpec  # noqa: E402
from qsp_inference.priors.theta_pool import (  # noqa: E402
    ThetaPoolSpec,
    get_theta_pool,
    theta_for_indices,
)

CSV_HEADER = "name,expected_value,units,distribution,dist_param1,dist_param2\n"
ROWS = [
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


def _write_csv(tmp_path, rows, name="priors.csv"):
    path = Path(tmp_path) / name
    with open(path, "w") as f:
        f.write(CSV_HEADER)
        for r in rows:
            f.write(f"{r['name']},1.0,1/day,{r['dist']},{r['p1']},{r['p2']}\n")
    return str(path)


def _write_yaml(tmp_path, data, name="submodel_priors.yaml"):
    from ruamel.yaml import YAML

    path = Path(tmp_path) / name
    with open(path, "w") as f:
        YAML().dump(data, f)
    return str(path)


@pytest.fixture
def prior_spec(tmp_path):
    return PriorSpec(
        priors_csv=_write_csv(tmp_path, ROWS),
        submodel_priors_yaml=_write_yaml(tmp_path, YAML_DATA),
    )


@pytest.fixture
def pool_spec(prior_spec):
    return ThetaPoolSpec(prior=prior_spec, seed=17, n_total=200)


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
class TestConstruction:
    def test_shape_and_positivity(self, pool_spec, tmp_path):
        pool = get_theta_pool(pool_spec, tmp_path / "pools")
        assert pool.shape == (200, 3)
        assert np.all(pool > 0), "lognormal draws must be in original space"

    def test_is_deterministic(self, pool_spec, tmp_path):
        a = get_theta_pool(pool_spec, tmp_path / "a")
        b = get_theta_pool(pool_spec, tmp_path / "b")
        assert np.array_equal(a, b)

    def test_cache_is_reused_not_regenerated(self, pool_spec, tmp_path):
        d = tmp_path / "pools"
        first = get_theta_pool(pool_spec, d)
        path = pool_spec.cache_path(d)
        assert path.exists()
        # Overwrite on disk; a second call must return what is cached.
        np.save(path, np.zeros_like(first))
        assert np.array_equal(get_theta_pool(pool_spec, d), np.zeros_like(first))

    def test_rejects_nonpositive_size(self, prior_spec):
        with pytest.raises(ValueError, match="n_total must be positive"):
            ThetaPoolSpec(prior=prior_spec, seed=0, n_total=0)

    def test_tempered_pool_is_wider(self, prior_spec, tmp_path):
        cold = ThetaPoolSpec(prior=prior_spec, seed=3, n_total=20_000)
        hot = ThetaPoolSpec(prior=prior_spec.at_temperature(9.0), seed=3, n_total=20_000)
        s_cold = np.log(get_theta_pool(cold, tmp_path / "c")).std(axis=0)
        s_hot = np.log(get_theta_pool(hot, tmp_path / "h")).std(axis=0)
        assert np.all(s_hot > 2.5 * s_cold)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------
class TestIdentity:
    def test_every_sampling_knob_moves_the_fingerprint(self, prior_spec):
        base = ThetaPoolSpec(prior=prior_spec, seed=17, n_total=200)
        variants = {
            "seed": ThetaPoolSpec(prior=prior_spec, seed=18, n_total=200),
            "n_total": ThetaPoolSpec(prior=prior_spec, seed=17, n_total=201),
            "temperature": ThetaPoolSpec(
                prior=prior_spec.at_temperature(4.0), seed=17, n_total=200
            ),
        }
        for name, v in variants.items():
            assert v.fingerprint() != base.fingerprint(), f"{name} did not key a pool"

    def test_search_effort_does_not_fragment_the_cache(self, prior_spec):
        """Oversample settings change how long the search runs, not the rows."""
        a = ThetaPoolSpec(prior=prior_spec, seed=1, n_total=50)
        b = ThetaPoolSpec(
            prior=prior_spec,
            seed=1,
            n_total=50,
            restriction_oversample_factor=9.0,
            restriction_max_oversample=64,
        )
        assert a.fingerprint() == b.fingerprint()

    def test_untempered_pool_hashes_as_if_temperature_never_existed(self, prior_spec):
        a = ThetaPoolSpec(prior=prior_spec, seed=5, n_total=10)
        b = ThetaPoolSpec(prior=prior_spec.at_temperature(1.0), seed=5, n_total=10)
        assert a.fingerprint() == b.fingerprint()

    def test_cache_filename_names_the_variant(self, prior_spec, tmp_path):
        hot = ThetaPoolSpec(prior=prior_spec.at_temperature(4.0), seed=1, n_total=9)
        name = hot.cache_path(tmp_path).name
        assert "_n9" in name and "T4" in name

    def test_editing_the_csv_keys_a_different_pool(self, tmp_path, pool_spec):
        edited_rows = [dict(ROWS[0], p2=0.9), ROWS[1], ROWS[2]]
        edited = PriorSpec(
            priors_csv=_write_csv(tmp_path, edited_rows, name="p2.csv"),
            submodel_priors_yaml=pool_spec.prior.submodel_priors_yaml,
        )
        other = ThetaPoolSpec(prior=edited, seed=17, n_total=200)
        assert other.fingerprint() != pool_spec.fingerprint()


class TestLegacyCache:
    def test_legacy_pool_is_read_rather_than_regenerated(self, pool_spec, tmp_path):
        """Pools are expensive; a rename must not throw one away."""
        d = tmp_path / "pools"
        d.mkdir()
        marker = np.full((pool_spec.n_total, 3), 4.2)
        np.save(pool_spec.legacy_cache_path(d), marker)
        assert np.array_equal(get_theta_pool(pool_spec, d), marker)

    def test_legacy_fallback_can_be_refused(self, pool_spec, tmp_path):
        d = tmp_path / "pools"
        d.mkdir()
        np.save(pool_spec.legacy_cache_path(d), np.full((pool_spec.n_total, 3), 4.2))
        pool = get_theta_pool(pool_spec, d, allow_legacy_cache=False)
        assert not np.allclose(pool, 4.2)

    def test_new_path_wins_over_legacy(self, pool_spec, tmp_path):
        d = tmp_path / "pools"
        d.mkdir()
        np.save(pool_spec.legacy_cache_path(d), np.full((pool_spec.n_total, 3), 4.2))
        np.save(pool_spec.cache_path(d), np.full((pool_spec.n_total, 3), 7.7))
        assert np.allclose(get_theta_pool(pool_spec, d), 7.7)

    def test_legacy_path_differs_from_the_new_one(self, pool_spec, tmp_path):
        assert pool_spec.legacy_cache_path(tmp_path) != pool_spec.cache_path(tmp_path)


class TestLegacyHashesAreGolden:
    """Pin the pre-consolidation hashes against fixed inputs.

    ``legacy_cache_path`` exists only to reproduce digests computed by code that
    has been deleted, so it has no other specification to check against. These
    values were verified byte-for-byte against
    ``qsp_hpc.simulation.theta_pool.theta_pool_cache_path`` before that function
    was retired; the cross-check below re-runs it wherever qsp-hpc-tools is
    installed, and these constants keep the guarantee everywhere else.

    A failure here means cached pools on HPC will silently be regenerated rather
    than reused. That is expensive, not incorrect, but it should be a decision.
    """

    HEADER = CSV_HEADER
    CSV_BODY = "k1,1.0,1/day,lognormal,-0.5,0.8\n"
    YAML_BODY = "metadata: {n_parameters: 0}\nparameters: []\n"
    VARY_BODY = "vary: [k1]\n"

    def _spec(self, tmp_path, **prior_kwargs):
        csv = tmp_path / "p.csv"
        csv.write_text(self.HEADER + self.CSV_BODY)
        ym = tmp_path / "s.yaml"
        ym.write_text(self.YAML_BODY)
        vp = tmp_path / "v.yaml"
        vp.write_text(self.VARY_BODY)
        if prior_kwargs.pop("with_vary", False):
            prior_kwargs["vary_policy"] = str(vp)
        return ThetaPoolSpec(
            prior=PriorSpec(priors_csv=str(csv), submodel_priors_yaml=str(ym), **prior_kwargs),
            seed=17,
            n_total=200,
        )

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            ({}, "theta_pool_3476a3924004533a_n200.npy"),
            ({"with_vary": True}, "theta_pool_b2c6622e06bacdf0_n200_overlay.npy"),
            (
                {"proposal_temperature": 4.0},
                "theta_pool_67d0815fa5903b5a_n200_T4.npy",
            ),
        ],
    )
    def test_matches_the_retired_implementation(self, tmp_path, kwargs, expected):
        spec = self._spec(tmp_path, **kwargs)
        assert spec.legacy_cache_path(tmp_path).name == expected

    def test_cross_check_against_qsp_hpc_when_installed(self, tmp_path):
        legacy_mod = pytest.importorskip(
            "qsp_hpc.simulation.theta_pool",
            reason="qsp-hpc-tools not installed; golden values above cover this",
        )
        fn = getattr(legacy_mod, "theta_pool_cache_path", None)
        if fn is None:  # already removed there
            pytest.skip("theta_pool_cache_path has been retired from qsp-hpc-tools")
        spec = self._spec(tmp_path, with_vary=True, proposal_temperature=2.5)
        want = fn(
            tmp_path,
            spec.prior.priors_csv,
            spec.prior.submodel_priors_yaml,
            seed=spec.seed,
            n_total=spec.n_total,
            vary_policy=spec.prior.vary_policy,
            derived_yaml=spec.prior.derived_yaml,
            proposal_temperature=spec.prior.proposal_temperature,
        )
        assert spec.legacy_cache_path(tmp_path).name == want.name


# ---------------------------------------------------------------------------
# indexing
# ---------------------------------------------------------------------------
class TestThetaForIndices:
    def test_returns_rows_in_the_order_asked_for(self, pool_spec, tmp_path):
        d = tmp_path / "pools"
        pool = get_theta_pool(pool_spec, d)
        idx = np.array([7, 3, 199, 0])
        assert np.array_equal(theta_for_indices(idx, pool_spec, d), pool[idx])

    def test_empty_index_is_allowed(self, pool_spec, tmp_path):
        out = theta_for_indices(np.array([], dtype=np.int64), pool_spec, tmp_path / "p")
        assert out.shape[0] == 0

    @pytest.mark.parametrize("bad", [[-1], [200], [0, 200]])
    def test_out_of_range_raises(self, pool_spec, tmp_path, bad):
        with pytest.raises(IndexError, match="out of range"):
            theta_for_indices(np.array(bad), pool_spec, tmp_path / "p")

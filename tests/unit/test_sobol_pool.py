"""Sobol draws over the copula: the marginals stay exact, the coverage improves.

The sequence goes in the independent latent space, before the Cholesky, so the
things worth testing are that the distribution is unchanged and that the
stratification it is there for actually shows up.
"""
import warnings

import numpy as np
import pytest
from scipy import stats

# copula_prior imports torch at module scope, and torch is the optional [sbi]
# extra. Skipping here rather than at each test keeps collection from failing
# where it is not installed.
pytest.importorskip("torch")

from qsp_inference.priors.copula_prior import GaussianCopulaPrior
from qsp_inference.priors.inference_prior import PriorSpec
from qsp_inference.priors.theta_pool import ThetaPoolSpec


def _prior(d=6, rho=0.4, lognormal=False):
    R = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    marg = [stats.lognorm(s=0.5) if lognormal else stats.norm(0.0, 1.0) for _ in range(d)]
    return GaussianCopulaPrior(marginals=marg, correlation=R,
                               param_names=[f"p{i}" for i in range(d)])


def _marginal_chi2(x, bins=32):
    """Mean per-column chi^2 of standardised draws against uniform. Lower is flatter."""
    z = (x - x.mean(0)) / x.std(0)
    u = stats.norm.cdf(z)
    e = len(x) / bins
    return float(np.mean([
        ((np.histogram(u[:, j], bins=bins, range=(0, 1))[0] - e) ** 2 / e).sum()
        for j in range(u.shape[1])
    ]))


class TestSampleSobol:
    def test_shape_and_finiteness(self):
        x = _prior().sample_sobol(1024, seed=0).numpy()
        assert x.shape == (1024, 6)
        assert np.isfinite(x).all()

    def test_it_is_deterministic_in_the_seed(self):
        g = _prior()
        a = g.sample_sobol(512, seed=3).numpy()
        assert np.array_equal(a, g.sample_sobol(512, seed=3).numpy())
        assert not np.array_equal(a, g.sample_sobol(512, seed=4).numpy())

    def test_marginals_and_correlation_are_preserved(self):
        g = _prior(d=4, rho=0.6)
        x = g.sample_sobol(8192, seed=1).numpy()
        assert np.allclose(x.mean(0), 0.0, atol=0.05)
        assert np.allclose(x.std(0), 1.0, atol=0.05)
        off = np.corrcoef(x.T)[np.triu_indices(4, 1)]
        assert np.allclose(off, 0.6, atol=0.05)

    def test_the_non_normal_path_also_works(self):
        """Lognormal marginals take the Phi / inverse-CDF route, not the fast path."""
        g = _prior(d=3, lognormal=True)
        x = g.sample_sobol(4096, seed=2).numpy()
        assert (x > 0).all()
        assert np.isclose(np.median(x[:, 0]), 1.0, atol=0.05)

    def test_it_stratifies_the_marginals_better_than_iid(self):
        g = _prior(d=8, rho=0.3)
        sob = g.sample_sobol(4096, seed=5).numpy()
        import torch

        torch.manual_seed(5)
        iid = g.sample((4096,)).numpy()
        assert _marginal_chi2(sob) < 0.5 * _marginal_chi2(iid)

    def test_a_non_power_of_two_warns(self):
        with pytest.warns(UserWarning, match="power of two"):
            _prior().sample_sobol(1000, seed=0)

    def test_a_power_of_two_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _prior().sample_sobol(1024, seed=0)


CSV = (
    "name,expected_value,units,distribution,dist_param1,dist_param2\n"
    "k1,1.0,1/day,lognormal,-0.5,0.8\n"
    "k2,1.0,1/day,lognormal,-1.0,0.3\n"
)


class TestPoolSpecSampler:
    @pytest.fixture(autouse=True)
    def _csv(self, tmp_path):
        p = tmp_path / "priors.csv"
        p.write_text(CSV)
        self._path = str(p)

    def _spec(self, **kw):
        return ThetaPoolSpec(prior=PriorSpec(priors_csv=self._path), seed=1, n_total=64, **kw)

    def test_sobol_changes_the_fingerprint(self):
        assert self._spec().fingerprint() != self._spec(sampler="sobol").fingerprint()

    def test_iid_contributes_nothing_to_the_fingerprint(self):
        """Pools cached before the field existed were drawn iid; do not orphan them."""
        assert self._spec().fingerprint() == self._spec(sampler="iid").fingerprint()

    def test_an_unknown_sampler_is_refused(self):
        with pytest.raises(ValueError, match="must be 'iid' or 'sobol'"):
            self._spec(sampler="latin")

    def test_sobol_with_a_classifier_warns(self, tmp_path):
        with pytest.warns(UserWarning, match="not a Sobol set"):
            self._spec(sampler="sobol", restriction_classifier_dir=str(tmp_path))

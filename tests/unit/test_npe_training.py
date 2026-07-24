"""Smoke tests for build_npe / train_npe on a trivial linear-Gaussian problem.

These exercise the plumbing (estimator construction, append, train, posterior
build, learning-curve collection), not inference quality. Kept small so they run
in a few seconds.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

from qsp_inference.inference.npe_training import (  # noqa: E402
    NpeTrainingResult,
    build_npe,
    train_npe,
)


def _toy(n=400, d=2, seed=0):
    """theta ~ U(-2, 2)^d, x = theta + small noise."""
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    prior = BoxUniform(low=-2 * torch.ones(d), high=2 * torch.ones(d))
    theta = prior.sample((n,))
    x = theta + 0.1 * torch.randn(theta.shape)
    return prior, theta, x


def test_train_npe_end_to_end():
    prior, theta, x = _toy()
    inference = build_npe(prior, model="maf", hidden_features=16, num_transforms=2)
    inference = inference.append_simulations(theta, x)
    result = train_npe(inference, training_batch_size=64, max_num_epochs=5)

    assert isinstance(result, NpeTrainingResult)
    assert result.n_epochs == len(result.train_losses) == len(result.val_losses)
    assert result.n_epochs >= 1
    assert result.n_nn_params > 0
    assert np.isfinite(result.best_val_loss)
    # posterior is usable: sampling at an observation returns the right shape
    samples = result.posterior.sample((50,), x=x[0], show_progress_bars=False)
    assert samples.shape == (50, theta.shape[1])


def test_learning_curve_shape_and_columns():
    prior, theta, x = _toy(seed=1)
    inference = build_npe(prior, hidden_features=16, num_transforms=2)
    inference = inference.append_simulations(theta, x)
    result = train_npe(inference, training_batch_size=64, max_num_epochs=4)

    lc = result.learning_curve()
    assert list(lc.columns) == ["epoch", "train_loss", "val_loss"]
    assert len(lc) == result.n_epochs
    assert lc["epoch"].tolist() == list(range(1, result.n_epochs + 1))


def test_hit_max_epochs_flag():
    prior, theta, x = _toy(seed=2)
    inference = build_npe(prior, hidden_features=16, num_transforms=2)
    inference = inference.append_simulations(theta, x)
    result = train_npe(inference, training_batch_size=64, max_num_epochs=3)
    # With a 3-epoch cap and no convergence, training stops at the cap.
    assert result.hit_max_epochs(3) == (result.n_epochs >= 3)


def test_nsf_takes_num_bins_without_error():
    """build_npe forwards num_bins only for nsf; constructing must not raise."""
    prior, theta, x = _toy(seed=3)
    inference = build_npe(prior, model="nsf", hidden_features=16, num_transforms=2, num_bins=4)
    inference = inference.append_simulations(theta, x)
    result = train_npe(inference, training_batch_size=64, max_num_epochs=2)
    assert result.n_nn_params > 0

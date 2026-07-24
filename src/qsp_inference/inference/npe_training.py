"""Train a neural posterior estimator and capture its learning curve.

A deliberate, thin wrapper over the sbi training loop: build the density
estimator and the NPE, run the loss, build the posterior, and read the per-epoch
train/validation losses back out of the inference object. Extracted so the flat
runner and any future end-to-end ``InferenceRun`` share one definition of "train
an NPE and record how it went" instead of each reaching into sbi's private
``_summary`` dict.

What stays with the caller, deliberately: appending simulations (so a
diagnostic can permute ``x`` first, or a pretrained estimator can skip training
entirely), the progress reporting, and any artifact writing. This module only
owns the estimator construction and the train/collect step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = ["NpeTrainingResult", "build_npe", "train_npe"]


@dataclass
class NpeTrainingResult:
    """A trained NPE plus the record of how training went.

    ``inference`` is the same object passed to :func:`train_npe`, returned so a
    caller that did not hold a reference can still reach it (e.g. to append more
    simulations for a sequential round).
    """

    posterior: object
    estimator: object
    inference: object
    train_losses: List[float]
    val_losses: List[float]
    best_val_loss: float
    n_epochs: int
    n_nn_params: int

    def learning_curve(self):
        """``(n_epochs, 3)`` DataFrame with columns ``epoch``, ``train_loss``, ``val_loss``."""
        import pandas as pd

        return pd.DataFrame(
            {
                "epoch": range(1, self.n_epochs + 1),
                "train_loss": self.train_losses,
                "val_loss": self.val_losses,
            }
        )

    def hit_max_epochs(self, max_num_epochs: int) -> bool:
        """True if training stopped on the epoch cap rather than by convergence."""
        return self.n_epochs >= max_num_epochs


def build_npe(
    prior,
    *,
    model: str = "maf",
    hidden_features: int = 50,
    num_transforms: int = 5,
    num_bins: int = 10,
):
    """Construct an sbi ``NPE`` over ``prior`` with a ``posterior_nn`` estimator.

    Args:
        prior: the distribution passed to ``NPE(prior=...)``. For amortized
            inference with a tempered proposal this is the *proposal* the training
            theta were drawn from; reporting reweights to the reporting prior
            afterwards. At temperature 1 the two coincide.
        model: density-estimator family (``maf`` | ``nsf`` | ``mdn`` | ...).
        hidden_features, num_transforms: flow capacity.
        num_bins: spline knots; forwarded only for ``nsf`` (other families reject it).

    Returns:
        An untrained ``NPE`` inference object. Call ``append_simulations`` then
        :func:`train_npe`.
    """
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    de_kwargs = dict(model=model, hidden_features=hidden_features, num_transforms=num_transforms)
    if model == "nsf":
        de_kwargs["num_bins"] = num_bins
    return NPE(prior=prior, density_estimator=posterior_nn(**de_kwargs))


def train_npe(
    inference,
    *,
    training_batch_size: int,
    max_num_epochs: int,
    show_train_summary: bool = True,
) -> NpeTrainingResult:
    """Train an NPE and read back its learning curve.

    ``inference`` must already have had ``append_simulations`` called (kept with
    the caller so it can permute ``x`` for a null-shuffle diagnostic, or add a
    sequential round's simulations, before training). Trains, builds the
    posterior, and collects the per-epoch losses from the inference summary so no
    caller has to reach into ``inference._summary`` itself.

    Args:
        inference: an ``NPE`` with simulations appended.
        training_batch_size, max_num_epochs: forwarded to ``inference.train``.
        show_train_summary: forwarded to ``inference.train``.

    Returns:
        An :class:`NpeTrainingResult`.
    """
    estimator = inference.train(
        training_batch_size=training_batch_size,
        max_num_epochs=max_num_epochs,
        show_train_summary=show_train_summary,
    )
    posterior = inference.build_posterior()

    summary = inference._summary
    train_losses = list(summary["training_loss"])
    val_losses = list(summary["validation_loss"])
    best_val_loss = float(summary["best_validation_loss"][-1])
    n_epochs = len(train_losses)
    n_nn_params = sum(p.numel() for p in estimator.parameters())

    return NpeTrainingResult(
        posterior=posterior,
        estimator=estimator,
        inference=inference,
        train_losses=train_losses,
        val_losses=val_losses,
        best_val_loss=best_val_loss,
        n_epochs=n_epochs,
        n_nn_params=n_nn_params,
    )

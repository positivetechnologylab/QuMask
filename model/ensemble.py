"""
Deep ensemble wrapper for uncertainty quantification.

Trains M independent ``QuMaskTransformer`` instances from different random
seeds and aggregates their predictions at inference time. Each member is
initialized and trained independently — no weight sharing, no distillation.

Uncertainty estimates
---------------------
At inference the ensemble produces:
  - p̂ = (1/M) Σₘ p̂⁽ᵐ⁾          (mean prediction, point estimate)
  - σᵢ² = (1/M) Σₘ (p̂ᵢ⁽ᵐ⁾ − p̂ᵢ)²  (per-bin variance)

These are approximate uncertainty intervals. For rigorous coverage guarantees,
pass ``member_preds`` to ``conformal.ConformalPredictor.calibrate``.

Training
--------
Members are trained sequentially by default (one GPU, one at a time). Each
member gets seed ``cfg["training"]["seed_base"] + m``. If multiple GPUs are
available, ``train_all`` can optionally dispatch members to separate devices.

Checkpointing
-------------
Each member's weights are saved to ``{checkpoint_dir}/member_{m}.pt`` after
training. ``load_ensemble`` restores all members from disk, enabling inference
without re-training.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from model.transformer import QuMaskTransformer


class QuMaskEnsemble:
    """Ensemble of M independent QuMaskTransformer models.

    Args:
        models: List of M ``QuMaskTransformer`` instances.
    """

    def __init__(self, models: list[QuMaskTransformer]) -> None:
        ...

    @property
    def M(self) -> int:
        """Number of ensemble members."""
        ...

    def predict(
        self,
        x: Tensor,
        device: torch.device | None = None,
    ) -> dict[str, Tensor]:
        """Run inference through all ensemble members.

        Each member produces a softmax probability vector. The ensemble
        aggregates these into a mean prediction and per-bin standard deviation.

        Args:
            x: Float tensor of shape ``(batch, n_blocks, F)``.
            device: Device to run inference on. If None, uses the device of
                the first model's parameters.

        Returns:
            Dict with keys:
            - ``"p_hat"``: mean prediction, shape ``(batch, 2**k)``, sums to 1
            - ``"sigma"``: per-bin std, shape ``(batch, 2**k)``
            - ``"member_preds"``: all members' predictions,
              shape ``(M, batch, 2**k)``, for conformal calibration
        """
        ...

    def save(self, checkpoint_dir: str) -> None:
        """Save all member weights to ``{checkpoint_dir}/member_{m}.pt``.

        Args:
            checkpoint_dir: Directory path. Created if it does not exist.
        """
        ...


def train_single_member(
    model: QuMaskTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float,
    epochs: int,
    weight_decay: float,
    patience: int,
    device: torch.device,
    seed: int,
) -> tuple[QuMaskTransformer, dict[str, list[float]]]:
    """Train one ensemble member to convergence.

    Uses AdamW with the forward KL loss: KL(p* ∥ p̂). Early stopping monitors
    validation loss and restores the best checkpoint.

    The KL loss is computed as:
        loss = F.kl_div(F.log_softmax(logits, dim=-1), p_star, reduction='batchmean')

    This is numerically stable and equivalent to KL(p* ∥ p̂) when ``p_star``
    is passed as the target (not log-target).

    Args:
        model: Uninitialized ``QuMaskTransformer`` (weights randomized at call time
            using ``seed``).
        train_loader: DataLoader yielding ``(features, p_star)`` batches.
        val_loader: DataLoader for validation loss monitoring.
        lr: AdamW learning rate.
        epochs: Maximum number of training epochs.
        weight_decay: L2 regularization coefficient.
        patience: Number of epochs without val improvement before early stopping.
        device: Torch device (CPU or CUDA).
        seed: RNG seed applied before weight initialization and data sampling.

    Returns:
        Tuple of:
        - Trained model with best validation-loss weights restored.
        - History dict with keys ``"train_loss"`` and ``"val_loss"``, each a
          list of per-epoch scalar values.
    """
    ...


def train_all(
    cfg: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    checkpoint_dir: str,
    device: torch.device | None = None,
) -> QuMaskEnsemble:
    """Train a full ensemble of M members sequentially.

    Instantiates M models via ``build_model_from_config``, trains each with
    ``train_single_member``, saves checkpoints, and returns a ``QuMaskEnsemble``.

    Args:
        cfg: Config dict from ``config/default.yaml``.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        checkpoint_dir: Directory to save per-member checkpoints.
        device: Training device. Defaults to CUDA if available, else CPU.

    Returns:
        Trained ``QuMaskEnsemble`` with all M members loaded.
    """
    ...


def load_ensemble(
    cfg: dict,
    checkpoint_dir: str,
    device: torch.device | None = None,
) -> QuMaskEnsemble:
    """Restore a saved ensemble from checkpoint files.

    Expects ``{checkpoint_dir}/member_{m}.pt`` for m in 0..M-1.

    Args:
        cfg: Config dict used to reconstruct model architecture.
        checkpoint_dir: Directory containing per-member checkpoint files.
        device: Device to map weights onto. Defaults to CPU.

    Returns:
        ``QuMaskEnsemble`` with all members in eval mode.
    """
    ...

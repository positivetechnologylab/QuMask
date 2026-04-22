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


def make_optimizer(model: QuMaskTransformer, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    """Construct the optimizer for a single ensemble member.

    Swap this function to change optimizer type or schedule without touching
    the training loop.

    Args:
        model: The model whose parameters will be optimized.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.

    Returns:
        Configured optimizer.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def compute_loss(logits: Tensor, p_star: Tensor) -> Tensor:
    """Compute the training loss from logits and target distribution.

    Swap this function to change the loss without touching the training loop.
    Current: forward KL — KL(p* ∥ p̂), computed as:
        kl_div(log_softmax(logits), p_star, reduction='batchmean')

    Args:
        logits: Raw model output, shape ``(batch, output_dim)``.
        p_star: Target distribution, shape ``(batch, output_dim)``, sums to 1.

    Returns:
        Scalar loss tensor.
    """
    return nn.functional.kl_div(
        nn.functional.log_softmax(logits, dim=-1), p_star, reduction="batchmean"
    )


class QuMaskEnsemble:
    """Ensemble of M independent QuMaskTransformer models.

    Args:
        models: List of M ``QuMaskTransformer`` instances.
    """

    def __init__(self, models: list[QuMaskTransformer]) -> None:
        self.models = models

    @property
    def M(self) -> int:
        """Number of ensemble members."""
        return len(self.models)

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
        if device is None:
            device = next(self.models[0].parameters()).device
        x = x.to(device)
        preds = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                logits = model(x)
                preds.append(torch.softmax(logits, dim=-1))
        member_preds = torch.stack(preds, dim=0)  # (M, batch, 2**k)
        p_hat = member_preds.mean(dim=0)
        sigma = member_preds.std(dim=0, correction=0)
        return {"p_hat": p_hat, "sigma": sigma, "member_preds": member_preds}

    def save(self, checkpoint_dir: str) -> None:
        """Save all member weights to ``{checkpoint_dir}/member_{m}.pt``.

        Args:
            checkpoint_dir: Directory path. Created if it does not exist.
        """
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        for m, model in enumerate(self.models):
            torch.save(model.state_dict(), Path(checkpoint_dir) / f"member_{m}.pt")


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
    torch.manual_seed(seed)
    model = model.to(device)
    optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        model.train()
        train_loss = 0.0
        for features, p_star in train_loader:
            features, p_star = features.to(device), p_star.to(device)
            optimizer.zero_grad()
            logits = model(features)
            loss = compute_loss(logits, p_star)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * features.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, p_star in val_loader:
                features, p_star = features.to(device), p_star.to(device)
                logits = model(features)
                loss = compute_loss(logits, p_star)
                val_loss += loss.item() * features.size(0)
        val_loss /= len(val_loader.dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


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
    from model.transformer import build_model_from_config

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    M = cfg["ensemble"]["M"]
    seed_base = cfg["training"]["seed_base"]
    tr = cfg["training"]
    checkpoint_path = Path(checkpoint_dir)

    trained_models = []
    for m in range(M):
        model = build_model_from_config(cfg)
        model, _ = train_single_member(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=tr["lr"],
            epochs=tr["epochs"],
            weight_decay=tr["weight_decay"],
            patience=tr["patience"],
            device=device,
            seed=seed_base + m,
        )
        trained_models.append(model)

    ensemble = QuMaskEnsemble(trained_models)
    ensemble.save(checkpoint_dir)
    return ensemble


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
    from model.transformer import build_model_from_config

    if device is None:
        device = torch.device("cpu")

    M = cfg["ensemble"]["M"]
    checkpoint_path = Path(checkpoint_dir)
    models = []
    for m in range(M):
        model = build_model_from_config(cfg)
        path = checkpoint_path / f"member_{m}.pt"
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        model.to(device).eval()
        models.append(model)
    return QuMaskEnsemble(models)

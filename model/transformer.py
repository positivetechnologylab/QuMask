"""
QuMask Transformer: distribution recovery from correlator block features.

Architecture overview
---------------------
The model treats the 1000 blocks as a set of exchangeable tokens. Each block's
correlator feature vector is projected into a shared embedding space, processed
by a stack of self-attention encoder layers, aggregated by mean pooling (which
is permutation-invariant — block order carries no information), and decoded to
a probability distribution over the 2^k target basis states.

Design choices
--------------
- No positional encoding: block ordering is meaningless; mean pooling after
  attention enforces permutation invariance at the aggregation step.
- Standard PyTorch TransformerEncoder: no need for a custom attention
  implementation at this scale.
- Logits (not probabilities) are returned from ``forward`` for numerical
  stability when computing KL loss with ``torch.nn.functional.kl_div``.
- The model is fully parameterized by (F, output_dim) derived from (n, k),
  making it straightforward to extend to other (n, k) configurations by
  changing the config and re-instantiating.

Hyperparameter guidance (k=3, n=10, F=175)
-------------------------------------------
Start with the Small config (d_model=64, n_layers=2). The signal is
low-dimensional and the "find the consistent subset" task does not require deep
representations. Scale up only if the validation loss plateaus early.

    Config  | d_model | n_heads | n_layers | d_ff | ~Params
    --------|---------|---------|----------|------|---------
    Small   |   64    |    4    |    2     |  128 |  ~200k
    Medium  |  128    |    4    |    4     |  256 |  ~800k
    Large   |  256    |    8    |    6     |  512 |   ~3M
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class QuMaskTransformer(nn.Module):
    """Transformer-based distribution recovery model for the QuMask protocol.

    Takes a ``(batch, n_blocks, F)`` correlator feature tensor and returns
    ``(batch, output_dim)`` logits over target basis states.

    Args:
        F: Input feature dimension per block. Equals ``feature_dim(n, k)``.
        output_dim: Number of output classes. Equals ``2**k``.
        d_model: Transformer embedding dimension. Must be divisible by ``n_heads``.
        n_heads: Number of attention heads.
        n_layers: Number of ``TransformerEncoderLayer`` blocks.
        d_ff: Feed-forward hidden dimension within each encoder layer.
        dropout: Dropout probability applied within the encoder and output head.
    """

    def __init__(
        self,
        F: int,
        output_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(batch, n_blocks, F)``. Values in [−1, 1]
               (correlator features; no further normalization is required).

        Returns:
            Float tensor of shape ``(batch, output_dim)`` containing unnormalized
            logits. Apply ``torch.softmax(..., dim=-1)`` at inference time to
            obtain probability estimates. Do NOT apply softmax before passing to
            the KL loss — use ``torch.nn.functional.kl_div`` with ``log_target=False``
            and ``log_softmax`` applied to the logits instead.
        """
        ...

    @property
    def n_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        ...


def build_model_from_config(cfg: dict) -> QuMaskTransformer:
    """Instantiate a ``QuMaskTransformer`` from a config dict.

    Computes ``F`` and ``output_dim`` from ``cfg["data"]["n"]`` and
    ``cfg["data"]["k"]`` using ``data.features.feature_dim``, then passes
    the model hyperparameters from ``cfg["model"]``.

    Args:
        cfg: Nested config dict as loaded from ``config/default.yaml``.

    Returns:
        An uninitialized (randomly weighted) ``QuMaskTransformer``.
    """
    ...

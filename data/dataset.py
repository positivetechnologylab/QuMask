"""
PyTorch Dataset and DataLoader wrappers for QuMask training instances.

Expects data in the format produced by ``simulate.generate_dataset`` and saved
via ``simulate.save_dataset``. Each item is one adversarial instance:
  - ``features``: the (n_blocks, F) correlator tensor — model input
  - ``p_star``: the (2**k,) true target distribution — training label

The raw bitstrings and target_positions arrays are retained in the .npz files
for evaluation-time oracle marginal extraction, but are not returned by the
Dataset (the model never sees them during training).

Splits
------
The dataset is split into train / val / test / calibration. The calibration
split is a subset of the test split reserved for conformal predictor fitting.
Split indices are determined by ``get_dataloaders`` and are reproducible given
a fixed seed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class QuMaskDataset(Dataset):
    """PyTorch Dataset for QuMask adversarial instances.

    Loads the compressed numpy archive produced by ``simulate.save_dataset``
    and exposes ``(feature_tensor, p_star)`` pairs. All arrays are kept as
    numpy until ``__getitem__`` converts them to tensors.

    Args:
        path: Path to a ``.npz`` file produced by ``simulate.save_dataset``.

    Attributes:
        features: Float32 array of shape ``(N, n_blocks, F)``.
        p_stars: Float64 array of shape ``(N, 2**k)``.
        bitstrings: Uint8 array of shape ``(N, n_blocks, shots_per_block, n)``.
            Retained for evaluation; not returned by ``__getitem__``.
        target_positions: Int array of shape ``(N, n_blocks, k)``.
            Retained for evaluation; not returned by ``__getitem__``.
        n_instances: Total number of instances in this file.
    """

    def __init__(self, path: str) -> None:
        ...

    def __len__(self) -> int:
        """Return the number of instances in the dataset."""
        ...

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (features, p_star) pair for instance ``idx``.

        Returns:
            features: Float32 tensor of shape ``(n_blocks, F)``.
            p_star: Float32 tensor of shape ``(2**k,)``.
        """
        ...

    def get_eval_arrays(self, idx: int) -> dict[str, np.ndarray]:
        """Return the raw arrays needed for evaluation-time metric computation.

        These are not used during training. Call this at test time to obtain
        the inputs for ``metrics.empirical_marginal`` and ``metrics.compute_all_metrics``.

        Args:
            idx: Instance index.

        Returns:
            Dict with keys:
            - ``"bitstrings"``: shape ``(n_blocks, shots_per_block, n)``, uint8
            - ``"target_positions"``: shape ``(n_blocks, k)``, int
            - ``"p_star"``: shape ``(2**k,)``, float64
        """
        ...


def get_dataloaders(
    train_path: str,
    val_path: str,
    test_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct train, validation, and test DataLoaders from pre-split .npz files.

    Data is pre-split at generation time (see ``simulate.generate_dataset``
    called separately for each split). This keeps split logic outside the
    Dataset and makes splits reproducible without relying on in-memory shuffles.

    Args:
        train_path: Path to training split ``.npz``.
        val_path: Path to validation split ``.npz``.
        test_path: Path to test split ``.npz``.
        batch_size: Batch size for all loaders.
        num_workers: Worker processes for data loading. 0 = main process only.
        pin_memory: If True, use pinned memory for faster GPU transfers.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.
        Train loader shuffles; val and test loaders do not.
    """
    ...

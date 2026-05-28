"""
Tests for per-member checkpoint skip and incremental save in model/ensemble.py.

These tests validate that:
  1. train_all skips training a member when its checkpoint already exists on disk.
  2. Each member's checkpoint is saved immediately after training (before the
     next member begins), so a crash mid-ensemble doesn't lose completed members.

Both tests use the @pytest.mark.slow marker since they invoke train_all.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch

from model.ensemble import train_all, QuMaskEnsemble, load_ensemble
from model.transformer import QuMaskTransformer
from data.dataset import QuMaskDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_loaders(tiny_dataset_path: str, batch_size: int = 8):
    """Split the 20-instance tiny dataset 14/6 train/val."""
    ds = QuMaskDataset(tiny_dataset_path)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [14, 6], generator=torch.Generator().manual_seed(0)
    )
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def _fast_cfg(default_cfg: dict, M: int = 2) -> dict:
    return {
        **default_cfg,
        "ensemble": {"M": M},
        "training": {**default_cfg["training"], "epochs": 2, "patience": 10},
    }


# ---------------------------------------------------------------------------
# Test 1: existing member checkpoint is not overwritten
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_skip_existing_member(tiny_dataset_path, default_cfg, tmp_path):
    """If member_0.pt already exists before train_all is called, it must not
    be overwritten. The mtime of the pre-existing file should be unchanged."""
    train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
    cfg = _fast_cfg(default_cfg, M=2)

    # Write a fake member_0.pt (just a small tensor state dict matching the arch).
    model_arch = QuMaskTransformer(
        F=cfg["data"]["n"] * 10,  # rough F, actual value doesn't matter for this test
        output_dim=2 ** cfg["data"]["k"],
        **cfg["model"],
    )
    # Build a real model to get a valid state dict shape.
    from model.transformer import build_model_from_config
    real_model = build_model_from_config(cfg)
    ckpt_path = tmp_path / "member_0.pt"
    torch.save(real_model.state_dict(), str(ckpt_path))

    # Record mtime before calling train_all.
    mtime_before = ckpt_path.stat().st_mtime

    # Small sleep so any write would produce a detectably different mtime.
    time.sleep(0.05)

    train_all(cfg, train_loader, val_loader, str(tmp_path))

    mtime_after = ckpt_path.stat().st_mtime
    assert mtime_after == mtime_before, (
        "member_0.pt was overwritten by train_all even though it already existed. "
        "Per-member checkpoint skip is not working."
    )

    # member_1.pt must be newly created.
    assert (tmp_path / "member_1.pt").exists(), "member_1.pt was not created."


# ---------------------------------------------------------------------------
# Test 2: per-member save happens before next member trains
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_per_member_save_before_next_member(tiny_dataset_path, default_cfg, tmp_path):
    """After training member m, member_m.pt must exist on disk before member
    m+1 begins training. We verify this by intercepting train_single_member
    calls and checking checkpoint existence at each call boundary."""
    from model import ensemble as ensemble_module

    train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
    cfg = _fast_cfg(default_cfg, M=3)

    checkpoint_existence_at_call: list[dict[int, bool]] = []
    original_train = ensemble_module.train_single_member

    call_count = [0]

    def recording_train(model, train_loader, val_loader, lr, epochs,
                        weight_decay, patience, device, seed):
        m = call_count[0]
        # Record which checkpoints exist at the START of this member's training.
        snapshot = {
            j: (tmp_path / f"member_{j}.pt").exists()
            for j in range(cfg["ensemble"]["M"])
        }
        checkpoint_existence_at_call.append(snapshot)
        call_count[0] += 1
        return original_train(model, train_loader, val_loader, lr, epochs,
                              weight_decay, patience, device, seed)

    with patch.object(ensemble_module, "train_single_member", side_effect=recording_train):
        train_all(cfg, train_loader, val_loader, str(tmp_path))

    # member_0 must not exist when member_0 starts training (it's being trained now).
    assert not checkpoint_existence_at_call[0][0], (
        "member_0.pt should not exist before member_0 trains."
    )
    # member_0 must exist by the time member_1 starts training.
    assert checkpoint_existence_at_call[1][0], (
        "member_0.pt must be saved before member_1 begins training. "
        "Per-member incremental save is not working."
    )
    # member_1 must exist by the time member_2 starts training.
    assert checkpoint_existence_at_call[2][1], (
        "member_1.pt must be saved before member_2 begins training."
    )

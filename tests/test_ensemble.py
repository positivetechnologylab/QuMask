"""Tests for model/ensemble.py"""

import pytest
import torch
import yaml
from pathlib import Path

from model.transformer import QuMaskTransformer
from model.ensemble import QuMaskEnsemble, train_single_member, train_all, load_ensemble
from data.dataset import QuMaskDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_models(M: int, F: int = 175, output_dim: int = 8) -> list[QuMaskTransformer]:
    return [QuMaskTransformer(F=F, output_dim=output_dim) for _ in range(M)]


def _tiny_loaders(tiny_dataset_path: str, batch_size: int = 8):
    """Split the 20-instance tiny dataset 14/6 train/val."""
    ds = QuMaskDataset(tiny_dataset_path)
    train_ds, val_ds = torch.utils.data.random_split(
        ds, [14, 6], generator=torch.Generator().manual_seed(0)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# TestQuMaskEnsemblePredict
# ---------------------------------------------------------------------------

class TestQuMaskEnsemblePredict:
    def test_output_keys(self, F):
        ensemble = QuMaskEnsemble(_random_models(3, F=F))
        x = torch.randn(4, 10, F)
        out = ensemble.predict(x)
        assert set(out.keys()) == {"p_hat", "sigma", "member_preds"}

    def test_output_shapes(self, F, output_dim):
        M = 3
        batch = 4
        ensemble = QuMaskEnsemble(_random_models(M, F=F, output_dim=output_dim))
        x = torch.randn(batch, 10, F)
        out = ensemble.predict(x)
        assert out["p_hat"].shape == (batch, output_dim)
        assert out["sigma"].shape == (batch, output_dim)
        assert out["member_preds"].shape == (M, batch, output_dim)

    def test_p_hat_sums_to_one(self, F, output_dim):
        ensemble = QuMaskEnsemble(_random_models(3, F=F, output_dim=output_dim))
        x = torch.randn(4, 10, F)
        out = ensemble.predict(x)
        assert torch.allclose(out["p_hat"].sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_sigma_non_negative(self, F, output_dim):
        ensemble = QuMaskEnsemble(_random_models(3, F=F, output_dim=output_dim))
        x = torch.randn(4, 10, F)
        out = ensemble.predict(x)
        assert (out["sigma"] >= 0).all()

    def test_sigma_nonzero(self, F, output_dim):
        # Random-weight members will disagree — sigma should not be all zeros.
        ensemble = QuMaskEnsemble(_random_models(5, F=F, output_dim=output_dim))
        x = torch.randn(4, 10, F)
        out = ensemble.predict(x)
        assert out["sigma"].max().item() > 0

    def test_M_property(self, F):
        ensemble = QuMaskEnsemble(_random_models(7, F=F))
        assert ensemble.M == 7


# ---------------------------------------------------------------------------
# TestTrainSingleMember
# ---------------------------------------------------------------------------

class TestTrainSingleMember:
    @pytest.mark.slow
    def test_loss_decreases(self, tiny_dataset_path, F, output_dim):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        model = QuMaskTransformer(F=F, output_dim=output_dim)
        device = torch.device("cpu")
        _, history = train_single_member(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=1e-3, epochs=3, weight_decay=1e-4, patience=10,
            device=device, seed=0,
        )
        assert history["train_loss"][-1] < history["train_loss"][0]

    @pytest.mark.slow
    def test_history_keys_and_length(self, tiny_dataset_path, F, output_dim):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        model = QuMaskTransformer(F=F, output_dim=output_dim)
        _, history = train_single_member(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=1e-3, epochs=3, weight_decay=1e-4, patience=10,
            device=torch.device("cpu"), seed=0,
        )
        assert set(history.keys()) == {"train_loss", "val_loss"}
        assert len(history["train_loss"]) == len(history["val_loss"])
        assert len(history["train_loss"]) <= 3

    @pytest.mark.slow
    def test_all_losses_finite(self, tiny_dataset_path, F, output_dim):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        model = QuMaskTransformer(F=F, output_dim=output_dim)
        _, history = train_single_member(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=1e-3, epochs=3, weight_decay=1e-4, patience=10,
            device=torch.device("cpu"), seed=0,
        )
        import math
        for loss in history["train_loss"] + history["val_loss"]:
            assert math.isfinite(loss)

    @pytest.mark.slow
    def test_early_stopping(self, tiny_dataset_path, F, output_dim):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        model = QuMaskTransformer(F=F, output_dim=output_dim)
        max_epochs = 20
        _, history = train_single_member(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=0.0,  # zero lr → val loss constant → triggers stopping at patience+1
            epochs=max_epochs, weight_decay=0.0, patience=1,
            device=torch.device("cpu"), seed=0,
        )
        assert len(history["train_loss"]) < max_epochs


# ---------------------------------------------------------------------------
# TestCheckpointRoundTrip
# ---------------------------------------------------------------------------

class TestCheckpointRoundTrip:
    @pytest.mark.slow
    def test_predict_identical_after_roundtrip(self, tiny_dataset_path, F, output_dim, tmp_path, default_cfg):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        model = QuMaskTransformer(F=F, output_dim=output_dim)
        model, _ = train_single_member(
            model=model, train_loader=train_loader, val_loader=val_loader,
            lr=1e-3, epochs=2, weight_decay=1e-4, patience=10,
            device=torch.device("cpu"), seed=0,
        )
        ensemble = QuMaskEnsemble([model])
        ensemble.save(str(tmp_path))

        cfg = dict(default_cfg)
        cfg = {**default_cfg, "ensemble": {"M": 1}}
        loaded = load_ensemble(cfg, str(tmp_path), device=torch.device("cpu"))

        x = torch.randn(2, 10, F)
        out_orig = ensemble.predict(x)
        out_loaded = loaded.predict(x)
        assert torch.allclose(out_orig["p_hat"], out_loaded["p_hat"], atol=1e-6)

    def test_load_nonexistent_raises(self, default_cfg, tmp_path):
        cfg = {**default_cfg, "ensemble": {"M": 1}}
        with pytest.raises(Exception):
            load_ensemble(cfg, str(tmp_path / "does_not_exist"), device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# TestTrainAll
# ---------------------------------------------------------------------------

class TestTrainAll:
    @pytest.mark.slow
    def test_returns_ensemble_with_M_members(self, tiny_dataset_path, default_cfg, tmp_path):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        cfg = {**default_cfg, "ensemble": {"M": 2}, "training": {**default_cfg["training"], "epochs": 2, "patience": 10}}
        ensemble = train_all(cfg, train_loader, val_loader, str(tmp_path))
        assert isinstance(ensemble, QuMaskEnsemble)
        assert ensemble.M == 2

    @pytest.mark.slow
    def test_checkpoint_files_exist(self, tiny_dataset_path, default_cfg, tmp_path):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        cfg = {**default_cfg, "ensemble": {"M": 2}, "training": {**default_cfg["training"], "epochs": 2, "patience": 10}}
        train_all(cfg, train_loader, val_loader, str(tmp_path))
        assert (tmp_path / "member_0.pt").exists()
        assert (tmp_path / "member_1.pt").exists()

    @pytest.mark.slow
    def test_members_have_different_weights(self, tiny_dataset_path, default_cfg, tmp_path):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        cfg = {**default_cfg, "ensemble": {"M": 2}, "training": {**default_cfg["training"], "epochs": 2, "patience": 10}}
        ensemble = train_all(cfg, train_loader, val_loader, str(tmp_path))
        sd0 = ensemble.models[0].state_dict()
        sd1 = ensemble.models[1].state_dict()
        any_differ = any(
            not torch.equal(sd0[k], sd1[k]) for k in sd0
        )
        assert any_differ


# ---------------------------------------------------------------------------
# TestLoadEnsemble
# ---------------------------------------------------------------------------

class TestLoadEnsemble:
    @pytest.mark.slow
    def test_correct_M(self, tiny_dataset_path, default_cfg, tmp_path):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        cfg = {**default_cfg, "ensemble": {"M": 2}, "training": {**default_cfg["training"], "epochs": 2, "patience": 10}}
        train_all(cfg, train_loader, val_loader, str(tmp_path))
        loaded = load_ensemble(cfg, str(tmp_path), device=torch.device("cpu"))
        assert loaded.M == 2

    @pytest.mark.slow
    def test_all_members_in_eval_mode(self, tiny_dataset_path, default_cfg, tmp_path):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        cfg = {**default_cfg, "ensemble": {"M": 2}, "training": {**default_cfg["training"], "epochs": 2, "patience": 10}}
        train_all(cfg, train_loader, val_loader, str(tmp_path))
        loaded = load_ensemble(cfg, str(tmp_path), device=torch.device("cpu"))
        for model in loaded.models:
            assert not model.training

    @pytest.mark.slow
    def test_predict_matches_before_save(self, tiny_dataset_path, default_cfg, tmp_path, F):
        train_loader, val_loader = _tiny_loaders(tiny_dataset_path)
        cfg = {**default_cfg, "ensemble": {"M": 2}, "training": {**default_cfg["training"], "epochs": 2, "patience": 10}}
        ensemble = train_all(cfg, train_loader, val_loader, str(tmp_path))
        loaded = load_ensemble(cfg, str(tmp_path), device=torch.device("cpu"))

        x = torch.randn(2, 10, F)
        out_orig = ensemble.predict(x)
        out_loaded = loaded.predict(x)
        assert torch.allclose(out_orig["p_hat"], out_loaded["p_hat"], atol=1e-6)

"""Tests for model/transformer.py"""

import torch
import torch.nn.functional as F
import pytest

from model.transformer import QuMaskTransformer, build_model_from_config
from data.features import feature_dim


def _make_model(F=175, output_dim=8, **kwargs) -> QuMaskTransformer:
    return QuMaskTransformer(F=F, output_dim=output_dim, **kwargs)


class TestQuMaskTransformerShape:
    def test_canonical_shape(self):
        model = _make_model()
        x = torch.randn(4, 1000, 175)
        out = model(x)
        assert out.shape == (4, 8)

    def test_batch_size_1(self):
        model = _make_model()
        x = torch.randn(1, 1000, 175)
        assert model(x).shape == (1, 8)

    def test_single_block(self):
        model = _make_model()
        x = torch.randn(2, 1, 175)
        assert model(x).shape == (2, 8)

    @pytest.mark.parametrize("cfg", [
        dict(d_model=64,  n_heads=4, n_layers=2, d_ff=128),
        dict(d_model=128, n_heads=4, n_layers=4, d_ff=256),
        dict(d_model=256, n_heads=8, n_layers=6, d_ff=512),
    ])
    def test_output_shape_all_configs(self, cfg):
        model = _make_model(**cfg)
        x = torch.randn(2, 50, 175)
        assert model(x).shape == (2, 8)


class TestQuMaskTransformerOutputType:
    def test_output_is_logits_not_probs(self):
        model = _make_model()
        model.eval()
        x = torch.randn(4, 100, 175)
        with torch.no_grad():
            out = model(x)
        # Logits: not all in [0,1] and rows don't sum to 1
        assert not (out >= 0).all() or not torch.allclose(out.sum(dim=-1), torch.ones(4))

    def test_softmax_sums_to_one(self):
        model = _make_model()
        model.eval()
        x = torch.randn(4, 100, 175)
        with torch.no_grad():
            out = model(x)
        probs = torch.softmax(out, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-5)

    def test_no_nan_or_inf(self):
        model = _make_model()
        model.eval()
        x = torch.randn(4, 100, 175)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()


class TestQuMaskTransformerPermutationInvariance:
    def test_permutation_invariant(self):
        torch.manual_seed(0)
        model = _make_model()
        model.eval()
        n_blocks = 50
        x = torch.randn(1, n_blocks, 175)
        perm = torch.randperm(n_blocks)
        x_perm = x[:, perm, :]
        with torch.no_grad():
            out = model(x)
            out_perm = model(x_perm)
        assert torch.allclose(out, out_perm, atol=1e-5)


class TestQuMaskTransformerGradients:
    def test_gradients_flow(self):
        torch.manual_seed(0)
        model = _make_model()
        x = torch.randn(4, 50, 175)
        target = torch.ones(4, 8) / 8  # uniform distribution

        logits = model(x)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()

        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_not_all_gradients_zero(self):
        torch.manual_seed(0)
        model = _make_model()
        x = torch.randn(4, 50, 175)
        target = torch.ones(4, 8) / 8

        logits = model(x)
        loss = F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
        loss.backward()

        all_zero = all(p.grad.abs().max().item() == 0.0 for p in model.parameters() if p.grad is not None)
        assert not all_zero


class TestNParameters:
    def test_positive_integer(self):
        model = _make_model()
        n = model.n_parameters
        assert isinstance(n, int)
        assert n > 0

    def test_small_config_ballpark(self):
        model = QuMaskTransformer(F=175, output_dim=8, d_model=64, n_heads=4, n_layers=2, d_ff=128)
        assert 50_000 <= model.n_parameters <= 400_000


class TestBuildModelFromConfig:
    def test_builds_without_error(self, default_cfg):
        model = build_model_from_config(default_cfg)
        assert isinstance(model, QuMaskTransformer)

    def test_output_dim_matches_k(self, default_cfg):
        model = build_model_from_config(default_cfg)
        k = default_cfg["data"]["k"]
        x = torch.randn(1, 10, feature_dim(default_cfg["data"]["n"], k))
        out = model(x)
        assert out.shape[-1] == 2 ** k

    def test_F_matches_feature_dim(self, default_cfg):
        n = default_cfg["data"]["n"]
        k = default_cfg["data"]["k"]
        F = feature_dim(n, k)
        model = build_model_from_config(default_cfg)
        x = torch.randn(1, 10, F)
        # If F is wrong this will raise a shape error
        model(x)

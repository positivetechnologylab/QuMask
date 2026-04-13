"""
Tests for model/transformer.py

Covers QuMaskTransformer and build_model_from_config. All tests use synthetic
random tensors — no Qiskit, no real data. None are marked slow, though the
larger model configs may take a second on CPU.


class TestQuMaskTransformerShape:
    # Test the canonical forward pass shape: input (batch=4, n_blocks=1000, F=175)
    #   → output (4, 8) for k=3.
    # Test with batch size 1 (edge case).
    # Test with n_blocks=1 (single block — degenerate but must not crash).
    # Test that output shape is (batch, 2**k) regardless of the model config
    #   (small/medium/large).


class TestQuMaskTransformerOutputType:
    # Test that the output is logits, not probabilities: values are not
    #   constrained to [0,1] and do not sum to 1 per row.
    # Test that after applying softmax(dim=-1), each row sums to 1.0.
    # Test that no output value is NaN or inf for random valid inputs.


class TestQuMaskTransformerPermutationInvariance:
    # This is the most important architectural correctness test.
    # Create a fixed input tensor of shape (1, n_blocks, F). Randomly permute
    #   the n_blocks axis to get a second input. Run both through the model.
    # The mean-pooled output (after the final attention layer) must be identical
    #   up to floating-point tolerance. This verifies that the model has no
    #   positional encoding and that mean pooling enforces permutation invariance.


class TestQuMaskTransformerGradients:
    # Test that a forward pass followed by loss.backward() produces non-None
    #   .grad attributes on all parameters (gradient flows through the full model).
    # Use a KL-style loss: F.kl_div(F.log_softmax(logits, dim=-1), target, reduction='batchmean').
    # Verify no gradients are zero for all parameters (a zero gradient on every
    #   parameter would indicate a dead network, though individual zeros are fine).


class TestNParameters:
    # Test that n_parameters returns a positive integer.
    # Test the approximate parameter count for the Small config (d_model=64,
    #   n_heads=4, n_layers=2, d_ff=128, F=175, output_dim=8): should be
    #   in the range [100_000, 400_000]. This is a ballpark check to catch
    #   obvious architecture bugs (e.g., missing a layer or wrong projection size).


class TestBuildModelFromConfig:
    # Test that build_model_from_config(default_cfg) returns a QuMaskTransformer
    #   without error.
    # Test that the model's output dimension matches 2**cfg["data"]["k"].
    # Test that the model's F matches feature_dim(cfg["data"]["n"], cfg["data"]["k"]).
"""

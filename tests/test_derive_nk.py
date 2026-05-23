"""
Unit tests for data/derive_nk.py.

All tests use small synthetic instances built from fixed-seed numpy arrays.
No Qiskit, no disk I/O except where explicitly noted.

The core correctness invariant: the k bits at derived target_positions in the
derived bitstrings must exactly match the k bits at original target_positions
in the master bitstrings (the target signal is preserved under projection).
"""

from __future__ import annotations

import numpy as np
import pytest

from data.derive_nk import _select_block_cols, derive_split
from data.features import feature_dim, compute_block_features


# ---------------------------------------------------------------------------
# Shared synthetic master fixture (no Qiskit)
# ---------------------------------------------------------------------------

N_INST = 4
N_BLOCKS = 5
SHOTS = 3
N_MASTER = 15
K = 3


@pytest.fixture(scope="module")
def master_arrays() -> dict[str, np.ndarray]:
    """Synthetic (N=4, n_blocks=5, shots=3, n=15, k=3) master arrays.

    target_positions are drawn without replacement from [0, 15) per block.
    p_stars are random simplex points.
    bitstrings are random uint8.
    """
    rng = np.random.default_rng(7)

    bitstrings = rng.integers(0, 2, size=(N_INST, N_BLOCKS, SHOTS, N_MASTER), dtype=np.uint8)
    target_positions = np.empty((N_INST, N_BLOCKS, K), dtype=int)
    for i in range(N_INST):
        for b in range(N_BLOCKS):
            target_positions[i, b] = rng.choice(N_MASTER, size=K, replace=False)

    # Random probability simplex points for p_stars.
    raw = rng.exponential(1.0, size=(N_INST, 2**K))
    p_stars = raw / raw.sum(axis=1, keepdims=True)

    return {
        "bitstrings": bitstrings,
        "target_positions": target_positions,
        "p_stars": p_stars,
    }


@pytest.fixture(scope="module")
def master_npz(tmp_path_factory, master_arrays) -> str:
    """Write master arrays to a temp .npz and return the path."""
    from data.features import compute_block_features as cbf
    path = tmp_path_factory.mktemp("derive") / "master_train.npz"

    # Compute features for the master so the file has the full schema.
    F_master = feature_dim(N_MASTER, K)
    features_master = np.empty((N_INST, N_BLOCKS, F_master), dtype=np.float32)
    for i in range(N_INST):
        features_master[i] = cbf(master_arrays["bitstrings"][i], N_MASTER, K)

    np.savez_compressed(
        str(path),
        bitstrings=master_arrays["bitstrings"],
        target_positions=master_arrays["target_positions"],
        p_stars=master_arrays["p_stars"],
        features=features_master,
    )
    return str(path)


def _run_derive(master_npz, master_arrays, n_prime, tmp_path_factory, seed=0) -> dict[str, np.ndarray]:
    """Helper: run derive_split for n_prime and return loaded result dict."""
    out_path = tmp_path_factory.mktemp(f"derive_n{n_prime}") / "out.npz"
    derive_split(
        master_path=master_npz,
        n_prime=n_prime,
        k=K,
        out_path=str(out_path),
        seed=seed,
    )
    data = np.load(str(out_path))
    return {k: data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Test 1: output shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_prime", [6, 9, 12])
def test_output_shapes(master_npz, master_arrays, n_prime, tmp_path_factory):
    result = _run_derive(master_npz, master_arrays, n_prime, tmp_path_factory)
    F_prime = feature_dim(n_prime, K)

    assert result["bitstrings"].shape == (N_INST, N_BLOCKS, SHOTS, n_prime)
    assert result["features"].shape == (N_INST, N_BLOCKS, F_prime)
    assert result["target_positions"].shape == (N_INST, N_BLOCKS, K)
    assert result["p_stars"].shape == (N_INST, 2**K)


# ---------------------------------------------------------------------------
# Test 2: target_positions in range [0, n_prime)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_prime", [6, 9, 12])
def test_target_positions_in_range(master_npz, master_arrays, n_prime, tmp_path_factory):
    result = _run_derive(master_npz, master_arrays, n_prime, tmp_path_factory)
    tp = result["target_positions"]
    assert tp.min() >= 0
    assert tp.max() < n_prime


# ---------------------------------------------------------------------------
# Test 3: target bits preserved (core correctness invariant)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_prime", [6, 9])
def test_target_bits_preserved(master_npz, master_arrays, n_prime, tmp_path_factory):
    result = _run_derive(master_npz, master_arrays, n_prime, tmp_path_factory)

    master_bits = master_arrays["bitstrings"]          # (N, n_blocks, shots, 15)
    master_tp = master_arrays["target_positions"]       # (N, n_blocks, k)
    derived_bits = result["bitstrings"]                 # (N, n_blocks, shots, n_prime)
    derived_tp = result["target_positions"]             # (N, n_blocks, k)

    for i in range(N_INST):
        for b in range(N_BLOCKS):
            # Original k bits at master target positions.
            orig = master_bits[i, b][:, master_tp[i, b]]   # (shots, k) — two-step
            # Derived k bits at derived target positions.
            deriv = derived_bits[i, b][:, derived_tp[i, b]]  # (shots, k) — two-step
            assert np.array_equal(orig, deriv), (
                f"Target bits mismatch at instance {i}, block {b}: "
                f"master_tp={master_tp[i,b]}, derived_tp={derived_tp[i,b]}"
            )


# ---------------------------------------------------------------------------
# Test 4: p_stars unchanged
# ---------------------------------------------------------------------------

def test_p_stars_unchanged(master_npz, master_arrays, tmp_path_factory):
    result = _run_derive(master_npz, master_arrays, 6, tmp_path_factory)
    assert np.array_equal(result["p_stars"], master_arrays["p_stars"])


# ---------------------------------------------------------------------------
# Test 5: features consistent with derived bitstrings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_prime", [6, 9])
def test_features_consistent_with_bitstrings(master_npz, master_arrays, n_prime, tmp_path_factory):
    result = _run_derive(master_npz, master_arrays, n_prime, tmp_path_factory)
    # Spot-check instance 0.
    recomputed = compute_block_features(result["bitstrings"][0], n_prime, K)
    assert np.allclose(result["features"][0], recomputed, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: decoy columns vary across blocks
# ---------------------------------------------------------------------------

def test_decoy_cols_vary_across_blocks(master_arrays):
    """For a fixed instance, the set of selected decoy columns should not be
    identical across all blocks (target positions vary, so available sets vary)."""
    n_prime = 6
    seen_decoys: list[frozenset] = []
    rng_base = 0
    i = 0  # instance 0

    for b in range(N_BLOCKS):
        target_pos = master_arrays["target_positions"][i, b]
        rng = np.random.default_rng([rng_base, i, b])
        _, decoy_cols = _select_block_cols(target_pos, N_MASTER, n_prime, K, rng)
        seen_decoys.append(frozenset(decoy_cols))

    # With 5 blocks and random target positions, all decoy sets being identical
    # would be astronomically unlikely. Assert at least 2 distinct sets.
    assert len(set(seen_decoys)) >= 2, (
        "Decoy column selections are identical across all blocks — "
        "per-block randomization is not working."
    )


# ---------------------------------------------------------------------------
# Test 7: no column collisions
# ---------------------------------------------------------------------------

def test_no_column_collisions(master_arrays):
    """selected_cols must have no duplicates and exactly n_prime entries."""
    n_prime = 8
    rng_base = 0
    for i in range(N_INST):
        for b in range(N_BLOCKS):
            target_pos = master_arrays["target_positions"][i, b]
            rng = np.random.default_rng([rng_base, i, b])
            target_cols, decoy_cols = _select_block_cols(target_pos, N_MASTER, n_prime, K, rng)
            selected = target_cols + decoy_cols
            assert len(selected) == n_prime
            assert len(set(selected)) == n_prime, (
                f"Duplicate columns at instance {i}, block {b}: {selected}"
            )


# ---------------------------------------------------------------------------
# Test 8: determinism
# ---------------------------------------------------------------------------

def test_determinism(master_npz, tmp_path_factory):
    """Two runs with the same seed produce byte-for-byte identical arrays."""
    out1 = tmp_path_factory.mktemp("det1") / "out.npz"
    out2 = tmp_path_factory.mktemp("det2") / "out.npz"

    derive_split(master_path=master_npz, n_prime=7, k=K, out_path=str(out1), seed=42)
    derive_split(master_path=master_npz, n_prime=7, k=K, out_path=str(out2), seed=42)

    d1 = np.load(str(out1))
    d2 = np.load(str(out2))
    for key in d1.files:
        assert np.array_equal(d1[key], d2[key]), f"Mismatch in key '{key}'"


# ---------------------------------------------------------------------------
# Test 9: empirical marginal approximates p_star (statistical, larger arrays)
# ---------------------------------------------------------------------------

def test_empirical_marginal_approximates_p_star(tmp_path_factory):
    """With enough blocks and shots, the empirical marginal over derived target
    positions should be close to p_star."""
    from utils.metrics import empirical_marginal, tvd

    rng = np.random.default_rng(99)
    n_blocks_large = 300
    shots_large = 20
    n_inst = 1

    # Build a master with a known p_star (peaked distribution).
    p_star = np.array([0.5, 0.1, 0.3, 0.05, 0.02, 0.01, 0.01, 0.01])  # 2**3
    assert abs(p_star.sum() - 1.0) < 1e-9

    # Sample bitstrings: target columns always at [0, 1, 2], shot from p_star.
    powers = 1 << np.arange(K)
    indices = rng.choice(2**K, size=(n_blocks_large, shots_large), p=p_star)
    target_bits = ((indices[:, :, None] >> np.arange(K)) & 1).astype(np.uint8)

    # Fill remaining columns with noise.
    bitstrings = rng.integers(0, 2, size=(n_inst, n_blocks_large, shots_large, N_MASTER), dtype=np.uint8)
    bitstrings[0, :, :, :K] = target_bits
    target_positions = np.tile(np.arange(K, dtype=int), (n_inst, n_blocks_large, 1))
    p_stars = p_star[None, :]

    F_master = feature_dim(N_MASTER, K)
    features_master = np.empty((n_inst, n_blocks_large, F_master), dtype=np.float32)
    for i in range(n_inst):
        features_master[i] = compute_block_features(bitstrings[i], N_MASTER, K)

    master_path = tmp_path_factory.mktemp("marginal") / "master.npz"
    np.savez_compressed(
        str(master_path),
        bitstrings=bitstrings,
        target_positions=target_positions,
        p_stars=p_stars,
        features=features_master,
    )

    out_path = master_path.parent / "derived.npz"
    derive_split(master_path=str(master_path), n_prime=6, k=K, out_path=str(out_path), seed=0)

    derived = np.load(str(out_path))
    p_hat_marginal = empirical_marginal(
        derived["bitstrings"][0], derived["target_positions"][0], K
    )
    distance = tvd(p_hat_marginal, p_star)
    assert distance < 0.15, (
        f"Empirical marginal too far from p_star: TVD={distance:.4f}. "
        "The target signal may not be preserved correctly."
    )

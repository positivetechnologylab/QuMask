"""
Smoke test for data/derive_nk.py using real Qiskit simulation.

Generates a tiny (N=2, n_blocks=5, shots=3, n=15, k=3) master dataset via
generate_instance, saves it to a temp .npz, derives n'=6 from it, and
verifies the output file is valid and loadable by QuMaskDataset.

Marked slow because it invokes Qiskit statevector simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.simulate import generate_instance
from data.features import compute_block_features, feature_dim
from data.derive_nk import derive_split
from data.dataset import QuMaskDataset


N_MASTER = 15
K = 3
N_PRIME = 6
N_INST = 2
N_BLOCKS = 5
SHOTS = 3


@pytest.mark.slow
def test_smoke_derive_end_to_end(tmp_path):
    """Full pipeline: generate (15,3) master -> derive (6,3) -> load as QuMaskDataset."""
    # --- Generate master instances via real Qiskit simulation. ---
    instances = [
        generate_instance(
            k=K, n=N_MASTER, n_blocks=N_BLOCKS, shots_per_block=SHOTS,
            target_depth_min=2, target_depth_max=3, seed=i,
        )
        for i in range(N_INST)
    ]

    F_master = feature_dim(N_MASTER, K)
    features_master = np.stack([inst["features"] for inst in instances])
    p_stars = np.stack([inst["p_star"] for inst in instances])
    bitstrings = np.stack([inst["bitstrings"] for inst in instances])
    target_positions = np.stack([inst["target_positions"] for inst in instances])

    master_path = tmp_path / "master.npz"
    np.savez_compressed(
        str(master_path),
        features=features_master,
        p_stars=p_stars,
        bitstrings=bitstrings,
        target_positions=target_positions,
    )

    # --- Derive (6,3) from master. ---
    out_path = tmp_path / "derived.npz"
    derive_split(
        master_path=str(master_path),
        n_prime=N_PRIME,
        k=K,
        out_path=str(out_path),
        seed=0,
    )

    assert out_path.exists(), "derive_split did not produce an output file."

    # --- Verify file structure. ---
    data = np.load(str(out_path))
    F_prime = feature_dim(N_PRIME, K)
    assert data["bitstrings"].shape == (N_INST, N_BLOCKS, SHOTS, N_PRIME)
    assert data["features"].shape == (N_INST, N_BLOCKS, F_prime)
    assert data["target_positions"].shape == (N_INST, N_BLOCKS, K)
    assert data["p_stars"].shape == (N_INST, 2**K)
    assert data["target_positions"].min() >= 0
    assert data["target_positions"].max() < N_PRIME

    # --- Verify loadable by QuMaskDataset. ---
    ds = QuMaskDataset(str(out_path))
    assert len(ds) == N_INST
    features_tensor, p_star_tensor = ds[0]
    assert features_tensor.shape == (N_BLOCKS, F_prime)
    assert p_star_tensor.shape == (2**K,)
    assert abs(p_star_tensor.sum().item() - 1.0) < 1e-5

    # --- Idempotency: running again skips (output exists). ---
    import time
    mtime_before = out_path.stat().st_mtime
    time.sleep(0.05)
    derive_split(
        master_path=str(master_path),
        n_prime=N_PRIME,
        k=K,
        out_path=str(out_path),
        seed=0,
    )
    assert out_path.stat().st_mtime == mtime_before, "derive_split overwrote existing output."

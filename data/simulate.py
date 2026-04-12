"""
Training data generation via QuMask protocol simulation.

Each instance produced here represents one adversarial observation: a full set
of 1000 blocks of 10 shots from a hidden k-qubit target circuit embedded in an
n-qubit system, along with the ground-truth target distribution p*.

The simulation mirrors the QuMask protocol exactly:
  - A random k-qubit target circuit is fixed for the instance.
  - Each block re-randomizes target qubit positions and decoy partitions.
  - Decoy registers are independent random Qiskit circuits.
  - The full n-qubit system is a product state; registers are sampled independently
    and interleaved by qubit position.

Integration with existing simulation code
------------------------------------------
If you already have logic that produces the 10k bitstrings and p*, adapt it to
return the following structure and pass it to ``features.compute_block_features``:

    {
        "bitstrings":        np.ndarray,  # shape (n_blocks, shots_per_block, n), uint8
        "p_star":            np.ndarray,  # shape (2**k,), float64
        "target_positions":  np.ndarray,  # shape (n_blocks, k), int
    }

``generate_instance`` below produces exactly this structure. If your code already
handles the Qiskit simulation, you can replace the body of ``generate_instance``
with a thin wrapper around your existing function.
"""

from __future__ import annotations

import numpy as np

from utils.circuits import (
    random_circuit,
    statevector_distribution,
    sample_shots,
    interleave_bitstrings,
)
from data.features import compute_block_features


def random_decoy_partition(
    available_qubits: np.ndarray,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Partition a set of qubit indices into disjoint groups via sequential coin flips.

    Qubits are processed in a random order. At each step, the current qubit is
    either appended to the current group (heads) or starts a new group (tails).
    This produces groups whose sizes follow a geometric-like distribution, with
    minimum group size 1 and maximum size ``len(available_qubits)``.

    Args:
        available_qubits: 1-D int array of qubit indices to partition (the n−k
            non-target qubits for this block).
        rng: A ``numpy.random.Generator`` instance for reproducibility.

    Returns:
        List of 1-D int arrays, each containing the qubit indices of one decoy
        group. The union of all groups equals ``available_qubits``.
    """
    ...


def generate_instance(
    k: int,
    n: int,
    n_blocks: int,
    shots_per_block: int,
    target_depth: int,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate one complete training instance.

    Simulates the QuMask protocol: fixes a random k-qubit target circuit, then
    for each block re-randomizes target qubit positions, constructs fresh decoy
    circuits on a random partition of the remaining qubits, samples bitstrings
    from the product-state joint distribution, and extracts correlator features.

    Args:
        k: Number of target qubits.
        n: Total system qubits.
        n_blocks: Number of blocks (default 1000).
        shots_per_block: Shots per block (default 10).
        target_depth: Base circuit depth. Decoy depths are sampled uniformly
            from ``[0.75 * target_depth, 1.25 * target_depth]``.
        seed: Optional RNG seed for full reproducibility of this instance.

    Returns:
        Dict with keys:
        - ``"features"``: float32 array of shape ``(n_blocks, F)`` where
          ``F = sum(C(n, j) for j in 1..k)`` — correlator features per block.
        - ``"p_star"``: float64 array of shape ``(2**k,)`` — exact target
          distribution from statevector simulation.
        - ``"bitstrings"``: uint8 array of shape ``(n_blocks, shots_per_block, n)``
          — raw bitstrings (retained for ``empirical_marginal`` at eval time).
        - ``"target_positions"``: int array of shape ``(n_blocks, k)`` — target
          qubit indices per block (retained for oracle marginal extraction).
    """
    ...


def generate_dataset(
    n_instances: int,
    k: int,
    n: int,
    n_blocks: int = 1000,
    shots_per_block: int = 10,
    target_depth: int = 4,
    seed_base: int = 0,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Generate a full dataset of N independent instances in parallel.

    Each instance gets seed ``seed_base + i`` for reproducibility. Instances
    are generated via ``generate_instance`` and stacked along a leading batch
    axis.

    Args:
        n_instances: Number of instances to generate.
        k: Number of target qubits.
        n: Total system qubits.
        n_blocks: Number of blocks per instance.
        shots_per_block: Shots per block.
        target_depth: Base target circuit depth.
        seed_base: Base seed; instance ``i`` uses ``seed_base + i``.
        n_jobs: Number of parallel worker processes (``joblib``). Set to 1 to
            disable parallelism (useful for debugging).

    Returns:
        Dict with keys:
        - ``"features"``: shape ``(n_instances, n_blocks, F)``, float32
        - ``"p_stars"``: shape ``(n_instances, 2**k)``, float64
        - ``"bitstrings"``: shape ``(n_instances, n_blocks, shots_per_block, n)``, uint8
        - ``"target_positions"``: shape ``(n_instances, n_blocks, k)``, int
    """
    ...


def save_dataset(dataset: dict[str, np.ndarray], path: str) -> None:
    """Save a dataset dict to a compressed numpy archive.

    Args:
        dataset: Dict returned by ``generate_dataset``.
        path: File path ending in ``.npz``.
    """
    ...


def load_dataset(path: str) -> dict[str, np.ndarray]:
    """Load a dataset dict from a compressed numpy archive.

    Args:
        path: File path ending in ``.npz``.

    Returns:
        Dict with the same keys as produced by ``generate_dataset``.
    """
    ...

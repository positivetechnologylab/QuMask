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
from joblib import Parallel, delayed

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
    shuffled = rng.permutation(available_qubits)
    D = len(shuffled)

    groups: list[np.ndarray] = []
    current: list[int] = []

    # D-1 independent coin flips determine whether to cut after each qubit
    # except the last (the last qubit always ends the final group).
    # rng.integers upper bound is exclusive, so [0, 2^(D-1)) covers all
    # possible split patterns uniformly.
    if D > 1:
        splits = int(rng.integers(0, 1 << (D - 1)))
    else:
        splits = 0

    for i, qubit in enumerate(shuffled):
        current.append(int(qubit))
        if i < D - 1 and (splits >> i) & 1:
            groups.append(np.array(current, dtype=int))
            current = []

    groups.append(np.array(current, dtype=int))
    return groups


def generate_instance(
    k: int,
    n: int,
    n_blocks: int,
    shots_per_block: int,
    target_depth_min: int,
    target_depth_max: int,
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
        target_depth_min: Lower bound (inclusive) of the per-instance target
            circuit depth range. Depth is drawn from
            ``U(target_depth_min, target_depth_max)`` once per instance so that
            the training set spans the full entropy spectrum.
        target_depth_max: Upper bound (inclusive) of the per-instance target
            circuit depth range.
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
    rng = np.random.default_rng(seed)

    # Draw target depth once per instance from the specified range.
    target_depth = int(rng.integers(target_depth_min, target_depth_max + 1))

    # Seed stream: reserve one seed for the target circuit, then one per block
    # for decoy circuit generation and shot sampling.
    target_circuit_seed = int(rng.integers(0, 2**31))
    target_circuit = random_circuit(k, target_depth, seed=target_circuit_seed)
    p_star = statevector_distribution(target_circuit)

    depth_lo = max(1, int(target_depth * 0.75))
    depth_hi = int(target_depth * 1.25)

    bitstrings = np.empty((n_blocks, shots_per_block, n), dtype=np.uint8)
    target_positions_all = np.empty((n_blocks, k), dtype=int)

    for b in range(n_blocks):
        # Per-block RNG state derived deterministically from the instance rng.
        block_seed = int(rng.integers(0, 2**31))
        block_rng = np.random.default_rng(block_seed)

        # Sample target qubit positions for this block.
        target_pos = block_rng.choice(n, size=k, replace=False)
        target_positions_all[b] = target_pos

        # Sample target bits from the fixed target distribution.
        target_shot_seed = int(block_rng.integers(0, 2**31))
        target_bits = sample_shots(p_star, shots_per_block, seed=target_shot_seed)

        # Partition remaining qubits into decoy groups.
        available = np.array([q for q in range(n) if q not in set(target_pos.tolist())])
        decoy_groups = random_decoy_partition(available, block_rng)

        # Generate and sample each decoy group independently.
        decoy_bits_list = []
        decoy_positions_list = []
        for group in decoy_groups:
            g = len(group)
            decoy_depth = int(block_rng.integers(depth_lo, depth_hi + 1))
            decoy_seed = int(block_rng.integers(0, 2**31))
            decoy_circuit = random_circuit(g, decoy_depth, seed=decoy_seed)
            p_decoy = statevector_distribution(decoy_circuit)
            decoy_shot_seed = int(block_rng.integers(0, 2**31))
            decoy_bits_list.append(sample_shots(p_decoy, shots_per_block, seed=decoy_shot_seed))
            decoy_positions_list.append(group)

        bitstrings[b] = interleave_bitstrings(
            target_bits, decoy_bits_list, target_pos, decoy_positions_list, n
        )

    features = compute_block_features(bitstrings, n, k)

    return {
        "features": features,
        "p_star": p_star,
        "bitstrings": bitstrings,
        "target_positions": target_positions_all,
    }


def generate_dataset(
    n_instances: int,
    k: int,
    n: int,
    n_blocks: int = 1000,
    shots_per_block: int = 10,
    target_depth_min: int = 1,
    target_depth_max: int = 8,
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
        target_depth_min: Lower bound of the per-instance target depth range.
        target_depth_max: Upper bound of the per-instance target depth range.
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
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_instance)(k, n, n_blocks, shots_per_block, target_depth_min, target_depth_max, seed=seed_base + i)
        for i in range(n_instances)
    )

    return {
        "features":         np.stack([r["features"] for r in results]),
        "p_stars":          np.stack([r["p_star"] for r in results]),
        "bitstrings":       np.stack([r["bitstrings"] for r in results]),
        "target_positions": np.stack([r["target_positions"] for r in results]),
    }


def run_red_herring_block(
    n: int,
    depth_lo: int,
    depth_hi: int,
    shots_per_block: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one block of shots with all n qubits running random walk circuits.

    Partitions all n qubits via random_decoy_partition (same method as decoy
    generation) and samples each group independently. There is no target signal.

    Returns:
        uint8 array of shape ``(shots_per_block, n)``.
    """
    all_qubits = np.arange(n)
    groups = random_decoy_partition(all_qubits, rng)

    bits_list = []
    positions_list = []
    for group in groups:
        g = len(group)
        depth = int(rng.integers(depth_lo, depth_hi + 1))
        circuit_seed = int(rng.integers(0, 2**31))
        circuit = random_circuit(g, depth, seed=circuit_seed)
        p = statevector_distribution(circuit)
        shot_seed = int(rng.integers(0, 2**31))
        bits_list.append(sample_shots(p, shots_per_block, seed=shot_seed))
        positions_list.append(group)

    # Interleave using a dummy empty target (no target qubits).
    dummy_target_bits = np.empty((shots_per_block, 0), dtype=np.uint8)
    dummy_target_pos = np.empty(0, dtype=int)
    return interleave_bitstrings(
        dummy_target_bits, bits_list, dummy_target_pos, positions_list, n
    )


def generate_instance_fixed(
    k: int,
    n: int,
    n_blocks: int,
    shots_per_block: int,
    target_depth_min: int,
    target_depth_max: int,
    num_red_herring_blocks: int = 0,
    seed: int | None = None,
) -> dict[str, np.ndarray]:
    """Generate one training instance with the target circuit at a fixed qubit position.

    Identical to ``generate_instance`` except the target qubit positions are
    drawn once before the block loop and held constant for all blocks.  Decoy
    partitions and decoy circuits are still re-randomized per block.  This
    mirrors the fixed-position obfuscation regime, under which the variance
    baseline is significantly more effective.

    Args:
        k: Number of target qubits.
        n: Total system qubits.
        n_blocks: Number of real (target-carrying) blocks.
        shots_per_block: Shots per block.
        target_depth_min: Lower bound of the per-instance target depth range.
        target_depth_max: Upper bound of the per-instance target depth range.
        num_red_herring_blocks: Number of fully-randomized blocks to interleave.
            These carry no target signal. Default 0 (no red herrings).
        seed: Optional RNG seed for full reproducibility.

    Returns:
        Dict with keys:
        - ``"features"``: float32 array of shape ``(total_blocks, F)`` where
          ``total_blocks = n_blocks + num_red_herring_blocks``.
        - ``"p_star"``: float64 array of shape ``(2**k,)``.
        - ``"bitstrings"``: uint8 array of shape ``(total_blocks, shots_per_block, n)``.
        - ``"target_positions"``: int array of shape ``(total_blocks, k)`` — every
          row is the fixed target position (RH rows store it too, for dtype consistency).
        - ``"red_herring_mask"``: bool array of shape ``(total_blocks,)`` — True
          for red herring slots, False for real blocks.
    """
    rng = np.random.default_rng(seed)

    target_depth = int(rng.integers(target_depth_min, target_depth_max + 1))

    target_circuit_seed = int(rng.integers(0, 2**31))
    target_circuit = random_circuit(k, target_depth, seed=target_circuit_seed)
    p_star = statevector_distribution(target_circuit)

    depth_lo = max(1, int(target_depth * 0.75))
    depth_hi = int(target_depth * 1.25)

    # Draw target position once; fixed for all real blocks.
    target_pos = rng.choice(n, size=k, replace=False)

    total_blocks = n_blocks + num_red_herring_blocks

    # Randomly assign which slots are red herrings.
    rh_mask = np.zeros(total_blocks, dtype=bool)
    if num_red_herring_blocks > 0:
        rh_indices = rng.permutation(total_blocks)[:num_red_herring_blocks]
        rh_mask[rh_indices] = True

    target_positions_all = np.tile(target_pos, (total_blocks, 1))
    bitstrings = np.empty((total_blocks, shots_per_block, n), dtype=np.uint8)

    for b in range(total_blocks):
        block_seed = int(rng.integers(0, 2**31))
        block_rng = np.random.default_rng(block_seed)

        if rh_mask[b]:
            bitstrings[b] = run_red_herring_block(n, depth_lo, depth_hi, shots_per_block, block_rng)
            continue

        target_shot_seed = int(block_rng.integers(0, 2**31))
        target_bits = sample_shots(p_star, shots_per_block, seed=target_shot_seed)

        available = np.array([q for q in range(n) if q not in set(target_pos.tolist())])
        decoy_groups = random_decoy_partition(available, block_rng)

        decoy_bits_list = []
        decoy_positions_list = []
        for group in decoy_groups:
            g = len(group)
            decoy_depth = int(block_rng.integers(depth_lo, depth_hi + 1))
            decoy_seed = int(block_rng.integers(0, 2**31))
            decoy_circuit = random_circuit(g, decoy_depth, seed=decoy_seed)
            p_decoy = statevector_distribution(decoy_circuit)
            decoy_shot_seed = int(block_rng.integers(0, 2**31))
            decoy_bits_list.append(sample_shots(p_decoy, shots_per_block, seed=decoy_shot_seed))
            decoy_positions_list.append(group)

        bitstrings[b] = interleave_bitstrings(
            target_bits, decoy_bits_list, target_pos, decoy_positions_list, n
        )

    features = compute_block_features(bitstrings, n, k)

    return {
        "features": features,
        "p_star": p_star,
        "bitstrings": bitstrings,
        "target_positions": target_positions_all,
        "red_herring_mask": rh_mask,
    }


def generate_dataset_fixed(
    n_instances: int,
    k: int,
    n: int,
    n_blocks: int = 1000,
    shots_per_block: int = 10,
    target_depth_min: int = 1,
    target_depth_max: int = 8,
    seed_base: int = 0,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """Generate a dataset of fixed-position instances in parallel.

    Each instance gets seed ``seed_base + i``.  Mirrors ``generate_dataset``
    but calls ``generate_instance_fixed`` so target qubit positions are held
    constant within each instance.

    Returns:
        Dict with keys:
        - ``"features"``: shape ``(n_instances, n_blocks, F)``, float32
        - ``"p_stars"``: shape ``(n_instances, 2**k)``, float64
        - ``"bitstrings"``: shape ``(n_instances, n_blocks, shots_per_block, n)``, uint8
        - ``"target_positions"``: shape ``(n_instances, n_blocks, k)``, int
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_instance_fixed)(k, n, n_blocks, shots_per_block, target_depth_min, target_depth_max, seed=seed_base + i)
        for i in range(n_instances)
    )

    return {
        "features":         np.stack([r["features"] for r in results]),
        "p_stars":          np.stack([r["p_star"] for r in results]),
        "bitstrings":       np.stack([r["bitstrings"] for r in results]),
        "target_positions": np.stack([r["target_positions"] for r in results]),
    }


def save_dataset(dataset: dict[str, np.ndarray], path: str) -> None:
    """Save a dataset dict to a compressed numpy archive.

    Args:
        dataset: Dict returned by ``generate_dataset``.
        path: File path ending in ``.npz``.
    """
    np.savez_compressed(path, **dataset)


def load_dataset(path: str) -> dict[str, np.ndarray]:
    """Load a dataset dict from a compressed numpy archive.

    Args:
        path: File path ending in ``.npz``.

    Returns:
        Dict with the same keys as produced by ``generate_dataset``.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}

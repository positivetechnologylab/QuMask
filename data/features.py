"""
Correlator feature extraction from raw bitstrings.

The feature representation for each block is the vector of all m-body Z-basis
correlators for m = 1, ..., k:

    φ_b = { ⟨Z_{i1} ··· Z_{im}⟩ : {i1,...,im} ⊆ [n], m = 1,...,k }

where qubits are encoded as Z_i = 1 - 2*x_i ∈ {−1, +1}, and each correlator
is the empirical mean of the product across all shots in the block.

Justification
-------------
A distribution over k qubits has 2^k − 1 free parameters. Recovering it from
marginal statistics requires correlators up to order k. Pairwise features are
insufficient for k ≥ 4.

Feature ordering
----------------
Features are ordered by subset size first, then lexicographically by qubit
indices within each size. The ordering is fixed by ``get_subset_index`` for a
given (n, k) and is consistent across all calls. The full index is precomputed
once and cached at module level.

Feature dimension
-----------------
F = Σⱼ₌₁ᵏ C(n, j)

For k=3, n=10: F = C(10,1) + C(10,2) + C(10,3) = 10 + 45 + 120 = 175
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from itertools import combinations

import numpy as np

CORRELATOR_MEMORY_WARN_BYTES = 500 * 1024 * 1024  # 500 MB


@lru_cache(maxsize=32)
def get_all_subsets(n: int, k: int) -> list[tuple[int, ...]]:
    """Return all non-empty subsets of [n] of size 1 through k, in feature order.

    Result is cached so the enumeration is only computed once per (n, k) pair.
    Feature order: subsets of size 1 first (lexicographic), then size 2, ..., k.

    Args:
        n: Total number of qubits.
        k: Maximum subset size (= number of target qubits).

    Returns:
        List of tuples, each a sorted tuple of qubit indices. Length equals
        F = Σⱼ₌₁ᵏ C(n, j).
    """
    subsets = []
    for m in range(1, k + 1):
        subsets.extend(combinations(range(n), m))
    return subsets


def feature_dim(n: int, k: int) -> int:
    """Return the feature dimension F = Σⱼ₌₁ᵏ C(n, j).

    Args:
        n: Total system qubits.
        k: Number of target qubits (maximum correlator order).

    Returns:
        Integer feature dimension.
    """
    return len(get_all_subsets(n, k))


def compute_correlators(bitstrings: np.ndarray, n: int, k: int) -> np.ndarray:
    """Compute all correlator features for one block.

    Encodes the ``(shots, n)`` bitstring matrix as ±1, then computes the
    empirical mean of each multi-qubit product across shots. The subset
    enumeration is obtained from ``get_all_subsets`` (cached).

    Args:
        bitstrings: Int array of shape ``(shots, n)``, values in {0, 1}.
            Qubit 0 is in column 0.
        n: Total number of qubits (must match ``bitstrings.shape[1]``).
        k: Maximum correlator order.

    Returns:
        Float32 array of shape ``(F,)`` with values in [−1, 1].
        Ordering matches ``get_all_subsets(n, k)``.
    """
    z = 1 - 2 * bitstrings.astype(np.float32)  # (shots, n)
    shots = z.shape[0]
    subsets = get_all_subsets(n, k)
    correlators = np.empty(len(subsets), dtype=np.float32)
    offset = 0
    for m in range(1, k + 1):
        group = [s for s in subsets if len(s) == m]
        if not group:
            continue
        nbytes = shots * len(group) * m * 4
        if nbytes > CORRELATOR_MEMORY_WARN_BYTES:
            warnings.warn(
                f"compute_correlators: order-{m} allocation is "
                f"{nbytes / 1024**2:.0f} MB (threshold "
                f"{CORRELATOR_MEMORY_WARN_BYTES // 1024**2} MB). "
                f"n={n}, k={k}, shots={shots}.",
                ResourceWarning,
                stacklevel=2,
            )
        idx = np.array(group)  # (C(n,m), m)
        correlators[offset : offset + len(group)] = z[:, idx].prod(axis=2).mean(axis=0)
        offset += len(group)
    return correlators


def compute_block_features(
    bitstrings_blocked: np.ndarray,
    n: int,
    k: int,
) -> np.ndarray:
    """Compute correlator features for all blocks in one instance.

    Applies ``compute_correlators`` to each block independently. This is the
    function called by ``simulate.generate_instance`` and by the inference
    pipeline when processing raw adversary-observed bitstrings.

    Args:
        bitstrings_blocked: Int array of shape ``(n_blocks, shots_per_block, n)``,
            values in {0, 1}.
        n: Total number of qubits.
        k: Maximum correlator order.

    Returns:
        Float32 array of shape ``(n_blocks, F)``.
    """
    n_blocks = bitstrings_blocked.shape[0]
    F = feature_dim(n, k)
    features = np.empty((n_blocks, F), dtype=np.float32)
    for b in range(n_blocks):
        features[b] = compute_correlators(bitstrings_blocked[b], n, k)
    return features

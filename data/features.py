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

from functools import lru_cache
from itertools import combinations

import numpy as np


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
    ...


def feature_dim(n: int, k: int) -> int:
    """Return the feature dimension F = Σⱼ₌₁ᵏ C(n, j).

    Args:
        n: Total system qubits.
        k: Number of target qubits (maximum correlator order).

    Returns:
        Integer feature dimension.
    """
    ...


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
    ...


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
    ...

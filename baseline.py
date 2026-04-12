"""
Classical variance-based extractor baseline.

This is the reference adversary that the ML model is benchmarked against. It
requires no training: given the 10k observed bitstrings blocked into 1000
groups of 10, it enumerates all C(n, k) candidate qubit subsets, computes the
empirical marginal for each subset independently per block, and selects the
subset whose per-block marginals are most consistent (minimum cross-block
variance). The pooled marginal over the selected subset is the baseline
estimate p̂_baseline.

Rationale
---------
This is the best a purely statistical adversary can do without position
knowledge. It will fail most badly for near-uniform target distributions
(weak consistency signal) and large n (many candidate subsets to confound it).
The gap between baseline TVD and model TVD is the headline result of the paper.

Complexity
----------
O(C(n,k) · n_blocks · shots_per_block · 2^k) — milliseconds for k=3, n=10.
"""

from __future__ import annotations

import numpy as np
from itertools import combinations


def marginal_for_subset(
    bitstrings_blocked: np.ndarray,
    qubit_subset: tuple[int, ...],
) -> np.ndarray:
    """Compute the pooled empirical marginal over a specific qubit subset.

    Extracts the columns corresponding to ``qubit_subset`` from every shot in
    every block and returns the normalized frequency vector.

    Args:
        bitstrings_blocked: Shape ``(n_blocks, shots_per_block, n)``, values {0,1}.
        qubit_subset: Tuple of k qubit indices to extract (in any order).

    Returns:
        Float64 array of shape ``(2**k,)`` summing to 1. Index ``i`` is the
        frequency of the k-bit string whose integer value is ``i``.
    """
    ...


def cross_block_variance(
    bitstrings_blocked: np.ndarray,
    qubit_subset: tuple[int, ...],
) -> float:
    """Compute the mean per-bin variance of block-level marginals for a subset.

    For each block, compute the empirical k-qubit marginal over ``qubit_subset``.
    Then compute the variance across blocks for each bin, and return the mean
    across bins. Low variance indicates the marginal is stable across blocks —
    the signature of the target register.

    Args:
        bitstrings_blocked: Shape ``(n_blocks, shots_per_block, n)``.
        qubit_subset: Tuple of k qubit indices.

    Returns:
        Non-negative scalar. Lower = more consistent across blocks.
    """
    ...


def variance_baseline(
    bitstrings_blocked: np.ndarray,
    k: int,
    n: int,
) -> dict[str, object]:
    """Run the classical variance-based extractor.

    Enumerates all C(n, k) qubit subsets, scores each by ``cross_block_variance``,
    selects the minimum-variance subset, and returns the pooled empirical
    marginal over that subset as the estimate p̂_baseline.

    Args:
        bitstrings_blocked: Shape ``(n_blocks, shots_per_block, n)``, values {0,1}.
        k: Number of target qubits.
        n: Total system qubits.

    Returns:
        Dict with keys:
        - ``"p_hat_baseline"``: float64 array of shape ``(2**k,)`` — the
          baseline distribution estimate.
        - ``"selected_subset"``: tuple of k int — the chosen qubit indices.
        - ``"variance_score"``: float — the cross-block variance of the
          selected subset.
        - ``"all_variances"``: dict mapping subset tuple → variance score,
          for diagnostic inspection.
    """
    ...


def run_baseline_on_dataset(
    dataset_path: str,
    k: int,
    n: int,
) -> dict[str, np.ndarray]:
    """Run the variance baseline on every instance in a dataset file.

    Loads the ``.npz`` file, runs ``variance_baseline`` on the ``bitstrings``
    array for each instance, and computes TVD against both p* and p̂_marginal
    (the oracle empirical marginal) for each instance.

    Args:
        dataset_path: Path to a ``.npz`` file produced by ``simulate.save_dataset``.
        k: Number of target qubits.
        n: Total system qubits.

    Returns:
        Dict with keys:
        - ``"p_hat_baselines"``: shape ``(N, 2**k)`` — one estimate per instance
        - ``"tvd_adv"``: shape ``(N,)`` — TV(p̂_baseline, p̂_marginal) per instance
        - ``"tvd_true"``: shape ``(N,)`` — TV(p̂_baseline, p*) per instance
        - ``"selected_subsets"``: list of N tuples
    """
    ...

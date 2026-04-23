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
    k = len(qubit_subset)
    # Flatten to (total_shots, n), extract subset columns
    bits = bitstrings_blocked.reshape(-1, bitstrings_blocked.shape[-1])
    extracted = bits[:, list(qubit_subset)]  # (total_shots, k)
    # LSB-first: qubit_subset[0] is the least significant bit
    powers = 1 << np.arange(k)
    indices = extracted @ powers
    counts = np.bincount(indices, minlength=2**k).astype(np.float64)
    return counts / bits.shape[0]


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
    k = len(qubit_subset)
    n_blocks, shots_per_block, _ = bitstrings_blocked.shape
    cols = list(qubit_subset)
    powers = 1 << np.arange(k)

    # block_marginals[b] = empirical marginal for block b
    block_marginals = np.empty((n_blocks, 2**k), dtype=np.float64)
    for b in range(n_blocks):
        extracted = bitstrings_blocked[b][:, cols]  # (shots_per_block, k)
        indices = extracted @ powers
        counts = np.bincount(indices, minlength=2**k).astype(np.float64)
        block_marginals[b] = counts / shots_per_block

    # Mean over bins of per-bin variance across blocks
    return float(np.mean(np.var(block_marginals, axis=0)))


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
    all_variances: dict[tuple[int, ...], float] = {}
    for subset in combinations(range(n), k):
        all_variances[subset] = cross_block_variance(bitstrings_blocked, subset)

    selected_subset = min(all_variances, key=all_variances.__getitem__)
    variance_score = all_variances[selected_subset]
    p_hat_baseline = marginal_for_subset(bitstrings_blocked, selected_subset)

    return {
        "p_hat_baseline": p_hat_baseline,
        "selected_subset": selected_subset,
        "variance_score": variance_score,
        "all_variances": all_variances,
    }


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
    from utils.metrics import tvd, empirical_marginal

    data = np.load(dataset_path)
    bitstrings_all = data["bitstrings"]       # (N, n_blocks, shots_per_block, n)
    p_stars = data["p_stars"]                 # (N, 2**k)
    target_positions_all = data["target_positions"]  # (N, n_blocks, k)

    N = bitstrings_all.shape[0]
    p_hat_baselines = np.empty((N, 2**k), dtype=np.float64)
    tvd_adv = np.empty(N, dtype=np.float64)
    tvd_true = np.empty(N, dtype=np.float64)
    selected_subsets = []

    for i in range(N):
        result = variance_baseline(bitstrings_all[i], k, n)
        p_hat = result["p_hat_baseline"]
        p_hat_baselines[i] = p_hat
        selected_subsets.append(result["selected_subset"])

        p_hat_marginal = empirical_marginal(bitstrings_all[i], target_positions_all[i], k)
        tvd_adv[i] = tvd(p_hat, p_hat_marginal)
        tvd_true[i] = tvd(p_hat, p_stars[i])

    return {
        "p_hat_baselines": p_hat_baselines,
        "p_stars": p_stars,
        "tvd_adv": tvd_adv,
        "tvd_true": tvd_true,
        "selected_subsets": selected_subsets,
    }

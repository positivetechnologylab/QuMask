"""
Evaluation metrics for distribution recovery.

All functions are pure numpy — no torch dependency. Designed to be called on
numpy arrays produced either from model inference or from the classical baseline.

Metric conventions
------------------
- ``p``, ``q``: 1-D float arrays of shape ``(2**k,)`` summing to 1.
- ``p_star``: the true target distribution (known from simulation).
- ``p_hat``: the model's point-estimate distribution.
- ``p_hat_marginal``: the empirical marginal extracted from the 10k bitstrings
  over the oracle target qubit positions. This is the primary adversarial
  reference — what a perfect classical extractor with position knowledge sees.
"""

from __future__ import annotations

import numpy as np


def tvd(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions.

    TVD(p, q) = 0.5 * Σ |p_i - q_i|

    Args:
        p: Probability vector, shape ``(d,)``, sums to 1.
        q: Probability vector, shape ``(d,)``, sums to 1.

    Returns:
        Scalar in [0, 1]. 0 means identical distributions.
    """
    ...


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Forward KL divergence KL(p ∥ q).

    KL(p ∥ q) = Σ p_i log(p_i / q_i)

    Entries where p_i = 0 contribute 0 (by convention 0 log 0 = 0). Entries
    where q_i = 0 but p_i > 0 are clipped via ``eps`` to avoid divergence to
    infinity during evaluation (though this should not occur for well-trained
    models with softmax outputs).

    Args:
        p: True distribution, shape ``(d,)``.
        q: Predicted distribution, shape ``(d,)``.
        eps: Small constant added to ``q`` before taking the log.

    Returns:
        Non-negative scalar. Unbounded above.
    """
    ...


def ece(
    p_hats: np.ndarray,
    p_stars: np.ndarray,
    sigmas: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error for per-bin uncertainty estimates.

    Measures whether the ensemble's per-bin standard deviations are honest
    predictors of actual per-bin errors. Bins predictions by confidence level
    and computes the weighted mean absolute gap between expected and observed
    error within each bin.

    Args:
        p_hats: Model point estimates, shape ``(N, 2**k)``.
        p_stars: True distributions, shape ``(N, 2**k)``.
        sigmas: Per-bin standard deviations from the ensemble, shape ``(N, 2**k)``.
        n_bins: Number of equal-width confidence bins.

    Returns:
        Non-negative scalar. Lower is better; 0 means perfect calibration.
    """
    ...


def empirical_marginal(
    bitstrings: np.ndarray,
    target_positions: np.ndarray,
    k: int,
) -> np.ndarray:
    """Compute the empirical k-qubit marginal over the target register.

    Pools all bitstrings, extracts the k target-qubit columns, and returns the
    normalized count vector. This is ``p̂_marginal`` — the oracle reference used
    as the primary adversarial evaluation target.

    Args:
        bitstrings: Shape ``(n_blocks, shots_per_block, n)`` or ``(total_shots, n)``,
            values in {0, 1}.
        target_positions: Shape ``(k,)`` or ``(n_blocks, k)``. If 2-D, each block
            uses its own set of target positions (the QuMask setting). If 1-D,
            all shots use the same positions.
        k: Number of target qubits.

    Returns:
        Float64 array of shape ``(2**k,)`` summing to 1. Index ``i`` corresponds
        to the k-bit string whose integer value is ``i``.
    """
    ...


def compute_all_metrics(
    p_hat: np.ndarray,
    p_hat_marginal: np.ndarray,
    p_star: np.ndarray,
    conformal_radius: float | None = None,
) -> dict[str, float]:
    """Compute the full metric suite for one inference instance.

    Args:
        p_hat: Model point estimate, shape ``(2**k,)``.
        p_hat_marginal: Empirical marginal from oracle-position extraction,
            shape ``(2**k,)``. Primary adversarial reference.
        p_star: True target distribution from simulation, shape ``(2**k,)``.
        conformal_radius: The conformal TV-ball radius around ``p_hat``, if
            conformal prediction has been calibrated. Pass ``None`` to omit.

    Returns:
        Dict with keys:
        - ``"tvd_adv"``: TV(p̂, p̂_marginal) — primary adversarial metric
        - ``"tvd_true"``: TV(p̂, p*) — absolute accuracy
        - ``"kl_true"``: KL(p* ∥ p̂)
        - ``"entropy_true"``: Shannon entropy of p* (instance difficulty proxy)
        - ``"conformal_radius"``: conformal TV-ball radius (if provided)
    """
    ...

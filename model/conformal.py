"""
Conformal predictor for distribution recovery uncertainty quantification.

Method: split conformal prediction (Venn-Abers / inductive CP).
Coverage guarantee: the TV-ball B(p̂, r) contains p* with probability ≥ 1−α
over the randomness of the calibration set, for any (k, n) configuration
and any finite N_cal.

Nonconformity score
-------------------
    s(x, y) = TV(p̂_ensemble(x), p*)

where p̂_ensemble(x) is the ensemble mean prediction and p* is the true label.
This is the TV distance between the model's prediction and the ground truth on
calibration instances. The conformal radius r is the (1−α)(1 + 1/N_cal)
empirical quantile of these scores.

At inference, the prediction set is the TV-ball:
    { q ∈ Δ(2^k) : TV(p̂, q) ≤ r }

This is a convex set; the radius r is the single number reported alongside p̂.

Usage
-----
    predictor = ConformalPredictor(alpha=0.05)
    predictor.calibrate(ensemble, cal_loader, device)
    radius = predictor.radius  # use this as the TV-ball half-width

References
----------
- Angelopoulos & Bates, "A Gentle Introduction to Conformal Prediction" (2022)
- Shafer & Vovk, "A Tutorial on Conformal Prediction" (2008)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.ensemble import QuMaskEnsemble


class ConformalPredictor:
    """Split-conformal predictor producing TV-ball coverage sets.

    Args:
        alpha: Miscoverage level. The predictor guarantees coverage ≥ 1−α
            over calibration set randomness.

    Attributes:
        radius: The calibrated TV-ball radius. Set by ``calibrate``; None
            before calibration.
        n_cal: Number of calibration instances used. Set by ``calibrate``.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        ...

    def calibrate(
        self,
        ensemble: QuMaskEnsemble,
        cal_loader: DataLoader,
        device: torch.device | None = None,
    ) -> None:
        """Fit the conformal predictor on a held-out calibration set.

        Runs the ensemble on all calibration instances, computes the
        nonconformity score TV(p̂, p*) for each, and stores the empirical
        quantile at level (1−α)(1 + 1/N_cal) as ``self.radius``.

        The calibration set must be disjoint from the training and validation
        sets. It is drawn from the test split before any model evaluation.

        Args:
            ensemble: Trained ``QuMaskEnsemble`` in eval mode.
            cal_loader: DataLoader over the calibration split. Yields
                ``(features, p_star)`` pairs.
            device: Inference device. Defaults to CPU.

        Side effects:
            Sets ``self.radius`` and ``self.n_cal``.
        """
        ...

    def prediction_interval(self, p_hat: np.ndarray) -> dict[str, object]:
        """Return the conformal prediction set for a single instance.

        The prediction set is the TV-ball B(p̂, r) = { q : TV(p̂, q) ≤ r }.
        This is reported as the center p̂ and the radius r, which together
        define the convex set.

        Args:
            p_hat: Model point estimate, shape ``(2**k,)``.

        Returns:
            Dict with keys:
            - ``"center"``: p_hat (the ensemble mean prediction)
            - ``"radius"``: self.radius (TV-ball half-width)
            - ``"alpha"``: self.alpha (nominal miscoverage level)

        Raises:
            RuntimeError: If called before ``calibrate``.
        """
        ...

    def empirical_coverage(
        self,
        p_hats: np.ndarray,
        p_stars: np.ndarray,
    ) -> float:
        """Compute empirical coverage on a test set.

        Checks what fraction of instances have p* inside the TV-ball B(p̂, r).
        Should be ≥ 1−α if the calibration and test distributions match.

        Args:
            p_hats: Model point estimates, shape ``(N, 2**k)``.
            p_stars: True distributions, shape ``(N, 2**k)``.

        Returns:
            Scalar in [0, 1]. The empirical fraction of instances covered.
        """
        ...

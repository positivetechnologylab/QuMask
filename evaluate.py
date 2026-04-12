"""
Evaluation entry point for the QuMask adversarial ML system.

Usage
-----
    python evaluate.py                          # uses config/default.yaml
    python evaluate.py --config path/to.yaml
    python evaluate.py --baseline-only          # run only the classical baseline
    python evaluate.py --skip-baseline          # skip baseline, run model only

Workflow
--------
1. Load config and restore trained ensemble from checkpoints.
2. Load test and calibration splits.
3. Calibrate the conformal predictor on the calibration split.
4. Run ensemble inference on the test split.
5. For each test instance, compute the full metric suite via
   ``metrics.compute_all_metrics``:
     - TV(p̂, p̂_marginal)  ← primary adversarial metric
     - TV(p̂, p*)           ← absolute accuracy
     - KL(p* ∥ p̂)
     - conformal radius
6. Run ``baseline.run_baseline_on_dataset`` on the same test split.
7. Print per-metric means ± std, stratified by entropy quintile of p*.
8. Print ECE and conformal empirical coverage.
9. Save all per-instance results to ``{results_dir}/results.npz``.

This script contains no model logic — it wires together the modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import torch

from data.dataset import QuMaskDataset
from model.ensemble import load_ensemble
from model.conformal import ConformalPredictor
from utils.metrics import (
    compute_all_metrics,
    empirical_marginal,
    ece,
)
from baseline import run_baseline_on_dataset


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with attributes:
        - ``config``: str — path to YAML config
        - ``baseline_only``: bool
        - ``skip_baseline``: bool
    """
    ...


def load_config(path: str) -> dict:
    """Load and return the YAML config as a nested dict.

    Args:
        path: Path to a YAML config file.

    Returns:
        Nested config dict. Same structure as ``train.load_config``.
    """
    ...


def run_model_evaluation(
    cfg: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Run the full ensemble evaluation on the test split.

    Loads the ensemble, calibrates the conformal predictor on the calibration
    split, then runs inference on every test instance. For each instance,
    computes the oracle empirical marginal from raw bitstrings and target
    positions, and then calls ``compute_all_metrics``.

    Args:
        cfg: Full config dict.
        device: Inference device.

    Returns:
        Dict with per-instance arrays:
        - ``"p_hats"``: shape ``(N_test, 2**k)``
        - ``"sigmas"``: shape ``(N_test, 2**k)``
        - ``"tvd_adv"``: shape ``(N_test,)``
        - ``"tvd_true"``: shape ``(N_test,)``
        - ``"kl_true"``: shape ``(N_test,)``
        - ``"entropy_true"``: shape ``(N_test,)``
        - ``"conformal_radius"``: scalar (same for all instances after calibration)
        - ``"conformal_coverage"``: scalar empirical coverage on test set
    """
    ...


def print_results_table(
    model_results: dict[str, np.ndarray],
    baseline_results: dict[str, np.ndarray] | None,
    n_entropy_bins: int = 5,
) -> None:
    """Print evaluation results to stdout in a human-readable table.

    Reports mean ± std for each metric, stratified by entropy quintile of p*.
    Prints model results and (if provided) baseline results side by side for
    direct comparison.

    Args:
        model_results: Dict returned by ``run_model_evaluation``.
        baseline_results: Dict returned by ``run_baseline_on_dataset``, or None
            if baseline was skipped.
        n_entropy_bins: Number of entropy bins for stratification.
    """
    ...


def main() -> None:
    """Main evaluation entry point.

    Loads config, restores model, runs evaluation and/or baseline, prints
    results table, and saves per-instance results to disk.
    """
    ...


if __name__ == "__main__":
    main()

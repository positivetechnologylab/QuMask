"""
Evaluation entry point for the QuMask adversarial ML system.

Usage
-----
    python3 evaluate.py                          # uses config/default.yaml
    python3 evaluate.py --config path/to.yaml
    python3 evaluate.py --baseline-only          # run only the classical baseline
    python3 evaluate.py --skip-baseline          # skip baseline, run model only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import QuMaskDataset
from model.ensemble import load_ensemble
from model.conformal import ConformalPredictor
from utils.metrics import (
    compute_all_metrics,
    empirical_marginal,
    sigma_calibration_error,
)
from baseline import run_baseline_on_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--baseline-only", action="store_true")
    p.add_argument("--skip-baseline", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_model_evaluation(
    cfg: dict,
    device: torch.device,
) -> dict[str, np.ndarray]:
    data_dir = Path(cfg["paths"]["data_dir"])
    k = cfg["data"]["k"]
    batch_size = cfg["training"]["batch_size"]

    ensemble = load_ensemble(cfg, cfg["paths"]["checkpoint_dir"], device=device)

    # Calibrate conformal predictor on the cal split.
    cal_loader = DataLoader(
        QuMaskDataset(str(data_dir / "cal.npz")),
        batch_size=batch_size,
        shuffle=False,
    )
    predictor = ConformalPredictor(alpha=cfg["conformal"]["alpha"])
    predictor.calibrate(ensemble, cal_loader, device=device)

    # Batched inference over the test set.
    test_ds = QuMaskDataset(str(data_dir / "test.npz"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    p_hats_list: list[np.ndarray] = []
    sigmas_list: list[np.ndarray] = []
    for features, _ in test_loader:
        out = ensemble.predict(features, device=device)
        p_hats_list.append(out["p_hat"].cpu().numpy())
        sigmas_list.append(out["sigma"].cpu().numpy())

    p_hats  = np.concatenate(p_hats_list, axis=0).astype(np.float64)   # (N, 2**k)
    sigmas  = np.concatenate(sigmas_list, axis=0).astype(np.float64)   # (N, 2**k)
    p_stars = test_ds.p_stars.astype(np.float64)                        # (N, 2**k)

    # Vectorized empirical marginals over all test instances.
    # bitstrings: (N, n_blocks, shots_per_block, n)
    # target_positions: (N, n_blocks, k)
    N = len(test_ds)
    n_blocks, shots_per_block = test_ds.bitstrings.shape[1], test_ds.bitstrings.shape[2]
    powers = 1 << np.arange(k)

    # Extract per-block target bits: (N, n_blocks, shots_per_block, k)
    bits = test_ds.bitstrings  # (N, n_blocks, shots, n), uint8
    tp   = test_ds.target_positions  # (N, n_blocks, k)

    # Two-step indexing to avoid advanced-indexing transpose: index instance+block first.
    p_hat_marginals = np.empty((N, 2**k), dtype=np.float64)
    for i in range(N):
        counts = np.zeros(2**k, dtype=np.float64)
        for b in range(n_blocks):
            extracted = bits[i, b][:, tp[i, b]]   # (shots_per_block, k)
            indices = extracted @ powers
            np.add.at(counts, indices, 1)
        p_hat_marginals[i] = counts / (n_blocks * shots_per_block)

    # Vectorized metrics.
    tvd_adv      = 0.5 * np.abs(p_hats - p_hat_marginals).sum(axis=-1)
    tvd_true     = 0.5 * np.abs(p_hats - p_stars).sum(axis=-1)
    safe_q       = np.where(p_hats > 0, p_hats, 1.0)
    kl_true      = np.sum(np.where(p_stars > 0, p_stars * np.log(p_stars / safe_q), 0.0), axis=-1)
    entropy_true = -np.sum(np.where(p_stars > 0, p_stars * np.log(p_stars), 0.0), axis=-1)

    conformal_coverage = predictor.empirical_coverage(p_hats, p_stars)

    return {
        "p_hats":              p_hats,
        "p_stars":             p_stars,
        "sigmas":              sigmas,
        "tvd_adv":             tvd_adv,
        "tvd_true":            tvd_true,
        "kl_true":             kl_true,
        "entropy_true":        entropy_true,
        "conformal_radius":    np.float64(predictor.radius),
        "conformal_coverage":  np.float64(conformal_coverage),
    }


def print_results_table(
    model_results: dict[str, np.ndarray] | None,
    baseline_results: dict[str, np.ndarray] | None,
    n_entropy_bins: int = 5,
) -> None:
    # Determine entropy quintile boundaries from whichever results are available.
    if model_results is not None:
        entropy = model_results["entropy_true"]
    elif baseline_results is not None and "p_stars" in baseline_results:
        p_stars = baseline_results["p_stars"]
        entropy = -np.sum(p_stars * np.log(p_stars + 1e-12), axis=-1)
    else:
        entropy = None

    def _row(label: str, values: np.ndarray) -> str:
        return f"  {label:<22s}  {values.mean():.4f} ± {values.std():.4f}"

    def _section(tag: str, mask: np.ndarray, model: dict | None, baseline: dict | None) -> None:
        print(f"\n{tag}")
        if model is not None:
            print("  [model]")
            print(_row("tvd_adv",  model["tvd_adv"][mask]))
            print(_row("tvd_true", model["tvd_true"][mask]))
            print(_row("kl_true",  model["kl_true"][mask]))
        if baseline is not None:
            print("  [baseline]")
            print(_row("tvd_adv",  baseline["tvd_adv"][mask]))
            print(_row("tvd_true", baseline["tvd_true"][mask]))

    N = len(model_results["tvd_adv"]) if model_results is not None else len(baseline_results["tvd_adv"])
    all_mask = np.ones(N, dtype=bool)

    print("\n=== Overall ===")
    _section("", all_mask, model_results, baseline_results)

    if entropy is not None:
        boundaries = np.percentile(entropy, np.linspace(0, 100, n_entropy_bins + 1))
        for i in range(n_entropy_bins):
            lo, hi = boundaries[i], boundaries[i + 1]
            mask = (entropy >= lo) & (entropy <= hi if i == n_entropy_bins - 1 else entropy < hi)
            tag = f"=== Entropy quintile {i+1}/{ n_entropy_bins} [{lo:.3f}, {hi:.3f}] ==="
            _section(tag, mask, model_results, baseline_results)

    if model_results is not None:
        sce_val = sigma_calibration_error(model_results["p_hats"], model_results["p_stars"], model_results["sigmas"])
        print(f"\n  sigma_cal_error    {sce_val:.4f}")
        print(f"  conformal_radius   {float(model_results['conformal_radius']):.4f}")
        print(f"  conformal_coverage {float(model_results['conformal_coverage']):.4f}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    data_dir = Path(cfg["paths"]["data_dir"])
    results_dir = Path(cfg["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    model_results = None
    baseline_results = None

    if not args.baseline_only:
        print("Running model evaluation...")
        model_results = run_model_evaluation(cfg, device)

    if not args.skip_baseline:
        print("Running baseline...")
        baseline_results = run_baseline_on_dataset(
            str(data_dir / "test.npz"),
            cfg["data"]["k"],
            cfg["data"]["n"],
        )

    print_results_table(model_results, baseline_results)

    save_dict = {}
    if model_results is not None:
        save_dict.update(model_results)
    if baseline_results is not None:
        save_dict.update({f"baseline_{k}": v for k, v in baseline_results.items()
                          if isinstance(v, np.ndarray)})
    if save_dict:
        np.savez_compressed(str(results_dir / "results.npz"), **save_dict)
        print(f"\nResults saved to {results_dir}/results.npz")


if __name__ == "__main__":
    main()

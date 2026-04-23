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

    # Calibrate conformal predictor on the cal split.
    cal_loader = DataLoader(
        QuMaskDataset(str(data_dir / "cal.npz")),
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
    )
    ensemble = load_ensemble(cfg, cfg["paths"]["checkpoint_dir"], device=device)
    predictor = ConformalPredictor(alpha=cfg["conformal"]["alpha"])
    predictor.calibrate(ensemble, cal_loader, device=device)

    # Run inference on test set instance-by-instance to pair model output
    # with oracle bitstrings/target_positions from get_eval_arrays.
    test_ds = QuMaskDataset(str(data_dir / "test.npz"))
    N = len(test_ds)
    output_dim = 2 ** cfg["data"]["k"]

    p_hats   = np.empty((N, output_dim), dtype=np.float64)
    p_stars  = np.empty((N, output_dim), dtype=np.float64)
    sigmas   = np.empty((N, output_dim), dtype=np.float64)
    tvd_adv  = np.empty(N, dtype=np.float64)
    tvd_true = np.empty(N, dtype=np.float64)
    kl_true  = np.empty(N, dtype=np.float64)
    entropy_true = np.empty(N, dtype=np.float64)

    for idx in range(N):
        features, _ = test_ds[idx]
        x = features.unsqueeze(0)  # (1, n_blocks, F)
        out = ensemble.predict(x, device=device)
        p_hat = out["p_hat"].squeeze(0).cpu().numpy()
        sigma = out["sigma"].squeeze(0).cpu().numpy()

        eval_arrays = test_ds.get_eval_arrays(idx)
        p_hat_marginal = empirical_marginal(
            eval_arrays["bitstrings"],
            eval_arrays["target_positions"],
            cfg["data"]["k"],
        )
        metrics = compute_all_metrics(
            p_hat=p_hat,
            p_hat_marginal=p_hat_marginal,
            p_star=eval_arrays["p_star"],
            conformal_radius=predictor.radius,
        )

        p_hats[idx]       = p_hat
        p_stars[idx]      = eval_arrays["p_star"]
        sigmas[idx]       = sigma
        tvd_adv[idx]      = metrics["tvd_adv"]
        tvd_true[idx]     = metrics["tvd_true"]
        kl_true[idx]      = metrics["kl_true"]
        entropy_true[idx] = metrics["entropy_true"]

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

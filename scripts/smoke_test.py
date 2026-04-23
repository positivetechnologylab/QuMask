"""
End-to-end smoke test: generate tiny data splits, train 2 ensemble members for
3 epochs, calibrate conformal predictor, run model + baseline evaluation.

Takes ~60-90 seconds on CPU. Catches import errors, shape mismatches, and
pipeline wiring bugs before committing to full data generation and training.

Usage
-----
    python3 scripts/smoke_test.py
    python3 scripts/smoke_test.py --keep   # don't delete temp files on exit
"""

from __future__ import annotations

import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Allow running from repo root or from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.dataset import QuMaskDataset
from data.simulate import generate_dataset, save_dataset
from model.conformal import ConformalPredictor
from model.ensemble import load_ensemble, train_all
from utils.metrics import compute_all_metrics, empirical_marginal, sigma_calibration_error
from baseline import run_baseline_on_dataset


SMOKE_CFG = {
    "data": {
        "k": 3,
        "n": 10,
        "n_blocks": 1000,
        "shots_per_block": 10,
        "target_depth_min": 1,
        "target_depth_max": 8,
        "n_train": 60,
        "n_val": 20,
        "n_cal": 20,
        "n_test": 20,
    },
    "model": {
        "d_model": 64,
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "dropout": 0.1,
    },
    "ensemble": {"M": 2},
    "training": {
        "lr": 1e-3,
        "batch_size": 16,
        "epochs": 3,
        "weight_decay": 1e-4,
        "patience": 10,
        "seed_base": 42,
    },
    "conformal": {"alpha": 0.05},
}


def step(msg: str) -> None:
    print(f"\n[smoke] {msg}")


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  {status}  {label}{suffix}")
    if not condition:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="keep temp files after exit")
    args = parser.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="qumask_smoke_"))
    data_dir = tmp / "data"
    ckpt_dir = tmp / "checkpoints"
    data_dir.mkdir()
    ckpt_dir.mkdir()

    cfg = copy.deepcopy(SMOKE_CFG)
    cfg["paths"] = {
        "data_dir": str(data_dir),
        "checkpoint_dir": str(ckpt_dir),
        "results_dir": str(tmp / "results"),
        "seed_train": 0,
        "seed_val": 100,
        "seed_cal": 200,
        "seed_test": 300,
    }

    try:
        # ------------------------------------------------------------------
        step("1/6  Generating data splits")
        # ------------------------------------------------------------------
        d = cfg["data"]
        splits = [
            ("train.npz", d["n_train"], cfg["paths"]["seed_train"]),
            ("val.npz",   d["n_val"],   cfg["paths"]["seed_val"]),
            ("cal.npz",   d["n_cal"],   cfg["paths"]["seed_cal"]),
            ("test.npz",  d["n_test"],  cfg["paths"]["seed_test"]),
        ]
        for fname, n_inst, seed_base in splits:
            ds = generate_dataset(
                n_instances=n_inst,
                k=d["k"], n=d["n"],
                n_blocks=d["n_blocks"],
                shots_per_block=d["shots_per_block"],
                target_depth_min=d["target_depth_min"],
                target_depth_max=d["target_depth_max"],
                seed_base=seed_base,
                n_jobs=1,
            )
            save_dataset(ds, str(data_dir / fname))
            print(f"    wrote {fname}  ({n_inst} instances)")

        for fname, n_inst, _ in splits:
            path = data_dir / fname
            check(f"{fname} exists", path.exists())
            arr = np.load(str(path))
            check(f"{fname} features shape",
                  arr["features"].shape == (n_inst, d["n_blocks"], 175),
                  str(arr["features"].shape))
            check(f"{fname} p_stars sum to 1",
                  np.allclose(arr["p_stars"].sum(axis=1), 1.0, atol=1e-6))

        # ------------------------------------------------------------------
        step("2/6  Training ensemble (2 members, 3 epochs)")
        # ------------------------------------------------------------------
        device = torch.device("cpu")
        kwargs = dict(batch_size=cfg["training"]["batch_size"],
                      num_workers=0, pin_memory=False)
        train_loader = DataLoader(
            QuMaskDataset(str(data_dir / "train.npz")), shuffle=True, **kwargs)
        val_loader = DataLoader(
            QuMaskDataset(str(data_dir / "val.npz")), shuffle=False, **kwargs)

        ensemble = train_all(cfg, train_loader, val_loader, str(ckpt_dir), device=device)

        check("ensemble has 2 members", ensemble.M == 2)
        check("member_0.pt exists", (ckpt_dir / "member_0.pt").exists())
        check("member_1.pt exists", (ckpt_dir / "member_1.pt").exists())

        # ------------------------------------------------------------------
        step("3/6  Loading ensemble from checkpoints")
        # ------------------------------------------------------------------
        loaded = load_ensemble(cfg, str(ckpt_dir), device=device)
        check("loaded M == 2", loaded.M == 2)
        all_eval = all(not m.training for m in loaded.models)
        check("all members in eval mode", all_eval)

        # ------------------------------------------------------------------
        step("4/6  Conformal calibration on cal split")
        # ------------------------------------------------------------------
        cal_loader = DataLoader(
            QuMaskDataset(str(data_dir / "cal.npz")), shuffle=False, **kwargs)
        predictor = ConformalPredictor(alpha=cfg["conformal"]["alpha"])
        predictor.calibrate(loaded, cal_loader, device=device)

        check("radius is finite and positive",
              predictor.radius is not None and np.isfinite(predictor.radius) and predictor.radius >= 0,
              f"radius={predictor.radius:.4f}")
        check("n_cal == 20", predictor.n_cal == 20)

        # ------------------------------------------------------------------
        step("5/6  Model evaluation on test split")
        # ------------------------------------------------------------------
        test_ds = QuMaskDataset(str(data_dir / "test.npz"))
        N = len(test_ds)
        output_dim = 2 ** d["k"]

        p_hats   = np.empty((N, output_dim))
        p_stars  = np.empty((N, output_dim))
        sigmas   = np.empty((N, output_dim))
        tvd_advs = np.empty(N)

        for idx in range(N):
            features, _ = test_ds[idx]
            out = loaded.predict(features.unsqueeze(0), device=device)
            p_hat = out["p_hat"].squeeze(0).cpu().numpy()
            sigma = out["sigma"].squeeze(0).cpu().numpy()
            eval_arrays = test_ds.get_eval_arrays(idx)
            p_hat_marginal = empirical_marginal(
                eval_arrays["bitstrings"], eval_arrays["target_positions"], d["k"])
            metrics = compute_all_metrics(p_hat, p_hat_marginal, eval_arrays["p_star"],
                                          conformal_radius=predictor.radius)
            p_hats[idx] = p_hat
            p_stars[idx] = eval_arrays["p_star"]
            sigmas[idx] = sigma
            tvd_advs[idx] = metrics["tvd_adv"]

        check("p_hats sum to 1",
              np.allclose(p_hats.sum(axis=1), 1.0, atol=1e-5))
        check("tvd_adv in [0, 1]",
              np.all(tvd_advs >= 0) and np.all(tvd_advs <= 1.0))

        coverage = predictor.empirical_coverage(p_hats, p_stars)
        sce = sigma_calibration_error(p_hats, p_stars, sigmas)
        check("conformal coverage is finite", np.isfinite(coverage),
              f"coverage={coverage:.3f}")
        check("sigma_calibration_error is finite", np.isfinite(sce))

        # ------------------------------------------------------------------
        step("6/6  Baseline evaluation on test split")
        # ------------------------------------------------------------------
        baseline = run_baseline_on_dataset(str(data_dir / "test.npz"), d["k"], d["n"])
        check("baseline tvd_adv in [0,1]",
              np.all(baseline["tvd_adv"] >= 0) and np.all(baseline["tvd_adv"] <= 1))
        check("baseline p_stars shape",
              baseline["p_stars"].shape == (N, output_dim))

        # ------------------------------------------------------------------
        print("\n" + "=" * 50)
        print("  SMOKE TEST PASSED")
        print("=" * 50)
        print(f"  model  mean tvd_adv : {tvd_advs.mean():.4f}")
        print(f"  baseline mean tvd_adv: {baseline['tvd_adv'].mean():.4f}")
        print(f"  conformal radius    : {predictor.radius:.4f}")
        print(f"  conformal coverage  : {coverage:.3f}")
        print(f"  sigma_cal_error     : {sce:.4f}")
        print("=" * 50)
        print()
        print("  Note: model metrics are meaningless at 3 epochs —")
        print("  this only confirms the pipeline runs without errors.")

    finally:
        if args.keep:
            print(f"\n  temp files kept at: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

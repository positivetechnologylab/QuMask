"""
Training entry point for the QuMask adversarial ML system.

Usage
-----
    python3 train.py                          # uses config/default.yaml
    python3 train.py --config path/to.yaml   # custom config
    python3 train.py --generate-data          # regenerate training data before training
    python3 train.py --generate-data --force-regenerate  # overwrite existing data files
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
import torch
from torch.utils.data import DataLoader

from data.dataset import QuMaskDataset
from data.simulate import generate_dataset, save_dataset
from model.ensemble import train_all


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--generate-data", action="store_true")
    p.add_argument("--force-regenerate", action="store_true")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def maybe_generate_data(cfg: dict, force: bool = False) -> None:
    data_dir = Path(cfg["paths"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    splits = [
        ("train.npz", cfg["data"]["n_train"], cfg["paths"]["seed_train"]),
        ("val.npz",   cfg["data"]["n_val"],   cfg["paths"]["seed_val"]),
        ("cal.npz",   cfg["data"]["n_cal"],   cfg["paths"]["seed_cal"]),
        ("test.npz",  cfg["data"]["n_test"],  cfg["paths"]["seed_test"]),
    ]

    d = cfg["data"]
    for filename, n_instances, seed_base in splits:
        path = data_dir / filename
        if path.exists() and not force:
            print(f"  {path} exists, skipping")
            continue
        print(f"  generating {path} ({n_instances} instances, seed_base={seed_base})")
        dataset = generate_dataset(
            n_instances=n_instances,
            k=d["k"],
            n=d["n"],
            n_blocks=d["n_blocks"],
            shots_per_block=d["shots_per_block"],
            target_depth_min=d["target_depth_min"],
            target_depth_max=d["target_depth_max"],
            seed_base=seed_base,
            n_jobs=-1,
        )
        save_dataset(dataset, str(path))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.generate_data:
        print("Generating data splits...")
        maybe_generate_data(cfg, force=args.force_regenerate)

    data_dir = Path(cfg["paths"]["data_dir"])
    tr = cfg["training"]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Training on {device}")

    kwargs = dict(batch_size=tr["batch_size"], num_workers=0, pin_memory=(device.type == "cuda"))

    train_loader = DataLoader(QuMaskDataset(str(data_dir / "train.npz")), shuffle=True, **kwargs)
    val_loader   = DataLoader(QuMaskDataset(str(data_dir / "val.npz")),   shuffle=False, **kwargs)

    ensemble = train_all(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=cfg["paths"]["checkpoint_dir"],
        device=device,
    )

    print(f"\nTrained {ensemble.M} members. Checkpoints saved to {cfg['paths']['checkpoint_dir']}/")


if __name__ == "__main__":
    main()

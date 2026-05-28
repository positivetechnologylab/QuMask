"""
Modal training script for the QuMask adversarial ML system.

Usage
-----
# Step 1: upload data to Modal volume (run once per (n,k) config)
    python3 modal_train.py upload --config config/nk_configs/n15_k3.yaml

# Step 2: run training on Modal GPU
    python3 modal_train.py train --config config/nk_configs/n15_k3.yaml

# Step 3: download checkpoints back to local machine
    python3 modal_train.py download --config config/nk_configs/n15_k3.yaml

The --config flag selects which (n,k) configuration to train. It defaults to
config/default.yaml for backwards compatibility. The config determines the
data_dir (where .npz files are read from the volume) and the checkpoint_dir
(where member_*.pt files are written on the volume).

Training is resumable: if member_m.pt already exists in the checkpoint dir on
the volume, that member is skipped and loaded from disk. Volume commits happen
after each member so progress survives job interruption.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Image — install all dependencies on top of a CUDA-enabled base
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.4.0",
        "numpy==2.2.3",
        "qiskit==2.3.0",
        "pyyaml",
        "joblib",
    )
    .env({"PYTHONPATH": "/workspace"})
    .add_local_dir(
        Path(__file__).parent,
        remote_path="/workspace",
        ignore=[
            ".git", "__pycache__", "data/processed", "checkpoints",
            "results", ".venv", "node_modules", "*.npz", "*.pt",
        ],
    )
)

# ---------------------------------------------------------------------------
# Persistent volume for data and checkpoints
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("qumask-data", create_if_missing=True)
REMOTE_ROOT = Path("/data")

app = modal.App("qumask-train", image=image)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    volumes={str(REMOTE_ROOT): volume},
    timeout=60 * 60 * 3,  # 3 hours ceiling
)
def train_on_modal(config_path: str = "/workspace/config/default.yaml"):
    """Train one (n,k) configuration on a Modal GPU.

    Loads the config from ``config_path``, overrides data_dir and
    checkpoint_dir to point at the Modal volume, and runs train_all with
    incremental volume commits after each ensemble member.

    Args:
        config_path: Path to the YAML config file inside the Modal image
            (i.e. under /workspace/). Defaults to the default config.
    """
    import yaml
    from pathlib import Path as P

    cfg_path = P(config_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Override paths to point at the volume mount.
    # data_dir comes from cfg but must be rooted at REMOTE_ROOT.
    # checkpoint_dir likewise — use what the config specifies but remap to /data.
    local_data_dir = P(cfg["paths"]["data_dir"])       # e.g. data/processed/n6_k3
    local_ckpt_dir = P(cfg["paths"]["checkpoint_dir"]) # e.g. checkpoints/n6_k3

    remote_data_dir = REMOTE_ROOT / local_data_dir
    remote_ckpt_dir = REMOTE_ROOT / local_ckpt_dir

    cfg["paths"]["data_dir"] = str(remote_data_dir)
    cfg["paths"]["checkpoint_dir"] = str(remote_ckpt_dir)

    import torch
    from torch.utils.data import DataLoader
    from data.dataset import QuMaskDataset
    from model.ensemble import train_all

    remote_ckpt_dir.mkdir(parents=True, exist_ok=True)

    tr = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}  config={config_path}")

    kwargs = dict(batch_size=tr["batch_size"], num_workers=0, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(
        QuMaskDataset(str(remote_data_dir / "train.npz")), shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        QuMaskDataset(str(remote_data_dir / "val.npz")), shuffle=False, **kwargs
    )

    ensemble = train_all(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(remote_ckpt_dir),
        device=device,
        volume=volume,  # enables per-member incremental commits
    )
    print(f"Trained {ensemble.M} members. Final commit.")
    volume.commit()
    print("Done.")


# ---------------------------------------------------------------------------
# Local entrypoints for upload / train / download
# ---------------------------------------------------------------------------

LOCAL_ROOT = Path(__file__).parent


def _load_local_cfg(config_path: str) -> dict:
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def cmd_upload(config_path: str) -> None:
    cfg = _load_local_cfg(config_path)
    local_data_dir = LOCAL_ROOT / cfg["paths"]["data_dir"]
    # Remote path mirrors the local data_dir structure under /data on the volume.
    remote_prefix = Path(cfg["paths"]["data_dir"])

    print(f"Uploading data from {local_data_dir} to volume:{remote_prefix} ...")
    with volume.batch_upload(force=True) as batch:
        for f in sorted(local_data_dir.glob("*.npz")):
            remote = remote_prefix / f.name
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
            batch.put_file(str(f), str(remote))
    print("Upload complete.")


def cmd_train(config_path: str) -> None:
    # Map local config path to its /workspace equivalent inside the Modal image.
    rel = Path(config_path)
    remote_cfg = f"/workspace/{rel}"
    print(f"Dispatching training job to Modal  config={remote_cfg} ...")
    with modal.enable_output():
        with app.run(detach=True):
            train_on_modal.remote(config_path=remote_cfg)


def cmd_download(config_path: str) -> None:
    cfg = _load_local_cfg(config_path)
    local_ckpt_dir = LOCAL_ROOT / cfg["paths"]["checkpoint_dir"]
    remote_ckpt_dir = cfg["paths"]["checkpoint_dir"]  # e.g. checkpoints/n6_k3

    local_ckpt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading checkpoints from volume:{remote_ckpt_dir} -> {local_ckpt_dir} ...")
    for entry in volume.listdir(remote_ckpt_dir):
        name = Path(entry.path).name
        dest = local_ckpt_dir / name
        print(f"  {name}")
        with open(dest, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
    print(f"Checkpoints saved to {local_ckpt_dir}/")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Modal training entrypoint for QuMask.")
    p.add_argument(
        "command", choices=["upload", "train", "download"],
        help="Action to perform.",
    )
    p.add_argument(
        "--config", default="config/default.yaml",
        help="Path to the YAML config file (default: config/default.yaml).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cmds = {
        "upload": cmd_upload,
        "train": cmd_train,
        "download": cmd_download,
    }
    cmds[args.command](args.config)

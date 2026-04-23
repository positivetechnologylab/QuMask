"""
Modal training script for the QuMask adversarial ML system.

Usage
-----
# Step 1: upload data to Modal volume (run once)
    /usr/local/bin/python3.12 modal_train.py upload

# Step 2: run training on Modal GPU
    /usr/local/bin/python3.12 modal_train.py train

# Step 3: download checkpoints back to local machine
    /usr/local/bin/python3.12 modal_train.py download
"""

from __future__ import annotations

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
REMOTE_DATA = REMOTE_ROOT / "processed"
REMOTE_CKPT = REMOTE_ROOT / "checkpoints"

app = modal.App("qumask-train", image=image)

# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    volumes={str(REMOTE_ROOT): volume},
    timeout=60 * 60 * 3,  # 3 hours ceiling
)
def train_on_modal():
    import yaml
    from pathlib import Path as P

    cfg_path = P("/workspace/config/default.yaml")
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    cfg["paths"]["data_dir"] = str(REMOTE_DATA)
    cfg["paths"]["checkpoint_dir"] = str(REMOTE_CKPT)

    import torch
    from torch.utils.data import DataLoader
    from data.dataset import QuMaskDataset
    from model.ensemble import train_all

    REMOTE_CKPT.mkdir(parents=True, exist_ok=True)

    tr = cfg["training"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    kwargs = dict(batch_size=tr["batch_size"], num_workers=0, pin_memory=(device.type == "cuda"))
    train_loader = DataLoader(QuMaskDataset(str(REMOTE_DATA / "train.npz")), shuffle=True, **kwargs)
    val_loader   = DataLoader(QuMaskDataset(str(REMOTE_DATA / "val.npz")),   shuffle=False, **kwargs)

    ensemble = train_all(
        cfg=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(REMOTE_CKPT),
        device=device,
    )
    print(f"Trained {ensemble.M} members. Committing checkpoints to volume.")
    volume.commit()
    print("Done.")


# ---------------------------------------------------------------------------
# Local entrypoints for upload / train / download
# ---------------------------------------------------------------------------

LOCAL_DATA = Path(__file__).parent / "data" / "processed"
LOCAL_CKPT = Path(__file__).parent / "checkpoints"


def cmd_upload():
    import modal as _modal
    print("Uploading data to Modal volume...")
    with volume.batch_upload(force=True) as batch:
        for f in sorted(LOCAL_DATA.glob("*.npz")):
            remote = Path("processed") / f.name
            print(f"  {f.name}  ({f.stat().st_size / 1e6:.0f} MB)")
            batch.put_file(str(f), str(remote))
    print("Upload complete.")


def cmd_train():
    print("Dispatching training job to Modal...")
    with modal.enable_output():
        with app.run(detach=True):
            train_on_modal.remote()


def cmd_download():
    LOCAL_CKPT.mkdir(exist_ok=True)
    print("Downloading checkpoints from Modal volume...")
    for entry in volume.listdir("checkpoints"):
        name = Path(entry.path).name
        dest = LOCAL_CKPT / name
        print(f"  {name}")
        with open(dest, "wb") as f:
            for chunk in volume.read_file(entry.path):
                f.write(chunk)
    print(f"Checkpoints saved to {LOCAL_CKPT}/")


if __name__ == "__main__":
    cmds = {"upload": cmd_upload, "train": cmd_train, "download": cmd_download}
    if len(sys.argv) != 2 or sys.argv[1] not in cmds:
        print(f"Usage: python3 modal_train.py [{' | '.join(cmds)}]")
        sys.exit(1)
    cmds[sys.argv[1]]()

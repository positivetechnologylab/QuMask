"""
Derive reduced-(n',k) datasets from a (15,k) master dataset.

Rather than re-running expensive Qiskit statevector simulation for every
(n',k) configuration, this module derives smaller-n datasets by subsetting
columns from an existing master dataset.

Algorithm
---------
For each instance i and block b in the master dataset:

  1. Identify the k target columns: ``target_positions[i, b]``.
  2. Identify the 15−k non-target (available) columns.
  3. Randomly select ``n' − k`` columns from the available set (per-block
     random selection, since target positions vary per block).
  4. Combine: ``selected_cols = sorted(target_cols) + sorted(decoy_cols)``.
  5. Extract the ``(shots, n')`` bitstring submatrix.
  6. Re-index target positions into ``[0, n')``: the new index of target qubit
     ``q`` is its position within ``selected_cols``.

Features are always recomputed from the n'-column submatrix via
``compute_block_features(submatrix, n', k)`` — never subsetted from the
master features, which have different dimension F and different semantics.

The target distribution ``p_star`` is position-invariant and copied unchanged.

Checkpointing
-------------
Each split (train/val/cal/test) is written atomically via a temp file +
``os.replace``. Partially-written files are never left on disk.

For large splits, instance-level processing is batched. Intermediate batch
shards are written to ``<out_dir>/.cache/`` so that a restart can skip
completed batches. The final step merges shards and deletes the cache.

Usage
-----
    python3 data/derive_nk.py \\
        --master-dir data/processed/n15_k3 \\
        --n-prime 6 \\
        --out-dir data/processed/n6_k3

    # dry-run (prints what would be done, no writes):
    python3 data/derive_nk.py --master-dir ... --n-prime 6 --out-dir ... --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np

from data.features import compute_block_features, feature_dim


# Batch size for shard-level checkpointing within a split.
_BATCH_SIZE = 500

# Splits to process, in order.
_SPLITS = ("train", "val", "cal", "test")


def _select_block_cols(
    target_pos: np.ndarray,
    n_master: int,
    n_prime: int,
    k: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Select n' columns from n_master qubits for one block.

    Picks all k target columns plus ``n' - k`` random columns from the
    non-target qubits. Returns them as two sorted lists so that callers can
    assemble ``selected_cols = target_cols + decoy_cols`` and re-index.

    Args:
        target_pos: Shape ``(k,)`` int array — absolute target qubit indices
            in ``[0, n_master)``.
        n_master: Total qubits in the master dataset (e.g. 15).
        n_prime: Target reduced qubit count (e.g. 6). Must satisfy
            ``k <= n_prime <= n_master``.
        k: Number of target qubits.
        rng: RNG used for decoy column selection.

    Returns:
        ``(target_cols, decoy_cols)`` — each a sorted list of ints.
        ``target_cols`` = sorted target indices (length k).
        ``decoy_cols`` = sorted selected non-target indices (length n' - k).
    """
    target_set = set(target_pos.tolist())
    available = sorted(set(range(n_master)) - target_set)
    n_decoy = n_prime - k
    chosen_decoy = sorted(rng.choice(available, size=n_decoy, replace=False).tolist())
    return sorted(target_pos.tolist()), chosen_decoy


def derive_split(
    master_path: str,
    n_prime: int,
    k: int,
    out_path: str,
    seed: int = 0,
    batch_size: int = _BATCH_SIZE,
) -> None:
    """Derive one split file from a master .npz file.

    Loads the master, applies per-block column subsetting for each instance,
    recomputes features, and writes the result to ``out_path`` atomically.

    If ``out_path`` already exists, skips immediately (split-level
    checkpointing). If the derivation is interrupted mid-way, no partial file
    is left on disk.

    Shard-level checkpointing: processes instances in batches of ``batch_size``
    and writes each batch to a ``.cache/`` subdirectory next to ``out_path``.
    On restart, completed batches are loaded from cache rather than recomputed.

    Args:
        master_path: Path to the master .npz file (e.g. n15_k3/train.npz).
        n_prime: Reduced qubit count for the output dataset.
        k: Number of target qubits (must match master).
        out_path: Destination .npz path. Written atomically.
        seed: Base RNG seed for reproducible decoy column selection.
        batch_size: Number of instances per shard.

    Raises:
        FileNotFoundError: If ``master_path`` does not exist.
        ValueError: If ``n_prime`` is out of range or k doesn't match master.
    """
    out_path = Path(out_path)
    if out_path.exists():
        print(f"    {out_path.name} exists, skipping")
        return

    master_path = Path(master_path)
    if not master_path.exists():
        raise FileNotFoundError(f"Master file not found: {master_path}")

    data = np.load(master_path)
    bitstrings_master = data["bitstrings"]     # (N, n_blocks, shots, n_master)
    target_positions_master = data["target_positions"]  # (N, n_blocks, k_master)
    p_stars = data["p_stars"]                  # (N, 2**k)

    N, n_blocks, shots, n_master = bitstrings_master.shape
    k_master = target_positions_master.shape[2]

    if k_master != k:
        raise ValueError(f"Master has k={k_master} but derive_split called with k={k}")
    if not (k <= n_prime <= n_master):
        raise ValueError(f"n_prime={n_prime} must be in [{k}, {n_master}]")

    F_prime = feature_dim(n_prime, k)
    cache_dir = out_path.parent / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_batches = (N + batch_size - 1) // batch_size

    # Process each batch, using cached shards where available.
    shard_paths: list[Path] = []
    for batch_idx in range(n_batches):
        shard_path = cache_dir / f"{out_path.stem}_batch{batch_idx}.npz"
        shard_paths.append(shard_path)

        if shard_path.exists():
            print(f"    batch {batch_idx}/{n_batches}: cached, skipping")
            continue

        i_start = batch_idx * batch_size
        i_end = min(i_start + batch_size, N)
        batch_n = i_end - i_start

        bs_batch = bitstrings_master[i_start:i_end]   # (batch_n, n_blocks, shots, n_master)
        tp_batch = target_positions_master[i_start:i_end]  # (batch_n, n_blocks, k)

        derived_bits = np.empty((batch_n, n_blocks, shots, n_prime), dtype=np.uint8)
        derived_tp = np.empty((batch_n, n_blocks, k), dtype=int)
        derived_feats = np.empty((batch_n, n_blocks, F_prime), dtype=np.float32)

        for i_local in range(batch_n):
            i_global = i_start + i_local
            for b in range(n_blocks):
                target_pos = tp_batch[i_local, b]  # (k,)
                rng = np.random.default_rng([seed, i_global, b])
                target_cols, decoy_cols = _select_block_cols(
                    target_pos, n_master, n_prime, k, rng
                )
                selected_cols = target_cols + decoy_cols  # length n_prime
                selected_arr = np.array(selected_cols, dtype=int)

                # Extract (shots, n_prime) submatrix.
                derived_bits[i_local, b] = bs_batch[i_local, b][:, selected_arr]

                # Re-index target positions into [0, n_prime).
                # For each original target qubit (in the original order given by
                # target_pos), find its position within selected_cols.
                for j, tc in enumerate(target_pos.tolist()):
                    derived_tp[i_local, b, j] = selected_cols.index(tc)

            # Recompute features from the n'-column bitstrings.
            derived_feats[i_local] = compute_block_features(derived_bits[i_local], n_prime, k)

        print(f"    batch {batch_idx + 1}/{n_batches}: instances {i_start}–{i_end - 1}")
        np.savez_compressed(
            str(shard_path),
            bitstrings=derived_bits,
            target_positions=derived_tp,
            features=derived_feats,
        )

    # Merge shards.
    all_bits = []
    all_tp = []
    all_feats = []
    for shard_path in shard_paths:
        shard = np.load(shard_path)
        all_bits.append(shard["bitstrings"])
        all_tp.append(shard["target_positions"])
        all_feats.append(shard["features"])

    merged_bits = np.concatenate(all_bits, axis=0)
    merged_tp = np.concatenate(all_tp, axis=0)
    merged_feats = np.concatenate(all_feats, axis=0)

    # Write atomically.
    tmp_path = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        str(tmp_path),
        features=merged_feats,
        p_stars=p_stars,
        bitstrings=merged_bits,
        target_positions=merged_tp,
    )
    os.replace(tmp_path, out_path)

    # Clean up cache shards for this split.
    for shard_path in shard_paths:
        shard_path.unlink(missing_ok=True)
    # Remove cache dir if now empty.
    try:
        cache_dir.rmdir()
    except OSError:
        pass  # other splits' shards still present — leave dir in place


def derive_all_splits(
    master_dir: str,
    n_prime: int,
    k: int,
    out_dir: str,
    seed: int = 0,
    splits: tuple[str, ...] = _SPLITS,
) -> None:
    """Derive all splits from a master directory.

    Expects ``{master_dir}/{split}.npz`` for each split in ``splits``.
    Writes derived files to ``{out_dir}/{split}.npz``.

    Args:
        master_dir: Directory containing master split files.
        n_prime: Reduced qubit count.
        k: Number of target qubits (must match master).
        out_dir: Output directory. Created if absent.
        seed: Base seed passed to ``derive_split``.
        splits: Tuple of split names to process.
    """
    master_dir = Path(master_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        master_path = master_dir / f"{split}.npz"
        out_path = out_dir / f"{split}.npz"
        print(f"  {split}: {master_path} -> {out_path}")
        derive_split(
            master_path=str(master_path),
            n_prime=n_prime,
            k=k,
            out_path=str(out_path),
            seed=seed,
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Derive a reduced-(n',k) dataset from a (n_master,k) master dataset."
    )
    p.add_argument("--master-dir", required=True,
                   help="Directory containing master {train,val,cal,test}.npz files.")
    p.add_argument("--n-prime", type=int, required=True,
                   help="Target reduced qubit count n'.")
    p.add_argument("--k", type=int, default=3,
                   help="Number of target qubits (must match master). Default: 3.")
    p.add_argument("--out-dir", required=True,
                   help="Output directory for derived .npz files.")
    p.add_argument("--seed", type=int, default=0,
                   help="Base RNG seed for decoy column selection. Default: 0.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without writing any files.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.dry_run:
        print(f"[dry-run] Would derive (n'={args.n_prime}, k={args.k}) from {args.master_dir} -> {args.out_dir}")
        for split in _SPLITS:
            src = Path(args.master_dir) / f"{split}.npz"
            dst = Path(args.out_dir) / f"{split}.npz"
            status = "EXISTS (skip)" if dst.exists() else "to derive"
            print(f"  {split}: {src} -> {dst}  [{status}]")
    else:
        derive_all_splits(
            master_dir=args.master_dir,
            n_prime=args.n_prime,
            k=args.k,
            out_dir=args.out_dir,
            seed=args.seed,
        )

"""
Synthetic sweep: variance baseline with uniform target and Dirichlet-random decoys.

No Qiskit. Target distribution is exactly uniform. Each decoy subset draws a fresh
Dirichlet(1,...,1) distribution every block, mimicking a fully random circuit.

Expected results (k=3, S=10, B=100):
  1st-place variance : 7/640 = 0.010938  (pure shot noise on uniform p*)
  margin             : driven by Dirichlet between-block shift

Usage
-----
    python3 scripts/sweep_baseline_synthetic.py
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baseline import variance_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-instances", type=int, default=200)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--n-blocks", type=int, default=100)
    p.add_argument("--shots-per-block", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def generate_synthetic_instance(
    k: int,
    n: int,
    n_blocks: int,
    shots_per_block: int,
    target_pos: tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate bitstrings where target is uniform and decoys are Dirichlet-random per block.

    Returns bitstrings of shape (n_blocks, shots_per_block, n).
    """
    output_dim = 2 ** k
    decoy_qubits = [q for q in range(n) if q not in set(target_pos)]

    p_target = np.ones(output_dim) / output_dim  # exactly uniform

    bitstrings = np.empty((n_blocks, shots_per_block, n), dtype=np.uint8)
    powers_target = 1 << np.arange(k - 1, -1, -1)

    for b in range(n_blocks):
        # Target: sample from uniform
        target_indices = rng.choice(output_dim, size=shots_per_block, p=p_target)
        target_bits = ((target_indices[:, None] & powers_target[None, :]) > 0).astype(np.uint8)
        for j, q in enumerate(target_pos):
            bitstrings[b, :, q] = target_bits[:, j]

        # Each decoy qubit gets an independent fresh Bernoulli(p) each block,
        # where p ~ Uniform(0,1). This gives genuine per-qubit between-block variance
        # without the marginal-collapsing effect of a joint high-dim Dirichlet.
        for q in decoy_qubits:
            p = rng.uniform(0, 1)
            bitstrings[b, :, q] = rng.binomial(1, p, size=shots_per_block).astype(np.uint8)

    return bitstrings


def main() -> None:
    args = parse_args()
    k, n = args.k, args.n
    rng = np.random.default_rng(args.seed)

    # Fixed target position for all instances (position doesn't affect the math)
    target_pos = tuple(range(k))
    true_subset = target_pos

    n_correct = 0
    margins = []
    first_place_scores = []

    for _ in range(args.n_instances):
        bitstrings = generate_synthetic_instance(k, n, args.n_blocks, args.shots_per_block, target_pos, rng)
        result = variance_baseline(bitstrings, k, n)

        selected = result["selected_subset"]
        all_variances = result["all_variances"]

        if selected == true_subset:
            n_correct += 1

        sorted_scores = sorted(all_variances.values())
        first_place_scores.append(sorted_scores[0])
        margins.append(sorted_scores[1] - sorted_scores[0])

    theoretical_first = (1/8) * (7/8) / args.shots_per_block  # 7/640

    print(f"instances : {args.n_instances}")
    print(f"correct   : {n_correct} / {args.n_instances}  ({100 * n_correct / args.n_instances:.1f}%)")
    print(f"mean 1st-place variance          : {np.mean(first_place_scores):.6f}  (theory: {theoretical_first:.6f})")
    print(f"mean margin (2nd - 1st variance) : {np.mean(margins):.6f}")


if __name__ == "__main__":
    main()

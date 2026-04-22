"""
Sweep the variance baseline over many fixed-position instances.

Metrics reported:
  - n_instances: total instances run
  - n_correct: how many times the baseline selected the true qubit subset
  - mean_margin: average variance gap between the best and second-best subset

Usage
-----
    python3 scripts/sweep_baseline_fixed.py
    python3 scripts/sweep_baseline_fixed.py --n-instances 100 --n-blocks 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.simulate import generate_instance_fixed
from baseline import variance_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-instances", type=int, default=50)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--n-blocks", type=int, default=100)
    p.add_argument("--shots-per-block", type=int, default=10)
    p.add_argument("--target-depth", type=int, default=4)
    p.add_argument("--seed-base", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k, n = args.k, args.n

    n_correct = 0
    margins = []
    first_place_scores = []

    for i in range(args.n_instances):
        inst = generate_instance_fixed(
            k=k, n=n,
            n_blocks=args.n_blocks,
            shots_per_block=args.shots_per_block,
            target_depth=args.target_depth,
            seed=args.seed_base + i,
        )

        true_subset = tuple(sorted(int(q) for q in inst["target_positions"][0]))
        result = variance_baseline(inst["bitstrings"], k, n)

        selected = result["selected_subset"]
        all_variances = result["all_variances"]

        correct = selected == true_subset
        if correct:
            n_correct += 1

        sorted_scores = sorted(all_variances.values())
        margins.append(sorted_scores[1] - sorted_scores[0])
        first_place_scores.append(sorted_scores[0])

    print(f"instances : {args.n_instances}")
    print(f"correct   : {n_correct} / {args.n_instances}  ({100 * n_correct / args.n_instances:.1f}%)")
    print(f"mean 1st-place variance          : {np.mean(first_place_scores):.6f}")
    print(f"mean margin (2nd - 1st variance) : {np.mean(margins):.6f}")


if __name__ == "__main__":
    main()

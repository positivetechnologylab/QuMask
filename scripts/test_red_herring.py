"""
Sweep adversary accuracy vs. red-herring fraction.

Holds total_blocks fixed at 1000 (by default) and varies the fraction that are
red herrings from 0% to 100% in steps of 10. For each x-point, runs n_trials
independent instances and records how often the variance baseline correctly
identifies the target qubit subset.

Output: JSON file with params and per-x-point results, ready for plotting.

Usage
-----
    python3 scripts/test_red_herring.py
    python3 scripts/test_red_herring.py --total-blocks 200 --n-trials 20 --out /tmp/rh.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.simulate import generate_instance_fixed
from baseline import variance_baseline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--total-blocks", type=int, default=1000)
    p.add_argument("--shots-per-block", type=int, default=10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--target-depth-min", type=int, default=4)
    p.add_argument("--target-depth-max", type=int, default=4)
    p.add_argument("--n-trials", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--out", type=str, default="results/test_red_herring.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    k, n = args.k, args.n
    rh_pcts = list(range(0, 101, 10))

    results = []
    for x_idx, rh_pct in enumerate(rh_pcts):
        n_rh = round(args.total_blocks * rh_pct / 100)
        n_real = args.total_blocks - n_rh

        n_correct = 0
        for i in range(args.n_trials):
            seed = args.seed_base + x_idx * args.n_trials + i
            inst = generate_instance_fixed(
                k=k, n=n,
                n_blocks=n_real,
                shots_per_block=args.shots_per_block,
                target_depth_min=args.target_depth_min,
                target_depth_max=args.target_depth_max,
                num_red_herring_blocks=n_rh,
                seed=seed,
            )
            true_subset = tuple(sorted(int(q) for q in inst["target_positions"][0]))
            result = variance_baseline(inst["bitstrings"], k, n)
            if result["selected_subset"] == true_subset:
                n_correct += 1

        accuracy = n_correct / args.n_trials
        results.append({
            "rh_pct": rh_pct,
            "n_real": n_real,
            "n_rh": n_rh,
            "n_correct": n_correct,
            "n_trials": args.n_trials,
            "accuracy": accuracy,
        })
        print(f"  rh={rh_pct:3d}%  n_real={n_real:4d}  n_rh={n_rh:4d}  "
              f"correct={n_correct}/{args.n_trials}  ({100*accuracy:.1f}%)")

    out = {
        "params": {
            "total_blocks": args.total_blocks,
            "shots_per_block": args.shots_per_block,
            "k": k,
            "n": n,
            "target_depth_min": args.target_depth_min,
            "target_depth_max": args.target_depth_max,
            "n_trials": args.n_trials,
            "seed_base": args.seed_base,
        },
        "results": results,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()

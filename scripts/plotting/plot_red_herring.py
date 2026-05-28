"""
Plot adversary accuracy vs. red-herring fraction.

Reads the JSON produced by scripts/test_red_herring.py and saves a line plot.

Usage
-----
    python3 scripts/plotting/plot_red_herring.py
    python3 scripts/plotting/plot_red_herring.py --in results/test_red_herring.json --out results/red_herring.png
"""

from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="input", type=str, default="results/test_red_herring.json")
    p.add_argument("--out", type=str, default="results/red_herring.png")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data = json.loads(Path(args.input).read_text())
    params = data["params"]
    results = data["results"]

    k, n = params["k"], params["n"]
    n_trials = params["n_trials"]

    rh_pcts = [r["rh_pct"] for r in results]
    accuracies = [r["accuracy"] for r in results]
    n_corrects = [r["n_correct"] for r in results]

    # 95% Wilson score confidence intervals
    lo, hi = [], []
    for correct, total in zip(n_corrects, [n_trials] * len(n_corrects)):
        p_hat = correct / total
        z = 1.96
        denom = 1 + z**2 / total
        center = (p_hat + z**2 / (2 * total)) / denom
        half = z * (p_hat * (1 - p_hat) / total + z**2 / (4 * total**2)) ** 0.5 / denom
        lo.append(max(0.0, center - half))
        hi.append(min(1.0, center + half))

    random_floor = 1.0 / comb(n, k)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(rh_pcts, accuracies, marker="o", linewidth=2, color="#2563eb", label="Adversary accuracy")
    ax.fill_between(rh_pcts, lo, hi, alpha=0.15, color="#2563eb", label="95% CI (Wilson)")
    ax.axhline(random_floor, linestyle="--", linewidth=1, color="#dc2626",
               label=f"Random-guess floor (1/C({n},{k}) ≈ {random_floor:.3f})")

    ax.set_xlabel("Red-herring shots (% of total shots)", fontsize=12)
    ax.set_ylabel("Adversary correct-identification rate", fontsize=12)
    ax.set_title(
        f"Variance-baseline attack vs. red-herring fraction\n"
        f"k={k}, n={n}, total_blocks={params['total_blocks']}, "
        f"shots/block={params['shots_per_block']}, trials={n_trials}",
        fontsize=11,
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(rh_pcts)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

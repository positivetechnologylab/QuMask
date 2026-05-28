"""
Traceable demo: classical variance baseline on the fixed-position obfuscation regime.

Generates one instance via generate_instance_fixed (target qubit positions held
constant across all blocks), runs variance_baseline, and writes a full trace to
an output .txt file.

Output sections:
  1. Instance parameters
  2. True target positions and p*
  3. All bitstrings (every block, every shot), with target columns labeled
  4. All C(n,k) subset variance scores, sorted ascending
  5. Baseline decision and TVD metrics
  6. Side-by-side distribution comparison (p*, p̂_marginal, p̂_baseline)

Usage
-----
    python3 scripts/demo_baseline_fixed.py
    python3 scripts/demo_baseline_fixed.py --seed 7 --n-blocks 50 --out results/demo.txt
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.simulate import generate_instance_fixed
from baseline import variance_baseline, marginal_for_subset
from utils.metrics import tvd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--n-blocks", type=int, default=100)
    p.add_argument("--shots-per-block", type=int, default=10)
    p.add_argument("--target-depth", type=int, default=4)
    p.add_argument("--num-red-herring-blocks", type=int, default=0)
    p.add_argument("--out", type=str, default="demo_baseline_fixed.txt")
    return p.parse_args()


def _bar(val: float, width: int = 24) -> str:
    filled = round(val * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {val:.6f}"


def main() -> None:
    args = parse_args()
    k, n = args.k, args.n
    sep = "=" * 80
    thin = "-" * 80

    inst = generate_instance_fixed(
        k=k, n=n,
        n_blocks=args.n_blocks,
        shots_per_block=args.shots_per_block,
        target_depth_min=args.target_depth,
        target_depth_max=args.target_depth,
        num_red_herring_blocks=args.num_red_herring_blocks,
        seed=args.seed,
    )

    bitstrings = inst["bitstrings"]              # (total_blocks, shots, n)
    p_star = inst["p_star"]                      # (2**k,) exact theoretical distribution
    target_positions = inst["target_positions"]  # (total_blocks, k) — all rows identical
    rh_mask = inst["red_herring_mask"]           # (total_blocks,) bool

    # true_subset_ordered: preserves the MSB-first circuit orientation, used for
    # marginal extraction (qubit order determines which bit maps to which bin).
    # true_subset_sorted: matches the combinations(range(n),k) key space used by
    # variance_baseline — the adversary identifies a set of qubits, not an ordering.
    true_subset_ordered = tuple(int(q) for q in target_positions[0])
    true_subset_sorted = tuple(sorted(true_subset_ordered))
    target_set = set(true_subset_ordered)

    # Run baseline
    result = variance_baseline(bitstrings, k, n)
    all_variances = result["all_variances"]
    selected = result["selected_subset"]
    correct = selected == true_subset_sorted

    # Oracle marginal: extract with MSB-first ordering so bin indices align with p*.
    p_hat_marginal = marginal_for_subset(bitstrings, true_subset_ordered)

    # Baseline distribution: if the baseline found the right qubit set, re-extract
    # with the same MSB-first ordering as p* so the per-bin comparison table is
    # in a consistent basis. If it selected the wrong set, bins are incommensurable
    # with p* regardless of ordering — note this in the output.
    if correct:
        p_hat_baseline = marginal_for_subset(bitstrings, true_subset_ordered)
    else:
        p_hat_baseline = result["p_hat_baseline"]

    tvd_vs_true = tvd(p_hat_baseline, p_star)
    tvd_vs_marginal = tvd(p_hat_baseline, p_hat_marginal)
    sorted_subsets = sorted(all_variances, key=all_variances.__getitem__)
    true_rank = sorted_subsets.index(true_subset_sorted) + 1

    lines: list[str] = []
    w = lines.append

    # -------------------------------------------------------------------------
    # 1. Instance parameters
    # -------------------------------------------------------------------------
    w(sep)
    w("INSTANCE PARAMETERS")
    w(sep)
    total_blocks = args.n_blocks + args.num_red_herring_blocks
    w(f"  seed                   : {args.seed}")
    w(f"  k                      : {k}")
    w(f"  n                      : {n}")
    w(f"  n_blocks               : {args.n_blocks}")
    w(f"  num_red_herring_blocks : {args.num_red_herring_blocks}")
    w(f"  total_blocks           : {total_blocks}")
    w(f"  shots_per_block        : {args.shots_per_block}")
    w(f"  target_depth           : {args.target_depth}")
    w(f"  obfuscation mode       : fixed (target qubit positions constant across blocks)")
    w("")

    # -------------------------------------------------------------------------
    # 2. True target
    # -------------------------------------------------------------------------
    w(sep)
    w("TRUE TARGET")
    w(sep)
    w(f"  Target qubit positions (0-indexed, MSB-first): {list(true_subset_ordered)}")
    w(f"  Fixed across all {args.n_blocks} real blocks ({args.num_red_herring_blocks} red herring blocks interleaved).")
    w("")
    w(f"  p* — EXACT THEORETICAL distribution from statevector simulation ({2**k} bins).")
    w(f"  This is the infinite-shot oracle: what the circuit produces with certainty.")
    w("")
    for i, prob in enumerate(p_star):
        bits = format(i, f"0{k}b")
        w(f"    |{bits}> (={i:2d})  {_bar(prob)}")
    w("")

    # -------------------------------------------------------------------------
    # 3. All bitstrings
    # -------------------------------------------------------------------------
    w(sep)
    w("ALL BITSTRINGS")
    w("  Columns marked T are target qubits; D are decoy qubits.")
    w("  Format per row: [block shot]  q0 q1 ... q{n-1}  | target_bits (decimal)")
    w(sep)

    # Header row
    col_labels = "".join(
        f"q{q:<2}{'T' if q in target_set else 'D'} " for q in range(n)
    )
    w(f"  {'block':>5} {'shot':>4}  {col_labels}  target_bits  decimal")
    w("  " + thin)

    powers = 1 << np.arange(k - 1, -1, -1)
    for b in range(total_blocks):
        rh_label = " [RH]" if rh_mask[b] else "     "
        for s in range(args.shots_per_block):
            row = bitstrings[b, s]
            bits_str = "".join(f"{row[q]}    " for q in range(n))
            target_bits = row[list(true_subset_ordered)]
            target_str = "".join(str(bit) for bit in target_bits)
            decimal = int(target_bits @ powers)
            w(f"  {b:>5}{rh_label} {s:>4}  {bits_str}  {target_str}         {decimal:>7}")
        if b < total_blocks - 1:
            w("")  # blank line between blocks for readability

    w("")

    # -------------------------------------------------------------------------
    # 4. Variance scores — all subsets
    # -------------------------------------------------------------------------
    n_subsets = sum(1 for _ in combinations(range(n), k))
    w(sep)
    w(f"VARIANCE SCORES — all C({n},{k}) = {n_subsets} subsets, sorted ascending")
    w("  Lower variance = more consistent marginal across blocks = stronger signal")
    w(sep)
    w(f"  {'rank':>5}  {'subset':<25}  {'variance':>14}  note")
    w("  " + thin)
    for rank, subset in enumerate(sorted_subsets, 1):
        var = all_variances[subset]
        note = "<-- TRUE TARGET" if subset == true_subset_sorted else ""
        selected_note = "<-- SELECTED" if subset == selected and subset != true_subset_sorted else ""
        w(f"  {rank:>5}  {str(list(subset)):<25}  {var:>14.8f}  {note}{selected_note}")
    w("")

    # -------------------------------------------------------------------------
    # 5. Baseline decision
    # -------------------------------------------------------------------------
    w(sep)
    w("BASELINE DECISION")
    w(sep)
    w(f"  Selected subset          : {list(selected)}")
    w(f"  True target subset       : {list(true_subset_sorted)}  (MSB-first order: {list(true_subset_ordered)})")
    w(f"  Correct selection        : {'YES' if correct else 'NO'}")
    w(f"  True subset rank         : {true_rank} / {n_subsets}")
    w(f"  Selected subset variance : {all_variances[selected]:.8f}")
    w(f"  True subset variance     : {all_variances[true_subset_sorted]:.8f}")
    w(f"  TVD(p_hat_baseline, p*)          : {tvd_vs_true:.6f}")
    w(f"  TVD(p_hat_baseline, p_hat_marginal): {tvd_vs_marginal:.6f}")
    w("")

    # -------------------------------------------------------------------------
    # 6. Distribution comparison
    # -------------------------------------------------------------------------
    w(sep)
    w("DISTRIBUTION COMPARISON")
    w("  p*           : exact theoretical distribution (statevector, infinite-shot oracle)")
    w(f"  p_hat_marg   : empirical marginal over oracle qubit positions ({total_blocks * args.shots_per_block} shots incl. RH, noisy estimate of p*)")
    w("  p_hat_base   : baseline estimate (no oracle position knowledge)")
    if correct:
        w("  All three distributions share the same bin basis (MSB-first qubit ordering).")
    else:
        w("  WARNING: baseline selected wrong qubit set. p_hat_base bins are NOT")
        w("  aligned with p* or p_hat_marg. Per-bin comparison is not meaningful;")
        w("  TVD values are still valid (permutation-invariant).")
    w(sep)
    w(f"  {'bin':<12}  {'p*':>10}  {'p_hat_marg':>12}  {'p_hat_base':>12}")
    w("  " + thin)
    for i in range(2**k):
        bits = format(i, f"0{k}b")
        w(f"  |{bits}> (={i:2d})  {p_star[i]:>10.6f}  {p_hat_marginal[i]:>12.6f}  {p_hat_baseline[i]:>12.6f}")
    w("")
    w(f"  TVD(p_hat_marg,  p*)      : {tvd(p_hat_marginal, p_star):.6f}  (shot noise floor)")
    w(f"  TVD(p_hat_base,  p*)      : {tvd_vs_true:.6f}")
    w(f"  TVD(p_hat_base,  p_hat_marg): {tvd_vs_marginal:.6f}")
    w(sep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()

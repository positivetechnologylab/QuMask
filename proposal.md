# Problem Statement

A quantum circuit of k qubits (the "target") is obfuscated within an n-qubit system using the QuMask protocol. Every 10 shots, the target circuit is randomly repositioned among n qubits; the remaining n−k qubits are randomly partitioned into disjoint groups, each running an independent random Qiskit circuit of depth 0.75x–1.25x the target depth. The full n-qubit system is a product state: the joint distribution factors as p_target ⊗ p_decoy_1 ⊗ p_decoy_2 ⊗ ···, with no entanglement between the target and decoy registers.

An adversary observes 10k total bitstrings of length n across 1000 blocks of 10 shots each. The block structure is known to the adversary (i.e., which shots belong to which block); shots within a block are i.i.d. The target circuit is fixed across all 10k shots. The adversary knows k but not the target qubit positions in any block.

**Goal:** Recover the probability distribution p* ∈ Δ(2^k) over k-bit computational basis states that the target circuit produces.

**Initial scope:** k=3, n=10. The implementation is designed to generalize to other (k, n) values.

---

# Considered Approaches & Decisions

**Statistical variance test — rejected.** Computing per-block marginals for all C(n,k) candidate qubit subsets and selecting the most cross-block-consistent one is computationally trivial but statistically fragile. With only 10 shots per block, individual marginal estimates have standard error ~0.16 per bin, making variance-based consistency detection unreliable — especially for near-uniform target distributions or when decoy circuits produce structured outputs due to their constrained depth.

**LSTM on raw bitstrings — rejected.** Treats the shot sequence as temporally dependent, which it isn't — shots within a block are i.i.d. and the target distribution does not evolve across blocks. The sequential inductive bias actively works against the problem structure.

**Transformer on higher-order correlator features — adopted.** The correct invariant is that the target distribution's correlator signature is stable across blocks despite positional re-randomization, while decoy correlators are freshly randomized each block. Self-attention across blocks (treating each block as a token) naturally learns to identify and aggregate this cross-block consistency signal without requiring fixed qubit-position identity.

---

# Feature Representation

For each block b, compute all m-body correlators up to order k from the 10 shots:

```
φ_b = { ⟨Z_{i1}···Z_{im}⟩ : {i1,...,im} ⊆ [n], m = 1,...,k }
```

where each correlator is the empirical mean of the product of ±1-encoded qubit values across 10 shots. This yields F = Σⱼ₌₁ᵏ C(n,j) features per block.

**Justification:** A distribution over k qubits has 2^k − 1 free parameters; recovering it requires correlators up to order k. Pairwise features alone are insufficient for k ≥ 4.

**Feature dimensions (n=10):**

| k | Output dim (2^k) | Feature dim F |
|---|-----------------|---------------|
| 3 | 8               | 175           |
| 5 | 32              | 637           |
| 10| 1024            | 1023          |

Model input tensor shape: (1000 blocks × F features).

---

# Model Architecture

Transformer with blocks as tokens:

- **Input:** (1000 × F) feature tensor, linearly projected to embedding dimension d
- **Positional encoding:** omitted — block order is meaningless
- **Self-attention layers:** L layers of multi-head attention across the 1000-block sequence; attention learns to upweight mutually consistent blocks (where the target signal dominates) and downweight noisy or confounded ones
- **Aggregation:** mean pooling across the block dimension after the final attention layer
- **Output head:** MLP → 2^k logits → softmax → p̂ ∈ Δ(2^k)
- **Training loss:** KL(p* ∥ p̂) — forward KL, which penalizes confident errors by diverging when p̂ assigns zero mass where p* has support. This is appropriate since p* is fully known at training time and mode-averaging behavior is preferred over mode-seeking.

**Uncertainty quantification — deep ensembles:**

Train M=10–20 independent transformers with different random seeds. At inference:

```
p̂   = (1/M) Σₘ p̂⁽ᵐ⁾            (point estimate)
σᵢ² = (1/M) Σₘ (p̂ᵢ⁽ᵐ⁾ − p̂ᵢ)²   (per-bin variance)
```

Report p̂ᵢ ± 2σᵢ as approximate 95% per-bin intervals. Apply conformal prediction on a held-out calibration set to obtain a rigorous coverage guarantee: a TV-ball around p̂ that contains p* with probability ≥ 1−α. The nonconformity score is the TV distance between the ensemble prediction and the empirical marginal estimated from held-out blocks.

---

# Training Data

All training data is simulated. Each training instance:

1. Sample a random k-qubit target circuit (Qiskit, random gate set) → compute exact output distribution p* via statevector simulation
2. For each of 1000 blocks:
   - Sample k target qubit positions uniformly from [n]
   - Randomly partition remaining n−k qubits into disjoint groups
   - Generate an independent random Qiskit circuit on each partition group at 0.75x–1.25x target depth
   - Sample 10 shots from the full n-qubit distribution (product state: p_target ⊗ p_decoy_1 ⊗ ···)
   - Compute φ_b (correlators up to order k)
3. Label: p*

Generate thousands of such instances across varying target circuits (spanning a range of output entropies) for a robust training set. Stratify by target circuit entropy to ensure adequate representation of near-uniform distributions, which are the hardest case.

---

# Evaluation

## Baseline

A classical extractor serves as the reference baseline: enumerate all C(n,k) candidate qubit subsets, compute the empirical marginal for each subset pooled across all 10k shots, and select the subset with minimum cross-block variance. This is computationally cheap and represents the best a simple statistic-based adversary can do.

## Primary Metric

**TVD against the empirical marginal** — the actual k-qubit marginal extracted from the 10k observed bitstrings over the true target qubits. This is the natural adversarial threat metric: it measures how close the ML adversary gets to what a perfect classical extractor with oracle knowledge of qubit positions would recover. Formally:

```
TVD_adv = TV(p̂,  p̂_marginal)
```

where p̂_marginal = (1/10k) Σ (empirical counts over target qubits across all 10k shots).

p* (the true distribution) is used only during training as the ground-truth label. It is not directly observable at inference time.

## Full Metric Suite

| Metric | What it measures |
|--------|-----------------|
| TV(p̂, p̂_marginal) | Primary adversarial accuracy vs. empirical marginal |
| TV(p̂, p*) | Absolute accuracy vs. true distribution (training signal reference) |
| KL(p* ∥ p̂) | Penalizes confident errors; unbounded |
| ECE | Whether uncertainty estimates are calibrated |
| Conformal coverage @ 95% | Rigorous interval validity |

All metrics stratified by: k, n, shots per block, and target circuit entropy. Near-uniform distributions must be reported separately as the hardest case.

---

# Complexity & Feasibility

- **Feature extraction:** O(1000 · F · shots) per inference — milliseconds for k=3, n=10
- **Attention:** O(1000² · d) per layer — tractable on a single GPU
- **Scaling wall:** Arises from growing n, not k. At n=20, k=10, F = Σⱼ₌₁¹⁰ C(20,j) ≈ 616k — sparse correlator approximations would be needed
- **Practical verdict:** Fully feasible as a research prototype for k=3, n=10 on standard GPU hardware; architecture is parameterized by (k, n) for extension

---

# Full Pipeline

```
[Simulation]
Generate k-qubit target circuit → p* (statevector)
  ↓
For 1000 blocks:
  Sample k target qubit positions uniformly from [n]
  Randomly partition remaining n−k qubits into disjoint groups
  Generate independent random Qiskit circuits per group (0.75x–1.25x target depth)
  Sample 10 shots from product distribution p_target ⊗ p_decoy_1 ⊗ ···
  Compute φ_b (correlators up to order k)
  ↓
Stack → input tensor (1000 × F), label p*

[Training]
Train ensemble of M transformers
Loss: KL(p* ∥ p̂), optimizer: AdamW
Validate on held-out simulation instances
Calibrate conformal predictor on calibration split

[Inference]
New QuMask dataset (10k bitstrings, 1000 blocks, block structure known)
  ↓
Extract φ_b per block → (1000 × F) tensor
  ↓
Forward pass through M ensemble members
  ↓
p̂ (mean), σ (per-bin std), conformal interval

[Evaluation]
TV(p̂, p̂_marginal) — primary adversarial metric
TV(p̂, p*), KL, ECE, conformal coverage
Stratified by k, n, entropy of p*
Compare against classical variance-based baseline
```

Phase 1 — Foundation (no ML, no Qiskit)
These have zero external dependencies and can be verified with hand-computed examples.

1. utils/metrics.py
Implement all five functions. These are pure numpy math — no dependencies anywhere — and every downstream result depends on them being correct.

Testing:

tvd: hand-verify TV([1,0,0,0], [0,1,0,0]) = 1.0, TV(p,p) = 0, TV = 0.5·L1
kl_divergence: verify KL(p,p) = 0, KL blows up when q has zero mass where p doesn't (before eps clipping), verify eps clipping behavior explicitly
empirical_marginal: construct a fake (n_blocks, shots, n) array where you know the true marginal, verify the output matches
compute_all_metrics: integration test — feed known p_hat, p_star, p_hat_marginal, verify all dict keys are present and values are finite
2. data/features.py
Implement get_all_subsets, feature_dim, compute_correlators, compute_block_features. Pure numpy, no Qiskit.

Testing:

get_all_subsets(n=3, k=2) should return exactly [(0,), (1,), (2,), (0,1), (0,2), (1,2)] — check length = C(3,1)+C(3,2) = 6
feature_dim(10, 3) = 175 — spot check the table from the proposal
compute_correlators: construct a trivial case — all-zeros bitstring block → all correlators should be +1 (since Z_i = 1−2·0 = +1); all-ones → all correlators = (−1)^m for order m
compute_correlators: 10 shots of alternating 0/1 on qubit 0 → ⟨Z_0⟩ ≈ 0; verify sign conventions throughout
compute_block_features: verify output shape (n_blocks, F) for (n=10,k=3)
Verify get_all_subsets is cached: call it twice, confirm it returns the same object (identity check)
Phase 2 — Simulation
3. utils/circuits.py
Implement random_circuit, statevector_distribution, sample_shots, interleave_bitstrings. First Qiskit dependency.

Testing:

statevector_distribution on the |+⟩ state (single H gate on 1 qubit) → [0.5, 0.5]; on |0⟩ (no gates) → [1.0, 0.0]
statevector_distribution output sums to 1 for random circuits of varying sizes
sample_shots: sample 10k shots from [0.5, 0.5], verify empirical frequency is within 3σ of 0.5
sample_shots with seed is deterministic; different seeds give different results
interleave_bitstrings: construct a 3-qubit case by hand — target on qubits [1,3], decoy on [0,2,4] — verify columns land in the right positions in the assembled 5-qubit output
4. data/simulate.py
Implement random_decoy_partition, generate_instance, generate_dataset, save_dataset, load_dataset.

Testing:

random_decoy_partition: verify output groups are disjoint, their union equals input, every group has ≥ 1 element; run 1000 times, check group-size distribution looks geometric
generate_instance: verify output dict has correct keys, shapes, and dtypes:
features: (1000, 175) float32 for k=3,n=10
p_star: (8,) float64 summing to 1
bitstrings: (1000, 10, 10) uint8 with values in {0,1}
target_positions: (1000, 3) int with values in [0,9], no duplicates within a row
generate_instance with same seed → identical output (determinism check)
save_dataset / load_dataset round-trip: save and reload, verify all arrays are bit-for-bit identical
Sanity check on feature values: all entries in [-1, 1]
Phase 3 — Baseline (validates the problem is solvable before touching ML)
5. baseline.py
Implement marginal_for_subset, cross_block_variance, variance_baseline, run_baseline_on_dataset.

Testing:

marginal_for_subset: on hand-crafted bitstrings where the answer is known, verify counts exactly
cross_block_variance: on a constant-marginal dataset (same distribution every block) → variance ≈ 0; on a random-marginal dataset → variance > 0
variance_baseline oracle test: generate one instance, pass in the actual target qubit positions as one of the C(n,k) candidates, verify the baseline at least considers the correct subset (its variance should be low). This is a sanity check that the scoring function works, not that the baseline succeeds.
End-to-end: run variance_baseline on 100 test instances, compute mean TVD against p̂_marginal, log the result. You don't need it to be good — you need it to be finite, and it gives you a concrete number to beat.
Phase 4 — Model
6. model/transformer.py
Implement QuMaskTransformer, build_model_from_config.

Testing:

Shape test: forward pass with (batch=4, n_blocks=1000, F=175) → output (4, 8) for k=3
Output is logits (not probabilities): values not necessarily in [0,1], not summing to 1
After softmax: output sums to 1 per instance
n_parameters returns a positive integer; for Small config (~200k), verify it's in the right ballpark
Gradient flow: compute a loss and call .backward(), verify all parameters have non-None .grad
No positional encoding: permuting the 1000-block axis of the input tensor should change individual attention outputs but the mean-pooled output should be invariant. Verify this with a fixed input.
7. model/ensemble.py
Implement train_single_member, train_all, QuMaskEnsemble.predict, QuMaskEnsemble.save, load_ensemble.

Testing:

train_single_member smoke test: train for 2 epochs on 50 instances, verify loss decreases from epoch 1 to epoch 2
Early stopping: construct a scenario where val loss doesn't improve (e.g., constant val set), verify training stops at patience epochs
Checkpoint round-trip: save and reload a member, verify predict output is bit-for-bit identical before and after
QuMaskEnsemble.predict output shapes: p_hat sums to 1 per instance; sigma is non-negative; member_preds has shape (M, batch, 2**k)
Diversity check: verify sigma > 0 for at least some instances (members are not identical)
Phase 5 — Uncertainty Quantification
8. model/conformal.py
Implement ConformalPredictor.calibrate, prediction_interval, empirical_coverage.

Testing:

Coverage guarantee: calibrate on N_cal=500 instances, run empirical_coverage on a fresh test set of 1000 instances, verify result ≥ 1−α (0.95). This is the correctness test — split conformal prediction has a guaranteed finite-sample coverage property, so if this fails, the implementation is wrong.
Radius is finite and positive after calibration
prediction_interval raises RuntimeError before calibrate is called
Monotonicity: for α=0.01 the radius should be ≥ radius at α=0.05 (tighter confidence requires a larger ball)
Phase 6 — Integration
9. train.py + evaluate.py
Testing:

End-to-end smoke test: generate 100 training instances, train 2 ensemble members for 3 epochs, run evaluation. Verify no crashes, all output files exist, metrics are finite.
Verify the results table prints model TVD and baseline TVD side by side
Verify results.npz is written with the expected keys and shapes
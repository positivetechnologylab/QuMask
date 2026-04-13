"""
Tests for utils/circuits.py

All tests here are marked @pytest.mark.slow because they invoke Qiskit.
Each test should be fast in isolation (small circuits, few shots) but the
Qiskit import and statevector simulation make them unsuitable for the fast
no-slow pass.


class TestRandomCircuit:
    # Test that the returned object is a Qiskit QuantumCircuit with the correct
    #   number of qubits (circuit.num_qubits == n_qubits).
    # Test that the circuit has no measurement gates (no circuit.measurements),
    #   since it must be statevector-compatible.
    # Test that the same (n_qubits, depth, seed) always produces the same circuit
    #   (determinism): call twice and compare the gate lists.
    # Test that different seeds produce different circuits (non-determinism across
    #   seeds): verify at least one gate differs between two seeded calls.
    # Test that depth=0 produces a valid (empty) circuit with the right qubit count.


class TestStatevectorDistribution:
    # Test the |0⟩ state (no gates applied): output must be [1.0, 0.0].
    # Test the |+⟩ state (single H gate on 1 qubit): output must be [0.5, 0.5].
    # Test the |1⟩ state (single X gate on 1 qubit): output must be [0.0, 1.0].
    # Test that the output sums to 1.0 for random circuits of varying sizes
    #   (n_qubits in {1, 2, 3, 4}).
    # Test output shape is (2**n_qubits,).
    # Test output dtype is float64.
    # Test all values are non-negative (Born probabilities).


class TestSampleShots:
    # Test output shape is (n_shots, n_qubits).
    # Test output dtype is uint8 and all values are in {0, 1}.
    # Test determinism: same distribution, same seed, same n_shots → identical array.
    # Test statistical correctness: sample 10_000 shots from [0.5, 0.5] (1 qubit),
    #   verify the empirical frequency of 0 is within 3 standard deviations of 0.5
    #   (expected std ≈ 0.005 for 10k shots).
    # Test that different seeds produce different samples.


class TestInterleaveBitstrings:
    # Test a fully hand-specified case: 2 qubits target at positions [1, 3],
    #   2 qubits decoy at positions [0, 2, 4] split into one group.
    #   Construct target_bits and decoy_bits by hand and verify that the assembled
    #   5-qubit output has each bit in the correct column.
    # Test that the union of target_positions and all decoy_positions equals [0..n-1]
    #   (no column is missing or duplicated). This is a property test: generate
    #   random valid position assignments and check the assembled array has no
    #   NaN/uninitialized columns.
    # Test output shape is (n_shots, n).
    # Test output dtype is uint8 with values in {0, 1}.
"""

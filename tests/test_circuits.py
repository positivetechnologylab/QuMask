"""
Tests for utils/circuits.py
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Measure

from utils.circuits import (
    interleave_bitstrings,
    random_circuit,
    sample_shots,
    statevector_distribution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_measurements(circuit: QuantumCircuit) -> bool:
    return any(isinstance(instr.operation, Measure) for instr in circuit.data)


def _single_qubit_circuit(gate: str) -> QuantumCircuit:
    """Return a 1-qubit circuit with the named gate applied."""
    qc = QuantumCircuit(1)
    getattr(qc, gate)(0)
    return qc


# ---------------------------------------------------------------------------
# TestRandomCircuit
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRandomCircuit:
    def test_num_qubits(self):
        for n in (1, 3, 5):
            c = random_circuit(n, depth=2, seed=0)
            assert c.num_qubits == n

    def test_no_measurements(self):
        c = random_circuit(4, depth=3, seed=0)
        assert not _has_measurements(c)

    def test_determinism(self):
        c1 = random_circuit(3, depth=4, seed=42)
        c2 = random_circuit(3, depth=4, seed=42)
        assert c1 == c2

    def test_different_seeds_differ(self):
        c1 = random_circuit(3, depth=4, seed=0)
        c2 = random_circuit(3, depth=4, seed=1)
        assert c1 != c2

    def test_depth_zero_returns_valid_circuit(self):
        c = random_circuit(3, depth=0, seed=0)
        assert isinstance(c, QuantumCircuit)
        assert c.num_qubits == 3
        assert not _has_measurements(c)
        assert len(c.data) == 0

    def test_depth_zero_statevector_compatible(self):
        # depth=0 circuit must produce a valid distribution (|000⟩ state)
        c = random_circuit(3, depth=0, seed=0)
        p = statevector_distribution(c)
        assert p[0] == pytest.approx(1.0)
        assert p[1:].sum() == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestStatevectorDistribution
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestStatevectorDistribution:
    def test_zero_state(self):
        # No gates → |0⟩, so p[0] = 1.0
        qc = QuantumCircuit(1)
        p = statevector_distribution(qc)
        assert p == pytest.approx([1.0, 0.0])

    def test_x_gate(self):
        # X|0⟩ = |1⟩
        p = statevector_distribution(_single_qubit_circuit("x"))
        assert p == pytest.approx([0.0, 1.0])

    def test_h_gate(self):
        # H|0⟩ = |+⟩, uniform over {0,1}
        p = statevector_distribution(_single_qubit_circuit("h"))
        assert p == pytest.approx([0.5, 0.5], abs=1e-12)

    def test_shape(self):
        for n in (1, 2, 3, 4):
            c = random_circuit(n, depth=2, seed=0)
            p = statevector_distribution(c)
            assert p.shape == (2**n,)

    def test_dtype(self):
        c = random_circuit(2, depth=2, seed=0)
        p = statevector_distribution(c)
        assert p.dtype == np.float64

    def test_sums_to_one(self):
        for n in (1, 2, 3, 4):
            c = random_circuit(n, depth=3, seed=n)
            p = statevector_distribution(c)
            assert p.sum() == pytest.approx(1.0, abs=1e-12)

    def test_non_negative(self):
        for n in (1, 2, 3, 4):
            c = random_circuit(n, depth=3, seed=n + 10)
            p = statevector_distribution(c)
            assert np.all(p >= 0)

    def test_bell_state(self):
        # (H ⊗ I) · CNOT → |Φ+⟩: p[00]=0.5, p[11]=0.5
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        p = statevector_distribution(qc)
        # Qiskit little-endian: index 0 = |00⟩, index 3 = |11⟩
        assert p[0] == pytest.approx(0.5, abs=1e-12)
        assert p[3] == pytest.approx(0.5, abs=1e-12)
        assert p[1] == pytest.approx(0.0, abs=1e-12)
        assert p[2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# TestSampleShots
# ---------------------------------------------------------------------------

class TestSampleShots:
    def test_shape(self):
        dist = np.array([0.5, 0.5])
        bits = sample_shots(dist, n_shots=20, seed=0)
        assert bits.shape == (20, 1)

    def test_shape_multi_qubit(self):
        dist = np.ones(8) / 8  # 3 qubits, uniform
        bits = sample_shots(dist, n_shots=50, seed=0)
        assert bits.shape == (50, 3)

    def test_dtype_and_values(self):
        dist = np.array([0.25, 0.25, 0.25, 0.25])
        bits = sample_shots(dist, n_shots=100, seed=0)
        assert bits.dtype == np.uint8
        assert set(bits.flatten().tolist()).issubset({0, 1})

    def test_determinism(self):
        dist = np.array([0.3, 0.3, 0.2, 0.2])
        a = sample_shots(dist, n_shots=50, seed=7)
        b = sample_shots(dist, n_shots=50, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        dist = np.array([0.5, 0.5])
        a = sample_shots(dist, n_shots=100, seed=0)
        b = sample_shots(dist, n_shots=100, seed=1)
        assert not np.array_equal(a, b)

    def test_statistical_correctness(self):
        # 1 qubit, fair coin: empirical p(0) should be within 3σ of 0.5
        dist = np.array([0.5, 0.5])
        bits = sample_shots(dist, n_shots=10_000, seed=0)
        freq_zero = (bits[:, 0] == 0).mean()
        # σ ≈ sqrt(0.25 / 10000) ≈ 0.005
        assert abs(freq_zero - 0.5) < 3 * 0.005

    def test_deterministic_distribution(self):
        # All weight on index 5 (binary 101 little-endian → bits [1,0,1])
        dist = np.zeros(8)
        dist[5] = 1.0
        bits = sample_shots(dist, n_shots=10, seed=0)
        expected = np.array([[1, 0, 1]] * 10, dtype=np.uint8)
        np.testing.assert_array_equal(bits, expected)

    def test_little_endian_bit_ordering(self):
        # Index 1 = 0b001 little-endian → qubit 0 = 1, qubit 1 = 0
        dist = np.zeros(4)
        dist[1] = 1.0
        bits = sample_shots(dist, n_shots=5, seed=0)
        assert np.all(bits[:, 0] == 1)
        assert np.all(bits[:, 1] == 0)

    def test_index_2_little_endian(self):
        # Index 2 = 0b010 little-endian → qubit 0 = 0, qubit 1 = 1
        dist = np.zeros(4)
        dist[2] = 1.0
        bits = sample_shots(dist, n_shots=5, seed=0)
        assert np.all(bits[:, 0] == 0)
        assert np.all(bits[:, 1] == 1)

    def test_single_shot(self):
        dist = np.array([0.5, 0.5])
        bits = sample_shots(dist, n_shots=1, seed=0)
        assert bits.shape == (1, 1)
        assert bits.dtype == np.uint8


# ---------------------------------------------------------------------------
# TestInterleaveBitstrings
# ---------------------------------------------------------------------------

class TestInterleaveBitstrings:
    def test_hand_specified_case(self):
        # Target at positions [1, 3], decoy group at positions [0, 2, 4]
        # n=5, n_shots=3
        n_shots = 3
        target_bits = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)
        decoy_bits = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.uint8)
        target_positions = np.array([1, 3])
        decoy_positions = [np.array([0, 2, 4])]

        out = interleave_bitstrings(
            target_bits, [decoy_bits], target_positions, decoy_positions, n=5
        )

        assert out.shape == (3, 5)
        # Column 1 and 3 come from target_bits
        np.testing.assert_array_equal(out[:, [1, 3]], target_bits)
        # Columns 0, 2, 4 come from decoy_bits
        np.testing.assert_array_equal(out[:, [0, 2, 4]], decoy_bits)

    def test_all_columns_covered_no_overlap(self):
        # Property: every column index 0..n-1 is assigned exactly once
        rng = np.random.default_rng(99)
        n, k, n_shots = 10, 3, 5
        perm = rng.permutation(n)
        target_positions = perm[:k]
        remaining = perm[k:]
        # Split remaining into two groups
        split = len(remaining) // 2
        decoy_positions = [remaining[:split], remaining[split:]]

        target_bits = rng.integers(0, 2, size=(n_shots, k), dtype=np.uint8)
        decoy_bits_list = [
            rng.integers(0, 2, size=(n_shots, len(g)), dtype=np.uint8)
            for g in decoy_positions
        ]

        out = interleave_bitstrings(
            target_bits, decoy_bits_list, target_positions, decoy_positions, n=n
        )
        assert out.shape == (n_shots, n)
        # No column should be uninitialized — verify all columns match their source
        np.testing.assert_array_equal(out[:, target_positions], target_bits)
        for bits, positions in zip(decoy_bits_list, decoy_positions):
            np.testing.assert_array_equal(out[:, positions], bits)

    def test_shape(self):
        n_shots, n, k = 7, 6, 2
        target_bits = np.zeros((n_shots, k), dtype=np.uint8)
        decoy_bits = np.zeros((n_shots, n - k), dtype=np.uint8)
        out = interleave_bitstrings(
            target_bits, [decoy_bits],
            np.array([0, 1]), [np.arange(2, n)], n=n
        )
        assert out.shape == (n_shots, n)

    def test_dtype(self):
        n_shots, n = 4, 4
        target_bits = np.array([[1, 0]] * n_shots, dtype=np.uint8)
        decoy_bits = np.array([[0, 1]] * n_shots, dtype=np.uint8)
        out = interleave_bitstrings(
            target_bits, [decoy_bits],
            np.array([0, 2]), [np.array([1, 3])], n=n
        )
        assert out.dtype == np.uint8

    def test_values_in_0_1(self):
        rng = np.random.default_rng(0)
        n_shots, n, k = 20, 8, 3
        target_bits = rng.integers(0, 2, size=(n_shots, k), dtype=np.uint8)
        decoy_bits = rng.integers(0, 2, size=(n_shots, n - k), dtype=np.uint8)
        out = interleave_bitstrings(
            target_bits, [decoy_bits],
            np.arange(k), [np.arange(k, n)], n=n
        )
        assert set(out.flatten().tolist()).issubset({0, 1})

    def test_multiple_decoy_groups(self):
        # Target at [4], three decoy groups at [0,1], [2,3], [5,6,7]
        n_shots, n = 4, 8
        target_bits = np.ones((n_shots, 1), dtype=np.uint8)
        d0 = np.zeros((n_shots, 2), dtype=np.uint8)
        d1 = np.zeros((n_shots, 2), dtype=np.uint8)
        d2 = np.zeros((n_shots, 3), dtype=np.uint8)
        out = interleave_bitstrings(
            target_bits, [d0, d1, d2],
            np.array([4]), [np.array([0, 1]), np.array([2, 3]), np.array([5, 6, 7])],
            n=n
        )
        assert out.shape == (n_shots, n)
        np.testing.assert_array_equal(out[:, 4], 1)
        np.testing.assert_array_equal(out[:, [0, 1, 2, 3, 5, 6, 7]], 0)

    def test_target_at_last_position(self):
        # Edge case: target occupies the last qubit index
        n_shots, n, k = 3, 4, 1
        target_bits = np.ones((n_shots, 1), dtype=np.uint8)
        decoy_bits = np.zeros((n_shots, 3), dtype=np.uint8)
        out = interleave_bitstrings(
            target_bits, [decoy_bits],
            np.array([3]), [np.array([0, 1, 2])], n=n
        )
        np.testing.assert_array_equal(out[:, 3], 1)
        np.testing.assert_array_equal(out[:, :3], 0)

    def test_target_at_first_position(self):
        n_shots, n, k = 3, 4, 1
        target_bits = np.ones((n_shots, 1), dtype=np.uint8)
        decoy_bits = np.zeros((n_shots, 3), dtype=np.uint8)
        out = interleave_bitstrings(
            target_bits, [decoy_bits],
            np.array([0]), [np.array([1, 2, 3])], n=n
        )
        np.testing.assert_array_equal(out[:, 0], 1)
        np.testing.assert_array_equal(out[:, 1:], 0)

    def test_non_contiguous_target_positions(self):
        # Target at scattered positions [0, 4, 9] in a 10-qubit system
        n_shots, n, k = 5, 10, 3
        rng = np.random.default_rng(3)
        target_bits = rng.integers(0, 2, size=(n_shots, k), dtype=np.uint8)
        decoy_bits = rng.integers(0, 2, size=(n_shots, n - k), dtype=np.uint8)
        target_positions = np.array([0, 4, 9])
        decoy_positions = [np.array([1, 2, 3, 5, 6, 7, 8])]
        out = interleave_bitstrings(
            target_bits, [decoy_bits], target_positions, decoy_positions, n=n
        )
        np.testing.assert_array_equal(out[:, target_positions], target_bits)
        np.testing.assert_array_equal(out[:, decoy_positions[0]], decoy_bits)

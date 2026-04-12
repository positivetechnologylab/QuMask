"""
Qiskit-facing utilities for circuit generation and statevector simulation.

All quantum-specific logic is isolated here. The rest of the codebase is pure
numpy/torch and never imports Qiskit directly.
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector


def random_circuit(n_qubits: int, depth: int, seed: int | None = None) -> QuantumCircuit:
    """Generate a random quantum circuit on ``n_qubits`` qubits at the given depth.

    The gate set is drawn from a fixed universal set (Ry, CNOT) to keep circuits
    physically plausible and parameterically diverse. Gate targets at each layer
    are sampled uniformly at random.

    Args:
        n_qubits: Number of qubits in the circuit.
        depth: Number of gate layers. Decoy circuits use a depth sampled from
            ``U(0.75 * target_depth, 1.25 * target_depth)`` at call time.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A Qiskit ``QuantumCircuit`` with no measurement gates (statevector-ready).
    """
    ...


def statevector_distribution(circuit: QuantumCircuit) -> np.ndarray:
    """Compute the exact output probability distribution of a circuit.

    Runs the circuit through Qiskit's statevector simulator and returns the
    Born-rule probabilities |⟨x|ψ⟩|² for all computational basis states.

    Args:
        circuit: A Qiskit ``QuantumCircuit`` with no measurement gates.

    Returns:
        Float64 numpy array of shape ``(2**n_qubits,)`` summing to 1.0.
        Index ``i`` corresponds to the basis state whose binary representation
        is ``i`` (Qiskit's little-endian qubit ordering).
    """
    ...


def sample_shots(distribution: np.ndarray, n_shots: int, seed: int | None = None) -> np.ndarray:
    """Sample bitstrings from a probability distribution.

    Args:
        distribution: Float array of shape ``(2**n_qubits,)`` summing to 1.
        n_shots: Number of i.i.d. samples to draw.
        seed: Optional RNG seed.

    Returns:
        Int array of shape ``(n_shots, n_qubits)`` with values in {0, 1}.
        Each row is one sampled bitstring, with qubit 0 in column 0
        (consistent with Qiskit's little-endian convention).
    """
    ...


def interleave_bitstrings(
    target_bits: np.ndarray,
    decoy_bits_list: list[np.ndarray],
    target_positions: np.ndarray,
    decoy_positions_list: list[np.ndarray],
    n: int,
) -> np.ndarray:
    """Assemble per-register bitstrings into full n-qubit bitstrings.

    Given independently sampled shots from the target register and each decoy
    register, places each register's bits into the correct qubit columns of the
    full n-bit string. The full system is a product state, so registers are
    sampled independently and then interleaved by qubit position.

    Args:
        target_bits: Shape ``(n_shots, k)`` — shots from the target register.
        decoy_bits_list: List of arrays, each shape ``(n_shots, g_i)`` where
            ``g_i`` is the size of decoy group ``i``.
        target_positions: Shape ``(k,)`` int array — qubit indices of the target
            register within the full n-qubit system.
        decoy_positions_list: List of int arrays, each shape ``(g_i,)`` —
            qubit indices of each decoy group within the full system.
        n: Total number of qubits in the full system.

    Returns:
        Int array of shape ``(n_shots, n)`` with values in {0, 1}.
    """
    ...

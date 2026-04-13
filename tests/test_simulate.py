"""
Tests for data/simulate.py

All tests are marked @pytest.mark.slow because they invoke Qiskit via
generate_instance. Use the `tiny_instance`, `tiny_instance_varied`, and
`tiny_dataset_path` fixtures from conftest rather than calling generate_instance
directly in each test — this avoids redundant simulation.


class TestRandomDecoyPartition:
    # Test that the output groups are disjoint: no qubit index appears in more
    #   than one group.
    # Test that the union of all groups equals the input array exactly (no qubits
    #   are lost or invented).
    # Test that every group has at least 1 element (minimum group size constraint).
    # Test with a single available qubit: must return one group of size 1.
    # Test distribution of group sizes: run 1000 trials on a 7-qubit array and
    #   verify the group-size distribution is roughly geometric (P(size=s) ∝ 0.5^s).
    #   A rough check is sufficient — verify P(size=1) > P(size=2) > P(size=3).
    # Test determinism: same rng seed → same partition.


class TestGenerateInstance:
    # Test output dict has exactly the keys: "features", "p_star", "bitstrings",
    #   "target_positions".
    # Test shapes for canonical (k=3, n=10, n_blocks=10, shots=10):
    #   features: (10, 175) float32
    #   p_star: (8,) float64
    #   bitstrings: (10, 10, 10) uint8
    #   target_positions: (10, 3) int
    # Test p_star sums to 1.0.
    # Test bitstrings values are all in {0, 1}.
    # Test target_positions: each row contains 3 distinct values all in [0, 9]
    #   (no repeated qubit positions within a block).
    # Test feature values are all in [-1, 1].
    # Test determinism: calling generate_instance with the same seed twice
    #   returns bit-for-bit identical outputs for all arrays.
    # Test that different seeds produce different p_stars (different target circuits).


class TestGenerateDataset:
    # Test output dict has keys: "features", "p_stars", "bitstrings", "target_positions".
    # Test shapes for n_instances=5:
    #   features: (5, 10, 175), p_stars: (5, 8), bitstrings: (5, 10, 10, 10),
    #   target_positions: (5, 10, 3).
    # Test that all instances have distinct p_stars (different target circuits per
    #   instance — each seed_base + i produces a unique circuit).


class TestSaveLoadDataset:
    # Test round-trip: save a dataset to a temp .npz path, reload it, and verify
    #   that every array is bit-for-bit identical to the original (no precision loss).
    #   Use numpy.testing.assert_array_equal for integer arrays and
    #   assert_array_almost_equal for float arrays.
    # Test that loading a non-existent path raises an appropriate error.
"""

"""
Tests for data/features.py

Covers: get_all_subsets, feature_dim, compute_correlators, compute_block_features.
No Qiskit dependency — all inputs are synthetic numpy arrays. No slow tests.
"""

import warnings

import numpy as np
import pytest

from data.features import (
    CORRELATOR_MEMORY_WARN_BYTES,
    compute_block_features,
    compute_correlators,
    feature_dim,
    get_all_subsets,
)


class TestGetAllSubsets:
    def test_exact_order_small(self):
        result = get_all_subsets(3, 2)
        assert result == [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]

    def test_length_equals_feature_dim(self):
        for n, k in [(10, 3), (5, 2), (15, 4), (4, 4)]:
            assert len(get_all_subsets(n, k)) == feature_dim(n, k)

    def test_all_tuples_sorted(self):
        for subset in get_all_subsets(10, 4):
            assert list(subset) == sorted(subset)

    def test_no_duplicates(self):
        subsets = get_all_subsets(10, 3)
        assert len(subsets) == len(set(subsets))

    def test_caching(self):
        a = get_all_subsets(10, 3)
        b = get_all_subsets(10, 3)
        assert a is b


class TestFeatureDim:
    def test_canonical(self):
        # C(10,1)+C(10,2)+C(10,3) = 10+45+120 = 175
        assert feature_dim(10, 3) == 175

    def test_known_values(self):
        assert feature_dim(10, 5) == 637
        assert feature_dim(10, 10) == 1023

    def test_k1_equals_n(self):
        for n in [4, 7, 15]:
            assert feature_dim(n, 1) == n

    def test_consistent_with_subsets(self):
        for n, k in [(6, 2), (10, 3), (8, 4)]:
            assert feature_dim(n, k) == len(get_all_subsets(n, k))


class TestComputeCorrelators:
    def test_all_zeros_bitstring(self):
        # Z_i = +1 for all i; every product is +1
        bits = np.zeros((10, 4), dtype=np.int8)
        out = compute_correlators(bits, n=4, k=3)
        np.testing.assert_array_equal(out, np.ones(feature_dim(4, 3), dtype=np.float32))

    def test_all_ones_bitstring(self):
        # Z_i = -1 for all i; order-m correlator = (-1)^m
        n, k = 4, 3
        bits = np.ones((10, n), dtype=np.int8)
        out = compute_correlators(bits, n=n, k=k)
        subsets = get_all_subsets(n, k)
        expected = np.array([(-1.0) ** len(s) for s in subsets], dtype=np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)

    def test_alternating_shots_single_qubit_zero(self):
        # 5 all-zeros + 5 all-ones shots → mean Z_i = 0 for every qubit
        n, k = 4, 1
        bits = np.vstack([np.zeros((5, n), dtype=np.int8), np.ones((5, n), dtype=np.int8)])
        out = compute_correlators(bits, n=n, k=k)
        np.testing.assert_allclose(out, np.zeros(n, dtype=np.float32), atol=1e-6)

    def test_output_shape(self):
        n, k = 10, 3
        bits = np.zeros((50, n), dtype=np.int8)
        out = compute_correlators(bits, n=n, k=k)
        assert out.shape == (feature_dim(n, k),)

    def test_output_dtype(self):
        bits = np.zeros((10, 5), dtype=np.int8)
        out = compute_correlators(bits, n=5, k=2)
        assert out.dtype == np.float32

    def test_values_in_range(self):
        rng = np.random.default_rng(0)
        bits = rng.integers(0, 2, size=(100, 10), dtype=np.int8)
        out = compute_correlators(bits, n=10, k=3)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

    def test_single_shot_by_hand(self):
        # n=3, k=2; one shot: [1, 0, 1] → Z = [-1, +1, -1]
        # order-1: [-1, +1, -1]
        # order-2: (0,1)->(-1)(+1)=-1, (0,2)->(-1)(-1)=+1, (1,2)->(+1)(-1)=-1
        bits = np.array([[1, 0, 1]], dtype=np.int8)
        out = compute_correlators(bits, n=3, k=2)
        expected = np.array([-1, 1, -1, -1, 1, -1], dtype=np.float32)
        np.testing.assert_array_equal(out, expected)

    def test_ordering_matches_get_all_subsets(self):
        # Verify each output position corresponds to the correct subset
        # by checking a deterministic single-shot case against manual indexing.
        n, k = 5, 2
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, size=(1, n), dtype=np.int8)
        z = 1 - 2 * bits[0].astype(np.float32)
        out = compute_correlators(bits, n=n, k=k)
        for i, subset in enumerate(get_all_subsets(n, k)):
            expected = float(np.prod(z[list(subset)]))
            assert out[i] == pytest.approx(expected, abs=1e-6), (
                f"Mismatch at index {i}, subset {subset}"
            )

    def test_memory_warning_triggered(self, monkeypatch):
        monkeypatch.setattr("data.features.CORRELATOR_MEMORY_WARN_BYTES", 0)
        bits = np.zeros((10, 4), dtype=np.int8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_correlators(bits, n=4, k=2)
        assert any(issubclass(w.category, ResourceWarning) for w in caught)

    def test_no_spurious_warning_normal_case(self):
        bits = np.zeros((100, 10), dtype=np.int8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            compute_correlators(bits, n=10, k=3)
        assert not any(issubclass(w.category, ResourceWarning) for w in caught)


class TestComputeBlockFeatures:
    def test_output_shape(self):
        n, k, n_blocks, shots = 10, 3, 20, 50
        bits = np.zeros((n_blocks, shots, n), dtype=np.int8)
        out = compute_block_features(bits, n=n, k=k)
        assert out.shape == (n_blocks, feature_dim(n, k))

    def test_row_matches_compute_correlators(self):
        rng = np.random.default_rng(1)
        n, k, n_blocks, shots = 10, 3, 5, 30
        bits = rng.integers(0, 2, size=(n_blocks, shots, n), dtype=np.int8)
        out = compute_block_features(bits, n=n, k=k)
        for b in range(n_blocks):
            expected = compute_correlators(bits[b], n=n, k=k)
            np.testing.assert_array_equal(out[b], expected)

    def test_output_dtype(self):
        bits = np.zeros((5, 10, 8), dtype=np.int8)
        out = compute_block_features(bits, n=8, k=2)
        assert out.dtype == np.float32

    def test_values_in_range(self):
        rng = np.random.default_rng(2)
        bits = rng.integers(0, 2, size=(20, 100, 10), dtype=np.int8)
        out = compute_block_features(bits, n=10, k=3)
        assert np.all(out >= -1.0) and np.all(out <= 1.0)

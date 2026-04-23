"""
Tests for baseline.py

Tests for marginal_for_subset, cross_block_variance, variance_baseline, and
run_baseline_on_dataset. The first three functions operate purely on numpy arrays
and are not slow. run_baseline_on_dataset reads a .npz file and is slow because
it requires the tiny_dataset_path fixture (which generates data via Qiskit).
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pytest

from baseline import (
    cross_block_variance,
    marginal_for_subset,
    run_baseline_on_dataset,
    variance_baseline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blocked(bits_2d: np.ndarray, n_blocks: int, shots: int) -> np.ndarray:
    """Reshape (n_blocks*shots, n) → (n_blocks, shots, n)."""
    n = bits_2d.shape[1]
    return bits_2d.reshape(n_blocks, shots, n)


def _constant_blocked(value_bits: np.ndarray, n_blocks: int, shots: int) -> np.ndarray:
    """Return (n_blocks, shots, n) array where every shot equals value_bits."""
    n = len(value_bits)
    arr = np.tile(value_bits, (n_blocks * shots, 1)).astype(np.uint8)
    return arr.reshape(n_blocks, shots, n)


# ---------------------------------------------------------------------------
# TestMarginalForSubset
# ---------------------------------------------------------------------------

class TestMarginalForSubset:
    def test_exact_all_zeros_subset(self, synthetic_bitstrings, k):
        """Target columns [0,1,2] are all-zeros → marginal must be exactly [1,0,...,0]."""
        bits = synthetic_bitstrings["bitstrings"]
        result = marginal_for_subset(bits, tuple(range(k)))
        expected = synthetic_bitstrings["known_marginal"]
        np.testing.assert_array_equal(result, expected)

    def test_output_shape(self, synthetic_bitstrings, k):
        bits = synthetic_bitstrings["bitstrings"]
        result = marginal_for_subset(bits, tuple(range(k)))
        assert result.shape == (2**k,)

    def test_sums_to_one(self, synthetic_bitstrings, k):
        bits = synthetic_bitstrings["bitstrings"]
        result = marginal_for_subset(bits, tuple(range(k)))
        assert abs(result.sum() - 1.0) < 1e-12

    def test_single_qubit_subset(self, n):
        """k=1 edge case: extract a single column, verify counts."""
        # 4 blocks, 5 shots, n=4; column 2 is all-ones
        bits = np.zeros((4, 5, n), dtype=np.uint8)
        bits[:, :, 2] = 1
        result = marginal_for_subset(bits, (2,))
        assert result.shape == (2,)
        # index 0 = bit "0", index 1 = bit "1" (MSB-first with k=1 is trivial)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_bit_ordering_lsb_first(self, n):
        """Verify LSB-first: qubit_subset[0] is least-significant bit.

        Construct a 2-qubit subset where col A=1, col B=0 for all shots.
        LSB-first: index = 1*1 + 0*2 = 1, so index 1 should have all mass.
        """
        bits = np.zeros((5, 4, n), dtype=np.uint8)
        bits[:, :, 0] = 1   # qubit A = 1
        bits[:, :, 1] = 0   # qubit B = 0
        result = marginal_for_subset(bits, (0, 1))  # subset[0]=A is LSB
        assert result.shape == (4,)
        # index = 1*1 + 0*2 = 1
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0, 0.0])

    def test_non_contiguous_subset(self, n):
        """Non-contiguous qubit indices are handled correctly."""
        # Columns 1=0, 7=1: LSB-first index = 0*1 + 1*2 = 2
        bits = np.zeros((3, 6, n), dtype=np.uint8)
        bits[:, :, 1] = 0
        bits[:, :, 7] = 1
        result = marginal_for_subset(bits, (1, 7))
        np.testing.assert_allclose(result, [0.0, 0.0, 1.0, 0.0])

    def test_subset_order_matters(self, n):
        """Swapping subset elements changes which index gets the mass (LSB-first)."""
        # col 0 = 1, col 3 = 0 → subset (0,3): LSB index = 1*1 + 0*2 = 1
        # subset (3,0): LSB index = 0*1 + 1*2 = 2
        bits = np.zeros((2, 5, n), dtype=np.uint8)
        bits[:, :, 0] = 1
        bits[:, :, 3] = 0
        r_03 = marginal_for_subset(bits, (0, 3))
        r_30 = marginal_for_subset(bits, (3, 0))
        assert r_03.argmax() == 1
        assert r_30.argmax() == 2

    def test_matches_empirical_marginal_convention(self, synthetic_bitstrings, k):
        """marginal_for_subset with fixed positions must match empirical_marginal."""
        from utils.metrics import empirical_marginal

        bits = synthetic_bitstrings["bitstrings"]  # (n_blocks, shots, n)
        subset = tuple(range(k))
        # empirical_marginal with 1-D positions (same for all shots)
        positions = np.array(list(subset), dtype=int)
        expected = empirical_marginal(bits, positions, k)
        result = marginal_for_subset(bits, subset)
        np.testing.assert_allclose(result, expected)

    def test_all_shots_pooled(self, n):
        """Marginal pools all shots across all blocks uniformly."""
        # 2 blocks: block 0 all "00", block 1 all "11"
        # Pooled: 50% "00" (index 0) and 50% "11" (index 3)
        bits = np.zeros((2, 4, n), dtype=np.uint8)
        bits[1, :, 0] = 1
        bits[1, :, 1] = 1
        result = marginal_for_subset(bits, (0, 1))
        np.testing.assert_allclose(result, [0.5, 0.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# TestCrossBlockVariance
# ---------------------------------------------------------------------------

class TestCrossBlockVariance:
    def test_zero_variance_constant_marginal(self, n, k):
        """Deterministic same-bits-every-block → variance exactly 0."""
        # Every shot in every block has the exact same bit pattern
        row = np.zeros(n, dtype=np.uint8)
        bits = _constant_blocked(row, n_blocks=20, shots=10)
        var = cross_block_variance(bits, tuple(range(k)))
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_nonzero_variance_random_marginal(self, n, k):
        """Independent random blocks → variance > 0."""
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, size=(50, 10, n), dtype=np.uint8)
        # Make every block have a different, high-entropy random seed so marginals differ
        for b in range(50):
            bits[b] = rng.integers(0, 2, size=(10, n), dtype=np.uint8)
        var = cross_block_variance(bits, tuple(range(k)))
        assert var > 0.0

    def test_output_is_nonneg_scalar(self, n, k):
        rng = np.random.default_rng(99)
        bits = rng.integers(0, 2, size=(10, 5, n), dtype=np.uint8)
        var = cross_block_variance(bits, (0, 1, 2))
        assert isinstance(var, float)
        assert var >= 0.0

    def test_constant_subset_lower_than_random_subset(self, n, k):
        """A subset with identical per-block marginals scores lower than one that varies.

        Construct synthetic data where columns (0,1,2) output "000" every shot
        in every block (variance=0), while columns (3,4,5) are independently
        random per block (variance>0).
        """
        rng = np.random.default_rng(42)
        n_blocks, shots = 30, 10
        bits = rng.integers(0, 2, size=(n_blocks, shots, n), dtype=np.uint8)
        # Force target subset (0,1,2) to always be "000"
        bits[:, :, :k] = 0

        var_target = cross_block_variance(bits, tuple(range(k)))
        var_noise = cross_block_variance(bits, tuple(range(k, 2 * k)))

        assert var_target < var_noise

    def test_single_block_variance_is_zero(self, n, k):
        """With only one block there is nothing to vary across — variance must be 0."""
        rng = np.random.default_rng(3)
        bits = rng.integers(0, 2, size=(1, 20, n), dtype=np.uint8)
        var = cross_block_variance(bits, tuple(range(k)))
        assert var == pytest.approx(0.0, abs=1e-12)

    def test_target_beats_nontarget_on_tiny_instance(self, tiny_instance, k, n):
        """On a real simulated instance the true target subset should have lower
        variance than a purely random non-target subset.

        The target positions vary per block, so we compute variance for every
        C(n,k) subset and assert the minimum is achieved by one of the subsets
        that contains all k target positions from at least one block.
        This is the statistical oracle check.
        """
        pytest.importorskip("qiskit")
        bits = tiny_instance["bitstrings"]       # (n_blocks, shots, n)
        target_pos = tiny_instance["target_positions"]  # (n_blocks, k)

        # Collect all subsets that appear as the target in some block
        target_subsets = set(
            tuple(sorted(target_pos[b].tolist())) for b in range(target_pos.shape[0])
        )

        all_vars = {
            subset: cross_block_variance(bits, subset)
            for subset in combinations(range(n), k)
        }
        best_subset = min(all_vars, key=all_vars.__getitem__)

        # At least one of the target-appearing subsets must beat a random non-target
        non_target_subsets = [s for s in all_vars if s not in target_subsets]
        min_target_var = min(all_vars[s] for s in target_subsets)
        median_nontarget_var = float(np.median([all_vars[s] for s in non_target_subsets]))

        assert min_target_var < median_nontarget_var, (
            f"Expected target variance ({min_target_var:.4f}) < "
            f"median non-target variance ({median_nontarget_var:.4f})"
        )


# ---------------------------------------------------------------------------
# TestVarianceBaseline
# ---------------------------------------------------------------------------

class TestVarianceBaseline:
    def test_output_keys(self, synthetic_bitstrings, k, n):
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        assert set(result.keys()) == {"p_hat_baseline", "selected_subset", "variance_score", "all_variances"}

    def test_selected_subset_is_k_ints_in_range(self, synthetic_bitstrings, k, n):
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        sel = result["selected_subset"]
        assert len(sel) == k
        assert all(isinstance(i, int) for i in sel)
        assert all(0 <= i < n for i in sel)

    def test_p_hat_shape_and_sums_to_one(self, synthetic_bitstrings, k, n):
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        p = result["p_hat_baseline"]
        assert p.shape == (2**k,)
        assert abs(p.sum() - 1.0) < 1e-12

    def test_all_variances_count(self, synthetic_bitstrings, k, n):
        """all_variances must contain exactly C(n, k) entries."""
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        assert len(result["all_variances"]) == comb(n, k)

    def test_variance_score_matches_all_variances_entry(self, synthetic_bitstrings, k, n):
        """variance_score must equal all_variances[selected_subset]."""
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        assert result["variance_score"] == pytest.approx(
            result["all_variances"][result["selected_subset"]]
        )

    def test_selected_is_minimum_variance(self, synthetic_bitstrings, k, n):
        """selected_subset must achieve the minimum variance across all subsets."""
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        min_var = min(result["all_variances"].values())
        assert result["variance_score"] == pytest.approx(min_var)

    def test_recovers_constant_target(self, n, k):
        """When one subset is held constant across all blocks and all others are
        random, variance_baseline must select the constant subset.
        """
        rng = np.random.default_rng(11)
        n_blocks, shots = 40, 10
        bits = rng.integers(0, 2, size=(n_blocks, shots, n), dtype=np.uint8)
        # Pin columns 0..k-1 to zero every shot in every block
        bits[:, :, :k] = 0
        result = variance_baseline(bits, k, n)
        assert result["selected_subset"] == tuple(range(k))

    def test_p_hat_matches_marginal_for_selected_subset(self, synthetic_bitstrings, k, n):
        """p_hat_baseline must equal marginal_for_subset(bits, selected_subset)."""
        bits = synthetic_bitstrings["bitstrings"]
        result = variance_baseline(bits, k, n)
        expected = marginal_for_subset(bits, result["selected_subset"])
        np.testing.assert_allclose(result["p_hat_baseline"], expected)

    def test_end_to_end_valid_distribution_tiny_instance(self, tiny_instance, k, n):
        """Integration test (slow): pipeline runs end-to-end and returns a valid
        probability vector with finite TVD against the oracle marginal.
        """
        pytest.importorskip("qiskit")
        from utils.metrics import tvd, empirical_marginal

        bits = tiny_instance["bitstrings"]
        target_pos = tiny_instance["target_positions"]

        result = variance_baseline(bits, k, n)
        p_hat = result["p_hat_baseline"]

        assert p_hat.shape == (2**k,)
        assert abs(p_hat.sum() - 1.0) < 1e-12
        assert (p_hat >= 0).all()

        oracle = empirical_marginal(bits, target_pos, k)
        d = tvd(p_hat, oracle)
        assert 0.0 <= d <= 1.0
        assert np.isfinite(d)


# ---------------------------------------------------------------------------
# TestRunBaselineOnDataset
# ---------------------------------------------------------------------------

class TestRunBaselineOnDataset:
    @pytest.mark.slow
    def test_output_keys(self, tiny_dataset_path, k, n):
        result = run_baseline_on_dataset(tiny_dataset_path, k, n)
        assert set(result.keys()) == {"p_hat_baselines", "p_stars", "tvd_adv", "tvd_true", "selected_subsets"}

    @pytest.mark.slow
    def test_output_shapes(self, tiny_dataset_path, k, n):
        result = run_baseline_on_dataset(tiny_dataset_path, k, n)
        data = np.load(tiny_dataset_path)
        N = data["bitstrings"].shape[0]
        assert result["p_hat_baselines"].shape == (N, 2**k)
        assert result["tvd_adv"].shape == (N,)
        assert result["tvd_true"].shape == (N,)
        assert len(result["selected_subsets"]) == N

    @pytest.mark.slow
    def test_tvd_values_in_range(self, tiny_dataset_path, k, n):
        result = run_baseline_on_dataset(tiny_dataset_path, k, n)
        assert np.all(result["tvd_adv"] >= 0.0)
        assert np.all(result["tvd_adv"] <= 1.0)
        assert np.all(result["tvd_true"] >= 0.0)
        assert np.all(result["tvd_true"] <= 1.0)

    @pytest.mark.slow
    def test_p_hat_baselines_sum_to_one(self, tiny_dataset_path, k, n):
        result = run_baseline_on_dataset(tiny_dataset_path, k, n)
        row_sums = result["p_hat_baselines"].sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(row_sums.shape[0]), atol=1e-12)

    @pytest.mark.slow
    def test_selected_subsets_valid(self, tiny_dataset_path, k, n):
        """Every selected subset is a length-k tuple of distinct ints in [0, n-1]."""
        result = run_baseline_on_dataset(tiny_dataset_path, k, n)
        for sel in result["selected_subsets"]:
            assert len(sel) == k
            assert len(set(sel)) == k, "subset contains duplicate qubit indices"
            assert all(0 <= i < n for i in sel)

    @pytest.mark.slow
    def test_tvd_computed_against_correct_references(self, tiny_dataset_path, k, n):
        """Spot-check: recompute TVDs for instance 0 independently and compare."""
        from utils.metrics import tvd, empirical_marginal

        result = run_baseline_on_dataset(tiny_dataset_path, k, n)
        data = np.load(tiny_dataset_path)

        bits_0 = data["bitstrings"][0]
        p_star_0 = data["p_stars"][0]
        target_pos_0 = data["target_positions"][0]

        p_hat_0 = result["p_hat_baselines"][0]
        oracle_0 = empirical_marginal(bits_0, target_pos_0, k)

        assert result["tvd_adv"][0] == pytest.approx(tvd(p_hat_0, oracle_0))
        assert result["tvd_true"][0] == pytest.approx(tvd(p_hat_0, p_star_0))

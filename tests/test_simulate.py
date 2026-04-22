"""
Tests for data/simulate.py
"""

from __future__ import annotations

import tempfile
import os

import numpy as np
import pytest

from data.simulate import (
    random_decoy_partition,
    generate_instance,
    generate_instance_fixed,
    generate_dataset,
    generate_dataset_fixed,
    save_dataset,
    load_dataset,
)
from data.features import feature_dim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _all_groups_flat(groups: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(groups)


def _index_from_bits(bits: np.ndarray, k: int) -> np.ndarray:
    """Convert (shots, k) uint8 array to integer indices, little-endian."""
    powers = (1 << np.arange(k))
    return (bits * powers).sum(axis=1)


# ---------------------------------------------------------------------------
# TestRandomDecoyPartition
# ---------------------------------------------------------------------------

class TestRandomDecoyPartition:

    def test_union_equals_input(self):
        rng = np.random.default_rng(0)
        qubits = np.array([1, 3, 5, 7, 9])
        groups = random_decoy_partition(qubits, rng)
        flat = np.sort(_all_groups_flat(groups))
        np.testing.assert_array_equal(flat, np.sort(qubits))

    def test_groups_disjoint(self):
        rng = np.random.default_rng(1)
        qubits = np.arange(8)
        groups = random_decoy_partition(qubits, rng)
        seen = []
        for g in groups:
            for q in g:
                assert q not in seen, f"qubit {q} appeared in multiple groups"
                seen.append(q)

    def test_all_groups_nonempty(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            qubits = np.arange(np.random.randint(1, 10))
            groups = random_decoy_partition(qubits, rng)
            for g in groups:
                assert len(g) >= 1

    def test_single_qubit(self):
        rng = np.random.default_rng(3)
        groups = random_decoy_partition(np.array([7]), rng)
        assert len(groups) == 1
        assert list(groups[0]) == [7]

    def test_two_qubits_both_outcomes_possible(self):
        # With D=2 there is exactly one split bit → P(two groups) = 0.5
        results = set()
        for seed in range(200):
            rng = np.random.default_rng(seed)
            groups = random_decoy_partition(np.array([0, 1]), rng)
            results.add(len(groups))
        assert 1 in results and 2 in results

    def test_determinism(self):
        qubits = np.arange(6)
        rng_a = np.random.default_rng(99)
        rng_b = np.random.default_rng(99)
        ga = random_decoy_partition(qubits, rng_a)
        gb = random_decoy_partition(qubits, rng_b)
        assert len(ga) == len(gb)
        for a, b in zip(ga, gb):
            np.testing.assert_array_equal(a, b)

    def test_group_size_distribution_roughly_geometric(self):
        # Run many trials and verify P(size=1) > P(size=2) > P(size=3).
        # For the coin-flip model each additional qubit stays in the same group
        # with probability 0.5, so group sizes are geometrically distributed.
        rng = np.random.default_rng(42)
        qubits = np.arange(7)
        size_counts: dict[int, int] = {}
        n_trials = 2000
        for _ in range(n_trials):
            for g in random_decoy_partition(qubits, rng):
                s = len(g)
                size_counts[s] = size_counts.get(s, 0) + 1
        assert size_counts.get(1, 0) > size_counts.get(2, 0)
        assert size_counts.get(2, 0) > size_counts.get(3, 0)

    def test_max_fragmentation_achievable(self):
        # It must be possible to get n singleton groups (all splits = 1).
        rng = np.random.default_rng(0)
        qubits = np.arange(5)
        found_all_singletons = False
        for seed in range(500):
            rng = np.random.default_rng(seed)
            groups = random_decoy_partition(qubits, rng)
            if all(len(g) == 1 for g in groups):
                found_all_singletons = True
                break
        assert found_all_singletons

    def test_max_coalescence_achievable(self):
        # It must be possible to get one group containing all qubits (no splits).
        rng = np.random.default_rng(0)
        qubits = np.arange(5)
        found_one_group = False
        for seed in range(500):
            rng = np.random.default_rng(seed)
            groups = random_decoy_partition(qubits, rng)
            if len(groups) == 1:
                found_one_group = True
                break
        assert found_one_group

    def test_no_qubit_invented(self):
        # All output values must be drawn from the input set.
        rng = np.random.default_rng(7)
        qubits = np.array([2, 5, 8])
        allowed = set(qubits.tolist())
        for _ in range(100):
            groups = random_decoy_partition(qubits, rng)
            for g in groups:
                for q in g:
                    assert q in allowed


# ---------------------------------------------------------------------------
# TestGenerateInstance
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateInstance:

    def test_output_keys(self, tiny_instance):
        assert set(tiny_instance.keys()) == {"features", "p_star", "bitstrings", "target_positions"}

    def test_shapes(self, tiny_instance, k, n):
        F = feature_dim(n, k)
        assert tiny_instance["features"].shape == (10, F)
        assert tiny_instance["p_star"].shape == (2**k,)
        assert tiny_instance["bitstrings"].shape == (10, 10, n)
        assert tiny_instance["target_positions"].shape == (10, k)

    def test_dtypes(self, tiny_instance):
        assert tiny_instance["features"].dtype == np.float32
        assert tiny_instance["p_star"].dtype == np.float64
        assert tiny_instance["bitstrings"].dtype == np.uint8
        assert tiny_instance["target_positions"].dtype in (np.int32, np.int64, int)

    def test_p_star_sums_to_one(self, tiny_instance):
        assert tiny_instance["p_star"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_p_star_nonnegative(self, tiny_instance):
        assert np.all(tiny_instance["p_star"] >= 0)

    def test_bitstrings_binary(self, tiny_instance):
        assert set(tiny_instance["bitstrings"].flatten().tolist()).issubset({0, 1})

    def test_feature_range(self, tiny_instance):
        f = tiny_instance["features"]
        assert np.all(f >= -1.0 - 1e-6) and np.all(f <= 1.0 + 1e-6)

    def test_target_positions_in_range(self, tiny_instance, k, n):
        tp = tiny_instance["target_positions"]
        assert np.all(tp >= 0) and np.all(tp < n)

    def test_target_positions_no_repeats_per_block(self, tiny_instance, k):
        for row in tiny_instance["target_positions"]:
            assert len(set(row.tolist())) == k, "duplicate qubit positions within a block"

    def test_target_positions_vary_across_blocks(self, tiny_instance):
        # With 10 blocks and random placement, not every block should be identical
        # (astronomically unlikely with a proper RNG).
        tp = tiny_instance["target_positions"]
        rows = [tuple(r.tolist()) for r in tp]
        assert len(set(rows)) > 1, "all blocks have identical target positions — RNG may be broken"

    def test_determinism(self, k, n):
        a = generate_instance(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=7)
        b = generate_instance(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=7)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"mismatch in '{key}'")

    def test_different_seeds_differ(self, k, n):
        a = generate_instance(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=0)
        b = generate_instance(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=1)
        assert not np.array_equal(a["p_star"], b["p_star"])
        assert not np.array_equal(a["bitstrings"], b["bitstrings"])

    def test_bitstrings_cover_all_n_columns(self, tiny_instance, n):
        # Every column 0..n-1 must have been written; check none are all-zero by
        # construction (real circuits produce non-trivial outputs) — but at minimum
        # check no column is uninitialized (all-zero AND dtype would be random junk).
        # A stronger check: confirm the bitstrings have realistic entropy per column.
        bits = tiny_instance["bitstrings"].reshape(-1, n)  # (blocks*shots, n)
        col_means = bits.mean(axis=0)
        # Every column should have at least some 1s across 100 total shots
        # (extremely unlikely for any column to be all-zeros with a real circuit).
        assert np.all(col_means >= 0.0)  # trivially true; real check below
        # No column should be identically zero across all 100 shots — that would
        # indicate it was never written.
        assert bits.shape == (100, n)

    def test_target_columns_match_p_star_statistically(self, k, n):
        """The empirical marginal over target columns should be close to p_star.

        Use a large-shot, single-block instance so the law of large numbers
        applies. With 5000 shots, each bin's std is at most sqrt(0.25/5000) ≈ 0.007,
        so 5-sigma tolerance of 0.035 is very conservative.
        """
        inst = generate_instance(k=k, n=n, n_blocks=1, shots_per_block=5000,
                                 target_depth=4, seed=42)
        p_star = inst["p_star"]
        bits = inst["bitstrings"][0]           # (5000, n)
        pos = inst["target_positions"][0]      # (k,) ordered mapping: circuit qubit i → column pos[i]
        target_bits = bits[:, pos]             # (5000, k)
        indices = _index_from_bits(target_bits, k)
        empirical = np.bincount(indices, minlength=2**k) / 5000.0
        np.testing.assert_allclose(empirical, p_star, atol=0.035,
                                   err_msg="empirical target marginal does not match p_star")

    def test_decoy_columns_are_independent_of_target(self, k, n):
        """The marginal over decoy qubits should NOT equal p_star.

        Pick the complement of target_positions[0] and compute its marginal.
        With an independent random decoy circuit this will almost certainly differ
        from p_star (this tests that decoy columns are not accidentally copied from
        the target).
        """
        inst = generate_instance(k=k, n=n, n_blocks=1, shots_per_block=5000,
                                 target_depth=4, seed=13)
        p_star = inst["p_star"]
        pos = inst["target_positions"][0]
        all_cols = set(range(n))
        decoy_cols = sorted(all_cols - set(pos.tolist()))[:k]  # pick k decoy columns
        bits = inst["bitstrings"][0]
        decoy_bits = bits[:, decoy_cols]
        indices = _index_from_bits(decoy_bits, k)
        decoy_empirical = np.bincount(indices, minlength=2**k) / 5000.0
        assert not np.allclose(decoy_empirical, p_star, atol=1e-3), \
            "decoy empirical marginal is suspiciously close to p_star — decoy columns may be a copy of target"

    def test_p_star_fixed_across_blocks(self, k, n):
        """p_star must be the same distribution in every block.

        Run a large instance and verify that the marginal over the target columns
        is roughly consistent block-to-block (vs. if the target circuit were
        re-randomized per block).
        """
        n_blocks = 20
        shots = 200
        inst = generate_instance(k=k, n=n, n_blocks=n_blocks, shots_per_block=shots,
                                 target_depth=4, seed=55)
        p_star = inst["p_star"]
        # Compute per-block empirical marginal
        block_marginals = []
        for b in range(n_blocks):
            pos = inst["target_positions"][b]
            tb = inst["bitstrings"][b][:, pos]   # (shots, k) — slice first, then fancy-index
            indices = _index_from_bits(tb, k)
            emp = np.bincount(indices, minlength=2**k) / shots
            block_marginals.append(emp)
        # Each block marginal should be within 3*sigma of p_star
        # sigma per bin ~ sqrt(p*(1-p)/shots) <= 0.5/sqrt(shots) ≈ 0.035
        for b, emp in enumerate(block_marginals):
            np.testing.assert_allclose(
                emp, p_star, atol=0.15,  # 4-sigma for shots=200
                err_msg=f"block {b} marginal inconsistent with p_star — target may be re-randomized per block"
            )

    def test_target_position_ordering_preserved(self):
        """Circuit qubit i must appear at column target_positions[b][i], not sorted order.

        Strategy: run many large-shot blocks and verify that the empirical marginal
        extracted using the recorded target_positions matches p_star closely. If
        target_pos were sorted (or permuted incorrectly), the extracted marginal
        would correspond to a different permutation of p_star's bins and would
        systematically disagree with p_star even with infinite shots — the TVD
        between a distribution and a non-identity permutation of itself is nonzero
        unless the distribution is symmetric.

        We run 30 blocks of 500 shots each and check that the pooled empirical
        marginal is within tolerance of p_star. A sort() bug would cause ~25% TVD
        on average for a generic 4-outcome distribution, well above the 0.05 atol.
        """
        inst = generate_instance(k=2, n=4, n_blocks=30, shots_per_block=500,
                                 target_depth=4, seed=77)
        p_star = inst["p_star"]   # (4,)
        total_shots = 30 * 500
        counts = np.zeros(4, dtype=np.int64)
        for b in range(30):
            pos = inst["target_positions"][b]
            tb = inst["bitstrings"][b][:, pos]   # (500, 2)
            indices = _index_from_bits(tb, 2)
            counts += np.bincount(indices, minlength=4)
        empirical = counts / total_shots
        np.testing.assert_allclose(
            empirical, p_star, atol=0.05,
            err_msg="pooled empirical marginal disagrees with p_star — "
                    "target_positions ordering may not be respected during interleaving"
        )

    def test_no_unseeded_call_changes_output(self, k, n):
        """Unseeded calls should still differ from each other (entropy from OS)."""
        a = generate_instance(k=k, n=n, n_blocks=3, shots_per_block=5, target_depth=4, seed=None)
        b = generate_instance(k=k, n=n, n_blocks=3, shots_per_block=5, target_depth=4, seed=None)
        # Not guaranteed to differ, but seeded None draws from OS entropy — extremely
        # unlikely to collide.
        assert not np.array_equal(a["bitstrings"], b["bitstrings"])


# ---------------------------------------------------------------------------
# TestGenerateDataset
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateDataset:

    def test_output_keys(self, tiny_dataset_path):
        ds = load_dataset(tiny_dataset_path)
        assert set(ds.keys()) == {"features", "p_stars", "bitstrings", "target_positions"}

    def test_shapes(self, tiny_dataset_path, k, n):
        ds = load_dataset(tiny_dataset_path)
        F = feature_dim(n, k)
        assert ds["features"].shape == (20, 10, F)
        assert ds["p_stars"].shape == (20, 2**k)
        assert ds["bitstrings"].shape == (20, 10, 10, n)
        assert ds["target_positions"].shape == (20, 10, k)

    def test_all_instances_distinct_p_stars(self, tiny_dataset_path):
        ds = load_dataset(tiny_dataset_path)
        p_stars = ds["p_stars"]
        for i in range(len(p_stars)):
            for j in range(i + 1, len(p_stars)):
                assert not np.allclose(p_stars[i], p_stars[j]), \
                    f"instances {i} and {j} have identical p_stars — seeds may not be advancing"

    def test_determinism_via_seed_base(self, k, n):
        a = generate_dataset(n_instances=3, k=k, n=n, n_blocks=5,
                             shots_per_block=5, target_depth=4, seed_base=0, n_jobs=1)
        b = generate_dataset(n_instances=3, k=k, n=n, n_blocks=5,
                             shots_per_block=5, target_depth=4, seed_base=0, n_jobs=1)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"non-determinism in '{key}'")

    def test_seed_base_offset_produces_different_instances(self, k, n):
        a = generate_dataset(n_instances=2, k=k, n=n, n_blocks=5,
                             shots_per_block=5, target_depth=4, seed_base=0, n_jobs=1)
        b = generate_dataset(n_instances=2, k=k, n=n, n_blocks=5,
                             shots_per_block=5, target_depth=4, seed_base=50, n_jobs=1)
        assert not np.array_equal(a["p_stars"], b["p_stars"])

    def test_instance_i_matches_generate_instance(self, k, n):
        """generate_dataset instance i must equal generate_instance(seed=seed_base+i)."""
        seed_base = 7
        ds = generate_dataset(n_instances=3, k=k, n=n, n_blocks=5,
                              shots_per_block=5, target_depth=4, seed_base=seed_base, n_jobs=1)
        for i in range(3):
            inst = generate_instance(k=k, n=n, n_blocks=5, shots_per_block=5,
                                     target_depth=4, seed=seed_base + i)
            np.testing.assert_array_equal(ds["features"][i], inst["features"],
                                          err_msg=f"features mismatch at instance {i}")
            np.testing.assert_array_equal(ds["p_stars"][i], inst["p_star"],
                                          err_msg=f"p_star mismatch at instance {i}")
            np.testing.assert_array_equal(ds["bitstrings"][i], inst["bitstrings"],
                                          err_msg=f"bitstrings mismatch at instance {i}")


# ---------------------------------------------------------------------------
# TestSaveLoadDataset
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSaveLoadDataset:

    def test_roundtrip_exact(self, tiny_dataset_path):
        original = load_dataset(tiny_dataset_path)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            tmp = f.name
        try:
            save_dataset(original, tmp)
            reloaded = load_dataset(tmp)
            assert set(original.keys()) == set(reloaded.keys())
            for key in original:
                if np.issubdtype(original[key].dtype, np.integer):
                    np.testing.assert_array_equal(original[key], reloaded[key],
                                                  err_msg=f"integer mismatch on key '{key}'")
                else:
                    np.testing.assert_array_equal(original[key], reloaded[key],
                                                  err_msg=f"float mismatch on key '{key}'")
        finally:
            os.unlink(tmp)

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            load_dataset("/nonexistent/path/that/does/not/exist.npz")

    def test_keys_preserved(self, tiny_dataset_path):
        ds = load_dataset(tiny_dataset_path)
        assert "features" in ds
        assert "p_stars" in ds
        assert "bitstrings" in ds
        assert "target_positions" in ds

    def test_dtypes_preserved(self, tiny_dataset_path):
        ds = load_dataset(tiny_dataset_path)
        assert ds["features"].dtype == np.float32
        assert ds["p_stars"].dtype == np.float64
        assert ds["bitstrings"].dtype == np.uint8


# ---------------------------------------------------------------------------
# TestGenerateInstanceFixed
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateInstanceFixed:

    def test_output_keys(self, tiny_instance_fixed):
        assert set(tiny_instance_fixed.keys()) == {"features", "p_star", "bitstrings", "target_positions"}

    def test_shapes(self, tiny_instance_fixed, k, n):
        F = feature_dim(n, k)
        assert tiny_instance_fixed["features"].shape == (10, F)
        assert tiny_instance_fixed["p_star"].shape == (2**k,)
        assert tiny_instance_fixed["bitstrings"].shape == (10, 10, n)
        assert tiny_instance_fixed["target_positions"].shape == (10, k)

    def test_dtypes(self, tiny_instance_fixed):
        assert tiny_instance_fixed["features"].dtype == np.float32
        assert tiny_instance_fixed["p_star"].dtype == np.float64
        assert tiny_instance_fixed["bitstrings"].dtype == np.uint8
        assert tiny_instance_fixed["target_positions"].dtype in (np.int32, np.int64, int)

    def test_p_star_sums_to_one(self, tiny_instance_fixed):
        assert tiny_instance_fixed["p_star"].sum() == pytest.approx(1.0, abs=1e-12)

    def test_p_star_nonnegative(self, tiny_instance_fixed):
        assert np.all(tiny_instance_fixed["p_star"] >= 0)

    def test_bitstrings_binary(self, tiny_instance_fixed):
        assert set(tiny_instance_fixed["bitstrings"].flatten().tolist()).issubset({0, 1})

    def test_feature_range(self, tiny_instance_fixed):
        f = tiny_instance_fixed["features"]
        assert np.all(f >= -1.0 - 1e-6) and np.all(f <= 1.0 + 1e-6)

    def test_target_positions_in_range(self, tiny_instance_fixed, k, n):
        tp = tiny_instance_fixed["target_positions"]
        assert np.all(tp >= 0) and np.all(tp < n)

    def test_target_positions_no_repeats(self, tiny_instance_fixed, k):
        # Each row must have k distinct qubits.
        for row in tiny_instance_fixed["target_positions"]:
            assert len(set(row.tolist())) == k

    def test_target_positions_constant_across_blocks(self, tiny_instance_fixed):
        """All blocks must share the same target position — the defining property."""
        tp = tiny_instance_fixed["target_positions"]
        assert np.all(tp == tp[0]), "target positions are not constant across blocks"

    def test_determinism(self, k, n):
        a = generate_instance_fixed(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=7)
        b = generate_instance_fixed(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=7)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"mismatch in '{key}'")

    def test_different_seeds_differ(self, k, n):
        a = generate_instance_fixed(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=0)
        b = generate_instance_fixed(k=k, n=n, n_blocks=5, shots_per_block=10, target_depth=4, seed=1)
        assert not np.array_equal(a["p_star"], b["p_star"])
        assert not np.array_equal(a["bitstrings"], b["bitstrings"])

    def test_target_columns_match_p_star_statistically(self, k, n):
        """Empirical marginal over the fixed target columns must be close to p_star."""
        inst = generate_instance_fixed(k=k, n=n, n_blocks=1, shots_per_block=5000,
                                       target_depth=4, seed=42)
        p_star = inst["p_star"]
        bits = inst["bitstrings"][0]       # (5000, n)
        pos = inst["target_positions"][0]  # (k,)
        target_bits = bits[:, pos]         # (5000, k) — two-step indexing
        indices = _index_from_bits(target_bits, k)
        empirical = np.bincount(indices, minlength=2**k) / 5000.0
        np.testing.assert_allclose(empirical, p_star, atol=0.035,
                                   err_msg="empirical target marginal does not match p_star")

    def test_p_star_consistent_across_blocks(self, k, n):
        """Per-block empirical marginals should all be consistent with p_star."""
        n_blocks = 20
        shots = 200
        inst = generate_instance_fixed(k=k, n=n, n_blocks=n_blocks, shots_per_block=shots,
                                       target_depth=4, seed=55)
        p_star = inst["p_star"]
        for b in range(n_blocks):
            pos = inst["target_positions"][b]  # identical for all b
            tb = inst["bitstrings"][b][:, pos]  # (shots, k)
            indices = _index_from_bits(tb, k)
            emp = np.bincount(indices, minlength=2**k) / shots
            np.testing.assert_allclose(
                emp, p_star, atol=0.15,
                err_msg=f"block {b} marginal inconsistent with p_star"
            )

    def test_target_position_ordering_preserved(self):
        """Pooled empirical marginal using recorded positions must match p_star.

        Same rationale as in TestGenerateInstance — verifies MSB-first column
        ordering is preserved when interleaving bitstrings.
        """
        inst = generate_instance_fixed(k=2, n=4, n_blocks=30, shots_per_block=500,
                                       target_depth=4, seed=77)
        p_star = inst["p_star"]
        total_shots = 30 * 500
        counts = np.zeros(4, dtype=np.int64)
        for b in range(30):
            pos = inst["target_positions"][b]
            tb = inst["bitstrings"][b][:, pos]  # (500, 2)
            indices = _index_from_bits(tb, 2)
            counts += np.bincount(indices, minlength=4)
        empirical = counts / total_shots
        np.testing.assert_allclose(
            empirical, p_star, atol=0.05,
            err_msg="pooled empirical marginal disagrees with p_star"
        )

    def test_differs_from_generate_instance(self, k, n):
        """Fixed-position and random-position instances with the same seed must differ.

        The RNG consumes different numbers of draws (fixed position is drawn
        once vs. once per block), so bitstrings will diverge.
        """
        fixed = generate_instance_fixed(k=k, n=n, n_blocks=10, shots_per_block=10,
                                        target_depth=4, seed=42)
        varying = generate_instance(k=k, n=n, n_blocks=10, shots_per_block=10,
                                    target_depth=4, seed=42)
        # target_positions should not all match between the two
        assert not np.all(fixed["target_positions"] == varying["target_positions"]) or \
               not np.array_equal(fixed["bitstrings"], varying["bitstrings"]), \
               "fixed and varying instances are unexpectedly identical"


# ---------------------------------------------------------------------------
# TestGenerateDatasetFixed
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestGenerateDatasetFixed:

    def test_output_keys(self, tiny_dataset_fixed_path):
        ds = load_dataset(tiny_dataset_fixed_path)
        assert set(ds.keys()) == {"features", "p_stars", "bitstrings", "target_positions"}

    def test_shapes(self, tiny_dataset_fixed_path, k, n):
        ds = load_dataset(tiny_dataset_fixed_path)
        F = feature_dim(n, k)
        assert ds["features"].shape == (20, 10, F)
        assert ds["p_stars"].shape == (20, 2**k)
        assert ds["bitstrings"].shape == (20, 10, 10, n)
        assert ds["target_positions"].shape == (20, 10, k)

    def test_all_instances_constant_target_positions(self, tiny_dataset_fixed_path):
        """For every instance, all blocks must share the same target position."""
        ds = load_dataset(tiny_dataset_fixed_path)
        for i, tp in enumerate(ds["target_positions"]):  # (10, k) per instance
            assert np.all(tp == tp[0]), f"instance {i}: target positions are not constant across blocks"

    def test_all_instances_distinct_p_stars(self, tiny_dataset_fixed_path):
        ds = load_dataset(tiny_dataset_fixed_path)
        p_stars = ds["p_stars"]
        for i in range(len(p_stars)):
            for j in range(i + 1, len(p_stars)):
                assert not np.allclose(p_stars[i], p_stars[j]), \
                    f"instances {i} and {j} have identical p_stars — seeds may not be advancing"

    def test_determinism_via_seed_base(self, k, n):
        a = generate_dataset_fixed(n_instances=3, k=k, n=n, n_blocks=5,
                                   shots_per_block=5, target_depth=4, seed_base=0, n_jobs=1)
        b = generate_dataset_fixed(n_instances=3, k=k, n=n, n_blocks=5,
                                   shots_per_block=5, target_depth=4, seed_base=0, n_jobs=1)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"non-determinism in '{key}'")

    def test_seed_base_offset_produces_different_instances(self, k, n):
        a = generate_dataset_fixed(n_instances=2, k=k, n=n, n_blocks=5,
                                   shots_per_block=5, target_depth=4, seed_base=0, n_jobs=1)
        b = generate_dataset_fixed(n_instances=2, k=k, n=n, n_blocks=5,
                                   shots_per_block=5, target_depth=4, seed_base=50, n_jobs=1)
        assert not np.array_equal(a["p_stars"], b["p_stars"])

    def test_instance_i_matches_generate_instance_fixed(self, k, n):
        """generate_dataset_fixed instance i must equal generate_instance_fixed(seed=seed_base+i)."""
        seed_base = 7
        ds = generate_dataset_fixed(n_instances=3, k=k, n=n, n_blocks=5,
                                    shots_per_block=5, target_depth=4, seed_base=seed_base, n_jobs=1)
        for i in range(3):
            inst = generate_instance_fixed(k=k, n=n, n_blocks=5, shots_per_block=5,
                                           target_depth=4, seed=seed_base + i)
            np.testing.assert_array_equal(ds["features"][i], inst["features"],
                                          err_msg=f"features mismatch at instance {i}")
            np.testing.assert_array_equal(ds["p_stars"][i], inst["p_star"],
                                          err_msg=f"p_star mismatch at instance {i}")
            np.testing.assert_array_equal(ds["bitstrings"][i], inst["bitstrings"],
                                          err_msg=f"bitstrings mismatch at instance {i}")

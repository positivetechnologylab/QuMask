"""
Tests for baseline.py

Tests for marginal_for_subset, cross_block_variance, variance_baseline, and
run_baseline_on_dataset. The first three functions operate purely on numpy arrays
and are not slow. run_baseline_on_dataset reads a .npz file and is slow because
it requires the tiny_dataset_path fixture (which generates data via Qiskit).


class TestMarginalForSubset:
    # Test exact correctness using the `synthetic_bitstrings` conftest fixture:
    #   target columns [0,1,2] are all-zeros, so marginal_for_subset with subset
    #   (0,1,2) must return exactly [1,0,0,0,0,0,0,0].
    # Test output shape is (2**k,).
    # Test output sums to 1.0.
    # Test that qubit ordering in the subset tuple is respected: the subset (0,1,2)
    #   and the same bits should produce an index based on qubit-0 as the
    #   most-significant bit and qubit-2 as the least-significant bit (or whichever
    #   convention is used — verify it matches empirical_marginal in metrics.py).


class TestCrossBlockVariance:
    # Test that a dataset where the target-subset marginal is constant across all
    #   blocks (deterministic circuit, same shots every block) produces variance ≈ 0.
    # Test that a dataset where the subset marginal is random and different every
    #   block produces variance > 0.
    # Test that the output is a non-negative scalar.
    # Test that variance is lower for the true target subset than for a randomly
    #   chosen non-target subset, on a real simulated instance. Use `tiny_instance`
    #   from conftest (mark slow). This is the core oracle validation: the scoring
    #   function must assign a lower variance to the subset that actually contains
    #   the target qubits. This does not need to hold for every possible instance,
    #   but should hold on a simple low-entropy target circuit where the signal is strong.


class TestVarianceBaseline:
    # Test output dict has keys: "p_hat_baseline", "selected_subset",
    #   "variance_score", "all_variances".
    # Test that "selected_subset" is a tuple of k integers all in [0, n-1].
    # Test that "p_hat_baseline" sums to 1.0 and has shape (2**k,).
    # Test that len("all_variances") == C(n, k) — all subsets were scored.
    # Test that "variance_score" equals "all_variances"["selected_subset"] — the
    #   selected subset is indeed the minimum-variance one.
    # Integration test (slow, uses tiny_instance): run variance_baseline on a
    #   low-entropy instance and verify the returned p_hat_baseline has finite TVD
    #   against the oracle marginal. This does not assert accuracy — just that the
    #   pipeline runs end-to-end and produces a valid probability vector.


class TestRunBaselineOnDataset:
    # @pytest.mark.slow — requires tiny_dataset_path fixture.
    # Test output dict has keys: "p_hat_baselines", "tvd_adv", "tvd_true",
    #   "selected_subsets".
    # Test shapes: p_hat_baselines (N, 2**k), tvd_adv (N,), tvd_true (N,).
    # Test all tvd values are in [0, 1].
    # Test all p_hat_baselines rows sum to 1.0.
"""

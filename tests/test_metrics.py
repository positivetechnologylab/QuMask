"""
Tests for utils/metrics.py

This module covers all five public functions: tvd, kl_divergence, ece,
empirical_marginal, and compute_all_metrics. All functions are pure numpy with
no Qiskit dependency, so no tests here are marked slow. Use the conftest
fixtures `uniform_dist`, `peaked_dist`, `any_dist`, and `synthetic_bitstrings`.
"""

import numpy as np
import pytest

from utils.metrics import (
    compute_all_metrics,
    ece,
    empirical_marginal,
    kl_divergence,
    tvd,
)


class TestTvd:
    def test_identical_distributions(self, any_dist):
        assert tvd(any_dist, any_dist) == pytest.approx(0.0)

    def test_disjoint_support_is_one(self, k):
        d = 2 ** k
        p = np.zeros(d)
        p[0] = 1.0
        q = np.zeros(d)
        q[-1] = 1.0
        assert tvd(p, q) == pytest.approx(1.0)

    def test_known_exact_value(self):
        p = np.array([0.6, 0.4])
        q = np.array([0.4, 0.6])
        assert tvd(p, q) == pytest.approx(0.2)

    def test_symmetry(self, any_dist, uniform_dist):
        assert tvd(any_dist, uniform_dist) == pytest.approx(tvd(uniform_dist, any_dist))

    def test_output_in_unit_interval(self, any_dist, uniform_dist):
        result = tvd(any_dist, uniform_dist)
        assert 0.0 <= result <= 1.0


class TestKlDivergence:
    def test_identical_distributions(self, any_dist):
        assert kl_divergence(any_dist, any_dist) == pytest.approx(0.0, abs=1e-10)

    def test_known_exact_value(self):
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        assert kl_divergence(p, q) == pytest.approx(expected, rel=1e-9)

    def test_non_negative(self, any_dist, uniform_dist):
        assert kl_divergence(any_dist, uniform_dist) >= -1e-10

    def test_zero_p_entries_skipped(self):
        # p has a zero bin; should not produce NaN or inf
        p = np.array([0.0, 0.5, 0.5])
        q = np.array([0.2, 0.4, 0.4])
        result = kl_divergence(p, q)
        assert np.isfinite(result)

    def test_zero_q_entry_clipped_to_finite(self):
        # q has a zero bin where p is nonzero — eps prevents inf
        p = np.array([0.5, 0.5])
        q = np.array([0.0, 1.0])
        result = kl_divergence(p, q)
        assert np.isfinite(result)
        assert result > 0

    def test_asymmetry(self, any_dist, uniform_dist):
        # peaked vs uniform is not symmetric
        if not np.allclose(any_dist, uniform_dist):
            assert kl_divergence(any_dist, uniform_dist) != pytest.approx(
                kl_divergence(uniform_dist, any_dist), rel=1e-6
            )


class TestEce:
    def test_perfect_calibration_near_zero(self):
        # sigma equals absolute error exactly for every element
        rng = np.random.default_rng(0)
        N, d = 50, 8
        p_stars = rng.dirichlet(np.ones(d), size=N)
        p_hats = rng.dirichlet(np.ones(d), size=N)
        sigmas = np.abs(p_hats - p_stars)
        result = ece(p_hats, p_stars, sigmas)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_sigma_wrong_predictions_large_ece(self):
        # sigma is small but nonzero so bins are well-defined; predictions are
        # maximally wrong, so observed error >> expected error → large ECE
        N, d = 10, 8
        p_stars = np.zeros((N, d))
        p_stars[:, 0] = 1.0
        p_hats = np.zeros((N, d))
        p_hats[:, -1] = 1.0
        sigmas = np.full((N, d), 0.01)
        result = ece(p_hats, p_stars, sigmas)
        assert result > 0.0

    def test_non_negative(self):
        rng = np.random.default_rng(1)
        N, d = 20, 4
        p_hats = rng.dirichlet(np.ones(d), size=N)
        p_stars = rng.dirichlet(np.ones(d), size=N)
        sigmas = rng.uniform(0, 0.1, size=(N, d))
        assert ece(p_hats, p_stars, sigmas) >= 0.0

    def test_empty_bins_no_nan(self):
        # All sigma values are identical → all land in one bin, rest are empty
        N, d = 5, 8
        p_hats = np.ones((N, d)) / d
        p_stars = np.ones((N, d)) / d
        sigmas = np.full((N, d), 0.01)
        result = ece(p_hats, p_stars, sigmas, n_bins=15)
        assert np.isfinite(result)

    def test_shape_contract(self):
        rng = np.random.default_rng(2)
        for N, k in [(10, 2), (30, 3), (5, 4)]:
            d = 2 ** k
            p_hats = rng.dirichlet(np.ones(d), size=N)
            p_stars = rng.dirichlet(np.ones(d), size=N)
            sigmas = rng.uniform(0, 0.05, size=(N, d))
            result = ece(p_hats, p_stars, sigmas)
            assert np.isscalar(result) or result.ndim == 0


class TestEmpiricalMarginal:
    def test_all_zeros_target_columns_2d_positions(self, synthetic_bitstrings, k):
        data = synthetic_bitstrings
        result = empirical_marginal(data["bitstrings"], data["target_positions"], k)
        assert result == pytest.approx(data["known_marginal"])

    def test_1d_target_positions_point_mass(self, k, n):
        total_shots = 200
        bits = np.zeros((total_shots, n), dtype=np.uint8)
        positions = np.arange(k, dtype=int)
        result = empirical_marginal(bits, positions, k)
        expected = np.zeros(2 ** k)
        expected[0] = 1.0
        assert result == pytest.approx(expected)

    def test_sums_to_one(self, synthetic_bitstrings, k):
        data = synthetic_bitstrings
        result = empirical_marginal(data["bitstrings"], data["target_positions"], k)
        assert result.sum() == pytest.approx(1.0)

    def test_output_shape(self, n):
        for k in [1, 2, 3]:
            bits = np.zeros((50, n), dtype=np.uint8)
            positions = np.arange(k, dtype=int)
            result = empirical_marginal(bits, positions, k)
            assert result.shape == (2 ** k,)

    def test_bit_ordering_index_convention(self, n):
        # Single shot: qubit pattern [0, 1, 0] → binary '010' = index 2
        k = 3
        bits = np.array([[0, 1, 0] + [0] * (n - k)], dtype=np.uint8)
        positions = np.array([0, 1, 2], dtype=int)
        result = empirical_marginal(bits, positions, k)
        expected = np.zeros(2 ** k)
        expected[2] = 1.0
        assert result == pytest.approx(expected)


class TestComputeAllMetrics:
    def test_expected_keys_present(self, uniform_dist, peaked_dist):
        result = compute_all_metrics(uniform_dist, peaked_dist, uniform_dist)
        for key in ("tvd_adv", "tvd_true", "kl_true", "entropy_true"):
            assert key in result

    def test_conformal_radius_absent_when_none(self, uniform_dist, peaked_dist):
        result = compute_all_metrics(uniform_dist, peaked_dist, uniform_dist)
        assert "conformal_radius" not in result

    def test_conformal_radius_present_when_provided(self, uniform_dist, peaked_dist):
        result = compute_all_metrics(uniform_dist, peaked_dist, uniform_dist, conformal_radius=0.15)
        assert "conformal_radius" in result
        assert result["conformal_radius"] == pytest.approx(0.15)

    def test_all_values_finite(self, uniform_dist, peaked_dist):
        result = compute_all_metrics(peaked_dist, uniform_dist, peaked_dist, conformal_radius=0.1)
        for val in result.values():
            assert np.isfinite(val)

    def test_entropy_true_point_mass_is_zero(self, k):
        d = 2 ** k
        p_star = np.zeros(d)
        p_star[0] = 1.0
        result = compute_all_metrics(p_star, p_star, p_star)
        # With eps=1e-12: -1*log(1+1e-12) ≈ 0
        assert result["entropy_true"] == pytest.approx(0.0, abs=1e-9)

    def test_entropy_true_uniform_is_log_d(self, uniform_dist, k):
        d = 2 ** k
        expected = np.log(d)  # eps contribution is negligible for uniform
        result = compute_all_metrics(uniform_dist, uniform_dist, uniform_dist)
        assert result["entropy_true"] == pytest.approx(expected, rel=1e-6)

    def test_identical_p_hat_and_p_star_zero_errors(self, any_dist):
        result = compute_all_metrics(any_dist, any_dist, any_dist)
        assert result["tvd_true"] == pytest.approx(0.0)
        assert result["kl_true"] == pytest.approx(0.0, abs=1e-10)

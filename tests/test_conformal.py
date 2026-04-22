"""Tests for model/conformal.py"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.conformal import ConformalPredictor
from model.transformer import QuMaskTransformer
from model.ensemble import QuMaskEnsemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ensemble(F: int = 175, output_dim: int = 8, M: int = 3) -> QuMaskEnsemble:
    """Untrained ensemble with random weights."""
    return QuMaskEnsemble([QuMaskTransformer(F=F, output_dim=output_dim) for _ in range(M)])


def _make_loader(features: torch.Tensor, p_stars: torch.Tensor, batch_size: int = 8) -> DataLoader:
    return DataLoader(TensorDataset(features, p_stars), batch_size=batch_size, shuffle=False)


def _tv(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised TV distance: 0.5 * sum |a - b|. Works for (N, D) inputs."""
    return 0.5 * np.abs(a - b).sum(axis=-1)


def _uniform(output_dim: int = 8) -> np.ndarray:
    return np.ones(output_dim, dtype=np.float32) / output_dim


def _peaked(output_dim: int = 8) -> np.ndarray:
    p = np.full(output_dim, 0.03 / (output_dim - 1), dtype=np.float32)
    p[0] = 0.97
    return p


# ---------------------------------------------------------------------------
# TestCalibrateQuantileLogic
# ---------------------------------------------------------------------------

class TestCalibrateQuantileLogic:
    """Core quantile computation tested against pre-computed scores.

    We bypass the ensemble by wiring a stub that returns known p_hat values,
    then verify the radius matches np.quantile at the split-conformal level.
    """

    def _calibrate_with_scores(
        self, scores: np.ndarray, alpha: float = 0.05
    ) -> ConformalPredictor:
        """Build a ConformalPredictor whose radius was set from known scores.

        Constructs a stub ensemble whose predict() returns p_hats such that
        TV(p_hat, p_star) exactly equals each element of `scores`, then
        calls calibrate() normally so the full code path is exercised.
        """
        output_dim = 8
        N = len(scores)

        # Construct p_stars as uniform and p_hats offset by `scores` in a
        # controlled direction: shift all mass from bin 0 to bin 1 by 2*s,
        # so TV = 0.5 * (2*s + 2*s) ... simpler: make p_hat differ from
        # p_star only in two bins.
        # TV(p_hat, p_star) = 0.5 * |d| + 0.5 * |d| = d  where d = scores[i].
        p_star_val = _uniform(output_dim)
        p_hats = np.tile(p_star_val, (N, 1)).astype(np.float32)
        p_stars = np.tile(p_star_val, (N, 1)).astype(np.float32)
        for i, s in enumerate(scores):
            s = float(s)
            # Move mass s from bin 1 to bin 0: TV = s.
            p_hats[i, 0] += s
            p_hats[i, 1] -= s

        features = torch.zeros(N, 10, 175)  # shape (N, n_blocks, F)
        p_stars_t = torch.tensor(p_stars)

        class _StubEnsemble:
            def predict(self, x, device=None):
                idx_start = _StubEnsemble._call_count * batch_size
                idx_end = min(idx_start + batch_size, N)
                batch = torch.tensor(p_hats[idx_start:idx_end])
                _StubEnsemble._call_count += 1
                return {"p_hat": batch, "sigma": torch.zeros_like(batch), "member_preds": batch.unsqueeze(0)}
            _call_count = 0

        batch_size = 4
        loader = _make_loader(features, p_stars_t, batch_size=batch_size)
        predictor = ConformalPredictor(alpha=alpha)
        predictor.calibrate(_StubEnsemble(), loader)
        return predictor, scores

    def test_radius_equals_expected_quantile(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 0.5, size=100).astype(np.float64)
        alpha = 0.05
        predictor, scores = self._calibrate_with_scores(scores, alpha=alpha)
        N = len(scores)
        q = min((1 - alpha) * (1 + 1 / N), 1.0)
        expected = float(np.quantile(scores, q, method='higher'))
        assert abs(predictor.radius - expected) < 1e-5

    def test_n_cal_set_correctly(self):
        scores = np.linspace(0, 0.4, 50)
        predictor, _ = self._calibrate_with_scores(scores)
        assert predictor.n_cal == 50

    def test_edge_case_n_cal_1(self):
        # With N=1, q = (1-alpha)*(1 + 1/1) = 2*(1-alpha) > 1, clamped to 1.0.
        # np.quantile at 1.0 returns the max (and only) value.
        scores = np.array([0.3])
        predictor, _ = self._calibrate_with_scores(scores, alpha=0.05)
        assert abs(predictor.radius - 0.3) < 1e-5

    def test_monotone_in_alpha(self):
        # Smaller alpha → larger quantile level → larger radius.
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 0.5, size=200)
        pred_strict, _ = self._calibrate_with_scores(scores, alpha=0.01)
        pred_loose, _ = self._calibrate_with_scores(scores, alpha=0.20)
        assert pred_strict.radius >= pred_loose.radius

    def test_radius_in_valid_range(self):
        # TV is bounded in [0, 1]; radius must stay in that range.
        rng = np.random.default_rng(2)
        scores = rng.uniform(0, 0.5, size=100)
        predictor, _ = self._calibrate_with_scores(scores, alpha=0.05)
        assert 0.0 <= predictor.radius <= 1.0

    def test_radius_above_alpha_fraction_of_scores(self):
        # By definition of quantile, at least (1-alpha) of scores ≤ radius.
        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 0.5, size=500)
        alpha = 0.10
        predictor, _ = self._calibrate_with_scores(scores, alpha=alpha)
        frac_covered = (scores <= predictor.radius).mean()
        assert frac_covered >= 1 - alpha

    def test_all_equal_scores(self):
        # All calibration instances have the same nonconformity score.
        # Radius should equal that score.
        scores = np.full(20, 0.25)
        predictor, _ = self._calibrate_with_scores(scores)
        assert abs(predictor.radius - 0.25) < 1e-5

    def test_scores_at_boundary_zero(self):
        # Perfect predictions: all scores = 0. Radius = 0.
        scores = np.zeros(50)
        predictor, _ = self._calibrate_with_scores(scores)
        assert predictor.radius == 0.0


# ---------------------------------------------------------------------------
# TestPredictionInterval
# ---------------------------------------------------------------------------

class TestPredictionInterval:

    def test_raises_before_calibration(self):
        predictor = ConformalPredictor(alpha=0.05)
        p_hat = _uniform()
        with pytest.raises(RuntimeError):
            predictor.prediction_interval(p_hat)

    def test_output_keys(self):
        predictor = ConformalPredictor(alpha=0.05)
        predictor.radius = 0.15
        predictor.n_cal = 100
        result = predictor.prediction_interval(_uniform())
        assert set(result.keys()) == {"center", "radius", "alpha"}

    def test_center_is_input_p_hat(self):
        predictor = ConformalPredictor(alpha=0.05)
        predictor.radius = 0.15
        predictor.n_cal = 100
        p_hat = _peaked()
        result = predictor.prediction_interval(p_hat)
        np.testing.assert_array_equal(result["center"], p_hat)

    def test_radius_matches_stored(self):
        predictor = ConformalPredictor(alpha=0.05)
        predictor.radius = 0.23
        predictor.n_cal = 100
        result = predictor.prediction_interval(_uniform())
        assert result["radius"] == 0.23

    def test_alpha_matches_constructor(self):
        predictor = ConformalPredictor(alpha=0.10)
        predictor.radius = 0.10
        predictor.n_cal = 100
        result = predictor.prediction_interval(_uniform())
        assert result["alpha"] == 0.10

    def test_center_not_copied(self):
        # prediction_interval returns the input array, not a defensive copy.
        # This is the documented contract; both the array and result["center"]
        # should be the same object.
        predictor = ConformalPredictor(alpha=0.05)
        predictor.radius = 0.1
        predictor.n_cal = 50
        p_hat = _uniform()
        result = predictor.prediction_interval(p_hat)
        assert result["center"] is p_hat

    def test_radius_zero_after_perfect_calibration(self):
        predictor = ConformalPredictor(alpha=0.05)
        predictor.radius = 0.0
        predictor.n_cal = 100
        result = predictor.prediction_interval(_uniform())
        assert result["radius"] == 0.0


# ---------------------------------------------------------------------------
# TestEmpiricalCoverage
# ---------------------------------------------------------------------------

class TestEmpiricalCoverage:

    def _predictor_with_radius(self, r: float, alpha: float = 0.05) -> ConformalPredictor:
        predictor = ConformalPredictor(alpha=alpha)
        predictor.radius = r
        predictor.n_cal = 100
        return predictor

    def test_exact_coverage_fraction(self):
        # Construct 100 instances where exactly 90 have TV ≤ radius.
        # The 90 covered instances: p_hat == p_star (TV = 0).
        # The 10 uncovered: TV = 0.4 > radius = 0.3.
        output_dim = 8
        N = 100
        p_star = _uniform(output_dim)
        p_hats = np.tile(p_star, (N, 1))
        p_stars = np.tile(p_star, (N, 1))
        for i in range(10):  # last 10 are outside the ball
            p_hats[-(i + 1), 0] += 0.4
            p_hats[-(i + 1), 1] -= 0.4

        predictor = self._predictor_with_radius(0.3)
        coverage = predictor.empirical_coverage(p_hats, p_stars)
        assert abs(coverage - 0.90) < 1e-9

    def test_coverage_in_unit_interval(self):
        rng = np.random.default_rng(10)
        output_dim = 8
        N = 50
        p_hats = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        p_stars = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        predictor = self._predictor_with_radius(0.2)
        coverage = predictor.empirical_coverage(p_hats, p_stars)
        assert 0.0 <= coverage <= 1.0

    def test_radius_zero_covers_only_identical(self):
        output_dim = 8
        p = _uniform(output_dim)
        p_hats = np.tile(p, (10, 1))
        p_stars = np.tile(p, (10, 1))
        p_hats[0, 0] += 0.1  # one instance is outside
        p_hats[0, 1] -= 0.1
        predictor = self._predictor_with_radius(0.0)
        coverage = predictor.empirical_coverage(p_hats, p_stars)
        assert abs(coverage - 0.9) < 1e-9

    def test_radius_one_covers_all(self):
        # TV ≤ 1 always holds since TV(p, q) ≤ 1 for any distributions.
        rng = np.random.default_rng(11)
        output_dim = 8
        N = 50
        p_hats = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        p_stars = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        predictor = self._predictor_with_radius(1.0)
        assert predictor.empirical_coverage(p_hats, p_stars) == 1.0

    def test_perfect_predictions_covered(self):
        # When p_hat == p_star for every instance, any radius ≥ 0 covers all.
        output_dim = 8
        p = _uniform(output_dim)
        p_hats = np.tile(p, (20, 1))
        p_stars = np.tile(p, (20, 1))
        predictor = self._predictor_with_radius(0.0)
        assert predictor.empirical_coverage(p_hats, p_stars) == 1.0

    def test_tv_symmetry(self):
        # TV is symmetric: coverage(p_hat, p_star) == coverage(p_star, p_hat).
        rng = np.random.default_rng(12)
        output_dim = 8
        N = 30
        p_hats = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        p_stars = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        predictor = self._predictor_with_radius(0.25)
        assert predictor.empirical_coverage(p_hats, p_stars) == \
               predictor.empirical_coverage(p_stars, p_hats)

    def test_coverage_at_exact_boundary(self):
        # Instances with TV == radius should be counted as covered (≤, not <).
        output_dim = 8
        p_star = _uniform(output_dim)
        p_hat = p_star.copy()
        r = 0.20
        p_hat[0] += r
        p_hat[1] -= r  # TV(p_hat, p_star) = r exactly
        tv_check = 0.5 * np.abs(p_hat - p_star).sum()
        assert abs(tv_check - r) < 1e-6, "test setup: TV should equal r"
        predictor = self._predictor_with_radius(r)
        coverage = predictor.empirical_coverage(
            p_hat[np.newaxis], p_star[np.newaxis]
        )
        assert coverage == 1.0

    def test_single_instance(self):
        output_dim = 8
        p_star = _uniform(output_dim)
        p_hat = p_star.copy()
        predictor = self._predictor_with_radius(0.1)
        assert predictor.empirical_coverage(p_hat[np.newaxis], p_star[np.newaxis]) == 1.0

    def test_coverage_monotone_in_radius(self):
        # Larger radius → coverage never decreases.
        rng = np.random.default_rng(13)
        output_dim = 8
        N = 100
        p_hats = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        p_stars = rng.dirichlet(np.ones(output_dim), size=N).astype(np.float32)
        radii = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
        coverages = []
        for r in radii:
            pred = ConformalPredictor(alpha=0.05)
            pred.radius = r
            pred.n_cal = N
            coverages.append(pred.empirical_coverage(p_hats, p_stars))
        for i in range(len(coverages) - 1):
            assert coverages[i] <= coverages[i + 1]



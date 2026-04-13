"""
Tests for model/conformal.py

Covers ConformalPredictor.calibrate, prediction_interval, and empirical_coverage.
Calibration requires running the ensemble on a dataset, so all tests that call
calibrate are marked slow. Tests against pre-computed scores (bypassing the
ensemble) are fast and test the quantile logic directly.


class TestCalibrateQuantileLogic:
    # This class tests the core quantile computation without running the ensemble.
    # Directly inject synthetic nonconformity scores (a numpy array of TVD values)
    #   by monkey-patching or subclassing ConformalPredictor to accept pre-computed
    #   scores. Then verify:
    #   - self.radius equals the (1-alpha)(1 + 1/N_cal) empirical quantile of the
    #     scores. For a known sorted score array, compute the expected quantile by
    #     hand and assert equality.
    #   - self.n_cal equals the number of scores passed.
    # Test the edge case N_cal=1: radius should equal the single score value
    #   (the only quantile available is 100% of the data).
    # Test monotonicity: for the same score array, a smaller alpha (e.g. 0.01)
    #   should produce a larger radius than a larger alpha (e.g. 0.20).


class TestPredictionInterval:
    # Test that calling prediction_interval before calibrate raises RuntimeError.
    # After calibrating with a known radius r, test that prediction_interval
    #   returns a dict with keys "center", "radius", "alpha".
    # Test that "radius" equals self.radius and "center" equals the input p_hat.
    # Test that "alpha" equals the alpha used at construction time.


class TestEmpiricalCoverage:
    # Test the coverage formula directly with synthetic data (no ensemble needed).
    # Construct p_hats and p_stars such that exactly 90% of instances have
    #   TVD(p_hat, p_star) ≤ radius. Set radius to the 90th percentile of the
    #   TVD values and verify empirical_coverage returns 0.90.
    # Test that empirical_coverage is in [0, 1].
    # Test the boundary: when radius is 0, coverage should be 0 (no instance
    #   has TVD exactly 0 unless p_hat == p_star); when radius is 1.0 (maximum
    #   possible TVD), coverage should be 1.0.


class TestCoverageGuarantee:
    # @pytest.mark.slow — requires tiny_dataset_path and a trained ensemble.
    # This is the statistical correctness test for split-conformal prediction.
    # Calibrate on N_cal=10 instances (from tiny_dataset_path), then evaluate
    #   empirical_coverage on a fresh set of 10 held-out instances.
    # Assert empirical_coverage >= 1 - alpha (0.95).
    # Note: with only 10 calibration points the guarantee is loose, but the
    #   finite-sample property of split-conformal prediction guarantees it holds
    #   in expectation. If this test fails, the quantile computation is wrong.
"""

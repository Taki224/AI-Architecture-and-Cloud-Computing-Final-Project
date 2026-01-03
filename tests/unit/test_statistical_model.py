"""
Unit tests for StatisticalAnomalyDetector.

Tests Z-score based anomaly detection.
"""

import pytest
import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.statistical_model import StatisticalAnomalyDetector


class TestStatisticalDetector:
    """Test Z-score anomaly detection."""

    def test_detector_initialization(self):
        """Test detector can be created with default threshold."""
        detector = StatisticalAnomalyDetector()
        assert detector.threshold == 3.0

    def test_custom_threshold(self):
        """Test detector accepts custom threshold."""
        detector = StatisticalAnomalyDetector(threshold=2.5)
        assert detector.threshold == 2.5

    def test_fit_learns_mean_and_std(self, normal_readings):
        """Test that fit learns mean and std from data."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        assert detector.mean_ is not None
        assert detector.std_ is not None
        # Mean should be close to 0
        assert abs(detector.mean_) < 0.3
        # Std should be close to 1
        assert abs(detector.std_ - 1.0) < 0.3

    def test_predict_normal_values(self, normal_readings):
        """Test that normal values are predicted as 1 (normal)."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        # Test on some normal values
        predictions = detector.predict([0.0, 0.5, -0.5, 1.0])
        
        # All should be normal (1)
        assert all(p == 1 for p in predictions)

    def test_predict_anomaly_values(self, normal_readings):
        """Test that anomaly values are predicted as -1 (anomaly)."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        # Test on anomaly values (beyond 3σ)
        predictions = detector.predict([5.0, -5.0, 10.0])
        
        # All should be anomalies (-1)
        assert all(p == -1 for p in predictions)

    def test_score_samples_returns_scores(self, normal_readings):
        """Test that score_samples returns numeric scores."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        scores = detector.score_samples([0.0, 5.0])
        
        assert len(scores) == 2
        assert isinstance(scores[0], (int, float, np.number))

    def test_decision_function_alias(self, normal_readings):
        """Test that decision_function is alias for score_samples."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        scores1 = detector.score_samples([0.0, 1.0])
        scores2 = detector.decision_function([0.0, 1.0])
        
        np.testing.assert_array_equal(scores1, scores2)

    def test_fit_with_labels(self):
        """Test fitting with labels (y parameter)."""
        detector = StatisticalAnomalyDetector()
        
        # Mixed data with labels
        X = np.array([0.0, 0.5, 5.0, -0.5, 10.0])
        y = np.array([1, 1, -1, 1, -1])  # 1=normal, -1=anomaly
        
        detector.fit(X, y)
        
        # Should learn from normal samples only
        assert detector.mean_ is not None
        assert detector.std_ is not None


class TestStatisticalDetectorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_value_prediction(self, normal_readings):
        """Test prediction on single value."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        prediction = detector.predict([0.5])
        assert len(prediction) == 1

    def test_batch_prediction(self, normal_readings, batch_of_10):
        """Test prediction on batch of values."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)
        
        predictions = detector.predict(batch_of_10)
        assert len(predictions) == 10

    def test_threshold_boundary(self, normal_readings):
        """Test behavior at threshold boundary."""
        detector = StatisticalAnomalyDetector(threshold=3.0)
        detector.fit(normal_readings)
        
        # Value exactly at 3σ from mean
        test_value = detector.mean_ + 3.0 * detector.std_
        prediction = detector.predict([test_value])
        
        # At exactly 3σ, score should be ~0
        score = detector.score_samples([test_value])[0]
        assert abs(score) < 0.1

    def test_works_with_numpy_arrays(self, normal_readings):
        """Test that detector works with numpy arrays."""
        detector = StatisticalAnomalyDetector()
        
        X = np.array(normal_readings)
        detector.fit(X)
        
        predictions = detector.predict(X[:10])
        assert len(predictions) == 10

    def test_works_with_python_lists(self, normal_readings):
        """Test that detector works with Python lists."""
        detector = StatisticalAnomalyDetector()
        detector.fit(normal_readings)  # normal_readings is already a list
        
        predictions = detector.predict(normal_readings[:10])
        assert len(predictions) == 10

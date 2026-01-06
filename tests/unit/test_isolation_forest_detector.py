"""
Unit tests for IsolationForestDetector.

Tests ML-based anomaly detection with sliding window features.
"""

import pytest
import sys
import os

# Add models directory to path
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
sys.path.insert(0, models_path)

from isolation_forest_detector import IsolationForestDetector


class TestIsolationForestInitialization:
    """Test detector initialization."""

    def test_default_initialization(self):
        """Test detector can be created with defaults."""
        detector = IsolationForestDetector()
        
        assert detector.contamination == 0.10
        assert detector.window_size == 50
        assert detector.n_estimators == 100
        assert detector.min_samples_for_fit == 200
        assert detector.z_threshold == 3.5
        assert detector.is_fitted is False

    def test_custom_parameters(self):
        """Test detector accepts custom parameters."""
        detector = IsolationForestDetector(
            contamination=0.01,
            window_size=30,
            n_estimators=50,
            min_samples_for_fit=50,
            z_threshold=4.0
        )
        
        assert detector.contamination == 0.01
        assert detector.window_size == 30
        assert detector.n_estimators == 50
        assert detector.min_samples_for_fit == 50
        assert detector.z_threshold == 4.0


class TestWarmupPhase:
    """Test behavior during warmup phase."""

    def test_warmup_status(self):
        """Test that detector reports warming_up status initially."""
        detector = IsolationForestDetector()
        
        result = detector.detect(0.5)
        
        assert result["status"] == "warming_up"
        assert result["is_anomaly"] is False

    def test_samples_needed_countdown(self):
        """Test that samples_needed decreases during warmup."""
        detector = IsolationForestDetector(min_samples_for_fit=10)
        
        result1 = detector.detect(0.5)
        result2 = detector.detect(0.6)
        
        assert result2["samples_needed"] < result1["samples_needed"]

    def test_warmup_to_fitted_transition(self):
        """Test transition from warmup to fitted state."""
        # Use smaller window size so we have enough feature samples
        detector = IsolationForestDetector(
            min_samples_for_fit=60,
            window_size=10
        )
        
        # Feed enough samples to trigger fitting
        for i in range(59):
            result = detector.detect(float(i) * 0.1)
            assert result["status"] == "warming_up"
        
        # 60th sample should trigger fitting
        result = detector.detect(5.9)
        assert result["status"] == "just_fitted"
        assert detector.is_fitted is True


class TestWindowFilling:
    """Test sliding window filling phase."""

    def test_filling_window_after_fit(self):
        """Test that detector works after fitting with Z-score detection."""
        detector = IsolationForestDetector(
            min_samples_for_fit=10,
            window_size=20
        )
        
        # Warm up (fit the model)
        for i in range(10):
            detector.detect(float(i) * 0.1)
        
        # After fitting, detector uses Z-score detection so it's immediately active
        result = detector.detect(0.5)
        
        # With the new Z-score based detection, status is active after fitting
        assert result["status"] in ["active", "just_fitted"]


class TestDetectionResults:
    """Test detection result structure."""

    def test_result_structure_during_warmup(self):
        """Test result contains required fields during warmup."""
        detector = IsolationForestDetector()
        result = detector.detect(0.5)
        
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "status" in result
        assert "samples_needed" in result

    def test_anomaly_score_range(self):
        """Test that anomaly_score is in valid range."""
        detector = IsolationForestDetector(min_samples_for_fit=5)
        
        # Warm up
        for i in range(10):
            result = detector.detect(float(i) * 0.1)
            assert 0.0 <= result["anomaly_score"] <= 1.0


class TestFeatureExtraction:
    """Test feature extraction from sliding window."""

    def test_window_accumulation(self):
        """Test that window accumulates readings."""
        detector = IsolationForestDetector(window_size=5)
        
        for i in range(3):
            detector.detect(float(i))
        
        assert len(detector.window) == 3

    def test_window_max_size(self):
        """Test that window doesn't exceed max size."""
        detector = IsolationForestDetector(window_size=5)
        
        for i in range(10):
            detector.detect(float(i))
        
        # Window should be capped at window_size
        assert len(detector.window) <= 5


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset clears all state."""
        detector = IsolationForestDetector(min_samples_for_fit=5)
        
        # Warm up
        for i in range(10):
            detector.detect(float(i) * 0.1)
        
        # Reset
        detector.reset()
        
        assert detector.is_fitted is False
        assert len(detector.window) == 0
        assert len(detector.training_data) == 0

    def test_detection_after_reset(self):
        """Test that detection works after reset."""
        detector = IsolationForestDetector(min_samples_for_fit=5)
        
        # Warm up
        for i in range(10):
            detector.detect(float(i) * 0.1)
        
        # Reset
        detector.reset()
        
        # Should be back to warmup phase
        result = detector.detect(0.5)
        assert result["status"] == "warming_up"


class TestNormalSequence:
    """Test detection on normal data sequences."""

    def test_normal_readings_sequence(self, normal_readings):
        """Test detector on sequence of normal readings."""
        detector = IsolationForestDetector(
            min_samples_for_fit=20,
            window_size=10  # Smaller window so ML can fit with enough samples
        )
        
        results = []
        for reading in normal_readings[:50]:
            result = detector.detect(reading)
            results.append(result)
        
        # Should complete warmup and be active
        assert any(r["status"] in ["just_fitted", "active"] 
                   for r in results)

    def test_training_data_collection(self):
        """Test that training data is collected during warmup."""
        detector = IsolationForestDetector(min_samples_for_fit=10)
        
        for i in range(5):
            detector.detect(float(i) * 0.1)
        
        assert len(detector.training_data) == 5


class TestEdgeCases:
    """Test edge cases."""

    def test_zero_value(self):
        """Test detection with zero value."""
        detector = IsolationForestDetector()
        result = detector.detect(0.0)
        
        assert result["is_anomaly"] is False
        assert result["status"] == "warming_up"

    def test_negative_value(self):
        """Test detection with negative value."""
        detector = IsolationForestDetector()
        result = detector.detect(-0.5)
        
        assert result["is_anomaly"] is False

    def test_large_value_during_warmup(self):
        """Test that large values during warmup don't break detector."""
        detector = IsolationForestDetector()
        result = detector.detect(100.0)
        
        assert "is_anomaly" in result
        assert result["status"] == "warming_up"

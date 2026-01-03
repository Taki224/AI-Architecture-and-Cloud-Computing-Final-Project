"""
Unit tests for HybridAnomalyDetector.

Tests the ensemble approach combining ML and statistical detection.
"""

import pytest
import sys
import os

# Add models directory to path
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
sys.path.insert(0, models_path)

from hybrid_detector import HybridAnomalyDetector


class TestHybridDetectorInitialization:
    """Test hybrid detector initialization."""

    def test_default_initialization(self):
        """Test detector can be created with defaults."""
        detector = HybridAnomalyDetector()
        
        assert detector.z_threshold == 3.0
        assert detector.contamination == 0.003
        assert detector._detection_count == 0
        assert detector._anomaly_count == 0

    def test_custom_parameters(self):
        """Test detector accepts custom parameters."""
        detector = HybridAnomalyDetector(
            z_threshold=2.5,
            contamination=0.01,
            window_size=30,
            n_estimators=50
        )
        
        assert detector.z_threshold == 2.5
        assert detector.contamination == 0.01


class TestStatisticalFallbackMode:
    """Test behavior during ML warmup (statistical fallback)."""

    def test_warmup_phase_uses_statistical(self):
        """Test that warmup phase uses statistical detection only."""
        detector = HybridAnomalyDetector()
        
        # During warmup, should use statistical fallback
        result = detector.detect(0.5)
        
        assert result["method"] == "statistical_fallback"
        assert "ml_status" in result
        assert result["ml_status"] in ["warming_up", "filling_window"]

    def test_normal_value_during_warmup(self):
        """Test normal value detection during warmup."""
        detector = HybridAnomalyDetector()
        
        result = detector.detect(0.5)
        
        assert result["is_anomaly"] is False
        assert result["stat_anomaly"] is False
        assert result["ml_anomaly"] is False

    def test_anomaly_during_warmup(self):
        """Test anomaly detection during warmup."""
        detector = HybridAnomalyDetector()
        
        # 5σ deviation should be flagged even during warmup
        result = detector.detect(5.5)
        
        assert result["is_anomaly"] is True
        assert result["stat_anomaly"] is True
        assert result["method"] == "statistical_fallback"


class TestDetectionResults:
    """Test detection result structure."""

    def test_result_contains_required_fields(self):
        """Test that detection result has all required fields."""
        detector = HybridAnomalyDetector()
        result = detector.detect(0.5)
        
        required_fields = [
            "is_anomaly",
            "method",
            "ml_anomaly",
            "stat_anomaly",
            "anomaly_score",
            "z_score",
            "confidence"
        ]
        
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_confidence_in_valid_range(self):
        """Test that confidence is between 0 and 1."""
        detector = HybridAnomalyDetector()
        
        for value in [0.0, 0.5, 1.0, 3.0, 5.0]:
            result = detector.detect(value)
            assert 0.0 <= result["confidence"] <= 1.0

    def test_z_score_calculation(self):
        """Test that z_score is calculated correctly."""
        detector = HybridAnomalyDetector()
        
        result = detector.detect(3.0)
        
        # For μ=0, σ=1, z_score should equal |value|
        assert abs(result["z_score"] - 3.0) < 0.1


class TestStatistics:
    """Test statistics tracking."""

    def test_get_stats_initial(self):
        """Test initial statistics."""
        detector = HybridAnomalyDetector()
        stats = detector.get_stats()
        
        assert stats["total_detections"] == 0
        assert stats["total_anomalies"] == 0
        assert stats["anomaly_rate"] == 0.0

    def test_stats_track_detections(self):
        """Test that statistics track detections."""
        detector = HybridAnomalyDetector()
        
        detector.detect(0.5)
        detector.detect(0.6)
        detector.detect(0.7)
        
        stats = detector.get_stats()
        assert stats["total_detections"] == 3

    def test_stats_track_anomalies(self):
        """Test that statistics track anomalies."""
        detector = HybridAnomalyDetector()
        
        detector.detect(0.5)  # Normal
        detector.detect(5.0)  # Anomaly
        detector.detect(0.6)  # Normal
        
        stats = detector.get_stats()
        assert stats["total_anomalies"] >= 1


class TestReset:
    """Test reset functionality."""

    def test_reset_clears_stats(self):
        """Test that reset clears statistics."""
        detector = HybridAnomalyDetector()
        
        # Generate some detections
        for i in range(10):
            detector.detect(float(i) * 0.1)
        
        # Reset
        detector.reset()
        
        stats = detector.get_stats()
        assert stats["total_detections"] == 0
        assert stats["total_anomalies"] == 0


class TestMultipleDetections:
    """Test behavior with multiple sequential detections."""

    def test_normal_sequence(self, normal_readings):
        """Test detection on sequence of normal readings."""
        detector = HybridAnomalyDetector()
        
        anomaly_count = 0
        for reading in normal_readings[:20]:
            result = detector.detect(reading)
            if result["is_anomaly"]:
                anomaly_count += 1
        
        # Most should be normal (allow up to 5 anomalies in 20 readings)
        assert anomaly_count < 5

    def test_mixed_sequence(self):
        """Test detection on mixed normal/anomaly sequence."""
        detector = HybridAnomalyDetector()
        
        # Mix of normal and anomalous values
        values = [0.5, 0.6, 5.0, 0.7, -5.0, 0.8, 0.9]
        
        results = [detector.detect(v) for v in values]
        
        # Should detect the large deviations (5.0 and -5.0)
        assert any(r["is_anomaly"] for r in results)

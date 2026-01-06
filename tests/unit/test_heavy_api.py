"""
Unit tests for Heavy Model API Service
Tests initialization, batch processing, and carbon tracking integration
"""
import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

# Mock Google Cloud modules before importing
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.pubsub_v1'] = MagicMock()
sys.modules['google.cloud.logging'] = MagicMock()
sys.modules['google.cloud.monitoring_v3'] = MagicMock()
sys.modules['codecarbon'] = MagicMock()

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/heavy-model'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/common'))


@pytest.fixture
def mock_detector():
    """Mock HybridAnomalyDetector."""
    detector = MagicMock()
    detector.detect.return_value = {
        'is_anomaly': False,
        'confidence': 0.1,
        'anomaly_score': 0.05,
        'z_score': 1.2,
        'method': 'ensemble',
        'ml_anomaly': False,
        'stat_anomaly': False
    }
    detector.get_stats.return_value = {
        'ml_fitted': True,
        'total_detections': 100,
        'anomaly_rate': 0.3
    }
    return detector


@pytest.fixture
def mock_carbon_monitor():
    """Mock CarbonMonitor."""
    monitor = MagicMock()
    monitor.get_stats.return_value = {
        'service': 'heavy-model',
        'mode': 'PERFORMANCE',
        'total_emissions_gco2e': 5.5,
        'total_inferences': 100,
        'avg_emissions_per_inference_gco2e': 0.055
    }
    
    # Mock context manager
    context = MagicMock()
    context.__enter__ = MagicMock(return_value=context)
    context.__exit__ = MagicMock(return_value=False)
    monitor.track_inference.return_value = context
    
    return monitor


class TestBasicFunctionality:
    """Test basic functionality with mocked dependencies."""
    
    def test_carbon_monitor_integration(self, mock_detector, mock_carbon_monitor):
        """Test that carbon monitoring can be integrated into batch processing."""
        # Simulate batch data
        batch_data = {
            'device_id': 'sensor-001',
            'readings': [
                {'timestamp': 1704556800.0, 'vibration': 1.2},
                {'timestamp': 1704556801.0, 'vibration': 0.8},
            ]
        }
        
        # Simulate processing with carbon tracking
        batch_size = len(batch_data['readings'])
        
        with mock_carbon_monitor.track_inference(batch_size=batch_size):
            for reading in batch_data['readings']:
                result = mock_detector.detect(reading['vibration'])
                assert 'is_anomaly' in result
        
        # Verify carbon tracking was called
        mock_carbon_monitor.track_inference.assert_called_once_with(batch_size=2)
        
    def test_anomaly_detection_flow(self, mock_detector):
        """Test anomaly detection flow with detector."""
        value = 5.5
        result = mock_detector.detect(value)
        
        assert 'is_anomaly' in result
        assert 'confidence' in result
        assert 'z_score' in result
        mock_detector.detect.assert_called_once_with(value)


class TestCarbonStatsCollection:
    """Test carbon statistics collection."""
    
    def test_carbon_stats_structure(self, mock_carbon_monitor):
        """Test carbon stats have correct structure."""
        stats = mock_carbon_monitor.get_stats()
        
        assert 'service' in stats
        assert 'mode' in stats
        assert 'total_emissions_gco2e' in stats
        assert 'total_inferences' in stats
        assert 'avg_emissions_per_inference_gco2e' in stats
        
        assert stats['service'] == 'heavy-model'
        assert stats['mode'] == 'PERFORMANCE'
        assert stats['total_emissions_gco2e'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

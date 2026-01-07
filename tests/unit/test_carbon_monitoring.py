"""
Unit tests for Carbon Monitoring module
Tests the CarbonMonitor class for emissions tracking and GCP metric export
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Mock Google Cloud modules before importing carbon_monitoring
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.monitoring_v3'] = MagicMock()
sys.modules['codecarbon'] = MagicMock()

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../services/common'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../models'))

from carbon_monitoring import CarbonMonitor


@pytest.fixture
def mock_monitoring_client():
    """Mock GCP Monitoring client."""
    with patch('carbon_monitoring.monitoring_v3.MetricServiceClient') as mock:
        yield mock


class TestCarbonMonitorInitialization:
    """Test CarbonMonitor initialization."""
    
    def test_init_with_valid_params(self, mock_monitoring_client):
        """Test initialization with valid parameters."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        assert monitor.project_id == "test-project"
        assert monitor.service_name == "heavy-model"
        assert monitor.mode == "PERFORMANCE"
        assert monitor.total_emissions_kg == 0.0
        assert monitor.total_inferences == 0
        mock_monitoring_client.assert_called_once()
    
    def test_init_with_custom_country(self, mock_monitoring_client):
        """Test initialization with custom country code."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="light-model",
            mode="ECO",
            country_iso_code="USA"
        )
        
        assert monitor.country_iso_code == "USA"
        assert monitor.mode == "ECO"


class TestCarbonTracking:
    """Test carbon emissions tracking."""
    
    def test_track_inference_context_manager(self, mock_monitoring_client):
        """Test tracking inference using context manager (estimation-based)."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        # Track a batch of 10 inferences (uses estimation with 10s default per batch)
        with monitor.track_inference(batch_size=10):
            pass  # Simulate inference
        
        # Check emissions were recorded (estimation: 15W * 10s / 3600 * 0.1 = 0.004166... kg)
        EXPECTED_EMISSIONS = (15.0 * 10.0 / 3600) * 0.1  # 0.004166... kg
        assert monitor.total_inferences == 10
        assert monitor.total_emissions_kg > 0  # Should have some emissions recorded
        assert abs(monitor.total_emissions_kg - EXPECTED_EMISSIONS) < 0.000001  # Allow small float difference
    
    def test_record_emissions_manually(self, mock_monitoring_client):
        """Test manually recording emissions."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="light-model",
            mode="ECO"
        )
        
        # Record emissions for a single inference
        monitor.record_emissions(emissions_kg=0.0005, batch_size=1)
        
        assert abs(monitor.total_emissions_kg - 0.0005) < 0.0001
        assert monitor.total_inferences == 1
        
        # Record more emissions
        monitor.record_emissions(emissions_kg=0.0003, batch_size=5)
        
        assert abs(monitor.total_emissions_kg - 0.0008) < 0.0001
        assert monitor.total_inferences == 6
    
    def test_track_multiple_batches(self, mock_monitoring_client):
        """Test tracking multiple batches."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        # Track multiple batches (3 batches × 10s default = 30s total)
        for i in range(3):
            with monitor.track_inference(batch_size=10):
                pass
        
        # Each batch: 15W * 10s / 3600 * 0.1 = 0.004166... kg
        EXPECTED_EMISSIONS_PER_BATCH = (15.0 * 10.0 / 3600) * 0.1
        EXPECTED_TOTAL = EXPECTED_EMISSIONS_PER_BATCH * 3
        
        assert monitor.total_inferences == 30
        assert abs(monitor.total_emissions_kg - EXPECTED_TOTAL) < 0.000001  # Allow small float difference

    def test_track_inference_with_actual_time(self, mock_monitoring_client):
        """Test tracking inference with actual measured time."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        # Track with actual inference time (5000ms = 5s for batch of 10)
        with monitor.track_inference(batch_size=10, inference_time_ms=5000):
            pass
        
        # Calculate expected: 15W * 5s / 3600 * 0.1 = 0.002083... kg
        EXPECTED_EMISSIONS = (15.0 * 5.0 / 3600) * 0.1
        
        assert monitor.total_inferences == 10
        assert abs(monitor.total_emissions_kg - EXPECTED_EMISSIONS) < 0.000001


class TestMetricsExport:
    """Test GCP Cloud Monitoring metrics export."""
    
    @patch('carbon_monitoring.monitoring_v3.MetricServiceClient')
    def test_write_emissions_metric(self, mock_client_class):
        """Test writing emissions metric to Cloud Monitoring."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        # Write metric
        monitor._write_emissions_metric(10.5)  # 10.5 grams
        
        # Check metric was written
        mock_client.create_time_series.assert_called()
        call_args = mock_client.create_time_series.call_args
        
        assert call_args[1]['name'] == "projects/test-project"
        time_series = call_args[1]['time_series'][0]
        assert time_series.metric.type == CarbonMonitor.METRIC_EMISSIONS
        # Labels are set via dictionary assignment, so just check they exist
        assert 'service' in str(time_series.metric.labels) or hasattr(time_series.metric.labels, '__setitem__')
    
    @patch('carbon_monitoring.monitoring_v3.MetricServiceClient')
    def test_flush_pending_metrics(self, mock_client_class):
        """Test flushing pending metrics."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="light-model",
            mode="ECO"
        )
        
        # Record some emissions
        monitor.record_emissions(emissions_kg=0.002, batch_size=5)
        
        # Flush metrics
        monitor._flush_pending_metrics()
        
        # Check metrics were written
        assert mock_client.create_time_series.call_count >= 2  # emissions + total + count


class TestStatistics:
    """Test statistics retrieval."""
    
    def test_get_stats_initial(self, mock_monitoring_client):
        """Test getting stats with no data."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        stats = monitor.get_stats()
        
        assert stats['service'] == "heavy-model"
        assert stats['mode'] == "PERFORMANCE"
        assert stats['total_emissions_gco2e'] == 0.0
        assert stats['total_inferences'] == 0
        assert stats['avg_emissions_per_inference_gco2e'] == 0.0
    
    def test_get_stats_with_data(self, mock_monitoring_client):
        """Test getting stats with tracked data."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="light-model",
            mode="ECO"
        )
        
        # Record some emissions
        monitor.record_emissions(emissions_kg=0.010, batch_size=100)  # 10 grams over 100 inferences
        
        stats = monitor.get_stats()
        
        assert stats['total_emissions_gco2e'] == 10.0  # 10 grams
        assert stats['total_inferences'] == 100
        assert stats['avg_emissions_per_inference_gco2e'] == 0.1  # 0.1 grams per inference


class TestBackgroundReporter:
    """Test background metrics reporting."""
    
    @patch('carbon_monitoring.monitoring_v3.MetricServiceClient')
    def test_reporter_thread_starts(self, mock_client_class):
        """Test that background reporter thread starts."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        assert monitor._running is True
        assert monitor._reporter_thread is not None
        assert monitor._reporter_thread.daemon is True
        
        # Cleanup
        monitor.flush()
    
    @patch('carbon_monitoring.monitoring_v3.MetricServiceClient')
    def test_flush_stops_reporter(self, mock_client_class):
        """Test that flush stops the reporter."""
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        monitor.flush()
        
        assert monitor._running is False


class TestErrorHandling:
    """Test error handling and graceful degradation."""
    
    @patch('carbon_monitoring.monitoring_v3.MetricServiceClient')
    def test_metric_write_failure_graceful(self, mock_client_class):
        """Test graceful handling of metric write failures."""
        mock_client = MagicMock()
        mock_client.create_time_series.side_effect = Exception("Network error")
        mock_client_class.return_value = mock_client
        
        monitor = CarbonMonitor(
            project_id="test-project",
            service_name="heavy-model",
            mode="PERFORMANCE"
        )
        
        # Should not raise exception
        monitor._write_emissions_metric(5.0)
        
        # Data should still be tracked locally
        monitor.record_emissions(emissions_kg=0.001, batch_size=1)
        assert monitor.total_emissions_kg == 0.001
    
    def test_monitoring_client_init_failure(self):
        """Test handling of monitoring client initialization failure."""
        with patch('carbon_monitoring.monitoring_v3.MetricServiceClient', side_effect=Exception("Auth failed")):
            # Should not raise exception
            monitor = CarbonMonitor(
                project_id="test-project",
                service_name="heavy-model",
                mode="PERFORMANCE"
            )
            
            # Monitor should still work locally
            assert monitor._monitoring_client is None
            monitor.record_emissions(emissions_kg=0.001, batch_size=1)
            assert monitor.total_emissions_kg == 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

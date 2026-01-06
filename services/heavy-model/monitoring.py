"""
Cloud Monitoring for Heavy Model Anomaly Detection
Logs anomaly detections to Cloud Logging and exports metrics to Cloud Monitoring
"""
import os
import time
import threading
from datetime import datetime
from typing import Optional
from collections import deque


class AnomalyMonitor:
    """
    Monitors and logs anomaly detections to GCP Cloud Logging and Cloud Monitoring.
    
    Features:
    - Structured JSON logging to Cloud Logging
    - 60-second rolling window anomaly rate metric
    - Custom metric export to Cloud Monitoring
    - Graceful degradation when Cloud APIs unavailable
    """
    
    METRIC_TYPE = "custom.googleapis.com/anomaly_detection/rate"
    WINDOW_SECONDS = 60
    
    def __init__(self, project_id: str):
        """
        Initialize the anomaly monitor.
        
        Args:
            project_id: GCP project ID for Cloud Logging/Monitoring
        """
        self.project_id = project_id
        self._logging_client = None
        self._monitoring_client = None
        self._logger = None
        
        # Rolling window for anomaly rate calculation
        self._anomaly_times = deque()
        self._lock = threading.Lock()
        
        # Statistics
        self.total_anomalies = 0
        self.total_readings = 0
        
        # Initialize Cloud clients
        self._init_logging()
        self._init_monitoring()
        
        # Start background metric reporter
        self._reporter_thread = None
        self._running = False
        self._start_reporter()
    
    def _init_logging(self):
        """Initialize Cloud Logging client."""
        try:
            from google.cloud import logging as cloud_logging
            
            self._logging_client = cloud_logging.Client(project=self.project_id)
            self._logger = self._logging_client.logger('anomaly-detection')
            print("[Monitor] Cloud Logging initialized")
            
        except ImportError:
            print("[Monitor] google-cloud-logging not installed, using stdout")
        except Exception as e:
            print(f"[Monitor] Cloud Logging unavailable: {e}")
    
    def _init_monitoring(self):
        """Initialize Cloud Monitoring client and create metric descriptors."""
        try:
            from google.cloud import monitoring_v3
            
            self._monitoring_client = monitoring_v3.MetricServiceClient()
            print("[Monitor] Cloud Monitoring initialized")
            self._create_metric_descriptor()
            
        except ImportError:
            print("[Monitor] google-cloud-monitoring not installed")
        except Exception as e:
            print(f"[Monitor] Cloud Monitoring unavailable: {e}")
    
    def _create_metric_descriptor(self):
        """Create custom metric descriptor for anomaly rate if it doesn't exist."""
        if not self._monitoring_client:
            return
        
        try:
            from google.cloud import monitoring_v3
            
            project_name = f"projects/{self.project_id}"
            
            descriptor = monitoring_v3.MetricDescriptor()
            descriptor.type = self.METRIC_TYPE
            descriptor.metric_kind = monitoring_v3.MetricDescriptor.MetricKind.GAUGE
            descriptor.value_type = monitoring_v3.MetricDescriptor.ValueType.DOUBLE
            descriptor.description = "Anomaly detection rate per minute (60-second rolling window)"
            descriptor.display_name = "Anomaly Detection Rate"
            descriptor.unit = "anomalies/min"
            
            self._monitoring_client.create_metric_descriptor(
                name=project_name,
                metric_descriptor=descriptor,
                timeout=10.0
            )
            print(f"[Monitor] Created metric descriptor: {self.METRIC_TYPE}")
            
        except Exception as e:
            # Ignore if descriptor already exists
            if "already exists" in str(e).lower():
                print(f"[Monitor] Metric descriptor already exists: {self.METRIC_TYPE}")
            else:
                print(f"[Monitor] Failed to create metric descriptor: {e}")
    
    def _start_reporter(self):
        """Start background thread for periodic metric reporting."""
        self._running = True
        self._reporter_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._reporter_thread.start()
    
    def _report_loop(self):
        """Background loop that reports metrics every 60 seconds."""
        while self._running:
            time.sleep(60)
            if self._running:
                self._report_anomaly_rate()
    
    def _report_anomaly_rate(self):
        """Calculate and report the anomaly rate metric."""
        with self._lock:
            # Clean up old entries outside the window
            cutoff = time.time() - self.WINDOW_SECONDS
            while self._anomaly_times and self._anomaly_times[0] < cutoff:
                self._anomaly_times.popleft()
            
            # Calculate rate (anomalies per minute)
            anomaly_rate = len(self._anomaly_times)
        
        print(f"[Monitor] Anomaly rate: {anomaly_rate}/min (last 60s)")
        
        # Export to Cloud Monitoring
        if self._monitoring_client:
            self._write_metric(anomaly_rate)
    
    def _write_metric(self, value: float):
        """
        Write anomaly rate metric to Cloud Monitoring.
        
        Args:
            value: Anomaly rate value (anomalies per minute)
        """
        if not self._monitoring_client:
            return
        
        try:
            from google.cloud import monitoring_v3
            from google.protobuf import timestamp_pb2
            
            project_name = f"projects/{self.project_id}"
            
            series = monitoring_v3.TimeSeries()
            series.metric.type = self.METRIC_TYPE
            series.resource.type = "global"
            
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)
            
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": seconds, "nanos": nanos}
            })
            
            point = monitoring_v3.Point({
                "interval": interval,
                "value": {"double_value": value}
            })
            
            series.points = [point]
            
            self._monitoring_client.create_time_series(
                name=project_name,
                time_series=[series],
                timeout=10.0  # 10 second timeout to prevent 504 errors
            )
            
        except Exception as e:
            print(f"[Monitor] Failed to write metric: {e}")
    
    def log_anomaly(
        self,
        timestamp: float,
        vibration: float,
        confidence: float,
        device_id: str
    ):
        """
        Log an anomaly detection event.
        
        Args:
            timestamp: Unix timestamp of the reading
            vibration: Vibration sensor value
            confidence: Model confidence score (0-1)
            device_id: Device identifier
        """
        self.total_anomalies += 1
        
        # Add to rolling window
        with self._lock:
            self._anomaly_times.append(time.time())
        
        # Create structured log entry
        log_entry = {
            'event_type': 'anomaly_detected',
            'timestamp': datetime.utcfromtimestamp(timestamp).isoformat(),
            'device_id': device_id,
            'vibration': vibration,
            'confidence': confidence,
            'total_anomalies': self.total_anomalies
        }
        
        # Log to Cloud Logging or stdout
        if self._logger:
            try:
                self._logger.log_struct(
                    log_entry,
                    severity='WARNING',
                    labels={
                        'device_id': device_id,
                        'event_type': 'anomaly'
                    }
                )
            except Exception as e:
                print(f"[Monitor] Cloud Logging error: {e}")
                self._log_to_stdout(log_entry)
        else:
            self._log_to_stdout(log_entry)
    
    def _log_to_stdout(self, entry: dict):
        """Fallback logging to stdout when Cloud Logging unavailable."""
        import json
        print(f"[ANOMALY] {json.dumps(entry)}")
    
    def log_reading(self, device_id: str):
        """
        Track a reading for statistics (normal or anomaly).
        
        Args:
            device_id: Device identifier
        """
        self.total_readings += 1
    
    def get_stats(self) -> dict:
        """Get current monitoring statistics."""
        with self._lock:
            # Clean up old entries
            cutoff = time.time() - self.WINDOW_SECONDS
            while self._anomaly_times and self._anomaly_times[0] < cutoff:
                self._anomaly_times.popleft()
            
            current_rate = len(self._anomaly_times)
        
        return {
            'total_anomalies': self.total_anomalies,
            'total_readings': self.total_readings,
            'anomaly_rate_per_minute': current_rate,
            'anomaly_percentage': (self.total_anomalies / self.total_readings * 100)
                                  if self.total_readings > 0 else 0
        }
    
    def flush(self):
        """Flush any pending logs/metrics and stop the reporter."""
        self._running = False
        
        # Final metric report
        self._report_anomaly_rate()
        
        # Flush Cloud Logging
        if self._logging_client:
            try:
                # The Python client auto-flushes, but we can force it
                pass
            except Exception:
                pass
        
        print("[Monitor] Flushed and stopped")

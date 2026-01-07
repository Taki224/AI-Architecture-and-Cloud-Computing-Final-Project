"""
Carbon Emissions Monitoring for Anomaly Detection Services
Uses CodeCarbon to measure gCO₂e emissions and exports to GCP Cloud Monitoring

This module provides:
- Per-batch/per-request carbon emission tracking
- Custom metrics export to Cloud Monitoring with service/mode labels
- Total and per-inference emissions tracking
"""
import os
import time
import threading
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

# CodeCarbon for emissions tracking
from codecarbon import EmissionsTracker

# GCP Cloud Monitoring
from google.cloud import monitoring_v3
from google.api import metric_pb2 as api_metric
from google.api import label_pb2


class CarbonMonitor:
    """
    Monitors and tracks carbon emissions for ML inference workloads.
    
    Features:
    - Wraps CodeCarbon EmissionsTracker for per-inference tracking
    - Exports custom metrics to GCP Cloud Monitoring
    - Labels metrics by service (heavy-model/light-model) and mode (PERFORMANCE/ECO)
    - Tracks total cumulative emissions per service
    """
    
    # Custom metric types for Cloud Monitoring
    METRIC_EMISSIONS = "custom.googleapis.com/carbon/emissions_gco2e"
    METRIC_TOTAL_EMISSIONS = "custom.googleapis.com/carbon/total_emissions_gco2e"
    METRIC_INFERENCE_COUNT = "custom.googleapis.com/carbon/inference_count"
    
    # Reporting interval in seconds
    REPORT_INTERVAL = 60
    
    def __init__(
        self,
        project_id: str,
        service_name: str,
        mode: str,
        country_iso_code: str = "AUT",  # Austria default, adjust as needed
        region: Optional[str] = None
    ):
        """
        Initialize the carbon monitor.
        
        Args:
            project_id: GCP project ID for Cloud Monitoring
            service_name: Service identifier ('heavy-model' or 'light-model')
            mode: Processing mode ('PERFORMANCE' or 'ECO')
            country_iso_code: ISO 3166-1 alpha-3 country code for carbon intensity
            region: Cloud region for more accurate carbon intensity (optional)
        """
        self.project_id = project_id
        self.service_name = service_name
        self.mode = mode
        self.country_iso_code = country_iso_code
        self.region = region
        
        # Cumulative tracking
        self.total_emissions_kg = 0.0
        self.total_inferences = 0
        self._lock = threading.Lock()
        
        # Pending emissions to report (accumulated between reporting intervals)
        self._pending_emissions_kg = 0.0
        self._pending_inferences = 0
        
        # Initialize Cloud Monitoring client
        self._monitoring_client = None
        self._init_monitoring()
        
        # Start background reporter thread
        self._running = False
        self._reporter_thread = None
        self._start_reporter()
        
        print(f"[CarbonMonitor] Initialized for {service_name} ({mode} mode)")
        print(f"[CarbonMonitor] Country: {country_iso_code}, Project: {project_id}")
    
    def _init_monitoring(self):
        """Initialize GCP Cloud Monitoring client and create metric descriptors."""
        try:
            self._monitoring_client = monitoring_v3.MetricServiceClient()
            print("[CarbonMonitor] Cloud Monitoring client initialized")
            self._create_metric_descriptors()
        except Exception as e:
            print(f"[CarbonMonitor] Failed to init Cloud Monitoring: {e}")
    
    def _create_metric_descriptors(self):
        """Create custom metric descriptors if they don't exist."""
        if not self._monitoring_client:
            return
        
        project_name = f"projects/{self.project_id}"
        
        descriptors = [
            {
                "type": self.METRIC_EMISSIONS,
                "display_name": "Carbon Emissions per Period",
                "description": "Carbon emissions in gCO2e per reporting period",
                "unit": "gCO2e"
            },
            {
                "type": self.METRIC_TOTAL_EMISSIONS,
                "display_name": "Total Carbon Emissions",
                "description": "Cumulative carbon emissions in gCO2e",
                "unit": "gCO2e"
            },
            {
                "type": self.METRIC_INFERENCE_COUNT,
                "display_name": "Inference Count",
                "description": "Number of inferences per reporting period",
                "unit": "1"
            }
        ]
        
        for desc_config in descriptors:
            try:
                descriptor = api_metric.MetricDescriptor()
                descriptor.type = desc_config["type"]
                descriptor.metric_kind = api_metric.MetricDescriptor.MetricKind.GAUGE
                descriptor.value_type = api_metric.MetricDescriptor.ValueType.DOUBLE if "emissions" in desc_config["type"] else api_metric.MetricDescriptor.ValueType.INT64
                descriptor.description = desc_config["description"]
                descriptor.display_name = desc_config["display_name"]
                descriptor.unit = desc_config["unit"]
                
                # Add labels
                descriptor.labels.append(label_pb2.LabelDescriptor(
                    key="service",
                    value_type=label_pb2.LabelDescriptor.ValueType.STRING,
                    description="Service name (heavy-model or light-model)"
                ))
                descriptor.labels.append(label_pb2.LabelDescriptor(
                    key="mode",
                    value_type=label_pb2.LabelDescriptor.ValueType.STRING,
                    description="Processing mode (PERFORMANCE or ECO)"
                ))
                
                self._monitoring_client.create_metric_descriptor(
                    name=project_name,
                    metric_descriptor=descriptor,
                    timeout=10.0
                )
                print(f"[CarbonMonitor] Created metric descriptor: {desc_config['type']}")
                
            except Exception as e:
                # Ignore if descriptor already exists
                if "already exists" in str(e).lower():
                    print(f"[CarbonMonitor] Metric descriptor already exists: {desc_config['type']}")
                else:
                    print(f"[CarbonMonitor] Failed to create descriptor {desc_config['type']}: {e}")
    
    def _start_reporter(self):
        """Start background thread for periodic metric reporting."""
        self._running = True
        self._reporter_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._reporter_thread.start()
        print(f"[CarbonMonitor] Background reporter thread started (daemon={self._reporter_thread.daemon})")
    
    def _report_loop(self):
        """Background loop that reports aggregated metrics periodically."""
        print(f"[CarbonMonitor] Reporter thread started, will report every {self.REPORT_INTERVAL}s")
        while self._running:
            time.sleep(self.REPORT_INTERVAL)
            if self._running:
                print(f"[CarbonMonitor] Reporter waking up to flush metrics...")
                self._flush_pending_metrics()
    
    def _flush_pending_metrics(self):
        """Flush pending emissions to Cloud Monitoring."""
        with self._lock:
            print(f"[CarbonMonitor] Checking pending metrics: {self._pending_emissions_kg:.6f} kg, {self._pending_inferences} inferences")
            if self._pending_emissions_kg <= 0 and self._pending_inferences <= 0:
                print(f"[CarbonMonitor] No pending metrics to report, skipping")
                return
            
            emissions_to_report = self._pending_emissions_kg
            inferences_to_report = self._pending_inferences
            total_emissions = self.total_emissions_kg
            
            # Reset pending counters
            self._pending_emissions_kg = 0.0
            self._pending_inferences = 0
        
        # Convert kg to grams for more readable values
        emissions_grams = emissions_to_report * 1000
        total_grams = total_emissions * 1000
        
        print(f"[CarbonMonitor] Reporting: {emissions_grams:.6f} gCO₂e "
              f"({inferences_to_report} inferences) | Total: {total_grams:.6f} gCO₂e")
        
        # Write metrics to Cloud Monitoring
        if self._monitoring_client:
            self._write_emissions_metric(emissions_grams)
            self._write_total_emissions_metric(total_grams)
            self._write_inference_count_metric(inferences_to_report)
    
    def _write_emissions_metric(self, value_grams: float):
        """Write per-period emissions to Cloud Monitoring."""
        self._write_metric(
            metric_type=self.METRIC_EMISSIONS,
            value=value_grams,
            value_type="double_value"
        )
    
    def _write_total_emissions_metric(self, value_grams: float):
        """Write cumulative total emissions to Cloud Monitoring."""
        self._write_metric(
            metric_type=self.METRIC_TOTAL_EMISSIONS,
            value=value_grams,
            value_type="double_value"
        )
    
    def _write_inference_count_metric(self, count: int):
        """Write inference count to Cloud Monitoring."""
        self._write_metric(
            metric_type=self.METRIC_INFERENCE_COUNT,
            value=count,
            value_type="int64_value"
        )
    
    def _write_metric(self, metric_type: str, value: float, value_type: str = "double_value"):
        """
        Write a metric to GCP Cloud Monitoring.
        
        Args:
            metric_type: Full metric type path
            value: Metric value
            value_type: 'double_value' or 'int64_value'
        """
        if not self._monitoring_client:
            return
        
        try:
            project_name = f"projects/{self.project_id}"
            
            series = monitoring_v3.TimeSeries()
            series.metric.type = metric_type
            series.metric.labels["service"] = self.service_name
            series.metric.labels["mode"] = self.mode
            
            # Use global resource type
            series.resource.type = "global"
            series.resource.labels["project_id"] = self.project_id
            
            # Set timestamp
            now = time.time()
            seconds = int(now)
            nanos = int((now - seconds) * 10**9)
            
            interval = monitoring_v3.TimeInterval({
                "end_time": {"seconds": seconds, "nanos": nanos}
            })
            
            # Set value based on type
            if value_type == "int64_value":
                point_value = {"int64_value": int(value)}
            else:
                point_value = {"double_value": float(value)}
            
            point = monitoring_v3.Point({
                "interval": interval,
                "value": point_value
            })
            
            series.points = [point]
            
            # Try to write with retries
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    self._monitoring_client.create_time_series(
                        name=project_name,
                        time_series=[series],
                        timeout=5.0  # Shorter timeout for faster failures
                    )
                    print(f"[CarbonMonitor] ✓ Wrote metric {metric_type.split('/')[-1]}: {value}")
                    break  # Success
                except Exception as retry_err:
                    if attempt < max_retries and "504" in str(retry_err):
                        print(f"[CarbonMonitor] Retry {attempt+1}/{max_retries} for {metric_type.split('/')[-1]} due to 504")
                        continue
                    else:
                        # Give up after retries
                        raise retry_err
            
        except Exception as e:
            # Log all errors for debugging
            print(f"[CarbonMonitor] ✗ Failed to write metric {metric_type.split('/')[-1]}: {e}")
    
    @contextmanager
    def track_inference(self, batch_size: int = 1):
        """
        Context manager to track carbon emissions for an inference operation.
        
        Usage:
            with carbon_monitor.track_inference(batch_size=10):
                results = model.predict(batch)
        
        Args:
            batch_size: Number of samples in this inference batch
            
        Yields:
            EmissionsTracker instance (can be ignored)
        """
        # Set country code via environment variable for CodeCarbon
        os.environ['CODECARBON_COUNTRY_ISO_CODE'] = self.country_iso_code
        
        # Create a temporary tracker for this inference
        tracker = EmissionsTracker(
            project_name=f"{self.service_name}-inference",
            measure_power_secs=0.5,  # Measure every 0.5s for short inferences
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
            log_level="error",  # Suppress verbose output
            tracking_mode="process"
        )
        
        try:
            tracker.start()
            yield tracker
        finally:
            emissions_kg = tracker.stop()
            
            if emissions_kg is None:
                emissions_kg = 0.0
            
            # Update totals
            with self._lock:
                self.total_emissions_kg += emissions_kg
                self.total_inferences += batch_size
                self._pending_emissions_kg += emissions_kg
                self._pending_inferences += batch_size
                
                print(f"[CarbonMonitor] Accumulated: pending={self._pending_emissions_kg:.9f} kg, "
                      f"total={self.total_emissions_kg:.9f} kg, inferences={self.total_inferences}")
            
            # Log if significant
            if emissions_kg > 0:
                emissions_grams = emissions_kg * 1000
                per_sample_grams = emissions_grams / batch_size if batch_size > 0 else 0
                print(f"[CarbonMonitor] Batch: {emissions_grams:.6f} gCO₂e "
                      f"({per_sample_grams:.6f} gCO₂e/sample, n={batch_size})")
    
    def track_single_inference(self) -> float:
        """
        Track a single inference and return emissions in grams CO₂e.
        
        This is a simpler alternative to the context manager for single predictions.
        Note: For very fast inferences, emissions may be negligible or zero.
        
        Returns:
            Emissions in grams CO₂e (may be 0 for very fast operations)
        """
        # Set country code via environment variable for CodeCarbon
        os.environ['CODECARBON_COUNTRY_ISO_CODE'] = self.country_iso_code
        
        tracker = EmissionsTracker(
            project_name=f"{self.service_name}-single",
            measure_power_secs=0.1,
            save_to_file=False,
            save_to_api=False,
            save_to_logger=False,
            log_level="error",
            tracking_mode="process"
        )
        
        tracker.start()
        # Caller should do inference between start/stop
        # This method is meant to be called as a wrapper
        return tracker
    
    def record_emissions(self, emissions_kg: float, batch_size: int = 1):
        """
        Manually record emissions (useful when using external tracker).
        
        Args:
            emissions_kg: Emissions in kilograms CO₂e
            batch_size: Number of samples processed
        """
        if emissions_kg is None:
            emissions_kg = 0.0
            
        with self._lock:
            self.total_emissions_kg += emissions_kg
            self.total_inferences += batch_size
            self._pending_emissions_kg += emissions_kg
            self._pending_inferences += batch_size
    
    def get_stats(self) -> dict:
        """Get current carbon monitoring statistics."""
        with self._lock:
            total_grams = self.total_emissions_kg * 1000
            avg_per_inference = (total_grams / self.total_inferences) if self.total_inferences > 0 else 0
        
        return {
            'service': self.service_name,
            'mode': self.mode,
            'total_emissions_gco2e': total_grams,
            'total_inferences': self.total_inferences,
            'avg_emissions_per_inference_gco2e': avg_per_inference,
            'country': self.country_iso_code
        }
    
    def flush(self):
        """Flush pending metrics and stop the background reporter."""
        self._running = False
        self._flush_pending_metrics()
        print(f"[CarbonMonitor] Flushed. Total: {self.total_emissions_kg * 1000:.6f} gCO₂e "
              f"over {self.total_inferences} inferences")
    
    def __del__(self):
        """Cleanup on destruction."""
        self._running = False

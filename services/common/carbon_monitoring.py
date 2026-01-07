"""
Carbon Emissions Monitoring for Anomaly Detection Services
Tracks gCO₂e emissions and exports to GCP Cloud Monitoring

This module provides:
- Per-batch/per-request carbon emission tracking (estimation-based for cloud reliability)
- Custom metrics export to Cloud Monitoring with service/mode labels
- Total and per-inference emissions tracking
- Optional CodeCarbon integration (set CODECARBON_ENABLED=true to enable)
"""
import os
import time
import threading
from datetime import datetime
from typing import Optional
from contextlib import contextmanager

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
        
        # Track last metric write timestamps per metric type to avoid "out of order" errors
        self._last_metric_write_times = {}
        self._min_metric_interval = 60  # seconds (GCP requires 60s minimum for GAUGE metrics)
        
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
                
            except Exception as e:
                # Ignore if descriptor already exists or other errors
                pass
    
    def _start_reporter(self):
        """Start background thread for periodic metric reporting."""
        self._running = True
        self._reporter_thread = threading.Thread(target=self._report_loop, daemon=True)
        self._reporter_thread.start()
    
    def _report_loop(self):
        """Background loop that reports aggregated metrics periodically."""
        while self._running:
            time.sleep(self.REPORT_INTERVAL)
            if self._running:
                self._flush_pending_metrics()
    
    def _flush_pending_metrics(self):
        """Flush pending emissions to Cloud Monitoring."""
        with self._lock:
            if self._pending_emissions_kg <= 0 and self._pending_inferences <= 0:
                return  # Nothing to report, skip silently
            
            emissions_to_report = self._pending_emissions_kg
            inferences_to_report = self._pending_inferences
            total_emissions = self.total_emissions_kg
            
            # Reset pending counters
            self._pending_emissions_kg = 0.0
            self._pending_inferences = 0
        
        # Convert kg to grams for more readable values
        emissions_grams = emissions_to_report * 1000
        total_grams = total_emissions * 1000
        
        print(f"[Carbon] Reporting: {emissions_grams:.4f} gCO₂e from {inferences_to_report} inferences | Cumulative: {total_grams:.4f} gCO₂e")
        
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
        
        # Ensure minimum interval between writes for the same metric type
        now = time.time()
        with self._lock:
            last_write = self._last_metric_write_times.get(metric_type, 0)
            if now - last_write < self._min_metric_interval:
                return  # Skip - too soon since last write
            self._last_metric_write_times[metric_type] = now
        
        # Fire and forget in background thread to avoid blocking
        def write_async():
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
                write_time = time.time()
                seconds = int(write_time)
                nanos = int((write_time - seconds) * 10**9)
                
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
                
                self._monitoring_client.create_time_series(
                    name=project_name,
                    time_series=[series],
                    timeout=30.0  # Longer timeout for cloud environments
                )
                
            except Exception as e:
                err_str = str(e).lower()
                if "504" in err_str or "deadline" in err_str:
                    print(f"[Carbon] Metric write timeout - skipping")
                elif "must be written in order" in err_str:
                    print(f"[Carbon] Metric out of order - skipping")
                elif "more frequently" in err_str:
                    print(f"[Carbon] Metric rate limited - skipping")
                else:
                    print(f"[Carbon] Metric write failed: {e}")
        
        # Run in background thread
        threading.Thread(target=write_async, daemon=True).start()
    
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
            None (CodeCarbon disabled for cloud - uses estimation)
        """
        # CodeCarbon's EmissionsTracker blocks on concurrent/repeated calls in cloud environments.
        # Use estimation-based approach instead for reliability.
        # Set CODECARBON_ENABLED=true to enable actual tracking (not recommended for production)
        use_codecarbon = os.getenv('CODECARBON_ENABLED', 'false').lower() == 'true'
        
        tracker = None
        if use_codecarbon:
            os.environ['CODECARBON_COUNTRY_ISO_CODE'] = self.country_iso_code
            try:
                from codecarbon import EmissionsTracker
                tracker = EmissionsTracker(
                    project_name=f"{self.service_name}-inference",
                    measure_power_secs=1,
                    save_to_file=False,
                    save_to_api=False,
                    save_to_logger=False,
                    log_level="error",
                    tracking_mode="machine",
                    allow_multiple_runs=True,
                )
                tracker.start()
            except Exception:
                tracker = None
        
        try:
            yield tracker
        finally:
            emissions_kg = 0.0
            
            if tracker:
                try:
                    emissions_kg = tracker.stop() or 0.0
                except Exception:
                    emissions_kg = 0.0
            
            # Use estimation-based calculation (more reliable in cloud environments)
            # Based on typical cloud vCPU: ~15W TDP, ~0.1 kgCO2e/kWh (low-carbon grid)
            # Estimate: ~0.000005 gCO2e per inference
            if emissions_kg == 0 or emissions_kg is None:
                emissions_kg = 0.000000005 * batch_size
            
            # Update totals
            with self._lock:
                self.total_emissions_kg += emissions_kg
                self.total_inferences += batch_size
                self._pending_emissions_kg += emissions_kg
                self._pending_inferences += batch_size
    
    def track_single_inference(self) -> float:
        """
        Track a single inference and return estimated emissions in grams CO₂e.
        
        This is a simpler alternative to the context manager for single predictions.
        Uses estimation-based calculation for cloud reliability.
        
        Returns:
            Emissions in grams CO₂e (estimated)
        """
        # Use estimation-based approach for reliability
        emissions_kg = 0.000000005  # ~0.005 mg CO2e per inference
        
        with self._lock:
            self.total_emissions_kg += emissions_kg
            self.total_inferences += 1
            self._pending_emissions_kg += emissions_kg
            self._pending_inferences += 1
        
        return emissions_kg * 1000  # Return in grams
    
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
    
    def __del__(self):
        """Cleanup on destruction."""
        self._running = False

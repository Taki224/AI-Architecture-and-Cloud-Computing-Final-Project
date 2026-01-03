"""
Local Sensor GUI - Edge Device Vibration Sensor Simulator
Carbon-Aware IoT Anomaly Detection System

This is the unified edge device application combining:
- Vibration sensor simulation with anomaly injection
- Real-time Tkinter GUI with Matplotlib visualization
- Google Cloud Pub/Sub integration for PERFORMANCE mode
- Local REST API integration for ECO mode

Supports two operation modes:
- PERFORMANCE: Streams data via Pub/Sub to cloud heavy model (GCP)
- ECO: Uses local REST API with light model (Docker)
"""

import json
import os
import time
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, List, Optional, Tuple
from datetime import datetime
from enum import Enum
from collections import deque

import numpy as np
import requests
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =============================================================================
# OPERATION MODE
# =============================================================================

class OperationMode(Enum):
    """Operation modes for carbon-aware processing"""
    PERFORMANCE = "PERFORMANCE"  # Low carbon - stream to cloud via Pub/Sub
    ECO = "ECO"  # High carbon - local inference via REST


# =============================================================================
# PUB/SUB CLIENT
# =============================================================================

class PubSubClient:
    """
    Base Pub/Sub client with automatic emulator detection.
    Uses google-cloud-pubsub library for GCP integration.
    """
    
    def __init__(self):
        """Initialize the Pub/Sub client with project and emulator detection."""
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')
        self.emulator_host = os.getenv('PUBSUB_EMULATOR_HOST')
        self._publisher = None
        self._subscriber = None
        self._connected = False
        
        if self.emulator_host:
            print(f"[PubSub] Using emulator at {self.emulator_host}")
        else:
            print(f"[PubSub] Using GCP project: {self.project_id}")
    
    def _get_publisher(self):
        """Lazy initialization of publisher client."""
        if self._publisher is None:
            try:
                from google.cloud import pubsub_v1
                self._publisher = pubsub_v1.PublisherClient()
                self._connected = True
            except Exception as e:
                print(f"[PubSub] Failed to create publisher: {e}")
                self._connected = False
        return self._publisher
    
    def _get_subscriber(self):
        """Lazy initialization of subscriber client."""
        if self._subscriber is None:
            try:
                from google.cloud import pubsub_v1
                self._subscriber = pubsub_v1.SubscriberClient()
                self._connected = True
            except Exception as e:
                print(f"[PubSub] Failed to create subscriber: {e}")
                self._connected = False
        return self._subscriber
    
    @property
    def is_connected(self) -> bool:
        """Check if Pub/Sub client is connected."""
        return self._connected


class SensorPublisher(PubSubClient):
    """
    Publishes batched sensor readings to Pub/Sub.
    Batches 10 readings (1 second of data) before publishing.
    """
    
    BATCH_SIZE = 10  # Number of readings per batch
    
    def __init__(self, device_id: str = None):
        """
        Initialize the sensor publisher.
        
        Args:
            device_id: Unique identifier for this edge device
        """
        super().__init__()
        self.device_id = device_id or os.getenv('DEVICE_ID', 'edge-001')
        self.topic_name = os.getenv('PUBSUB_SENSOR_TOPIC', 'sensor-data')
        self._batch_buffer: List[Dict] = []
        self._lock = threading.Lock()
        
        print(f"[SensorPublisher] Device: {self.device_id}, Topic: {self.topic_name}")
    
    def _get_topic_path(self) -> str:
        """Get the full topic path."""
        publisher = self._get_publisher()
        if publisher:
            return publisher.topic_path(self.project_id, self.topic_name)
        return None
    
    def add_reading(self, timestamp: float, vibration: float) -> bool:
        """
        Add a sensor reading to the batch buffer.
        Automatically publishes when batch is full.
        
        Args:
            timestamp: Unix timestamp of the reading
            vibration: Vibration sensor value
            
        Returns:
            True if batch was published, False otherwise
        """
        reading = {
            'timestamp': timestamp,
            'vibration': vibration
        }
        
        with self._lock:
            self._batch_buffer.append(reading)
            
            if len(self._batch_buffer) >= self.BATCH_SIZE:
                return self._publish_batch()
        
        return False
    
    def _publish_batch(self) -> bool:
        """
        Publish the current batch to Pub/Sub.
        
        Returns:
            True if publish succeeded, False otherwise
        """
        if not self._batch_buffer:
            return False
        
        publisher = self._get_publisher()
        topic_path = self._get_topic_path()
        
        if not publisher or not topic_path:
            print("[SensorPublisher] Publisher not available, clearing batch")
            self._batch_buffer.clear()
            return False
        
        # Create batch message
        message = {
            'device_id': self.device_id,
            'readings': self._batch_buffer.copy(),
            'count': len(self._batch_buffer),
            'published_at': datetime.utcnow().isoformat()
        }
        
        try:
            # Publish message
            data = json.dumps(message).encode('utf-8')
            future = publisher.publish(topic_path, data)
            message_id = future.result(timeout=5)  # Wait up to 5 seconds
            
            print(f"[SensorPublisher] Published batch of {len(self._batch_buffer)} readings (msg_id: {message_id})")
            self._batch_buffer.clear()
            return True
            
        except Exception as e:
            print(f"[SensorPublisher] Failed to publish batch: {e}")
            self._batch_buffer.clear()  # Clear to avoid memory buildup
            return False
    
    def flush(self) -> bool:
        """
        Force publish any remaining readings in the buffer.
        
        Returns:
            True if publish succeeded or buffer was empty, False otherwise
        """
        with self._lock:
            if self._batch_buffer:
                return self._publish_batch()
        return True


class AnomalySubscriber(PubSubClient):
    """
    Subscribes to anomaly detection results from the cloud.
    Handles batched responses from the heavy model service.
    """
    
    def __init__(self, callback: Callable[[Dict], None] = None):
        """
        Initialize the anomaly subscriber.
        
        Args:
            callback: Function to call when anomaly results are received
        """
        super().__init__()
        self.subscription_name = os.getenv('PUBSUB_ANOMALY_SUBSCRIPTION', 'anomaly-results-sub')
        self.callback = callback
        self._streaming_future = None
        self._running = False
        
        print(f"[AnomalySubscriber] Subscription: {self.subscription_name}")
    
    def _get_subscription_path(self) -> str:
        """Get the full subscription path."""
        subscriber = self._get_subscriber()
        if subscriber:
            return subscriber.subscription_path(self.project_id, self.subscription_name)
        return None
    
    def _message_callback(self, message):
        """
        Internal callback for processing received messages.
        
        Args:
            message: Pub/Sub message object
        """
        try:
            # Decode message
            data = json.loads(message.data.decode('utf-8'))
            
            # Process batch of results
            readings = data.get('readings', [])
            anomaly_count = sum(1 for r in readings if r.get('is_anomaly', False))
            
            print(f"[AnomalySubscriber] Received {len(readings)} results, {anomaly_count} anomalies")
            
            # Call user callback if provided
            if self.callback:
                self.callback(data)
            
            # Acknowledge message after successful processing
            message.ack()
            
        except Exception as e:
            print(f"[AnomalySubscriber] Error processing message: {e}")
            # Still acknowledge to avoid redelivery loops
            message.ack()
    
    def start(self):
        """Start listening for anomaly results in a background thread."""
        if self._running:
            print("[AnomalySubscriber] Already running")
            return
        
        subscriber = self._get_subscriber()
        subscription_path = self._get_subscription_path()
        
        if not subscriber or not subscription_path:
            print("[AnomalySubscriber] Subscriber not available")
            return
        
        try:
            self._streaming_future = subscriber.subscribe(
                subscription_path,
                callback=self._message_callback
            )
            self._running = True
            print(f"[AnomalySubscriber] Listening on {subscription_path}")
            
        except Exception as e:
            print(f"[AnomalySubscriber] Failed to start: {e}")
            self._running = False
    
    def stop(self):
        """Stop listening for anomaly results."""
        if self._streaming_future:
            self._streaming_future.cancel()
            self._streaming_future = None
        self._running = False
        print("[AnomalySubscriber] Stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if subscriber is currently running."""
        return self._running


def create_publisher(device_id: str = None) -> Optional[SensorPublisher]:
    """
    Factory function to create a sensor publisher.
    Returns None if Pub/Sub is not available.
    
    Args:
        device_id: Optional device identifier
        
    Returns:
        SensorPublisher instance or None
    """
    try:
        publisher = SensorPublisher(device_id)
        # Test connection by getting publisher client
        if publisher._get_publisher():
            return publisher
        return None
    except Exception as e:
        print(f"[PubSub] Failed to create publisher: {e}")
        return None


def create_subscriber(callback: Callable[[Dict], None] = None) -> Optional[AnomalySubscriber]:
    """
    Factory function to create an anomaly subscriber.
    Returns None if Pub/Sub is not available.
    
    Args:
        callback: Function to call when results are received
        
    Returns:
        AnomalySubscriber instance or None
    """
    try:
        subscriber = AnomalySubscriber(callback)
        # Test connection by getting subscriber client
        if subscriber._get_subscriber():
            return subscriber
        return None
    except Exception as e:
        print(f"[PubSub] Failed to create subscriber: {e}")
        return None


# =============================================================================
# VIBRATION SENSOR SIMULATOR
# =============================================================================

class VibrationSensor:
    """
    Simulates a vibration sensor generating time-series data with anomalies.
    
    Data characteristics:
    - Normal operation: μ=0, σ=1 (normal distribution)
    - Anomalies: ~2-3 per minute at 100ms intervals
    - Small anomalies: 3-4σ deviation (easier to detect)
    - Large anomalies: 5-8σ deviation (obvious anomalies)
    
    Operation modes:
    - PERFORMANCE: Batches 10 readings, publishes via Pub/Sub to cloud
    - ECO: Sends individual readings to local REST API
    """
    
    def __init__(
        self,
        mean: float = 0.0,
        std_dev: float = 1.0,
        anomaly_rate: float = 0.003,  # ~2 per minute at 100ms intervals
        small_anomaly_range: Tuple[float, float] = (3.0, 4.0),
        large_anomaly_range: Tuple[float, float] = (5.0, 8.0),
        large_anomaly_ratio: float = 0.4  # 40% of anomalies are large
    ):
        """
        Initialize the vibration sensor simulator.
        
        Args:
            mean: Mean value for normal vibration readings
            std_dev: Standard deviation for normal readings
            anomaly_rate: Probability of generating an anomaly (0.003 ≈ 2/min)
            small_anomaly_range: Sigma range for small anomalies (min, max)
            large_anomaly_range: Sigma range for large anomalies (min, max)
            large_anomaly_ratio: Proportion of anomalies that are large
        """
        self.mean = mean
        self.std_dev = std_dev
        self.anomaly_rate = anomaly_rate
        self.small_anomaly_range = small_anomaly_range
        self.large_anomaly_range = large_anomaly_range
        self.large_anomaly_ratio = large_anomaly_ratio
        
        self.mode = OperationMode.PERFORMANCE
        self.is_running = False
        self.timestamp = 0
        
        # ML Model API configuration (for ECO mode)
        self.ml_api_url = "http://localhost:5001"  # Default for local testing
        self.use_light_model = True  # Use light model for ECO mode
        
        # Pub/Sub configuration (for PERFORMANCE mode)
        self._pubsub_publisher: Optional[SensorPublisher] = None
        self._pubsub_subscriber: Optional[AnomalySubscriber] = None
        
        # Statistics tracking
        self.total_readings = 0
        self.total_anomalies = 0
        self.small_anomalies = 0
        self.large_anomalies = 0
        self.api_anomalies_detected = 0  # Anomalies detected by local API (ECO mode)
        self.cloud_anomalies_detected = 0  # Anomalies detected by cloud (PERFORMANCE mode)
        self.pubsub_batches_sent = 0  # Number of batches sent via Pub/Sub
        
    def generate_reading(self) -> Tuple[float, bool, str]:
        """
        Generate a single sensor reading with possible anomaly.
        
        Returns:
            Tuple of (value, is_anomaly, severity)
            - value: The vibration reading
            - is_anomaly: True if this is an anomaly
            - severity: 'normal', 'small', or 'large'
        """
        self.total_readings += 1
        self.timestamp = time.time()
        
        # Determine if this reading is an anomaly
        is_anomaly = np.random.random() < self.anomaly_rate
        
        if is_anomaly:
            self.total_anomalies += 1
            
            # Determine anomaly severity
            is_large = np.random.random() < self.large_anomaly_ratio
            
            if is_large:
                # Large anomaly: 5-8 sigma deviation
                self.large_anomalies += 1
                sigma_multiplier = np.random.uniform(*self.large_anomaly_range)
                severity = 'large'
            else:
                # Small anomaly: 3-4 sigma deviation
                self.small_anomalies += 1
                sigma_multiplier = np.random.uniform(*self.small_anomaly_range)
                severity = 'small'
            
            # Random direction (positive or negative)
            direction = np.random.choice([-1, 1])
            value = self.mean + direction * sigma_multiplier * self.std_dev
            
            return value, True, severity
        else:
            # Normal reading
            value = np.random.normal(self.mean, self.std_dev)
            return value, False, 'normal'
    
    def set_mode(self, mode: OperationMode):
        """
        Set the operation mode (PERFORMANCE or ECO).
        
        Args:
            mode: The operation mode to switch to
        """
        self.mode = mode
        print(f"[Sensor] Switched to {mode.value} mode")
    
    def start(self):
        """Start the sensor simulation"""
        self.is_running = True
        print("[Sensor] Started")
    
    def stop(self):
        """Stop the sensor simulation"""
        self.is_running = False
        print("[Sensor] Stopped")
    
    def set_ml_api_url(self, url: str):
        """
        Set the ML model API URL.
        
        Args:
            url: The base URL of the ML model API (e.g., 'http://ml-model:5000')
        """
        self.ml_api_url = url
        print(f"[Sensor] ML API URL set to: {url}")
    
    def get_statistics(self) -> dict:
        """
        Get current statistics about sensor readings.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'total_readings': self.total_readings,
            'total_anomalies': self.total_anomalies,
            'small_anomalies': self.small_anomalies,
            'large_anomalies': self.large_anomalies,
            'api_anomalies_detected': self.api_anomalies_detected,
            'cloud_anomalies_detected': self.cloud_anomalies_detected,
            'pubsub_batches_sent': self.pubsub_batches_sent,
            'anomaly_rate': (self.total_anomalies / self.total_readings * 100) 
                           if self.total_readings > 0 else 0
        }
    
    # ========== Pub/Sub Integration Methods ==========
    
    def set_pubsub_publisher(self, publisher: SensorPublisher):
        """
        Set the Pub/Sub publisher for PERFORMANCE mode.
        
        Args:
            publisher: SensorPublisher instance
        """
        self._pubsub_publisher = publisher
        print("[Sensor] Pub/Sub publisher configured")
    
    def set_pubsub_subscriber(self, subscriber: AnomalySubscriber):
        """
        Set the Pub/Sub subscriber for receiving cloud results.
        
        Args:
            subscriber: AnomalySubscriber instance
        """
        self._pubsub_subscriber = subscriber
        print("[Sensor] Pub/Sub subscriber configured")
    
    def publish_to_pubsub(self, value: float) -> bool:
        """
        Publish sensor reading to Pub/Sub for cloud processing.
        Called in PERFORMANCE mode.
        
        Args:
            value: The sensor reading value
            
        Returns:
            True if batch was published, False otherwise
        """
        if not self._pubsub_publisher:
            return False
        
        try:
            batch_published = self._pubsub_publisher.add_reading(
                timestamp=self.timestamp,
                vibration=value
            )
            
            if batch_published:
                self.pubsub_batches_sent += 1
                print(f"[Sensor] Batch #{self.pubsub_batches_sent} sent to cloud")
            
            return batch_published
            
        except Exception as e:
            print(f"[Sensor] ⚠️  Pub/Sub publish error: {e}")
            return False
    
    @property
    def pubsub_available(self) -> bool:
        """Check if Pub/Sub is available for PERFORMANCE mode."""
        return self._pubsub_publisher is not None and self._pubsub_publisher.is_connected
    
    def run_local_inference(self, value: float) -> Dict[str, any]:
        """
        Run anomaly detection using local ML model API.
        Called in ECO mode.
        
        Args:
            value: The sensor reading to analyze
        
        Returns:
            Dictionary with prediction results or None if API call fails
        
        Sends data to ML model API for inference:
        - Uses light model by default for ECO mode efficiency
        - Returns anomaly prediction and score
        - Handles connection errors gracefully
        """
        try:
            # Prepare payload
            payload = {
                "value": float(value),
                "model_type": "light" if self.use_light_model else "heavy"
            }
            
            # Debug: log API call
            print(f"[DEBUG] Calling API: {self.ml_api_url}/predict with value={value:.4f}")
            
            # Call ML model API
            response = requests.post(
                f"{self.ml_api_url}/predict",
                json=payload,
                timeout=2  # 2 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Track API-detected anomalies
                if result.get('is_anomaly', False):
                    self.api_anomalies_detected += 1
                
                return result
            else:
                print(f"[Sensor] API error: {response.status_code}")
                return None
                
        except requests.exceptions.ConnectionError:
            print(f"[Sensor] ⚠️  ML API not reachable at {self.ml_api_url}")
            return None
        except requests.exceptions.Timeout:
            print(f"[Sensor] ⚠️  ML API timeout")
            return None
        except Exception as e:
            print(f"[Sensor] ⚠️  Error calling ML API: {e}")
            return None


# =============================================================================
# GUI APPLICATION
# =============================================================================

class Application:
    """
    Main GUI application for vibration sensor visualization.
    Displays real-time chart with color-coded anomalies.
    Supports Pub/Sub status display for PERFORMANCE mode.
    """
    
    def __init__(self, root: tk.Tk, sensor: VibrationSensor, pubsub_connected: bool = False):
        """
        Initialize the GUI application.
        
        Args:
            root: The tkinter root window
            sensor: The VibrationSensor instance
            pubsub_connected: Whether Pub/Sub is connected for PERFORMANCE mode
        """
        self.root = root
        self.sensor = sensor
        self.pubsub_connected = pubsub_connected
        
        # Configure window
        self.root.title("Edge Device - Vibration Sensor Simulator")
        self.root.geometry("1200x700")
        
        # Data storage for visualization (30 seconds at 100ms = 300 points)
        self.max_points = 300
        self.times = deque(maxlen=self.max_points)
        self.values = deque(maxlen=self.max_points)
        self.anomalies = deque(maxlen=self.max_points)
        self.severities = deque(maxlen=self.max_points)
        
        self.start_time = time.time()
        self.update_interval = 100  # milliseconds
        
        # Create GUI components
        self._create_widgets()
        
        # Animation for real-time updates
        self.animation = None
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # ========== Control Panel ==========
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, padx=5)
        
        self.start_button = ttk.Button(
            button_frame,
            text="▶ Start Sensor",
            command=self._start_sensor,
            width=15
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⏸ Stop Sensor",
            command=self._stop_sensor,
            state=tk.DISABLED,
            width=15
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Carbon Mode Toggle
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=0, column=1, padx=20)
        
        ttk.Label(mode_frame, text="Carbon Mode:").grid(row=0, column=0, padx=5)
        
        self.mode_var = tk.StringVar(value="PERFORMANCE")
        self.mode_button = ttk.Button(
            mode_frame,
            text="🌍 PERFORMANCE",
            command=self._toggle_mode,
            width=18
        )
        self.mode_button.grid(row=0, column=1, padx=5)
        
        # Statistics display
        stats_frame = ttk.Frame(control_frame)
        stats_frame.grid(row=0, column=2, padx=20)
        
        self.status_label = ttk.Label(
            stats_frame,
            text="Status: Stopped",
            font=('Arial', 10, 'bold')
        )
        self.status_label.grid(row=0, column=0, padx=5)
        
        self.anomaly_label = ttk.Label(
            stats_frame,
            text="Anomalies: 0 (0 small, 0 large)",
            font=('Arial', 9)
        )
        self.anomaly_label.grid(row=0, column=1, padx=5)
        
        # Pub/Sub connection status
        self.pubsub_label = ttk.Label(
            stats_frame,
            text="Pub/Sub: " + ("Connected ✓" if self.pubsub_connected else "Disconnected"),
            font=('Arial', 9),
            foreground='green' if self.pubsub_connected else 'gray'
        )
        self.pubsub_label.grid(row=0, column=2, padx=10)
        
        # ========== Chart Panel ==========
        chart_frame = ttk.LabelFrame(main_frame, text="Real-Time Vibration Data (30s Window)", padding="10")
        chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # Configure plot appearance
        self.ax.set_xlabel('Time (seconds)', fontsize=10)
        self.ax.set_ylabel('Vibration Amplitude', fontsize=10)
        self.ax.set_title('Vibration Sensor Data Stream', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(-10, 10)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Legend info
        legend_frame = ttk.Frame(chart_frame)
        legend_frame.grid(row=1, column=0, pady=5)
        
        ttk.Label(legend_frame, text="Note:", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5)
        
        ttk.Label(legend_frame, text="Raw sensor data shown continuously.", font=('Arial', 9)).grid(row=0, column=1, padx=5)
        ttk.Label(legend_frame, text="Subtle markers indicate injected anomalies (for ground truth).", 
                 font=('Arial', 9, 'italic'), foreground='gray').grid(row=0, column=2, padx=5)
        
    def _start_sensor(self):
        """Start the sensor simulation and data collection"""
        self.sensor.start()
        self.start_time = time.time()
        
        # Clear previous data to avoid crossed lines when restarting
        self.times.clear()
        self.values.clear()
        self.anomalies.clear()
        self.severities.clear()
        
        # Update button states
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start data collection
        self._collect_data()
        
    def _stop_sensor(self):
        """Stop the sensor simulation"""
        self.sensor.stop()
        
        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        
    def _toggle_mode(self):
        """Toggle between PERFORMANCE and ECO modes"""
        if self.mode_var.get() == "PERFORMANCE":
            self.mode_var.set("ECO")
            self.sensor.set_mode(OperationMode.ECO)
            self.mode_button.config(text="🌱 ECO")
        else:
            self.mode_var.set("PERFORMANCE")
            self.sensor.set_mode(OperationMode.PERFORMANCE)
            self.mode_button.config(text="🌍 PERFORMANCE")
    
    def _collect_data(self):
        """Collect data from sensor and update visualization"""
        if not self.sensor.is_running:
            return
        
        # Generate new reading
        value, is_anomaly, severity = self.sensor.generate_reading()
        
        # Store data
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        self.values.append(value)
        self.anomalies.append(is_anomaly)
        self.severities.append(severity)
        
        # Mode-specific processing
        if self.sensor.mode == OperationMode.PERFORMANCE:
            # PERFORMANCE mode: publish to Pub/Sub for cloud processing
            if self.sensor.pubsub_available:
                self.sensor.publish_to_pubsub(value)
            else:
                # Fallback: log that Pub/Sub is not available
                if is_anomaly:
                    print(f"⚠️  [PERFORMANCE Mode] Pub/Sub unavailable - anomaly not sent to cloud")
        elif self.sensor.mode == OperationMode.ECO:
            # ECO mode: send data to local ML model API for analysis
            api_result = self.sensor.run_local_inference(value)
            if api_result:
                # Always log API anomaly detections
                if api_result['is_anomaly']:
                    print(f"\n🚨 [ECO Mode] ANOMALY DETECTED BY API")
                    print(f"   Value: {value:.4f}")
                    print(f"   Anomaly Score: {api_result['anomaly_score']:.4f}")
                    print(f"   Ground Truth: {'ANOMALY (' + severity + ')' if is_anomaly else 'Normal (False Positive)'}")
                    print(f"   Timestamp: {time.time():.2f}\n")
                # Log missed anomalies
                elif is_anomaly:
                    print(f"⚠️  [ECO Mode] API missed anomaly (ground truth: {severity}) - Value: {value:.4f}")
            else:
                # Log when API call fails
                if is_anomaly:
                    print(f"⚠️  [ECO Mode] API not available - missed anomaly: {severity}, Value: {value:.4f}")
        
        # Update chart
        self._update_chart()
        
        # Update statistics
        self._update_statistics()
        
        # Schedule next collection
        self.root.after(self.update_interval, self._collect_data)
    
    def _update_chart(self):
        """Update the matplotlib chart with current data"""
        self.ax.clear()
        
        # Configure plot
        self.ax.set_xlabel('Time (seconds)', fontsize=10)
        self.ax.set_ylabel('Vibration Amplitude', fontsize=10)
        self.ax.set_title('Vibration Sensor Data Stream', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        if len(self.times) > 0:
            times_list = list(self.times)
            values_list = list(self.values)
            anomalies_list = list(self.anomalies)
            severities_list = list(self.severities)
            
            # Plot continuous line through ALL points (normal and anomalies)
            self.ax.plot(times_list, values_list, 'b-', linewidth=1, alpha=0.7)
            
            # Plot all points with subtle markers
            self.ax.scatter(times_list, values_list, c='blue', s=15, alpha=0.5, zorder=3)
            
            # Subtly highlight small anomalies (slightly larger, faint orange edge)
            small_times = [t for t, a, s in zip(times_list, anomalies_list, severities_list) 
                          if a and s == 'small']
            small_values = [v for v, a, s in zip(values_list, anomalies_list, severities_list) 
                           if a and s == 'small']
            
            if small_times:
                self.ax.scatter(small_times, small_values, c='blue', s=25, 
                              edgecolors='orange', linewidths=1, alpha=0.6, zorder=4)
            
            # Subtly highlight large anomalies (slightly larger, faint red edge)
            large_times = [t for t, a, s in zip(times_list, anomalies_list, severities_list) 
                          if a and s == 'large']
            large_values = [v for v, a, s in zip(values_list, anomalies_list, severities_list) 
                           if a and s == 'large']
            
            if large_times:
                self.ax.scatter(large_times, large_values, c='blue', s=35, 
                              edgecolors='red', linewidths=1.5, alpha=0.7, zorder=5)
            
            # Auto-scale y-axis with some padding
            if values_list:
                y_min = min(values_list) - 1
                y_max = max(values_list) + 1
                self.ax.set_ylim(y_min, y_max)
            
            # Set x-axis to show last 30 seconds
            if times_list:
                x_max = times_list[-1]
                x_min = max(0, x_max - 30)
                self.ax.set_xlim(x_min, x_max)
        
        self.canvas.draw()
    
    def _update_statistics(self):
        """Update the statistics display"""
        stats = self.sensor.get_statistics()
        
        # Update status
        mode = self.mode_var.get()
        self.status_label.config(
            text=f"Status: Running ({mode})",
            foreground='green'
        )
        
        # Update anomaly count
        api_detected = stats.get('api_anomalies_detected', 0)
        cloud_detected = stats.get('cloud_anomalies_detected', 0)
        batches_sent = stats.get('pubsub_batches_sent', 0)
        
        anomaly_text = (f"Anomalies: {stats['total_anomalies']} "
                       f"({stats['small_anomalies']} small, {stats['large_anomalies']} large)")
        
        # Add mode-specific detection count
        if mode == "ECO" and api_detected > 0:
            anomaly_text += f" | API: {api_detected}"
        elif mode == "PERFORMANCE":
            anomaly_text += f" | Cloud: {cloud_detected} | Batches: {batches_sent}"
        
        self.anomaly_label.config(text=anomaly_text)
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Initialize and run the vibration sensor simulator GUI"""
    
    # Initialize sensor with demo-friendly parameters
    # ~2-3 anomalies per minute at 100ms intervals
    sensor = VibrationSensor(
        mean=0.0,
        std_dev=1.0,
        anomaly_rate=0.003,  # ~2 per minute
        small_anomaly_range=(3.0, 4.0),  # Easier to detect
        large_anomaly_range=(5.0, 8.0),  # Very obvious
        large_anomaly_ratio=0.4  # 40% are large
    )
    
    # Configure ML API URL from environment variable or use default
    ml_api_url = os.getenv('ML_API_URL', 'http://localhost:5001')
    sensor.set_ml_api_url(ml_api_url)
    
    # Initialize Pub/Sub for PERFORMANCE mode
    device_id = os.getenv('DEVICE_ID', 'edge-001')
    publisher = create_publisher(device_id)
    
    if publisher:
        sensor.set_pubsub_publisher(publisher)
        print(f"✓ Pub/Sub publisher initialized for device: {device_id}")
    else:
        print("⚠ Pub/Sub not available - PERFORMANCE mode will be limited")
    
    # Create anomaly subscriber for receiving cloud results
    def on_anomaly_results(data):
        """Callback for anomaly results from cloud."""
        readings = data.get('readings', [])
        for reading in readings:
            if reading.get('is_anomaly'):
                sensor.cloud_anomalies_detected += 1
                print(f"🚨 [CLOUD] Anomaly detected: vibration={reading['vibration']:.4f}, "
                      f"confidence={reading.get('confidence', 0):.4f}")
    
    subscriber = create_subscriber(on_anomaly_results)
    if subscriber:
        subscriber.start()
        sensor.set_pubsub_subscriber(subscriber)
        print("✓ Pub/Sub subscriber started for anomaly results")
    
    # Create tkinter root window
    root = tk.Tk()
    
    # Initialize GUI application with Pub/Sub status
    app = Application(root, sensor, pubsub_connected=publisher is not None)
    
    # Start the GUI
    print("=" * 60)
    print("Edge Device - Vibration Sensor Simulator")
    print("Carbon-Aware IoT Anomaly Detection System")
    print("=" * 60)
    print("\nStarting GUI...")
    print("- Click 'Start Sensor' to begin data generation")
    print("- Toggle between PERFORMANCE and ECO modes")
    print("- PERFORMANCE mode: Streams to cloud via Pub/Sub")
    print("- ECO mode: Uses local light model via REST")
    print("- Watch for color-coded anomalies:")
    print("  • Blue = Normal readings")
    print("  • Orange = Small anomalies (3-4σ)")
    print("  • Red = Large anomalies (5-8σ)")
    print("=" * 60)
    
    try:
        app.run()
    finally:
        # Cleanup on exit
        if publisher:
            publisher.flush()
        if subscriber:
            subscriber.stop()


if __name__ == "__main__":
    main()

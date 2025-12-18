"""
Vibration Sensor Simulator for Edge Device
Generates realistic sensor data with anomalies for carbon-aware anomaly detection system

Supports two operation modes:
- PERFORMANCE: Publishes sensor data to Pub/Sub for cloud processing
- ECO: Sends data to local REST API for light model inference
"""
import numpy as np
import time
import requests
from typing import Optional, Tuple, Dict, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from pubsub_client import SensorPublisher, AnomalySubscriber


class OperationMode(Enum):
    """Operation modes for carbon-aware processing"""
    PERFORMANCE = "PERFORMANCE"  # Low carbon - stream to cloud via Pub/Sub
    ECO = "ECO"  # High carbon - local inference via REST


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
        self.ml_api_url = "http://localhost:5000"  # Default for local testing
        self.use_light_model = True  # Use light model for ECO mode
        
        # Pub/Sub configuration (for PERFORMANCE mode)
        self._pubsub_publisher: Optional['SensorPublisher'] = None
        self._pubsub_subscriber: Optional['AnomalySubscriber'] = None
        
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
    
    def set_pubsub_publisher(self, publisher: 'SensorPublisher'):
        """
        Set the Pub/Sub publisher for PERFORMANCE mode.
        
        Args:
            publisher: SensorPublisher instance
        """
        self._pubsub_publisher = publisher
        print("[Sensor] Pub/Sub publisher configured")
    
    def set_pubsub_subscriber(self, subscriber: 'AnomalySubscriber'):
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

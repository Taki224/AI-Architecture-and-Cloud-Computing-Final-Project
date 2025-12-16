"""
Vibration Sensor Simulator for Edge Device
Generates realistic sensor data with anomalies for carbon-aware anomaly detection system
"""
import numpy as np
import time
from typing import Optional, Tuple
from enum import Enum


class OperationMode(Enum):
    """Operation modes for carbon-aware processing"""
    PERFORMANCE = "PERFORMANCE"  # Low carbon - stream to cloud
    ECO = "ECO"  # High carbon - local inference


class VibrationSensor:
    """
    Simulates a vibration sensor generating time-series data with anomalies.
    
    Data characteristics:
    - Normal operation: μ=0, σ=1 (normal distribution)
    - Anomalies: ~2-3 per minute at 100ms intervals
    - Small anomalies: 3-4σ deviation (easier to detect)
    - Large anomalies: 5-8σ deviation (obvious anomalies)
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
        
        # Statistics tracking
        self.total_readings = 0
        self.total_anomalies = 0
        self.small_anomalies = 0
        self.large_anomalies = 0
        
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
            'anomaly_rate': (self.total_anomalies / self.total_readings * 100) 
                           if self.total_readings > 0 else 0
        }
    
    # ========== Placeholder methods for future integration ==========
    
    def publish_to_pubsub(self, data: dict, topic: str):
        """
        Placeholder: Publish data to Google Cloud Pub/Sub.
        
        Args:
            data: The data payload to publish
            topic: The Pub/Sub topic ('raw-data' or 'confirmed-anomalies')
        
        Future implementation will:
        - Serialize data to JSON
        - Publish to GCP Pub/Sub topic
        - Handle connection errors and retries
        """
        pass  # TODO: Implement Pub/Sub publishing
    
    def run_local_inference(self, value: float) -> bool:
        """
        Placeholder: Run local anomaly detection using lightweight model.
        
        Args:
            value: The sensor reading to analyze
        
        Returns:
            True if anomaly detected, False otherwise
        
        Future implementation will:
        - Load model_light.pkl (Isolation Forest n=10)
        - Run inference on the reading
        - Return anomaly prediction
        """
        pass  # TODO: Implement local model inference
    
    def handle_mode_command(self, command: dict):
        """
        Placeholder: Handle mode switch commands from Pub/Sub.
        
        Args:
            command: Dictionary with 'mode' key
        
        Future implementation will:
        - Subscribe to 'edge-commands' topic
        - Parse mode switch messages
        - Update self.mode accordingly
        """
        pass  # TODO: Implement command handling

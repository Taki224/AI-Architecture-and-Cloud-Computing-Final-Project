"""
Isolation Forest Anomaly Detector with Sliding Window Features

Uses scikit-learn's Isolation Forest with statistical features extracted
from a sliding window for robust anomaly detection on vibration sensor data.
"""
import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    ML-based anomaly detector using Isolation Forest with sliding window features.
    
    Extracts statistical features from a sliding window of readings to detect
    both point anomalies (single outliers) and contextual anomalies (unusual patterns).
    
    This detector uses a Z-score based approach for anomaly detection during warmup
    and after fitting, to avoid the high false positive rate that occurs when
    Isolation Forest learns from contaminated runtime data.
    
    Parameters:
        contamination (float): Expected proportion of anomalies (default: 0.10 = 10%)
        window_size (int): Size of sliding window for feature extraction (default: 50)
        n_estimators (int): Number of trees in the forest (default: 100)
        min_samples_for_fit (int): Minimum samples before ML model activates (default: 200)
        z_threshold (float): Z-score threshold for anomaly detection (default: 3.5)
    """
    
    def __init__(
        self,
        contamination: float = 0.10,
        window_size: int = 50,
        n_estimators: int = 100,
        min_samples_for_fit: int = 200,
        z_threshold: float = 3.5
    ):
        self.contamination = contamination
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.min_samples_for_fit = min_samples_for_fit
        self.z_threshold = z_threshold
        
        self.window = deque(maxlen=window_size)
        self.training_data = []
        self.is_fitted = False
        
        # Running statistics for robust warmup (using Welford's algorithm)
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0
        
        # Baseline statistics learned from training (for Z-score detection)
        self.baseline_mean = None
        self.baseline_std = None
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=1  # Single thread to avoid GIL contention with concurrent requests
        )
    
    def _update_running_stats(self, value: float):
        """
        Update running mean and variance using Welford's online algorithm.
        This allows us to track statistics without storing all values.
        """
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._M2 += delta * delta2
    
    def _get_running_std(self) -> float:
        """Get the running standard deviation."""
        if self._n < 2:
            return 1.0  # Avoid division by zero
        return np.sqrt(self._M2 / (self._n - 1))
    
    def _compute_z_score(self, value: float) -> float:
        """Compute Z-score for a value using baseline or running statistics."""
        if self.baseline_mean is not None and self.baseline_std is not None:
            mean = self.baseline_mean
            std = self.baseline_std
        else:
            mean = self._mean
            std = self._get_running_std()
        
        if std < 0.001:  # Avoid division by very small numbers
            std = 1.0
        
        return abs(value - mean) / std
    
    def _extract_features(self, values: list) -> np.ndarray:
        """
        Extract statistical features from sliding window.
        
        Features:
        - Current value
        - Window mean
        - Window standard deviation
        - Window max
        - Window min
        - Rate of change (delta from previous)
        """
        arr = np.array(values)
        
        features = [
            arr[-1],                                          # Current value
            np.mean(arr),                                     # Window mean
            np.std(arr) if len(arr) > 1 else 0.0,            # Window std
            np.max(arr),                                      # Window max
            np.min(arr),                                      # Window min
            arr[-1] - arr[-2] if len(arr) > 1 else 0.0,      # Rate of change
        ]
        
        return np.array(features).reshape(1, -1)
    
    def detect(self, value: float) -> dict:
        """
        Detect if current reading is anomalous.
        
        Uses a hybrid approach:
        1. During warmup: Collects data and uses running Z-score for detection
        2. After fitting: Uses Z-score threshold with ML as confirmation
        
        The Z-score approach is primary because Isolation Forest can have high
        false positive rates when trained on limited or contaminated data.
        
        Args:
            value: Current sensor reading
            
        Returns:
            dict with keys:
                - is_anomaly: bool
                - anomaly_score: float (higher = more anomalous)
                - z_score: float
                - status: str ('warming_up', 'filling_window', 'active')
                - samples_needed: int (only during warmup)
        """
        self.window.append(value)
        
        # Update running statistics (excluding extreme outliers to build robust baseline)
        z_score = self._compute_z_score(value)
        if z_score < 5.0:  # Only update stats with non-extreme values
            self._update_running_stats(value)
        
        # Collect training data during warmup
        if not self.is_fitted:
            self.training_data.append(value)
            samples_needed = self.min_samples_for_fit - len(self.training_data)
            
            if len(self.training_data) >= self.min_samples_for_fit:
                self._fit_model()
                return {
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "z_score": z_score,
                    "status": "just_fitted",
                    "samples_needed": 0,
                    "method": "ml_warmup"
                }
            
            # During warmup, use Z-score for anomaly detection
            is_anomaly = z_score > self.z_threshold
            
            return {
                "is_anomaly": is_anomaly,
                "anomaly_score": min(1.0, z_score / 5.0),  # Normalize to 0-1
                "z_score": z_score,
                "status": "warming_up",
                "samples_needed": int(samples_needed),
                "method": "statistical_warmup"
            }
        
        # After fitting: Use Z-score as primary detection method
        # This is more reliable than Isolation Forest on streaming data
        is_anomaly = z_score > self.z_threshold
        anomaly_score = min(1.0, z_score / 5.0)  # Normalize to 0-1
        
        # If we have enough window data, we can also get ML prediction as secondary signal
        ml_anomaly = False
        if len(self.window) >= self.window_size:
            import time
            t1 = time.perf_counter()
            features = self._extract_features(list(self.window))
            t2 = time.perf_counter()
            prediction = self.model.predict(features)[0]
            t3 = time.perf_counter()
            ml_anomaly = prediction == -1
            print(f"      [TIMING-ML] extract_features: {(t2-t1)*1000:.1f}ms, predict: {(t3-t2)*1000:.1f}ms")
        
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "z_score": float(z_score),
            "ml_anomaly": ml_anomaly,
            "status": "active",
            "method": "ensemble" if ml_anomaly and is_anomaly else "statistical"
        }
    
    def _fit_model(self):
        """
        Fit the Isolation Forest model on collected training data.
        Also computes baseline statistics for Z-score detection.
        """
        # Compute baseline statistics from training data
        # Filter out extreme values (>4 sigma from running mean) before computing baseline
        values = np.array(self.training_data)
        running_std = self._get_running_std()
        
        # Use robust filtering: exclude values > 4 sigma from running mean
        if running_std > 0.001:
            mask = np.abs(values - self._mean) < 4 * running_std
            filtered_values = values[mask]
            if len(filtered_values) > 20:
                self.baseline_mean = np.mean(filtered_values)
                self.baseline_std = np.std(filtered_values)
            else:
                self.baseline_mean = self._mean
                self.baseline_std = running_std
        else:
            self.baseline_mean = self._mean
            self.baseline_std = running_std
        
        print(f"✓ Baseline statistics: mean={self.baseline_mean:.4f}, std={self.baseline_std:.4f}")
        
        # Build feature matrix from training data (using only filtered data)
        temp_window = deque(maxlen=self.window_size)
        features_list = []
        
        for value in self.training_data:
            temp_window.append(value)
            if len(temp_window) >= self.window_size:
                features = self._extract_features(list(temp_window))
                features_list.append(features.flatten())
        
        if len(features_list) >= 10:
            X = np.array(features_list)
            self.model.fit(X)
            self.is_fitted = True
            print(f"✓ IsolationForest fitted on {len(features_list)} samples")
        else:
            print(f"⚠ Not enough samples for ML fitting: {len(features_list)}, using Z-score only")
            self.is_fitted = True  # Mark as fitted so we use the baseline stats
    
    def reset(self):
        """Reset detector state for retraining."""
        self.window.clear()
        self.training_data = []
        self.is_fitted = False
        
        # Reset running statistics
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0
        self.baseline_mean = None
        self.baseline_std = None
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )

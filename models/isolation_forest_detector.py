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
    
    Parameters:
        contamination (float): Expected proportion of anomalies (default: 0.003 = 0.3%)
        window_size (int): Size of sliding window for feature extraction (default: 50)
        n_estimators (int): Number of trees in the forest (default: 100)
        min_samples_for_fit (int): Minimum samples before ML model activates (default: 100)
    """
    
    def __init__(
        self,
        contamination: float = 0.003,
        window_size: int = 50,
        n_estimators: int = 100,
        min_samples_for_fit: int = 100
    ):
        self.contamination = contamination
        self.window_size = window_size
        self.n_estimators = n_estimators
        self.min_samples_for_fit = min_samples_for_fit
        
        self.window = deque(maxlen=window_size)
        self.training_data = []
        self.is_fitted = False
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
    
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
        
        Args:
            value: Current sensor reading
            
        Returns:
            dict with keys:
                - is_anomaly: bool
                - anomaly_score: float (higher = more anomalous)
                - status: str ('warming_up', 'filling_window', 'active')
                - samples_needed: int (only during warmup)
        """
        self.window.append(value)
        
        # Collect training data during warmup
        if not self.is_fitted:
            self.training_data.append(value)
            samples_needed = self.min_samples_for_fit - len(self.training_data)
            
            if len(self.training_data) >= self.min_samples_for_fit:
                self._fit_model()
                return {
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "status": "just_fitted",
                    "samples_needed": 0,
                    "method": "ml_warmup"
                }
            
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "status": "warming_up",
                "samples_needed": int(samples_needed),
                "method": "ml_warmup"
            }
        
        # Need full window for feature extraction
        if len(self.window) < self.window_size:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "status": "filling_window",
                "samples_needed": int(self.window_size - len(self.window)),
                "method": "ml_warmup"
            }
        
        # Extract features and predict
        features = self._extract_features(list(self.window))
        prediction = self.model.predict(features)[0]
        raw_score = self.model.score_samples(features)[0]
        
        # Convert score: Isolation Forest returns negative scores, more negative = more anomalous
        # Normalize to 0-1 range where higher = more anomalous
        anomaly_score = max(0.0, min(1.0, -raw_score))
        
        return {
            "is_anomaly": bool(prediction == -1),
            "anomaly_score": float(anomaly_score),
            "raw_score": float(raw_score),
            "status": "active",
            "method": "ml_only"
        }
    
    def _fit_model(self):
        """Fit the Isolation Forest model on collected training data."""
        # Build feature matrix from training data
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
            print(f"⚠ Not enough samples for fitting: {len(features_list)}")
    
    def reset(self):
        """Reset detector state for retraining."""
        self.window.clear()
        self.training_data = []
        self.is_fitted = False
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )

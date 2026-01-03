"""
Hybrid Anomaly Detector

Combines Isolation Forest (ML) with Z-score statistical detection for
maximum accuracy and reliability. Uses ensemble voting to minimize
false negatives.

During warmup (first 100 samples), falls back to statistical detection only.
Once ML model is fitted, uses ensemble approach: flags anomaly if EITHER
detector triggers, maximizing recall.
"""
import numpy as np
from isolation_forest_detector import IsolationForestDetector
from statistical_model import StatisticalAnomalyDetector


class HybridAnomalyDetector:
    """
    Ensemble anomaly detector combining ML and statistical approaches.
    
    Parameters:
        z_threshold (float): Z-score threshold for statistical detector (default: 3.0)
        contamination (float): Expected anomaly rate for Isolation Forest (default: 0.003)
        window_size (int): Sliding window size for feature extraction (default: 50)
        n_estimators (int): Number of trees in Isolation Forest (default: 100)
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        contamination: float = 0.003,
        window_size: int = 50,
        n_estimators: int = 100
    ):
        self.z_threshold = z_threshold
        self.contamination = contamination
        
        # Initialize both detectors
        self.ml_detector = IsolationForestDetector(
            contamination=contamination,
            window_size=window_size,
            n_estimators=n_estimators
        )
        
        self.stat_detector = StatisticalAnomalyDetector(threshold=z_threshold)
        # Fit statistical detector with expected distribution (μ=0, σ=1)
        self.stat_detector.mean_ = 0.0
        self.stat_detector.std_ = 1.0
        
        self._detection_count = 0
        self._anomaly_count = 0
    
    def detect(self, value: float) -> dict:
        """
        Detect anomaly using ensemble approach.
        
        Args:
            value: Current sensor reading
            
        Returns:
            dict with comprehensive detection results including:
                - is_anomaly: bool (ensemble decision)
                - method: str ('statistical_fallback' or 'ensemble')
                - ml_anomaly: bool (ML detector result)
                - stat_anomaly: bool (statistical detector result)
                - anomaly_score: float (ML score, higher = more anomalous)
                - z_score: float (statistical z-score)
                - confidence: float (0-1 confidence in the detection)
        """
        self._detection_count += 1
        
        # Get ML detector result
        ml_result = self.ml_detector.detect(value)
        
        # Get statistical detector result
        z_score = abs((value - self.stat_detector.mean_) / self.stat_detector.std_)
        stat_anomaly = z_score > self.z_threshold
        
        # During warmup, use statistical only
        if ml_result.get("status") in ["warming_up", "filling_window", "just_fitted"]:
            if stat_anomaly:
                self._anomaly_count += 1
            
            return {
                "is_anomaly": bool(stat_anomaly),
                "method": "statistical_fallback",
                "ml_anomaly": False,
                "stat_anomaly": bool(stat_anomaly),
                "anomaly_score": 0.0,
                "z_score": float(z_score),
                "confidence": float(min(1.0, z_score / 5.0) if stat_anomaly else 1.0 - (z_score / self.z_threshold)),
                "ml_status": ml_result.get("status"),
                "samples_needed": int(ml_result.get("samples_needed", 0))
            }
        
        # Ensemble: flag if EITHER detector finds anomaly (maximize recall)
        ml_anomaly = ml_result.get("is_anomaly", False)
        is_anomaly = ml_anomaly or stat_anomaly
        
        if is_anomaly:
            self._anomaly_count += 1
        
        # Calculate confidence based on agreement
        anomaly_score = ml_result.get("anomaly_score", 0.0)
        if ml_anomaly and stat_anomaly:
            confidence = min(1.0, (anomaly_score + z_score / 5.0) / 2.0)
        elif ml_anomaly:
            confidence = anomaly_score * 0.8  # Slightly lower confidence for ML-only
        elif stat_anomaly:
            confidence = min(1.0, z_score / 5.0) * 0.8  # Slightly lower for stat-only
        else:
            confidence = 1.0 - max(anomaly_score, z_score / self.z_threshold)
        
        return {
            "is_anomaly": bool(is_anomaly),
            "method": "ensemble",
            "ml_anomaly": bool(ml_anomaly),
            "stat_anomaly": bool(stat_anomaly),
            "anomaly_score": float(anomaly_score),
            "z_score": float(z_score),
            "confidence": float(confidence),
            "raw_score": float(ml_result.get("raw_score", 0.0))
        }
    
    def get_stats(self) -> dict:
        """Get detection statistics."""
        return {
            "total_detections": self._detection_count,
            "total_anomalies": self._anomaly_count,
            "anomaly_rate": self._anomaly_count / self._detection_count if self._detection_count > 0 else 0.0,
            "ml_fitted": self.ml_detector.is_fitted
        }
    
    def reset(self):
        """Reset both detectors."""
        self.ml_detector.reset()
        self._detection_count = 0
        self._anomaly_count = 0

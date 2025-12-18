"""
Statistical Anomaly Detector for API Service
"""
import numpy as np


class StatisticalAnomalyDetector:
    """
    Simple statistical anomaly detector using Z-score threshold.
    Optimized for Gaussian-distributed vibration sensor data.
    """
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X, y=None):
        """Fit the model by learning mean and std from training data."""
        X = np.array(X).reshape(-1)
        
        if y is not None:
            y = np.array(y).reshape(-1)
            normal_mask = (y == 1)
            X_normal = X[normal_mask]
        else:
            X_normal = X
            
        self.mean_ = np.mean(X_normal)
        self.std_ = np.std(X_normal)
        
        return self
    
    def predict(self, X):
        """Predict anomalies: 1 = normal, -1 = anomaly"""
        X = np.array(X).reshape(-1, 1)
        scores = self.score_samples(X)
        predictions = np.where(scores < 0, -1, 1)
        return predictions
    
    def score_samples(self, X):
        """Compute anomaly scores (negative = anomaly)"""
        X = np.array(X).reshape(-1)
        z_scores = np.abs((X - self.mean_) / self.std_)
        scores = self.threshold - z_scores
        return scores
    
    def decision_function(self, X):
        """Alias for score_samples"""
        return self.score_samples(X)

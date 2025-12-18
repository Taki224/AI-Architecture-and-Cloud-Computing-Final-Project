"""
Statistical Threshold Model for Vibration Anomaly Detection
Uses Z-score thresholding optimized for Gaussian sensor data
"""
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import time


class StatisticalAnomalyDetector:
    """
    Simple statistical anomaly detector using Z-score threshold.
    Optimized for Gaussian-distributed vibration sensor data.
    """
    
    def __init__(self, threshold=3.0):
        """
        Initialize detector.
        
        Args:
            threshold: Number of standard deviations for anomaly threshold
                      (3.0 = catches ~3σ anomalies, 4.0 = only very large anomalies)
        """
        self.threshold = threshold
        self.mean_ = None
        self.std_ = None
        
    def fit(self, X, y=None):
        """
        Fit the model by learning mean and std from training data.
        Only learns from normal samples.
        """
        X = np.array(X).reshape(-1)
        
        # If labels provided, only use normal samples
        if y is not None:
            y = np.array(y).reshape(-1)
            normal_mask = (y == 1)  # 1 = normal, -1 = anomaly
            X_normal = X[normal_mask]
        else:
            X_normal = X
            
        self.mean_ = np.mean(X_normal)
        self.std_ = np.std(X_normal)
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies based on Z-score threshold.
        
        Returns:
            Array of predictions: 1 = normal, -1 = anomaly
        """
        X = np.array(X).reshape(-1, 1)
        scores = self.score_samples(X)
        
        # Convert to predictions: negative score = anomaly
        predictions = np.where(scores < 0, -1, 1)
        return predictions
    
    def score_samples(self, X):
        """
        Compute anomaly scores.
        Negative scores indicate anomalies.
        
        Returns:
            Anomaly scores (negative = anomaly, positive = normal)
        """
        X = np.array(X).reshape(-1)
        
        # Calculate Z-scores
        z_scores = np.abs((X - self.mean_) / self.std_)
        
        # Convert to sklearn-style scores: negative = anomaly
        # Invert so that high z-scores get negative values
        scores = self.threshold - z_scores
        
        return scores
    
    def decision_function(self, X):
        """Alias for score_samples for sklearn compatibility"""
        return self.score_samples(X)


def load_data(filepath: str):
    """Load training data from CSV file."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    X = df[['value']].values
    y_raw = df['is_anomaly'].values
    y = np.where(y_raw == 1, -1, 1)  # Convert: 1->-1 (anomaly), 0->1 (normal)
    
    print(f"Loaded {len(X):,} samples")
    print(f"  - Normal: {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.2f}%)")
    print(f"  - Anomalies: {sum(y == -1):,} ({sum(y == -1)/len(y)*100:.2f}%)")
    
    return X, y, df


def evaluate_model(model, X, y, model_name="Model"):
    """Evaluate model performance."""
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', pos_label=-1)
    
    print(f"\n{model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    cm = confusion_matrix(y, y_pred, labels=[1, -1])
    print("\n  Confusion Matrix:")
    print(f"                Predicted Normal  Predicted Anomaly")
    print(f"  Actual Normal       {cm[0][0]:7d}           {cm[0][1]:7d}")
    print(f"  Actual Anomaly      {cm[1][0]:7d}           {cm[1][1]:7d}")
    
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n  False Positive Rate: {fpr:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")


def main():
    print("=" * 70)
    print("Statistical Anomaly Detection Model Training")
    print("Z-Score Threshold Method (Optimized for Gaussian Data)")
    print("=" * 70)
    
    # Load data
    print("\n" + "=" * 70)
    print("STEP 1: Load Training Data")
    print("=" * 70)
    X_train, y_train, df_train = load_data("training_data.csv")
    
    # Train heavy model (lower threshold = more sensitive)
    print("\n" + "=" * 70)
    print("STEP 2: Train Heavy Model (Cloud - More Sensitive)")
    print("=" * 70)
    print("Using threshold=2.5σ (catches smaller anomalies)")
    
    model_heavy = StatisticalAnomalyDetector(threshold=2.5)
    model_heavy.fit(X_train, y_train)
    
    print(f"\nLearned parameters:")
    print(f"  Mean: {model_heavy.mean_:.4f}")
    print(f"  Std Dev: {model_heavy.std_:.4f}")
    print(f"  Threshold: {model_heavy.threshold}σ")
    
    evaluate_model(model_heavy, X_train, y_train, "Heavy Model (Training)")
    
    # Train light model (higher threshold = less sensitive, fewer false positives)
    print("\n" + "=" * 70)
    print("STEP 3: Train Light Model (Edge - Conservative)")
    print("=" * 70)
    print("Using threshold=3.0σ (only obvious anomalies)")
    
    model_light = StatisticalAnomalyDetector(threshold=3.0)
    model_light.fit(X_train, y_train)
    
    print(f"\nLearned parameters:")
    print(f"  Mean: {model_light.mean_:.4f}")
    print(f"  Std Dev: {model_light.std_:.4f}")
    print(f"  Threshold: {model_light.threshold}σ")
    
    evaluate_model(model_light, X_train, y_train, "Light Model (Training)")
    
    # Validate
    print("\n" + "=" * 70)
    print("STEP 4: Validation Set Evaluation")
    print("=" * 70)
    X_val, y_val, df_val = load_data("validation_data.csv")
    
    print("\n--- Heavy Model Validation ---")
    evaluate_model(model_heavy, X_val, y_val, "Heavy Model")
    
    print("\n--- Light Model Validation ---")
    evaluate_model(model_light, X_val, y_val, "Light Model")
    
    # Save models
    print("\n" + "=" * 70)
    print("STEP 5: Save Models")
    print("=" * 70)
    
    joblib.dump(model_heavy, "model_heavy.pkl")
    print("✓ Saved model_heavy.pkl")
    
    joblib.dump(model_light, "model_light.pkl")
    print("✓ Saved model_light.pkl")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\nModel type: Statistical Z-score threshold")
    print("Benefits:")
    print("  ✓ Simple and interpretable")
    print("  ✓ Fast inference (microseconds)")
    print("  ✓ Optimal for Gaussian sensor data")
    print("  ✓ No complex hyperparameters")
    print("  ✓ Guaranteed to catch large anomalies (>3σ)")
    print("=" * 70)


if __name__ == "__main__":
    main()

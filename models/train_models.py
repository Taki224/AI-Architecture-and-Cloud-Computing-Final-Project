"""
Model Training Script for Anomaly Detection
Trains both heavy (cloud) and light (edge) models using:
- Heavy: HybridAnomalyDetector (Isolation Forest + Z-score ensemble)
- Light: IsolationForestDetector (lightweight ML only)
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
import time
import os

from hybrid_detector import HybridAnomalyDetector
from isolation_forest_detector import IsolationForestDetector


def load_data(filepath: str):
    """
    Load training data from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Tuple of (X, y) where X is features and y is labels
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Extract features (just the vibration value for now)
    X = df[['value']].values.flatten()
    
    # Extract labels (1 for anomaly, -1 for normal - Isolation Forest convention)
    # Note: CSV has 0 for normal, 1 for anomaly
    y_raw = df['is_anomaly'].values
    y = np.where(y_raw == 1, -1, 1)  # Convert: 1->-1 (anomaly), 0->1 (normal)
    
    print(f"Loaded {len(X):,} samples")
    print(f"  - Normal: {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.2f}%)")
    print(f"  - Anomalies: {sum(y == -1):,} ({sum(y == -1)/len(y)*100:.2f}%)")
    
    return X, y, df


def train_hybrid_model(X, y, n_estimators: int, model_name: str, 
                       z_threshold: float = 3.0, contamination: float = 0.003,
                       window_size: int = 50):
    """
    Train a HybridAnomalyDetector (Isolation Forest + Z-score ensemble).
    
    Args:
        X: Training features (1D array of vibration values)
        y: Training labels
        n_estimators: Number of trees in Isolation Forest
        model_name: Name for display purposes
        z_threshold: Z-score threshold for statistical detector
        contamination: Expected anomaly rate for Isolation Forest
        window_size: Sliding window size for feature extraction
        
    Returns:
        Trained HybridAnomalyDetector
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Model: HybridAnomalyDetector (Isolation Forest + Z-score)")
    print(f"Parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - z_threshold: {z_threshold}")
    print(f"  - contamination: {contamination}")
    print(f"  - window_size: {window_size}")
    
    # Initialize model
    detector = HybridAnomalyDetector(
        z_threshold=z_threshold,
        contamination=contamination,
        window_size=window_size,
        n_estimators=n_estimators
    )
    
    # Train by passing all samples through the detector
    start_time = time.time()
    print(f"\nFitting model on {len(X):,} samples...")
    
    for i, value in enumerate(X):
        detector.detect(value)
        if (i + 1) % 1000 == 0:
            stats = detector.get_stats()
            print(f"  Processed {i+1:,}/{len(X):,} samples | ML fitted: {stats['ml_fitted']}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    stats = detector.get_stats()
    print(f"  ML model fitted: {stats['ml_fitted']}")
    print(f"  Total detections: {stats['total_detections']:,}")
    print(f"  Anomaly rate: {stats['anomaly_rate']*100:.2f}%")
    
    return detector


def train_light_model(X, y, n_estimators: int, model_name: str,
                      contamination: float = 0.003, window_size: int = 50):
    """
    Train an IsolationForestDetector (ML only, for edge deployment).
    
    Args:
        X: Training features (1D array of vibration values)
        y: Training labels
        n_estimators: Number of trees in Isolation Forest
        model_name: Name for display purposes
        contamination: Expected anomaly rate
        window_size: Sliding window size
        
    Returns:
        Trained IsolationForestDetector
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Model: IsolationForestDetector (lightweight ML)")
    print(f"Parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - contamination: {contamination}")
    print(f"  - window_size: {window_size}")
    
    # Initialize model
    detector = IsolationForestDetector(
        contamination=contamination,
        window_size=window_size,
        n_estimators=n_estimators,
        min_samples_for_fit=100
    )
    
    # Train by passing all samples through the detector
    start_time = time.time()
    print(f"\nFitting model on {len(X):,} samples...")
    
    for i, value in enumerate(X):
        detector.detect(value)
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{len(X):,} samples | ML fitted: {detector.is_fitted}")
    
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    print(f"  ML model fitted: {detector.is_fitted}")
    
    return detector


def evaluate_detector(detector, X, y, detector_type: str = "hybrid"):
    """
    Evaluate detector performance.
    
    Args:
        detector: Trained detector
        X: Features
        y: True labels (-1 for anomaly, 1 for normal)
        detector_type: 'hybrid' or 'light'
    """
    print("\nEvaluating detector...")
    
    predictions = []
    for value in X:
        result = detector.detect(value)
        pred = -1 if result['is_anomaly'] else 1
        predictions.append(pred)
    
    y_pred = np.array(predictions)
    
    # Calculate metrics
    accuracy = accuracy_score(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', pos_label=-1)
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred, labels=[1, -1])
    print("\n  Confusion Matrix:")
    print(f"                Predicted Normal  Predicted Anomaly")
    print(f"  Actual Normal       {cm[0][0]:7d}           {cm[0][1]:7d}")
    print(f"  Actual Anomaly      {cm[1][0]:7d}           {cm[1][1]:7d}")
    
    # Calculate false positive and false negative rates
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n  False Positive Rate: {fpr:.4f}")
    print(f"  False Negative Rate: {fnr:.4f}")


def save_model(model, filepath: str):
    """
    Save model to disk.
    
    Args:
        model: Trained model
        filepath: Output filepath
    """
    print(f"\nSaving model to {filepath}...")
    joblib.dump(model, filepath)
    
    # Check file size
    size_mb = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Model saved successfully ({size_mb:.2f} MB)")


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("Hybrid Anomaly Detector Training Pipeline")
    print("Carbon-Aware IoT Anomaly Detection System")
    print("=" * 60)
    
    # Paths
    training_data_path = "data/training_data.csv"
    validation_data_path = "data/validation_data.csv"
    
    # Load training data
    print("\n" + "=" * 60)
    print("STEP 1: Load Training Data")
    print("=" * 60)
    X_train, y_train, df_train = load_data(training_data_path)
    
    # Calculate actual contamination from training data
    actual_contamination = sum(y_train == -1) / len(y_train)
    print(f"\nActual contamination rate: {actual_contamination:.4f}")
    
    # Train heavy model (cloud) - HybridAnomalyDetector with more estimators
    print("\n" + "=" * 60)
    print("STEP 2: Train Heavy Model (Cloud)")
    print("=" * 60)
    model_heavy = train_hybrid_model(
        X_train, 
        y_train, 
        n_estimators=200,  # More trees for cloud
        model_name="Heavy Model (Cloud - Hybrid Ensemble)",
        z_threshold=3.0,
        contamination=0.003,
        window_size=50
    )
    
    # Train light model (edge) - IsolationForestDetector only
    print("\n" + "=" * 60)
    print("STEP 3: Train Light Model (Edge)")
    print("=" * 60)
    model_light = train_light_model(
        X_train, 
        y_train, 
        n_estimators=50,  # Fewer trees for edge
        model_name="Light Model (Edge - ML Only)",
        contamination=0.003,
        window_size=50
    )
    
    # Validate on validation set
    if os.path.exists(validation_data_path):
        print("\n" + "=" * 60)
        print("STEP 4: Validation Set Evaluation")
        print("=" * 60)
        X_val, y_val, df_val = load_data(validation_data_path)
        
        print("\n--- Heavy Model Validation Performance ---")
        evaluate_detector(model_heavy, X_val, y_val, detector_type="hybrid")
        
        print("\n--- Light Model Validation Performance ---")
        evaluate_detector(model_light, X_val, y_val, detector_type="light")
    
    # Save models
    print("\n" + "=" * 60)
    print("STEP 5: Save Models")
    print("=" * 60)
    save_model(model_heavy, "model_heavy.pkl")
    save_model(model_light, "model_light.pkl")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nModels saved:")
    print("  ✓ model_heavy.pkl (HybridAnomalyDetector - 200 estimators for cloud)")
    print("  ✓ model_light.pkl (IsolationForestDetector - 50 estimators for edge)")
    print("\nNext steps:")
    print("  1. Rebuild Docker image with trained models")
    print("  2. Deploy heavy model to Cloud Run")
    print("  3. Test with edge device GUI")
    print("=" * 60)


if __name__ == "__main__":
    main()

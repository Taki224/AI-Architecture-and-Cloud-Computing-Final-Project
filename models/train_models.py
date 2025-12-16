"""
Model Training Script for Isolation Forest Anomaly Detection
Trains both heavy (cloud) and light (edge) models
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib
import time
import os


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
    X = df[['value']].values
    
    # Extract labels (1 for anomaly, -1 for normal - Isolation Forest convention)
    # Note: CSV has 0 for normal, 1 for anomaly
    y_raw = df['is_anomaly'].values
    y = np.where(y_raw == 1, -1, 1)  # Convert: 1->-1 (anomaly), 0->1 (normal)
    
    print(f"Loaded {len(X):,} samples")
    print(f"  - Normal: {sum(y == 1):,} ({sum(y == 1)/len(y)*100:.2f}%)")
    print(f"  - Anomalies: {sum(y == -1):,} ({sum(y == -1)/len(y)*100:.2f}%)")
    
    return X, y, df


def train_model(X, y, n_estimators: int, model_name: str, contamination: float = 0.003):
    """
    Train an Isolation Forest model.
    
    Args:
        X: Training features
        y: Training labels
        n_estimators: Number of trees in the forest
        model_name: Name for display purposes
        contamination: Expected proportion of anomalies
        
    Returns:
        Trained model
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    print(f"Parameters:")
    print(f"  - n_estimators: {n_estimators}")
    print(f"  - contamination: {contamination}")
    print(f"  - max_samples: auto")
    print(f"  - random_state: 42")
    
    # Initialize model
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples='auto',
        random_state=42,
        n_jobs=-1,  # Use all CPU cores
        verbose=0
    )
    
    # Train
    start_time = time.time()
    model.fit(X)
    elapsed = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed:.2f} seconds")
    
    # Evaluate on training data
    print("\nTraining Set Performance:")
    evaluate_model(model, X, y)
    
    return model


def evaluate_model(model, X, y):
    """
    Evaluate model performance.
    
    Args:
        model: Trained Isolation Forest model
        X: Features
        y: True labels
    """
    # Predict
    y_pred = model.predict(X)
    
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
    print("Isolation Forest Model Training Pipeline")
    print("Carbon-Aware IoT Anomaly Detection System")
    print("=" * 60)
    
    # Paths
    training_data_path = "training_data.csv"
    validation_data_path = "validation_data.csv"
    
    # Load training data
    print("\n" + "=" * 60)
    print("STEP 1: Load Training Data")
    print("=" * 60)
    X_train, y_train, df_train = load_data(training_data_path)
    
    # Calculate actual contamination from training data
    actual_contamination = sum(y_train == -1) / len(y_train)
    print(f"\nActual contamination rate: {actual_contamination:.4f}")
    
    # Train heavy model (cloud)
    print("\n" + "=" * 60)
    print("STEP 2: Train Heavy Model (Cloud)")
    print("=" * 60)
    model_heavy = train_model(
        X_train, 
        y_train, 
        n_estimators=200, 
        model_name="Heavy Model (Cloud - 200 estimators)",
        contamination=actual_contamination
    )
    
    # Train light model (edge)
    print("\n" + "=" * 60)
    print("STEP 3: Train Light Model (Edge)")
    print("=" * 60)
    model_light = train_model(
        X_train, 
        y_train, 
        n_estimators=10, 
        model_name="Light Model (Edge - 10 estimators)",
        contamination=actual_contamination
    )
    
    # Validate on validation set
    if os.path.exists(validation_data_path):
        print("\n" + "=" * 60)
        print("STEP 4: Validation Set Evaluation")
        print("=" * 60)
        X_val, y_val, df_val = load_data(validation_data_path)
        
        print("\n--- Heavy Model Validation Performance ---")
        evaluate_model(model_heavy, X_val, y_val)
        
        print("\n--- Light Model Validation Performance ---")
        evaluate_model(model_light, X_val, y_val)
    
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
    print("  ✓ model_heavy.pkl (200 estimators - for cloud deployment)")
    print("  ✓ model_light.pkl (10 estimators - for edge deployment)")
    print("\nNext steps:")
    print("  1. Test models with edge device GUI")
    print("  2. Integrate with Pub/Sub for cloud communication")
    print("  3. Deploy heavy model to Cloud Run")
    print("  4. Implement carbon-aware mode switching")
    print("=" * 60)


if __name__ == "__main__":
    main()

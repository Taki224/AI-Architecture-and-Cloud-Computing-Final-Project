"""
ML Model API Service
Provides anomaly detection inference using trained Statistical models
"""
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from datetime import datetime
from statistical_model import StatisticalAnomalyDetector

app = Flask(__name__)

# Global variables for models
light_model = None
heavy_model = None

# Model paths
LIGHT_MODEL_PATH = '/app/models/model_light.pkl'
HEAVY_MODEL_PATH = '/app/models/model_heavy.pkl'


def load_models():
    """Load trained models on startup"""
    global light_model, heavy_model
    
    # Check environment variable to determine which models to load
    # LOAD_MODELS can be: "light", "heavy", or "both" (default: "light")
    load_mode = os.getenv('LOAD_MODELS', 'light').lower()
    
    try:
        # Always load light model (for edge/local inference)
        if load_mode in ['light', 'both']:
            if os.path.exists(LIGHT_MODEL_PATH):
                light_model = joblib.load(LIGHT_MODEL_PATH)
                print(f"✓ Light model loaded from {LIGHT_MODEL_PATH}")
                # Get model size
                import sys
                size_mb = sys.getsizeof(light_model) / (1024 * 1024)
                print(f"  Model size: ~{size_mb:.2f} MB in memory")
            else:
                print(f"⚠ Light model not found at {LIGHT_MODEL_PATH}")
        
        # Only load heavy model if explicitly requested
        if load_mode in ['heavy', 'both']:
            if os.path.exists(HEAVY_MODEL_PATH):
                heavy_model = joblib.load(HEAVY_MODEL_PATH)
                print(f"✓ Heavy model loaded from {HEAVY_MODEL_PATH}")
            else:
                print(f"⚠ Heavy model not found at {HEAVY_MODEL_PATH}")
        else:
            print(f"ℹ Heavy model not loaded (LOAD_MODELS={load_mode})")
            
    except Exception as e:
        print(f"✗ Error loading models: {e}")


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'light_model_loaded': light_model is not None,
        'heavy_model_loaded': heavy_model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Analyze vibration data and detect anomalies
    
    Expected JSON payload:
    {
        "value": 2.5,
        "model_type": "light"  // optional: "light" or "heavy", defaults to "light"
    }
    
    Or batch prediction:
    {
        "values": [1.2, 2.5, 3.1],
        "model_type": "light"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Determine which model to use
        model_type = data.get('model_type', 'light')
        model = light_model if model_type == 'light' else heavy_model
        
        if model is None:
            return jsonify({'error': f'{model_type} model not loaded'}), 503
        
        # Handle single value or batch
        if 'value' in data:
            values = [[data['value']]]
        elif 'values' in data:
            values = [[v] for v in data['values']]
        else:
            return jsonify({'error': 'Missing "value" or "values" field'}), 400
        
        # Convert to numpy array
        X = np.array(values)
        
        # Make prediction (-1 = anomaly, 1 = normal)
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Process results
        results = []
        for i, (pred, score, val) in enumerate(zip(predictions, scores, values)):
            is_anomaly = (pred == -1)
            result = {
                'value': float(val[0]),
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(score),
                'prediction': int(pred),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
            
            # Log ALL predictions to console for debugging
            status = "🚨 ANOMALY" if is_anomaly else "✓ Normal"
            print(f"{status} | Value: {result['value']:7.4f} | Score: {result['anomaly_score']:8.6f} | Model: {model_type.upper()}")
            
            # Extra prominent logging for anomalies
            if is_anomaly:
                print("=" * 70)
                print(f"🚨 🚨 🚨  ANOMALY DETECTED  🚨 🚨 🚨")
                print(f"   Timestamp: {result['timestamp']}")
                print(f"   Value: {result['value']:.4f}")
                print(f"   Anomaly Score: {result['anomaly_score']:.6f}")
                print(f"   Model: {model_type.upper()}")
                print("=" * 70)
        
        # Return single result or array based on input
        if 'value' in data:
            return jsonify(results[0])
        else:
            return jsonify({'predictions': results, 'count': len(results)})
            
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze batch of sensor readings
    
    Expected JSON payload:
    {
        "readings": [
            {"timestamp": "2025-12-17T10:30:00", "value": 1.2},
            {"timestamp": "2025-12-17T10:30:01", "value": 5.5},
            {"timestamp": "2025-12-17T10:30:02", "value": 0.8}
        ],
        "model_type": "heavy"  // optional
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'readings' not in data:
            return jsonify({'error': 'Missing "readings" field'}), 400
        
        readings = data['readings']
        model_type = data.get('model_type', 'light')
        model = light_model if model_type == 'light' else heavy_model
        
        if model is None:
            return jsonify({'error': f'{model_type} model not loaded'}), 503
        
        # Extract values
        values = [[r['value']] for r in readings]
        X = np.array(values)
        
        # Predict
        predictions = model.predict(X)
        scores = model.score_samples(X)
        
        # Combine results
        results = []
        anomaly_count = 0
        for reading, pred, score in zip(readings, predictions, scores):
            is_anomaly = (pred == -1)
            result = {
                'timestamp': reading.get('timestamp'),
                'value': reading['value'],
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(score)
            }
            results.append(result)
            
            if is_anomaly:
                anomaly_count += 1
                print(f"🚨 ANOMALY: t={reading.get('timestamp')}, value={reading['value']:.4f}, score={score:.6f}")
        
        summary = {
            'total_readings': len(results),
            'anomalies_detected': anomaly_count,
            'anomaly_rate': anomaly_count / len(results) if results else 0,
            'model_type': model_type
        }
        
        if anomaly_count > 0:
            print(f"\n📊 Batch Analysis Summary: {anomaly_count}/{len(results)} anomalies detected ({summary['anomaly_rate']*100:.2f}%)\n")
        
        return jsonify({
            'summary': summary,
            'results': results
        })
        
    except Exception as e:
        print(f"✗ Batch analysis error: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("ML Model API Service - Anomaly Detection (EDGE/LOCAL)")
    print("=" * 70)
    
    # Load models on startup
    load_models()
    
    print("\nStarting Flask API server...")
    print("Available endpoints:")
    print("  - GET  /health          - Health check")
    print("  - POST /predict         - Single/batch prediction")
    print("  - POST /analyze/batch   - Batch analysis with timestamps")
    print("\nOptimized for EDGE deployment - using light model only")
    print("Set LOAD_MODELS=both to enable heavy model")
    print("=" * 70)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)

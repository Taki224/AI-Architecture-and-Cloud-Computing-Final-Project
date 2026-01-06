"""
ML Model API Service
Provides anomaly detection inference using Hybrid ML + Statistical detection
"""
from flask import Flask, request, jsonify
import numpy as np
import os
import sys
from datetime import datetime

# Add models directory to path
sys.path.insert(0, '/app/models')

from isolation_forest_detector import IsolationForestDetector

# Import carbon monitoring
try:
    sys.path.insert(0, '/app/common')
    from carbon_monitoring import CarbonMonitor
    CARBON_MONITORING_AVAILABLE = True
except ImportError:
    CarbonMonitor = None
    CARBON_MONITORING_AVAILABLE = False

app = Flask(__name__)

# Global detector instance
detector = None
carbon_monitor = None

# Configuration
CONTAMINATION = float(os.getenv('CONTAMINATION', '0.003'))
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '50'))
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', '50'))
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'project-bf5303f9-c6e6-4630-a08')


def init_detector():
    """Load the pre-trained light model from disk."""
    global detector
    
    model_path = os.getenv('LIGHT_MODEL_PATH', '/app/models/model_light.pkl')
    
    try:
        import joblib
        detector = joblib.load(model_path)
        print(f"✓ Loaded pre-trained IsolationForestDetector from {model_path}")
        print(f"  - ML model fitted: {detector.is_fitted}")
        return True
    except FileNotFoundError:
        print(f"✗ Model file not found: {model_path}")
        print("  Falling back to runtime initialization...")
        return init_detector_fallback()
    except Exception as e:
        print(f"✗ Failed to load detector: {e}")
        print("  Falling back to runtime initialization...")
        return init_detector_fallback()


def init_detector_fallback():
    """Fallback: Initialize detector at runtime if pre-trained model unavailable."""
    global detector
    
    try:
        detector = IsolationForestDetector(
            contamination=CONTAMINATION,
            window_size=WINDOW_SIZE,
            n_estimators=N_ESTIMATORS
        )
        print(f"✓ IsolationForestDetector initialized (runtime fallback)")
        print(f"  - Contamination: {CONTAMINATION}")
        print(f"  - Window size: {WINDOW_SIZE}")
        print(f"  - Isolation Forest estimators: {N_ESTIMATORS}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        return False


def init_carbon_monitoring():
    """Initialize Carbon Monitoring for emissions tracking."""
    global carbon_monitor
    
    if CARBON_MONITORING_AVAILABLE and CarbonMonitor:
        try:
            carbon_monitor = CarbonMonitor(
                project_id=PROJECT_ID,
                service_name="light-model",
                mode="ECO",
                country_iso_code=os.getenv('CARBON_COUNTRY_CODE', 'AUT')
            )
            print("✓ Carbon Monitoring initialized (ECO mode)")
            return True
        except Exception as e:
            print(f"⚠ Carbon Monitoring not available: {e}")
    
    return False


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    carbon_stats = carbon_monitor.get_stats() if carbon_monitor else {}
    return jsonify({
        'status': 'healthy' if detector else 'degraded',
        'detector': 'isolation_forest',
        'ml_fitted': detector.is_fitted if detector else False,
        'carbon_monitoring_enabled': carbon_monitor is not None,
        'carbon_emissions_gco2e': carbon_stats.get('total_emissions_gco2e', 0),
        'carbon_mode': carbon_stats.get('mode', 'ECO'),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Analyze vibration data and detect anomalies using Isolation Forest ML.
    
    Expected JSON payload:
    {
        "value": 2.5
    }
    
    Returns:
    {
        "value": 2.5,
        "is_anomaly": false,
        "method": "ensemble",
        "ml_anomaly": false,
        "stat_anomaly": false,
        "anomaly_score": 0.15,
        "z_score": 2.5,
        "confidence": 0.85
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if detector is None:
            return jsonify({'error': 'Detector not initialized'}), 503
        
        if 'value' not in data:
            return jsonify({'error': 'Missing "value" field'}), 400
        
        value = float(data['value'])
        
        # Track carbon emissions for this inference
        if carbon_monitor:
            with carbon_monitor.track_inference(batch_size=1):
                result = detector.detect(value)
        else:
            result = detector.detect(value)
        
        # Log detection
        if result['is_anomaly']:
            method_info = f"ML={result.get('ml_anomaly')}, Stat={result.get('stat_anomaly')}"
            method = result.get('method', 'ml_only')
            print(f"🚨 ANOMALY | Value: {value:7.4f} | Method: {method} | {method_info}")
            print(f"   Score: {result.get('anomaly_score', 0):.4f} | Z: {result.get('z_score', 0):.2f} | Conf: {result.get('confidence', 0):.2%}")
        else:
            print(f"✓ Normal  | Value: {value:7.4f} | Z: {result.get('z_score', 0):.2f}")
        
        # Get carbon stats for response
        carbon_stats = carbon_monitor.get_stats() if carbon_monitor else {}
        
        return jsonify({
            'value': float(value),
            'is_anomaly': bool(result['is_anomaly']),
            'method': result.get('method'),
            'ml_anomaly': bool(result.get('ml_anomaly', False)),
            'stat_anomaly': bool(result.get('stat_anomaly', False)),
            'anomaly_score': float(result.get('anomaly_score', 0.0)),
            'z_score': float(result.get('z_score', 0.0)),
            'confidence': float(result.get('confidence', 0.0)),
            'ml_status': result.get('ml_status'),
            'carbon_emissions_gco2e': carbon_stats.get('total_emissions_gco2e', 0),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"✗ Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """Alias for /predict for compatibility."""
    return predict()
@app.route('/analyze/batch', methods=['POST'])
def analyze_batch():
    """
    Analyze batch of sensor readings using hybrid ML + statistical detection.
    
    Expected JSON payload:
    {
        "readings": [
            {"timestamp": "2025-12-17T10:30:00", "value": 1.2},
            {"timestamp": "2025-12-17T10:30:01", "value": 5.5},
            {"timestamp": "2025-12-17T10:30:02", "value": 0.8}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'readings' not in data:
            return jsonify({'error': 'Missing "readings" field'}), 400
        
        if detector is None:
            return jsonify({'error': 'Detector not initialized'}), 503
        
        readings = data['readings']
        results = []
        anomaly_count = 0
        
        # Track carbon emissions for batch processing
        if carbon_monitor:
            with carbon_monitor.track_inference(batch_size=len(readings)):
                for reading in readings:
                    value = float(reading['value'])
                    result = detector.detect(value)
                    
                    is_anomaly = result['is_anomaly']
                    if is_anomaly:
                        anomaly_count += 1
                        print(f"🚨 ANOMALY: t={reading.get('timestamp')}, value={value:.4f}, "
                              f"method={result['method']}, score={result.get('anomaly_score', 0):.4f}")
                    
                    results.append({
                        'timestamp': reading.get('timestamp'),
                        'value': value,
                        'is_anomaly': is_anomaly,
                        'method': result.get('method'),
                        'anomaly_score': result.get('anomaly_score', 0.0),
                        'z_score': result.get('z_score', 0.0),
                        'confidence': result.get('confidence', 0.0)
                    })
        else:
            for reading in readings:
                value = float(reading['value'])
                result = detector.detect(value)
                
                is_anomaly = result['is_anomaly']
                if is_anomaly:
                    anomaly_count += 1
                    print(f"🚨 ANOMALY: t={reading.get('timestamp')}, value={value:.4f}, "
                          f"method={result['method']}, score={result.get('anomaly_score', 0):.4f}")
                
                results.append({
                    'timestamp': reading.get('timestamp'),
                    'value': value,
                    'is_anomaly': is_anomaly,
                    'method': result.get('method'),
                    'anomaly_score': result.get('anomaly_score', 0.0),
                    'z_score': result.get('z_score', 0.0),
                    'confidence': result.get('confidence', 0.0)
                })
        
        stats = detector.get_stats()
        summary = {
            'total_readings': len(results),
            'anomalies_detected': anomaly_count,
            'anomaly_rate': anomaly_count / len(results) if results else 0,
            'detector': 'hybrid_isolation_forest_zscore',
            'ml_fitted': stats.get('ml_fitted', False)
        }
        
        if anomaly_count > 0:
            print(f"\n📊 Batch Summary: {anomaly_count}/{len(results)} anomalies ({summary['anomaly_rate']*100:.2f}%)\n")
        
        return jsonify({
            'summary': summary,
            'results': results
        })
        
    except Exception as e:
        print(f"✗ Batch analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def get_stats():
    """Get detector statistics."""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 503
    
    stats = detector.get_stats()
    
    # Add carbon stats if available
    if carbon_monitor:
        stats['carbon'] = carbon_monitor.get_stats()
    
    return jsonify(stats)


@app.route('/reset', methods=['POST'])
def reset_detector():
    """Reset the detector (clears training data, requires re-warmup)."""
    if detector is None:
        return jsonify({'error': 'Detector not initialized'}), 503
    
    detector.reset()
    return jsonify({'status': 'reset', 'message': 'Detector reset, warmup required'})


if __name__ == '__main__':
    print("=" * 70)
    print("ML Model API Service - Hybrid Anomaly Detection (EDGE/LOCAL)")
    print("=" * 70)
    
    # Initialize detector on startup
    init_detector()
    
    # Initialize carbon monitoring
    init_carbon_monitoring()
    
    print("\nStarting Flask API server...")
    print("Available endpoints:")
    print("  - GET  /health          - Health check with detector status")
    print("  - POST /predict         - Single value prediction")
    print("  - POST /analyze         - Alias for /predict")
    print("  - POST /analyze/batch   - Batch analysis with timestamps")
    print("  - GET  /stats           - Detector statistics")
    print("  - POST /reset           - Reset detector (requires re-warmup)")
    print("\nDetection: Hybrid Isolation Forest + Z-score (threshold=3.0σ)")
    print("Warmup: First 100 samples use statistical fallback")
    print("Carbon: CodeCarbon tracking → GCP Cloud Monitoring (ECO mode)")
    print("=" * 70)
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False)

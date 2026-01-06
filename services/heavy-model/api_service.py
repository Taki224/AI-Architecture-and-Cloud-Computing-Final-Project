"""
Heavy Model API Service for GCP Cloud Run
Provides anomaly detection using Hybrid ML (Isolation Forest + Z-score) via Pub/Sub

This service:
1. Subscribes to 'sensor-readings' Pub/Sub topic for batched sensor readings
2. Runs hybrid ML inference (Isolation Forest + Z-score ensemble)
3. Publishes results to 'anomaly-results' topic
4. Exposes /health endpoint for Cloud Run health checks
5. Logs anomaly detections to Cloud Logging/Monitoring
"""
import os
import sys
import json
import threading
import signal
import warnings
from datetime import datetime
from concurrent.futures import TimeoutError as FuturesTimeoutError

# Suppress sklearn warnings to reduce log noise
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')

from flask import Flask, jsonify
import numpy as np

# Add parent directory for shared model code
sys.path.insert(0, '/app/models')

from hybrid_detector import HybridAnomalyDetector

# Import monitoring (optional - gracefully degrades if unavailable)
try:
    from monitoring import AnomalyMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    AnomalyMonitor = None
    MONITORING_AVAILABLE = False

# Import carbon monitoring
try:
    sys.path.insert(0, '/app/common')
    from carbon_monitoring import CarbonMonitor
    CARBON_MONITORING_AVAILABLE = True
except ImportError:
    CarbonMonitor = None
    CARBON_MONITORING_AVAILABLE = False

# Flask app for health checks
app = Flask(__name__)

# Global state
detector = None
publisher = None
subscriber = None
monitor = None
carbon_monitor = None
shutdown_event = threading.Event()

# Configuration from environment
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'local-project')
SENSOR_TOPIC = os.getenv('PUBSUB_SENSOR_TOPIC', 'sensor-data')
SENSOR_SUBSCRIPTION = os.getenv('PUBSUB_SENSOR_SUBSCRIPTION', 'sensor-data-sub')
ANOMALY_TOPIC = os.getenv('PUBSUB_ANOMALY_TOPIC', 'anomaly-results')

# ML Configuration (heavier settings for cloud)
Z_THRESHOLD = float(os.getenv('Z_THRESHOLD', '3.0'))
CONTAMINATION = float(os.getenv('CONTAMINATION', '0.003'))
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '50'))
N_ESTIMATORS = int(os.getenv('N_ESTIMATORS', '200'))  # More estimators for cloud


def init_detector():
    """Load the pre-trained hybrid anomaly detector from disk."""
    global detector
    
    model_path = os.getenv('HEAVY_MODEL_PATH', '/app/models/model_heavy.pkl')
    
    try:
        import joblib
        detector = joblib.load(model_path)
        stats = detector.get_stats()
        print(f"✓ Loaded pre-trained HybridAnomalyDetector from {model_path}")
        print(f"  - ML model fitted: {stats.get('ml_fitted', False)}")
        print(f"  - Z-score threshold: {detector.z_threshold}")
        print(f"  - Contamination: {detector.contamination}")
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
        detector = HybridAnomalyDetector(
            z_threshold=Z_THRESHOLD,
            contamination=CONTAMINATION,
            window_size=WINDOW_SIZE,
            n_estimators=N_ESTIMATORS
        )
        print(f"✓ HybridAnomalyDetector initialized (runtime fallback)")
        print(f"  - Z-score threshold: {Z_THRESHOLD}")
        print(f"  - Contamination: {CONTAMINATION}")
        print(f"  - Window size: {WINDOW_SIZE}")
        print(f"  - Isolation Forest estimators: {N_ESTIMATORS}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        return False


def init_pubsub():
    """Initialize Pub/Sub publisher and subscriber."""
    global publisher, subscriber
    
    try:
        from google.cloud import pubsub_v1
        
        # Publisher for anomaly results
        publisher = pubsub_v1.PublisherClient()
        print(f"✓ Pub/Sub publisher initialized")
        
        # Subscriber for sensor data
        subscriber = pubsub_v1.SubscriberClient()
        print(f"✓ Pub/Sub subscriber initialized")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to initialize Pub/Sub: {e}")
        return False


def init_monitoring():
    """Initialize Cloud Monitoring for anomaly tracking."""
    global monitor
    
    if MONITORING_AVAILABLE and AnomalyMonitor:
        try:
            monitor = AnomalyMonitor(PROJECT_ID)
            print("✓ Cloud Monitoring initialized")
            return True
        except Exception as e:
            print(f"⚠ Cloud Monitoring not available: {e}")
    
    return False


def init_carbon_monitoring():
    """Initialize Carbon Monitoring for emissions tracking."""
    global carbon_monitor
    
    if CARBON_MONITORING_AVAILABLE and CarbonMonitor:
        try:
            carbon_monitor = CarbonMonitor(
                project_id=PROJECT_ID,
                service_name="heavy-model",
                mode="PERFORMANCE",
                country_iso_code=os.getenv('CARBON_COUNTRY_CODE', 'HUN')
            )
            print("✓ Carbon Monitoring initialized (PERFORMANCE mode)")
            return True
        except Exception as e:
            print(f"⚠ Carbon Monitoring not available: {e}")
    
    return False


def process_batch(message_data: dict) -> dict:
    """
    Process a batch of sensor readings through the hybrid ML detector.
    
    Args:
        message_data: Dict containing 'device_id' and 'readings' array
        
    Returns:
        Dict with processed results including anomaly predictions
    """
    global detector, monitor, carbon_monitor
    
    device_id = message_data.get('device_id', 'unknown')
    readings = message_data.get('readings', [])
    
    if not readings:
        return {'device_id': device_id, 'readings': [], 'error': 'No readings provided'}
    
    if detector is None:
        return {'device_id': device_id, 'readings': [], 'error': 'Detector not initialized'}
    
    results = []
    anomaly_count = 0
    batch_size = len(readings)
    
    # Use carbon tracking context manager if available
    carbon_context = carbon_monitor.track_inference(batch_size=batch_size) if carbon_monitor else None
    
    try:
        if carbon_context:
            carbon_context.__enter__()
        
        for reading in readings:
            timestamp = reading.get('timestamp', datetime.utcnow().timestamp())
            vibration = reading.get('vibration', reading.get('value', 0.0))
            
            try:
                # Run hybrid detection
                detection = detector.detect(vibration)
                
                is_anomaly = detection['is_anomaly']
                confidence = detection.get('confidence', 0.0)
                anomaly_score = detection.get('anomaly_score', 0.0)
                z_score = detection.get('z_score', 0.0)
                method = detection.get('method', 'unknown')
                
                result = {
                    'timestamp': float(timestamp),
                    'vibration': float(vibration),
                    'is_anomaly': bool(is_anomaly),
                    'confidence': float(confidence),
                    'anomaly_score': float(anomaly_score),
                    'z_score': float(z_score),
                    'method': str(method),
                    'ml_anomaly': bool(detection.get('ml_anomaly', False)),
                    'stat_anomaly': bool(detection.get('stat_anomaly', False))
                }
                results.append(result)
                
                if is_anomaly:
                    anomaly_count += 1
                    print(f"🚨 ANOMALY | Device: {device_id} | Value: {vibration:7.4f} | "
                          f"Method: {method} | Score: {anomaly_score:.4f} | Z: {z_score:.2f}")
                    
                    # Log to Cloud Monitoring
                    if monitor:
                        monitor.log_anomaly(
                            timestamp=timestamp,
                            vibration=vibration,
                            confidence=confidence,
                            device_id=device_id
                        )
                else:
                    print(f"✓ Normal  | Device: {device_id} | Value: {vibration:7.4f} | Z: {z_score:.2f}")
                
            except Exception as e:
                print(f"✗ Error processing reading: {e}")
                results.append({
                    'timestamp': timestamp,
                    'vibration': vibration,
                    'is_anomaly': False,
                    'confidence': 0.0,
                    'error': str(e)
                })
    finally:
        # Exit carbon tracking context
        if carbon_context:
            carbon_context.__exit__(None, None, None)
    
    # Summary logging
    stats = detector.get_stats()
    carbon_stats = carbon_monitor.get_stats() if carbon_monitor else {}
    
    if anomaly_count > 0:
        print(f"\n📊 Batch Summary: {anomaly_count}/{len(readings)} anomalies detected "
              f"({anomaly_count/len(readings)*100:.1f}%) | ML fitted: {stats.get('ml_fitted', False)}")
    
    if carbon_stats:
        print(f"🌱 Carbon: {carbon_stats.get('total_emissions_gco2e', 0):.6f} gCO₂e total | "
              f"{carbon_stats.get('avg_emissions_per_inference_gco2e', 0):.6f} gCO₂e/inference\n")
    
    return {
        'device_id': device_id,
        'readings': results,
        'count': len(results),
        'anomalies_detected': anomaly_count,
        'ml_fitted': stats.get('ml_fitted', False),
        'carbon_emissions_gco2e': carbon_stats.get('total_emissions_gco2e', 0) if carbon_stats else None,
        'processed_at': datetime.utcnow().isoformat()
    }


def publish_results(results: dict):
    """
    Publish processing results to the anomaly-results topic.
    
    Args:
        results: Dict containing processed readings with anomaly predictions
    """
    global publisher
    
    if publisher is None:
        print("⚠ Publisher not available, results not sent")
        return False
    
    try:
        topic_path = publisher.topic_path(PROJECT_ID, ANOMALY_TOPIC)
        data = json.dumps(results).encode('utf-8')
        
        future = publisher.publish(topic_path, data)
        message_id = future.result(timeout=10)
        
        print(f"[Publisher] Results sent (msg_id: {message_id})")
        return True
        
    except Exception as e:
        print(f"✗ Failed to publish results: {e}")
        return False


def message_callback(message):
    """
    Callback for processing incoming Pub/Sub messages.
    
    Args:
        message: Pub/Sub message object
    """
    try:
        # Decode message
        data = json.loads(message.data.decode('utf-8'))
        
        device_id = data.get('device_id', 'unknown')
        reading_count = data.get('count', len(data.get('readings', [])))
        
        print(f"\n[Subscriber] Received batch: {reading_count} readings from {device_id}")
        
        # Process through heavy model
        results = process_batch(data)
        
        # Publish results
        publish_success = publish_results(results)
        
        # Only acknowledge after successful processing and publishing
        if publish_success or results.get('error'):
            message.ack()
            print(f"[Subscriber] Message acknowledged")
        else:
            # Don't ack - will be redelivered
            print(f"[Subscriber] Message NOT acknowledged - will retry")
            message.nack()
            
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in message: {e}")
        message.ack()  # Ack to avoid infinite retry of bad messages
        
    except Exception as e:
        print(f"✗ Error processing message: {e}")
        message.nack()  # Retry on unexpected errors


def start_subscriber():
    """Start the Pub/Sub subscriber in a background thread."""
    global subscriber
    
    if subscriber is None:
        print("✗ Subscriber not initialized")
        return None
    
    subscription_path = subscriber.subscription_path(PROJECT_ID, SENSOR_SUBSCRIPTION)
    
    print(f"\n[Subscriber] Listening on {subscription_path}...")
    
    streaming_pull_future = subscriber.subscribe(
        subscription_path,
        callback=message_callback
    )
    
    return streaming_pull_future


# ========== Flask Health Check Endpoints ==========

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run."""
    stats = detector.get_stats() if detector else {}
    carbon_stats = carbon_monitor.get_stats() if carbon_monitor else {}
    return jsonify({
        'status': 'healthy' if detector else 'degraded',
        'detector': 'hybrid_isolation_forest_zscore',
        'ml_fitted': stats.get('ml_fitted', False),
        'total_detections': stats.get('total_detections', 0),
        'anomaly_rate': stats.get('anomaly_rate', 0.0),
        'pubsub_connected': publisher is not None and subscriber is not None,
        'monitoring_enabled': monitor is not None,
        'carbon_monitoring_enabled': carbon_monitor is not None,
        'carbon_emissions_gco2e': carbon_stats.get('total_emissions_gco2e', 0),
        'carbon_mode': carbon_stats.get('mode', 'PERFORMANCE'),
        'n_estimators': N_ESTIMATORS,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        'service': 'heavy-model-api',
        'version': '2.0.0',
        'description': 'Cloud anomaly detection using Hybrid ML (Isolation Forest + Z-score)',
        'detector': 'HybridAnomalyDetector',
        'n_estimators': N_ESTIMATORS,
        'endpoints': {
            '/health': 'Health check with detector status',
            '/': 'Service info'
        }
    })


def run_flask():
    """Run Flask app in a separate thread."""
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print(f"\n[Service] Received signal {signum}, shutting down...")
    shutdown_event.set()


def main():
    """Main entry point for the heavy model service."""
    print("=" * 70)
    print("Heavy Model API Service - Hybrid ML Cloud Anomaly Detection")
    print("=" * 70)
    print(f"Project: {PROJECT_ID}")
    print(f"Sensor Topic: {SENSOR_TOPIC}")
    print(f"Anomaly Topic: {ANOMALY_TOPIC}")
    print(f"Detector: HybridAnomalyDetector")
    print(f"  - Isolation Forest: {N_ESTIMATORS} estimators (heavy)")
    print(f"  - Z-score threshold: {Z_THRESHOLD}")
    print(f"  - Window size: {WINDOW_SIZE}")
    print("=" * 70)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize components
    if not init_detector():
        print("⚠ Warning: Detector not initialized, service will return errors")
    
    if not init_pubsub():
        print("✗ Failed to initialize Pub/Sub, exiting...")
        sys.exit(1)
    
    init_monitoring()
    init_carbon_monitoring()
    
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    print(f"\n✓ Flask health check server started on port {os.getenv('PORT', 8080)}")
    
    # Start Pub/Sub subscriber
    streaming_future = start_subscriber()
    
    if streaming_future is None:
        print("✗ Failed to start subscriber, exiting...")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("Service ready - waiting for sensor data...")
    print("=" * 70 + "\n")
    
    # Block until shutdown
    try:
        while not shutdown_event.is_set():
            try:
                streaming_future.result(timeout=1.0)
            except FuturesTimeoutError:
                continue
            except Exception as e:
                print(f"✗ Subscriber error: {e}")
                break
    finally:
        print("\n[Service] Shutting down...")
        streaming_future.cancel()
        
        if monitor:
            monitor.flush()
        
        if carbon_monitor:
            carbon_monitor.flush()
        
        print("[Service] Goodbye!")


if __name__ == '__main__':
    main()

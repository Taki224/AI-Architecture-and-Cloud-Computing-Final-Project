"""
Heavy Model API Service for GCP Cloud Run
Provides anomaly detection using the heavy (robust) model via Pub/Sub

This service:
1. Subscribes to 'sensor-data' Pub/Sub topic for batched sensor readings
2. Runs heavy model inference on each reading
3. Publishes results to 'anomaly-results' topic
4. Exposes /health endpoint for Cloud Run health checks
5. Logs anomaly detections to Cloud Logging/Monitoring
"""
import os
import sys
import json
import threading
import signal
from datetime import datetime
from concurrent.futures import TimeoutError as FuturesTimeoutError

from flask import Flask, jsonify
import joblib
import numpy as np

# Add parent directory for shared model code
sys.path.insert(0, '/app/models')

try:
    from statistical_model import StatisticalAnomalyDetector
except ImportError:
    StatisticalAnomalyDetector = None

# Import monitoring (optional - gracefully degrades if unavailable)
try:
    from monitoring import AnomalyMonitor
    MONITORING_AVAILABLE = True
except ImportError:
    AnomalyMonitor = None
    MONITORING_AVAILABLE = False

# Flask app for health checks
app = Flask(__name__)

# Global state
heavy_model = None
publisher = None
subscriber = None
monitor = None
shutdown_event = threading.Event()

# Configuration from environment
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'local-project')
SENSOR_TOPIC = os.getenv('PUBSUB_SENSOR_TOPIC', 'sensor-data')
SENSOR_SUBSCRIPTION = os.getenv('PUBSUB_SENSOR_SUBSCRIPTION', 'sensor-data-sub')
ANOMALY_TOPIC = os.getenv('PUBSUB_ANOMALY_TOPIC', 'anomaly-results')
MODEL_PATH = os.getenv('HEAVY_MODEL_PATH', '/app/models/model_heavy.pkl')


def load_model():
    """Load the heavy model on startup."""
    global heavy_model
    
    try:
        if os.path.exists(MODEL_PATH):
            heavy_model = joblib.load(MODEL_PATH)
            print(f"✓ Heavy model loaded from {MODEL_PATH}")
            
            # Log model info
            import sys
            size_mb = sys.getsizeof(heavy_model) / (1024 * 1024)
            print(f"  Model size: ~{size_mb:.2f} MB in memory")
            return True
        else:
            print(f"✗ Heavy model not found at {MODEL_PATH}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading heavy model: {e}")
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


def process_batch(message_data: dict) -> dict:
    """
    Process a batch of sensor readings through the heavy model.
    
    Args:
        message_data: Dict containing 'device_id' and 'readings' array
        
    Returns:
        Dict with processed results including anomaly predictions
    """
    global heavy_model, monitor
    
    device_id = message_data.get('device_id', 'unknown')
    readings = message_data.get('readings', [])
    
    if not readings:
        return {'device_id': device_id, 'readings': [], 'error': 'No readings provided'}
    
    if heavy_model is None:
        return {'device_id': device_id, 'readings': [], 'error': 'Model not loaded'}
    
    results = []
    anomaly_count = 0
    
    for reading in readings:
        timestamp = reading.get('timestamp', datetime.utcnow().timestamp())
        vibration = reading.get('vibration', 0.0)
        
        try:
            # Prepare input for model
            X = np.array([[vibration]])
            
            # Get prediction (-1 = anomaly, 1 = normal)
            prediction = heavy_model.predict(X)[0]
            score = heavy_model.score_samples(X)[0]
            
            is_anomaly = (prediction == -1)
            
            # Calculate confidence (normalize score to 0-1 range)
            # Lower scores = more anomalous, typical range is -10 to 0
            confidence = min(1.0, max(0.0, (score + 10) / 10))
            if is_anomaly:
                confidence = 1.0 - confidence  # Invert for anomalies
            
            result = {
                'timestamp': timestamp,
                'vibration': vibration,
                'is_anomaly': is_anomaly,
                'confidence': float(confidence),
                'anomaly_score': float(score)
            }
            results.append(result)
            
            if is_anomaly:
                anomaly_count += 1
                print(f"🚨 ANOMALY | Device: {device_id} | Value: {vibration:7.4f} | "
                      f"Score: {score:8.4f} | Confidence: {confidence:.2%}")
                
                # Log to Cloud Monitoring
                if monitor:
                    monitor.log_anomaly(
                        timestamp=timestamp,
                        vibration=vibration,
                        confidence=confidence,
                        device_id=device_id
                    )
            else:
                print(f"✓ Normal  | Device: {device_id} | Value: {vibration:7.4f} | Score: {score:8.4f}")
                
        except Exception as e:
            print(f"✗ Error processing reading: {e}")
            results.append({
                'timestamp': timestamp,
                'vibration': vibration,
                'is_anomaly': False,
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Summary logging
    if anomaly_count > 0:
        print(f"\n📊 Batch Summary: {anomaly_count}/{len(readings)} anomalies detected "
              f"({anomaly_count/len(readings)*100:.1f}%)\n")
    
    return {
        'device_id': device_id,
        'readings': results,
        'count': len(results),
        'anomalies_detected': anomaly_count,
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
    return jsonify({
        'status': 'healthy',
        'model_loaded': heavy_model is not None,
        'pubsub_connected': publisher is not None and subscriber is not None,
        'monitoring_enabled': monitor is not None,
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint."""
    return jsonify({
        'service': 'heavy-model-api',
        'version': '1.0.0',
        'description': 'Cloud anomaly detection service using heavy model via Pub/Sub',
        'endpoints': {
            '/health': 'Health check',
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
    print("Heavy Model API Service - Cloud Anomaly Detection")
    print("=" * 70)
    print(f"Project: {PROJECT_ID}")
    print(f"Sensor Topic: {SENSOR_TOPIC}")
    print(f"Anomaly Topic: {ANOMALY_TOPIC}")
    print(f"Model Path: {MODEL_PATH}")
    print("=" * 70)
    
    # Register signal handlers
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize components
    if not load_model():
        print("⚠ Warning: Model not loaded, service will return errors")
    
    if not init_pubsub():
        print("✗ Failed to initialize Pub/Sub, exiting...")
        sys.exit(1)
    
    init_monitoring()
    
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
        
        print("[Service] Goodbye!")


if __name__ == '__main__':
    main()

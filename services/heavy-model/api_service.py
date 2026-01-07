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
import time
from datetime import datetime
from concurrent.futures import TimeoutError as FuturesTimeoutError

# Suppress sklearn warnings to reduce log noise
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', message='.*sklearn.utils.parallel.*')

from flask import Flask, jsonify
import numpy as np
from google.cloud import pubsub_v1

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
batch_semaphore = threading.Semaphore(1)  # Only allow 1 batch at a time
batch_counter = 0  # Counter for batch IDs

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
                country_iso_code=os.getenv('CARBON_COUNTRY_CODE', 'AUT')
            )
            print("✓ Carbon Monitoring initialized (PERFORMANCE mode)")
            return True
        except Exception as e:
            print(f"⚠ Carbon Monitoring not available: {e}")
    
    return False


def process_batch(message_data: dict) -> dict:
    """
    Process a batch of sensor readings through the hybrid ML detector.
    Processes readings one-by-one with logging after each.
    
    Args:
        message_data: Dict containing 'device_id' and 'readings' array
        
    Returns:
        Dict with processed results including anomaly predictions
    """
    global detector, monitor, carbon_monitor
    
    device_id = message_data.get('device_id', 'unknown')
    readings = message_data.get('readings', [])
    batch_size = len(readings)
    
    if not readings:
        return {'device_id': device_id, 'readings': [], 'error': 'No readings provided'}
    
    if detector is None:
        return {'device_id': device_id, 'readings': [], 'error': 'Detector not initialized'}
    
    results = []
    anomaly_count = 0
    
    # Process each reading one by one
    for i, reading in enumerate(readings, 1):
        timestamp = reading.get('timestamp', datetime.utcnow().timestamp())
        vibration = reading.get('vibration', reading.get('value', 0.0))
        
        print(f"  Processing {i}/{batch_size}...")
        
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
            
            # Log result
            status = "🚨 ANOMALY" if is_anomaly else "✓ normal"
            print(f"  Finished {i}/{batch_size}: v={vibration:.4f} → {status}")
            
            if is_anomaly:
                anomaly_count += 1
                # Log to Cloud Monitoring (async, won't block)
                if monitor:
                    monitor.log_anomaly(
                        timestamp=timestamp,
                        vibration=vibration,
                        confidence=confidence,
                        device_id=device_id
                    )
                    
        except Exception as e:
            print(f"  ✗ Error processing {i}/{batch_size}: {e}")
            results.append({
                'timestamp': timestamp,
                'vibration': vibration,
                'is_anomaly': False,
                'confidence': 0.0,
                'error': str(e)
            })
    
    # Track carbon after all readings processed
    if carbon_monitor:
        print(f"  Tracking carbon emissions for {batch_size} inferences...")
        with carbon_monitor.track_inference(batch_size=batch_size):
            pass  # Just tracking the batch
        print(f"  ✓ Carbon tracked")
    
    # Get stats
    stats = detector.get_stats()
    carbon_stats = carbon_monitor.get_stats() if carbon_monitor else {}
    
    print(f"  ✓ Batch complete: {anomaly_count}/{batch_size} anomalies")
    
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
        
    Returns:
        bool: True if publish succeeded, False otherwise
    """
    global publisher
    
    if publisher is None:
        return False
    
    try:
        topic_path = publisher.topic_path(PROJECT_ID, ANOMALY_TOPIC)
        data = json.dumps(results).encode('utf-8')
        
        future = publisher.publish(topic_path, data)
        future.result(timeout=10)
        return True
        
    except Exception as e:
        print(f"  ✗ Publish failed: {e}")
        return False


def message_callback(message):
    """
    Callback for processing incoming Pub/Sub messages.
    Uses semaphore to block ALL other callbacks until this one completes.
    ACKs only after complete processing.
    
    Args:
        message: Pub/Sub message object
    """
    global batch_counter, batch_semaphore
    
    import threading
    thread_id = threading.current_thread().ident
    print(f"[Thread-{thread_id}] Attempting to acquire semaphore...")
    
    # Block here until no other batch is processing - this is THE critical section
    batch_semaphore.acquire()
    print(f"[Thread-{thread_id}] Semaphore acquired!")
    
    try:
        # Decode message
        data = json.loads(message.data.decode('utf-8'))
        
        device_id = data.get('device_id', 'unknown')
        reading_count = data.get('count', len(data.get('readings', [])))
        
        # Assign batch ID for tracking
        batch_counter += 1
        batch_id = batch_counter
        
        print(f"\n{'='*60}")
        print(f"[Batch #{batch_id} / Thread-{thread_id}] Received {reading_count} readings from {device_id}")
        print(f"{'='*60}")
        
        # Process through heavy model
        start_time = time.perf_counter()
        results = process_batch(data)
        processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Publish results
        print(f"  Publishing results to anomaly-results topic...")
        publish_success = publish_results(results)
        
        # Summary
        print(f"\n[Batch #{batch_id}] COMPLETE")
        print(f"  Time: {processing_time_ms:.1f} ms ({processing_time_ms/reading_count:.1f} ms/reading)")
        print(f"  Anomalies: {results.get('anomalies_detected', 0)}/{reading_count}")
        print(f"  Published: {'✓' if publish_success else '✗'}")
        print(f"{'='*60}\n")
        
        # Acknowledge AFTER processing completes
        message.ack()
            
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON in message: {e}")
        message.ack()
        
    except Exception as e:
        print(f"✗ Error processing message: {e}")
        import traceback
        traceback.print_exc()
        message.ack()
    
    finally:
        # Always release semaphore to allow next batch
        print(f"[Thread-{thread_id}] Releasing semaphore...")
        batch_semaphore.release()
        print(f"[Thread-{thread_id}] Semaphore released!")


def start_subscriber():
    """Start the Pub/Sub subscriber with automatic stream restart on timeout."""
    global subscriber
    
    if subscriber is None:
        print("✗ Subscriber not initialized")
        return None
    
    subscription_path = subscriber.subscription_path(PROJECT_ID, SENSOR_SUBSCRIPTION)
    
    print(f"\n[Subscriber] Listening on {subscription_path}...")
    print(f"[Subscriber] Mode: Sequential (1 batch at a time)")
    
    # Configure flow control: only 1 message at a time for sequential processing
    flow_control = pubsub_v1.types.FlowControl(
        max_messages=1,  # Process exactly 1 batch at a time - fully sequential
        max_bytes=10 * 1024 * 1024,  # 10 MB
    )
    
    # Use a single-threaded executor to ensure truly sequential processing
    from concurrent.futures import ThreadPoolExecutor
    executor = ThreadPoolExecutor(max_workers=1)
    
    streaming_pull_future = subscriber.subscribe(
        subscription_path,
        callback=message_callback,
        flow_control=flow_control,
        scheduler=pubsub_v1.subscriber.scheduler.ThreadScheduler(executor=executor)
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
    
    # Block until shutdown with automatic stream restart
    try:
        while not shutdown_event.is_set():
            try:
                # Wait for stream completion with 1-second timeout
                streaming_future.result(timeout=1.0)
                # If we get here, stream ended unexpectedly
                print("⚠ Subscriber stream ended, restarting...")
                streaming_future = start_subscriber()
            except FuturesTimeoutError:
                # Normal - stream still running
                continue
            except Exception as e:
                error_msg = str(e)
                # Check for known timeout/session expiry errors
                if "OutOfRange" in error_msg or "maximum allowed duration" in error_msg:
                    print(f"⚠ Stream timeout (1h limit reached), restarting subscriber...")
                    try:
                        streaming_future.cancel()
                    except:
                        pass
                    streaming_future = start_subscriber()
                    if streaming_future:
                        print("✓ Subscriber restarted successfully")
                        continue
                    else:
                        print("✗ Failed to restart subscriber")
                        break
                else:
                    print(f"✗ Subscriber error: {e}")
                    break
    finally:
        print("\n[Service] Shutting down...")
        try:
            streaming_future.cancel()
        except:
            pass
        
        if monitor:
            monitor.flush()
        
        if carbon_monitor:
            carbon_monitor.flush()
        
        print("[Service] Goodbye!")


if __name__ == '__main__':
    main()

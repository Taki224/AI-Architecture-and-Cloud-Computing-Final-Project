"""
Main entry point for Edge Device Vibration Sensor Simulator
Carbon-Aware IoT Anomaly Detection System

Supports two operation modes:
- PERFORMANCE: Streams data via Pub/Sub to cloud heavy model
- ECO: Uses local REST API with light model
"""
import tkinter as tk
import os
from sensor_simulator import VibrationSensor
from gui import Application
from pubsub_client import create_publisher, create_subscriber


def main():
    """Initialize and run the vibration sensor simulator GUI"""
    
    # Initialize sensor with demo-friendly parameters
    # ~2-3 anomalies per minute at 100ms intervals
    sensor = VibrationSensor(
        mean=0.0,
        std_dev=1.0,
        anomaly_rate=0.003,  # ~2 per minute
        small_anomaly_range=(3.0, 4.0),  # Easier to detect
        large_anomaly_range=(5.0, 8.0),  # Very obvious
        large_anomaly_ratio=0.4  # 40% are large
    )
    
    # Configure ML API URL from environment variable or use default
    ml_api_url = os.getenv('ML_API_URL', 'http://localhost:5001')
    sensor.set_ml_api_url(ml_api_url)
    
    # Initialize Pub/Sub for PERFORMANCE mode
    device_id = os.getenv('DEVICE_ID', 'edge-001')
    publisher = create_publisher(device_id)
    
    if publisher:
        sensor.set_pubsub_publisher(publisher)
        print(f"✓ Pub/Sub publisher initialized for device: {device_id}")
    else:
        print("⚠ Pub/Sub not available - PERFORMANCE mode will be limited")
    
    # Create anomaly subscriber for receiving cloud results
    def on_anomaly_results(data):
        """Callback for anomaly results from cloud."""
        readings = data.get('readings', [])
        for reading in readings:
            if reading.get('is_anomaly'):
                sensor.cloud_anomalies_detected += 1
                print(f"🚨 [CLOUD] Anomaly detected: vibration={reading['vibration']:.4f}, "
                      f"confidence={reading.get('confidence', 0):.4f}")
    
    subscriber = create_subscriber(on_anomaly_results)
    if subscriber:
        subscriber.start()
        sensor.set_pubsub_subscriber(subscriber)
        print("✓ Pub/Sub subscriber started for anomaly results")
    
    # Create tkinter root window
    root = tk.Tk()
    
    # Initialize GUI application with Pub/Sub status
    app = Application(root, sensor, pubsub_connected=publisher is not None)
    
    # Start the GUI
    print("=" * 60)
    print("Edge Device - Vibration Sensor Simulator")
    print("Carbon-Aware IoT Anomaly Detection System")
    print("=" * 60)
    print("\nStarting GUI...")
    print("- Click 'Start Sensor' to begin data generation")
    print("- Toggle between PERFORMANCE and ECO modes")
    print("- PERFORMANCE mode: Streams to cloud via Pub/Sub")
    print("- ECO mode: Uses local light model via REST")
    print("- Watch for color-coded anomalies:")
    print("  • Blue = Normal readings")
    print("  • Orange = Small anomalies (3-4σ)")
    print("  • Red = Large anomalies (5-8σ)")
    print("=" * 60)
    
    try:
        app.run()
    finally:
        # Cleanup on exit
        if publisher:
            publisher.flush()
        if subscriber:
            subscriber.stop()


if __name__ == "__main__":
    main()

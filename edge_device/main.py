"""
Main entry point for Edge Device Vibration Sensor Simulator
Carbon-Aware IoT Anomaly Detection System
"""
import tkinter as tk
from sensor_simulator import VibrationSensor
from gui import Application


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
    
    # Create tkinter root window
    root = tk.Tk()
    
    # Initialize GUI application
    app = Application(root, sensor)
    
    # Start the GUI
    print("=" * 60)
    print("Edge Device - Vibration Sensor Simulator")
    print("Carbon-Aware IoT Anomaly Detection System")
    print("=" * 60)
    print("\nStarting GUI...")
    print("- Click 'Start Sensor' to begin data generation")
    print("- Toggle between PERFORMANCE and ECO modes")
    print("- Watch for color-coded anomalies:")
    print("  • Blue = Normal readings")
    print("  • Orange = Small anomalies (3-4σ)")
    print("  • Red = Large anomalies (5-8σ)")
    print("=" * 60)
    
    app.run()


if __name__ == "__main__":
    main()

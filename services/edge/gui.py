"""
Tkinter GUI for Vibration Sensor Simulator
Real-time visualization of sensor data with anomaly detection
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation
from collections import deque
from typing import Optional
import time

from sensor_simulator import VibrationSensor, OperationMode


class Application:
    """
    Main GUI application for vibration sensor visualization.
    Displays real-time chart with color-coded anomalies.
    Supports Pub/Sub status display for PERFORMANCE mode.
    """
    
    def __init__(self, root: tk.Tk, sensor: VibrationSensor, pubsub_connected: bool = False):
        """
        Initialize the GUI application.
        
        Args:
            root: The tkinter root window
            sensor: The VibrationSensor instance
            pubsub_connected: Whether Pub/Sub is connected for PERFORMANCE mode
        """
        self.root = root
        self.sensor = sensor
        self.pubsub_connected = pubsub_connected
        
        # Configure window
        self.root.title("Edge Device - Vibration Sensor Simulator")
        self.root.geometry("1200x700")
        
        # Data storage for visualization (30 seconds at 100ms = 300 points)
        self.max_points = 300
        self.times = deque(maxlen=self.max_points)
        self.values = deque(maxlen=self.max_points)
        self.anomalies = deque(maxlen=self.max_points)
        self.severities = deque(maxlen=self.max_points)
        
        self.start_time = time.time()
        self.update_interval = 100  # milliseconds
        
        # Create GUI components
        self._create_widgets()
        
        # Animation for real-time updates
        self.animation = None
        
    def _create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # ========== Control Panel ==========
        control_frame = ttk.LabelFrame(main_frame, text="Control Panel", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Start/Stop buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=0, column=0, padx=5)
        
        self.start_button = ttk.Button(
            button_frame,
            text="▶ Start Sensor",
            command=self._start_sensor,
            width=15
        )
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame,
            text="⏸ Stop Sensor",
            command=self._stop_sensor,
            state=tk.DISABLED,
            width=15
        )
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Carbon Mode Toggle
        mode_frame = ttk.Frame(control_frame)
        mode_frame.grid(row=0, column=1, padx=20)
        
        ttk.Label(mode_frame, text="Carbon Mode:").grid(row=0, column=0, padx=5)
        
        self.mode_var = tk.StringVar(value="PERFORMANCE")
        self.mode_button = ttk.Button(
            mode_frame,
            text="🌍 PERFORMANCE",
            command=self._toggle_mode,
            width=18
        )
        self.mode_button.grid(row=0, column=1, padx=5)
        
        # Statistics display
        stats_frame = ttk.Frame(control_frame)
        stats_frame.grid(row=0, column=2, padx=20)
        
        self.status_label = ttk.Label(
            stats_frame,
            text="Status: Stopped",
            font=('Arial', 10, 'bold')
        )
        self.status_label.grid(row=0, column=0, padx=5)
        
        self.anomaly_label = ttk.Label(
            stats_frame,
            text="Anomalies: 0 (0 small, 0 large)",
            font=('Arial', 9)
        )
        self.anomaly_label.grid(row=0, column=1, padx=5)
        
        # Pub/Sub connection status
        self.pubsub_label = ttk.Label(
            stats_frame,
            text="Pub/Sub: " + ("Connected ✓" if self.pubsub_connected else "Disconnected"),
            font=('Arial', 9),
            foreground='green' if self.pubsub_connected else 'gray'
        )
        self.pubsub_label.grid(row=0, column=2, padx=10)
        
        # ========== Chart Panel ==========
        chart_frame = ttk.LabelFrame(main_frame, text="Real-Time Vibration Data (30s Window)", padding="10")
        chart_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        
        # Configure plot appearance
        self.ax.set_xlabel('Time (seconds)', fontsize=10)
        self.ax.set_ylabel('Vibration Amplitude', fontsize=10)
        self.ax.set_title('Vibration Sensor Data Stream', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim(-10, 10)
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Legend info
        legend_frame = ttk.Frame(chart_frame)
        legend_frame.grid(row=1, column=0, pady=5)
        
        ttk.Label(legend_frame, text="Note:", font=('Arial', 9, 'bold')).grid(row=0, column=0, padx=5)
        
        ttk.Label(legend_frame, text="Raw sensor data shown continuously.", font=('Arial', 9)).grid(row=0, column=1, padx=5)
        ttk.Label(legend_frame, text="Subtle markers indicate injected anomalies (for ground truth).", 
                 font=('Arial', 9, 'italic'), foreground='gray').grid(row=0, column=2, padx=5)
        
    def _start_sensor(self):
        """Start the sensor simulation and data collection"""
        self.sensor.start()
        self.start_time = time.time()
        
        # Update button states
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Start data collection
        self._collect_data()
        
    def _stop_sensor(self):
        """Stop the sensor simulation"""
        self.sensor.stop()
        
        # Update button states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        
    def _toggle_mode(self):
        """Toggle between PERFORMANCE and ECO modes"""
        if self.mode_var.get() == "PERFORMANCE":
            self.mode_var.set("ECO")
            self.sensor.set_mode(OperationMode.ECO)
            self.mode_button.config(text="🌱 ECO")
        else:
            self.mode_var.set("PERFORMANCE")
            self.sensor.set_mode(OperationMode.PERFORMANCE)
            self.mode_button.config(text="🌍 PERFORMANCE")
    
    def _collect_data(self):
        """Collect data from sensor and update visualization"""
        if not self.sensor.is_running:
            return
        
        # Generate new reading
        value, is_anomaly, severity = self.sensor.generate_reading()
        
        # Store data
        current_time = time.time() - self.start_time
        self.times.append(current_time)
        self.values.append(value)
        self.anomalies.append(is_anomaly)
        self.severities.append(severity)
        
        # Mode-specific processing
        if self.sensor.mode == OperationMode.PERFORMANCE:
            # PERFORMANCE mode: publish to Pub/Sub for cloud processing
            if self.sensor.pubsub_available:
                self.sensor.publish_to_pubsub(value)
            else:
                # Fallback: log that Pub/Sub is not available
                if is_anomaly:
                    print(f"⚠️  [PERFORMANCE Mode] Pub/Sub unavailable - anomaly not sent to cloud")
        elif self.sensor.mode == OperationMode.ECO:
            # ECO mode: send data to local ML model API for analysis
            api_result = self.sensor.run_local_inference(value)
            if api_result:
                # Always log API anomaly detections
                if api_result['is_anomaly']:
                    print(f"\n🚨 [ECO Mode] ANOMALY DETECTED BY API")
                    print(f"   Value: {value:.4f}")
                    print(f"   Anomaly Score: {api_result['anomaly_score']:.4f}")
                    print(f"   Ground Truth: {'ANOMALY (' + severity + ')' if is_anomaly else 'Normal (False Positive)'}")
                    print(f"   Timestamp: {time.time():.2f}\n")
                # Log missed anomalies
                elif is_anomaly:
                    print(f"⚠️  [ECO Mode] API missed anomaly (ground truth: {severity}) - Value: {value:.4f}")
            else:
                # Log when API call fails
                if is_anomaly:
                    print(f"⚠️  [ECO Mode] API not available - missed anomaly: {severity}, Value: {value:.4f}")
        
        # Update chart
        self._update_chart()
        
        # Update statistics
        self._update_statistics()
        
        # Schedule next collection
        self.root.after(self.update_interval, self._collect_data)
    
    def _update_chart(self):
        """Update the matplotlib chart with current data"""
        self.ax.clear()
        
        # Configure plot
        self.ax.set_xlabel('Time (seconds)', fontsize=10)
        self.ax.set_ylabel('Vibration Amplitude', fontsize=10)
        self.ax.set_title('Vibration Sensor Data Stream', fontsize=12, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        if len(self.times) > 0:
            times_list = list(self.times)
            values_list = list(self.values)
            anomalies_list = list(self.anomalies)
            severities_list = list(self.severities)
            
            # Plot continuous line through ALL points (normal and anomalies)
            self.ax.plot(times_list, values_list, 'b-', linewidth=1, alpha=0.7)
            
            # Plot all points with subtle markers
            self.ax.scatter(times_list, values_list, c='blue', s=15, alpha=0.5, zorder=3)
            
            # Subtly highlight small anomalies (slightly larger, faint orange edge)
            small_times = [t for t, a, s in zip(times_list, anomalies_list, severities_list) 
                          if a and s == 'small']
            small_values = [v for v, a, s in zip(values_list, anomalies_list, severities_list) 
                           if a and s == 'small']
            
            if small_times:
                self.ax.scatter(small_times, small_values, c='blue', s=25, 
                              edgecolors='orange', linewidths=1, alpha=0.6, zorder=4)
            
            # Subtly highlight large anomalies (slightly larger, faint red edge)
            large_times = [t for t, a, s in zip(times_list, anomalies_list, severities_list) 
                          if a and s == 'large']
            large_values = [v for v, a, s in zip(values_list, anomalies_list, severities_list) 
                           if a and s == 'large']
            
            if large_times:
                self.ax.scatter(large_times, large_values, c='blue', s=35, 
                              edgecolors='red', linewidths=1.5, alpha=0.7, zorder=5)
            
            # Auto-scale y-axis with some padding
            if values_list:
                y_min = min(values_list) - 1
                y_max = max(values_list) + 1
                self.ax.set_ylim(y_min, y_max)
            
            # Set x-axis to show last 30 seconds
            if times_list:
                x_max = times_list[-1]
                x_min = max(0, x_max - 30)
                self.ax.set_xlim(x_min, x_max)
        
        self.canvas.draw()
    
    def _update_statistics(self):
        """Update the statistics display"""
        stats = self.sensor.get_statistics()
        
        # Update status
        mode = self.mode_var.get()
        self.status_label.config(
            text=f"Status: Running ({mode})",
            foreground='green'
        )
        
        # Update anomaly count
        api_detected = stats.get('api_anomalies_detected', 0)
        cloud_detected = stats.get('cloud_anomalies_detected', 0)
        batches_sent = stats.get('pubsub_batches_sent', 0)
        
        anomaly_text = (f"Anomalies: {stats['total_anomalies']} "
                       f"({stats['small_anomalies']} small, {stats['large_anomalies']} large)")
        
        # Add mode-specific detection count
        if mode == "ECO" and api_detected > 0:
            anomaly_text += f" | API: {api_detected}"
        elif mode == "PERFORMANCE":
            anomaly_text += f" | Cloud: {cloud_detected} | Batches: {batches_sent}"
        
        self.anomaly_label.config(text=anomaly_text)
    
    def run(self):
        """Start the GUI main loop"""
        self.root.mainloop()

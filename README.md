# Carbon-Aware IoT Anomaly Detection System

A dynamic hybrid edge-cloud system that optimizes anomaly detection deployment based on electricity grid carbon intensity.

## Project Overview

This system simulates an edge device with vibration sensors and implements carbon-aware computing by switching between:
- **Performance Mode** (Low Carbon): Stream raw data to cloud for heavy model inference
- **Eco Mode** (High Carbon): Run lightweight model locally on edge device

## Current Implementation Status

### ✅ Completed
- **Edge Device Simulator** with realistic vibration sensor data generation
- **Real-time GUI** for sensor visualization
- **Training Data Generation** pipeline
- **Isolation Forest Models** (Heavy: 200 estimators, Light: 10 estimators)

### 🚧 Coming Next
- Google Cloud Pub/Sub integration
- Cloud Run deployment for heavy model
- CarbonController service
- CI/CD pipeline with GitHub Actions

## Project Structure

```
FinalProject/
├── edge_device/           # Edge device simulation
│   ├── sensor_simulator.py    # Sensor data generation logic
│   ├── gui.py                  # Tkinter GUI application
│   ├── main.py                 # Entry point for GUI
│   ├── generate_training_data.py  # Training data generator
│   ├── requirements.txt
│   ├── training_data.csv       # Generated training set
│   └── validation_data.csv     # Generated validation set
│
├── models/                # Machine learning models
│   ├── train_models.py         # Model training script
│   ├── model_heavy.pkl         # Heavy model (200 estimators)
│   ├── model_light.pkl         # Light model (10 estimators)
│   └── requirements.txt
│
└── plans/                 # Architecture documentation
    ├── adr.md                  # Architectural decision records
    ├── uml.plantuml
    ├── sequence.plantuml
    └── development.plantuml
```

## Getting Started

### 1. Run Edge Device GUI

Visualize real-time sensor data with anomaly generation:

```bash
cd edge_device
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

**GUI Controls:**
- Click "▶ Start Sensor" to begin data generation
- Toggle between "🌍 PERFORMANCE" and "🌱 ECO" modes
- Watch for subtle anomaly markers (faint orange/red edges)
- Data updates every 100ms with ~2-3 anomalies per minute

### 2. Generate Training Data

Create datasets for model training:

```bash
cd edge_device
python generate_training_data.py
```

This generates:
- `training_data.csv`: 100,000 samples (~300 anomalies)
- `validation_data.csv`: 20,000 samples (~60 anomalies)

### 3. Train Models

Train both heavy (cloud) and light (edge) models:

```bash
cd models
pip install -r requirements.txt
python train_models.py
```

Output:
- `model_heavy.pkl`: 200 estimators (cloud deployment)
- `model_light.pkl`: 10 estimators (edge deployment)

## Sensor Data Characteristics

- **Distribution**: Normal (μ=0, σ=1)
- **Sampling Rate**: 100ms intervals
- **Anomaly Types**:
  - Small anomalies: 3-4σ deviation (easier to detect)
  - Large anomalies: 5-8σ deviation (obvious)
- **Anomaly Rate**: ~0.3% (2-3 per minute for demos)

## Model Performance

After training, both models are evaluated on:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- False Positive/Negative Rates

Expected trade-off:
- **Heavy Model**: Higher accuracy, more computational cost
- **Light Model**: Lower accuracy, faster inference, lower power

## Architecture Principles

Based on **ADR 001** (see [plans/adr.md](plans/adr.md)):
- Dynamic mode switching based on grid carbon intensity
- Offline resilience with local inference capability
- Trade accuracy for sustainability during high-carbon periods
- Fan-in architecture with Google Cloud Pub/Sub

## Development Notes

- Placeholder methods exist for Pub/Sub integration (not yet functional)
- GUI shows raw sensor data; anomaly markers are ground truth references
- Models trained on single-feature data (vibration amplitude only)

---

**Status**: Phase 1 Complete - Data Generation & Model Training
**Next**: Cloud Integration & Carbon-Aware Controller

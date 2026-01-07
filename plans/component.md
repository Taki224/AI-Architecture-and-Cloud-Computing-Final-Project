# Component Diagram

This document describes the logical component architecture of the Carbon-Aware IoT Anomaly Detection System.

---

## System Components Overview

```mermaid
flowchart TB
    subgraph Edge[Edge Device]
        GUI[GUI - Tkinter]
        Controller[CarbonAwareController]
        Sensor[SensorSimulator]
        EdgePubSub[PubSubClient]
    end

    subgraph LightService[Light Model Service]
        LightAPI[Flask REST API]
        LightDetector[IsolationForestDetector<br/>50 estimators]
        LightCarbon[CarbonMonitor ECO]
    end

    subgraph HeavyService[Heavy Model Service]
        HeavySubscriber[PubSub Subscriber]
        HeavyDetector[HybridDetector<br/>200 estimators]
        HeavyPublisher[PubSub Publisher]
        HeavyCarbon[CarbonMonitor PERF]
        AnomalyMon[AnomalyMonitor]
    end

    subgraph Messaging[Message Broker]
        SensorTopic[sensor-data]
        AnomalyTopic[anomaly-results]
    end

    subgraph CloudObservability[GCP Observability]
        CloudMonitoring[Cloud Monitoring<br/>Custom Metrics]
        CloudLogging[Cloud Logging<br/>Structured Logs]
    end

    GUI <--> Controller
    Controller <--> Sensor
    Controller <--> EdgePubSub

    Controller -->|ECO POST /analyze| LightAPI
    LightAPI --> LightDetector
    LightDetector --> LightAPI
    LightAPI -->|JSON response| Controller
    LightDetector --> LightCarbon
    LightCarbon -->|carbon metrics| CloudMonitoring

    EdgePubSub -->|publish batch| SensorTopic
    SensorTopic --> HeavySubscriber
    HeavySubscriber --> HeavyDetector
    HeavyDetector --> HeavyPublisher
    HeavyDetector --> HeavyCarbon
    HeavyDetector --> AnomalyMon
    HeavyPublisher --> AnomalyTopic
    AnomalyTopic -->|predictions| EdgePubSub
    
    HeavyCarbon -->|carbon metrics| CloudMonitoring
    AnomalyMon -->|anomaly rate| CloudMonitoring
    AnomalyMon -->|structured logs| CloudLogging
```

**Component Details:**

| Component | Port | Description |
|-----------|------|-------------|
| GUI | - | Tkinter + Matplotlib, 1200x700 window |
| Controller | - | Mode switching: PERFORMANCE or ECO |
| Sensor | - | Gaussian noise (mean=0, std=1), 10 Hz |
| PubSubClient | - | Batches 10 readings for cloud |
| Light API | 5001 | Flask REST: /analyze, /health, /stats, /reset |
| Light Detector | - | IsolationForest (50 estimators) + Z-score |
| Heavy Detector | 8080 | HybridDetector (200 estimators IF + Z-score) |
| CarbonMonitor | - | Per-inference carbon tracking, GCP metrics export |
| AnomalyMonitor | - | 60s rolling window rate, Cloud Logging integration |

---

## ML Detector Component Structure

```mermaid
flowchart TB
    subgraph HybridDetector[HybridAnomalyDetector]
        Input[detect value]
        IsoForest[IsolationForestDetector]
        StatDetector[StatisticalDetector]
        Ensemble[OR Gate Ensemble]
        Output[return result]
    end

    Input --> IsoForest
    Input --> StatDetector
    IsoForest -->|ml_anomaly| Ensemble
    StatDetector -->|stat_anomaly| Ensemble
    Ensemble --> Output
```

**Detector Configuration:**

| Detector | Light Model | Heavy Model |
|----------|-------------|-------------|
| IsolationForest | 100 estimators | 200 estimators |
| Window Size | 50 samples | 50 samples |
| Z-score Threshold | 3.0 sigma | 3.0 sigma |
| Ensemble Logic | OR gate | OR gate |

---

## Component-to-Code Mapping

| Component | Source File | Description |
|-----------|-------------|-------------|
| GUI | `services/edge/local_sensor_gui.py` | Tkinter GUI class with Matplotlib chart |
| CarbonAwareController | `services/edge/local_sensor_gui.py` | Mode switching and orchestration logic |
| SensorSimulator | `services/edge/local_sensor_gui.py` | Gaussian sensor with anomaly injection |
| PubSubClient | `services/edge/local_sensor_gui.py` | Pub/Sub publish/subscribe wrapper |
| LightModelAPI | `services/light-model/api_service.py` | Flask REST endpoints |
| HeavyModelAPI | `services/heavy-model/api_service.py` | Flask health + Pub/Sub subscriber |
| HybridAnomalyDetector | `models/hybrid_detector.py` | Ensemble detector orchestration |
| IsolationForestDetector | `models/isolation_forest_detector.py` | Online-trained Isolation Forest |
| StatisticalAnomalyDetector | `models/statistical_model.py` | Z-score threshold detector |
| CarbonMonitor | `services/common/carbon_monitoring.py` | Carbon emissions tracking and GCP export |
| AnomalyMonitor | `services/heavy-model/monitoring.py` | Anomaly rate calculation and Cloud Logging |

---

## Interface Definitions

### Light Model REST API (Port 5001)

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/health` | GET | - | `{"status": "healthy", "ml_fitted": bool}` |
| `/analyze` | POST | `{"value": float}` | `{"is_anomaly": bool, "method": str, "scores": {...}}` |
| `/analyze/batch` | POST | `{"readings": [...]}` | `{"summary": {...}, "results": [...]}` |
| `/stats` | GET | - | `{"total_detections": int, "anomaly_rate": float}` |
| `/reset` | POST | - | `{"status": "reset"}` |

### Heavy Model Service (Port 8080)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check for Cloud Run |
| `/` | GET | Service info |

### Pub/Sub Message Formats

| Topic | Direction | Message Format |
|-------|-----------|----------------|
| `sensor-data` | Edge to Cloud | `{"device_id": str, "readings": [...], "batch_id": str}` |
| `anomaly-results` | Cloud to Edge | `{"batch_id": str, "predictions": [...], "ml_fitted": bool}` |

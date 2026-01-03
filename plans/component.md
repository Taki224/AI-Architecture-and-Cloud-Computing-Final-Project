# Component Diagram

This document describes the logical component architecture of the Carbon-Aware IoT Anomaly Detection System.

---

## System Components Overview

```mermaid
flowchart TB
    subgraph Edge["Edge Device"]
        GUI["GUI<br/>───<br/>Tkinter + Matplotlib<br/>1200×700 window"]
        Controller["CarbonAwareController<br/>───<br/>Mode: PERFORMANCE | ECO"]
        Sensor["SensorSimulator<br/>───<br/>μ=0, σ=1, 10 Hz"]
        EdgePubSub["PubSubClient<br/>───<br/>batch_size=10"]
    end

    subgraph LightService["Light Model Service :5001"]
        LightAPI["Flask REST API<br/>───<br/>/analyze, /health"]
        LightDetector["HybridAnomalyDetector<br/>───<br/>100 estimators"]
    end

    subgraph HeavyService["Heavy Model Service :8080"]
        HeavySubscriber["Pub/Sub Subscriber"]
        HeavyDetector["HybridAnomalyDetector<br/>───<br/>200 estimators"]
        HeavyPublisher["Pub/Sub Publisher"]
    end

    subgraph Messaging["Message Broker (Pub/Sub)"]
        SensorTopic["sensor-readings topic"]
        AnomalyTopic["anomaly-results topic"]
    end

    %% Edge internal connections
    GUI <--> Controller
    Controller <--> Sensor
    Controller <--> EdgePubSub

    %% ECO mode flow (local)
    Controller -->|"ECO: POST /analyze"| LightAPI
    LightAPI --> LightDetector
    LightDetector --> LightAPI
    LightAPI -->|"JSON response"| Controller

    %% PERFORMANCE mode flow (cloud)
    EdgePubSub -->|"publish batch"| SensorTopic
    SensorTopic --> HeavySubscriber
    HeavySubscriber --> HeavyDetector
    HeavyDetector --> HeavyPublisher
    HeavyPublisher --> AnomalyTopic
    AnomalyTopic -->|"predictions"| EdgePubSub
```

---

## ML Detector Component Structure

```mermaid
flowchart TB
    subgraph HybridDetector["HybridAnomalyDetector"]
        direction TB
        
        Input["detect(value)"]
        
        subgraph Detectors["Detection Engines"]
            IsoForest["IsolationForestDetector<br/>───<br/>n_estimators: 100 or 200<br/>window_size: 50"]
            StatDetector["StatisticalAnomalyDetector<br/>───<br/>z_threshold: 3.0σ"]
        end
        
        Ensemble["Ensemble: OR gate<br/>───<br/>is_anomaly = ml OR stat"]
        
        Output["return {is_anomaly, method, scores}"]
    end

    Input --> IsoForest
    Input --> StatDetector
    IsoForest -->|"ml_anomaly"| Ensemble
    StatDetector -->|"stat_anomaly"| Ensemble
    Ensemble --> Output
```

---

## Component-to-Code Mapping

| Component | Source File | Description |
|-----------|-------------|-------------|
| GUI | \`services/edge/local_sensor_gui.py\` | Tkinter GUI class with Matplotlib chart |
| CarbonAwareController | \`services/edge/local_sensor_gui.py\` | Mode switching and orchestration logic |
| SensorSimulator | \`services/edge/local_sensor_gui.py\` | Gaussian sensor with anomaly injection |
| PubSubClient | \`services/edge/local_sensor_gui.py\` | Pub/Sub publish/subscribe wrapper |
| LightModelAPI | \`services/light-model/api_service.py\` | Flask REST endpoints |
| HeavyModelAPI | \`services/heavy-model/api_service.py\` | Flask health + Pub/Sub subscriber |
| HybridAnomalyDetector | \`models/hybrid_detector.py\` | Ensemble detector orchestration |
| IsolationForestDetector | \`models/isolation_forest_detector.py\` | Online-trained Isolation Forest |
| StatisticalAnomalyDetector | \`models/statistical_model.py\` | Z-score threshold detector |

---

## Interface Definitions

### Light Model REST API (Port 5001)

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| \`/health\` | GET | - | \`{"status": "healthy", "detector": "hybrid", "ml_fitted": bool}\` |
| \`/analyze\` | POST | \`{"value": float}\` | \`{"is_anomaly": bool, "method": str, "anomaly_score": float, "z_score": float}\` |
| \`/analyze/batch\` | POST | \`{"readings": [{value, timestamp}]}\` | \`{"summary": {...}, "results": [...]}\` |
| \`/stats\` | GET | - | \`{"total_detections": int, "anomaly_rate": float, "ml_fitted": bool}\` |
| \`/reset\` | POST | - | \`{"status": "reset"}\` |

### Heavy Model Service (Port 8080)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| \`/health\` | GET | Health check with ML status for Cloud Run |
| \`/\` | GET | Service info with detector configuration |

### Pub/Sub Message Formats

| Topic | Direction | Message Format |
|-------|-----------|----------------|
| \`sensor-readings\` | Edge → Cloud | \`{"device_id": str, "readings": [{value, timestamp}], "batch_id": str}\` |
| \`anomaly-results\` | Cloud → Edge | \`{"batch_id": str, "predictions": [{is_anomaly, method, scores}], "ml_fitted": bool}\` |

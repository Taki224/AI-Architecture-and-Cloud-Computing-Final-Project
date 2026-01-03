# Sequence Diagrams

This document contains UML sequence diagrams showing the runtime interactions between system components. Each diagram focuses on a single scenario with clear actor roles and complete request-response cycles.

---

## System Initialization

```mermaid
sequenceDiagram
    actor Operator
    participant GUI as GUI
    participant Controller as CarbonAwareController
    participant Sensor as SensorSimulator
    participant PubSub as PubSubClient

    Operator->>GUI: Launch application
    activate GUI
    GUI->>Controller: create()
    activate Controller
    Controller->>Sensor: create(μ=0, σ=1, rate=0.003)
    Sensor-->>Controller: sensor ready
    Controller->>PubSub: create(project="local-project")
    PubSub->>PubSub: init publisher (sensor-readings)
    PubSub->>PubSub: init subscriber (anomaly-results-sub)
    PubSub-->>Controller: client ready
    Controller-->>GUI: controller ready
    deactivate Controller
    GUI-->>Operator: Display window (1200×700, PERFORMANCE mode)
    deactivate GUI
```

---

## ECO Mode: Local Inference

When the operator selects ECO mode, each sensor reading is processed locally via REST API.

```mermaid
sequenceDiagram
    actor Operator
    participant GUI as GUI
    participant Controller as CarbonAwareController
    participant Sensor as SensorSimulator
    participant LightAPI as LightModelAPI :5001
    participant Detector as HybridAnomalyDetector

    Operator->>GUI: Click "🌱 ECO" toggle
    GUI->>Controller: set_mode(ECO)
    Controller-->>GUI: mode = ECO
    GUI-->>Operator: Show ECO indicator

    Operator->>GUI: Click "▶ Start"
    activate GUI
    
    loop Every 100ms (10 Hz)
        GUI->>Controller: tick()
        Controller->>Sensor: generate_reading()
        Sensor-->>Controller: value (float)
        Controller->>GUI: update_chart(value)
        
        Controller->>LightAPI: POST /analyze {"value": v}
        activate LightAPI
        LightAPI->>Detector: detect(value)
        activate Detector
        
        alt Warmup (samples < 100)
            Detector->>Detector: z_score = |value - μ| / σ
            Detector-->>LightAPI: {method: "statistical_fallback", z_score}
        else ML Active
            Detector->>Detector: extract_features(window)
            Detector->>Detector: IsolationForest.predict()
            Detector->>Detector: Z-score check
            Detector->>Detector: ensemble = OR(ml, stat)
            Detector-->>LightAPI: {method: "ensemble", is_anomaly, scores}
        end
        deactivate Detector
        
        LightAPI-->>Controller: {is_anomaly, method, anomaly_score, z_score}
        deactivate LightAPI
        
        alt is_anomaly = true
            Controller->>GUI: show_alert(prediction)
            GUI-->>Operator: Display anomaly indicator
        end
    end
    deactivate GUI
```

---

## PERFORMANCE Mode: Cloud Batch Processing

When the operator selects PERFORMANCE mode, readings are batched and sent to the cloud via Pub/Sub.

```mermaid
sequenceDiagram
    actor Operator
    participant GUI as GUI
    participant Controller as CarbonAwareController
    participant Sensor as SensorSimulator
    participant PubSub as PubSubClient
    participant Topic as sensor-readings
    participant Heavy as HeavyModelService :8080
    participant Detector as HybridAnomalyDetector
    participant Results as anomaly-results

    Operator->>GUI: Click "⚡ PERFORMANCE" toggle
    GUI->>Controller: set_mode(PERFORMANCE)
    Controller-->>GUI: mode = PERFORMANCE
    GUI-->>Operator: Show PERFORMANCE indicator

    Operator->>GUI: Click "▶ Start"
    activate GUI
    
    loop Every 100ms (10 Hz)
        GUI->>Controller: tick()
        Controller->>Sensor: generate_reading()
        Sensor-->>Controller: value (float)
        Controller->>GUI: update_chart(value)
        Controller->>PubSub: add_to_batch(value, timestamp)
        
        alt Batch size = 10
            PubSub->>Topic: publish(batch)
            Topic-->>PubSub: ack
        end
    end

    Note over Topic,Heavy: Async cloud processing
    
    Topic->>Heavy: deliver(batch message)
    activate Heavy
    
    loop For each reading in batch
        Heavy->>Detector: detect(value)
        Detector-->>Heavy: {is_anomaly, method, scores}
    end
    
    Heavy->>Results: publish(predictions)
    Results-->>Heavy: ack
    deactivate Heavy

    Results->>PubSub: deliver(predictions)
    PubSub->>Controller: on_results(predictions)
    
    loop For each prediction
        alt is_anomaly = true
            Controller->>GUI: show_alert(prediction)
            GUI-->>Operator: Display anomaly indicator
        end
    end
    deactivate GUI
```

---

## Mode Toggle

```mermaid
sequenceDiagram
    actor Operator
    participant GUI as GUI
    participant Controller as CarbonAwareController
    participant PubSub as PubSubClient
    participant LightAPI as LightModelAPI :5001

    alt Toggle to PERFORMANCE
        Operator->>GUI: Click mode button
        GUI->>Controller: set_mode(PERFORMANCE)
        Controller->>PubSub: enable_batching()
        PubSub-->>Controller: batching enabled
        Controller-->>GUI: mode = PERFORMANCE
        GUI-->>Operator: Show "⚡ PERFORMANCE"
    else Toggle to ECO
        Operator->>GUI: Click mode button
        GUI->>Controller: set_mode(ECO)
        Controller->>PubSub: flush_remaining_batch()
        PubSub-->>Controller: batch flushed
        Controller->>LightAPI: GET /health
        LightAPI-->>Controller: {status: "healthy", ml_fitted: bool}
        Controller-->>GUI: mode = ECO
        GUI-->>Operator: Show "🌱 ECO"
    end
```

---

## ML Warmup Sequence

This diagram shows how the HybridAnomalyDetector transitions from statistical-only detection to full ensemble mode.

```mermaid
sequenceDiagram
    participant Caller as API/Service
    participant Hybrid as HybridAnomalyDetector
    participant IsoForest as IsolationForestDetector
    participant Stat as StatisticalAnomalyDetector

    Note over Hybrid: Warmup Phase (samples 1-99)
    
    loop Samples 1-99
        Caller->>Hybrid: detect(value)
        activate Hybrid
        Hybrid->>IsoForest: detect(value)
        IsoForest->>IsoForest: training_data.append(value)
        IsoForest-->>Hybrid: {status: "warming_up", samples: N/100}
        Hybrid->>Stat: predict(value)
        Stat->>Stat: z = |value - μ| / σ
        Stat-->>Hybrid: z_score, is_anomaly
        Hybrid-->>Caller: {method: "statistical_fallback", is_anomaly, z_score}
        deactivate Hybrid
    end

    Note over IsoForest: Sample 100: Training trigger
    
    Caller->>Hybrid: detect(value)
    activate Hybrid
    Hybrid->>IsoForest: detect(value)
    activate IsoForest
    IsoForest->>IsoForest: _fit_model() on 100 samples
    IsoForest-->>Hybrid: {status: "just_fitted"}
    deactivate IsoForest
    Hybrid->>Stat: predict(value)
    Stat-->>Hybrid: z_score, stat_anomaly
    Hybrid->>Hybrid: ensemble = OR(ml_anomaly, stat_anomaly)
    Hybrid-->>Caller: {method: "ensemble", is_anomaly, scores}
    deactivate Hybrid

    Note over Hybrid: Active Phase (samples 101+)
    
    loop Subsequent samples
        Caller->>Hybrid: detect(value)
        activate Hybrid
        Hybrid->>IsoForest: detect(value)
        IsoForest->>IsoForest: update sliding window
        IsoForest->>IsoForest: extract features
        IsoForest->>IsoForest: model.predict()
        IsoForest-->>Hybrid: {is_anomaly, anomaly_score, status: "active"}
        Hybrid->>Stat: predict(value)
        Stat-->>Hybrid: z_score, stat_anomaly
        Hybrid->>Hybrid: ensemble = OR(ml_anomaly, stat_anomaly)
        Hybrid-->>Caller: {method: "ensemble", is_anomaly, scores}
        deactivate Hybrid
    end
```

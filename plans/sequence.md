# Sequence Diagrams

This document contains UML sequence diagrams showing the runtime interactions between system components.

---

## System Initialization

```mermaid
sequenceDiagram
    actor Operator
    participant GUI
    participant Controller
    participant Sensor
    participant PubSub

    Operator->>GUI: Launch application
    GUI->>Controller: create
    Controller->>Sensor: create sensor
    Sensor-->>Controller: ready
    Controller->>PubSub: create client
    PubSub-->>Controller: ready
    Controller-->>GUI: ready
    GUI-->>Operator: Display window
```

---

## ECO Mode - Local Inference

When the operator selects ECO mode, each sensor reading is processed locally via REST API.

```mermaid
sequenceDiagram
    actor Operator
    participant GUI
    participant Controller
    participant Sensor
    participant LightAPI
    participant Detector

    Operator->>GUI: Select ECO mode
    GUI->>Controller: set_mode ECO
    Controller-->>GUI: mode changed
    GUI-->>Operator: Show ECO indicator

    Operator->>GUI: Start sensor
    
    loop Every 100ms
        GUI->>Controller: tick
        Controller->>Sensor: generate_reading
        Sensor-->>Controller: value
        Controller->>GUI: update_chart
        
        Controller->>LightAPI: POST /analyze
        LightAPI->>Detector: detect
        Detector-->>LightAPI: result
        LightAPI-->>Controller: response
        
        alt is_anomaly true
            Controller->>GUI: show_alert
            GUI-->>Operator: Display anomaly
        end
    end
```

---

## PERFORMANCE Mode - Cloud Batch Processing

When the operator selects PERFORMANCE mode, readings are batched and sent to the cloud via Pub/Sub.

```mermaid
sequenceDiagram
    actor Operator
    participant GUI
    participant Controller
    participant Sensor
    participant PubSub
    participant Topic
    participant Heavy
    participant Results

    Operator->>GUI: Select PERFORMANCE mode
    GUI->>Controller: set_mode PERFORMANCE
    Controller-->>GUI: mode changed
    GUI-->>Operator: Show PERFORMANCE indicator

    Operator->>GUI: Start sensor
    
    loop Every 100ms
        GUI->>Controller: tick
        Controller->>Sensor: generate_reading
        Sensor-->>Controller: value
        Controller->>GUI: update_chart
        Controller->>PubSub: add_to_batch
        
        alt Batch size equals 10
            PubSub->>Topic: publish batch
            Topic-->>PubSub: ack
        end
    end

    Note over Topic,Heavy: Async cloud processing
    
    Topic->>Heavy: deliver batch
    Heavy->>Heavy: detect all values
    Heavy->>Results: publish predictions
    Results-->>Heavy: ack

    Results->>PubSub: deliver predictions
    PubSub->>Controller: on_results
    
    loop For each prediction
        alt is_anomaly true
            Controller->>GUI: show_alert
            GUI-->>Operator: Display anomaly
        end
    end
```

---

## Mode Toggle

```mermaid
sequenceDiagram
    actor Operator
    participant GUI
    participant Controller
    participant PubSub
    participant LightAPI

    alt Toggle to PERFORMANCE
        Operator->>GUI: Click mode button
        GUI->>Controller: set_mode PERFORMANCE
        Controller->>PubSub: enable_batching
        PubSub-->>Controller: enabled
        Controller-->>GUI: mode PERFORMANCE
        GUI-->>Operator: Show PERFORMANCE
    else Toggle to ECO
        Operator->>GUI: Click mode button
        GUI->>Controller: set_mode ECO
        Controller->>PubSub: flush_batch
        PubSub-->>Controller: flushed
        Controller->>LightAPI: GET /health
        LightAPI-->>Controller: healthy
        Controller-->>GUI: mode ECO
        GUI-->>Operator: Show ECO
    end
```

---

## ML Warmup Sequence

This diagram shows how the HybridAnomalyDetector transitions from statistical-only detection to full ensemble mode.

```mermaid
sequenceDiagram
    participant Caller
    participant Hybrid
    participant IsoForest
    participant Stat

    Note over Hybrid: Warmup Phase samples 1 to 99
    
    loop Samples 1 to 99
        Caller->>Hybrid: detect value
        Hybrid->>IsoForest: detect value
        IsoForest->>IsoForest: collect training data
        IsoForest-->>Hybrid: warming_up
        Hybrid->>Stat: predict value
        Stat-->>Hybrid: z_score
        Hybrid-->>Caller: statistical_fallback result
    end

    Note over IsoForest: Sample 100 Training trigger
    
    Caller->>Hybrid: detect value
    Hybrid->>IsoForest: detect value
    IsoForest->>IsoForest: fit_model
    IsoForest-->>Hybrid: just_fitted
    Hybrid->>Stat: predict value
    Stat-->>Hybrid: z_score
    Hybrid->>Hybrid: ensemble OR
    Hybrid-->>Caller: ensemble result

    Note over Hybrid: Active Phase samples 101 plus
    
    loop Subsequent samples
        Caller->>Hybrid: detect value
        Hybrid->>IsoForest: detect value
        IsoForest->>IsoForest: predict
        IsoForest-->>Hybrid: ml_anomaly score
        Hybrid->>Stat: predict value
        Stat-->>Hybrid: stat_anomaly
        Hybrid->>Hybrid: ensemble OR
        Hybrid-->>Caller: ensemble result
    end
```

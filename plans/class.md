# Class Diagram

## Core Classes

```mermaid
classDiagram
    class GUI {
        -window_size: 1200x700
        -chart_window: 30 seconds
        -max_data_points: 300
        -update_interval: 100ms
        -y_axis_range: -10 to 10
        +start_sensor()
        +stop_sensor()
        +toggle_mode()
        +update_chart(value, timestamp)
        +show_anomaly_alert(prediction)
    }

    class CarbonAwareController {
        -mode: PERFORMANCE | ECO
        -device_id: "edge-001"
        -ml_api_url: "http://localhost:5001"
        +set_mode(mode)
        +on_reading(value, timestamp)
        +on_anomaly_result(predictions)
        -process_performance_mode(reading)
        -process_eco_mode(reading)
    }

    class SensorSimulator {
        -normal_mean: 0.0
        -normal_std: 1.0
        -anomaly_rate: 0.003
        -small_anomaly_range: 3.0σ to 4.0σ
        -large_anomaly_range: 5.0σ to 8.0σ
        -large_anomaly_ratio: 0.4
        +generate_reading(): float
        +is_anomaly(value): bool
        -inject_anomaly(): float
    }

    class PubSubClient {
        -project_id: "local-project"
        -device_id: "edge-001"
        -batch_size: 10
        -publish_timeout: 5 seconds
        -sensor_topic: "sensor-readings"
        -anomaly_subscription: "anomaly-results-sub"
        +add_to_batch(reading)
        +flush_batch()
        +subscribe_to_results(callback)
        -on_message(message)
    }

    class HybridAnomalyDetector {
        -z_threshold: 3.0
        -contamination: 0.003
        -ml_detector: IsolationForestDetector
        -stat_detector: StatisticalAnomalyDetector
        -detection_count: int
        -anomaly_count: int
        +detect(value): dict
        +get_stats(): dict
        +reset()
    }

    class IsolationForestDetector {
        -contamination: 0.003
        -window_size: 50
        -n_estimators: 100 | 200
        -min_samples_for_fit: 100
        -is_fitted: bool
        -window: deque
        -training_data: list
        +detect(value): dict
        +reset()
        -_extract_features(values): ndarray
        -_fit_model()
    }

    class StatisticalAnomalyDetector {
        -threshold: 3.0
        -mean_: 0.0
        -std_: 1.0
        +fit(X, y)
        +predict(X): array
        +score_samples(X): array
    }

    class LightModelAPI {
        -port: 5000
        -detector: HybridAnomalyDetector
        -n_estimators: 100
        +health(): dict
        +predict(value): dict
        +analyze(value): dict
        +analyze_batch(readings): dict
        +get_stats(): dict
        +reset(): dict
    }

    class HeavyModelService {
        -port: 8080
        -detector: HybridAnomalyDetector
        -n_estimators: 200
        -sensor_subscription: "sensor-readings-sub"
        -anomaly_topic: "anomaly-results"
        -publish_timeout: 10 seconds
        +health(): dict
        +process_batch(readings): dict
        -on_sensor_message(message)
    }

    class AnomalyMonitor {
        -window_size: 60 seconds
        -metric_name: "anomaly_detection/rate"
        +record_prediction(is_anomaly)
        +get_anomaly_rate(): float
        +log_to_cloud(message)
        +push_metric(value)
    }

    %% Relationships
    GUI --> CarbonAwareController : controls
    CarbonAwareController --> SensorSimulator : reads from
    CarbonAwareController --> PubSubClient : uses for PERFORMANCE
    CarbonAwareController --> LightModelAPI : calls for ECO
    
    LightModelAPI --> HybridAnomalyDetector : uses
    HeavyModelService --> HybridAnomalyDetector : uses
    
    HybridAnomalyDetector --> IsolationForestDetector : ML detection
    HybridAnomalyDetector --> StatisticalAnomalyDetector : statistical fallback
    
    HeavyModelService --> AnomalyMonitor : reports to
    PubSubClient ..> HeavyModelService : via Pub/Sub
```

## ML Detection Flow

```mermaid
flowchart TD
    A[Sensor Reading] --> B{Warmup Complete?}
    B -->|No, samples < 100| C[Statistical Only]
    B -->|Yes| D[Hybrid Ensemble]
    
    C --> E[Z-score Detection]
    E -->|z > 3.0σ| F[Anomaly]
    E -->|z ≤ 3.0σ| G[Normal]
    
    D --> H[Isolation Forest]
    D --> I[Z-score]
    H --> J{ML Anomaly?}
    I --> K{Stat Anomaly?}
    
    J -->|Yes| L[Flag Anomaly]
    K -->|Yes| L
    J -->|No| M{Other Triggered?}
    K -->|No| M
    M -->|Yes| L
    M -->|No| N[Normal]
    
    subgraph Features["Sliding Window Features (50 samples)"]
        F1[Current Value]
        F2[Window Mean]
        F3[Window Std]
        F4[Window Max/Min]
        F5[Rate of Change]
    end
    
    H -.-> Features
```

## Data Classes

```mermaid
classDiagram
    class SensorReading {
        +value: float
        +timestamp: datetime
        +device_id: str
    }

    class BatchMessage {
        +device_id: str
        +batch_id: str
        +readings: List~SensorReading~
        +mode: str
    }

    class DetectionResult {
        +is_anomaly: bool
        +method: "ensemble" | "statistical_fallback"
        +ml_anomaly: bool
        +stat_anomaly: bool
        +anomaly_score: float
        +z_score: float
        +confidence: float
        +ml_status: str
    }

    class AnomalyResultMessage {
        +batch_id: str
        +device_id: str
        +predictions: List~DetectionResult~
        +ml_fitted: bool
        +processing_time_ms: float
    }

    BatchMessage o-- SensorReading : contains
    AnomalyResultMessage o-- DetectionResult : contains
```

## Model Configuration

| Model | Location | Estimators | Window | Contamination | Warmup |
|-------|----------|------------|--------|---------------|--------|
| Light (ECO) | Docker :5001 | 100 | 50 | 0.003 | 100 samples |
| Heavy (Cloud) | Cloud Run :8080 | 200 | 50 | 0.003 | 100 samples |

| Detection Method | Threshold | Fallback | Ensemble Logic |
|------------------|-----------|----------|----------------|
| Isolation Forest | contamination=0.003 | N/A | Flag if score indicates anomaly |
| Z-score | 3.0σ | Primary during warmup | Flag if z > threshold |
| Hybrid | N/A | Stat-only first 100 samples | Flag if EITHER triggers |

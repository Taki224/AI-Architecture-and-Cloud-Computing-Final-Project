# Architecture Decision Records

## ADR 001: Hybrid Edge-Cloud Architecture for Carbon Optimization

### Context
The project requires an IoT anomaly detection system capable of processing real-time vibration sensor data (Gaussian distribution μ=0, σ=1). A critical non-functional requirement is **sustainability**—the ability to adapt system behavior to minimize carbon emissions based on electricity grid carbon intensity.

We evaluated three deployment strategies:
1. **Cloud-Only:** Stream all raw data to the cloud for processing
2. **Edge-Only:** Process all data locally on the device
3. **Hybrid (Dynamic):** Switch between edge and cloud based on operator choice (manual toggle)

The system must maintain operational reliability with sub-second latency for anomaly detection.

### Decision
We adopt a **Dynamic Hybrid Edge-Cloud Architecture** with manual mode switching via GUI.

* **PERFORMANCE Mode:** Edge device batches 10 readings and publishes to `sensor-data` Pub/Sub topic. Cloud Run service processes with **HybridAnomalyDetector** (200-estimator Isolation Forest + Z-score ensemble).
* **ECO Mode:** Edge device sends individual readings to local REST API (port 5001) for inference with **IsolationForestDetector** (50-estimator Isolation Forest with Z-score statistical detector).

PERFORMANCE mode uses a full hybrid ensemble (Isolation Forest + Z-score) while ECO mode uses a lighter Isolation Forest-only detector. Mode switching is controlled manually by the operator through the GUI. Carbon consumption measurement is integrated via CarbonMonitor class for observability (logging gCO₂e per mode), with manual mode switching only.

### Consequences

**Positive:**
- Carbon reduction during high-intensity periods by avoiding cloud compute and network transmission
- Cost efficiency through reduced Pub/Sub message volume and Cloud Run invocations in ECO mode
- Resilience: edge retains full ML detection capability during cloud outages
- Consistent detection quality across modes (both use hybrid ML)

**Negative:**
- Higher complexity: both services require same ML dependencies
- Warmup period: first 100 samples use statistical fallback during ML training
- Split observability: metrics from edge and cloud must be correlated

---

## ADR 002: Hybrid ML Detection with Isolation Forest + Z-Score Ensemble

### Context
The system requires ML-based anomaly detection to meet accuracy requirements. We evaluated:

1. **Isolation Forest only:** ML-based, requires training data, cold start problem
2. **Z-Score only:** Statistical, instant startup, limited pattern detection
3. **Hybrid ensemble:** Combine both for maximum accuracy and reliability

Requirements:
- Use machine learning for detection (project requirement)
- Minimize false negatives (maximize recall)
- Handle cold start gracefully
- Work without pre-trained model files

### Decision
Implement **HybridAnomalyDetector** combining:

1. **IsolationForestDetector:** Online-trained Isolation Forest with sliding window features
   - Light model (ECO): 100 estimators
   - Heavy model (PERFORMANCE): 200 estimators
   - Window size: 50 samples for feature extraction
   - Contamination: 0.003 (matches 0.3% anomaly rate)
   - Features: value, window mean, std, max, min, rate of change

2. **StatisticalAnomalyDetector:** Z-score with 3.0σ threshold
   - Provides detection during ML warmup
   - Catches simple outliers ML might miss

3. **Ensemble logic:** Flag anomaly if EITHER detector triggers (OR gate)
   - Maximizes recall (catches more anomalies)
   - Accepts slightly higher false positive rate for safety

**Warmup behavior:**
- First 100 samples: Statistical-only detection
- Sample 100: IsolationForest trains on collected data
- Sample 101+: Full ensemble active

### Consequences

**Positive:**
- ML-based detection meets project requirements
- No pre-trained model files needed (online training)
- Graceful cold start via statistical fallback
- Higher recall through ensemble voting
- Interpretable: both z-score and ML score available

**Negative:**
- 100-sample warmup with reduced accuracy
- Slightly higher false positive rate due to OR ensemble
- More complex than single-model approach
- Memory overhead for sliding window

---

## ADR 003: Google Cloud Pub/Sub for Edge-Cloud Communication

### Context
PERFORMANCE mode requires reliable message delivery from edge to cloud. We evaluated:
1. **Direct REST API calls:** Simple but synchronous, blocks sensor loop
2. **WebSocket:** Persistent connection, complex state management
3. **Cloud Pub/Sub:** Managed async messaging with guaranteed delivery

Requirements:
- Batch processing (10 readings per message)
- Async operation (don't block 100ms sensor loop)
- Result delivery back to edge
- Local development without GCP costs

### Decision
Use **Google Cloud Pub/Sub** with:
- `sensor-data` topic: Edge → Cloud (batched readings)
- `anomaly-results` topic: Cloud → Edge (predictions)
- `sensor-data-sub` / `anomaly-results-sub` subscriptions
- Local Pub/Sub emulator for development (port 8085)

Configuration:
- Batch size: 10 readings
- Publish timeout: 5 seconds
- Project ID: Configured via `GOOGLE_CLOUD_PROJECT` environment variable

### Consequences

**Positive:**
- Decoupled architecture: edge and cloud operate independently
- Guaranteed delivery with acknowledgment
- Native GCP integration with Cloud Run
- Local emulator eliminates development costs

**Negative:**
- Added latency (~1-2s round trip) compared to direct REST
- Requires Pub/Sub emulator for local development
- Message ordering not guaranteed (acceptable for this use case)

---

## ADR 004: Flask for REST API Services

### Context
Both light-model and heavy-model services expose HTTP endpoints. We evaluated:
1. **Flask:** Simple, synchronous, mature ecosystem
2. **FastAPI:** Async, automatic OpenAPI, Pydantic validation
3. **Starlette:** Lightweight async, less batteries-included

Requirements:
- Health check endpoint for Docker/Cloud Run
- JSON request/response for predictions
- Simple deployment in containers

### Decision
Use **Flask** for both services:

Light Model Service (port 5001):
- `GET /health` → Health check
- `POST /analyze` → Single prediction
- `POST /analyze/batch` → Batch prediction
- `GET /stats` → Detection statistics
- `POST /reset` → Reset detector state

Heavy Model Service (port 8080):
- `GET /health` → Health check for Cloud Run
- `GET /` → Service info

### Consequences

**Positive:**
- Simple, well-documented, team familiarity
- Synchronous model fits inference workload (CPU-bound)
- Easy Docker health check integration

**Negative:**
- No async support (acceptable for inference workload)
- Manual OpenAPI documentation required
- No built-in request validation (handled manually)

---

## ADR 005: Manual Mode Toggle with Carbon Observability

### Context
The system must switch between PERFORMANCE and ECO modes. Two approaches were considered:
1. **Manual Toggle:** Operator controls mode via GUI button
2. **Automated:** System queries carbon intensity API, switches automatically

### Decision
Manual toggle via GUI:
- Operator clicks mode button to switch between "⚡ PERFORMANCE" and "🌱 ECO"
- No external carbon intensity data source required
- Full operator control for testing and demonstration

Carbon consumption measurement:
- CarbonMonitor class measures gCO₂e per inference
- Custom metrics exported to GCP Cloud Monitoring with service/mode labels
- Provides visibility into environmental impact of each mode
- CodeCarbon enabled for local development, estimation-based for cloud
- **No automated switching**—measurement is for observability only

### Consequences

**Positive:**
- Simple implementation, no external dependencies for switching
- Operator can test both modes at will
- Clear demonstration of mode differences
- Carbon metrics enable informed manual decisions

**Negative:**
- No automatic carbon optimization
- Requires operator attention to switch modes
- Carbon savings depend on operator awareness and action

---

## ADR 006: Online Training vs Pre-trained Model Files

### Context
ML models require training before inference. Two approaches were considered:

1. **Pre-trained models:** Train offline, save to `.pkl` files, load on startup
2. **Online training:** Collect data at runtime, train when sufficient samples arrive

Requirements:
- No dependency on external training pipeline for deployment
- Handle fresh deployments without historical data
- Adapt to potentially different data distributions per deployment

### Decision
Use **pre-trained models with online training fallback**:

**Primary approach (production):**
- Pre-train models offline using `train_models.py` with synthetic data
- Save models as `model_heavy.pkl` and `model_light.pkl`
- Load pre-trained models at service startup (via joblib)
- Immediate inference capability with no warmup period

**Fallback approach (if .pkl files unavailable):**
- Initialize detector at runtime with online training
- Collect first 100 samples in `training_data` list
- At sample 100, call `_fit_model()` to train IsolationForest
- Use sliding window of 50 samples for ongoing feature extraction

```python
min_samples_for_fit = 100  # Samples before ML activates (fallback only)
window_size = 50           # Sliding window for feature extraction
```

### Consequences

**Positive (Pre-trained models):**
- Immediate inference capability with no warmup period
- Consistent performance across all deployments
- Models trained on comprehensive synthetic dataset (10,000 samples)
- Reproducible behavior—same model weights every time

**Positive (Online training fallback):**
- Graceful degradation if model files unavailable
- Adapts to local data distribution automatically
- No hard dependency on model artifact files

**Negative:**
- Requires pre-training step before deployment (automated in CI/CD)
- Model files (~1-2 MB) must be included in Docker images
- Fallback mode has 100-sample warmup period (~10 seconds at 10 Hz)
- Cannot adapt to distribution drift without retraining
# Architecture Decision Records

## ADR 001: Hybrid Edge-Cloud Architecture for Carbon Optimization

### Status
Accepted

### Context
The project requires an IoT anomaly detection system capable of processing real-time vibration sensor data (Gaussian distribution μ=0, σ=1). A critical non-functional requirement is **sustainability**—the ability to adapt system behavior to minimize carbon emissions based on electricity grid carbon intensity.

We evaluated three deployment strategies:
1. **Cloud-Only:** Stream all raw data to the cloud for processing
2. **Edge-Only:** Process all data locally on the device
3. **Hybrid (Dynamic):** Switch between edge and cloud based on operator choice (manual toggle)

The system must maintain operational reliability with sub-second latency for anomaly detection.

### Decision
We adopt a **Dynamic Hybrid Edge-Cloud Architecture** with manual mode switching via GUI.

* **PERFORMANCE Mode:** Edge device batches 10 readings and publishes to `sensor-readings` Pub/Sub topic. Cloud Run service processes with **HybridAnomalyDetector** (200-estimator Isolation Forest + Z-score ensemble).
* **ECO Mode:** Edge device sends individual readings to local REST API (port 5001) for inference with **HybridAnomalyDetector** (100-estimator Isolation Forest + Z-score ensemble).

Both modes use the same hybrid ML approach with different model sizes. Mode switching is controlled manually by the operator through the GUI. Future phases will integrate carbon consumption measurement via Python library for observability (logging gCO₂e per mode), but switching will remain manual.

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

### Status
Accepted (supersedes original statistical-only approach)

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

### Status
Accepted

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
- `sensor-readings` topic: Edge → Cloud (batched readings)
- `anomaly-results` topic: Cloud → Edge (predictions)
- `sensor-readings-sub` / `anomaly-results-sub` subscriptions
- Local Pub/Sub emulator for development (port 8085)

Configuration:
- Batch size: 10 readings
- Publish timeout: 5 seconds
- Project ID: `local-project` (emulator)

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

### Status
Accepted

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

Light Model Service (port 5000, exposed as 5001):
- `GET /health` → Health check
- `POST /analyze` → Single prediction
- `POST /analyze/batch` → Batch prediction

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

### Status
Accepted

### Context
The system must switch between PERFORMANCE and ECO modes. Two approaches were considered:
1. **Manual Toggle:** Operator controls mode via GUI button
2. **Automated:** System queries carbon intensity API, switches automatically

### Decision
**All Phases:** Manual toggle via GUI.
- Operator clicks mode button to switch between "⚡ PERFORMANCE" and "🌱 ECO"
- No external carbon intensity data source required
- Full operator control for testing and demonstration

**Phase 4 Addition:** Carbon consumption measurement for observability.
- Integrate Python library to measure gCO₂e per inference
- Log carbon metrics per mode for comparison
- Provide visibility into environmental impact
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

### Status
Accepted

### Context
ML models require training before inference. Two approaches were considered:

1. **Pre-trained models:** Train offline, save to `.pkl` files, load on startup
2. **Online training:** Collect data at runtime, train when sufficient samples arrive

Requirements:
- No dependency on external training pipeline for deployment
- Handle fresh deployments without historical data
- Adapt to potentially different data distributions per deployment

### Decision
Use **online training** for the IsolationForestDetector:

```python
min_samples_for_fit = 100  # Samples before ML activates
window_size = 50           # Sliding window for feature extraction
```

Training flow:
1. Collect first 100 samples in `training_data` list
2. At sample 100, call `_fit_model()` to train IsolationForest
3. Use sliding window of 50 samples for ongoing feature extraction
4. No `.pkl` files required—model trains fresh each run

### Consequences

**Positive:**
- Zero deployment dependencies—no model files to manage
- Adapts to local data distribution automatically
- Fresh start on each deployment/restart
- Simpler CI/CD pipeline (no model artifact management)

**Negative:**
- 100-sample warmup period on each restart (~10 seconds at 10 Hz)
- Statistical-only detection during warmup
- Cannot leverage historical training data
- Slightly inconsistent behavior across restarts (different training samples)
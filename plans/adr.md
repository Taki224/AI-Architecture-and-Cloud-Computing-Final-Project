# ADR 001: Adoption of Hybrid Edge-Cloud Architecture for Carbon Optimization


## Context
The project requires an IoT anomaly detection system capable of processing real-time sensor data. A critical non-functional requirement is **sustainability**, specifically the ability to adapt system behavior to minimize carbon emissions ($gCO_{2}e$) based on the current carbon intensity of the electricity grid.

We evaluated three potential deployment strategies:
1.  **Cloud-Only:** Stream all raw data to the cloud for processing.
2.  **Edge-Only:** Process all data locally on the device.
3.  **Hybrid (Dynamic):** Switch between edge and cloud based on external signals.

The system must also maintain a specific Service Level Objective (SLO) regarding processing latency ($p95 \le 300ms$) and ensure high detection accuracy when conditions permit.

## Decision
We have decided to adopt a **Dynamic Hybrid Edge-Cloud Architecture**.

In this model, the "Edge" device (simulated via local Docker container) will operate in two distinct modes, controlled by a central cloud-based service monitoring grid carbon intensity:

* **Performance Mode (Low Carbon):** The edge device acts as a pass-through gateway, streaming raw telemetry to Google Cloud Pub/Sub. The heavy-lifting inference is performed by a serverless worker (Cloud Run) using a high-precision model (`model_heavy.pkl`).
* **Eco Mode (High Carbon):** The edge device activates a lightweight, quantized local model (`model_light.pkl`). It performs inference locally and only transmits confirmed anomalies to the cloud, significantly reducing network egress and cloud compute usage.

## Consequences

### Positive Consequences
* **Carbon Reduction:** We can actively reduce the system's carbon footprint during "dirty" energy periods by shifting compute to the low-power edge device and reducing data transmission[cite: 3219, 3220].
* **Cost Efficiency:** Reducing data ingress/egress and cloud compute execution time during high-carbon windows (which often correlate with peak pricing) optimizes operational costs[cite: 2931].
* **Resilience:** The system retains basic anomaly detection capabilities at the edge even if the connection to the cloud is severed (offline operation)[cite: 3204].

### Negative Consequences
* **Operational Complexity:** We must manage and version two distinct models (Light vs. Heavy) and ensure the switching logic in the `EdgeSimulator` is robust.
* **Accuracy Trade-off:** The local `model_light.pkl` (Isolation Forest with fewer estimators) is inherently less accurate than the cloud version. [cite_start]We accept a temporarily higher false-negative rate in exchange for carbon savings during high-intensity windows[cite: 606].
* **Observability Challenges:** Metrics are now split between the local device and the cloud. [cite_start]We must implement a mechanism for the edge device to batch-send its "local inference counts" to the cloud dashboard to maintain a complete view of the system[cite: 3210].

## Compliance
[cite_start]This decision aligns with the requirement to demonstrate "Sustainability considerations in deployment, including trade-offs between scalability, energy efficiency, and cost" [cite: 585] [cite_start]and explicitly addresses the "Carbon-aware behavior driven by a simulated carbon-intensity signal" requirement[cite: 587].

***

**Would you like me to generate the Python code for the `EdgeSimulator` next, specifically the part that switches models based on the Pub/Sub command?**
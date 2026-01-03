# Deployment Diagram

This document describes the runtime deployment architecture of the Carbon-Aware IoT Anomaly Detection System using UML deployment notation, focusing on execution environments, nodes, and deployed artifacts.

---

## Production Deployment (GCP)

```mermaid
graph TB
    subgraph EdgeNode["&lt;&lt;device&gt;&gt;<br/>Edge Device<br/>(Raspberry Pi / Industrial PC)"]
        EdgeRuntime["&lt;&lt;execution environment&gt;&gt;<br/>Python 3.11"]
        EdgeArtifact["&lt;&lt;artifact&gt;&gt;<br/>local_sensor_gui.py"]
    end

    subgraph GCPNode["&lt;&lt;execution environment&gt;&gt;<br/>Google Cloud Platform (us-central1)"]
        subgraph PubSubNode["&lt;&lt;infrastructure&gt;&gt;<br/>Cloud Pub/Sub<br/>99.9% SLA"]
            SensorTopic["&lt;&lt;topic&gt;&gt;<br/>sensor-readings"]
            AnomalyTopic["&lt;&lt;topic&gt;&gt;<br/>anomaly-results"]
            SensorSub["&lt;&lt;subscription&gt;&gt;<br/>sensor-readings-sub"]
            AnomalySub["&lt;&lt;subscription&gt;&gt;<br/>anomaly-results-sub"]
        end

        subgraph CloudRunNode["&lt;&lt;execution environment&gt;&gt;<br/>Cloud Run<br/>Auto-scale: 0-10 instances"]
            HeavyContainer["&lt;&lt;container&gt;&gt;<br/>heavy-model-service"]
            HeavyDetector["&lt;&lt;component&gt;&gt;<br/>HybridAnomalyDetector<br/>(200 estimators, z=3.0σ)"]
        end

        subgraph OpsNode["&lt;&lt;infrastructure&gt;&gt;<br/>Operations Suite"]
            Logging["&lt;&lt;service&gt;&gt;<br/>Cloud Logging"]
            Monitoring["&lt;&lt;service&gt;&gt;<br/>Cloud Monitoring"]
        end

        subgraph RegistryNode["&lt;&lt;repository&gt;&gt;<br/>Artifact Registry"]
            HeavyImage["&lt;&lt;artifact&gt;&gt;<br/>heavy-model:latest"]
        end
    end

    EdgeArtifact -->|"gRPC/TLS :443<br/>publish batch"| SensorTopic
    SensorTopic --> SensorSub
    SensorSub -->|"push"| HeavyContainer
    HeavyContainer --> HeavyDetector
    HeavyContainer -->|"publish"| AnomalyTopic
    AnomalyTopic --> AnomalySub
    AnomalySub -->|"gRPC/TLS :443<br/>pull results"| EdgeArtifact
    HeavyContainer -->|"logs"| Logging
    HeavyContainer -->|"metrics"| Monitoring
    HeavyImage -.->|"deploy"| HeavyContainer
```

---

## Local Development Deployment

```mermaid
graph TB
    subgraph HostNode["&lt;&lt;device&gt;&gt;<br/>Developer Workstation (macOS/Linux)"]
        subgraph PythonEnv["&lt;&lt;execution environment&gt;&gt;<br/>Python 3.11 Virtual Environment"]
            EdgeApp["&lt;&lt;artifact&gt;&gt;<br/>local_sensor_gui.py"]
        end

        subgraph DockerNode["&lt;&lt;execution environment&gt;&gt;<br/>Docker Engine"]
            subgraph Network["&lt;&lt;network&gt;&gt;<br/>carbon-aware-network"]
                subgraph EmulatorContainer["&lt;&lt;container&gt;&gt;<br/>pubsub-emulator :8085"]
                    Emulator["&lt;&lt;service&gt;&gt;<br/>gcloud-pubsub-emulator"]
                    Topics["&lt;&lt;topic&gt;&gt;<br/>sensor-readings<br/>anomaly-results"]
                end

                subgraph LightContainer["&lt;&lt;container&gt;&gt;<br/>light-model-service :5001"]
                    LightAPI["&lt;&lt;service&gt;&gt;<br/>Flask REST API"]
                    LightDetector["&lt;&lt;component&gt;&gt;<br/>HybridAnomalyDetector<br/>(100 estimators, z=3.0σ)"]
                end

                subgraph HeavyContainer["&lt;&lt;container&gt;&gt;<br/>heavy-model-service :8080"]
                    HeavyAPI["&lt;&lt;service&gt;&gt;<br/>Flask Health API"]
                    Subscriber["&lt;&lt;component&gt;&gt;<br/>Pub/Sub Subscriber"]
                    HeavyDetectorLocal["&lt;&lt;component&gt;&gt;<br/>HybridAnomalyDetector<br/>(200 estimators, z=3.0σ)"]
                end
            end
        end
    end

    EdgeApp -->|"HTTP POST /analyze<br/>(ECO mode)"| LightAPI
    LightAPI --> LightDetector
    LightDetector -->|"response"| EdgeApp
    
    EdgeApp -->|"gRPC :8085<br/>(PERFORMANCE mode)"| Emulator
    Emulator --> Subscriber
    Subscriber --> HeavyDetectorLocal
    HeavyDetectorLocal -->|"publish results"| Emulator
    Emulator -->|"pull subscription"| EdgeApp

    LightContainer -.->|"depends_on"| EmulatorContainer
    HeavyContainer -.->|"depends_on"| EmulatorContainer
```

---

## Node Specifications

### Edge Device Node

| Property | Value |
|----------|-------|
| **Node Type** | Physical device (Raspberry Pi) or developer workstation |
| **OS** | Linux (Raspberry Pi OS) / macOS / Windows |
| **Runtime** | Python 3.11 |
| **Deployed Artifact** | \`local_sensor_gui.py\` |
| **Network Requirements** | Internet connectivity to GCP (production) or localhost (development) |

### Cloud Run Execution Environment

| Property | Value |
|----------|-------|
| **Node Type** | Serverless container instance |
| **Scaling** | 0–10 instances (auto-scale based on Pub/Sub backlog) |
| **Memory** | 512 MB per instance |
| **CPU** | 1 vCPU per instance |
| **Deployed Artifact** | \`heavy-model-service:latest\` container image |
| **Image Registry** | Artifact Registry (\`gcr.io/PROJECT/heavy-model\`) |

### Docker Containers (Local Development)

| Container | Base Image | Port Mapping | Health Check |
|-----------|------------|--------------|--------------|
| \`pubsub-emulator\` | \`google/cloud-sdk:slim\` | 8085:8085 (gRPC) | 5s interval, 10 retries |
| \`light-model-service\` | \`python:3.11-slim\` | 5001:5000 (HTTP) | 10s interval, 5 retries |
| \`heavy-model-service\` | \`python:3.11-slim\` | 8080:8080 (HTTP) | 10s interval, 5 retries |

---

## Communication Channels

| Source | Target | Protocol | Port | Purpose |
|--------|--------|----------|------|---------|
| Edge Device | Cloud Pub/Sub | gRPC/TLS | 443 | Publish sensor batches |
| Cloud Pub/Sub | Cloud Run | gRPC (push) | Internal | Subscription delivery |
| Cloud Run | Cloud Pub/Sub | gRPC | Internal | Publish detection results |
| Cloud Pub/Sub | Edge Device | gRPC/TLS | 443 | Pull anomaly results |
| Edge Device | Light Model API | HTTP/REST | 5001 | ECO mode local inference |

---

## Deployment Artifacts

| Artifact | Type | Location | Description |
|----------|------|----------|-------------|
| \`local_sensor_gui.py\` | Python script | \`services/edge/\` | Unified edge application (GUI + sensor + controller) |
| \`light-model-service\` | Docker image | \`deployment/docker/Dockerfile.light\` | Local ML inference service |
| \`heavy-model-service\` | Docker image | \`deployment/docker/Dockerfile.heavy\` | Cloud ML inference service |
| \`hybrid_detector.py\` | Python module | \`models/\` | Shared ML detection logic |

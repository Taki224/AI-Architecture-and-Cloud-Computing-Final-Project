# GCP Setup Guide - Heavy Model Deployment

This guide walks you through deploying the heavy model service to Google Cloud Platform (GCP) using Cloud Run and Pub/Sub for messaging.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Setup](#project-setup)
3. [Create Pub/Sub Resources](#create-pubsub-resources)
4. [Service Account Setup](#service-account-setup)
5. [Local Development](#local-development)
6. [Build & Deploy to Cloud Run](#build--deploy-to-cloud-run)
7. [Testing](#testing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Tools

1. **Google Cloud CLI (gcloud)**
   ```bash
   # macOS
   brew install --cask google-cloud-sdk
   
   # Or download from https://cloud.google.com/sdk/docs/install
   ```

2. **Docker** (for local testing)
   ```bash
   # Verify installation
   docker --version
   ```

3. **Python 3.11+** (for local development)
   ```bash
   python3 --version
   ```

### Verify gcloud Installation

```bash
gcloud version
gcloud auth login
```

---

## Project Setup

### 1. Create or Select a GCP Project

```bash
# Create a new project (optional)
gcloud projects create YOUR_PROJECT_ID --name="Carbon-Aware Anomaly Detection"

# Set the active project
gcloud config set project YOUR_PROJECT_ID

# Verify
gcloud config get-value project
```

### 2. Enable Required APIs

```bash
# Enable Pub/Sub, Cloud Run, Artifact Registry, Cloud Build, Logging, Monitoring
gcloud services enable \
    pubsub.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com
```

### 3. Set Default Region

```bash
gcloud config set run/region europe-north1
gcloud config set artifacts/location europe-north1
```

---

## Create Pub/Sub Resources

### Create Topics

```bash
# Sensor data topic (edge device -> cloud)
gcloud pubsub topics create sensor-data

# Anomaly results topic (cloud -> edge device)
gcloud pubsub topics create anomaly-results
```

### Create Subscriptions

```bash
# Subscription for heavy model service
gcloud pubsub subscriptions create sensor-data-sub \
    --topic=sensor-data \
    --ack-deadline=60 \
    --message-retention-duration=1h

# Subscription for edge device to receive results
gcloud pubsub subscriptions create anomaly-results-sub \
    --topic=anomaly-results \
    --ack-deadline=60 \
    --message-retention-duration=1h
```

### Verify Resources

```bash
# List topics
gcloud pubsub topics list

# List subscriptions
gcloud pubsub subscriptions list
```

---

## Service Account Setup

### 1. Create Service Account

```bash
# Create service account for heavy model service
gcloud iam service-accounts create heavy-model-sa \
    --display-name="Heavy Model Service Account"
```

### 2. Grant Required Roles

```bash
PROJECT_ID=$(gcloud config get-value project)

# Pub/Sub Subscriber (for sensor-data topic)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:heavy-model-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/pubsub.subscriber"

# Pub/Sub Publisher (for anomaly-results topic)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:heavy-model-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/pubsub.publisher"

# Cloud Logging Writer (for anomaly logs)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:heavy-model-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/logging.logWriter"

# Cloud Monitoring Metric Writer (for anomaly rate metrics)
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:heavy-model-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/monitoring.metricWriter"
```

### 3. (Optional) Download Key for Local Testing

```bash
# Only needed for local testing outside Docker
gcloud iam service-accounts keys create key.json \
    --iam-account=heavy-model-sa@${PROJECT_ID}.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/key.json"
```

> ⚠️ **Security Note**: Never commit `key.json` to version control. Add it to `.gitignore`.

---

## Local Development

### Option 1: Use Pub/Sub Emulator (Recommended)

The docker-compose setup includes a Pub/Sub emulator for local testing:

```bash
# Start all services with emulator
docker-compose up -d

# View logs
docker-compose logs -f heavy-model

# Stop services
docker-compose down
```

### Option 2: Use Application Default Credentials

```bash
# Authenticate with your Google account
gcloud auth application-default login

# Set project
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

### Environment Variables for Local Development

Create a `.env` file (copy from `.env.example`):

```bash
# GCP Configuration
GOOGLE_CLOUD_PROJECT=your-project-id

# Pub/Sub Topics
PUBSUB_SENSOR_TOPIC=sensor-data
PUBSUB_ANOMALY_TOPIC=anomaly-results
PUBSUB_SENSOR_SUBSCRIPTION=sensor-data-sub
PUBSUB_ANOMALY_SUBSCRIPTION=anomaly-results-sub

# Device Configuration
DEVICE_ID=edge-001

# For local emulator (set by docker-compose)
# PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
```

---

## Build & Deploy to Cloud Run

### 1. Train the Models

Before building the Docker image, train the HybridAnomalyDetector models:

```bash
cd models
python train_models.py
cd ..
```

This will create:
- `models/model_heavy.pkl` - HybridAnomalyDetector (Isolation Forest + Z-score, 200 estimators)
- `models/model_light.pkl` - IsolationForestDetector (50 estimators for edge)

### 2. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create carbon-aware \
    --repository-format=docker \
    --location=europe-north1 \
    --description="Carbon-Aware Anomaly Detection Images"
```

### 3. Configure Docker Authentication

```bash
gcloud auth configure-docker europe-north1-docker.pkg.dev
```

### 4. Build and Push Image

```bash
PROJECT_ID=$(gcloud config get-value project)
IMAGE_URL="europe-north1-docker.pkg.dev/${PROJECT_ID}/carbon-aware/heavy-model:latest"

# Build image (includes pre-trained model)
# Note: Use --platform linux/amd64 for Cloud Run compatibility (especially on Apple Silicon/ARM Macs)
docker build --platform linux/amd64 -f deployment/docker/Dockerfile.heavy -t $IMAGE_URL .

# Push to Artifact Registry
docker push $IMAGE_URL
```

### 5. Deploy to Cloud Run

```bash
PROJECT_ID=$(gcloud config get-value project)
IMAGE_URL="europe-north1-docker.pkg.dev/${PROJECT_ID}/carbon-aware/heavy-model:latest"

gcloud run deploy heavy-model-service \
    --image=$IMAGE_URL \
    --region=europe-north1 \
    --platform=managed \
    --service-account=heavy-model-sa@${PROJECT_ID}.iam.gserviceaccount.com \
    --set-env-vars="GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --set-env-vars="PUBSUB_SENSOR_TOPIC=sensor-data" \
    --set-env-vars="PUBSUB_SENSOR_SUBSCRIPTION=sensor-data-sub" \
    --set-env-vars="PUBSUB_ANOMALY_TOPIC=anomaly-results" \
    --allow-unauthenticated \
    --port=8080 \
    --memory=512Mi \
    --cpu=1 \
    --min-instances=1 \
    --max-instances=10
```

> **Note**: `--min-instances=1` keeps one instance warm to avoid cold starts. Remove for cost savings in development.

### 6. Verify Deployment

```bash
# Get service URL
gcloud run services describe heavy-model-service --region=europe-north1 --format='value(status.url)'

# Test health endpoint
SERVICE_URL=$(gcloud run services describe heavy-model-service --region=europe-north1 --format='value(status.url)')
curl $SERVICE_URL/health
```

---

## Testing

### 1. Publish Test Message

```bash
# Publish a test batch to sensor-data topic
gcloud pubsub topics publish sensor-data --message='{
  "device_id": "test-device",
  "readings": [
    {"timestamp": 1702900000, "vibration": 0.5},
    {"timestamp": 1702900001, "vibration": 5.5},
    {"timestamp": 1702900002, "vibration": -0.3}
  ],
  "count": 3
}'
```

### 2. Check Cloud Logging

```bash
# View recent logs from heavy-model-service
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=heavy-model-service" \
    --limit=50 \
    --format="table(timestamp, textPayload)"
```

### 3. Check Anomaly Results

```bash
# Pull messages from anomaly-results subscription
gcloud pubsub subscriptions pull anomaly-results-sub --auto-ack --limit=10
```

### 4. Verify Cloud Monitoring Metrics

1. Go to [Cloud Monitoring Console](https://console.cloud.google.com/monitoring)
2. Navigate to Metrics Explorer
3. Search for `custom.googleapis.com/anomaly_detection/rate`
4. You should see the anomaly rate metric being reported

---

## Troubleshooting

### Common Issues

#### 1. "Permission denied" errors

```bash
# Verify service account roles
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:heavy-model-sa@" \
    --format="table(bindings.role)"
```

#### 2. Pub/Sub messages not being received

```bash
# Check subscription backlog
gcloud pubsub subscriptions describe sensor-data-sub \
    --format="value(numUndeliveredMessages)"

# Verify subscription is attached to correct topic
gcloud pubsub subscriptions describe sensor-data-sub \
    --format="value(topic)"
```

#### 3. Cloud Run cold starts

- Set `--min-instances=1` to keep one instance warm
- Increase memory if model loading is slow
- Check logs for initialization errors

#### 4. Model not loading

```bash
# Check if model file exists in container
docker run --rm -it $IMAGE_URL ls -la /app/models/
```

#### 5. Emulator not working locally

```bash
# Check emulator is running
docker-compose ps pubsub-emulator

# View emulator logs
docker-compose logs pubsub-emulator

# Restart emulator
docker-compose restart pubsub-emulator pubsub-init
```

### Useful Commands

```bash
# View Cloud Run logs in real-time
gcloud beta run services logs tail heavy-model-service --region=europe-north1

# Describe Cloud Run service
gcloud run services describe heavy-model-service --region=europe-north1

# List all Pub/Sub subscriptions with message counts
gcloud pubsub subscriptions list --format="table(name, topic, ackDeadlineSeconds)"

# Delete and recreate subscription (if stuck)
gcloud pubsub subscriptions delete sensor-data-sub
gcloud pubsub subscriptions create sensor-data-sub --topic=sensor-data --ack-deadline=60
```

---

## Cost Optimization

1. **Development**: Use `--min-instances=0` and the Pub/Sub emulator locally
2. **Staging**: Use `--min-instances=0` with real GCP Pub/Sub
3. **Production**: Use `--min-instances=1` to avoid cold starts

Estimated monthly costs (europe-north1):
- Cloud Run: ~$5-20 (depends on traffic)
- Pub/Sub: ~$0.50 per million messages
- Cloud Logging: Free tier usually sufficient
- Cloud Monitoring: Free tier usually sufficient

---

## Next Steps

1. Set up CI/CD with Cloud Build (`deployment/gcp/cloudbuild.yaml`)
2. Configure alerting for anomaly rate spikes
3. Set up VPC connector for private networking
4. Implement dead letter queue for failed messages

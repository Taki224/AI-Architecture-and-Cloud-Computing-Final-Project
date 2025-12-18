#!/bin/bash
# Pub/Sub Topic and Subscription Initialization Script
# Works with both local emulator and real GCP

set -e

# Configuration
PROJECT_ID="${PUBSUB_PROJECT_ID:-local-project}"
SENSOR_TOPIC="sensor-data"
ANOMALY_TOPIC="anomaly-results"
SENSOR_SUBSCRIPTION="sensor-data-sub"
ANOMALY_SUBSCRIPTION="anomaly-results-sub"

echo "========================================"
echo "Pub/Sub Initialization Script"
echo "========================================"
echo "Project: $PROJECT_ID"

# Detect if using emulator
if [ -n "$PUBSUB_EMULATOR_HOST" ]; then
    echo "Mode: Emulator ($PUBSUB_EMULATOR_HOST)"
    
    # Wait for emulator to be ready
    echo "Waiting for Pub/Sub emulator..."
    MAX_RETRIES=30
    RETRY_COUNT=0
    
    while ! curl -s "http://${PUBSUB_EMULATOR_HOST}" > /dev/null 2>&1; do
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
            echo "ERROR: Pub/Sub emulator not available after $MAX_RETRIES attempts"
            exit 1
        fi
        echo "  Attempt $RETRY_COUNT/$MAX_RETRIES - waiting..."
        sleep 2
    done
    echo "✓ Pub/Sub emulator is ready"
    
    # Use curl to create topics and subscriptions via emulator HTTP API
    EMULATOR_URL="http://${PUBSUB_EMULATOR_HOST}"
    
    # Create sensor-data topic
    echo ""
    echo "Creating topic: $SENSOR_TOPIC"
    curl -s -X PUT "${EMULATOR_URL}/v1/projects/${PROJECT_ID}/topics/${SENSOR_TOPIC}" \
        -H "Content-Type: application/json" \
        -d '{}' || true
    echo "✓ Topic $SENSOR_TOPIC created"
    
    # Create anomaly-results topic
    echo "Creating topic: $ANOMALY_TOPIC"
    curl -s -X PUT "${EMULATOR_URL}/v1/projects/${PROJECT_ID}/topics/${ANOMALY_TOPIC}" \
        -H "Content-Type: application/json" \
        -d '{}' || true
    echo "✓ Topic $ANOMALY_TOPIC created"
    
    # Create sensor-data subscription
    echo ""
    echo "Creating subscription: $SENSOR_SUBSCRIPTION"
    curl -s -X PUT "${EMULATOR_URL}/v1/projects/${PROJECT_ID}/subscriptions/${SENSOR_SUBSCRIPTION}" \
        -H "Content-Type: application/json" \
        -d "{\"topic\": \"projects/${PROJECT_ID}/topics/${SENSOR_TOPIC}\", \"ackDeadlineSeconds\": 60}" || true
    echo "✓ Subscription $SENSOR_SUBSCRIPTION created"
    
    # Create anomaly-results subscription
    echo "Creating subscription: $ANOMALY_SUBSCRIPTION"
    curl -s -X PUT "${EMULATOR_URL}/v1/projects/${PROJECT_ID}/subscriptions/${ANOMALY_SUBSCRIPTION}" \
        -H "Content-Type: application/json" \
        -d "{\"topic\": \"projects/${PROJECT_ID}/topics/${ANOMALY_TOPIC}\", \"ackDeadlineSeconds\": 60}" || true
    echo "✓ Subscription $ANOMALY_SUBSCRIPTION created"

else
    echo "Mode: GCP Production"
    
    # Use gcloud CLI for real GCP
    # Create topics (--quiet suppresses prompts, || true ignores "already exists" errors)
    echo ""
    echo "Creating topics..."
    gcloud pubsub topics create $SENSOR_TOPIC --project=$PROJECT_ID --quiet 2>/dev/null || echo "  Topic $SENSOR_TOPIC already exists"
    gcloud pubsub topics create $ANOMALY_TOPIC --project=$PROJECT_ID --quiet 2>/dev/null || echo "  Topic $ANOMALY_TOPIC already exists"
    
    # Create subscriptions
    echo ""
    echo "Creating subscriptions..."
    gcloud pubsub subscriptions create $SENSOR_SUBSCRIPTION \
        --topic=$SENSOR_TOPIC \
        --project=$PROJECT_ID \
        --ack-deadline=60 \
        --quiet 2>/dev/null || echo "  Subscription $SENSOR_SUBSCRIPTION already exists"
    
    gcloud pubsub subscriptions create $ANOMALY_SUBSCRIPTION \
        --topic=$ANOMALY_TOPIC \
        --project=$PROJECT_ID \
        --ack-deadline=60 \
        --quiet 2>/dev/null || echo "  Subscription $ANOMALY_SUBSCRIPTION already exists"
fi

echo ""
echo "========================================"
echo "Pub/Sub Setup Complete!"
echo "========================================"
echo "Topics:"
echo "  - $SENSOR_TOPIC (edge device -> cloud)"
echo "  - $ANOMALY_TOPIC (cloud -> edge device)"
echo ""
echo "Subscriptions:"
echo "  - $SENSOR_SUBSCRIPTION (heavy model subscribes)"
echo "  - $ANOMALY_SUBSCRIPTION (edge device subscribes)"
echo "========================================"

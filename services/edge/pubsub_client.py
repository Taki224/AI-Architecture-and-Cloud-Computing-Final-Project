"""
Google Cloud Pub/Sub Client for Edge Device
Handles publishing sensor data and subscribing to anomaly results
"""
import json
import os
import time
import threading
from typing import Callable, Dict, List, Optional
from datetime import datetime


class PubSubClient:
    """
    Base Pub/Sub client with automatic emulator detection.
    Uses google-cloud-pubsub library for GCP integration.
    """
    
    def __init__(self):
        """Initialize the Pub/Sub client with project and emulator detection."""
        self.project_id = os.getenv('GOOGLE_CLOUD_PROJECT', 'local-project')
        self.emulator_host = os.getenv('PUBSUB_EMULATOR_HOST')
        self._publisher = None
        self._subscriber = None
        self._connected = False
        
        if self.emulator_host:
            print(f"[PubSub] Using emulator at {self.emulator_host}")
        else:
            print(f"[PubSub] Using GCP project: {self.project_id}")
    
    def _get_publisher(self):
        """Lazy initialization of publisher client."""
        if self._publisher is None:
            try:
                from google.cloud import pubsub_v1
                self._publisher = pubsub_v1.PublisherClient()
                self._connected = True
            except Exception as e:
                print(f"[PubSub] Failed to create publisher: {e}")
                self._connected = False
        return self._publisher
    
    def _get_subscriber(self):
        """Lazy initialization of subscriber client."""
        if self._subscriber is None:
            try:
                from google.cloud import pubsub_v1
                self._subscriber = pubsub_v1.SubscriberClient()
                self._connected = True
            except Exception as e:
                print(f"[PubSub] Failed to create subscriber: {e}")
                self._connected = False
        return self._subscriber
    
    @property
    def is_connected(self) -> bool:
        """Check if Pub/Sub client is connected."""
        return self._connected


class SensorPublisher(PubSubClient):
    """
    Publishes batched sensor readings to Pub/Sub.
    Batches 10 readings (1 second of data) before publishing.
    """
    
    BATCH_SIZE = 10  # Number of readings per batch
    
    def __init__(self, device_id: str = None):
        """
        Initialize the sensor publisher.
        
        Args:
            device_id: Unique identifier for this edge device
        """
        super().__init__()
        self.device_id = device_id or os.getenv('DEVICE_ID', 'edge-001')
        self.topic_name = os.getenv('PUBSUB_SENSOR_TOPIC', 'sensor-data')
        self._batch_buffer: List[Dict] = []
        self._lock = threading.Lock()
        
        print(f"[SensorPublisher] Device: {self.device_id}, Topic: {self.topic_name}")
    
    def _get_topic_path(self) -> str:
        """Get the full topic path."""
        publisher = self._get_publisher()
        if publisher:
            return publisher.topic_path(self.project_id, self.topic_name)
        return None
    
    def add_reading(self, timestamp: float, vibration: float) -> bool:
        """
        Add a sensor reading to the batch buffer.
        Automatically publishes when batch is full.
        
        Args:
            timestamp: Unix timestamp of the reading
            vibration: Vibration sensor value
            
        Returns:
            True if batch was published, False otherwise
        """
        reading = {
            'timestamp': timestamp,
            'vibration': vibration
        }
        
        with self._lock:
            self._batch_buffer.append(reading)
            
            if len(self._batch_buffer) >= self.BATCH_SIZE:
                return self._publish_batch()
        
        return False
    
    def _publish_batch(self) -> bool:
        """
        Publish the current batch to Pub/Sub.
        
        Returns:
            True if publish succeeded, False otherwise
        """
        if not self._batch_buffer:
            return False
        
        publisher = self._get_publisher()
        topic_path = self._get_topic_path()
        
        if not publisher or not topic_path:
            print("[SensorPublisher] Publisher not available, clearing batch")
            self._batch_buffer.clear()
            return False
        
        # Create batch message
        message = {
            'device_id': self.device_id,
            'readings': self._batch_buffer.copy(),
            'count': len(self._batch_buffer),
            'published_at': datetime.utcnow().isoformat()
        }
        
        try:
            # Publish message
            data = json.dumps(message).encode('utf-8')
            future = publisher.publish(topic_path, data)
            message_id = future.result(timeout=5)  # Wait up to 5 seconds
            
            print(f"[SensorPublisher] Published batch of {len(self._batch_buffer)} readings (msg_id: {message_id})")
            self._batch_buffer.clear()
            return True
            
        except Exception as e:
            print(f"[SensorPublisher] Failed to publish batch: {e}")
            self._batch_buffer.clear()  # Clear to avoid memory buildup
            return False
    
    def flush(self) -> bool:
        """
        Force publish any remaining readings in the buffer.
        
        Returns:
            True if publish succeeded or buffer was empty, False otherwise
        """
        with self._lock:
            if self._batch_buffer:
                return self._publish_batch()
        return True


class AnomalySubscriber(PubSubClient):
    """
    Subscribes to anomaly detection results from the cloud.
    Handles batched responses from the heavy model service.
    """
    
    def __init__(self, callback: Callable[[Dict], None] = None):
        """
        Initialize the anomaly subscriber.
        
        Args:
            callback: Function to call when anomaly results are received
        """
        super().__init__()
        self.subscription_name = os.getenv('PUBSUB_ANOMALY_SUBSCRIPTION', 'anomaly-results-sub')
        self.callback = callback
        self._streaming_future = None
        self._running = False
        
        print(f"[AnomalySubscriber] Subscription: {self.subscription_name}")
    
    def _get_subscription_path(self) -> str:
        """Get the full subscription path."""
        subscriber = self._get_subscriber()
        if subscriber:
            return subscriber.subscription_path(self.project_id, self.subscription_name)
        return None
    
    def _message_callback(self, message):
        """
        Internal callback for processing received messages.
        
        Args:
            message: Pub/Sub message object
        """
        try:
            # Decode message
            data = json.loads(message.data.decode('utf-8'))
            
            # Process batch of results
            readings = data.get('readings', [])
            anomaly_count = sum(1 for r in readings if r.get('is_anomaly', False))
            
            print(f"[AnomalySubscriber] Received {len(readings)} results, {anomaly_count} anomalies")
            
            # Call user callback if provided
            if self.callback:
                self.callback(data)
            
            # Acknowledge message after successful processing
            message.ack()
            
        except Exception as e:
            print(f"[AnomalySubscriber] Error processing message: {e}")
            # Still acknowledge to avoid redelivery loops
            message.ack()
    
    def start(self):
        """Start listening for anomaly results in a background thread."""
        if self._running:
            print("[AnomalySubscriber] Already running")
            return
        
        subscriber = self._get_subscriber()
        subscription_path = self._get_subscription_path()
        
        if not subscriber or not subscription_path:
            print("[AnomalySubscriber] Subscriber not available")
            return
        
        try:
            self._streaming_future = subscriber.subscribe(
                subscription_path,
                callback=self._message_callback
            )
            self._running = True
            print(f"[AnomalySubscriber] Listening on {subscription_path}")
            
        except Exception as e:
            print(f"[AnomalySubscriber] Failed to start: {e}")
            self._running = False
    
    def stop(self):
        """Stop listening for anomaly results."""
        if self._streaming_future:
            self._streaming_future.cancel()
            self._streaming_future = None
        self._running = False
        print("[AnomalySubscriber] Stopped")
    
    @property
    def is_running(self) -> bool:
        """Check if subscriber is currently running."""
        return self._running


def create_publisher(device_id: str = None) -> Optional[SensorPublisher]:
    """
    Factory function to create a sensor publisher.
    Returns None if Pub/Sub is not available.
    
    Args:
        device_id: Optional device identifier
        
    Returns:
        SensorPublisher instance or None
    """
    try:
        publisher = SensorPublisher(device_id)
        # Test connection by getting publisher client
        if publisher._get_publisher():
            return publisher
        return None
    except Exception as e:
        print(f"[PubSub] Failed to create publisher: {e}")
        return None


def create_subscriber(callback: Callable[[Dict], None] = None) -> Optional[AnomalySubscriber]:
    """
    Factory function to create an anomaly subscriber.
    Returns None if Pub/Sub is not available.
    
    Args:
        callback: Function to call when results are received
        
    Returns:
        AnomalySubscriber instance or None
    """
    try:
        subscriber = AnomalySubscriber(callback)
        # Test connection by getting subscriber client
        if subscriber._get_subscriber():
            return subscriber
        return None
    except Exception as e:
        print(f"[PubSub] Failed to create subscriber: {e}")
        return None

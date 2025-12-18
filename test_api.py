"""
Test script to send data to the ML Model API
Demonstrates how the edge device can communicate with the model service
"""
import requests
import json
import time

# API endpoint (use 'localhost' when testing, or 'ml-model' from within Docker network)
API_URL = "http://localhost:5000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*70)
    print("Testing Health Endpoint")
    print("="*70)
    
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_single_prediction():
    """Test single value prediction"""
    print("\n" + "="*70)
    print("Testing Single Prediction")
    print("="*70)
    
    # Test normal value
    normal_data = {"value": 1.2, "model_type": "light"}
    print(f"\nSending normal value: {normal_data}")
    response = requests.post(f"{API_URL}/predict", json=normal_data)
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test anomaly value
    anomaly_data = {"value": 6.5, "model_type": "light"}
    print(f"\nSending anomaly value: {anomaly_data}")
    response = requests.post(f"{API_URL}/predict", json=anomaly_data)
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*70)
    print("Testing Batch Prediction")
    print("="*70)
    
    batch_data = {
        "values": [0.5, 1.2, 7.8, -0.3, 5.5, 1.0],
        "model_type": "heavy"
    }
    
    print(f"\nSending batch: {batch_data['values']}")
    response = requests.post(f"{API_URL}/predict", json=batch_data)
    result = response.json()
    
    print(f"\nTotal predictions: {result['count']}")
    anomalies = [r for r in result['predictions'] if r['is_anomaly']]
    print(f"Anomalies detected: {len(anomalies)}")
    
    for pred in result['predictions']:
        status = "🚨 ANOMALY" if pred['is_anomaly'] else "✓ Normal"
        print(f"  {status}: value={pred['value']:.2f}, score={pred['anomaly_score']:.4f}")


def test_batch_analysis():
    """Test batch analysis with timestamps"""
    print("\n" + "="*70)
    print("Testing Batch Analysis")
    print("="*70)
    
    readings = [
        {"timestamp": "2025-12-17T10:30:00", "value": 1.2},
        {"timestamp": "2025-12-17T10:30:01", "value": 0.8},
        {"timestamp": "2025-12-17T10:30:02", "value": 6.5},
        {"timestamp": "2025-12-17T10:30:03", "value": 1.1},
        {"timestamp": "2025-12-17T10:30:04", "value": 7.2},
    ]
    
    data = {
        "readings": readings,
        "model_type": "light"
    }
    
    print(f"\nSending {len(readings)} readings...")
    response = requests.post(f"{API_URL}/analyze/batch", json=data)
    result = response.json()
    
    print(f"\nSummary:")
    print(json.dumps(result['summary'], indent=2))


def simulate_streaming():
    """Simulate streaming sensor data"""
    print("\n" + "="*70)
    print("Simulating Streaming Data (10 readings)")
    print("="*70)
    
    import random
    
    for i in range(10):
        # Generate reading (mostly normal, occasionally anomalous)
        if random.random() < 0.2:  # 20% anomalies
            value = random.uniform(5.0, 8.0)  # Anomaly
        else:
            value = random.gauss(0.0, 1.0)  # Normal
        
        data = {"value": value, "model_type": "light"}
        
        print(f"\n[{i+1}/10] Sending value: {value:.4f}")
        response = requests.post(f"{API_URL}/predict", json=data)
        result = response.json()
        
        if result['is_anomaly']:
            print(f"        🚨 ANOMALY DETECTED! Score: {result['anomaly_score']:.4f}")
        else:
            print(f"        ✓ Normal (score: {result['anomaly_score']:.4f})")
        
        time.sleep(0.5)  # Wait between readings


if __name__ == "__main__":
    try:
        print("\n" + "="*70)
        print("ML Model API Test Suite")
        print("="*70)
        print("Make sure the ML model service is running:")
        print("  docker-compose up ml-model")
        print("="*70)
        
        # Run all tests
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_batch_analysis()
        simulate_streaming()
        
        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to ML Model API")
        print("   Make sure the service is running: docker-compose up ml-model")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")

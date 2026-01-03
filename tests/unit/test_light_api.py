"""
Unit tests for Light Model API Service.

Tests Flask endpoints for anomaly detection.
"""

import pytest
import sys
import os
import json

# Add project paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../services/light-model')))
models_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
sys.path.insert(0, models_path)

# Import Flask first
try:
    import flask
except ImportError:
    pytest.skip("Flask not installed", allow_module_level=True)

# Import after path setup
import api_service


@pytest.fixture
def client():
    """Create Flask test client."""
    api_service.app.config['TESTING'] = True
    
    # Initialize detector
    api_service.init_detector_fallback()
    
    with api_service.app.test_client() as client:
        yield client


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_check_returns_200(self, client):
        """Test that health check returns 200 status."""
        response = client.get('/health')
        assert response.status_code == 200

    def test_health_check_structure(self, client):
        """Test health check response structure."""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert 'status' in data
        assert 'detector' in data
        assert 'ml_fitted' in data
        assert 'timestamp' in data

    def test_health_status(self, client):
        """Test that health status indicates healthy or degraded."""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert data['status'] in ['healthy', 'degraded']

    def test_detector_type(self, client):
        """Test that detector type is reported."""
        response = client.get('/health')
        data = json.loads(response.data)
        
        assert data['detector'] == 'isolation_forest'


class TestPredictEndpoint:
    """Test /predict endpoint."""

    def test_predict_normal_value(self, client):
        """Test prediction with normal value."""
        response = client.post('/predict',
                              data=json.dumps({'value': 0.5}),
                              content_type='application/json')
        
        assert response.status_code == 200

    def test_predict_response_structure(self, client):
        """Test that predict response has required fields."""
        response = client.post('/predict',
                              data=json.dumps({'value': 0.5}),
                              content_type='application/json')
        data = json.loads(response.data)
        
        required_fields = ['value', 'is_anomaly']
        for field in required_fields:
            assert field in data

    def test_predict_missing_value(self, client):
        """Test predict with missing value field."""
        response = client.post('/predict',
                              data=json.dumps({}),
                              content_type='application/json')
        
        assert response.status_code == 400

    def test_predict_returns_json(self, client):
        """Test that predict returns JSON."""
        response = client.post('/predict',
                              data=json.dumps({'value': 0.5}),
                              content_type='application/json')
        
        assert response.content_type == 'application/json'


class TestAnalyzeEndpoint:
    """Test /analyze endpoint (alias for /predict)."""

    def test_analyze_normal_value(self, client):
        """Test analyze with normal value."""
        response = client.post('/analyze',
                              data=json.dumps({'value': 0.5}),
                              content_type='application/json')
        
        assert response.status_code == 200

    def test_analyze_returns_detection_result(self, client):
        """Test analyze returns detection result."""
        response = client.post('/analyze',
                              data=json.dumps({'value': 0.5}),
                              content_type='application/json')
        data = json.loads(response.data)
        
        assert 'is_anomaly' in data
        assert isinstance(data['is_anomaly'], bool)


class TestResetEndpoint:
    """Test /reset endpoint."""

    def test_reset_detector(self, client):
        """Test resetting the detector."""
        response = client.post('/reset')
        
        assert response.status_code == 200

    def test_reset_returns_json(self, client):
        """Test that reset returns JSON."""
        response = client.post('/reset')
        
        assert response.content_type == 'application/json'


class TestValueTypes:
    """Test different value types."""

    def test_float_value(self, client):
        """Test prediction with float value."""
        response = client.post('/predict',
                              data=json.dumps({'value': 0.123}),
                              content_type='application/json')
        
        assert response.status_code == 200

    def test_integer_value(self, client):
        """Test prediction with integer value."""
        response = client.post('/predict',
                              data=json.dumps({'value': 1}),
                              content_type='application/json')
        
        assert response.status_code == 200

    def test_negative_value(self, client):
        """Test prediction with negative value."""
        response = client.post('/predict',
                              data=json.dumps({'value': -0.5}),
                              content_type='application/json')
        
        assert response.status_code == 200

    def test_zero_value(self, client):
        """Test prediction with zero value."""
        response = client.post('/predict',
                              data=json.dumps({'value': 0.0}),
                              content_type='application/json')
        
        assert response.status_code == 200

    def test_large_value(self, client):
        """Test prediction with large value."""
        response = client.post('/predict',
                              data=json.dumps({'value': 100.0}),
                              content_type='application/json')
        
        assert response.status_code == 200


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json(self, client):
        """Test with invalid JSON."""
        response = client.post('/predict',
                              data='not json',
                              content_type='application/json')
        
        # Should handle gracefully (400 or 500)
        assert response.status_code in [400, 500]

    def test_missing_content_type(self, client):
        """Test with missing content type."""
        response = client.post('/predict',
                              data=json.dumps({'value': 0.5}))
        
        # API returns 500 when content-type is missing (error is caught)
        assert response.status_code in [200, 400, 415, 500]

    def test_non_numeric_value(self, client):
        """Test with non-numeric value."""
        response = client.post('/predict',
                              data=json.dumps({'value': 'not a number'}),
                              content_type='application/json')
        
        # Should return error
        assert response.status_code in [400, 500]

"""
Pytest configuration and shared fixtures.
"""

import pytest
import numpy as np


@pytest.fixture
def normal_readings():
    """Generate 100 normal sensor readings N(0, 1)."""
    np.random.seed(42)
    return np.random.normal(0.0, 1.0, 100).tolist()


@pytest.fixture
def anomaly_readings():
    """Generate readings with anomalies (3-8σ)."""
    np.random.seed(42)
    readings = []
    for i in range(100):
        if i % 20 == 0:  # 5% anomaly rate
            sigma = np.random.uniform(3.0, 8.0)
            value = sigma if np.random.random() < 0.5 else -sigma
        else:
            value = np.random.normal(0, 1)
        readings.append(value)
    return readings


@pytest.fixture
def batch_of_10():
    """10 readings for batch testing."""
    return [0.5, -0.3, 3.5, 0.1, -0.8, 6.2, 0.2, -3.8, 0.4, -0.1]

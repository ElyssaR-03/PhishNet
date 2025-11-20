"""Integration tests converted from debug helper.

These tests use FastAPI's TestClient to exercise important API flows
that were previously invoked by `backend/scripts/debug_api_call.py`.
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_train_endpoint_integration():
    """Call the /train endpoint with a small synthetic request and assert success."""
    resp = client.post('/train', json={'n_samples': 50})
    assert resp.status_code == 200
    data = resp.json()
    assert 'success' in data
    assert data['success'] is True
    assert 'accuracies' in data


def test_models_info_available():
    """Verify /models/info is available and reports training state."""
    resp = client.get('/models/info')
    assert resp.status_code == 200
    data = resp.json()
    assert 'available_models' in data
    assert 'models_trained' in data


def test_analyze_url_smoke():
    """Smoke test analyze URL endpoint to ensure prediction flow works."""
    resp = client.post('/analyze/url', json={'url': 'https://www.example.com', 'model': 'random_forest'})
    assert resp.status_code == 200
    data = resp.json()
    assert 'is_phishing' in data
    assert 'confidence' in data

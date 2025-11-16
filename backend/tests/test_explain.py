"""Tests for explanation endpoints and PhishingDetector.explain"""
import numpy as np
from fastapi.testclient import TestClient
from main import app, detector

client = TestClient(app)


def setup_detector():
    if not detector.is_trained:
        # train on small synthetic data compatible with the feature vector size (16)
        X = np.random.rand(200, 16)
        y = np.random.randint(0, 2, 200)
        detector.train(X, y)


def test_explain_url_endpoint():
    setup_detector()
    response = client.post("/explain/url", json={"url": "https://www.example.com", "model": "random_forest"})
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert isinstance(data["top_features"], list)


def test_explain_email_endpoint():
    setup_detector()
    response = client.post("/explain/email", json={"content": "Please verify your account.", "sender": "noreply@example.com", "model": "random_forest"})
    assert response.status_code == 200
    data = response.json()
    assert "top_features" in data
    assert isinstance(data["top_features"], list)


def test_explain_method_direct():
    setup_detector()
    X = np.random.rand(16)
    res = detector.explain(X, model_name='random_forest', top_k=3)
    assert isinstance(res, dict)
    assert 'top_features' in res
    assert isinstance(res['top_features'], list)

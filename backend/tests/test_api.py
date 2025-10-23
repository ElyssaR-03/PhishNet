"""
Unit tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
from main import app, detector
import numpy as np


# Create test client
client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_models():
    """Setup: train models before each test."""
    if not detector.is_trained:
        X = np.random.rand(100, 16)
        y = np.random.randint(0, 2, 100)
        detector.train(X, y)


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["models_loaded"] is True


class TestAnalysisEndpoints:
    """Test analysis endpoints."""
    
    def test_analyze_url_basic(self):
        """Test URL analysis with basic URL."""
        response = client.post(
            "/analyze/url",
            json={"url": "https://www.example.com", "model": "random_forest"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_phishing" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "features" in data
        assert isinstance(data["is_phishing"], bool)
        assert 0 <= data["confidence"] <= 1
    
    def test_analyze_url_suspicious(self):
        """Test URL analysis with suspicious URL."""
        response = client.post(
            "/analyze/url",
            json={"url": "http://192.168.1.1/login-verify-account", "model": "random_forest"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_phishing" in data
    
    def test_analyze_url_ensemble(self):
        """Test URL analysis with ensemble model."""
        response = client.post(
            "/analyze/url",
            json={"url": "https://example.com", "model": "ensemble"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "individual_predictions" in data
        assert data["model_used"] == "ensemble"
    
    def test_analyze_email_basic(self):
        """Test email analysis."""
        response = client.post(
            "/analyze/email",
            json={
                "content": "Hello, this is a test email.",
                "sender": "test@example.com",
                "model": "random_forest"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_phishing" in data
        assert "confidence" in data
    
    def test_analyze_email_suspicious(self):
        """Test email analysis with suspicious content."""
        response = client.post(
            "/analyze/email",
            json={
                "content": "URGENT! Verify account! Prize $1000!!!",
                "sender": "noreply123@suspicious.com",
                "model": "random_forest"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "is_phishing" in data


class TestModelsEndpoints:
    """Test model-related endpoints."""
    
    def test_models_info(self):
        """Test models info endpoint."""
        response = client.get("/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "available_models" in data
        assert "default_model" in data
        assert "models_trained" in data
        assert "svm" in data["available_models"]
        assert "random_forest" in data["available_models"]
        assert "logistic_regression" in data["available_models"]
    
    def test_train_endpoint(self):
        """Test training endpoint."""
        response = client.post(
            "/train",
            json={"n_samples": 100}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "accuracies" in data


class TestEducationEndpoints:
    """Test education endpoints."""
    
    def test_education_tips(self):
        """Test education tips endpoint."""
        response = client.get("/education/tips")
        assert response.status_code == 200
        data = response.json()
        assert "tips" in data
        assert isinstance(data["tips"], list)
        assert len(data["tips"]) > 0
        assert "title" in data["tips"][0]
        assert "description" in data["tips"][0]

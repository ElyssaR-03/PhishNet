"""
Unit tests for ML models module.
"""
import pytest
import numpy as np
from models.ml_models import PhishingDetector


class TestPhishingDetector:
    """Test suite for PhishingDetector class."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = PhishingDetector()
        
        assert detector is not None
        assert 'svm' in detector.models
        assert 'random_forest' in detector.models
        assert 'logistic_regression' in detector.models
        assert detector.is_trained is False
    
    def test_train_basic(self):
        """Test basic training functionality."""
        detector = PhishingDetector()
        
        # Create simple training data
        X = np.random.rand(100, 16)
        y = np.random.randint(0, 2, 100)
        
        scores = detector.train(X, y)
        
        assert detector.is_trained is True
        assert 'svm' in scores
        assert 'random_forest' in scores
        assert 'logistic_regression' in scores
        assert all(0 <= score <= 1 for score in scores.values())
    
    def test_predict_before_training(self):
        """Test that prediction fails before training."""
        detector = PhishingDetector()
        X = np.random.rand(16)
        
        with pytest.raises(ValueError, match="Models must be trained"):
            detector.predict(X)
    
    def test_predict_after_training(self):
        """Test prediction after training."""
        detector = PhishingDetector()
        
        # Train with simple data
        X_train = np.random.rand(100, 16)
        y_train = np.random.randint(0, 2, 100)
        detector.train(X_train, y_train)
        
        # Make prediction
        X_test = np.random.rand(16)
        prediction, confidence = detector.predict(X_test, 'random_forest')
        
        assert prediction in [0, 1]
        assert 0 <= confidence <= 1
    
    def test_predict_invalid_model(self):
        """Test prediction with invalid model name."""
        detector = PhishingDetector()
        
        # Train first
        X_train = np.random.rand(100, 16)
        y_train = np.random.randint(0, 2, 100)
        detector.train(X_train, y_train)
        
        X_test = np.random.rand(16)
        
        with pytest.raises(ValueError, match="Unknown model"):
            detector.predict(X_test, 'invalid_model')
    
    def test_predict_ensemble(self):
        """Test ensemble prediction."""
        detector = PhishingDetector()
        
        # Train with simple data
        X_train = np.random.rand(100, 16)
        y_train = np.random.randint(0, 2, 100)
        detector.train(X_train, y_train)
        
        # Make ensemble prediction
        X_test = np.random.rand(16)
        prediction, confidence, individual = detector.predict_ensemble(X_test)
        
        assert prediction in [0, 1]
        assert 0 <= confidence <= 1
        assert 'svm' in individual
        assert 'random_forest' in individual
        assert 'logistic_regression' in individual
    
    def test_predict_2d_input(self):
        """Test prediction with 2D input array."""
        detector = PhishingDetector()
        
        # Train
        X_train = np.random.rand(100, 16)
        y_train = np.random.randint(0, 2, 100)
        detector.train(X_train, y_train)
        
        # Test with 2D input
        X_test = np.random.rand(1, 16)
        prediction, confidence = detector.predict(X_test)
        
        assert prediction in [0, 1]
        assert 0 <= confidence <= 1
    
    def test_save_before_training(self):
        """Test that saving fails before training."""
        detector = PhishingDetector()
        
        with pytest.raises(ValueError, match="Models must be trained"):
            detector.save_models()
    
    def test_train_different_test_sizes(self):
        """Test training with different test sizes."""
        detector = PhishingDetector()
        
        X = np.random.rand(100, 16)
        y = np.random.randint(0, 2, 100)
        
        scores = detector.train(X, y, test_size=0.3)
        
        assert all(0 <= score <= 1 for score in scores.values())

"""
Unit tests for feature extraction module.
"""
import pytest
from feature_extractor import FeatureExtractor


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""
    
    def test_extract_url_features_basic(self):
        """Test basic URL feature extraction."""
        extractor = FeatureExtractor()
        url = "https://www.example.com/path?query=value"
        features = extractor.extract_url_features(url)
        
        # Check that all expected features are present
        assert 'url_length' in features
        assert 'num_dots' in features
        assert 'is_https' in features
        assert 'domain_length' in features
        
        # Check specific values
        assert features['url_length'] == len(url)
        assert features['is_https'] == 1.0
        assert features['num_dots'] == 2
    
    def test_extract_url_features_suspicious(self):
        """Test URL with suspicious characteristics."""
        extractor = FeatureExtractor()
        url = "http://192.168.1.1/login-verify-account"
        features = extractor.extract_url_features(url)
        
        assert features['has_ip'] == 1.0
        assert features['is_https'] == 0.0
        assert features['has_suspicious_keywords'] == 1.0
    
    def test_extract_url_features_complex(self):
        """Test URL with many special characters."""
        extractor = FeatureExtractor()
        url = "https://example.com/path/to/page?param1=val1&param2=val2&user=test@example.com"
        features = extractor.extract_url_features(url)
        
        assert features['num_slashes'] > 0
        assert features['num_ampersands'] > 0
        assert features['num_equals'] > 0
        assert features['num_at'] > 0
    
    def test_extract_email_features_basic(self):
        """Test basic email feature extraction."""
        extractor = FeatureExtractor()
        content = "Hello, this is a test email."
        sender = "test@example.com"
        features = extractor.extract_email_features(content, sender)
        
        # Check that all expected features are present
        assert 'content_length' in features
        assert 'num_urls' in features
        assert 'sender_length' in features
        
        # Check specific values
        assert features['content_length'] == len(content)
        assert features['sender_length'] == len(sender)
    
    def test_extract_email_features_suspicious(self):
        """Test email with suspicious characteristics."""
        extractor = FeatureExtractor()
        content = "URGENT! Verify your account NOW! Click here: http://phishing.com Prize: $1000!!!"
        sender = "noreply123@suspicious.com"
        features = extractor.extract_email_features(content, sender)
        
        assert features['num_suspicious_keywords'] > 0
        assert features['has_money_keywords'] == 1.0
        assert features['num_exclamations'] > 0
        assert features['sender_has_numbers'] == 1.0
        assert features['num_urls'] == 1
    
    def test_extract_email_features_no_sender(self):
        """Test email feature extraction without sender."""
        extractor = FeatureExtractor()
        content = "Test email content"
        features = extractor.extract_email_features(content)
        
        assert features['sender_length'] == 0
        assert features['sender_has_numbers'] == 0.0
    
    def test_get_feature_names(self):
        """Test getting feature names."""
        extractor = FeatureExtractor()
        feature_names = extractor.get_feature_names()
        
        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert 'url_length' in feature_names
        assert 'content_length' in feature_names

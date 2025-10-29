"""
Unit tests for PhishTank service integration.
"""
import pytest
from unittest.mock import Mock, patch
from phishtank_service import PhishTankService


class TestPhishTankService:
    """Test PhishTank service functionality."""
    
    def test_service_initialization_without_key(self):
        """Test service initialization without API key."""
        service = PhishTankService(api_key="")
        assert service.is_enabled() is False
    
    def test_service_initialization_with_key(self):
        """Test service initialization with API key."""
        service = PhishTankService(api_key="test_key_123")
        assert service.is_enabled() is True
    
    def test_check_url_disabled(self):
        """Test URL check when service is disabled."""
        service = PhishTankService(api_key="")
        result = service.check_url("https://example.com")
        
        assert result['success'] is False
        assert result['in_database'] is False
        assert 'error' in result
        assert 'not configured' in result['error']
    
    @patch('phishtank_service.requests.post')
    def test_check_url_not_in_database(self, mock_post):
        """Test URL check when URL is not in PhishTank database."""
        # Mock successful response with URL not in database
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {
                'in_database': False
            }
        }
        mock_post.return_value = mock_response
        
        service = PhishTankService(api_key="test_key")
        result = service.check_url("https://example.com")
        
        assert result['success'] is True
        assert result['in_database'] is False
    
    @patch('phishtank_service.requests.post')
    def test_check_url_in_database_verified(self, mock_post):
        """Test URL check when URL is in database and verified."""
        # Mock successful response with URL in database
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {
                'in_database': True,
                'verified': True,
                'phish_id': 12345,
                'verified_at': '2024-01-01T12:00:00+00:00',
                'phish_detail_page': 'https://www.phishtank.com/phish_detail.php?phish_id=12345'
            }
        }
        mock_post.return_value = mock_response
        
        service = PhishTankService(api_key="test_key")
        result = service.check_url("https://phishing-site.com")
        
        assert result['success'] is True
        assert result['in_database'] is True
        assert result['verified'] is True
        assert result['phish_id'] == 12345
        assert 'verified_at' in result
        assert 'detail_url' in result
    
    @patch('phishtank_service.requests.post')
    def test_check_url_in_database_unverified(self, mock_post):
        """Test URL check when URL is in database but not verified."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': {
                'in_database': True,
                'verified': False,
                'phish_id': 67890
            }
        }
        mock_post.return_value = mock_response
        
        service = PhishTankService(api_key="test_key")
        result = service.check_url("https://suspicious-site.com")
        
        assert result['success'] is True
        assert result['in_database'] is True
        assert result['verified'] is False
        assert result['phish_id'] == 67890
    
    @patch('phishtank_service.requests.post')
    def test_check_url_api_error(self, mock_post):
        """Test URL check when API returns error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response
        
        service = PhishTankService(api_key="test_key")
        result = service.check_url("https://example.com")
        
        assert result['success'] is False
        assert result['in_database'] is False
        assert 'error' in result
        assert '500' in result['error']
    
    @patch('phishtank_service.requests.post')
    def test_check_url_timeout(self, mock_post):
        """Test URL check with timeout."""
        import requests
        mock_post.side_effect = requests.exceptions.Timeout()
        
        service = PhishTankService(api_key="test_key")
        result = service.check_url("https://example.com")
        
        assert result['success'] is False
        assert result['in_database'] is False
        assert 'error' in result
        assert 'timed out' in result['error']
    
    @patch('phishtank_service.requests.post')
    def test_check_url_network_error(self, mock_post):
        """Test URL check with network error."""
        import requests
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        
        service = PhishTankService(api_key="test_key")
        result = service.check_url("https://example.com")
        
        assert result['success'] is False
        assert result['in_database'] is False
        assert 'error' in result
        assert 'request failed' in result['error'].lower()

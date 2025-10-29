"""
PhishTank API integration service for PhishNet.
Provides functionality to check URLs against the PhishTank database.
"""
import requests
import hashlib
from typing import Dict, Optional
from config import settings


class PhishTankService:
    """Service for interacting with PhishTank API."""
    
    BASE_URL = "https://checkurl.phishtank.com/checkurl/"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PhishTank service.
        
        Args:
            api_key: PhishTank API key. If not provided, uses value from settings.
        """
        self.api_key = api_key or settings.PHISHTANK_API_KEY
        self.enabled = bool(self.api_key)
    
    def check_url(self, url: str, timeout: int = 10) -> Dict:
        """
        Check if a URL is in the PhishTank database.
        
        Args:
            url: The URL to check
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary containing PhishTank check results:
            {
                'success': bool,
                'in_database': bool,
                'verified': bool,
                'verified_at': str (optional),
                'phish_id': int (optional),
                'error': str (optional)
            }
        """
        if not self.enabled:
            return {
                'success': False,
                'in_database': False,
                'error': 'PhishTank API key not configured'
            }
        
        try:
            # Prepare the request
            data = {
                'url': url,
                'format': 'json',
                'app_key': self.api_key
            }
            
            # Make the API request
            response = requests.post(
                self.BASE_URL,
                data=data,
                timeout=timeout
            )
            
            # Check response status
            if response.status_code != 200:
                return {
                    'success': False,
                    'in_database': False,
                    'error': f'PhishTank API returned status {response.status_code}'
                }
            
            # Parse response
            result = response.json()
            
            # Extract relevant information
            results_data = result.get('results', {})
            in_database = results_data.get('in_database', False)
            
            response_data = {
                'success': True,
                'in_database': in_database,
            }
            
            # Add additional info if URL is in database
            if in_database:
                response_data['verified'] = results_data.get('verified', False)
                response_data['phish_id'] = results_data.get('phish_id')
                
                # Add verification timestamp if available
                if results_data.get('verified_at'):
                    response_data['verified_at'] = results_data.get('verified_at')
                
                # Add detail URL if available
                if results_data.get('phish_detail_page'):
                    response_data['detail_url'] = results_data.get('phish_detail_page')
            
            return response_data
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'in_database': False,
                'error': 'PhishTank API request timed out'
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'in_database': False,
                'error': f'PhishTank API request failed: {str(e)}'
            }
        except (ValueError, KeyError) as e:
            return {
                'success': False,
                'in_database': False,
                'error': f'Failed to parse PhishTank response: {str(e)}'
            }
    
    def is_enabled(self) -> bool:
        """Check if PhishTank service is enabled."""
        return self.enabled


# Global PhishTank service instance
phishtank_service = PhishTankService()

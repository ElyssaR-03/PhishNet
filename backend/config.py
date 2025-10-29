"""
Configuration management for PhishNet.
Handles environment variables and API keys.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)


class Settings:
    """Application settings."""
    
    # PhishTank API Configuration
    PHISHTANK_API_KEY: str = os.getenv('PHISHTANK_API_KEY', '')
    PHISHTANK_ENABLED: bool = bool(PHISHTANK_API_KEY)
    
    # API Configuration
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    
    @classmethod
    def is_phishtank_configured(cls) -> bool:
        """Check if PhishTank API is properly configured."""
        return bool(cls.PHISHTANK_API_KEY)


# Global settings instance
settings = Settings()

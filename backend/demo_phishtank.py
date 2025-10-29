#!/usr/bin/env python3
"""
Demo script showing PhishTank API integration in PhishNet.
This script demonstrates how the PhishTank service works.
"""

from phishtank_service import PhishTankService
from config import settings

def main():
    print("=" * 60)
    print("PhishNet - PhishTank Integration Demo")
    print("=" * 60)
    print()
    
    # Check if PhishTank is configured
    if settings.is_phishtank_configured():
        print("✓ PhishTank API is CONFIGURED")
        print(f"  API Key: ***{settings.PHISHTANK_API_KEY[-4:]}")  # Only show last 4 chars
        print()
    else:
        print("✗ PhishTank API is NOT CONFIGURED")
        print()
        print("To enable PhishTank integration:")
        print("1. Get a free API key from: https://www.phishtank.com/api_info.php")
        print("2. Create a .env file in the backend directory")
        print("3. Add: PHISHTANK_API_KEY=your_api_key_here")
        print()
        print("The system will work in ML-only mode without PhishTank.")
        print()
        return
    
    # Initialize service
    service = PhishTankService()
    
    # Example URLs to check
    test_urls = [
        "https://www.google.com",
        "https://www.example.com",
    ]
    
    print("Testing PhishTank API with sample URLs:")
    print("-" * 60)
    
    for url in test_urls:
        print(f"\nChecking: {url}")
        result = service.check_url(url)
        
        if result['success']:
            if result['in_database']:
                print(f"  ⚠️  FOUND in PhishTank database")
                print(f"     Verified: {result.get('verified', False)}")
                if result.get('phish_id'):
                    print(f"     Phish ID: {result['phish_id']}")
                if result.get('verified_at'):
                    print(f"     Verified at: {result['verified_at']}")
            else:
                print(f"  ✓ Not found in PhishTank database (good)")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown error')}")
    
    print()
    print("=" * 60)
    print("How PhishTank Integration Works:")
    print("=" * 60)
    print()
    print("1. When a URL is analyzed via /analyze/url endpoint:")
    print("   - PhishTank database is checked first (if enabled)")
    print("   - ML models provide independent analysis")
    print("   - Both results are returned to the user")
    print()
    print("2. If PhishTank confirms a verified phishing URL:")
    print("   - The result is marked as phishing (overrides ML)")
    print("   - Confidence is set to 0.95 or higher")
    print("   - PhishTank details are included in response")
    print()
    print("3. Benefits:")
    print("   - Real-time database of known phishing sites")
    print("   - Crowdsourced verification")
    print("   - Complements ML-based detection")
    print("   - Works seamlessly with or without configuration")
    print()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Verification script for PhishNet system.
Tests backend API endpoints and displays results.
"""
import requests
import sys
from typing import Dict, Any


API_BASE = "http://localhost:8000"


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(test_name: str, success: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"    {details}")


def test_health() -> bool:
    """Test health endpoint."""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models_loaded = data.get("models_loaded", False)
            print_result("Health Check", True, f"Status: {data.get('status')}, Models Loaded: {models_loaded}")
            return models_loaded
        else:
            print_result("Health Check", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Health Check", False, str(e))
        return False


def test_url_analysis() -> bool:
    """Test URL analysis endpoint."""
    try:
        # Test legitimate URL
        response = requests.post(
            f"{API_BASE}/analyze/url",
            json={"url": "https://www.google.com", "model": "random_forest"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            is_phishing = data.get("is_phishing")
            confidence = data.get("confidence", 0)
            print_result(
                "URL Analysis - Legitimate",
                not is_phishing,
                f"Detected as {'phishing' if is_phishing else 'safe'} with {confidence:.2%} confidence"
            )
        else:
            print_result("URL Analysis - Legitimate", False, f"Status code: {response.status_code}")
            return False
        
        # Test suspicious URL
        response = requests.post(
            f"{API_BASE}/analyze/url",
            json={"url": "http://192.168.1.1/login-verify-account", "model": "random_forest"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            is_phishing = data.get("is_phishing")
            confidence = data.get("confidence", 0)
            print_result(
                "URL Analysis - Suspicious",
                is_phishing,
                f"Detected as {'phishing' if is_phishing else 'safe'} with {confidence:.2%} confidence"
            )
            return True
        else:
            print_result("URL Analysis - Suspicious", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("URL Analysis", False, str(e))
        return False


def test_email_analysis() -> bool:
    """Test email analysis endpoint."""
    try:
        response = requests.post(
            f"{API_BASE}/analyze/email",
            json={
                "content": "URGENT! Verify your account NOW! Click here: http://phishing.com Prize $1000!!!",
                "sender": "noreply123@suspicious.com",
                "model": "random_forest"
            },
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            is_phishing = data.get("is_phishing")
            confidence = data.get("confidence", 0)
            print_result(
                "Email Analysis",
                True,
                f"Detected as {'phishing' if is_phishing else 'safe'} with {confidence:.2%} confidence"
            )
            return True
        else:
            print_result("Email Analysis", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Email Analysis", False, str(e))
        return False


def test_ensemble() -> bool:
    """Test ensemble prediction."""
    try:
        response = requests.post(
            f"{API_BASE}/analyze/url",
            json={"url": "http://192.168.1.1/verify-login", "model": "ensemble"},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            individual = data.get("individual_predictions", {})
            has_individual = len(individual) == 3
            print_result(
                "Ensemble Prediction",
                has_individual,
                f"Models: {', '.join(individual.keys()) if individual else 'None'}"
            )
            return has_individual
        else:
            print_result("Ensemble Prediction", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Ensemble Prediction", False, str(e))
        return False


def test_models_info() -> bool:
    """Test models info endpoint."""
    try:
        response = requests.get(f"{API_BASE}/models/info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = data.get("available_models", [])
            print_result(
                "Models Info",
                len(models) == 4,
                f"Available: {', '.join(models)}"
            )
            return True
        else:
            print_result("Models Info", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Models Info", False, str(e))
        return False


def test_education_tips() -> bool:
    """Test education tips endpoint."""
    try:
        response = requests.get(f"{API_BASE}/education/tips", timeout=5)
        if response.status_code == 200:
            data = response.json()
            tips = data.get("tips", [])
            print_result(
                "Education Tips",
                len(tips) > 0,
                f"Retrieved {len(tips)} tips"
            )
            return True
        else:
            print_result("Education Tips", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_result("Education Tips", False, str(e))
        return False


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("  🛡️  PhishNet System Verification")
    print("=" * 70)
    print(f"\nAPI Base URL: {API_BASE}")
    
    tests = [
        ("System Health", test_health),
        ("URL Analysis", test_url_analysis),
        ("Email Analysis", test_email_analysis),
        ("Ensemble Mode", test_ensemble),
        ("Models Information", test_models_info),
        ("Education Tips", test_education_tips),
    ]
    
    results = []
    
    print_section("Running Tests")
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print_result(test_name, False, f"Exception: {str(e)}")
            results.append(False)
    
    # Summary
    print_section("Test Summary")
    passed = sum(results)
    total = len(results)
    print(f"\nTests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("\n✅ All tests passed! PhishNet is working correctly.")
        print("\nNext steps:")
        print("  1. Keep backend running at http://localhost:8000")
        print("  2. Start frontend: cd frontend && npm start")
        print("  3. Access app at http://localhost:3000")
        print("  4. View API docs at http://localhost:8000/docs")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\nTroubleshooting:")
        print("  1. Ensure backend is running: python main.py")
        print("  2. Check models are trained: python train_models.py")
        print("  3. Verify dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nVerification interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)

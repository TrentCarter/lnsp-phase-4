#!/usr/bin/env python3
"""
Check n8n authentication status and provide solutions
"""

import urllib.request
import urllib.error
import json

def test_api_endpoints():
    """Test different n8n API endpoints to understand auth status"""

    endpoints = [
        ("/rest/workflows", "Workflows API"),
        ("/rest/settings", "Settings API"),
        ("/rest/owner", "Owner API"),
        ("/rest/login", "Login API"),
        ("/healthz", "Health Check"),
        ("/", "Main Page")
    ]

    print("🔍 Testing n8n API Endpoints")
    print("=" * 40)

    for endpoint, description in endpoints:
        url = f"http://localhost:5678{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                status = response.getcode()
                print(f"✅ {description:15} - {status}")
        except urllib.error.HTTPError as e:
            print(f"❌ {description:15} - {e.code} {e.reason}")
        except Exception as e:
            print(f"❓ {description:15} - Error: {str(e)[:30]}")

    print()

def check_owner_setup():
    """Check if n8n owner is set up"""
    try:
        with urllib.request.urlopen("http://localhost:5678/rest/owner", timeout=5) as response:
            data = json.loads(response.read().decode())
            return data
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return {"error": "unauthorized"}
        return {"error": f"http_{e.code}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🧪 n8n Authentication Diagnostic Tool")
    print("=" * 45)

    # Test endpoints
    test_api_endpoints()

    # Check owner setup
    print("🔍 Checking Owner Setup")
    print("-" * 25)
    owner_status = check_owner_setup()

    if "error" in owner_status:
        if owner_status["error"] == "unauthorized":
            print("❌ Owner API requires authentication")
            print("   This means user management is ENABLED")
        else:
            print(f"❓ Owner API error: {owner_status['error']}")
    else:
        print("✅ Owner API accessible")
        print(f"   Data: {owner_status}")

    print()
    print("💡 SOLUTIONS:")
    print("=" * 15)

    print("1. 🔄 Complete n8n Restart:")
    print("   pkill -f n8n")
    print("   N8N_USER_MANAGEMENT_DISABLED=true N8N_SECURE_COOKIE=false n8n start")
    print()

    print("2. 🏠 Reset n8n User Data (nuclear option):")
    print("   pkill -f n8n")
    print("   rm -rf ~/.n8n")
    print("   N8N_USER_MANAGEMENT_DISABLED=true N8N_SECURE_COOKIE=false n8n start")
    print()

    print("3. 🌐 Use Browser Method (safest):")
    print("   python3 n8n_workflows/test_no_auth.py")
    print()

    print("4. 🔍 Check Environment Variables:")
    print("   env | grep N8N")

if __name__ == "__main__":
    main()
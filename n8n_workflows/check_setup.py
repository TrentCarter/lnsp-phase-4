#!/usr/bin/env python3
"""
Check n8n setup and webhook availability
"""

import urllib.request
import urllib.error
import json

def check_n8n_running():
    """Check if n8n is accessible"""
    try:
        with urllib.request.urlopen("http://localhost:5678", timeout=5) as response:
            return True
    except:
        return False

def check_webhook_active():
    """Check if webhook endpoint is available"""
    try:
        # Try to access the webhook endpoint
        req = urllib.request.Request(
            "http://localhost:5678/webhook/vec2text",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return True
    except urllib.error.HTTPError as e:
        # 405 Method Not Allowed is expected for GET on POST endpoint
        if e.code == 405:
            return True
        return False
    except:
        return False

def main():
    print("🔍 n8n Setup Checker")
    print("=" * 30)

    # Check n8n
    print("1. Checking n8n status...")
    if check_n8n_running():
        print("   ✅ n8n is running at http://localhost:5678")
    else:
        print("   ❌ n8n is not running")
        print("   💡 Start with: N8N_SECURE_COOKIE=false n8n start")
        return

    # Check webhook
    print("2. Checking webhook endpoint...")
    if check_webhook_active():
        print("   ✅ Webhook endpoint is active at /webhook/vec2text")
        print("   🎯 Ready to test with: python n8n_workflows/test_webhook_simple.py")
    else:
        print("   ❌ Webhook endpoint not found")
        print("   💡 Steps to activate:")
        print("      1. Open http://localhost:5678")
        print("      2. Find 'Vec2Text API Webhook' workflow")
        print("      3. Toggle it to 'Active'")
        print("      4. Execute it once to register the webhook")

if __name__ == "__main__":
    main()
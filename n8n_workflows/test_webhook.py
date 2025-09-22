#!/usr/bin/env python3
"""
Simple test script for the n8n vec2text webhook API
"""

import requests
import json
import time
import sys
from typing import Dict, Any, Optional

# Configuration
N8N_BASE_URL = "http://localhost:5678"
WEBHOOK_PATH = "/webhook/vec2text"
WEBHOOK_URL = f"{N8N_BASE_URL}{WEBHOOK_PATH}"

# Test data
TEST_TEXTS = [
    "What is the role of glucose in diabetes?",
    "Artificial intelligence and machine learning",
    "Neural networks process information",
    "One day, a little girl named Lily found",
    "The quick brown fox jumps over the lazy dog"
]

def test_webhook(text: str, **kwargs) -> Dict[str, Any]:
    """Test the webhook with given text and parameters"""

    payload = {
        "text": text,
        "subscribers": kwargs.get("subscribers", "jxe,ielab"),
        "steps": kwargs.get("steps", 1),
        "backend": kwargs.get("backend", "isolated"),
        "format": kwargs.get("format", "json")
    }

    try:
        print(f"📤 Testing: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   Payload: {json.dumps(payload, indent=2)}")

        response = requests.post(
            WEBHOOK_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )

        response.raise_for_status()
        result = response.json()

        print(f"✅ Status: {result.get('status', 'unknown')}")

        if result.get('status') == 'success':
            print(f"   Result: {json.dumps(result.get('result', {}), indent=2)}")
        elif result.get('status') == 'error':
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        else:
            print(f"⚠️  Warning: {result.get('message', 'Unknown warning')}")

        return result

    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed. Is n8n running at {N8N_BASE_URL}?")
        return {"error": "Connection failed"}
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (>30s)")
        return {"error": "Timeout"}
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return {"error": f"HTTP Error: {e}"}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON response: {response.text}")
        return {"error": "Invalid JSON response"}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return {"error": f"Unexpected error: {e}"}

def check_n8n_status() -> bool:
    """Check if n8n is running and accessible"""
    try:
        response = requests.get(N8N_BASE_URL, timeout=5)
        return response.status_code in [200, 401]  # 401 is also OK (auth required)
    except:
        return False

def main():
    """Main test function"""
    print("🚀 n8n Vec2Text Webhook Test")
    print("=" * 50)

    # Check n8n status
    print("🔍 Checking n8n status...")
    if not check_n8n_status():
        print(f"❌ n8n is not accessible at {N8N_BASE_URL}")
        print("   Please start n8n with: N8N_SECURE_COOKIE=false n8n start")
        sys.exit(1)

    print(f"✅ n8n is running at {N8N_BASE_URL}")
    print(f"🎯 Testing webhook at: {WEBHOOK_URL}")
    print()

    # Parse command line arguments
    if len(sys.argv) > 1:
        # Test with custom text
        custom_text = " ".join(sys.argv[1:])
        print(f"Testing with custom text: {custom_text}")
        test_webhook(custom_text)
    else:
        # Test with predefined texts
        print(f"Testing with {len(TEST_TEXTS)} predefined texts:")
        print()

        results = []
        for i, text in enumerate(TEST_TEXTS, 1):
            print(f"--- Test {i}/{len(TEST_TEXTS)} ---")
            result = test_webhook(text)
            results.append({"text": text, "result": result})
            print()

            # Brief pause between tests
            if i < len(TEST_TEXTS):
                time.sleep(1)

        # Summary
        print("📊 SUMMARY")
        print("=" * 20)
        successful = len([r for r in results if r["result"].get("status") == "success"])
        print(f"✅ Successful: {successful}/{len(results)}")
        print(f"❌ Failed: {len(results) - successful}/{len(results)}")

        if successful < len(results):
            print("\n🔍 Failed tests:")
            for result in results:
                if result["result"].get("status") != "success":
                    print(f"   - '{result['text'][:30]}...': {result['result'].get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
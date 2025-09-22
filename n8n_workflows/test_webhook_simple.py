#!/usr/bin/env python3
"""
Quick and simple webhook test - minimal dependencies
"""

import urllib.request
import urllib.parse
import json

# Test the webhook
def test_webhook():
    url = "http://localhost:5678/webhook/vec2text"
    data = {
        "text": "What is AI?",
        "subscribers": "jxe,ielab",
        "steps": 1
    }

    # Convert to JSON and encode
    json_data = json.dumps(data).encode('utf-8')

    # Create request
    req = urllib.request.Request(
        url,
        data=json_data,
        headers={'Content-Type': 'application/json'}
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            print("✅ Success!")
            print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Simple webhook test...")
    test_webhook()
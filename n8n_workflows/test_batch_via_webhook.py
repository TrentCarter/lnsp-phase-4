#!/usr/bin/env python3
"""
Batch test via webhook - processes multiple texts through vec2text
"""

import urllib.request
import json
import time

# Test texts
TEST_TEXTS = [
    "What is the role of glucose in diabetes?",
    "Artificial intelligence and machine learning",
    "Neural networks process information",
    "One day, a little girl named Lily found",
    "The quick brown fox jumps over the lazy dog"
]

def call_webhook(text):
    """Call the webhook with a single text"""
    url = "http://localhost:5678/webhook/vec2text"
    data = {
        "text": text,
        "subscribers": "jxe,ielab",
        "steps": 1
    }

    json_data = json.dumps(data).encode('utf-8')
    req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return {"error": str(e)}

def main():
    print("🧪 Batch Vec2Text Test via Webhook")
    print("=" * 40)

    results = []

    for i, text in enumerate(TEST_TEXTS, 1):
        print(f"\n📤 Test {i}/{len(TEST_TEXTS)}: {text[:50]}...")

        result = call_webhook(text)
        results.append({"input": text, "result": result})

        if "error" in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   ✅ Success: {result.get('status', 'processed')}")

        # Brief pause between requests
        if i < len(TEST_TEXTS):
            time.sleep(1)

    # Summary
    print(f"\n📊 BATCH RESULTS SUMMARY")
    print("=" * 30)

    successful = len([r for r in results if "error" not in r["result"]])
    print(f"✅ Successful: {successful}/{len(results)}")
    print(f"❌ Failed: {len(results) - successful}/{len(results)}")

    # Show detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for i, item in enumerate(results, 1):
        print(f"\n{i}. Input: {item['input']}")
        if "error" in item["result"]:
            print(f"   ❌ Error: {item['result']['error']}")
        else:
            print(f"   ✅ Result: {json.dumps(item['result'], indent=2)}")

if __name__ == "__main__":
    main()
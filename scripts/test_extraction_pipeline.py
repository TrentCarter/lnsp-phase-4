#!/usr/bin/env python3
"""
This is a DIAGNOSTIC script to definitively identify the root cause of the
Ollama connection issue. It does NOT run the full pipeline.

It performs the following checks:
1. Reports the environment variables it detects.
2. Tests connectivity to the base Ollama URL.
3. Tests connectivity to standard Ollama API endpoints.
4. Provides a clear summary and next steps.

Usage:
  ./venv/bin/python3 scripts/test_extraction_pipeline.py
"""

import os
import requests
import sys

def main():
    print("--- Running Ollama Connection Diagnostic Tool ---")

    # 1. Report Environment Variables
    print("\n[STEP 1] Checking Environment Variables...")
    ollama_url_env = os.getenv("OLLAMA_URL")
    ollama_host_env = os.getenv("OLLAMA_HOST")
    print(f"  -> OLLAMA_URL: {ollama_url_env if ollama_url_env else '[Not Set]'}")
    print(f"  -> OLLAMA_HOST: {ollama_host_env if ollama_host_env else '[Not Set]'}")

    # Determine the base URL to test
    base_url = ollama_url_env or ollama_host_env or "http://localhost:11434"
    # Clean up the base URL to ensure it's just the host and port
    if '/api' in base_url:
        base_url = base_url.split('/api')[0]
    base_url = base_url.rstrip('/')
    print(f"  -> Using Base URL for tests: {base_url}")

    # 2. Test Base Connection
    print(f"\n[STEP 2] Testing base connection to {base_url}...")
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200 and "Ollama is running" in response.text:
            print(f"  [SUCCESS] Connected to base URL. Ollama service is running. Status: {response.status_code}")
            base_conn_ok = True
        else:
            print(f"  [FAILURE] Connected, but did not get expected response. Status: {response.status_code}, Response: {response.text[:100]}...")
            base_conn_ok = False
    except requests.exceptions.RequestException as e:
        print(f"  [FAILURE] Could not connect to {base_url}. Error: {e}")
        base_conn_ok = False

    # 3. Test Standard API Endpoints
    endpoints_to_test = {
        "List Models (/api/tags)": f"{base_url}/api/tags",
        "Chat Endpoint (/api/chat)": f"{base_url}/api/chat",
    }
    
    print("\n[STEP 3] Testing standard API endpoints...")
    all_endpoints_ok = True
    if not base_conn_ok:
        print("  -> Skipping endpoint tests because base connection failed.")
        all_endpoints_ok = False
    else:
        for name, url in endpoints_to_test.items():
            try:
                # For /api/chat, we use POST with a proper body as GET is not allowed
                if "/api/chat" in url:
                    # Use a minimal valid chat request with llama3.1:8b
                    res = requests.post(url, json={
                        "model": "llama3.1:8b",  # Use the correct llama model name
                        "messages": [{"role": "user", "content": "test"}],
                        "stream": False
                    }, headers={"Content-Type": "application/json"}, timeout=10)
                else:
                    res = requests.get(url, timeout=5)
                
                if res.status_code == 200:
                    print(f"  [SUCCESS] Reached {name}. Status: {res.status_code}")
                elif "/api/chat" in url and res.status_code in [400, 500]:
                    # For chat endpoint, 400/500 might indicate model not found but endpoint exists
                    print(f"  [PARTIAL] Reached {name} but got {res.status_code}. Endpoint exists but may need valid model. Status: {res.status_code}")
                    # Don't fail for this - endpoint is reachable
                elif res.status_code == 404:
                    print(f"  [FAILURE] Reached {name} but got 404 Not Found. This endpoint may not exist at this path. Status: {res.status_code}")
                    all_endpoints_ok = False
                else:
                    print(f"  [FAILURE] Reached {name} but got an unexpected status. Status: {res.status_code}, Response: {res.text[:100]}...")
                    all_endpoints_ok = False

            except requests.exceptions.RequestException as e:
                print(f"  [FAILURE] Could not connect to {name}. Error: {e}")
                all_endpoints_ok = False

    # 4. Final Summary and Next Steps
    print("\n--- Diagnostic Summary ---")
    if base_conn_ok and all_endpoints_ok:
        print("✅ All checks passed. The Ollama service appears to be running and all standard endpoints are reachable.")
        print("The original error may have been intermittent or due to a specific model not being available.")
        print("Please try running the original script again.")
    elif base_conn_ok and not all_endpoints_ok:
        print("❌ Partial failure. The Ollama service is running, but one or more API endpoints are not found or are failing.")
        print("This suggests a potential issue with the Ollama installation or a non-standard configuration.")
        print("Please check your Ollama server logs for errors.")
    else:
        print("❌ Critical failure. The script could not connect to the Ollama service at all.")
        print("Please take the following steps:")
        print("  1. Ensure the Ollama application is running on your machine.")
        print("  2. If it is running, check if it's using a non-standard address (e.g., a different port than 11434).")
        print("  3. If you are using a non-standard address, set the OLLAMA_HOST environment variable accordingly.")
        print("     Example: export OLLAMA_HOST='http://localhost:12345'")

    print("\n--- Diagnostic Complete ---")

if __name__ == "__main__":
    main()

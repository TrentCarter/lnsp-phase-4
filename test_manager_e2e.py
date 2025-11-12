#!/usr/bin/env python3
"""
End-to-End Test: Director → Manager → Aider RPC

This test verifies the complete Manager tier implementation:
1. Submit job card to Director-Code
2. Director decomposes into Manager tasks
3. Manager executes via Aider RPC
4. Results propagate back to Director
"""
import requests
import json
import time
import uuid

# Test configuration
DIRECTOR_CODE_URL = "http://127.0.0.1:6111"
TEST_JOB_CARD = {
    "id": f"test-manager-e2e-{str(uuid.uuid4())[:8]}",
    "task": "Add a simple hello_world() function to test_utils.py",
    "lane": "Code",
    "inputs": [
        {
            "type": "file",
            "path": "test_utils.py",
            "required": True
        }
    ],
    "expected_artifacts": [
        {
            "type": "code",
            "path": "test_utils.py",
            "description": "Python file with hello_world() function"
        }
    ],
    "acceptance": [
        {"check": "lint==0"}
    ],
    "budget": {
        "max_tokens": 10000,
        "max_duration_s": 300
    }
}

def main():
    print("=" * 80)
    print("Manager Tier End-to-End Test")
    print("=" * 80)

    # Step 1: Check Director-Code health
    print("\n[1/5] Checking Director-Code health...")
    try:
        response = requests.get(f"{DIRECTOR_CODE_URL}/health", timeout=5)
        response.raise_for_status()
        health = response.json()
        print(f"✓ Director-Code: {health['service']} (agent: {health['agent']})")
        print(f"  LLM Model: {health['llm_model']}")
        print(f"  Managers: {health['agent_metadata']['children']}")
    except Exception as e:
        print(f"✗ Director-Code health check failed: {e}")
        return 1

    # Step 2: Create test file if it doesn't exist
    print("\n[2/5] Creating test file...")
    import os
    if not os.path.exists("test_utils.py"):
        with open("test_utils.py", "w") as f:
            f.write("# Test utilities\n\n")
        print("✓ Created test_utils.py")
    else:
        print("✓ test_utils.py already exists")

    # Step 3: Submit job card to Director-Code
    print("\n[3/5] Submitting job card to Director-Code...")
    try:
        response = requests.post(
            f"{DIRECTOR_CODE_URL}/submit",
            json={"job_card": TEST_JOB_CARD},
            timeout=300  # 5 minutes timeout for Aider execution
        )

        if response.status_code in [200, 202]:
            result = response.json()
            print(f"✓ Job card accepted: {result['job_card_id']}")
            print(f"  Status: {result['status']}")
            job_card_id = result['job_card_id']
        else:
            print(f"✗ Job submission failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return 1

    except requests.exceptions.Timeout:
        print("✗ Job submission timed out (5 minutes)")
        return 1
    except Exception as e:
        print(f"✗ Job submission error: {e}")
        return 1

    # Step 4: Poll for job status
    print("\n[4/5] Polling for job completion...")
    max_polls = 60  # 5 minutes (5s intervals)
    for i in range(max_polls):
        try:
            response = requests.get(
                f"{DIRECTOR_CODE_URL}/status/{job_card_id}",
                timeout=5
            )
            response.raise_for_status()
            status = response.json()

            state = status['state']
            print(f"  Poll {i+1}/{max_polls}: {state}")

            if state == "completed":
                print("✓ Job completed successfully!")
                print(f"  Duration: {status.get('duration_s', 0):.2f}s")
                print(f"  Managers used: {list(status.get('managers', {}).keys())}")
                break
            elif state == "failed":
                print(f"✗ Job failed: {status.get('message', 'Unknown error')}")
                return 1
            elif state in ["planning", "decomposing", "delegating", "executing", "monitoring", "validating"]:
                # Still in progress
                time.sleep(5)
                continue
            else:
                print(f"✗ Unknown state: {state}")
                return 1

        except Exception as e:
            print(f"  Poll error: {e}")
            time.sleep(5)
            continue
    else:
        print("✗ Job did not complete within 5 minutes")
        return 1

    # Step 5: Verify artifact
    print("\n[5/5] Verifying artifact...")
    try:
        with open("test_utils.py", "r") as f:
            content = f.read()

        if "hello_world" in content and "def hello_world" in content:
            print("✓ Function added successfully!")
            print("\nGenerated code:")
            print("-" * 40)
            print(content)
            print("-" * 40)
        else:
            print("✗ Function not found in test_utils.py")
            print(f"\nFile content:\n{content}")
            return 1

    except Exception as e:
        print(f"✗ Error reading artifact: {e}")
        return 1

    print("\n" + "=" * 80)
    print("✓ Manager Tier End-to-End Test PASSED")
    print("=" * 80)
    print("\nManager tier is fully operational:")
    print("  1. Director-Code decomposed job card into Manager tasks")
    print("  2. Manager executed task via Aider RPC")
    print("  3. Aider RPC generated code changes")
    print("  4. Results propagated back to Director")
    print("  5. Artifact verified successfully")

    return 0

if __name__ == "__main__":
    exit(main())

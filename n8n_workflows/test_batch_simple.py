#!/usr/bin/env python3
"""
Simple test for the Vec2Text Testing Workflow (batch processing)
Minimal dependencies version
"""

import urllib.request
import json
import time

def test_batch_workflow():
    """Execute the batch workflow and show results"""
    base_url = "http://localhost:5678"

    try:
        # 1. Get list of workflows to find our workflow ID
        print("🔍 Finding Vec2Text Testing Workflow...")
        req = urllib.request.Request(f"{base_url}/rest/workflows")
        with urllib.request.urlopen(req, timeout=10) as response:
            workflows = json.loads(response.read().decode())

        workflow_id = None
        for workflow in workflows:
            if workflow.get('name') == 'Vec2Text Testing Workflow':
                workflow_id = workflow.get('id')
                break

        if not workflow_id:
            print("❌ Workflow not found. Make sure you imported vec2text_test_workflow.json")
            return

        print(f"✅ Found workflow: {workflow_id}")

        # 2. Execute the workflow
        print("🚀 Executing batch workflow...")
        execute_url = f"{base_url}/rest/workflows/{workflow_id}/execute"
        req = urllib.request.Request(execute_url, method="POST")
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            execution_id = result.get('data', {}).get('executionId')

        if not execution_id:
            print("❌ Failed to start execution")
            return

        print(f"⏳ Execution started: {execution_id}")

        # 3. Wait for completion (simple polling)
        for i in range(30):  # Wait up to 60 seconds
            time.sleep(2)
            status_url = f"{base_url}/rest/executions/{execution_id}"
            req = urllib.request.Request(status_url)

            with urllib.request.urlopen(req, timeout=10) as response:
                execution = json.loads(response.read().decode())

            if execution.get('finished'):
                if execution.get('success'):
                    print("✅ Workflow completed successfully!")
                    print(f"\n📊 Results summary:")
                    print(f"   Execution ID: {execution_id}")
                    print(f"   View at: {base_url}/execution/{execution_id}")
                else:
                    print("❌ Workflow completed with errors")
                return

            print(".", end="", flush=True)

        print("\n⏰ Timeout - workflow may still be running")
        print(f"Check status at: {base_url}/execution/{execution_id}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Simple Vec2Text Batch Workflow Test")
    print("=" * 40)
    test_batch_workflow()
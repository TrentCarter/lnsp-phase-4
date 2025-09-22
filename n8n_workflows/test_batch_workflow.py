#!/usr/bin/env python3
"""
Test script for the "Vec2Text Testing Workflow" (batch processing workflow)
This script triggers and monitors the batch workflow execution.
"""

import urllib.request
import urllib.parse
import urllib.error
import json
import time
import sys

N8N_BASE_URL = "http://localhost:5678"

def find_workflow_id():
    """Find the workflow ID for 'Vec2Text Testing Workflow'"""
    try:
        req = urllib.request.Request(f"{N8N_BASE_URL}/rest/workflows")
        with urllib.request.urlopen(req, timeout=10) as response:
            workflows = json.loads(response.read().decode())

        for workflow in workflows:
            if workflow.get('name') == 'Vec2Text Testing Workflow':
                return workflow.get('id')

        return None
    except Exception as e:
        print(f"❌ Error finding workflow: {e}")
        return None

def execute_workflow(workflow_id):
    """Execute the workflow and return execution ID"""
    try:
        # Execute workflow
        execute_url = f"{N8N_BASE_URL}/rest/workflows/{workflow_id}/execute"
        req = urllib.request.Request(execute_url, method="POST")
        req.add_header('Content-Type', 'application/json')

        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            return result.get('data', {}).get('executionId')

    except Exception as e:
        print(f"❌ Error executing workflow: {e}")
        return None

def get_execution_status(execution_id):
    """Get the status and results of a workflow execution"""
    try:
        status_url = f"{N8N_BASE_URL}/rest/executions/{execution_id}"
        req = urllib.request.Request(status_url)

        with urllib.request.urlopen(req, timeout=10) as response:
            execution = json.loads(response.read().decode())
            return execution

    except Exception as e:
        print(f"❌ Error getting execution status: {e}")
        return None

def wait_for_completion(execution_id, max_wait=60):
    """Wait for workflow execution to complete"""
    print(f"⏳ Waiting for execution {execution_id} to complete...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        execution = get_execution_status(execution_id)

        if not execution:
            print("❌ Could not get execution status")
            return None

        status = execution.get('finished', False)

        if status:
            success = execution.get('success', False)
            if success:
                print("✅ Workflow completed successfully!")
            else:
                print("❌ Workflow completed with errors")
            return execution

        print(".", end="", flush=True)
        time.sleep(2)

    print(f"\n⏰ Timeout after {max_wait}s")
    return None

def parse_results(execution):
    """Parse and display workflow results"""
    if not execution:
        return

    print("\n📊 WORKFLOW RESULTS")
    print("=" * 50)

    # Get execution data
    execution_data = execution.get('data', {})
    result_data = execution_data.get('resultData', {})

    if not result_data:
        print("No result data found")
        return

    # Look for the "Display Results" node or similar
    for node_name, node_data in result_data.items():
        print(f"\n🔍 Node: {node_name}")
        print("-" * 30)

        if isinstance(node_data, list) and node_data:
            for i, item in enumerate(node_data):
                if isinstance(item, list) and item:
                    for j, data_item in enumerate(item):
                        if 'json' in data_item:
                            json_data = data_item['json']
                            print(f"Item {i+1}.{j+1}:")

                            # Pretty print relevant fields
                            if 'original_text' in json_data:
                                print(f"  Input: {json_data['original_text']}")

                            if 'processing_result' in json_data:
                                result = json_data['processing_result']
                                if isinstance(result, dict):
                                    print(f"  Result: {json.dumps(result, indent=4)}")
                                else:
                                    print(f"  Result: {result}")

                            if 'success' in json_data:
                                status = "✅ Success" if json_data['success'] else "❌ Failed"
                                print(f"  Status: {status}")

                            if 'error' in json_data:
                                print(f"  Error: {json_data['error']}")

                            print()

def check_n8n_status():
    """Check if n8n is running"""
    try:
        with urllib.request.urlopen(f"{N8N_BASE_URL}", timeout=5) as response:
            return True
    except:
        return False

def main():
    """Main test function"""
    print("🧪 Vec2Text Testing Workflow - Batch Test")
    print("=" * 50)

    # Check n8n status
    if not check_n8n_status():
        print(f"❌ n8n is not accessible at {N8N_BASE_URL}")
        print("   Start with: N8N_SECURE_COOKIE=false n8n start")
        sys.exit(1)

    print(f"✅ n8n is running at {N8N_BASE_URL}")

    # Find workflow
    print("🔍 Looking for 'Vec2Text Testing Workflow'...")
    workflow_id = find_workflow_id()

    if not workflow_id:
        print("❌ Workflow not found!")
        print("   Make sure you've imported: vec2text_test_workflow.json")
        sys.exit(1)

    print(f"✅ Found workflow ID: {workflow_id}")

    # Execute workflow
    print("🚀 Executing workflow...")
    execution_id = execute_workflow(workflow_id)

    if not execution_id:
        print("❌ Failed to execute workflow")
        sys.exit(1)

    print(f"✅ Execution started: {execution_id}")

    # Wait for completion
    execution = wait_for_completion(execution_id, max_wait=120)

    if execution:
        parse_results(execution)

        # Show execution summary
        success = execution.get('success', False)
        if success:
            print("\n🎉 Batch test completed successfully!")
        else:
            print("\n💥 Batch test completed with errors")

        print(f"🔗 View details at: {N8N_BASE_URL}/execution/{execution_id}")
    else:
        print("\n⚠️  Execution may still be running")
        print(f"🔗 Check status at: {N8N_BASE_URL}/execution/{execution_id}")

if __name__ == "__main__":
    main()
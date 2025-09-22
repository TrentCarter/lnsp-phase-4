#!/usr/bin/env python3
"""
Test script that prompts for n8n authentication if needed
"""

import urllib.request
import urllib.error
import json
import sys
import getpass
import base64

def get_credentials():
    """Prompt for n8n credentials"""
    print("🔐 n8n requires authentication")
    email = input("Email: ")
    password = getpass.getpass("Password: ")
    return email, password

def make_authenticated_request(url, method="GET", data=None, email=None, password=None):
    """Make request with basic auth if credentials provided"""
    req = urllib.request.Request(url, method=method)

    if data:
        req.add_header('Content-Type', 'application/json')
        req.data = json.dumps(data).encode('utf-8')

    if email and password:
        credentials = f"{email}:{password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        req.add_header('Authorization', f'Basic {encoded_credentials}')

    return urllib.request.urlopen(req, timeout=10)

def test_api_access():
    """Test if we can access n8n API"""
    try:
        with urllib.request.urlopen("http://localhost:5678/rest/workflows", timeout=5) as response:
            return True, None
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return False, "authentication_required"
        return False, f"http_error_{e.code}"
    except Exception as e:
        return False, str(e)

def find_and_execute_workflow():
    """Find and execute the workflow with authentication if needed"""
    base_url = "http://localhost:5678"
    email, password = None, None

    # Test API access
    accessible, error = test_api_access()

    if not accessible:
        if error == "authentication_required":
            print("🔐 n8n API requires authentication")
            email, password = get_credentials()
        else:
            print(f"❌ Cannot access n8n API: {error}")
            return False

    try:
        # Get workflows
        with make_authenticated_request(f"{base_url}/rest/workflows", email=email, password=password) as response:
            workflows = json.loads(response.read().decode())

        # Find our workflow
        workflow_id = None
        for workflow in workflows:
            if workflow.get('name') == 'Vec2Text Testing Workflow':
                workflow_id = workflow.get('id')
                break

        if not workflow_id:
            print("❌ 'Vec2Text Testing Workflow' not found")
            print("   Available workflows:")
            for wf in workflows[:5]:  # Show first 5
                print(f"   - {wf.get('name', 'Unnamed')}")
            return False

        print(f"✅ Found workflow: {workflow_id}")

        # Execute workflow
        print("🚀 Executing workflow...")
        execute_url = f"{base_url}/rest/workflows/{workflow_id}/execute"

        with make_authenticated_request(execute_url, method="POST", email=email, password=password) as response:
            result = json.loads(response.read().decode())
            execution_id = result.get('data', {}).get('executionId')

        if execution_id:
            print(f"✅ Execution started: {execution_id}")
            print(f"🔗 View at: {base_url}/execution/{execution_id}")
            return True
        else:
            print("❌ Failed to start execution")
            return False

    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("❌ Authentication failed")
        else:
            print(f"❌ HTTP Error {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🧪 Vec2Text Workflow Test (with auth support)")
    print("=" * 50)

    # Check if n8n is running
    try:
        with urllib.request.urlopen("http://localhost:5678", timeout=5) as response:
            print("✅ n8n is running")
    except:
        print("❌ n8n is not running")
        print("   Start with: N8N_SECURE_COOKIE=false n8n start")
        sys.exit(1)

    # Try to execute workflow
    if find_and_execute_workflow():
        print("\n🎉 Workflow execution started successfully!")
        print("\n💡 For real-time monitoring, use the n8n web interface:")
        print("   http://localhost:5678/executions")
    else:
        print("\n🔧 Alternative: Manual execution")
        print("   1. Open: http://localhost:5678/workflow/vec2text_test_001")
        print("   2. Click 'Execute Workflow'")

if __name__ == "__main__":
    main()
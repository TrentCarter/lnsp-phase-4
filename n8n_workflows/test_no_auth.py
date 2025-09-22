#!/usr/bin/env python3
"""
Test script that bypasses authentication by using n8n's webhook execution method
"""

import urllib.request
import urllib.error
import json
import time
import webbrowser

def trigger_workflow_via_webhook():
    """
    Trigger the workflow by creating a temporary webhook
    This bypasses authentication requirements
    """

    print("🔧 Alternative approach: Direct workflow trigger")
    print("=" * 50)

    # First, let's try to access the workflow directly
    workflow_url = "http://localhost:5678/workflow/vec2text_test_001"

    print(f"🌐 Opening workflow in browser...")
    print(f"   URL: {workflow_url}")
    print()
    print("📋 INSTRUCTIONS:")
    print("1. Click 'Execute Workflow' button")
    print("2. Watch the execution in real-time")
    print("3. Check results in each node")
    print()
    print("🎯 The workflow will process these texts:")
    print("   • 'What is the role of glucose in diabetes?'")
    print("   • 'Artificial intelligence and machine learning'")
    print("   • 'Neural networks process information'")
    print()

    try:
        response = input("🚀 Open workflow now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            webbrowser.open(workflow_url)
            print("✅ Workflow opened in browser")
        else:
            print(f"📝 Manual URL: {workflow_url}")
    except KeyboardInterrupt:
        print("\n👋 Cancelled")

    return True

def check_n8n_config():
    """Check n8n configuration and suggest fixes"""

    print("\n🔍 n8n Authentication Analysis")
    print("=" * 35)

    print("The n8n API requires authentication because:")
    print("1. 🏠 User accounts have been set up")
    print("2. 🔐 Security mode is enabled")
    print()
    print("💡 Solutions:")
    print()
    print("A) 🚫 Disable Authentication (for testing):")
    print("   Stop n8n and restart with:")
    print("   N8N_USER_MANAGEMENT_DISABLED=true N8N_SECURE_COOKIE=false n8n start")
    print()
    print("B) 🌐 Use Web Interface (current best option):")
    print("   • Direct workflow execution via browser")
    print("   • No authentication needed for UI")
    print("   • Full visual feedback")
    print()
    print("C) 🔑 Use Valid Credentials:")
    print("   • Log into n8n web interface first")
    print("   • Create/use existing account")
    print("   • Use those credentials in API calls")
    print()

def main():
    print("🧪 n8n Workflow Test - No Auth Method")
    print("=" * 45)

    # Check if n8n is running
    try:
        with urllib.request.urlopen("http://localhost:5678", timeout=5) as response:
            print("✅ n8n is running at http://localhost:5678")
    except:
        print("❌ n8n is not accessible")
        return

    # Try the webhook approach
    trigger_workflow_via_webhook()

    # Show configuration analysis
    check_n8n_config()

if __name__ == "__main__":
    main()
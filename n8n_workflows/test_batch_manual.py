#!/usr/bin/env python3
"""
Manual workflow test - provides instructions for manual execution
since n8n API requires authentication
"""

import urllib.request
import webbrowser
import sys

def check_n8n_status():
    """Check if n8n is accessible"""
    try:
        with urllib.request.urlopen("http://localhost:5678", timeout=5) as response:
            return True
    except:
        return False

def main():
    print("🧪 Vec2Text Testing Workflow - Manual Test Guide")
    print("=" * 55)

    # Check n8n status
    if not check_n8n_status():
        print("❌ n8n is not accessible at http://localhost:5678")
        print("   Start with: N8N_SECURE_COOKIE=false n8n start")
        sys.exit(1)

    print("✅ n8n is running")
    print()

    # Manual testing instructions
    print("📋 MANUAL TEST STEPS:")
    print("=" * 30)
    print("1. Open n8n workflow editor:")
    print("   🔗 http://localhost:5678/workflow/vec2text_test_001")
    print()
    print("2. Click the 'Execute Workflow' button")
    print()
    print("3. Watch the execution progress:")
    print("   ➤ Generate Test Data")
    print("   ➤ Execute Vec2Text")
    print("   ➤ Parse Results")
    print("   ➤ Display Results")
    print()
    print("4. Check results in each node:")
    print("   • Click on each node to see its output")
    print("   • Look for vec2text processing results")
    print("   • Verify JXE and IELab decoder outputs")
    print()

    # Test texts that will be processed
    test_texts = [
        "What is the role of glucose in diabetes?",
        "Artificial intelligence and machine learning",
        "Neural networks process information"
    ]

    print("📝 TEST TEXTS (predefined in workflow):")
    print("-" * 40)
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    print()

    print("🎯 EXPECTED RESULTS:")
    print("-" * 20)
    print("• Each text should be processed through:")
    print("  - GTR-T5 encoder (text → vector)")
    print("  - JXE decoder (vector → text)")
    print("  - IELab decoder (vector → text)")
    print("• JSON output with processing results")
    print("• Success/error status for each step")
    print()

    # Offer to open browser
    try:
        response = input("🌐 Open workflow in browser? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            workflow_url = "http://localhost:5678/workflow/vec2text_test_001"
            print(f"🔗 Opening: {workflow_url}")
            webbrowser.open(workflow_url)
        else:
            print("📋 Workflow URL: http://localhost:5678/workflow/vec2text_test_001")
    except KeyboardInterrupt:
        print("\n👋 Cancelled")
    except:
        print("📋 Workflow URL: http://localhost:5678/workflow/vec2text_test_001")

    print()
    print("💡 TIP: If the workflow fails, check:")
    print("   • Python venv is activated")
    print("   • Vec2text dependencies are installed")
    print("   • JXE and IELab adapters are available")

if __name__ == "__main__":
    main()
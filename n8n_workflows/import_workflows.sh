#!/bin/bash

# n8n Workflow Import Script
# This script imports n8n workflows using the CLI

echo "🚀 n8n Workflow Import Tool"
echo "=========================="

# Check if n8n is installed
if ! command -v n8n &> /dev/null; then
    echo "❌ n8n is not installed. Please install it first:"
    echo "   npm install -g n8n"
    exit 1
fi

# Set the workflows directory
WORKFLOW_DIR="$(dirname "$0")"
cd "$WORKFLOW_DIR" || exit 1

echo "📁 Working directory: $WORKFLOW_DIR"

# Function to import a workflow
import_workflow() {
    local workflow_file=$1
    local workflow_name=$(basename "$workflow_file" .json)

    if [ ! -f "$workflow_file" ]; then
        echo "❌ File not found: $workflow_file"
        return 1
    fi

    echo "📥 Importing: $workflow_name"

    # Use n8n CLI to import the workflow
    # Note: n8n import command requires the n8n instance to be running
    n8n import:workflow --input="$workflow_file" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✅ Successfully imported: $workflow_name"
    else
        echo "⚠️  Could not import via CLI. Please ensure n8n is running or import manually."
        echo "    Alternative: Use the n8n web UI at http://localhost:5678"
    fi
}

# Check if n8n is running
check_n8n_running() {
    if curl -s http://localhost:5678 > /dev/null 2>&1; then
        echo "✅ n8n is running at http://localhost:5678"
        return 0
    else
        echo "⚠️  n8n doesn't appear to be running"
        echo "    Start it with: N8N_SECURE_COOKIE=false n8n start"
        return 1
    fi
}

# Main execution
echo ""
echo "🔍 Checking n8n status..."
if ! check_n8n_running; then
    echo ""
    echo "Would you like to start n8n now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Starting n8n in background..."
        N8N_SECURE_COOKIE=false n8n start &>/dev/null &
        N8N_PID=$!
        echo "Waiting for n8n to start (PID: $N8N_PID)..."
        sleep 5

        if check_n8n_running; then
            echo "✅ n8n started successfully"
        else
            echo "❌ Failed to start n8n"
            exit 1
        fi
    else
        echo "Please start n8n manually and run this script again."
        exit 0
    fi
fi

echo ""
echo "📋 Available workflows:"
echo "1. vec2text_test_workflow.json - Basic testing workflow"
echo "2. webhook_api_workflow.json - API webhook endpoint"
echo ""

# Import workflows
echo "Importing workflows..."
echo "====================="

# Import each workflow
for workflow in vec2text_test_workflow.json webhook_api_workflow.json; do
    if [ -f "$workflow" ]; then
        import_workflow "$workflow"
    fi
done

echo ""
echo "🎉 Import process complete!"
echo ""
echo "📌 Next steps:"
echo "1. Open n8n at http://localhost:5678"
echo "2. Check your workflows in the UI"
echo "3. Activate the webhook workflow if needed"
echo "4. Test the workflows with your vec2text system"
echo ""
echo "📚 See README.md for detailed usage instructions"
# n8n Workflows for LNSP Phase 4

This directory contains n8n workflows designed to test and interact with the vec2text processing system.

## Available Workflows

### 1. Vec2Text Testing Workflow (`vec2text_test_workflow.json`)
A simple workflow that tests the vec2text pipeline with predefined test texts.

**Features:**
- Processes multiple test texts through the vec2text system
- Uses JXE and IELAB subscribers for decoding
- Parses and displays results in a readable format
- Handles errors gracefully

**How to use:**
1. Import into n8n
2. Execute manually to test the vec2text pipeline
3. Check the results in the Display Results node

### 2. Vec2Text API Webhook (`webhook_api_workflow.json`)
A webhook-triggered workflow that exposes the vec2text system as an API endpoint.

**Features:**
- POST endpoint at `/vec2text`
- Validates incoming requests
- Configurable parameters (subscribers, steps, backend, format)
- Returns JSON responses

**API Usage:**
```bash
curl -X POST http://localhost:5678/webhook/vec2text \
  -H "Content-Type: application/json" \
  -d '{
    "text": "What is the role of glucose in diabetes?",
    "subscribers": "jxe,ielab",
    "steps": 1,
    "backend": "isolated",
    "format": "json"
  }'
```

## Prerequisites

1. **n8n Installation:**
   ```bash
   # Install n8n globally
   npm install -g n8n

   # Or use npx
   npx n8n
   ```

2. **Start n8n:**
   ```bash
   # Start with secure cookie disabled (for local testing)
   N8N_SECURE_COOKIE=false n8n start
   ```

3. **Access n8n:**
   - Open browser at `http://localhost:5678`
   - Create account if first time

## Importing Workflows

### Method 1: CLI Import (Automated)

Use the provided import scripts:

```bash
# Using bash script
./n8n_workflows/import_workflows.sh

# Or using Node.js script (with REST API)
node n8n_workflows/cli_import.js

# With authentication (if configured)
N8N_API_KEY=your-api-key node n8n_workflows/cli_import.js
```

### Method 2: n8n CLI

```bash
# Import individual workflow
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json

# Import all workflows
n8n import:workflow --input=n8n_workflows/
```

### Method 3: Manual Import (Web UI)

1. Open n8n in your browser (`http://localhost:5678`)
2. Click on "Workflows" in the left sidebar
3. Click "Add workflow" → "Import from File"
4. Select the workflow JSON file from this directory
5. Click "Save" to save the workflow

## Activating Webhooks

For the webhook workflow:
1. Import the workflow
2. Open the workflow
3. Click "Execute Workflow" once to register the webhook
4. Toggle the "Active" switch to activate the workflow
5. The webhook URL will be displayed in the Webhook node

## Testing

### Test the basic workflow:
1. Import `vec2text_test_workflow.json`
2. Test with Python scripts:
   ```bash
   # Full test with detailed results
   python n8n_workflows/test_batch_workflow.py

   # Simple test
   python n8n_workflows/test_batch_simple.py
   ```

3. Or test manually in n8n UI:
   - Open the workflow
   - Click "Execute Workflow"
   - Check results in each node

### Test the webhook API:
1. Import and activate `webhook_api_workflow.json`
2. Test with Python scripts:
   ```bash
   # Full test suite
   python n8n_workflows/test_webhook.py

   # Simple test
   python n8n_workflows/test_webhook_simple.py

   # Custom text
   python n8n_workflows/test_webhook.py "Your custom text here"
   ```

3. Or test with curl:
   ```bash
   curl -X POST http://localhost:5678/webhook/vec2text \
     -H "Content-Type: application/json" \
     -d '{"text": "Neural networks process information"}'
   ```

## Customization

You can modify these workflows to:
- Add different subscribers (e.g., add `vmmoe` if available)
- Change the number of processing steps
- Add data persistence (save to database)
- Send results to other services (Slack, email, etc.)
- Create scheduled jobs for batch processing

## Troubleshooting

1. **Command execution fails:**
   - Ensure the venv is activated
   - Check the path to the project directory
   - Verify Python and dependencies are installed

2. **Webhook not responding:**
   - Ensure the workflow is activated
   - Check n8n is running
   - Verify the webhook URL is correct

3. **JSON parsing errors:**
   - Check the vec2text script output format
   - Adjust the JSON extraction regex if needed

## Notes

- The workflows use the CPU device for compatibility
- Processing is set to 1 step for speed (increase for quality)
- The isolated backend is used as per CLAUDE.md requirements
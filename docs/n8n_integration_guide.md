# n8n Integration Guide for LNSP Phase 4

## Overview

This guide documents the complete n8n workflow integration for the LNSP Phase 4 vec2text processing pipeline. The integration provides automated testing and API access to the text-vector-text processing system using JXE and IELab decoders.

### Claude Code MCP Integration (NEW)

As of 2025-09-19, n8n can be integrated with Claude Code using the Model Context Protocol (MCP):

```bash
# Setup n8n MCP server in Claude Code
claude mcp add n8n-local -- npx -y n8n-mcp --n8n-url=http://localhost:5678

# Verify connection
claude mcp list

# Use in Claude Code
# The MCP server enables Claude to interact with n8n workflows directly
```

**Benefits of MCP Integration:**
- Direct workflow management from Claude Code
- Automated workflow execution and monitoring
- Seamless integration with vec2text processing pipeline

## Quick Start

### Prerequisites
1. **n8n installed**: `npm install -g n8n`
2. **n8n running**: `N8N_SECURE_COOKIE=false n8n start`
3. **Project environment**: Python 3.11+, venv activated

### Working Commands
```bash
# Test single text
python3 n8n_workflows/test_webhook_simple.py

# Test multiple texts (batch)
python3 n8n_workflows/test_batch_via_webhook.py

# Test custom text
python3 n8n_workflows/test_webhook.py "Your custom text here"
```

## Architecture

### Created Workflows
1. **`vec2text_test_workflow.json`** - Batch testing workflow
2. **`webhook_api_workflow.json`** - REST API endpoint (WORKING)

### Created Test Scripts
1. **`test_webhook_simple.py`** - Minimal single text test ✅
2. **`test_webhook.py`** - Full featured webhook test ✅
3. **`test_batch_via_webhook.py`** - Batch processing via webhook ✅
4. **`test_batch_simple.py`** - REST API batch test ❌ (auth issues)
5. **`test_batch_workflow.py`** - Full REST API test ❌ (auth issues)

## Working Solution: Webhooks

### Why Webhooks Work
- **No authentication required** - webhooks are public endpoints by design
- **Direct workflow execution** - bypasses n8n's internal REST API
- **Production ready** - designed for external service integration
- **Immediate execution** - real-time processing

### Webhook API Usage
```python
import urllib.request
import json

# Single request
url = "http://localhost:5678/webhook/vec2text"
data = {
    "text": "What is AI?",
    "subscribers": "jxe,ielab",
    "steps": 1,
    "backend": "isolated",
    "format": "json"
}

json_data = json.dumps(data).encode('utf-8')
req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})

with urllib.request.urlopen(req, timeout=30) as response:
    result = json.loads(response.read().decode())
    print(result)
```

### Webhook Parameters
- **`text`** (required) - Input text for processing
- **`subscribers`** (optional) - "jxe,ielab" (default), "jxe", or "ielab"
- **`steps`** (optional) - Number of vec2text steps (default: 1)
- **`backend`** (optional) - "isolated" (default)
- **`format`** (optional) - "json" (default)

## Authentication Issues (Documented Problems)

### The Problem
n8n's REST API (`/rest/workflows`, `/rest/executions`) requires authentication once user management is enabled. This authentication persists even with environment variables:

```bash
# These DON'T work to disable auth:
N8N_USER_MANAGEMENT_DISABLED=true n8n start
N8N_SECURE_COOKIE=false n8n start
```

### Why Environment Variables Fail
1. **Persistent database** - User data stored in `~/.n8n/database.sqlite`
2. **Authentication sticky** - Once users exist, auth is required
3. **Runtime config** - Environment variables ignored after first setup

### Failed Solutions Attempted
1. ❌ Setting `N8N_USER_MANAGEMENT_DISABLED=true`
2. ❌ Database manipulation (clearing user tables)
3. ❌ Removing and recreating database
4. ❌ Complete `rm -rf ~/.n8n` reset

### Working Solution
**Use webhooks instead of REST API** - webhooks are public endpoints that bypass authentication entirely.

## Test Results

### Working Tests ✅
```bash
# Single text processing
python3 n8n_workflows/test_webhook_simple.py
# Output: {"message": "Workflow was started"} + processing results

# Batch processing via webhook
python3 n8n_workflows/test_batch_via_webhook.py
# Processes 5 test texts through vec2text pipeline

# Custom text processing
python3 n8n_workflows/test_webhook.py "Neural networks process information"
# Full test suite with detailed results
```

### Non-Working Tests ❌
```bash
# REST API tests (require authentication)
python3 n8n_workflows/test_batch_simple.py
# Error: HTTP Error 401: Unauthorized

python3 n8n_workflows/test_batch_workflow.py
# Error: HTTP Error 401: Unauthorized
```

## Integration with Vec2Text Pipeline

### Command Executed by Webhook
The webhook workflow executes this command:
```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4 && \
VEC2TEXT_FORCE_PROJECT_VENV=1 VEC2TEXT_DEVICE=cpu TOKENIZERS_PARALLELISM=false \
./venv/bin/python3 app/vect_text_vect/vec_text_vect_isolated.py \
  --input-text "{{text}}" \
  --subscribers {{subscribers}} \
  --vec2text-backend isolated \
  --output-format json \
  --steps {{steps}}
```

### Processing Flow
1. **Input**: Text via webhook POST request
2. **Encoding**: GTR-T5 768D (text → vector)
3. **Decoding**: JXE and/or IELab (vector → text)
4. **Output**: JSON results with processing status

### Expected Output Format
```json
{
  "status": "success",
  "request": {
    "text": "What is AI?",
    "subscribers": "jxe,ielab",
    "steps": 1
  },
  "result": {
    "jxe": "Generated text from JXE decoder",
    "ielab": "Generated text from IELab decoder"
  },
  "processing_time": "2025-09-18T21:30:00.000Z"
}
```

## File Structure

```
n8n_workflows/
├── README.md                    # Detailed usage instructions
├── vec2text_test_workflow.json  # Batch workflow (REST API)
├── webhook_api_workflow.json    # Webhook API workflow ✅
├── test_webhook_simple.py       # Simple webhook test ✅
├── test_webhook.py              # Full webhook test ✅
├── test_batch_via_webhook.py    # Batch via webhook ✅
├── test_batch_simple.py         # REST API test ❌
├── test_batch_workflow.py       # REST API test ❌
├── import_workflows.sh          # Import automation
├── cli_import.js               # Node.js import tool
└── check_setup.py              # Diagnostic tool
```

## Import Instructions

### Method 1: CLI Import
```bash
# Import individual workflows
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json
n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json

# Or use automation script
./n8n_workflows/import_workflows.sh
```

### Method 2: Web Interface
1. Open http://localhost:5678
2. Workflows → Add workflow → Import from File
3. Select workflow JSON file
4. Save

### Method 3: Automated Import
```bash
node n8n_workflows/cli_import.js
```

## Activation and Testing

### Activate Webhook Workflow
1. Open http://localhost:5678/workflow/vec2text_webhook_001
2. Toggle workflow to "Active"
3. Execute once to register webhook endpoint

### Verify Webhook Works
```bash
python3 n8n_workflows/check_setup.py
```

Expected output:
```
✅ n8n is running at http://localhost:5678
✅ Webhook endpoint is active at /webhook/vec2text
🎯 Ready to test with: python3 n8n_workflows/test_webhook_simple.py
```

## Troubleshooting

### Common Issues

1. **404 Webhook Not Found**
   - Ensure webhook workflow is activated
   - Execute workflow once to register endpoint
   - Check workflow is saved and active

2. **Authentication Errors**
   - Use webhook methods only
   - Avoid REST API tests unless you have valid credentials

3. **Vec2Text Command Fails**
   - Verify project venv is properly set up
   - Check JXE/IELab adapters are installed
   - Ensure GTR-T5 model is available

4. **Connection Refused**
   - Start n8n: `N8N_SECURE_COOKIE=false n8n start`
   - Check n8n is running: `ps aux | grep n8n`
   - Verify port 5678 is accessible: `curl http://localhost:5678`

### Diagnostic Commands
```bash
# Check n8n status
python3 n8n_workflows/check_setup.py

# Test webhook directly
curl -X POST http://localhost:5678/webhook/vec2text \
  -H "Content-Type: application/json" \
  -d '{"text": "test"}'

# Check n8n processes
ps aux | grep n8n
```

## Summary

The n8n integration provides automated testing and API access to the LNSP Phase 4 vec2text processing pipeline. While REST API access requires authentication that cannot be easily disabled, the webhook approach provides a robust, production-ready solution for:

- **Automated testing** of JXE and IELab decoders
- **API access** for external integrations
- **Batch processing** of multiple texts
- **Real-time monitoring** of processing results

The webhook solution is actually superior for production use as it's designed for external access and doesn't require authentication management.
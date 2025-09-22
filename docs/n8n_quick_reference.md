# n8n Quick Reference - LNSP Phase 4

## Claude Code MCP Integration

### Setup n8n MCP Server
```bash
# Add n8n MCP server to Claude Code
claude mcp add n8n-local -- npx -y n8n-mcp --n8n-url=http://localhost:5678

# Check MCP server status
claude mcp list

# Remove MCP server if needed
claude mcp remove n8n-local
```

### MCP Server Status
- ✅ **Configured**: n8n-local MCP server connected
- ✅ **Package**: Using `n8n-mcp` (v2.11.3)
- ✅ **Tested**: 2025-09-19

## Start n8n
```bash
N8N_SECURE_COOKIE=false n8n start
```

## Import Workflows
```bash
n8n import:workflow --input=n8n_workflows/webhook_api_workflow.json
n8n import:workflow --input=n8n_workflows/vec2text_test_workflow.json
```

## Activate Webhook
1. Open: http://localhost:5678/workflow/vec2text_webhook_001
2. Toggle to "Active"
3. Execute once to register

## Working Tests ✅

### Single Text Test
```bash
python3 n8n_workflows/test_webhook_simple.py
```

### Batch Test (5 texts)
```bash
python3 n8n_workflows/test_batch_via_webhook.py
```

### Custom Text
```bash
python3 n8n_workflows/test_webhook.py "Your custom text here"
```

### Direct API Call
```bash
curl -X POST http://localhost:5678/webhook/vec2text \
  -H "Content-Type: application/json" \
  -d '{"text": "What is AI?", "subscribers": "jxe,ielab", "steps": 1}'
```

## Non-Working Tests ❌
```bash
# These require authentication
python3 n8n_workflows/test_batch_simple.py
python3 n8n_workflows/test_batch_workflow.py
```

## Troubleshooting
```bash
# Check setup
python3 n8n_workflows/check_setup.py

# Verify n8n is running
ps aux | grep n8n

# Test basic connectivity
curl http://localhost:5678
```

## Key URLs
- **n8n Interface**: http://localhost:5678
- **Webhook Endpoint**: http://localhost:5678/webhook/vec2text
- **Workflow Editor**: http://localhost:5678/workflow/vec2text_webhook_001

## Parameters
- **text** (required): Input text to process
- **subscribers** (optional): "jxe,ielab" (default), "jxe", or "ielab"
- **steps** (optional): Number of vec2text steps (default: 1)
- **backend** (optional): "isolated" (default)
- **format** (optional): "json" (default)
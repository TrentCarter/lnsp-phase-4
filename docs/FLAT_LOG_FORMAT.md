# Flat Log Format Specification

## Overview

All parent-child communications, status updates, and commands in the PAS system are logged to flat `.txt` files for easy parsing and human readability.

## Log Format

```
<timestamp>|<from>|<to>|<type>|<message>|<llm_model>|<run_id>|<status>|<progress>|<metadata>
```

### Field Definitions

| Field | Type | Required | Description | Example |
|-------|------|----------|-------------|---------|
| `timestamp` | ISO8601 | ✅ | UTC timestamp with milliseconds | `2025-11-10T15:30:45.123Z` |
| `from` | string | ✅ | Source agent/service name | `Gateway`, `PAS Root`, `Aider-LCO` |
| `to` | string | ✅ | Destination agent/service name | `Mgr Backend`, `Dir Code` |
| `type` | enum | ✅ | Message type | `CMD`, `STATUS`, `HEARTBEAT`, `RESPONSE` |
| `message` | string | ✅ | Human-readable message | `Execute Prime Directive` |
| `llm_model` | string | ❌ | LLM model used (if applicable) | `ollama/qwen2.5-coder:7b-instruct` |
| `run_id` | UUID | ❌ | Run identifier | `abc123-def456` |
| `status` | enum | ❌ | Agent status | `running`, `completed`, `error` |
| `progress` | float | ❌ | Progress ratio (0.0-1.0) | `0.75` |
| `metadata` | JSON | ❌ | Additional key-value data (URL-encoded) | `{"tokens":1500,"cost_usd":0.02}` |

### Message Types

- **CMD**: Command from parent to child (e.g., "start task", "pause", "abort")
- **STATUS**: Status update from child to parent (e.g., "started", "progress 50%")
- **HEARTBEAT**: Periodic health check (e.g., "alive", token usage, resource metrics)
- **RESPONSE**: Response to a command (e.g., "acknowledged", "completed", "error")

### Special Values

- Empty fields: Use `-` for missing optional fields
- Pipe character in message: Escaped as `\|`
- Newline in message: Escaped as `\n`
- Metadata JSON: URL-encoded (no pipes or newlines)

## Examples

### 1. Prime Directive Submission
```
2025-11-10T15:30:45.123Z|Gateway|PAS Root|CMD|Submit Prime Directive: Add docstrings|-|abc123-def456|-|-|-
```

### 2. Task Start
```
2025-11-10T15:30:46.234Z|PAS Root|Aider-LCO|CMD|Execute: Add docstrings to services/gateway/app.py|ollama/qwen2.5-coder:7b-instruct|abc123-def456|queued|0.0|-
```

### 3. Heartbeat
```
2025-11-10T15:31:00.456Z|Aider-LCO|PAS Root|HEARTBEAT|Task running|ollama/qwen2.5-coder:7b-instruct|abc123-def456|running|0.35|{"tokens":1500,"ctx_limit":131072}
```

### 4. Status Update
```
2025-11-10T15:31:15.678Z|Aider-LCO|PAS Root|STATUS|Completed 3 of 5 files|ollama/qwen2.5-coder:7b-instruct|abc123-def456|running|0.60|{"files_done":3,"files_total":5}
```

### 5. Completion
```
2025-11-10T15:32:10.890Z|Aider-LCO|PAS Root|RESPONSE|Task completed successfully|ollama/qwen2.5-coder:7b-instruct|abc123-def456|completed|1.0|{"duration_s":85,"rc":0}
```

### 6. Multi-Agent Coordination
```
2025-11-10T15:30:50.123Z|Dir Code|Mgr Backend|CMD|Implement API endpoint /users|claude-3-7-sonnet|task456-backend|-|-|-
2025-11-10T15:30:51.234Z|Mgr Backend|Prog 005|CMD|Write handler function|ollama/qwen2.5-coder:7b-instruct|task456-backend|-|-|-
2025-11-10T15:31:20.456Z|Prog 005|Mgr Backend|STATUS|Handler complete, writing tests|ollama/qwen2.5-coder:7b-instruct|task456-backend|running|0.50|-
2025-11-10T15:31:45.678Z|Prog 005|Mgr Backend|RESPONSE|All tests passing|ollama/qwen2.5-coder:7b-instruct|task456-backend|completed|1.0|{"tests":5,"passed":5}
2025-11-10T15:31:46.789Z|Mgr Backend|Dir Code|RESPONSE|Endpoint complete|ollama/qwen2.5-coder:7b-instruct|task456-backend|completed|1.0|-
```

## Log File Location

**Production Logs:**
```
artifacts/logs/pas_comms_<date>.txt
```

Example: `artifacts/logs/pas_comms_2025-11-10.txt`

**Per-Run Logs:**
```
artifacts/runs/<run_id>/comms.txt
```

Example: `artifacts/runs/abc123-def456/comms.txt`

## Parsing Examples

### Shell (grep)
```bash
# All commands to Mgr Backend
grep "|Mgr Backend|CMD|" artifacts/logs/pas_comms_2025-11-10.txt

# All heartbeats
grep "|HEARTBEAT|" artifacts/logs/pas_comms_2025-11-10.txt

# All errors
grep "|error|" artifacts/logs/pas_comms_2025-11-10.txt

# All messages using Claude
grep "claude-3-7-sonnet" artifacts/logs/pas_comms_2025-11-10.txt
```

### Python
```python
import csv
from urllib.parse import unquote

with open('artifacts/logs/pas_comms_2025-11-10.txt') as f:
    reader = csv.reader(f, delimiter='|')
    for row in reader:
        timestamp, from_, to, type_, message, llm, run_id, status, progress, metadata = row
        if llm != '-':
            print(f"{timestamp}: {from_} → {to} using {llm}")
```

## Rotation & Retention

- **Daily rotation**: New file created at 00:00 UTC
- **Retention**: Keep 30 days (configurable)
- **Compression**: Gzip files older than 7 days
- **Archive location**: `artifacts/logs/archive/`

## Performance

- **Write mode**: Append-only (no locking needed)
- **Buffering**: 4KB line buffer, flush every 100 lines or 5 seconds
- **Max line size**: 64KB (truncate message if exceeded)
- **Max file size**: 100MB (rotate early if exceeded)

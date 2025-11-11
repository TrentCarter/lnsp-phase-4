# PAS Communication Logging System

**Status**: ✅ Production Ready (Nov 10, 2025)

Complete parent-child communication logging for the PAS (Project Agentic System) with flat `.txt` logs, LLM metadata tracking, and real-time parsing.

---

## 🎯 Features

- **Flat Log Format**: Pipe-delimited `.txt` files for easy parsing with `grep`, `awk`, or Python
- **LLM Tracking**: Every agent action includes which LLM model was used
- **Parent-Child Hierarchy**: Track commands flowing from Gateway → PAS Root → Aider-LCO
- **Per-Run Logs**: Separate log file for each run ID (in `artifacts/runs/<run-id>/comms.txt`)
- **Global Daily Logs**: All communications in `artifacts/logs/pas_comms_<date>.txt`
- **Real-Time Tailing**: Watch logs as they happen (`--tail` mode)
- **Multiple Output Formats**: Text (colored) or JSON
- **Advanced Filtering**: Filter by run ID, agent, LLM model, message type, status

---

## 📂 File Locations

### Global Logs (Daily Rotation)
```
artifacts/logs/pas_comms_2025-11-10.txt
artifacts/logs/pas_comms_2025-11-11.txt
...
```

### Per-Run Logs
```
artifacts/runs/<run-id>/comms.txt
artifacts/runs/abc123-def456/comms.txt
```

### Code Locations
- **Logger Module**: `services/common/comms_logger.py`
- **Parser Tool**: `tools/parse_comms_log.py`
- **Log Format Spec**: `docs/FLAT_LOG_FORMAT.md`
- **Test Script**: `tests/test_comms_logger.py`

---

## 📝 Log Format

```
timestamp|from|to|type|message|llm_model|run_id|status|progress|metadata
```

### Example Log Lines

```txt
2025-11-10T18:31:37.429Z|Gateway|PAS Root|CMD|Submit Prime Directive: Add docstrings|-|test-run-001|-|-|%7B%22files%22%3A%5B%22app.py%22%5D%7D
2025-11-10T18:31:37.430Z|PAS Root|Aider-LCO|CMD|Execute: Add docstrings to app.py|ollama/qwen2.5-coder:7b-instruct|test-run-001|queued|0.0|-
2025-11-10T18:31:45.123Z|Aider-LCO|PAS Root|HEARTBEAT|Processing file 3 of 5|ollama/qwen2.5-coder:7b-instruct|test-run-001|running|0.60|%7B%22files_done%22%3A3%7D
2025-11-10T18:32:10.456Z|Aider-LCO|PAS Root|RESPONSE|Execution completed successfully|ollama/qwen2.5-coder:7b-instruct|test-run-001|completed|1.0|%7B%22rc%22%3A0%7D
```

### Field Definitions

| Field | Required | Description | Example |
|-------|----------|-------------|---------|
| `timestamp` | ✅ | ISO8601 timestamp (UTC) | `2025-11-10T18:31:37.429Z` |
| `from` | ✅ | Source agent name | `Gateway`, `PAS Root`, `Aider-LCO` |
| `to` | ✅ | Destination agent name | `PAS Root`, `Aider-LCO`, `Mgr Backend` |
| `type` | ✅ | Message type | `CMD`, `STATUS`, `HEARTBEAT`, `RESPONSE` |
| `message` | ✅ | Human-readable message | `Submit Prime Directive: Add docstrings` |
| `llm_model` | ❌ | LLM model name | `ollama/qwen2.5-coder:7b-instruct` |
| `run_id` | ❌ | Run identifier (UUID) | `abc123-def456` |
| `status` | ❌ | Agent status | `queued`, `running`, `completed`, `error` |
| `progress` | ❌ | Progress (0.0-1.0) | `0.75` |
| `metadata` | ❌ | URL-encoded JSON | `%7B%22files%22%3A%5B%22app.py%22%5D%7D` |

---

## 🚀 Usage

### 1. Basic Viewing (Colored Output)

```bash
# View all logs for today
./tools/parse_comms_log.py

# View last 20 entries
./tools/parse_comms_log.py --limit 20

# View without colors
./tools/parse_comms_log.py --no-color
```

**Output:**
```
2025-11-10 18:31:37 CMD        Gateway         → PAS Root        Submit Prime Directive: Add docstrings
2025-11-10 18:31:37 STATUS     PAS Root        → Gateway         Started execution [running] [10%]
2025-11-10 18:31:37 HEARTBEAT  Aider-LCO       → PAS Root        Processing file 3 of 5 [running] [60%] [qwen2.5-coder]
2025-11-10 18:31:37 RESPONSE   Aider-LCO       → PAS Root        Execution completed successfully [completed] [qwen2.5-coder]
```

### 2. Filtering

```bash
# Filter by run ID
./tools/parse_comms_log.py --run-id abc123-def456

# Filter by message type
./tools/parse_comms_log.py --type CMD
./tools/parse_comms_log.py --type HEARTBEAT

# Filter by agent (from or to)
./tools/parse_comms_log.py --agent "Aider-LCO"
./tools/parse_comms_log.py --agent "Mgr Backend"

# Filter by LLM model (substring match)
./tools/parse_comms_log.py --llm claude
./tools/parse_comms_log.py --llm qwen

# Filter by status
./tools/parse_comms_log.py --status running
./tools/parse_comms_log.py --status error

# Combine filters
./tools/parse_comms_log.py --run-id abc123 --type STATUS --llm qwen
```

### 3. JSON Output

```bash
# Export to JSON
./tools/parse_comms_log.py --format json > logs.json

# Filter + JSON
./tools/parse_comms_log.py --run-id abc123 --format json | jq '.[] | {from, to, message}'
```

**Output:**
```json
[
  {
    "timestamp": "2025-11-10T18:31:37.430Z",
    "from": "Aider-LCO",
    "to": "PAS Root",
    "type": "RESPONSE",
    "message": "Execution completed successfully",
    "llm_model": "ollama/qwen2.5-coder:7b-instruct",
    "run_id": "test-run-001",
    "status": "completed",
    "progress": null,
    "metadata": {
      "duration_s": 42.5,
      "rc": 0
    }
  }
]
```

### 4. Real-Time Tailing (Like `tail -f`)

```bash
# Watch logs in real-time
./tools/parse_comms_log.py --tail

# Watch with filters
./tools/parse_comms_log.py --tail --type HEARTBEAT
./tools/parse_comms_log.py --tail --agent "Aider-LCO"
./tools/parse_comms_log.py --tail --llm claude
```

Press `Ctrl+C` to stop watching.

### 5. Shell Parsing (Advanced)

```bash
# All commands to Mgr Backend
grep "|Mgr Backend|CMD|" artifacts/logs/pas_comms_2025-11-10.txt

# All heartbeats
grep "|HEARTBEAT|" artifacts/logs/pas_comms_2025-11-10.txt

# All errors
grep "|error|" artifacts/logs/pas_comms_2025-11-10.txt

# All messages using Claude
grep "claude" artifacts/logs/pas_comms_2025-11-10.txt

# Count messages by type
awk -F'|' '{print $4}' artifacts/logs/pas_comms_2025-11-10.txt | sort | uniq -c
```

---

## 👨‍💻 Developer Usage

### Logging in Your Code

```python
from services.common.comms_logger import get_logger, MessageType

logger = get_logger()

# Log a command (capture log_id for parent-child tracking)
parent_log_id = logger.log_cmd(
    from_agent="Gateway",
    to_agent="PAS Root",
    message="Submit Prime Directive: Add docstrings",
    run_id="abc123-def456",
    metadata={"files": ["app.py", "utils.py"]}
)
# Returns: log_id (int) for use as parent_log_id in child logs

# Log a status update (link to parent command)
status_log_id = logger.log_status(
    from_agent="PAS Root",
    to_agent="Gateway",
    message="Started execution",
    run_id="abc123-def456",
    status="running",
    progress=0.1,
    parent_log_id=parent_log_id  # Link to parent command
)

# Log a delegation to child agent (link to status)
aider_cmd_log_id = logger.log_cmd(
    from_agent="PAS Root",
    to_agent="Aider-LCO",
    message="Execute Prime Directive",
    run_id="abc123-def456",
    parent_log_id=status_log_id  # Link to parent status
)

# Log a heartbeat with LLM info (link to command)
logger.log_heartbeat(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="Processing file 3 of 5",
    llm_model="ollama/qwen2.5-coder:7b-instruct",
    run_id="abc123-def456",
    status="running",
    progress=0.6,
    metadata={"files_done": 3, "files_total": 5},
    parent_log_id=aider_cmd_log_id  # Link to command that started this work
)

# Log a response (link to command)
logger.log_response(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="Execution completed successfully",
    llm_model="ollama/qwen2.5-coder:7b-instruct",
    run_id="abc123-def456",
    status="completed",
    metadata={"duration_s": 42.5, "rc": 0},
    parent_log_id=aider_cmd_log_id  # Link to command that this is responding to
)

# Generic log (full control)
logger.log(
    from_agent="Dir Code",
    to_agent="Mgr Backend",
    msg_type=MessageType.CMD,
    message="Implement API endpoint",
    llm_model="anthropic/claude-3-7-sonnet",
    run_id="xyz789",
    metadata={"lane": "backend"}
)
```

### Parent-Child Tracking (✅ Nov 10, 2025)

All logging methods return a `log_id` that can be used as `parent_log_id` for child logs.

**Example Flow:**
```python
# 1. Gateway submits Prime Directive (root message)
gateway_log_id = logger.log_cmd(
    from_agent="Gateway",
    to_agent="PAS Root",
    message="Submit Prime Directive: Add docstrings",
    run_id="abc123"
)

# 2. PAS Root acknowledges (child of gateway message)
status_log_id = logger.log_status(
    from_agent="PAS Root",
    to_agent="Gateway",
    message="Started execution",
    run_id="abc123",
    parent_log_id=gateway_log_id  # ← Links to parent
)

# 3. PAS Root delegates to Aider-LCO (child of status)
aider_cmd_log_id = logger.log_cmd(
    from_agent="PAS Root",
    to_agent="Aider-LCO",
    message="Execute Prime Directive",
    run_id="abc123",
    parent_log_id=status_log_id  # ← Links to parent
)

# 4. Aider-LCO responds (child of Aider command)
logger.log_response(
    from_agent="Aider-LCO",
    to_agent="PAS Root",
    message="Completed successfully",
    run_id="abc123",
    parent_log_id=aider_cmd_log_id  # ← Links to parent
)
```

**Resulting Hierarchy:**
```
log_id 2417: Gateway → PAS Root (parent_log_id=NULL)
├─ log_id 2418: PAS Root → Gateway "Queued" (parent_log_id=2417)
├─ log_id 2419: PAS Root → Gateway "Started" (parent_log_id=2417)
   └─ log_id 2420: PAS Root → Aider-LCO "Execute" (parent_log_id=2419)
      └─ log_id 2421: Aider-LCO → PAS Root "Completed" (parent_log_id=2420)
```

**Benefits:**
- ✅ Complete audit trail for every request
- ✅ Error traceability (trace errors back through call chain)
- ✅ Performance analysis (measure latency at each delegation level)
- ✅ Tree visualization in HMI

### Current Integration Points

The logger is integrated with full parent-child tracking in:

1. **Aider-LCO RPC** (`services/tools/aider_rpc/app.py`)
   - Captures `parent_log_id` from incoming requests
   - Links all responses and errors to parent command
   - Includes LLM model info (`llm_model`, `llm_provider`)

2. **PAS Root** (`services/pas/root/app.py`)
   - Captures log_id when Gateway submits Prime Directive
   - Passes parent_log_id to all child operations
   - Links all status updates and delegations to parent
   - Includes LLM model from Aider response

3. **Registry Service** (`services/registry/registry_service.py`)
   - Stores `parent_log_id` in `action_logs` table
   - Provides `/action_logs/task/{task_id}` endpoint for tree building

---

## 📊 Schema Updates

The following schemas have been updated to include LLM metadata:

### 1. Heartbeat Schema
**Files**: `contracts/heartbeat.schema.json`, `schemas/heartbeat.schema.json`

**New Fields:**
```json
{
  "llm_model": "ollama/qwen2.5-coder:7b-instruct",
  "llm_provider": "ollama",
  "parent_agent": "PAS Root",
  "children_agents": ["Mgr Backend", "Mgr Api"]
}
```

### 2. Status Update Schema
**Files**: `contracts/status_update.schema.json`, `schemas/status_update.schema.json`

**New Fields:**
```json
{
  "llm_model": "ollama/qwen2.5-coder:7b-instruct",
  "llm_provider": "ollama",
  "parent_agent": "PAS Root",
  "progress": 0.75
}
```

---

## 🧪 Testing

### Run Tests

```bash
# Test logger functionality
./.venv/bin/python tests/test_comms_logger.py

# Expected output:
# ✓ Logged CMD
# ✓ Logged STATUS
# ✓ Logged HEARTBEAT
# ✓ Logged RESPONSE
# ✓ Logged with LLM metadata
# ✅ All tests passed!
```

### Verify Logs

```bash
# Check global log
cat artifacts/logs/pas_comms_$(date +%Y-%m-%d).txt

# Check per-run log
cat artifacts/runs/test-run-001/comms.txt

# Parse with tool
./tools/parse_comms_log.py --run-id test-run-001
```

---

## 🔧 Configuration

### Log Rotation

- **Daily rotation**: New file created at 00:00 UTC
- **Retention**: Keep 30 days (configurable in `comms_logger.py`)
- **Compression**: Gzip files older than 7 days (future)
- **Max file size**: 100MB (rotate early if exceeded)

### Performance

- **Write mode**: Append-only (no file locking needed)
- **Buffering**: 4KB line buffer, flush every 100 lines or 5 seconds
- **Max line size**: 64KB (truncate message if exceeded)
- **Thread safety**: Safe for concurrent writes (append-only)

---

## 📖 Examples

### Example 1: Debug a Failing Run

```bash
# Find all logs for a specific run
./tools/parse_comms_log.py --run-id abc123-def456

# Find errors only
./tools/parse_comms_log.py --run-id abc123-def456 --status error

# Export to JSON for analysis
./tools/parse_comms_log.py --run-id abc123-def456 --format json > run_abc123.json
```

### Example 2: Monitor LLM Usage

```bash
# See which LLMs are being used
./tools/parse_comms_log.py --llm claude
./tools/parse_comms_log.py --llm qwen
./tools/parse_comms_log.py --llm gpt

# Watch live LLM activity
./tools/parse_comms_log.py --tail --llm claude
```

### Example 3: Track Agent Communication

```bash
# See all commands to Mgr Backend
./tools/parse_comms_log.py --agent "Mgr Backend" --type CMD

# See all heartbeats from Aider-LCO
./tools/parse_comms_log.py --agent "Aider-LCO" --type HEARTBEAT
```

### Example 4: Build HMI Visualization

```python
import json

# Load logs as JSON
logs = json.loads(subprocess.check_output([
    "./tools/parse_comms_log.py",
    "--run-id", run_id,
    "--format", "json"
]))

# Build timeline
for entry in logs:
    timestamp = entry["timestamp"]
    from_agent = entry["from"]
    to_agent = entry["to"]
    llm_model = entry["llm_model"]
    status = entry["status"]
    progress = entry["progress"]

    # Render in HMI sequencer (see PRD_Human_Machine_Interface_HMI.md)
    render_task_note(
        agent=from_agent,
        timestamp=timestamp,
        status=status,
        progress=progress,
        llm_model=llm_model
    )
```

---

## 🚦 Next Steps

### Immediate (Ready Now)
- ✅ Logging system production ready
- ✅ All schemas updated with LLM metadata
- ✅ Parser tool with filtering and real-time tailing
- ✅ Integrated in Aider-LCO and PAS Root

### Future Enhancements
- [ ] Integrate with HMI sequencer for visual timeline
- [ ] Add log compression for files older than 7 days
- [ ] Add metrics aggregation (avg duration, success rate)
- [ ] Add log search API endpoint (REST)
- [ ] Add WebSocket streaming for real-time HMI updates

---

## 📚 Related Documentation

- **Log Format Spec**: `docs/FLAT_LOG_FORMAT.md`
- **HMI PRD**: `docs/PRDs/PRD_Human_Machine_Interface_HMI.md`
- **P0 Integration**: `docs/P0_END_TO_END_INTEGRATION.md`
- **Schema Contracts**: `contracts/heartbeat.schema.json`, `contracts/status_update.schema.json`

---

**Last Updated**: 2025-11-10
**Status**: ✅ Production Ready

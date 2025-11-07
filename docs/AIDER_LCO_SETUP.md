# Aider-LCO (Local Code Operator) Setup Guide

**Status**: P0 Complete (2025-11-07)
**Version**: 0.1.0

This guide covers the complete setup and integration of Aider as the primary coding agent for PAS (Project Agentic System).

---

## Overview

Aider-LCO is a FastAPI-based RPC wrapper around the [Aider](https://github.com/paul-gauthier/aider) coding agent, providing:

- **JSON-RPC/HTTP API** for PAS integration
- **Command sandboxing** via allowlist and file ACLs
- **Secrets redaction** for logs and diffs
- **Cost/KPI tracking** with receipts for Token Governor
- **Registry integration** with automatic heartbeat
- **Git-native workflow** with auto-commit and multi-file editing

---

## Architecture

```
User/DirEng
    ↓
PAS Gateway (port 6120)
    ↓ (routes by role="execution", caps=["git-edit"])
Aider-LCO RPC (port 6150)
    ↓ (subprocess execution)
Aider CLI (installed globally)
    ↓ (connects to model provider)
Ollama/OpenAI/Anthropic (local or remote)
```

**Key Components**:
- `tools/aider_rpc/server.py` - Original scaffold (stub execution)
- `tools/aider_rpc/server_enhanced.py` - Full implementation with Aider execution
- `tools/aider_rpc/allowlist.py` - Command sandboxing
- `tools/aider_rpc/redact.py` - Secrets scrubbing
- `tools/aider_rpc/receipts.py` - Cost/KPI tracking
- `tools/aider_rpc/heartbeat.py` - Registry client
- `configs/pas/aider.yaml` - Configuration

---

## Installation

### Prerequisites

1. **Python 3.10-3.12** (Aider doesn't support 3.13 yet)
2. **Git** (for Aider's auto-commit features)
3. **Ollama** (for local models) or API keys (OpenAI/Anthropic)

### Step 1: Install Aider CLI

**Latest Version**: v0.86.1 (August 2025)
**Python Requirements**: 3.10-3.12 (3.13 not yet supported)

**Option A: Using pipx (recommended)**
```bash
# Install pipx if not already installed
brew install pipx
pipx ensurepath

# Install latest aider (uses separate venv, won't conflict with lnsp-phase-4)
pipx install aider-chat

# Verify installation (should show v0.86.1 or later)
aider --version
```

**Option B: Global pip install**
```bash
# Install aider globally
pip3 install aider-chat

# Verify installation
aider --version
```

**Why not in project venv?**
Aider requires Python 3.9-3.12, but this project uses Python 3.13. Installing globally via `pipx` creates an isolated environment with the correct Python version.

### Step 2: Setup Model Provider

**Option A: Ollama (Local, Free)**
```bash
# Install Ollama (if not already)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull recommended model (7B, fast, free)
ollama pull qwen2.5-coder:7b-instruct

# Alternative: DeepSeek R1 (best quality for local, 14B)
# ollama pull deepseek-r1:14b

# Start Ollama server
ollama serve &

# Verify
curl http://localhost:11434/api/tags
```

**Option B: OpenAI (Remote, Paid)**
```bash
export OPENAI_API_KEY="sk-..."
# Recommended models: gpt-4o, o1, o3-mini, gpt-4.1
```

**Option C: Anthropic (Remote, Paid)**
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Recommended models: claude-3-7-sonnet (best), claude-sonnet-4, claude-opus-4
```

**Option D: DeepSeek (Remote, Low Cost)**
```bash
export DEEPSEEK_API_KEY="sk-..."
# Recommended: deepseek-r1, deepseek-chat-v3 (best quality/cost ratio)
```

### Step 3: Install Python Dependencies (for RPC server)

```bash
cd /Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4

# Already in your venv
./.venv/bin/pip install fastapi uvicorn httpx pydantic
```

---

## Configuration

Edit `configs/pas/aider.yaml` to customize:

```yaml
# Network
network:
  port: 6150  # Default RPC port

# Model
model:
  primary: "ollama/qwen2.5-coder:7b-instruct"  # Change to your model

# Registry (optional)
registry:
  enabled: true
  url: "http://127.0.0.1:6121"
```

**Environment Variable Overrides** (highest priority):
```bash
export PAS_PORT=6150
export AIDER_MODEL="ollama/qwen2.5-coder:7b-instruct"
export PAS_REGISTRY_URL="http://127.0.0.1:6121"  # Optional
```

---

## Running the Server

### Basic Startup (Stub Mode)

```bash
# Start original server (stub execution, no actual Aider)
export PAS_PORT=6150
./.venv/bin/python tools/aider_rpc/server.py
```

### Full Startup (Enhanced Mode with Aider)

```bash
# Ensure Aider is installed
which aider  # Should print path

# Start enhanced server (actual Aider execution)
export PAS_PORT=6150
export AIDER_MODEL="ollama/qwen2.5-coder:7b-instruct"
export REPO_ROOT=$(pwd)
./.venv/bin/python tools/aider_rpc/server_enhanced.py
```

**With Registry Integration**:
```bash
export PAS_REGISTRY_URL="http://127.0.0.1:6121"
./.venv/bin/python tools/aider_rpc/server_enhanced.py
```

### Health Check

```bash
curl http://localhost:6150/health | jq
# {
#   "status": "ok",
#   "service_id": "...",
#   "name": "Aider-LCO",
#   "role": "execution",
#   "model": "ollama/qwen2.5-coder:7b-instruct"
# }
```

---

## Usage Examples

### Example 1: Code Refactoring

```bash
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "target": {
    "name": "Aider-LCO",
    "type": "agent",
    "role": "execution"
  },
  "payload": {
    "task": "code_refactor",
    "message": "Add type hints to all functions in src/utils.py",
    "files": ["src/utils.py"],
    "auto_commit": true
  },
  "policy": {
    "timeout_s": 120,
    "require_caps": ["git-edit"]
  },
  "run_id": "refactor-001"
}' | jq
```

**Response**:
```json
{
  "upstream": {
    "outputs": {
      "status": "ok",
      "output": "...",
      "log_file": "/tmp/aider-xxxxx.log",
      "receipt": {
        "run_id": "refactor-001",
        "usage": {"total_tokens": 6500},
        "cost": {"total_cost": 0.0},
        "kpis": {
          "files_changed": 1,
          "duration_seconds": 15.3
        }
      }
    }
  },
  "routing_receipt": {
    "run_id": "refactor-001",
    "status": "ok",
    "cost_estimate": {"usd": 0.0}
  }
}
```

### Example 2: Documentation Update

```bash
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "target": {"role": "execution"},
  "payload": {
    "message": "Convert all doc comments to active voice in docs/PRDs/",
    "files": ["docs/PRDs/*.md"]
  },
  "policy": {"timeout_s": 180},
  "run_id": "doc-update-001"
}' | jq '.routing_receipt.status'
```

### Example 3: Test Fix Loop

```bash
curl -s http://localhost:6150/invoke -H 'Content-Type: application/json' -d '{
  "payload": {
    "message": "Fix failing tests in tests/test_retrieval.py",
    "files": ["tests/test_retrieval.py", "src/retrieval.py"],
    "auto_commit": true
  },
  "policy": {"timeout_s": 300},
  "run_id": "test-fix-001"
}' | jq
```

---

## Receipt Storage

All execution receipts are stored in `artifacts/costs/`:

```bash
ls -lh artifacts/costs/
# refactor-001.json
# doc-update-001.json
# test-fix-001.json

cat artifacts/costs/refactor-001.json | jq '.usage'
# {
#   "input_tokens": 5000,
#   "output_tokens": 1500,
#   "total_tokens": 6500
# }
```

---

## Security Features (P0)

### 1. Command Sandboxing

**Allowlist** (safe commands only):
- `git`, `python`, `pytest`, `ls`, `cat`, `grep`, etc.
- **Blocked**: `rm -rf`, `sudo`, `curl | sh`, etc.

Test:
```bash
./.venv/bin/python tools/aider_rpc/allowlist.py
# ✓ ls -la                => True
# ✗ rm -rf /              => False (dangerous pattern)
```

### 2. Secrets Redaction

**Auto-redacts**:
- API keys (OpenAI, Anthropic, GitHub, etc.)
- JWTs, Bearer tokens
- Connection strings with passwords
- .env file contents

Test:
```bash
./.venv/bin/python tools/aider_rpc/redact.py
# ✓ OPENAI_API_KEY=sk-abc...  => OPENAI_API_KEY=[REDACTED]
```

### 3. File ACL Enforcement

**Allowed roots** (configurable in `aider.yaml`):
- `/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4`
- `/tmp`
- `/var/tmp`

**Blocked**:
- Writes to `.env`, `credentials.json`, etc.
- Access outside allowed roots

---

## Integration with PAS

### Gateway Routing

When PAS Gateway receives a job card with:
```json
{
  "role": "execution",
  "capabilities": ["git-edit"]
}
```

It routes to Aider-LCO (assuming it's registered).

### Token Governor Integration

Aider-LCO emits receipts with:
- Token usage (input/output/thinking)
- Cost estimates (USD)
- Duration (seconds)

Token Governor can:
- Enforce budgets per job
- Track cumulative spend
- Trigger save-state at 75% context usage

### PLMS Integration

Receipts include KPIs for lane-specific validation:
- `files_changed`
- `lines_added` / `lines_removed`
- `test_pass_rate`
- `lint_errors` / `type_errors`

---

## Troubleshooting

### Issue: "aider command not found"

**Solution**: Install aider via pipx (see Step 1 above).

```bash
pipx install aider-chat
aider --version
```

### Issue: "Module 'pkgutil' has no attribute 'ImpImporter'"

**Cause**: Aider doesn't support Python 3.13 yet.

**Solution**: Use pipx (installs in separate venv with correct Python version).

### Issue: Server fails to start with "/mnt/data" error

**Solution**: Fixed in current version. Use relative path:

```bash
export PAS_COST_DIR="artifacts/costs"
```

### Issue: Model connection fails

**For Ollama**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve &
```

**For OpenAI/Anthropic**:
```bash
# Verify API key is set
echo $OPENAI_API_KEY  # or $ANTHROPIC_API_KEY
```

---

## Next Steps (P1 - Tomorrow)

1. **Gateway Integration**
   - Register Aider-LCO with PAS Registry
   - Route job cards from Gateway to Aider-LCO
   - Test end-to-end flow

2. **HMI Integration**
   - Stream heartbeats to Tree + Sequencer visualization
   - Show context usage (node size = ctx load)
   - Display cost in real-time

3. **Multi-Agent Collaboration**
   - Claude Code → Gateway → Aider-LCO for code tasks
   - Gemini CLI → Gateway → Aider-LCO for doc tasks

---

## Next Steps (P2 - This Week)

1. **Hardening**
   - Signed audit logs (HMAC)
   - Air-gapped mode (disable external calls)
   - Rate limiting per service_id

2. **Advanced Features**
   - WebSocket transport (streaming updates)
   - Multi-file change planning
   - Diff preview before execution

---

## File Manifest

```
lnsp-phase-4/
├── tools/aider_rpc/
│   ├── __init__.py              # Module initialization
│   ├── server.py                # Original scaffold (stub)
│   ├── server_enhanced.py       # Full implementation ⭐
│   ├── allowlist.py             # Command sandboxing
│   ├── redact.py                # Secrets scrubbing
│   ├── receipts.py              # Cost/KPI tracking
│   └── heartbeat.py             # Registry client
├── configs/pas/
│   └── aider.yaml               # Configuration
├── artifacts/costs/             # Receipt storage (auto-created)
├── schemas/                     # JSON schemas (existing)
│   ├── job_card.schema.json
│   ├── routing_receipt.schema.json
│   └── ...
└── docs/
    ├── AIDER_LCO_SETUP.md       # This file ⭐
    └── research/
        ├── RESEARCH_Claude_Code_Like_Open_Source_Options.md
        └── AIDER_RPC_README.txt
```

---

## References

- **Aider GitHub**: https://github.com/paul-gauthier/aider
- **Aider Docs**: https://aider.chat/
- **PAS PRD**: `docs/PRDs/PRD_PAS_Project_Agentic_System.md`
- **LCO PRD**: `docs/PRDs/PRD_Local_Code_Operator_LCO.md`
- **Research**: `docs/research/RESEARCH_Claude_Code_Like_Open_Source_Options.md`

---

**Last Updated**: 2025-11-07
**Authors**: DirEng (Claude Code)
**Status**: ✅ P0 Complete - Ready for P1 integration

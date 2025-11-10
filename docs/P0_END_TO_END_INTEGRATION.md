# P0 End-to-End Integration: Prime Directive → Aider Execution

**Version**: 1.0
**Date**: 2025-11-10
**Status**: Implementation Ready
**Goal**: Wire complete flow from HMI/CLI → Gateway → PAS Root → Aider-LCO → Aider CLI

---

## 🎯 Overview

This document describes the **Production P0 scaffold** that connects all components in the Prime Directive execution pipeline:

```
┌─────────────────────────────────────────────────────────────────┐
│ User Interface Layer                                            │
├─────────────────────────────────────────────────────────────────┤
│ • HMI (Flask @ 6101) - Web UI with "Execute" button            │
│ • Verdict CLI (bin/verdict) - Command-line interface           │
└─────────────────────┬───────────────────────────────────────────┘
                      │
                      ▼ POST /prime_directives
         ┌────────────────────────────────────────┐
         │  Gateway (FastAPI @ 6120)              │
         │  - Single client entrypoint            │
         │  - Idempotency key handling            │
         │  - Future: receipts, approvals         │
         └────────────┬───────────────────────────┘
                      │
                      ▼ POST /pas/prime_directives
         ┌────────────────────────────────────────┐
         │  PAS Root (FastAPI @ 6100)             │
         │  - Orchestration layer (NO AI)         │
         │  - Background task execution           │
         │  - Run status tracking                 │
         └────────────┬───────────────────────────┘
                      │
                      ▼ POST /aider/edit
         ┌────────────────────────────────────────┐
         │  Aider-LCO RPC (FastAPI @ 6130)        │
         │  - Filesystem allowlist checking       │
         │  - Command allowlist checking          │
         │  - Secrets redaction                   │
         │  - Timeout enforcement                 │
         └────────────┬───────────────────────────┘
                      │
                      ▼ subprocess.run([aider, ...])
         ┌────────────────────────────────────────┐
         │  Aider CLI (pipx install)              │
         │  - AI-powered code editing             │
         │  - Multi-file support                  │
         │  - Git auto-commit                     │
         └────────────┬───────────────────────────┘
                      │
            ┌─────────┴─────────┐
            │                   │
            ▼                   ▼
    ┌──────────────┐    ┌──────────────┐
    │ Git Ops      │    │ File Ops     │
    │ - git add    │    │ - read       │
    │ - git commit │    │ - write      │
    │ - git diff   │    │ - patch      │
    └──────────────┘    └──────────────┘
```

---

## 📋 Component Responsibilities

### 1. Gateway (Port 6120)
- **Role**: Single entry point for all Prime Directive submissions
- **AI**: None (pure routing)
- **Key Features**:
  - Validates incoming requests
  - Forwards to PAS Root
  - Handles idempotency keys (future)
  - Attaches receipts/metrics (future)
- **File**: `services/gateway/app.py`

### 2. PAS Root (Port 6100)
- **Role**: Orchestration layer (no AI logic)
- **AI**: None (delegates to tools)
- **Key Features**:
  - Creates unique run IDs
  - Spawns background tasks
  - Tracks run status (in-memory for P0)
  - Calls Aider-LCO RPC with instructions
- **File**: `services/pas/root/app.py`

### 3. Aider-LCO RPC (Port 6130)
- **Role**: Guardrail wrapper around Aider CLI
- **AI**: None (calls Aider which uses AI)
- **Key Features**:
  - Filesystem allowlist enforcement
  - Command allowlist enforcement
  - Secrets redaction (env vars)
  - Timeout enforcement (default 900s)
  - Subprocess isolation
- **File**: `services/tools/aider_rpc/app.py`

### 4. Aider CLI (External Tool)
- **Role**: AI-powered code editor
- **AI**: Yes (Claude/GPT/Gemini/Llama via API)
- **Key Features**:
  - Multi-file editing
  - Context-aware code generation
  - Git integration (auto-commit)
  - Repo map for large codebases
- **Install**: `pipx install aider-chat`

---

## 🔒 Safety Layers

### Layer 1: Filesystem Allowlist (`configs/pas/fs_allowlist.yaml`)

**Purpose**: Prevent unauthorized file access

**Rules**:
- ✅ **Allowed**: Project workspace files (Python, docs, configs)
- ❌ **Denied**: System files (`/etc`, `~/.ssh`), secrets (`.env`, `.pem`, `.key`)

**Example**:
```yaml
roots:
  - "/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4"
allow:
  - "**/*.py"
  - "**/*.txt"
  - "docs/**/*.md"
  - "services/**/*.py"
deny:
  - "**/.ssh/**"
  - "/etc/**"
  - "**/*.key"
  - "**/*.pem"
  - "**/.env"
```

### Layer 2: Command Allowlist (`configs/pas/cmd_allowlist.yaml`)

**Purpose**: Prevent destructive commands

**Rules**:
- ✅ **Allowed**: Safe git operations, testing, linting
- ❌ **Denied**: Force push, destructive deletes, external network access

**Example**:
```yaml
allow:
  - "git status"
  - "git add *"
  - "git commit *"
  - "pytest *"
  - "ruff *"
deny:
  - "git push --force"
  - "rm -rf *"
  - "curl *metadata*"
  - "ssh *"
```

### Layer 3: Environment Isolation

**Purpose**: Prevent secret leakage

**Rules**:
- Only whitelist safe env vars (PATH, HOME, USER)
- Redact API keys from logs/output
- No full `os.environ` passthrough

### Layer 4: Timeout Enforcement

**Purpose**: Kill runaway processes

**Rules**:
- Default: 900s (15 minutes)
- Configurable per request
- Subprocess killed on timeout

---

## 🚀 Quick Start (30 seconds)

### 1. Install Dependencies
```bash
# Install Aider CLI (one-time)
pipx install aider-chat

# Verify installation
aider --version  # Should show v0.86.1+

# Set API key (for Claude/GPT models)
export ANTHROPIC_API_KEY=your_key_here  # or OPENAI_API_KEY
```

### 2. Start All Services
```bash
# Start the full stack (Gateway + PAS Root + Aider-LCO + HMI)
bash scripts/run_stack.sh
```

**Expected Output**:
```
[Aider-LCO] Started on http://127.0.0.1:6130
[PAS Root]  Started on http://127.0.0.1:6100
[Gateway]   Started on http://127.0.0.1:6120
[HMI]       Started on http://127.0.0.1:6101
```

### 3. Test via CLI
```bash
./bin/verdict send \
  --title "Add docstrings" \
  --description "Add docstrings to all functions" \
  --repo-root "/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4" \
  --goal "Add docstrings to all functions in services/gateway/app.py" \
  --entry-file "services/gateway/app.py"

# Expected output:
# {
#   "run_id": "abc123-uuid-here",
#   "status": "queued"
# }

# Check status
./bin/verdict status --run-id abc123-uuid-here
```

### 4. Test via HMI
1. Open browser: `http://localhost:6101`
2. Click "Execute Prime Directive" button
3. Fill in form:
   - **Title**: "Quick test"
   - **Goal**: "Add a TODO comment to services/gateway/app.py"
   - **Repo Root**: `/Users/trentcarter/Artificial_Intelligence/AI_Projects/lnsp-phase-4`
   - **Entry Files**: `services/gateway/app.py`
4. Click "Submit"
5. Check artifacts: `artifacts/runs/<uuid>/aider_stdout.txt`

---

## 📂 File Structure

```
lnsp-phase-4/
├── bin/
│   └── verdict                          # CLI entrypoint (Python script)
├── configs/
│   └── pas/
│       ├── fs_allowlist.yaml           # Filesystem allowlist
│       ├── cmd_allowlist.yaml          # Command allowlist
│       └── aider.yaml                  # Aider configuration
├── services/
│   ├── gateway/
│   │   └── app.py                      # Gateway (port 6120)
│   ├── pas/
│   │   └── root/
│   │       └── app.py                  # PAS Root (port 6100)
│   └── tools/
│       └── aider_rpc/
│           └── app.py                  # Aider-LCO RPC (port 6130)
├── scripts/
│   └── run_stack.sh                    # Start all services
└── artifacts/
    └── runs/
        └── <run_id>/
            └── aider_stdout.txt        # Aider output logs
```

---

## 🔄 Request Flow Example

### Example: "Add type hints to src/utils.py"

**Step 1: User submits via CLI**
```bash
./bin/verdict send \
  --title "Add type hints" \
  --goal "Add type hints to all functions in src/utils.py" \
  --entry-file "src/utils.py"
```

**Step 2: Verdict CLI → Gateway**
```http
POST http://127.0.0.1:6120/prime_directives
Content-Type: application/json

{
  "title": "Add type hints",
  "description": "CLI submission",
  "repo_root": "/Users/trentcarter/.../lnsp-phase-4",
  "goal": "Add type hints to all functions in src/utils.py",
  "entry_files": ["src/utils.py"]
}
```

**Step 3: Gateway → PAS Root**
```http
POST http://127.0.0.1:6100/pas/prime_directives
Content-Type: application/json

{
  "title": "Add type hints",
  "description": "CLI submission",
  "repo_root": "/Users/trentcarter/.../lnsp-phase-4",
  "goal": "Add type hints to all functions in src/utils.py",
  "entry_files": ["src/utils.py"]
}
```

**Step 4: PAS Root (background task) → Aider-LCO RPC**
```http
POST http://127.0.0.1:6130/aider/edit
Content-Type: application/json

{
  "message": "You are refactoring/creating files to satisfy: Add type hints to all functions in src/utils.py. Repo root is /Users/trentcarter/.../lnsp-phase-4. Respect code style and docs.",
  "files": ["/Users/trentcarter/.../lnsp-phase-4/src/utils.py"],
  "dry_run": false
}
```

**Step 5: Aider-LCO RPC checks allowlists**
- ✅ Filesystem: `src/utils.py` matches `**/*.py` pattern
- ✅ Command: (no commands yet, just file edits)
- ✅ Environment: Redact API keys, pass only safe vars

**Step 6: Aider-LCO RPC → Aider CLI**
```bash
aider --yes \
  --model claude-3-5-sonnet \
  /Users/trentcarter/.../lnsp-phase-4/src/utils.py \
  --message "You are refactoring/creating files to satisfy: Add type hints..."
```

**Step 7: Aider CLI execution**
- Reads `src/utils.py`
- Calls Claude API with prompt
- Applies edits to file
- Runs `git diff` to show changes
- Runs `git add src/utils.py && git commit -m "Add type hints to src/utils.py"`

**Step 8: Aider-LCO RPC returns**
```json
{
  "ok": true,
  "rc": 0,
  "stdout": "Applied edits to src/utils.py\nCommitted: Add type hints to src/utils.py",
  "duration_s": 12.5
}
```

**Step 9: PAS Root updates status**
```python
RUNS[run_id]["status"] = "completed"
RUNS[run_id]["message"] = "Prime Directive executed via Aider"
```

**Step 10: User checks status**
```bash
./bin/verdict status --run-id abc123-uuid-here

# Output:
# {
#   "run_id": "abc123-uuid-here",
#   "status": "completed",
#   "message": "Prime Directive executed via Aider"
# }
```

---

## 🧪 Testing Strategy

### 1. Unit Tests (Per Component)

**Gateway** (`tests/gateway/test_gateway.py`):
```python
def test_prime_directive_validation():
    """Test that gateway validates required fields"""
    response = client.post("/prime_directives", json={})
    assert response.status_code == 422  # Missing required fields

def test_prime_directive_forwarding():
    """Test that gateway forwards to PAS Root"""
    mock_pas_root.expect_post("/pas/prime_directives")
    response = client.post("/prime_directives", json={...})
    assert response.status_code == 200
```

**Aider-LCO RPC** (`tests/aider_rpc/test_allowlists.py`):
```python
def test_fs_allowlist_blocks_secrets():
    """Test that .env files are blocked"""
    response = client.post("/aider/edit", json={
        "files": [".env"],
        "message": "Add TODO"
    })
    assert response.status_code == 403
    assert "not allowed" in response.json()["detail"]

def test_cmd_allowlist_blocks_force_push():
    """Test that git push --force is blocked"""
    # (future: when command execution is added)
    pass
```

### 2. Integration Tests (End-to-End)

**Smoke Test** (`tests/integration/test_e2e_smoke.py`):
```python
@pytest.mark.integration
def test_e2e_prime_directive_execution():
    """Test full flow from Gateway to Aider CLI"""
    # 1. Submit Prime Directive via Gateway
    response = requests.post("http://localhost:6120/prime_directives", json={
        "title": "Test PD",
        "goal": "Add a TODO comment to test_file.py",
        "entry_files": ["test_file.py"]
    })
    run_id = response.json()["run_id"]

    # 2. Wait for completion (max 30s)
    for _ in range(30):
        status = requests.get(f"http://localhost:6120/runs/{run_id}").json()
        if status["status"] in ["completed", "error"]:
            break
        time.sleep(1)

    # 3. Verify execution
    assert status["status"] == "completed"
    assert "Aider" in status["message"]

    # 4. Verify artifacts
    assert Path(f"artifacts/runs/{run_id}/aider_stdout.txt").exists()
```

### 3. Manual Testing Checklist

- [ ] Gateway health check: `curl http://localhost:6120/health`
- [ ] PAS Root health check: `curl http://localhost:6100/health`
- [ ] Aider-LCO RPC health check: `curl http://localhost:6130/health`
- [ ] CLI submission: `./bin/verdict send --title "Test" --goal "..." --entry-file "..."`
- [ ] HMI submission: Click "Execute" button in browser
- [ ] Status polling: `./bin/verdict status --run-id <uuid>`
- [ ] Artifacts created: `ls artifacts/runs/<uuid>/`
- [ ] Git commits visible: `git log -1`

---

## 🚧 Known Limitations (P0)

### 1. In-Memory State
- **Issue**: PAS Root stores run status in `RUNS = {}` dict
- **Impact**: State lost on restart
- **Future**: Persist to SQLite (`artifacts/registry/registry.db`)

### 2. No Receipts/Cost Tracking
- **Issue**: No token usage or cost estimates
- **Impact**: No budget enforcement
- **Future**: Add `receipts.py` module, save to `artifacts/costs/{run_id}.json`

### 3. No Approvals
- **Issue**: All Prime Directives auto-execute
- **Impact**: No human review for risky operations
- **Future**: Add `/approve` endpoint in Gateway, require confirmation

### 4. No SSE/WebSocket
- **Issue**: HMI must poll for status updates
- **Impact**: Delayed UI feedback
- **Future**: Add Event Stream (port 6102), push status to Sequencer

### 5. No Multi-Tier AI (Architect/Director/Manager)
- **Issue**: PAS Root directly calls Aider-LCO RPC
- **Impact**: No task decomposition, single-shot execution
- **Future**: Add PAS Architect (Tier 1), Directors (Tier 2), Managers (Tier 3)

---

## 📈 Next Increments (Priority Order)

### Week 1: P0 Stabilization
- ✅ All services running (Gateway, PAS Root, Aider-LCO RPC)
- ✅ Verdict CLI working
- ✅ HMI button working
- ✅ Allowlists enforced
- ✅ Artifacts saved

### Week 2: Receipts & Metrics
- [ ] Add `receipts.py` module to Aider-LCO RPC
- [ ] Parse Aider output for token usage
- [ ] Save cost estimates to `artifacts/costs/{run_id}.json`
- [ ] Gateway attaches routing receipts
- [ ] HMI displays cost/duration

### Week 3: Registry & Persistence
- [ ] Registry API (port 6121) - SQLite backend
- [ ] PAS Root writes to Registry DB
- [ ] Aider-LCO RPC sends heartbeats
- [ ] Gateway stores idempotency keys
- [ ] Run status survives restarts

### Week 4: SSE & HMI Integration
- [ ] Event Stream (port 6102) - SSE broadcast
- [ ] PAS Root publishes status changes
- [ ] HMI Sequencer subscribes to SSE
- [ ] Tree view shows live progress
- [ ] Completion banner triggers instantly

### Week 5-8: Multi-Tier AI (PAS Full)
- [ ] PAS Architect (Tier 1) - Task decomposition
- [ ] PAS Directors (Tier 2) - Lane selection
- [ ] PAS Managers (Tier 3) - Task execution
- [ ] PAS Programmers (Tier 4) - Code editing
- [ ] Hierarchical receipts (bubble-up)

---

## 🎓 Key Concepts

### Prime Directive
A **user-facing project request** with:
- **Title**: Short name (e.g., "Add JWT auth")
- **Description**: Context for humans
- **Goal**: Natural language instruction for AI (e.g., "Implement JWT authentication with bcrypt hashing")
- **Repo Root**: Workspace path
- **Entry Files**: Starting point files for Aider

### Run ID
A **UUID** that tracks a Prime Directive execution from submission to completion:
- Created by PAS Root on `/pas/prime_directives`
- Used to query status via `/runs/{run_id}`
- Used to find artifacts in `artifacts/runs/{run_id}/`

### Allowlist
A **YAML configuration** that defines safe filesystem paths and commands:
- **Filesystem**: Whitelist workspace files, blacklist secrets
- **Commands**: Whitelist safe git ops, blacklist destructive commands
- Enforced by Aider-LCO RPC before subprocess execution

### Receipt
A **JSON artifact** that records execution metadata:
- **Routing Receipt**: Gateway-level (which service handled it)
- **Execution Receipt**: Aider-LCO level (tokens, cost, duration)
- **KPI Receipt**: Quality metrics (files changed, tests passed, etc.)
- Saved to `artifacts/costs/{run_id}/` for audit trail

---

## 🔗 Related Documents

- **Architecture Overview**: `docs/OPTIONS_SENDING_PRIME_DIRECTIVES.md`
- **Aider-LCO Setup**: `docs/AIDER_LCO_SETUP.md`
- **Aider-LCO Quickstart**: `docs/AIDER_LCO_QUICKSTART.md`
- **Security Review**: `docs/SECURITY_REVIEW_AIDER_LCO.md`
- **HMI PRD**: `docs/PRDs/PRD_Human_Machine_Interface_HMI.md`
- **PLMS PRD**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`
- **DirEng Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md`
- **PEX Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md`

---

## 🎯 Success Criteria (P0 Complete)

- [x] All 4 services running (Gateway, PAS Root, Aider-LCO RPC, HMI)
- [x] CLI can submit Prime Directives and check status
- [x] HMI button can trigger Prime Directives
- [x] Aider CLI executes with real AI model (Claude/GPT)
- [x] Filesystem allowlist blocks unauthorized paths
- [x] Command allowlist blocks destructive commands
- [x] Artifacts saved to `artifacts/runs/{run_id}/`
- [x] Git commits created by Aider

**When all boxes checked**: P0 is production-ready for local development use!

---

**Version**: 1.0
**Last Updated**: 2025-11-10
**Status**: ✅ Ready for Implementation

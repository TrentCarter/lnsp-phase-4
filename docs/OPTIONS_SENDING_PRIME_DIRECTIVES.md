# Options for Sending Prime Directives (Projects) to PAS Root

**Version**: 1.2
**Date**: 2025-11-10
**Status**: Ready for Implementation
**Context**: Two-tier AI architecture (DirEng → PEX → PAS)

**Changelog**:
- **v1.2** (2025-11-10): Added Q0 (AI tool access path via Verdict/Aider-LCO)
- **v1.1** (2025-11-09): Added FAQ (Q1-Q4), Complete Request Flow diagram, Quick Reference Table
- **v1.0** (2025-11-08): Initial version with 6 options

---

## 🎯 Overview

This document outlines **6 distinct options** for submitting Prime Directives (project requests) to the PAS Root orchestrator. Each option varies in complexity, user experience, and integration depth.

**Current Status**:
- ✅ **PAS Stub**: 12 endpoints operational (port 6200)
- ✅ **PLMS Tier 1**: Multi-run support, Bayesian calibration, risk visualization
- ✅ **DirEng + PEX Contracts**: Defined (Nov 7, 2025)
- ⏳ **Full PAS**: To be implemented (Weeks 5-8)

---

## ❓ FAQ: Architecture Questions

### Q0: What is the path for AI to get filesystem/tool access?

**Short Answer**: Yes, **Verdict = Aider-LCO** (Aider re-skinned with PAS guardrails)

**Complete Tool Access Path**:

```
┌─────────────────────────────────────────────────────────────────┐
│ OPTION 1 (Direct API): cURL → PAS ROOT                         │
│ OPTION 2 (Python): pas.submit() → PAS ROOT                     │
│ OPTION 3 (CLI): verdict invoke → PAS ROOT                      │
│ OPTION 4 (NL): DirEng (Claude Code) → PEX → PAS ROOT          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │        PAS ROOT (Tier 0)              │
         │  - No AI model (pure orchestration)   │
         │  - Spawns background worker           │
         │  - Routes to lane-specific AI         │
         └───────────────┬───────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │   PAS Architect/Director/Manager      │
         │   (Claude/GPT-4/GPT-3.5)              │
         │   - Decomposes tasks                  │
         │   - Selects lane (code/data/docs)     │
         │   - Chooses executor                  │
         └───────────────┬───────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────────┐
         │   VERDICT CLI (Aider-LCO Wrapper)     │
         │   Location: bin/verdict               │
         │   Backend: tools/verdict_cli.py       │
         │   Role: RPC client for Aider-LCO      │
         └───────────────┬───────────────────────┘
                         │
                         ▼ HTTP POST to http://127.0.0.1:6150/invoke
         ┌───────────────────────────────────────┐
         │   AIDER-LCO RPC SERVER                │
         │   Location: tools/aider_rpc/          │
         │   - server_enhanced.py (FastAPI)      │
         │   - allowlist.py (sandboxing)         │
         │   - redact.py (secrets scrubbing)     │
         │   - receipts.py (cost tracking)       │
         │   - heartbeat.py (registry client)    │
         └───────────────┬───────────────────────┘
                         │
                         ▼ subprocess.run([aider, ...])
         ┌───────────────────────────────────────┐
         │   ACTUAL AIDER CLI (v0.86.1+)         │
         │   Installed: pipx install aider-chat  │
         │   Model: ollama/qwen2.5-coder:7b      │
         │   Role: AI-powered code editor        │
         └───────────────┬───────────────────────┘
                         │
            ┌────────────┴────────────┐
            │                         │
            ▼                         ▼
    ┌──────────────┐        ┌──────────────────┐
    │ Git Commands │        │ File Operations  │
    │ - git add    │        │ - fs.read()      │
    │ - git commit │        │ - fs.write()     │
    │ - git diff   │        │ - fs.patch()     │
    └──────────────┘        └──────────────────┘
```

**Key Components**:

1. **Verdict** (`bin/verdict`):
   - CLI wrapper around Aider-LCO RPC server
   - Provides PAS-native interface (run_id, cost tracking, HMI integration)
   - Example: `verdict invoke --message "Add docstrings" --files src/api.py`

2. **Aider-LCO RPC Server** (`tools/aider_rpc/server_enhanced.py`):
   - FastAPI service (port 6150)
   - Wraps real Aider CLI with guardrails:
     - ✅ Command allowlist (`allowlist.py`)
     - ✅ Secrets redaction (`redact.py`)
     - ✅ Cost receipts (`receipts.py`)
     - ✅ Registry heartbeat (`heartbeat.py`)

3. **Real Aider CLI** (installed via `pipx install aider-chat`):
   - Version: v0.86.1+ (latest from August 2025)
   - Model: `ollama/qwen2.5-coder:7b-instruct` (default)
   - Capabilities: multi-file edit, git auto-commit, pytest loop, repo-map

**Safety Layers**:

| Layer | Enforced By | What It Does |
|-------|-------------|-------------|
| **Filesystem Allowlist** | `allowlist.py` | Only allows read/write to workspace paths (blocks `/etc`, `~/.ssh`, etc.) |
| **Command Allowlist** | `allowlist.py` | Only permits safe git commands (blocks `git push --force`, `rm -rf`, etc.) |
| **Secrets Redaction** | `redact.py` | Scrubs API keys, tokens from logs/diffs before storing receipts |
| **Environment Isolation** | `server_enhanced.py` | Only passes whitelisted env vars to Aider subprocess (PATH, HOME, model keys) |
| **Timeout Guard** | `execute_aider()` | Kills runaway processes after timeout (default 120s) |
| **Cost Tracking** | `receipts.py` | Logs token usage + cost estimates to `artifacts/costs/{run_id}.json` |

**"Aider Re-skinned Re-branded as Verdict"**: ✅ **CORRECT**

- **Verdict** is the PAS-facing CLI (brand name, user interface)
- **Aider-LCO** is the RPC server (guardrails, integration glue)
- **Aider CLI** is the actual tool (does the file editing)

**Example End-to-End Flow**:

```bash
# User (Option 3: CLI)
$ verdict invoke --message "Add type hints to src/utils.py" --files src/utils.py --run-id dev-001

# Verdict CLI (bin/verdict)
# → POST http://127.0.0.1:6150/invoke with payload

# Aider-LCO RPC Server (tools/aider_rpc/server_enhanced.py)
# → Validates file access: src/utils.py (✅ allowed)
# → Builds command: ["aider", "--model", "ollama/qwen2.5-coder:7b", "--message", "...", "src/utils.py"]
# → Executes: await asyncio.create_subprocess_exec(...)

# Real Aider CLI (from pipx)
# → Loads file src/utils.py
# → Calls Ollama API with code editing prompt
# → Applies edits to file
# → Runs git diff, git add, git commit (if auto_commit=true)
# → Returns output

# Aider-LCO RPC Server
# → Parses output for token usage
# → Redacts secrets from log
# → Saves receipt to artifacts/costs/dev-001.json
# → Returns response to Verdict CLI

# Verdict CLI
# → Displays result: "✓ Status: ok, Files changed: 1, Cost: $0.0032"
# → Optionally opens HMI dashboard (if --open-hmi flag)
```

**Configuration Files**:

- **Aider-LCO Config**: `configs/pas/aider.yaml`
- **Allowlists**: `services/pas/executors/allowlists/*.yaml` (future, currently inline)
- **Environment**: `PAS_PORT=6150`, `AIDER_MODEL=ollama/qwen2.5-coder:7b-instruct`

**Documentation**:

- **Setup Guide**: `docs/AIDER_LCO_SETUP.md` (complete installation + config)
- **Quickstart**: `docs/AIDER_LCO_QUICKSTART.md` (30-second test)
- **Security Review**: `docs/SECURITY_REVIEW_AIDER_LCO.md` (threat model + mitigations)

---

### Q1: What is the AI that processes the request?

**Short Answer**: It depends on which tier handles the request:
- **DirEng (Tier 1)**: Uses **Claude Sonnet 4.5** (you, the conversational assistant)
- **PEX (Tier 2)**: Uses **configurable AI provider** (defaults to Claude Sonnet 4.5, can use local Llama 3.1)
- **PAS Workers**: Use **provider matrix** (Anthropic, OpenAI, local LLMs) based on lane

**Detailed Answer**:

| Component | AI Model | Purpose | Who Chooses Model? |
|-----------|----------|---------|-------------------|
| **DirEng** | Claude Sonnet 4.5 | Human conversation, exploration | Fixed (you are DirEng) |
| **PEX** | Claude Sonnet 4.5 or Llama 3.1:8b | Project orchestration | User via `LNSP_LLM_MODEL` env var |
| **PAS Architect** | Claude Sonnet 4.5 | Task decomposition | Provider matrix in PLMS |
| **PAS Directors** | Claude Sonnet 4.5 or GPT-4 | Lane coordination | Provider matrix |
| **PAS Managers** | Claude Haiku or GPT-3.5 | Task execution | Provider matrix |
| **PAS Programmers** | Local Llama 3.1:8b | Code editing | Provider matrix (cost optimization) |

**Provider Matrix** (configured in PLMS per project):
```json
{
  "code": {
    "provider": "anthropic",
    "model": "claude-sonnet-4.5",
    "version": "20250929"
  },
  "narrative": {
    "provider": "openai",
    "model": "gpt-4-turbo"
  },
  "data-loader": {
    "provider": "local",
    "model": "llama3.1:8b",
    "endpoint": "http://localhost:11434"
  }
}
```

### Q2: Who selects which AI to use for the top level?

**Answer**: The **user** or **project creator** selects the provider matrix when creating a project in PLMS.

**Selection Flow**:
1. User submits Prime Directive (via any of the 6 options)
2. PLMS creates project entry in `projects` table
3. User (or defaults) defines **provider matrix** per lane
4. PEX reads provider matrix during task decomposition
5. PAS Architect assigns lanes → Directors → Managers → Workers
6. Each worker uses the AI specified in provider matrix for its lane

**Default Behavior** (if no provider matrix specified):
- **Tier 0 (PAS ROOT)**: No AI (orchestration logic only)
- **Tier 1 (Architect)**: Claude Sonnet 4.5
- **Tier 2 (Directors)**: Claude Sonnet 4.5
- **Tier 3 (Managers)**: Claude Haiku (cost optimization)
- **Tier 4 (Programmers)**: Local Llama 3.1:8b (free, runs on localhost)

**Override Example**:
```bash
# Use local Llama for all tiers (free, slower)
export LNSP_LLM_PROVIDER="local_llama"
export LNSP_LLM_MODEL="llama3.1:8b"
export LNSP_LLM_ENDPOINT="http://localhost:11434"

# Start project with this config
vp run --project-id 42 --run-kind baseline
```

### Q3: Is the top level called PAS ROOT?

**Yes!** The top level orchestrator is called **PAS ROOT** (or **PAS_ROOT** in code).

**Hierarchy** (from top to bottom):

```
Tier 0: PAS ROOT (orchestration logic, no AI model)
        ↓ Delegates to
Tier 1: PAS Architect (Claude Sonnet 4.5) - Task decomposition
        ↓ Creates lanes
Tier 2: PAS Directors (per lane: Code, Data, Narrative, Vector-Ops, Graph-Ops)
        ↓ Spawn managers
Tier 3: PAS Managers (per task cluster)
        ↓ Spawn workers
Tier 4: PAS Programmers/Workers (execute atomic tasks)
```

**PAS ROOT Responsibilities**:
- Receives Prime Directives (from cURL, VP CLI, DirEng, PEX, HMI)
- Validates budget caps and KPI gates
- Starts/pauses/resumes/terminates runs
- Emits completion signals to HMI
- Tracks portfolio status (all active runs)
- **Does NOT use an AI model** (pure orchestration)

**Evidence from Code**:
```python
# services/pas/stub/app.py:283
"from_agent": "PAS_ROOT",  # Identifies completion signal
```

```javascript
// services/webui/templates/sequencer.html:940
if (task.from_agent === 'PAS_ROOT' && task.action_type === 'directive_complete') {
    handleDirectiveComplete(task);
}
```

### Q4: Where are the system prompts for the agents stored?

**Answer**: System prompts are stored in **`docs/contracts/`** directory (Markdown files).

**Current Contracts**:
1. **DirEng Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md` (15KB, 400+ lines)
   - Role: Human-facing conversational assistant
   - Tools: fs, git, shell, rag
   - Tone: Conversational, proactive, honest

2. **PEX Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md` (8.6KB, 204 lines)
   - Role: Project orchestration layer
   - APIs: PLMS, PAS, FS, LightRAG
   - Constraints: Allowlists, budget caps, KPI gates

**Future Contracts** (to be implemented with full PAS):
- `docs/contracts/PAS_ARCHITECT_PROMPT.md` - Task decomposition
- `docs/contracts/PAS_DIRECTOR_CODE_PROMPT.md` - Code lane coordination
- `docs/contracts/PAS_DIRECTOR_DATA_PROMPT.md` - Data lane coordination
- `docs/contracts/PAS_DIRECTOR_NARRATIVE_PROMPT.md` - Documentation
- `docs/contracts/PAS_MANAGER_PROMPT.md` - Task cluster management
- `docs/contracts/PAS_PROGRAMMER_PROMPT.md` - Atomic code editing

**Contract Loading** (future implementation):
```python
# services/pas/agents/architect.py
import pathlib

PROMPT_PATH = pathlib.Path(__file__).parent.parent.parent / "docs/contracts/PAS_ARCHITECT_PROMPT.md"

def load_system_prompt() -> str:
    """Load Architect system prompt from contract file."""
    return PROMPT_PATH.read_text()

# Use in LLM call
response = llm.chat(
    messages=[
        {"role": "system", "content": load_system_prompt()},
        {"role": "user", "content": user_message}
    ]
)
```

**Contract Versioning**:
Each contract includes version string at top:
```markdown
**Contract Version:** `2025-11-07-001`
```

This allows replay passports to lock to specific contract versions for deterministic replay.

---

## 🔄 Complete Request Flow (ASCII Diagram)

This diagram shows the **complete path** from user cURL command all the way down to the Programmer, then back up the hierarchy to PAS ROOT, and finally notification to the user.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         OPTION 1: cURL Command                          │
│                                                                         │
│  $ curl -X POST http://localhost:6200/pas/v1/runs/start \              │
│      -H "Content-Type: application/json" \                             │
│      -H "Idempotency-Key: $(uuidgen)" \                                │
│      -d '{"project_id": 42, "run_id": "run-001", ...}'                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP POST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PAS ROOT (Tier 0)                               │
│                      Port 6200 (FastAPI Stub)                           │
│                                                                         │
│  Responsibilities:                                                      │
│  • Validate idempotency key                                            │
│  • Create run entry in RUNS dict                                       │
│  • Start background worker: _execute_run(run_id)                       │
│  • Return: {"status": "executing", "run_id": "run-001"}                │
│                                                                         │
│  AI Model: NONE (pure orchestration logic)                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Spawns background thread
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      Background Worker Thread                           │
│                      _execute_run(run_id)                               │
│                                                                         │
│  Loop through tasks in DAG[run_id]:                                     │
│    1. Check dependencies (topological order)                            │
│    2. _execute_task_synthetic(task_id, lane, payload)                  │
│    3. Store receipt + KPIs                                              │
│    4. Update task status                                                │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ For each task
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    PAS Architect (Tier 1)                               │
│                    [STUB: Skipped in current impl]                      │
│                                                                         │
│  Responsibilities (future):                                             │
│  • Decompose Prime Directive → task tree                                │
│  • Assign lanes (Code-Impl, Data-Schema, etc.)                         │
│  • Estimate tokens/cost per task                                       │
│  • Create job cards                                                     │
│                                                                         │
│  AI Model: Claude Sonnet 4.5                                            │
│  System Prompt: docs/contracts/PAS_ARCHITECT_PROMPT.md (future)        │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Delegates to lane Director
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PAS Director - Code Lane (Tier 2)                     │
│                   [STUB: Skipped in current impl]                       │
│                                                                         │
│  Responsibilities (future):                                             │
│  • Coordinate all Code-Impl tasks                                       │
│  • Spawn Managers for task clusters                                     │
│  • Track lane budget (e.g., $5.00 cap)                                 │
│  • Report to Architect                                                  │
│                                                                         │
│  AI Model: Claude Sonnet 4.5 or GPT-4                                   │
│  System Prompt: docs/contracts/PAS_DIRECTOR_CODE_PROMPT.md (future)    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Spawns Manager
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PAS Manager - Backend Cluster (Tier 3)                │
│                   [STUB: Skipped in current impl]                       │
│                                                                         │
│  Responsibilities (future):                                             │
│  • Manage task cluster (e.g., "Implement JWT auth")                    │
│  • Spawn Programmers for atomic edits                                   │
│  • Validate outputs (linting, tests)                                    │
│  • Report to Director                                                   │
│                                                                         │
│  AI Model: Claude Haiku (cost optimization)                             │
│  System Prompt: docs/contracts/PAS_MANAGER_PROMPT.md (future)          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Spawns Programmer
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   PAS Programmer (Tier 4)                               │
│                   [STUB: _execute_task_synthetic()]                     │
│                                                                         │
│  Current Implementation (Stub):                                         │
│  • Sleep for random duration (5-15 seconds)                             │
│  • Emit synthetic receipt:                                              │
│    - tokens_in: 500-2000                                                │
│    - tokens_out: 200-800                                                │
│    - cost_usd: $0.05-$0.30                                              │
│    - echo_cos: 0.80-0.95                                                │
│    - status: "succeeded" (90%) or "failed" (10%)                        │
│  • Emit synthetic KPIs (lane-specific)                                  │
│                                                                         │
│  Future Implementation:                                                 │
│  • Read files from fs                                                   │
│  • Apply unified diff patch                                             │
│  • Run tests, linters                                                   │
│  • Commit changes (if auto-commit enabled)                              │
│                                                                         │
│  AI Model: Local Llama 3.1:8b (free, runs on localhost)                │
│  System Prompt: docs/contracts/PAS_PROGRAMMER_PROMPT.md (future)       │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Emits receipt
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Task Receipt Storage                            │
│                                                                         │
│  Stub: RECEIPTS list (in-memory)                                        │
│  Stub: KPI_RECEIPTS list (in-memory)                                    │
│                                                                         │
│  Future: Registry DB (artifacts/registry/registry.db)                   │
│    • action_logs table (task actions)                                   │
│    • task_receipts table (execution metrics)                            │
│    • kpi_receipts table (quality validation)                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ After all tasks complete
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│               Completion Bubble-Up (Child → Parent)                     │
│                                                                         │
│  Tier 4 (Programmer) → Tier 3 (Manager):                                │
│    "Task completed, test_pass_rate=0.95"                                │
│                                                                         │
│  Tier 3 (Manager) → Tier 2 (Director):                                  │
│    "Cluster completed, 8/8 tasks succeeded"                             │
│                                                                         │
│  Tier 2 (Director) → Tier 1 (Architect):                                │
│    "Code lane completed, $4.65 spent"                                   │
│                                                                         │
│  Tier 1 (Architect) → Tier 0 (PAS ROOT):                                │
│    "All lanes completed, validation_pass=true"                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Final completion
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     PAS ROOT (Tier 0) - Completion                      │
│                                                                         │
│  _notify_directive_complete():                                          │
│  • Calculate final metrics (duration, cost, success rate)               │
│  • Create completion action_log entry:                                  │
│    {                                                                    │
│      "task_id": "run-001",                                              │
│      "from_agent": "PAS_ROOT",                                          │
│      "to_agent": "HMI",                                                 │
│      "action_type": "directive_complete",                               │
│      "action_data": {                                                   │
│        "tasks_total": 8,                                                │
│        "tasks_succeeded": 8,                                            │
│        "duration_seconds": 127.5,                                       │
│        "validation_pass": true                                          │
│      }                                                                  │
│    }                                                                    │
│  • POST to Registry: http://localhost:6121/action_logs                 │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP POST
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Registry Service (Port 6121)                         │
│                    SQLite DB: artifacts/registry/registry.db            │
│                                                                         │
│  • Inserts completion entry into action_logs table                      │
│  • Triggers SSE broadcast to all connected clients                      │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ SSE push
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    HMI Service (Port 6101)                              │
│                    FastAPI: services/webui/hmi_app.py                   │
│                                                                         │
│  Background Thread: poll_action_logs() [every 1 second]                 │
│  • Detects new completion entry                                         │
│  • Pushes to SSE stream: /stream_action_logs                            │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ SSE stream
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Browser (Sequencer UI)                               │
│                    http://localhost:6101/sequencer                      │
│                                                                         │
│  JavaScript: sequencer.html                                             │
│  • Listens to SSE stream                                                │
│  • Detects: task.from_agent === 'PAS_ROOT' &&                          │
│              task.action_type === 'directive_complete'                  │
│  • Calls: handleDirectiveComplete(task)                                 │
│                                                                         │
│  handleDirectiveComplete():                                             │
│  1. Stop timeline auto-scroll (isPlaying = false)                       │
│  2. Show "END OF PROJECT" banner                                        │
│  3. Stop polling (clearInterval)                                        │
│  4. Scroll to end of timeline                                           │
│  5. Display final report (tasks, cost, duration, KPIs)                  │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ Browser alert / visual feedback
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER (You)                                      │
│                                                                         │
│  Sees:                                                                  │
│  • "END OF PROJECT" banner in Sequencer                                 │
│  • Final report card:                                                   │
│    - ✅ 8/8 tasks succeeded                                             │
│    - Duration: 2m 7s                                                    │
│    - Cost: $4.65 / $5.00 budget                                         │
│    - KPIs: All passed ✓                                                 │
│  • Timeline frozen at end                                               │
│  • Option to download receipts, view logs, replay run                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Observations**:
1. **PAS ROOT does NOT use an AI model** (pure orchestration)
2. **Each tier reports back to parent** (child → parent bubble-up)
3. **Completion signal is special action_log entry** (from_agent="PAS_ROOT")
4. **HMI listens via SSE stream** (no polling delay, instant push)
5. **Current stub skips Tier 1-3** (directly executes tasks synthetically)

---

## Option 1: Direct HTTP API (Raw cURL)

### Description
Send JSON payload directly to PAS REST API using `curl` or similar HTTP client.

### Usage
```bash
# Start a new project run
curl -X POST http://localhost:6200/pas/v1/runs/start \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{
    "project_id": 42,
    "run_id": "run-baseline-001",
    "run_kind": "baseline",
    "rehearsal_pct": 0.0,
    "budget_caps": {
      "Code-Impl": 5.0,
      "Narrative": 2.0
    }
  }'

# Submit job cards (tasks)
curl -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(uuidgen)" \
  -d '{
    "project_id": 42,
    "run_id": "run-baseline-001",
    "lane": "Code-Impl",
    "priority": 0.8,
    "deps": [],
    "payload": {
      "instruction": "Implement user authentication with JWT",
      "files": ["src/auth.py", "src/api/routes.py"]
    },
    "budget_usd": 5.0,
    "ci_width_hint": 0.3
  }'

# Check run status
curl "http://localhost:6200/pas/v1/runs/status?run_id=run-baseline-001"

# Monitor portfolio
curl http://localhost:6200/pas/v1/portfolio/status
```

### Pros
- ✅ **Simplest implementation** (already works with PAS Stub)
- ✅ **No dependencies** (just HTTP client)
- ✅ **Direct control** over all API parameters
- ✅ **Script-friendly** for automation

### Cons
- ❌ **No abstraction** (must construct JSON manually)
- ❌ **No validation** (typos cause runtime errors)
- ❌ **No user-friendly prompts** (requires API knowledge)
- ❌ **No DirEng/PEX integration** (bypasses AI layer)

### Best For
- **Power users** who want direct API control
- **CI/CD pipelines** and automation scripts
- **Debugging** and testing PAS endpoints
- **Quick prototyping** without tooling overhead

---

## Option 2: Python Client Library (`pas_client.py`)

### Description
Thin Python wrapper around PAS HTTP API with type hints and validation.

### Usage
```python
from services.pas.client import PASClient

# Initialize client
pas = PASClient(base_url="http://localhost:6200")

# Start run (auto-generates idempotency key)
run = pas.start_run(
    project_id=42,
    run_id="run-baseline-001",
    run_kind="baseline",
    budget_caps={"Code-Impl": 5.0, "Narrative": 2.0}
)
print(f"Run started: {run.run_id}, Status: {run.status}")

# Submit job card
task = pas.submit_jobcard(
    project_id=42,
    run_id="run-baseline-001",
    lane="Code-Impl",
    priority=0.8,
    payload={
        "instruction": "Implement user authentication with JWT",
        "files": ["src/auth.py", "src/api/routes.py"]
    },
    budget_usd=5.0
)
print(f"Task submitted: {task.task_id}")

# Poll for completion
while True:
    status = pas.run_status("run-baseline-001")
    print(f"Progress: {status.tasks_completed}/{status.tasks_total}")
    if status.status in ["completed", "needs_review", "terminated"]:
        break
    time.sleep(5)

print(f"Final status: {status.status}")
print(f"Cost: ${status.spend_usd:.2f}")
print(f"KPI violations: {status.kpi_violations}")
```

### Implementation
```python
# services/pas/client.py
import requests
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel

class RunStatus(BaseModel):
    run_id: str
    status: str
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    spend_usd: float
    runway_minutes: int
    kpi_violations: List[Dict]

class PASClient:
    def __init__(self, base_url: str = "http://localhost:6200"):
        self.base_url = base_url
        self.session = requests.Session()

    def start_run(self, project_id: int, run_id: str,
                  run_kind: str = "baseline",
                  rehearsal_pct: float = 0.0,
                  budget_caps: Optional[Dict[str, float]] = None) -> Dict:
        """Start a new run."""
        resp = self.session.post(
            f"{self.base_url}/pas/v1/runs/start",
            json={
                "project_id": project_id,
                "run_id": run_id,
                "run_kind": run_kind,
                "rehearsal_pct": rehearsal_pct,
                "budget_caps": budget_caps or {}
            }
        )
        resp.raise_for_status()
        return resp.json()

    def submit_jobcard(self, project_id: int, run_id: str,
                       lane: str, priority: float,
                       payload: Dict, budget_usd: float,
                       deps: Optional[List[str]] = None) -> Dict:
        """Submit a job card (task)."""
        resp = self.session.post(
            f"{self.base_url}/pas/v1/jobcards",
            headers={"Idempotency-Key": str(uuid.uuid4())},
            json={
                "project_id": project_id,
                "run_id": run_id,
                "lane": lane,
                "priority": priority,
                "deps": deps or [],
                "payload": payload,
                "budget_usd": budget_usd,
                "ci_width_hint": 0.3
            }
        )
        resp.raise_for_status()
        return resp.json()

    def run_status(self, run_id: str) -> RunStatus:
        """Get run status."""
        resp = self.session.get(
            f"{self.base_url}/pas/v1/runs/status",
            params={"run_id": run_id}
        )
        resp.raise_for_status()
        return RunStatus(**resp.json())
```

### Pros
- ✅ **Type safety** (Pydantic validation)
- ✅ **Easy to use** (Pythonic interface)
- ✅ **Auto-handles** idempotency keys
- ✅ **Good for notebooks** and scripting

### Cons
- ❌ **Still requires** API knowledge
- ❌ **No natural language** interface
- ❌ **No DirEng integration** (manual orchestration)

### Best For
- **Python scripts** and notebooks
- **Automated workflows** with validation
- **Library consumers** who want type hints

---

## Option 3: CLI Tool (`vp` - Verdict Pro)

### Description
Command-line interface that mimics `verdict` but targets PAS directly (not Aider RPC).

### Usage
```bash
# Initialize new project
vp new --prd "Build user authentication system" \
       --budget 50 \
       --output artifacts/projects/auth.json

# Start baseline run
vp run --project-id 42 \
       --run-kind baseline \
       --budget-caps "Code-Impl=5.0,Narrative=2.0"

# Submit task
vp task --run-id run-baseline-001 \
        --lane Code-Impl \
        --instruction "Implement JWT auth" \
        --files src/auth.py src/api/routes.py \
        --budget 3.0

# Check status
vp status --run-id run-baseline-001

# Simulate rehearsal (1% canary)
vp simulate --run-id run-baseline-001 --pct 0.01

# View receipts
vp receipts --run-id run-baseline-001 --format json

# Open HMI dashboard
vp dashboard --run-id run-baseline-001
```

### Implementation
```python
#!/usr/bin/env python3
"""vp: Verdict Pro - CLI for PAS Prime Directives"""
import argparse
import json
import sys
from services.pas.client import PASClient

def cmd_new(args):
    """Create new project in PLMS."""
    # Call PLMS API to create project
    print(f"✓ Project created: ID={args.project_id}")
    return 0

def cmd_run(args):
    """Start a run."""
    pas = PASClient()
    result = pas.start_run(
        project_id=args.project_id,
        run_id=f"run-{args.run_kind}-{args.project_id:03d}",
        run_kind=args.run_kind,
        budget_caps=_parse_budget_caps(args.budget_caps)
    )
    print(f"✓ Run started: {result['run_id']}")
    return 0

def cmd_task(args):
    """Submit task (job card)."""
    pas = PASClient()
    result = pas.submit_jobcard(
        project_id=args.project_id,
        run_id=args.run_id,
        lane=args.lane,
        priority=0.8,
        payload={
            "instruction": args.instruction,
            "files": args.files
        },
        budget_usd=args.budget
    )
    print(f"✓ Task submitted: {result['task_id']}")
    return 0

def cmd_status(args):
    """Check run status."""
    pas = PASClient()
    status = pas.run_status(args.run_id)
    print(f"Status: {status.status}")
    print(f"Progress: {status.tasks_completed}/{status.tasks_total}")
    print(f"Cost: ${status.spend_usd:.2f}")
    print(f"Runway: {status.runway_minutes} min")
    if status.kpi_violations:
        print(f"⚠️  KPI violations: {len(status.kpi_violations)}")
    return 0

def main():
    parser = argparse.ArgumentParser(prog="vp")
    subparsers = parser.add_subparsers(dest="command")

    # vp new
    p_new = subparsers.add_parser("new")
    p_new.add_argument("--prd", required=True)
    p_new.add_argument("--budget", type=float, default=50.0)
    p_new.set_defaults(func=cmd_new)

    # vp run
    p_run = subparsers.add_parser("run")
    p_run.add_argument("--project-id", type=int, required=True)
    p_run.add_argument("--run-kind", default="baseline")
    p_run.add_argument("--budget-caps")
    p_run.set_defaults(func=cmd_run)

    # ... (other subcommands)

    args = parser.parse_args()
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
```

### Pros
- ✅ **User-friendly** CLI interface
- ✅ **Quick commands** for common operations
- ✅ **HMI integration** (open dashboard)
- ✅ **Shell-scriptable** for automation

### Cons
- ❌ **Still requires** understanding of lanes/job cards
- ❌ **No natural language** (structured commands)
- ❌ **Medium complexity** (CLI parser + client)

### Best For
- **Terminal users** who want fast workflows
- **DevOps scripts** that need simple CLI
- **Power users** comfortable with structured commands

---

## Option 4: DirEng Natural Language Interface (Conversational)

### Description
Talk to DirEng (you!) in natural language, who translates intent to PAS job cards.

### Usage
```
You: "Hey DirEng, I need to implement user authentication with JWT tokens.
      Use src/auth.py and src/api/routes.py. Budget is $5."

DirEng: I'll submit this to PAS as a Code-Impl task. Let me break it down:

        Project: User Authentication (new project ID: 42)
        Task: Implement JWT authentication
        Lane: Code-Impl
        Files: src/auth.py, src/api/routes.py
        Budget: $5.00

        Should I start a baseline run or a rehearsal first? (1% rehearsal = $0.05)

You: "Do rehearsal first"

DirEng: ✓ Starting 1% rehearsal run...
        ⏳ Executing canary task (1% of files)...
        ✓ Rehearsal completed in 12s

        Projected metrics (extrapolated):
        - Tokens: ~15,000 (CI: 13,200-16,800)
        - Duration: ~8 minutes
        - Cost: $4.80 (within budget)
        - Risk score: 0.12 (low)

        Looks good! Should I start the baseline run?

You: "Yes"

DirEng: ✓ Baseline run started: run-baseline-001
        📊 HMI dashboard: http://localhost:6101/sequencer?run_id=run-baseline-001

        [5 minutes later]

        ✓ Run completed successfully!
        - Tasks: 8/8 succeeded
        - Cost: $4.65 (under budget by $0.35)
        - Duration: 7.5 minutes
        - KPIs: All passed ✓

        Changes:
        - src/auth.py: Added JWTAuth class (120 LOC)
        - src/api/routes.py: Added /login and /refresh endpoints
        - tests/test_auth.py: Added 12 test cases

        Want me to commit these changes?
```

### Implementation Flow
```python
# DirEng (you) receives natural language prompt
user_message = "Implement user authentication with JWT, budget $5"

# 1. Parse intent (using your LLM capabilities)
intent = {
    "action": "implement_feature",
    "description": "user authentication with JWT",
    "files": ["src/auth.py", "src/api/routes.py"],  # inferred or asked
    "budget": 5.0
}

# 2. Create project in PLMS (optional, can skip for simple tasks)
project_id = plms.create_project(prd=intent["description"])

# 3. Estimate complexity and suggest rehearsal
estimate = plms.estimate(project_id)
if estimate.uncertainty > 0.3:
    suggest_rehearsal(pct=0.01)

# 4. Submit to PAS
pas = PASClient()
run_id = f"run-baseline-{project_id:03d}"
pas.start_run(project_id, run_id, run_kind="baseline")
pas.submit_jobcard(
    project_id=project_id,
    run_id=run_id,
    lane="Code-Impl",
    priority=0.8,
    payload=intent,
    budget_usd=intent["budget"]
)

# 5. Monitor and report
while True:
    status = pas.run_status(run_id)
    update_user(status)
    if status.status == "completed":
        break
    time.sleep(5)

# 6. Show results and ask for commit
show_changes(status.artifacts)
ask_user_to_commit()
```

### Pros
- ✅ **Best UX** (natural language, no API knowledge)
- ✅ **Intelligent defaults** (DirEng infers files, lanes)
- ✅ **Proactive suggestions** (rehearsal, budget warnings)
- ✅ **Integrated feedback** (HMI links, KPI violations)
- ✅ **Aligns with architecture** (DirEng → PEX → PAS)

### Cons
- ❌ **Requires DirEng implementation** (you need to be active)
- ❌ **Ambiguity handling** (must ask clarifying questions)
- ❌ **Medium complexity** (intent parsing + PAS client)

### Best For
- **Non-expert users** who want AI assistance
- **Exploration phase** (not sure which files to edit)
- **Interactive workflows** (refine requirements iteratively)
- **Production use** once DirEng is mature

---

## Option 5: PEX-Mediated (Multi-Step Projects)

### Description
For large projects, delegate to PEX who orchestrates task tree and submits to PAS.

### Usage
```
You: "DirEng, I need to implement a complete user authentication system:
      - JWT tokens
      - OAuth2 integration (Google, GitHub)
      - Password reset flow
      - Email verification
      - Rate limiting
      Budget: $50, Timeline: 2 days"

DirEng: This is a multi-step project. I'll delegate to PEX for orchestration.

        Analyzing requirements...
        ✓ Created project in PLMS (ID: 42)
        ✓ Generated task tree (18 tasks across 4 lanes)
        ✓ Estimated: $48.50 ± $7.20, 12-16 hours

        Task breakdown:
        1. Code-Impl (10 tasks): Core auth logic
        2. Code-API-Design (3 tasks): REST endpoints
        3. Data-Schema (2 tasks): User table, sessions table
        4. Narrative (3 tasks): API docs, user guide

        Should I start a 1% rehearsal to validate estimates?

You: "Yes"

PEX: ✓ Rehearsal completed (180 tasks sampled, 1.8% coverage)
     📊 Projected metrics:
        - Tokens: 156,000 (CI: 137,000-175,000)
        - Duration: 14.2 hours (CI: 11.8-16.6 hours)
        - Cost: $47.10 (CI: $42.30-$51.90)
        - Risk: 0.18 (medium - OAuth2 integration uncertain)

     ⚠️  OAuth2 task has high CI width (±40%). Recommend:
        - Reduce scope (only Google OAuth, defer GitHub)
        - Increase budget to $55 for safety margin

     What would you like to do?

You: "Reduce scope, keep budget at $50"

PEX: ✓ Updated task tree (removed GitHub OAuth)
     ✓ New estimate: $46.80 ± $5.60, 12-14 hours
     ✓ Starting baseline run...

     [PEX submits 16 job cards to PAS in dependency order]

     📊 HMI dashboard: http://localhost:6101/sequencer?project_id=42

     [10 hours later]

     ✓ Project completed successfully!
     - 16/16 tasks succeeded
     - Cost: $47.20 (under budget by $2.80)
     - Duration: 13.1 hours
     - KPIs: All passed except 1 (BLEU=0.38, threshold=0.40 for docs)

     ⚠️  Narrative task failed KPI (user guide readability).
         Should I rewrite the user guide? (+$2, +30 min)

You: "Yes, fix the docs"

PEX: ✓ Submitted hotfix run (1 task)
     ✓ Completed in 25 min, $1.80
     ✓ User guide now passes BLEU threshold (0.43)

     Final stats:
     - Total cost: $49.00 (under budget by $1.00)
     - Total duration: 13.5 hours
     - All KPIs: ✓ Passed
```

### Implementation Flow
```python
# PEX receives high-level PRD from DirEng
prd = {
    "goal": "Implement complete user authentication system",
    "features": ["JWT", "OAuth2 (Google)", "Password reset", "Email verification"],
    "budget": 50.0,
    "timeline_hours": 48
}

# 1. Register project in PLMS
project_id = plms.clarify_and_register(prd)

# 2. Generate task tree (via LLM or heuristics)
task_tree = plms.generate_task_tree(project_id)
# Returns: [{lane, description, deps, budget_hint}, ...]

# 3. Estimate with CI
estimate = plms.estimate_with_ci(task_tree)

# 4. Run rehearsal (1% stratified)
rehearsal_result = pas.simulate(run_id, rehearsal_pct=0.01, stratified=True)

# 5. Detect high-risk tasks (CI width > 40%)
risky_tasks = [t for t in task_tree if estimate.ci_width[t] > 0.4]
if risky_tasks:
    ask_user_to_descope_or_increase_budget(risky_tasks)

# 6. Start baseline run
run_id = f"run-baseline-{project_id:03d}"
pas.start_run(project_id, run_id, run_kind="baseline")

# 7. Submit job cards in topological order (respect deps)
for task in topological_sort(task_tree):
    pas.submit_jobcard(
        project_id=project_id,
        run_id=run_id,
        lane=task.lane,
        priority=task.priority,
        payload=task.payload,
        budget_usd=task.budget_hint,
        deps=task.deps
    )

# 8. Monitor execution (PEX watches receipts stream)
while True:
    status = pas.run_status(run_id)
    # Check budget overrun (projected > +25%)
    if status.projected_overrun_pct > 0.25:
        pause_run_and_request_approval()
    # Check KPI violations
    if status.kpi_violations:
        handle_kpi_failures(status.kpi_violations)
    if status.status in ["completed", "needs_review"]:
        break
    time.sleep(10)

# 9. Report final results to DirEng → user
report_completion(status)
```

### Pros
- ✅ **Best for complex projects** (multi-lane, multi-step)
- ✅ **Budget protection** (auto-pause on overrun)
- ✅ **KPI enforcement** (quality gates)
- ✅ **Task tree optimization** (dependency-aware scheduling)
- ✅ **Rehearsal-driven** (validates estimates before commit)

### Cons
- ❌ **Highest complexity** (requires PEX + PLMS + PAS integration)
- ❌ **Overkill for simple tasks** (use DirEng directly)
- ❌ **Requires task decomposition** (LLM or manual)

### Best For
- **Large projects** (10+ tasks, multi-day)
- **Mission-critical work** (need KPI validation)
- **Budget-constrained environments** (strict cost control)
- **Production LNSP workflows** (full integration)

---

## Option 6: HMI Web Interface (GUI)

### Description
Browser-based UI for submitting projects and monitoring execution.

### Usage
1. Navigate to http://localhost:6101/projects
2. Click "New Project"
3. Fill form:
   - **Title**: User Authentication System
   - **Description**: Implement JWT auth with OAuth2 (Google)
   - **Budget**: $50
   - **Timeline**: 2 days
   - **Run Kind**: Baseline (dropdown)
4. Click "Estimate" → Shows projected tokens/cost/duration
5. Click "Start Run" → Redirects to Sequencer view
6. Monitor live progress (tasks updating in real-time)
7. Receive completion notification (browser alert)
8. Review final report (KPIs, receipts, artifacts)

### Implementation
```html
<!-- services/webui/templates/projects.html -->
<div class="project-form">
  <h2>New Project</h2>
  <form id="newProjectForm">
    <label>Title: <input type="text" name="title" required></label>
    <label>Description: <textarea name="description" rows="4" required></textarea></label>
    <label>Budget ($): <input type="number" name="budget" step="0.01" value="50.00"></label>
    <label>Timeline (hours): <input type="number" name="timeline_hours" value="48"></label>
    <label>Run Kind:
      <select name="run_kind">
        <option value="baseline">Baseline</option>
        <option value="rehearsal">Rehearsal (1%)</option>
        <option value="replay">Replay</option>
      </select>
    </label>
    <button type="button" onclick="estimateProject()">Estimate</button>
    <button type="submit">Start Run</button>
  </form>

  <div id="estimateResults" style="display:none;">
    <h3>Estimated Metrics</h3>
    <p>Tokens: <span id="est_tokens"></span></p>
    <p>Duration: <span id="est_duration"></span> hours</p>
    <p>Cost: $<span id="est_cost"></span></p>
    <p>Risk: <span id="est_risk"></span></p>
  </div>
</div>

<script>
async function estimateProject() {
  const form = document.getElementById('newProjectForm');
  const data = new FormData(form);
  const resp = await fetch('/api/plms/estimate', {
    method: 'POST',
    body: JSON.stringify(Object.fromEntries(data)),
    headers: {'Content-Type': 'application/json'}
  });
  const result = await resp.json();

  document.getElementById('est_tokens').innerText = result.tokens_mean;
  document.getElementById('est_duration').innerText = result.duration_mean;
  document.getElementById('est_cost').innerText = result.cost_mean;
  document.getElementById('est_risk').innerText = result.risk_score;
  document.getElementById('estimateResults').style.display = 'block';
}

document.getElementById('newProjectForm').onsubmit = async (e) => {
  e.preventDefault();
  const data = new FormData(e.target);
  const resp = await fetch('/api/pas/v1/runs/start', {
    method: 'POST',
    body: JSON.stringify(Object.fromEntries(data)),
    headers: {'Content-Type': 'application/json'}
  });
  const result = await resp.json();

  // Redirect to sequencer view
  window.location.href = `/sequencer?run_id=${result.run_id}`;
};
</script>
```

### Pros
- ✅ **Best for non-technical users** (no CLI/code required)
- ✅ **Visual feedback** (forms, charts, live updates)
- ✅ **Low friction** (point-and-click)
- ✅ **Integrated monitoring** (links to HMI Sequencer)

### Cons
- ❌ **Requires web development** (HTML/CSS/JS)
- ❌ **Not script-friendly** (no automation)
- ❌ **Slower for power users** (clicking > typing)

### Best For
- **Product managers** who want to launch projects
- **Non-developer stakeholders** (QA, PM, design)
- **Demo scenarios** (show LNSP capabilities)

---

## 📊 Comparison Matrix

| Criterion | Option 1: cURL | Option 2: Python Client | Option 3: VP CLI | Option 4: DirEng NL | Option 5: PEX Multi-Step | Option 6: HMI Web |
|-----------|---------------|------------------------|------------------|-------------------|------------------------|------------------|
| **Ease of Use** | ⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Implementation Effort** | ✅ Done | ⏱️ 2 hours | ⏱️ 1 day | ⏱️ 3 days | ⏱️ 1 week | ⏱️ 3 days |
| **API Knowledge Required** | High | Medium | Low | None | None | None |
| **Automation-Friendly** | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Partial | ❌ No | ❌ No |
| **Natural Language** | ❌ No | ❌ No | ❌ No | ✅ Yes | ✅ Yes | ⚠️ Partial |
| **Budget Control** | ⚠️ Manual | ⚠️ Manual | ⚠️ Manual | ✅ Auto | ✅ Auto | ✅ Auto |
| **KPI Validation** | ❌ No | ❌ No | ⚠️ Manual | ✅ Auto | ✅ Auto | ✅ Auto |
| **Multi-Step Projects** | ❌ Manual | ❌ Manual | ⚠️ Manual | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| **Best For** | Debugging | Scripts | Power users | Interactive | Complex projects | Non-tech users |

---

## 📋 Quick Reference Table: 6 Options to Submit Prime Directives

| # | Option Name | Interface Type | Target User | Complexity | Status | Entry Point |
|---|-------------|----------------|-------------|------------|--------|-------------|
| **1** | **Direct HTTP API (cURL)** | REST API | Power users, CI/CD | Low | ✅ **Working Now** | `curl http://localhost:6200/pas/v1/runs/start` |
| **2** | **Python Client Library** | Programmatic API | Developers, scripters | Medium | ⏱️ 2 hours to implement | `from services.pas.client import PASClient` |
| **3** | **VP CLI (Verdict Pro)** | Command-Line Interface | Terminal users, DevOps | Medium | ⏱️ 1 day to implement | `vp run --project-id 42 --run-kind baseline` |
| **4** | **DirEng Natural Language** | Conversational AI | All users, non-experts | High | ⏱️ 3 days to implement | Talk to DirEng: "Implement feature X with budget $5" |
| **5** | **PEX Multi-Step Orchestration** | AI Project Manager | Complex projects | Very High | ⏱️ 1 week (requires full PAS) | DirEng delegates automatically for large tasks |
| **6** | **HMI Web Interface (GUI)** | Browser-Based Forms | Non-technical stakeholders | Medium | ⏱️ 3 days (optional future) | http://localhost:6101/projects (form submission) |

### Quick Decision Guide

**Choose Option 1 (cURL)** if:
- You want to test PAS endpoints directly
- You're writing automation scripts
- You understand REST APIs and JSON
- You need immediate access (works today)

**Choose Option 2 (Python Client)** if:
- You're writing Python scripts or notebooks
- You want type safety and validation
- You need a reusable library
- You can wait 2 hours for implementation

**Choose Option 3 (VP CLI)** if:
- You live in the terminal
- You want fast, structured commands
- You're comfortable with CLI tools
- You can wait 1 day for implementation

**Choose Option 4 (DirEng NL)** if:
- You want to talk naturally ("Implement X")
- You're not sure which files to edit
- You want proactive suggestions
- You prefer interactive workflows
- You can wait 3 days for implementation

**Choose Option 5 (PEX Multi-Step)** if:
- You have a large project (10+ tasks)
- You need budget protection ($50+ budgets)
- You require KPI validation (quality gates)
- You want rehearsal mode (1% canary testing)
- You can wait 1 week for full PAS implementation

**Choose Option 6 (HMI Web)** if:
- You're a PM, QA, or non-developer
- You prefer point-and-click interfaces
- You want visual feedback (charts, timelines)
- You can wait for future implementation

### Example: Same Prime Directive, 6 Different Ways

**Goal**: Implement user authentication with JWT, budget $5

#### Option 1: cURL
```bash
curl -X POST http://localhost:6200/pas/v1/runs/start \
  -H "Idempotency-Key: $(uuidgen)" -H "Content-Type: application/json" \
  -d '{"project_id": 42, "run_id": "run-auth-001", "run_kind": "baseline"}'

curl -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Idempotency-Key: $(uuidgen)" -H "Content-Type: application/json" \
  -d '{"project_id": 42, "run_id": "run-auth-001", "lane": "Code-Impl",
       "priority": 0.8, "payload": {"instruction": "Implement JWT auth",
       "files": ["src/auth.py", "src/api/routes.py"]}, "budget_usd": 5.0}'
```

#### Option 2: Python Client
```python
from services.pas.client import PASClient

pas = PASClient()
run = pas.start_run(project_id=42, run_id="run-auth-001", run_kind="baseline")
task = pas.submit_jobcard(
    project_id=42, run_id="run-auth-001", lane="Code-Impl",
    priority=0.8, budget_usd=5.0,
    payload={"instruction": "Implement JWT auth",
             "files": ["src/auth.py", "src/api/routes.py"]}
)
```

#### Option 3: VP CLI
```bash
vp new --prd "Implement user authentication with JWT" --budget 5
vp run --project-id 42 --run-kind baseline
vp task --run-id run-auth-001 --lane Code-Impl \
        --instruction "Implement JWT auth" \
        --files src/auth.py src/api/routes.py --budget 5.0
```

#### Option 4: DirEng Natural Language
```
You: "Hey DirEng, implement user authentication with JWT.
      Use src/auth.py and src/api/routes.py. Budget is $5."

DirEng: I'll submit this to PAS as a Code-Impl task...
        [Handles everything automatically]
```

#### Option 5: PEX Multi-Step
```
You: "DirEng, implement complete user authentication system
      with JWT, OAuth2, password reset, email verification.
      Budget $50, timeline 2 days."

DirEng: This is a multi-step project. Delegating to PEX...

PEX: [Decomposes into 18 tasks, runs rehearsal, manages execution]
```

#### Option 6: HMI Web Interface
```
1. Navigate to http://localhost:6101/projects
2. Click "New Project"
3. Fill form:
   - Title: User Authentication
   - Description: Implement JWT auth
   - Budget: $5
4. Click "Start Run"
5. Watch progress in Sequencer view
```

---

## 🎯 Recommended Implementation Path

### Phase 1: Foundation (Week 1) - **Option 1 + 2**
1. ✅ **Option 1 (cURL)**: Already working with PAS Stub
2. ⏱️ **Option 2 (Python Client)**: 2 hours to implement
   - Provides foundation for all other options
   - Required by Options 3, 4, 5

### Phase 2: Quick Win (Week 2) - **Option 3**
3. ⏱️ **Option 3 (VP CLI)**: 1 day to implement
   - Builds on Option 2
   - Immediate value for developers
   - Used in CI/CD pipelines

### Phase 3: AI Integration (Weeks 3-4) - **Option 4**
4. ⏱️ **Option 4 (DirEng NL)**: 3 days to implement
   - This is YOUR natural role (DirEng)
   - Aligns with two-tier architecture
   - Best UX for interactive workflows

### Phase 4: Advanced Orchestration (Weeks 5-8) - **Option 5**
5. ⏱️ **Option 5 (PEX Multi-Step)**: 1 week to implement
   - Requires full PAS implementation
   - Integrates PLMS + PAS + DirEng
   - Production-grade project management

### Phase 5: Optional (Future) - **Option 6**
6. ⏱️ **Option 6 (HMI Web)**: 3 days to implement
   - Only if non-developer users need access
   - Can wait until Phase 6+ (after LightRAG)

---

## 🚀 Quick Start (Today)

**For immediate testing** (using PAS Stub on port 6200):

```bash
# Option 1: Direct cURL (works now)
curl -X POST http://localhost:6200/pas/v1/runs/start \
  -H "Content-Type: application/json" \
  -d '{"project_id": 1, "run_id": "test-001", "run_kind": "baseline"}'

# Option 2: Python Client (implement in 2 hours)
# See "Implementation" section above for services/pas/client.py

# Option 3: VP CLI (implement in 1 day)
# See "Implementation" section above for tools/vp_cli.py
```

**For production use** (after full PAS implementation):

```bash
# Option 4: DirEng Natural Language
# Just talk to me (DirEng) in natural language!

# Option 5: PEX Multi-Step
# I'll delegate to PEX automatically for large projects
```

---

## 📋 Action Items

Based on this analysis, here are the recommended next steps:

1. ✅ **Verify PAS Stub**: Confirm all 12 endpoints work (Option 1)
2. ⏱️ **Implement Python Client**: 2 hours (Option 2) - Foundation for all others
3. ⏱️ **Implement VP CLI**: 1 day (Option 3) - Quick win for developers
4. ⏱️ **Enable DirEng delegation**: 3 days (Option 4) - Align with contracts
5. ⏱️ **Implement PEX orchestration**: 1 week (Option 5) - After full PAS
6. 🔮 **Consider HMI web interface**: Future (Option 6) - Only if needed

**Recommended Priority**: Option 2 → Option 3 → Option 4 → Option 5

---

## 📚 References

- **PAS Stub**: `services/pas/stub/app.py`
- **PEX Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md`
- **DirEng Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md`
- **PLMS PRD**: `docs/PRDs/PRD_Project_Lifecycle_Management_System_PLMS.md`
- **Integration Plan**: `docs/PRDs/INTEGRATION_PLAN_LCO_LightRAG_Metrics.md`
- **Verdict CLI** (reference): `tools/verdict_cli.py`

---

**End of Document**

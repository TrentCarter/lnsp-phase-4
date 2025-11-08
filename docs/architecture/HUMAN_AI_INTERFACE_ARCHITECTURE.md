# Human ↔ AI Interface Architecture

**Version**: 2025-11-07-001
**Status**: Design Document

---

## Overview

This document defines the **two-tier AI interface** for the LNSP system:

1. **DirEng** (Director of Engineering AI): Human-facing conversational assistant
2. **PEX** (Project Executive): Project-facing orchestration layer

**Key Insight**: You talk to **DirEng** (like Claude Code), DirEng delegates to **PEX** (like a project manager) when needed.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                           USER (You)                             │
│                                                                  │
│  Natural language: "Implement feature X", "Where is Y?", etc.   │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Conversational
                             │ Interface
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DirEng (Tier 1)                            │
│                  "Director of Engineering AI"                     │
│                                                                  │
│  Role: Human-facing conversational assistant                     │
│  Analogue: Claude Code (explore, answer, small edits)           │
│                                                                  │
│  Direct Tools:                                                   │
│  - fs.read/write/patch/search/glob                              │
│  - git.status/diff/commit                                       │
│  - shell.exec (with approval)                                   │
│  - rag.query (LightRAG semantic/graph)                          │
│                                                                  │
│  Decision Logic:                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Small task (<5 min, 1-3 files)?                          │  │
│  │   YES → Handle directly                                  │  │
│  │   NO  → Delegate to PEX                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Delegation
                             │ (when task is large/complex)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        PEX (Tier 2)                              │
│                   "Project Executive"                            │
│                                                                  │
│  Role: Project-facing orchestration layer                       │
│  Operates via: Strict contracts, allowlists, KPI gates          │
│                                                                  │
│  Services Orchestrated:                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐               │
│  │   PLMS     │  │    PAS     │  │ Vector-Ops │               │
│  │            │  │            │  │            │               │
│  │ Estimate   │  │ Execute    │  │ LightRAG   │               │
│  │ Budget     │  │ Monitor    │  │ Refresh    │               │
│  │ Calibrate  │  │ KPI Gates  │  │ Index      │               │
│  └────────────┘  └────────────┘  └────────────┘               │
│                                                                  │
│  Constraints:                                                    │
│  - Budget caps + runway monitoring                              │
│  - Command allowlists (no raw shell)                            │
│  - Sandboxed executors (bubblewrap/sandbox-exec)                │
│  - KPI validation (echo_cos, test_pass_rate, etc.)             │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Status Updates
                             │ (progress, spend, KPIs)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                       DirEng (Tier 1)                            │
│                                                                  │
│  Relays status back to user in conversational format:           │
│  "PEX is implementing JWT auth (run abc123):                    │
│   - ✅ Created User model                                        │
│   - 🔄 Writing tests (current)                                  │
│   - ⏳ Integration tests                                         │
│   Spend: $1.25 / $5.00 cap"                                     │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             │ Results
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                           USER (You)                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Comparison: DirEng vs PEX

| Aspect | DirEng (Tier 1) | PEX (Tier 2) |
|--------|-----------------|--------------|
| **User** | Human (you) | DirEng |
| **Interface** | Natural language | Structured API (JSON) |
| **Scope** | Single task | Multi-task project |
| **Duration** | Seconds to minutes | Minutes to hours |
| **Tools** | Direct FS/shell/git | Sandboxed executors |
| **Safety** | Asks permission | Strict allowlists |
| **Budget** | No tracking | Tracked with caps |
| **KPIs** | N/A | Enforced gates |
| **Output** | Conversational | Structured (receipts) |
| **Analogue** | Claude Code | GitHub Actions + PM |

---

## When DirEng Delegates to PEX

### Decision Matrix

| Task Type | Example | Handler |
|-----------|---------|---------|
| **Exploration** | "Where is X defined?" | DirEng (direct) |
| **Small Edit** | "Fix typo in file Y" | DirEng (direct) |
| **Local Op** | "Run tests" | DirEng (direct) |
| **Medium Task** | "Refactor 3 files to use pattern Z" | DirEng (ask user) |
| **Large Task** | "Implement user auth with JWT" | DirEng → PEX |
| **Estimation** | "How long will this take?" | DirEng → PEX (PLMS) |
| **Budget Tracking** | "What's my spend so far?" | DirEng → PEX (PLMS) |

### Trigger Phrases (Auto-Delegate)

User says one of these → DirEng automatically delegates to PEX:
- "Implement feature X"
- "Refactor module Y to use pattern Z"
- "Run full test suite and fix all errors"
- "Estimate how long this will take"
- "Add logging to all API endpoints"
- "Generate API documentation from code"

---

## Information Flow

### Example: Large Task Delegation

**Step 1: User → DirEng**
```
User: "Implement user authentication with JWT, including tests and docs"
```

**Step 2: DirEng → PEX** (Delegation)
```json
{
  "from": "direng",
  "to": "pex",
  "intent": "Implement user authentication with JWT",
  "context": {
    "current_branch": "feature/auth",
    "modified_files": ["src/api/routes.py"],
    "user_preferences": {"test_framework": "pytest"}
  },
  "constraints": {
    "max_cost_usd": 5.0,
    "timeout_minutes": 30
  }
}
```

**Step 3: PEX → PLMS** (Estimate)
```
PEX calls PLMS API:
POST /api/projects/{id}/estimate
→ Returns: 25k tokens, $2.50, 15 minutes
```

**Step 4: PEX → DirEng** (Confirmation)
```json
{
  "status": "awaiting_approval",
  "estimate": {
    "tokens": 25000,
    "cost_usd": 2.50,
    "duration_min": 15
  }
}
```

**Step 5: DirEng → User** (Request Approval)
```
DirEng: "PEX estimates this will take:
  - Tokens: 25,000
  - Cost: $2.50
  - Duration: ~15 minutes

Proceed? (y/n)"
```

**Step 6: User → DirEng** (Approval)
```
User: "y"
```

**Step 7: DirEng → PEX** (Approved)
```json
{"approved": true, "idempotency_key": "uuid"}
```

**Step 8: PEX → PAS** (Execute)
```
PEX calls PAS API:
POST /pas/v1/runs/start
→ Returns: run_id = "abc123"
```

**Step 9: PAS → PEX** (Status Updates, periodic)
```json
{
  "run_id": "abc123",
  "progress": {
    "tasks_completed": 3,
    "tasks_total": 8,
    "current_task": "Writing unit tests"
  },
  "spend": {"tokens_used": 15000, "cost_usd": 1.25}
}
```

**Step 10: PEX → DirEng** (Relay)
```json
{
  "status": "in_progress",
  "progress": {...},
  "spend": {...}
}
```

**Step 11: DirEng → User** (Conversational Update)
```
DirEng: "PEX is implementing JWT authentication (run abc123):
  - ✅ Created User model
  - ✅ Added JWT signing/verification
  - ✅ Implemented /login endpoint
  - 🔄 Writing unit tests (current)
  - ⏳ Integration tests
  - ⏳ API documentation

Spend: $1.25 / $5.00 cap
ETA: 8 minutes remaining"
```

**Step 12: PAS → PEX** (Completion)
```json
{
  "status": "completed",
  "kpis": {"echo_cos": 0.89, "test_pass_rate": 1.0},
  "spend": {"tokens_used": 24500, "cost_usd": 2.45}
}
```

**Step 13: PEX → DirEng** (Final Report)
```json
{
  "status": "completed",
  "summary": "Implemented JWT auth with 12/12 tests passing",
  "artifacts": ["src/auth.py", "tests/test_auth.py", "docs/auth.md"],
  "kpis": {...},
  "spend": {...}
}
```

**Step 14: DirEng → User** (Final Summary)
```
DirEng: "✅ JWT authentication complete!

Files changed:
  - src/auth.py (new)
  - src/api/routes.py (modified)
  - tests/test_auth.py (new)
  - docs/auth.md (new)

Tests: 12/12 passing
Cost: $2.45 (under $5.00 cap)
Duration: 14 minutes

Want me to commit these changes?"
```

---

## Implementation Path

### Phase 1: DirEng MVP (Weeks 1-2)

**Goal**: Get DirEng operational as a standalone conversational assistant (no PEX delegation yet)

**Deliverables**:
1. DirEng CLI/REPL interface
2. Direct tool integration (fs, git, shell, rag)
3. Context tracking (current branch, modified files)
4. Basic approval flow (ask before risky ops)

**Test**: User can explore codebase, make small edits, run tests

---

### Phase 2: PEX Delegation (Weeks 3-4)

**Goal**: Wire DirEng → PEX delegation for large tasks

**Deliverables**:
1. Delegation protocol (JSON format)
2. Task complexity heuristic (when to delegate)
3. Status relay (PEX → DirEng → User)
4. Approval flow (estimate → user confirmation → execute)

**Test**: User says "Implement feature X" → DirEng delegates to PEX → status updates → completion

---

### Phase 3: LightRAG Integration (Weeks 1-2, parallel)

**Goal**: DirEng uses LightRAG for semantic/graph queries

**Deliverables**:
1. LightRAG code index (tree-sitter → Neo4j → FAISS)
2. Query verbs (where_defined, who_calls, impact_set, nn_snippet)
3. DirEng integration (`rag.query()`)

**Test**: User asks "Where is X?" → DirEng uses LightRAG → fast, accurate results

---

### Phase 4: Full PAS (Weeks 5-8)

**Goal**: Replace PAS stub with full implementation

**Deliverables**:
1. Lane executors (Code-Impl, Data-Schema, Vector-Ops, etc.)
2. Sandboxing (bubblewrap, cgroups v2, allowlists)
3. Auth/secrets/artifact store
4. KPI validators

**Test**: PEX delegates to PAS → multi-lane execution → KPI validation → completion

---

## DirEng Interface Options

### Option A: REPL (Interactive Shell)

```bash
$ direng

DirEng> where is the database connection defined?
Found at src/db.py:15-25 [shows code]

DirEng> fix the typo on line 17
Applied patch [shows diff]

DirEng> commit this change
✅ Committed: "Fix DB host env var typo"

DirEng> implement jwt authentication
This is a large task (estimate: $2.50, 15 min).
Delegate to PEX? (y/n) y
✅ Delegated to PEX (run abc123)
[Status updates stream here...]
✅ Complete! Files changed: 4, Tests: 12/12
```

### Option B: VS Code Extension

```
User: Opens command palette → "DirEng: Ask"
User: Types "Where is the database connection?"
→ DirEng responds in sidebar with code snippet + line numbers
User: Clicks "Fix typo" button
→ DirEng applies patch, shows diff in editor
```

### Option C: CLI (One-Shot Commands)

```bash
$ direng ask "Where is the database connection?"
[Shows result]

$ direng fix "Fix typo in src/db.py:17"
[Shows diff, asks for confirmation]

$ direng implement "JWT authentication"
[Delegates to PEX, streams status]
```

### Recommendation: Start with REPL (Option A)

- **Why**: Conversational flow, easy to prototype, no IDE lock-in
- **Later**: Add VS Code extension (Option B) for visual users
- **CLI (Option C)**: Good for scripting, but less conversational

---

## Security Considerations

### DirEng (Tier 1)
- **Direct shell access**: YES (with approval for risky ops)
- **File write access**: YES (with user confirmation)
- **Network access**: YES (with approval for outbound requests)
- **Secrets**: Can read `.env` (warns user if secrets detected)

**Rationale**: User trusts DirEng like Claude Code (full access, but asks permission)

### PEX (Tier 2)
- **Direct shell access**: NO (only allowlisted commands)
- **File write access**: YES (via sandboxed executors, allowlisted paths)
- **Network access**: NO (unless lane policy permits, e.g., Vector-Ops may allow localhost)
- **Secrets**: NO (fetched from vault, never in prompts/logs)

**Rationale**: PEX operates autonomously, needs strict boundaries

---

## Open Questions

1. **DirEng Persistence**: Should DirEng remember conversations across sessions?
   - **Option A**: Ephemeral (like Claude Code, fresh each time)
   - **Option B**: Persistent (saves conversation history, project context)

2. **Multi-User**: If multiple users work on same project, how to handle?
   - **Option A**: Each user gets their own DirEng instance (no sharing)
   - **Option B**: Shared DirEng (team context)

3. **DirEng Model**: What LLM powers DirEng?
   - **Option A**: Same as PEX (Anthropic Sonnet 4.5)
   - **Option B**: Faster model for quick queries (Haiku), escalate to Sonnet for complex

4. **Approval Fatigue**: How to reduce "ask permission" prompts without sacrificing safety?
   - **Option A**: User sets trust level (low/medium/high)
   - **Option B**: DirEng learns from past approvals

---

## References

- **DirEng Contract**: `docs/contracts/DIRENG_SYSTEM_PROMPT.md`
- **PEX Contract**: `docs/contracts/PEX_SYSTEM_PROMPT.md`
- **Integration Plan**: `docs/PRDs/INTEGRATION_PLAN_LCO_LightRAG_Metrics.md`
- **Security Design**: `docs/design/SECURITY_INTEGRATION_PLAN.md`

---

**Next Steps**:

1. Build DirEng REPL (Week 1-2)
2. Wire to LightRAG (Week 1-2, parallel)
3. Add PEX delegation (Week 3-4)
4. Test end-to-end flow (Week 4)

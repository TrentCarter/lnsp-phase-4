# PRD: Project Lifecycle Management System (PLMS)

**Version**: 1.0
**Date**: 2025-11-06
**Status**: Draft
**Owner**: LNSP Core Team

---

## Executive Summary

**PLMS** is a thin lifecycle coordination layer that sits **above** the Polyglot Agent Swarm (PAS) to provide human-facing project management: idea → clarification → PRD → estimation → approval → execution → comparison.

**What PLMS Is**:
- Pre-execution planning layer (ideation, Q&A, PRD generation, estimation)
- Human approval gates with budget/quality SLOs
- Post-execution aggregation (actual vs estimated, receipts ledger)
- HMI overlays on existing Tree/Sequencer views

**What PLMS Is NOT**:
- A parallel orchestrator (PAS handles all execution via job cards, receipts, heartbeats)
- A duplicate token budget system (uses Token Governor)
- A new UI stack (extends existing HMI views)

**Core Innovation**: TMD-aware estimation with lane-specific priors, Echo-Loop quality SLOs, and seamless PAS integration for execution transparency.

---

## 1. Problem Statement

### Current State
- Users submit raw ideas → manual planning → ad-hoc execution
- No structured estimation of tokens/time/cost before execution
- No visibility into project lifecycle (ideation → completion)
- PAS execution is powerful but lacks a "project wrapper" for human workflows

### Pain Points
1. **No pre-execution clarity**: Users can't answer "What will this cost?" before approving work
2. **No structured ideation**: Ideas go straight to execution without clarification or PRD
3. **No quality gates**: Execution can succeed (green dashboard) but produce low-quality outputs (red semantics)
4. **No post-mortem aggregation**: Hard to compare estimated vs actual, or audit decision chains

### Why Now?
- PAS infrastructure is mature (Architect, Directors, Managers, receipts, heartbeats, Token Governor)
- HMI is production-ready (Tree, Sequencer, Audio, live WS/SSE streams)
- Need human-facing lifecycle layer to unlock non-expert users

---

## 2. Goals & Success Criteria

### Goals
1. **Pre-execution planning**: Idea → Q&A → PRD → Estimation with ≤10min human time
2. **Accurate estimates**: Actual vs estimated within ±20% (tokens), ±30% (time) averaged over 10 projects
3. **Quality SLOs**: Projects define Echo-Loop thresholds (e.g., ≥0.82 cosine) and block completion until met
4. **PAS integration**: Zero duplication—PLMS translates PRD → PAS job cards, consumes PAS receipts/heartbeats
5. **HMI transparency**: Live project status via overlays on existing Tree/Sequencer (no new UI stack)

### Success Criteria (V1 MVP)
- [ ] 3 test projects complete full lifecycle (ideation → completion)
- [ ] Estimation accuracy: ≤30% error on tokens/time (mean absolute percentage error)
- [ ] **Tier 1**: 90% credible intervals tracked, estimates improve after 10 projects
- [ ] Echo-Loop SLO enforcement: Projects with quality gates block completion when violated
- [ ] **Tier 1**: Lane-specific KPIs enforced (test pass rates, schema diffs, etc.)
- [ ] PAS integration: All execution uses PAS job cards, receipts visible in HMI
- [ ] HMI overlays: Project badges, budget gauge, SLO status visible in existing views
- [ ] **Tier 1**: Budget runway gauge + risk heatmap visible in Project Detail view
- [ ] **Tier 1**: 1% rehearsal mode functional (canary execution before full commit)
- [ ] **Tier 1**: Multi-run support (baseline, rehearsal, replay, hotfix)

### Non-Goals (V1)
- Semantic Critical Path (graph centrality for task prioritization) → V2
- ~~Rehearsal Mode (1-5% slice execution for SLO preview)~~ → **Moved to V1 (Tier 1)**
- PRD-Delta Watcher (auto-replan on PRD changes) → V2
- ~~Deterministic Replay (byte-for-byte run reconstruction)~~ → **Foundation in V1 (replay passport), Full implementation V2**

---

## 3. Architecture & Design

### 3.1 System Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    PLMS (Lifecycle Layer)                      │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │  Ideation   │→│ Clarification │→│  PRD Gen     │          │
│  │  Agent      │  │  Agent (Q&A) │  │  Agent       │          │
│  └─────────────┘  └──────────────┘  └──────────────┘         │
│                            ↓                                    │
│                   ┌──────────────┐                             │
│                   │  Estimation  │  (TMD-aware, lane priors)   │
│                   │  Agent       │                             │
│                   └──────────────┘                             │
│                            ↓                                    │
│                   ┌──────────────┐                             │
│                   │   Approval   │  (Human gate)               │
│                   │   Gate       │                             │
│                   └──────────────┘                             │
│                            ↓                                    │
│                   ┌──────────────┐                             │
│                   │ PAS Job Card │  (Translate PRD→Job Cards)  │
│                   │ Generator    │                             │
│                   └──────────────┘                             │
└────────────────────────┬───────────────────────────────────────┘
                         │ Job Cards
                         ▼
┌────────────────────────────────────────────────────────────────┐
│             PAS (Polyglot Agent Swarm) - EXISTING              │
│                                                                 │
│  Architect → Directors → Managers → Exec Agents                │
│  • Token Governor (budgets, save/resume)                       │
│  • Provider Router (model selection, receipts)                 │
│  • Registry (discovery, heartbeats, TTL)                       │
│  • Resource Manager (quotas, approvals)                        │
│                                                                 │
│  Outputs: receipts, heartbeats, artifacts, approvals           │
└────────────────────────┬───────────────────────────────────────┘
                         │ Streams
                         ▼
┌────────────────────────────────────────────────────────────────┐
│               HMI (Monitoring Layer) - EXISTING                │
│                                                                 │
│  Tree View, Sequencer, Audio Control                           │
│  + NEW: Project Overlays (badges, SLOs, budget gauge)          │
│                                                                 │
│  Consumes: receipts, heartbeats, artifacts (from PAS)          │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Lifecycle Phases

```
┌─────────────────────────────────────────────────────────────────┐
│                     PROJECT LIFECYCLE                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. IDEATION (status: 'ideation')                               │
│     • User submits: title, description, tags, initial context   │
│     • Store in `projects` table                                 │
│     • Ideation Agent parses intent, suggests question strategy  │
│                                                                  │
│  2. CLARIFICATION (status: 'clarifying')                        │
│     • Clarification Agent asks 3-10 questions                   │
│     • Interactive Q&A (stored in `project_questions`)           │
│     • Loop until: sufficient detail OR user signals "done"      │
│                                                                  │
│  3. PRD GENERATION (status: 'planning')                         │
│     • PRD Agent synthesizes Q&A into structured PRD             │
│     • Saves to `docs/PRDs/PRD_{project_name}.md`               │
│     • User reviews, edits if needed, approves PRD               │
│                                                                  │
│  4. ESTIMATION (status: 'estimating')                           │
│     • Estimation Agent decomposes PRD into task tree            │
│     • Assigns TMD lane, store hint, complexity per task         │
│     • Calculates tokens/time/cost using lane-specific priors    │
│     • Stores in `task_estimates` table                          │
│     • Outputs: total_tokens, total_minutes, total_cost_usd      │
│                                                                  │
│  5. APPROVAL (status: 'pending_approval')                       │
│     • User reviews: PRD + task tree + estimates                 │
│     • Sets budget caps: max_tokens, max_cost_usd                │
│     • Defines quality SLOs: min_echo_cos, lane-specific gates   │
│     • Action: Approve / Reject / Request Replan                 │
│     • On approval → status: 'approved'                          │
│                                                                  │
│  6. EXECUTION (status: 'executing')                             │
│     • PAS Job Card Generator creates job cards from task tree   │
│     • Submits to PAS Architect                                  │
│     • PAS orchestrates: Directors → Managers → Execs            │
│     • Token Governor enforces budget caps                       │
│     • Heartbeats, receipts, artifacts flow to HMI               │
│     • PLMS polls PAS status, updates `project_runs`             │
│                                                                  │
│  7. VALIDATION (status: 'validating')                           │
│     • Echo-Loop checks quality SLOs on produced artifacts       │
│     • If SLOs violated: status → 'needs_review' (human gate)    │
│     • If SLOs met: status → 'completed'                         │
│                                                                  │
│  8. COMPLETION (status: 'completed')                            │
│     • Aggregate: actual_tokens, actual_minutes, actual_cost     │
│     • Compare: actual vs estimated (error %, visual charts)     │
│     • Link: receipts ledger, artifacts, heartbeat stats         │
│     • Post-mortem: lessons learned, accuracy metrics            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Database Schema

### 4.1 New Tables (Planning Artifacts)

```sql
-- Main projects table (lifecycle metadata)
CREATE TABLE projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT,
    tags TEXT, -- JSON array of tags
    status TEXT NOT NULL, -- 'ideation', 'clarifying', 'planning', 'estimating', 'pending_approval', 'approved', 'executing', 'validating', 'needs_review', 'completed', 'cancelled'
    prd_path TEXT, -- e.g., 'docs/PRDs/PRD_calculator_api.md'

    -- Budget caps (set during approval)
    budget_tokens_max INTEGER,
    budget_cost_usd_max REAL,
    breach_policy TEXT DEFAULT 'pause', -- 'pause', 'ask', 'kill'

    -- Quality SLOs (set during approval)
    min_echo_cos REAL DEFAULT 0.82, -- Minimum Echo-Loop cosine threshold
    quality_slo_gates TEXT, -- JSON: lane-specific SLOs

    -- Estimates (from Estimation Agent)
    estimated_tokens INTEGER,
    estimated_duration_minutes INTEGER,
    estimated_cost_usd REAL,

    -- Actuals (from PAS execution)
    actual_tokens INTEGER,
    actual_duration_minutes INTEGER,
    actual_cost_usd REAL,

    -- Accuracy metrics
    token_error_pct REAL, -- abs((actual - estimated) / estimated) * 100
    time_error_pct REAL,
    cost_error_pct REAL,

    -- Timestamps
    created_at TEXT NOT NULL,
    prd_generated_at TEXT,
    approved_at TEXT,
    started_at TEXT,
    completed_at TEXT,

    -- Metadata
    created_by TEXT DEFAULT 'user',
    template_used TEXT -- If created from template
);

-- Q&A for clarification phase
CREATE TABLE project_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT,
    asked_at TEXT NOT NULL,
    answered_at TEXT,
    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Task estimates (pre-execution decomposition)
CREATE TABLE task_estimates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    task_name TEXT NOT NULL,
    task_description TEXT,
    parent_task_id INTEGER, -- Hierarchical tasks
    order_index INTEGER, -- Execution order (topological sort)

    -- TMD routing (LNSP-specific)
    tmd_lane INTEGER, -- 16-bit TMD lane ID (Task/Modifier/Domain)
    store_hint TEXT, -- 'vec', 'graph', 'text', 'hybrid'
    expected_echo_cos REAL, -- Expected Echo-Loop cosine for this task

    -- Tier 1: Lane-specific KPIs (beyond Echo-Loop)
    kpi_formula TEXT, -- JSON: {"test_pass_rate": {">=": 0.9}, "schema_diff": {"==": 0}}

    -- Estimation
    complexity TEXT, -- 'trivial', 'simple', 'moderate', 'complex', 'very_complex'
    estimated_tokens INTEGER,
    estimated_duration_minutes INTEGER,
    estimated_cost_usd REAL,

    -- Lane-specific priors used (JSON)
    estimation_priors TEXT, -- e.g., '{"P5_ms_per_concept": 500, "P7_ms_per_text": 50, "batch_size": 32}'

    -- Execution tracking (links to PAS)
    action_log_id INTEGER, -- Links to action_logs when PAS executes this task
    pas_job_card_id TEXT, -- PAS job card UUID
    status TEXT DEFAULT 'pending', -- 'pending', 'in_progress', 'completed', 'failed', 'blocked'

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_task_id) REFERENCES task_estimates(id)
);
```

### 4.2 Extensions to Existing PAS Tables

```sql
-- Link action_logs to projects
ALTER TABLE action_logs ADD COLUMN project_id INTEGER;

-- New table: project_runs (links PLMS projects to PAS execution runs)
CREATE TABLE project_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    run_id TEXT NOT NULL, -- PAS run UUID
    status TEXT, -- 'running', 'completed', 'failed', 'paused'
    started_at TEXT,
    completed_at TEXT,

    -- Tier 1: Multi-run support + rehearsal + deterministic replay
    run_kind TEXT DEFAULT 'baseline', -- 'baseline', 'rehearsal', 'replay', 'hotfix'
    rehearsal_pct REAL, -- NULL for baseline, 0.01 for 1% canary, etc.
    provider_matrix_json TEXT, -- Snapshot of provider versions (deterministic replay)
    capability_snapshot TEXT, -- Portfolio/capabilities at start
    risk_score REAL, -- Computed at /simulate

    -- Audit trails (links to PAS outputs)
    cost_receipt_path TEXT, -- Path to Provider Router receipts JSON
    heartbeat_ok_ratio REAL, -- % of heartbeats that were healthy
    token_budget_breaches INTEGER DEFAULT 0, -- # of Token Governor budget violations

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Project artifacts (deterministic replay + provenance)
CREATE TABLE project_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL, -- PAS run UUID
    project_id INTEGER NOT NULL,
    artifact_path TEXT NOT NULL,
    artifact_kind TEXT, -- 'prd', 'code', 'config', 'model', 'logs', 'receipt'
    size_bytes INTEGER,
    sha256 TEXT, -- For deterministic replay
    created_at TEXT,

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Project approvals (human-in-the-loop gates)
CREATE TABLE project_approvals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL,
    run_id TEXT, -- NULL for pre-execution approvals (PRD, estimates)
    approver TEXT NOT NULL, -- Username or 'system'
    gate TEXT NOT NULL, -- 'prd_review', 'budget_approval', 'quality_override', 'replan_request'
    status TEXT NOT NULL, -- 'approved', 'rejected', 'pending'
    reason TEXT,
    approved_at TEXT,

    FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
);

-- Tier 1: Lane override feedback (active learning)
CREATE TABLE lane_overrides (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_estimate_id INTEGER NOT NULL,
    task_description TEXT,
    predicted_lane INTEGER,
    corrected_lane INTEGER,
    corrected_by TEXT, -- Username
    corrected_at TEXT,

    FOREIGN KEY (task_estimate_id) REFERENCES task_estimates(id)
);

-- Tier 1: Bayesian estimate versions (calibration)
CREATE TABLE estimate_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER,
    lane_id INTEGER, -- TMD lane (nullable for global priors)
    provider_name TEXT, -- 'openai/gpt-4', 'local/llama-8b', etc.
    created_at TEXT NOT NULL,
    priors_hash TEXT, -- SHA256 of priors JSON (detect changes)

    -- Prior distributions (Bayesian)
    tokens_mean REAL,
    tokens_stddev REAL,
    duration_ms_mean REAL,
    duration_ms_stddev REAL,
    cost_usd_mean REAL,
    cost_usd_stddev REAL,

    -- Error statistics (actual observations)
    n_observations INTEGER DEFAULT 0,
    mean_absolute_error_tokens REAL,
    mean_absolute_error_duration_ms REAL,

    FOREIGN KEY (project_id) REFERENCES projects(id)
);

-- Tier 1: Performance indexes
CREATE INDEX idx_task_estimates_project_status ON task_estimates(project_id, status, order_index);
CREATE INDEX idx_estimate_versions_lane_provider ON estimate_versions(lane_id, provider_name, created_at);
CREATE INDEX idx_lane_overrides_task ON lane_overrides(task_estimate_id);
CREATE INDEX idx_project_runs_kind_time ON project_runs(project_id, run_kind, started_at);
```

---

## 5. API Endpoints

### 5.1 Project Lifecycle

```python
# Create new project from idea
POST /api/projects/create
Body: {
    "name": "Calculator API",
    "description": "Build a simple REST API for basic arithmetic",
    "tags": ["api", "math", "backend"],
    "initial_context": "Need JSON input/output, error handling, unit tests"
}
Response: {"project_id": 42, "status": "ideation"}

# List all projects (with filters)
GET /api/projects?status=executing&tags=api
Response: [{"id": 42, "name": "Calculator API", "status": "executing", ...}]

# Get project details
GET /api/projects/{id}
Response: {
    "id": 42,
    "name": "Calculator API",
    "status": "executing",
    "prd_path": "docs/PRDs/PRD_calculator_api.md",
    "estimated_tokens": 19000,
    "actual_tokens": 17500,
    "budget_tokens_max": 25000,
    "quality_slo": {"min_echo_cos": 0.82},
    "tasks": [...],
    "artifacts": [...],
    "receipts": [...]
}

# Delete/cancel project
DELETE /api/projects/{id}
Response: {"status": "cancelled"}
```

### 5.2 Clarification Phase

```python
# Start clarification (agent asks questions)
POST /api/projects/{id}/clarify/start
Response: {
    "questions": [
        {"id": 1, "text": "What arithmetic operations? (basic, scientific, both)"},
        {"id": 2, "text": "Input format? (JSON, query params, both)"},
        {"id": 3, "text": "Authentication required?"}
    ]
}

# Submit answers
POST /api/projects/{id}/clarify/answer
Body: {
    "answers": [
        {"question_id": 1, "answer": "Basic only (add, subtract, multiply, divide)"},
        {"question_id": 2, "answer": "JSON POST requests"},
        {"question_id": 3, "answer": "No authentication needed"}
    ]
}
Response: {"status": "clarifying", "questions_remaining": 0}

# Finalize clarification (move to PRD generation)
POST /api/projects/{id}/clarify/finalize
Response: {"status": "planning"}
```

### 5.3 PRD & Estimation

```python
# Generate PRD from Q&A
POST /api/projects/{id}/generate-prd
Response: {
    "prd_path": "docs/PRDs/PRD_calculator_api.md",
    "status": "planning"
}

# User edits PRD (manual step via IDE), then triggers estimation

# Generate estimates from PRD
POST /api/projects/{id}/estimate
Response: {
    "status": "estimating",
    "task_tree": [
        {
            "id": 1,
            "name": "Design API schema",
            "tmd_lane": 4201, # Code-API-Design
            "store_hint": "text",
            "complexity": "simple",
            "estimated_tokens": 3000,
            "estimated_duration_minutes": 5,
            "estimated_cost_usd": 0.18
        },
        {
            "id": 2,
            "name": "Implement endpoints",
            "parent_task_id": 1,
            "tmd_lane": 4202, # Code-API-Implementation
            "store_hint": "vec",
            "complexity": "moderate",
            "estimated_tokens": 8000,
            "estimated_duration_minutes": 15,
            "estimated_cost_usd": 0.48
        }
        # ... more tasks
    ],
    "totals": {
        "estimated_tokens": 19000,
        "estimated_duration_minutes": 35,
        "estimated_cost_usd": 1.14
    }
}
```

### 5.4 Approval & Execution

```python
# Approve project (set budget caps, quality SLOs)
POST /api/projects/{id}/approve
Body: {
    "budget_tokens_max": 25000,
    "budget_cost_usd_max": 2.00,
    "breach_policy": "pause",
    "quality_slo": {
        "min_echo_cos": 0.82,
        "lane_gates": {
            "4201": {"min_echo_cos": 0.85}, # Stricter for Code-API-Design
            "4202": {"min_echo_cos": 0.80}
        }
    }
}
Response: {"status": "approved"}

# Start execution (translates to PAS job cards) - Tier 1: Idempotency + Replay Passport
POST /api/projects/{id}/start
Headers:
    Idempotency-Key: <uuid>  # Required for safe retries
Body: {
    "run_kind": "baseline"  # or "rehearsal", "replay", "hotfix"
}
Response: {
    "status": "executing",
    "run_id": "a3f8b2e1-...",
    "provider_matrix": {
        "openai/gpt-4": "gpt-4-0613",
        "local/llama-8b": "llama-3.1-8b-v1.2"
    },
    "replay_passport_path": "artifacts/project-42/replay_passport.json",
    "pas_job_cards": ["card-001", "card-002", ...]
}

# Get execution status
GET /api/projects/{id}/status
Response: {
    "status": "executing",
    "progress": {
        "completed_tasks": 2,
        "total_tasks": 4,
        "current_task": "Implement endpoints",
        "tokens_used": 11200,
        "tokens_budget": 25000,
        "budget_remaining_pct": 55.2
    },
    "quality_slo_status": {
        "min_echo_cos": 0.82,
        "current_mean_cos": 0.87,
        "violations": []
    }
}
```

### 5.5 Lifecycle Controls (V1 MVP)

```python
# Simulate execution - Tier 1: Rehearsal Mode (1% canary)
POST /api/projects/{id}/simulate?rehearsal_pct=0.01&write_sandbox=false
Body: {
    "rehearsal_pct": 0.01,  # Execute 1% of tasks (stratified by complexity)
    "write_sandbox": false  # Suppress DB/artifact writes (V1.5 feature)
}
Response: {
    "simulation_results": {
        "rehearsal_actual": {
            "tokens": 190,
            "duration_ms": 21000,
            "cost_usd": 0.011,
            "echo_cos_mean": 0.84
        },
        "extrapolated_full": {
            "tokens": 19000,
            "duration_ms": 2100000,
            "cost_usd": 1.10,
            "echo_cos_mean": 0.84,
            "ci_90": {
                "tokens": [17100, 20900],
                "cost_usd": [0.99, 1.21]
            }
        },
        "risk_factors": [
            {"lane": 4202, "issue": "high_variance", "samples": 3}
        ]
    }
}

# Pause execution
POST /api/projects/{id}/pause
Response: {"status": "paused", "checkpoint_saved": true}

# Resume execution
POST /api/projects/{id}/resume
Response: {"status": "executing"}

# Request replan (PRD changed, need new estimates)
POST /api/projects/{id}/replan
Body: {"reason": "Added authentication requirement"}
Response: {"status": "estimating", "old_estimate": {...}, "new_estimate": {...}}

# Terminate execution
POST /api/projects/{id}/terminate
Body: {"reason": "Budget exceeded, need to rethink approach"}
Response: {"status": "cancelled"}

# Update budget mid-execution
POST /api/projects/{id}/budget
Body: {"budget_tokens_max": 30000, "budget_cost_usd_max": 2.50}
Response: {"status": "executing", "budget_updated": true}
```

### 5.6 Post-Execution

```python
# Get metrics - Tier 1: With Credible Intervals (Bayesian calibration)
GET /api/projects/{id}/metrics?with_ci=1
Response: {
    "estimated": {
        "tokens": {"mean": 19000, "ci_90": [17100, 20900]},
        "duration_ms": {"mean": 2100000, "ci_90": [1890000, 2310000]},
        "cost_usd": {"mean": 1.14, "ci_90": [1.03, 1.25]}
    },
    "actual": {"tokens": 17500, "duration_ms": 1920000, "cost_usd": 1.05},
    "accuracy": {
        "tokens_in_ci": true,  # Actual within 90% CI
        "duration_ms_in_ci": true,
        "cost_usd_in_ci": true,
        "token_error_pct": 7.9,
        "time_error_pct": 8.6,
        "cost_error_pct": 7.9
    },
    "calibration": {
        "lane_specific_errors": [
            {"lane_id": 4201, "lane_name": "Code-API-Design", "mae_tokens": 150, "mae_pct": 5.3},
            {"lane_id": 4202, "lane_name": "Code-API-Impl", "mae_tokens": 620, "mae_pct": 7.8}
        ]
    },
    "quality_slo": {
        "target_echo_cos": 0.82,
        "actual_mean_cos": 0.87,
        "violations": []
    }
}

# Get receipts ledger (audit trail)
GET /api/projects/{id}/receipts
Response: [
    {
        "task_id": 1,
        "task_name": "Design API schema",
        "provider": "openai/gpt-4",
        "tokens_used": 2800,
        "cost_usd": 0.17,
        "receipt_path": "artifacts/receipts/task-001.json"
    },
    # ... more receipts
]

# Get artifacts
GET /api/projects/{id}/artifacts
Response: [
    {
        "path": "src/calculator_api.py",
        "kind": "code",
        "size_bytes": 4523,
        "sha256": "a3f8b2e1..."
    },
    # ... more artifacts
]

# Get lane overrides - Tier 1: Active Learning Feedback
GET /api/projects/{id}/lane-overrides
Response: [
    {
        "task_id": 2,
        "task_description": "Implement JWT authentication middleware",
        "predicted_lane": 4202,  # Code-API-Implementation
        "corrected_lane": 5301,  # Security-Auth-Implementation
        "corrected_by": "user@example.com",
        "corrected_at": "2025-11-07T14:32:00Z"
    }
]
```

---

## 6. Estimation Algorithm (TMD-Aware, Lane-Specific Priors)

### 6.1 Lane-Specific Priors (From LNSP Measured Bottlenecks)

```python
# P5: LLM Interrogation (CPESH generation, semantic extraction)
P5_MS_PER_CONCEPT = 500  # milliseconds per concept
P5_TOKENS_PER_CONCEPT = 1500  # tokens per concept (input + output)

# P7: GTR-T5 Embeddings (768D vectors)
P7_MS_PER_TEXT = 50  # milliseconds per text (batchable)
P7_BATCH_SIZE = 32  # typical batch size
P7_TOKENS_OVERHEAD = 100  # API overhead per batch

# P15: LNSP Training (if project involves model training)
P15_MS_PER_BATCH = 2000  # milliseconds per training batch
P15_TOKENS_PER_BATCH = 0  # No LLM tokens (GPU work)

# TMD Lane Multipliers (complexity adjustments)
TMD_LANE_MULTIPLIERS = {
    "Math-Derivation": 1.5,  # Math proofs are complex
    "Code-API-Design": 1.2,
    "Code-API-Implementation": 1.0,  # Baseline
    "Narrative-Story": 0.8,  # Simpler generation
    "Data-Ingestion": 0.6,  # Mostly mechanical
}

# Store Hint Overhead (fan-out to multiple stores)
STORE_OVERHEAD = {
    "vec": 1.0,  # Baseline (FAISS only)
    "graph": 1.3,  # Neo4j writes are slower
    "text": 0.8,  # PostgreSQL text is fast
    "hybrid": 1.5,  # Write to all 3 stores
}
```

### 6.2 Estimation Formula

```python
def estimate_task(task: TaskEstimate, priors: Dict) -> Dict:
    """
    Estimate tokens, duration, cost for a single task.

    Args:
        task: Task from PRD decomposition (name, description, tmd_lane, store_hint)
        priors: Lane-specific priors (P5/P7/P15 defaults)

    Returns:
        {"estimated_tokens": int, "estimated_duration_ms": int, "estimated_cost_usd": float}
    """

    # Step 1: Parse task description to infer work type
    work_type = infer_work_type(task.description)  # "llm_generation", "embedding", "training", "mixed"

    # Step 2: Estimate base tokens/time by work type
    if work_type == "llm_generation":
        # Use P5 priors (LLM interrogation)
        n_concepts = estimate_concept_count(task.description)  # Heuristic: count nouns, keywords
        base_tokens = n_concepts * priors["P5_TOKENS_PER_CONCEPT"]
        base_duration_ms = n_concepts * priors["P5_MS_PER_CONCEPT"]

    elif work_type == "embedding":
        # Use P7 priors (embeddings)
        n_texts = estimate_text_count(task.description)
        base_tokens = (n_texts // priors["P7_BATCH_SIZE"]) * priors["P7_TOKENS_OVERHEAD"]
        base_duration_ms = n_texts * priors["P7_MS_PER_TEXT"] / priors["P7_BATCH_SIZE"]

    elif work_type == "training":
        # Use P15 priors (model training)
        n_batches = estimate_batch_count(task.description)
        base_tokens = 0  # No LLM tokens for training
        base_duration_ms = n_batches * priors["P15_MS_PER_BATCH"]

    elif work_type == "mixed":
        # Decompose further or use average multiplier
        base_tokens = 5000  # Conservative default
        base_duration_ms = 10000  # 10 seconds

    # Step 3: Apply TMD lane multiplier
    lane_name = lookup_tmd_lane_name(task.tmd_lane)  # e.g., "Code-API-Implementation"
    lane_multiplier = priors["TMD_LANE_MULTIPLIERS"].get(lane_name, 1.0)
    adjusted_tokens = base_tokens * lane_multiplier
    adjusted_duration_ms = base_duration_ms * lane_multiplier

    # Step 4: Apply store overhead
    store_multiplier = priors["STORE_OVERHEAD"][task.store_hint]
    final_duration_ms = adjusted_duration_ms * store_multiplier

    # Step 5: Calculate cost (assume $0.06 per 1K tokens, blended rate)
    cost_usd = (adjusted_tokens / 1000) * 0.06

    return {
        "estimated_tokens": int(adjusted_tokens),
        "estimated_duration_ms": int(final_duration_ms),
        "estimated_cost_usd": round(cost_usd, 2)
    }
```

### 6.3 Task Tree Decomposition (PRD → Task Tree)

```python
def decompose_prd_to_tasks(prd_text: str) -> List[TaskEstimate]:
    """
    Parse PRD and decompose into hierarchical task tree with TMD lanes.

    Steps:
    1. Extract requirements sections (Objectives, Scope, Requirements, Success Criteria)
    2. Identify high-level tasks (e.g., "Design API", "Implement Endpoints", "Write Tests")
    3. For each high-level task, infer TMD lane based on keywords
    4. Recursively decompose complex tasks into subtasks
    5. Assign order_index via topological sort (dependencies)
    6. Assign store_hint based on task nature (code→vec, docs→text, relations→graph)
    7. Run estimate_task() for each task

    Returns:
        List of TaskEstimate objects (with parent_task_id for hierarchy)
    """

    # Example output for "Calculator API" PRD:
    return [
        TaskEstimate(
            id=1,
            name="Design API schema",
            tmd_lane=4201,  # Code-API-Design
            store_hint="text",
            complexity="simple",
            parent_task_id=None,
            order_index=0
        ),
        TaskEstimate(
            id=2,
            name="Implement endpoints",
            tmd_lane=4202,  # Code-API-Implementation
            store_hint="vec",
            complexity="moderate",
            parent_task_id=1,  # Depends on task 1
            order_index=1
        ),
        # ... subtasks for each endpoint
        TaskEstimate(
            id=3,
            name="Write unit tests",
            tmd_lane=4203,  # Code-Testing
            store_hint="vec",
            complexity="moderate",
            parent_task_id=2,  # Depends on task 2
            order_index=2
        ),
        TaskEstimate(
            id=4,
            name="Generate OpenAPI docs",
            tmd_lane=4204,  # Code-Documentation
            store_hint="text",
            complexity="simple",
            parent_task_id=2,  # Depends on task 2
            order_index=3
        )
    ]
```

### 6.4 Uncertainty & Calibration (Bayesian Estimator) - Tier 1

**Problem**: Point estimates drift without feedback. After 10 projects, estimates should improve.

**Solution**: Bayesian calibration loop that updates lane- and provider-specific priors from PAS receipts.

#### Calibration Loop

```python
def update_priors_after_run(project_id: int, run_id: str):
    """
    After PAS run completes, update Bayesian priors for each lane/provider used.

    Steps:
    1. Load actual metrics from receipts (tokens, duration, cost per task)
    2. Load estimated metrics from task_estimates
    3. For each (lane, provider) pair:
       a. Compute error: delta = actual - estimated
       b. Update prior using exponential smoothing: new_prior = old_prior * (1 - α) + actual * α
       c. Update variance: track stddev for credible intervals
       d. Store in estimate_versions table
    4. Next project using this (lane, provider) gets updated priors
    """

    α = 0.3  # Learning rate (tune based on variance)

    for task in get_completed_tasks(project_id):
        lane_id = task.tmd_lane
        provider = get_provider_from_receipt(task.receipt_path)

        # Load current prior
        prior = get_latest_prior(lane_id, provider)

        # Compute error
        delta_tokens = task.actual_tokens - task.estimated_tokens
        delta_duration = task.actual_duration_ms - task.estimated_duration_ms

        # Update mean (exponential smoothing)
        new_tokens_mean = prior.tokens_mean * (1 - α) + task.actual_tokens * α
        new_duration_mean = prior.duration_ms_mean * (1 - α) + task.actual_duration_ms * α

        # Update variance (Welford's online algorithm)
        new_tokens_variance = update_variance(
            prior.tokens_stddev**2,
            prior.n_observations,
            task.actual_tokens,
            new_tokens_mean
        )

        # Store new prior
        store_prior(
            lane_id=lane_id,
            provider=provider,
            tokens_mean=new_tokens_mean,
            tokens_stddev=sqrt(new_tokens_variance),
            duration_ms_mean=new_duration_mean,
            n_observations=prior.n_observations + 1,
            mae_tokens=abs(delta_tokens)
        )
```

#### Credible Intervals (90% CI)

```python
def estimate_task_with_uncertainty(task: TaskEstimate, priors: Dict) -> Dict:
    """
    Estimate with calibrated uncertainty (90% credible intervals).

    Returns:
        {
            "tokens": {"mean": 3000, "ci_90": [2700, 3300]},
            "duration_ms": {"mean": 300000, "ci_90": [270000, 330000]},
            "cost_usd": {"mean": 0.18, "ci_90": [0.16, 0.20]}
        }
    """

    # Get Bayesian prior for this (lane, provider)
    prior = get_prior(task.tmd_lane, task.provider)

    # Point estimate (same as §6.2)
    tokens_mean = compute_base_estimate(task, priors)

    # Credible interval (±1.645 stddev for 90% CI)
    tokens_stddev = prior.tokens_stddev
    tokens_ci_90 = [
        tokens_mean - 1.645 * tokens_stddev,
        tokens_mean + 1.645 * tokens_stddev
    ]

    return {
        "tokens": {"mean": tokens_mean, "ci_90": tokens_ci_90},
        # ... same for duration, cost
    }
```

#### Cold-Start Problem (New Lanes)

For new lanes with no observations:
- Use global priors (average across all lanes)
- High uncertainty (wide CIs: ±50%)
- After 3 observations, switch to lane-specific priors
- After 10 observations, CIs narrow to ±20%

---

## 7. Echo-Loop Quality SLOs

### 7.1 What is Echo-Loop?

**Echo-Loop** is LNSP's semantic validation technique:
1. Encode concept text → 768D vector (via GTR-T5)
2. Decode vector → reconstructed text (via vec2text IELab/JXE)
3. Re-encode reconstructed text → 768D vector
4. Compute cosine similarity: `cos(original_vector, re_encoded_vector)`

**Threshold**: Projects should achieve **≥0.82 cosine** mean across all concepts. Lower scores indicate semantic drift or low-quality embeddings.

### 7.2 Project-Level Quality SLOs

```python
class QualitySLO:
    """
    Quality SLO definition for a project.
    """
    min_echo_cos: float = 0.82  # Global minimum (default)
    lane_gates: Dict[int, float] = {}  # TMD lane-specific overrides

    # Example:
    # {
    #     4201: 0.85,  # Code-API-Design requires stricter SLO
    #     4202: 0.80,  # Code-API-Implementation can be looser
    # }
```

### 7.3 Validation Phase (After Execution)

```python
def validate_project_quality(project_id: int) -> ValidationResult:
    """
    After PAS execution completes, validate all produced concepts against Echo-Loop SLOs.

    Steps:
    1. Retrieve all concepts produced during execution (from action_logs)
    2. For each concept:
       a. Load vector from FAISS/NPZ
       b. Run Echo-Loop (encode→decode→re-encode)
       c. Compute cosine similarity
    3. Group by TMD lane
    4. Check: mean_cos ≥ min_echo_cos (global and per-lane)
    5. If violations found: status → 'needs_review'
    6. If all pass: status → 'completed'

    Returns:
        ValidationResult(
            passed: bool,
            mean_echo_cos: float,
            violations: List[Violation]  # [(task_id, lane, actual_cos, expected_cos)]
        )
    """
    pass
```

### 7.4 Lane-Specific KPIs (Beyond Echo-Loop) - Tier 1

**Problem**: Echo-Loop cosine ≥0.82 is necessary but insufficient. Code needs test pass rates, Data needs schema diffs, Narrative needs coherence.

**Solution**: Per-lane KPI library that extends quality SLOs.

#### KPI Library (V1 MVP)

| Lane | KPI Name | Formula | Threshold | Validator |
|------|----------|---------|-----------|-----------|
| Code-* | test_pass_rate | passed_tests / total_tests | ≥0.90 | `code_tests(artifact_path)` |
| Code-* | linter_pass | lint_errors == 0 | true | `code_linter(artifact_path)` |
| Data-* | schema_diff | diff(expected_schema, actual_schema) | == 0 | `schema_diff(expected, actual)` |
| Data-* | row_count_delta | abs(actual_rows - expected_rows) / expected_rows | ≤0.05 | `row_count_check(table)` |
| Graph-* | edge_count_delta | abs(actual_edges - expected_edges) / expected_edges | ≤0.10 | `graph_edge_count(neo4j)` |
| Narrative-* | BLEU_score | BLEU(generated, reference) | ≥0.40 | `bleu_score(gen, ref)` |
| Narrative-* | readability | Flesch-Kincaid grade level | ≤12 | `readability(text)` |

#### Quality SLO with Lane KPIs

```python
class QualitySLO:
    """
    Extended quality SLO with lane-specific KPIs.
    """
    min_echo_cos: float = 0.82  # Global Echo-Loop threshold

    lane_gates: Dict[int, Dict] = {
        4201: {  # Code-API-Design
            "min_echo_cos": 0.85,  # Stricter Echo-Loop
            "kpis": [
                {"name": "test_pass_rate", "threshold": 0.90, "operator": ">="},
                {"name": "linter_pass", "threshold": True, "operator": "=="}
            ]
        },
        5100: {  # Data-Schema
            "min_echo_cos": 0.80,  # Looser Echo-Loop (structure matters more)
            "kpis": [
                {"name": "schema_diff", "threshold": 0, "operator": "=="},
                {"name": "row_count_delta", "threshold": 0.05, "operator": "<="}
            ]
        }
    }
```

#### Validation Phase (Extended)

```python
def validate_project_quality(project_id: int) -> ValidationResult:
    """
    Extended validation: Echo-Loop + lane-specific KPIs.

    Steps:
    1. Run Echo-Loop validation (existing §7.3)
    2. For each completed task:
       a. Load lane-specific KPIs from quality_slo_gates
       b. Run KPI validators on artifacts
       c. Check: actual_value meets threshold
    3. Aggregate violations by lane
    4. If critical violations: status → 'needs_review'
    5. If all pass: status → 'completed'
    """

    violations = []

    for task in get_completed_tasks(project_id):
        lane_gates = get_lane_gates(task.tmd_lane)

        # Check Echo-Loop
        echo_cos = compute_echo_loop(task.concept_vectors)
        if echo_cos < lane_gates.get("min_echo_cos", 0.82):
            violations.append({
                "task_id": task.id,
                "kpi": "echo_cos",
                "actual": echo_cos,
                "expected": lane_gates["min_echo_cos"]
            })

        # Check lane-specific KPIs
        for kpi in lane_gates.get("kpis", []):
            validator = KPI_VALIDATORS[kpi["name"]]
            actual_value = validator(task.artifacts)

            if not check_threshold(actual_value, kpi["threshold"], kpi["operator"]):
                violations.append({
                    "task_id": task.id,
                    "kpi": kpi["name"],
                    "actual": actual_value,
                    "expected": kpi["threshold"]
                })

    return ValidationResult(
        passed=(len(violations) == 0),
        violations=violations
    )
```

#### KPI Validators (Implementation)

```python
# Ship as: services/plms/kpi_validators.py

def code_tests(artifact_path: str) -> float:
    """Run pytest, return pass rate."""
    result = subprocess.run(["pytest", artifact_path, "--json-report"], capture_output=True)
    report = json.loads(result.stdout)
    return report["summary"]["passed"] / report["summary"]["total"]

def schema_diff(expected_schema: Dict, actual_table: str) -> int:
    """Compare expected vs actual schema, return diff count."""
    actual_schema = get_table_schema(actual_table)  # Query PostgreSQL
    return len(DeepDiff(expected_schema, actual_schema))

def graph_edge_count(neo4j_query: str) -> int:
    """Count edges in Neo4j graph."""
    result = run_cypher(neo4j_query)
    return result[0]["edge_count"]

def bleu_score(generated: str, reference: str) -> float:
    """Compute BLEU score (MT evaluation metric)."""
    from nltk.translate.bleu_score import sentence_bleu
    return sentence_bleu([reference.split()], generated.split())
```

---

## 8. HMI Integration (Overlays on Existing Views)

### 8.1 Existing HMI Views (DO NOT DUPLICATE)

- **Tree View** (`/tree`): Hierarchical agent execution tree
- **Sequencer** (`/sequencer`): Timeline view of agent activities
- **Audio Control** (`/audio`): Audio playback controls
- **Actions View** (`/actions`): Task flow logging (NEW as of Nov 6, 2025)

### 8.2 New: Project Overlays (Minimal UI Changes)

#### A. Projects Dashboard (`/projects`)

**New standalone view** (list all projects):

```html
<!-- Reuses existing Bootstrap theme, card layout -->
<div class="container">
  <h1>Projects</h1>
  <div class="filters">
    Status: [All | Executing | Completed]
    Tags: [api, ml, backend, ...]
  </div>

  <div class="project-cards">
    <div class="card">
      <h3>Calculator API</h3>
      <span class="badge">Executing</span>
      <p>Estimated: 19K tokens, $1.14 | Actual: 11K tokens, $0.66 (so far)</p>
      <div class="progress-bar">
        <div style="width: 60%"></div> <!-- 60% complete -->
      </div>
      <a href="/projects/42">View Details</a>
    </div>
    <!-- More project cards -->
  </div>
</div>
```

#### B. Project Detail View (`/projects/{id}`)

**New standalone view** (tabs for Overview, Planning, Execution, Metrics):

```html
<div class="container">
  <h1>Project: Calculator API</h1>
  <span class="badge badge-info">Executing</span>

  <ul class="nav nav-tabs">
    <li class="active"><a href="#overview">Overview</a></li>
    <li><a href="#planning">Planning</a></li>
    <li><a href="#execution">Execution</a></li>
    <li><a href="#metrics">Metrics</a></li>
  </ul>

  <div id="overview" class="tab-pane active">
    <h3>Summary</h3>
    <p><strong>Description:</strong> Build a simple REST API for basic arithmetic</p>
    <p><strong>PRD:</strong> <a href="/PRDs/PRD_calculator_api.md">View PRD</a></p>

    <h3>Budget</h3>
    <div class="budget-gauge">
      Tokens: 11,200 / 25,000 (44.8% used)
      Cost: $0.66 / $2.00 (33% used)
    </div>

    <h3>Quality SLOs</h3>
    <p>Min Echo-Loop Cosine: 0.82</p>
    <p>Current Mean: 0.87 ✓</p>
  </div>

  <div id="planning" class="tab-pane">
    <h3>Q&A History</h3>
    <ul>
      <li><strong>Q:</strong> What operations? <strong>A:</strong> Basic only</li>
      <li><strong>Q:</strong> Input format? <strong>A:</strong> JSON POST</li>
    </ul>

    <h3>Task Tree</h3>
    <!-- Hierarchical task tree (similar to Actions view) -->
  </div>

  <div id="execution" class="tab-pane">
    <!-- Embed existing Actions view, filtered by project_id -->
    <iframe src="/actions?project_id=42" style="width:100%; height:800px;"></iframe>
  </div>

  <div id="metrics" class="tab-pane">
    <h3>Actual vs Estimated</h3>
    <table>
      <tr><th>Metric</th><th>Estimated</th><th>Actual</th><th>Error %</th></tr>
      <tr><td>Tokens</td><td>19,000</td><td>17,500</td><td>7.9%</td></tr>
      <tr><td>Duration (min)</td><td>35</td><td>32</td><td>8.6%</td></tr>
      <tr><td>Cost (USD)</td><td>$1.14</td><td>$1.05</td><td>7.9%</td></tr>
    </table>

    <!-- Chart: Estimated vs Actual (bar chart) -->
    <canvas id="metricsChart"></canvas>
  </div>
</div>
```

#### C. Enhanced Actions View (`/actions?project_id={id}`)

**Extend existing Actions view** (minimal changes):

```html
<!-- Add project filter to existing view -->
<div class="filters">
  Project:
  <select id="project-filter">
    <option value="">All Projects</option>
    <option value="42" selected>Calculator API</option>
    <!-- More projects -->
  </select>
</div>

<!-- Existing action tree, filtered by project_id -->
```

#### D. Navigation (Add "Projects" Link to Existing Navbar)

```html
<!-- Extend existing HMI navbar -->
<nav>
  <a href="/tree">Tree</a>
  <a href="/sequencer">Sequencer</a>
  <a href="/audio">Audio</a>
  <a href="/actions">Actions</a>
  <a href="/projects">Projects</a> <!-- NEW -->
</nav>
```

### 8.3 Risk Visualization (Budget Runway + Lane Heatmap) - Tier 1

**Problem**: Users see current spend but not "when will budget run out?" Can't see which lanes/phases are high-risk before starting.

**Solution**: Budget runway gauge + lane risk heatmap.

#### Budget Runway Gauge

```html
<div class="budget-runway">
  <h4>Budget Runway</h4>
  <div class="gauge">
    <div class="gauge-bar" style="width: 55%"></div>
    <span class="gauge-label">$1.10 / $2.00 spent</span>
  </div>

  <div class="projection">
    <strong>Projected Depletion:</strong> 12 minutes
    <span class="warning">⚠️ High burn rate</span>
  </div>

  <div class="breakdown">
    <p>Current rate: $0.083/min</p>
    <p>Remaining budget: $0.90</p>
    <p>Estimated completion: $1.42 (⚠️ 29% over budget)</p>
  </div>
</div>
```

#### Lane Risk Heatmap

```html
<div class="risk-heatmap">
  <h4>Risk Heatmap (Lane × Phase)</h4>
  <table>
    <thead>
      <tr>
        <th>Lane</th>
        <th>Estimation</th>
        <th>Execution</th>
        <th>Validation</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Code-API (4200)</td>
        <td class="risk-low">✓ Low</td>
        <td class="risk-medium">⚠️ Medium</td>
        <td class="risk-low">✓ Low</td>
      </tr>
      <tr>
        <td>Data-Schema (5100)</td>
        <td class="risk-high">⚠️ High</td>
        <td class="risk-medium">⚠️ Medium</td>
        <td class="risk-high">⚠️ High</td>
      </tr>
    </tbody>
  </table>

  <div class="legend">
    <span class="risk-low">✓ Low: MAE &lt; 15%, CI width &lt; 30%</span>
    <span class="risk-medium">⚠️ Medium: MAE 15-30%, CI width 30-50%</span>
    <span class="risk-high">⚠️ High: MAE &gt; 30%, CI width &gt; 50%</span>
  </div>
</div>
```

#### Estimation Drift Sparkline (Per Lane)

```html
<div class="estimation-drift">
  <h4>Estimation Drift (Code-API Lane)</h4>
  <canvas id="drift-sparkline-4200"></canvas>
  <p>Last 10 projects: MAE 22% → 18% → 15% → 12% (✓ improving)</p>
</div>

<script>
// Chart.js sparkline
new Chart(document.getElementById('drift-sparkline-4200'), {
  type: 'line',
  data: {
    labels: ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10'],
    datasets: [{
      label: 'MAE %',
      data: [22, 18, 15, 19, 14, 12, 11, 13, 10, 12],
      borderColor: 'green',
      tension: 0.4
    }]
  }
});
</script>
```

#### HMI JSON Contracts (For Implementation)

**Budget Runway** (GET `/api/projects/{id}/budget-runway`):
```json
{
  "budget": {"usd_max": 2.00, "usd_spent": 1.10, "burn_per_min": 0.083},
  "runway": {"minutes_to_depletion": 12.0, "projected_overrun_usd": 0.32},
  "status": "warning"  // "ok" | "warning" | "critical"
}
```

**Risk Heatmap** (GET `/api/projects/{id}/risk-heatmap`):
```json
{
  "lanes": [
    {
      "lane_id": 4200,
      "name": "Code-API",
      "estimation_risk": "low",
      "execution_risk": "medium",
      "validation_risk": "low",
      "signals": {"mae_pct": 0.12, "ci_width_pct": 0.28}
    },
    {
      "lane_id": 5100,
      "name": "Data-Schema",
      "estimation_risk": "high",
      "execution_risk": "medium",
      "validation_risk": "high",
      "signals": {"mae_pct": 0.37, "ci_width_pct": 0.55}
    }
  ],
  "legend": {
    "low":    {"mae_pct_lt": 0.15, "ci_width_lt": 0.30},
    "medium": {"mae_pct_le": 0.30, "ci_width_le": 0.50},
    "high":   {"else": true}
  }
}
```

---

## 9. Agent Specifications

### 9.1 Clarification Agent

**Purpose**: Ask targeted questions to gather missing requirements from PRD.

**Inputs**:
- Project ID
- Initial description
- Tags/context

**Outputs**:
- List of questions (3-10 questions)
- Stores in `project_questions` table

**Algorithm**:
1. Parse initial description, extract entities/keywords
2. Identify missing dimensions: scope, constraints, success criteria, non-functional requirements
3. Generate targeted questions using LLM (template-based)
4. Interactive Q&A loop until: sufficient detail OR user signals "done"

**Example Questions**:
- "What is the expected input format? (JSON, CSV, XML, other)"
- "What error handling is required? (validation, retries, logging)"
- "What are the performance requirements? (latency, throughput)"
- "Are there authentication/authorization needs?"

### 9.2 PRD Generation Agent

**Purpose**: Synthesize Q&A into structured PRD document.

**Inputs**:
- Project ID
- Q&A history (from `project_questions`)
- Initial description

**Outputs**:
- PRD markdown file (`docs/PRDs/PRD_{project_name}.md`)
- Updates `projects.prd_path`

**PRD Template**:
```markdown
# PRD: {Project Name}

## Objective
{1-2 sentence goal}

## Scope
- In scope: {features/capabilities}
- Out of scope: {explicitly excluded}

## Requirements
### Functional Requirements
1. {Requirement 1}
2. {Requirement 2}

### Non-Functional Requirements
- Performance: {latency, throughput targets}
- Reliability: {uptime, error rate targets}
- Security: {auth, encryption, compliance}

## Success Criteria
- {Measurable outcome 1}
- {Measurable outcome 2}

## Dependencies
- {External systems, libraries, services}

## Risks & Mitigations
- Risk: {description} → Mitigation: {strategy}
```

### 9.3 Estimation Agent

**Purpose**: Decompose PRD into task tree with TMD lanes and lane-specific estimates.

**Inputs**:
- Project ID
- PRD text

**Outputs**:
- Task tree (stored in `task_estimates`)
- Totals: estimated_tokens, estimated_duration_minutes, estimated_cost_usd

**Algorithm**:
1. Parse PRD sections (Requirements, Success Criteria)
2. Identify high-level tasks (e.g., "Design", "Implement", "Test", "Document")
3. For each task:
   a. Infer TMD lane (keyword matching: "API" → Code-API, "database" → Data-Schema)
   b. Assign store hint (code/config → vec, docs → text, relations → graph)
   c. Estimate complexity (trivial/simple/moderate/complex/very_complex)
   d. Run `estimate_task()` with lane-specific priors
4. Build task tree (parent_task_id for hierarchy)
5. Topological sort for order_index (dependencies)
6. Sum totals across all leaf tasks

**TMD Lane Inference** (keyword heuristics):
- "API", "REST", "endpoint" → Code-API (lane 4200-4299)
- "database", "schema", "SQL" → Data-Schema (lane 5100-5199)
- "algorithm", "computation" → Math-Algorithm (lane 1100-1199)
- "narrative", "story", "summary" → Narrative-Story (lane 6100-6199)

### 9.4 PAS Job Card Generator

**Purpose**: Translate task tree into PAS job cards for execution.

**Inputs**:
- Project ID
- Task tree (from `task_estimates`)
- Budget caps (from `projects`)

**Outputs**:
- PAS job cards (JSON submitted to PAS Architect)
- Run ID (from PAS)
- Updates `project_runs` table

**Job Card Schema** (PAS-compatible):
```json
{
  "run_id": "a3f8b2e1-4c7d-4e9f-8b2a-1f6d3e5c9a2b",
  "project_id": 42,
  "tasks": [
    {
      "task_id": "task-001",
      "task_name": "Design API schema",
      "tmd_lane": 4201,
      "store_hint": "text",
      "budget_tokens": 3000,
      "estimated_duration_ms": 300000,
      "dependencies": []
    },
    {
      "task_id": "task-002",
      "task_name": "Implement endpoints",
      "tmd_lane": 4202,
      "store_hint": "vec",
      "budget_tokens": 8000,
      "estimated_duration_ms": 900000,
      "dependencies": ["task-001"]
    }
  ],
  "global_budget": {
    "max_tokens": 25000,
    "max_cost_usd": 2.00,
    "breach_policy": "pause"
  },
  "quality_slo": {
    "min_echo_cos": 0.82,
    "lane_gates": {
      "4201": 0.85
    }
  }
}
```

---

## 10. Future Enhancements (V2+)

### 10.1 Semantic Critical Path (SCP)

**Concept**: Compute project "critical path" based on concept graph centrality, not just task durations.

**Algorithm**:
1. Extract all concepts mentioned in PRD
2. Query Neo4j for relations between concepts (6-degree graph)
3. Compute centrality (PageRank, betweenness) for each concept
4. Prioritize tasks that produce/consume high-centrality concepts
5. Reschedule tasks to minimize critical path length

**Use Case**: In a "Build Recommendation Engine" project, prioritize tasks that produce "user embeddings" and "item embeddings" (high centrality) before "UI rendering" (low centrality).

### 10.2 Rehearsal Mode

**Concept**: Run 1-5% of tasks end-to-end, measure quality SLOs, then scale to full execution.

**Steps**:
1. User triggers `/simulate` with `rehearsal_pct=0.05`
2. PLMS samples 5% of tasks (stratified by complexity)
3. Execute sampled tasks via PAS
4. Measure: actual tokens/time/cost, Echo-Loop cosine, error rates
5. Extrapolate to full project: `full_cost = sample_cost / 0.05`
6. Present variance/confidence intervals to user
7. User approves or requests adjustments

**Benefits**:
- Catch estimation errors early (before committing full budget)
- Discover quality issues in task definitions
- Reduce risk of runaway costs

### 10.3 PRD-Delta Watcher

**Concept**: Detect PRD changes mid-execution, auto-compute task deltas, trigger replan.

**Steps**:
1. Watch `docs/PRDs/PRD_{project_name}.md` for file changes (inotify)
2. On change detected:
   a. Parse new PRD
   b. Re-run Estimation Agent
   c. Diff: old task tree vs new task tree
   d. Identify: added tasks, removed tasks, modified tasks
3. Pause execution if critical tasks changed
4. Show human diff view: "3 tasks added, 1 task removed, 2 tasks modified"
5. User approves replan or reverts PRD
6. Resume execution with updated task tree

**Benefits**:
- Agile response to changing requirements
- No need to cancel/restart entire project

### 10.4 Deterministic Replay

**Concept**: Replay any project run byte-for-byte to a known commit + provider matrix.

**Requirements**:
1. Store in `project_artifacts`:
   - Git commit SHA
   - Provider versions (model names, API versions)
   - All input files (checksums)
   - All intermediate artifacts (checksums)
2. Replay command: `plms replay --run-id a3f8b2e1 --verify-checksums`
3. Re-execute all tasks with identical inputs
4. Verify: output checksums match original run

**Benefits**:
- Audit compliance (prove decisions)
- Regression testing (detect model drift)
- Debugging (reproduce failures exactly)

### 10.5 Lane-Specialist Routing by TMD

**Concept**: Route each task to a tiny specialist agent per TMD lane (Math-Derivation, Code-API, Med-Guideline, etc.).

**Implementation**:
1. Extend PAS Director roster with lane-specific agents:
   - `DirectorCodeAPI` (lane 4200-4299)
   - `DirectorMathDerivation` (lane 1100-1199)
   - `DirectorNarrativeStory` (lane 6100-6199)
2. PAS Architect routes tasks by TMD lane → specialist Director
3. Specialist Directors use lane-tuned prompts, tools, models
4. Estimation Agent uses specialist-specific priors (more accurate)

**Benefits**:
- Better estimates (specialists know lane bottlenecks)
- Better quality (specialists use lane-tuned prompts)
- Scalability (add new lanes without retraining Architect)

### 10.6 Budget-Aware Replanning

**Concept**: When Token Governor projects overrun, auto-replan: downshift providers or split tasks.

**Triggers**:
1. Token Governor detects: `tokens_used + projected_remaining > budget_tokens_max`
2. Signal to PLMS: `budget_breach_imminent`
3. PLMS pauses execution, runs Estimation Agent on remaining tasks
4. Replanning strategies:
   a. **Downshift provider**: Use cheaper model (e.g., Sonnet → local Llama 8B) for low-risk tasks
   b. **Split tasks**: Break large tasks into smaller batches (reduce per-task budget)
   c. **Skip optional tasks**: Mark "nice-to-have" tasks as skipped
5. Present replan options to user
6. User approves replan or increases budget

**Benefits**:
- Graceful degradation (finish project within budget)
- No need to cancel entire project

### 10.7 SLA-Driven Provider Picks

**Concept**: Add SLA fields (p95 latency, reliability) to task estimates; Provider Router chooses cheapest model that meets SLA.

**Example**:
```python
TaskEstimate(
    name="Generate API docs",
    sla={"p95_latency_ms": 5000, "reliability": 0.99},
    estimated_tokens=2000
)

# Provider Router logic:
# - GPT-4: p95=2000ms, reliability=0.999, cost=$0.12 → MEETS SLA
# - Llama 8B: p95=1000ms, reliability=0.95, cost=$0.02 → FAILS SLA (reliability)
# → Choose GPT-4 (cheapest that meets SLA)
```

**Benefits**:
- Cost optimization (don't overpay for performance you don't need)
- Quality guarantees (SLA violations tracked)

### 10.8 Hybrid Store Hints (Automatic)

**Concept**: Auto-infer store hints from TMD lane + task type.

**Heuristics**:
- Code/Config → `vec` (FAISS for similarity search)
- Docs/Narrative → `text` (PostgreSQL full-text search)
- Relations/Ontology → `graph` (Neo4j traversal)
- Multi-modal → `hybrid` (all 3 stores)

**Estimation Impact**:
- `vec`: 1.0x duration (baseline)
- `graph`: 1.3x duration (Neo4j writes slower)
- `text`: 0.8x duration (PostgreSQL faster)
- `hybrid`: 1.5x duration (fan-out to 3 stores)

### 10.9 Echo-Loop per Task (Fine-Grained Quality)

**Concept**: Run Echo-Loop validation after each task completes, not just at project end.

**Benefits**:
- Early detection of quality issues (stop execution before compounding errors)
- Per-task quality metrics (identify weak agents)
- Adaptive SLOs (relax/tighten based on task criticality)

### 10.10 Project Templates

**Concept**: Pre-defined project templates for common workflows (e.g., "Build REST API", "Train ML Model", "Ingest Ontology").

**Template Schema**:
```json
{
  "template_name": "Build REST API",
  "description": "Standard workflow for designing/implementing a REST API",
  "default_questions": [
    "What operations does the API support?",
    "What is the input/output format?",
    "Are there authentication requirements?"
  ],
  "task_tree_template": [
    {
      "name": "Design API schema",
      "tmd_lane": 4201,
      "complexity": "simple"
    },
    {
      "name": "Implement endpoints",
      "tmd_lane": 4202,
      "complexity": "moderate"
    }
  ],
  "default_slo": {"min_echo_cos": 0.82}
}
```

**Benefits**:
- Faster project creation (skip ideation/clarification for common patterns)
- Better estimates (templates use historical data)
- Consistency (enforce best practices)

---

## 11. Testing & Validation (V1 MVP)

### 11.1 Test Projects (3 Required for MVP)

1. **Test Project 1: Calculator API** (simple, well-scoped)
   - Objective: Build REST API for basic arithmetic
   - Operations: add, subtract, multiply, divide
   - Expected: ~19K tokens, ~35 minutes, ~$1.14
   - Success: Estimation error ≤30%, Echo-Loop ≥0.82

2. **Test Project 2: Ontology Ingestion** (LNSP-native, moderate complexity)
   - Objective: Ingest 1,000 concepts from SWO ontology
   - Tasks: Parse RDF, extract triples, generate CPESH, compute TMD, encode vectors
   - Expected: ~500K tokens, ~120 minutes, ~$30
   - Success: Estimation error ≤30%, Echo-Loop ≥0.82, data sync verified

3. **Test Project 3: Narrative Summarization** (high variability, stress test)
   - Objective: Summarize 50 Wikipedia articles into 100-word summaries
   - Tasks: Fetch articles, extract key facts, generate summaries, validate coherence
   - Expected: ~250K tokens, ~60 minutes, ~$15
   - Success: Estimation error ≤40% (higher variance expected), Echo-Loop ≥0.80

### 11.2 Acceptance Criteria

- [ ] All 3 test projects complete full lifecycle (ideation → completion)
- [ ] Mean estimation error ≤30% across all 3 projects (tokens + time)
- [ ] Echo-Loop SLOs enforced (projects block on violations)
- [ ] PAS integration: All execution uses job cards, receipts visible
- [ ] HMI overlays: Project badges, budget gauge, SLO status visible
- [ ] No data loss: All receipts, heartbeats, artifacts stored
- [ ] Lifecycle controls work: /pause, /resume, /terminate functional

---

## 12. Implementation Roadmap

### Phase 1: Foundation (Week 1) - Tier 1 Enhanced
- [ ] Create database tables (projects, project_questions, task_estimates, project_runs, project_artifacts, project_approvals)
- [ ] **Tier 1**: Add estimate_versions, lane_overrides tables (Bayesian calibration + active learning)
- [ ] **Tier 1**: Extend project_runs with run_kind, rehearsal_pct, provider_matrix_json (multi-run support)
- [ ] **Tier 1**: Extend task_estimates with kpi_formula (lane-specific KPIs)
- [ ] Extend `action_logs` with `project_id` column
- [ ] Implement Project API endpoints (CRUD, lifecycle controls)
- [ ] **Tier 1**: Add Idempotency-Key support to POST /start
- [ ] **Tier 1**: Add GET /metrics?with_ci=1 (credible intervals)
- [ ] **Tier 1**: Add POST /simulate?rehearsal_pct=0.01 (1% canary)
- [ ] **Tier 1**: Add GET /lane-overrides (active learning feedback)
- [ ] Add Projects Dashboard UI (list view, filters)
- [ ] **Tier 1**: Add budget runway gauge + risk heatmap to Project Detail view

### Phase 2: Planning Agents (Week 2)
- [ ] Implement Clarification Agent (Q&A loop)
- [ ] Implement PRD Generation Agent (Q&A → PRD)
- [ ] Implement Estimation Agent (PRD → task tree with TMD lanes)
- [ ] Test: Manual flow through ideation → PRD → estimation

### Phase 3: PAS Integration (Week 3)
- [ ] Implement PAS Job Card Generator (task tree → job cards)
- [ ] Submit job cards to PAS Architect, track run_id
- [ ] Poll PAS status, update project_runs table
- [ ] Wire receipts, heartbeats, artifacts to project tables
- [ ] Test: End-to-end execution with Test Project 1 (Calculator API)

### Phase 4: Quality SLOs (Week 4)
- [ ] Implement Echo-Loop validation (per-task and project-level)
- [ ] Add quality_slo enforcement (block completion on violations)
- [ ] Add HMI SLO indicators (current vs target cosine)
- [ ] Test: Run Test Project 2 (Ontology Ingestion) with SLO gates

### Phase 5: HMI & Polish (Week 5)
- [ ] Add Project Detail view (tabs: Overview, Planning, Execution, Metrics)
- [ ] Add project badges/overlays to Tree/Sequencer/Actions
- [ ] Add Actual vs Estimated charts (metrics view)
- [ ] Add Receipts Ledger view (audit trail)
- [ ] Test: Run Test Project 3 (Narrative Summarization)

### Phase 6: Validation & Docs (Week 6)
- [ ] Run all 3 test projects, measure accuracy
- [ ] Document: API reference, agent specs, SLO tuning guide
- [ ] Record: Demo video (idea → completion)
- [ ] Write: Migration guide (how to adopt PLMS for existing PAS users)

---

## 13. Risks & Mitigations

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| **Estimation drift** (actual >> estimated) | High | Medium | Use conservative priors, add 20% buffer, tune after 10 projects |
| **PAS integration bugs** (job cards malformed) | High | Low | Validate job card schema, add integration tests |
| **Echo-Loop SLO too strict** (projects never pass) | Medium | Medium | Start with 0.75 threshold, tune up to 0.82 after data |
| **HMI clutter** (too many overlays) | Low | Medium | Minimize new UI, use existing views with filters |
| **TMD lane inference errors** (wrong lane → bad estimates) | Medium | Medium | Manual override in UI, log mismatches for retraining |
| **User fatigue** (too many questions in clarification) | Low | High | Limit to 5 questions max for V1, use smart defaults |

---

## 14. Open Questions

1. **How to handle PRD edits mid-execution?**
   - V1: Reject edits (block PRD file), require cancel/replan
   - V2: PRD-Delta Watcher (auto-replan on changes)

2. **Should estimation be interactive?**
   - V1: Fully automated (agent decomposes PRD)
   - V2: Show task tree to user, allow manual adjustments

3. **What if user disagrees with task tree?**
   - V1: Allow manual edits in UI before approval
   - V2: Agent learns from edits (feedback loop)

4. **Should PLMS support multi-run projects?**
   - V1: One run per project (complete → archive)
   - V2: Allow re-runs (e.g., retrain model with new data)

5. **How to handle concurrent projects?**
   - V1: No explicit limit (Token Governor enforces global quotas)
   - V2: Add priority queue, user sets project priority

---

## 15. Appendix

### A. Glossary

- **PLMS**: Project Lifecycle Management System
- **PAS**: Polyglot Agent Swarm (existing execution engine)
- **TMD**: Task/Modifier/Domain (16-bit lane ID in LNSP)
- **Echo-Loop**: LNSP semantic validation (encode→decode→re-encode, measure cosine)
- **SLO**: Service Level Objective (quality gate)
- **Receipt**: Provider Router audit record (model used, tokens, cost)
- **Heartbeat**: Registry health check (agent liveliness)

### B. References

- **PAS PRD**: `docs/PRDs/PRD_Polyglot_Agent_Swarm.md`
- **HMI PRD**: `docs/PRDs/PRD_Human_Machine_Interface_HMI.md`
- **LNSP Long-Term Memory**: `LNSP_LONG_TERM_MEMORY.md`
- **Actions View Session**: `docs/SESSION_SUMMARY_2025_11_06_ACTIONS_VIEW.md`
- **Database Locations**: `docs/DATABASE_LOCATIONS.md`
- **Echo-Loop Validation**: `docs/design_documents/echo_loop_validation.md`

### C. Example Workflows

#### Workflow 1: Simple Project (Calculator API)
```
1. User: POST /api/projects/create {"name": "Calculator API", ...}
2. Agent: POST /api/projects/42/clarify/start → 5 questions
3. User: POST /api/projects/42/clarify/answer → answers
4. Agent: POST /api/projects/42/generate-prd → PRD created
5. User: Reviews PRD in IDE, approves
6. Agent: POST /api/projects/42/estimate → task tree + estimates
7. User: POST /api/projects/42/approve → sets budget caps
8. User: POST /api/projects/42/start → PAS executes
9. HMI: User watches /actions?project_id=42 (live progress)
10. Agent: Echo-Loop validation → SLOs pass
11. Status: 'completed' → User views /projects/42/metrics (actual vs estimated)
```

#### Workflow 2: Budget Overrun (Replan)
```
1-7. (Same as Workflow 1)
8. PAS executes, Token Governor detects: projected overrun
9. Agent: POST /api/projects/42/pause → execution paused
10. Agent: POST /api/projects/42/replan → new estimates (cheaper models)
11. User: Reviews replan options, approves downshift
12. User: POST /api/projects/42/resume → execution continues
13. Status: 'completed' (within budget)
```

#### Workflow 3: Quality SLO Violation (Human Override)
```
1-9. (Same as Workflow 1)
10. Agent: Echo-Loop validation → mean_cos = 0.78 (below 0.82 threshold)
11. Status: 'needs_review' → User notified
12. User: Reviews artifacts, decides: acceptable trade-off
13. User: POST /api/projects/42/approvals {"gate": "quality_override", "status": "approved"}
14. Status: 'completed' (with quality exception logged)
```

---

## 16. Conclusion

**PLMS** provides a thin, human-facing lifecycle layer above PAS that enables:
- Structured ideation → PRD → estimation → approval workflow
- TMD-aware estimation with lane-specific priors (accurate cost forecasting)
- Echo-Loop quality SLOs (semantic validation gates)
- Seamless PAS integration (job cards, receipts, heartbeats)
- Transparent HMI monitoring (overlays on existing views)

**V1 MVP** delivers:
- 3 test projects (Calculator API, Ontology Ingestion, Narrative Summarization)
- ≤30% estimation error (tokens + time)
- Echo-Loop SLO enforcement (≥0.82 cosine)
- Full audit trail (receipts, artifacts, heartbeats)

**V2+** unlocks:
- Semantic Critical Path (graph centrality)
- Rehearsal Mode (1-5% preview)
- PRD-Delta Watcher (auto-replan)
- Deterministic Replay (byte-for-byte)
- Lane-Specialist routing (TMD-aware agents)

**PLMS is NOT**:
- A parallel orchestrator (PAS handles execution)
- A duplicate budget system (uses Token Governor)
- A new UI stack (extends HMI with overlays)

**Next Steps**: Begin Phase 1 (Foundation) — create database schema, implement Project API, build Projects Dashboard UI.

---

**Document Status**: Draft (awaiting review)
**Last Updated**: 2025-11-06
**Authors**: LNSP Core Team + Claude Code

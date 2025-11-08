# PRD: PAS (Project Agentic System)

**Version**: 1.0
**Date**: November 7, 2025
**Status**: Ready for Implementation
**Owner**: Engineering

---

## 1) Purpose (What PAS Is)

**PAS** is the **execution spine** under PLMS/HMI. It receives **job cards** from the VP/PEX (your local code operator), schedules them, executes via lane-specialized agents, streams heartbeats/receipts/KPIs, and enforces budget + quality gates. It's **model/provider-agnostic** and treats local/API LLMs as pluggable tools.

**Key Responsibilities**:
- Accept job submissions from LCO (Local Code Operator)
- Schedule tasks with fair-share, lane caps, deadline awareness
- Execute via specialized agents (Architect → Directors → Managers → Executors)
- Stream real-time telemetry (heartbeats, receipts, KPIs)
- Enforce budget guardrails and quality gates
- Emit run passports for deterministic replay
- Integrate with PLMS for planning/estimation/calibration

---

## 2) Non-Goals (V1)

- ❌ **No GUI** (HMI consumes PAS APIs)
- ❌ **No deep enterprise policy engine** (basic RBAC + allowlists only)
- ❌ **No vendor lock-in** (providers are adapters)
- ❌ **No distributed execution** (single-process scheduler for V1)
- ❌ **No fine-grained corporate RBAC** (simple scopes only)

---

## 3) System Context

```
Human → PEX/VP Agent (local) → PLMS (plan/estimate/approve) → PAS (execute)
                                                         ↑ receipts/heartbeats/KPIs ↓
                                                         HMI (dashboards, controls)
```

**Data Flow**:
1. Human: Declares initiative via `vp new`
2. LCO (VP Agent): Collects context, files plan with PLMS
3. PLMS: Estimates, stratifies, returns budget/CI bands
4. LCO: Submits job card to PAS
5. PAS: Schedules → Executes → Emits telemetry → Validates KPIs
6. PLMS: Receives receipts, updates calibration priors
7. HMI: Visualizes progress, budget runway, risk heatmap

---

## 4) Agent Hierarchy (Runtime Roles)

### 4.1 Architect
- **Role**: Expands job card → task graph
- **Responsibilities**:
  - Decompose PRD/goal into tasks
  - Resolve dependencies (topological order)
  - Estimate complexity per task
  - Allocate to Directors by lane

### 4.2 Director (Lane-Specific)
- **Role**: Assigns lanes/providers, sets budgets, batches similar tasks
- **Types**:
  - `director-code`: Owns Code-API-Design, Code-Impl lanes
  - `director-data`: Owns Data-Schema, Data-Pipeline lanes
  - `director-docs`: Owns Narrative, Documentation lanes
  - `director-graph`: Owns Graph-Ops (Neo4j operations)
  - `director-vector`: Owns Vector-Ops (LightRAG maintenance/query)

### 4.3 Manager
- **Role**: Orchestrates step-by-step execution, retries/backoff, emits heartbeats
- **Responsibilities**:
  - Execute task sequence (topological order)
  - Retry failed tasks (exponential backoff)
  - Emit heartbeats every 30s
  - Stream receipts/KPIs to PLMS

### 4.4 Executor
- **Role**: Does the work (tool/LLM/model call, script, test run)
- **Types**:
  - `executor-llm`: Calls LLM (Ollama, OpenAI, etc.)
  - `executor-tool`: Runs tools (pytest, ruff, make, npm, etc.)
  - `executor-git`: Git operations (commit, push, PR)
  - `executor-lightrag`: LightRAG refresh/query

### 4.5 Lanes (Examples)

| Lane | Purpose | Director | Typical Executors |
|------|---------|----------|-------------------|
| Code-API-Design | API endpoint design | director-code | executor-llm, executor-tool (openapi) |
| Code-Impl | Implementation | director-code | executor-llm, executor-tool (pytest, ruff) |
| Data-Schema | Database schema | director-data | executor-tool (psql, cypher-shell) |
| Narrative | Documentation | director-docs | executor-llm |
| Graph-Ops | Neo4j operations | director-graph | executor-tool (cypher-shell) |
| Vector-Ops | LightRAG refresh/query | director-vector | executor-lightrag |

---

## 5) State Machine (Project/Run)

### Run States
```
pending → planning → executing → validating → (completed | needs_review | failed | terminated | paused)
```

### Task States
```
queued → running → waiting_io → (succeeded | failed | skipped)
```

### State Transitions

| From | To | Trigger |
|------|-----|---------|
| pending | planning | `POST /runs/start` |
| planning | executing | Architect completes task graph |
| executing | validating | All tasks completed |
| validating | completed | KPI gates pass |
| validating | needs_review | KPI gates fail |
| * | paused | `POST /runs/pause` |
| paused | executing | `POST /runs/resume` |
| * | terminated | `POST /runs/terminate` or budget exceeded |

---

## 6) Core Data Model (Ties to PLMS)

### 6.1 Tables

```sql
-- Extends PLMS project_runs table
ALTER TABLE project_runs ADD COLUMN provider_matrix_json TEXT;  -- {lane: provider}
ALTER TABLE project_runs ADD COLUMN budget_cap_usd REAL;
ALTER TABLE project_runs ADD COLUMN energy_cap_kwh REAL;
ALTER TABLE project_runs ADD COLUMN rehearsal_pct REAL DEFAULT 0.0;
ALTER TABLE project_runs ADD COLUMN write_sandbox BOOLEAN DEFAULT FALSE;

-- PAS-specific tables
CREATE TABLE job_cards (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    lane TEXT NOT NULL,
    priority REAL DEFAULT 0.5,
    deps TEXT,  -- JSON array of task IDs
    payload_json TEXT,  -- Task-specific payload
    budget_usd REAL,
    ci_width_hint REAL,  -- From PLMS estimates
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id)
);

CREATE TABLE task_receipts (
    task_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    lane TEXT NOT NULL,
    provider TEXT NOT NULL,  -- "ollama:llama3.1:8b" or "openai:gpt-4"
    tokens_in INTEGER DEFAULT 0,
    tokens_out INTEGER DEFAULT 0,
    tokens_think INTEGER DEFAULT 0,  -- Reasoning tokens
    time_ms INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    energy_kwh REAL DEFAULT 0.0,
    echo_cos REAL,  -- Echo-loop cosine similarity
    status TEXT NOT NULL,  -- succeeded | failed | skipped
    artifacts_path TEXT,  -- Path to output artifacts
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id)
);

CREATE TABLE kpi_receipts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    kpi_name TEXT NOT NULL,  -- test_pass_rate, build_success, schema_diff, etc.
    value_json TEXT,  -- Actual value (may be float, bool, array)
    pass_bool BOOLEAN NOT NULL,  -- Did it meet threshold?
    threshold_json TEXT,  -- Expected threshold
    logs_path TEXT,  -- Path to detailed logs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES task_receipts(task_id),
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id)
);

CREATE TABLE run_passports (
    run_id TEXT PRIMARY KEY,
    provider_matrix_json TEXT NOT NULL,  -- Provider versions
    env_snapshot_json TEXT NOT NULL,  -- Environment variables, system info
    prd_sha TEXT,  -- Git SHA of PRD document
    git_commit TEXT NOT NULL,  -- Codebase commit at run start
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES project_runs(run_id)
);

-- Already exists in PLMS (action_logs extends this)
-- CREATE TABLE action_logs (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     run_id TEXT NOT NULL,
--     task_id TEXT,
--     ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
--     level TEXT,  -- DEBUG | INFO | WARN | ERROR
--     message TEXT,
--     meta_json TEXT
-- );
```

### 6.2 Indexes

```sql
CREATE INDEX idx_job_cards_run_lane_status ON job_cards(run_id, lane, status);
CREATE INDEX idx_task_receipts_task ON task_receipts(task_id);
CREATE INDEX idx_kpi_receipts_kpi_pass ON kpi_receipts(kpi_name, pass_bool);
CREATE INDEX idx_task_receipts_lane_provider ON task_receipts(lane, provider);
```

---

## 7) External API (Stable Contracts)

### 7.1 Submit Job Card

**Endpoint**: `POST /pas/v1/jobcards`

**Request**:
```json
{
  "project_id": 42,
  "run_id": "r-2025-11-07-001",
  "lane": "Code-Impl",
  "priority": 0.65,
  "deps": ["task-11", "task-12"],
  "payload": {
    "repo": "/path/to/repo",
    "goal": "implement /login endpoint",
    "tests": ["tests/test_login.py"]
  },
  "budget_usd": 1.50,
  "ci_width_hint": 0.3,
  "idempotency_key": "uuid-1234"
}
```

**Response**: `201 Created`
```json
{
  "task_id": "task-17"
}
```

**Headers**:
- `Idempotency-Key: <uuid>` (required for idempotency)
- `Idempotent-Replay: true` (returned if duplicate key)

---

### 7.2 Control Plane

**Start Run**: `POST /pas/v1/runs/start`
```json
{
  "project_id": 42,
  "run_id": "r-2025-11-07-001",
  "run_kind": "baseline",
  "rehearsal_pct": 0.0,
  "budget_caps": {
    "budget_usd": 50.0,
    "energy_kwh": 2.0
  }
}
```

**Pause Run**: `POST /pas/v1/runs/pause`
```json
{
  "run_id": "r-2025-11-07-001",
  "reason": "Budget threshold exceeded"
}
```

**Resume Run**: `POST /pas/v1/runs/resume`
```json
{
  "run_id": "r-2025-11-07-001"
}
```

**Terminate Run**: `POST /pas/v1/runs/terminate`
```json
{
  "run_id": "r-2025-11-07-001",
  "reason": "User cancellation"
}
```

**Simulate (Rehearsal)**: `POST /pas/v1/runs/simulate`
```json
{
  "run_id": "r-2025-11-07-001",
  "rehearsal_pct": 0.01,
  "stratified": true
}
```

**Response**:
```json
{
  "strata_coverage": 1.0,
  "rehearsal_tokens": 150,
  "projected_tokens": 15000,
  "ci_lower": 13200,
  "ci_upper": 16800,
  "risk_score": 0.12
}
```

---

### 7.3 Status & Monitoring

**Run Status**: `GET /pas/v1/runs/status?run_id=<run_id>`

**Response**:
```json
{
  "run_id": "r-2025-11-07-001",
  "status": "executing",
  "tasks_total": 10,
  "tasks_completed": 6,
  "tasks_failed": 1,
  "spend_usd": 12.50,
  "spend_energy_kwh": 0.45,
  "runway_minutes": 45,
  "kpi_violations": [
    {
      "task_id": "task-5",
      "kpi_name": "test_pass_rate",
      "expected": 0.95,
      "actual": 0.87,
      "pass": false
    }
  ]
}
```

**Portfolio Status**: `GET /pas/v1/portfolio/status`

**Response**:
```json
{
  "active_runs": 3,
  "queued_tasks": 42,
  "lane_utilization": {
    "Code-Impl": 0.85,
    "Data-Schema": 0.40,
    "Narrative": 0.60
  },
  "lane_caps": {
    "Code-Impl": 6,
    "Data-Schema": 2,
    "Narrative": 4
  },
  "fairness_weights": {
    "project-101": 0.33,
    "project-102": 0.33,
    "project-103": 0.34
  }
}
```

---

### 7.4 Telemetry I/O

**Heartbeat**: `POST /pas/v1/heartbeats`
```json
{
  "run_id": "r-2025-11-07-001",
  "task_id": "task-5",
  "ts": "2025-11-07T14:30:00Z",
  "progress_pct": 0.65,
  "est_remaining_ms": 12000
}
```

**Task Receipt**: `POST /pas/v1/receipts`
```json
{
  "task_id": "task-5",
  "run_id": "r-2025-11-07-001",
  "lane": "Code-Impl",
  "provider": "ollama:llama3.1:8b",
  "tokens_in": 1200,
  "tokens_out": 450,
  "tokens_think": 80,
  "time_ms": 8500,
  "cost_usd": 0.12,
  "energy_kwh": 0.015,
  "echo_cos": 0.87,
  "status": "succeeded",
  "artifacts_path": "/artifacts/task-5/"
}
```

**KPI Receipt**: `POST /pas/v1/kpis`
```json
{
  "task_id": "task-5",
  "run_id": "r-2025-11-07-001",
  "kpi_name": "test_pass_rate",
  "value": 0.87,
  "pass": false,
  "threshold": {"min": 0.95},
  "logs_path": "/artifacts/task-5/pytest_output.txt"
}
```

---

### 7.5 Webhooks → PLMS

**Run Completed**: `POST {PLMS_API}/hooks/pas/run_completed`
```json
{
  "run_id": "r-2025-11-07-001",
  "validation_pass": true,
  "summary": {
    "tasks_total": 10,
    "tasks_succeeded": 9,
    "tasks_failed": 1,
    "spend_usd": 25.50,
    "duration_minutes": 120
  }
}
```

**Task Completed**: `POST {PLMS_API}/hooks/pas/task_completed`
```json
{
  "task_id": "task-5",
  "run_id": "r-2025-11-07-001",
  "status": "succeeded",
  "metrics": {
    "tokens_total": 1730,
    "time_ms": 8500,
    "cost_usd": 0.12
  },
  "kpis": [
    {"name": "test_pass_rate", "value": 0.87, "pass": false}
  ]
}
```

**Auth**: mTLS or signed JWT (service accounts). All write APIs accept `Idempotency-Key`.

---

## 8) Scheduler (Portfolio + Lane Caps)

### 8.1 Objective
Minimize risk (CI width, historical MAE), respect deadlines/budgets, enforce lane caps, preserve fair-share across projects.

### 8.2 Priority Score
```python
score = (
    w_deadline * urgency +
    w_ci * ci_width +
    w_budget * runway_risk +
    w_proj * (1 - fairshare_quota)
)
```

**Weights** (default):
- `w_deadline = 0.4` (deadline urgency)
- `w_ci = 0.2` (estimation uncertainty)
- `w_budget = 0.3` (budget runway risk)
- `w_proj = 0.1` (fair-share quota)

### 8.3 Dispatch Policy
- **Stratified batching** by lane/complexity
- **Reserve rehearsal slots** (1–5%) when `simulate` requested
- **Lane caps enforced**: No lane exceeds `max_concurrent` (from PLMS `lane_configs`)
- **Fair-share**: No project starved (≥20% utilization for active projects)

---

## 9) Rehearsal & Deterministic Replay

### 9.1 Rehearsal Execution
- Executes **1–5% stratified sample** of tasks
- Returns **CI bands** and **risk vectors**
- **Strata coverage must be 1.0** (all strata represented)
- If coverage < 1.0, auto-bump `rehearsal_pct` to 5% (cap)

### 9.2 Run Passport
**Required fields**:
- `provider_matrix_json`: Provider versions (e.g., `{"Code-Impl": "ollama:llama3.1:8b:v1.2.3"}`)
- `env_snapshot_json`: Environment variables, system info
- `prd_sha`: Git SHA of PRD document
- `git_commit`: Codebase commit at run start

**Enforcement**:
- **Must exist before first task starts**
- **Required for calibration inclusion** (baseline/hotfix only)
- **Stored in `run_passports` table**

---

## 10) Quality Gates (Echo + KPIs)

### 10.1 Global Echo Threshold
- **Global**: `echo_cos ≥ 0.82`
- **Lane overrides allowed** (e.g., Narrative may use BLEU instead)

### 10.2 Lane KPIs Table

| Lane | KPI | Threshold | Validator |
|------|-----|-----------|-----------|
| Code-Impl | `test_pass_rate` | ≥ 0.90 | `pytest --json` |
| Code-Impl | `linter_pass` | `true` | `ruff check` |
| Data-Schema | `schema_diff` | `== 0` | `pg_dump \| diff` |
| Data-Schema | `migration_success` | `true` | Migration script exit code |
| Narrative | `BLEU` | ≥ 0.40 | BLEU score vs reference |
| Graph-Ops | `edge_count_delta` | `< 5%` | Neo4j relationship count |
| Vector-Ops | `index_freshness` | `≤ 2 min` | Last refresh timestamp |

### 10.3 Validation Logic
```python
def validate(run_id: str) -> bool:
    """Aggregate KPI violations; return validation_pass."""
    violations = db.query(
        "SELECT * FROM kpi_receipts WHERE run_id = ? AND pass_bool = false",
        (run_id,)
    )

    if violations:
        db.execute(
            "UPDATE project_runs SET validation_pass = false WHERE run_id = ?",
            (run_id,)
        )
        return False

    db.execute(
        "UPDATE project_runs SET validation_pass = true WHERE run_id = ?",
        (run_id,)
    )
    return True
```

**Consequences**:
- `validation_pass = false` **blocks completion** and **calibration**
- Run moves to `needs_review` state
- Human intervention required to fix + re-run

---

## 11) Safety / Sandboxing

### 11.1 Executor Tool Allowlists (Per Lane)

| Lane | Allowed Tools |
|------|---------------|
| Code-Impl | `pytest`, `ruff`, `make`, `npm`, `node`, `python`, `mypy` |
| Data-Schema | `psql`, `cypher-shell`, `pg_dump`, `pg_restore` |
| Narrative | (LLM only, no shell tools) |
| Graph-Ops | `cypher-shell`, `neo4j-admin` |
| Vector-Ops | `curl` (LightRAG API) |

**Enforcement**: Executor rejects commands not in allowlist

### 11.2 File Access Sandbox
- **Bounded to workspace root(s)**: No traversal outside repo
- **Read-only by default**: Write requires explicit `write_sandbox = true`
- **Dry-run mode**: Always store unified diff + backup before applying

### 11.3 Secrets Scrubber
- **Outbound LLM context**: Mask env vars, API keys, tokens
- **Patterns**: `API_KEY=`, `SECRET=`, `TOKEN=`, `PASSWORD=`
- **Redact before sending to LLM**

### 11.4 Budget Guard
- **Auto-pause when projected spend > 25% over cap** (unless override approved)
- **Alert**: Slack + PagerDuty (if configured)
- **Formula**: `projected_spend = (tokens_so_far / tasks_completed) * tasks_total`

---

## 12) Observability

### 12.1 Metrics (Per Task)
Emit **all** of the following for every task:
- `time_ms`: Execution time
- `tokens_in`, `tokens_out`, `tokens_think`: Token breakdown
- `cost_usd`: Provider cost
- `energy_kwh`: Estimated energy (CPU/GPU coefficients)

**Already aligned with PLMS Tier 1** (multi-metric telemetry addendum)

### 12.2 Heartbeats
- **Frequency**: Every ≤ 30 seconds
- **Payload**: `progress_pct`, `est_remaining_ms`
- **Purpose**: HMI progress bars, timeout detection

### 12.3 Observability Metrics
- **P95 task latency** (by lane)
- **Failure taxonomies** (by lane, provider, task type)
- **Retry counters** (exponential backoff stats)
- **Lane utilization** (active tasks / lane cap)

### 12.4 HMI Integration
HMI consumes:
- `/portfolio/status` → Global queue, lane caps, fairness weights
- `/runs/status` → DAG status, KPIs, spend, runway
- `task_receipts`, `kpi_receipts` tables → Metrics visualization

---

## 13) Minimal PAS Stub (To Unblock LCO MVP)

**Purpose**: Allow Phase 3 (LCO MVP) to run while full PAS is built (Weeks 5-8)

### 13.1 Capabilities
- ✅ In-memory queue (single process worker)
- ✅ Accepts `/jobcards`, tracks DAG in memory
- ✅ "Executes" tasks by sleeping and emitting synthetic receipts/KPIs
- ✅ Provides `/runs/simulate` with CI extrapolation
- ✅ All endpoints match production API (stable contract)

### 13.2 Stub Contract (Python FastAPI Pseudocode)

```python
# POST /pas/v1/jobcards
def submit_jobcard(card: JobCard):
    tid = str(uuid.uuid4())
    DAG.add(tid, deps=card.deps, lane=card.lane, payload=card.payload)
    return {"task_id": tid}

# POST /pas/v1/runs/start
def start_run(run):
    RUNS[run.run_id] = {**run.dict(), "status": "executing"}

    # Background worker thread:
    # - Topological order
    # - time.sleep(simulated duration)
    # - Emit receipts/KPIs to /v1/receipts and /v1/kpis internally
    # - Update RUNS status

    threading.Thread(target=_execute_run, args=(run.run_id,)).start()
    return {"status": "executing"}

# GET /pas/v1/runs/status?run_id=...
def run_status(run_id):
    return DAG.snapshot(run_id)  # tasks, states, spend, KPIs
```

**Synthetic Execution**:
```python
def _execute_task_synthetic(task_id, lane, payload):
    """Simulate task execution with realistic delays."""
    # Simulate execution time based on lane
    delays = {
        "Code-Impl": (5, 15),  # 5-15 seconds
        "Data-Schema": (3, 8),
        "Narrative": (10, 20),
        "Vector-Ops": (2, 5),
    }
    delay = random.uniform(*delays.get(lane, (5, 10)))
    time.sleep(delay)

    # Emit synthetic receipt
    receipt = {
        "task_id": task_id,
        "lane": lane,
        "provider": "synthetic:stub",
        "tokens_in": random.randint(500, 2000),
        "tokens_out": random.randint(200, 800),
        "tokens_think": random.randint(50, 200),
        "time_ms": int(delay * 1000),
        "cost_usd": round(random.uniform(0.05, 0.30), 2),
        "energy_kwh": round(random.uniform(0.01, 0.05), 3),
        "echo_cos": round(random.uniform(0.80, 0.95), 2),
        "status": "succeeded" if random.random() > 0.1 else "failed",
    }

    # Emit synthetic KPIs (lane-specific)
    kpis = []
    if lane == "Code-Impl":
        kpis.append({
            "kpi_name": "test_pass_rate",
            "value": round(random.uniform(0.85, 1.0), 2),
            "pass": receipt["status"] == "succeeded",
            "threshold": {"min": 0.90}
        })

    return receipt, kpis
```

**This lets `vp plan/estimate/status` + HMI operate NOW; swap in real scheduler/executors later.**

---

## 14) Rollout (Weeks 5–8 from Integration Plan)

### Week 5: PAS PRD + Stub API
- ✅ Create `PRD_PAS_Project_Agentic_System.md` (this document)
- ✅ Bootstrap repo: `services/pas/`, `services/pas/stub/`
- ✅ Stub API online: `make run-pas-stub`
- ✅ LCO wired to stub: `vp start` works end-to-end

### Week 6: Real Scheduler
- ✅ Lane caps enforcement
- ✅ Fair-share scheduling (portfolio)
- ✅ Receipts/KPIs persisted (SQLite → PostgreSQL)
- ✅ Heartbeat streaming (SSE)

### Week 7: Executors
- ✅ Code-Impl executor (pytest, ruff, git)
- ✅ Data-Schema executor (psql, cypher-shell)
- ✅ Graph-Ops executor (Neo4j operations)
- ✅ Vector-Ops executor (LightRAG refresh/query)

### Week 8: Quality Gates + Webhooks
- ✅ Rehearsal mode (stratified sampling)
- ✅ Run passports (deterministic replay)
- ✅ Webhooks to PLMS (run/task completion)
- ✅ Quality gates enforced (echo + KPIs)

---

## 15) Acceptance Criteria (V1)

### Functional Requirements
- [ ] End-to-end: `vp start` → PAS executes → KPIs green → PLMS marks completed
- [ ] Budget guardrails fire when projected spend > 25% over cap
- [ ] Portfolio fairness visible (no lane <20% utilization for active projects)
- [ ] Rehearsal CI matches realized cost within CI band (≥80% of runs)

### Quality Requirements
- [ ] Replay passport present on every baseline/hotfix run
- [ ] Calibration excludes dirty runs (rehearsal/sandbox/failed)
- [ ] KPI gates enforced (no "green echo, red tests" slips)
- [ ] Heartbeat latency P95 ≤ 30s

### Operational Requirements
- [ ] Two concurrent projects complete with fair-share scheduling
- [ ] Stub API → Real API migration (zero downtime)
- [ ] All endpoints return OpenAPI-compliant responses

---

## Addendum: Three Enhancements (LightRAG + Planner Learning + Multi-Metric)

### A) LightRAG Integration (Vector Manager Agent)

**Goal**: Maintain a semantic + graph index of the **codebase** (not data); refresh on every commit

**Components**:
1. **LightRAG Code Service** (port 7500)
   - Endpoints: `/refresh`, `/query`, `/snapshot`
   - Storage: `artifacts/lightrag_code_index/`
   - Git hook: `.git/hooks/post-commit` → `curl http://localhost:7500/refresh`

2. **Vector Manager Agent** (`services/lightrag_code/vector_manager.py`)
   - Runs every 5 minutes (cron)
   - Checks: index freshness ≤ 2 min, coverage ≥ 98%, no drift warnings
   - Alerts: Slack if violations

3. **Queries**:
   - `where_defined(symbol)` → file:line
   - `who_calls(fn)` → caller list
   - `impact_set(file|symbol)` → transitive dependents
   - `nearest_neighbors(snippet)` → semantic matches

**SLOs**:
- Index freshness ≤ 2 min from commit
- Query latency P95 ≤ 300 ms (local)
- Coverage ≥ 98% of `.py` files indexed

---

### B) Planner Learning LLM

**Goal**: Train project-experience model on completed runs (LOCAL + GLOBAL partitions)

**Pipeline**:
1. **After completion**: PLMS emits `planner_training_pack.json`
   - Sanitized: task tree, lane ids, provider matrix, rehearsal outcomes, KPI results, spend/time/token/energy
2. **Trainer agent** fine-tunes or LoRA-adapts Planner model (Llama 3.1 base)
   - **Dual partitions**: `artifacts/planner_local/{repo_id}/` + `artifacts/planner_global/`
3. **A/B validation**: Re-run same template with updated Planner (no human), compare units (time, tokens, cost, energy)
   - Target: ≥15% improvement median after 10 projects

**Serving**:
- Planner uses **GLOBAL first**, overlays **LOCAL deltas** if repo/team matches
- Cold-start: fallback to default priors + CI bands

**KPIs**:
- Estimation MAE% drops over time (goal: ≤20% at 10 projects)
- Rework rate ↓, KPI violations ↓, budget overruns ↓

---

### C) Multi-Metric Telemetry & Visualization

**Goal**: Track separately: time, tokens (in/out/think), cost, energy, carbon

**Data Model** (extends `task_receipts`):
```sql
ALTER TABLE task_receipts ADD COLUMN tokens_in INTEGER DEFAULT 0;
ALTER TABLE task_receipts ADD COLUMN tokens_out INTEGER DEFAULT 0;
ALTER TABLE task_receipts ADD COLUMN tokens_think INTEGER DEFAULT 0;
ALTER TABLE task_receipts ADD COLUMN energy_kwh REAL DEFAULT 0.0;
ALTER TABLE task_receipts ADD COLUMN carbon_kg REAL DEFAULT 0.0;
```

**Energy Estimator**:
```python
def estimate_energy(tokens: int, model: str, device: str) -> float:
    """E ≈ (GPU_kW × active_time) + (CPU_kW × active_time)"""
    coefficients = {
        "llama3.1:8b": {"cpu": 0.05, "gpu": 0.15},  # kWh per 1K tokens
        "gpt-4": {"cpu": 0.0, "gpu": 0.20},
    }
    coeff = coefficients.get(model, {}).get(device, 0.10)
    return (tokens / 1000) * coeff
```

**HMI**:
- Stacked bars per task/lane (time / token types / cost / energy)
- Budget runway + **carbon overlay** (for "green" stakeholders)
- Compare runs: `GET /compare?runA=…&runB=…` → % deltas with significance flags

**APIs**:
- `GET /metrics?with_ci=1&breakdown=all` → returns mean + CI for each metric and token subtype
- `GET /compare?runA=…&runB=…` → structured diff with significance flags

**KPIs**:
- Visualization latency ≤ 1s for recent projects
- Metrics completeness ≥ 99% of steps report all four classes (time, tokens, cost, energy)

---

## Critical Notes (So We Don't Paint Ourselves Into a Corner)

1. **Idempotency Everywhere**
   - All write endpoints require `Idempotency-Key` header
   - Replay header honored (already in PLMS Tier 1)

2. **Provider Snapshot First**
   - Refuse `start` without run passport snapshot
   - Passport must include provider versions, env, PRD SHA, git commit

3. **Rehearsal Isolation**
   - **NEVER include rehearsal/sandbox/failed runs in calibration**
   - Enforced in PAS, verified by nightly invariants

4. **Vector-Ops Lane**
   - Treat LightRAG refresh/query as **first-class tasks**
   - Budgets/metrics apply (not "free" operations)

---

## What You Can Do Next, Immediately

1. **Generate FastAPI PAS stub** (single file, ~300 LOC)
   - All endpoints from Section 7 (stable contract)
   - Synthetic executors for Code-Impl + Vector-Ops
   - OpenAPI schema auto-generated

2. **Add Makefile target**: `make run-pas-stub`
   - Start stub on port 6200
   - Health check: `curl http://localhost:6200/health`

3. **Wire LCO to PAS stub**
   - `vp start` → `POST /pas/v1/runs/start`
   - `vp status` → `GET /pas/v1/runs/status`
   - `vp logs` → `GET /pas/v1/runs/status` (task logs)

4. **End-to-end demo script**
   - `vp new --name demo-project`
   - `vp estimate`
   - `vp start`
   - `vp status` (watch progress)
   - `vp logs --tail 20`

---

**END OF PRD**

_This PRD defines the execution spine that PLMS, LCO, and HMI all depend on._
_Implementation: Weeks 5-8 (10 weeks total with Phases 1-3 from Integration Plan)_
_Owner: Engineering + Operations_

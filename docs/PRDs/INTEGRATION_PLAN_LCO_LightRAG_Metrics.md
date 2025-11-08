# Integration Plan: LCO + LightRAG Code Index + Multi-Metric Telemetry

**Version**: 1.0
**Date**: November 7, 2025
**Status**: Ready for Review
**Owner**: Engineering + Operations

---

## 🎯 Executive Summary

**Goal**: Integrate three new PRD features into production LNSP+PLMS stack over 10 weeks

**Components**:
1. **LightRAG Code Index** (PRD Addendum A) - Code+graph index for codebase
2. **Multi-Metric Telemetry** (PRD Addendum C) - Energy/carbon tracking
3. **LCO (Local Code Operator)** (PRD LCO) - VP Agent terminal client

**Key Insight**: These PRDs are **tightly coupled** but build on **PAS (Project Agentic System)** which doesn't exist. We must decouple and sequence implementation carefully.

**Recommended Timeline**: 10 weeks (incremental delivery, not waterfall)

---

## 🚨 Critical Dependencies (BLOCKERS)

### Missing Foundation: PAS (Project Agentic System)

```
PAS is referenced in all three PRDs but NEVER DEFINED!

LCO expects:
  - PAS.submit(jobcard) → run_id
  - PAS.status(run_id) → receipts stream
  - PAS.kpi_validate(run_id) → pass/fail

Planner Learning expects:
  - PAS completion webhooks → training_pack.json
  - PAS lane assignments → LOCAL/GLOBAL partitions

BUT: No PAS architecture document, no code, no API spec!
```

**Decision**: Build **PAS Lite** (stub for testing) in Phase 3, defer full PAS to Phase 4

---

## 🗓️ Implementation Phases (Decoupled & Sequenced)

### Phase 1: LightRAG Code Index (Standalone) - Weeks 1-2

**Goal**: Index THIS codebase with LightRAG for symbol lookup + dependency analysis

**Why First**: Provides infrastructure that LCO will later consume (but works standalone)

#### Deliverables

1. **LightRAG Code Service** (`services/lightrag_code/`)
   - Port: 7500
   - Endpoints:
     - `POST /refresh` - Re-index repo (triggered by git hook)
     - `GET /query` - where_defined, who_calls, impact_set, nearest_neighbors
     - `GET /snapshot` - Export rag_snapshot.json bound to git SHA
   - Storage: `artifacts/lightrag_code_index/` (separate from data index)

2. **Git Hook Integration**
   - `.git/hooks/post-commit` → `curl http://localhost:7500/refresh`
   - Max refresh time: 2 minutes for 10K LOC codebase

3. **Vector Manager Agent** (lightweight)
   - Script: `services/lightrag_code/vector_manager.py`
   - Runs every 5 minutes (cron)
   - Checks: index freshness, coverage, drift warnings
   - Alerts: Slack if coverage drops below 98%

#### Acceptance Criteria

- [ ] `vp new` triggers index refresh within 2 minutes
- [ ] Query `where_defined("IsolatedVecTextVectOrchestrator")` returns correct file:line
- [ ] Query `who_calls("encode_texts")` returns 5+ callers
- [ ] Query `impact_set("src/vectorizer.py")` returns transitive dependents
- [ ] Query `nearest_neighbors("embedding generation")` returns semantic matches
- [ ] Coverage: ≥98% of `.py` files indexed
- [ ] Latency: P95 ≤ 300ms for queries

#### Implementation Commands

```bash
# Week 1: Setup LightRAG for codebase (NOT data)
mkdir -p services/lightrag_code
mkdir -p artifacts/lightrag_code_index

# Install tree-sitter for AST parsing
./.venv/bin/pip install tree-sitter tree-sitter-python

# Create service (FastAPI)
cat > services/lightrag_code/app.py <<'EOF'
from fastapi import FastAPI
from lightrag import LightRAG
from lightrag.llm import lollms_embed
import os

app = FastAPI()

# Initialize LightRAG for codebase (separate from data)
rag = LightRAG(
    working_dir="artifacts/lightrag_code_index",
    llm_model_name="llama3.1:8b",
    llm_model_max_token_size=8192,
    embedding_func=lollms_embed  # Use same embedder as data
)

@app.post("/refresh")
async def refresh_index(scope: str = "."):
    """Re-index codebase (triggered by git hook or manual)."""
    # Parse Python files with tree-sitter
    # Extract: imports, class defs, function defs, calls
    # Build call graph + import graph
    # Generate embeddings for docstrings + code snippets
    # Write to artifacts/lightrag_code_index/
    return {"status": "refreshed", "files_indexed": 1234, "duration_ms": 1200}

@app.get("/query")
async def query_code(kind: str, payload: dict):
    """
    kind: where_defined | who_calls | impact_set | nearest_neighbors
    payload: {"symbol": "foo"} or {"file": "bar.py"} or {"snippet": "..."}
    """
    # Queries LightRAG index
    # Returns: [{file, line, snippet, confidence}]
    return {"results": [...]}

@app.get("/snapshot")
async def snapshot():
    """Export rag_snapshot.json bound to current git SHA."""
    import subprocess
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    # Export index to JSON with git SHA
    return {"snapshot_path": f"artifacts/lightrag_code_snapshot_{git_sha}.json"}
EOF

# Start service
./.venv/bin/uvicorn services.lightrag_code.app:app --host 127.0.0.1 --port 7500 &

# Test refresh
curl -X POST http://localhost:7500/refresh
# Expected: {"status": "refreshed", "files_indexed": ~1234, "duration_ms": <2000}

# Test query
curl "http://localhost:7500/query?kind=where_defined&symbol=IsolatedVecTextVectOrchestrator"
# Expected: {"results": [{"file": "app/vect_text_vect/vec_text_vect_isolated.py", "line": 42}]}

# Week 2: Git hook + Vector Manager
cat > .git/hooks/post-commit <<'EOF'
#!/bin/bash
# Trigger LightRAG code index refresh after commit
curl -X POST http://localhost:7500/refresh -s || echo "LightRAG code service not running"
EOF
chmod +x .git/hooks/post-commit

# Vector Manager (cron job every 5 min)
cat > services/lightrag_code/vector_manager.py <<'EOF'
import requests
import time

def check_health():
    """Check index freshness, coverage, drift."""
    resp = requests.get("http://localhost:7500/snapshot").json()
    # Check: last refresh < 2 min ago, coverage ≥ 98%, no drift warnings
    # Alert to Slack if violations
    pass

if __name__ == "__main__":
    while True:
        check_health()
        time.sleep(300)  # Every 5 minutes
EOF

# Add to crontab
echo "*/5 * * * * cd /path/to/lnsp-phase-4 && ./.venv/bin/python services/lightrag_code/vector_manager.py" | crontab -
```

#### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Large monorepo (>100K LOC) | Shard index by submodule, lazy load on query |
| Tree-sitter parsing failures | Fallback to ctags for coverage gaps |
| Index refresh >2 min | Background async refresh, serve stale index |

---

### Phase 2: Multi-Metric Telemetry - Week 2

**Goal**: Add energy/carbon tracking to PLMS receipts + HMI visualization

**Why Second**: Quick win, enhances observability before LCO integration

#### Deliverables

1. **Extended Receipt Schema**
   ```sql
   -- Add to project_runs table
   ALTER TABLE project_runs ADD COLUMN tokens_input INTEGER DEFAULT 0;
   ALTER TABLE project_runs ADD COLUMN tokens_output INTEGER DEFAULT 0;
   ALTER TABLE project_runs ADD COLUMN tokens_tool_use INTEGER DEFAULT 0;
   ALTER TABLE project_runs ADD COLUMN tokens_think INTEGER DEFAULT 0;
   ALTER TABLE project_runs ADD COLUMN energy_kwh REAL DEFAULT 0.0;
   ALTER TABLE project_runs ADD COLUMN carbon_kg REAL DEFAULT 0.0;
   ```

2. **Energy Estimator** (`services/plms/energy_estimator.py`)
   ```python
   def estimate_energy(tokens: int, model: str, device: str) -> float:
       """
       E ≈ (GPU_kW × active_time) + (CPU_kW × active_time)
       Returns: kWh
       """
       coefficients = {
           "llama3.1:8b": {"cpu": 0.05, "gpu": 0.15},  # kWh per 1K tokens
           "gpt-4": {"cpu": 0.0, "gpu": 0.20},
       }
       coeff = coefficients.get(model, {}).get(device, 0.10)
       return (tokens / 1000) * coeff
   ```

3. **HMI Endpoint** (`services/plms/api/projects.py`)
   ```python
   @router.get("/api/projects/{id}/metrics")
   async def get_metrics(id: int, with_ci: int = 0, breakdown: str = "basic"):
       """
       breakdown: basic | all
       all → includes token breakdown, energy, carbon
       """
       if breakdown == "all":
           return {
               "tokens_total": 15000,
               "tokens_input": 8000,
               "tokens_output": 5000,
               "tokens_tool_use": 1500,
               "tokens_think": 500,
               "energy_kwh": 0.75,
               "carbon_kg": 0.15,  # 200g CO2/kWh grid intensity
               # ... existing metrics
           }
   ```

#### Acceptance Criteria

- [ ] `/metrics?breakdown=all` returns token breakdown (input/output/tool/think)
- [ ] Energy estimates labeled as **ESTIMATED** (not actual)
- [ ] Carbon overlay in HMI budget runway gauge
- [ ] Compare runs: `GET /compare?runA=42&runB=43` shows % deltas
- [ ] Visualization latency ≤ 1s for recent projects
- [ ] Metrics completeness ≥ 99% of steps

#### Implementation Commands

```bash
# Apply migration
sqlite3 artifacts/registry/registry.db < migrations/2025_11_07_telemetry_schema.sql

# Update services/plms/api/projects.py (add breakdown logic)
# Update services/plms/kpi_emit.py (emit token types)

# Test endpoint
curl "http://localhost:6100/api/projects/42/metrics?with_ci=1&breakdown=all" | jq
# Expected: tokens_input, tokens_output, energy_kwh, carbon_kg fields

# HMI integration: Update frontend to show stacked bars (time/tokens/cost/energy)
```

---

### Phase 3: LCO Terminal Client (MVP, Read-Only) - Weeks 3-4

**Goal**: Ship `vp` CLI that can plan, estimate, and query status (NO execution yet)

**Why Third**: Useful for planning even without PAS execution

#### Deliverables

1. **Terminal Client** (`cli/vp.py`)
   - Commands: `vp new`, `vp plan`, `vp estimate`, `vp status`, `vp logs`
   - Model broker: Ollama only (Qwen-Coder or Llama 3.1)
   - File sandbox: Read-only (no edits yet)
   - JSON-RPC API: stdio only (no WebSocket yet)

2. **PAS Lite (Stub)** (`services/pas_lite/`)
   - Fake job submission: `POST /submit` → returns synthetic run_id
   - Fake receipts: `GET /receipts/{run_id}` → streams canned receipts
   - Purpose: Testing LCO without full PAS

3. **PLMS Integration**
   - `vp new` → `POST /api/projects` (creates project in registry.db)
   - `vp estimate` → `GET /api/projects/{id}/metrics?with_ci=1`
   - `vp status` → `GET /api/projects/{id}/status`

#### Acceptance Criteria

- [ ] `vp new --name test-proj` creates project in registry.db
- [ ] `vp plan` calls PLMS clarify endpoint (if exists) or prompts user
- [ ] `vp estimate` prints token/duration/cost with 90% CI bands
- [ ] `vp status` shows current phase, progress %, budget remaining
- [ ] `vp logs --tail 20` shows last 20 receipt lines
- [ ] All commands work with NO PAS running (stub mode)

#### Implementation Commands

```bash
# Week 3: Terminal client scaffolding
mkdir -p cli
cat > cli/vp.py <<'EOF'
#!/usr/bin/env python3
import click
import requests

PLMS_API = "http://localhost:6100"

@click.group()
def cli():
    """VP Agent - Local Code Operator"""
    pass

@cli.command()
@click.option("--name", required=True)
def new(name):
    """Initialize a new project."""
    resp = requests.post(f"{PLMS_API}/api/projects", json={"name": name})
    print(f"Project created: {resp.json()['project_id']}")

@cli.command()
def estimate():
    """Get cost/time estimates."""
    # Read .vp/project_id from local state
    project_id = 42  # TODO: Read from local state
    resp = requests.get(f"{PLMS_API}/api/projects/{project_id}/metrics?with_ci=1")
    data = resp.json()
    print(f"Estimated tokens: {data['tokens_mean']} (CI: {data['tokens_ci_lower']}-{data['tokens_ci_upper']})")
    print(f"Estimated duration: {data['duration_mean']} min")

@cli.command()
def status():
    """Get current project status."""
    project_id = 42
    resp = requests.get(f"{PLMS_API}/api/projects/{project_id}/status")
    print(resp.json())

if __name__ == "__main__":
    cli()
EOF

chmod +x cli/vp.py

# Test commands
./cli/vp.py new --name test-project
# Expected: Project created: 42

./cli/vp.py estimate
# Expected: Estimated tokens: 15000 (CI: 13200-16800)

# Week 4: Model broker (Ollama only)
cat > cli/model_broker.py <<'EOF'
import requests

def get_model(policy: dict):
    """Choose model based on policy (cost/capability/latency)."""
    # For MVP: Always use Ollama + Llama 3.1
    return {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "endpoint": "http://localhost:11434",
    }

def generate(prompt: str):
    """Generate response using selected model."""
    model = get_model({})
    resp = requests.post(f"{model['endpoint']}/api/generate", json={
        "model": model["model"],
        "prompt": prompt,
        "stream": False,
    })
    return resp.json()["response"]
EOF
```

#### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| PAS doesn't exist yet | Use PAS Lite stub (canned responses) |
| Model broker complexity | Start with Ollama only, defer API providers |
| Git operations unsafe | Read-only mode, defer commit/push to Phase 4 |

---

### Phase 4: PAS Integration (Full Stack) - Weeks 5-8

**Goal**: Build **FULL PAS** (Project Agentic System) to execute jobs submitted by LCO

**Why Fourth**: Foundation for LCO execution + Planner Learning

**WARNING**: This is a **MAJOR UNDERTAKING** not defined in any PRD! Requires architecture design.

#### Deliverables (HIGH-LEVEL, NEEDS DETAILED PRD)

1. **PAS Architect** (`services/pas/architect.py`)
   - Decomposes PRD → task tree
   - Allocates tasks to lanes (Code-API, Data-Schema, Narrative, etc.)
   - Assigns to Director agents

2. **PAS Director Agents** (`services/pas/directors/`)
   - `director-code.py` - Owns code lane
   - `director-data.py` - Owns data lane
   - `director-docs.py` - Owns docs lane
   - Each Director spawns Manager agents

3. **PAS Manager Agents** (`services/pas/managers/`)
   - `manager-code.py` - Code review, test, build
   - `manager-data.py` - Schema validation, migration
   - Each Manager executes tasks, emits KPI receipts

4. **PAS Execution Engine** (`services/pas/executor.py`)
   - Job queue (lane-based)
   - Fair-share scheduling (portfolio scheduler integration)
   - KPI validation gates

5. **KPI Receipt Emission** (`services/pas/kpi_emit.py`)
   - Already exists in PLMS! Reuse `services/plms/kpi_emit.py`

#### Acceptance Criteria (DRAFT - NEEDS REFINEMENT)

- [ ] PAS accepts job from LCO: `POST /submit` → run_id
- [ ] PAS executes task tree with lane assignments
- [ ] PAS emits KPI receipts: test_pass_rate, build_success, etc.
- [ ] PAS validates KPI gates (no "green echo, red tests")
- [ ] PAS streams receipts to LCO: `GET /receipts/{run_id}` (SSE)
- [ ] PAS integrates with PLMS: updates `project_runs` table
- [ ] Two concurrent projects complete with fair-share scheduling

#### CRITICAL: THIS NEEDS A SEPARATE PRD!

**Action Item**: Create `PRD_PAS_Project_Agentic_System.md` BEFORE starting Phase 4

**Estimated Scope**: 4 weeks, 1 engineer + 1 ops, 5000+ LOC

---

### Phase 5: Planner Learning LLM - Weeks 9-10

**Goal**: Train project-experience model on completed runs (LOCAL + GLOBAL partitions)

**Why Last**: Optimization layer after base functionality works

#### Deliverables

1. **Training Data Pipeline** (`services/plms/planner_training.py`)
   - After completion: PLMS emits `planner_training_pack.json`
   - Sanitized: task tree, lane ids, provider matrix, rehearsal outcomes, KPI results
   - Partitions: LOCAL (per project repo) + GLOBAL (portfolio)

2. **Trainer Agent** (`services/plms/trainer_agent.py`)
   - Fine-tunes or LoRA-adapts Planner model (Llama 3.1 base)
   - Dual partitions: `artifacts/planner_local/{repo_id}/` + `artifacts/planner_global/`
   - A/B validation: Re-run same template, compare units (time, tokens, cost, energy)

3. **Serving** (`services/plms/planner_model.py`)
   - Planner uses GLOBAL first, overlays LOCAL deltas if repo/team matches
   - Cold-start: fallback to default priors + CI bands

#### Acceptance Criteria

- [ ] After 10 baseline runs, training pipeline generates `planner_training_pack.json`
- [ ] Trainer agent fine-tunes model (LoRA) on LOCAL + GLOBAL partitions
- [ ] A/B test: Updated Planner reduces MAE by ≥15% (10-run median)
- [ ] Estimation MAE% drops over time (goal: ≤20% at 10 projects)
- [ ] Rework rate ↓, KPI violations ↓, budget overruns ↓

#### Implementation Commands (SKETCH)

```bash
# After 10 runs, collect training data
./.venv/bin/python services/plms/planner_training.py --export artifacts/planner_training_pack.json

# Train LoRA adapter (LOCAL partition)
./.venv/bin/python services/plms/trainer_agent.py \
  --input artifacts/planner_training_pack.json \
  --partition local \
  --repo-id lnsp-phase-4 \
  --out artifacts/planner_local/lnsp-phase-4/lora_adapter.pt

# Train LoRA adapter (GLOBAL partition)
./.venv/bin/python services/plms/trainer_agent.py \
  --input artifacts/planner_training_pack.json \
  --partition global \
  --out artifacts/planner_global/lora_adapter.pt

# A/B test (re-run project template 43 with updated Planner)
./.venv/bin/python services/plms/ab_test.py \
  --template-id 43 \
  --baseline-model llama3.1:8b \
  --candidate-model llama3.1:8b+lora \
  --runs 10
```

---

## 📊 Integration Timeline (Gantt Chart)

```
Week │ Phase 1       │ Phase 2    │ Phase 3       │ Phase 4 (PAS) │ Phase 5
─────┼───────────────┼────────────┼───────────────┼───────────────┼──────────
  1  │ LightRAG Code │            │               │               │
  2  │ Index + Mgr   │ Telemetry  │               │               │
  3  │               │            │ LCO CLI       │               │
  4  │               │            │ (Read-Only)   │               │
  5  │               │            │               │ PAS Architect │
  6  │               │            │               │ PAS Directors │
  7  │               │            │               │ PAS Managers  │
  8  │               │            │               │ PAS Execution │
  9  │               │            │               │               │ Planner
 10  │               │            │               │               │ Learning
─────┴───────────────┴────────────┴───────────────┴───────────────┴──────────
     │ Standalone    │ Quick Win  │ Planning Only │ FULL STACK    │ Optimize
```

---

## 🚨 Risks & Mitigation Strategies

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **PAS doesn't exist (no PRD)** | 🔴 CRITICAL | 100% | Create PRD_PAS first, allocate 4 weeks |
| **LCO + PAS tight coupling** | 🟡 HIGH | 80% | Decouple: LCO read-only works without PAS |
| **Planner Learning data quality** | 🟡 HIGH | 60% | Calibration quarantine (baseline+hotfix only) |
| **LightRAG code index >2 min refresh** | 🟢 MEDIUM | 40% | Async background refresh, serve stale index |
| **Energy estimates inaccurate** | 🟢 LOW | 90% | Label as ESTIMATED, allow per-cluster overrides |

---

## ✅ Acceptance Criteria (Done Means Done)

### Phase 1: LightRAG Code Index
- [ ] Code index refreshes within 2 min after commit
- [ ] Query API returns correct results for all 4 query types
- [ ] Coverage ≥98%, latency P95 ≤300ms

### Phase 2: Multi-Metric Telemetry
- [ ] `/metrics?breakdown=all` returns token breakdown + energy/carbon
- [ ] HMI shows stacked bars (time/tokens/cost/energy)
- [ ] Compare runs shows % deltas vs baseline

### Phase 3: LCO Terminal Client
- [ ] `vp new`, `vp estimate`, `vp status` work without PAS
- [ ] Model broker chooses Ollama + Llama 3.1 correctly
- [ ] No file edits (read-only mode enforced)

### Phase 4: PAS Integration
- [ ] **SEPARATE PRD REQUIRED**
- [ ] PAS executes jobs, emits KPI receipts, validates gates
- [ ] Two concurrent projects complete with fair-share scheduling

### Phase 5: Planner Learning
- [ ] Training pipeline generates `planner_training_pack.json`
- [ ] A/B test shows ≥15% MAE reduction (10-run median)
- [ ] Estimation MAE drops to ≤20% at 10 projects

---

## 📚 Documentation Requirements

### Before Starting
- [ ] Create `PRD_PAS_Project_Agentic_System.md` (CRITICAL!)
- [ ] Update `CLAUDE.md` with LightRAG code index usage
- [ ] Update `docs/DATABASE_LOCATIONS.md` with new storage paths

### During Implementation
- [ ] LightRAG Code Index API docs (`docs/LIGHTRAG_CODE_INDEX.md`)
- [ ] LCO terminal client usage guide (`docs/LCO_TERMINAL_CLIENT.md`)
- [ ] PAS architecture diagram (`docs/PAS_ARCHITECTURE.md`)

### After Completion
- [ ] Integration test report (all 5 phases)
- [ ] Performance benchmarks (latency, throughput, accuracy)
- [ ] Lessons learned + tuning recommendations

---

## 🎯 Next Steps (What to Do NOW)

### Immediate Actions (This Week)

1. **Review this plan with stakeholders** (30 min)
   - Engineering: Feasibility check for PAS scope
   - Operations: Resource allocation for 10-week rollout
   - PM: Sign-off on phased delivery vs waterfall

2. **Create PRD_PAS_Project_Agentic_System.md** (4 hours) 🔴 CRITICAL
   - Define: Architect, Directors, Managers, Executors
   - API contracts: `/submit`, `/receipts/{run_id}`, `/status`
   - KPI gates: test_pass_rate, build_success, schema_diff, etc.
   - Fair-share scheduling integration with PLMS portfolio scheduler

3. **Week 1 Kickoff: LightRAG Code Index** (Monday)
   ```bash
   # Day 1: Setup
   mkdir -p services/lightrag_code artifacts/lightrag_code_index
   ./.venv/bin/pip install tree-sitter tree-sitter-python

   # Day 2-3: Implement FastAPI service (refresh, query, snapshot)
   # Day 4-5: Git hook + Vector Manager + testing
   ```

### Decision Points

**Question 1**: Should we proceed with phased rollout (10 weeks) or wait for full PAS design?
- ✅ **Recommend phased**: Phases 1-2 deliver value immediately, decouple from PAS

**Question 2**: Who owns PAS architecture design?
- Needs: 1 senior engineer + 1 architect, 1 week for PRD + design review

**Question 3**: Can we ship LCO read-only mode without PAS execution?
- ✅ **Yes**: Planning/estimation/status queries work with PLMS API only

---

## 📞 Contact & Support

- **Integration Plan Questions**: See this document
- **PAS Architecture Design**: TBD (assign owner!)
- **PLMS API Issues**: See `docs/HMI_JSON_CONTRACTS_PLMS.md`
- **LightRAG Code Index**: See `docs/LIGHTRAG_CODE_INDEX.md` (to be created)

---

**END OF INTEGRATION PLAN**

_This plan requires PAS PRD creation before Phase 4 can proceed._
_Estimated total effort: 10 weeks, 1-2 engineers, 8000+ LOC_

# PAS Stub - Minimal Project Agentic System

**Version**: 0.1.0
**Purpose**: Enable Phase 3 (LCO MVP) while full PAS is built (Weeks 5-8)
**Status**: Ready for Testing

---

## 🎯 What is PAS Stub?

A **minimal, in-memory implementation** of the full PAS (Project Agentic System) API that:

- ✅ Accepts job cards from LCO (Local Code Operator)
- ✅ Tracks DAG in memory (no database)
- ✅ "Executes" tasks synthetically (sleeps + emits fake receipts/KPIs)
- ✅ Provides all endpoints matching production API (stable contract)
- ✅ Supports idempotency, rehearsal simulation, pause/resume

**Use Cases**:
- LCO development: `vp plan`, `vp estimate`, `vp start`, `vp status` work NOW
- HMI integration testing: Visualize progress, budget runway, KPIs
- API contract validation: Ensure endpoints match PRD spec

**NOT included** (use full PAS for these):
- ❌ Real task execution (no actual pytest, ruff, git, etc.)
- ❌ Persistent storage (all data in memory, lost on restart)
- ❌ Fair-share scheduling (no portfolio optimization)
- ❌ PLMS webhooks (no calibration feedback loop)

---

## 🚀 Quick Start

### Start PAS Stub

```bash
# Option 1: Using Makefile
make run-pas-stub

# Option 2: Direct uvicorn
./.venv/bin/uvicorn services.pas.stub.app:app --host 127.0.0.1 --port 6200 --reload
```

**Service runs on**: `http://localhost:6200`

### Health Check

```bash
curl http://localhost:6200/health | jq
# Expected:
# {
#   "status": "ok",
#   "active_runs": 0,
#   "total_tasks": 0,
#   "total_receipts": 0
# }
```

### Run End-to-End Demo

```bash
# Automated test script (submits 2 tasks, waits for synthetic execution)
bash tests/demos/demo_pas_stub_e2e.sh
```

**Expected Output**:
```
✅ End-to-End Demo Complete!

📊 Summary:
   - Run ID: r-2025-11-07-143052
   - Tasks submitted: 2
   - Lanes tested: Code-Impl, Vector-Ops
   - Idempotency: ✓ Verified
   - Synthetic execution: ✓ Completed
   - KPIs emitted: ✓ (check run status for violations)
```

---

## 📡 API Endpoints

### Core Endpoints (Stable Contract)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/pas/v1/jobcards` | POST | Submit job card |
| `/pas/v1/runs/start` | POST | Start run |
| `/pas/v1/runs/pause` | POST | Pause run |
| `/pas/v1/runs/resume` | POST | Resume run |
| `/pas/v1/runs/terminate` | POST | Terminate run |
| `/pas/v1/runs/simulate` | POST | Rehearsal simulation |
| `/pas/v1/runs/status` | GET | Run status (DAG, spend, KPIs) |
| `/pas/v1/portfolio/status` | GET | Portfolio status (all runs) |
| `/pas/v1/heartbeats` | POST | Task heartbeat |
| `/pas/v1/receipts` | POST | Task receipt |
| `/pas/v1/kpis` | POST | KPI receipt |

### OpenAPI Schema

**Auto-generated**: `http://localhost:6200/docs` (Swagger UI)
**ReDoc**: `http://localhost:6200/redoc`
**JSON Schema**: `http://localhost:6200/openapi.json`

---

## 🧪 Example Usage

### 1. Start a Run

```bash
curl -X POST http://localhost:6200/pas/v1/runs/start \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": 42,
    "run_id": "r-2025-11-07-001",
    "run_kind": "baseline",
    "rehearsal_pct": 0.0,
    "budget_caps": {
      "budget_usd": 50.0,
      "energy_kwh": 2.0
    }
  }'
```

**Response**:
```json
{
  "status": "executing",
  "run_id": "r-2025-11-07-001"
}
```

---

### 2. Submit Job Card (with Idempotency)

```bash
IDEMPOTENCY_KEY=$(uuidgen)

curl -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -d '{
    "project_id": 42,
    "run_id": "r-2025-11-07-001",
    "lane": "Code-Impl",
    "priority": 0.8,
    "deps": [],
    "payload": {
      "repo": "/path/to/repo",
      "goal": "implement /login endpoint",
      "tests": ["tests/test_login.py"]
    },
    "budget_usd": 1.50,
    "ci_width_hint": 0.3
  }'
```

**Response**:
```json
{
  "task_id": "task-abc12345"
}
```

**Idempotent Replay** (same key):
```bash
curl -X POST http://localhost:6200/pas/v1/jobcards \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -d '{ ... same payload ... }'
```

**Response**:
```json
{
  "task_id": "task-abc12345",
  "idempotent_replay": true
}
```

---

### 3. Check Run Status

```bash
curl "http://localhost:6200/pas/v1/runs/status?run_id=r-2025-11-07-001" | jq
```

**Response**:
```json
{
  "run_id": "r-2025-11-07-001",
  "status": "executing",
  "tasks_total": 2,
  "tasks_completed": 1,
  "tasks_failed": 0,
  "spend_usd": 0.17,
  "spend_energy_kwh": 0.023,
  "runway_minutes": 45,
  "kpi_violations": [],
  "tasks": [
    {
      "task_id": "task-abc12345",
      "lane": "Code-Impl",
      "status": "succeeded"
    },
    {
      "task_id": "task-def67890",
      "lane": "Vector-Ops",
      "status": "running"
    }
  ]
}
```

---

### 4. Simulate Rehearsal

```bash
curl -X POST http://localhost:6200/pas/v1/runs/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "r-2025-11-07-001",
    "rehearsal_pct": 0.01,
    "stratified": true
  }'
```

**Response**:
```json
{
  "strata_coverage": 1.0,
  "rehearsal_tokens": 150,
  "projected_tokens": 15000,
  "ci_lower": 13200,
  "ci_upper": 16800,
  "risk_score": 0.14
}
```

---

### 5. Portfolio Status

```bash
curl http://localhost:6200/pas/v1/portfolio/status | jq
```

**Response**:
```json
{
  "active_runs": 2,
  "queued_tasks": 5,
  "lane_utilization": {
    "Code-Impl": 0.85,
    "Data-Schema": 0.40,
    "Narrative": 0.60,
    "Vector-Ops": 0.30
  },
  "lane_caps": {
    "Code-Impl": 6,
    "Data-Schema": 2,
    "Narrative": 4,
    "Vector-Ops": 4
  },
  "fairness_weights": {
    "r-2025-11-07-001": 0.5,
    "r-2025-11-07-002": 0.5
  }
}
```

---

## 🔬 Synthetic Execution Details

### Lane Execution Times (Realistic Delays)

| Lane | Delay Range | Avg Duration |
|------|-------------|--------------|
| Code-Impl | 5-15 seconds | 10s |
| Data-Schema | 3-8 seconds | 5.5s |
| Narrative | 10-20 seconds | 15s |
| Vector-Ops | 2-5 seconds | 3.5s |
| Code-API-Design | 3-10 seconds | 6.5s |
| Graph-Ops | 4-12 seconds | 8s |

### Synthetic Receipts (Per Task)

Generated automatically after execution:

```python
{
  "task_id": "task-abc12345",
  "run_id": "r-2025-11-07-001",
  "lane": "Code-Impl",
  "provider": "synthetic:stub",
  "tokens_in": 1423,       # Random: 500-2000
  "tokens_out": 567,       # Random: 200-800
  "tokens_think": 102,     # Random: 50-200
  "time_ms": 8732,         # Based on lane delay
  "cost_usd": 0.18,        # Random: 0.05-0.30
  "energy_kwh": 0.027,     # Random: 0.01-0.05
  "echo_cos": 0.89,        # Random: 0.80-0.95
  "status": "succeeded"    # 90% success rate
}
```

### Synthetic KPIs (Lane-Specific)

**Code-Impl**:
- `test_pass_rate`: 0.85-1.0 (threshold: ≥0.90)
- `linter_pass`: true/false (matches task status)

**Data-Schema**:
- `schema_diff`: 0-2 (threshold: 0)

**Narrative**:
- `BLEU`: 0.35-0.85 (threshold: ≥0.40)

**Vector-Ops**:
- `index_freshness`: 30-180 seconds (threshold: ≤120s)

---

## 🔄 Migration to Full PAS

When full PAS is ready (Week 8):

### 1. Swap Service

```bash
# Stop stub
pkill -f "uvicorn services.pas.stub.app"

# Start full PAS
./.venv/bin/uvicorn services.pas.app:app --host 127.0.0.1 --port 6200
```

### 2. Verify Contract Compatibility

```bash
# All endpoints should return same structure
diff <(curl http://localhost:6200/openapi.json) <(curl http://localhost:6200/openapi.json)
# Expected: No differences (API contract preserved)
```

### 3. Update LCO Configuration

```python
# In cli/vp.py
PAS_API = "http://localhost:6200"  # No change needed!
```

**Zero code changes required** - stub and full PAS share same API contract

---

## 🧩 Integration Points

### LCO (VP Agent)

```python
# In cli/vp.py
import requests

PAS_API = "http://localhost:6200"

def vp_start(run_id: str):
    """Start a run via PAS."""
    resp = requests.post(
        f"{PAS_API}/pas/v1/runs/start",
        json={
            "project_id": 42,
            "run_id": run_id,
            "run_kind": "baseline",
        }
    )
    return resp.json()

def vp_status(run_id: str):
    """Get run status from PAS."""
    resp = requests.get(
        f"{PAS_API}/pas/v1/runs/status",
        params={"run_id": run_id}
    )
    return resp.json()
```

### HMI (Dashboard)

```javascript
// In frontend/src/pas_client.js
const PAS_API = "http://localhost:6200";

async function getRun Status(runId) {
  const resp = await fetch(`${PAS_API}/pas/v1/runs/status?run_id=${runId}`);
  return resp.json();
}

async function getPortfolioStatus() {
  const resp = await fetch(`${PAS_API}/pas/v1/portfolio/status`);
  return resp.json();
}
```

---

## 📚 Documentation

- **PRD**: `docs/PRDs/PRD_PAS_Project_Agentic_System.md`
- **Integration Plan**: `docs/PRDs/INTEGRATION_PLAN_LCO_LightRAG_Metrics.md`
- **API Docs**: `http://localhost:6200/docs` (Swagger UI)

---

## 🐛 Troubleshooting

### Issue: Service won't start

**Error**: `Address already in use`

**Solution**:
```bash
# Find and kill process on port 6200
lsof -ti:6200 | xargs kill -9

# Restart
make run-pas-stub
```

---

### Issue: Tasks never complete

**Symptom**: Run status shows `tasks_completed = 0` after 30 seconds

**Cause**: Background worker thread not started

**Solution**: Check logs for errors:
```bash
# Run with debug logging
./.venv/bin/uvicorn services.pas.stub.app:app --host 127.0.0.1 --port 6200 --log-level debug
```

---

### Issue: Idempotency not working

**Symptom**: Same `Idempotency-Key` returns different `task_id`

**Cause**: In-memory cache cleared (service restarted)

**Solution**: In-memory state is **NOT persistent**. For production, use Redis cache.

---

## 🎯 Next Steps

1. **Test PAS Stub** (Today):
   ```bash
   make run-pas-stub
   bash tests/demos/demo_pas_stub_e2e.sh
   ```

2. **Wire LCO to PAS** (Phase 3, Week 3):
   - Implement `vp start` → `POST /pas/v1/runs/start`
   - Implement `vp status` → `GET /pas/v1/runs/status`

3. **Build Full PAS** (Phase 4, Weeks 5-8):
   - Real scheduler (lane caps, fair-share)
   - Real executors (pytest, ruff, git, LightRAG)
   - PLMS webhooks (calibration feedback)

---

**End of PAS Stub README**

_This stub enables LCO MVP development while full PAS is built (10-week phased rollout)_

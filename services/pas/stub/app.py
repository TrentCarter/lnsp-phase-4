"""
PAS Stub - Minimal Project Agentic System for LCO MVP

This stub provides all PAS API endpoints with synthetic execution,
allowing Phase 3 (LCO MVP) to run while full PAS is built (Weeks 5-8).

Capabilities:
- In-memory queue (single process worker)
- Accepts /jobcards, tracks DAG in memory
- "Executes" tasks by sleeping and emitting synthetic receipts/KPIs
- Provides /runs/simulate with CI extrapolation
- All endpoints match production API (stable contract)

Start: make run-pas-stub
Health: curl http://localhost:6200/health
"""

import asyncio
import json
import random
import requests
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# ============================================================================
# Data Models
# ============================================================================


class JobCard(BaseModel):
    """Job card submitted by LCO."""

    project_id: int
    run_id: str
    lane: str
    priority: float = 0.5
    deps: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    budget_usd: float = 1.0
    ci_width_hint: float = 0.3
    idempotency_key: Optional[str] = None


class RunStart(BaseModel):
    """Start run request."""

    project_id: int
    run_id: str
    run_kind: str = "baseline"
    rehearsal_pct: float = 0.0
    budget_caps: Optional[Dict[str, float]] = None


class RunControl(BaseModel):
    """Pause/Resume/Terminate request."""

    run_id: str
    reason: Optional[str] = None


class SimulateRequest(BaseModel):
    """Rehearsal simulation request."""

    run_id: str
    rehearsal_pct: float = 0.01
    stratified: bool = True


class TaskReceipt(BaseModel):
    """Task execution receipt."""

    task_id: str
    run_id: str
    lane: str
    provider: str
    tokens_in: int
    tokens_out: int
    tokens_think: int
    time_ms: int
    cost_usd: float
    energy_kwh: float
    echo_cos: float
    status: str
    artifacts_path: Optional[str] = None


class KPIReceipt(BaseModel):
    """KPI validation receipt."""

    task_id: str
    run_id: str
    kpi_name: str
    value: Any
    pass_bool: bool
    threshold: Dict[str, Any]
    logs_path: Optional[str] = None


# ============================================================================
# In-Memory State (Stub)
# ============================================================================

RUNS: Dict[str, Dict] = {}  # run_id → run state
TASKS: Dict[str, Dict] = {}  # task_id → task state
DAG: Dict[str, List[str]] = defaultdict(list)  # run_id → [task_ids]
RECEIPTS: List[TaskReceipt] = []
KPI_RECEIPTS: List[KPIReceipt] = []
IDEMPOTENCY_CACHE: Dict[str, str] = {}  # idempotency_key → task_id


# ============================================================================
# Synthetic Execution Logic
# ============================================================================


def _execute_task_synthetic(task_id: str, lane: str, payload: Dict) -> tuple:
    """
    Simulate task execution with realistic delays.

    Returns: (receipt, kpis)
    """
    # Simulate execution time based on lane
    delays = {
        "Code-Impl": (5, 15),  # 5-15 seconds
        "Data-Schema": (3, 8),
        "Narrative": (10, 20),
        "Vector-Ops": (2, 5),
        "Code-API-Design": (3, 10),
        "Graph-Ops": (4, 12),
    }
    delay = random.uniform(*delays.get(lane, (5, 10)))
    time.sleep(delay)

    # Emit synthetic receipt
    receipt = TaskReceipt(
        task_id=task_id,
        run_id=TASKS[task_id]["run_id"],
        lane=lane,
        provider="synthetic:stub",
        tokens_in=random.randint(500, 2000),
        tokens_out=random.randint(200, 800),
        tokens_think=random.randint(50, 200),
        time_ms=int(delay * 1000),
        cost_usd=round(random.uniform(0.05, 0.30), 2),
        energy_kwh=round(random.uniform(0.01, 0.05), 3),
        echo_cos=round(random.uniform(0.80, 0.95), 2),
        status="succeeded" if random.random() > 0.1 else "failed",
    )

    # Emit synthetic KPIs (lane-specific)
    kpis = []
    if lane == "Code-Impl":
        test_pass_rate = round(random.uniform(0.85, 1.0), 2)
        kpis.append(
            KPIReceipt(
                task_id=task_id,
                run_id=TASKS[task_id]["run_id"],
                kpi_name="test_pass_rate",
                value=test_pass_rate,
                pass_bool=test_pass_rate >= 0.90,
                threshold={"min": 0.90},
            )
        )
        kpis.append(
            KPIReceipt(
                task_id=task_id,
                run_id=TASKS[task_id]["run_id"],
                kpi_name="linter_pass",
                value=receipt.status == "succeeded",
                pass_bool=receipt.status == "succeeded",
                threshold={"expected": True},
            )
        )
    elif lane == "Data-Schema":
        schema_diff = random.randint(0, 2)
        kpis.append(
            KPIReceipt(
                task_id=task_id,
                run_id=TASKS[task_id]["run_id"],
                kpi_name="schema_diff",
                value=schema_diff,
                pass_bool=schema_diff == 0,
                threshold={"expected": 0},
            )
        )
    elif lane == "Narrative":
        bleu = round(random.uniform(0.35, 0.85), 2)
        kpis.append(
            KPIReceipt(
                task_id=task_id,
                run_id=TASKS[task_id]["run_id"],
                kpi_name="BLEU",
                value=bleu,
                pass_bool=bleu >= 0.40,
                threshold={"min": 0.40},
            )
        )
    elif lane == "Vector-Ops":
        index_freshness = random.randint(30, 180)  # seconds
        kpis.append(
            KPIReceipt(
                task_id=task_id,
                run_id=TASKS[task_id]["run_id"],
                kpi_name="index_freshness",
                value=index_freshness,
                pass_bool=index_freshness <= 120,
                threshold={"max": 120},
            )
        )

    return receipt, kpis


def _execute_run(run_id: str):
    """Background worker to execute a run (topological order)."""
    start_time = time.time()
    tasks = DAG[run_id]

    for task_id in tasks:
        task = TASKS[task_id]

        # Check dependencies (simplified - assume linear DAG for stub)
        task["status"] = "running"

        # Execute (synthetic)
        receipt, kpis = _execute_task_synthetic(
            task_id, task["lane"], task["payload"]
        )

        # Store receipts
        RECEIPTS.append(receipt)
        KPI_RECEIPTS.extend(kpis)

        # Update task status
        task["status"] = receipt.status
        task["receipt"] = receipt.dict()
        task["kpis"] = [kpi.dict() for kpi in kpis]

    # Update run status
    failed_tasks = [t for t in tasks if TASKS[t]["status"] == "failed"]
    if failed_tasks:
        RUNS[run_id]["status"] = "needs_review"
        RUNS[run_id]["validation_pass"] = False
    else:
        RUNS[run_id]["status"] = "completed"
        RUNS[run_id]["validation_pass"] = True

    # 🆕 Notify HMI of Prime Directive completion
    duration = time.time() - start_time
    _notify_directive_complete(run_id, duration, tasks, failed_tasks)


def _notify_directive_complete(
    run_id: str, duration: float, tasks: List[str], failed_tasks: List[str]
):
    """
    Send Prime Directive completion signal to HMI via Registry action_logs.

    This creates a special action log entry with:
    - action_type: "directive_complete"
    - from_agent: "PAS_ROOT"
    - action_data: JSON with run summary

    The HMI will detect this entry and:
    1. Stop timeline auto-scroll
    2. Show "END OF PROJECT" banner
    3. Display final report
    """
    try:
        run = RUNS[run_id]

        completion_log = {
            "task_id": run_id,  # Use run_id as pseudo-task for grouping
            "parent_log_id": None,
            "timestamp": datetime.utcnow().isoformat(),
            "from_agent": "PAS_ROOT",
            "to_agent": "HMI",
            "action_type": "directive_complete",
            "action_name": "Prime Directive Complete",
            "action_data": json.dumps(
                {
                    "run_id": run_id,
                    "project_id": run["project_id"],
                    "tasks_total": len(tasks),
                    "tasks_succeeded": len(tasks) - len(failed_tasks),
                    "tasks_failed": len(failed_tasks),
                    "duration_seconds": round(duration, 2),
                    "validation_pass": run["validation_pass"],
                    "status": run["status"],
                }
            ),
            "status": "done",
            "tier_from": 0,  # PAS ROOT is tier 0 (above all agents)
            "tier_to": None,
        }

        # POST to Registry action_logs endpoint
        response = requests.post(
            "http://localhost:6121/action_logs",
            json=completion_log,
            timeout=2,
        )
        response.raise_for_status()

        print(f"✅ [PAS] Notified HMI of Prime Directive completion: {run_id}")
        print(
            f"   Tasks: {len(tasks) - len(failed_tasks)}/{len(tasks)} succeeded, "
            f"Duration: {duration:.1f}s, Validation: {run['validation_pass']}"
        )

    except requests.exceptions.RequestException as e:
        # Log warning but don't fail the run (notification is non-critical)
        print(f"⚠️ [PAS] Failed to notify HMI of completion (Registry unavailable): {e}")
    except Exception as e:
        print(f"⚠️ [PAS] Failed to notify HMI of completion (unexpected error): {e}")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="PAS Stub",
    description="Minimal Project Agentic System for LCO MVP",
    version="0.1.0",
)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "active_runs": len([r for r in RUNS.values() if r["status"] == "executing"]),
        "total_tasks": len(TASKS),
        "total_receipts": len(RECEIPTS),
    }


# ============================================================================
# 7.1 Submit Job Card
# ============================================================================


@app.post("/pas/v1/jobcards")
async def submit_jobcard(
    card: JobCard, idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key")
):
    """Submit a job card to PAS."""
    # Idempotency check
    if idempotency_key and idempotency_key in IDEMPOTENCY_CACHE:
        return {
            "task_id": IDEMPOTENCY_CACHE[idempotency_key],
            "idempotent_replay": True,
        }

    # Create task
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    TASKS[task_id] = {
        "task_id": task_id,
        "run_id": card.run_id,
        "lane": card.lane,
        "priority": card.priority,
        "deps": card.deps,
        "payload": card.payload,
        "budget_usd": card.budget_usd,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }

    # Add to DAG
    DAG[card.run_id].append(task_id)

    # Store idempotency key
    if idempotency_key:
        IDEMPOTENCY_CACHE[idempotency_key] = task_id

    return {"task_id": task_id}


# ============================================================================
# 7.2 Control Plane
# ============================================================================


@app.post("/pas/v1/runs/start")
async def start_run(run: RunStart):
    """Start a run."""
    if run.run_id in RUNS:
        raise HTTPException(status_code=409, detail="Run already started")

    RUNS[run.run_id] = {
        "run_id": run.run_id,
        "project_id": run.project_id,
        "run_kind": run.run_kind,
        "rehearsal_pct": run.rehearsal_pct,
        "budget_caps": run.budget_caps or {},
        "status": "executing",
        "started_at": datetime.utcnow().isoformat(),
    }

    # Start background execution
    threading.Thread(target=_execute_run, args=(run.run_id,), daemon=True).start()

    return {"status": "executing", "run_id": run.run_id}


@app.post("/pas/v1/runs/pause")
async def pause_run(req: RunControl):
    """Pause a run."""
    if req.run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    RUNS[req.run_id]["status"] = "paused"
    RUNS[req.run_id]["pause_reason"] = req.reason
    return {"status": "paused", "run_id": req.run_id}


@app.post("/pas/v1/runs/resume")
async def resume_run(req: RunControl):
    """Resume a run."""
    if req.run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    RUNS[req.run_id]["status"] = "executing"
    # Would restart background worker in production
    return {"status": "executing", "run_id": req.run_id}


@app.post("/pas/v1/runs/terminate")
async def terminate_run(req: RunControl):
    """Terminate a run."""
    if req.run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    RUNS[req.run_id]["status"] = "terminated"
    RUNS[req.run_id]["termination_reason"] = req.reason
    return {"status": "terminated", "run_id": req.run_id}


@app.post("/pas/v1/runs/simulate")
async def simulate_run(req: SimulateRequest):
    """Simulate a rehearsal run (returns CI extrapolation)."""
    if req.run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    # Synthetic rehearsal calculation
    rehearsal_tokens = int(15000 * req.rehearsal_pct)
    projected_tokens = 15000
    ci_lower = int(projected_tokens * 0.88)
    ci_upper = int(projected_tokens * 1.12)

    return {
        "strata_coverage": 1.0 if req.stratified else 0.8,
        "rehearsal_tokens": rehearsal_tokens,
        "projected_tokens": projected_tokens,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "risk_score": round(random.uniform(0.10, 0.25), 2),
    }


# ============================================================================
# 7.3 Status & Monitoring
# ============================================================================


@app.get("/pas/v1/runs/status")
async def run_status(run_id: str):
    """Get run status."""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    run = RUNS[run_id]
    tasks = [TASKS[tid] for tid in DAG.get(run_id, [])]

    tasks_total = len(tasks)
    tasks_completed = len([t for t in tasks if t["status"] in ["succeeded", "failed"]])
    tasks_failed = len([t for t in tasks if t["status"] == "failed"])

    # Calculate spend (from receipts)
    spend_usd = sum(
        r.cost_usd for r in RECEIPTS if r.run_id == run_id
    )
    spend_energy_kwh = sum(
        r.energy_kwh for r in RECEIPTS if r.run_id == run_id
    )

    # KPI violations
    kpi_violations = [
        {
            "task_id": kpi.task_id,
            "kpi_name": kpi.kpi_name,
            "expected": kpi.threshold,
            "actual": kpi.value,
            "pass": kpi.pass_bool,
        }
        for kpi in KPI_RECEIPTS
        if kpi.run_id == run_id and not kpi.pass_bool
    ]

    # Runway calculation (simplified)
    runway_minutes = 45 if tasks_completed < tasks_total else 0

    return {
        "run_id": run_id,
        "status": run["status"],
        "tasks_total": tasks_total,
        "tasks_completed": tasks_completed,
        "tasks_failed": tasks_failed,
        "spend_usd": round(spend_usd, 2),
        "spend_energy_kwh": round(spend_energy_kwh, 3),
        "runway_minutes": runway_minutes,
        "kpi_violations": kpi_violations,
        "tasks": [
            {
                "task_id": t["task_id"],
                "lane": t["lane"],
                "status": t["status"],
            }
            for t in tasks
        ],
    }


@app.get("/pas/v1/portfolio/status")
async def portfolio_status():
    """Get portfolio status (all runs)."""
    active_runs = [r for r in RUNS.values() if r["status"] == "executing"]
    queued_tasks = [t for t in TASKS.values() if t["status"] == "queued"]

    # Lane utilization (synthetic)
    lane_utilization = {
        "Code-Impl": 0.85,
        "Data-Schema": 0.40,
        "Narrative": 0.60,
        "Vector-Ops": 0.30,
    }
    lane_caps = {
        "Code-Impl": 6,
        "Data-Schema": 2,
        "Narrative": 4,
        "Vector-Ops": 4,
    }

    # Fairness weights (equal for stub)
    fairness_weights = {
        run["run_id"]: 1.0 / len(active_runs) if active_runs else 0
        for run in active_runs
    }

    return {
        "active_runs": len(active_runs),
        "queued_tasks": len(queued_tasks),
        "lane_utilization": lane_utilization,
        "lane_caps": lane_caps,
        "fairness_weights": fairness_weights,
    }


# ============================================================================
# 7.4 Telemetry I/O
# ============================================================================


@app.post("/pas/v1/heartbeats")
async def heartbeat(data: Dict[str, Any]):
    """Receive heartbeat from executor."""
    # Stub: Just acknowledge
    return {"status": "acknowledged"}


@app.post("/pas/v1/receipts")
async def submit_receipt(receipt: TaskReceipt):
    """Submit task receipt."""
    RECEIPTS.append(receipt)
    return {"status": "stored", "task_id": receipt.task_id}


@app.post("/pas/v1/kpis")
async def submit_kpi(kpi: KPIReceipt):
    """Submit KPI receipt."""
    KPI_RECEIPTS.append(kpi)
    return {"status": "stored", "task_id": kpi.task_id, "kpi_name": kpi.kpi_name}


# ============================================================================
# Main (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=6200)

#!/usr/bin/env python3
"""
PAS Root (P0 Production Scaffold)
Orchestration layer with NO AI logic.

Responsibilities:
- Accept Prime Directives
- Create run IDs
- Spawn background tasks
- Track run status
- Call Aider-LCO RPC for execution

Port: 6100
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
import httpx
import os
import json
import time
import pathlib

AIDER_RPC = os.getenv("AIDER_RPC_URL", "http://127.0.0.1:6130")

app = FastAPI(title="PAS Root", version="1.0")

class PrimeDirective(BaseModel):
    title: str
    description: str
    repo_root: str
    entry_files: List[str] = []
    goal: str
    budget_tokens_max: Optional[int] = 25000
    budget_cost_usd_max: Optional[float] = 2.00


class RunStatus(BaseModel):
    run_id: str
    status: str  # queued, running, completed, error
    message: Optional[str] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_s: Optional[float] = None


# In-memory run tracking (P0 only, will move to SQLite in P1)
RUNS: Dict[str, Dict[str, Any]] = {}


def _artifact_dir(run_id: str) -> pathlib.Path:
    """Create and return artifacts directory for run"""
    p = pathlib.Path(f"artifacts/runs/{run_id}")
    p.mkdir(parents=True, exist_ok=True)
    return p


async def _execute_prime_directive(run_id: str, pd: PrimeDirective):
    """
    Execute Prime Directive via Aider-LCO RPC (background task).

    Steps:
    1. Mark run as "running"
    2. Build instruction message for Aider
    3. Call Aider-LCO RPC
    4. Save artifacts
    5. Update run status
    """
    started_at = time.time()
    RUNS[run_id].update({
        "status": "running",
        "started_at": started_at
    })

    # Step 1: Build instruction for Aider
    nl = (
        f"You are refactoring/creating files to satisfy: {pd.goal}. "
        f"Repo root is {pd.repo_root}. "
        f"Respect existing code style and documentation conventions."
    )

    # Step 2: Resolve file paths
    files = [str(pathlib.Path(pd.repo_root) / f) for f in pd.entry_files]

    # Step 3: Call Aider-LCO RPC
    payload = {
        "message": nl,
        "files": files,
        "dry_run": False
    }

    try:
        async with httpx.AsyncClient(timeout=1200) as client:
            r = await client.post(f"{AIDER_RPC}/aider/edit", json=payload)
        r.raise_for_status()
        out = r.json()

        # Step 4: Save artifacts
        artifact_dir = _artifact_dir(run_id)
        artifact_dir.joinpath("aider_stdout.txt").write_text(out.get("stdout", ""))
        artifact_dir.joinpath("aider_response.json").write_text(json.dumps(out, indent=2))
        artifact_dir.joinpath("prime_directive.json").write_text(json.dumps(pd.dict(), indent=2))

        # Step 5: Update run status
        completed_at = time.time()
        RUNS[run_id].update({
            "status": "completed",
            "message": "Prime Directive executed via Aider",
            "completed_at": completed_at,
            "duration_s": round(completed_at - started_at, 2),
            "aider_duration_s": out.get("duration_s"),
            "aider_rc": out.get("rc")
        })

    except httpx.HTTPStatusError as e:
        # HTTP error from Aider-LCO RPC
        completed_at = time.time()
        error_detail = e.response.text if hasattr(e.response, 'text') else str(e)
        RUNS[run_id].update({
            "status": "error",
            "message": f"Aider-LCO RPC error: {error_detail}",
            "completed_at": completed_at,
            "duration_s": round(completed_at - started_at, 2)
        })

    except Exception as e:
        # Other errors (network, timeout, etc.)
        completed_at = time.time()
        RUNS[run_id].update({
            "status": "error",
            "message": f"Execution error: {str(e)}",
            "completed_at": completed_at,
            "duration_s": round(completed_at - started_at, 2)
        })


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "PAS Root",
        "port": 6100,
        "aider_rpc_url": AIDER_RPC,
        "runs_active": len([r for r in RUNS.values() if r["status"] == "running"]),
        "runs_total": len(RUNS)
    }


@app.post("/pas/prime_directives", response_model=RunStatus)
async def submit_prime_directive(pd: PrimeDirective, bg: BackgroundTasks):
    """
    Submit a Prime Directive for execution.

    Returns immediately with run_id and "queued" status.
    Actual execution happens in background.
    """
    run_id = str(uuid.uuid4())
    RUNS[run_id] = {
        "status": "queued",
        "title": pd.title,
        "goal": pd.goal,
        "ts": time.time()
    }

    # Spawn background task
    bg.add_task(_execute_prime_directive, run_id, pd)

    return RunStatus(run_id=run_id, status="queued")


@app.get("/pas/runs/{run_id}", response_model=RunStatus)
def get_status(run_id: str):
    """Get run status by ID"""
    if run_id not in RUNS:
        raise HTTPException(404, "run not found")

    st = RUNS[run_id]
    return RunStatus(
        run_id=run_id,
        status=st["status"],
        message=st.get("message"),
        started_at=st.get("started_at"),
        completed_at=st.get("completed_at"),
        duration_s=st.get("duration_s")
    )


@app.get("/pas/runs")
def list_runs(limit: int = 50, status: Optional[str] = None):
    """List recent runs (optionally filtered by status)"""
    runs = []
    for run_id, data in sorted(RUNS.items(), key=lambda x: x[1].get("ts", 0), reverse=True)[:limit]:
        if status and data["status"] != status:
            continue
        runs.append({
            "run_id": run_id,
            "status": data["status"],
            "title": data.get("title"),
            "ts": data.get("ts")
        })
    return {"runs": runs, "total": len(runs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6100)

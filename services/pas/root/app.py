#!/usr/bin/env python3
"""
PAS Root (P0 Production Scaffold)
Orchestration layer with NO AI logic.

Responsibilities:
- Accept Prime Directives
- Create run IDs
- Spawn background tasks
- Track run status
- Call Prog-Aider RPC for execution

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
import sys

# Add services/common to path for comms_logger and title_generator
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from common.comms_logger import get_logger, MessageType
from common.title_generator import generate_short_title_async

AIDER_RPC = os.getenv("AIDER_RPC_URL", "http://127.0.0.1:6130")

app = FastAPI(title="PAS Root", version="1.0")

# Initialize comms logger
logger = get_logger()

# Get programmer agent name from Aider RPC (fallback to "Prog-Qwen-001")
PROG_AGENT_NAME = "Prog-Qwen-001"  # Default
PROG_METADATA = {}  # Agent metadata (role, parent, tool, tier)
try:
    with httpx.Client(timeout=2.0) as client:
        resp = client.get(f"{AIDER_RPC}/health")
        if resp.status_code == 200:
            health_data = resp.json()
            # Use agent field directly (e.g., "Prog-Qwen-001")
            PROG_AGENT_NAME = health_data.get("agent", PROG_AGENT_NAME)
            PROG_METADATA = health_data.get("agent_metadata", {})
except Exception:
    pass  # Use default if health check fails

# P0: Stub Dir/Mgr hierarchy (will be full agents in Phase 1+)
# For now, PAS Root acts as Dir-Code → Mgr-Code-01 → Prog-{LLM}-{N}
DIR_CODE_NAME = PROG_METADATA.get("grandparent", "Dir-Code")
MGR_CODE_NAME = PROG_METADATA.get("parent", "Mgr-Code-01")

class PrimeDirective(BaseModel):
    title: str  # User-provided title (can be long)
    description: str
    repo_root: str
    entry_files: List[str] = []
    goal: str  # Full goal description (used for LLM short name generation)
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
    Execute Prime Directive via Prog-Aider RPC (background task).

    Steps:
    1. Mark run as "running"
    2. Build instruction message for Aider
    3. Call Prog-Aider RPC
    4. Save artifacts
    5. Update run status
    """
    started_at = time.time()
    RUNS[run_id].update({
        "status": "running",
        "started_at": started_at
    })

    # Get short_name and gateway_log_id from RUNS
    short_name = RUNS[run_id].get("short_name", pd.title)
    gateway_log_id = RUNS[run_id].get("gateway_log_id")

    # Log START separator (visual break in logs)
    logger.log_separator(f"START: {short_name} ({run_id[:8]})", run_id=run_id)

    # Log status change - link to gateway submission
    status_log_id = logger.log_status(
        from_agent="PAS Root",
        to_agent="Gateway",
        message=f"Started execution: {short_name}",
        run_id=run_id,
        status="running",
        progress=0.0,
        parent_log_id=gateway_log_id
    )

    # Step 1: Build instruction for Aider
    nl = (
        f"You are refactoring/creating files to satisfy: {pd.goal}. "
        f"Repo root is {pd.repo_root}. "
        f"Respect existing code style and documentation conventions."
    )

    # Step 2: Resolve file paths
    files = [str(pathlib.Path(pd.repo_root) / f) for f in pd.entry_files]

    # P0: Log delegation chain: Gateway → PAS Root → Architect → Dir → Mgr → Prog
    # (In full PAS, Architect/Dir/Mgr will be real AI agents. For P0, we log the chain for proper hierarchy)

    # Step A: PAS Root → Architect (top-level AI coordinator)
    architect_cmd_log_id = logger.log_cmd(
        from_agent="PAS Root",
        to_agent="Architect",
        message=f"Analyze and delegate Prime Directive: {short_name}",
        run_id=run_id,
        metadata={"task_type": "prime_directive", "goal": pd.goal},
        parent_log_id=status_log_id
    )

    # Step B: Architect → Dir-Code
    dir_cmd_log_id = logger.log_cmd(
        from_agent="Architect",
        to_agent=DIR_CODE_NAME,
        message=f"Delegate to Code Director: {short_name}",
        run_id=run_id,
        metadata={"task_type": "code_edit", "files_count": len(files)},
        parent_log_id=architect_cmd_log_id
    )

    # Step C: Dir-Code → Mgr-Code-01
    mgr_cmd_log_id = logger.log_cmd(
        from_agent=DIR_CODE_NAME,
        to_agent=MGR_CODE_NAME,
        message=f"Assign to Code Manager: {short_name}",
        run_id=run_id,
        metadata={"lane": "code", "files": files},
        parent_log_id=dir_cmd_log_id
    )

    # Step D: Mgr-Code-01 → Prog-{LLM}-{N}
    prog_cmd_log_id = logger.log_cmd(
        from_agent=MGR_CODE_NAME,
        to_agent=PROG_AGENT_NAME,
        message=f"Execute Prime Directive: {pd.goal[:100]}",
        run_id=run_id,
        metadata={"files": files, "short_name": short_name, "original_title": pd.title},
        parent_log_id=mgr_cmd_log_id
    )

    # Step 3: Call Prog RPC with parent_log_id
    payload = {
        "message": nl,
        "files": files,
        "dry_run": False,
        "run_id": run_id,  # Pass run_id for logging
        "parent_log_id": prog_cmd_log_id  # Pass parent_log_id for hierarchy tracking
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
        duration_s = round(completed_at - started_at, 2)
        llm_model = out.get("llm_model", "unknown")

        RUNS[run_id].update({
            "status": "completed",
            "message": "Prime Directive executed via Aider",
            "completed_at": completed_at,
            "duration_s": duration_s,
            "aider_duration_s": out.get("duration_s"),
            "aider_rc": out.get("rc"),
            "llm_model": llm_model
        })

        # Log completion chain (reverse): Prog → Mgr → Dir → Architect → PAS Root → Gateway
        # (Prog → Mgr is logged by Aider RPC)

        # Mgr-Code-01 → Dir-Code
        mgr_resp_log_id = logger.log_response(
            from_agent=MGR_CODE_NAME,
            to_agent=DIR_CODE_NAME,
            message=f"Task completed: {short_name}",
            llm_model=llm_model,
            run_id=run_id,
            status="completed",
            metadata={"duration_s": duration_s},
            parent_log_id=mgr_cmd_log_id
        )

        # Dir-Code → Architect
        dir_resp_log_id = logger.log_response(
            from_agent=DIR_CODE_NAME,
            to_agent="Architect",
            message=f"Code directive completed: {short_name}",
            llm_model=llm_model,
            run_id=run_id,
            status="completed",
            parent_log_id=dir_cmd_log_id
        )

        # Architect → PAS Root
        architect_resp_log_id = logger.log_response(
            from_agent="Architect",
            to_agent="PAS Root",
            message=f"Prime Directive analysis complete: {short_name}",
            llm_model=llm_model,
            run_id=run_id,
            status="completed",
            parent_log_id=architect_cmd_log_id
        )

        # PAS Root → Gateway
        logger.log_response(
            from_agent="PAS Root",
            to_agent="Gateway",
            message=f"Completed: {short_name}",
            llm_model=llm_model,
            run_id=run_id,
            status="completed",
            metadata={"duration_s": duration_s, "rc": out.get("rc")},
            parent_log_id=architect_resp_log_id
        )

        # Log END separator (visual break in logs)
        logger.log_separator(f"END: {short_name} ({run_id[:8]}) - {duration_s}s", run_id=run_id)

    except httpx.HTTPStatusError as e:
        # HTTP error from Prog-Aider RPC
        completed_at = time.time()
        duration_s = round(completed_at - started_at, 2)
        error_detail = e.response.text if hasattr(e.response, 'text') else str(e)

        RUNS[run_id].update({
            "status": "error",
            "message": f"Prog-Aider RPC error: {error_detail}",
            "completed_at": completed_at,
            "duration_s": duration_s
        })

        # Log error - link to Aider command
        logger.log_response(
            from_agent="PAS Root",
            to_agent="Gateway",
            message=f"Error: {error_detail[:200]}",
            run_id=run_id,
            status="error",
            metadata={"duration_s": duration_s},
            parent_log_id=prog_cmd_log_id
        )

        # Log END separator (error case)
        logger.log_separator(f"END: {short_name} ({run_id[:8]}) - ERROR after {duration_s}s", run_id=run_id)

    except Exception as e:
        # Other errors (network, timeout, etc.)
        completed_at = time.time()
        duration_s = round(completed_at - started_at, 2)

        RUNS[run_id].update({
            "status": "error",
            "message": f"Execution error: {str(e)}",
            "completed_at": completed_at,
            "duration_s": duration_s
        })

        # Log error - link to Aider command
        logger.log_response(
            from_agent="PAS Root",
            to_agent="Gateway",
            message=f"Execution error: {str(e)[:200]}",
            run_id=run_id,
            status="error",
            metadata={"duration_s": duration_s},
            parent_log_id=prog_cmd_log_id
        )

        # Log END separator (error case)
        logger.log_separator(f"END: {short_name} ({run_id[:8]}) - ERROR after {duration_s}s", run_id=run_id)


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

    Automatically generates a short, human-readable name from the goal.
    """
    run_id = str(uuid.uuid4())

    # Generate short name using LLM (falls back to truncated goal if LLM unavailable)
    llm_endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
    short_name = await generate_short_title_async(pd.goal, llm_endpoint)

    RUNS[run_id] = {
        "status": "queued",
        "title": pd.title,          # User-provided title (original)
        "short_name": short_name,   # LLM-generated short name (for HMI display)
        "goal": pd.goal,
        "ts": time.time()
    }

    # Log submission (use short_name for display) - capture log_id for parent tracking
    gateway_log_id = logger.log_cmd(
        from_agent="Gateway",
        to_agent="PAS Root",
        message=f"Submit Prime Directive: {short_name}",
        run_id=run_id,
        metadata={
            "goal": pd.goal,
            "short_name": short_name,
            "original_title": pd.title,
            "files": pd.entry_files,
            "budget_tokens_max": pd.budget_tokens_max,
            "budget_cost_usd_max": pd.budget_cost_usd_max
        }
    )

    # Store gateway_log_id in RUNS for background task
    RUNS[run_id]["gateway_log_id"] = gateway_log_id

    # Log response (queued) - link to parent
    logger.log_response(
        from_agent="PAS Root",
        to_agent="Gateway",
        message=f"Queued: {short_name}",
        run_id=run_id,
        status="queued",
        parent_log_id=gateway_log_id
    )

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
            "short_name": data.get("short_name"),  # LLM-generated short name for HMI
            "ts": data.get("ts")
        })
    return {"runs": runs, "total": len(runs)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6100)

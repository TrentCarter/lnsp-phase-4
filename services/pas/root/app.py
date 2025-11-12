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
import requests
import os
import json
import time
import pathlib
import sys
import asyncio
import sqlite3
import subprocess
import signal
from datetime import datetime
from typing import Optional

# Add services/common to path for comms_logger and title_generator
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from common.comms_logger import get_logger, MessageType
from common.title_generator import generate_short_title_async
from common.heartbeat import get_monitor, AgentState

AIDER_RPC = os.getenv("AIDER_RPC_URL", "http://127.0.0.1:6130")
ARCHITECT_URL = os.getenv("ARCHITECT_URL", "http://127.0.0.1:6110")

app = FastAPI(title="PAS Root", version="1.0")

# Initialize comms logger and heartbeat monitor
logger = get_logger()
heartbeat_monitor = get_monitor()

# Register PAS Root agent
heartbeat_monitor.register_agent(
    agent="PAS Root",
    parent=None,  # Root has no parent
    llm_model=None,  # PAS Root doesn't use LLM directly
    role="orchestrator",
    tier="root"
)

# HHMRS Phase 2 constants
# MAX_FAILED_TASKS moved to settings (artifacts/pas_settings.json) as max_llm_retries
GATEWAY_URL = os.getenv("GATEWAY_URL", "http://127.0.0.1:6120")

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
    artifacts: Optional[Dict[str, Any]] = None
    acceptance_results: Optional[Dict[str, Any]] = None
    actuals: Optional[Dict[str, Any]] = None
    lanes: Optional[Dict[str, Any]] = None


# In-memory run tracking (P0 only, will move to SQLite in P1)
RUNS: Dict[str, Dict[str, Any]] = {}


# === HHMRS Event Emission Helper ===

def _emit_hhmrs_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Emit HHMRS event to Event Stream for HMI chimes and TRON visualization

    Args:
        event_type: One of: 'hhmrs_timeout', 'hhmrs_restart', 'hhmrs_escalation', 'hhmrs_failure'
        data: Event data containing agent_id, message, etc.
    """
    try:
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()

        payload = {
            "event_type": event_type,
            "data": data
        }

        response = requests.post(
            "http://localhost:6102/broadcast",
            json=payload,
            timeout=1.0
        )

        if response.status_code != 200:
            print(f"Warning: Failed to emit HHMRS event {event_type}, status={response.status_code}")

    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        print(f"Warning: Event Stream not available for HHMRS event {event_type}")
    except Exception as e:
        print(f"Error emitting HHMRS event {event_type}: {e}")


def _artifact_dir(run_id: str) -> pathlib.Path:
    """Create and return artifacts directory for run"""
    p = pathlib.Path(f"artifacts/runs/{run_id}")
    p.mkdir(parents=True, exist_ok=True)
    return p


# === HHMRS Phase 2: Database Helper Functions ===

def _get_failure_count(agent_id: str) -> int:
    """Get failure count for agent from heartbeat monitor"""
    return heartbeat_monitor._failure_counts.get(agent_id, 0)


def _increment_failure_count(agent_id: str) -> int:
    """Increment and return failure count for agent"""
    current = _get_failure_count(agent_id)
    new_count = current + 1
    heartbeat_monitor._failure_counts[agent_id] = new_count
    return new_count


def _record_retry(agent_id: str, retry_type: str, retry_count: int, reason: str,
                  old_config: dict, new_config: dict, run_id: Optional[str] = None) -> None:
    """Record retry in retry_history table"""
    try:
        db_path = pathlib.Path("artifacts/registry/registry.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO retry_history (
                run_id, task_id, agent_id, retry_type, retry_count,
                reason, old_config, new_config, timestamp, outcome
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            None,  # task_id
            agent_id,
            retry_type,
            retry_count,
            reason,
            json.dumps(old_config) if old_config else None,
            json.dumps(new_config) if new_config else None,
            datetime.now().isoformat(),
            "pending"  # Will be updated after retry attempt
        ))

        conn.commit()
        conn.close()

    except Exception as e:
        logger.log_status(
            from_agent="PAS Root",
            to_agent="PAS Root",
            message=f"Failed to record retry in database: {e}",
            run_id=run_id,
            metadata={"error": str(e)}
        )


def _get_agent_port(agent_id: str) -> Optional[int]:
    """Get port for agent from AGENT_PORT_MAP"""
    port_map = {
        "Architect": 6110,
        "Director-Code": 6111,
        "Director-Models": 6112,
        "Director-Data": 6113,
        "Director-DevSecOps": 6114,
        "Director-Docs": 6115,
    }
    return port_map.get(agent_id)


async def _execute_prime_directive(run_id: str, pd: PrimeDirective):
    """
    Execute Prime Directive via Architect (background task).

    Steps:
    1. Mark run as "running"
    2. Submit Prime Directive to Architect
    3. Poll Architect for completion
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

    # Step 1: Resolve file paths
    files = [str(pathlib.Path(pd.repo_root) / f) for f in pd.entry_files]

    # Step 2: Log delegation to Architect
    architect_cmd_log_id = logger.log_cmd(
        from_agent="PAS Root",
        to_agent="Architect",
        message=f"Submit Prime Directive to Architect: {short_name}",
        run_id=run_id,
        metadata={"task_type": "prime_directive", "goal": pd.goal, "files": files},
        parent_log_id=status_log_id
    )

    # Step 3: Submit to Architect
    architect_payload = {
        "run_id": run_id,
        "prd": pd.goal,  # Full PRD text
        "title": short_name,
        "entry_files": files,
        "budget": {
            "tokens_max": pd.budget_tokens_max or 25000,
            "cost_usd_max": pd.budget_cost_usd_max or 2.00
        },
        "policy": {
            "require_cross_vendor_review": True,
            "protected_paths": ["app/", "contracts/", "scripts/", "docs/PRDs/"]
        },
        "approval_mode": "auto"
    }

    try:
        # Submit to Architect
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{ARCHITECT_URL}/submit", json=architect_payload)
        r.raise_for_status()
        architect_response = r.json()

        # Log Architect acceptance
        logger.log_response(
            from_agent="Architect",
            to_agent="PAS Root",
            message=f"Accepted Prime Directive: {short_name}",
            run_id=run_id,
            status="planning",
            parent_log_id=architect_cmd_log_id
        )

        # Step 4: Poll Architect for completion
        poll_interval = 10  # seconds
        max_wait_time = 3600  # 1 hour
        elapsed = 0

        while elapsed < max_wait_time:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    status_r = await client.get(f"{ARCHITECT_URL}/status/{run_id}")
                status_r.raise_for_status()
                architect_status = status_r.json()

                current_state = architect_status.get("state")

                if current_state in ["completed", "failed"]:
                    # Architect finished
                    break

            except Exception:
                # Ignore polling errors, continue waiting
                pass

        # Step 5: Save artifacts
        artifact_dir = _artifact_dir(run_id)
        artifact_dir.joinpath("architect_plan.json").write_text(
            json.dumps(architect_status, indent=2) if 'architect_status' in locals() else "{}"
        )
        artifact_dir.joinpath("prime_directive.json").write_text(json.dumps(pd.dict(), indent=2))

        # Step 6: Update run status
        completed_at = time.time()
        duration_s = round(completed_at - started_at, 2)

        if 'architect_status' in locals() and architect_status.get("state") == "completed":
            RUNS[run_id].update({
                "status": "completed",
                "message": "Prime Directive executed via Architect",
                "completed_at": completed_at,
                "duration_s": duration_s
            })

            # Log completion: Architect → PAS Root → Gateway
            architect_resp_log_id = logger.log_response(
                from_agent="Architect",
                to_agent="PAS Root",
                message=f"Prime Directive completed: {short_name}",
                run_id=run_id,
                status="completed",
                parent_log_id=architect_cmd_log_id
            )

            # PAS Root → Gateway
            logger.log_response(
                from_agent="PAS Root",
                to_agent="Gateway",
                message=f"Completed: {short_name}",
                run_id=run_id,
                status="completed",
                metadata={"duration_s": duration_s},
                parent_log_id=architect_resp_log_id
            )

            # Log END separator
            logger.log_separator(f"END: {short_name} ({run_id[:8]}) - {duration_s}s", run_id=run_id)
        else:
            # Failed or timeout
            RUNS[run_id].update({
                "status": "error",
                "message": "Architect execution failed or timed out",
                "completed_at": completed_at,
                "duration_s": duration_s
            })

            # Log error
            logger.log_response(
                from_agent="PAS Root",
                to_agent="Gateway",
                message=f"Error: Architect failed or timed out",
                run_id=run_id,
                status="error",
                metadata={"duration_s": duration_s},
                parent_log_id=architect_cmd_log_id
            )

            # Log END separator (error case)
            logger.log_separator(f"END: {short_name} ({run_id[:8]}) - ERROR after {duration_s}s", run_id=run_id)

    except httpx.HTTPStatusError as e:
        # HTTP error from Architect
        completed_at = time.time()
        duration_s = round(completed_at - started_at, 2)
        error_detail = e.response.text if hasattr(e.response, 'text') else str(e)

        RUNS[run_id].update({
            "status": "error",
            "message": f"Architect RPC error: {error_detail}",
            "completed_at": completed_at,
            "duration_s": duration_s
        })

        # Log error
        logger.log_response(
            from_agent="PAS Root",
            to_agent="Gateway",
            message=f"Error: {error_detail[:200]}",
            run_id=run_id,
            status="error",
            metadata={"duration_s": duration_s},
            parent_log_id=architect_cmd_log_id
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

        # Log error
        logger.log_response(
            from_agent="PAS Root",
            to_agent="Gateway",
            message=f"Execution error: {str(e)[:200]}",
            run_id=run_id,
            status="error",
            metadata={"duration_s": duration_s},
            parent_log_id=architect_cmd_log_id
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
        "architect_url": ARCHITECT_URL,
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
async def get_status(run_id: str):
    """Get run status by ID"""
    if run_id not in RUNS:
        raise HTTPException(404, "run not found")

    st = RUNS[run_id]

    # Query Architect for detailed lane information (artifacts, acceptance_results, actuals)
    artifacts = None
    acceptance_results = None
    actuals = None
    lanes = None

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            architect_r = await client.get(f"{ARCHITECT_URL}/status/{run_id}")

        if architect_r.status_code == 200:
            architect_data = architect_r.json()
            lanes = architect_data.get("lanes", {})

            # Extract artifacts, acceptance_results, actuals from completed lanes
            # For now, take data from the "Code" lane (primary lane in P0)
            if lanes:
                for lane_name, lane_data in lanes.items():
                    if lane_data.get("state") in ["completed", "failed"]:
                        # Use the first completed lane's data
                        # (In multi-lane scenarios, might want to aggregate)
                        artifacts = lane_data.get("artifacts")
                        acceptance_results = lane_data.get("acceptance_results")
                        actuals = lane_data.get("actuals")
                        break

    except Exception:
        # If Architect query fails, just return basic status
        # (run might not have reached Architect yet)
        pass

    return RunStatus(
        run_id=run_id,
        status=st["status"],
        message=st.get("message"),
        started_at=st.get("started_at"),
        completed_at=st.get("completed_at"),
        duration_s=st.get("duration_s"),
        artifacts=artifacts,
        acceptance_results=acceptance_results,
        actuals=actuals,
        lanes=lanes
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


# === HHMRS Phase 2: Grandchild Failure Handler ===

class GrandchildFailureAlert(BaseModel):
    """Grandchild failure alert from parent (Architect/Director)"""
    type: str  # "grandchild_failure"
    grandchild_id: str
    parent_id: str
    failure_count: int
    reason: str


async def mark_task_failed(agent_id: str, run_id: Optional[str] = None) -> dict:
    """Mark task as permanently failed after max failures exceeded"""

    max_llm_retries = heartbeat_monitor.max_llm_retries

    # Emit permanent failure event for HMI chimes and TRON visualization
    _emit_hhmrs_event('hhmrs_failure', {
        'agent_id': agent_id,
        'run_id': run_id,
        'max_llm_retries': max_llm_retries,
        'message': f"{agent_id} permanently failed after {max_llm_retries} LLM retry attempts"
    })

    logger.log_status(
        from_agent="PAS Root",
        to_agent="PAS Root",
        message=f"Task {agent_id} permanently failed after {max_llm_retries} attempts",
        run_id=run_id,
        metadata={"agent_id": agent_id, "max_failures": max_llm_retries}
    )

    # Update run status if we have a run_id
    if run_id and run_id in RUNS:
        RUNS[run_id].update({
            "status": "failed",
            "message": f"Permanently failed: {agent_id} (max failures exceeded)",
            "completed_at": time.time()
        })

    # Alert Gateway
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{GATEWAY_URL}/notify_run_failed",
                json={
                    "agent_id": agent_id,
                    "run_id": run_id,
                    "reason": "max_failures_exceeded"
                }
            )

        logger.log_status(
            from_agent="PAS Root",
            to_agent="Gateway",
            message=f"Notified Gateway of permanent failure: {agent_id}",
            run_id=run_id,
            metadata={"agent_id": agent_id}
        )
    except Exception as e:
        logger.log_status(
            from_agent="PAS Root",
            to_agent="PAS Root",
            message=f"Failed to notify Gateway: {e}",
            run_id=run_id,
            metadata={"error": str(e)}
        )

    return {"status": "permanently_failed", "agent_id": agent_id}


@app.post("/handle_grandchild_failure")
async def handle_grandchild_failure(alert: GrandchildFailureAlert):
    """
    Handle grandchild failure escalation from Architect/Directors

    HHMRS Phase 2 escalation strategy:
    - failure_count < 3: Retry with different LLM (Anthropic ↔ Ollama)
    - failure_count >= 3: Mark as permanently failed
    """
    grandchild_id = alert.grandchild_id
    parent_id = alert.parent_id

    # Get current failure count
    failure_count = _get_failure_count(grandchild_id)

    logger.log_status(
        from_agent=parent_id,
        to_agent="PAS Root",
        message=f"Grandchild escalation: {grandchild_id} (failure_count={failure_count})",
        run_id=None,
        metadata={
            "grandchild_id": grandchild_id,
            "parent_id": parent_id,
            "failure_count": failure_count,
            "reason": alert.reason
        }
    )

    # Check if we've exceeded max failures (load from settings)
    max_llm_retries = heartbeat_monitor.max_llm_retries
    if failure_count >= max_llm_retries:
        # Emit failure event for HMI chimes and TRON visualization
        _emit_hhmrs_event('hhmrs_failure', {
            'agent_id': grandchild_id,
            'parent_id': parent_id,
            'grandparent_id': 'PAS Root',
            'failure_count': failure_count,
            'reason': 'max_llm_retries_exceeded',
            'message': f"{grandchild_id} permanently failed after {failure_count} LLM retries"
        })
        # Permanent failure
        return await mark_task_failed(grandchild_id, run_id=None)

    # Try different LLM
    # Phase 2 simplified: Just log the LLM change intent
    # In full implementation, would restart agent with different LLM
    try:
        # Determine current and new LLM
        # For Phase 2, we'll just simulate the LLM switch
        old_llm = "claude-sonnet-4-5"  # Assume current is Anthropic
        new_llm = "llama3.1:8b"  # Switch to Ollama

        # If failure_count is even, switch back to Anthropic
        if failure_count % 2 == 1:
            old_llm = "llama3.1:8b"
            new_llm = "claude-sonnet-4-5"

        logger.log_status(
            from_agent="PAS Root",
            to_agent=grandchild_id,
            message=f"Retrying {grandchild_id} with different LLM: {old_llm} → {new_llm}",
            run_id=None,
            metadata={
                "grandchild_id": grandchild_id,
                "old_llm": old_llm,
                "new_llm": new_llm,
                "retry_count": failure_count + 1
            }
        )

        # Emit retry event for HMI chimes and TRON visualization
        _emit_hhmrs_event('hhmrs_restart', {
            'agent_id': grandchild_id,
            'parent_id': parent_id,
            'grandparent_id': 'PAS Root',
            'retry_type': 'llm_change',
            'old_llm': old_llm,
            'new_llm': new_llm,
            'retry_count': failure_count + 1,
            'message': f"Retrying {grandchild_id} with {new_llm} (attempt {failure_count + 1})"
        })

        # Increment failure count
        new_failure_count = _increment_failure_count(grandchild_id)

        # Record retry in database
        _record_retry(
            agent_id=grandchild_id,
            retry_type="llm_change",
            retry_count=new_failure_count,
            reason="max_restarts_exceeded",
            old_config={"llm": old_llm},
            new_config={"llm": new_llm},
            run_id=None
        )

        # TODO Phase 2: Implement actual process restart with different LLM
        # For now, just acknowledge the escalation
        return {
            "status": "retrying_with_different_llm",
            "grandchild_id": grandchild_id,
            "old_llm": old_llm,
            "new_llm": new_llm,
            "failure_count": new_failure_count,
            "note": "Phase 2: LLM switching not yet fully implemented"
        }

    except Exception as e:
        logger.log_status(
            from_agent="PAS Root",
            to_agent="PAS Root",
            message=f"Failed to retry {grandchild_id} with different LLM: {e}",
            run_id=None,
            metadata={"grandchild_id": grandchild_id, "error": str(e)}
        )

        # If retry failed, mark as permanently failed
        return await mark_task_failed(grandchild_id, run_id=None)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6100)

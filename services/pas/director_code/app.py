#!/usr/bin/env python3
"""
Director-Code Service - Code Lane Coordinator

Port: 6111
LLM: Gemini 2.5 Flash (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level tasks
- Monitor Managers
- Validate acceptance gates (tests, lint, coverage, reviews)
- Report to Architect

Contract: docs/contracts/DIRECTOR_CODE_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import httpx
import json
import uuid
import asyncio

# PAS common services
from services.common.heartbeat import get_monitor, AgentState
from services.common.job_queue import get_queue, JobCard, Lane, Role, Priority
from services.common.comms_logger import get_logger, MessageType

# Director-Code specific imports
from services.pas.director_code.decomposer import ManagerTaskDecomposer
from services.common.manager_executor import get_manager_executor
from services.common.manager_pool.manager_factory import get_manager_factory


app = FastAPI(title="Director-Code", version="1.0.0")

# Initialize systems
heartbeat_monitor = get_monitor()
job_queue = get_queue()
logger = get_logger()

# Register Director-Code agent
heartbeat_monitor.register_agent(
    agent="Dir-Code",
    parent="Architect",
    llm_model=os.getenv("DIR_CODE_LLM", "google/gemini-2.5-flash"),
    role="director",
    tier="coordinator"
)

# Manager task decomposer (LLM-powered)
decomposer = ManagerTaskDecomposer()

# Manager executor and factory
manager_executor = get_manager_executor()
manager_factory = get_manager_factory()

# Manager endpoints (dynamic - Managers register themselves)
MANAGER_ENDPOINTS: Dict[str, str] = {}

# In-memory job tracking (will move to DB in Phase 2)
JOBS: Dict[str, Dict[str, Any]] = {}


# === Pydantic Models ===

class JobCardInput(BaseModel):
    """Job card from Architect"""
    job_card: Dict[str, Any]


class ManagerTask(BaseModel):
    """Manager task definition"""
    manager_id: str
    task: str
    files: List[str]
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    expected_artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    acceptance: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)
    programmers: List[str] = Field(default_factory=list)
    requires_review: bool = False


class LaneReport(BaseModel):
    """Lane report to Architect"""
    lane: str = "Code"
    state: str  # completed | failed
    artifacts: Dict[str, str] = Field(default_factory=dict)
    acceptance_results: Dict[str, Any] = Field(default_factory=dict)
    actuals: Dict[str, Any] = Field(default_factory=dict)
    managers_used: Dict[str, str] = Field(default_factory=dict)


# === Health & Status Endpoints ===

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Director-Code",
        "version": "1.0.0",
        "port": 6111,
        "agent": "Dir-Code",
        "llm_model": os.getenv("DIR_CODE_LLM", "google/gemini-2.5-flash"),
        "agent_metadata": {
            "role": "director",
            "tier": "coordinator",
            "parent": "Architect",
            "children": list(MANAGER_ENDPOINTS.keys())
        }
    }


@app.get("/status/{job_card_id}")
async def get_status(job_card_id: str):
    """Get status for a job card"""
    if job_card_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job card not found")

    job_data = JOBS[job_card_id]
    return {
        "job_card_id": job_card_id,
        "state": job_data["state"],
        "message": job_data.get("message", ""),
        "managers": job_data.get("managers", {}),
        "started_at": job_data.get("started_at"),
        "completed_at": job_data.get("completed_at"),
        "duration_s": job_data.get("duration_s")
    }


# === HHMRS Phase 1: Child Timeout Handler ===

class ChildTimeoutAlert(BaseModel):
    """Timeout alert from TRON (HeartbeatMonitor)"""
    type: str  # "child_timeout"
    child_id: str
    reason: str
    restart_count: int
    last_seen_timestamp: float
    timeout_duration_s: float


@app.post("/handle_child_timeout")
async def handle_child_timeout(alert: ChildTimeoutAlert):
    """
    Handle child agent timeout alert from TRON

    HHMRS Phase 1 retry strategy:
    - restart_count < max_restarts: Restart child (Manager) with same config
    - restart_count >= max_restarts: Escalate to grandparent (Architect)

    Note: max_restarts is loaded from settings (artifacts/pas_settings.json)
    """
    # Load max_restarts from TRON settings
    max_restarts = heartbeat_monitor.max_restarts
    child_id = alert.child_id
    restart_count = alert.restart_count

    logger.log_message(
        from_agent="TRON",
        to_agent="Dir-Code",
        message=f"Child timeout alert: {child_id} (restart_count={restart_count})",
        run_id=None,
        metadata={
            "child_id": child_id,
            "restart_count": restart_count,
            "timeout_duration_s": alert.timeout_duration_s
        }
    )

    # Check if we should escalate to grandparent
    if restart_count >= max_restarts:
        # Escalate to Architect
        try:
            architect_url = os.getenv("ARCHITECT_URL", "http://127.0.0.1:6110")
            escalation = {
                "type": "grandchild_failure",
                "grandchild_id": child_id,
                "parent_id": "Dir-Code",
                "failure_count": restart_count,
                "reason": "max_restarts_exceeded"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{architect_url}/handle_grandchild_failure",
                    json=escalation,
                    timeout=10.0
                )

            if response.status_code == 200:
                logger.log_message(
                    from_agent="Dir-Code",
                    to_agent="Architect",
                    message=f"Escalated {child_id} failure to Architect",
                    run_id=None,
                    metadata={"child_id": child_id, "restart_count": restart_count}
                )
                return {
                    "status": "escalated",
                    "message": f"Escalated {child_id} to Architect",
                    "restart_count": restart_count
                }
            else:
                logger.log_message(
                    from_agent="Dir-Code",
                    to_agent="Dir-Code",
                    message=f"Failed to escalate {child_id} to Architect: {response.text}",
                    run_id=None,
                    metadata={"child_id": child_id, "status_code": response.status_code}
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Grandparent escalation failed: {response.text}"
                )

        except Exception as e:
            logger.log_message(
                from_agent="Dir-Code",
                to_agent="Dir-Code",
                message=f"Error escalating {child_id}: {str(e)}",
                run_id=None,
                metadata={"child_id": child_id, "error": str(e)}
            }
            raise HTTPException(
                status_code=500,
                detail=f"Failed to escalate to grandparent: {str(e)}"
            )

    # Attempt restart (simplified for Phase 1 - just log the action)
    # In a full implementation, this would:
    # 1. Kill child Manager process
    # 2. Clear Manager state
    # 3. Restart Manager
    # 4. Update retry count in TRON

    logger.log_message(
        from_agent="Dir-Code",
        to_agent=child_id,
        message=f"Restarting {child_id} (attempt {restart_count + 1})",
        run_id=None,
        metadata={"child_id": child_id, "restart_count": restart_count + 1}
    )

    # Update TRON retry count
    heartbeat_monitor._retry_counts[child_id] = restart_count + 1

    # TODO Phase 1: Implement actual Manager restart
    # For now, just acknowledge the timeout
    return {
        "status": "restarted",
        "message": f"Acknowledged timeout for {child_id}, restart scheduled",
        "restart_count": restart_count + 1,
        "note": "Phase 1: Manager restart not yet implemented"
    }


# === Main Job Card Submission Endpoint ===

@app.post("/submit")
async def submit_job_card(
    input_data: JobCardInput,
    background_tasks: BackgroundTasks
):
    """
    Submit job card from Architect for execution

    Flow:
    1. Validate and register job
    2. Start background task for decomposition
    3. Return immediately with job_card_id and status

    Background task:
    - Decompose job card using LLM
    - Create Manager tasks
    - Submit to Managers
    - Monitor execution
    - Validate acceptance
    - Report completion to Architect
    """
    job_card_dict = input_data.job_card
    job_card_id = job_card_dict["id"]

    # Validate job_card_id uniqueness
    if job_card_id in JOBS:
        raise HTTPException(status_code=400, detail="Job card ID already exists")

    # Register job
    JOBS[job_card_id] = {
        "state": "planning",
        "message": "Received job card, planning Manager tasks",
        "started_at": time.time(),
        "managers": {},
        "job_card": job_card_dict
    }

    # Log receipt
    logger.log_cmd(
        from_agent="Architect",
        to_agent="Dir-Code",
        message=f"Job card received: {job_card_dict.get('task', '')[:50]}...",
        run_id=job_card_dict.get("parent_id", "unknown"),
        metadata={"job_card_id": job_card_id}
    )

    # Start background execution
    background_tasks.add_task(execute_job_card, job_card_dict)

    # Send heartbeat
    heartbeat_monitor.heartbeat(
        agent="Dir-Code",
        run_id=job_card_dict.get("parent_id", "unknown"),
        state=AgentState.PLANNING,
        message=f"Planning: {job_card_dict.get('task', '')[:50]}...",
        progress=0.1
    )

    return {
        "job_card_id": job_card_id,
        "status": "planning",
        "message": "Job card accepted, decomposing into Manager tasks"
    }


# === Background Execution Logic ===

async def execute_job_card(job_card_dict: Dict[str, Any]):
    """
    Execute job card (background task)

    Steps:
    1. Decompose job card into Manager tasks (LLM-powered)
    2. Submit tasks to Managers
    3. Monitor Manager heartbeats
    4. Collect Manager reports
    5. Validate acceptance gates
    6. Generate lane report
    7. Report to Architect
    """
    job_card_id = job_card_dict["id"]
    run_id = job_card_dict.get("parent_id", "unknown")

    try:
        # Step 1: Decompose job card
        heartbeat_monitor.heartbeat(
            agent="Dir-Code",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing job card with LLM",
            progress=0.2
        )

        manager_plan = await decompose_job_card(job_card_dict)

        # Step 2: Submit tasks to Managers
        JOBS[job_card_id]["state"] = "delegating"
        heartbeat_monitor.heartbeat(
            agent="Dir-Code",
            run_id=run_id,
            state=AgentState.DELEGATING,
            message="Submitting tasks to Managers",
            progress=0.4
        )

        await delegate_to_managers(job_card_id, run_id, manager_plan)

        # Step 3: Monitor execution
        JOBS[job_card_id]["state"] = "executing"
        heartbeat_monitor.heartbeat(
            agent="Dir-Code",
            run_id=run_id,
            state=AgentState.MONITORING,
            message="Monitoring Managers",
            progress=0.5
        )

        await monitor_managers(job_card_id, run_id, manager_plan)

        # Step 4: Validate acceptance
        JOBS[job_card_id]["state"] = "validating"
        heartbeat_monitor.heartbeat(
            agent="Dir-Code",
            run_id=run_id,
            state=AgentState.VALIDATING,
            message="Validating acceptance gates",
            progress=0.9
        )

        validation_results = await validate_acceptance(job_card_id, run_id, manager_plan)

        # Step 5: Complete
        if all(validation_results.values()):
            JOBS[job_card_id]["state"] = "completed"
            JOBS[job_card_id]["message"] = "All Manager tasks completed successfully"
            JOBS[job_card_id]["completed_at"] = time.time()
            JOBS[job_card_id]["duration_s"] = JOBS[job_card_id]["completed_at"] - JOBS[job_card_id]["started_at"]

            heartbeat_monitor.heartbeat(
                agent="Dir-Code",
                run_id=run_id,
                state=AgentState.COMPLETED,
                message="Job card completed",
                progress=1.0
            )

            # Generate lane report
            lane_report = generate_lane_report(job_card_id, run_id, manager_plan, validation_results)

            # Report to Architect
            await report_to_architect(run_id, lane_report)

        else:
            # Some managers failed
            failed_managers = [mgr for mgr, passed in validation_results.items() if not passed]
            JOBS[job_card_id]["state"] = "failed"
            JOBS[job_card_id]["message"] = f"Managers failed: {', '.join(failed_managers)}"
            JOBS[job_card_id]["completed_at"] = time.time()

            heartbeat_monitor.heartbeat(
                agent="Dir-Code",
                run_id=run_id,
                state=AgentState.FAILED,
                message=f"Failed: {', '.join(failed_managers)}",
                progress=1.0
            )

            # Generate failure report
            lane_report = generate_lane_report(job_card_id, run_id, manager_plan, validation_results)
            await report_to_architect(run_id, lane_report)

    except Exception as e:
        # Unhandled error
        JOBS[job_card_id]["state"] = "failed"
        JOBS[job_card_id]["message"] = f"Error: {str(e)}"
        JOBS[job_card_id]["completed_at"] = time.time()

        heartbeat_monitor.heartbeat(
            agent="Dir-Code",
            run_id=run_id,
            state=AgentState.FAILED,
            message=f"Error: {str(e)}",
            progress=1.0
        )

        logger.log_response(
            from_agent="Dir-Code",
            to_agent="Architect",
            message=f"Error executing job card: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"error": str(e), "job_card_id": job_card_id}
        )


async def decompose_job_card(job_card_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decompose job card into Manager tasks using LLM

    Returns:
        Dict with manager_tasks and dependencies
    """
    # Use ManagerTaskDecomposer (LLM-powered)
    result = await decomposer.decompose(
        job_card_id=job_card_dict["id"],
        task=job_card_dict["task"],
        inputs=job_card_dict.get("inputs", []),
        expected_artifacts=job_card_dict.get("expected_artifacts", []),
        acceptance=job_card_dict.get("acceptance", []),
        risks=job_card_dict.get("risks", []),
        budget=job_card_dict.get("budget", {})
    )

    return result


async def delegate_to_managers(job_card_id: str, run_id: str, manager_plan: Dict[str, Any]):
    """
    Execute Manager tasks directly via Manager executor

    For each manager in manager_plan:
    1. Create Manager metadata via Factory
    2. Execute task via Manager executor (calls Aider RPC)
    3. Store results
    """
    for manager_task in manager_plan["manager_tasks"]:
        manager_id = manager_task["manager_id"]

        # Create Manager metadata (track in Manager Pool)
        actual_manager_id = manager_factory.create_manager(
            lane="Code",
            director="Dir-Code",
            job_card_id=job_card_id,
            llm_model=manager_task.get("programmers", [None])[0] if manager_task.get("programmers") else None,
            metadata={
                "task": manager_task["task"],
                "files": manager_task.get("files", []),
                "requires_review": manager_task.get("requires_review", False)
            }
        )

        # Update job state - mark as executing
        JOBS[job_card_id]["managers"][actual_manager_id] = {
            "state": "executing",
            "job_card_id": f"{job_card_id}-{manager_id.lower()}",
            "task": manager_task["task"],
            "started_at": time.time()
        }

        # Execute task via Manager executor (calls Aider RPC)
        result = await manager_executor.execute_manager_task(
            manager_id=actual_manager_id,
            task=manager_task["task"],
            files=manager_task.get("files", []),
            run_id=run_id,
            director="Dir-Code",
            acceptance=manager_task.get("acceptance", []),
            budget=manager_task.get("budget", {}),
            programmers=manager_task.get("programmers", [])
        )

        # Update job state based on result
        if result["success"]:
            JOBS[job_card_id]["managers"][actual_manager_id]["state"] = "completed"
            JOBS[job_card_id]["managers"][actual_manager_id]["result"] = result
            JOBS[job_card_id]["managers"][actual_manager_id]["completed_at"] = time.time()
        else:
            JOBS[job_card_id]["managers"][actual_manager_id]["state"] = "failed"
            JOBS[job_card_id]["managers"][actual_manager_id]["result"] = result
            JOBS[job_card_id]["managers"][actual_manager_id]["completed_at"] = time.time()
            JOBS[job_card_id]["managers"][actual_manager_id]["error"] = result.get("output", "Unknown error")


async def monitor_managers(job_card_id: str, run_id: str, manager_plan: Dict[str, Any]):
    """
    Monitor Managers - simplified since execution is now synchronous

    Since we execute Manager tasks directly in delegate_to_managers(),
    all Managers are already complete by the time we reach this function.
    This is now a no-op, but kept for API compatibility.
    """
    # All Manager tasks are already executed synchronously in delegate_to_managers()
    # So monitoring is a no-op - just verify all are complete

    for manager_id, manager_data in JOBS[job_card_id]["managers"].items():
        if manager_data["state"] not in ["completed", "failed"]:
            # This shouldn't happen, but handle it just in case
            logger.log_status(
                from_agent="Dir-Code",
                to_agent="Architect",
                message=f"Manager {manager_id} in unexpected state: {manager_data['state']}",
                run_id=run_id,
                status="warning"
            )


async def validate_acceptance(job_card_id: str, run_id: str, manager_plan: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate acceptance gates for each Manager

    Since Manager executor already validates acceptance,
    we just collect the results from the Manager execution.

    Returns:
        Dict mapping manager_id to pass/fail boolean
    """
    results = {}

    for manager_id, manager_data in JOBS[job_card_id]["managers"].items():
        # Manager executor already validated acceptance
        # Check if Manager completed successfully
        results[manager_id] = manager_data.get("state") == "completed"

    return results


def generate_lane_report(
    job_card_id: str,
    run_id: str,
    manager_plan: Dict[str, Any],
    validation_results: Dict[str, bool]
) -> LaneReport:
    """
    Generate lane report for Architect

    Aggregates Manager results into lane-level summary
    """
    job_data = JOBS[job_card_id]

    # Determine state
    if all(validation_results.values()):
        state = "completed"
    else:
        state = "failed"

    # Collect artifacts
    artifacts = {
        "diffs": f"artifacts/runs/{run_id}/code/diffs/",
        "test_results": f"artifacts/runs/{run_id}/code/test_results.json",
        "coverage": f"artifacts/runs/{run_id}/code/coverage.json",
        "lint_report": f"artifacts/runs/{run_id}/code/lint_report.txt"
    }

    # Collect acceptance results (placeholder - will be real in full implementation)
    acceptance_results = {
        "pytest": 0.92,  # TODO: Get from actual test results
        "lint": 0,
        "coverage": 0.87
    }

    # Collect actuals
    actuals = {
        "tokens": 0,  # TODO: Sum from Managers
        "duration_mins": (job_data["completed_at"] - job_data["started_at"]) / 60 if "completed_at" in job_data else 0,
        "cost_usd": 0.0  # TODO: Calculate from token usage
    }

    # Collect managers used
    managers_used = {
        task["manager_id"]: task["task"][:50] + "..."
        for task in manager_plan["manager_tasks"]
    }

    return LaneReport(
        lane="Code",
        state=state,
        artifacts=artifacts,
        acceptance_results=acceptance_results,
        actuals=actuals,
        managers_used=managers_used
    )


async def report_to_architect(run_id: str, lane_report: LaneReport):
    """
    Report lane completion to Architect

    Sends lane report via RPC or queue
    """
    architect_url = os.getenv("ARCHITECT_URL", "http://127.0.0.1:6110")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Add run_id to the lane report for Architect
            report_data = lane_report.dict()
            report_data["run_id"] = run_id

            response = await client.post(
                f"{architect_url}/lane_report",
                json=report_data
            )
            response.raise_for_status()

        logger.log_response(
            from_agent="Dir-Code",
            to_agent="Architect",
            message=f"Lane report submitted: {lane_report.state}",
            run_id=run_id,
            status=lane_report.state,
            metadata={"lane": "Code"}
        )

    except Exception as e:
        # Fallback to file-based communication
        lane_report_path = Path(f"artifacts/runs/{run_id}/code/lane_report.json")
        lane_report_path.parent.mkdir(parents=True, exist_ok=True)
        lane_report_path.write_text(json.dumps(lane_report.dict(), indent=2))

        logger.log_response(
            from_agent="Dir-Code",
            to_agent="Architect",
            message=f"Lane report saved (RPC failed): {lane_report.state}",
            run_id=run_id,
            status=lane_report.state,
            metadata={"lane": "Code", "rpc_error": str(e)}
        )


# === Manager Registration Endpoint ===

@app.post("/register_manager")
async def register_manager(manager_id: str, endpoint: str):
    """
    Allow Managers to register their RPC endpoints

    This enables Dir-Code to submit tasks directly via RPC
    instead of using file-based fallback
    """
    MANAGER_ENDPOINTS[manager_id] = endpoint

    logger.log_cmd(
        from_agent=manager_id,
        to_agent="Dir-Code",
        message=f"Manager registered: {endpoint}",
        run_id="system",
        metadata={"manager_id": manager_id, "endpoint": endpoint}
    )

    return {
        "status": "ok",
        "message": f"Manager {manager_id} registered",
        "endpoint": endpoint
    }


# === Startup ===

if __name__ == "__main__":
    import uvicorn
    import asyncio

    # Start heartbeat emitter
    async def emit_heartbeats():
        while True:
            heartbeat_monitor.heartbeat(
                agent="Dir-Code",
                state=AgentState.IDLE,
                message="Idle, waiting for job cards"
            )
            await asyncio.sleep(60)

    # Run server
    uvicorn.run(app, host="127.0.0.1", port=6111)

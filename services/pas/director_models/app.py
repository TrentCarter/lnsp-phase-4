#!/usr/bin/env python3
"""
Director-Code Service - Models Lane Coordinator

Port: 6112
LLM: Gemini 2.5 Flash (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level tasks
- Monitor Managers
- Validate acceptance gates (training, evaluation, KPI gates)
- Report to Architect

Contract: docs/contracts/DIRECTOR_MODELS_SYSTEM_PROMPT.md
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

# Director-Models specific imports
from services.pas.director_models.decomposer import ManagerTaskDecomposer
from services.common.manager_executor import get_manager_executor
from services.common.manager_pool.manager_factory import get_manager_factory

# Agent chat for Parent-Child communication
from services.common.agent_chat import get_agent_chat_client, AgentChatMessage

# LLM with tool support (Phase 3)
from services.common.llm_tool_caller import call_llm_with_tools, LLMResponse
from services.common.llm_tools import get_ask_parent_tool, validate_ask_parent_args, get_system_prompt_with_ask_parent


app = FastAPI(title="Director-Models", version="1.0.0")

# Initialize systems
heartbeat_monitor = get_monitor()
job_queue = get_queue()
logger = get_logger()
agent_chat = get_agent_chat_client()

# Manager executor and factory
manager_executor = get_manager_executor()
manager_factory = get_manager_factory()

# Register Director-Code agent
heartbeat_monitor.register_agent(
    agent="Dir-Models",
    parent="Architect",
    llm_model=os.getenv("DIR_MODELS_LLM", "anthropic/claude-sonnet-4-5"),
    role="director",
    tier="coordinator"
)

# Manager task decomposer (LLM-powered)
decomposer = ManagerTaskDecomposer()

# Manager endpoints (dynamic - Managers register themselves)
MANAGER_ENDPOINTS: Dict[str, str] = {}

# In-memory job tracking (will move to DB in Phase 2)
JOBS: Dict[str, Dict[str, Any]] = {}


# === HHMRS Event Emission Helper ===

def _emit_hhmrs_event(event_type: str, data: Dict[str, Any]) -> None:
    """
    Emit HHMRS event to Event Stream for HMI chimes and TRON visualization

    Args:
        event_type: One of: 'hhmrs_timeout', 'hhmrs_restart', 'hhmrs_escalation', 'hhmrs_failure'
        data: Event data containing agent_id, message, etc.
    """
    try:
        import requests
        from datetime import datetime

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
    lane: str = "Models"
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
        "service": "Director-Models",
        "version": "1.0.0",
        "port": 6112,
        "agent": "Dir-Models",
        "llm_model": os.getenv("DIR_MODELS_LLM", "anthropic/claude-sonnet-4-5"),
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
        to_agent="Dir-Models",
        message=f"Job card received: {job_card_dict.get('task', '')[:50]}...",
        run_id=job_card_dict.get("parent_id", "unknown"),
        metadata={"job_card_id": job_card_id}
    )

    # Start background execution
    background_tasks.add_task(execute_job_card, job_card_dict)

    # Send heartbeat
    heartbeat_monitor.heartbeat(
        agent="Dir-Models",
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
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing job card with LLM",
            progress=0.2
        )

        manager_plan = await decompose_job_card(job_card_dict)

        # Step 2: Submit tasks to Managers
        JOBS[job_card_id]["state"] = "delegating"
        heartbeat_monitor.heartbeat(
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.DELEGATING,
            message="Submitting tasks to Managers",
            progress=0.4
        )

        await delegate_to_managers(job_card_id, run_id, manager_plan)

        # Step 3: Monitor execution
        JOBS[job_card_id]["state"] = "executing"
        heartbeat_monitor.heartbeat(
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.MONITORING,
            message="Monitoring Managers",
            progress=0.5
        )

        await monitor_managers(job_card_id, run_id, manager_plan)

        # Step 4: Validate acceptance
        JOBS[job_card_id]["state"] = "validating"
        heartbeat_monitor.heartbeat(
            agent="Dir-Models",
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
                agent="Dir-Models",
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
                agent="Dir-Models",
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
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.FAILED,
            message=f"Error: {str(e)}",
            progress=1.0
        )

        logger.log_response(
            from_agent="Dir-Models",
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
    Submit tasks to Managers

    For each manager in manager_plan:
    1. Create Manager job card
    2. Submit via RPC (or queue fallback)
    3. Log delegation
    """
    for manager_task in manager_plan["manager_tasks"]:
        manager_id = manager_task["manager_id"]

        # Create Manager job card
        mgr_job_card = JobCard(
            id=f"{job_card_id}-{manager_id.lower()}",
            parent_id=job_card_id,
            role=Role.MANAGER,
            lane=Lane.MODELS,
            task=manager_task["task"],
            inputs=manager_task.get("inputs", []),
            expected_artifacts=manager_task.get("expected_artifacts", []),
            acceptance=manager_task.get("acceptance", []),
            risks=[],
            budget=manager_task.get("budget", {}),
            metadata={
                "files": manager_task.get("files", []),
                "programmers": manager_task.get("programmers", []),
                "requires_review": manager_task.get("requires_review", False)
            },
            submitted_by="Dir-Models"
        )

        # Try RPC first (if Manager is registered)
        try:
            endpoint = MANAGER_ENDPOINTS.get(manager_id)
            if endpoint:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{endpoint}/submit", json={"job_card": mgr_job_card.dict()})
                    response.raise_for_status()

                logger.log_cmd(
                    from_agent="Dir-Models",
                    to_agent=manager_id,
                    message=f"Task submitted: {manager_task['task'][:50]}...",
                    run_id=run_id,
                    metadata={"job_card_id": mgr_job_card.id}
                )
            else:
                # Fallback to queue
                job_queue.submit(mgr_job_card, target_agent=manager_id, use_file_fallback=True)

        except Exception as e:
            # Fallback to file queue
            job_queue.submit(mgr_job_card, target_agent=manager_id, use_file_fallback=True)

            logger.log_cmd(
                from_agent="Dir-Models",
                to_agent=manager_id,
                message=f"Task queued (RPC failed): {manager_task['task'][:50]}...",
                run_id=run_id,
                metadata={"job_card_id": mgr_job_card.id, "rpc_error": str(e)}
            )

        # Update job state
        JOBS[job_card_id]["managers"][manager_id] = {
            "state": "delegated",
            "job_card_id": mgr_job_card.id,
            "task": manager_task["task"]
        }


async def monitor_managers(job_card_id: str, run_id: str, manager_plan: Dict[str, Any]):
    """
    Monitor Managers until all complete or timeout

    Polls Manager status every 10 seconds
    Checks heartbeats for liveness
    Escalates on 2 missed heartbeats
    """
    manager_ids = [task["manager_id"] for task in manager_plan["manager_tasks"]]
    timeout_s = 1800  # 30 minutes max
    start_time = time.time()

    while time.time() - start_time < timeout_s:
        all_complete = True

        for manager_id in manager_ids:
            mgr_state = JOBS[job_card_id]["managers"][manager_id]["state"]

            if mgr_state not in ["completed", "failed"]:
                all_complete = False

                # Check Manager health
                health = heartbeat_monitor.get_health(manager_id)

                if not health.healthy:
                    # Manager unhealthy - escalate
                    JOBS[job_card_id]["managers"][manager_id]["state"] = "failed"
                    JOBS[job_card_id]["managers"][manager_id]["message"] = health.reason

                    logger.log_status(
                        from_agent="Dir-Models",
                        to_agent="Architect",
                        message=f"Manager {manager_id} unhealthy: {health.reason}",
                        run_id=run_id,
                        status="failed"
                    )

        if all_complete:
            break

        # Wait before next poll
        await asyncio.sleep(10)

    # Check for timeout
    if time.time() - start_time >= timeout_s:
        for manager_id in manager_ids:
            if JOBS[job_card_id]["managers"][manager_id]["state"] not in ["completed", "failed"]:
                JOBS[job_card_id]["managers"][manager_id]["state"] = "failed"
                JOBS[job_card_id]["managers"][manager_id]["message"] = "Timeout"


async def validate_acceptance(job_card_id: str, run_id: str, manager_plan: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate acceptance gates for each Manager

    Returns:
        Dict mapping manager_id to pass/fail boolean
    """
    results = {}

    for manager_task in manager_plan["manager_tasks"]:
        manager_id = manager_task["manager_id"]

        # Get Manager report
        # (In full implementation, Managers would report artifacts + results)
        # For now, assume success if Manager completed
        mgr_state = JOBS[job_card_id]["managers"].get(manager_id, {}).get("state")
        results[manager_id] = mgr_state == "completed"

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
        lane="Models",
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
            response = await client.post(
                f"{architect_url}/lane_report",
                json=lane_report.dict()
            )
            response.raise_for_status()

        logger.log_response(
            from_agent="Dir-Models",
            to_agent="Architect",
            message=f"Lane report submitted: {lane_report.state}",
            run_id=run_id,
            status=lane_report.state,
            metadata={"lane": "Models"}
        )

    except Exception as e:
        # Fallback to file-based communication
        lane_report_path = Path(f"artifacts/runs/{run_id}/code/lane_report.json")
        lane_report_path.parent.mkdir(parents=True, exist_ok=True)
        lane_report_path.write_text(json.dumps(lane_report.dict(), indent=2))

        logger.log_response(
            from_agent="Dir-Models",
            to_agent="Architect",
            message=f"Lane report saved (RPC failed): {lane_report.state}",
            run_id=run_id,
            status=lane_report.state,
            metadata={"lane": "Models", "rpc_error": str(e)}
        )


# === Manager Registration Endpoint ===

@app.post("/register_manager")
async def register_manager(manager_id: str, endpoint: str):
    """
    Allow Managers to register their RPC endpoints

    This enables Dir-Models to submit tasks directly via RPC
    instead of using file-based fallback
    """
    MANAGER_ENDPOINTS[manager_id] = endpoint

    logger.log_cmd(
        from_agent=manager_id,
        to_agent="Dir-Models",
        message=f"Manager registered: {endpoint}",
        run_id="system",
        metadata={"manager_id": manager_id, "endpoint": endpoint}
    )

    return {
        "status": "ok",
        "message": f"Manager {manager_id} registered",
        "endpoint": endpoint
    }


# === Agent Chat Endpoint ===

@app.post("/agent_chat/receive")
async def receive_agent_chat_message(
    message: AgentChatMessage,
    background_tasks: BackgroundTasks
):
    """
    Receive agent chat message from Architect

    This endpoint handles Parent-Child communication via conversation threads.
    When Architect delegates a task via agent chat, this endpoint:
    1. Loads full thread history
    2. Processes with LLM (with ask_parent tool available)
    3. Executes task in background
    4. Sends status updates and asks questions as needed

    Different from /submit which uses traditional job cards.

    Flow:
    - Architect creates thread with delegation message
    - Dir-Models receives via this endpoint
    - Dir-Models can ask questions using agent_chat.send_message()
    - Dir-Models sends status updates during execution
    - Dir-Models closes thread on completion/error
    """
    thread_id = message.thread_id

    # Load full thread history for context
    try:
        thread = await agent_chat.get_thread(thread_id)
    except Exception as e:
        logger.log_status(
            from_agent="Dir-Models",
            to_agent="Architect",
            message=f"Error loading thread {thread_id}: {str(e)}",
            run_id=thread_id,
            status="error"
        )
        raise HTTPException(status_code=500, detail=f"Error loading thread: {str(e)}")

    # Log receipt
    logger.log_cmd(
        from_agent="Architect",
        to_agent="Dir-Models",
        message=f"Agent chat message received (thread={thread_id})",
        run_id=thread_id,
        metadata={"thread_id": thread_id, "message_content": message.content[:100]}
    )

    # Process in background
    background_tasks.add_task(process_agent_chat_message, message)

    return {"status": "ok", "thread_id": thread_id}


async def process_agent_chat_message(request: AgentChatMessage):
    """
    Process agent chat message in background

    Steps:
    1. Load thread history
    2. Determine task from thread
    3. Execute task (decompose → delegate → monitor)
    4. Send status updates
    5. Close thread on completion
    """
    thread_id = request.thread_id

    try:
        # Load thread
        thread = await agent_chat.get_thread(thread_id)

        # Extract task from latest message
        task_description = request.content

        # Send acknowledgment
        thread = await agent_chat.get_thread(thread_id)

        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-Models",
            to_agent="Architect",
            content=f"Task received: {task_description[:100]}... Starting decomposition.",
            parent_message_id=request.message_id
        )

        # Create synthetic job card for execution
        job_card_dict = {
            "id": f"agent-chat-{thread_id}",
            "parent_id": thread_id,
            "task": task_description,
            "inputs": [],
            "expected_artifacts": [],
            "acceptance": [],
            "risks": [],
            "budget": {}
        }

        # Execute job card logic
        run_id = thread_id

        # Step 1: Decompose
        heartbeat_monitor.heartbeat(
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing task with LLM",
            progress=0.2
        )

        manager_plan = await decompose_job_card(job_card_dict)

        # Send status update
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-Models",
            to_agent="Architect",
            content=f"Decomposition complete. Identified {len(manager_plan['manager_tasks'])} manager tasks."
        )

        # Step 2: Delegate
        heartbeat_monitor.heartbeat(
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.DELEGATING,
            message="Delegating to Managers",
            progress=0.4
        )

        # Register temporary job for tracking
        JOBS[job_card_dict["id"]] = {
            "state": "delegating",
            "started_at": time.time(),
            "managers": {},
            "job_card": job_card_dict
        }

        await delegate_to_managers(job_card_dict["id"], run_id, manager_plan)

        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-Models",
            to_agent="Architect",
            content="Tasks delegated to Managers. Monitoring execution..."
        )

        # Step 3: Monitor
        heartbeat_monitor.heartbeat(
            agent="Dir-Models",
            run_id=run_id,
            state=AgentState.MONITORING,
            message="Monitoring Managers",
            progress=0.6
        )

        await monitor_managers(job_card_dict["id"], run_id, manager_plan)

        # Step 4: Validate
        validation_results = await validate_acceptance(job_card_dict["id"], run_id, manager_plan)

        # Step 5: Complete
        if all(validation_results.values()):
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Dir-Models",
                to_agent="Architect",
                content=f"✅ Task completed successfully. All {len(manager_plan['manager_tasks'])} manager tasks passed validation."
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                closed_by="Dir-Models",
                reason="Task completed successfully"
            )

            heartbeat_monitor.heartbeat(
                agent="Dir-Models",
                run_id=run_id,
                state=AgentState.COMPLETED,
                message="Task completed",
                progress=1.0
            )

        else:
            # Some managers failed
            failed_managers = [mgr for mgr, passed in validation_results.items() if not passed]

            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Dir-Models",
                to_agent="Architect",
                content=f"❌ Task failed. Managers that failed: {', '.join(failed_managers)}"
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                closed_by="Dir-Models",
                reason=f"Task failed: {', '.join(failed_managers)}"
            )

            heartbeat_monitor.heartbeat(
                agent="Dir-Models",
                run_id=run_id,
                state=AgentState.FAILED,
                message=f"Failed: {', '.join(failed_managers)}",
                progress=1.0
            )

    except Exception as e:
        # Error during execution
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-Models",
            to_agent="Architect",
            content=f"❌ Error during execution: {str(e)}"
        )

        await agent_chat.close_thread(
            thread_id=thread_id,
            closed_by="Dir-Models",
            reason=f"Error: {str(e)}"
        )

        logger.log_status(
            from_agent="Dir-Models",
            to_agent="Architect",
            message=f"Error processing agent chat message: {str(e)}",
            run_id=thread_id,
            status="error"
        )


# === Startup ===

if __name__ == "__main__":
    import uvicorn
    import asyncio

    # Start heartbeat emitter
    async def emit_heartbeats():
        while True:
            heartbeat_monitor.heartbeat(
                agent="Dir-Models",
                state=AgentState.IDLE,
                message="Idle, waiting for job cards"
            )
            await asyncio.sleep(60)

    # Run server
    uvicorn.run(app, host="127.0.0.1", port=6111)

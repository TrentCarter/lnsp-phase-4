#!/usr/bin/env python3
"""
Director-Code Service - DevSecOps Lane Coordinator

Port: 6114
LLM: Gemini 2.5 Flash (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive job cards from Architect
- Decompose into Manager-level tasks
- Monitor Managers
- Validate acceptance gates (security scans, SBOM, CI/CD gates)
- Report to Architect

Contract: docs/contracts/DIRECTOR_DEVSECOPS_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
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
from services.pas.director_devsecops.decomposer import ManagerTaskDecomposer

# Agent chat for Parent-Child communication
from services.common.agent_chat import get_agent_chat_client, AgentChatMessage

# LLM with tool support (Phase 3)
from services.common.llm_tool_caller import call_llm_with_tools, LLMResponse
from services.common.llm_tools import get_ask_parent_tool, validate_ask_parent_args, get_system_prompt_with_ask_parent


app = FastAPI(title="Director-DevSecOps", version="1.0.0")

# Add CORS middleware to allow browser requests from HMI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Initialize systems
heartbeat_monitor = get_monitor()
job_queue = get_queue()
logger = get_logger()
agent_chat = get_agent_chat_client()

# Register Director-Code agent
heartbeat_monitor.register_agent(
    agent="Dir-DevSecOps",
    parent="Architect",
    llm_model=os.getenv("DIR_DEVSECOPS_LLM", "google/gemini-2.5-flash"),
    role="director",
    tier="coordinator"
)

# Manager task decomposer (LLM-powered)
decomposer = ManagerTaskDecomposer()

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
    lane: str = "DevSecOps"
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
        "service": "Director-DevSecOps",
        "version": "1.0.0",
        "port": 6114,
        "agent": "Dir-DevSecOps",
        "llm_model": os.getenv("DIR_DEVSECOPS_LLM", "google/gemini-2.5-flash"),
        "agent_metadata": {
            "role": "director",
            "tier": "coordinator",
            "parent": "Architect",
            "children": list(MANAGER_ENDPOINTS.keys())
        }
    }


@app.post("/chat/stream")
async def chat_stream(request: dict):
    """Streaming chat endpoint for LLM Chat interface."""
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            messages = request.get("messages", [])
            model = request.get("model", os.getenv("DIR_DEVSECOPS_LLM", "google/gemini-2.5-flash"))
            if not messages:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No messages provided'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return
            health_data = await health()
            agent_name = health_data.get("agent", "Dir-DevSecOps")
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': f'{agent_name} processing...'})}\n\n"
            system_prompt = f"""You are {agent_name}, a DevSecOps Director in the PAS (Project Automation System).

Your role:
- Coordinate CI/CD, security, and infrastructure tasks
- Guide deployment, monitoring, and security practices
- Ensure compliance and operational excellence
- Manage infrastructure as code and security gates

You have access to the filesystem via Aider for DevSecOps operations."""
            full_messages = [{"role": "system", "content": system_prompt}] + messages
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"
            gateway_url = "http://localhost:6120/chat/stream"
            payload = {"session_id": str(uuid.uuid4()), "message_id": str(uuid.uuid4()), "agent_id": "direct", "model": model, "content": messages[-1]["content"] if messages else "", "messages": [{"role": msg["role"], "content": msg["content"]} for msg in full_messages]}
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", gateway_url, json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                            try:
                                event_data = json.loads(line[6:])
                                if event_data.get('type') == 'done':
                                    break
                            except json.JSONDecodeError:
                                pass
        except Exception as e:
            print(f"[Dir-DevSecOps] Chat stream error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")


class AgentChatRequest(BaseModel):
    """Agent chat message for testing and inter-agent communication"""
    sender_agent: str
    message_type: str  # question, delegation, status, etc.
    content: str
    metadata: Optional[Dict[str, Any]] = None


@app.post("/agent/chat/send")
async def agent_chat_send(request: AgentChatRequest):
    """
    Receive agent chat message (for testing and inter-agent communication).

    This endpoint allows external systems (HMI tests, other agents) to send
    messages to this agent for testing connectivity and basic interactions.
    """
    # Get agent name from health endpoint metadata
    health_data = await health()
    agent_name = health_data.get("agent", "Unknown")

    # Log the received message
    print(f"[{agent_name}] Received {request.message_type} from {request.sender_agent}: {request.content}")

    # Return acknowledgment
    return {
        "status": "received",
        "agent": agent_name,
        "message_id": str(uuid.uuid4()),
        "sender": request.sender_agent,
        "message_type": request.message_type,
        "response": f"{agent_name} received: {request.content}"
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
        to_agent="Dir-DevSecOps",
        message=f"Job card received: {job_card_dict.get('task', '')[:50]}...",
        run_id=job_card_dict.get("parent_id", "unknown"),
        metadata={"job_card_id": job_card_id}
    )

    # Start background execution
    background_tasks.add_task(execute_job_card, job_card_dict)

    # Send heartbeat
    heartbeat_monitor.heartbeat(
        agent="Dir-DevSecOps",
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
            agent="Dir-DevSecOps",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing job card with LLM",
            progress=0.2
        )

        manager_plan = await decompose_job_card(job_card_dict)

        # Step 2: Submit tasks to Managers
        JOBS[job_card_id]["state"] = "delegating"
        heartbeat_monitor.heartbeat(
            agent="Dir-DevSecOps",
            run_id=run_id,
            state=AgentState.DELEGATING,
            message="Submitting tasks to Managers",
            progress=0.4
        )

        await delegate_to_managers(job_card_id, run_id, manager_plan)

        # Step 3: Monitor execution
        JOBS[job_card_id]["state"] = "executing"
        heartbeat_monitor.heartbeat(
            agent="Dir-DevSecOps",
            run_id=run_id,
            state=AgentState.MONITORING,
            message="Monitoring Managers",
            progress=0.5
        )

        await monitor_managers(job_card_id, run_id, manager_plan)

        # Step 4: Validate acceptance
        JOBS[job_card_id]["state"] = "validating"
        heartbeat_monitor.heartbeat(
            agent="Dir-DevSecOps",
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
                agent="Dir-DevSecOps",
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
                agent="Dir-DevSecOps",
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
            agent="Dir-DevSecOps",
            run_id=run_id,
            state=AgentState.FAILED,
            message=f"Error: {str(e)}",
            progress=1.0
        )

        logger.log_response(
            from_agent="Dir-DevSecOps",
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
            lane=Lane.DEVSECOPS,
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
            submitted_by="Dir-DevSecOps"
        )

        # Try RPC first (if Manager is registered)
        try:
            endpoint = MANAGER_ENDPOINTS.get(manager_id)
            if endpoint:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{endpoint}/submit", json={"job_card": mgr_job_card.dict()})
                    response.raise_for_status()

                logger.log_cmd(
                    from_agent="Dir-DevSecOps",
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
                from_agent="Dir-DevSecOps",
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
                        from_agent="Dir-DevSecOps",
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
        lane="DevSecOps",
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
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message=f"Lane report submitted: {lane_report.state}",
            run_id=run_id,
            status=lane_report.state,
            metadata={"lane": "DevSecOps"}
        )

    except Exception as e:
        # Fallback to file-based communication
        lane_report_path = Path(f"artifacts/runs/{run_id}/code/lane_report.json")
        lane_report_path.parent.mkdir(parents=True, exist_ok=True)
        lane_report_path.write_text(json.dumps(lane_report.dict(), indent=2))

        logger.log_response(
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message=f"Lane report saved (RPC failed): {lane_report.state}",
            run_id=run_id,
            status=lane_report.state,
            metadata={"lane": "DevSecOps", "rpc_error": str(e)}
        )


# === Agent Chat Endpoint (Parent ↔ Child Communication) ===

@app.post("/agent_chat/receive")
async def receive_agent_message(
    request: AgentChatMessage,
    background_tasks: BackgroundTasks
):
    """
    Receive message from Architect via Agent Chat thread.

    This is the RECOMMENDED way for Architect to communicate with Dir-DevSecOps.
    Enables bidirectional Q&A, status updates, and context preservation.

    Flow:
    - Architect creates thread with delegation message
    - Dir-DevSecOps receives via this endpoint
    - Dir-DevSecOps can ask questions using agent_chat.send_message()
    - Dir-DevSecOps sends status updates during execution
    - Dir-DevSecOps closes thread on completion/error

    Alternative: /submit endpoint (job card only, no conversation)
    """
    thread_id = request.thread_id

    # Load thread to get run_id
    try:
        thread = await agent_chat.get_thread(thread_id)
        run_id = thread.run_id
    except Exception:
        run_id = "unknown"

    logger.log_cmd(
        from_agent="Architect",
        to_agent="Dir-DevSecOps",
        message=f"Agent chat message received: {request.message_type}",
        run_id=run_id,
        metadata={
            "thread_id": thread_id,
            "message_type": request.message_type,
            "from_agent": request.from_agent
        }
    )

    # Delegate to background task for LLM processing
    background_tasks.add_task(process_agent_chat_message, request)

    return {
        "status": "ok",
        "thread_id": thread_id,
        "message": "Agent chat message received, processing with LLM"
    }


async def process_agent_chat_message(request: AgentChatMessage):
    """
    Process agent chat message with LLM (background task).

    Uses LLM with ask_parent tool to enable bidirectional communication.
    """
    thread_id = request.thread_id

    # Load thread to get run_id
    try:
        thread = await agent_chat.get_thread(thread_id)
        run_id = thread.run_id
    except Exception:
        run_id = "unknown"

    try:
        # Load full thread history for context
        thread = await agent_chat.get_thread(thread_id)

        # Send initial status
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message_type="status",
            content="Received task, analyzing with LLM..."
        )

        # Build LLM context from thread history
        llm_messages = []
        for msg in thread.messages:
            role = "user" if msg.from_agent == "Architect" else "assistant"
            llm_messages.append({
                "role": role,
                "content": msg.content
            })

        # Add system prompt with ask_parent tool
        system_prompt = get_system_prompt_with_ask_parent(
            agent_name="Dir-DevSecOps",
            parent_name="Architect",
            role_description="DevSecOps Lane Director responsible for security scans, SBOM, and CI/CD gates"
        )

        # Call LLM with tools
        llm_response = await call_llm_with_tools(
            messages=llm_messages,
            system_prompt=system_prompt,
            tools=[get_ask_parent_tool()],
            model=os.getenv("DIR_DEVSECOPS_LLM", "google/gemini-2.5-flash")
        )

        # Handle tool calls (ask_parent)
        if llm_response.tool_calls:
            for tool_call in llm_response.tool_calls:
                if tool_call["name"] == "ask_parent":
                    args = tool_call["args"]
                    validation_error = validate_ask_parent_args(args)

                    if validation_error:
                        # Invalid args - log error
                        logger.log_status(
                            from_agent="Dir-DevSecOps",
                            to_agent="Architect",
                            message=f"LLM generated invalid ask_parent call: {validation_error}",
                            run_id=run_id,
                            status="failed"
                        )
                        continue

                    # Send question to Architect
                    await agent_chat.send_message(
                        thread_id=thread_id,
                        from_agent="Dir-DevSecOps",
                        to_agent="Architect",
                        message_type="question",
                        content=args["question"],
                        metadata={"urgency": args.get("urgency", "normal")}
                    )

                    logger.log_cmd(
                        from_agent="Dir-DevSecOps",
                        to_agent="Architect",
                        message=f"Question: {args['question'][:100]}...",
                        run_id=run_id,
                        metadata={"thread_id": thread_id, "urgency": args.get("urgency")}
                    )

                    # Wait for answer (Architect will call this endpoint again with answer)
                    # Don't proceed with job card creation yet
                    return

        # No questions asked - proceed with decomposition
        # Extract task from initial delegation message
        task_content = thread.messages[0].content

        # Create job card from LLM analysis
        job_card_id = f"job-{run_id}-devsecops-{uuid.uuid4().hex[:8]}"

        # TODO: Use LLM to extract entry_files, acceptance criteria, etc.
        # For now, use defaults
        entry_files = llm_response.data.get("entry_files", [])

        # Step 4: Send progress update
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message_type="status",
            content=f"Starting task execution (Job ID: {job_card_id})"
        )

        # Create job card
        job_card_dict = {
            "id": job_card_id,
            "parent_id": run_id,
            "role": "director",
            "lane": "devsecops",
            "task": task_content,
            "inputs": [],
            "expected_artifacts": [],
            "acceptance": [],
            "risks": [],
            "budget": {},
            "metadata": {"entry_files": entry_files, "thread_id": thread_id},
            "submitted_by": "Architect"
        }

        # Register job
        JOBS[job_card_id] = {
            "state": "planning",
            "message": "Decomposing into Manager tasks",
            "started_at": time.time(),
            "managers": {},
            "job_card": job_card_dict,
            "thread_id": thread_id
        }

        # Execute job card (same as /submit flow)
        await execute_job_card_with_chat(job_card_dict, thread_id)

    except Exception as e:
        # Send error message to Architect
        try:
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Dir-DevSecOps",
                to_agent="Architect",
                message_type="error",
                content=f"Error processing task: {str(e)}"
            )
        except:
            pass

        logger.log_status(
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message=f"Error processing agent chat message: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"thread_id": thread_id, "error": str(e)}
        )


async def execute_job_card_with_chat(job_card_dict: Dict[str, Any], thread_id: str):
    """
    Execute job card with agent chat status updates.

    Same as execute_job_card() but sends status updates via agent chat.
    """
    job_card_id = job_card_dict["id"]
    run_id = job_card_dict.get("parent_id", "unknown")

    try:
        # Step 1: Decompose job card
        heartbeat_monitor.heartbeat(
            agent="Dir-DevSecOps",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing job card with LLM",
            progress=0.2
        )

        manager_plan = await decompose_job_card(job_card_dict)

        # Send status update via agent chat
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message_type="status",
            content=f"Decomposed into {len(manager_plan['manager_tasks'])} Manager tasks"
        )

        # Step 2: Submit tasks to Managers
        JOBS[job_card_id]["state"] = "delegating"
        await delegate_to_managers(job_card_id, run_id, manager_plan)

        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent="Dir-DevSecOps",
            to_agent="Architect",
            message_type="status",
            content="Delegated tasks to Managers, monitoring execution..."
        )

        # Step 3: Monitor execution
        JOBS[job_card_id]["state"] = "executing"
        await monitor_managers(job_card_id, run_id, manager_plan)

        # Step 4: Validate acceptance
        JOBS[job_card_id]["state"] = "validating"
        validation_results = await validate_acceptance(job_card_id, run_id, manager_plan)

        # Step 5: Complete
        if all(validation_results.values()):
            JOBS[job_card_id]["state"] = "completed"
            JOBS[job_card_id]["completed_at"] = time.time()

            # Send completion message
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Dir-DevSecOps",
                to_agent="Architect",
                message_type="completion",
                content=f"✅ DevSecOps lane tasks completed successfully. All Manager tasks passed validation."
            )

            # Close thread
            await agent_chat.close_thread(
                thread_id=thread_id,
                status="completed",
                result="All Manager tasks completed successfully"
            )

        else:
            # Some managers failed
            failed_managers = [mgr for mgr, passed in validation_results.items() if not passed]
            JOBS[job_card_id]["state"] = "failed"

            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Dir-DevSecOps",
                to_agent="Architect",
                message_type="error",
                content=f"❌ DevSecOps lane tasks failed. Failed managers: {', '.join(failed_managers)}"
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=f"Managers failed: {', '.join(failed_managers)}"
            )

    except Exception as e:
        JOBS[job_card_id]["state"] = "failed"

        try:
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Dir-DevSecOps",
                to_agent="Architect",
                message_type="error",
                content=f"Error executing job card: {str(e)}"
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=str(e)
            )
        except:
            pass


# === Manager Registration Endpoint ===

@app.post("/register_manager")
async def register_manager(manager_id: str, endpoint: str):
    """
    Allow Managers to register their RPC endpoints

    This enables Dir-DevSecOps to submit tasks directly via RPC
    instead of using file-based fallback
    """
    MANAGER_ENDPOINTS[manager_id] = endpoint

    logger.log_cmd(
        from_agent=manager_id,
        to_agent="Dir-DevSecOps",
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
                agent="Dir-DevSecOps",
                state=AgentState.IDLE,
                message="Idle, waiting for job cards"
            )
            await asyncio.sleep(60)

    # Run server
    uvicorn.run(app, host="127.0.0.1", port=6111)

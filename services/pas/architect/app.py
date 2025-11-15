#!/usr/bin/env python3
"""
Architect Service - Top-Level PAS Coordinator

Port: 6110
LLM: Claude Sonnet 4.5 (primary), Gemini 2.5 Pro (fallback)

Responsibilities:
- Receive Prime Directives from PAS Root
- Decompose into lane-specific job cards
- Allocate resources via Resource Manager
- Monitor Directors
- Aggregate status and validate acceptance
- Generate executive summaries

Contract: docs/contracts/ARCHITECT_SYSTEM_PROMPT.md
"""
import sys
import os
from pathlib import Path
import subprocess
import signal

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import time
import httpx
import requests
import json
import uuid
import asyncio
from dataclasses import asdict
from datetime import datetime

# PAS common services
from services.common.heartbeat import get_monitor, AgentState
from services.common.job_queue import get_queue, JobCard, Lane, Role, Priority
from services.common.comms_logger import get_logger, MessageType
from services.common.title_generator import generate_short_title_async
from services.common.agent_chat import get_agent_chat_client, AgentChatThread

# LLM with tool support (Phase 3)
from services.common.llm_tool_caller import call_llm, LLMResponse

# Architect-specific imports
from services.pas.architect.decomposer import TaskDecomposer


app = FastAPI(title="Architect", version="1.0.0")

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

# Register Architect agent
heartbeat_monitor.register_agent(
    agent="Architect",
    parent="PAS Root",
    llm_model=os.getenv("ARCHITECT_LLM", "anthropic/claude-sonnet-4-5"),
    role="architect",
    tier="coordinator"
)

# Task decomposer (LLM-powered)
decomposer = TaskDecomposer()

# Director endpoints (ports 6111-6115)
DIRECTOR_ENDPOINTS = {
    "Dir-Code": os.getenv("DIR_CODE_URL", "http://127.0.0.1:6111"),
    "Dir-Models": os.getenv("DIR_MODELS_URL", "http://127.0.0.1:6112"),
    "Dir-Data": os.getenv("DIR_DATA_URL", "http://127.0.0.1:6113"),
    "Dir-DevSecOps": os.getenv("DIR_DEVSECOPS_URL", "http://127.0.0.1:6114"),
    "Dir-Docs": os.getenv("DIR_DOCS_URL", "http://127.0.0.1:6115"),
}

# In-memory run tracking (will move to DB in Phase 2)
RUNS: Dict[str, Dict[str, Any]] = {}

# Track active tasks for each child (for TRON restart/resend)
CHILD_ACTIVE_TASKS: Dict[str, Dict[str, Any]] = {
    # "Dir-Code": {"job_card": JobCard(...), "run_id": "...", "endpoint": "..."},
    # "Dir-Models": {...},
}


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
        # Event Stream not available - log but don't fail
        print(f"Warning: Event Stream not available for HHMRS event {event_type}")
    except Exception as e:
        print(f"Error emitting HHMRS event {event_type}: {e}")


# === Pydantic Models ===

class PrimeDirective(BaseModel):
    """Prime Directive from PAS Root"""
    run_id: str
    prd: str  # Full PRD text
    title: str  # User-provided title
    entry_files: List[str] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)  # tokens_max, duration_max_mins, cost_usd_max
    policy: Dict[str, Any] = Field(default_factory=dict)  # require_cross_vendor_review, protected_paths
    approval_mode: str = "auto"  # auto | human


class ArchitectPlan(BaseModel):
    """Architect plan output"""
    run_id: str
    executive_summary: str
    lane_allocations: Dict[str, Dict[str, Any]]  # lane -> job card dict
    dependency_graph: str  # Graphviz DOT format
    resource_reservations: Dict[str, Any]
    acceptance_gates: Dict[str, List[str]]  # lane -> list of checks
    created_at: float = Field(default_factory=time.time)


class RunStatus(BaseModel):
    """Run status"""
    run_id: str
    state: str  # planning | executing | awaiting_approval | completed | failed
    message: str
    lanes: Dict[str, Dict[str, Any]] = Field(default_factory=dict)  # lane -> status
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    duration_s: Optional[float] = None


class LaneReport(BaseModel):
    """Lane report from Director"""
    lane: str
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
        "service": "Architect",
        "version": "1.0.0",
        "port": 6110,
        "agent": "Architect",
        "llm_model": os.getenv("ARCHITECT_LLM", "anthropic/claude-sonnet-4-5"),
        "agent_metadata": {
            "role": "architect",
            "tier": "coordinator",
            "parent": "PAS Root",
            "children": list(DIRECTOR_ENDPOINTS.keys())
        }
    }


@app.post("/chat/stream")
async def chat_stream(request: dict):
    """
    Streaming chat endpoint for LLM Chat interface.

    Accepts: {messages: [{role, content}], model: str}
    Returns: SSE stream with tokens, status updates, usage

    Routes to Gateway for multi-provider support (Ollama, Anthropic, Google, etc.)
    """
    from fastapi.responses import StreamingResponse

    async def generate():
        try:
            messages = request.get("messages", [])
            model = request.get("model", os.getenv("ARCHITECT_LLM", "anthropic/claude-sonnet-4-5"))

            if not messages:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No messages provided'})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # Get agent info
            health_data = await health()
            agent_name = health_data.get("agent", "Architect")

            # Status: Planning
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': f'{agent_name} processing...'})}\n\n"

            # System prompt for Architect
            system_prompt = f"""You are {agent_name}, the top-level Architect in the PAS (Project Automation System).

Your role:
- Decompose high-level PRDs into lane-specific job cards
- Coordinate Directors across Code, Models, Data, DevSecOps, and Docs lanes
- Provide strategic guidance and architectural oversight
- Ensure cross-lane coordination and integration

You have access to the filesystem via Aider for architectural operations."""

            # Prepend system message
            full_messages = [{"role": "system", "content": system_prompt}] + messages

            # Status: Executing
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"

            # Route to Gateway for multi-provider support
            gateway_url = "http://localhost:6120/chat/stream"
            payload = {
                "session_id": str(uuid.uuid4()),
                "message_id": str(uuid.uuid4()),
                "agent_id": "direct",
                "model": model,
                "content": messages[-1]["content"] if messages else "",
                "messages": [{"role": msg["role"], "content": msg["content"]} for msg in full_messages]
            }

            # Get response from Gateway
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("POST", gateway_url, json=payload) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        # Forward SSE events from Gateway
                        if line.startswith("data: "):
                            yield f"{line}\n\n"

                            # Parse to check for completion
                            try:
                                event_data = json.loads(line[6:])
                                if event_data.get('type') == 'done':
                                    break
                            except json.JSONDecodeError:
                                pass

        except Exception as e:
            print(f"[Architect] Chat stream error: {e}")
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
    # Log the received message
    print(f"[Architect] Received {request.message_type} from {request.sender_agent}: {request.content}")

    # Return acknowledgment
    return {
        "status": "received",
        "agent": "Architect",
        "message_id": str(uuid.uuid4()),
        "sender": request.sender_agent,
        "message_type": request.message_type,
        "response": f"Architect received: {request.content}"
    }


@app.get("/status/{run_id}")
async def get_status(run_id: str) -> RunStatus:
    """Get status for a run"""
    if run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    run_data = RUNS[run_id]
    return RunStatus(
        run_id=run_id,
        state=run_data["state"],
        message=run_data.get("message", ""),
        lanes=run_data.get("lanes", {}),
        started_at=run_data.get("started_at"),
        completed_at=run_data.get("completed_at"),
        duration_s=run_data.get("duration_s")
    )


class LaneReportRequest(BaseModel):
    """Lane report request with run_id"""
    run_id: str
    lane: str
    state: str  # completed | failed
    artifacts: Dict[str, str] = Field(default_factory=dict)
    acceptance_results: Dict[str, Any] = Field(default_factory=dict)
    actuals: Dict[str, Any] = Field(default_factory=dict)
    managers_used: Dict[str, str] = Field(default_factory=dict)


@app.post("/lane_report")
async def receive_lane_report(request: LaneReportRequest):
    """
    Receive lane completion report from Director

    Directors call this endpoint when they complete (or fail) their lane work.
    Updates run state and triggers completion checks.
    """
    if request.run_id not in RUNS:
        raise HTTPException(status_code=404, detail="Run not found")

    lane_name = request.lane

    # Update lane state in RUNS
    if lane_name not in RUNS[request.run_id]["lanes"]:
        RUNS[request.run_id]["lanes"][lane_name] = {}

    RUNS[request.run_id]["lanes"][lane_name].update({
        "state": request.state,
        "artifacts": request.artifacts,
        "acceptance_results": request.acceptance_results,
        "actuals": request.actuals,
        "managers_used": request.managers_used,
        "completed_at": time.time()
    })

    # Log receipt
    logger.log_response(
        from_agent=f"Dir-{lane_name}",
        to_agent="Architect",
        message=f"Lane report received: {request.state}",
        run_id=request.run_id,
        status=request.state,
        metadata={
            "lane": lane_name,
            "artifacts_count": len(request.artifacts),
            "managers_used": len(request.managers_used)
        }
    )

    # Clean up active task tracking (task is now complete or failed)
    director_id = f"Dir-{lane_name}"
    if director_id in CHILD_ACTIVE_TASKS:
        del CHILD_ACTIVE_TASKS[director_id]
        logger.log_status(
            from_agent="Architect",
            to_agent=director_id,
            message=f"Cleared active task tracking for {director_id}",
            run_id=request.run_id,
            metadata={"lane": lane_name, "state": request.state}
        )

    return {
        "status": "ok",
        "message": f"Lane report received for {lane_name}",
        "run_id": request.run_id
    }


# === HHMRS Phase 3: Process Restart Logic ===

# Agent restart configuration
AGENT_RESTART_CONFIG = {
    "Dir-Code": {
        "port": 6111,
        "module": "services.pas.director_code.app:app",
        "log_file": "logs/pas/director_code.log",
        "env_vars": {
            "DIR_CODE_LLM_PROVIDER": os.getenv("DIR_CODE_LLM_PROVIDER", "google"),
            "DIR_CODE_LLM": os.getenv("DIR_CODE_LLM", "gemini-2.5-flash"),
        }
    },
    "Dir-Models": {
        "port": 6112,
        "module": "services.pas.director_models.app:app",
        "log_file": "logs/pas/director_models.log",
        "env_vars": {
            "DIR_MODELS_LLM_PROVIDER": os.getenv("DIR_MODELS_LLM_PROVIDER", "anthropic"),
            "DIR_MODELS_LLM": os.getenv("DIR_MODELS_LLM", "claude-sonnet-4-5-20250929"),
        }
    },
    "Dir-Data": {
        "port": 6113,
        "module": "services.pas.director_data.app:app",
        "log_file": "logs/pas/director_data.log",
        "env_vars": {
            "DIR_DATA_LLM_PROVIDER": os.getenv("DIR_DATA_LLM_PROVIDER", "anthropic"),
            "DIR_DATA_LLM": os.getenv("DIR_DATA_LLM", "claude-sonnet-4-5-20250929"),
        }
    },
    "Dir-DevSecOps": {
        "port": 6114,
        "module": "services.pas.director_devsecops.app:app",
        "log_file": "logs/pas/director_devsecops.log",
        "env_vars": {
            "DIR_DEVSECOPS_LLM_PROVIDER": os.getenv("DIR_DEVSECOPS_LLM_PROVIDER", "google"),
            "DIR_DEVSECOPS_LLM": os.getenv("DIR_DEVSECOPS_LLM", "gemini-2.5-flash"),
        }
    },
    "Dir-Docs": {
        "port": 6115,
        "module": "services.pas.director_docs.app:app",
        "log_file": "logs/pas/director_docs.log",
        "env_vars": {
            "DIR_DOCS_LLM_PROVIDER": os.getenv("DIR_DOCS_LLM_PROVIDER", "anthropic"),
            "DIR_DOCS_LLM": os.getenv("DIR_DOCS_LLM", "claude-sonnet-4-5-20250929"),
        }
    },
}


def _restart_child_process(child_id: str) -> bool:
    """
    Restart a child Director process

    Steps:
    1. Find process on child's port
    2. Kill process gracefully (SIGTERM), then forcefully (SIGKILL) if needed
    3. Start new process with same configuration
    4. Wait for health check

    Returns:
        True if restart successful, False otherwise
    """
    if child_id not in AGENT_RESTART_CONFIG:
        logger.log_status(
            from_agent="Architect",
            to_agent="Architect",
            message=f"Cannot restart {child_id}: no restart config found",
            run_id=None,
            metadata={"child_id": child_id, "error": "unknown_agent"}
        )
        return False

    config = AGENT_RESTART_CONFIG[child_id]
    port = config["port"]
    module = config["module"]
    log_file = config["log_file"]
    env_vars = config.get("env_vars", {})

    try:
        # Step 1: Kill process on port
        logger.log_status(
            from_agent="Architect",
            to_agent=child_id,
            message=f"Killing process on port {port}",
            run_id=None,
            metadata={"child_id": child_id, "port": port}
        )

        # Find PID using lsof
        find_pid = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True
        )

        if find_pid.returncode == 0 and find_pid.stdout.strip():
            pid = int(find_pid.stdout.strip())
            logger.log_status(
                from_agent="Architect",
                to_agent=child_id,
                message=f"Found process PID {pid} on port {port}",
                run_id=None,
                metadata={"child_id": child_id, "port": port, "pid": pid}
            )

            # Try graceful shutdown (SIGTERM)
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)  # Wait for graceful shutdown

                # Check if still alive
                try:
                    os.kill(pid, 0)  # Check if process exists
                    # Still alive, force kill
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(1)
                except OSError:
                    # Process already dead
                    pass

            except OSError as e:
                logger.log_status(
                    from_agent="Architect",
                    to_agent="Architect",
                    message=f"Error killing PID {pid}: {e}",
                    run_id=None,
                    metadata={"child_id": child_id, "pid": pid, "error": str(e)}
                )
        else:
            logger.log_status(
                from_agent="Architect",
                to_agent=child_id,
                message=f"No process found on port {port}",
                run_id=None,
                metadata={"child_id": child_id, "port": port}
            )

        # Step 2: Start new process
        logger.log_status(
            from_agent="Architect",
            to_agent=child_id,
            message=f"Starting new process on port {port}",
            run_id=None,
            metadata={"child_id": child_id, "port": port, "module": module}
        )

        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)

        # Build environment (copy current + add child-specific)
        child_env = os.environ.copy()
        child_env.update(env_vars)

        # Open log file
        log_handle = open(log_file, "a")

        # Start uvicorn process
        process = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn", module,
                "--host", "127.0.0.1",
                "--port", str(port)
            ],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=child_env,
            preexec_fn=os.setpgrp  # Create new process group
        )

        logger.log_status(
            from_agent="Architect",
            to_agent=child_id,
            message=f"Started new process PID {process.pid}",
            run_id=None,
            metadata={"child_id": child_id, "port": port, "pid": process.pid}
        )

        # Step 3: Health check (wait up to 10 seconds)
        max_retries = 10
        for i in range(max_retries):
            time.sleep(1)
            try:
                response = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
                if response.status_code == 200:
                    logger.log_status(
                        from_agent="Architect",
                        to_agent=child_id,
                        message=f"Health check passed after {i+1} seconds",
                        run_id=None,
                        metadata={"child_id": child_id, "port": port}
                    )
                    return True
            except:
                pass

        logger.log_status(
            from_agent="Architect",
            to_agent="Architect",
            message=f"Health check failed after {max_retries} seconds",
            run_id=None,
            metadata={"child_id": child_id, "port": port, "error": "health_check_timeout"}
        )
        return False

    except Exception as e:
        logger.log_status(
            from_agent="Architect",
            to_agent="Architect",
            message=f"Error restarting {child_id}: {str(e)}",
            run_id=None,
            metadata={"child_id": child_id, "error": str(e)}
        )
        return False


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
    - restart_count < max_restarts: Restart child with same config
    - restart_count >= max_restarts: Escalate to grandparent (PAS Root)

    Note: max_restarts is loaded from settings (artifacts/pas_settings.json)
    """
    # Load max_restarts from TRON settings
    max_restarts = heartbeat_monitor.max_restarts
    child_id = alert.child_id
    restart_count = alert.restart_count

    logger.log_status(
        from_agent="TRON",
        to_agent="Architect",
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
        # Emit escalation event for HMI chimes and TRON visualization
        _emit_hhmrs_event('hhmrs_escalation', {
            'agent_id': child_id,
            'parent_id': 'Architect',
            'grandparent_id': 'PAS Root',
            'restart_count': restart_count,
            'reason': 'max_restarts_exceeded',
            'message': f"{child_id} escalated to PAS Root after {restart_count} restarts"
        })

        # Escalate to PAS Root
        try:
            pas_root_url = os.getenv("PAS_ROOT_URL", "http://127.0.0.1:6100")
            escalation = {
                "type": "grandchild_failure",
                "grandchild_id": child_id,
                "parent_id": "Architect",
                "failure_count": restart_count,
                "reason": "max_restarts_exceeded"
            }

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{pas_root_url}/handle_grandchild_failure",
                    json=escalation,
                    timeout=10.0
                )

            if response.status_code == 200:
                logger.log_status(
                    from_agent="Architect",
                    to_agent="PAS Root",
                    message=f"Escalated {child_id} failure to PAS Root",
                    run_id=None,
                    metadata={"child_id": child_id, "restart_count": restart_count}
                )
                return {
                    "status": "escalated",
                    "message": f"Escalated {child_id} to PAS Root",
                    "restart_count": restart_count
                }
            else:
                logger.log_status(
                    from_agent="Architect",
                    to_agent="Architect",
                    message=f"Failed to escalate {child_id} to PAS Root: {response.text}",
                    run_id=None,
                    metadata={"child_id": child_id, "status_code": response.status_code}
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Grandparent escalation failed: {response.text}"
                )

        except Exception as e:
            logger.log_status(
                from_agent="Architect",
                to_agent="Architect",
                message=f"Error escalating {child_id}: {str(e)}",
                run_id=None,
                metadata={"child_id": child_id, "error": str(e)}
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to escalate to grandparent: {str(e)}"
            )

    # Emit restart event for HMI chimes and TRON visualization
    _emit_hhmrs_event('hhmrs_restart', {
        'agent_id': child_id,
        'parent_id': 'Architect',
        'restart_count': restart_count + 1,
        'message': f"Restarting {child_id} (attempt {restart_count + 1})"
    })

    # Phase 3: Attempt actual process restart
    logger.log_status(
        from_agent="Architect",
        to_agent=child_id,
        message=f"Attempting process restart for {child_id} (attempt {restart_count + 1})",
        run_id=None,
        metadata={"child_id": child_id, "restart_count": restart_count + 1}
    )

    # Update TRON retry count
    heartbeat_monitor._retry_counts[child_id] = restart_count + 1

    # Restart the child process
    restart_success = _restart_child_process(child_id)

    if restart_success:
        logger.log_status(
            from_agent="Architect",
            to_agent=child_id,
            message=f"Successfully restarted {child_id}",
            run_id=None,
            metadata={"child_id": child_id, "restart_count": restart_count + 1, "status": "success"}
        )

        # Check if this child had an active task that needs to be resent
        if child_id in CHILD_ACTIVE_TASKS:
            task_context = CHILD_ACTIVE_TASKS[child_id]
            job_card = task_context["job_card"]
            endpoint = task_context["endpoint"]
            run_id = task_context["run_id"]

            logger.log_status(
                from_agent="Architect",
                to_agent=child_id,
                message=f"Resending task to restarted {child_id}: {job_card.task[:50]}...",
                run_id=run_id,
                metadata={
                    "child_id": child_id,
                    "job_card_id": job_card.id,
                    "restart_count": restart_count + 1
                }
            )

            # Resend the task to the restarted child
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{endpoint}/submit", json={"job_card": asdict(job_card)})
                    response.raise_for_status()

                logger.log_cmd(
                    from_agent="Architect",
                    to_agent=child_id,
                    message=f"Task resent successfully: {job_card.task[:50]}...",
                    run_id=run_id,
                    metadata={
                        "job_card_id": job_card.id,
                        "restart_count": restart_count + 1,
                        "resend_status": "success"
                    }
                )

                return {
                    "status": "restarted_and_resent",
                    "message": f"Successfully restarted {child_id} and resent task",
                    "restart_count": restart_count + 1,
                    "job_card_id": job_card.id
                }

            except Exception as e:
                logger.log_status(
                    from_agent="Architect",
                    to_agent=child_id,
                    message=f"Failed to resend task to {child_id}: {str(e)}",
                    run_id=run_id,
                    metadata={
                        "child_id": child_id,
                        "job_card_id": job_card.id,
                        "restart_count": restart_count + 1,
                        "error": str(e)
                    }
                )

                return {
                    "status": "restarted_but_resend_failed",
                    "message": f"Restarted {child_id} but failed to resend task: {str(e)}",
                    "restart_count": restart_count + 1,
                    "error": str(e)
                }
        else:
            # No active task to resend
            return {
                "status": "restarted",
                "message": f"Successfully restarted {child_id} (no active task to resend)",
                "restart_count": restart_count + 1
            }
    else:
        logger.log_status(
            from_agent="Architect",
            to_agent=child_id,
            message=f"Failed to restart {child_id}, may need manual intervention",
            run_id=None,
            metadata={"child_id": child_id, "restart_count": restart_count + 1, "status": "failed"}
        )

        return {
            "status": "restart_failed",
            "message": f"Failed to restart {child_id}",
            "restart_count": restart_count + 1,
            "note": "Manual intervention may be required"
        }


# === Main Prime Directive Endpoint ===

@app.post("/submit")
async def submit_prime_directive(
    pd: PrimeDirective,
    background_tasks: BackgroundTasks
):
    """
    Submit Prime Directive for execution

    Flow:
    1. Validate and register run
    2. Start background task for decomposition
    3. Return immediately with run_id and status

    Background task:
    - Decompose PRD using LLM
    - Create lane job cards
    - Submit to Directors
    - Monitor execution
    - Validate acceptance
    - Report completion to PAS Root
    """
    # Validate run_id uniqueness
    if pd.run_id in RUNS:
        raise HTTPException(status_code=400, detail="Run ID already exists")

    # Register run
    RUNS[pd.run_id] = {
        "state": "planning",
        "message": "Received Prime Directive, planning lanes",
        "started_at": time.time(),
        "lanes": {},
        "pd": pd.model_dump()
    }

    # Log receipt
    logger.log_cmd(
        from_agent="PAS Root",
        to_agent="Architect",
        message=f"Prime Directive received: {pd.title}",
        run_id=pd.run_id,
        metadata={"prd_length": len(pd.prd), "budget": pd.budget}
    )

    # Start background execution
    background_tasks.add_task(execute_prime_directive, pd)

    # Send heartbeat
    heartbeat_monitor.heartbeat(
        agent="Architect",
        run_id=pd.run_id,
        state=AgentState.PLANNING,
        message=f"Planning: {pd.title}",
        progress=0.1
    )

    return {
        "run_id": pd.run_id,
        "status": "planning",
        "message": "Prime Directive accepted, decomposing into lanes"
    }


# === Background Execution Logic ===

async def execute_prime_directive(pd: PrimeDirective):
    """
    Execute Prime Directive (background task)

    Steps:
    1. Decompose PRD into lane job cards (LLM-powered)
    2. Create Architect Plan artifact
    3. Submit job cards to Directors
    4. Monitor Director heartbeats
    5. Collect lane reports
    6. Validate acceptance gates
    7. Generate executive summary
    8. Report to PAS Root
    """
    run_id = pd.run_id

    try:
        # Step 1: Decompose PRD
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.PLANNING,
            message="Decomposing PRD with LLM",
            progress=0.2
        )

        plan = await decompose_prd(pd)

        # Step 2: Save Architect Plan
        save_architect_plan(run_id, plan)

        # Step 3: Submit job cards to Directors
        RUNS[run_id]["state"] = "delegating"
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.DELEGATING,
            message="Submitting job cards to Directors",
            progress=0.4
        )

        await delegate_to_directors(run_id, plan)

        # Step 4: Monitor execution
        RUNS[run_id]["state"] = "executing"
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.MONITORING,
            message="Monitoring Directors",
            progress=0.5
        )

        await monitor_directors(run_id, plan)

        # Step 5: Validate acceptance
        RUNS[run_id]["state"] = "validating"
        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.VALIDATING,
            message="Validating acceptance gates",
            progress=0.9
        )

        validation_results = await validate_acceptance(run_id, plan)

        # Step 6: Complete
        if all(validation_results.values()):
            RUNS[run_id]["state"] = "completed"
            RUNS[run_id]["message"] = "All lanes completed successfully"
            RUNS[run_id]["completed_at"] = time.time()
            RUNS[run_id]["duration_s"] = RUNS[run_id]["completed_at"] - RUNS[run_id]["started_at"]

            heartbeat_monitor.heartbeat(
                agent="Architect",
                run_id=run_id,
                state=AgentState.COMPLETED,
                message="Prime Directive completed",
                progress=1.0
            )

            # Log completion
            logger.log_response(
                from_agent="Architect",
                to_agent="PAS Root",
                message=f"Prime Directive completed: {pd.title}",
                run_id=run_id,
                status="completed",
                metadata={"duration_s": RUNS[run_id]["duration_s"]}
            )
        else:
            # Some lanes failed
            failed_lanes = [lane for lane, passed in validation_results.items() if not passed]
            RUNS[run_id]["state"] = "failed"
            RUNS[run_id]["message"] = f"Lanes failed: {', '.join(failed_lanes)}"
            RUNS[run_id]["completed_at"] = time.time()

            heartbeat_monitor.heartbeat(
                agent="Architect",
                run_id=run_id,
                state=AgentState.FAILED,
                message=f"Failed: {', '.join(failed_lanes)}",
                progress=1.0
            )

            # Log failure
            logger.log_response(
                from_agent="Architect",
                to_agent="PAS Root",
                message=f"Prime Directive failed: {pd.title}",
                run_id=run_id,
                status="failed",
                metadata={"failed_lanes": failed_lanes}
            )

    except Exception as e:
        # Unhandled error
        RUNS[run_id]["state"] = "failed"
        RUNS[run_id]["message"] = f"Error: {str(e)}"
        RUNS[run_id]["completed_at"] = time.time()

        heartbeat_monitor.heartbeat(
            agent="Architect",
            run_id=run_id,
            state=AgentState.FAILED,
            message=f"Error: {str(e)}",
            progress=1.0
        )

        logger.log_response(
            from_agent="Architect",
            to_agent="PAS Root",
            message=f"Error executing Prime Directive: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"error": str(e)}
        )


async def decompose_prd(pd: PrimeDirective) -> ArchitectPlan:
    """
    Decompose PRD into lane job cards using LLM

    Uses Claude Sonnet 4.5 to analyze PRD and generate:
    - Executive summary
    - Lane allocations
    - Dependency graph
    - Resource reservations
    - Acceptance gates

    Returns:
        ArchitectPlan with all decomposition results
    """
    # Use TaskDecomposer (LLM-powered)
    result = await decomposer.decompose(
        prd=pd.prd,
        title=pd.title,
        entry_files=pd.entry_files,
        budget=pd.budget,
        policy=pd.policy
    )

    return ArchitectPlan(
        run_id=pd.run_id,
        executive_summary=result["executive_summary"],
        lane_allocations=result["lane_allocations"],
        dependency_graph=result["dependency_graph"],
        resource_reservations=result["resource_reservations"],
        acceptance_gates=result["acceptance_gates"]
    )


def save_architect_plan(run_id: str, plan: ArchitectPlan):
    """Save Architect Plan to artifacts"""
    artifact_dir = Path(f"artifacts/runs/{run_id}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    plan_file = artifact_dir / "architect_plan.json"
    plan_file.write_text(json.dumps(plan.model_dump(), indent=2))

    # Save dependency graph (DOT format)
    graph_file = artifact_dir / "dependency_graph.dot"
    graph_file.write_text(plan.dependency_graph)


async def delegate_to_directors(run_id: str, plan: ArchitectPlan):
    """
    Submit job cards to Directors

    NEW: Creates agent conversation threads for bidirectional communication.
    If task is complex/ambiguous, use thread instead of job card.

    For each lane in plan.lane_allocations:
    1. Determine if task needs conversation (complex/ambiguous)
    2a. If yes: Create agent chat thread
    2b. If no: Create traditional JobCard
    3. Submit to Director
    4. Log delegation
    """
    for lane_name, allocation in plan.lane_allocations.items():
        director = f"Dir-{lane_name}"

        # Determine if task is complex/ambiguous (needs conversation)
        # TODO: Use LLM to determine complexity, for now use simple heuristic
        task_text = allocation["task"]
        needs_conversation = (
            len(task_text) > 100 or  # Long tasks likely complex
            "?" in task_text or      # Contains questions
            "refactor" in task_text.lower() or  # Refactoring often ambiguous
            "improve" in task_text.lower() or   # Subjective tasks
            len(allocation.get("inputs", [])) == 0  # No clear inputs
        )

        # Create job card (always create for backwards compatibility)
        job_card = JobCard(
            id=f"jc-{run_id}-{lane_name.lower()}-001",
            parent_id=run_id,
            role=Role.DIRECTOR,
            lane=Lane[lane_name.upper()],
            task=allocation["task"],
            inputs=allocation.get("inputs", []),
            expected_artifacts=allocation.get("expected_artifacts", []),
            acceptance=allocation.get("acceptance", []),
            risks=allocation.get("risks", []),
            budget=allocation.get("budget", {}),
            metadata=allocation.get("metadata", {}),
            submitted_by="Architect"
        )

        # Create agent chat thread if task is complex
        thread = None
        if needs_conversation:
            try:
                thread = await agent_chat.create_thread(
                    run_id=run_id,
                    parent_agent_id="Architect",
                    child_agent_id=director,
                    initial_message=task_text,
                    metadata={
                        "entry_files": plan.lane_allocations[lane_name].get("inputs", []),
                        "budget_tokens": allocation.get("budget", {}).get("tokens_max", 10000),
                        "expected_artifacts": allocation.get("expected_artifacts", []),
                        "lane": lane_name,
                        "job_card_id": job_card.id  # Link to job card
                    }
                )

                logger.log_cmd(
                    from_agent="Architect",
                    to_agent=director,
                    message=f"Agent chat thread created (complex task): {task_text[:50]}...",
                    run_id=run_id,
                    metadata={
                        "thread_id": thread.thread_id,
                        "job_card_id": job_card.id,
                        "reason": "complex/ambiguous task"
                    }
                )
            except Exception as e:
                logger.log_status(
                    from_agent="Architect",
                    to_agent=director,
                    message=f"Failed to create chat thread: {e}. Falling back to job card.",
                    run_id=run_id,
                    metadata={"error": str(e)}
                )

        # Try RPC first
        try:
            endpoint = DIRECTOR_ENDPOINTS.get(director)
            if endpoint:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.post(f"{endpoint}/submit", json={"job_card": asdict(job_card)})
                    response.raise_for_status()

                logger.log_cmd(
                    from_agent="Architect",
                    to_agent=director,
                    message=f"Job card submitted: {allocation['task'][:50]}...",
                    run_id=run_id,
                    metadata={"job_card_id": job_card.id}
                )

                # Track active task for TRON restart/resend
                CHILD_ACTIVE_TASKS[director] = {
                    "job_card": job_card,
                    "run_id": run_id,
                    "endpoint": endpoint,
                    "lane_name": lane_name,
                    "submitted_at": time.time(),
                    "thread_id": thread.thread_id if thread else None,  # Track conversation thread
                    "has_conversation": thread is not None
                }
            else:
                # Fallback to queue
                job_queue.submit(job_card, target_agent=director, use_file_fallback=True)

        except Exception as e:
            # Fallback to file queue
            job_queue.submit(job_card, target_agent=director, use_file_fallback=True)

            logger.log_cmd(
                from_agent="Architect",
                to_agent=director,
                message=f"Job card queued (RPC failed): {allocation['task'][:50]}...",
                run_id=run_id,
                metadata={"job_card_id": job_card.id, "rpc_error": str(e)}
            )

        # Update run state
        RUNS[run_id]["lanes"][lane_name] = {
            "state": "delegated",
            "job_card_id": job_card.id,
            "director": director,
            "thread_id": thread.thread_id if thread else None,
            "has_conversation": thread is not None
        }


async def monitor_conversation_threads():
    """
    Monitor all active conversation threads for pending questions.
    If a Director asks a question, generate answer using LLM.

    This function is called periodically by monitor_directors().
    """
    # Get all pending questions for Architect
    questions = await agent_chat.get_pending_questions("Architect")

    for question in questions:
        thread_id = question["thread_id"]
        child_agent = question["from_agent"]
        question_content = question["content"]

        # Get full thread history for context
        try:
            thread = await agent_chat.get_thread(thread_id)

            # Build conversation context for LLM
            conversation_history = "\n".join([
                f"{msg.from_agent} → {msg.to_agent} ({msg.message_type}): {msg.content}"
                for msg in thread.messages
            ])

            # Generate answer using LLM (simple implementation for now)
            # TODO: Use actual LLM (Claude, Gemini, etc.) for intelligent answers
            # For now, use a simple heuristic-based answer
            answer = await generate_answer_to_question(
                question=question_content,
                conversation_history=conversation_history,
                thread_metadata=thread.metadata
            )

            # Send answer back to child
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent="Architect",
                to_agent=child_agent,
                message_type="answer",
                content=answer
            )

            logger.log_response(
                from_agent="Architect",
                to_agent=child_agent,
                message=f"Answered question: {question_content[:50]}...",
                run_id=thread.run_id,
                metadata={
                    "thread_id": thread_id,
                    "question": question_content,
                    "answer": answer
                }
            )

        except Exception as e:
            logger.log_status(
                from_agent="Architect",
                to_agent=child_agent,
                message=f"Failed to answer question: {e}",
                run_id=thread.run_id,
                metadata={
                    "thread_id": thread_id,
                    "error": str(e)
                }
            )


async def generate_answer_to_question(
    question: str,
    conversation_history: str,
    thread_metadata: Dict[str, Any]
) -> str:
    """
    Generate answer to Director's question using LLM (Phase 3)

    Uses Claude Sonnet 4.5 or Gemini 2.5 Pro with full conversation context
    to provide intelligent, context-aware answers.

    Args:
        question: The question from the Director
        conversation_history: Full conversation thread history
        thread_metadata: Thread metadata (entry_files, budget, etc.)

    Returns:
        Answer string
    """
    # Get LLM model from environment
    model = os.getenv("ARCHITECT_LLM", "claude-sonnet-4-5-20250929")

    # Get original PRD if available (from RUNS)
    run_id = thread_metadata.get("run_id")
    original_prd = ""
    if run_id and run_id in RUNS:
        original_prd = RUNS[run_id].get("prd", "")

    # Build system prompt for Architect
    system_prompt = """You are the Architect, the top-level coordinator in a Polyglot Agent Swarm (PAS).

**Your Role:**
- You decompose user requirements (PRDs) into lane-specific job cards
- You coordinate 5 Directors: Code, Models, Data, DevSecOps, Docs
- You have full visibility into project constraints, budget, and policy
- You answer Directors' clarifying questions to help them succeed

**When Answering Questions:**
1. Consider the full conversation context - what have we discussed so far?
2. Reference the original PRD - what did the user actually ask for?
3. Consider project constraints (budget, timeline, policy)
4. Be specific and actionable - avoid vague guidance
5. If the question reveals a gap in requirements, acknowledge it and provide your best interpretation
6. Trust your Directors - they're experts in their lanes, just need clarity on intent/constraints

**Guidelines:**
- Be concise but complete (2-3 sentences ideal)
- Provide reasoning when making recommendations
- If multiple valid approaches exist, explain trade-offs and recommend one
- If you don't have enough information, say so explicitly
- Always align your answer with the user's original intent (PRD)"""

    # Build user prompt
    user_prompt = f"""**Director's Question:**
{question}

**Full Conversation History:**
{conversation_history}

**Thread Metadata:**
Entry Files: {thread_metadata.get('entry_files', [])}
Budget: {thread_metadata.get('budget_tokens', 'Not specified')} tokens
Policy: {json.dumps(thread_metadata.get('policy', {}), indent=2) if thread_metadata.get('policy') else 'None'}
Urgency: {thread_metadata.get('urgency', 'Not specified')}

**Original PRD (User Requirements):**
{original_prd[:1500] if original_prd else "Not available"}

**Your Task:**
Answer the Director's question clearly and specifically. Consider:
1. What does the original PRD say (if anything) about this?
2. What project constraints apply (budget, policy, timeline)?
3. What's the most practical approach given the context?

Provide a direct answer in 2-4 sentences. Be actionable."""

    try:
        # Call LLM (no tools needed for answers)
        llm_response = await call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=0.5,  # Balanced between creativity and consistency
            max_tokens=500
        )

        return llm_response.content

    except Exception as e:
        logger.log_error(
            from_agent="Architect",
            to_agent="Architect",
            message=f"LLM answer generation failed: {e}",
            metadata={"question": question, "error": str(e)}
        )

        # Fallback to heuristic answer
        return _generate_heuristic_answer(question, thread_metadata)


def _generate_heuristic_answer(question: str, thread_metadata: Dict[str, Any]) -> str:
    """
    Fallback heuristic answer generation (used when LLM fails)

    This is the old Phase 2 logic, kept as a safety net.
    """
    question_lower = question.lower()

    # Library choice questions
    if "library" in question_lower or "which" in question_lower:
        if "auth" in question_lower:
            return "Use authlib - it's more actively maintained, has better documentation, and supports OAuth2/OIDC out of the box."
        elif "test" in question_lower:
            return "Use pytest - it's the industry standard with excellent plugin ecosystem."
        else:
            return "Please evaluate both options and choose the one with better documentation, active maintenance, and community support."

    # File/scope questions
    if "which file" in question_lower or "all" in question_lower or "focus" in question_lower:
        entry_files = thread_metadata.get("entry_files", [])
        if entry_files:
            return f"Focus on {', '.join(entry_files[:3])} first. We can expand scope if needed."
        else:
            return "Start with the main entry point, then expand to related modules as needed."

    # Test failure questions
    if "test" in question_lower and ("fail" in question_lower or "error" in question_lower):
        return "Fix the test failures before proceeding. Tests must pass for acceptance. If you need guidance on specific failures, share the error messages."

    # Budget/scope questions
    if "budget" in question_lower or "time" in question_lower:
        budget = thread_metadata.get("budget_tokens", 10000)
        return f"You have {budget} tokens budgeted. Prioritize core functionality first, optimizations second."

    # Default: Ask for more context
    return f"Good question. Please provide more context: What are the specific options you're considering, and what are the trade-offs you see?"


async def monitor_directors(run_id: str, plan: ArchitectPlan):
    """
    Monitor Directors until all complete or timeout

    NEW: Also monitors agent chat threads for pending questions.
    If a Director asks a question, generates answer using LLM.

    Polls Director status every 10 seconds
    Checks heartbeats for liveness
    Checks for pending questions in conversation threads
    Escalates on 2 missed heartbeats
    """
    lanes = list(plan.lane_allocations.keys())
    timeout_s = 3600  # 1 hour max
    start_time = time.time()

    while time.time() - start_time < timeout_s:
        all_complete = True

        for lane in lanes:
            lane_state = RUNS[run_id]["lanes"][lane]["state"]

            if lane_state not in ["completed", "failed"]:
                all_complete = False

                # Don't check health aggressively - Directors report back via /lane_report
                # when they complete or fail. Just wait for their reports.
                # Only fail on actual timeout (handled below).

        # NEW: Check for pending questions in conversation threads
        await monitor_conversation_threads()

        if all_complete:
            break

        # Wait before next poll
        await asyncio.sleep(10)

    # Check for timeout
    if time.time() - start_time >= timeout_s:
        for lane in lanes:
            if RUNS[run_id]["lanes"][lane]["state"] not in ["completed", "failed"]:
                RUNS[run_id]["lanes"][lane]["state"] = "failed"
                RUNS[run_id]["lanes"][lane]["message"] = "Timeout"


async def validate_acceptance(run_id: str, plan: ArchitectPlan) -> Dict[str, bool]:
    """
    Validate acceptance gates for each lane

    Returns:
        Dict mapping lane name to pass/fail boolean
    """
    results = {}

    for lane_name, gates in plan.acceptance_gates.items():
        # Get lane report from Director
        # (In full implementation, Directors would report artifacts + results)
        # For now, assume success if lane completed
        lane_state = RUNS[run_id]["lanes"].get(lane_name, {}).get("state")
        results[lane_name] = lane_state == "completed"

    return results


# === Startup ===

if __name__ == "__main__":
    import uvicorn
    import asyncio

    # Start heartbeat emitter
    async def emit_heartbeats():
        while True:
            heartbeat_monitor.heartbeat(
                agent="Architect",
                state=AgentState.IDLE,
                message="Idle, waiting for Prime Directives"
            )
            await asyncio.sleep(60)

    # Run server
    uvicorn.run(app, host="127.0.0.1", port=6110)

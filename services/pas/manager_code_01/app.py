#!/usr/bin/env python3
"""
Manager-Code-01 Service - Code Lane Task Breakdown

Port: 6141
Tier: 4 (Manager)
LLM: Gemini 2.5 Flash (primary), Claude Haiku 4 (fallback)

Responsibilities:
- Receive job cards from Director-Code
- Break down tasks into surgical Programmer-level tasks
- Delegate to Programmers (up to 5 parallel)
- Monitor Programmer progress
- Validate outputs
- Report back to Director

Architecture:
- FastAPI HTTP service (like Director)
- LLM-powered task decomposition (surgical, atomic tasks)
- Programmer Pool integration (load balancing)
- Parallel execution (up to 5 Programmers)
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
import requests
import json
from datetime import datetime
import uuid
import asyncio

# PAS common services
from services.common.heartbeat import get_monitor, AgentState
from services.common.job_queue import get_queue, JobCard, Lane, Role, Priority
from services.common.comms_logger import get_logger, MessageType
from services.common.programmer_pool import get_programmer_pool

app = FastAPI(title="Manager-Code-01", version="1.0.0")

# Initialize systems
heartbeat_monitor = get_monitor()
job_queue = get_queue()
logger = get_logger()
programmer_pool = get_programmer_pool()

# Service configuration
SERVICE_NAME = "Manager-Code-01"
SERVICE_PORT = 6141
AGENT_ID = "Mgr-Code-01"
PARENT_AGENT = "Dir-Code"
TIER = "manager"
LANE = "Code"

# LLM Configuration (load from env or config file)
PRIMARY_LLM = os.getenv("MGR_CODE_01_PRIMARY_LLM", "google/gemini-2.5-flash")
BACKUP_LLM = os.getenv("MGR_CODE_01_BACKUP_LLM", "anthropic/claude-haiku-4")

# Register Manager agent with Heartbeat Monitor
heartbeat_monitor.register_agent(
    agent=AGENT_ID,
    parent=PARENT_AGENT,
    llm_model=PRIMARY_LLM,
    role="manager",
    tier=TIER
)

# Programmer endpoints (will load from Programmer Pool)
PROGRAMMER_ENDPOINTS: Dict[str, str] = {}

# In-memory job tracking
JOBS: Dict[str, Dict[str, Any]] = {}


# === Pydantic Models ===

class JobCardInput(BaseModel):
    """Job card from Director"""
    job_card: Dict[str, Any]


class ProgrammerTask(BaseModel):
    """Programmer task definition (surgical, atomic)"""
    programmer_id: str
    task: str
    files: List[str]
    operation: str  # 'create', 'modify', 'delete', 'refactor'
    context: List[str] = Field(default_factory=list)  # Related files for context
    acceptance: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)
    timeout_s: int = 300


class ManagerReport(BaseModel):
    """Manager report to Director"""
    manager: str = AGENT_ID
    state: str  # completed | failed | partial
    artifacts: Dict[str, str] = Field(default_factory=dict)
    programmers_used: Dict[str, str] = Field(default_factory=dict)
    actuals: Dict[str, Any] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)


# === Health & Status Endpoints ===

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": "1.0.0",
        "port": SERVICE_PORT,
        "agent": AGENT_ID,
        "parent": PARENT_AGENT,
        "tier": TIER,
        "lane": LANE,
        "llm_model": PRIMARY_LLM,
        "agent_metadata": {
            "role": "manager",
            "tier": "manager",
            "parent": PARENT_AGENT,
            "grandparent": "Architect",
            "max_programmers": 5
        }
    }


@app.get("/status/{job_card_id}")
async def get_status(job_card_id: str):
    """Get job card status"""
    if job_card_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job card {job_card_id} not found")

    job = JOBS[job_card_id]
    return {
        "job_card_id": job_card_id,
        "state": job["state"],
        "manager": AGENT_ID,
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "duration_s": job.get("duration_s"),
        "programmers": job.get("programmers", {}),
        "artifacts": job.get("artifacts", {}),
        "errors": job.get("errors", [])
    }


# === Job Submission Endpoint ===

@app.post("/submit")
async def submit_job_card(input: JobCardInput, background_tasks: BackgroundTasks):
    """
    Submit job card to Manager

    Flow:
    1. Accept job card from Director
    2. Decompose into surgical Programmer tasks (LLM-powered)
    3. Delegate to Programmers (parallel execution)
    4. Monitor Programmer progress
    5. Validate outputs
    6. Report back to Director
    """
    job_card = input.job_card
    job_card_id = job_card.get("id", str(uuid.uuid4()))

    # Log job card acceptance
    logger.log(
        from_agent=PARENT_AGENT,
        to_agent=AGENT_ID,
        msg_type=MessageType.CMD,
        message=f"Job card received: {job_card.get('task', 'Unknown')[:50]}...",
        run_id=job_card_id,
        metadata={"job_card_id": job_card_id}
    )

    # Initialize job tracking
    JOBS[job_card_id] = {
        "state": "planning",
        "job_card": job_card,
        "started_at": datetime.now().isoformat(),
        "programmers": {},
        "artifacts": {},
        "errors": []
    }

    # Update heartbeat
    heartbeat_monitor.heartbeat(
        agent=AGENT_ID,
        state=AgentState.PLANNING,
        message=f"Processing {job_card_id}",
        metadata={"job_card_id": job_card_id}
    )

    # Process in background
    background_tasks.add_task(process_job_card, job_card_id)

    # Log acceptance
    logger.log(
        from_agent=AGENT_ID,
        to_agent=PARENT_AGENT,
        msg_type=MessageType.RESPONSE,
        message=f"Accepted job card: {job_card.get('task', 'Unknown')[:50]}...",
        run_id=job_card_id,
        status="planning"
    )

    return {
        "status": "accepted",
        "job_card_id": job_card_id,
        "manager": AGENT_ID,
        "state": "planning"
    }


# === Background Processing ===

async def process_job_card(job_card_id: str):
    """
    Background task to process job card

    Steps:
    1. Decompose into Programmer tasks (LLM)
    2. Delegate to Programmers (parallel)
    3. Monitor progress
    4. Validate outputs
    5. Report to Director
    """
    job = JOBS[job_card_id]
    job_card = job["job_card"]

    try:
        # Step 1: Decompose into Programmer tasks (LLM-powered)
        job["state"] = "decomposing"
        logger.log(
            from_agent=AGENT_ID,
            to_agent=AGENT_ID,
            msg_type=MessageType.STATUS,
            message=f"Decomposing job card into Programmer tasks",
            run_id=job_card_id,
            status="decomposing"
        )

        programmer_tasks = await decompose_into_programmer_tasks(job_card)
        job["programmer_tasks"] = programmer_tasks
        job["programmers"] = {}  # Will be populated by delegate_to_programmers

        logger.log(
            from_agent=AGENT_ID,
            to_agent=AGENT_ID,
            msg_type=MessageType.STATUS,
            message=f"Decomposed into {len(programmer_tasks)} Programmer tasks",
            run_id=job_card_id,
            status="decomposing",
            metadata={"task_count": len(programmer_tasks)}
        )

        # Step 2: Delegate to Programmers (parallel execution)
        job["state"] = "delegating"
        logger.log(
            from_agent=AGENT_ID,
            to_agent=AGENT_ID,
            msg_type=MessageType.STATUS,
            message=f"Delegating to Programmers",
            run_id=job_card_id,
            status="delegating"
        )

        results = await delegate_to_programmers(job_card_id, programmer_tasks)

        # Update programmers dict with results
        for result in results:
            prog_id = result.get("programmer_id", "unknown")
            status = "completed" if result["success"] else "failed"
            job["programmers"][prog_id] = status

        # Step 3: Validate results
        job["state"] = "validating"
        all_success = all(r["success"] for r in results)

        if all_success:
            job["state"] = "completed"
            job["completed_at"] = datetime.now().isoformat()
            job["duration_s"] = (
                datetime.fromisoformat(job["completed_at"]) -
                datetime.fromisoformat(job["started_at"])
            ).total_seconds()

            logger.log(
                from_agent=AGENT_ID,
                to_agent=PARENT_AGENT,
                msg_type=MessageType.RESPONSE,
                message=f"Job completed successfully",
                run_id=job_card_id,
                status="completed",
                metadata={"duration_s": job["duration_s"]}
            )
        else:
            job["state"] = "failed"
            job["errors"] = [r.get("error") for r in results if not r["success"]]

            logger.log(
                from_agent=AGENT_ID,
                to_agent=PARENT_AGENT,
                msg_type=MessageType.RESPONSE,
                message=f"Job failed",
                run_id=job_card_id,
                status="failed",
                metadata={"errors": job["errors"]}
            )

        # Update heartbeat
        heartbeat_monitor.heartbeat(
            agent=AGENT_ID,
            state=AgentState.IDLE,
            message=f"Completed {job_card_id}"
        )

    except Exception as e:
        job["state"] = "error"
        job["errors"].append(str(e))

        logger.log(
            from_agent=AGENT_ID,
            to_agent=PARENT_AGENT,
            msg_type=MessageType.RESPONSE,
            message=f"Job error: {str(e)}",
            run_id=job_card_id,
            status="error"
        )

        heartbeat_monitor.heartbeat(
            agent=AGENT_ID,
            state=AgentState.FAILED,
            message=f"Error: {str(e)}"
        )


async def decompose_into_programmer_tasks(job_card: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Decompose job card into surgical Programmer tasks using LLM

    This is where the Manager's intelligence lives. The LLM breaks down
    the Director's job card into atomic, surgical tasks for Programmers.

    Example:
        Director: "Add authentication to the API"
        Manager decomposes into:
        1. Create auth middleware (app/middleware/auth.py)
        2. Add JWT token generation (app/utils/jwt.py)
        3. Add login endpoint (app/api/auth.py)
        4. Update existing endpoints to require auth (app/api/*.py)
        5. Add tests (tests/test_auth.py)
    """
    # TODO: Implement LLM-powered decomposition (Phase 2)
    # For now, create a simple 1:1 mapping (P0 fallback)

    task = job_card.get("task", "")
    files = job_card.get("inputs", [])
    file_paths = [f["path"] for f in files if f.get("type") == "file"]

    # Simple decomposition: 1 Programmer task per file
    programmer_tasks = []
    for idx, file_path in enumerate(file_paths):
        programmer_tasks.append({
            "task": f"{task} in {file_path}",
            "files": [file_path],
            "operation": "modify",
            "context": file_paths,  # All files for context
            "acceptance": job_card.get("acceptance", []),
            "budget": job_card.get("budget", {}),
            "timeout_s": 300
        })

    # If no files, create a single generic task
    if not programmer_tasks:
        programmer_tasks.append({
            "task": task,
            "files": [],
            "operation": "create",
            "context": [],
            "acceptance": job_card.get("acceptance", []),
            "budget": job_card.get("budget", {}),
            "timeout_s": 300
        })

    return programmer_tasks


async def delegate_to_programmers(job_card_id: str, programmer_tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Delegate tasks to Programmers in parallel using Programmer Pool

    Steps:
    1. Discover available Programmers from pool
    2. For each task, assign to next available Programmer (round-robin)
    3. POST to Programmer's /execute endpoint
    4. Execute in parallel (asyncio.gather)
    5. Collect results
    """
    # Discover Programmers (auto-discovery)
    programmer_pool.discover_programmers()

    # Execute tasks in parallel
    async def execute_task(task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute single task on assigned Programmer"""
        try:
            # Assign Programmer from pool (round-robin)
            programmer_info = programmer_pool.assign_task(task["task"])

            if not programmer_info:
                return {
                    "success": False,
                    "programmer_id": task.get("programmer_id", "unknown"),
                    "task": task["task"],
                    "error": "No available Programmers in pool"
                }

            # POST to Programmer's /execute endpoint
            async with httpx.AsyncClient(timeout=task["timeout_s"] + 10) as client:
                response = await client.post(
                    f"{programmer_info.endpoint}/execute",
                    json={
                        "task": task["task"],
                        "files": task["files"],
                        "llm_provider": "ollama",  # Default to free local model
                        "llm_model": "qwen2.5-coder:7b-instruct",
                        "run_id": job_card_id,
                        "timeout_s": task["timeout_s"]
                    }
                )

                # Release Programmer back to pool
                success = response.status_code == 200
                programmer_pool.release_task(programmer_info.agent_id, success=success)

                if success:
                    result = response.json()
                    logger.log(
                        from_agent=programmer_info.agent_id,
                        to_agent=AGENT_ID,
                        msg_type=MessageType.RESPONSE,
                        message=f"Task completed: {task['task'][:50]}...",
                        run_id=job_card_id,
                        status="completed"
                    )

                    return {
                        "success": True,
                        "programmer_id": programmer_info.agent_id,
                        "task": task["task"],
                        "result": result,
                        "metrics": result.get("metrics", {})
                    }
                else:
                    logger.log(
                        from_agent=programmer_info.agent_id,
                        to_agent=AGENT_ID,
                        msg_type=MessageType.RESPONSE,
                        message=f"Task failed: {task['task'][:50]}...",
                        run_id=job_card_id,
                        status="failed",
                        metadata={"status_code": response.status_code}
                    )

                    return {
                        "success": False,
                        "programmer_id": programmer_info.agent_id,
                        "task": task["task"],
                        "error": f"Programmer returned {response.status_code}"
                    }

        except Exception as e:
            logger.log(
                from_agent=AGENT_ID,
                to_agent=AGENT_ID,
                msg_type=MessageType.STATUS,
                message=f"Task error: {str(e)}",
                run_id=job_card_id,
                status="error"
            )

            return {
                "success": False,
                "programmer_id": task.get("programmer_id", "unknown"),
                "task": task["task"],
                "error": str(e)
            }

    # Execute all tasks in parallel
    results = await asyncio.gather(*[execute_task(task) for task in programmer_tasks])
    return list(results)


# === Startup Event ===

@app.on_event("startup")
async def startup_event():
    """Service startup"""
    logger.log(
        from_agent=AGENT_ID,
        to_agent="System",
        msg_type=MessageType.STATUS,
        message=f"Manager service started on port {SERVICE_PORT}",
        status="started",
        metadata={
            "agent": AGENT_ID,
            "port": SERVICE_PORT,
            "tier": TIER,
            "lane": LANE,
            "llm_model": PRIMARY_LLM
        }
    )

    print(f"✓ {SERVICE_NAME} started on port {SERVICE_PORT}")
    print(f"  Agent: {AGENT_ID}")
    print(f"  Parent: {PARENT_AGENT}")
    print(f"  Tier: {TIER}")
    print(f"  Lane: {LANE}")
    print(f"  LLM: {PRIMARY_LLM}")


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    logger.log(
        from_agent=AGENT_ID,
        to_agent="System",
        msg_type=MessageType.STATUS,
        message=f"Manager service shutdown",
        status="shutdown"
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=SERVICE_PORT)

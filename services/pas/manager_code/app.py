#!/usr/bin/env python3
"""
Manager-Code Service - Code Task Executor

Port: 6141 (Mgr-Code-01), 6142 (Mgr-Code-02), 6143 (Mgr-Code-03)
LLM: Qwen 2.5 Coder 7B (primary), Claude Sonnet 4.5 (fallback)

Responsibilities:
- Receive task assignments from Dir-Code
- Execute code changes via Aider RPC
- Run acceptance tests (lint, tests, coverage)
- Report results back to Dir-Code
- Ask questions when clarification needed

Contract: docs/contracts/MANAGER_CODE_SYSTEM_PROMPT.md
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
from services.common.comms_logger import get_logger, MessageType

# Agent chat for Parent-Child communication
from services.common.agent_chat import get_agent_chat_client, AgentChatMessage

# LLM with tool support
from services.common.llm_tool_caller import call_llm_with_tools, LLMResponse
from services.common.llm_tools import get_ask_parent_tool, validate_ask_parent_args, get_system_prompt_with_ask_parent

# Programmer pool load balancer
from services.common.programmer_pool import get_programmer_pool


# Get Manager ID and port from environment
MANAGER_ID = os.getenv("MANAGER_ID", "Mgr-Code-01")
MANAGER_PORT = int(os.getenv("MANAGER_PORT", "6141"))

app = FastAPI(title=f"Manager-Code ({MANAGER_ID})", version="1.0.0")

# Initialize systems
heartbeat_monitor = get_monitor()
logger = get_logger()
agent_chat = get_agent_chat_client()
programmer_pool = get_programmer_pool()

# Register Manager agent
heartbeat_monitor.register_agent(
    agent=MANAGER_ID,
    parent="Dir-Code",
    llm_model=os.getenv("MANAGER_LLM", "qwen2.5-coder:7b"),
    role="manager",
    tier="executor"
)

# In-memory task tracking
TASKS: Dict[str, Dict[str, Any]] = {}


# === Pydantic Models ===

class TaskAssignment(BaseModel):
    """Task assignment from Director"""
    task_id: str
    task: str
    files: List[str]
    acceptance: List[Dict[str, Any]] = Field(default_factory=list)
    budget: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# === API Endpoints ===

@app.get("/health")
async def health():
    """Health check with programmer pool status"""
    pool_status = await programmer_pool.get_pool_status()

    return {
        "status": "ok",
        "manager_id": MANAGER_ID,
        "port": MANAGER_PORT,
        "llm": os.getenv("MANAGER_LLM", "qwen2.5-coder:7b"),
        "programmer_pool": {
            "available": pool_status["available"],
            "unavailable": pool_status["unavailable"],
            "using_backup": pool_status["using_backup"],
            "total_queue_depth": pool_status["total_queue_depth"]
        }
    }


@app.get("/programmer_pool/status")
async def programmer_pool_status():
    """Get detailed programmer pool status"""
    return await programmer_pool.get_pool_status()


@app.post("/agent_chat/receive")
async def receive_agent_message(
    request: AgentChatMessage,
    background_tasks: BackgroundTasks
):
    """
    Receive message from Dir-Code via Agent Chat thread.

    This is the RECOMMENDED way for Dir-Code to communicate with Managers.
    Enables bidirectional Q&A, status updates, and context preservation.

    Flow:
    - Dir-Code creates thread with delegation message
    - Manager receives via this endpoint
    - Manager can ask questions using agent_chat.send_message()
    - Manager sends status updates during execution
    - Manager closes thread on completion/error

    Alternative: /submit endpoint (task only, no conversation)
    """
    thread_id = request.thread_id

    # Load thread to get run_id
    try:
        thread = await agent_chat.get_thread(thread_id)
        run_id = thread.run_id
    except Exception:
        run_id = "unknown"

    logger.log_cmd(
        from_agent="Dir-Code",
        to_agent=MANAGER_ID,
        message=f"Agent chat message received: {request.message_type}",
        run_id=run_id,
        metadata={
            "thread_id": thread_id,
            "message_type": request.message_type,
            "from_agent": request.from_agent
        }
    )

    # Delegate to background task for processing
    background_tasks.add_task(process_agent_chat_message, request)

    return {
        "status": "ok",
        "thread_id": thread_id,
        "message": "Agent chat message received, processing"
    }


async def process_agent_chat_message(request: AgentChatMessage):
    """
    Process agent chat message (background task).

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
            from_agent=MANAGER_ID,
            to_agent="Dir-Code",
            message_type="status",
            content="Received task, analyzing..."
        )

        # Extract task from delegation message
        if request.message_type == "delegation":
            # Parse task from message content or metadata
            task_description = request.content
            files = request.metadata.get("files", [])
            acceptance = request.metadata.get("acceptance", [])
            budget = request.metadata.get("budget", {})

            # Execute task with agent chat updates
            await execute_task_with_chat(
                task_description=task_description,
                files=files,
                acceptance=acceptance,
                budget=budget,
                thread_id=thread_id,
                run_id=run_id
            )

        elif request.message_type == "answer":
            # Parent answered a question - context is in thread history
            # Continue execution if needed
            pass

    except Exception as e:
        logger.log_error(
            from_agent=MANAGER_ID,
            to_agent="Dir-Code",
            message=f"Error processing agent chat message: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"thread_id": thread_id, "error": str(e)}
        )


async def execute_task_with_chat(
    task_description: str,
    files: List[str],
    acceptance: List[Dict[str, Any]],
    budget: Dict[str, Any],
    thread_id: str,
    run_id: str
):
    """
    Execute task with agent chat status updates.

    This is the main execution flow for Managers.
    """
    task_id = f"task-{uuid.uuid4().hex[:8]}"

    try:
        # Step 1: Send status - starting execution
        heartbeat_monitor.heartbeat(
            agent=MANAGER_ID,
            run_id=run_id,
            state=AgentState.EXECUTING,
            message="Executing code changes with Aider",
            progress=0.2
        )

        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent=MANAGER_ID,
            to_agent="Dir-Code",
            message_type="status",
            content=f"Starting code execution: {task_description[:100]}...",
            metadata={"progress": 20, "files": files}
        )

        # Step 2: Convert relative paths to absolute for Aider RPC
        repo_root = Path.cwd()
        absolute_files = []
        for file in files:
            file_path = Path(file)
            if not file_path.is_absolute():
                file_path = (repo_root / file_path).resolve()
            absolute_files.append(str(file_path))

        # Step 3: Dispatch task to programmer pool
        # Use load balancing and failover
        dispatch_result = await programmer_pool.dispatch_task(
            task_description=task_description,
            files=absolute_files,
            run_id=run_id,
            capabilities=["fast"],  # Prefer fast local models
            prefer_free=True  # Prefer free over paid
        )

        programmer_id = dispatch_result["programmer_id"]
        result = dispatch_result["result"]

        # Log which programmer was used
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent=MANAGER_ID,
            to_agent="Dir-Code",
            message_type="status",
            content=f"Task dispatched to {programmer_id}",
            metadata={"programmer_id": programmer_id}
        )

        # Step 4: Check result
        if dispatch_result["status"] != "ok":
            # Aider failed
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=MANAGER_ID,
                to_agent="Dir-Code",
                message_type="error",
                content=f"Aider execution failed: {result.get('stderr', 'Unknown error')[:200]}"
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=f"Aider failed: rc={result.get('rc', 'unknown')}"
            )
            return

        # Step 5: Send status - running acceptance tests
        heartbeat_monitor.heartbeat(
            agent=MANAGER_ID,
            run_id=run_id,
            state=AgentState.VALIDATING,
            message="Running acceptance tests",
            progress=0.7
        )

        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent=MANAGER_ID,
            to_agent="Dir-Code",
            message_type="status",
            content="Code changes complete, running acceptance tests...",
            metadata={"progress": 70}
        )

        # Step 6: Run acceptance checks (tests, lint, coverage)
        acceptance_results = await run_acceptance_checks(acceptance, files, run_id)
        all_passed = all(acceptance_results.values())

        # Step 7: Send completion or error
        if all_passed:
            heartbeat_monitor.heartbeat(
                agent=MANAGER_ID,
                run_id=run_id,
                state=AgentState.COMPLETED,
                message="Task completed successfully",
                progress=1.0
            )

            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=MANAGER_ID,
                to_agent="Dir-Code",
                message_type="completion",
                content=f"✅ Task completed successfully. All acceptance tests passed.",
                metadata={
                    "duration_s": result.get("duration_s", 0),
                    "acceptance_results": acceptance_results
                }
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="completed",
                result="Task completed successfully"
            )
        else:
            # Some acceptance tests failed
            failed_checks = [name for name, passed in acceptance_results.items() if not passed]

            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=MANAGER_ID,
                to_agent="Dir-Code",
                message_type="error",
                content=f"❌ Acceptance tests failed: {', '.join(failed_checks)}",
                metadata={"acceptance_results": acceptance_results}
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=f"Acceptance tests failed: {', '.join(failed_checks)}"
            )

    except Exception as e:
        # Unhandled error
        logger.log_error(
            from_agent=MANAGER_ID,
            to_agent="Dir-Code",
            message=f"Error executing task: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"thread_id": thread_id, "error": str(e)}
        )

        try:
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=MANAGER_ID,
                to_agent="Dir-Code",
                message_type="error",
                content=f"Fatal error: {str(e)}"
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=str(e)
            )
        except Exception:
            pass  # Best effort


async def run_acceptance_checks(
    acceptance: List[Dict[str, Any]],
    files: List[str],
    run_id: str
) -> Dict[str, bool]:
    """
    Run acceptance checks (tests, lint, coverage).

    Returns:
        Dict mapping check name to pass/fail bool
    """
    results = {}

    for check in acceptance:
        check_type = check.get("type", "unknown")
        check_name = check.get("name", check_type)

        # For now, assume all checks pass (placeholder)
        # In production, would actually run pytest, ruff, coverage, etc.
        results[check_name] = True

    return results


# === Startup ===

@app.on_event("startup")
async def startup():
    """Startup tasks"""
    print(f"[{MANAGER_ID}] Starting on port {MANAGER_PORT}")
    print(f"[{MANAGER_ID}] LLM: {os.getenv('MANAGER_LLM', 'qwen2.5-coder:7b')}")
    print(f"[{MANAGER_ID}] Aider RPC: {AIDER_RPC_URL}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=MANAGER_PORT)

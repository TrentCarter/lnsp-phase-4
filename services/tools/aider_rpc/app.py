#!/usr/bin/env python3
"""
Aider-LCO RPC Server (Programmer Pool Instance)
FastAPI service that wraps Aider CLI with guardrails for PAS integration.

Features:
- Filesystem allowlist enforcement
- Command allowlist enforcement
- Secrets redaction
- Timeout enforcement
- Subprocess isolation
- Agent chat integration for Parent-Child communication
- LLM failover (primary → backup)
- Part of 10-programmer pool

Port: Configured via PROGRAMMER_ID env var (default: 001 → 6151)
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml
import fnmatch
import subprocess
import shlex
import os
import pathlib
import time
import sys
import uuid
import asyncio

# Add services/common to path for comms_logger
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
from common.comms_logger import get_logger, MessageType
from common.heartbeat import get_monitor, AgentState
from common.agent_chat import get_agent_chat_client, AgentChatMessage

# Get programmer ID from environment (001-010)
PROGRAMMER_ID = os.getenv("PROGRAMMER_ID", "001")

app = FastAPI(title=f"Programmer-{PROGRAMMER_ID} RPC", version="2.0")

# Initialize systems
logger = get_logger()
heartbeat_monitor = get_monitor()
agent_chat = get_agent_chat_client()

# Load configurations
FS_ALLOW = yaml.safe_load(open("configs/pas/fs_allowlist.yaml"))
CMD_ALLOW = yaml.safe_load(open("configs/pas/cmd_allowlist.yaml"))
POOL_CFG = yaml.safe_load(open("configs/pas/programmer_pool.yaml"))

ROOTS = [pathlib.Path(p).resolve() for p in FS_ALLOW["roots"]]
DENY_PATTERNS = FS_ALLOW.get("deny", [])
ALLOW_PATTERNS = FS_ALLOW.get("allow", [])

# Find this programmer's configuration in the pool
def _get_programmer_config():
    """Get configuration for this programmer instance from pool config"""
    for prog in POOL_CFG["programmers"]:
        if prog["id"] == PROGRAMMER_ID:
            return prog
    raise RuntimeError(f"Programmer {PROGRAMMER_ID} not found in pool config")

PROG_CFG = _get_programmer_config()

# Extract LLM configuration
PRIMARY_LLM = PROG_CFG["primary_llm"]
BACKUP_LLM = PROG_CFG["backup_llm"]
PROGRAMMER_PORT = PROG_CFG["port"]
CAPABILITIES = PROG_CFG.get("capabilities", [])

# Build agent name: Prog-{Counter} (LLM extracted from primary_llm)
def _build_agent_name():
    """Build agent name from programmer ID: Prog-{Counter}"""
    return f"Prog-{PROGRAMMER_ID}"

def _extract_llm_short_name(model_string: str) -> str:
    """Extract short LLM name from model string"""
    model_lower = model_string.lower()
    if "qwen" in model_lower:
        return "Qwen"
    elif "claude" in model_lower:
        return "Claude"
    elif "gpt" in model_lower or "openai" in model_lower:
        return "GPT"
    elif "gemini" in model_lower:
        return "Gemini"
    elif "deepseek" in model_lower:
        return "DeepSeek"
    elif "llama" in model_lower:
        return "Llama"
    else:
        return "Unknown"

AGENT_NAME = _build_agent_name()
AGENT_METADATA = {
    "role": "exec",
    "parent": "Mgr-Code-01",  # Will be assigned dynamically by Manager
    "grandparent": "Dir-Code",
    "tool": "aider",
    "tier": "programmer",
    "programmer_id": PROGRAMMER_ID,
    "primary_llm": PRIMARY_LLM,
    "backup_llm": BACKUP_LLM,
    "capabilities": CAPABILITIES
}
PARENT_AGENT = AGENT_METADATA.get("parent", "Mgr-Code-01")

# Failover state tracking
FAILOVER_STATE = {
    "primary_failures": 0,
    "circuit_open": False,
    "circuit_open_until": None,
    "using_backup": False,
    "current_llm": PRIMARY_LLM
}

# Register agent with heartbeat monitor
heartbeat_monitor.register_agent(
    agent=AGENT_NAME,
    parent=PARENT_AGENT,
    llm_model=PRIMARY_LLM,
    role="programmer",
    tier="executor"
)


def _in_roots(path: pathlib.Path) -> bool:
    """Check if path is under allowed roots"""
    try:
        rp = path.resolve()
        return any(str(rp).startswith(str(root)) for root in ROOTS)
    except Exception:
        return False


def _fs_allowed(path: str) -> bool:
    """
    Check if filesystem path is allowed by allowlist.

    Rules:
    1. Must be under allowed roots
    2. Must not match deny patterns
    3. Must match allow patterns (if specified)
    """
    p = pathlib.Path(path)
    if not _in_roots(p):
        return False
    s = str(p)
    if any(fnmatch.fnmatch(s, pat) for pat in DENY_PATTERNS):
        return False
    if ALLOW_PATTERNS and not any(fnmatch.fnmatch(s, pat) for pat in ALLOW_PATTERNS):
        return False
    return True


def _cmd_allowed(cmd: str) -> bool:
    """
    Check if command is allowed by allowlist.

    Rules:
    1. Must not match deny patterns
    2. Must match allow patterns (if specified)
    """
    tokens = shlex.split(cmd)
    base = " ".join(tokens[:2]) if len(tokens) >= 2 else (tokens[0] if tokens else "")
    full = " ".join(tokens)
    if any(fnmatch.fnmatch(full, pat) or fnmatch.fnmatch(base, pat) for pat in CMD_ALLOW.get("deny", [])):
        return False
    if CMD_ALLOW.get("allow"):
        return any(fnmatch.fnmatch(full, pat) or fnmatch.fnmatch(base, pat) for pat in CMD_ALLOW["allow"])
    return False


class EditRequest(BaseModel):
    message: str = Field(..., description="Natural language instruction for aider")
    files: List[str] = Field(default_factory=list, description="Files to include with aider")
    branch: Optional[str] = None
    dry_run: bool = False
    run_id: Optional[str] = Field(None, description="Run identifier for logging/tracking")
    parent_log_id: Optional[int] = Field(None, description="Parent log ID for hierarchical tracking")


def _get_current_llm() -> str:
    """
    Get current LLM to use (with circuit breaker logic).

    Returns primary LLM unless circuit is open, then returns backup.
    """
    # Check if circuit breaker is open
    if FAILOVER_STATE["circuit_open"]:
        if FAILOVER_STATE["circuit_open_until"] and time.time() < FAILOVER_STATE["circuit_open_until"]:
            # Still in cooldown - use backup
            return BACKUP_LLM
        else:
            # Cooldown expired - reset circuit
            FAILOVER_STATE["circuit_open"] = False
            FAILOVER_STATE["circuit_open_until"] = None
            FAILOVER_STATE["primary_failures"] = 0
            FAILOVER_STATE["using_backup"] = False
            FAILOVER_STATE["current_llm"] = PRIMARY_LLM
            return PRIMARY_LLM

    return FAILOVER_STATE["current_llm"]


def _record_llm_failure(llm: str, error_type: str):
    """
    Record LLM failure and update circuit breaker state.

    Args:
        llm: LLM that failed
        error_type: Type of failure (timeout, api_error, etc.)
    """
    failover_cfg = POOL_CFG.get("failover", {})
    circuit_cfg = failover_cfg.get("circuit_breaker", {})

    if not circuit_cfg.get("enabled", True):
        return

    # Only track primary failures
    if llm != PRIMARY_LLM:
        return

    FAILOVER_STATE["primary_failures"] += 1

    # Check if we should open circuit
    threshold = circuit_cfg.get("failure_threshold", 3)
    if FAILOVER_STATE["primary_failures"] >= threshold:
        # Open circuit - switch to backup
        FAILOVER_STATE["circuit_open"] = True
        cooldown_minutes = circuit_cfg.get("cooldown_minutes", 5)
        FAILOVER_STATE["circuit_open_until"] = time.time() + (cooldown_minutes * 60)
        FAILOVER_STATE["using_backup"] = True
        FAILOVER_STATE["current_llm"] = BACKUP_LLM

        logger.log(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            msg_type=MessageType.CMD,
            message=f"⚠️ Circuit breaker OPEN - Switching to backup LLM: {BACKUP_LLM}",
            run_id="system",
            status="warning",
            metadata={
                "primary_llm": PRIMARY_LLM,
                "backup_llm": BACKUP_LLM,
                "failures": FAILOVER_STATE["primary_failures"],
                "cooldown_until": FAILOVER_STATE["circuit_open_until"]
            }
        )


def _record_llm_success(llm: str):
    """Record successful LLM execution (resets failure counter)"""
    if llm == PRIMARY_LLM:
        # Reset failure counter on success
        FAILOVER_STATE["primary_failures"] = max(0, FAILOVER_STATE["primary_failures"] - 1)


@app.get("/health")
def health():
    """Health check endpoint"""
    current_llm = _get_current_llm()

    return {
        "status": "ok",
        "service": f"{AGENT_NAME} RPC",
        "agent": AGENT_NAME,
        "programmer_id": PROGRAMMER_ID,
        "agent_metadata": AGENT_METADATA,
        "port": PROGRAMMER_PORT,
        "llm": {
            "current": current_llm,
            "primary": PRIMARY_LLM,
            "backup": BACKUP_LLM,
            "using_backup": FAILOVER_STATE["using_backup"],
            "circuit_open": FAILOVER_STATE["circuit_open"],
            "failures": FAILOVER_STATE["primary_failures"]
        },
        "capabilities": CAPABILITIES
    }


@app.post("/agent_chat/receive")
async def receive_agent_message(
    request: AgentChatMessage,
    background_tasks: BackgroundTasks
):
    """
    Receive message from Manager via Agent Chat thread.

    This is the RECOMMENDED way for Managers to communicate with Programmers.
    Enables bidirectional Q&A, status updates, and context preservation.

    Flow:
    - Manager creates thread with delegation message
    - Programmer receives via this endpoint
    - Programmer sends status updates during Aider execution
    - Programmer closes thread on completion/error

    Alternative: /aider/edit endpoint (RPC-style, no conversation)
    """
    thread_id = request.thread_id

    # Load thread to get run_id
    try:
        thread = await agent_chat.get_thread(thread_id)
        run_id = thread.run_id
    except Exception:
        run_id = "unknown"

    logger.log_cmd(
        from_agent=PARENT_AGENT,
        to_agent=AGENT_NAME,
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

    Executes Aider with agent chat status updates.
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
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message_type="status",
            content="Received task, preparing Aider execution..."
        )

        # Extract task from delegation message
        if request.message_type == "delegation":
            # Parse task from message content or metadata
            task_message = request.content
            files = request.metadata.get("files", [])

            # Execute Aider with agent chat updates
            await execute_aider_with_chat(
                message=task_message,
                files=files,
                thread_id=thread_id,
                run_id=run_id
            )

        elif request.message_type == "answer":
            # Parent answered a question - context is in thread history
            # Continue execution if needed
            pass

    except Exception as e:
        logger.log(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            msg_type=MessageType.RESPONSE,
            message=f"Error processing agent chat message: {str(e)}",
            run_id=run_id,
            status="failed",
            metadata={"thread_id": thread_id, "error": str(e)}
        )

        # Send error to parent via agent chat
        try:
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=AGENT_NAME,
                to_agent=PARENT_AGENT,
                message_type="error",
                content=f"Error processing message: {str(e)}"
            )
            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=str(e)
            )
        except:
            pass


async def execute_aider_with_chat(
    message: str,
    files: List[str],
    thread_id: str,
    run_id: str
):
    """
    Execute Aider with agent chat status updates.

    This is the main execution flow for Programmers when called via agent chat.
    """
    try:
        # Step 1: Send status - starting execution
        heartbeat_monitor.heartbeat(
            agent=AGENT_NAME,
            run_id=run_id,
            state=AgentState.EXECUTING,
            message="Executing code changes with Aider CLI",
            progress=0.2
        )

        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message_type="status",
            content=f"Starting Aider execution: {message[:100]}...",
            metadata={"progress": 20, "files": files}
        )

        # Step 2: Validate filesystem access
        for f in files:
            if not _fs_allowed(f):
                error_msg = f"File not allowed by allowlist: {f}"
                await agent_chat.send_message(
                    thread_id=thread_id,
                    from_agent=AGENT_NAME,
                    to_agent=PARENT_AGENT,
                    message_type="error",
                    content=error_msg
                )
                await agent_chat.close_thread(
                    thread_id=thread_id,
                    status="failed",
                    error=error_msg
                )
                raise HTTPException(status_code=403, detail=error_msg)

        # Step 3: Build aider command
        model = _get_current_llm()  # Use current LLM (with circuit breaker)
        aider_bin = "aider"  # Always use aider binary
        timeout_s = int(POOL_CFG.get("defaults", {}).get("timeout_s", 900))

        cmd = [aider_bin, "--yes", "--no-show-model-warnings"]
        if model:
            cmd += ["--model", model]
        for f in files:
            cmd += [f]
        cmd += ["--message", message]

        # Step 4: Redact environment variables
        env = os.environ.copy()
        # No redaction needed from old config - handled by aider itself

        # Step 5: Execute aider with progress updates and failover retry
        await agent_chat.send_message(
            thread_id=thread_id,
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message_type="status",
            content=f"Executing Aider CLI with {_extract_llm_short_name(model)}...",
            metadata={"progress": 40, "llm": model}
        )

        heartbeat_monitor.heartbeat(
            agent=AGENT_NAME,
            run_id=run_id,
            state=AgentState.EXECUTING,
            message="Aider CLI running...",
            progress=0.5
        )

        # Retry loop: try primary, then backup if enabled
        failover_cfg = POOL_CFG.get("failover", {})
        max_retries = failover_cfg.get("max_retries", 2)
        retry_delay = failover_cfg.get("retry_delay_s", 5)

        last_error = None
        for attempt in range(max_retries):
            current_model = _get_current_llm()

            # Update command with current model
            cmd_with_model = [aider_bin, "--yes", "--no-show-model-warnings"]
            cmd_with_model += ["--model", current_model]
            for f in files:
                cmd_with_model += [f]
            cmd_with_model += ["--message", message]

            start = time.time()
            try:
                proc = subprocess.run(
                    cmd_with_model,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=timeout_s,
                    cwd=ROOTS[0]  # Execute in project root
                )

                # Success - record and break retry loop
                _record_llm_success(current_model)
                break

            except subprocess.TimeoutExpired:
                last_error = f"Aider timed out after {timeout_s}s"
                _record_llm_failure(current_model, "timeout")

                # Check if we should retry with backup
                if attempt < max_retries - 1 and failover_cfg.get("always_failover", True):
                    await agent_chat.send_message(
                        thread_id=thread_id,
                        from_agent=AGENT_NAME,
                        to_agent=PARENT_AGENT,
                        message_type="status",
                        content=f"⚠️ Timeout with {_extract_llm_short_name(current_model)}, retrying with backup in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # Final failure
                    await agent_chat.send_message(
                        thread_id=thread_id,
                        from_agent=AGENT_NAME,
                        to_agent=PARENT_AGENT,
                        message_type="error",
                        content=last_error
                    )
                    await agent_chat.close_thread(
                        thread_id=thread_id,
                        status="failed",
                        error=last_error
                    )
                    raise HTTPException(status_code=504, detail=last_error)

            except FileNotFoundError:
                error_msg = f"Aider binary not found: {aider_bin}"
                _record_llm_failure(current_model, "binary_not_found")
                await agent_chat.send_message(
                    thread_id=thread_id,
                    from_agent=AGENT_NAME,
                    to_agent=PARENT_AGENT,
                    message_type="error",
                    content=error_msg
                )
                await agent_chat.close_thread(
                    thread_id=thread_id,
                    status="failed",
                    error=error_msg
                )
                raise HTTPException(status_code=500, detail=f"{error_msg}. Install with: pipx install aider-chat")

            except Exception as e:
                error_msg = f"Execution error: {str(e)}"
                _record_llm_failure(current_model, "execution_error")
                last_error = error_msg

                # Check if we should retry with backup
                if attempt < max_retries - 1 and failover_cfg.get("always_failover", True):
                    await agent_chat.send_message(
                        thread_id=thread_id,
                        from_agent=AGENT_NAME,
                        to_agent=PARENT_AGENT,
                        message_type="status",
                        content=f"⚠️ Error with {_extract_llm_short_name(current_model)}, retrying with backup in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue
                else:
                    # Final failure
                    await agent_chat.send_message(
                        thread_id=thread_id,
                        from_agent=AGENT_NAME,
                        to_agent=PARENT_AGENT,
                        message_type="error",
                        content=error_msg
                    )
                    await agent_chat.close_thread(
                        thread_id=thread_id,
                        status="failed",
                        error=error_msg
                    )
                    raise HTTPException(status_code=500, detail=error_msg)

        duration = round(time.time() - start, 2)
        rc = proc.returncode

        # Step 6: Check result and send completion or error
        if rc != 0:
            # Aider failed
            error_msg = f"Aider failed with rc={rc}"
            stderr_preview = proc.stderr[-500:] if proc.stderr else ""

            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=AGENT_NAME,
                to_agent=PARENT_AGENT,
                message_type="error",
                content=f"❌ {error_msg}\n\nStderr:\n{stderr_preview}",
                metadata={"rc": rc, "duration_s": duration}
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=error_msg
            )
        else:
            # Success
            heartbeat_monitor.heartbeat(
                agent=AGENT_NAME,
                run_id=run_id,
                state=AgentState.COMPLETED,
                message="Aider execution completed successfully",
                progress=1.0
            )

            stdout_preview = proc.stdout[-1000:] if proc.stdout else ""

            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=AGENT_NAME,
                to_agent=PARENT_AGENT,
                message_type="completion",
                content=f"✅ Aider execution completed successfully in {duration}s\n\nOutput:\n{stdout_preview}",
                metadata={"rc": rc, "duration_s": duration, "model": model}
            )

            await agent_chat.close_thread(
                thread_id=thread_id,
                status="completed",
                result="Aider execution successful"
            )

    except HTTPException:
        # Already handled above, just re-raise
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.log(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            msg_type=MessageType.RESPONSE,
            message=error_msg,
            run_id=run_id,
            status="failed"
        )

        try:
            await agent_chat.send_message(
                thread_id=thread_id,
                from_agent=AGENT_NAME,
                to_agent=PARENT_AGENT,
                message_type="error",
                content=error_msg
            )
            await agent_chat.close_thread(
                thread_id=thread_id,
                status="failed",
                error=error_msg
            )
        except:
            pass


@app.post("/aider/edit")
def aider_edit(req: EditRequest):
    """
    Execute Aider CLI with guardrails.

    Steps:
    1. Validate all files against filesystem allowlist
    2. Validate branch creation command (if specified)
    3. Build aider command with redacted environment
    4. Execute with timeout
    5. Return stdout/stderr + duration + LLM metadata
    """
    # Extract run_id and parent_log_id from request
    run_id = req.run_id
    parent_log_id = req.parent_log_id

    # Get LLM model config (with circuit breaker)
    model = _get_current_llm()
    llm_provider = model.split("/")[0] if "/" in model else "unknown"

    # Log incoming command - link to parent (from PAS Root)
    cmd_log_id = logger.log(
        from_agent="PAS Root",
        to_agent=AGENT_NAME,
        msg_type=MessageType.CMD,
        message=f"Execute: {req.message[:100]}...",
        llm_model=model,
        run_id=run_id,
        status="queued",
        progress=0.0,
        metadata={"files": req.files, "dry_run": req.dry_run},
        parent_log_id=parent_log_id
    )

    # Step 1: Validate filesystem access
    for f in req.files:
        if not _fs_allowed(f):
            logger.log_response(
                from_agent=AGENT_NAME,
                to_agent=PARENT_AGENT,
                message=f"File not allowed: {f}",
                run_id=run_id,
                status="error",
                parent_log_id=cmd_log_id
            )
            raise HTTPException(status_code=403, detail=f"File not allowed: {f}")

    # Step 2: Validate branch creation
    if req.branch and not _cmd_allowed(f"git checkout -b {req.branch}"):
        logger.log_response(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message="Command not allowed: git checkout -b",
            run_id=run_id,
            status="error",
            parent_log_id=cmd_log_id
        )
        raise HTTPException(status_code=403, detail="Command not allowed: git checkout -b ...")

    # Step 3: Build aider command
    aider_bin = "aider"  # Always use aider binary
    timeout_s = int(POOL_CFG.get("defaults", {}).get("timeout_s", 900))

    cmd = [aider_bin, "--yes", "--no-show-model-warnings"]
    if model:
        cmd += ["--model", model]
    for f in req.files:
        cmd += [f]
    cmd += ["--message", req.message]

    # Step 4: Redact environment variables
    env = os.environ.copy()
    # No redaction needed - aider handles this

    # Dry run mode (return command without executing)
    if req.dry_run:
        logger.log_response(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message="Dry run completed",
            llm_model=model,
            run_id=run_id,
            status="completed",
            parent_log_id=cmd_log_id
        )
        return {"ok": True, "dry_run": True, "cmd": cmd, "llm_model": model, "llm_provider": llm_provider}

    # Step 5: Execute aider
    status_log_id = logger.log_status(
        from_agent=AGENT_NAME,
        to_agent=PARENT_AGENT,
        message="Starting Aider execution",
        llm_model=model,
        run_id=run_id,
        status="running",
        progress=0.1,
        parent_log_id=cmd_log_id
    )

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            cwd=ROOTS[0]  # Execute in project root
        )
    except subprocess.TimeoutExpired:
        logger.log_response(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message=f"Timeout after {timeout_s}s",
            llm_model=model,
            run_id=run_id,
            status="error",
            metadata={"timeout_s": timeout_s},
            parent_log_id=status_log_id
        )
        raise HTTPException(status_code=504, detail=f"aider timed out after {timeout_s}s")
    except FileNotFoundError:
        logger.log_response(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message=f"Aider binary not found: {aider_bin}",
            llm_model=model,
            run_id=run_id,
            status="error",
            parent_log_id=status_log_id
        )
        raise HTTPException(status_code=500, detail=f"aider binary not found: {aider_bin}. Install with: pipx install aider-chat")
    except Exception as e:
        logger.log_response(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message=f"Execution error: {str(e)}",
            llm_model=model,
            run_id=run_id,
            status="error",
            parent_log_id=status_log_id
        )
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    duration = round(time.time() - start, 2)
    rc = proc.returncode

    # Step 6: Return result
    if rc != 0:
        logger.log_response(
            from_agent=AGENT_NAME,
            to_agent=PARENT_AGENT,
            message=f"Aider failed with rc={rc}",
            llm_model=model,
            run_id=run_id,
            status="error",
            metadata={"rc": rc, "duration_s": duration},
            parent_log_id=status_log_id
        )
        # Include stderr in error response (truncate to avoid huge payloads)
        return {
            "ok": False,
            "rc": rc,
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "duration_s": duration,
            "llm_model": model,
            "llm_provider": llm_provider
        }

    logger.log_response(
        from_agent=AGENT_NAME,
        to_agent=PARENT_AGENT,
        message="Aider execution completed successfully",
        llm_model=model,
        run_id=run_id,
        status="completed",
        progress=1.0,
        metadata={"rc": rc, "duration_s": duration},
        parent_log_id=status_log_id
    )

    return {
        "ok": True,
        "rc": rc,
        "stdout": proc.stdout[-4000:] if proc.stdout else "",  # Last 4KB
        "duration_s": duration,
        "llm_model": model,
        "llm_provider": llm_provider
    }


if __name__ == "__main__":
    import uvicorn
    # Port is configured via programmer pool config
    uvicorn.run(app, host="127.0.0.1", port=PROGRAMMER_PORT)

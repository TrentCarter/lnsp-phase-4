#!/usr/bin/env python3
"""
Programmer-001 Service - LLM-Agnostic Code Execution Agent

Port: 6151
Tier: 5 (Programmer)
LLM: Runtime-selectable (local Ollama or API providers)

Responsibilities:
- Execute atomic code tasks using Aider CLI
- Support multiple LLM providers (Ollama, Anthropic, OpenAI, Google)
- Track comprehensive metrics (tokens, cost, time, quality)
- Generate receipts for cost tracking and auditing
- Enforce filesystem and command allowlists
- Integrate with Heartbeat Monitor and Communication Logger

Architecture:
- FastAPI HTTP service (like Managers/Directors)
- Runtime LLM selection (not hardcoded at startup)
- Aider CLI wrapper with guardrails
- Comprehensive metrics tracking and receipt generation
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
import subprocess
import shlex
import json
import yaml
from datetime import datetime
import uuid
import fnmatch

# PAS common services
from services.common.heartbeat import get_monitor, AgentState
from services.common.comms_logger import get_logger, MessageType

# Receipt tracking (reuse from Aider RPC)
from tools.aider_rpc.receipts import (
    AiderReceipt, TokenUsage, CostEstimate, ProviderSnapshot, KPIMetrics,
    estimate_cost, now_iso
)

app = FastAPI(title="Programmer-001", version="1.0.0")

# Initialize systems
heartbeat_monitor = get_monitor()
logger = get_logger()

# Service configuration
SERVICE_NAME = "Programmer-009"
SERVICE_PORT = 6159
AGENT_ID = "Prog-009"
PARENT_AGENT = "Mgr-Data-01"  # Will be set by Manager at runtime
TIER = "programmer"
LANE = "Code"

# Load allowlists
CONFIG_DIR = Path(__file__).parent.parent.parent.parent / "configs" / "pas"
FS_ALLOW = yaml.safe_load(open(CONFIG_DIR / "fs_allowlist.yaml"))
CMD_ALLOW = yaml.safe_load(open(CONFIG_DIR / "cmd_allowlist.yaml"))

ROOTS = [Path(p).resolve() for p in FS_ALLOW["roots"]]
DENY_PATTERNS = FS_ALLOW.get("deny", [])
ALLOW_PATTERNS = FS_ALLOW.get("allow", [])

# Receipts directory
RECEIPTS_DIR = Path("artifacts/programmer_receipts")
RECEIPTS_DIR.mkdir(parents=True, exist_ok=True)

# Register Programmer agent with Heartbeat Monitor
heartbeat_monitor.register_agent(
    agent=AGENT_ID,
    parent=PARENT_AGENT,
    llm_model="runtime-selectable",
    role="programmer",
    tier=TIER
)

# In-memory job tracking
JOBS: Dict[str, Dict[str, Any]] = {}


# === Pydantic Models ===

class LLMParams(BaseModel):
    """LLM inference parameters (optional overrides)"""
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=128000)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1)


class ExecuteRequest(BaseModel):
    """Programmer task execution request"""
    task: str = Field(..., description="Natural language task description")
    files: List[str] = Field(default_factory=list, description="Files to operate on")
    llm_provider: str = Field("ollama", description="LLM provider: ollama, anthropic, openai, google")
    llm_model: str = Field("qwen2.5-coder:7b-instruct", description="Model identifier")
    llm_params: Optional[LLMParams] = Field(None, description="LLM parameter overrides")
    run_id: Optional[str] = Field(None, description="Run identifier for tracking")
    job_id: Optional[str] = Field(None, description="Job card ID")
    parent_log_id: Optional[int] = Field(None, description="Parent log ID for hierarchical tracking")
    budget_usd: Optional[float] = Field(None, ge=0, description="Budget limit (USD)")
    timeout_s: int = Field(300, ge=10, le=3600, description="Timeout in seconds")
    dry_run: bool = Field(False, description="Dry run mode (no actual execution)")


class ExecuteResponse(BaseModel):
    """Programmer task execution response"""
    status: str  # success, error, timeout, budget_exceeded
    run_id: str
    job_id: Optional[str]
    artifacts: List[str] = Field(default_factory=list)
    metrics: Dict[str, Any]
    receipt_path: str
    error: Optional[str] = None


class StatusResponse(BaseModel):
    """Job status response"""
    job_id: str
    status: str
    progress: float = Field(ge=0.0, le=1.0)
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


# === Filesystem & Command Allowlist Enforcement ===

def _in_roots(path: Path) -> bool:
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
    p = Path(path)
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


# === LLM Provider Configuration ===

def get_llm_config(provider: str, model: str, params: Optional[LLMParams]) -> Dict[str, Any]:
    """
    Build LLM configuration for Aider CLI based on provider and model.

    Returns:
        Dictionary with Aider CLI arguments
    """
    config = {}

    if provider == "ollama":
        config["model"] = f"ollama/{model}"
        config["api_base"] = os.getenv("OLLAMA_URL", "http://localhost:11434")

    elif provider == "anthropic":
        config["model"] = f"anthropic/{model}"
        config["api_key"] = os.getenv("ANTHROPIC_API_KEY")
        if not config["api_key"]:
            raise ValueError("ANTHROPIC_API_KEY not set")

    elif provider == "openai":
        config["model"] = f"openai/{model}"
        config["api_key"] = os.getenv("OPENAI_API_KEY")
        if not config["api_key"]:
            raise ValueError("OPENAI_API_KEY not set")

    elif provider == "google":
        config["model"] = f"google/{model}"
        config["api_key"] = os.getenv("GEMINI_API_KEY")
        if not config["api_key"]:
            raise ValueError("GEMINI_API_KEY not set")
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Add parameter overrides
    if params:
        if params.temperature is not None:
            config["temperature"] = params.temperature
        if params.max_tokens is not None:
            config["max_tokens"] = params.max_tokens

    return config


def build_aider_command(task: str, files: List[str], llm_config: Dict[str, Any], dry_run: bool = False) -> List[str]:
    """
    Build Aider CLI command with LLM configuration.

    Returns:
        List of command arguments for subprocess
    """
    cmd = ["aider"]

    # Add model
    if "model" in llm_config:
        cmd.extend(["--model", llm_config["model"]])

    # Add API configuration
    if "api_key" in llm_config:
        # Set via environment, not command line (security)
        pass

    if "api_base" in llm_config:
        cmd.extend(["--api-base", llm_config["api_base"]])

    # Add parameters
    if "temperature" in llm_config:
        cmd.extend(["--temperature", str(llm_config["temperature"])])

    if "max_tokens" in llm_config:
        cmd.extend(["--max-tokens", str(llm_config["max_tokens"])])

    # Add files
    for f in files:
        cmd.append(f)

    # Add task as message
    cmd.extend(["--message", task])

    # Add flags
    cmd.append("--yes")  # Auto-confirm
    cmd.append("--no-auto-commits")  # Don't auto-commit (Manager handles commits)

    if dry_run:
        cmd.append("--dry-run")

    return cmd


# === Task Execution ===

def execute_aider(
    task: str,
    files: List[str],
    llm_provider: str,
    llm_model: str,
    llm_params: Optional[LLMParams],
    timeout_s: int,
    dry_run: bool
) -> Dict[str, Any]:
    """
    Execute Aider CLI with specified LLM configuration.

    Returns:
        Dictionary with execution results and metrics
    """
    start_time = time.time()

    # Build LLM config
    try:
        llm_config = get_llm_config(llm_provider, llm_model, llm_params)
    except ValueError as e:
        return {
            "status": "error",
            "error": str(e),
            "duration_s": time.time() - start_time
        }

    # Validate files
    for f in files:
        if not _fs_allowed(f):
            return {
                "status": "error",
                "error": f"File not allowed: {f}",
                "duration_s": time.time() - start_time
            }

    # Build command
    cmd = build_aider_command(task, files, llm_config, dry_run)

    # Prepare environment (for API keys)
    env = os.environ.copy()
    if "api_key" in llm_config:
        if llm_provider == "anthropic":
            env["ANTHROPIC_API_KEY"] = llm_config["api_key"]
        elif llm_provider == "openai":
            env["OPENAI_API_KEY"] = llm_config["api_key"]
        elif llm_provider == "google":
            env["GEMINI_API_KEY"] = llm_config["api_key"]

    # Execute Aider
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(Path.cwd())
        )

        duration_s = time.time() - start_time

        # Parse Aider output for metrics
        # TODO: Parse token usage from Aider output
        # For now, estimate based on file sizes
        tokens = TokenUsage()
        tokens.input_tokens = sum(len(Path(f).read_text()) // 4 for f in files if Path(f).exists())
        tokens.output_tokens = len(result.stdout) // 4
        tokens.update_total()

        return {
            "status": "success" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "duration_s": duration_s,
            "tokens": tokens,
            "llm_model": llm_model
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error": f"Execution exceeded timeout of {timeout_s}s",
            "duration_s": timeout_s
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "duration_s": time.time() - start_time
        }


# === API Endpoints ===

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "agent": AGENT_ID,
        "port": SERVICE_PORT,
        "tier": TIER,
        "lane": LANE,
        "llm_mode": "runtime-selectable",
        "supported_providers": ["ollama", "anthropic", "openai", "google"]
    }


@app.post("/execute", response_model=ExecuteResponse)
async def execute(request: ExecuteRequest, background_tasks: BackgroundTasks):
    """
    Execute code task with specified LLM.

    This endpoint:
    1. Validates request and files
    2. Executes Aider CLI with LLM configuration
    3. Tracks comprehensive metrics (tokens, cost, time, quality)
    4. Generates receipt for auditing
    5. Returns execution results
    """
    # Generate run_id if not provided
    run_id = request.run_id or str(uuid.uuid4())
    job_id = request.job_id or f"job-{run_id[:8]}"

    # Update heartbeat state
    heartbeat_monitor.update_state(AGENT_ID, AgentState.BUSY, run_id=run_id)

    # Log task start
    log_id = logger.log(
        from_agent=AGENT_ID,
        to_agent=PARENT_AGENT,
        msg_type=MessageType.TASK_START,
        message=f"Executing: {request.task[:100]}",
        run_id=run_id,
        status="in_progress"
    )

    # Initialize receipt
    receipt = AiderReceipt(
        run_id=run_id,
        job_id=job_id,
        task=request.task[:200],
        status="in_progress",
        created_at=now_iso()
    )

    # Set provider snapshot
    receipt.provider.provider = request.llm_provider
    receipt.provider.model = request.llm_model
    if request.llm_params:
        if request.llm_params.temperature is not None:
            receipt.provider.temperature = request.llm_params.temperature
        if request.llm_params.max_tokens is not None:
            receipt.provider.max_tokens = request.llm_params.max_tokens

    # Execute task
    result = execute_aider(
        task=request.task,
        files=request.files,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        llm_params=request.llm_params,
        timeout_s=request.timeout_s,
        dry_run=request.dry_run
    )

    # Update receipt with results
    receipt.status = result["status"]
    receipt.completed_at = now_iso()

    if "tokens" in result:
        receipt.usage = result["tokens"]
        receipt.cost = estimate_cost(receipt.usage, result["llm_model"])

        # Check budget
        if request.budget_usd and receipt.cost.total_cost > request.budget_usd:
            receipt.status = "budget_exceeded"
            result["status"] = "budget_exceeded"
            result["error"] = f"Cost ${receipt.cost.total_cost:.4f} exceeds budget ${request.budget_usd:.4f}"

    if "duration_s" in result:
        receipt.kpis.duration_seconds = result["duration_s"]

    # TODO: Parse git diff for quality metrics
    receipt.kpis.files_changed = len(request.files)

    # Save receipt
    receipt_path = RECEIPTS_DIR / f"{run_id}.json"
    receipt.save(receipt_path)

    # Log completion
    logger.log(
        from_agent=AGENT_ID,
        to_agent=PARENT_AGENT,
        msg_type=MessageType.TASK_COMPLETE if receipt.status == "success" else MessageType.ERROR,
        message=f"Status: {receipt.status}, Tokens: {receipt.usage.total_tokens}, Cost: ${receipt.cost.total_cost:.4f}",
        run_id=run_id,
        status=receipt.status
    )

    # Update heartbeat state
    heartbeat_monitor.update_state(AGENT_ID, AgentState.IDLE)

    # Prepare response
    return ExecuteResponse(
        status=receipt.status,
        run_id=run_id,
        job_id=job_id,
        artifacts=[],  # TODO: Collect artifact paths
        metrics={
            "tokens": {
                "input": receipt.usage.input_tokens,
                "output": receipt.usage.output_tokens,
                "total": receipt.usage.total_tokens
            },
            "cost_usd": receipt.cost.total_cost,
            "duration_s": receipt.kpis.duration_seconds,
            "files_changed": receipt.kpis.files_changed,
            "lines_added": receipt.kpis.lines_added,
            "lines_removed": receipt.kpis.lines_removed
        },
        receipt_path=str(receipt_path),
        error=result.get("error")
    )


@app.get("/status/{job_id}", response_model=StatusResponse)
def get_status(job_id: str):
    """Get job status"""
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = JOBS[job_id]
    return StatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        metrics=job.get("metrics"),
        error=job.get("error")
    )


@app.get("/metrics")
def get_metrics():
    """Get service metrics"""
    # Calculate aggregate metrics from receipts
    receipts = []
    for receipt_file in RECEIPTS_DIR.glob("*.json"):
        try:
            receipt = AiderReceipt.load(receipt_file)
            receipts.append(receipt)
        except Exception:
            continue

    total_tasks = len(receipts)
    total_cost = sum(r.cost.total_cost for r in receipts)
    total_tokens = sum(r.usage.total_tokens for r in receipts)
    total_duration = sum(r.kpis.duration_seconds for r in receipts)

    return {
        "agent": AGENT_ID,
        "total_tasks": total_tasks,
        "total_cost_usd": round(total_cost, 4),
        "total_tokens": total_tokens,
        "total_duration_s": round(total_duration, 2),
        "avg_cost_per_task": round(total_cost / total_tasks, 4) if total_tasks > 0 else 0,
        "avg_tokens_per_task": round(total_tokens / total_tasks, 0) if total_tasks > 0 else 0
    }


if __name__ == "__main__":
    import uvicorn
    print(f"Starting {SERVICE_NAME} on port {SERVICE_PORT}...")
    print(f"Agent ID: {AGENT_ID}")
    print(f"LLM Mode: Runtime-selectable")
    print(f"Supported Providers: ollama, anthropic, openai, google")
    uvicorn.run(app, host="127.0.0.1", port=SERVICE_PORT)

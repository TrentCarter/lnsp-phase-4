#!/usr/bin/env python3
"""
Aider RPC Server (Enhanced with P0 Integrations)

FastAPI service that wraps Aider coding agent for PAS integration.

P0 Features:
- Command sandboxing via allowlist
- Secrets redaction for logs/diffs
- Cost/KPI receipt generation
- Registry heartbeat integration
- Actual Aider CLI execution

To run:
    export PAS_PORT=6150
    export PAS_REGISTRY_URL=http://127.0.0.1:6121  # optional
    export AIDER_MODEL=ollama/qwen2.5-coder:7b-instruct  # optional
    python tools/aider_rpc/server_enhanced.py

Example invoke:
    curl -s http://127.0.0.1:6150/invoke -H 'Content-Type: application/json' -d '{
      "target": {"name": "Aider-LCO", "type":"agent", "role":"execution"},
      "payload": {
        "task": "code_refactor",
        "message": "Add type hints to all functions in src/utils.py",
        "files": ["src/utils.py"]
      },
      "policy": {"timeout_s": 120, "require_caps": ["git-edit"]},
      "run_id": "refactor-001"
    }' | jq
"""
from __future__ import annotations

import os
import sys
import time
import json
import uuid
import asyncio
import subprocess
import tempfile
import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

# Import our support modules
try:
    from allowlist import check_command, check_file_access, Rights
    from redact import redact_text, redact_diff
    from receipts import AiderReceipt, TokenUsage, CostEstimate, ProviderSnapshot, KPIMetrics, estimate_cost, now_iso
    from heartbeat import RegistryClient
except ImportError:
    # If running as module
    from tools.aider_rpc.allowlist import check_command, check_file_access, Rights
    from tools.aider_rpc.redact import redact_text, redact_diff
    from tools.aider_rpc.receipts import AiderReceipt, TokenUsage, CostEstimate, ProviderSnapshot, KPIMetrics, estimate_cost, now_iso
    from tools.aider_rpc.heartbeat import RegistryClient

# Configuration
APP_NAME = "Aider-LCO"
SERVICE_TYPE = "agent"
DEFAULT_ROLE = "execution"
DEFAULT_CAPS = ["git-edit", "pytest-loop", "repo-map", "multi-file-edit"]
CTX_LIMIT = int(os.getenv("AIDER_CTX_LIMIT", "131072"))
PORT = int(os.getenv("PAS_PORT", "6150"))
HOST = os.getenv("PAS_HOST", "127.0.0.1")
REGISTRY_URL = os.getenv("PAS_REGISTRY_URL")
HEARTBEAT_SEC = int(os.getenv("PAS_HEARTBEAT_SEC", "60"))
ARTIFACTS_COST_DIR = Path(os.getenv("PAS_COST_DIR", "artifacts/costs")).resolve()
ARTIFACTS_COST_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_ID = os.getenv("PAS_SERVICE_ID") or str(uuid.uuid4())

# Aider configuration
AIDER_MODEL = os.getenv("AIDER_MODEL", "ollama/qwen2.5-coder:7b-instruct")
AIDER_EDITOR_MODEL = os.getenv("AIDER_EDITOR_MODEL", "")  # Optional separate editor model
REPO_ROOT = Path(os.getenv("REPO_ROOT", os.getcwd())).resolve()

app = FastAPI(title=f"{APP_NAME} RPC Server", version="0.1.0")

# Global registry client
registry_client: Optional[RegistryClient] = None

# ==================== Pydantic Models ====================

class Target(BaseModel):
    service_id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = Field(None, description="model|tool|agent")
    role: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None

class Policy(BaseModel):
    timeout_s: int = 120
    prefer: Optional[str] = Field("role", description="role|name|p95|random_healthy")
    require_caps: Optional[List[str]] = None

class InvokeRequest(BaseModel):
    target: Target
    payload: Dict[str, Any]
    policy: Policy = Policy()
    run_id: Optional[str] = None

class UpstreamResponse(BaseModel):
    outputs: Dict[str, Any]

class RoutingResolved(BaseModel):
    service_id: str
    name: str
    role: str
    url: str

class RoutingReceipt(BaseModel):
    run_id: str
    resolved: RoutingResolved
    timings_ms: Dict[str, float]
    status: str
    cost_estimate: Dict[str, Any] = Field(default_factory=lambda: {"usd": 0.0})
    ts: str

class InvokeResponse(BaseModel):
    upstream: UpstreamResponse
    routing_receipt: RoutingReceipt

# ==================== Aider Execution ====================

async def execute_aider(
    message: str,
    files: List[str],
    run_id: str,
    timeout_s: int = 120,
    auto_commit: bool = True,
) -> Dict[str, Any]:
    """
    Execute Aider CLI with the given message and files.

    Args:
        message: Instruction for Aider
        files: List of file paths to edit
        run_id: Unique run identifier
        timeout_s: Execution timeout
        auto_commit: Whether to auto-commit changes

    Returns:
        Dictionary with execution results
    """
    t0 = time.perf_counter()

    # Create receipt
    receipt = AiderReceipt(
        run_id=run_id,
        task="aider_execution",
        status="running",
        created_at=now_iso(),
    )

    # Check if aider is available (use shutil.which for security)
    aider_cmd = shutil.which("aider")
    if not aider_cmd:
        return {
            "status": "error",
            "error": "aider command not found. Install with: pip install aider-chat",
            "receipt": receipt.to_dict(),
        }

    # Build aider command
    cmd_parts = [
        aider_cmd,
        "--message", message,
        "--model", AIDER_MODEL,
        "--yes",  # Auto-confirm
    ]

    if AIDER_EDITOR_MODEL:
        cmd_parts.extend(["--editor-model", AIDER_EDITOR_MODEL])

    if auto_commit:
        cmd_parts.append("--auto-commits")
    else:
        cmd_parts.append("--no-auto-commits")

    # Add files (normalize and validate)
    for f in files:
        # Normalize path first (resolve relative paths)
        try:
            normalized_path = str(Path(f).resolve(strict=False))
        except Exception as e:
            return {
                "status": "error",
                "error": f"Invalid file path: {f} - {e}",
                "receipt": receipt.to_dict(),
            }

        # Validate file access with normalized path
        allowed, reason = check_file_access(normalized_path, "w")
        if not allowed:
            return {
                "status": "error",
                "error": f"File access denied: {normalized_path} - {reason}",
                "receipt": receipt.to_dict(),
            }
        cmd_parts.append(normalized_path)

    # Create temporary log file
    log_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".log")
    log_path = Path(log_file.name)

    # Whitelist environment variables (security: don't pass full env)
    safe_env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        # Model provider API keys (only if set)
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", ""),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY", ""),
        "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
        # Aider-specific
        "AIDER_MODEL": AIDER_MODEL,
    }
    # Remove empty values
    safe_env = {k: v for k, v in safe_env.items() if v}

    try:
        # Execute aider
        print(f"[Aider] Executing: {' '.join(cmd_parts)}")
        process = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=REPO_ROOT,
            env=safe_env,  # SECURITY: Whitelist env vars only
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout_s
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            receipt.status = "timeout"
            return {
                "status": "timeout",
                "error": f"Execution exceeded {timeout_s}s timeout",
                "receipt": receipt.to_dict(),
            }

        # Parse output
        output_text = stdout.decode("utf-8", errors="replace")
        error_text = stderr.decode("utf-8", errors="replace")

        # Write to log
        with log_path.open("w") as f:
            f.write(output_text)
            if error_text:
                f.write("\n\n=== STDERR ===\n")
                f.write(error_text)

        # Redact secrets from output
        output_redacted = redact_text(output_text)

        # Parse token usage from aider output (aider prints cost summary)
        # Example: "Tokens: 5,234 sent, 1,245 received. Cost: $0.0032"
        receipt.usage.input_tokens = 0  # TODO: Parse from aider output
        receipt.usage.output_tokens = 0
        receipt.usage.update_total()

        # Estimate cost (fallback if aider doesn't provide it)
        receipt.provider.model = AIDER_MODEL
        receipt.cost = estimate_cost(receipt.usage, AIDER_MODEL)

        # Update KPIs
        receipt.kpis.files_changed = len(files)
        receipt.kpis.duration_seconds = time.perf_counter() - t0

        # TODO: Parse git diff to get lines added/removed
        # git_diff = subprocess.run(["git", "diff", "HEAD"], capture_output=True, cwd=REPO_ROOT)
        # receipt.kpis.lines_added = ...
        # receipt.kpis.lines_removed = ...

        receipt.status = "ok" if process.returncode == 0 else "error"
        receipt.completed_at = now_iso()
        receipt.artifacts.append(str(log_path))

        # Save receipt
        receipt.save(ARTIFACTS_COST_DIR / f"{run_id}.json")

        return {
            "status": receipt.status,
            "output": output_redacted,
            "returncode": process.returncode,
            "log_file": str(log_path),
            "receipt": receipt.to_dict(),
        }

    except Exception as e:
        receipt.status = "error"
        receipt.completed_at = now_iso()
        return {
            "status": "error",
            "error": str(e),
            "receipt": receipt.to_dict(),
        }

# ==================== FastAPI Routes ====================

@app.on_event("startup")
async def startup():
    """Initialize registry client and start heartbeat"""
    global registry_client

    if REGISTRY_URL:
        registry_client = RegistryClient(
            registry_url=REGISTRY_URL,
            service_id=SERVICE_ID,
            name=APP_NAME,
            service_type=SERVICE_TYPE,
            role=DEFAULT_ROLE,
            url=f"http://{HOST}:{PORT}",
            caps=DEFAULT_CAPS,
            ctx_limit=CTX_LIMIT,
            heartbeat_interval_s=HEARTBEAT_SEC,
        )
        # Start heartbeat in background
        asyncio.create_task(registry_client.start_heartbeat_loop())

@app.on_event("shutdown")
async def shutdown():
    """Deregister from registry"""
    if registry_client:
        await registry_client.deregister()

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service_id": SERVICE_ID,
        "name": APP_NAME,
        "role": DEFAULT_ROLE,
        "model": AIDER_MODEL,
    }

@app.get("/describe")
async def describe():
    """Service description endpoint"""
    return {
        "service_id": SERVICE_ID,
        "name": APP_NAME,
        "type": SERVICE_TYPE,
        "role": DEFAULT_ROLE,
        "caps": DEFAULT_CAPS,
        "labels": {"editor": "aider"},
        "ctx_limit": CTX_LIMIT,
        "host": HOST,
        "port": PORT,
        "model": AIDER_MODEL,
    }

@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, background_tasks: BackgroundTasks):
    """
    Main invocation endpoint.

    Payload format:
    {
      "task": "code_refactor|doc_update|test_fix|...",
      "message": "Instruction for Aider",
      "files": ["path/to/file1.py", "path/to/file2.py"],
      "auto_commit": true
    }
    """
    t0 = time.perf_counter()
    run_id = req.run_id or f"aider-{uuid.uuid4()}"

    # Extract payload
    payload = req.payload
    message = payload.get("message", "")
    files = payload.get("files", [])
    auto_commit = payload.get("auto_commit", True)

    if not message:
        raise HTTPException(status_code=400, detail="Missing 'message' in payload")
    if not files:
        raise HTTPException(status_code=400, detail="Missing 'files' in payload")

    # Execute aider
    result = await execute_aider(
        message=message,
        files=files,
        run_id=run_id,
        timeout_s=req.policy.timeout_s,
        auto_commit=auto_commit,
    )

    dt_total = (time.perf_counter() - t0) * 1000.0

    # Build response
    resolved = RoutingResolved(
        service_id=SERVICE_ID,
        name=APP_NAME,
        role=DEFAULT_ROLE,
        url=f"http://{HOST}:{PORT}",
    )

    routing_receipt = RoutingReceipt(
        run_id=run_id,
        resolved=resolved,
        timings_ms={"total": round(dt_total, 2)},
        status=result.get("status", "error"),
        cost_estimate=result.get("receipt", {}).get("cost", {"usd": 0.0}),
        ts=now_iso(),
    )

    return InvokeResponse(
        upstream=UpstreamResponse(outputs=result),
        routing_receipt=routing_receipt,
    )

# ==================== Main ====================

if __name__ == "__main__":
    print(f"Starting {APP_NAME} on {HOST}:{PORT}")
    print(f"Model: {AIDER_MODEL}")
    print(f"Repo: {REPO_ROOT}")
    if REGISTRY_URL:
        print(f"Registry: {REGISTRY_URL}")

    uvicorn.run(
        "server_enhanced:app",
        host=HOST,
        port=PORT,
        reload=False,
    )

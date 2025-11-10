#!/usr/bin/env python3
"""
Aider-LCO RPC Server (P0 Production Scaffold)
FastAPI service that wraps Aider CLI with guardrails for PAS integration.

Features:
- Filesystem allowlist enforcement
- Command allowlist enforcement
- Secrets redaction
- Timeout enforcement
- Subprocess isolation

Port: 6130
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import yaml
import fnmatch
import subprocess
import shlex
import os
import pathlib
import time

app = FastAPI(title="Aider-LCO RPC", version="1.0")

# Load configurations
FS_ALLOW = yaml.safe_load(open("configs/pas/fs_allowlist.yaml"))
CMD_ALLOW = yaml.safe_load(open("configs/pas/cmd_allowlist.yaml"))
AIDER_CFG = yaml.safe_load(open("configs/pas/aider.yaml"))

ROOTS = [pathlib.Path(p).resolve() for p in FS_ALLOW["roots"]]
DENY_PATTERNS = FS_ALLOW.get("deny", [])
ALLOW_PATTERNS = FS_ALLOW.get("allow", [])


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


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "Aider-LCO RPC",
        "port": 6130,
        "aider_model": AIDER_CFG.get("model", {}).get("primary", "unknown")
    }


@app.post("/aider/edit")
def aider_edit(req: EditRequest):
    """
    Execute Aider CLI with guardrails.

    Steps:
    1. Validate all files against filesystem allowlist
    2. Validate branch creation command (if specified)
    3. Build aider command with redacted environment
    4. Execute with timeout
    5. Return stdout/stderr + duration
    """
    # Step 1: Validate filesystem access
    for f in req.files:
        if not _fs_allowed(f):
            raise HTTPException(status_code=403, detail=f"File not allowed: {f}")

    # Step 2: Validate branch creation
    if req.branch and not _cmd_allowed(f"git checkout -b {req.branch}"):
        raise HTTPException(status_code=403, detail="Command not allowed: git checkout -b ...")

    # Step 3: Build aider command
    aider_bin = AIDER_CFG["model"].get("primary", "aider")  # Use model.primary or fallback
    if "aider_bin" in AIDER_CFG:
        aider_bin = AIDER_CFG["aider_bin"]
    else:
        # Default to "aider" binary
        aider_bin = "aider"

    model = AIDER_CFG.get("model", {}).get("primary", "claude-3-5-sonnet-20241022")
    timeout_s = int(AIDER_CFG.get("timeout_s", 900))

    cmd = [aider_bin, "--yes"]
    if model:
        cmd += ["--model", model]
    for f in req.files:
        cmd += [f]
    cmd += ["--message", req.message]

    # Step 4: Redact environment variables
    env = os.environ.copy()
    for k in AIDER_CFG.get("redact_env", []):
        env.pop(k, None)

    # Dry run mode (return command without executing)
    if req.dry_run:
        return {"ok": True, "dry_run": True, "cmd": cmd}

    # Step 5: Execute aider
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
        raise HTTPException(status_code=504, detail=f"aider timed out after {timeout_s}s")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"aider binary not found: {aider_bin}. Install with: pipx install aider-chat")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(e)}")

    duration = round(time.time() - start, 2)
    rc = proc.returncode

    # Step 6: Return result
    if rc != 0:
        # Include stderr in error response (truncate to avoid huge payloads)
        return {
            "ok": False,
            "rc": rc,
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
            "stdout": proc.stdout[-2000:] if proc.stdout else "",
            "duration_s": duration
        }

    return {
        "ok": True,
        "rc": rc,
        "stdout": proc.stdout[-4000:] if proc.stdout else "",  # Last 4KB
        "duration_s": duration
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6130)

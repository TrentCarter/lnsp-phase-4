#!/usr/bin/env python3
"""
Aider RPC Scaffold (PAS wrapper)

FastAPI service that exposes a thin, uniform API so PAS Gateway/Registry
can route jobs to an Aider-backed coding agent.

Features (scaffold):
- /health, /describe
- /invoke (accepts a Gateway-like request shape)
- Routing receipts persisted to artifacts/costs/<run_id>.json
- Optional self-register + heartbeat to PAS Registry (env PAS_REGISTRY_URL)
- Token/context budget hints (non-enforcing here; Token Governor enforces globally)

To run:
    export PAS_PORT=6150
    python tools/aider_rpc/server.py

Example invoke:
    curl -s http://127.0.0.1:${PAS_PORT:-6150}/invoke -H 'Content-Type: application/json' -d '{
      "target": {"name": "Aider-LCO", "type":"agent", "role":"execution"},
      "payload": {"command": "doc_update", "paths": ["docs/**/*.md"], "style": "active-voice"},
      "policy": {"timeout_s": 120, "require_caps": ["git-edit","pytest-loop"]},
      "run_id": "demo-run-001"
    }' | jq

NOTE: This scaffold does NOT actually invoke the aider CLI yet.
Hook points are marked with TODO:AIDER.
"""
from __future__ import annotations

import os
import time
import json
import uuid
import asyncio
import socket
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx
from pathlib import Path

APP_NAME = "Aider-LCO"
SERVICE_TYPE = "agent"  # agent|tool|model
DEFAULT_ROLE = "execution"
DEFAULT_CAPS = ["git-edit","pytest-loop","repo-map"]
CTX_LIMIT = int(os.getenv("AIDER_CTX_LIMIT", "131072"))
PORT = int(os.getenv("PAS_PORT", "6150"))
HOST = os.getenv("PAS_HOST", "127.0.0.1")
REGISTRY_URL = os.getenv("PAS_REGISTRY_URL")  # e.g., http://127.0.0.1:6121
HEARTBEAT_SEC = int(os.getenv("PAS_HEARTBEAT_SEC", "60"))
ARTIFACTS_COST_DIR = Path(os.getenv("PAS_COST_DIR", "artifacts/costs")).resolve()
ARTIFACTS_COST_DIR.mkdir(parents=True, exist_ok=True)

SERVICE_ID = os.getenv("PAS_SERVICE_ID") or str(uuid.uuid4())

app = FastAPI(title=f"{APP_NAME} RPC Scaffold", version="0.1.0")

# ---------------------------
# Pydantic models (aligned with PRD)
# ---------------------------
class Target(BaseModel):
    service_id: Optional[str] = None
    name: Optional[str] = None
    type: Optional[str] = Field(None, description="model|tool|agent")
    role: Optional[str] = None
    labels: Optional[Dict[str, Any]] = None

class Policy(BaseModel):
    timeout_s: int = 60
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

# ---------------------------
# Helpers
# ---------------------------
async def write_receipt(receipt: RoutingReceipt):
    path = ARTIFACTS_COST_DIR / f"{receipt.run_id}.json"
    with path.open("w") as f:
        json.dump(receipt.dict(), f, indent=2)

def now_iso() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

async def try_registry_register():
    if not REGISTRY_URL:
        return
    data = {
        "name": APP_NAME,
        "type": SERVICE_TYPE,
        "role": DEFAULT_ROLE,
        "url": f"http://{HOST}:{PORT}",
        "caps": DEFAULT_CAPS,
        "labels": {"editor":"aider","space":"n/a"},
        "ctx_limit": CTX_LIMIT,
        "cost_hint": {"usd_per_1k": 0.00},
        "heartbeat_interval_s": HEARTBEAT_SEC,
        "ttl_s": HEARTBEAT_SEC + 30,
        "service_id": SERVICE_ID
    }
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(f"{REGISTRY_URL.rstrip('/')}/register", json=data)
            r.raise_for_status()
    except Exception as e:
        print(f"[WARN] Registry register failed: {e}")

async def heartbeat_task():
    if not REGISTRY_URL:
        return
    while True:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                hb = {
                    "service_id": SERVICE_ID,
                    "status": "ok",
                    "p95_ms": 800,
                    "queue_depth": 0,
                    "load": 0.10
                }
                r = await client.put(f"{REGISTRY_URL.rstrip('/')}/heartbeat", json=hb)
                # ignore errors for now
        except Exception as e:
            print(f"[WARN] Heartbeat failed: {e}")
        await asyncio.sleep(HEARTBEAT_SEC)

# ---------------------------
# Hook to Aider (stub)
# ---------------------------
async def aider_run(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO:AIDER
    This function should call the aider CLI / Python API to perform the requested action.
    For now, we fake an execution and echo inputs with a synthetic diff/result.
    """
    # Simulate some work
    await asyncio.sleep(0.2)
    return {
        "result": "ok",
        "summary": "aider stub executed",
        "echo_payload": payload,
        "diff": "--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n-Hello\n+Hello (edited by Aider-LCO)"
    }

# ---------------------------
# Routes
# ---------------------------
@app.on_event("startup")
async def _startup():
    if REGISTRY_URL:
        await try_registry_register()
        asyncio.create_task(heartbeat_task())

@app.get("/health")
async def health():
    return {"status": "ok", "service_id": SERVICE_ID, "name": APP_NAME, "role": DEFAULT_ROLE}

@app.get("/describe")
async def describe():
    return {
        "service_id": SERVICE_ID,
        "name": APP_NAME,
        "type": SERVICE_TYPE,
        "role": DEFAULT_ROLE,
        "caps": DEFAULT_CAPS,
        "labels": {"editor":"aider"},
        "ctx_limit": CTX_LIMIT,
        "host": HOST,
        "port": PORT
    }

@app.post("/invoke", response_model=InvokeResponse)
async def invoke(req: InvokeRequest, background_tasks: BackgroundTasks):
    t0 = time.perf_counter()

    # Resolve target locally (thin; Gateway usually does this)
    resolved = RoutingResolved(
        service_id=SERVICE_ID,
        name=APP_NAME if not req.target.name else req.target.name,
        role=DEFAULT_ROLE if not req.target.role else req.target.role,
        url=f"http://{HOST}:{PORT}"
    )

    # Execute (stubbed)
    upstream_out = await aider_run(req.payload)

    dt_upstream = (time.perf_counter() - t0) * 1000.0
    run_id = req.run_id or f"gw-{uuid.uuid4()}"

    receipt = RoutingReceipt(
        run_id=run_id,
        resolved=resolved,
        timings_ms={"queue": 0.0, "upstream": round(dt_upstream,2), "total": round(dt_upstream,2)},
        status="ok",
        ts=now_iso(),
    )
    background_tasks.add_task(write_receipt, receipt)

    return InvokeResponse(
        upstream=UpstreamResponse(outputs=upstream_out),
        routing_receipt=receipt
    )

if __name__ == "__main__":
    uvicorn.run("tools.aider_rpc.server:app", host=HOST, port=PORT, reload=False)

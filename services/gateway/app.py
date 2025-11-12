#!/usr/bin/env python3
"""
PAS Gateway (P0 Production Scaffold)
Single entrypoint for all Prime Directive submissions.

Responsibilities:
- Validate incoming requests
- Forward to PAS Root
- Handle idempotency keys (future)
- Attach receipts/metrics (future)

Port: 6120
"""
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import httpx
import os

PAS_ROOT = os.getenv("PAS_ROOT_URL", "http://127.0.0.1:6100")

app = FastAPI(title="PAS Gateway", version="1.0")


class PrimeDirectiveIn(BaseModel):
    title: str
    description: str
    repo_root: str
    entry_files: List[str] = []
    goal: str
    budget_tokens_max: Optional[int] = 25000
    budget_cost_usd_max: Optional[float] = 2.0


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "PAS Gateway",
        "port": 6120,
        "pas_root_url": PAS_ROOT
    }


@app.post("/prime_directives")
async def prime_directives(
    pd: PrimeDirectiveIn,
    idempotency_key: Optional[str] = Header(default=None)
):
    """
    Submit a Prime Directive for execution.

    Future enhancements:
    - Idempotency key caching (prevent duplicate submissions)
    - Receipt attachment (routing metadata)
    - Budget validation (check against PLMS limits)
    - Approval workflows (require human confirmation for risky ops)
    """
    # P0: Simple pass-through to PAS Root
    # Future: Add validation, caching, receipts, etc.

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{PAS_ROOT}/pas/prime_directives",
                json=pd.dict()
            )
        r.raise_for_status()
        return r.json()

    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(503, f"PAS Root unreachable: {str(e)}")


@app.get("/runs/{run_id}")
async def run_status(run_id: str):
    """Get run status by ID"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(f"{PAS_ROOT}/pas/runs/{run_id}")
        r.raise_for_status()
        return r.json()

    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(503, f"PAS Root unreachable: {str(e)}")


@app.get("/runs")
async def list_runs(limit: int = 50, status: Optional[str] = None):
    """List recent runs"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                f"{PAS_ROOT}/pas/runs",
                params={"limit": limit, "status": status} if status else {"limit": limit}
            )
        r.raise_for_status()
        return r.json()

    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code, e.response.text)
    except httpx.RequestError as e:
        raise HTTPException(503, f"PAS Root unreachable: {str(e)}")


class RunFailureNotification(BaseModel):
    """Notification of permanent run failure from PAS Root"""
    run_id: str
    prime_directive: str
    reason: str  # "max_restarts_exceeded" | "max_llm_retries_exceeded" | "manual_abort"
    failure_details: dict  # agent_id, restart_count, failure_count, last_error, etc.
    retry_history: List[dict] = []  # list of all retry attempts


@app.post("/notify_run_failed")
async def notify_run_failed(notification: RunFailureNotification):
    """
    Receive notification of permanent run failure from PAS Root.

    Called by PAS Root when:
    - All restart attempts exhausted (Level 1: 3 restarts)
    - All LLM retries exhausted (Level 2: 3 LLM switches)
    - Task marked as permanently failed (Level 3)

    HHMRS Phase 3: This endpoint receives failure notifications and:
    1. Logs the failure for audit trail
    2. Updates internal run tracking (if applicable)
    3. Triggers user notifications (email, Slack, etc.) - future enhancement
    4. Updates HMI status display (via WebSocket) - future enhancement

    See: docs/PRDs/PRD_Hierarchical_Health_Monitoring_Retry_System.md
    """
    import logging
    from datetime import datetime

    logger = logging.getLogger("gateway")

    # Log the failure
    logger.error(
        f"PERMANENT FAILURE: run_id={notification.run_id}, "
        f"reason={notification.reason}, "
        f"agent={notification.failure_details.get('agent_id', 'unknown')}, "
        f"restarts={notification.failure_details.get('restart_count', 0)}, "
        f"llm_retries={notification.failure_details.get('failure_count', 0)}"
    )

    # Log detailed retry history
    for attempt in notification.retry_history:
        logger.info(
            f"  Retry attempt {attempt.get('retry_count', '?')}: "
            f"type={attempt.get('retry_type', 'unknown')}, "
            f"reason={attempt.get('reason', 'unknown')}, "
            f"timestamp={attempt.get('timestamp', 'unknown')}"
        )

    # Future enhancements:
    # 1. Send email/Slack notification to user
    # 2. Update HMI via WebSocket (show red "FAILED" badge)
    # 3. Trigger chime/alert sound in HMI (if enabled in settings)
    # 4. Update run status in Gateway's own database (if we add one)
    # 5. Store failure analytics for HHMRS Phase 5 metrics

    # For now, just acknowledge receipt
    return {
        "status": "acknowledged",
        "run_id": notification.run_id,
        "timestamp": datetime.utcnow().isoformat(),
        "message": f"Permanent failure recorded for run {notification.run_id}"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6120)

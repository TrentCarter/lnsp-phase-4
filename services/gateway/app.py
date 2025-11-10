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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6120)

"""
Gateway Service - Central routing hub with cost tracking

This service:
- Routes requests to appropriate AI providers
- Tracks costs and generates receipts
- Integrates with Provider Router for provider selection
- Broadcasts cost events to Event Stream for HMI display
"""

import asyncio
import httpx
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from cost_tracker import CostTracker


# Pydantic Models for API

class ProviderRequirements(BaseModel):
    """Requirements for provider selection"""
    model: str
    context_window: Optional[int] = None
    features: Optional[List[str]] = None


class BudgetConstraints(BaseModel):
    """Budget constraints for request"""
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[int] = None


class RoutingRequest(BaseModel):
    """Request to route through gateway"""
    request_id: str = Field(..., min_length=1)
    run_id: Optional[str] = None
    agent: str = Field(..., min_length=1)
    requirements: ProviderRequirements
    budget: Optional[BudgetConstraints] = None
    optimization: str = Field(default="cost", pattern="^(cost|latency|balanced)$")
    payload: Optional[Dict[str, Any]] = None


# Initialize FastAPI app
app = FastAPI(
    title="Gateway",
    description="Central Routing Hub with Cost Tracking",
    version="1.0.0"
)

# Initialize cost tracker
cost_tracker = CostTracker()

# HTTP client for provider router and event stream
http_client = httpx.AsyncClient(timeout=30.0)

# Configuration
PROVIDER_ROUTER_URL = "http://localhost:6103"
EVENT_STREAM_URL = "http://localhost:6102"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "gateway",
        "port": 6120,
        "timestamp": datetime.utcnow().isoformat(),
        "dependencies": {
            "provider_router": PROVIDER_ROUTER_URL,
            "event_stream": EVENT_STREAM_URL
        }
    }


@app.post("/route")
async def route_request(request: RoutingRequest):
    """
    Route a request through the gateway

    This endpoint:
    1. Selects appropriate provider via Provider Router
    2. Routes request to selected provider
    3. Tracks costs and generates receipt
    4. Broadcasts cost event to Event Stream
    5. Returns response with cost metadata

    Args:
        request: Routing request with requirements and payload

    Returns:
        Response from provider with cost metadata
    """
    start_time = time.time()

    # Check budget if run_id specified
    if request.run_id:
        budget_status = cost_tracker.get_budget_status(request.run_id)
        if budget_status.get('budget_set') and not budget_status.get('can_proceed'):
            raise HTTPException(
                status_code=429,
                detail=f"Budget exceeded: {budget_status['percent_used']:.1f}% used"
            )

    # Step 1: Select provider via Provider Router
    try:
        selection_request = {
            "requirements": request.requirements.model_dump(),
            "optimization": request.optimization
        }

        if request.budget:
            if request.budget.max_cost_usd:
                selection_request["max_cost_usd"] = request.budget.max_cost_usd
            if request.budget.max_latency_ms:
                selection_request["max_latency_ms"] = request.budget.max_latency_ms

        response = await http_client.post(
            f"{PROVIDER_ROUTER_URL}/select",
            json=selection_request
        )
        response.raise_for_status()
        selection = response.json()

        selected_provider = selection['selected_provider']
        alternatives = selection.get('alternatives', [])

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Provider Router unavailable: {str(e)}"
        )

    # Step 2: Route request to selected provider
    # For now, we'll simulate the provider response
    # In production, this would call the actual provider endpoint
    provider_response = await _call_provider(
        selected_provider,
        request.payload or {}
    )

    latency_ms = int((time.time() - start_time) * 1000)

    # Extract token usage from provider response
    # This structure assumes OpenAI-style response format
    usage = provider_response.get('usage', {})
    input_tokens = usage.get('prompt_tokens', 0)
    output_tokens = usage.get('completion_tokens', 0)
    status = "success" if provider_response.get('choices') else "error"

    # Step 3: Track costs and generate receipt
    receipt = cost_tracker.record_request(
        request_id=request.request_id,
        run_id=request.run_id,
        agent=request.agent,
        provider=selected_provider['name'],
        model=selected_provider['model'],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_per_input_token=selected_provider['cost_per_input_token'],
        cost_per_output_token=selected_provider['cost_per_output_token'],
        latency_ms=latency_ms,
        status=status
    )

    # Step 4: Broadcast cost event to Event Stream
    await _broadcast_cost_event(receipt)

    # Step 5: Return response with cost metadata
    return {
        "request_id": request.request_id,
        "run_id": request.run_id,
        "status": status,
        "provider": selected_provider['name'],
        "model": selected_provider['model'],
        "response": provider_response,
        "cost": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": receipt['cost_usd'],
            "latency_ms": latency_ms
        },
        "alternatives": [p['name'] for p in alternatives],
        "timestamp": receipt['timestamp']
    }


async def _call_provider(provider: Dict, payload: Dict) -> Dict:
    """
    Call the selected provider endpoint

    For MVP, this simulates a provider response.
    In production, this would make actual HTTP calls to provider APIs.

    Args:
        provider: Provider info from registry
        payload: Request payload to forward

    Returns:
        Provider response
    """
    # Simulate provider response
    # In production, this would be:
    # response = await http_client.post(provider['endpoint'], json=payload)
    # return response.json()

    # Simulated response (OpenAI-style format)
    return {
        "id": "chatcmpl-simulated",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": provider['model'],
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a simulated response from the gateway."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 25,
            "total_tokens": 75
        }
    }


async def _broadcast_cost_event(receipt: Dict):
    """
    Broadcast cost event to Event Stream for HMI display

    Args:
        receipt: Cost receipt to broadcast
    """
    try:
        event_payload = {
            "event_type": "cost_update",
            "data": receipt
        }

        await http_client.post(
            f"{EVENT_STREAM_URL}/broadcast",
            json=event_payload
        )
    except httpx.HTTPError:
        # Non-critical: Event Stream unavailable
        # Log but don't fail the request
        pass


@app.get("/metrics")
async def get_metrics(window: str = "minute"):
    """
    Get cost metrics for a rolling window

    Args:
        window: Time window (minute/hour/day)

    Returns:
        Cost metrics for the window
    """
    try:
        metrics = cost_tracker.get_metrics(window)
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/receipts/{run_id}")
async def get_receipts(run_id: str):
    """
    Get all receipts for a run

    Args:
        run_id: Run identifier

    Returns:
        List of receipts
    """
    receipts = cost_tracker.get_all_receipts(run_id)
    return {
        "run_id": run_id,
        "receipts": receipts,
        "count": len(receipts),
        "total_cost_usd": sum(r['cost_usd'] for r in receipts)
    }


@app.post("/budget")
async def set_budget(run_id: str, budget_usd: float):
    """
    Set budget for a run

    Args:
        run_id: Run identifier
        budget_usd: Budget in USD

    Returns:
        Budget confirmation
    """
    if budget_usd <= 0:
        raise HTTPException(status_code=400, detail="Budget must be positive")

    cost_tracker.set_budget(run_id, budget_usd)

    return {
        "run_id": run_id,
        "budget_usd": budget_usd,
        "status": "Budget set successfully"
    }


@app.get("/budget/{run_id}")
async def get_budget_status(run_id: str):
    """
    Get budget status for a run

    Args:
        run_id: Run identifier

    Returns:
        Budget status with alerts
    """
    return cost_tracker.get_budget_status(run_id)


@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Gateway",
        "version": "1.0.0",
        "description": "Central Routing Hub with Cost Tracking",
        "endpoints": {
            "health": "/health",
            "route": "POST /route",
            "metrics": "GET /metrics?window=minute|hour|day",
            "receipts": "GET /receipts/{run_id}",
            "set_budget": "POST /budget?run_id=X&budget_usd=Y",
            "get_budget": "GET /budget/{run_id}",
            "docs": "/docs"
        }
    }


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    await http_client.aclose()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6120)

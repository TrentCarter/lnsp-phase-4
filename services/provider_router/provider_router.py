"""
Provider Router Service - Manages AI provider registration and selection

This service handles:
- Provider registration and discovery
- Capability matching (model, context window, features)
- Provider selection based on cost, latency, and availability
"""

import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime

from provider_registry import ProviderRegistry


# Pydantic Models for API

class ProviderRegistration(BaseModel):
    """Provider registration request"""
    name: str = Field(..., min_length=1, max_length=100)
    model: str = Field(..., min_length=1, max_length=100)
    context_window: int = Field(..., ge=1, le=2000000)
    cost_per_input_token: float = Field(..., ge=0)
    cost_per_output_token: float = Field(..., ge=0)
    endpoint: str = Field(..., description="Base URL for provider API")
    features: List[str] = Field(default_factory=list)
    slo: Optional[Dict] = None
    metadata: Optional[Dict] = None


class ProviderRequirements(BaseModel):
    """Requirements for provider selection"""
    model: str
    context_window: Optional[int] = None
    features: Optional[List[str]] = None


class SelectionRequest(BaseModel):
    """Request for provider selection"""
    requirements: ProviderRequirements
    optimization: str = Field(default="cost", pattern="^(cost|latency|balanced)$")
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[int] = None


# Initialize FastAPI app
app = FastAPI(
    title="Provider Router",
    description="AI Provider Registration and Selection Service",
    version="1.0.0"
)

# Initialize provider registry
registry = ProviderRegistry()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    stats = registry.get_stats()
    return {
        "status": "ok",
        "service": "provider_router",
        "port": 6103,
        "timestamp": datetime.utcnow().isoformat(),
        "registry_stats": stats
    }


@app.post("/register")
async def register_provider(provider: ProviderRegistration):
    """
    Register a new AI provider

    Args:
        provider: Provider registration data

    Returns:
        Registration status and provider info
    """
    result = registry.register_provider(provider.model_dump())

    if result['status'] == 'error':
        raise HTTPException(status_code=400, detail=result['error'])

    return result


@app.get("/providers")
async def list_providers(
    model: Optional[str] = None,
    min_context: Optional[int] = None
):
    """
    List all active providers with optional filtering

    Args:
        model: Filter by model name (exact match)
        min_context: Filter by minimum context window

    Returns:
        List of matching providers
    """
    providers = registry.list_providers(model=model, min_context=min_context)
    return {
        "providers": providers,
        "count": len(providers),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/providers/{name}")
async def get_provider(name: str):
    """
    Get provider details by name

    Args:
        name: Provider name

    Returns:
        Provider details
    """
    provider = registry.get_provider(name)

    if not provider:
        raise HTTPException(status_code=404, detail=f"Provider '{name}' not found")

    return provider


@app.post("/select")
async def select_provider(request: SelectionRequest):
    """
    Select best provider based on requirements and optimization strategy

    Args:
        request: Selection request with requirements and optimization

    Returns:
        Selected provider with selection reasoning
    """
    providers = registry.find_matching_providers(request.requirements.model_dump())

    if not providers:
        raise HTTPException(
            status_code=404,
            detail=f"No providers found matching requirements: {request.requirements}"
        )

    # Filter by budget constraints if specified
    if request.max_cost_usd is not None:
        # Estimate cost for 1000 input + 500 output tokens
        providers = [
            p for p in providers
            if (1000 * p['cost_per_input_token'] + 500 * p['cost_per_output_token']) <= request.max_cost_usd
        ]

    if not providers:
        raise HTTPException(
            status_code=404,
            detail="No providers found within budget constraints"
        )

    # Select provider based on optimization strategy
    if request.optimization == "cost":
        # Already sorted by cost in find_matching_providers
        selected = providers[0]
        reason = "Lowest cost provider"
    elif request.optimization == "latency":
        # Sort by latency SLO if available
        providers_with_slo = [p for p in providers if p.get('slo', {}).get('latency_p95_ms')]
        if providers_with_slo:
            selected = min(providers_with_slo, key=lambda p: p['slo']['latency_p95_ms'])
            reason = "Lowest latency SLO"
        else:
            selected = providers[0]
            reason = "No latency SLO available, selected lowest cost"
    else:  # balanced
        # Score = normalized_cost + normalized_latency
        # For now, default to cost
        selected = providers[0]
        reason = "Balanced optimization (cost-weighted)"

    return {
        "selected_provider": selected,
        "alternatives": providers[1:3] if len(providers) > 1 else [],
        "selection_reason": reason,
        "optimization": request.optimization,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.delete("/providers/{name}")
async def deactivate_provider(name: str):
    """
    Deactivate a provider (soft delete)

    Args:
        name: Provider name

    Returns:
        Deactivation status
    """
    result = registry.deactivate_provider(name)

    if result['status'] == 'error':
        raise HTTPException(status_code=404, detail=result['error'])

    return result


@app.get("/stats")
async def get_stats():
    """Get provider registry statistics"""
    return registry.get_stats()


# For testing/debugging
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Provider Router",
        "version": "1.0.0",
        "description": "AI Provider Registration and Selection Service",
        "endpoints": {
            "health": "/health",
            "register": "POST /register",
            "list": "GET /providers",
            "get": "GET /providers/{name}",
            "select": "POST /select",
            "deactivate": "DELETE /providers/{name}",
            "stats": "GET /stats",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6103)

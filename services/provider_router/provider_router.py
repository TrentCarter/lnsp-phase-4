"""
Provider Router Service - Manages AI provider registration and selection

This service handles:
- Provider registration and discovery
- Capability matching (model, context window, features)
- Provider selection based on cost, latency, and availability
- Integration with Model Pool Manager for dynamic model routing
- Per-agent model preferences and inference parameter management
"""

import asyncio
import httpx
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from provider_registry import ProviderRegistry


# Configuration
MODEL_POOL_URL = "http://localhost:8050"
# Use absolute paths relative to project root (2 levels up from this file)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_PREFERENCES_FILE = _PROJECT_ROOT / "configs" / "pas" / "model_preferences.json"
ADVANCED_SETTINGS_FILE = _PROJECT_ROOT / "configs" / "pas" / "advanced_model_settings.json"

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


class ModelRouteRequest(BaseModel):
    """Request for model routing (Model Pool integration)"""
    agent_class: Optional[str] = Field(default=None, description="Agent class (Architect, Programmer, etc.)")
    model_id: Optional[str] = Field(default=None, description="Specific model ID to use")
    prompt: str = Field(..., description="Prompt to send to the model")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Override inference parameters")


# Initialize FastAPI app
app = FastAPI(
    title="Provider Router",
    description="AI Provider Registration and Selection Service",
    version="1.0.0"
)

# Initialize provider registry
registry = ProviderRegistry()


# ============================================================================
# Model Pool Integration Helpers
# ============================================================================

def load_model_preferences() -> Dict:
    """Load model preferences from config file"""
    if MODEL_PREFERENCES_FILE.exists():
        with open(MODEL_PREFERENCES_FILE, 'r') as f:
            return json.load(f)
    return {"agent_preferences": {}, "model_specific_settings": {}}


def load_advanced_settings() -> Dict:
    """Load advanced model settings from config file"""
    if ADVANCED_SETTINGS_FILE.exists():
        with open(ADVANCED_SETTINGS_FILE, 'r') as f:
            return json.load(f)
    return {}


async def get_model_pool_status() -> Dict:
    """Query Model Pool Manager for active models"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MODEL_POOL_URL}/models")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        return {"models": [], "error": str(e)}


def select_model_for_agent(agent_class: str, preferences: Dict) -> tuple[str, str]:
    """
    Select model based on agent class preferences

    Returns:
        tuple: (primary_model_id, fallback_model_id)
    """
    agent_prefs = preferences.get("agent_preferences", {})

    # Get preferences for this agent class, or use default
    agent_config = agent_prefs.get(agent_class, agent_prefs.get("default", {}))

    primary = agent_config.get("primary", "llama3.1:8b")
    fallback = agent_config.get("fallback", "llama3.1:8b")

    return primary, fallback


def get_model_endpoint(model_id: str, pool_status: Dict) -> Optional[str]:
    """
    Find endpoint for a specific model in the pool

    Returns:
        str: Endpoint URL (e.g., "http://localhost:8051") or None if not found
    """
    models = pool_status.get("models", [])

    for model in models:
        if model.get("model_id") == model_id and model.get("state") == "HOT":
            port = model.get("port")
            if port:
                return f"http://localhost:{port}"

    return None


def merge_inference_parameters(
    model_id: str,
    preferences: Dict,
    advanced_settings: Dict,
    override_params: Optional[Dict] = None
) -> Dict:
    """
    Merge inference parameters from multiple sources

    Priority (highest to lowest):
    1. Override parameters from request
    2. Model-specific settings from preferences
    3. Global advanced settings
    """
    # Start with global advanced settings
    params = advanced_settings.copy()

    # Override with model-specific settings
    model_settings = preferences.get("model_specific_settings", {}).get(model_id, {})
    for key in ["temperature", "maxTokens", "topP", "topK", "frequencyPenalty", "presencePenalty"]:
        if key in model_settings:
            params[key] = model_settings[key]

    # Override with request-specific parameters
    if override_params:
        params.update(override_params)

    return params


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


# ============================================================================
# Model Pool Integration Endpoints
# ============================================================================

@app.get("/model-pool/status")
async def model_pool_status():
    """
    Get current Model Pool status with all active models

    Returns:
        Model pool status including all loaded models and their states
    """
    pool_status = await get_model_pool_status()

    if "error" in pool_status:
        raise HTTPException(
            status_code=503,
            detail=f"Model Pool Manager unavailable: {pool_status['error']}"
        )

    return {
        "status": "ok",
        "pool_manager_url": MODEL_POOL_URL,
        "timestamp": datetime.utcnow().isoformat(),
        **pool_status
    }


@app.get("/model-pool/preferences")
async def get_model_preferences():
    """
    Get model preferences configuration

    Returns:
        Model preferences for each agent class and model-specific settings
    """
    preferences = load_model_preferences()
    return {
        "status": "ok",
        "preferences": preferences,
        "config_file": str(MODEL_PREFERENCES_FILE),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/model-pool/route")
async def route_to_model(request: ModelRouteRequest):
    """
    Route request to appropriate model based on agent class or explicit model ID

    This is the main endpoint for Model Pool integration. It:
    1. Determines which model to use (explicit or based on agent class)
    2. Checks if model is loaded in the pool (HOT state)
    3. Loads model if needed
    4. Applies inference parameters
    5. Routes request to model service
    6. Returns model response

    Args:
        request: Routing request with agent_class or model_id, prompt, and optional parameters

    Returns:
        Model response with routing metadata
    """
    # Load configurations
    preferences = load_model_preferences()
    advanced_settings = load_advanced_settings()

    # Determine which model to use
    if request.model_id:
        # Explicit model requested
        primary_model = request.model_id
        fallback_model = None
        selection_reason = f"Explicit model requested: {request.model_id}"
    elif request.agent_class:
        # Select based on agent class
        primary_model, fallback_model = select_model_for_agent(request.agent_class, preferences)
        selection_reason = f"Selected for agent class '{request.agent_class}'"
    else:
        # Use default
        primary_model, fallback_model = select_model_for_agent("default", preferences)
        selection_reason = "Using default model (no agent_class or model_id specified)"

    # Get current pool status
    pool_status = await get_model_pool_status()

    if "error" in pool_status:
        raise HTTPException(
            status_code=503,
            detail=f"Model Pool Manager unavailable: {pool_status['error']}"
        )

    # Try primary model
    endpoint = get_model_endpoint(primary_model, pool_status)

    if not endpoint and fallback_model:
        # Try fallback model
        endpoint = get_model_endpoint(fallback_model, pool_status)
        if endpoint:
            primary_model = fallback_model
            selection_reason += f" (fallback to {fallback_model})"

    if not endpoint:
        # Model not loaded - trigger load
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                load_response = await client.post(f"{MODEL_POOL_URL}/models/{primary_model}/load")
                load_response.raise_for_status()
                load_data = load_response.json()

                # Wait for model to become HOT (poll for up to 60 seconds)
                for _ in range(12):  # 12 * 5 = 60 seconds
                    await asyncio.sleep(5)
                    pool_status = await get_model_pool_status()
                    endpoint = get_model_endpoint(primary_model, pool_status)
                    if endpoint:
                        break

                if not endpoint:
                    raise HTTPException(
                        status_code=504,
                        detail=f"Model {primary_model} failed to load within timeout"
                    )

        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to load model {primary_model}: {str(e)}"
            )

    # Merge inference parameters
    params = merge_inference_parameters(
        primary_model,
        preferences,
        advanced_settings,
        request.parameters
    )

    # Route request to model service
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Use OpenAI-compatible /v1/chat/completions endpoint
            model_request = {
                "model": primary_model,
                "messages": [{"role": "user", "content": request.prompt}],
                "temperature": params.get("temperature", 0.8),
                "max_tokens": params.get("maxTokens", 3000),
                "top_p": params.get("topP", 0.95),
            }

            # Add optional parameters if present
            if "topK" in params:
                model_request["top_k"] = params["topK"]
            if "frequencyPenalty" in params:
                model_request["frequency_penalty"] = params["frequencyPenalty"]
            if "presencePenalty" in params:
                model_request["presence_penalty"] = params["presencePenalty"]

            response = await client.post(
                f"{endpoint}/v1/chat/completions",
                json=model_request
            )
            response.raise_for_status()
            model_response = response.json()

            return {
                "status": "success",
                "model_used": primary_model,
                "endpoint": endpoint,
                "selection_reason": selection_reason,
                "parameters_used": params,
                "agent_class": request.agent_class,
                "response": model_response,
                "timestamp": datetime.utcnow().isoformat()
            }

    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Model service error: {str(e)}"
        )


@app.post("/model-pool/route/stream")
async def route_to_model_stream(request: ModelRouteRequest):
    """
    Route request to model with streaming response (for long completions)

    Same as /route but returns Server-Sent Events (SSE) for streaming
    """
    # TODO: Implement streaming support
    raise HTTPException(
        status_code=501,
        detail="Streaming not yet implemented - use /model-pool/route for non-streaming"
    )


# For testing/debugging
@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "Provider Router",
        "version": "1.0.0",
        "description": "AI Provider Registration and Selection Service with Model Pool Integration",
        "endpoints": {
            "health": "/health",
            "register": "POST /register",
            "list": "GET /providers",
            "get": "GET /providers/{name}",
            "select": "POST /select",
            "deactivate": "DELETE /providers/{name}",
            "stats": "GET /stats",
            "model_pool_status": "GET /model-pool/status",
            "model_pool_preferences": "GET /model-pool/preferences",
            "model_pool_route": "POST /model-pool/route",
            "model_pool_route_stream": "POST /model-pool/route/stream",
            "docs": "/docs"
        },
        "model_pool_integration": {
            "enabled": True,
            "pool_manager_url": MODEL_POOL_URL,
            "preferences_file": str(MODEL_PREFERENCES_FILE),
            "advanced_settings_file": str(ADVANCED_SETTINGS_FILE)
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6103)

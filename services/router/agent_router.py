"""
Agent Router Service - Routes requests to agents based on capabilities

This service:
- Routes requests to agents by name or capability
- Queries Registry for agent discovery
- Supports multiple transport mechanisms (RPC, REST, file-based, MCP)
- Tracks agent invocations for monitoring
- Integrates with Gateway for cost tracking
"""

import asyncio
import httpx
import time
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


# Pydantic Models for API

class AgentInvocationRequest(BaseModel):
    """Request to invoke an agent"""
    request_id: str = Field(..., min_length=1)
    run_id: Optional[str] = None
    agent_name: Optional[str] = None
    capabilities: Optional[List[str]] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    timeout_s: Optional[int] = 30
    transport: Optional[Literal["rpc", "file", "mcp", "rest"]] = None


class AgentInvocationResponse(BaseModel):
    """Response from agent invocation"""
    request_id: str
    agent_name: str
    status: Literal["success", "error", "timeout"]
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    latency_ms: int


class AgentDiscoveryRequest(BaseModel):
    """Request to discover agents by capability"""
    capabilities: Optional[List[str]] = None
    agent_role: Optional[Literal["coord", "exec", "system"]] = None
    tier: Optional[int] = None
    limit: Optional[int] = 10


# Initialize FastAPI app
app = FastAPI(
    title="Agent Router",
    description="Routes requests to agents based on capabilities",
    version="1.0.0"
)

# HTTP client for Registry and other services
http_client = httpx.AsyncClient(timeout=30.0)

# Configuration
REGISTRY_URL = "http://localhost:6121"
GATEWAY_URL = "http://localhost:6120"
EVENT_STREAM_URL = "http://localhost:6102"

# In-memory cache for agent definitions
agent_cache: Dict[str, Dict[str, Any]] = {}


@app.on_event("startup")
async def startup_event():
    """Load agent definitions on startup"""
    await refresh_agent_cache()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await http_client.aclose()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "service": "agent_router",
        "port": 6119,
        "timestamp": datetime.utcnow().isoformat(),
        "cached_agents": len(agent_cache),
        "dependencies": {
            "registry": REGISTRY_URL,
            "gateway": GATEWAY_URL,
            "event_stream": EVENT_STREAM_URL
        }
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "agent_router",
        "version": "1.0.0",
        "description": "Routes requests to agents based on capabilities"
    }


async def refresh_agent_cache():
    """Refresh agent cache from Registry"""
    try:
        response = await http_client.get(f"{REGISTRY_URL}/services")
        response.raise_for_status()
        data = response.json()

        # Filter for agents only
        services = data.get('items', [])
        agents = [s for s in services if s.get('type') == 'agent']

        # Cache by service_id and name
        for agent in agents:
            agent_name = agent['service_id'].replace('agent-', '')
            agent_cache[agent_name] = agent

        return len(agent_cache)
    except Exception as e:
        print(f"Warning: Failed to refresh agent cache: {e}")
        return 0


@app.post("/discover")
async def discover_agents(request: AgentDiscoveryRequest):
    """
    Discover agents by capability, role, or tier

    Args:
        request: Discovery criteria

    Returns:
        List of matching agents
    """
    await refresh_agent_cache()

    matching_agents = []

    for agent_name, agent_data in agent_cache.items():
        # Filter by role
        if request.agent_role:
            agent_role = agent_data.get('labels', {}).get('agent_role')
            if agent_role != request.agent_role:
                continue

        # Filter by tier
        if request.tier:
            tier = agent_data.get('labels', {}).get('tier')
            if tier and int(tier) != request.tier:
                continue

        # Filter by capabilities
        if request.capabilities:
            agent_caps = set(agent_data.get('caps', []))
            required_caps = set(request.capabilities)
            if not required_caps.issubset(agent_caps):
                continue

        matching_agents.append({
            "agent_name": agent_name,
            "display_name": agent_data.get('name'),
            "role": agent_data.get('labels', {}).get('agent_role'),
            "tier": agent_data.get('labels', {}).get('tier'),
            "capabilities": agent_data.get('caps', []),
            "url": agent_data.get('url'),
            "status": agent_data.get('labels', {}).get('status', 'available')
        })

    # Limit results
    if request.limit:
        matching_agents = matching_agents[:request.limit]

    return {
        "count": len(matching_agents),
        "agents": matching_agents
    }


@app.post("/invoke")
async def invoke_agent(request: AgentInvocationRequest):
    """
    Invoke an agent by name or capability

    This endpoint:
    1. Discovers agent (by name or capability)
    2. Determines transport mechanism
    3. Invokes agent via appropriate handler
    4. Returns response with metadata

    Args:
        request: Invocation request with agent name or capabilities

    Returns:
        Agent response with metadata
    """
    start_time = time.time()

    # Step 1: Discover agent
    agent_name = None
    agent_data = None

    if request.agent_name:
        # Direct lookup by name
        await refresh_agent_cache()
        agent_data = agent_cache.get(request.agent_name)
        if not agent_data:
            raise HTTPException(
                status_code=404,
                detail=f"Agent not found: {request.agent_name}"
            )
        agent_name = request.agent_name
    elif request.capabilities:
        # Discover by capability
        discovery = await discover_agents(
            AgentDiscoveryRequest(capabilities=request.capabilities, limit=1)
        )
        if discovery['count'] == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No agent found with capabilities: {request.capabilities}"
            )
        agent_name = discovery['agents'][0]['agent_name']
        agent_data = agent_cache.get(agent_name)
    else:
        raise HTTPException(
            status_code=400,
            detail="Must specify agent_name or capabilities"
        )

    # Step 2: Determine transport mechanism
    transport = request.transport
    if not transport:
        # Use first transport from agent definition
        # We'll need to look this up from the agent's markdown file
        # For now, default to RPC
        transport = "rpc"

    # Step 3: Invoke agent via handler
    try:
        result = await _invoke_via_transport(
            agent_name=agent_name,
            agent_data=agent_data,
            transport=transport,
            payload=request.payload,
            timeout_s=request.timeout_s or 30
        )

        latency_ms = int((time.time() - start_time) * 1000)

        # Broadcast invocation event
        await _broadcast_invocation_event(
            request_id=request.request_id,
            run_id=request.run_id,
            agent_name=agent_name,
            status="success",
            latency_ms=latency_ms
        )

        return AgentInvocationResponse(
            request_id=request.request_id,
            agent_name=agent_name,
            status="success",
            result=result,
            metadata={
                "transport": transport,
                "agent_role": agent_data.get('labels', {}).get('agent_role'),
                "tier": agent_data.get('labels', {}).get('tier')
            },
            latency_ms=latency_ms
        )

    except asyncio.TimeoutError:
        latency_ms = int((time.time() - start_time) * 1000)
        await _broadcast_invocation_event(
            request_id=request.request_id,
            run_id=request.run_id,
            agent_name=agent_name,
            status="timeout",
            latency_ms=latency_ms
        )
        return AgentInvocationResponse(
            request_id=request.request_id,
            agent_name=agent_name,
            status="timeout",
            error=f"Agent invocation timed out after {request.timeout_s}s",
            metadata={},
            latency_ms=latency_ms
        )
    except Exception as e:
        latency_ms = int((time.time() - start_time) * 1000)
        await _broadcast_invocation_event(
            request_id=request.request_id,
            run_id=request.run_id,
            agent_name=agent_name,
            status="error",
            latency_ms=latency_ms
        )
        return AgentInvocationResponse(
            request_id=request.request_id,
            agent_name=agent_name,
            status="error",
            error=str(e),
            metadata={},
            latency_ms=latency_ms
        )


async def _invoke_via_transport(
    agent_name: str,
    agent_data: Dict[str, Any],
    transport: str,
    payload: Dict[str, Any],
    timeout_s: int
) -> Any:
    """
    Invoke agent via specified transport mechanism

    Args:
        agent_name: Name of agent to invoke
        agent_data: Agent metadata from Registry
        transport: Transport mechanism (rpc, file, mcp, rest)
        payload: Request payload
        timeout_s: Timeout in seconds

    Returns:
        Agent response
    """
    if transport == "rpc":
        return await _invoke_via_rpc(agent_name, agent_data, payload, timeout_s)
    elif transport == "file":
        return await _invoke_via_file(agent_name, agent_data, payload, timeout_s)
    elif transport == "mcp":
        return await _invoke_via_mcp(agent_name, agent_data, payload, timeout_s)
    elif transport == "rest":
        return await _invoke_via_rest(agent_name, agent_data, payload, timeout_s)
    else:
        raise ValueError(f"Unsupported transport: {transport}")


async def _invoke_via_rpc(
    agent_name: str,
    agent_data: Dict[str, Any],
    payload: Dict[str, Any],
    timeout_s: int
) -> Any:
    """
    Invoke agent via RPC (Task tool in Claude Code)

    This is a stub implementation. In production, this would:
    1. Use Claude Code's Task tool to spawn sub-agent
    2. Pass context and payload
    3. Wait for response
    4. Return result

    For now, we'll return a simulated response.
    """
    # Simulate agent processing
    await asyncio.sleep(0.1)

    return {
        "message": f"Agent {agent_name} processed request via RPC",
        "payload_received": payload,
        "transport": "rpc",
        "timestamp": datetime.utcnow().isoformat()
    }


async def _invoke_via_file(
    agent_name: str,
    agent_data: Dict[str, Any],
    payload: Dict[str, Any],
    timeout_s: int
) -> Any:
    """
    Invoke agent via file-based transport (inbox/outbox pattern)

    This writes a request file to agent's inbox, waits for response in outbox.
    """
    inbox_dir = Path(f"artifacts/agent_inbox/{agent_name}")
    outbox_dir = Path(f"artifacts/agent_outbox/{agent_name}")

    inbox_dir.mkdir(parents=True, exist_ok=True)
    outbox_dir.mkdir(parents=True, exist_ok=True)

    request_id = payload.get('request_id', 'unknown')
    request_file = inbox_dir / f"{request_id}.json"
    response_file = outbox_dir / f"{request_id}.json"

    # Write request
    request_file.write_text(json.dumps(payload, indent=2))

    # Wait for response (with timeout)
    start = time.time()
    while time.time() - start < timeout_s:
        if response_file.exists():
            response_data = json.loads(response_file.read_text())
            response_file.unlink()  # Cleanup
            request_file.unlink()
            return response_data
        await asyncio.sleep(0.1)

    # Timeout
    raise asyncio.TimeoutError(f"No response from agent {agent_name}")


async def _invoke_via_mcp(
    agent_name: str,
    agent_data: Dict[str, Any],
    payload: Dict[str, Any],
    timeout_s: int
) -> Any:
    """
    Invoke agent via MCP (Model Context Protocol)

    This is for future integration with MCP servers.
    """
    # Stub implementation
    return {
        "message": f"MCP invocation for {agent_name} (not yet implemented)",
        "payload": payload
    }


async def _invoke_via_rest(
    agent_name: str,
    agent_data: Dict[str, Any],
    payload: Dict[str, Any],
    timeout_s: int
) -> Any:
    """
    Invoke agent via REST API

    This is for system agents that run as HTTP services.
    """
    url = agent_data.get('url')
    if not url or url.startswith('internal://'):
        raise ValueError(f"Agent {agent_name} does not have a REST endpoint")

    # Call agent's endpoint
    try:
        response = await http_client.post(
            f"{url}/invoke",
            json=payload,
            timeout=timeout_s
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        raise Exception(f"REST invocation failed: {str(e)}")


async def _broadcast_invocation_event(
    request_id: str,
    run_id: Optional[str],
    agent_name: str,
    status: str,
    latency_ms: int
):
    """Broadcast invocation event to Event Stream"""
    try:
        event = {
            "type": "agent_invocation",
            "request_id": request_id,
            "run_id": run_id,
            "agent_name": agent_name,
            "status": status,
            "latency_ms": latency_ms,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }

        await http_client.post(
            f"{EVENT_STREAM_URL}/broadcast",
            json=event
        )
    except Exception as e:
        # Don't fail the request if event broadcast fails
        print(f"Warning: Failed to broadcast invocation event: {e}")


@app.get("/stats")
async def get_stats():
    """Get agent router statistics"""
    await refresh_agent_cache()

    # Group agents by role
    by_role = {}
    by_tier = {}

    for agent_name, agent_data in agent_cache.items():
        role = agent_data.get('labels', {}).get('agent_role', 'unknown')
        tier = agent_data.get('labels', {}).get('tier', 'unknown')

        by_role[role] = by_role.get(role, 0) + 1
        by_tier[tier] = by_tier.get(tier, 0) + 1

    return {
        "total_agents": len(agent_cache),
        "by_role": by_role,
        "by_tier": by_tier,
        "cache_updated": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6119)

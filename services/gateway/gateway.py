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
import json
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

from services.gateway.cost_tracker import CostTracker

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path)
print(f"[GATEWAY] Loaded .env from: {env_path}")


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


class ChatMessage(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., min_length=1)


class ChatStreamRequest(BaseModel):
    """Request to stream chat response"""
    session_id: str = Field(..., min_length=1)
    message_id: str = Field(..., min_length=1)
    agent_id: str = Field(..., min_length=1)
    model: str = Field(default="llama3.1:8b")
    # Legacy field for backward compatibility (single message)
    content: str = Field(default="", description="Legacy: single user message")
    # New field for conversation history
    messages: list[ChatMessage] = Field(default_factory=list, description="Full conversation history")


# Initialize FastAPI app
app = FastAPI(
    title="Gateway",
    description="Central Routing Hub with Cost Tracking",
    version="1.0.0"
)

# Configure CORS to allow requests from HMI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:6101", "http://127.0.0.1:6101"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize cost tracker
cost_tracker = CostTracker()

# HTTP client for provider router and event stream
http_client = httpx.AsyncClient(timeout=30.0)

# Configuration
PROVIDER_ROUTER_URL = "http://localhost:6103"
EVENT_STREAM_URL = "http://localhost:6102"
OLLAMA_URL = "http://localhost:11434"


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


@app.post("/chat/stream")
async def chat_stream_post(request: ChatStreamRequest):
    """
    POST endpoint for streaming chat responses (legacy support)

    Routes based on agent_id:
    - agent_id="direct" → Route to LLM provider directly
    - agent_id="architect" → Route to PAS Architect (port 6110) with Aider
    - agent_id="dir-code" → Route to Dir-Code (port 6111) with Aider
    - agent_id="dir-models" → Route to Dir-Models (port 6112) with Aider
    - agent_id="dir-data" → Route to Dir-Data (port 6113) with Aider
    - agent_id="dir-devsecops" → Route to Dir-DevSecOps (port 6114) with Aider
    - agent_id="dir-docs" → Route to Dir-Docs (port 6115) with Aider

    Then routes to LLM provider based on model prefix:
    - ollama/* → Ollama local models
    - anthropic/* → Anthropic API (Claude)
    - openai/* → OpenAI API (GPT)
    - google/* → Google API (Gemini)
    - kimi/* → Kimi API (Moonshot)
    - auto → Auto-select provider

    SSE Event Types:
    - status_update: Task progress (planning, executing, complete)
    - token: Streaming text chunks
    - usage: Token/cost tracking
    - done: Stream complete signal
    """
    # Check if routing to PAS agent (not "direct")
    agent_id = request.agent_id.lower()
    if agent_id != "direct":
        # Route to PAS agent endpoint
        return StreamingResponse(
            _stream_pas_agent_response(request),
            media_type="text/event-stream"
        )

    # Direct chat mode - route based on model prefix
    model = request.model.lower()

    if model.startswith("ollama/"):
        return StreamingResponse(
            _stream_ollama_response(request),
            media_type="text/event-stream"
        )
    elif model.startswith("anthropic/"):
        return StreamingResponse(
            _stream_anthropic_response(request),
            media_type="text/event-stream"
        )
    elif model.startswith("openai/"):
        return StreamingResponse(
            _stream_openai_response(request),
            media_type="text/event-stream"
        )
    elif model.startswith("google/"):
        return StreamingResponse(
            _stream_google_response(request),
            media_type="text/event-stream"
        )
    elif model.startswith("kimi/"):
        return StreamingResponse(
            _stream_kimi_response(request),
            media_type="text/event-stream"
        )
    elif model == "auto":
        # Auto-select: default to Ollama qwen2.5-coder
        # Create a modified request with a specific model
        auto_request = ChatStreamRequest(
            session_id=request.session_id,
            message_id=request.message_id,
            agent_id=request.agent_id,
            model="ollama/qwen2.5-coder:7b-instruct",  # Default auto-select model
            content=request.content
        )
        return StreamingResponse(
            _stream_ollama_response(auto_request),
            media_type="text/event-stream"
        )
    else:
        # Unknown provider: try Ollama as fallback
        return StreamingResponse(
            _stream_ollama_response(request),
            media_type="text/event-stream"
        )


@app.get("/chat/stream/{session_id}")
async def chat_stream_get(session_id: str):
    """
    GET endpoint for streaming chat responses (PRD-aligned)

    Note: This endpoint expects chat context to be retrieved from
    a session store. For now, returns an error directing to POST endpoint.
    """
    raise HTTPException(
        status_code=501,
        detail="GET /chat/stream/{session_id} not implemented. Use POST /chat/stream with full context."
    )


async def _stream_pas_agent_response(request: ChatStreamRequest):
    """
    Stream chat response from PAS agent (Architect, Directors) with filesystem access via Aider.

    Routes to agent's FastAPI endpoint which handles:
    1. LLM interaction for planning/reasoning
    2. Aider RPC calls for filesystem operations
    3. Job card management and delegation

    Agent endpoint mapping:
    - architect → http://127.0.0.1:6110/chat
    - dir-code → http://127.0.0.1:6111/chat
    - dir-models → http://127.0.0.1:6112/chat
    - dir-data → http://127.0.0.1:6113/chat
    - dir-devsecops → http://127.0.0.1:6114/chat
    - dir-docs → http://127.0.0.1:6115/chat

    Yields SSE events from agent (forwarded transparently)
    """
    # Map agent_id to endpoint
    agent_endpoints = {
        "architect": "http://127.0.0.1:6110",
        "dir-code": "http://127.0.0.1:6111",
        "dir-models": "http://127.0.0.1:6112",
        "dir-data": "http://127.0.0.1:6113",
        "dir-devsecops": "http://127.0.0.1:6114",
        "dir-docs": "http://127.0.0.1:6115",
    }

    agent_id = request.agent_id.lower()
    agent_url = agent_endpoints.get(agent_id)

    if not agent_url:
        # Unknown agent - return error
        yield f"data: {json.dumps({'type': 'error', 'message': f'Unknown PAS agent: {agent_id}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return

    # Status: Routing to agent
    yield f"data: {json.dumps({'type': 'status_update', 'status': 'routing', 'detail': f'Routing to {agent_id.upper()}...'})}\n\n"

    # Prepare message for agent
    # Use messages array if provided, otherwise fall back to legacy content field
    if request.messages:
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    else:
        # Legacy: single user message
        messages = [{"role": "user", "content": request.content}]

    # Get the latest user message for task description
    latest_message = messages[-1]["content"] if messages else request.content

    try:
        # TODO Phase 2: Implement proper PAS agent chat endpoints
        # For now, agents don't have /chat endpoints - they use job cards
        # This is a placeholder that returns an informative message

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': f'{agent_id.upper()} received task...'})}\n\n"

        # Return informative message about agent capabilities
        response_text = f"""**{agent_id.upper()} Agent Ready** 🔧

I'm a PAS agent with filesystem access via Aider. However, the direct chat interface for PAS agents is not yet implemented.

**Current Capabilities:**
- Receive job cards via Prime Directive workflow
- Delegate to Managers and Programmers
- Execute code changes via Aider RPC
- Track costs and generate reports

**To use me:**
1. Submit a task via Verdict CLI: `./bin/verdict send --title "Task" --goal "Description"`
2. Monitor progress in TRON (http://localhost:6101/tron)
3. View results in artifacts/runs/

**Coming Soon:**
- Direct chat interface with streaming responses
- Interactive filesystem exploration
- Real-time code editing assistance

For now, please use the **Direct Chat** option for conversational LLM access, or submit structured tasks via Verdict CLI for filesystem operations.
"""

        # Stream response as tokens
        for char in response_text:
            yield f"data: {json.dumps({'type': 'token', 'content': char})}\n\n"
            await asyncio.sleep(0.01)  # Simulate streaming

        # Usage metadata (free - no LLM call yet)
        yield f"data: {json.dumps({'type': 'usage', 'tokens': 0, 'cost_usd': 0.0})}\n\n"

        # Done
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response complete'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        print(f"[GATEWAY] Error streaming from PAS agent {agent_id}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': f'Agent error: {str(e)}'})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_ollama_response(request: ChatStreamRequest):
    """
    Stream chat response from Ollama with SSE formatting.

    Yields SSE events:
    - status_update: Progress indicators
    - token: Text chunks from LLM
    - usage: Token/cost metadata
    - done: Completion signal
    """
    try:
        # Status: Planning
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': 'Preparing request...'})}\n\n"

        # Prepare Ollama request (use /api/chat for conversational models)
        # Strip ollama/ prefix if present
        model_name = request.model
        if model_name.startswith("ollama/"):
            model_name = model_name[7:]  # Remove "ollama/" prefix

        # Use messages array if provided, otherwise fall back to legacy content field
        if request.messages:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        else:
            # Legacy: single user message
            messages = [{"role": "user", "content": request.content}]

        ollama_payload = {
            "model": model_name,
            "messages": messages,
            "stream": True
        }

        # Debug logging
        print(f"[GATEWAY] Ollama request: model={request.model}, prompt_len={len(request.content)}, url={OLLAMA_URL}/api/chat")

        # Status: Executing
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"

        # Track tokens for cost calculation
        total_tokens = 0
        response_text = ""

        # Stream from Ollama
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_URL}/api/chat",
                json=ollama_payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    try:
                        chunk = json.loads(line)

                        # Extract token from Ollama /api/chat response
                        if "message" in chunk and "content" in chunk["message"]:
                            token = chunk["message"]["content"]
                            if token:
                                response_text += token
                                total_tokens += 1

                                # Send token event
                                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                        # Check if done
                        if chunk.get("done", False):
                            # Status: Complete
                            yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response generated'})}\n\n"

                            # Usage tracking (Ollama returns actual counts in final chunk)
                            prompt_tokens = chunk.get("prompt_eval_count", 0)
                            completion_tokens = chunk.get("eval_count", total_tokens)

                            # Calculate cost (Llama 3.1 local is free, but track for consistency)
                            usage_data = {
                                'type': 'usage',
                                'usage': {
                                    'prompt_tokens': prompt_tokens,
                                    'completion_tokens': completion_tokens,
                                    'total_tokens': prompt_tokens + completion_tokens
                                },
                                'cost_usd': 0.0,  # Local LLM is free
                                'model': request.model
                            }
                            yield f"data: {json.dumps(usage_data)}\n\n"

                            # Done signal
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            break

                    except json.JSONDecodeError:
                        # Skip malformed lines
                        continue

    except httpx.HTTPError as e:
        # Error status
        import traceback
        error_msg = f"Ollama unavailable: {str(e)}"
        print(f"[GATEWAY ERROR] {error_msg}")
        print(f"[GATEWAY ERROR] Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        # Unexpected error
        import traceback
        error_msg = f"Streaming error: {str(e)}"
        print(f"[GATEWAY ERROR] {error_msg}")
        print(f"[GATEWAY ERROR] Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_anthropic_response(request: ChatStreamRequest):
    """Stream chat response from Anthropic API (Claude)"""
    try:
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': 'Preparing Anthropic request...'})}\n\n"

        # Check for API key
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            error_msg = "Anthropic API key not configured. Set ANTHROPIC_API_KEY environment variable."
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Strip anthropic/ prefix
        model_name = request.model[10:] if request.model.startswith("anthropic/") else request.model

        # Import Anthropic SDK
        try:
            from anthropic import Anthropic
        except ImportError:
            error_msg = "Anthropic SDK not installed. Run: pip install anthropic"
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"

        # Call Anthropic API
        client = Anthropic(api_key=anthropic_api_key)
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        with client.messages.stream(
            model=model_name,
            max_tokens=1024,
            messages=[{"role": "user", "content": request.content}]
        ) as stream:
            for text in stream.text_stream:
                response_text += text
                completion_tokens += 1
                yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response generated'})}\n\n"

        # Usage tracking
        usage_data = {
            'type': 'usage',
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens},
            'cost_usd': 0.0,  # TODO: Calculate actual Anthropic cost
            'model': request.model
        }
        yield f"data: {json.dumps(usage_data)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        import traceback
        error_msg = f"Anthropic API error: {str(e)}"
        print(f"[GATEWAY ERROR] {error_msg}")
        print(f"[GATEWAY ERROR] Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_openai_response(request: ChatStreamRequest):
    """Stream chat response from OpenAI API (GPT)"""
    try:
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': 'Preparing OpenAI request...'})}\n\n"

        # Check for API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key or openai_api_key.startswith('your_'):
            error_msg = "OpenAI API key not configured. Set OPENAI_API_KEY environment variable."
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Strip openai/ prefix
        model_name = request.model[7:] if request.model.startswith("openai/") else request.model

        # Import OpenAI SDK
        try:
            from openai import OpenAI
        except ImportError:
            error_msg = "OpenAI SDK not installed. Run: pip install openai"
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"

        # Call OpenAI API
        client = OpenAI(api_key=openai_api_key)
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        stream = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": request.content}],
            stream=True,
            max_tokens=1024
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                response_text += text
                completion_tokens += 1
                yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response generated'})}\n\n"

        # Usage tracking
        usage_data = {
            'type': 'usage',
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens},
            'cost_usd': 0.0,  # TODO: Calculate actual OpenAI cost
            'model': request.model
        }
        yield f"data: {json.dumps(usage_data)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        import traceback
        error_msg = f"OpenAI API error: {str(e)}"
        print(f"[GATEWAY ERROR] {error_msg}")
        print(f"[GATEWAY ERROR] Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_google_response(request: ChatStreamRequest):
    """Stream chat response from Google API (Gemini)"""
    try:
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': 'Preparing Google Gemini request...'})}\n\n"

        # Check for API key
        gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gemini_api_key or gemini_api_key.startswith('your_'):
            error_msg = "Google Gemini API key not configured. Set GEMINI_API_KEY environment variable."
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Strip google/ prefix
        model_name = request.model[7:] if request.model.startswith("google/") else request.model

        # Import Google Generative AI SDK
        try:
            import google.generativeai as genai
        except ImportError:
            error_msg = "Google Generative AI SDK not installed. Run: pip install google-generativeai"
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"

        # Configure and call Google Gemini API
        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel(model_name)
        response_text = ""
        completion_tokens = 0

        response = model.generate_content(request.content, stream=True)
        for chunk in response:
            if chunk.text:
                text = chunk.text
                response_text += text
                completion_tokens += len(text.split())
                yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response generated'})}\n\n"

        # Usage tracking
        usage_data = {
            'type': 'usage',
            'usage': {'prompt_tokens': 0, 'completion_tokens': completion_tokens, 'total_tokens': completion_tokens},
            'cost_usd': 0.0,  # TODO: Calculate actual Google cost
            'model': request.model
        }
        yield f"data: {json.dumps(usage_data)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        import traceback
        error_msg = f"Google Gemini API error: {str(e)}"
        print(f"[GATEWAY ERROR] {error_msg}")
        print(f"[GATEWAY ERROR] Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _stream_kimi_response(request: ChatStreamRequest):
    """Stream chat response from Kimi API (Moonshot)"""
    try:
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'planning', 'detail': 'Preparing Kimi request...'})}\n\n"

        # Check for API key
        kimi_api_key = os.getenv("KIMI_API_KEY")
        if not kimi_api_key or kimi_api_key.startswith('your_'):
            error_msg = "Kimi API key not configured. Set KIMI_API_KEY environment variable."
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        # Strip kimi/ prefix
        model_name = request.model[5:] if request.model.startswith("kimi/") else request.model

        # Kimi uses OpenAI-compatible API (Moonshot AI)
        try:
            from openai import OpenAI
        except ImportError:
            error_msg = "OpenAI SDK not installed (required for Kimi). Run: pip install openai"
            yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'executing', 'detail': 'Generating response...'})}\n\n"

        # Call Kimi API (Moonshot AI uses OpenAI-compatible endpoint)
        client = OpenAI(
            api_key=kimi_api_key,
            base_url="https://api.moonshot.ai/v1"  # Correct: .ai not .cn
        )
        response_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        # Get actual model name from env or use default
        actual_model = os.getenv("KIMI_MODEL_NAME", "moonshot-v1-8k")

        # Use messages array if provided, otherwise fall back to legacy content field
        if request.messages:
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        else:
            # Legacy: single user message
            messages = [{"role": "user", "content": request.content}]

        stream = client.chat.completions.create(
            model=actual_model,
            messages=messages,
            stream=True,
            max_tokens=1024
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                response_text += text
                completion_tokens += 1
                yield f"data: {json.dumps({'type': 'token', 'content': text})}\n\n"

        yield f"data: {json.dumps({'type': 'status_update', 'status': 'complete', 'detail': 'Response generated'})}\n\n"

        # Usage tracking
        usage_data = {
            'type': 'usage',
            'usage': {'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens, 'total_tokens': prompt_tokens + completion_tokens},
            'cost_usd': 0.0,  # TODO: Calculate actual Kimi cost
            'model': request.model
        }
        yield f"data: {json.dumps(usage_data)}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        import traceback
        error_msg = f"Kimi API error: {str(e)}"
        print(f"[GATEWAY ERROR] {error_msg}")
        print(f"[GATEWAY ERROR] Traceback: {traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'status_update', 'status': 'error', 'detail': error_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"


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
            "chat_stream_post": "POST /chat/stream",
            "chat_stream_get": "GET /chat/stream/{session_id}",
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

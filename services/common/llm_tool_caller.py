#!/usr/bin/env python3
"""
LLM Tool Calling Infrastructure

Unified interface for calling LLMs with tool support across multiple providers:
- Anthropic Claude (native tool support)
- Google Gemini (function calling)
- Ollama (tool support in recent versions)

Used in Phase 3 of Parent-Child Agent Chat Communications.
"""
import os
import json
import httpx
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """Unified LLM response structure"""
    content: str  # Text response from LLM
    tool_calls: List[Dict[str, Any]]  # List of tool calls made
    stop_reason: str  # "tool_use", "end_turn", "max_tokens", etc.
    provider: str  # Which provider was used
    model: str  # Which model was used
    usage: Optional[Dict[str, int]] = None  # Token usage stats


def detect_provider(model: str) -> str:
    """
    Detect LLM provider from model string

    Args:
        model: Model identifier (e.g., "claude-sonnet-4-5", "gemini-2.5-flash", "llama3.1:8b")

    Returns:
        Provider name: "anthropic", "google", or "ollama"
    """
    model_lower = model.lower()

    if "claude" in model_lower:
        return "anthropic"
    elif "gemini" in model_lower or "google" in model_lower:
        return "google"
    elif "llama" in model_lower or ":" in model:  # Ollama uses "model:tag" format
        return "ollama"
    else:
        # Default to Ollama for unknown models (local fallback)
        return "ollama"


async def call_llm_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 2000
) -> LLMResponse:
    """
    Call LLM with tool support (unified interface across providers)

    Args:
        system_prompt: System prompt (agent role/instructions)
        user_prompt: User prompt (task/question)
        tools: List of tool definitions (provider-specific format)
        model: Model identifier (auto-detects provider)
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum response tokens

    Returns:
        LLMResponse with content, tool_calls, and metadata
    """
    # Auto-detect model if not provided
    if model is None:
        model = os.getenv("LNSP_LLM_MODEL", "llama3.1:8b")

    provider = detect_provider(model)

    # Route to provider-specific implementation
    if provider == "anthropic":
        return await _call_anthropic_with_tools(
            system_prompt, user_prompt, tools, model, temperature, max_tokens
        )
    elif provider == "google":
        return await _call_google_with_tools(
            system_prompt, user_prompt, tools, model, temperature, max_tokens
        )
    elif provider == "ollama":
        return await _call_ollama_with_tools(
            system_prompt, user_prompt, tools, model, temperature, max_tokens
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")


async def call_llm(
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> LLMResponse:
    """
    Call LLM without tools (simple text completion)

    Args:
        system_prompt: System prompt
        user_prompt: User prompt
        model: Model identifier
        temperature: Sampling temperature
        max_tokens: Maximum response tokens

    Returns:
        LLMResponse (tool_calls will be empty)
    """
    # Call with empty tools list
    return await call_llm_with_tools(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=[],
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )


# === Anthropic Claude Implementation ===

async def _call_anthropic_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int
) -> LLMResponse:
    """Call Anthropic Claude API with tool support"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    endpoint = "https://api.anthropic.com/v1/messages"

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }

    body = {
        "model": model,
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    # Add tools if provided
    if tools:
        body["tools"] = tools

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(endpoint, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    # Parse response
    content_text = ""
    tool_calls = []

    for block in data.get("content", []):
        if block["type"] == "text":
            content_text += block["text"]
        elif block["type"] == "tool_use":
            tool_calls.append({
                "id": block["id"],
                "name": block["name"],
                "args": block["input"]
            })

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        stop_reason=data.get("stop_reason", "end_turn"),
        provider="anthropic",
        model=model,
        usage=data.get("usage")
    )


# === Google Gemini Implementation ===

async def _call_google_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int
) -> LLMResponse:
    """Call Google Gemini API with function calling"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not set")

    # Gemini uses "gemini-2.5-flash" format, extract version
    model_clean = model.replace("google/", "")
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_clean}:generateContent?key={api_key}"

    # Combine system + user prompts (Gemini doesn't have separate system role)
    full_prompt = f"{system_prompt}\n\n{user_prompt}"

    body = {
        "contents": [
            {"role": "user", "parts": [{"text": full_prompt}]}
        ],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens
        }
    }

    # Add tools if provided (Gemini calls them "function declarations")
    if tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool.get("parameters", tool.get("input_schema", {}))
                    }
                    for tool in tools
                ]
            }
        ]

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(endpoint, json=body)
        response.raise_for_status()
        data = response.json()

    # Parse response
    content_text = ""
    tool_calls = []

    candidates = data.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        content = candidate.get("content", {})

        for part in content.get("parts", []):
            if "text" in part:
                content_text += part["text"]
            elif "functionCall" in part:
                func_call = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",  # Gemini doesn't provide IDs
                    "name": func_call["name"],
                    "args": func_call.get("args", {})
                })

        stop_reason = "tool_use" if tool_calls else candidate.get("finishReason", "STOP")
    else:
        stop_reason = "error"

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        provider="google",
        model=model,
        usage=data.get("usageMetadata")
    )


# === Ollama Implementation ===

async def _call_ollama_with_tools(
    system_prompt: str,
    user_prompt: str,
    tools: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int
) -> LLMResponse:
    """
    Call Ollama with tool support

    Note: Ollama's tool support varies by model. Llama 3.1+ supports tools.
    For older models, we fall back to prompt-based tool calling.
    """
    endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
    chat_url = f"{endpoint.rstrip('/')}/api/chat"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens
        }
    }

    # Add tools if supported (Llama 3.1+)
    if tools and _ollama_supports_tools(model):
        body["tools"] = tools

    async with httpx.AsyncClient(timeout=120.0) as client:  # Ollama can be slower
        response = await client.post(chat_url, json=body)
        response.raise_for_status()
        data = response.json()

    # Parse response
    message = data.get("message", {})
    content_text = message.get("content", "")
    tool_calls = []

    # Check for tool calls in response
    if "tool_calls" in message:
        for tc in message["tool_calls"]:
            tool_calls.append({
                "id": tc.get("id", f"call_{len(tool_calls)}"),
                "name": tc["function"]["name"],
                "args": tc["function"].get("arguments", {})
            })

    # If no native tool calls but tools were provided, try to parse from text
    # (fallback for older models)
    if not tool_calls and tools and content_text:
        tool_calls = _parse_tool_calls_from_text(content_text, tools)

    stop_reason = "tool_use" if tool_calls else data.get("done_reason", "stop")

    return LLMResponse(
        content=content_text,
        tool_calls=tool_calls,
        stop_reason=stop_reason,
        provider="ollama",
        model=model,
        usage={
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "completion_tokens": data.get("eval_count", 0),
            "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0)
        }
    )


def _ollama_supports_tools(model: str) -> bool:
    """Check if Ollama model supports native tool calling"""
    # Llama 3.1+ supports tools natively
    tool_supported_models = ["llama3.1", "llama3.2", "mistral-nemo", "qwen2.5"]
    return any(m in model.lower() for m in tool_supported_models)


def _parse_tool_calls_from_text(text: str, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parse tool calls from LLM text output (fallback for models without native tool support)

    Looks for patterns like:
    ```
    ask_parent(question="...", urgency="blocking")
    ```
    """
    import re

    tool_calls = []

    for tool in tools:
        tool_name = tool["name"]
        # Look for tool_name(...) pattern
        pattern = rf'{tool_name}\((.*?)\)'
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            args_str = match.group(1)
            # Parse simple key=value pairs
            args = {}
            for arg_match in re.finditer(r'(\w+)="([^"]*)"', args_str):
                key, value = arg_match.groups()
                args[key] = value

            if args:
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "name": tool_name,
                    "args": args
                })

    return tool_calls

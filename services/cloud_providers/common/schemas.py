"""
Pydantic schemas for cloud provider adapters
OpenAI-compatible format for consistency across all providers
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum


# ============================================================================
# Chat Completion Models (OpenAI-compatible)
# ============================================================================

class ChatRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    role: ChatRole
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, ge=1)
    top_p: float = Field(1.0, ge=0, le=1)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "tool_calls"]


class Usage(BaseModel):
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage
    system_fingerprint: Optional[str] = None


# ============================================================================
# Health & Info Models
# ============================================================================

class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    status: HealthStatus
    service: str
    port: int = Field(..., ge=1024, le=65535)
    provider: str
    model: str
    api_key_present: bool
    uptime_seconds: float = Field(..., ge=0)
    last_request_ms: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    provider: str
    context_window: int
    cost_per_input_token: float
    cost_per_output_token: float
    capabilities: List[str]


class PerformanceInfo(BaseModel):
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None


class ServiceInfo(BaseModel):
    service_name: str
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    provider: str
    model: ModelInfo
    performance: PerformanceInfo
    endpoints: List[str]

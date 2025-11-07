"""
Pydantic models for LLM service API contracts
Aligned with contracts/llm_service.schema.json
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from enum import Enum


# ============================================================================
# Chat Completion Models
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
    max_tokens: Optional[int] = Field(None, ge=1, le=8192)
    top_p: float = Field(1.0, ge=0, le=1)
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: float = Field(0, ge=-2, le=2)
    frequency_penalty: float = Field(0, ge=-2, le=2)


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "content_filter", "null"]


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
# Generate (Ollama-style) Models
# ============================================================================

class GenerateOptions(BaseModel):
    temperature: Optional[float] = Field(None, ge=0, le=2)
    top_k: Optional[int] = Field(None, ge=1)
    top_p: Optional[float] = Field(None, ge=0, le=1)
    num_predict: Optional[int] = Field(None, ge=-2)
    repeat_penalty: Optional[float] = Field(None, ge=0)
    seed: Optional[int] = None


class GenerateRequest(BaseModel):
    model: str
    prompt: str = Field(..., min_length=1)
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    stream: bool = False
    raw: bool = False
    format: Optional[Literal["json"]] = None
    options: Optional[GenerateOptions] = None
    keep_alive: Optional[Union[str, int]] = None


class GenerateResponse(BaseModel):
    model: str
    created_at: str
    response: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


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
    ollama_backend: str
    model: str
    model_loaded: bool
    uptime_seconds: float = Field(..., ge=0)
    last_request_ms: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    parameters: Optional[str] = None
    quantization: Optional[str] = None
    context_length: Optional[int] = None
    embedding_dim: Optional[int] = None


class PerformanceInfo(BaseModel):
    avg_throughput_tok_s: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None


class ServiceInfo(BaseModel):
    service_name: str
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    model: ModelInfo
    capabilities: List[str]
    performance: PerformanceInfo
    endpoints: List[str]


# ============================================================================
# TLC Domain Classifier Models
# ============================================================================

class DomainClassificationRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(3, ge=1, le=10)
    context_docs: Optional[List[str]] = None


class DomainScore(BaseModel):
    domain: str
    confidence: float = Field(..., ge=0, le=1)


class DomainClassificationResponse(BaseModel):
    query: str
    domains: List[DomainScore] = Field(..., min_length=1)
    primary_domain: str
    metadata: Optional[dict] = None


class TaskType(str, Enum):
    RETRIEVE = "RETRIEVE"
    ANSWER = "ANSWER"
    VERIFY = "VERIFY"
    COMPARE = "COMPARE"
    SUMMARIZE = "SUMMARIZE"


class MethodType(str, Enum):
    DENSE = "DENSE"
    SPARSE = "SPARSE"
    HYBRID = "HYBRID"
    GRAPH = "GRAPH"


class TMD(BaseModel):
    task: TaskType
    method: MethodType
    domain: str


class TMDConfidence(BaseModel):
    task: float = Field(..., ge=0, le=1)
    method: float = Field(..., ge=0, le=1)
    domain: float = Field(..., ge=0, le=1)


class TMDExtractionRequest(BaseModel):
    query: str = Field(..., min_length=1)
    context_docs: Optional[List[str]] = None
    method_hint: Optional[MethodType] = None


class TMDExtractionResponse(BaseModel):
    query: str
    tmd: TMD
    confidence: Optional[TMDConfidence] = None
    metadata: Optional[dict] = None

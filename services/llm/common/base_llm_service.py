"""
Base FastAPI service for LLM wrappers
Provides common functionality for all LLM services
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import httpx
import time
from typing import List, Optional
from .schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    GenerateRequest, GenerateResponse,
    HealthResponse, HealthStatus, ServiceInfo,
    ModelInfo, PerformanceInfo
)
from .ollama_client import OllamaClient


class BaseLLMService:
    """Base class for LLM wrapper services"""

    def __init__(
        self,
        service_name: str,
        model: str,
        port: int,
        capabilities: List[str],
        ollama_url: str = "http://localhost:11434",
        agent_id: Optional[str] = None,
        version: str = "1.0.0"
    ):
        """
        Initialize base LLM service

        Args:
            service_name: Human-readable service name
            model: Ollama model identifier (e.g., "llama3.1:8b")
            port: Service port number
            capabilities: List of agent capabilities
            ollama_url: Ollama backend URL
            agent_id: Unique agent ID for registry
            version: Service version
        """
        self.service_name = service_name
        self.model = model
        self.port = port
        self.capabilities = capabilities
        self.ollama_url = ollama_url
        self.agent_id = agent_id or f"llm_{model.replace(':', '_').replace('.', '_')}"
        self.version = version
        self.start_time = time.time()
        self.last_request_time: Optional[float] = None

        # Initialize FastAPI app
        self.app = FastAPI(
            title=service_name,
            description=f"LLM wrapper service for {model}",
            version=version
        )

        # Initialize Ollama client
        self.ollama_client = OllamaClient(base_url=ollama_url)

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup FastAPI routes"""
        self.app.post("/chat/completions", response_model=ChatCompletionResponse)(
            self._chat_completions
        )
        self.app.post("/generate", response_model=GenerateResponse)(
            self._generate
        )
        self.app.get("/health")(self._health)
        self.app.get("/info")(self._info)

    async def _chat_completions(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Chat completions endpoint (OpenAI-compatible)

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        self.last_request_time = time.time()

        try:
            # Override model if different from service model
            if request.model != self.model:
                request.model = self.model

            response = await self.ollama_client.chat(request)
            return response

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: {str(e)}"
            )

    async def _generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Generate endpoint (Ollama-compatible)

        Args:
            request: Generate request

        Returns:
            Generate response
        """
        self.last_request_time = time.time()

        try:
            # Override model if different from service model
            if request.model != self.model:
                request.model = self.model

            response = await self.ollama_client.generate(request)
            return response

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Ollama error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: {str(e)}"
            )

    async def _health(self) -> HealthResponse:
        """
        Health check endpoint

        Returns:
            Health status
        """
        try:
            # Check Ollama backend
            is_healthy = await self.ollama_client.health_check()
            model_loaded = await self.ollama_client.is_model_loaded(self.model)

            status = HealthStatus.HEALTHY if (is_healthy and model_loaded) else HealthStatus.DEGRADED

            uptime = time.time() - self.start_time
            last_request_ms = (
                (time.time() - self.last_request_time) * 1000
                if self.last_request_time
                else None
            )

            return HealthResponse(
                status=status,
                service=self.agent_id,
                port=self.port,
                ollama_backend=self.ollama_url,
                model=self.model,
                model_loaded=model_loaded,
                uptime_seconds=uptime,
                last_request_ms=last_request_ms
            )

        except Exception as e:
            return HealthResponse(
                status=HealthStatus.UNHEALTHY,
                service=self.agent_id,
                port=self.port,
                ollama_backend=self.ollama_url,
                model=self.model,
                model_loaded=False,
                uptime_seconds=time.time() - self.start_time,
                last_request_ms=None
            )

    async def _info(self) -> ServiceInfo:
        """
        Service info endpoint

        Returns:
            Service metadata
        """
        return ServiceInfo(
            service_name=self.service_name,
            version=self.version,
            model=self._get_model_info(),
            capabilities=self.capabilities,
            performance=self._get_performance_info(),
            endpoints=[
                "/chat/completions",
                "/generate",
                "/health",
                "/info"
            ]
        )

    def _get_model_info(self) -> ModelInfo:
        """
        Get model metadata

        Returns:
            Model info (override in subclasses for specific details)
        """
        return ModelInfo(
            name=self.model,
            parameters=None,
            quantization=None,
            context_length=None,
            embedding_dim=None
        )

    def _get_performance_info(self) -> PerformanceInfo:
        """
        Get performance metadata

        Returns:
            Performance info (override in subclasses for specific metrics)
        """
        return PerformanceInfo(
            avg_throughput_tok_s=None,
            avg_latency_ms=None,
            p95_latency_ms=None
        )

    async def register_with_registry(self, registry_url: str = "http://localhost:6121"):
        """
        Register service with Agent Registry

        Args:
            registry_url: Agent Registry URL
        """
        # Use Registry service registration format
        registration = {
            "service_id": f"agent-{self.agent_id}",
            "name": self.service_name,
            "type": "agent",  # Service type: agent
            "role": "production",  # Registry role: production/staging/canary/experimental
            "url": f"http://localhost:{self.port}",
            "caps": self.capabilities,
            "labels": {
                "tier": "2",  # Local LLM = Tier 2
                "mode": "service",
                "agent_role": "execution",  # Agent type: coordinator/execution/system
                "model": self.model,
                "provider": "ollama",
                "version": self.version,
                "cost_per_1k_tokens": "0.0",
                "ollama_backend": self.ollama_url
            },
            "heartbeat_interval_s": 60,
            "ttl_s": 120
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{registry_url}/register",
                    json=registration,
                    timeout=10.0
                )
                if response.status_code == 200:
                    print(f"✅ Registered {registration['service_id']} with registry at {registry_url}")
                else:
                    print(f"⚠️ Registration failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Registry connection failed: {e}")

    async def startup(self):
        """
        Service startup hook
        Override in subclasses for custom initialization
        """
        # Check Ollama health
        is_healthy = await self.ollama_client.health_check()
        if not is_healthy:
            print(f"⚠️ Warning: Ollama backend at {self.ollama_url} is not responding")
        else:
            print(f"✅ Ollama backend healthy at {self.ollama_url}")

        # Check model availability
        model_loaded = await self.ollama_client.is_model_loaded(self.model)
        if not model_loaded:
            print(f"⚠️ Warning: Model '{self.model}' not found in Ollama")
            print(f"   Run: ollama pull {self.model}")
        else:
            print(f"✅ Model '{self.model}' is loaded")

        # Register with registry (non-blocking)
        try:
            await self.register_with_registry()
        except Exception as e:
            print(f"⚠️ Registry registration skipped: {e}")

    async def shutdown(self):
        """
        Service shutdown hook
        Override in subclasses for custom cleanup
        """
        await self.ollama_client.close()
        print(f"✅ {self.service_name} shut down gracefully")

"""
Base Cloud Provider Adapter
Abstract base class for all cloud LLM provider wrappers
"""

from abc import ABC, abstractmethod
from fastapi import FastAPI, HTTPException
import httpx
import time
from typing import List, Optional
from .schemas import (
    ChatCompletionRequest, ChatCompletionResponse,
    HealthResponse, HealthStatus, ServiceInfo,
    ModelInfo, PerformanceInfo
)
from .credential_manager import CredentialManager


class BaseCloudAdapter(ABC):
    """
    Abstract base class for cloud LLM provider adapters

    Responsibilities:
    - Wrap cloud provider API
    - Provide OpenAI-compatible endpoints
    - Auto-register with Provider Router
    - Load credentials from .env
    - Health checks and error handling
    """

    def __init__(
        self,
        provider_name: str,
        service_name: str,
        model: str,
        port: int,
        api_key_env_var: str,
        capabilities: List[str],
        context_window: int,
        cost_per_input_token: float,
        cost_per_output_token: float,
        version: str = "1.0.0",
        api_base_url: Optional[str] = None
    ):
        """
        Initialize base cloud adapter

        Args:
            provider_name: Provider identifier (e.g., "openai", "anthropic")
            service_name: Human-readable service name
            model: Model identifier (e.g., "gpt-5-codex")
            port: Service port number
            api_key_env_var: Environment variable name for API key
            capabilities: List of agent capabilities
            context_window: Maximum context window in tokens
            cost_per_input_token: Cost per 1k input tokens (USD)
            cost_per_output_token: Cost per 1k output tokens (USD)
            version: Service version
            api_base_url: Optional custom API base URL
        """
        self.provider_name = provider_name
        self.service_name = service_name
        self.model = model
        self.port = port
        self.api_key_env_var = api_key_env_var
        self.capabilities = capabilities
        self.context_window = context_window
        self.cost_per_input_token = cost_per_input_token
        self.cost_per_output_token = cost_per_output_token
        self.version = version
        self.api_base_url = api_base_url

        self.start_time = time.time()
        self.last_request_time: Optional[float] = None

        # Load credentials
        self.credential_manager = CredentialManager()
        self.api_key = self._load_api_key()

        # Initialize FastAPI app
        self.app = FastAPI(
            title=service_name,
            description=f"Cloud LLM adapter for {provider_name} ({model})",
            version=version
        )

        # HTTP client for provider API
        self.http_client = httpx.AsyncClient(timeout=60.0)

        # Setup routes
        self._setup_routes()

    def _load_api_key(self) -> str:
        """
        Load API key from environment

        Returns:
            API key string

        Raises:
            ValueError: If API key not found
        """
        return self.credential_manager.get_key(
            self.api_key_env_var,
            required=True,
            provider_name=self.provider_name
        )

    def _setup_routes(self):
        """Setup FastAPI routes"""
        self.app.post("/chat/completions", response_model=ChatCompletionResponse)(
            self._chat_completions
        )
        self.app.get("/health")(self._health)
        self.app.get("/info")(self._info)
        self.app.get("/")(self._root)

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

            # Call provider-specific implementation
            response = await self._call_provider_api(request)
            return response

        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"{self.provider_name} API error: {e.response.text}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: {str(e)}"
            )

    @abstractmethod
    async def _call_provider_api(
        self, request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Call provider-specific API

        Must be implemented by subclasses to call actual cloud provider API

        Args:
            request: Chat completion request

        Returns:
            Chat completion response
        """
        pass

    async def _health(self) -> HealthResponse:
        """
        Health check endpoint

        Returns:
            Health status
        """
        try:
            # Check API key presence
            api_key_present = bool(self.api_key)

            # Optionally ping provider API (implement in subclass if needed)
            status = HealthStatus.HEALTHY if api_key_present else HealthStatus.UNHEALTHY

            uptime = time.time() - self.start_time
            last_request_ms = (
                (time.time() - self.last_request_time) * 1000
                if self.last_request_time
                else None
            )

            return HealthResponse(
                status=status,
                service=f"{self.provider_name}-{self.model}",
                port=self.port,
                provider=self.provider_name,
                model=self.model,
                api_key_present=api_key_present,
                uptime_seconds=uptime,
                last_request_ms=last_request_ms
            )

        except Exception as e:
            return HealthResponse(
                status=HealthStatus.UNHEALTHY,
                service=f"{self.provider_name}-{self.model}",
                port=self.port,
                provider=self.provider_name,
                model=self.model,
                api_key_present=False,
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
            provider=self.provider_name,
            model=ModelInfo(
                name=self.model,
                provider=self.provider_name,
                context_window=self.context_window,
                cost_per_input_token=self.cost_per_input_token,
                cost_per_output_token=self.cost_per_output_token,
                capabilities=self.capabilities
            ),
            performance=PerformanceInfo(
                avg_latency_ms=None,
                p95_latency_ms=None
            ),
            endpoints=[
                "/chat/completions",
                "/health",
                "/info",
                "/docs"
            ]
        )

    async def _root(self):
        """Root endpoint with service info"""
        return {
            "service": self.service_name,
            "version": self.version,
            "provider": self.provider_name,
            "model": self.model,
            "endpoints": {
                "chat": "POST /chat/completions",
                "health": "GET /health",
                "info": "GET /info",
                "docs": "GET /docs"
            }
        }

    async def register_with_router(
        self,
        router_url: str = "http://localhost:6103"
    ):
        """
        Register provider with Provider Router

        Args:
            router_url: Provider Router URL
        """
        registration = {
            "name": f"{self.provider_name}-{self.model.replace(':', '-').replace('.', '-')}",
            "model": self.model,
            "context_window": self.context_window,
            "cost_per_input_token": self.cost_per_input_token,
            "cost_per_output_token": self.cost_per_output_token,
            "endpoint": f"http://localhost:{self.port}",
            "features": self.capabilities,
            "slo": None,
            "metadata": {
                "provider": self.provider_name,
                "version": self.version,
                "api_base_url": self.api_base_url or "default"
            }
        }

        try:
            response = await self.http_client.post(
                f"{router_url}/register",
                json=registration,
                timeout=10.0
            )
            if response.status_code == 200:
                print(f"✅ Registered {registration['name']} with Provider Router")
            else:
                print(f"⚠️ Registration failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Provider Router connection failed: {e}")

    async def startup(self):
        """
        Service startup hook
        Override in subclasses for custom initialization
        """
        # Validate API key
        if not self.api_key:
            print(f"⚠️ Warning: {self.api_key_env_var} not set")
        else:
            masked_key = CredentialManager.mask_key(self.api_key)
            print(f"✅ API key loaded: {masked_key}")

        # Register with Provider Router (non-blocking)
        try:
            await self.register_with_router()
        except Exception as e:
            print(f"⚠️ Provider Router registration skipped: {e}")

    async def shutdown(self):
        """
        Service shutdown hook
        Override in subclasses for custom cleanup
        """
        await self.http_client.aclose()
        print(f"✅ {self.service_name} shut down gracefully")

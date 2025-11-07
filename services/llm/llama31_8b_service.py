"""
Llama 3.1 8B Wrapper Service (Port 8050)
General-purpose reasoning and complex task processing
"""

import uvicorn
from .common.base_llm_service import BaseLLMService
from .common.schemas import ModelInfo, PerformanceInfo


class Llama31Service(BaseLLMService):
    """Llama 3.1 8B wrapper service"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        super().__init__(
            service_name="Llama 3.1 8B Service",
            model="llama3.1:8b",
            port=8050,
            capabilities=[
                "reasoning",
                "planning",
                "code_review",
                "explanation",
                "general_completion"
            ],
            ollama_url=ollama_url,
            agent_id="llm_llama31_8b",
            version="1.0.0"
        )

    def _get_model_info(self) -> ModelInfo:
        """Get Llama 3.1 8B model metadata"""
        return ModelInfo(
            name="llama3.1:8b",
            parameters="8B",
            quantization="Q4_K_M",
            context_length=8192,
            embedding_dim=4096
        )

    def _get_performance_info(self) -> PerformanceInfo:
        """Get performance characteristics"""
        return PerformanceInfo(
            avg_throughput_tok_s=73.0,  # M4 Max benchmark
            avg_latency_ms=None,
            p95_latency_ms=None
        )


# Create service instance
service = Llama31Service()
app = service.app


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting Llama 3.1 8B Service (Port 8050)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("Llama 3.1 8B Service Ready")
    print("  URL: http://localhost:8050")
    print("  Model: llama3.1:8b")
    print("  Capabilities: reasoning, planning, code_review")
    print("  OpenAPI Docs: http://localhost:8050/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    print("\nShutting down Llama 3.1 8B Service...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.llm.llama31_8b_service:app",
        host="127.0.0.1",
        port=8050,
        log_level="info"
    )

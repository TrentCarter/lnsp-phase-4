"""
TinyLlama 1.1B Wrapper Service (Port 8051)
Fast, lightweight classification and simple tasks
"""

import uvicorn
from .common.base_llm_service import BaseLLMService
from .common.schemas import ModelInfo, PerformanceInfo


class TinyLlamaService(BaseLLMService):
    """TinyLlama 1.1B wrapper service"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        super().__init__(
            service_name="TinyLlama 1.1B Service",
            model="tinyllama:1.1b",
            port=8051,
            capabilities=[
                "classification",
                "tagging",
                "extraction",
                "filtering",
                "simple_completion"
            ],
            ollama_url=ollama_url,
            agent_id="llm_tinyllama_1b",
            version="1.0.0"
        )

    def _get_model_info(self) -> ModelInfo:
        """Get TinyLlama 1.1B model metadata"""
        return ModelInfo(
            name="tinyllama:1.1b",
            parameters="1.1B",
            quantization="Q4_0",
            context_length=2048,
            embedding_dim=2048
        )

    def _get_performance_info(self) -> PerformanceInfo:
        """Get performance characteristics"""
        return PerformanceInfo(
            avg_throughput_tok_s=277.0,  # M4 Max benchmark (3.8x faster than Llama 3.1)
            avg_latency_ms=None,
            p95_latency_ms=None
        )


# Create service instance
service = TinyLlamaService()
app = service.app


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting TinyLlama 1.1B Service (Port 8051)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("TinyLlama 1.1B Service Ready")
    print("  URL: http://localhost:8051")
    print("  Model: tinyllama:1.1b")
    print("  Capabilities: classification, tagging, extraction")
    print("  Performance: 277 tok/s (3.8x faster than Llama 3.1)")
    print("  OpenAPI Docs: http://localhost:8051/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    print("\nShutting down TinyLlama 1.1B Service...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.llm.tinyllama_service:app",
        host="127.0.0.1",
        port=8051,
        log_level="info"
    )

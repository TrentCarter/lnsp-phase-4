"""
TLC Domain Classifier Service (Port 8052)
Specialized TMD (Task-Method-Domain) domain classification
Uses TinyLlama backend for fast classification
"""

import uvicorn
import json
from fastapi import HTTPException
from .common.base_llm_service import BaseLLMService
from .common.schemas import (
    ModelInfo, PerformanceInfo,
    DomainClassificationRequest, DomainClassificationResponse, DomainScore,
    TMDExtractionRequest, TMDExtractionResponse, TMD, TMDConfidence,
    TaskType, MethodType
)
from typing import List
import time


# Domain taxonomy from TMD-LS PRD
DOMAINS = [
    "FACTOIDWIKI",
    "ARTIFICIAL_INTELLIGENCE",
    "MEDICINE",
    "BIOLOGY",
    "PHYSICS",
    "CHEMISTRY",
    "MATHEMATICS",
    "COMPUTER_SCIENCE",
    "ENGINEERING",
    "HISTORY",
    "LITERATURE",
    "PHILOSOPHY",
    "LAW",
    "ECONOMICS",
    "BUSINESS",
    "PSYCHOLOGY",
    "SOCIOLOGY",
    "GEOGRAPHY",
    "SPORTS",
    "ENTERTAINMENT",
    "OTHER"
]


class TLCClassifierService(BaseLLMService):
    """TLC Domain Classifier service"""

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        super().__init__(
            service_name="TLC Domain Classifier",
            model="tinyllama:1.1b",
            port=8052,
            capabilities=[
                "domain_classification",
                "tmd_extraction",
                "domain_validation"
            ],
            ollama_url=ollama_url,
            agent_id="tlc_domain_classifier",
            version="1.0.0"
        )

        # Add custom routes for TLC-specific endpoints
        self.app.post("/classify_domain", response_model=DomainClassificationResponse)(
            self._classify_domain
        )
        self.app.post("/extract_tmd", response_model=TMDExtractionResponse)(
            self._extract_tmd
        )

    def _get_model_info(self) -> ModelInfo:
        """Get model metadata"""
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
            avg_throughput_tok_s=277.0,
            avg_latency_ms=None,
            p95_latency_ms=None
        )

    async def _classify_domain(
        self, request: DomainClassificationRequest
    ) -> DomainClassificationResponse:
        """
        Classify query domain using TinyLlama

        Args:
            request: Domain classification request

        Returns:
            Domain classification response with top-k domains
        """
        start_time = time.time()

        try:
            # Build context from docs if provided
            context_str = ""
            if request.context_docs:
                context_str = "\n\nContext documents:\n" + "\n".join(
                    f"- {doc[:200]}" for doc in request.context_docs[:3]
                )

            # Build prompt for domain classification
            system_prompt = f"""You are a domain classification expert. Given a query, classify it into one or more of these domains:

{', '.join(DOMAINS)}

Return a JSON object with the top {request.top_k} most relevant domains and their confidence scores (0-1).

Format:
{{
  "domains": [
    {{"domain": "DOMAIN_NAME", "confidence": 0.95}},
    ...
  ]
}}

Only return valid JSON, no other text."""

            user_prompt = f"Query: {request.query}{context_str}"

            # Call LLM
            result = await self.ollama_client.extract_with_prompt(
                model=self.model,
                system_prompt=system_prompt,
                user_query=user_prompt,
                temperature=0.0,
                format_json=True
            )

            # Parse LLM response
            domains_data = result.get("domains", [])
            if not domains_data:
                # Fallback: classify as FACTOIDWIKI
                domains_data = [{"domain": "FACTOIDWIKI", "confidence": 0.5}]

            # Validate domains against taxonomy
            validated_domains = []
            for d in domains_data[:request.top_k]:
                domain_name = d.get("domain", "OTHER").upper()
                if domain_name not in DOMAINS:
                    domain_name = "OTHER"
                validated_domains.append(
                    DomainScore(
                        domain=domain_name,
                        confidence=min(1.0, max(0.0, d.get("confidence", 0.5)))
                    )
                )

            # Sort by confidence
            validated_domains.sort(key=lambda x: x.confidence, reverse=True)

            processing_time = (time.time() - start_time) * 1000

            return DomainClassificationResponse(
                query=request.query,
                domains=validated_domains,
                primary_domain=validated_domains[0].domain,
                metadata={
                    "processing_time_ms": processing_time,
                    "model_used": self.model
                }
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Classification failed: {str(e)}"
            )

    async def _extract_tmd(
        self, request: TMDExtractionRequest
    ) -> TMDExtractionResponse:
        """
        Extract Task-Method-Domain triplet from query

        Args:
            request: TMD extraction request

        Returns:
            TMD extraction response
        """
        start_time = time.time()

        try:
            # Build context from docs if provided
            context_str = ""
            if request.context_docs:
                context_str = "\n\nContext documents:\n" + "\n".join(
                    f"- {doc[:200]}" for doc in request.context_docs[:3]
                )

            # Build prompt for TMD extraction
            system_prompt = f"""You are a query analysis expert. Extract the Task-Method-Domain (TMD) triplet from a query.

Task types: RETRIEVE, ANSWER, VERIFY, COMPARE, SUMMARIZE
Method types: DENSE, SPARSE, HYBRID, GRAPH
Domains: {', '.join(DOMAINS)}

Return a JSON object:
{{
  "task": "TASK_TYPE",
  "method": "METHOD_TYPE",
  "domain": "DOMAIN_NAME",
  "confidence": {{
    "task": 0.95,
    "method": 0.88,
    "domain": 0.92
  }}
}}

Only return valid JSON, no other text."""

            method_hint = request.method_hint.value if request.method_hint else "HYBRID"
            user_prompt = f"Query: {request.query}\nSuggested method: {method_hint}{context_str}"

            # Call LLM
            result = await self.ollama_client.extract_with_prompt(
                model=self.model,
                system_prompt=system_prompt,
                user_query=user_prompt,
                temperature=0.0,
                format_json=True
            )

            # Parse LLM response
            task_str = result.get("task", "RETRIEVE").upper()
            method_str = result.get("method", method_hint).upper()
            domain_str = result.get("domain", "FACTOIDWIKI").upper()

            # Validate task
            try:
                task = TaskType(task_str)
            except ValueError:
                task = TaskType.RETRIEVE

            # Validate method
            try:
                method = MethodType(method_str)
            except ValueError:
                method = MethodType.HYBRID

            # Validate domain
            if domain_str not in DOMAINS:
                domain_str = "FACTOIDWIKI"

            # Extract confidence scores
            confidence_data = result.get("confidence", {})
            confidence = TMDConfidence(
                task=min(1.0, max(0.0, confidence_data.get("task", 0.7))),
                method=min(1.0, max(0.0, confidence_data.get("method", 0.7))),
                domain=min(1.0, max(0.0, confidence_data.get("domain", 0.7)))
            )

            processing_time = (time.time() - start_time) * 1000

            return TMDExtractionResponse(
                query=request.query,
                tmd=TMD(
                    task=task,
                    method=method,
                    domain=domain_str
                ),
                confidence=confidence,
                metadata={
                    "processing_time_ms": processing_time,
                    "model_used": self.model,
                    "llm_raw_response": json.dumps(result)
                }
            )

        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"TMD extraction failed: {str(e)}"
            )

    def _get_model_info(self) -> ModelInfo:
        """Override to show TLC-specific info"""
        return ModelInfo(
            name="tinyllama:1.1b",
            parameters="1.1B (TLC Specialist)",
            quantization="Q4_0",
            context_length=2048,
            embedding_dim=2048
        )


# Create service instance
service = TLCClassifierService()
app = service.app


# Override info endpoint to include TLC-specific endpoints
@app.get("/info")
async def info():
    """Service info with TLC-specific endpoints"""
    base_info = await service._info()
    base_info.endpoints.extend([
        "/classify_domain",
        "/extract_tmd"
    ])
    return base_info


# Lifecycle events
@app.on_event("startup")
async def startup_event():
    """Service startup"""
    print("=" * 60)
    print("Starting TLC Domain Classifier (Port 8052)")
    print("=" * 60)
    await service.startup()
    print("=" * 60)
    print("TLC Domain Classifier Ready")
    print("  URL: http://localhost:8052")
    print("  Model: tinyllama:1.1b (TLC Specialist)")
    print("  Capabilities: domain_classification, tmd_extraction")
    print(f"  Domains: {len(DOMAINS)} supported")
    print("  OpenAPI Docs: http://localhost:8052/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Service shutdown"""
    print("\nShutting down TLC Domain Classifier...")
    await service.shutdown()


if __name__ == "__main__":
    uvicorn.run(
        "services.llm.tlc_classifier_service:app",
        host="127.0.0.1",
        port=8052,
        log_level="info"
    )

#!/usr/bin/env python3
"""
TMD Router FastAPI Service

Exposes TMD extraction and lane routing as RESTful API.
Routes concepts to appropriate lane specialist based on Domain/Task/Modifier codes.
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.tmd_router import (
    route_concept,
    select_lane,
    get_cache_stats,
    clear_cache,
    get_lane_prompt
)

app = FastAPI(
    title="TMD Router API",
    description="Domain/Task/Modifier extraction and lane specialist routing",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class RouteRequest(BaseModel):
    """Request for routing a concept"""
    concept_text: str = Field(..., description="Concept text to route")
    use_cache: bool = Field(default=True, description="Use TMD cache")
    llm_endpoint: Optional[str] = Field(default=None, description="LLM endpoint (defaults to env)")
    llm_model: str = Field(default="qwen2.5:1.5b", description="LLM model for TMD extraction")


class RouteResponse(BaseModel):
    """Response with routing information"""
    concept_text: str
    domain_code: int
    domain_name: str
    task_code: int
    modifier_code: int
    lane_model: str
    lane_port: int
    specialist_prompt_id: str
    temperature: float
    max_tokens: int
    is_fallback: bool
    cache_hit: bool


class BatchRouteRequest(BaseModel):
    """Request for routing multiple concepts"""
    concepts: List[str] = Field(..., description="List of concepts to route")
    use_cache: bool = Field(default=True, description="Use TMD cache")
    llm_endpoint: Optional[str] = Field(default=None, description="LLM endpoint")
    llm_model: str = Field(default="qwen2.5:1.5b", description="LLM model")


class BatchRouteResponse(BaseModel):
    """Response with batch routing results"""
    results: List[RouteResponse]
    count: int


class ExtractTMDRequest(BaseModel):
    """Request for TMD extraction only (no lane routing)"""
    concept_text: str = Field(..., description="Concept text to analyze")
    llm_endpoint: Optional[str] = Field(default=None, description="LLM endpoint")
    llm_model: str = Field(default="qwen2.5:1.5b", description="LLM model")


class ExtractTMDResponse(BaseModel):
    """Response with TMD codes only"""
    concept_text: str
    domain_code: int
    domain_name: str
    task_code: int
    modifier_code: int


class SelectLaneRequest(BaseModel):
    """Request for lane selection from TMD codes"""
    domain_code: int = Field(..., ge=0, le=15, description="Domain code (0-15)")
    task_code: int = Field(..., ge=0, le=31, description="Task code (0-31)")
    modifier_code: int = Field(..., ge=0, le=63, description="Modifier code (0-63)")
    allow_fallback: bool = Field(default=True, description="Allow fallback to Llama 3.1")


class SelectLaneResponse(BaseModel):
    """Response with lane selection"""
    domain_code: int
    domain_name: str
    task_code: int
    modifier_code: int
    lane_model: str
    lane_port: int
    specialist_prompt_id: str
    temperature: float
    max_tokens: int
    is_fallback: bool


class CacheStatsResponse(BaseModel):
    """Cache statistics"""
    size: int
    maxsize: int
    hits: int
    misses: int
    hit_rate: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    cache_enabled: bool
    default_llm_endpoint: str
    default_llm_model: str


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/", tags=["info"])
async def root():
    """API information"""
    return {
        "name": "TMD Router API",
        "version": "1.0.0",
        "description": "Domain/Task/Modifier extraction and lane specialist routing",
        "endpoints": [
            "POST /route - Route concept to lane specialist",
            "POST /route/batch - Batch routing",
            "POST /extract-tmd - Extract TMD codes only",
            "POST /select-lane - Select lane from TMD codes",
            "GET /cache/stats - Cache statistics",
            "DELETE /cache - Clear cache",
            "GET /health - Health check"
        ]
    }


@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        cache_enabled=True,
        default_llm_endpoint=os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434"),
        default_llm_model=os.getenv("LNSP_LLM_MODEL", "qwen2.5:1.5b")
    )


@app.post("/route", response_model=RouteResponse, tags=["routing"])
async def route_concept_endpoint(request: RouteRequest):
    """
    Route concept to appropriate lane specialist.

    Extracts Domain/Task/Modifier codes and selects the best lane specialist model.
    """
    try:
        result = route_concept(
            concept_text=request.concept_text,
            use_cache=request.use_cache,
            llm_endpoint=request.llm_endpoint,
            llm_model=request.llm_model
        )

        return RouteResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Routing failed: {str(e)}"
        )


@app.post("/route/batch", response_model=BatchRouteResponse, tags=["routing"])
async def batch_route_concepts(request: BatchRouteRequest):
    """
    Route multiple concepts in batch.

    More efficient than individual calls when routing many concepts.
    """
    try:
        results = []
        for concept in request.concepts:
            result = route_concept(
                concept_text=concept,
                use_cache=request.use_cache,
                llm_endpoint=request.llm_endpoint,
                llm_model=request.llm_model
            )
            results.append(RouteResponse(**result))

        return BatchRouteResponse(
            results=results,
            count=len(results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch routing failed: {str(e)}"
        )


@app.post("/extract-tmd", response_model=ExtractTMDResponse, tags=["tmd"])
async def extract_tmd_endpoint(request: ExtractTMDRequest):
    """
    Extract TMD codes without lane routing.

    Useful for analysis or manual lane selection.
    """
    try:
        result = route_concept(
            concept_text=request.concept_text,
            use_cache=False,  # Don't cache since we're not using lane info
            llm_endpoint=request.llm_endpoint,
            llm_model=request.llm_model
        )

        return ExtractTMDResponse(
            concept_text=result['concept_text'],
            domain_code=result['domain_code'],
            domain_name=result['domain_name'],
            task_code=result['task_code'],
            modifier_code=result['modifier_code']
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TMD extraction failed: {str(e)}"
        )


@app.post("/select-lane", response_model=SelectLaneResponse, tags=["routing"])
async def select_lane_endpoint(request: SelectLaneRequest):
    """
    Select lane specialist from TMD codes.

    Useful when you already have TMD codes and just need lane selection.
    """
    try:
        lane_info = select_lane(
            domain_code=request.domain_code,
            task_code=request.task_code,
            modifier_code=request.modifier_code,
            allow_fallback=request.allow_fallback
        )

        return SelectLaneResponse(
            domain_code=request.domain_code,
            domain_name=lane_info['domain_name'],
            task_code=request.task_code,
            modifier_code=request.modifier_code,
            lane_model=lane_info['model'],
            lane_port=lane_info['port'],
            specialist_prompt_id=lane_info['specialist_prompt_id'],
            temperature=lane_info['temperature'],
            max_tokens=lane_info['max_tokens'],
            is_fallback=lane_info['is_fallback']
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lane selection failed: {str(e)}"
        )


@app.get("/cache/stats", response_model=CacheStatsResponse, tags=["cache"])
async def cache_stats_endpoint():
    """Get TMD cache statistics"""
    stats = get_cache_stats()
    return CacheStatsResponse(**stats)


@app.delete("/cache", tags=["cache"])
async def clear_cache_endpoint():
    """Clear TMD cache"""
    clear_cache()
    return {"status": "success", "message": "Cache cleared"}


@app.get("/prompt/{specialist_prompt_id}", tags=["prompts"])
async def get_prompt_endpoint(specialist_prompt_id: str):
    """Get prompt template for a lane specialist"""
    prompt = get_lane_prompt(specialist_prompt_id)

    if prompt is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt not found: {specialist_prompt_id}"
        )

    return {
        "specialist_prompt_id": specialist_prompt_id,
        "template": prompt
    }


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup():
    """Initialize service on startup"""
    print("🚀 TMD Router API starting...")
    print(f"   LLM Endpoint: {os.getenv('LNSP_LLM_ENDPOINT', 'http://localhost:11434')}")
    print(f"   LLM Model: {os.getenv('LNSP_LLM_MODEL', 'llama3.1:8b')}")
    print("✅ TMD Router API ready")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    print("👋 TMD Router API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8002,
        log_level="info"
    )

#!/usr/bin/env python3
"""
Episode Chunker FastAPI Service

Converts long documents into coherent episodes based on semantic similarity.
Detects low-coherence transitions and splits into separate episodes.

Port: 8900
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import requests

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

app = FastAPI(
    title="Episode Chunker API",
    description="Converts documents into coherent episodes with sequence metadata",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Models
# ============================================================================

class Episode(BaseModel):
    """A coherent span of text"""
    episode_id: str
    text: str
    start_index: int
    end_index: int
    coherence_score: float
    chunk_count: int


class EpisodeChunkRequest(BaseModel):
    """Request to chunk document into episodes"""
    document_id: str = Field(..., description="Document identifier (e.g., 'wikipedia_12345')")
    text: str = Field(..., description="Document text to chunk")
    coherence_threshold: float = Field(0.6, description="Coherence threshold (0.0-1.0)", ge=0.0, le=1.0)
    min_episode_length: int = Field(3, description="Minimum chunks per episode", ge=1)
    max_episode_length: int = Field(20, description="Maximum chunks per episode", ge=1)
    embedding_api: Optional[str] = Field(None, description="Embedding API URL (default: localhost:8767)")


class EpisodeChunkResponse(BaseModel):
    """Response with episodes"""
    document_id: str
    episodes: List[Episode]
    total_episodes: int
    total_chunks: int
    avg_coherence: float


# ============================================================================
# Episode Chunker Logic
# ============================================================================

def get_embeddings(texts: List[str], api_url: str) -> np.ndarray:
    """Get embeddings from GTR-T5 API"""
    response = requests.post(
        f"{api_url}/embed",
        json={"texts": texts},
        timeout=60
    )
    response.raise_for_status()
    return np.array(response.json()["embeddings"], dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs (simple chunking for episode detection)"""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    if not paragraphs:
        # Fallback: split by sentences
        paragraphs = [s.strip() for s in text.split('. ') if s.strip()]
    return paragraphs


def build_episodes(
    chunks: List[str],
    embeddings: np.ndarray,
    coherence_threshold: float,
    min_len: int,
    max_len: int,
    document_id: str
) -> List[Episode]:
    """Build episodes from chunks and embeddings"""

    if len(chunks) < min_len:
        # Single episode for short documents
        return [Episode(
            episode_id=f"{document_id}_ep0",
            text='\n\n'.join(chunks),
            start_index=0,
            end_index=len(chunks) - 1,
            coherence_score=1.0,
            chunk_count=len(chunks)
        )]

    episodes = []
    episode_start = 0
    episode_id = 0

    for i in range(1, len(chunks)):
        # Compute coherence with previous chunk
        similarity = cosine_similarity(embeddings[i-1], embeddings[i])

        # Check if we should split here
        should_split = (
            similarity < coherence_threshold or
            (i - episode_start) >= max_len
        )

        if should_split and (i - episode_start) >= min_len:
            # Create episode
            episode_chunks = chunks[episode_start:i]
            episode_text = '\n\n'.join(episode_chunks)

            # Compute average coherence for episode
            coherences = []
            for j in range(episode_start + 1, i):
                coherences.append(cosine_similarity(embeddings[j-1], embeddings[j]))
            avg_coherence = float(np.mean(coherences)) if coherences else 1.0

            episodes.append(Episode(
                episode_id=f"{document_id}_ep{episode_id}",
                text=episode_text,
                start_index=episode_start,
                end_index=i - 1,
                coherence_score=avg_coherence,
                chunk_count=len(episode_chunks)
            ))

            episode_start = i
            episode_id += 1

    # Final episode
    if episode_start < len(chunks):
        episode_chunks = chunks[episode_start:]
        episode_text = '\n\n'.join(episode_chunks)

        coherences = []
        for j in range(episode_start + 1, len(chunks)):
            coherences.append(cosine_similarity(embeddings[j-1], embeddings[j]))
        avg_coherence = float(np.mean(coherences)) if coherences else 1.0

        episodes.append(Episode(
            episode_id=f"{document_id}_ep{episode_id}",
            text=episode_text,
            start_index=episode_start,
            end_index=len(chunks) - 1,
            coherence_score=avg_coherence,
            chunk_count=len(episode_chunks)
        ))

    return episodes


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "name": "Episode Chunker API",
        "version": "1.0.0",
        "description": "Converts documents into coherent episodes",
        "endpoints": [
            "POST /chunk - Chunk document into episodes",
            "GET /health - Health check"
        ]
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/chunk", response_model=EpisodeChunkResponse)
async def chunk_into_episodes(request: EpisodeChunkRequest):
    """
    Chunk document into coherent episodes.

    1. Splits document into paragraphs
    2. Embeds paragraphs (via GTR-T5 API)
    3. Detects low-coherence transitions
    4. Returns episodes with metadata
    """

    # Default embedding API
    embedding_api = request.embedding_api or "http://localhost:8767"

    try:
        # Step 1: Split into paragraphs
        chunks = split_into_paragraphs(request.text)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks extracted from text")

        # Step 2: Get embeddings
        embeddings = get_embeddings(chunks, embedding_api)

        # Step 3: Build episodes
        episodes = build_episodes(
            chunks=chunks,
            embeddings=embeddings,
            coherence_threshold=request.coherence_threshold,
            min_len=request.min_episode_length,
            max_len=request.max_episode_length,
            document_id=request.document_id
        )

        # Step 4: Compute stats
        total_chunks = sum(ep.chunk_count for ep in episodes)
        avg_coherence = float(np.mean([ep.coherence_score for ep in episodes]))

        return EpisodeChunkResponse(
            document_id=request.document_id,
            episodes=episodes,
            total_episodes=len(episodes),
            total_chunks=total_chunks,
            avg_coherence=avg_coherence
        )

    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Embedding API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Episode chunking failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8900)

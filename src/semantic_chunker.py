#!/usr/bin/env python3
"""
Semantic Chunker: Concept-based text chunking for TMD-LS pipeline.

This module provides three chunking strategies:
1. Semantic: Embedding-based semantic boundary detection (fast, GTR-T5)
2. Proposition: LLM-based atomic proposition extraction (high-quality, slow)
3. Hybrid: Semantic splitting + optional proposition refinement

Integration with TMD-LS:
- Chunks are routed to TMD router for domain/task/modifier extraction
- Chunk size optimized for CPESH generation (180-320 words recommended)
- Preserves semantic coherence for accurate domain routing
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

import numpy as np
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Import existing LNSP components
try:
    from src.vectorizer import EmbeddingBackend
except ImportError:
    # Fallback for testing
    EmbeddingBackend = None

try:
    from src.llm.local_llama_client import LocalLlamaClient
except ImportError:
    LocalLlamaClient = None


logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a semantic chunk of text."""
    text: str
    chunk_id: str
    chunk_index: int
    word_count: int
    char_count: int
    chunking_mode: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class SemanticChunker:
    """
    Embedding-based semantic chunker using LlamaIndex SemanticSplitter.

    Uses GTR-T5 embeddings (same as LNSP vecRAG) to find natural semantic
    boundaries between sentences.

    Parameters:
        embed_model_name: HuggingFace model name (default: gtr-t5-base)
        buffer_size: Number of sentences before/after to compare (default: 1)
        breakpoint_percentile_threshold: Similarity threshold for splits (default: 95)
        min_chunk_size: Minimum characters per chunk (default: 500)
    """

    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/gtr-t5-base",
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        min_chunk_size: int = 500
    ):
        self.embed_model_name = embed_model_name
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.min_chunk_size = min_chunk_size

        # Initialize LlamaIndex embedding model
        logger.info(f"Initializing SemanticChunker with {embed_model_name}")
        self.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

        # Initialize semantic splitter
        self.splitter = SemanticSplitterNodeParser(
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
            embed_model=self.embed_model
        )

        logger.info("SemanticChunker initialized successfully")

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Split text into semantic chunks.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects with semantic boundaries
        """
        if not text or not text.strip():
            return []

        # Create LlamaIndex document
        doc = Document(text=text, metadata=metadata or {})

        # Split into semantic chunks
        nodes = self.splitter.get_nodes_from_documents([doc])

        # Convert to Chunk objects
        chunks = []
        for idx, node in enumerate(nodes):
            chunk_text = node.get_content()

            # Filter out tiny chunks
            if len(chunk_text) < self.min_chunk_size:
                logger.debug(f"Skipping tiny chunk ({len(chunk_text)} chars): {chunk_text[:50]}...")
                continue

            chunk = Chunk(
                text=chunk_text,
                chunk_id=self._generate_chunk_id(chunk_text, idx),
                chunk_index=idx,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                chunking_mode="semantic",
                metadata={
                    **(metadata or {}),
                    "embedding_model": self.embed_model_name,
                    "buffer_size": self.buffer_size,
                    "breakpoint_threshold": self.breakpoint_percentile_threshold
                }
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} semantic chunks from {len(text)} chars")
        return chunks

    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate stable chunk ID from text content and index."""
        content = f"{index}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class PropositionChunker:
    """
    LLM-based proposition chunker for extracting atomic semantic units.

    Uses local LLM (TinyLlama/Llama) to extract self-contained propositions
    from text. Each proposition is:
    - Atomic: Cannot be further subdivided
    - Self-contained: Includes all necessary context
    - Distinct: Represents a single factoid

    Parameters:
        llm_endpoint: Ollama endpoint (default: http://localhost:11434)
        llm_model: Model name (default: tinyllama:1.1b)
        max_propositions: Max propositions per document (default: 50)
    """

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:11434",
        llm_model: str = "tinyllama:1.1b",
        max_propositions: int = 50
    ):
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.max_propositions = max_propositions

        # Initialize LLM client
        if LocalLlamaClient is None:
            raise ImportError("LocalLlamaClient not available. Check src/llm/local_llama_client.py")

        logger.info(f"Initializing PropositionChunker with {llm_model}")
        self.llm_client = LocalLlamaClient(
            endpoint=llm_endpoint,
            model=llm_model
        )
        logger.info("PropositionChunker initialized successfully")

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Extract atomic propositions from text using LLM.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks

        Returns:
            List of Chunk objects (one per proposition)
        """
        if not text or not text.strip():
            return []

        # Build extraction prompt
        prompt = self._build_extraction_prompt(text)

        # Call LLM using Ollama API
        try:
            import requests
            response = requests.post(
                f"{self.llm_endpoint}/api/generate",
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 2000}
                },
                timeout=30
            )
            response.raise_for_status()

            # Parse JSON response
            response_text = response.json().get("response", "")
            propositions = self._parse_propositions(response_text)

        except Exception as e:
            logger.error(f"LLM proposition extraction failed: {e}")
            return []

        # Convert to Chunk objects
        chunks = []
        for idx, prop_text in enumerate(propositions[:self.max_propositions]):
            chunk = Chunk(
                text=prop_text,
                chunk_id=self._generate_chunk_id(prop_text, idx),
                chunk_index=idx,
                word_count=len(prop_text.split()),
                char_count=len(prop_text),
                chunking_mode="proposition",
                metadata={
                    **(metadata or {}),
                    "llm_model": self.llm_model,
                    "llm_endpoint": self.llm_endpoint
                }
            )
            chunks.append(chunk)

        logger.info(f"Extracted {len(chunks)} propositions from {len(text)} chars")
        return chunks

    def _build_extraction_prompt(self, text: str) -> str:
        """Build LLM prompt for proposition extraction."""
        return f"""Extract atomic semantic propositions from the text below.

**Requirements:**
1. Each proposition must be a **distinct, standalone statement**
2. Each must be **self-contained** with all necessary context
3. Each must be **minimal and indivisible** (cannot be further split)
4. Each must represent a **single factoid or concept**

**Output Format:**
Return a JSON array of strings (one per proposition).

**Example:**
Input: "Photosynthesis is how plants make food. It requires sunlight and water."
Output: ["Photosynthesis is the process by which plants produce food", "Photosynthesis requires sunlight as an energy source", "Photosynthesis requires water as a raw material"]

**Text:**
{text}

**Output (JSON array only, no explanation):**"""

    def _parse_propositions(self, response_text: str) -> List[str]:
        """Parse propositions from LLM response."""
        try:
            # Try to extract JSON array
            if "[" in response_text and "]" in response_text:
                start_idx = response_text.index("[")
                end_idx = response_text.rindex("]") + 1
                json_str = response_text[start_idx:end_idx]
                propositions = json.loads(json_str)

                if isinstance(propositions, list):
                    return [str(p).strip() for p in propositions if p and str(p).strip()]
        except Exception as e:
            logger.warning(f"Failed to parse JSON propositions: {e}")

        # Fallback: split by newlines
        lines = [line.strip() for line in response_text.split("\n") if line.strip()]
        # Filter out non-proposition lines
        propositions = [
            line for line in lines
            if len(line) > 20 and not line.startswith(("{", "[", "Output", "**"))
        ]
        return propositions

    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate stable chunk ID from text content and index."""
        content = f"{index}:{text[:100]}"
        return hashlib.md5(content.encode()).hexdigest()[:16]


class HybridChunker:
    """
    Hybrid chunker combining semantic splitting + proposition refinement.

    Strategy:
    1. Use SemanticChunker for fast initial splitting
    2. For chunks > refine_threshold words, extract propositions
    3. Otherwise keep semantic chunks as-is

    This balances speed (semantic) with quality (proposition) for optimal
    TMD-LS pipeline throughput.

    Parameters:
        refine_threshold: Word count threshold for proposition refinement (default: 150)
        refine_domains: Domain codes to always refine (e.g., Law=11, Medicine=4)
    """

    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/gtr-t5-base",
        llm_endpoint: str = "http://localhost:11434",
        llm_model: str = "tinyllama:1.1b",
        refine_threshold: int = 150,
        refine_domains: Optional[List[int]] = None
    ):
        self.refine_threshold = refine_threshold
        self.refine_domains = refine_domains or []

        # Initialize both chunkers
        logger.info("Initializing HybridChunker with semantic + proposition modes")
        self.semantic_chunker = SemanticChunker(embed_model_name=embed_model_name)
        self.proposition_chunker = PropositionChunker(
            llm_endpoint=llm_endpoint,
            llm_model=llm_model
        )
        logger.info("HybridChunker initialized successfully")

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        force_refine: bool = False
    ) -> List[Chunk]:
        """
        Chunk text using hybrid semantic + proposition strategy.

        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            force_refine: Force proposition refinement for all chunks

        Returns:
            List of Chunk objects (mix of semantic and proposition chunks)
        """
        if not text or not text.strip():
            return []

        # Stage 1: Semantic splitting
        semantic_chunks = self.semantic_chunker.chunk(text, metadata)

        # Stage 2: Selective proposition refinement
        refined_chunks = []
        for chunk in semantic_chunks:
            should_refine = (
                force_refine or
                chunk.word_count > self.refine_threshold or
                metadata and metadata.get("domain_code") in self.refine_domains
            )

            if should_refine:
                # Extract propositions from this chunk
                logger.debug(f"Refining chunk {chunk.chunk_id} ({chunk.word_count} words)")
                propositions = self.proposition_chunker.chunk(chunk.text, chunk.metadata)

                if propositions:
                    # Use propositions instead of original chunk
                    for prop in propositions:
                        prop.metadata["refined_from"] = chunk.chunk_id
                        prop.chunking_mode = "hybrid"
                    refined_chunks.extend(propositions)
                else:
                    # Keep original if proposition extraction failed
                    chunk.chunking_mode = "hybrid"
                    refined_chunks.append(chunk)
            else:
                # Keep semantic chunk as-is
                chunk.chunking_mode = "hybrid"
                refined_chunks.append(chunk)

        # Re-index chunks
        for idx, chunk in enumerate(refined_chunks):
            chunk.chunk_index = idx

        logger.info(
            f"Hybrid chunking: {len(semantic_chunks)} semantic → "
            f"{len(refined_chunks)} final chunks"
        )
        return refined_chunks


def analyze_chunks(chunks: List[Chunk]) -> Dict[str, Any]:
    """
    Analyze chunk statistics for quality checking.

    Returns:
        Dictionary with chunk statistics (word counts, modes, etc.)
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "mean_words": 0,
            "min_words": 0,
            "max_words": 0,
            "p95_words": 0,
            "chunking_modes": {}
        }

    word_counts = [c.word_count for c in chunks]
    word_counts_sorted = sorted(word_counts)

    total = len(word_counts)
    mean_words = sum(word_counts) / total if total > 0 else 0
    p95_index = int(total * 0.95)
    p95_words = word_counts_sorted[p95_index] if p95_index < total else word_counts_sorted[-1]

    # Count chunking modes
    mode_counts = {}
    for chunk in chunks:
        mode = chunk.chunking_mode
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        "total_chunks": total,
        "mean_words": round(mean_words, 1),
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "p95_words": p95_words,
        "chunking_modes": mode_counts,
        "word_distribution": {
            "0-100": sum(1 for w in word_counts if w < 100),
            "100-200": sum(1 for w in word_counts if 100 <= w < 200),
            "200-300": sum(1 for w in word_counts if 200 <= w < 300),
            "300+": sum(1 for w in word_counts if w >= 300)
        }
    }


# CLI test
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    sample_text = """
    Photosynthesis is the process by which plants convert light energy into chemical energy.
    This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
    carbon dioxide as inputs. During photosynthesis, plants absorb light energy through
    chlorophyll molecules. This energy is used to convert carbon dioxide and water into
    glucose and oxygen. The glucose serves as food for the plant, while oxygen is released
    as a byproduct. This process is essential for life on Earth, as it produces the oxygen
    we breathe and forms the base of most food chains. Plants typically perform photosynthesis
    during daylight hours when sunlight is available. The rate of photosynthesis can be
    affected by factors such as light intensity, temperature, and carbon dioxide concentration.
    Different plants have adapted various strategies to maximize their photosynthetic efficiency
    in different environments. Some plants, called C4 plants, have evolved special mechanisms
    to concentrate carbon dioxide, making them more efficient in hot, dry conditions. Others,
    known as CAM plants, open their stomata at night to minimize water loss while still
    maintaining photosynthesis. Understanding photosynthesis is crucial for agriculture,
    climate science, and biotechnology research.
    """

    print("Testing Semantic Chunker")
    print("=" * 60)

    # Test semantic chunking
    semantic_chunker = SemanticChunker()
    semantic_chunks = semantic_chunker.chunk(sample_text)

    print(f"\nSemantic Chunks: {len(semantic_chunks)}")
    for i, chunk in enumerate(semantic_chunks[:3]):  # Show first 3
        print(f"\nChunk {i+1}: {chunk.word_count} words")
        print(f"Text: {chunk.text[:150]}...")

    # Analyze
    stats = analyze_chunks(semantic_chunks)
    print(f"\nStatistics:")
    print(f"  Mean words: {stats['mean_words']}")
    print(f"  Range: {stats['min_words']}-{stats['max_words']} words")
    print(f"  Distribution: {stats['word_distribution']}")

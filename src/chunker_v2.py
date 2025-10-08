#!/usr/bin/env python3
"""
Chunker V2: Advanced text chunking with multiple strategies.

Provides both simple sentence-based chunking and advanced semantic chunking:
- Simple: Fast sentence aggregation (180-320 word targets)
- Semantic: Embedding-based semantic boundary detection (GTR-T5)
- Proposition: LLM-extracted atomic propositions
- Hybrid: Semantic splitting + selective proposition refinement

For TMD-LS pipeline integration, use semantic/proposition/hybrid modes.
For backward compatibility, simple chunking functions are preserved.
"""

import re
import hashlib
import json
import logging
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Optional imports for advanced chunking
try:
    from llama_index.core import Document
    from llama_index.core.node_parser import SemanticSplitterNodeParser
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    logging.warning("LlamaIndex not available. Semantic chunking disabled. Install with: pip install llama-index llama-index-embeddings-huggingface")

try:
    from src.llm.local_llama_client import LocalLlamaClient
    LLM_CLIENT_AVAILABLE = True
except ImportError:
    LLM_CLIENT_AVAILABLE = False
    logging.warning("LocalLlamaClient not available. Proposition chunking disabled.")


logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Data Classes
# ============================================================================

class ChunkingMode(str, Enum):
    """Chunking strategy modes."""
    SIMPLE = "simple"           # Fast sentence aggregation
    SEMANTIC = "semantic"       # Embedding-based semantic boundaries
    PROPOSITION = "proposition" # LLM-extracted propositions
    HYBRID = "hybrid"           # Semantic + proposition refinement


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


# ============================================================================
# Simple Chunking Functions (Backward Compatible)
# ============================================================================

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using improved regex patterns."""
    # Handle common abbreviations and edge cases
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Sr|Jr)\.\s*', r'\1<PERIOD> ', text)
    text = re.sub(r'\b([A-Z])\.\s*([A-Z])\.\s*', r'\1<PERIOD> \2<PERIOD> ', text)
    text = re.sub(r'\.{3,}', '<ELLIPSIS>', text)

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore periods and ellipses
    sentences = [s.replace('<PERIOD>', '.').replace('<ELLIPSIS>', '...') for s in sentences]

    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]

    return sentences


def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def generate_chunk_id(text: str, index: int) -> str:
    """Generate stable chunk ID from text content and index."""
    content = f"{index}:{text[:100]}"  # Use index and first 100 chars
    return hashlib.md5(content.encode()).hexdigest()[:16]


def create_chunks(
    text: str,
    min_words: int = 180,
    max_words: int = 320,
    overlap_sentences: int = 1
) -> List[Dict[str, Any]]:
    """
    Create properly-sized chunks from text using simple sentence aggregation.

    Args:
        text: Input text to chunk
        min_words: Minimum words per chunk (default 180)
        max_words: Maximum words per chunk (default 320)
        overlap_sentences: Number of sentences to overlap between chunks

    Returns:
        List of chunk dictionaries with text, word count, and position
    """
    if not text or not text.strip():
        return []

    sentences = split_into_sentences(text)
    if not sentences:
        return []

    chunks = []
    i = 0
    chunk_index = 0

    while i < len(sentences):
        current_chunk = []
        current_words = 0

        # Build chunk by adding sentences
        j = i
        while j < len(sentences) and current_words < min_words:
            sentence = sentences[j]
            sentence_words = count_words(sentence)

            # Check if adding this sentence would exceed max
            if current_words + sentence_words > max_words and current_chunk:
                # Only break if we already have content
                break

            current_chunk.append(sentence)
            current_words += sentence_words
            j += 1

        # Handle edge cases
        if not current_chunk:
            # Single sentence too long, split it
            if i < len(sentences):
                words = sentences[i].split()
                if len(words) > max_words:
                    # Split long sentence into max_words chunks
                    for k in range(0, len(words), max_words):
                        chunk_words = words[k:k+max_words]
                        chunk_text = ' '.join(chunk_words)
                        chunks.append({
                            'text': chunk_text,
                            'word_count': len(chunk_words),
                            'chunk_index': chunk_index,
                            'start_sentence': i,
                            'end_sentence': i,
                            'chunk_id': generate_chunk_id(chunk_text, chunk_index),
                            'chunking_mode': 'simple'
                        })
                        chunk_index += 1
                    i += 1
                    continue
                else:
                    # Include the sentence even if it's short
                    current_chunk = [sentences[i]]
                    current_words = count_words(sentences[i])
                    j = i + 1

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'word_count': current_words,
                'chunk_index': chunk_index,
                'start_sentence': i,
                'end_sentence': j - 1,
                'chunk_id': generate_chunk_id(chunk_text, chunk_index),
                'chunking_mode': 'simple'
            })
            chunk_index += 1

            # Move forward with overlap
            i = max(i + 1, j - overlap_sentences)
        else:
            i += 1

    return chunks


def merge_short_chunks(chunks: List[Dict[str, Any]], min_words: int = 180, max_words: int = 320) -> List[Dict[str, Any]]:
    """
    Merge consecutive short chunks to reach target size.
    Used for fixing existing data.
    """
    if not chunks:
        return []

    merged = []
    current_buffer = []
    current_words = 0

    for chunk in chunks:
        chunk_words = chunk.get('word_count', count_words(chunk.get('text', '')))

        # If this chunk alone is in range, add it directly
        if min_words <= chunk_words <= max_words:
            # First flush buffer if any
            if current_buffer:
                merged_text = ' '.join([c.get('text', '') for c in current_buffer])
                merged.append({
                    'text': merged_text,
                    'word_count': current_words,
                    'chunk_index': len(merged),
                    'chunk_id': generate_chunk_id(merged_text, len(merged)),
                    'merged_from': [c.get('chunk_id', '') for c in current_buffer]
                })
                current_buffer = []
                current_words = 0

            # Add this chunk as-is
            merged.append(chunk)

        # If adding this would exceed max, flush buffer first
        elif current_words + chunk_words > max_words and current_buffer:
            merged_text = ' '.join([c.get('text', '') for c in current_buffer])
            merged.append({
                'text': merged_text,
                'word_count': current_words,
                'chunk_index': len(merged),
                'chunk_id': generate_chunk_id(merged_text, len(merged)),
                'merged_from': [c.get('chunk_id', '') for c in current_buffer]
            })
            current_buffer = [chunk]
            current_words = chunk_words

        # Otherwise add to buffer
        else:
            current_buffer.append(chunk)
            current_words += chunk_words

            # Check if buffer reached minimum
            if current_words >= min_words:
                merged_text = ' '.join([c.get('text', '') for c in current_buffer])
                merged.append({
                    'text': merged_text,
                    'word_count': current_words,
                    'chunk_index': len(merged),
                    'chunk_id': generate_chunk_id(merged_text, len(merged)),
                    'merged_from': [c.get('chunk_id', '') for c in current_buffer]
                })
                current_buffer = []
                current_words = 0

    # Flush remaining buffer
    if current_buffer:
        merged_text = ' '.join([c.get('text', '') for c in current_buffer])
        merged.append({
            'text': merged_text,
            'word_count': current_words,
            'chunk_index': len(merged),
            'chunk_id': generate_chunk_id(merged_text, len(merged)),
            'merged_from': [c.get('chunk_id', '') for c in current_buffer]
        })

    return merged


def analyze_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze chunk statistics for quality checking."""
    if not chunks:
        return {
            'total_chunks': 0,
            'mean_words': 0,
            'min_words': 0,
            'max_words': 0,
            'p95_words': 0,
            'short_chunks': 0,
            'target_range': 0,
            'chunking_modes': {}
        }

    word_counts = [c.get('word_count', count_words(c.get('text', ''))) for c in chunks]
    word_counts.sort()

    total = len(word_counts)
    mean_words = sum(word_counts) / total if total > 0 else 0
    p95_index = int(total * 0.95)
    p95_words = word_counts[p95_index] if p95_index < total else word_counts[-1]

    short_chunks = sum(1 for w in word_counts if w < 120)
    target_range = sum(1 for w in word_counts if 180 <= w <= 320)

    # Count chunking modes
    mode_counts = {}
    for chunk in chunks:
        mode = chunk.get('chunking_mode', 'unknown')
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        'total_chunks': total,
        'mean_words': round(mean_words, 1),
        'min_words': min(word_counts) if word_counts else 0,
        'max_words': max(word_counts) if word_counts else 0,
        'p95_words': p95_words,
        'short_chunks': short_chunks,
        'short_pct': round(100 * short_chunks / total, 1) if total > 0 else 0,
        'target_range': target_range,
        'target_pct': round(100 * target_range / total, 1) if total > 0 else 0,
        'chunking_modes': mode_counts,
        'word_distribution': {
            '0-100': sum(1 for w in word_counts if w < 100),
            '100-200': sum(1 for w in word_counts if 100 <= w < 200),
            '200-300': sum(1 for w in word_counts if 200 <= w < 300),
            '300+': sum(1 for w in word_counts if w >= 300)
        }
    }


# ============================================================================
# Advanced Semantic Chunking Classes
# ============================================================================

class SemanticChunker:
    """
    Embedding-based semantic chunker using LlamaIndex SemanticSplitter.

    Uses GTR-T5 embeddings (same as LNSP vecRAG) to find natural semantic
    boundaries between sentences.
    """

    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/gtr-t5-base",
        buffer_size: int = 1,
        breakpoint_percentile_threshold: int = 95,
        min_chunk_size: int = 500
    ):
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError("LlamaIndex not available. Install with: pip install llama-index llama-index-embeddings-huggingface")

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

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Split text into semantic chunks."""
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
                chunk_id=generate_chunk_id(chunk_text, idx),
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


class PropositionChunker:
    """
    LLM-based proposition chunker for extracting atomic semantic units.
    """

    def __init__(
        self,
        llm_endpoint: str = "http://localhost:11434",
        llm_model: str = "tinyllama:1.1b",
        max_propositions: int = 50
    ):
        if not LLM_CLIENT_AVAILABLE:
            raise ImportError("LocalLlamaClient not available. Check src/llm/local_llama_client.py")

        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model
        self.max_propositions = max_propositions

        logger.info(f"Initializing PropositionChunker with {llm_model}")
        self.llm_client = LocalLlamaClient(endpoint=llm_endpoint, model=llm_model)
        logger.info("PropositionChunker initialized successfully")

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """Extract atomic propositions from text using LLM."""
        if not text or not text.strip():
            return []

        # Build extraction prompt
        prompt = self._build_extraction_prompt(text)

        # Call LLM
        try:
            response = self.llm_client.chat(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )

            response_text = response.get("message", {}).get("content", "")
            propositions = self._parse_propositions(response_text)

        except Exception as e:
            logger.error(f"LLM proposition extraction failed: {e}")
            return []

        # Convert to Chunk objects
        chunks = []
        for idx, prop_text in enumerate(propositions[:self.max_propositions]):
            chunk = Chunk(
                text=prop_text,
                chunk_id=generate_chunk_id(prop_text, idx),
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
        propositions = [
            line for line in lines
            if len(line) > 20 and not line.startswith(("{", "[", "Output", "**"))
        ]
        return propositions


class HybridChunker:
    """
    Hybrid chunker combining semantic splitting + proposition refinement.
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
        """Chunk text using hybrid semantic + proposition strategy."""
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
                logger.debug(f"Refining chunk {chunk.chunk_id} ({chunk.word_count} words)")
                propositions = self.proposition_chunker.chunk(chunk.text, chunk.metadata)

                if propositions:
                    for prop in propositions:
                        prop.metadata["refined_from"] = chunk.chunk_id
                        prop.chunking_mode = "hybrid"
                    refined_chunks.extend(propositions)
                else:
                    chunk.chunking_mode = "hybrid"
                    refined_chunks.append(chunk)
            else:
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


# ============================================================================
# Unified Chunker Interface
# ============================================================================

class UnifiedChunker:
    """
    Unified interface for all chunking modes.

    Automatically selects the appropriate chunker based on mode.
    """

    def __init__(
        self,
        mode: ChunkingMode = ChunkingMode.SIMPLE,
        embed_model_name: str = "sentence-transformers/gtr-t5-base",
        llm_endpoint: str = "http://localhost:11434",
        llm_model: str = "tinyllama:1.1b"
    ):
        self.mode = mode
        self.embed_model_name = embed_model_name
        self.llm_endpoint = llm_endpoint
        self.llm_model = llm_model

        # Initialize chunkers on-demand
        self._semantic_chunker = None
        self._proposition_chunker = None
        self._hybrid_chunker = None

        logger.info(f"UnifiedChunker initialized with mode: {mode}")

    def chunk(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        min_words: int = 180,
        max_words: int = 320
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using configured mode.

        Returns list of chunk dictionaries for compatibility.
        """
        if self.mode == ChunkingMode.SIMPLE:
            # Use simple sentence-based chunking
            return create_chunks(text, min_words=min_words, max_words=max_words)

        elif self.mode == ChunkingMode.SEMANTIC:
            # Use semantic chunker
            if self._semantic_chunker is None:
                self._semantic_chunker = SemanticChunker(embed_model_name=self.embed_model_name)
            chunks = self._semantic_chunker.chunk(text, metadata)
            return [c.to_dict() for c in chunks]

        elif self.mode == ChunkingMode.PROPOSITION:
            # Use proposition chunker
            if self._proposition_chunker is None:
                self._proposition_chunker = PropositionChunker(
                    llm_endpoint=self.llm_endpoint,
                    llm_model=self.llm_model
                )
            chunks = self._proposition_chunker.chunk(text, metadata)
            return [c.to_dict() for c in chunks]

        elif self.mode == ChunkingMode.HYBRID:
            # Use hybrid chunker
            if self._hybrid_chunker is None:
                self._hybrid_chunker = HybridChunker(
                    embed_model_name=self.embed_model_name,
                    llm_endpoint=self.llm_endpoint,
                    llm_model=self.llm_model
                )
            chunks = self._hybrid_chunker.chunk(text, metadata)
            return [c.to_dict() for c in chunks]

        else:
            raise ValueError(f"Invalid chunking mode: {self.mode}")


# ============================================================================
# CLI Test
# ============================================================================

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

    print("Testing Chunker V2 - Multiple Modes")
    print("=" * 70)

    # Test 1: Simple chunking (backward compatible)
    print("\n1. Simple Chunking (Fast Sentence Aggregation)")
    print("-" * 70)
    simple_chunks = create_chunks(sample_text, min_words=180, max_words=320)
    stats = analyze_chunks(simple_chunks)
    print(f"Created {stats['total_chunks']} chunks")
    print(f"Mean words: {stats['mean_words']}")
    print(f"Range: {stats['min_words']}-{stats['max_words']} words")
    print(f"In target range (180-320): {stats['target_pct']}%")

    # Test 2: Semantic chunking (if available)
    if LLAMA_INDEX_AVAILABLE:
        print("\n2. Semantic Chunking (GTR-T5 Embeddings)")
        print("-" * 70)
        try:
            semantic_chunker = SemanticChunker()
            semantic_chunks = semantic_chunker.chunk(sample_text)
            semantic_dicts = [c.to_dict() for c in semantic_chunks]
            stats = analyze_chunks(semantic_dicts)
            print(f"Created {stats['total_chunks']} chunks")
            print(f"Mean words: {stats['mean_words']}")
            print(f"Range: {stats['min_words']}-{stats['max_words']} words")
            print(f"Mode distribution: {stats['chunking_modes']}")
        except Exception as e:
            print(f"Semantic chunking failed: {e}")
    else:
        print("\n2. Semantic Chunking: NOT AVAILABLE")
        print("   Install with: pip install llama-index llama-index-embeddings-huggingface")

    # Test 3: Unified interface
    print("\n3. Unified Chunker Interface")
    print("-" * 70)
    try:
        unified = UnifiedChunker(mode=ChunkingMode.SIMPLE)
        chunks = unified.chunk(sample_text)
        print(f"Mode: {unified.mode}")
        print(f"Created {len(chunks)} chunks")
    except Exception as e:
        print(f"Unified chunker failed: {e}")

    print("\n" + "=" * 70)
    print("✓ Chunker V2 test complete")
    print("\nAvailable modes:")
    print("  - simple: Fast sentence aggregation (always available)")
    print("  - semantic: Embedding-based boundaries (requires llama-index)")
    print("  - proposition: LLM-extracted propositions (requires llama-index + Ollama)")
    print("  - hybrid: Semantic + proposition refinement (requires both)")

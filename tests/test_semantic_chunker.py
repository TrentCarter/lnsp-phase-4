#!/usr/bin/env python3
"""
Tests for semantic chunker module.

Tests semantic, proposition, and hybrid chunking strategies.
"""

import pytest
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.semantic_chunker import (
    SemanticChunker,
    PropositionChunker,
    HybridChunker,
    Chunk,
    analyze_chunks
)


# ============================================================================
# Test Data
# ============================================================================

SAMPLE_TEXT_SHORT = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
carbon dioxide as inputs.
"""

SAMPLE_TEXT_MEDIUM = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
carbon dioxide as inputs. During photosynthesis, plants absorb light energy through
chlorophyll molecules. This energy is used to convert carbon dioxide and water into
glucose and oxygen. The glucose serves as food for the plant, while oxygen is released
as a byproduct. This process is essential for life on Earth, as it produces the oxygen
we breathe and forms the base of most food chains. Plants typically perform photosynthesis
during daylight hours when sunlight is available.
"""

SAMPLE_TEXT_LONG = """
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


# ============================================================================
# Semantic Chunker Tests
# ============================================================================

class TestSemanticChunker:
    """Test semantic chunker functionality."""

    @pytest.fixture
    def semantic_chunker(self):
        """Create semantic chunker instance."""
        return SemanticChunker(
            embed_model_name="sentence-transformers/gtr-t5-base"
        )

    def test_init(self, semantic_chunker):
        """Test chunker initialization."""
        assert semantic_chunker.embed_model is not None
        assert semantic_chunker.splitter is not None
        assert semantic_chunker.embed_model_name == "sentence-transformers/gtr-t5-base"

    def test_chunk_empty_text(self, semantic_chunker):
        """Test chunking empty text."""
        chunks = semantic_chunker.chunk("")
        assert len(chunks) == 0

        chunks = semantic_chunker.chunk("   ")
        assert len(chunks) == 0

    def test_chunk_short_text(self, semantic_chunker):
        """Test chunking short text."""
        chunks = semantic_chunker.chunk(SAMPLE_TEXT_SHORT)

        # Should create at least 1 chunk
        assert len(chunks) >= 1

        # Check chunk structure
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.text
            assert chunk.chunk_id
            assert chunk.chunk_index >= 0
            assert chunk.word_count > 0
            assert chunk.char_count > 0
            assert chunk.chunking_mode == "semantic"

    def test_chunk_medium_text(self, semantic_chunker):
        """Test chunking medium-length text."""
        chunks = semantic_chunker.chunk(SAMPLE_TEXT_MEDIUM)

        # Should create multiple chunks
        assert len(chunks) >= 1

        # Verify chunk IDs are unique
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))

    def test_chunk_with_metadata(self, semantic_chunker):
        """Test chunking with metadata."""
        metadata = {"source": "biology_textbook", "chapter": 3}
        chunks = semantic_chunker.chunk(SAMPLE_TEXT_MEDIUM, metadata=metadata)

        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "biology_textbook"
            assert chunk.metadata["chapter"] == 3

    def test_chunk_minimum_size(self, semantic_chunker):
        """Test minimum chunk size filtering."""
        chunks = semantic_chunker.chunk(SAMPLE_TEXT_LONG)

        # All chunks should meet minimum size
        for chunk in chunks:
            assert chunk.char_count >= semantic_chunker.min_chunk_size


# ============================================================================
# Proposition Chunker Tests
# ============================================================================

class TestPropositionChunker:
    """Test proposition chunker functionality."""

    @pytest.fixture
    def proposition_chunker(self):
        """Create proposition chunker instance."""
        # Check if Ollama is available
        llm_endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
        llm_model = os.getenv("LNSP_LLM_MODEL", "tinyllama:1.1b")

        try:
            return PropositionChunker(
                llm_endpoint=llm_endpoint,
                llm_model=llm_model
            )
        except ImportError:
            pytest.skip("LocalLlamaClient not available")

    @pytest.mark.slow
    def test_init(self, proposition_chunker):
        """Test chunker initialization."""
        assert proposition_chunker.llm_client is not None
        assert proposition_chunker.llm_model

    @pytest.mark.slow
    def test_chunk_empty_text(self, proposition_chunker):
        """Test chunking empty text."""
        chunks = proposition_chunker.chunk("")
        assert len(chunks) == 0

    @pytest.mark.slow
    def test_chunk_text(self, proposition_chunker):
        """Test proposition extraction."""
        chunks = proposition_chunker.chunk(SAMPLE_TEXT_MEDIUM)

        # Should extract multiple propositions
        assert len(chunks) >= 1

        # Check proposition structure
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.text
            assert chunk.chunking_mode == "proposition"
            # Propositions should be self-contained statements
            assert len(chunk.text.split()) >= 5  # At least 5 words

    @pytest.mark.slow
    def test_proposition_limit(self):
        """Test max propositions limit."""
        chunker = PropositionChunker(
            llm_endpoint=os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434"),
            llm_model=os.getenv("LNSP_LLM_MODEL", "tinyllama:1.1b"),
            max_propositions=3
        )

        chunks = chunker.chunk(SAMPLE_TEXT_LONG)

        # Should not exceed max limit
        assert len(chunks) <= 3


# ============================================================================
# Hybrid Chunker Tests
# ============================================================================

class TestHybridChunker:
    """Test hybrid chunker functionality."""

    @pytest.fixture
    def hybrid_chunker(self):
        """Create hybrid chunker instance."""
        llm_endpoint = os.getenv("LNSP_LLM_ENDPOINT", "http://localhost:11434")
        llm_model = os.getenv("LNSP_LLM_MODEL", "tinyllama:1.1b")

        try:
            return HybridChunker(
                embed_model_name="sentence-transformers/gtr-t5-base",
                llm_endpoint=llm_endpoint,
                llm_model=llm_model,
                refine_threshold=100  # Lower threshold for testing
            )
        except ImportError:
            pytest.skip("Hybrid chunker dependencies not available")

    def test_init(self, hybrid_chunker):
        """Test chunker initialization."""
        assert hybrid_chunker.semantic_chunker is not None
        assert hybrid_chunker.proposition_chunker is not None

    @pytest.mark.slow
    def test_chunk_without_refinement(self, hybrid_chunker):
        """Test hybrid chunking without triggering refinement."""
        chunks = hybrid_chunker.chunk(SAMPLE_TEXT_SHORT)

        # Short text should not trigger refinement
        assert len(chunks) >= 1

        for chunk in chunks:
            assert chunk.chunking_mode == "hybrid"

    @pytest.mark.slow
    def test_chunk_with_refinement(self, hybrid_chunker):
        """Test hybrid chunking with refinement triggered."""
        chunks = hybrid_chunker.chunk(SAMPLE_TEXT_LONG)

        # Should create multiple chunks
        assert len(chunks) >= 1

        # Some chunks may be refined
        assert all(c.chunking_mode == "hybrid" for c in chunks)

    @pytest.mark.slow
    def test_force_refine(self, hybrid_chunker):
        """Test forced refinement."""
        chunks = hybrid_chunker.chunk(
            SAMPLE_TEXT_MEDIUM,
            force_refine=True
        )

        # Should refine all chunks
        assert len(chunks) >= 1

        for chunk in chunks:
            assert chunk.chunking_mode == "hybrid"


# ============================================================================
# Analysis Tests
# ============================================================================

class TestAnalyzeChunks:
    """Test chunk analysis functionality."""

    def test_analyze_empty(self):
        """Test analyzing empty chunk list."""
        stats = analyze_chunks([])

        assert stats["total_chunks"] == 0
        assert stats["mean_words"] == 0

    def test_analyze_chunks(self):
        """Test analyzing chunk statistics."""
        # Create sample chunks
        chunks = [
            Chunk(
                text="Test chunk one",
                chunk_id="id1",
                chunk_index=0,
                word_count=3,
                char_count=15,
                chunking_mode="semantic",
                metadata={}
            ),
            Chunk(
                text="Test chunk two with more words",
                chunk_id="id2",
                chunk_index=1,
                word_count=6,
                char_count=31,
                chunking_mode="semantic",
                metadata={}
            ),
            Chunk(
                text="Test chunk three",
                chunk_id="id3",
                chunk_index=2,
                word_count=3,
                char_count=16,
                chunking_mode="proposition",
                metadata={}
            )
        ]

        stats = analyze_chunks(chunks)

        assert stats["total_chunks"] == 3
        assert stats["mean_words"] == 4.0  # (3 + 6 + 3) / 3
        assert stats["min_words"] == 3
        assert stats["max_words"] == 6
        assert stats["chunking_modes"] == {"semantic": 2, "proposition": 1}


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for chunking pipeline."""

    @pytest.mark.slow
    def test_semantic_to_dict(self):
        """Test converting semantic chunks to dict."""
        chunker = SemanticChunker()
        chunks = chunker.chunk(SAMPLE_TEXT_MEDIUM)

        # Convert to dict
        chunk_dicts = [c.to_dict() for c in chunks]

        assert len(chunk_dicts) == len(chunks)
        for chunk_dict in chunk_dicts:
            assert "text" in chunk_dict
            assert "chunk_id" in chunk_dict
            assert "chunk_index" in chunk_dict
            assert "word_count" in chunk_dict
            assert "chunking_mode" in chunk_dict

    def test_chunk_id_stability(self):
        """Test that chunk IDs are stable across runs."""
        chunker = SemanticChunker()

        chunks1 = chunker.chunk(SAMPLE_TEXT_MEDIUM)
        chunks2 = chunker.chunk(SAMPLE_TEXT_MEDIUM)

        # Chunk IDs should be identical for same input
        ids1 = [c.chunk_id for c in chunks1]
        ids2 = [c.chunk_id for c in chunks2]

        assert ids1 == ids2


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance benchmarks for chunkers."""

    @pytest.mark.slow
    def test_semantic_chunker_speed(self):
        """Test semantic chunker performance."""
        import time

        chunker = SemanticChunker()

        start_time = time.time()
        chunks = chunker.chunk(SAMPLE_TEXT_LONG)
        elapsed_time = time.time() - start_time

        # Should be reasonably fast (< 5 seconds for short text)
        assert elapsed_time < 5.0
        assert len(chunks) > 0

        print(f"\nSemantic chunker: {len(chunks)} chunks in {elapsed_time:.2f}s")


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

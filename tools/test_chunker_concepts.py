#!/usr/bin/env python3
"""
Quick test: Chunk texts with 1, 2, and 3 distinct concepts.

Tests how the chunker handles different conceptual densities.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.chunker_v2 import (
    create_chunks,
    SemanticChunker,
    UnifiedChunker,
    ChunkingMode,
    analyze_chunks,
    LLAMA_INDEX_AVAILABLE
)


# ============================================================================
# Test Texts
# ============================================================================

# Test 1: Single Concept (Photosynthesis)
TEXT_1_CONCEPT = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. During photosynthesis, plants absorb
light energy through chlorophyll molecules. This energy is used to convert carbon dioxide
and water into glucose and oxygen. The glucose serves as food for the plant, while oxygen
is released as a byproduct.
"""

# Test 2: Two Concepts (Photosynthesis + Cellular Respiration)
TEXT_2_CONCEPTS = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. During photosynthesis, plants absorb
light energy through chlorophyll molecules. This energy is used to convert carbon dioxide
and water into glucose and oxygen. The glucose serves as food for the plant.

Cellular respiration is the metabolic process that converts glucose into ATP energy.
This process occurs in the mitochondria of cells. During cellular respiration, glucose
is broken down through glycolysis, the Krebs cycle, and the electron transport chain.
Oxygen is consumed and carbon dioxide is produced as a waste product. This process
provides energy for all cellular activities.
"""

# Test 3: Three Concepts (Photosynthesis + Cellular Respiration + Fermentation)
TEXT_3_CONCEPTS = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. During photosynthesis, plants absorb
light energy through chlorophyll molecules. This energy is used to convert carbon dioxide
and water into glucose and oxygen.

Cellular respiration is the metabolic process that converts glucose into ATP energy.
This process occurs in the mitochondria of cells. During cellular respiration, glucose
is broken down through glycolysis, the Krebs cycle, and the electron transport chain.
Oxygen is consumed and carbon dioxide is produced as a waste product.

Fermentation is an anaerobic process that allows cells to produce energy without oxygen.
In alcoholic fermentation, yeast converts glucose into ethanol and carbon dioxide.
In lactic acid fermentation, muscle cells convert glucose into lactic acid during
intense exercise. This process produces less ATP than cellular respiration but can
occur when oxygen is limited.
"""


# ============================================================================
# Test Functions
# ============================================================================

def test_simple_chunking():
    """Test simple sentence-based chunking."""
    print("\n" + "=" * 80)
    print("SIMPLE CHUNKING (Sentence Aggregation)")
    print("=" * 80)

    for i, (text, concept_count) in enumerate([
        (TEXT_1_CONCEPT, 1),
        (TEXT_2_CONCEPTS, 2),
        (TEXT_3_CONCEPTS, 3)
    ], 1):
        print(f"\n{i}. Testing {concept_count} concept(s) ({len(text.split())} words)")
        print("-" * 80)

        chunks = create_chunks(text, min_words=50, max_words=150)
        stats = analyze_chunks(chunks)

        print(f"   Chunks created: {stats['total_chunks']}")
        print(f"   Mean words: {stats['mean_words']}")
        print(f"   Range: {stats['min_words']}-{stats['max_words']} words")
        print(f"   Chunking mode: {stats['chunking_modes']}")

        # Show first chunk
        if chunks:
            print(f"\n   First chunk preview:")
            print(f"   '{chunks[0]['text'][:100]}...'")


def test_semantic_chunking():
    """Test semantic embedding-based chunking."""
    if not LLAMA_INDEX_AVAILABLE:
        print("\n" + "=" * 80)
        print("SEMANTIC CHUNKING: SKIPPED (llama-index not available)")
        print("=" * 80)
        return

    print("\n" + "=" * 80)
    print("SEMANTIC CHUNKING (GTR-T5 Embedding-Based)")
    print("=" * 80)

    chunker = SemanticChunker(min_chunk_size=200)  # Lower threshold for testing

    for i, (text, concept_count) in enumerate([
        (TEXT_1_CONCEPT, 1),
        (TEXT_2_CONCEPTS, 2),
        (TEXT_3_CONCEPTS, 3)
    ], 1):
        print(f"\n{i}. Testing {concept_count} concept(s) ({len(text.split())} words)")
        print("-" * 80)

        try:
            chunks = chunker.chunk(text)
            chunk_dicts = [c.to_dict() for c in chunks]
            stats = analyze_chunks(chunk_dicts)

            print(f"   Chunks created: {stats['total_chunks']}")
            print(f"   Mean words: {stats['mean_words']}")
            print(f"   Range: {stats['min_words']}-{stats['max_words']} words")
            print(f"   Chunking mode: {stats['chunking_modes']}")

            # Show chunk boundaries
            if chunks:
                print(f"\n   Chunk boundaries:")
                for idx, chunk in enumerate(chunks):
                    print(f"   [{idx+1}] {chunk.word_count} words: '{chunk.text[:80]}...'")

        except Exception as e:
            print(f"   Error: {e}")


def test_unified_chunking():
    """Test unified chunker interface."""
    print("\n" + "=" * 80)
    print("UNIFIED CHUNKER (Mode Comparison)")
    print("=" * 80)

    # Test with 2-concept text
    text = TEXT_2_CONCEPTS
    print(f"\nTesting with 2-concept text ({len(text.split())} words)")
    print("-" * 80)

    modes_to_test = [ChunkingMode.SIMPLE]
    if LLAMA_INDEX_AVAILABLE:
        modes_to_test.append(ChunkingMode.SEMANTIC)

    for mode in modes_to_test:
        print(f"\n   Mode: {mode.value}")
        print(f"   {'-' * 70}")

        try:
            chunker = UnifiedChunker(mode=mode)
            chunks = chunker.chunk(text, min_words=50, max_words=150)
            stats = analyze_chunks(chunks)

            print(f"   Chunks: {stats['total_chunks']}")
            print(f"   Mean words: {stats['mean_words']}")
            print(f"   Distribution: {stats['word_distribution']}")

        except Exception as e:
            print(f"   Error: {e}")


def show_detailed_comparison():
    """Show detailed side-by-side comparison."""
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON: Simple vs Semantic")
    print("=" * 80)

    text = TEXT_3_CONCEPTS
    print(f"\nUsing 3-concept text ({len(text.split())} words)")
    print(f"Concepts: Photosynthesis, Cellular Respiration, Fermentation")
    print("-" * 80)

    # Simple chunking
    print("\n[SIMPLE MODE]")
    simple_chunks = create_chunks(text, min_words=60, max_words=120)
    for i, chunk in enumerate(simple_chunks, 1):
        print(f"\nChunk {i}: {chunk['word_count']} words")
        print(f"Text: {chunk['text'][:150]}...")

    # Semantic chunking
    if LLAMA_INDEX_AVAILABLE:
        print("\n[SEMANTIC MODE]")
        try:
            chunker = SemanticChunker(min_chunk_size=200)
            semantic_chunks = chunker.chunk(text)
            for i, chunk in enumerate(semantic_chunks, 1):
                print(f"\nChunk {i}: {chunk.word_count} words")
                print(f"Text: {chunk.text[:150]}...")
        except Exception as e:
            print(f"Error: {e}")


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🧪 CHUNKER CONCEPT TEST" + "\n")
    print("Testing chunker behavior with 1, 2, and 3 distinct concepts")

    # Run tests
    test_simple_chunking()
    test_semantic_chunking()
    test_unified_chunking()
    show_detailed_comparison()

    # Summary
    print("\n" + "=" * 80)
    print("✓ CONCEPT TEST COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    print("  - Simple mode: Chunks by word count (may split concepts)")
    if LLAMA_INDEX_AVAILABLE:
        print("  - Semantic mode: Chunks by semantic similarity (preserves concepts)")
    else:
        print("  - Semantic mode: Not tested (install llama-index to enable)")
    print("\nRecommendation:")
    print("  - Use semantic/hybrid mode for concept-based chunking")
    print("  - Use simple mode for fast processing when concept boundaries don't matter")
    print()

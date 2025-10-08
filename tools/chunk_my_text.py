#!/usr/bin/env python3
"""
Simple Chunker Test - Put Your Text Here!

Just replace the YOUR_TEXT_HERE section below with your own text
and run this script to see how it gets chunked.

Usage:
    python tools/chunk_my_text.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.chunker_v2 import SemanticChunker, create_chunks, analyze_chunks


# ============================================================================
# 👇 PUT YOUR TEXT HERE 👇
# ============================================================================

YOUR_TEXT_HERE = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
This occurs in the chloroplasts of plant cells. The process requires sunlight, water, and
carbon dioxide as inputs. During photosynthesis, plants absorb light energy through
chlorophyll molecules.

Cellular respiration is the metabolic process that converts glucose into ATP energy.
This process occurs in the mitochondria of cells. During cellular respiration, glucose
is broken down through glycolysis, the Krebs cycle, and the electron transport chain.

The water cycle describes how water moves between Earth's surface and atmosphere.
Water evaporates from oceans and lakes, forms clouds, and returns as precipitation.
This continuous cycle is essential for life on Earth and regulates global climate.
"""

# ============================================================================
# 👆 PUT YOUR TEXT ABOVE 👆
# ============================================================================


def chunk_text(text: str, show_comparison: bool = False):
    """
    Chunk the text using semantic chunking.

    Args:
        text: Your input text
        show_comparison: If True, show both simple and semantic modes
    """

    print("\n" + "=" * 80)
    print("🔍 SEMANTIC CHUNKING RESULTS")
    print("=" * 80)
    print(f"\nInput text: {len(text)} characters, {len(text.split())} words")
    print("-" * 80)

    # Semantic chunking (recommended)
    print("\n📊 SEMANTIC MODE (Concept-Based Boundaries)")
    print("-" * 80)

    try:
        chunker = SemanticChunker(min_chunk_size=200)
        chunks = chunker.chunk(text)

        if not chunks:
            print("⚠️  No chunks created (text too short)")
            return

        print(f"✓ Created {len(chunks)} chunks")
        print()

        for i, chunk in enumerate(chunks, 1):
            print(f"{'─' * 80}")
            print(f"CHUNK {i} of {len(chunks)}")
            print(f"{'─' * 80}")
            print(f"Words: {chunk.word_count}")
            print(f"Characters: {chunk.char_count}")
            print(f"Chunk ID: {chunk.chunk_id}")
            print()
            print("TEXT:")
            print(chunk.text.strip())
            print()

        # Statistics
        chunk_dicts = [c.to_dict() for c in chunks]
        stats = analyze_chunks(chunk_dicts)

        print("=" * 80)
        print("📈 STATISTICS")
        print("=" * 80)
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Mean words per chunk: {stats['mean_words']}")
        print(f"Word count range: {stats['min_words']} - {stats['max_words']}")
        print(f"Word distribution: {stats['word_distribution']}")
        print()

    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("💡 Tip: Make sure llama-index is installed:")
        print("   pip install llama-index llama-index-embeddings-huggingface")
        return

    # Optional: Show comparison with simple mode
    if show_comparison:
        print("\n" + "=" * 80)
        print("📊 COMPARISON: Simple vs Semantic")
        print("=" * 80)

        print("\n[SIMPLE MODE - Word Count Based]")
        print("-" * 80)
        simple_chunks = create_chunks(text, min_words=50, max_words=150)
        print(f"Chunks created: {len(simple_chunks)}")
        for i, chunk in enumerate(simple_chunks, 1):
            print(f"\nChunk {i}: {chunk['word_count']} words")
            print(f"Preview: {chunk['text'][:100]}...")

        print("\n[SEMANTIC MODE - Concept Based]")
        print("-" * 80)
        print(f"Chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks, 1):
            print(f"\nChunk {i}: {chunk.word_count} words")
            print(f"Preview: {chunk.text[:100].strip()}...")


# ============================================================================
# Run the chunker
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk your text using semantic chunking")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison between simple and semantic modes"
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to chunk (overrides YOUR_TEXT_HERE in script)"
    )

    args = parser.parse_args()

    # Use command-line text if provided, otherwise use YOUR_TEXT_HERE
    text = args.text if args.text else YOUR_TEXT_HERE

    # Check if text is just the placeholder
    if text.strip() == "":
        print("\n❌ ERROR: No text provided!")
        print("\n📝 To use this script:")
        print("   1. Edit this file and replace YOUR_TEXT_HERE with your text")
        print("   2. Or run: python tools/chunk_my_text.py --text 'Your text here'")
        print()
        sys.exit(1)

    # Run chunking
    chunk_text(text, show_comparison=args.compare)

    print("\n" + "=" * 80)
    print("✓ DONE")
    print("=" * 80)
    print("\n💡 Tips:")
    print("   - Edit YOUR_TEXT_HERE in this script to test different texts")
    print("   - Run with --compare to see simple vs semantic modes")
    print("   - Run with --text 'your text' to chunk from command line")
    print()

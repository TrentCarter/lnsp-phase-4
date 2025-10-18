#!/usr/bin/env python3
"""Test semantic chunking API with new parameters (min=40, max=200)"""

import requests
import json

def test_chunking():
    # Test text with multiple sentences
    test_text = """
    This is a test sentence. This is another sentence to test chunking behavior.
    We want to verify that chunks are properly sized between 40 and 200 characters.
    This continues the text to create a longer chunk that might need to be split.
    And here is even more text to ensure we test the splitting behavior at the
    200 character boundary which should split on sentence boundaries for clean breaks.
    Adding more sentences here to really test the limits. This should create multiple
    chunks all within the target range of 40 to 200 characters as specified.
    """

    print("🧪 Testing Semantic Chunking API")
    print("=" * 60)
    print(f"Input text length: {len(test_text)} chars")
    print()

    # Call chunking API
    response = requests.post(
        "http://localhost:8001/chunk",
        json={
            "text": test_text,
            "mode": "semantic",
            "min_chunk_size": 40,
            "max_chunk_size": 200,
            "breakpoint_threshold": 75
        },
        timeout=30
    )

    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        return

    data = response.json()
    chunks = data.get("chunks", [])

    print(f"✅ Received {len(chunks)} chunks\n")

    # Analyze chunk sizes
    sizes = []
    for i, chunk in enumerate(chunks, 1):
        chunk_text = chunk["text"]
        size = len(chunk_text)
        sizes.append(size)

        # Determine if size is ideal
        status = "✅" if 40 <= size <= 200 else "❌"

        print(f"{status} Chunk {i}: {size} chars")
        print(f"   \"{chunk_text[:70]}...\"")
        print()

    # Summary
    print("=" * 60)
    print("📊 Summary:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Min size: {min(sizes) if sizes else 0} chars")
    print(f"   Max size: {max(sizes) if sizes else 0} chars")
    print(f"   Avg size: {sum(sizes)/len(sizes) if sizes else 0:.1f} chars")

    in_range = sum(1 for s in sizes if 40 <= s <= 200)
    print(f"   In range (40-200): {in_range}/{len(chunks)} ({100*in_range/len(chunks) if chunks else 0:.1f}%)")

    if all(40 <= s <= 200 for s in sizes):
        print("\n✅ All chunks within target range!")
    else:
        print("\n⚠️  Some chunks outside target range")

if __name__ == "__main__":
    test_chunking()

#!/usr/bin/env python3
"""
Test vec2text library directly using official API.

This bypasses all our wrappers to determine if the problem is:
1. Our wrapper/server infrastructure
2. The vec2text library itself
"""

import torch
from sentence_transformers import SentenceTransformer

print("\n" + "=" * 80)
print("Testing Vec2Text Official API")
print("=" * 80)
print()

# Test cases
test_texts = [
    "Water reflects light very differently from typical terrestrial materials.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can predict the next vector in a sequence.",
]

print("Step 1: Loading GTR-T5 encoder...")
encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')
print("✓ GTR-T5 encoder loaded")
print()

print("Step 2: Attempting to import vec2text...")
try:
    import vec2text
    print(f"✓ vec2text imported successfully (version: {vec2text.__version__ if hasattr(vec2text, '__version__') else 'unknown'})")
except ImportError as e:
    print(f"✗ vec2text import failed: {e}")
    print("\nTo install vec2text:")
    print("  pip install vec2text")
    exit(1)
print()

print("Step 3: Loading vec2text model...")
try:
    # Try loading JXE model
    vec2text_model = vec2text.models.InversionModel.from_pretrained("jxe/gtr__nq__32")
    print("✓ Vec2text model loaded (jxe/gtr__nq__32)")
except Exception as e:
    print(f"✗ Failed to load vec2text model: {e}")
    print("\nTrying alternative model path...")
    try:
        vec2text_model = vec2text.models.InversionModel.from_pretrained("jxmorris25/vec2text")
        print("✓ Vec2text model loaded (jxmorris25/vec2text)")
    except Exception as e2:
        print(f"✗ Alternative model also failed: {e2}")
        exit(1)
print()

print("Step 4: Testing round-trip on sample texts...")
print()

for i, text in enumerate(test_texts, 1):
    print(f"Test {i}/{len(test_texts)}")
    print(f"Original: {text}")

    # Encode with GTR-T5
    embedding = encoder.encode([text], convert_to_tensor=True)
    print(f"  → Encoded to shape {embedding.shape}")

    # Decode with vec2text
    try:
        decoded = vec2text_model.invert(
            embeddings=embedding,
            num_steps=20  # Use more steps for better quality
        )
        print(f"Decoded:  {decoded[0]}")

        # Calculate word overlap
        orig_words = set(text.lower().split())
        decoded_words = set(decoded[0].lower().split())
        overlap = len(orig_words & decoded_words)
        total = len(orig_words)
        print(f"Word overlap: {overlap}/{total} ({100*overlap/total:.1f}%)")

    except Exception as e:
        print(f"  ✗ Decode failed: {e}")

    print()

print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()
print("If this test SUCCEEDS:")
print("  → Vec2text library works correctly")
print("  → Problem is in our wrapper/server infrastructure")
print("  → Need to investigate app/api/vec2text_server.py and wrappers")
print()
print("If this test FAILS:")
print("  → Vec2text library itself has issues")
print("  → May need to reinstall vec2text or use different model checkpoint")
print("  → Check HuggingFace model hub for correct model paths")
print()

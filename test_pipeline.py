#!/usr/bin/env python3
"""
Quick test of the ingestion pipeline components.
"""

import sys
import os
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Test imports
try:
    from src.prompt_extractor import extract_cpe_from_text
    from src.tmd_encoder import pack_tmd, unpack_tmd
    from src.vectorizer import EmbeddingBackend
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test basic functionality
print("\nTesting TMD encoding...")
bits = pack_tmd(9, 0, 5)  # art, fact_retrieval, historical
domain, task, modifier = unpack_tmd(bits)
print(f"✓ TMD: domain={domain}, task={task}, modifier={modifier}")

print("\nTesting embedding...")
embedder = EmbeddingBackend()
vec = embedder.encode(["test text"])
print(f"✓ Embedding shape: {vec.shape}")

print("\nTesting extraction...")
result = extract_cpe_from_text("This is a test album by a singer.")
print(f"✓ Extraction keys: {list(result.keys())}")
print(f"✓ Concept: {result['concept'][:50]}...")
print(f"✓ Domain: {result['domain']}")

print("\n🎉 All tests passed!")

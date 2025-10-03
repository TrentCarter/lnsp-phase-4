#!/usr/bin/env python3
"""Simple test to verify RAG bench components work."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Test imports
print("Testing imports...")
try:
    from src.db_faiss import FaissDB
    print("✓ FaissDB import OK")
except Exception as e:
    print(f"✗ FaissDB import failed: {e}")
    sys.exit(1)

try:
    from src.vectorizer import EmbeddingBackend
    print("✓ EmbeddingBackend import OK")
except Exception as e:
    print(f"✗ EmbeddingBackend import failed: {e}")
    sys.exit(1)

try:
    from rank_bm25 import BM25Okapi
    print("✓ BM25Okapi import OK")
except Exception as e:
    print(f"✗ BM25Okapi import failed: {e}")
    sys.exit(1)

# Test BM25
print("\nTesting BM25...")
corpus = ["hello world", "foo bar baz", "test document"]
tokenized = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized)
scores = bm25.get_scores("hello test".split())
print(f"✓ BM25 scores: {scores}")

print("\nAll tests passed!")

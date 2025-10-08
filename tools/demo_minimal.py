#!/usr/bin/env python3
"""
Minimal LVM Pipeline Demo - Direct FAISS Usage

Shows raw FAISS retrieval vs LLM-smoothed responses with citations.
"""
import os
import sys
from pathlib import Path
import numpy as np
import faiss
import requests
import time

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB
from src.vectorizer import EmbeddingBackend
from src.utils.norms import l2_normalize


def search_faiss(index, query_vec, k=5):
    """Search FAISS index directly."""
    q = l2_normalize(query_vec.reshape(1, -1)).astype(np.float32)
    scores, ids = index.search(q, k)
    return scores[0], ids[0]


def generate_with_llm(query, concepts):
    """Simple LLM smoothing with citation validation."""
    concepts_text = "\n".join([f"- ({c['id']}): {c['text']}" for c in concepts])

    prompt = f"""You are answering a scientific query using retrieved ontology concepts.

Query: {query}

Retrieved Concepts:
{concepts_text}

CRITICAL REQUIREMENTS:
1. EVERY factual claim MUST cite a concept ID in (id:text) format.
2. Use ONLY the IDs and texts provided above.
3. Keep response concise: 2-3 sentences maximum.
4. Do NOT invent facts not in the retrieved concepts.

Example Format:
"Neural networks (cpe_001:neural networks) enable deep learning (cpe_002:deep learning)."

Response (with mandatory citations):"""

    try:
        resp = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.1:8b",
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            },
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["message"]["content"].strip()
        return f"LLM Error: {resp.status_code}"
    except Exception as e:
        return f"LLM Error: {e}"


def main():
    print("=" * 70)
    print("MINIMAL LVM PIPELINE DEMO")
    print("=" * 70)
    print()

    # Load FAISS index and metadata
    npz_path = "artifacts/ontology_13k.npz"
    index_path = "artifacts/ontology_13k_ivf_flat_ip_rebuilt.index"

    if not Path(npz_path).exists():
        print(f"❌ NPZ file not found: {npz_path}")
        sys.exit(1)
    if not Path(index_path).exists():
        print(f"❌ Index file not found: {index_path}")
        sys.exit(1)

    print(f"Loading FAISS index from {index_path}...")
    index = faiss.read_index(str(index_path))
    print(f"✓ Loaded index with {index.ntotal} vectors (dim={getattr(index, 'd', 'unknown')})")

    print(f"Loading metadata from {npz_path}...")
    npz = np.load(npz_path, allow_pickle=True)
    cpe_ids = npz['cpe_ids']
    concept_texts = npz['concept_texts']
    print(f"✓ Loaded metadata for {len(cpe_ids)} concepts")

    print("Initializing embedding backend (may take 2-5 minutes on first run)...")
    startup_start = time.time()
    embedder = EmbeddingBackend()
    startup_time = time.time() - startup_start
    print(f"✓ Embedding backend ready (startup took {startup_time:.1f}s)")
    print()
    print("=" * 70)
    print("BENCHMARK STARTING (startup time excluded from results)")
    print("=" * 70)
    print()

    # Test queries
    queries = [
        "protein function",
        "enzyme catalysis and regulation",
        "cell membrane structure",
    ]

    # Track query timing (excluding startup)
    query_times = []

    index_dim = int(getattr(index, 'd', 768))

    for query in queries:
        print("=" * 70)
        print(f"Query: {query}")
        print("=" * 70)

        # Time the query processing
        query_start = time.time()

        # Encode query (768D from GTR-T5)
        query_vec = embedder.encode([query])[0].astype(np.float32)

        # Adapt to index dimension
        if index_dim == 784:
            # Prepend 16D TMD zeros to form fused 784D, then normalize
            tmd_zeros = np.zeros((16,), dtype=np.float32)
            fused = np.concatenate([tmd_zeros, query_vec]).astype(np.float32)
            query_vec_adapted = l2_normalize(fused.reshape(1, -1)).astype(np.float32)[0]
        elif index_dim == 768:
            query_vec_adapted = l2_normalize(query_vec.reshape(1, -1)).astype(np.float32)[0]
        else:
            raise ValueError(f"Unsupported index dim {index_dim}; expected 768 or 784")

        # Search FAISS
        scores, ids = search_faiss(index, query_vec_adapted, k=5)

        query_time = time.time() - query_start

        # Display raw results
        print("\nBEFORE (Raw FAISS Retrieval):")
        print("-" * 70)
        concepts = []
        for i, (score, idx) in enumerate(zip(scores, ids), 1):
            if idx < len(cpe_ids):
                cpe_id = str(cpe_ids[idx])
                text = str(concept_texts[idx])
                print(f"{i}. [{score:.4f}] {text}")
                concepts.append({'id': cpe_id, 'text': text, 'score': score})
            else:
                print(f"{i}. [INVALID INDEX: {idx}]")

        # LLM smoothing
        print("\nAFTER (LLM-Smoothed Response):")
        print("-" * 70)
        response = generate_with_llm(query, concepts)
        print(response)

        query_times.append(query_time)
        print(f"\n⏱️  Query time: {query_time:.3f}s (excluding startup)")
        print()

    # Summary
    print()
    print("=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Startup time: {startup_time:.1f}s (one-time cost, excluded from metrics)")
    print(f"Queries processed: {len(query_times)}")
    print(f"Average query time: {sum(query_times)/len(query_times):.3f}s")
    print(f"Min query time: {min(query_times):.3f}s")
    print(f"Max query time: {max(query_times):.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()

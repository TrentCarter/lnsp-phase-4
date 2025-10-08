#!/usr/bin/env python3
"""
BEFORE/AFTER Demo - Shows raw vecRAG vs LLM-smoothed results
"""
import sys
from pathlib import Path
import numpy as np
import faiss
import requests
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.vectorizer import EmbeddingBackend
from src.utils.norms import l2_normalize


def search_faiss(index, query_vec, concept_texts, cpe_ids, k=5):
    """Search FAISS and return results."""
    q = l2_normalize(query_vec.reshape(1, -1)).astype(np.float32)
    scores, ids = index.search(q, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < len(concept_texts):
            results.append({
                'text': str(concept_texts[idx]),
                'cpe_id': str(cpe_ids[idx]),
                'score': float(score)
            })
    return results


def generate_with_llm(query, concepts):
    """LLM smoothing with citations."""
    concepts_text = "\n".join([f"- {c['text']}" for c in concepts])

    prompt = f"""You are a bioinformatics expert. Answer this query using ONLY the provided ontology concepts.

Query: {query}

Retrieved Concepts:
{concepts_text}

Requirements:
1. Write a natural, coherent 2-3 sentence answer
2. Use ONLY the concepts listed above
3. Cite concepts by putting them in quotes, like: "ClustalW"
4. Be factual and precise

Answer:"""

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
        return f"[LLM Error: {resp.status_code}]"
    except Exception as e:
        return f"[LLM Error: {e}]"


def main():
    print("=" * 80)
    print("BEFORE/AFTER DEMO - vecRAG with LLM Smoothing")
    print("=" * 80)
    print()

    # Load data
    print("Loading FAISS index and metadata...")
    index = faiss.read_index("artifacts/ontology_13k_ivf_flat_ip_rebuilt.index")
    npz = np.load("artifacts/ontology_13k.npz", allow_pickle=True)
    concept_texts = npz['concept_texts']
    cpe_ids = npz['cpe_ids']
    print(f"✓ Loaded {index.ntotal} vectors, {len(concept_texts)} concepts")

    print("\nInitializing embedding backend (may take 2-5 min)...")
    startup_start = time.time()
    embedder = EmbeddingBackend()
    startup_time = time.time() - startup_start
    print(f"✓ Ready (startup: {startup_time:.1f}s)")
    print()

    # Single-concept queries
    single_queries = [
        "microarray data analysis",
        "sequence alignment software",
        "gene expression quantification",
        "R statistical package",
        "proteomics data format"
    ]

    # Dual-concept queries
    dual_queries = [
        "microarray and gene expression",
        "sequence alignment and clustering",
        "R software and bioinformatics",
        "data formats and processing",
        "quantification and statistical analysis"
    ]

    print("=" * 80)
    print("PART 1: SINGLE-CONCEPT QUERIES (5 examples)")
    print("=" * 80)
    print()

    for i, query in enumerate(single_queries, 1):
        print(f"--- Query {i}: \"{query}\" ---")
        print()

        # Encode and search
        query_vec = embedder.encode([query])[0]
        results = search_faiss(index, query_vec, concept_texts, cpe_ids, k=5)

        # BEFORE: Raw vecRAG
        print("BEFORE (Raw vecRAG Retrieval):")
        for j, r in enumerate(results, 1):
            print(f"  {j}. [{r['score']:.3f}] {r['text']}")
        print()

        # AFTER: LLM smoothed
        print("AFTER (LLM-Smoothed Response):")
        response = generate_with_llm(query, results)
        print(f"  {response}")
        print()
        print()

    print("=" * 80)
    print("PART 2: DUAL-CONCEPT QUERIES (5 examples)")
    print("=" * 80)
    print()

    for i, query in enumerate(dual_queries, 1):
        print(f"--- Query {i}: \"{query}\" ---")
        print()

        # Encode and search
        query_vec = embedder.encode([query])[0]
        results = search_faiss(index, query_vec, concept_texts, cpe_ids, k=5)

        # BEFORE: Raw vecRAG
        print("BEFORE (Raw vecRAG Retrieval):")
        for j, r in enumerate(results, 1):
            print(f"  {j}. [{r['score']:.3f}] {r['text']}")
        print()

        # AFTER: LLM smoothed
        print("AFTER (LLM-Smoothed Response):")
        response = generate_with_llm(query, results)
        print(f"  {response}")
        print()
        print()


if __name__ == "__main__":
    main()

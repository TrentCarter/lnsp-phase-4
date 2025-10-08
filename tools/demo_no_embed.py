#!/usr/bin/env python3
"""
BEFORE/AFTER Demo - Using pre-computed query vectors (no EmbeddingBackend initialization)
"""
import sys
from pathlib import Path
import numpy as np
import faiss
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    print("Using pre-sampled query vectors from actual data")
    print("=" * 80)
    print()

    # Load data
    print("Loading FAISS index and metadata...")
    index = faiss.read_index("artifacts/ontology_13k_ivf_flat_ip_rebuilt.index")
    npz = np.load("artifacts/ontology_13k.npz", allow_pickle=True)
    concept_texts = npz['concept_texts']
    cpe_ids = npz['cpe_ids']

    # Use actual fused vectors as "queries" to show retrieval
    fused_vecs = npz['fused']  # 784D vectors (concept + TMD)

    print(f"✓ Loaded {index.ntotal} vectors, {len(concept_texts)} concepts")
    print(f"  Using {fused_vecs.shape[1]}D fused vectors")
    print()

    # Pick interesting concepts as queries (manually selected indices with good content)
    single_query_indices = [228, 285, 1116, 501, 178]  # Based on earlier sampling
    dual_query_pairs = [
        (285, 228),  # microarray + quantification
        (1116, 407),  # ClustalW + dot plot
        (501, 563),  # R software concepts
        (178, 54),   # formats + OMICS
        (228, 285),  # quantification + processing
    ]

    print("=" * 80)
    print("PART 1: SINGLE-CONCEPT QUERIES (5 examples)")
    print("=" * 80)
    print()

    for i, idx in enumerate(single_query_indices, 1):
        query_text = concept_texts[idx]
        print(f"--- Query {i}: \"{query_text}\" (index {idx}) ---")
        print()

        # Use the concept's fused vector as query
        query_vec = fused_vecs[idx]
        results = search_faiss(index, query_vec, concept_texts, cpe_ids, k=5)

        # BEFORE: Raw vecRAG
        print("BEFORE (Raw vecRAG Retrieval):")
        for j, r in enumerate(results, 1):
            print(f"  {j}. [{r['score']:.3f}] {r['text']}")
        print()

        # AFTER: LLM smoothed
        print("AFTER (LLM-Smoothed Response):")
        response = generate_with_llm(query_text, results)
        print(f"  {response}")
        print()
        print()

    print("=" * 80)
    print("PART 2: DUAL-CONCEPT QUERIES (5 examples)")
    print("=" * 80)
    print()

    for i, (idx1, idx2) in enumerate(dual_query_pairs, 1):
        text1 = concept_texts[idx1]
        text2 = concept_texts[idx2]
        query_text = f"{text1} and {text2}"
        print(f"--- Query {i}: \"{text1}\" + \"{text2}\" ---")
        print()

        # Average the two fused vectors
        query_vec = (fused_vecs[idx1] + fused_vecs[idx2]) / 2
        results = search_faiss(index, query_vec, concept_texts, cpe_ids, k=5)

        # BEFORE: Raw vecRAG
        print("BEFORE (Raw vecRAG Retrieval):")
        for j, r in enumerate(results, 1):
            print(f"  {j}. [{r['score']:.3f}] {r['text']}")
        print()

        # AFTER: LLM smoothed
        print("AFTER (LLM-Smoothed Response):")
        response = generate_with_llm(query_text, results)
        print(f"  {response}")
        print()
        print()


if __name__ == "__main__":
    main()

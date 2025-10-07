#!/usr/bin/env python3
"""
Simple Query Demo: BEFORE/AFTER LLM Smoothing

Shows raw FAISS retrieval vs LLM-smoothed responses with citations.

Usage:
    python tools/demo_simple_queries.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.db_faiss import FaissDB
from src.vectorizer import EmbeddingBackend
import requests
import json
import re


def generate_with_citations(query, concepts, llm_endpoint="http://localhost:11434", llm_model="llama3.1:8b"):
    """Simple LLM smoothing with citation validation."""
    # Build prompt
    concepts_text = "\n".join([f"- ({c['id']}): {c['text']}" for c in concepts])

    prompt = f"""You are answering a scientific query using retrieved ontology concepts.

Query: {query}

Retrieved Concepts:
{concepts_text}

CRITICAL REQUIREMENTS:
1. EVERY factual claim MUST cite a concept ID in (id:text) format.
2. Use ONLY the IDs and texts provided above (e.g., "({concepts[0]['id']}:{concepts[0]['text']})").
3. Keep response concise: 2-3 sentences maximum.
4. Do NOT invent facts not in the retrieved concepts.

Example Format:
"Neural networks (cpe_001:neural networks) enable deep learning (cpe_002:deep learning), a key technique in AI."

Response (with mandatory citations):"""

    try:
        # Call Ollama
        response = requests.post(
            f"{llm_endpoint}/api/generate",
            json={
                "model": llm_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "num_predict": 150
                }
            },
            timeout=30
        )
        response.raise_for_status()
        text = response.json()["response"].strip()

        # Extract citations
        citation_pattern = r'\(([^:]+):([^)]+)\)'
        citations_found = re.findall(citation_pattern, text)
        citation_rate = len(citations_found) / max(len(concepts), 1)

        return {
            "text": text,
            "citations_found": len(citations_found),
            "total_concepts": len(concepts),
            "citation_rate": citation_rate
        }
    except Exception as e:
        return {
            "text": f"Error: {e}",
            "citations_found": 0,
            "total_concepts": len(concepts),
            "citation_rate": 0.0
        }


def main():
    print("=" * 70)
    print("LVM PIPELINE DEMO: BEFORE/AFTER LLM SMOOTHING")
    print("=" * 70)
    print()

    # Initialize
    npz_path = "artifacts/ontology_13k.npz"
    index_path = "artifacts/ontology_13k_ivf_flat_ip.index"

    if not Path(npz_path).exists() or not Path(index_path).exists():
        print(f"❌ FAISS files not found")
        print(f"   NPZ: {npz_path}")
        print(f"   Index: {index_path}")
        sys.exit(1)

    print("Initializing...")
    faiss_db = FaissDB(index_path=index_path, meta_npz_path=npz_path)
    faiss_db.load()  # Must call load() to load index and metadata
    embedding_backend = EmbeddingBackend()

    # Check if Ollama is available
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=1)
        if resp.status_code == 200:
            print("✓ LLM smoother ready (Ollama + Llama 3.1:8b)")
            llm_available = True
        else:
            print("⚠ Ollama not responding - showing raw results only")
            llm_available = False
    except:
        print("⚠ Ollama not available - showing raw results only")
        llm_available = False

    print("✓ FAISS loaded")
    print()

    # Part 1: Single-concept queries
    print("=" * 70)
    print("PART 1: SINGLE-CONCEPT QUERIES (5x)")
    print("=" * 70)

    single_queries = [
        "protein function",
        "enzyme catalysis",
        "cell membrane structure",
        "DNA replication process",
        "metabolic pathway regulation",
    ]

    for i, query in enumerate(single_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query #{i}: \"{query}\"")
        print(f"{'─' * 70}")

        # Encode and search
        query_vec = embedding_backend.encode([query])[0]
        results = faiss_db.search_legacy(query_vec, topk=5)

        # Show BEFORE (raw retrieval)
        print("\n📊 BEFORE (Raw Retrieval):")
        print(f"   Top 5 concepts:")
        for j, result in enumerate(results, 1):
            concept_text = result.get("metadata", {}).get("concept_text", "")
            score = result.get("score", 0.0)
            print(f"   {j}. {concept_text[:60]}... (score: {score:.3f})")

        # Show AFTER (LLM smoothing)
        if llm_available:
            concepts = [
                {
                    "id": r.get("cpe_id", "unknown"),
                    "text": r.get("metadata", {}).get("concept_text", ""),
                }
                for r in results
            ]

            smoothed = generate_with_citations(query=query, concepts=concepts)

            print("\n✨ AFTER (LLM Smoothing):")
            print(f"   Response:")
            for line in smoothed['text'].split('\n'):
                if line.strip():
                    print(f"   {line}")

            print(f"\n   📌 Citations:")
            print(f"      Rate: {smoothed['citation_rate']:.1%}")
            print(f"      Found: {smoothed['citations_found']}/{smoothed['total_concepts']}")

            if smoothed['citation_rate'] >= 0.9:
                print(f"      Status: ✅ PASS (≥90%)")
            else:
                print(f"      Status: ⚠ LOW (<90%)")

    # Part 2: Dual-concept queries
    print("\n\n" + "=" * 70)
    print("PART 2: DUAL-CONCEPT QUERIES (5x)")
    print("=" * 70)

    dual_queries = [
        "protein and enzyme relationship",
        "cell membrane and transport mechanisms",
        "DNA and RNA synthesis",
        "metabolic pathways and energy production",
        "gene expression and regulation",
    ]

    for i, query in enumerate(dual_queries, 1):
        print(f"\n{'─' * 70}")
        print(f"Query #{i}: \"{query}\"")
        print(f"{'─' * 70}")

        # Encode and search
        query_vec = embedding_backend.encode([query])[0]
        results = faiss_db.search_legacy(query_vec, topk=5)

        # Show BEFORE (raw retrieval)
        print("\n📊 BEFORE (Raw Retrieval):")
        print(f"   Top 5 concepts:")
        for j, result in enumerate(results, 1):
            concept_text = result.get("metadata", {}).get("concept_text", "")
            score = result.get("score", 0.0)
            print(f"   {j}. {concept_text[:60]}... (score: {score:.3f})")

        # Show AFTER (LLM smoothing)
        if llm_available:
            concepts = [
                {
                    "id": r.get("cpe_id", "unknown"),
                    "text": r.get("metadata", {}).get("concept_text", ""),
                }
                for r in results
            ]

            smoothed = generate_with_citations(query=query, concepts=concepts)

            print("\n✨ AFTER (LLM Smoothing):")
            print(f"   Response:")
            for line in smoothed['text'].split('\n'):
                if line.strip():
                    print(f"   {line}")

            print(f"\n   📌 Citations:")
            print(f"      Rate: {smoothed['citation_rate']:.1%}")
            print(f"      Found: {smoothed['citations_found']}/{smoothed['total_concepts']}")

            if smoothed['citation_rate'] >= 0.9:
                print(f"      Status: ✅ PASS (≥90%)")
            else:
                print(f"      Status: ⚠ LOW (<90%)")

    print("\n" + "=" * 70)
    print("✅ DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

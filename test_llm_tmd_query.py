#!/usr/bin/env python3
"""
Quick test: Compare LLM-based vs pattern-based TMD query extraction.

This tests query-time TMD generation WITHOUT re-ingesting the corpus.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Setup paths
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Set LLM environment
os.environ['LNSP_LLM_ENDPOINT'] = 'http://localhost:11434'
os.environ['LNSP_LLM_MODEL'] = 'llama3.1:8b'

from RAG.vecrag_tmd_rerank import generate_tmd_for_query
from src.llm_tmd_extractor import extract_tmd_with_llm
from src.tmd_extractor_v2 import extract_tmd_from_text as extract_tmd_pattern

# Test queries from ontology domain
TEST_QUERIES = [
    "What is the function of sodium-glucose cotransporter 2?",
    "Define apoptosis in cellular biology",
    "How does CRISPR-Cas9 gene editing work?",
    "Explain the role of mitochondria in energy production",
    "What are the symptoms of type 2 diabetes?"
]


def main():
    print("=" * 80)
    print("Testing LLM-based vs Pattern-based TMD Query Extraction")
    print("=" * 80)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}] {query}")
        print("-" * 80)

        # Pattern-based extraction
        pattern_tmd_dict = extract_tmd_pattern(query)
        pattern_vec = generate_tmd_for_query(query, use_llm=False)

        # LLM-based extraction
        llm_tmd_dict = extract_tmd_with_llm(query)
        llm_vec = generate_tmd_for_query(query, use_llm=True)

        # Compare results
        print("Pattern-based:")
        print(f"  Domain: {pattern_tmd_dict.get('domain_code')} | "
              f"Task: {pattern_tmd_dict.get('task_code')} | "
              f"Modifier: {pattern_tmd_dict.get('modifier_code')}")
        print(f"  Vector norm: {np.linalg.norm(pattern_vec):.4f}")

        print("\nLLM-based:")
        print(f"  Domain: {llm_tmd_dict.get('domain_code')} | "
              f"Task: {llm_tmd_dict.get('task_code')} | "
              f"Modifier: {llm_tmd_dict.get('modifier_code')}")
        print(f"  Vector norm: {np.linalg.norm(llm_vec):.4f}")

        # Cosine similarity between vectors
        if np.linalg.norm(pattern_vec) > 0 and np.linalg.norm(llm_vec) > 0:
            similarity = np.dot(pattern_vec, llm_vec) / (
                np.linalg.norm(pattern_vec) * np.linalg.norm(llm_vec)
            )
            print(f"\nCosine similarity: {similarity:.4f}")
        else:
            print("\nCosine similarity: N/A (zero vector)")

    print("\n" + "=" * 80)
    print("✓ Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

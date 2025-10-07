#!/usr/bin/env python3
"""
LVM Pipeline Demo: Show Before/After LLM Smoothing

Demonstrates the complete pipeline with:
1. Single-concept queries (5x)
2. Dual-concept queries (5x)
3. Shows raw retrieval results vs smoothed LLM output
4. Citation validation and formatting

Usage:
    python tools/demo_pipeline_with_smoothing.py
    python tools/demo_pipeline_with_smoothing.py --verbose
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vectorizer import EmbeddingBackend
from src.faiss_db import FaissDB
from src.lvm.llm_smoothing import LLMSmoother


class PipelineDemo:
    """Demonstrate complete pipeline with LLM smoothing."""

    def __init__(self, npz_path: str, index_path: str):
        """Initialize demo with FAISS index and LLM smoother."""
        print("=" * 70)
        print("LVM PIPELINE DEMO: BEFORE/AFTER LLM SMOOTHING")
        print("=" * 70)
        print()

        # Initialize components
        print("Initializing components...")
        self.embedding_backend = EmbeddingBackend()
        self.faiss_db = FaissDB(npz_path=npz_path, index_path=index_path)

        # Check if Ollama is available
        try:
            import requests
            resp = requests.get("http://localhost:11434/api/tags", timeout=1)
            if resp.status_code == 200:
                self.llm_smoother = LLMSmoother(
                    llm_endpoint="http://localhost:11434",
                    llm_model="llama3.1:8b",
                    citation_threshold=0.9,
                )
                print("✓ LLM smoother initialized (Ollama + Llama 3.1:8b)")
                self.llm_available = True
            else:
                print("⚠ Ollama not responding - will skip LLM smoothing")
                self.llm_available = False
        except:
            print("⚠ Ollama not available - will skip LLM smoothing")
            self.llm_available = False

        print("✓ FAISS index loaded")
        print("✓ Embedding backend ready")
        print()

    def run_query(self, query_text: str, k: int = 5) -> Dict[str, Any]:
        """Run a single query through the pipeline."""
        # 1. Encode query
        query_vec = self.embedding_backend.encode([query_text])[0]

        # 2. FAISS search
        results = self.faiss_db.search(query_vec, k=k)

        # 3. Extract concepts
        retrieved_concepts = []
        for result in results:
            concept_data = {
                "id": result.get("cpe_id", f"idx_{result.get('idx', 'unknown')}"),
                "text": result.get("concept_text", ""),
                "score": result.get("score", 0.0),
            }
            retrieved_concepts.append(concept_data)

        # 4. LLM smoothing (if available)
        if self.llm_available:
            smoothed = self.llm_smoother.generate_with_citations(
                query=query_text,
                concepts=retrieved_concepts,
            )
        else:
            # Fallback: simple concatenation
            smoothed = {
                "text": f"Retrieved concepts: {', '.join([c['text'] for c in retrieved_concepts])}",
                "citation_rate": 0.0,
                "citations_found": 0,
                "total_concepts": len(retrieved_concepts),
            }

        return {
            "query": query_text,
            "retrieved_concepts": retrieved_concepts,
            "raw_response": self._format_raw_response(retrieved_concepts),
            "smoothed_response": smoothed,
        }

    def _format_raw_response(self, concepts: List[Dict]) -> str:
        """Format raw retrieval results as text."""
        lines = []
        for i, concept in enumerate(concepts, 1):
            lines.append(f"{i}. {concept['text']} (score: {concept['score']:.3f})")
        return "\n".join(lines)

    def print_result(self, result: Dict[str, Any], index: int):
        """Pretty print a single result."""
        print(f"\n{'─' * 70}")
        print(f"Query #{index}: {result['query']}")
        print(f"{'─' * 70}")

        # Show raw retrieval
        print("\n📊 BEFORE (Raw Retrieval):")
        print(f"   Top {len(result['retrieved_concepts'])} concepts retrieved:")
        print()
        for i, concept in enumerate(result['retrieved_concepts'], 1):
            print(f"   {i}. {concept['text']}")
            print(f"      ID: {concept['id']}")
            print(f"      Score: {concept['score']:.3f}")

        # Show smoothed response
        print("\n✨ AFTER (LLM Smoothing):")
        smoothed = result['smoothed_response']
        print(f"   Response:")

        # Indent the response text
        response_text = smoothed['text']
        for line in response_text.split('\n'):
            print(f"   {line}")

        # Show citation metrics
        if self.llm_available:
            print()
            print(f"   📌 Citations:")
            print(f"      Rate: {smoothed['citation_rate']:.1%}")
            print(f"      Found: {smoothed['citations_found']}/{smoothed['total_concepts']}")

            # Validate citation format
            if smoothed['citation_rate'] >= 0.9:
                print(f"      Status: ✅ PASS (≥90% cited)")
            else:
                print(f"      Status: ⚠ LOW (<90% cited)")

    def run_single_concept_queries(self):
        """Run 5 single-concept queries."""
        print("\n" + "=" * 70)
        print("PART 1: SINGLE-CONCEPT QUERIES (5x)")
        print("=" * 70)

        queries = [
            "protein function",
            "enzyme catalysis",
            "cell membrane structure",
            "DNA replication process",
            "metabolic pathway regulation",
        ]

        results = []
        for i, query in enumerate(queries, 1):
            result = self.run_query(query, k=5)
            results.append(result)
            self.print_result(result, i)

        return results

    def run_dual_concept_queries(self):
        """Run 5 dual-concept queries."""
        print("\n" + "=" * 70)
        print("PART 2: DUAL-CONCEPT QUERIES (5x)")
        print("=" * 70)

        queries = [
            "protein and enzyme relationship",
            "cell membrane and transport mechanisms",
            "DNA and RNA synthesis",
            "metabolic pathways and energy production",
            "gene expression and regulation",
        ]

        results = []
        for i, query in enumerate(queries, 1):
            result = self.run_query(query, k=5)
            results.append(result)
            self.print_result(result, i)

        return results

    def run_demo(self):
        """Run complete demonstration."""
        # Part 1: Single-concept queries
        single_results = self.run_single_concept_queries()

        # Part 2: Dual-concept queries
        dual_results = self.run_dual_concept_queries()

        # Summary
        self.print_summary(single_results, dual_results)

        return {
            "single_concept_results": single_results,
            "dual_concept_results": dual_results,
        }

    def print_summary(self, single_results: List[Dict], dual_results: List[Dict]):
        """Print summary statistics."""
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)

        if not self.llm_available:
            print("\n⚠ LLM smoothing was not available - summary limited")
            return

        # Calculate citation rates
        single_citation_rates = [
            r['smoothed_response']['citation_rate']
            for r in single_results
        ]
        dual_citation_rates = [
            r['smoothed_response']['citation_rate']
            for r in dual_results
        ]

        # Overall stats
        all_citation_rates = single_citation_rates + dual_citation_rates

        print("\n📊 Citation Rate Statistics:")
        print(f"   Single-concept queries:")
        print(f"      Mean: {np.mean(single_citation_rates):.1%}")
        print(f"      Min:  {np.min(single_citation_rates):.1%}")
        print(f"      Max:  {np.max(single_citation_rates):.1%}")

        print(f"\n   Dual-concept queries:")
        print(f"      Mean: {np.mean(dual_citation_rates):.1%}")
        print(f"      Min:  {np.min(dual_citation_rates):.1%}")
        print(f"      Max:  {np.max(dual_citation_rates):.1%}")

        print(f"\n   Overall (10 queries):")
        print(f"      Mean: {np.mean(all_citation_rates):.1%}")
        print(f"      Target: ≥90%")

        if np.mean(all_citation_rates) >= 0.9:
            print(f"      Status: ✅ PASS")
        else:
            print(f"      Status: ⚠ BELOW TARGET")

        # Retrieval quality
        print("\n🎯 Retrieval Quality:")
        print(f"   Average concepts retrieved: 5")
        avg_top_score = np.mean([
            r['retrieved_concepts'][0]['score']
            for r in single_results + dual_results
        ])
        print(f"   Average top-1 score: {avg_top_score:.3f}")

        print("\n" + "=" * 70)
        print("✅ DEMO COMPLETE")
        print("=" * 70)


def main():
    """Main demo entry point."""
    # Check for required files
    npz_path = "artifacts/ontology_4k_full.npz"
    index_path = "artifacts/ontology_4k_full.index"

    if not Path(npz_path).exists():
        print(f"❌ NPZ file not found: {npz_path}")
        print("   Run ingestion first: ./scripts/ingest_ontologies_limited.sh")
        sys.exit(1)

    if not Path(index_path).exists():
        print(f"❌ Index file not found: {index_path}")
        print("   Build FAISS index first")
        sys.exit(1)

    # Run demo
    demo = PipelineDemo(npz_path=npz_path, index_path=index_path)
    results = demo.run_demo()

    # Optionally save results
    output_file = "RAG/results/pipeline_demo_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()

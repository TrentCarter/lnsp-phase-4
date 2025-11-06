#!/usr/bin/env python3
"""
Comprehensive test of sentence-aware retrieval system.

Runs 3 iterations of testing with different configurations:
1. Baseline (paragraph-only)
2. Sentence reranking
3. Sentence reranking + directional adapter

Logs all results to a table for comparison.

Usage:
    python tools/test_sentence_retrieval.py \
        --sent-bank artifacts/arxiv_sentence_bank.npz \
        --para-npz artifacts/lvm/arxiv_papers_210_768d.npz \
        --adapter artifacts/directional_adapter.npz \
        --out-table artifacts/sentence_retrieval_results.json
"""
import argparse
import json
import numpy as np
import os
import sys
from typing import List, Tuple, Dict

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retriever.sent_para_rerank import SentenceParaReranker


class RetrievalTester:
    """Test sentence-aware retrieval with multiple configurations."""

    def __init__(
        self,
        para_npz: str,
        sent_bank: str,
        adapter_npz: str = None
    ):
        """Load data for testing."""
        print("="*60)
        print("Loading test data...")

        # Load paragraph-level data
        print(f"  Paragraphs: {para_npz}")
        para_data = np.load(para_npz, allow_pickle=True)
        self.para_vecs = para_data["vectors"].astype(np.float32)
        self.para_texts = para_data.get("concept_texts", [])
        self.para_ids = np.arange(len(self.para_vecs))

        # Normalize paragraph vectors
        self.para_vecs /= (np.linalg.norm(self.para_vecs, axis=1, keepdims=True) + 1e-9)

        print(f"    Loaded {len(self.para_vecs)} paragraphs")

        # Load sentence bank
        print(f"  Sentence bank: {sent_bank}")
        self.reranker = SentenceParaReranker(sent_bank, top_n_sents=5)

        # Load directional adapter (optional)
        self.adapter_W = None
        if adapter_npz and os.path.exists(adapter_npz):
            print(f"  Adapter: {adapter_npz}")
            adapter_data = np.load(adapter_npz, allow_pickle=True)
            self.adapter_W = adapter_data["W"].astype(np.float32)
            print(f"    Loaded adapter {self.adapter_W.shape}")
        else:
            print("  Adapter: None (will be tested without adapter)")

        print("="*60)

    def create_test_queries(self, n_queries: int = 50) -> List[Tuple[np.ndarray, int]]:
        """
        Create test queries from paragraph data.

        Each query is a paragraph vector, target is the next paragraph.

        Returns:
            List of (query_vec, target_para_id)
        """
        queries = []

        # Use paragraphs 0..N-1 as queries, paragraphs 1..N as targets
        for i in range(min(n_queries, len(self.para_vecs) - 1)):
            query_vec = self.para_vecs[i]
            target_id = i + 1
            queries.append((query_vec, target_id))

        return queries

    def retrieve_top_k(
        self,
        query_vec: np.ndarray,
        k: int = 10,
        use_adapter: bool = False
    ) -> List[Tuple[int, float]]:
        """
        Retrieve top-K paragraphs using cosine similarity.

        Args:
            query_vec: Query vector [768]
            k: Number of results
            use_adapter: Apply directional adapter

        Returns:
            List of (para_id, cosine_score) sorted by score descending
        """
        if use_adapter and self.adapter_W is not None:
            # Apply adapter
            query_adapted = query_vec @ self.adapter_W
            query_adapted /= (np.linalg.norm(query_adapted) + 1e-9)

            para_adapted = self.para_vecs @ self.adapter_W
            para_adapted /= (np.linalg.norm(para_adapted, axis=1, keepdims=True) + 1e-9)

            scores = para_adapted @ query_adapted
        else:
            # Standard cosine
            scores = self.para_vecs @ query_vec

        # Get top-K
        top_k_indices = np.argsort(scores)[-k:][::-1]
        results = [(int(idx), float(scores[idx])) for idx in top_k_indices]

        return results

    def compute_metrics(
        self,
        results: List[List[Tuple[int, float]]],
        targets: List[int],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Compute retrieval metrics.

        Args:
            results: List of retrieval results for each query
            targets: List of target paragraph IDs
            k_values: K values for R@K

        Returns:
            Dict of metrics
        """
        metrics = {}

        for k in k_values:
            hits = 0
            for result, target in zip(results, targets):
                retrieved_ids = [r[0] for r in result[:k]]
                if target in retrieved_ids:
                    hits += 1

            recall = hits / len(targets) if targets else 0.0
            metrics[f"R@{k}"] = recall

        # MRR (Mean Reciprocal Rank)
        mrr_sum = 0.0
        for result, target in zip(results, targets):
            retrieved_ids = [r[0] for r in result]
            if target in retrieved_ids:
                rank = retrieved_ids.index(target) + 1
                mrr_sum += 1.0 / rank

        metrics["MRR"] = mrr_sum / len(targets) if targets else 0.0

        return metrics

    def test_configuration(
        self,
        config_name: str,
        use_reranker: bool,
        use_adapter: bool,
        n_queries: int = 50,
        k: int = 10
    ) -> Dict:
        """
        Test a specific configuration.

        Args:
            config_name: Configuration name
            use_reranker: Use sentence-based reranking
            use_adapter: Use directional adapter
            n_queries: Number of test queries
            k: Top-K retrieval

        Returns:
            Results dict
        """
        print(f"\nTesting: {config_name}")
        print(f"  Reranker: {use_reranker}, Adapter: {use_adapter}")

        # Create test queries
        queries = self.create_test_queries(n_queries)
        targets = [q[1] for q in queries]

        # Retrieve for each query
        all_results = []

        for query_vec, target_id in queries:
            # Stage 1: Paragraph retrieval
            para_results = self.retrieve_top_k(query_vec, k=k, use_adapter=use_adapter)

            # Stage 2: Sentence reranking (if enabled)
            if use_reranker:
                # Convert to format expected by reranker
                para_candidates = [(pid, score) for pid, score in para_results]
                para_vecs = self.para_vecs[[pid for pid, _ in para_results]]

                reranked = self.reranker.rerank_with_sentences(
                    query_vec, para_candidates, para_vecs
                )

                # Convert back to (para_id, score) format
                results = [(pid, fused_score) for pid, fused_score, _, _ in reranked]
            else:
                results = para_results

            all_results.append(results)

        # Compute metrics
        metrics = self.compute_metrics(all_results, targets, k_values=[1, 5, 10])

        result = {
            "config": config_name,
            "use_reranker": use_reranker,
            "use_adapter": use_adapter,
            "n_queries": n_queries,
            "k": k,
            "metrics": metrics
        }

        print(f"  R@1: {metrics['R@1']:.3f}")
        print(f"  R@5: {metrics['R@5']:.3f}")
        print(f"  R@10: {metrics['R@10']:.3f}")
        print(f"  MRR: {metrics['MRR']:.3f}")

        return result


def main():
    ap = argparse.ArgumentParser(description="Test sentence-aware retrieval")
    ap.add_argument("--sent-bank", required=True,
                   help="Sentence bank NPZ")
    ap.add_argument("--para-npz", required=True,
                   help="Paragraph-level data NPZ")
    ap.add_argument("--adapter", default=None,
                   help="Directional adapter NPZ (optional)")
    ap.add_argument("--out-table", required=True,
                   help="Output JSON table")
    ap.add_argument("--n-queries", type=int, default=50,
                   help="Number of test queries")
    args = ap.parse_args()

    # Initialize tester
    tester = RetrievalTester(
        para_npz=args.para_npz,
        sent_bank=args.sent_bank,
        adapter_npz=args.adapter
    )

    # Run 3 configurations
    results = []

    # Config 1: Baseline (paragraph-only, no adapter)
    results.append(tester.test_configuration(
        config_name="1_baseline_para_only",
        use_reranker=False,
        use_adapter=False,
        n_queries=args.n_queries
    ))

    # Config 2: Sentence reranking (no adapter)
    results.append(tester.test_configuration(
        config_name="2_sent_rerank",
        use_reranker=True,
        use_adapter=False,
        n_queries=args.n_queries
    ))

    # Config 3: Sentence reranking + adapter (if available)
    if tester.adapter_W is not None:
        results.append(tester.test_configuration(
            config_name="3_sent_rerank_adapter",
            use_reranker=True,
            use_adapter=True,
            n_queries=args.n_queries
        ))
    else:
        print("\n⚠️  Skipping adapter test (no adapter available)")

    # Save results
    os.makedirs(os.path.dirname(args.out_table) or ".", exist_ok=True)
    with open(args.out_table, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Config':<30} {'R@1':<8} {'R@5':<8} {'R@10':<8} {'MRR':<8}")
    print("-"*60)

    for r in results:
        m = r["metrics"]
        print(f"{r['config']:<30} {m['R@1']:<8.3f} {m['R@5']:<8.3f} {m['R@10']:<8.3f} {m['MRR']:<8.3f}")

    print("="*60)
    print(f"\n✓ Results saved to: {args.out_table}")

    # Check acceptance gates
    if len(results) >= 2:
        baseline = results[0]["metrics"]
        best = results[-1]["metrics"]

        r5_improvement = (best["R@5"] - baseline["R@5"]) * 100
        mrr_improvement = best["MRR"] - baseline["MRR"]

        print("\nAcceptance Gates:")
        print(f"  R@5 improvement: {r5_improvement:+.1f}pp (target: +7pp)")
        print(f"  MRR improvement: {mrr_improvement:+.3f} (target: +0.04)")

        if r5_improvement >= 7.0 and mrr_improvement >= 0.04:
            print("  ✅ PASSED all gates!")
        else:
            print("  ⚠️  Did not meet all gates")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Retrieval Evaluation Framework
===============================

Evaluates LVM-based retrieval with standard IR metrics:
- Recall@1, Recall@5, Recall@10
- MRR@10 (Mean Reciprocal Rank)
- Latency (P50, P95, P99)

Supports baseline vs. improved comparison to measure reranking lift.
"""

import torch
import torch.nn as nn
import numpy as np
import faiss
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple, Optional
import json
import sys

# Import AMN model
sys.path.insert(0, 'app/lvm')
from models import AttentionMixtureNetwork

# Import reranking strategies
from rerank_strategies import rerank_pipeline


class RetrievalEvaluator:
    """Evaluate retrieval quality for LVM predictions."""

    def __init__(
        self,
        model_path: str = "artifacts/lvm/production_model",
        vectors_path: str = "artifacts/wikipedia_500k_corrected_vectors.npz",
        test_sequences_path: str = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    ):
        print("🔬 Loading Retrieval Evaluation Suite...")
        print("=" * 80)

        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load AMN model
        print(f"\n📦 Loading AMN model from {model_path}...")
        self.model = self._load_amn_model(model_path)
        print(f"✅ AMN model loaded")

        # Load vectors and texts
        print(f"\n📦 Loading Wikipedia vectors from {vectors_path}...")
        self.vectors_data = np.load(vectors_path, allow_pickle=True)
        self.concept_texts = self.vectors_data['concept_texts']
        self.vectors = self.vectors_data['vectors']
        print(f"✅ Loaded {len(self.concept_texts):,} concepts")

        # Build metadata lookup
        print(f"\n📦 Building metadata index...")
        self._build_metadata_index()
        print(f"✅ Metadata index ready")

        # Load test sequences
        print(f"\n📦 Loading test sequences from {test_sequences_path}...")
        self.test_data = np.load(test_sequences_path, allow_pickle=True)
        self.test_X = self.test_data['context_sequences']  # (N, 5, 768)
        self.test_y = self.test_data['target_vectors']     # (N, 768)
        self.test_article_ids = self.test_data.get('article_ids', None)
        print(f"✅ Loaded {len(self.test_X):,} test sequences")

        # Build FAISS index
        print(f"\n📦 Building FAISS index...")
        self._build_faiss_index()
        print(f"✅ FAISS index built ({self.index.ntotal:,} vectors)")

        print("\n" + "=" * 80)
        print("🎯 Evaluation Suite Ready!")
        print("=" * 80 + "\n")

    def _load_amn_model(self, model_path: str) -> nn.Module:
        """Load AMN model from checkpoint."""
        model_dir = Path(model_path)
        checkpoint_path = model_dir / "best_model.pt"

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        hparams = checkpoint.get('hyperparameters', {})

        model = AttentionMixtureNetwork(
            input_dim=hparams.get('input_dim', 768),
            d_model=hparams.get('d_model', 256),
            hidden_dim=hparams.get('hidden_dim', 512)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    def _build_metadata_index(self):
        """Build lookup from vector index to (article_index, chunk_index)."""
        # For now, assume sequential indexing
        # TODO: Load actual metadata if available
        self.vec_idx_to_meta = {}
        for i in range(len(self.vectors)):
            # Placeholder: would normally load from database
            self.vec_idx_to_meta[i] = {
                "article_index": i // 100,  # Rough estimate
                "chunk_index": i % 100,
                "vector_index": i
            }

    def _build_faiss_index(self):
        """Build FAISS index for nearest neighbor search."""
        # Normalize vectors for cosine similarity (inner product after normalization)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)

        # Use inner product index
        self.index = faiss.IndexFlatIP(768)
        self.index.add(vectors_norm.astype('float32'))

    def evaluate(
        self,
        use_reranking: bool = False,
        k_retrieve: int = 50,
        k_final: int = 10,
        mmr_lambda: float = 0.7,
        w_cos: float = 1.0,
        w_same_article: float = 0.05,
        w_next_gap: float = 0.12,
        tau: float = 3.0,
        n_samples: Optional[int] = None
    ) -> Dict:
        """
        Evaluate retrieval on test set.

        Args:
            use_reranking: Whether to apply reranking strategies
            k_retrieve: Number of candidates to retrieve initially
            k_final: Final number of candidates after reranking
            mmr_lambda: MMR lambda parameter
            w_cos: Cosine weight for sequence reranking
            w_same_article: Same-article bonus
            w_next_gap: Next-chunk bonus weight
            tau: Gap penalty temperature
            n_samples: Number of test samples (None = all)

        Returns:
            Dictionary of metrics
        """
        print(f"\n🔬 Evaluating retrieval...")
        print(f"   Mode: {'WITH reranking' if use_reranking else 'BASELINE (no reranking)'}")
        print(f"   Test samples: {n_samples or len(self.test_X):,}")
        print(f"   K (retrieve): {k_retrieve}, K (final): {k_final}")

        # Metrics
        r1 = r5 = r10 = 0
        mrr = 0.0
        latencies = []
        n = 0

        # Sample test set if requested
        test_indices = range(len(self.test_X)) if n_samples is None else range(min(n_samples, len(self.test_X)))

        for idx in test_indices:
            context_vecs = self.test_X[idx]  # (5, 768)
            target_vec = self.test_y[idx]    # (768,)

            # Run AMN inference
            with torch.no_grad():
                context_t = torch.from_numpy(context_vecs).float().unsqueeze(0).to(self.device)
                pred_vec = self.model(context_t).cpu().numpy()[0]  # (768,)

            # Measure retrieval latency
            t0 = perf_counter()

            # Normalize for FAISS
            pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)

            # Initial retrieval
            D, I = self.index.search(pred_norm.reshape(1, -1).astype('float32'), k_retrieve)
            distances = D[0]
            indices = I[0]

            # Build candidates list
            candidates = []
            for i, (dist, vec_idx) in enumerate(zip(distances, indices)):
                meta = self.vec_idx_to_meta.get(int(vec_idx), {})
                candidates.append((
                    self.concept_texts[vec_idx],  # text
                    float(dist),                   # cosine similarity
                    self.vectors[vec_idx],         # vector
                    meta                           # metadata
                ))

            # Apply reranking if requested
            if use_reranking:
                # Get last context metadata
                # For now, use placeholder (would normally extract from test data)
                last_ctx_meta = {
                    "article_index": meta.get("article_index", 0),
                    "chunk_index": meta.get("chunk_index", 0) - 1
                }

                # Apply integrated reranking pipeline
                reranked = rerank_pipeline(
                    query_vec=pred_norm,
                    candidates=candidates,
                    last_ctx_meta=last_ctx_meta,
                    k_final=k_final,
                    mmr_lambda=mmr_lambda,
                    w_cos=w_cos,
                    w_same_article=w_same_article,
                    w_next_gap=w_next_gap,
                    tau=tau
                )

                # Extract final candidates
                final_candidates = [(text, cos, meta) for _, text, cos, meta in reranked]
            else:
                # Baseline: just use top-k by cosine
                final_candidates = [(text, cos, meta) for text, cos, _, meta in candidates[:k_final]]

            dt = (perf_counter() - t0) * 1000  # ms
            latencies.append(dt)

            # Find ground truth in results
            target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)
            target_match_idx = None

            # Check if any candidate matches ground truth (high cosine similarity)
            for rank, (text, cos, meta) in enumerate(final_candidates):
                # Compute similarity to ground truth
                cand_vec = self.vectors[meta.get("vector_index", -1)] if "vector_index" in meta else None
                if cand_vec is not None:
                    cand_norm = cand_vec / (np.linalg.norm(cand_vec) + 1e-8)
                    sim_to_truth = float(target_norm @ cand_norm)

                    # If very close to ground truth (> 0.99 similarity), consider it a match
                    if sim_to_truth > 0.99:
                        target_match_idx = rank
                        break

            # Update metrics
            if target_match_idx is not None:
                if target_match_idx == 0:
                    r1 += 1
                if target_match_idx < 5:
                    r5 += 1
                if target_match_idx < 10:
                    r10 += 1
                mrr += 1.0 / (target_match_idx + 1)

            n += 1

            # Progress indicator
            if (n % 100) == 0:
                print(f"   Processed {n:,} samples...")

        # Compute metrics
        metrics = {
            "n_samples": n,
            "recall@1": r1 / n if n > 0 else 0.0,
            "recall@5": r5 / n if n > 0 else 0.0,
            "recall@10": r10 / n if n > 0 else 0.0,
            "mrr@10": mrr / n if n > 0 else 0.0,
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p95_ms": float(np.percentile(latencies, 95)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "latency_mean_ms": float(np.mean(latencies)),
            "use_reranking": use_reranking
        }

        return metrics

    def compare_baseline_vs_improved(
        self,
        n_samples: Optional[int] = None,
        k_retrieve: int = 50,
        k_final: int = 10
    ) -> Tuple[Dict, Dict]:
        """
        Run baseline vs. improved evaluation and compare.

        Args:
            n_samples: Number of test samples (None = all)
            k_retrieve: Number of candidates to retrieve
            k_final: Final number of candidates

        Returns:
            (baseline_metrics, improved_metrics)
        """
        print("\n" + "=" * 80)
        print("📊 BASELINE VS. IMPROVED COMPARISON")
        print("=" * 80)

        # Baseline (no reranking)
        print("\n[1/2] Running BASELINE evaluation...")
        baseline = self.evaluate(
            use_reranking=False,
            k_retrieve=k_retrieve,
            k_final=k_final,
            n_samples=n_samples
        )

        # Improved (with reranking)
        print("\n[2/2] Running IMPROVED evaluation (with reranking)...")
        improved = self.evaluate(
            use_reranking=True,
            k_retrieve=k_retrieve,
            k_final=k_final,
            n_samples=n_samples
        )

        # Display comparison
        self._display_comparison(baseline, improved)

        return baseline, improved

    def _display_comparison(self, baseline: Dict, improved: Dict):
        """Display side-by-side comparison of metrics."""
        print("\n" + "=" * 80)
        print("📈 RESULTS COMPARISON")
        print("=" * 80)

        metrics = [
            ("Recall@1", "recall@1", "↑"),
            ("Recall@5", "recall@5", "↑"),
            ("Recall@10", "recall@10", "↑"),
            ("MRR@10", "mrr@10", "↑"),
            ("Latency P50", "latency_p50_ms", "↓"),
            ("Latency P95", "latency_p95_ms", "↓")
        ]

        print(f"\n{'Metric':<20} {'Baseline':<15} {'Improved':<15} {'Lift':<15}")
        print("-" * 80)

        for name, key, direction in metrics:
            base_val = baseline[key]
            imp_val = improved[key]

            # Compute lift
            if "latency" in key:
                # For latency, show absolute difference
                lift = imp_val - base_val
                lift_str = f"{lift:+.2f} ms"
            else:
                # For recall/MRR, show percentage point difference
                lift = (imp_val - base_val) * 100
                lift_str = f"{lift:+.1f} pp"

            # Format values
            if "latency" in key:
                base_str = f"{base_val:.2f} ms"
                imp_str = f"{imp_val:.2f} ms"
            else:
                base_str = f"{base_val * 100:.1f}%"
                imp_str = f"{imp_val * 100:.1f}%"

            # Determine if improvement is good
            is_good = (lift > 0 and direction == "↑") or (lift < 0 and direction == "↓")
            lift_indicator = "✅" if is_good else "⚠️"

            print(f"{name:<20} {base_str:<15} {imp_str:<15} {lift_indicator} {lift_str}")

        print("=" * 80 + "\n")


def main():
    """Run evaluation comparison."""
    evaluator = RetrievalEvaluator()

    # Run comparison on full test set (or sample)
    baseline, improved = evaluator.compare_baseline_vs_improved(
        n_samples=None,  # Use all test samples (7,145)
        k_retrieve=50,
        k_final=10
    )

    # Save results
    results = {
        "baseline": baseline,
        "improved": improved,
        "lift": {
            "recall@1_pp": (improved["recall@1"] - baseline["recall@1"]) * 100,
            "recall@5_pp": (improved["recall@5"] - baseline["recall@5"]) * 100,
            "recall@10_pp": (improved["recall@10"] - baseline["recall@10"]) * 100,
            "mrr@10": improved["mrr@10"] - baseline["mrr@10"],
            "latency_p95_ms": improved["latency_p95_ms"] - baseline["latency_p95_ms"]
        }
    }

    output_file = "artifacts/lvm/retrieval_eval_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    main()

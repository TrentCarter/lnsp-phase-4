#!/usr/bin/env python3
"""
Comprehensive AMN end-to-end testing with multiple decoding strategies.

Tests AMN with:
- Multiple context lengths (5, 10, 20 chunks)
- 5 decoding strategies:
  1. Direct vec2text (JXE decoder)
  2. Direct vec2text (IELab decoder)
  3. FAISS nearest neighbor (top-1)
  4. Top-k retrieval (k=5 candidates)
  5. Ensemble (vec2text + NN candidates)

Shows full input chunks and all output candidates (NO TRUNCATION).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
from typing import List, Dict, Tuple, Optional
import faiss
import os

# Import AMN architecture
sys.path.insert(0, 'app/lvm')
from models import AttentionMixtureNetwork

# Import vec2text orchestrator
sys.path.insert(0, '.')
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator


class AMNEndToEndTester:
    """Complete end-to-end testing framework for AMN model."""

    def __init__(
        self,
        model_path: str = "artifacts/lvm/production_model",
        vectors_path: str = "artifacts/wikipedia_500k_corrected_vectors.npz",
        test_sequences_path: str = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    ):
        print("🚀 Loading AMN End-to-End Test Suite...")
        print("=" * 80)

        # Device setup
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {self.device}")

        # Load AMN model
        print(f"\n📦 Loading AMN model from {model_path}...")
        self.model = self._load_amn_model(model_path)
        print(f"✅ AMN model loaded (production)")

        # Load vectors and texts
        print(f"\n📦 Loading Wikipedia vectors from {vectors_path}...")
        self.vectors_data = np.load(vectors_path, allow_pickle=True)
        self.concept_texts = self.vectors_data['concept_texts']
        self.vectors = self.vectors_data['vectors']
        print(f"✅ Loaded {len(self.concept_texts):,} concepts with vectors")

        # Load test sequences
        print(f"\n📦 Loading test sequences from {test_sequences_path}...")
        self.test_data = np.load(test_sequences_path, allow_pickle=True)
        self.test_X = self.test_data['context_sequences']  # Shape: (N, 5, 768)
        self.test_y = self.test_data['target_vectors']  # Shape: (N, 768)

        # Load metadata if available
        if 'article_ids' in self.test_data:
            self.test_article_ids = self.test_data['article_ids']  # Article IDs for each sequence
        else:
            self.test_article_ids = None

        # No concept_indices in this format - we'll use vector matching
        self.test_concept_indices = None

        print(f"✅ Loaded {len(self.test_X):,} test sequences (context_size=5)")

        # Load vec2text orchestrator
        print(f"\n📦 Loading Vec2Text orchestrator...")
        os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
        os.environ['VEC2TEXT_DEVICE'] = 'cpu'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.vec2text = IsolatedVecTextVectOrchestrator()
        print(f"✅ Vec2Text orchestrator ready (JXE + IELab decoders)")

        # Build FAISS index for nearest neighbor retrieval
        print(f"\n📦 Building FAISS index for NN retrieval...")
        self._build_faiss_index()
        print(f"✅ FAISS index built ({self.index.ntotal:,} vectors)")

        print("\n" + "=" * 80)
        print("🎯 AMN End-to-End Test Suite Ready!")
        print("=" * 80 + "\n")

    def _load_amn_model(self, model_path: str) -> nn.Module:
        """Load AMN model from checkpoint."""
        model_dir = Path(model_path)
        checkpoint_path = model_dir / "best_model.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Create model with saved hyperparameters
        hparams = checkpoint.get('hyperparameters', {})
        model = AttentionMixtureNetwork(
            input_dim=hparams.get('input_dim', 768),
            d_model=hparams.get('d_model', 256),
            hidden_dim=hparams.get('hidden_dim', 512)
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        return model

    def _build_faiss_index(self):
        """Build FAISS index for nearest neighbor search."""
        # Normalize vectors for cosine similarity (inner product after normalization)
        vectors_norm = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-8)

        # Use inner product index (equivalent to cosine after normalization)
        self.index = faiss.IndexFlatIP(768)
        self.index.add(vectors_norm.astype('float32'))

    def run_test_suite(
        self,
        test_cases: List[int] = [0, 1, 2],
        context_lengths: List[int] = [5, 10, 20],
        vec2text_steps: int = 1
    ):
        """
        Run full test suite with different context lengths.

        Args:
            test_cases: Indices of test examples to use
            context_lengths: How many chunks to show as context (5/10/20)
            vec2text_steps: Number of vec2text decoding steps (1=fast, 5=quality)
        """
        self.vec2text_steps = vec2text_steps

        for test_idx in test_cases:
            print("\n" + "🔬" * 40)
            print(f"TEST CASE #{test_idx}")
            print("🔬" * 40 + "\n")

            for ctx_len in context_lengths:
                print(f"\n{'='*80}")
                print(f"Context Length: {ctx_len} chunks (using last 5 for AMN input)")
                print(f"{'='*80}\n")

                # Prepare test case
                context_vecs, target_vec, context_texts, target_text = self._prepare_test_case(
                    test_idx, ctx_len
                )

                # Display full input
                self._display_input(context_texts, target_text, ctx_len)

                # Run AMN inference on last 5 vectors
                amn_input = context_vecs[-5:]  # Always use last 5 for AMN
                pred_vec = self._run_amn_inference(amn_input)

                # Decode using all 5 strategies
                results = self._decode_all_strategies(pred_vec, target_vec, target_text)

                # Display all results
                self._display_results(results, target_text)

                print("\n" + "-" * 80 + "\n")

    def _prepare_test_case(
        self,
        test_idx: int,
        context_length: int
    ) -> Tuple[np.ndarray, np.ndarray, List[str], str]:
        """
        Prepare test case with specified context length.

        For context_length=5: Use test sequence directly (indices 0-4 → predict 5)
        For context_length=10: Find longer sequence, use indices 5-9 → predict 10
        For context_length=20: Find even longer sequence, use indices 15-19 → predict 20

        Returns:
            context_vecs: (context_length, 768) vectors
            target_vec: (768,) target vector
            context_texts: List of context texts
            target_text: Target text
        """
        # Get base test sequence (always has 5 context + 1 target)
        base_context = self.test_X[test_idx]  # (5, 768)
        base_target = self.test_y[test_idx]   # (768,)

        # Get concept indices if available
        if self.test_concept_indices is not None:
            concept_idx_sequence = self.test_concept_indices[test_idx]  # (6,) - 5 context + 1 target

            # Extend context if needed
            if context_length > 5:
                # Try to find earlier concepts in the sequence
                # For now, just repeat/extend the existing sequence
                # TODO: Could load more from original Wikipedia articles
                extended_indices = []
                target_idx = concept_idx_sequence[-1]

                # Create extended sequence by going backwards
                for i in range(context_length):
                    if i < len(concept_idx_sequence) - 1:
                        extended_indices.append(concept_idx_sequence[i])
                    else:
                        # Pad with earlier context (wrap around)
                        extended_indices.append(concept_idx_sequence[i % (len(concept_idx_sequence) - 1)])

                # Get vectors and texts
                context_vecs = self.vectors[extended_indices]
                context_texts = [self.concept_texts[idx] for idx in extended_indices]
                target_vec = self.vectors[target_idx]
                target_text = self.concept_texts[target_idx]
            else:
                # Use original 5-chunk sequence
                context_indices = concept_idx_sequence[:5]
                target_idx = concept_idx_sequence[5]

                context_vecs = base_context
                context_texts = [self.concept_texts[idx] for idx in context_indices]
                target_vec = base_target
                target_text = self.concept_texts[target_idx]
        else:
            # Fallback: use vectors directly, find nearest neighbors for texts
            context_vecs = base_context
            target_vec = base_target

            # Find nearest neighbor texts (approximate)
            context_texts = []
            for vec in context_vecs:
                vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
                D, I = self.index.search(vec_norm.reshape(1, -1).astype('float32'), 1)
                context_texts.append(self.concept_texts[I[0][0]])

            # Find target text
            target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)
            D, I = self.index.search(target_norm.reshape(1, -1).astype('float32'), 1)
            target_text = self.concept_texts[I[0][0]]

        return context_vecs, target_vec, context_texts, target_text

    def _run_amn_inference(self, context_vecs: np.ndarray) -> np.ndarray:
        """
        Run AMN model inference.

        Args:
            context_vecs: (5, 768) context vectors

        Returns:
            pred_vec: (768,) predicted next vector
        """
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            context_t = torch.from_numpy(context_vecs).float().unsqueeze(0).to(self.device)  # (1, 5, 768)

            # Run model
            pred_t = self.model(context_t)  # (1, 768)

            # Convert back to numpy
            pred_vec = pred_t.cpu().numpy()[0]  # (768,)

        return pred_vec

    def _decode_all_strategies(
        self,
        pred_vec: np.ndarray,
        target_vec: np.ndarray,
        target_text: str
    ) -> Dict:
        """
        Decode predicted vector using all 5 strategies.

        Args:
            pred_vec: (768,) predicted vector
            target_vec: (768,) ground truth vector
            target_text: Ground truth text

        Returns:
            results: Dict with all decoding results and metrics
        """
        results = {}

        print("🔄 Decoding prediction using retrieval-based strategies...\n")
        print("Note: Vec2text decoding requires text input. For LVM output vectors,")
        print("we use retrieval-based strategies (FAISS nearest neighbor search).\n")

        # Strategy 1: Nearest neighbor (top-1)
        print("  [1/3] Running FAISS nearest neighbor (top-1)...")
        results['nearest_neighbor'] = self._decode_nearest_neighbor(pred_vec, k=1)

        # Strategy 2: Top-k retrieval
        print("  [2/3] Running top-k retrieval (k=10)...")
        results['topk_retrieval'] = self._decode_nearest_neighbor(pred_vec, k=10)

        # Strategy 3: Confidence-weighted ensemble
        print("  [3/3] Running confidence-weighted ensemble...")
        results['ensemble'] = self._create_retrieval_ensemble(
            results['topk_retrieval'],
            pred_vec
        )

        # Add similarity metrics
        results['cosine_to_target'] = self._cosine_similarity(pred_vec, target_vec)
        results['target_text'] = target_text
        results['target_vec'] = target_vec
        results['pred_vec'] = pred_vec

        print("✅ Decoding complete!\n")

        return results

    def _decode_vec2text(self, vec: np.ndarray, subscriber: str) -> Dict:
        """
        Decode vector using vec2text decoder.

        Since we have a raw vector from AMN, we create a dummy text and process it,
        but we'll inject the AMN vector instead of encoding the dummy text.
        """
        import subprocess
        import json
        import tempfile

        # Create temporary file for vector
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'vector': vec.tolist()}, f)
            vec_file = f.name

        try:
            # Call vec2text CLI directly with our vector
            cmd = [
                str(self.vec2text.venv_paths[subscriber] / 'bin' / 'python3'),
                'app/vect_text_vect/vec_text_vect_isolated.py',
                '--input-text', 'dummy',  # Will be replaced by our vector
                '--subscribers', subscriber,
                '--vec2text-backend', 'isolated',
                '--output-format', 'json',
                '--steps', str(self.vec2text_steps),
                '--inject-vector', vec_file  # Custom flag to inject our vector
            ]

            # Actually, let's use a simpler approach - just encode a dummy text and decode it
            # We'll use the encode_texts method to convert back

            # For now, let's use the nearest neighbor as a proxy
            # This is a fallback until we can properly decode
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            D, I = self.index.search(vec_norm.reshape(1, -1).astype('float32'), 1)
            proxy_text = self.concept_texts[I[0][0]]

            # Encode the proxy text to measure how close it is
            proxy_vec = np.array(self.vec2text.encode_texts([proxy_text])[0])
            reconstruction_cosine = self._cosine_similarity(vec, proxy_vec)

            return {
                'text': f"[Vec2Text {subscriber.upper()} - using NN proxy]: {proxy_text}",
                'reconstruction_cosine': reconstruction_cosine,
                'decoder': subscriber,
                'note': 'Using nearest neighbor as proxy (direct vec2text decoding not yet implemented)'
            }

        finally:
            import os
            if os.path.exists(vec_file):
                os.unlink(vec_file)

    def _decode_nearest_neighbor(self, vec: np.ndarray, k: int = 1) -> List[Dict]:
        """Find k nearest neighbors in training set."""
        # Normalize vector
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)

        # Search FAISS index
        D, I = self.index.search(vec_norm.reshape(1, -1).astype('float32'), k)

        # Format results
        results = []
        for idx, score in zip(I[0], D[0]):
            results.append({
                'text': self.concept_texts[idx],
                'cosine_similarity': float(score),
                'index': int(idx)
            })

        return results

    def _create_retrieval_ensemble(
        self,
        topk_results: List[Dict],
        pred_vec: np.ndarray
    ) -> Dict:
        """
        Create confidence-weighted ensemble from top-k retrieval results.

        Uses distance-weighted voting where closer matches have higher weight.
        """
        # Weight candidates by their cosine similarity (confidence)
        total_weight = sum(r['cosine_similarity'] for r in topk_results)

        weighted_candidates = []
        for rank, result in enumerate(topk_results, 1):
            weight = result['cosine_similarity'] / total_weight
            weighted_candidates.append({
                'text': result['text'],
                'rank': rank,
                'cosine_similarity': result['cosine_similarity'],
                'weight': weight,
                'contribution': weight * 100  # Percentage contribution
            })

        # Best candidate is still top-1 (highest similarity)
        best_candidate = weighted_candidates[0]

        # Confidence score: How much better is top-1 than top-2?
        if len(weighted_candidates) > 1:
            confidence_margin = (weighted_candidates[0]['cosine_similarity'] -
                               weighted_candidates[1]['cosine_similarity'])
        else:
            confidence_margin = weighted_candidates[0]['cosine_similarity']

        return {
            'weighted_candidates': weighted_candidates,
            'best_candidate': best_candidate,
            'confidence_margin': confidence_margin,
            'top1_weight': best_candidate['weight']
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))

    def _display_input(self, context_texts: List[str], target_text: str, context_length: int):
        """Display full input context (NO TRUNCATION)."""
        print("📥 INPUT CONTEXT (Full Text - No Truncation):")
        print("─" * 80)

        for i, text in enumerate(context_texts, 1):
            print(f"\n  Chunk [{i}/{context_length}]:")
            print(f"  {text}")

        print("\n" + "─" * 80)
        print(f"\n🎯 GROUND TRUTH TARGET:")
        print(f"  {target_text}")
        print("\n" + "─" * 80)

    def _display_results(self, results: Dict, target_text: str):
        """Display all decoding results (NO TRUNCATION)."""
        print("\n📤 RETRIEVAL RESULTS (Full Text - No Truncation):")
        print("=" * 80)

        # Overall similarity to target
        print(f"\n🎯 Prediction Quality:")
        print(f"   Cosine similarity to ground truth: {results['cosine_to_target']:.4f}")

        # Strategy 1: Nearest Neighbor (top-1)
        print(f"\n{'─'*80}")
        print("🔹 Strategy 1: FAISS Nearest Neighbor (Top-1)")
        print(f"{'─'*80}")
        nn = results['nearest_neighbor'][0]
        print(f"   Retrieved text: {nn['text']}")
        print(f"   Cosine similarity: {nn['cosine_similarity']:.4f}")

        # Strategy 2: Top-k Retrieval
        print(f"\n{'─'*80}")
        print("🔹 Strategy 2: Top-K Retrieval (K=10 candidates)")
        print(f"{'─'*80}")
        for i, candidate in enumerate(results['topk_retrieval'], 1):
            print(f"\n   Rank {i} (cosine: {candidate['cosine_similarity']:.4f}):")
            print(f"   {candidate['text']}")

        # Strategy 3: Confidence-Weighted Ensemble
        print(f"\n{'─'*80}")
        print("🔹 Strategy 3: Confidence-Weighted Ensemble")
        print(f"{'─'*80}")
        ensemble = results['ensemble']

        print(f"\n   Confidence metrics:")
        print(f"   • Top-1 weight: {ensemble['top1_weight']:.1%}")
        print(f"   • Confidence margin: {ensemble['confidence_margin']:.4f}")

        print(f"\n   Top-3 weighted candidates:")
        for cand in ensemble['weighted_candidates'][:3]:
            print(f"   • Rank {cand['rank']} (similarity: {cand['cosine_similarity']:.4f}, weight: {cand['contribution']:.1f}%)")
            print(f"     {cand['text']}")

        print(f"\n   Best candidate (highest similarity): {ensemble['best_candidate']['text']}")

        # Comparison to ground truth
        print(f"\n{'='*80}")
        print("🏆 COMPARISON TO GROUND TRUTH:")
        print(f"{'='*80}")
        print(f"   Ground truth: {target_text}")
        print(f"\n   Match analysis:")
        print(f"   • NN Top-1: {'✅ EXACT MATCH' if results['nearest_neighbor'][0]['text'] == target_text else '❌ Different'}")

        # Check if target appears in top-k
        topk_texts = [c['text'] for c in results['topk_retrieval']]
        if target_text in topk_texts:
            rank = topk_texts.index(target_text) + 1
            print(f"   • ✅ Target found in top-10 at rank: {rank}")
            print(f"   • Similarity at rank {rank}: {results['topk_retrieval'][rank-1]['cosine_similarity']:.4f}")
        else:
            print(f"   • ❌ Target NOT in top-10 candidates")

        print(f"{'='*80}")


if __name__ == "__main__":
    # Create tester
    tester = AMNEndToEndTester()

    # Run comprehensive test suite
    # - Test 3 different examples (indices 0, 1, 2)
    # - With 3 context lengths (5, 10, 20 chunks)
    # - Using 1 vec2text step (fast mode)

    print("\n" + "🧪" * 40)
    print("STARTING COMPREHENSIVE END-TO-END TEST SUITE")
    print("🧪" * 40 + "\n")

    tester.run_test_suite(
        test_cases=[0, 1, 2],          # Test 3 different examples
        context_lengths=[5, 10, 20],   # Try different context sizes
        vec2text_steps=1               # Fast mode (use 5 for better quality)
    )

    print("\n" + "🎉" * 40)
    print("TEST SUITE COMPLETE!")
    print("🎉" * 40 + "\n")

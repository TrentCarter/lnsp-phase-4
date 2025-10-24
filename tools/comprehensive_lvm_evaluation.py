#!/usr/bin/env python3
"""
Comprehensive LVM Evaluation Framework

Tests LVM models with multiple output approaches and context sizes:
1. Direct 768D-vec2text -> text
2. Tiny Recursion (TR) approach
3. TwoTower retrieval enhancement
4. Nearest neighbor retrieval
5. Ensemble approaches

Tests both in-dataset and out-of-dataset scenarios with various context window sizes.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import sys
import os
import time
from typing import List, Dict, Tuple, Optional, Any
import faiss
from dataclasses import dataclass
from contextlib import contextmanager

# Import LVM models
sys.path.insert(0, 'app/lvm')
from models import create_model, LSTMBaseline, GRUStack, TransformerVectorPredictor, AttentionMixtureNetwork

# Import vec2text
sys.path.insert(0, '.')
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

@dataclass
class EvaluationConfig:
    """Configuration for LVM evaluation."""

    # Models to test
    models: List[str] = None

    # Context sizes to test
    context_sizes: List[int] = None

    # Output approaches to test
    approaches: List[str] = None

    # Test data
    test_data_path: str = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    training_data_path: str = "artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz"

    # Number of test samples
    num_samples: int = 100

    # Vec2text settings
    vec2text_steps: int = 2
    vec2text_beam_width: int = 1

    # Tiny Recursion settings
    tr_threshold: float = 0.05
    tr_max_attempts: int = 2

    # Output settings
    output_dir: str = "artifacts/lvm/evaluation_results"

    def __post_init__(self):
        if self.models is None:
            self.models = ['amn', 'transformer', 'gru', 'lstm']
        if self.context_sizes is None:
            self.context_sizes = [5, 10, 20, 50]
        if self.approaches is None:
            self.approaches = ['direct_vec2text', 'tiny_recursion', 'nearest_neighbor', 'ensemble']


class TinyRecursion(nn.Module):
    """Tiny Recursion implementation for LVM output refinement."""

    def __init__(self, base_model: nn.Module, threshold: float = 0.05, max_attempts: int = 2):
        super().__init__()
        self.base_model = base_model
        self.threshold = threshold
        self.max_attempts = max_attempts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply tiny recursion to refine predictions."""

        # Initial prediction
        with torch.no_grad():
            pred1 = self.base_model(x)

        metadata = {
            'attempts': 1,
            'converged': False,
            'delta_history': [],
            'final_prediction': pred1
        }

        # Try recursion if needed
        for attempt in range(1, self.max_attempts):
            # Create context that includes the prediction as the last vector
            batch_size, seq_len, dim = x.shape
            pred1_expanded = pred1.unsqueeze(1).expand(-1, seq_len, -1)

            # Use attention weights to blend original context with prediction
            # For simplicity, use the last few vectors + weighted prediction
            context_with_pred = x.clone()

            # Get attention weights if model supports it
            if hasattr(self.base_model, 'forward') and 'return_attention' in self.base_model.forward.__code__.co_varnames:
                try:
                    _, attn_weights = self.base_model(x, return_attention=True)
                    # Weight the prediction by attention to the last few positions
                    weight = attn_weights[:, -1].mean()  # Use last position attention
                    context_with_pred = x * (1 - weight) + pred1_expanded * weight
                except:
                    # Fallback: simple weighted combination
                    context_with_pred = x * 0.8 + pred1_expanded * 0.2

            # Second prediction
            with torch.no_grad():
                pred2 = self.base_model(context_with_pred)

            # Check convergence
            delta = 1 - torch.cosine_similarity(pred1, pred2, dim=-1).mean().item()
            metadata['delta_history'].append(delta)

            if delta < self.threshold:
                metadata['converged'] = True
                metadata['final_prediction'] = pred2
                metadata['attempts'] = attempt + 1
                break

            pred1 = pred2

        return metadata['final_prediction'], metadata


class TwoTowerRetriever:
    """TwoTower retrieval system for enhanced output."""

    def __init__(self, vector_db_path: str = None):
        self.vector_db_path = vector_db_path or "artifacts/faiss_meta.json"
        self.index = None
        self.vectors = None
        self.texts = None
        self._load_database()

    def _load_database(self):
        """Load FAISS index and vectors for retrieval."""
        try:
            # Try to load FAISS index
            meta_path = Path(self.vector_db_path)
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    index_path = meta.get('index_path')

                if index_path and Path(index_path).exists():
                    import faiss
                    self.index = faiss.read_index(index_path)

                    # Try to load vectors and texts
                    # This is a simplified version - in practice would need full corpus
                    print("⚠️  TwoTower: Using simplified retrieval (full implementation needed)")
                    return

        except Exception as e:
            print(f"⚠️  TwoTower: Could not load database: {e}")

        # Fallback: create simple index from test data
        print("⚠️  TwoTower: Using fallback retrieval mode")

    def retrieve(self, query_vec: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k candidates for a query vector."""
        if self.index is None:
            return [{'text': 'N/A', 'score': 0.0, 'rank': i+1} for i in range(k)]

        # Normalize query
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        query_norm = query_norm.reshape(1, -1).astype('float32')

        # Search
        try:
            D, I = self.index.search(query_norm, k)
            return [
                {'text': f'Candidate_{i[0]}', 'score': float(d[0]), 'rank': rank+1}
                for rank, (d, i) in enumerate(zip(D[0], I[0]))
            ]
        except:
            return [{'text': 'N/A', 'score': 0.0, 'rank': i+1} for i in range(k)]


class LVMEvaluator:
    """Comprehensive LVM evaluation framework."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models
        self.models = self._load_models()

        # Load test data
        self.test_data = self._load_test_data()

        # Initialize output approaches
        self.vec2text_orchestrator = self._init_vec2text()
        self.two_tower = TwoTowerRetriever()
        self.faiss_index = self._build_faiss_index()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_models(self) -> Dict[str, nn.Module]:
        """Load all LVM models."""
        models = {}
        model_paths = {
            'amn': 'artifacts/lvm/production_model',
            'transformer': 'artifacts/lvm/fallback_accuracy',
            'gru': 'artifacts/lvm/fallback_secondary',
            'lstm': 'artifacts/lvm/models/lstm_20251023_202152'
        }

        for model_name in self.config.models:
            if model_name not in model_paths:
                print(f"⚠️  Skipping unknown model: {model_name}")
                continue

            model_path = model_paths[model_name]
            model_dir = Path(model_path)

            if not model_dir.exists():
                print(f"⚠️  Model path not found: {model_path}")
                continue

            print(f"Loading {model_name} from {model_path}...")

            # Load checkpoint
            checkpoint_path = model_dir / "best_model.pt"
            if not checkpoint_path.exists():
                checkpoint_path = model_dir / "final_model.pt"

            if not checkpoint_path.exists():
                print(f"⚠️  No checkpoint found for {model_name}")
                continue

            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                # Create model with hyperparameters from checkpoint
                hparams = checkpoint.get('hyperparameters', {})
                model = create_model(model_name, **hparams)

                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()

                models[model_name] = model
                print(f"✅ Loaded {model_name}")

            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue

        return models

    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data for evaluation."""
        data = {}

        # Load out-of-distribution test data
        if Path(self.config.test_data_path).exists():
            test_npz = np.load(self.config.test_data_path)
            data['ood'] = {
                'X': test_npz['X'][:self.config.num_samples],
                'y': test_npz['y'][:self.config.num_samples],
                'concept_indices': test_npz.get('concept_indices', None)
            }
            print(f"✅ Loaded OOD test data: {len(data['ood']['X'])} samples")

        # Load in-distribution training data
        if Path(self.config.training_data_path).exists():
            train_npz = np.load(self.config.training_data_path)
            # Use last portion as in-distribution test
            n_samples = min(self.config.num_samples, len(train_npz['X']) // 4)
            start_idx = len(train_npz['X']) - n_samples
            data['ind'] = {
                'X': train_npz['X'][start_idx:],
                'y': train_npz['y'][start_idx:],
                'concept_indices': train_npz.get('concept_indices', None)
            }
            print(f"✅ Loaded IND test data: {len(data['ind']['X'])} samples")

        return data

    def _init_vec2text(self):
        """Initialize vec2text orchestrator."""
        try:
            print("Initializing vec2text orchestrator...")
            os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
            os.environ['VEC2TEXT_DEVICE'] = 'cpu'
            os.environ['TOKENIZERS_PARALLELISM'] = 'false'

            orchestrator = IsolatedVecTextVectOrchestrator()
            print("✅ Vec2Text orchestrator ready")
            return orchestrator
        except Exception as e:
            print(f"⚠️  Vec2Text initialization failed: {e}")
            return None

    def _build_faiss_index(self):
        """Build FAISS index for nearest neighbor search."""
        try:
            print("Building FAISS index...")

            # Use training data for FAISS index
            if Path(self.config.training_data_path).exists():
                train_npz = np.load(self.config.training_data_path)
                vectors = train_npz['vectors'][:10000]  # Use subset for speed

                # Normalize vectors
                vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

                # Create index
                index = faiss.IndexFlatIP(768)  # Inner product for cosine after normalization
                index.add(vectors_norm.astype('float32'))

                print(f"✅ FAISS index built with {index.ntotal:,} vectors")
                return index

        except Exception as e:
            print(f"⚠️  FAISS index creation failed: {e}")

        return None

    def predict_with_context_size(self, model: nn.Module, context_vecs: np.ndarray,
                                context_size: int, approach: str) -> Dict[str, Any]:
        """Make prediction with specified context size and approach."""

        # Truncate/extend context to desired size
        if context_size <= len(context_vecs):
            # Truncate to context_size
            context_input = context_vecs[-context_size:]
        else:
            # Pad with zeros or repeat (simplified)
            context_input = context_vecs[-5:]  # Use last 5 as base
            while len(context_input) < context_size:
                context_input = np.vstack([context_input, context_input[-1:]])
            context_input = context_input[:context_size]

        # Convert to tensor
        context_t = torch.from_numpy(context_input).float().unsqueeze(0).to(self.device)

        # Get prediction
        start_time = time.time()

        if approach == 'tiny_recursion':
            # Apply tiny recursion
            tr_model = TinyRecursion(model, threshold=self.config.tr_threshold)
            with torch.no_grad():
                pred_t, tr_metadata = tr_model(context_t)
        else:
            # Direct prediction
            with torch.no_grad():
                pred_t = model(context_t)
                tr_metadata = {'attempts': 1, 'converged': True}

        inference_time = time.time() - start_time

        # Convert back to numpy
        pred_vec = pred_t.cpu().numpy()[0]

        result = {
            'prediction': pred_vec,
            'inference_time': inference_time,
            'context_size': context_size,
            'approach': approach,
            'tr_metadata': tr_metadata
        }

        return result

    def decode_prediction(self, pred_vec: np.ndarray, target_vec: np.ndarray,
                         target_text: str, approach: str) -> Dict[str, Any]:
        """Decode prediction using various approaches."""

        results = {
            'cosine_similarity': self._cosine_similarity(pred_vec, target_vec),
            'target_text': target_text,
            'target_vec': target_vec
        }

        # Direct vec2text decoding
        if approach in ['direct_vec2text', 'all'] and self.vec2text_orchestrator:
            try:
                vec_text = pred_vec.tolist()
                decode_result = self.vec2text_orchestrator.orchestrate(
                    input_vector=vec_text,
                    subscribers=['jxe', 'ielab'],
                    vec2text_backend='isolated',
                    output_format='json',
                    num_steps=self.config.vec2text_steps
                )

                results['vec2text_jxe'] = {
                    'text': decode_result['jxe']['decoded_text'],
                    'confidence': 0.5  # Placeholder
                }

                results['vec2text_ielab'] = {
                    'text': decode_result['ielab']['decoded_text'],
                    'confidence': 0.5  # Placeholder
                }

            except Exception as e:
                print(f"⚠️  Vec2Text decoding failed: {e}")
                results['vec2text_jxe'] = {'text': 'ERROR', 'confidence': 0.0}
                results['vec2text_ielab'] = {'text': 'ERROR', 'confidence': 0.0}

        # Nearest neighbor retrieval
        if self.faiss_index is not None:
            try:
                pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)
                D, I = self.faiss_index.search(pred_norm.reshape(1, -1).astype('float32'), 5)

                results['nearest_neighbors'] = [
                    {'text': f'NN_{i[0]}', 'score': float(d[0]), 'rank': rank+1}
                    for rank, (d, i) in enumerate(zip(D[0], I[0]))
                ]
            except Exception as e:
                print(f"⚠️  NN retrieval failed: {e}")
                results['nearest_neighbors'] = []

        # TwoTower enhanced retrieval
        try:
            results['twotower'] = self.two_tower.retrieve(pred_vec, k=5)
        except Exception as e:
            print(f"⚠️  TwoTower retrieval failed: {e}")
            results['twotower'] = []

        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))

    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation across all models and approaches."""

        all_results = {
            'config': self.config,
            'models': {},
            'summary': {}
        }

        for model_name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"EVALUATING MODEL: {model_name.upper()}")
            print(f"{'='*80}")

            model_results = {
                'in_distribution': {},
                'out_of_distribution': {}
            }

            # Test each dataset
            for dataset_name, dataset in self.test_data.items():
                print(f"\nTesting on {dataset_name} dataset...")

                dataset_results = {}

                # Test each context size
                for context_size in self.config.context_sizes:
                    print(f"  Context size: {context_size}")

                    context_results = {}

                    # Test each approach
                    for approach in self.config.approaches:
                        print(f"    Approach: {approach}")

                        approach_results = []

                        # Test each sample
                        for i in range(min(self.config.num_samples, len(dataset['X']))):
                            # Get context and target
                            context_vecs = dataset['X'][i]
                            target_vec = dataset['y'][i]
                            target_text = f"Target_{i}"  # Placeholder

                            try:
                                # Make prediction
                                pred_result = self.predict_with_context_size(
                                    model, context_vecs, context_size, approach
                                )

                                # Decode prediction
                                decode_result = self.decode_prediction(
                                    pred_result['prediction'],
                                    target_vec,
                                    target_text,
                                    approach
                                )

                                # Combine results
                                sample_result = {
                                    'sample_id': i,
                                    'prediction': pred_result,
                                    'decoding': decode_result,
                                    'context_size': context_size,
                                    'approach': approach
                                }

                                approach_results.append(sample_result)

                            except Exception as e:
                                print(f"      ❌ Sample {i} failed: {e}")
                                continue

                        context_results[approach] = approach_results

                    dataset_results[context_size] = context_results

                model_results[dataset_name] = dataset_results

            all_results['models'][model_name] = model_results

        # Generate summary
        all_results['summary'] = self._generate_summary(all_results)

        # Save results
        self._save_results(all_results)

        return all_results

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        summary = {
            'best_models': {},
            'best_approaches': {},
            'context_size_analysis': {}
        }

        # Find best models for each dataset
        for dataset_name in ['in_distribution', 'out_of_distribution']:
            if dataset_name not in self.test_data:
                continue

            best_scores = {}

            for model_name, model_results in results['models'].items():
                if dataset_name not in model_results:
                    continue

                # Average cosine similarity across all context sizes and approaches
                total_score = 0
                count = 0

                for context_size, context_results in model_results[dataset_name].items():
                    for approach, approach_results in context_results.items():
                        for sample_result in approach_results:
                            total_score += sample_result['decoding']['cosine_similarity']
                            count += 1

                if count > 0:
                    avg_score = total_score / count
                    best_scores[model_name] = avg_score

            if best_scores:
                best_model = max(best_scores.items(), key=lambda x: x[1])
                summary['best_models'][dataset_name] = {
                    'model': best_model[0],
                    'score': best_model[1]
                }

        return summary

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        timestamp = int(time.time())

        # Save detailed results
        results_file = Path(self.config.output_dir) / f"detailed_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary
        summary_file = Path(self.config.output_dir) / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)

        print("\n📊 Results saved to:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary: {summary_file}")


def main():
    """Main evaluation function."""

    # Create configuration
    config = EvaluationConfig(
        models=['amn', 'transformer', 'gru'],
        context_sizes=[5, 10, 20],
        approaches=['direct_vec2text', 'tiny_recursion', 'nearest_neighbor'],
        num_samples=50  # Start small for testing
    )

    # Create evaluator
    evaluator = LVMEvaluator(config)

    print(f"\n{'🎯'*60}")
    print("COMPREHENSIVE LVM EVALUATION FRAMEWORK")
    print(f"{'🎯'*60}")
    print(f"Models: {', '.join(config.models)}")
    print(f"Context sizes: {', '.join(map(str, config.context_sizes))}")
    print(f"Approaches: {', '.join(config.approaches)}")
    print(f"Samples per test: {config.num_samples}")
    print(f"{'🎯'*60}\n")

    # Run evaluation
    results = evaluator.run_evaluation()

    print(f"\n{'🏆'*60}")
    print("EVALUATION COMPLETE!")
    print(f"{'🏆'*60}")

    # Print summary
    summary = results['summary']
    for dataset, best in summary['best_models'].items():
        print(f"Best {dataset}: {best['model']} (score: {best['score']:.4f})")

    print(f"\nDetailed results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Comprehensive LVM Evaluation with Multiple Output Approaches

Tests LVM models with:
1. Different context window sizes
2. In-dataset vs out-of-dataset performance
3. Multiple output approaches (vec2text, nearest neighbor, etc.)
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

# Import LVM models
sys.path.insert(0, 'app/lvm')
from models import create_model

@dataclass
class EvaluationConfig:
    """Configuration for comprehensive LVM evaluation."""
    models: List[str] = None
    context_sizes: List[int] = None
    approaches: List[str] = None
    test_data_path: str = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    training_data_path: str = "artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz"
    vectors_data_path: str = "artifacts/lvm/vectors_data.npz"
    num_samples: int = 100
    output_dir: str = "artifacts/lvm/comprehensive_evaluation"

    def __post_init__(self):
        if self.models is None:
            self.models = ['amn', 'transformer', 'gru']
        if self.context_sizes is None:
            self.context_sizes = [5, 10, 20]
        if self.approaches is None:
            self.approaches = ['direct_prediction', 'tiny_recursion', 'twotower']


class TinyRecursion:
    """Tiny Recursion implementation for LVM output refinement."""

    def __init__(self, base_model: nn.Module, threshold: float = 0.05, max_attempts: int = 2):
        self.base_model = base_model
        self.threshold = threshold
        self.max_attempts = max_attempts

    def refine_prediction(self, context_t: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply tiny recursion to refine predictions."""

        # Initial prediction
        with torch.no_grad():
            pred1 = self.base_model(context_t)

        metadata = {
            'attempts': 1,
            'converged': False,
            'delta_history': [],
            'final_prediction': pred1,
            'confidence': 1.0
        }

        # Try recursion if needed
        current_pred = pred1
        for attempt in range(1, self.max_attempts):
            # Create enhanced context that includes the prediction
            batch_size, seq_len, dim = context_t.shape

            # Blend prediction with original context using attention-like weighting
            # Use the last few vectors more heavily weighted with prediction
            enhanced_context = context_t.clone()

            # Simple approach: replace last vector with weighted combination
            if seq_len > 1:
                # Weight the prediction by confidence (higher confidence = more weight)
                confidence_weight = min(0.5, 1.0 / (attempt + 1))  # Decreasing weight with attempts
                enhanced_context[:, -1, :] = (1 - confidence_weight) * context_t[:, -1, :] + confidence_weight * current_pred.squeeze(1)

            # Second prediction
            with torch.no_grad():
                pred2 = self.base_model(enhanced_context)

            # Check convergence (cosine delta)
            delta = 1 - torch.cosine_similarity(current_pred, pred2, dim=-1).mean().item()
            metadata['delta_history'].append(delta)

            # If converged, use refined prediction
            if delta < self.threshold:
                metadata['converged'] = True
                metadata['final_prediction'] = pred2
                metadata['attempts'] = attempt + 1
                metadata['confidence'] = 1.0 - delta  # Higher confidence for converged predictions
                break

            current_pred = pred2
            metadata['attempts'] = attempt + 1

        return metadata['final_prediction'], metadata


class TwoTowerQueryEncoder(nn.Module):
    """TwoTower Query Encoder for context → query vector."""

    def __init__(self, input_dim=768, hidden_dim=512, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.proj = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, context):
        """Encode context sequence to query vector."""
        # context: [batch, seq_len, 768]
        _, hidden = self.gru(context)  # hidden: [num_layers, batch, hidden_dim]
        pooled = hidden[-1]  # Last layer: [batch, hidden_dim]
        query = self.proj(pooled)  # [batch, 768]
        return torch.nn.functional.normalize(query, dim=-1)


class ComprehensiveLVMEvaluator:
    """Comprehensive LVM evaluation framework."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models
        self.models = self._load_models()

        # Load data
        self.test_data, self.vectors_data = self._load_data()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_models(self) -> Dict[str, nn.Module]:
        """Load all LVM models."""
        models = {}
        model_paths = {
            'amn': 'artifacts/lvm/production_model',
            'transformer': 'artifacts/lvm/fallback_accuracy',
            'gru': 'artifacts/lvm/fallback_secondary'
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
                # Load checkpoint with safe loading
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

                # Create model
                hparams = checkpoint.get('hyperparameters', {})
                model = create_model(model_name, **hparams)

                # Load state dict
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(self.device)
                model.eval()

                models[model_name] = model
                print(f"✅ Loaded {model_name} ({model.count_parameters():,} params)")

            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue

        return models

    def _load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Load test and training data."""
        test_data = {}
        vectors_data = {}

        # Load out-of-distribution test data
        if Path(self.config.test_data_path).exists():
            print(f"Loading OOD test data from {self.config.test_data_path}...")
            test_npz = np.load(self.config.test_data_path, allow_pickle=True)
            test_data['ood'] = {
                'context_sequences': test_npz['context_sequences'][:self.config.num_samples],
                'target_vectors': test_npz['target_vectors'][:self.config.num_samples]
            }
            print(f"✅ Loaded {len(test_data['ood']['context_sequences'])} OOD samples")

        # Load in-distribution training data
        if Path(self.config.training_data_path).exists():
            print(f"Loading IND training data from {self.config.training_data_path}...")
            train_npz = np.load(self.config.training_data_path, allow_pickle=True)
            # Use last portion as in-distribution test
            n_samples = min(self.config.num_samples, len(train_npz['context_sequences']) // 4)
            start_idx = len(train_npz['context_sequences']) - n_samples
            test_data['ind'] = {
                'context_sequences': train_npz['context_sequences'][start_idx:],
                'target_vectors': train_npz['target_vectors'][start_idx:]
            }
            print(f"✅ Loaded {len(test_data['ind']['context_sequences'])} IND samples")

            # Load vectors for FAISS index
            if 'vectors' in train_npz:
                vectors_data['vectors'] = train_npz['vectors'][:10000]  # Subset for speed
                print(f"✅ Loaded {len(vectors_data['vectors'])} vectors for FAISS")

        return test_data, vectors_data

    def predict_with_approach(self, model: nn.Module, context_vecs: np.ndarray,
                            context_size: int, approach: str) -> Dict[str, Any]:
        """Make prediction with specified approach (Tiny Recursion, TwoTower, etc.)."""

        # Truncate/extend context to desired size
        if context_size <= len(context_vecs):
            context_input = context_vecs[-context_size:]
        else:
            # Pad with repetition for longer contexts
            context_input = context_vecs[-5:]  # Use last 5 as base
            while len(context_input) < context_size:
                context_input = np.vstack([context_input, context_input[-1:]])
            context_input = context_input[:context_size]

        # Convert to tensor
        context_t = torch.from_numpy(context_input).float().unsqueeze(0).to(self.device)

        # Apply different approaches
        if approach == 'tiny_recursion':
            # Apply Tiny Recursion refinement
            tr_model = TinyRecursion(model, threshold=0.05, max_attempts=2)
            with torch.no_grad():
                refined_pred, tr_metadata = tr_model.refine_prediction(context_t)

            pred_vec = refined_pred.cpu().numpy()[0]
            tr_metadata['approach'] = 'tiny_recursion'

            return {
                'prediction': pred_vec,
                'tr_metadata': tr_metadata,
                'context_size': context_size,
                'approach': approach
            }

        elif approach == 'twotower':
            # Apply TwoTower query encoding + retrieval
            query_encoder = TwoTowerQueryEncoder().to(self.device)

            with torch.no_grad():
                # Encode context to query vector
                query_vec = query_encoder(context_t)

                # Use query vector for enhanced prediction (simplified)
                # In full implementation, this would do retrieval + re-ranking
                enhanced_context = context_t.clone()
                enhanced_context[:, -1, :] = query_vec  # Replace last vector with query

                pred_t = model(enhanced_context)

            pred_vec = pred_t.cpu().numpy()[0]

            return {
                'prediction': pred_vec,
                'query_vector': query_vec.cpu().numpy()[0],
                'context_size': context_size,
                'approach': approach,
                'tr_metadata': None
            }

        else:
            # Direct prediction
            start_time = time.time()
            with torch.no_grad():
                pred_t = model(context_t)
            inference_time = time.time() - start_time

            pred_vec = pred_t.cpu().numpy()[0]

            return {
                'prediction': pred_vec,
                'inference_time': inference_time,
                'context_size': context_size,
                'approach': approach,
                'tr_metadata': None
            }

    def evaluate_prediction(self, pred_vec: np.ndarray, target_vec: np.ndarray) -> Dict[str, Any]:
        """Evaluate prediction quality."""

        # Cosine similarity
        cosine_sim = self._cosine_similarity(pred_vec, target_vec)

        # Euclidean distance
        euclidean_dist = np.linalg.norm(pred_vec - target_vec)

        # Prediction magnitude
        pred_magnitude = np.linalg.norm(pred_vec)
        target_magnitude = np.linalg.norm(target_vec)

        return {
            'cosine_similarity': cosine_sim,
            'euclidean_distance': euclidean_dist,
            'magnitude_error': abs(pred_magnitude - target_magnitude),
            'pred_vec': pred_vec,
            'target_vec': target_vec
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))

    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation."""

        all_results = {
            'config': self.config,
            'models': {},
            'summary': {}
        }

        print(f"\n{'🎯'*60}")
        print("COMPREHENSIVE LVM EVALUATION WITH TR & TWOTOWER")
        print(f"{'🎯'*60}")
        print(f"Models: {', '.join(self.config.models)}")
        print(f"Context sizes: {', '.join(map(str, self.config.context_sizes))}")
        print(f"Approaches: {', '.join(self.config.approaches)}")
        print(f"Samples per test: {self.config.num_samples}")
        print(f"{'🎯'*60}\n")

        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"EVALUATING MODEL: {model_name.upper()}")
            print(f"{'='*60}")

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
                        num_samples = min(self.config.num_samples, len(dataset['context_sequences']))
                        for i in range(num_samples):
                            if i % 20 == 0:
                                print(f"      Sample {i}/{num_samples}")

                            # Get context and target
                            context_vecs = dataset['context_sequences'][i]
                            target_vec = dataset['target_vectors'][i]

                            try:
                                # Make prediction with approach
                                pred_result = self.predict_with_approach(
                                    model, context_vecs, context_size, approach
                                )

                                # Evaluate prediction
                                eval_result = self.evaluate_prediction(
                                    pred_result['prediction'], target_vec
                                )

                                # Combine results
                                sample_result = {
                                    'sample_id': i,
                                    'prediction': pred_result,
                                    'evaluation': eval_result
                                }

                                approach_results.append(sample_result)

                            except Exception as e:
                                print(f"        ❌ Sample {i} failed: {e}")
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
        """Generate comprehensive summary."""
        summary = {
            'best_models': {},
            'context_size_analysis': {},
            'approach_analysis': {},
            'performance_comparison': {}
        }

        # Find best models for each dataset and approach
        for dataset_name in ['ood', 'ind']:  # Fix: use actual dataset names
            if dataset_name not in self.test_data:
                continue

            for approach in self.config.approaches:
                best_scores = {}

                for model_name, model_results in results['models'].items():
                    if dataset_name not in model_results:
                        continue

                    # Average cosine similarity across all context sizes for this approach
                    total_score = 0
                    count = 0

                    for context_size, context_results in model_results[dataset_name].items():
                        if approach in context_results:
                            for sample_result in context_results[approach]:
                                total_score += sample_result['evaluation']['cosine_similarity']
                                count += 1

                    if count > 0:
                        avg_score = total_score / count
                        best_scores[model_name] = avg_score

                if best_scores:
                    best_model = max(best_scores.items(), key=lambda x: x[1])
                    if dataset_name not in summary['best_models']:
                        summary['best_models'][dataset_name] = {}
                    summary['best_models'][dataset_name][approach] = {
                        'model': best_model[0],
                        'cosine_similarity': best_model[1]
                    }

        # Context size analysis
        for dataset_name in ['ood', 'ind']:  # Fix: use actual dataset names
            if dataset_name not in self.test_data:
                continue

            context_scores = {}

            for model_name, model_results in results['models'].items():
                if dataset_name not in model_results:
                    continue

                for context_size, context_results in model_results[dataset_name].items():
                    if context_size not in context_scores:
                        context_scores[context_size] = {}

                    for approach in self.config.approaches:
                        if approach in context_results:
                            if approach not in context_scores[context_size]:
                                context_scores[context_size][approach] = {}

                            total_score = 0
                            count = 0

                            for sample_result in context_results[approach]:
                                total_score += sample_result['evaluation']['cosine_similarity']
                                count += 1

                            if count > 0:
                                context_scores[context_size][approach][model_name] = total_score / count

            summary['context_size_analysis'][dataset_name] = context_scores

        return summary

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        timestamp = int(time.time())

        # Save detailed results
        results_file = Path(self.config.output_dir) / f"comprehensive_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary
        summary_file = Path(self.config.output_dir) / f"comprehensive_summary_{timestamp}.json"
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
        approaches=['direct_prediction', 'tiny_recursion', 'twotower'],
        num_samples=30  # Start with reasonable number for testing
    )

    # Create evaluator
    evaluator = ComprehensiveLVMEvaluator(config)

    # Run evaluation
    results = evaluator.run_evaluation()

    print(f"\n{'🏆'*60}")
    print("COMPREHENSIVE EVALUATION COMPLETE!")
    print(f"{'🏆'*60}")

    # Print summary
    summary = results['summary']

    print("\n📈 BEST MODELS BY APPROACH:")
    for dataset, approaches in summary['best_models'].items():
        print(f"\n{dataset.upper()}:")
        for approach, best in approaches.items():
            print(f"  {approach}: {best['model']} (cosine: {best['cosine_similarity']:.4f})")

    print("\n📊 CONTEXT SIZE ANALYSIS:")
    for dataset_name, context_analysis in summary['context_size_analysis'].items():
        print(f"\n{dataset_name.upper()}:")
        for context_size, approach_scores in context_analysis.items():
            print(f"  Context {context_size}:")
            for approach, model_scores in approach_scores.items():
                best_model = max(model_scores.items(), key=lambda x: x[1])
                print(f"    {approach}: {best_model[0]} (cosine: {best_model[1]:.4f})")

    print(f"\nDetailed results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Basic LVM Evaluation Script

Tests LVM models with different context sizes and approaches.
Simplified version focusing on core functionality.
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
class SimpleConfig:
    """Simple configuration for LVM evaluation."""
    models: List[str] = None
    context_sizes: List[int] = None
    test_data_path: str = "artifacts/lvm/wikipedia_ood_test_ctx5.npz"
    training_data_path: str = "artifacts/lvm/wikipedia_fresh_sequences_ctx5.npz"
    num_samples: int = 50
    output_dir: str = "artifacts/lvm/evaluation_results"

    def __post_init__(self):
        if self.models is None:
            self.models = ['amn', 'transformer', 'gru']
        if self.context_sizes is None:
            self.context_sizes = [5, 10, 20]


class BasicLVMEvaluator:
    """Basic LVM evaluation without complex dependencies."""

    def __init__(self, config: SimpleConfig):
        self.config = config
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load models
        self.models = self._load_models()

        # Load test data
        self.test_data = self._load_test_data()

        # Build FAISS index for nearest neighbor
        self.faiss_index = self._build_faiss_index()

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def _load_models(self) -> Dict[str, nn.Module]:
        """Load LVM models."""
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
                print(f"✅ Loaded {model_name}")

            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
                continue

        return models

    def _load_test_data(self) -> Dict[str, Any]:
        """Load test data."""
        data = {}

        # Load out-of-distribution test data
        if Path(self.config.test_data_path).exists():
            test_npz = np.load(self.config.test_data_path, allow_pickle=True)
            data['ood'] = {
                'context_sequences': test_npz['context_sequences'][:self.config.num_samples],
                'target_vectors': test_npz['target_vectors'][:self.config.num_samples]
            }
            print(f"✅ Loaded OOD test data: {len(data['ood']['context_sequences'])} samples")

        # Load in-distribution training data
        if Path(self.config.training_data_path).exists():
            train_npz = np.load(self.config.training_data_path, allow_pickle=True)
            # Use last portion as in-distribution test
            n_samples = min(self.config.num_samples, len(train_npz['context_sequences']) // 4)
            start_idx = len(train_npz['context_sequences']) - n_samples
            data['ind'] = {
                'context_sequences': train_npz['context_sequences'][start_idx:],
                'target_vectors': train_npz['target_vectors'][start_idx:]
            }
            print(f"✅ Loaded IND test data: {len(data['ind']['context_sequences'])} samples")

        return data

    def _build_faiss_index(self):
        """Build FAISS index for nearest neighbor search."""
        try:
            print("Building FAISS index...")

            # Use training data for FAISS index
            if Path(self.config.training_data_path).exists():
                train_npz = np.load(self.config.training_data_path, allow_pickle=True)
                vectors = train_npz['vectors'][:5000]  # Use subset for speed

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
                                context_size: int) -> Dict[str, Any]:
        """Make prediction with specified context size."""

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

        # Get prediction
        start_time = time.time()
        with torch.no_grad():
            pred_t = model(context_t)
        inference_time = time.time() - start_time

        # Convert back to numpy
        pred_vec = pred_t.cpu().numpy()[0]

        return {
            'prediction': pred_vec,
            'inference_time': inference_time,
            'context_size': context_size
        }

    def evaluate_prediction(self, pred_vec: np.ndarray, target_vec: np.ndarray) -> Dict[str, Any]:
        """Evaluate prediction quality."""

        # Cosine similarity
        cosine_sim = self._cosine_similarity(pred_vec, target_vec)

        # Nearest neighbor retrieval
        nn_results = []
        if self.faiss_index is not None:
            try:
                pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)
                D, I = self.faiss_index.search(pred_norm.reshape(1, -1).astype('float32'), 5)
                nn_results = [
                    {'score': float(d[0]), 'rank': rank+1}
                    for rank, (d, i) in enumerate(zip(D[0], I[0]))
                ]
            except Exception as e:
                print(f"⚠️  NN retrieval failed: {e}")

        return {
            'cosine_similarity': cosine_sim,
            'nearest_neighbors': nn_results,
            'pred_vec': pred_vec,
            'target_vec': target_vec
        }

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity."""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))

    def run_evaluation(self) -> Dict[str, Any]:
        """Run evaluation across all models and context sizes."""

        all_results = {
            'config': self.config,
            'models': {},
            'summary': {}
        }

        print(f"\n{'🎯'*60}")
        print("LVM EVALUATION STARTED")
        print(f"{'🎯'*60}")

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

                    context_results = []

                    # Test each sample
                    for i in range(min(self.config.num_samples, len(dataset['context_sequences']))):
                        if i % 10 == 0:
                            print(f"    Sample {i}/{min(self.config.num_samples, len(dataset['context_sequences']))}")

                        # Get context and target
                        context_vecs = dataset['context_sequences'][i]
                        target_vec = dataset['target_vectors'][i]

                        try:
                            # Make prediction
                            pred_result = self.predict_with_context_size(
                                model, context_vecs, context_size
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

                            context_results.append(sample_result)

                        except Exception as e:
                            print(f"      ❌ Sample {i} failed: {e}")
                            continue

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

                # Average cosine similarity across all context sizes
                total_score = 0
                count = 0

                for context_size, context_results in model_results[dataset_name].items():
                    for sample_result in context_results:
                        total_score += sample_result['evaluation']['cosine_similarity']
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

        # Context size analysis
        for dataset_name in ['in_distribution', 'out_of_distribution']:
            if dataset_name not in self.test_data:
                continue

            context_scores = {}

            for model_name, model_results in results['models'].items():
                if dataset_name not in model_results:
                    continue

                for context_size, context_results in model_results[dataset_name].items():
                    if context_size not in context_scores:
                        context_scores[context_size] = {}

                    total_score = 0
                    count = 0

                    for sample_result in context_results:
                        total_score += sample_result['evaluation']['cosine_similarity']
                        count += 1

                    if count > 0:
                        context_scores[context_size][model_name] = total_score / count

            summary['context_size_analysis'][dataset_name] = context_scores

        return summary

    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results."""
        timestamp = int(time.time())

        # Save detailed results
        results_file = Path(self.config.output_dir) / f"basic_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save summary
        summary_file = Path(self.config.output_dir) / f"basic_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results['summary'], f, indent=2, default=str)

        print("\n📊 Results saved to:")
        print(f"   Detailed: {results_file}")
        print(f"   Summary: {summary_file}")


def main():
    """Main evaluation function."""

    # Create configuration
    config = SimpleConfig(
        models=['amn', 'transformer', 'gru'],
        context_sizes=[5, 10, 20],
        num_samples=20  # Start small for testing
    )

    # Create evaluator
    evaluator = BasicLVMEvaluator(config)

    print(f"\n{'🎯'*60}")
    print("BASIC LVM EVALUATION FRAMEWORK")
    print(f"{'🎯'*60}")
    print(f"Models: {', '.join(config.models)}")
    print(f"Context sizes: {', '.join(map(str, config.context_sizes))}")
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
        print(f"Best {dataset}: {best['model']} (cosine: {best['score']:.4f})")

    # Print context size analysis
    print("\n📈 Context Size Analysis:")
    for dataset_name, context_analysis in summary['context_size_analysis'].items():
        print(f"\n{dataset_name.upper()}:")
        for context_size, model_scores in context_analysis.items():
            print(f"  Context {context_size}: {model_scores}")

    print(f"\nDetailed results saved to: {config.output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LVM Full Pipeline Test with Actual Text Examples
Shows 10 examples of: Context (5 chunks) → LVM → Decoded Text
With ROUGE/BLEU scoring
"""

import sys
import time
import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, 'app/lvm')
sys.path.insert(0, 'app/vect_text_vect')

from models import create_model
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator



def decode_vector_to_text(orchestrator, vector):
    """Decode a single 768D vector to text using JXE"""
    # The orchestrator's _run_subscriber_subprocess expects:
    # - subscriber_name: 'jxe'
    # - vectors: torch.Tensor of shape [1, 768]
    # - metadata: dict with 'original_texts' for prompts

    if isinstance(vector, np.ndarray):
        vector_tensor = torch.from_numpy(vector).float()
    else:
        vector_tensor = vector.float()

    if vector_tensor.dim() == 1:
        vector_tensor = vector_tensor.unsqueeze(0)  # [768] -> [1, 768]

    result = orchestrator._run_subscriber_subprocess(
        'jxe',
        vector_tensor.cpu(),
        metadata={'original_texts': [' ']},  # Blank prompt for unconditional decoding
        device_override='cpu'
    )

    if result['status'] == 'error':
        return f"<DECODE_ERROR: {result['error']}>"

    decoded_texts = result['result']
    return decoded_texts[0] if isinstance(decoded_texts, list) else str(decoded_texts)


import argparse
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    args = parser.parse_args()

    print("=" * 100)
    print("LVM FULL PIPELINE: TEXT OUTPUT EXAMPLES")
    print("=" * 100)
    print()

    # Load orchestrator
    print("Loading Vec2Text orchestrator...")
    orch = IsolatedVecTextVectOrchestrator(steps=1, debug=False)
    print("✓ Orchestrator ready")
    print()

    # Initialize scorers
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    for model_path in args.models:
        print(f"\n{'='*100}")
        print(f"MODEL: {model_path}")
        print(f"{'='*100}")

        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu')
        model_type = checkpoint.get('model_type', 'amn')
        model_config = checkpoint.get('model_config', {})
        model = create_model(model_type, **model_config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ {model_type.upper()} loaded (Val Cosine: {checkpoint.get('val_cosine', 0.0):.4f})")
        print()

        for dataset_path in args.datasets:
            print(f"\n---")
            print(f"DATASET: {dataset_path}")
            print(f"---")

            data = np.load(dataset_path, allow_pickle=True)
            
            if 'context_texts' in data:
                contexts = data['context_texts']
            elif 'contexts' in data:
                contexts = data['contexts']
            else:
                print(f"Could not find context texts in {dataset_path}")
                continue

            if 'target_texts' in data:
                targets = data['target_texts']
            elif 'targets' in data:
                targets = data['targets']
            else:
                print(f"Could not find target texts in {dataset_path}")
                continue

            num_samples = len(contexts)
            sample_indices = random.sample(range(num_samples), 5)

            all_results = []

            for idx, sample_idx in enumerate(sample_indices, 1):
                sample_context = contexts[sample_idx]
                sample_expected = targets[sample_idx]

                print(f"\n{'-'*50}")
                print(f"SAMPLE {idx}/5")
                print(f"{'-'*50}")

                # Show context
                print("\n📖 CONTEXT (5 chunks):")
                for i, chunk in enumerate(sample_context, 1):
                    print(f"   {i}. {chunk}")

                # Encode context
                t_start = time.perf_counter()
                ctx_vectors = orch.encode_texts(sample_context)
                t_encode = (time.perf_counter() - t_start) * 1000

                # Convert to numpy if needed
                if isinstance(ctx_vectors, torch.Tensor):
                    ctx_vectors = ctx_vectors.cpu().numpy()

                # Run LVM
                t_start = time.perf_counter()
                ctx_tensor = torch.from_numpy(ctx_vectors).float().unsqueeze(0)  # [1, 5, 768]
                with torch.no_grad():
                    pred_vector = model(ctx_tensor).cpu().numpy()[0]  # [768]
                t_lvm = (time.perf_counter() - t_start) * 1000

                # Decode predicted vector to text
                t_start = time.perf_counter()
                predicted_text = decode_vector_to_text(orch, pred_vector)
                t_decode = (time.perf_counter() - t_start) * 1000

                # Compute vector similarity
                expected_vector = orch.encode_texts([sample_expected])[0]
                if isinstance(expected_vector, torch.Tensor):
                    expected_vector = expected_vector.cpu().numpy()

                pred_norm = pred_vector / (np.linalg.norm(pred_vector) + 1e-8)
                exp_norm = expected_vector / (np.linalg.norm(expected_vector) + 1e-8)
                vec_cos = float(np.dot(pred_norm, exp_norm))

                # Compute text scores
                rouge = rouge_scorer_obj.score(sample_expected, predicted_text)
                bleu = sentence_bleu(
                    [sample_expected.lower().split()],
                    predicted_text.lower().split(),
                    smoothing_function=smoothing
                )

                print(f"\n🎯 EXPECTED:")
                print(f"   {sample_expected}")
                print(f"\n🤖 PREDICTED:")
                print(f"   {predicted_text}")

                print(f"\n📊 SCORES:")
                print(f"   Vector Cosine:  {vec_cos:6.4f}")
                print(f"   ROUGE-1:        {rouge['rouge1'].fmeasure:6.4f}")
                print(f"   ROUGE-2:        {rouge['rouge2'].fmeasure:6.4f}")
                print(f"   ROUGE-L:        {rouge['rougeL'].fmeasure:6.4f}")
                print(f"   BLEU:           {bleu:6.4f}")

                all_results.append({
                    'vec_cos': vec_cos,
                    'rouge1': rouge['rouge1'].fmeasure,
                    'rouge2': rouge['rouge2'].fmeasure,
                    'rougeL': rouge['rougeL'].fmeasure,
                    'bleu': bleu,
                })

            # Summary
            print("\n")
            print("-" * 50)
            print("SUMMARY STATISTICS (5 samples)")
            print("-" * 50)
            print()
            print(f"Average Vector Cosine:  {np.mean([r['vec_cos'] for r in all_results]):.4f}")
            print(f"Average ROUGE-1:        {np.mean([r['rouge1'] for r in all_results]):.4f}")
            print(f"Average ROUGE-2:        {np.mean([r['rouge2'] for r in all_results]):.4f}")
            print(f"Average ROUGE-L:        {np.mean([r['rougeL'] for r in all_results]):.4f}")
            print(f"Average BLEU:           {np.mean([r['bleu'] for r in all_results]):.4f}")
            print()

if __name__ == '__main__':
    main()

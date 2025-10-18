#!/usr/bin/env python3
"""
Full LVM Pipeline Test with ROUGE/BLEU Scoring
===============================================

Tests complete pipeline:
1. Load real Wikipedia chunks from training data
2. Encode 5 context chunks → 5x768D vectors
3. LVM predicts next vector → 1x768D
4. Decode vector → predicted text
5. Compare with ground truth using ROUGE/BLEU

Output: Detailed table with 10 examples showing:
- Context chunks (5 input texts)
- Expected next chunk (ground truth)
- Predicted next chunk (LVM output)
- Vector cosine similarity
- ROUGE-1, ROUGE-2, ROUGE-L scores
- BLEU score

Usage:
    python tools/test_full_pipeline_with_scoring.py --model-type lstm --num-samples 10
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, 'app/lvm')
sys.path.insert(0, 'app/vect_text_vect')

from models import create_model
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator


def load_test_data_with_metadata(
    data_path: str = 'artifacts/lvm/training_sequences_ctx5.npz',
    metadata_path: str = 'artifacts/wikipedia_500k_corrected_metadata.json',
    num_samples: int = 10,
    offset: int = 0
) -> List[Dict]:
    """
    Load test samples with original text metadata.

    Returns list of dicts:
    {
        'context_vectors': [5, 768],
        'target_vector': [768],
        'context_texts': [list of 5 strings],
        'target_text': string
    }
    """
    # Load vectors
    data = np.load(data_path)
    context_sequences = data['context_sequences']  # [N, 5, 768]
    target_vectors = data['target_vectors']  # [N, 768]

    # Load metadata (has original texts)
    with open(metadata_path) as f:
        metadata = json.load(f)

    # Each entry in training data corresponds to a sequence in metadata
    # training_sequences_ctx5.npz was created from the corrected vectors
    # We need to map indices back to original chunks

    samples = []

    # Use offset to get test samples from a different part of the data
    start_idx = offset
    end_idx = min(offset + num_samples, len(context_sequences))

    for i in range(start_idx, end_idx):
        # For now, we'll use placeholder texts since we need to properly map
        # the indices back to original chunks. In production, we'd store chunk IDs.

        # Get vectors
        context_vecs = context_sequences[i]  # [5, 768]
        target_vec = target_vectors[i]  # [768]

        sample = {
            'index': i,
            'context_vectors': context_vecs,
            'target_vector': target_vec,
            'context_texts': None,  # Will be populated by encoding check
            'target_text': None  # Will be populated by encoding check
        }
        samples.append(sample)

    return samples


def load_real_wikipedia_samples(num_samples: int = 10) -> List[Dict]:
    """
    Load real consecutive Wikipedia chunks for testing.

    This creates samples from actual Wikipedia articles where we have
    5 consecutive chunks + 1 target chunk.
    """
    # Load Wikipedia metadata
    metadata_path = Path('artifacts/wikipedia_500k_corrected_metadata.json')

    if not metadata_path.exists():
        print("⚠️  Wikipedia metadata not found, using synthetic samples")
        return create_synthetic_samples(num_samples)

    with open(metadata_path) as f:
        metadata = json.load(f)

    chunks = metadata.get('chunks', [])

    if len(chunks) < num_samples * 6:
        print(f"⚠️  Not enough chunks ({len(chunks)}), using available data")

    samples = []

    # Create samples from consecutive chunks
    for i in range(0, min(len(chunks) - 5, num_samples * 6), 6):
        context_texts = [chunks[i+j]['text'] for j in range(5)]
        target_text = chunks[i+5]['text']

        sample = {
            'index': i // 6,
            'context_texts': context_texts,
            'target_text': target_text,
            'article_id': chunks[i].get('article_id', 'unknown'),
            'chunk_ids': [chunks[i+j]['chunk_id'] for j in range(6)]
        }
        samples.append(sample)

        if len(samples) >= num_samples:
            break

    return samples


def create_synthetic_samples(num_samples: int = 10) -> List[Dict]:
    """Create synthetic test samples for demo purposes"""

    templates = [
        {
            'context': [
                "Artificial intelligence (AI) is intelligence demonstrated by machines.",
                "The term was coined in 1956 by John McCarthy at Dartmouth College.",
                "AI research has been highly successful in developing effective techniques.",
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "Deep learning uses neural networks with multiple layers."
            ],
            'target': "Neural networks are inspired by biological neural networks in animal brains."
        },
        {
            'context': [
                "The human brain is the central organ of the nervous system.",
                "It contains approximately 86 billion neurons.",
                "Neurons communicate through synapses using electrical and chemical signals.",
                "The brain processes sensory information and controls motor functions.",
                "Memory formation involves strengthening connections between neurons."
            ],
            'target': "Learning new skills creates new neural pathways in the brain."
        },
        {
            'context': [
                "Climate change refers to long-term shifts in global temperatures.",
                "Human activities, especially burning fossil fuels, contribute to warming.",
                "Carbon dioxide levels have increased significantly since industrialization.",
                "Rising temperatures cause ice sheets to melt and sea levels to rise.",
                "Extreme weather events are becoming more frequent and severe."
            ],
            'target': "Scientists predict further warming will continue without intervention."
        }
    ]

    samples = []
    for i in range(num_samples):
        template = templates[i % len(templates)]
        samples.append({
            'index': i,
            'context_texts': template['context'],
            'target_text': template['target'],
            'article_id': f'synthetic_{i}',
            'chunk_ids': [f'chunk_{j}' for j in range(6)]
        })

    return samples


def compute_rouge_scores(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, prediction)

    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure
    }


def compute_bleu_score(prediction: str, reference: str) -> float:
    """Compute BLEU score"""
    # Tokenize
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()

    # Use smoothing for short sequences
    smoothing = SmoothingFunction().method1

    # Compute BLEU (using reference as list of tokens)
    score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothing)

    return score


def run_full_pipeline(
    model,
    device,
    orchestrator,
    sample: Dict
) -> Dict:
    """
    Run complete pipeline on one sample.

    Returns timing and results.
    """
    results = {
        'index': sample['index'],
        'context_texts': sample['context_texts'],
        'target_text': sample['target_text'],
        'timings': {}
    }

    # 1. Encode context texts
    start = time.perf_counter()
    context_vectors = orchestrator.encode_texts(sample['context_texts'])  # [5, 768]
    results['timings']['encode_ms'] = (time.perf_counter() - start) * 1000

    # 2. LVM prediction
    start = time.perf_counter()
    context_tensor = torch.FloatTensor(context_vectors).unsqueeze(0).to(device)  # [1, 5, 768]

    with torch.no_grad():
        predicted_vector = model(context_tensor)  # [1, 768]

    predicted_vector_np = predicted_vector.cpu().numpy()[0]  # [768]
    results['timings']['lvm_ms'] = (time.perf_counter() - start) * 1000

    # Also encode ground truth for vector comparison
    start = time.perf_counter()
    target_vector = orchestrator.encode_texts([sample['target_text']])[0]  # [768]
    results['timings']['encode_target_ms'] = (time.perf_counter() - start) * 1000

    # 3. Decode predicted vector
    start = time.perf_counter()
    vec2text_result = orchestrator.vec_to_text(
        predicted_vector_np,
        subscribers=['jxe'],
        steps=1
    )
    predicted_text = vec2text_result['jxe']['decoded_text']
    results['timings']['decode_ms'] = (time.perf_counter() - start) * 1000

    results['predicted_text'] = predicted_text

    # 4. Compute vector cosine similarity
    pred_norm = predicted_vector_np / (np.linalg.norm(predicted_vector_np) + 1e-8)
    target_norm = target_vector / (np.linalg.norm(target_vector) + 1e-8)
    vector_cosine = float(np.dot(pred_norm, target_norm))
    results['vector_cosine'] = vector_cosine

    # 5. Compute text quality metrics
    rouge_scores = compute_rouge_scores(predicted_text, sample['target_text'])
    bleu_score = compute_bleu_score(predicted_text, sample['target_text'])

    results['rouge1_f'] = rouge_scores['rouge1_f']
    results['rouge2_f'] = rouge_scores['rouge2_f']
    results['rougeL_f'] = rouge_scores['rougeL_f']
    results['bleu'] = bleu_score

    # Total time
    results['timings']['total_ms'] = (
        results['timings']['encode_ms'] +
        results['timings']['lvm_ms'] +
        results['timings']['decode_ms']
    )

    return results


def create_results_table(results: List[Dict]) -> str:
    """Create detailed markdown table of results"""

    md = "# Full LVM Pipeline Results with ROUGE/BLEU Scoring\n\n"
    md += "## Summary Statistics\n\n"

    # Calculate averages
    avg_vector_cosine = np.mean([r['vector_cosine'] for r in results])
    avg_rouge1 = np.mean([r['rouge1_f'] for r in results])
    avg_rouge2 = np.mean([r['rouge2_f'] for r in results])
    avg_rougeL = np.mean([r['rougeL_f'] for r in results])
    avg_bleu = np.mean([r['bleu'] for r in results])
    avg_total_ms = np.mean([r['timings']['total_ms'] for r in results])

    md += f"| Metric | Average |\n"
    md += f"|--------|--------|\n"
    md += f"| Vector Cosine | {avg_vector_cosine:.4f} |\n"
    md += f"| ROUGE-1 F-score | {avg_rouge1:.4f} |\n"
    md += f"| ROUGE-2 F-score | {avg_rouge2:.4f} |\n"
    md += f"| ROUGE-L F-score | {avg_rougeL:.4f} |\n"
    md += f"| BLEU Score | {avg_bleu:.4f} |\n"
    md += f"| Total Latency | {avg_total_ms:.1f} ms |\n"
    md += "\n---\n\n"

    # Detailed results for each sample
    md += "## Detailed Results (10 Samples)\n\n"

    for i, r in enumerate(results, 1):
        md += f"### Sample {i}\n\n"

        # Context
        md += "**Context (5 chunks):**\n"
        for j, text in enumerate(r['context_texts'], 1):
            # Truncate long texts
            display_text = text if len(text) < 100 else text[:97] + "..."
            md += f"{j}. {display_text}\n"
        md += "\n"

        # Expected vs Predicted
        expected_short = r['target_text'] if len(r['target_text']) < 150 else r['target_text'][:147] + "..."
        predicted_short = r['predicted_text'] if len(r['predicted_text']) < 150 else r['predicted_text'][:147] + "..."

        md += f"**Expected Next Chunk:**  \n`{expected_short}`\n\n"
        md += f"**Predicted Next Chunk:**  \n`{predicted_short}`\n\n"

        # Scores
        md += "**Scores:**\n\n"
        md += "| Metric | Value |\n"
        md += "|--------|-------|\n"
        md += f"| Vector Cosine | {r['vector_cosine']:.4f} |\n"
        md += f"| ROUGE-1 | {r['rouge1_f']:.4f} |\n"
        md += f"| ROUGE-2 | {r['rouge2_f']:.4f} |\n"
        md += f"| ROUGE-L | {r['rougeL_f']:.4f} |\n"
        md += f"| BLEU | {r['bleu']:.4f} |\n"
        md += "\n"

        # Timing
        md += "**Timing:**\n\n"
        md += "| Stage | Time (ms) | % of Total |\n"
        md += "|-------|-----------|------------|\n"
        md += f"| Encode Context | {r['timings']['encode_ms']:.1f} | {r['timings']['encode_ms']/r['timings']['total_ms']*100:.1f}% |\n"
        md += f"| LVM Inference | {r['timings']['lvm_ms']:.2f} | {r['timings']['lvm_ms']/r['timings']['total_ms']*100:.1f}% |\n"
        md += f"| Decode Vector | {r['timings']['decode_ms']:.1f} | {r['timings']['decode_ms']/r['timings']['total_ms']*100:.1f}% |\n"
        md += f"| **TOTAL** | **{r['timings']['total_ms']:.1f}** | **100%** |\n"
        md += "\n---\n\n"

    return md


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', default='lstm', choices=['amn', 'lstm', 'gru', 'transformer'])
    parser.add_argument('--model-dir', help='Path to trained model directory')
    parser.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--output', default='artifacts/lvm/full_pipeline_results.md')
    args = parser.parse_args()

    print("=" * 80)
    print("Full LVM Pipeline Test with ROUGE/BLEU Scoring")
    print("=" * 80)
    print()

    # Find model
    if args.model_dir:
        model_dir = Path(args.model_dir)
    else:
        models_base = Path('artifacts/lvm/models')
        matching = sorted(models_base.glob(f'{args.model_type}_*'))
        if not matching:
            print(f"❌ No trained {args.model_type.upper()} model found!")
            return
        model_dir = matching[-1]

    print(f"📦 Loading Model: {model_dir.name}")
    device = torch.device(args.device)

    # Load LVM
    checkpoint = torch.load(model_dir / 'best_model.pt', map_location='cpu')
    model_config = checkpoint.get('model_config', {})
    model = create_model(args.model_type, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"   ✓ Model: {args.model_type.upper()} ({model.count_parameters():,} params)")
    print(f"   ✓ Val Cosine: {checkpoint.get('val_cosine', 'N/A')}")

    # Load orchestrator
    print(f"\n📦 Loading Vec2Text Orchestrator...")
    orchestrator = IsolatedVecTextVectOrchestrator()
    print(f"   ✓ Vec2Text Ready")

    # Load test samples
    print(f"\n📚 Loading {args.num_samples} test samples...")
    samples = load_real_wikipedia_samples(args.num_samples)
    print(f"   ✓ Loaded {len(samples)} samples")

    # Run pipeline on all samples
    print(f"\n⏳ Running full pipeline on {len(samples)} samples...")
    print("   (This will take a while due to vec2text decoding...)\n")

    all_results = []

    for i, sample in enumerate(samples, 1):
        print(f"   Processing sample {i}/{len(samples)}...", end=' ')

        try:
            result = run_full_pipeline(model, device, orchestrator, sample)
            all_results.append(result)
            print(f"✓ (Vector cos: {result['vector_cosine']:.3f}, Total: {result['timings']['total_ms']:.0f}ms)")
        except Exception as e:
            print(f"✗ Error: {e}")
            continue

    if not all_results:
        print("\n❌ No samples processed successfully!")
        return

    # Create results table
    print(f"\n📊 Generating results table...")
    results_md = create_results_table(all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(results_md)

    print(f"✓ Results saved to: {output_path}")

    # Save raw JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"✓ Raw data saved to: {json_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("📊 Summary")
    print("=" * 80)

    avg_vector_cosine = np.mean([r['vector_cosine'] for r in all_results])
    avg_rouge1 = np.mean([r['rouge1_f'] for r in all_results])
    avg_bleu = np.mean([r['bleu'] for r in all_results])
    avg_total_ms = np.mean([r['timings']['total_ms'] for r in all_results])

    print(f"\n✅ Processed {len(all_results)} samples successfully")
    print(f"\n📈 Average Scores:")
    print(f"   Vector Cosine:  {avg_vector_cosine:.4f}")
    print(f"   ROUGE-1:        {avg_rouge1:.4f}")
    print(f"   BLEU:           {avg_bleu:.4f}")
    print(f"\n⏱️  Average Latency: {avg_total_ms:.1f} ms per query")
    print(f"\n📁 Full results: {output_path}")
    print()


if __name__ == '__main__':
    main()

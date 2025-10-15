#!/usr/bin/env python3
"""
Text Reconstruction Quality Evaluation
======================================

Evaluates LVM predictions using TEXT quality metrics:
1. Semantic Similarity: Sentence-BERT cosine, BERTScore
2. Token Overlap: BLEU, ROUGE-L
"""

import sys
sys.path.insert(0, 'app/lvm')

import torch
import numpy as np
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# Import model architectures
from train_lstm_baseline import LSTMVectorPredictor
from train_mamba2 import Mamba2VectorPredictor
from train_transformer import TransformerVectorPredictor


def decode_vector_to_text(vector, backend='jxe', steps=1):
    """Decode a vector using vec2text FastAPI server."""
    try:
        response = requests.post(
            'http://localhost:8766/decode',
            json={
                'vectors': [vector.tolist()],
                'subscribers': backend,
                'steps': steps,
                'device': 'cpu'
            },
            timeout=60
        )

        if response.status_code != 200:
            return None

        data = response.json()
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            if 'subscribers' in result:
                key = f"gtr → {backend}"
                if key in result['subscribers']:
                    return result['subscribers'][key].get('output', None)
        return None

    except Exception as e:
        print(f"Decode error: {e}")
        return None


def compute_text_metrics(reference, candidate, sbert_model):
    """Compute all text quality metrics."""

    # 1. Semantic Similarity (Sentence-BERT)
    ref_emb = sbert_model.encode([reference], convert_to_numpy=True)
    cand_emb = sbert_model.encode([candidate], convert_to_numpy=True)
    sbert_cosine = cos_sim(ref_emb, cand_emb)[0][0]

    # 2. BLEU Score (token overlap)
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    smoothing = SmoothingFunction().method1
    bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)

    # 3. ROUGE-L (longest common subsequence)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    rouge_l = rouge_scores['rougeL'].fmeasure

    # 4. BERTScore (contextual embedding similarity)
    P, R, F1 = bert_score([candidate], [reference], lang='en', verbose=False)
    bert_f1 = F1.item()

    return {
        'sbert_cosine': sbert_cosine,
        'bleu': bleu_score,
        'rouge_l': rouge_l,
        'bert_f1': bert_f1
    }


def evaluate_model_text_quality(model_name, model_path, model_class, model_kwargs,
                                 n_samples=5, device='cpu'):
    """Evaluate text reconstruction quality for a single model."""

    print(f"\n{'='*80}")
    print(f"Evaluating {model_name}: Text Reconstruction Quality")
    print(f"{'='*80}\n")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load data
    train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
    context_sequences = train_data['context_sequences']
    target_vectors = train_data['target_vectors']
    target_texts = train_data['target_texts']

    # Use validation split
    split_idx = int(0.9 * len(context_sequences))
    val_contexts = context_sequences[split_idx:split_idx + n_samples]
    val_targets = target_vectors[split_idx:split_idx + n_samples]
    val_texts = target_texts[split_idx:split_idx + n_samples]

    # Load Sentence-BERT model for semantic similarity
    print("Loading Sentence-BERT model...")
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print(f"\nEvaluating {n_samples} samples...\n")

    results = []

    for i in range(n_samples):
        context = torch.FloatTensor(val_contexts[i:i+1]).to(device)
        ground_truth_text = str(val_texts[i])
        ground_truth_vec = val_targets[i]

        # LVM prediction
        with torch.no_grad():
            pred_raw, pred_normed = model(context, return_raw=True)
        predicted_raw = pred_raw.cpu().numpy()[0]
        predicted_norm = pred_normed.cpu().numpy()[0]

        # Decode to text
        predicted_text = decode_vector_to_text(predicted_raw, backend='jxe', steps=1)

        if predicted_text is None:
            print(f"[{i+1}] FAILED to decode")
            continue

        # Compute metrics
        metrics = compute_text_metrics(ground_truth_text, predicted_text, sbert_model)
        metrics['pred_raw_norm'] = float(np.linalg.norm(predicted_raw))
        metrics['pred_norm'] = float(np.linalg.norm(predicted_norm))

        # Baseline: decode ground-truth embedding directly
        baseline_text = decode_vector_to_text(ground_truth_vec, backend='jxe', steps=1)
        baseline_metrics = None
        if baseline_text is not None:
            baseline_metrics = compute_text_metrics(ground_truth_text, baseline_text, sbert_model)
            metrics.update({
                'baseline_sbert_cosine': baseline_metrics['sbert_cosine'],
                'baseline_bert_f1': baseline_metrics['bert_f1'],
                'baseline_bleu': baseline_metrics['bleu'],
                'baseline_rouge_l': baseline_metrics['rouge_l'],
            })

        # Store result
        sample_record = {
            'sample_idx': i,
            'ground_truth': ground_truth_text,
            'prediction': predicted_text,
            **metrics
        }
        if baseline_text is not None:
            sample_record['baseline_prediction'] = baseline_text
        results.append(sample_record)

        # Print example
        print(f"{'─'*80}")
        print(f"Sample {i+1}/{n_samples}")
        print(f"{'─'*80}")
        print(f"Ground Truth:\n  {ground_truth_text}\n")
        print(f"Prediction:\n  {predicted_text}\n")
        print(f"Metrics:")
        print(f"  • Semantic Similarity (SBERT):  {metrics['sbert_cosine']:.4f}")
        print(f"  • BERTScore F1:                 {metrics['bert_f1']:.4f}")
        print(f"  • BLEU Score:                   {metrics['bleu']:.4f}")
        print(f"  • ROUGE-L F1:                   {metrics['rouge_l']:.4f}")
        print(f"  • Predicted raw norm:           {metrics['pred_raw_norm']:.3f}")
        print(f"  • Predicted L2 norm:            {metrics['pred_norm']:.3f}")
        if baseline_text is not None:
            print(f"  • Baseline text:                {baseline_text[:80]}...")
            print(f"    ∘ Baseline SBERT cosine:     {baseline_metrics['sbert_cosine']:.4f}")
            print(f"    ∘ Baseline BERTScore F1:     {baseline_metrics['bert_f1']:.4f}")
            print(f"    ∘ Baseline BLEU:             {baseline_metrics['bleu']:.4f}")
            print(f"    ∘ Baseline ROUGE-L:          {baseline_metrics['rouge_l']:.4f}")
        print()

    # Aggregate statistics
    if results:
        avg_metrics = {
            'sbert_cosine': np.mean([r['sbert_cosine'] for r in results]),
            'bert_f1': np.mean([r['bert_f1'] for r in results]),
            'bleu': np.mean([r['bleu'] for r in results]),
            'rouge_l': np.mean([r['rouge_l'] for r in results]),
            'pred_raw_norm': np.mean([r['pred_raw_norm'] for r in results]),
            'pred_norm': np.mean([r['pred_norm'] for r in results]),
        }

        baseline_present = [r for r in results if 'baseline_sbert_cosine' in r]
        if baseline_present:
            avg_metrics.update({
                'baseline_sbert_cosine': np.mean([r['baseline_sbert_cosine'] for r in baseline_present]),
                'baseline_bert_f1': np.mean([r['baseline_bert_f1'] for r in baseline_present]),
                'baseline_bleu': np.mean([r['baseline_bleu'] for r in baseline_present]),
                'baseline_rouge_l': np.mean([r['baseline_rouge_l'] for r in baseline_present]),
            })

        print(f"{'='*80}")
        print(f"AVERAGE METRICS ({len(results)} samples)")
        print(f"{'='*80}")
        print(f"  • Semantic Similarity (SBERT):  {avg_metrics['sbert_cosine']:.4f}")
        print(f"  • BERTScore F1:                 {avg_metrics['bert_f1']:.4f}")
        print(f"  • BLEU Score:                   {avg_metrics['bleu']:.4f}")
        print(f"  • ROUGE-L F1:                   {avg_metrics['rouge_l']:.4f}")
        print(f"  • Avg predicted raw norm:       {avg_metrics['pred_raw_norm']:.3f}")
        print(f"  • Avg predicted L2 norm:        {avg_metrics['pred_norm']:.3f}")
        if baseline_present:
            print(f"  • Baseline SBERT cosine:        {avg_metrics['baseline_sbert_cosine']:.4f}")
            print(f"  • Baseline BERTScore F1:        {avg_metrics['baseline_bert_f1']:.4f}")
            print(f"  • Baseline BLEU:                {avg_metrics['baseline_bleu']:.4f}")
            print(f"  • Baseline ROUGE-L F1:          {avg_metrics['baseline_rouge_l']:.4f}")
        print()

        return {
            'model': model_name,
            'samples': results,
            'averages': avg_metrics
        }

    return {'model': model_name, 'samples': [], 'averages': {}}


def main():
    print("\n" + "█"*80)
    print("█  TEXT RECONSTRUCTION QUALITY EVALUATION".center(80, " ") + "█")
    print("█"*80 + "\n")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Vec2text: CPU (required)\n")

    models = [
        {
            'name': 'LSTM',
            'path': 'artifacts/lvm/models/lstm_vec2text/best_model.pt',
            'class': LSTMVectorPredictor,
            'kwargs': {'input_dim': 768, 'hidden_dim': 512, 'num_layers': 2},
        },
        {
            'name': 'GRU',
            'path': 'artifacts/lvm/models/gru_vec2text/best_model.pt',
            'class': Mamba2VectorPredictor,
            'kwargs': {'input_dim': 768, 'd_model': 512, 'num_layers': 4},
        },
        {
            'name': 'Transformer',
            'path': 'artifacts/lvm/models/transformer_vec2text/best_model.pt',
            'class': TransformerVectorPredictor,
            'kwargs': {'input_dim': 768, 'd_model': 512, 'nhead': 8, 'num_layers': 4},
        },
    ]

    all_results = []
    for cfg in models:
        result = evaluate_model_text_quality(
            cfg['name'],
            cfg['path'],
            cfg['class'],
            cfg['kwargs'],
            n_samples=5,
            device=device,
        )
        all_results.append(result)

    # Save results
    import json
    output_path = Path('artifacts/lvm/evaluation/text_reconstruction_quality.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for res in all_results:
        samples_serialized = []
        for s in res['samples']:
            entry = {
                'sample_idx': s['sample_idx'],
                'ground_truth': s['ground_truth'],
                'prediction': s['prediction'],
                'sbert_cosine': float(s['sbert_cosine']),
                'bert_f1': float(s['bert_f1']),
                'bleu': float(s['bleu']),
                'rouge_l': float(s['rouge_l']),
                'pred_raw_norm': float(s['pred_raw_norm']),
                'pred_norm': float(s['pred_norm']),
            }
            if 'baseline_prediction' in s:
                entry['baseline_prediction'] = s['baseline_prediction']
                entry['baseline_sbert_cosine'] = float(s['baseline_sbert_cosine'])
                entry['baseline_bert_f1'] = float(s['baseline_bert_f1'])
                entry['baseline_bleu'] = float(s['baseline_bleu'])
                entry['baseline_rouge_l'] = float(s['baseline_rouge_l'])
            samples_serialized.append(entry)

        averages_serialized = {k: float(v) for k, v in res['averages'].items()}

        serializable.append({
            'model': res['model'],
            'samples': samples_serialized,
            'averages': averages_serialized,
        })

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"✓ Results saved to: {output_path}\n")


if __name__ == '__main__':
    main()

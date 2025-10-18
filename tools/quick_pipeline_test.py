#!/usr/bin/env python3
"""Quick full pipeline test with results table"""

import sys, time
import numpy as np
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, 'app/lvm')
sys.path.insert(0, 'app/vect_text_vect')

from models import create_model
from vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

print("Loading AMN model...")
checkpoint = torch.load('artifacts/lvm/models/amn_20251016_133427/best_model.pt', map_location='cpu')
model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ Model loaded (Val Cosine: {checkpoint['val_cosine']:.4f})\n")

print("Loading Vec2Text...")
orch = IsolatedVecTextVectOrchestrator()
print("✓ Ready\n")

samples = [
    {
        'context': [
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "The term was coined in 1956 by John McCarthy.",
            "AI research has been highly successful.",
            "Machine learning enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers."
        ],
        'target': "Neural networks are inspired by biological neural networks."
    },
    {
        'context': [
            "The human brain is the central organ of the nervous system.",
            "It contains approximately 86 billion neurons.",
            "Neurons communicate through synapses using chemical signals.",
            "The brain processes sensory information and controls functions.",
            "Memory formation involves strengthening neural connections."
        ],
        'target': "Learning new skills creates new neural pathways in the brain."
    }
]

print("=" * 80)
print("Full Pipeline Test (Text → Vec → LVM → Vec → Text)")
print("=" * 80)
print()

results = []

for idx, sample in enumerate(samples, 1):
    print(f"Sample {idx}:")
    print("-" * 80)

    # Show context
    print("Context (5 chunks):")
    for i, txt in enumerate(sample['context'], 1):
        print(f"  {i}. {txt}")
    print()

    # 1. Encode
    t0 = time.perf_counter()
    ctx_vecs = orch.encode_texts(sample['context'])
    t_encode = (time.perf_counter() - t0) * 1000

    # 2. LVM
    t0 = time.perf_counter()
    # Ensure ctx_vecs is numpy array
    if torch.is_tensor(ctx_vecs):
        ctx_vecs = ctx_vecs.cpu().numpy()
    ctx_tensor = torch.from_numpy(ctx_vecs).float().unsqueeze(0)
    with torch.no_grad():
        pred_vec = model(ctx_tensor).cpu().numpy()[0]
    t_lvm = (time.perf_counter() - t0) * 1000

    # 3. Decode
    t0 = time.perf_counter()
    result = orch.vec_to_text(pred_vec, subscribers=['jxe'], steps=1)
    pred_text = result['jxe']['decoded_text']
    t_decode = (time.perf_counter() - t0) * 1000

    # 4. Scores
    target_vec = orch.encode_texts([sample['target']])[0]
    pred_norm = pred_vec / (np.linalg.norm(pred_vec) + 1e-8)
    target_norm = target_vec / (np.linalg.norm(target_vec) + 1e-8)
    vec_cos = float(np.dot(pred_norm, target_norm))

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge = scorer.score(sample['target'], pred_text)

    bleu = sentence_bleu(
        [sample['target'].lower().split()],
        pred_text.lower().split(),
        smoothing_function=SmoothingFunction().method1
    )

    print(f"Expected:  {sample['target']}")
    print(f"Predicted: {pred_text}")
    print()
    print("Scores:")
    print(f"  Vector Cosine: {vec_cos:.4f}")
    print(f"  ROUGE-1:       {rouge['rouge1'].fmeasure:.4f}")
    print(f"  ROUGE-2:       {rouge['rouge2'].fmeasure:.4f}")
    print(f"  ROUGE-L:       {rouge['rougeL'].fmeasure:.4f}")
    print(f"  BLEU:          {bleu:.4f}")
    print()
    total = t_encode + t_lvm + t_decode
    print("Latency Breakdown:")
    print(f"  Encoding:   {t_encode:>8.1f} ms  ({t_encode/total*100:.1f}%)")
    print(f"  LVM:        {t_lvm:>8.2f} ms  ({t_lvm/total*100:.1f}%)")
    print(f"  Decoding:   {t_decode:>8.1f} ms  ({t_decode/total*100:.1f}%)")
    print(f"  ─────────────────────────")
    print(f"  TOTAL:      {total:>8.1f} ms")
    print()

    results.append({
        'vec_cos': vec_cos,
        'rouge1': rouge['rouge1'].fmeasure,
        'rouge2': rouge['rouge2'].fmeasure,
        'rougeL': rouge['rougeL'].fmeasure,
        'bleu': bleu,
        'total_ms': total,
        'encode_ms': t_encode,
        'lvm_ms': t_lvm,
        'decode_ms': t_decode
    })

print("=" * 80)
print("Summary Statistics")
print("=" * 80)
print()
print(f"Average Vector Cosine: {np.mean([r['vec_cos'] for r in results]):.4f}")
print(f"Average ROUGE-1:       {np.mean([r['rouge1'] for r in results]):.4f}")
print(f"Average ROUGE-2:       {np.mean([r['rouge2'] for r in results]):.4f}")
print(f"Average ROUGE-L:       {np.mean([r['rougeL'] for r in results]):.4f}")
print(f"Average BLEU:          {np.mean([r['bleu'] for r in results]):.4f}")
print()
print(f"Average Total Latency: {np.mean([r['total_ms'] for r in results]):.1f} ms")
print(f"  Encoding:            {np.mean([r['encode_ms'] for r in results]):.1f} ms")
print(f"  LVM:                 {np.mean([r['lvm_ms'] for r in results]):.2f} ms")
print(f"  Decoding:            {np.mean([r['decode_ms'] for r in results]):.1f} ms")
print()
print("=" * 80)
print("✓ Full pipeline works end-to-end!")
print("=" * 80)

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

# Sample data: Wikipedia-style sequential chunks
SAMPLES = [
    {
        'context': [
            "Artificial intelligence (AI) is intelligence demonstrated by machines.",
            "The term artificial intelligence was coined in 1956 by John McCarthy.",
            "AI research has been highly successful in developing effective techniques.",
            "Machine learning algorithms build a model based on sample data.",
            "Deep learning is part of a broader family of machine learning methods."
        ],
        'expected': "Neural networks are computing systems inspired by biological neural networks."
    },
    {
        'context': [
            "The human brain contains approximately 86 billion neurons.",
            "Neurons communicate through synapses using chemical signals.",
            "Synaptic connections can be strengthened or weakened over time.",
            "This process is called synaptic plasticity.",
            "Synaptic plasticity is essential for learning and memory formation."
        ],
        'expected': "Long-term potentiation strengthens synaptic connections permanently."
    },
    {
        'context': [
            "Python is a high-level programming language.",
            "It was created by Guido van Rossum in 1991.",
            "Python emphasizes code readability with significant whitespace.",
            "The language provides constructs for clear programming.",
            "Python supports multiple programming paradigms."
        ],
        'expected': "Object-oriented programming is one of Python's core paradigms."
    },
    {
        'context': [
            "Climate change refers to long-term shifts in temperatures and weather patterns.",
            "These shifts may be natural or human-caused.",
            "Since the 1800s, human activities have been the main driver.",
            "Burning fossil fuels generates greenhouse gas emissions.",
            "These emissions act like a blanket wrapped around Earth."
        ],
        'expected': "The trapped heat causes global temperatures to rise steadily."
    },
    {
        'context': [
            "The Internet is a global system of interconnected computer networks.",
            "It uses the Internet protocol suite to link devices worldwide.",
            "The Internet carries an extensive range of information resources.",
            "These include websites, email, and file sharing.",
            "The World Wide Web is accessed via the Internet."
        ],
        'expected': "Web browsers allow users to navigate between web pages easily."
    },
    {
        'context': [
            "DNA carries genetic information in living organisms.",
            "It consists of two strands forming a double helix.",
            "The strands are made of nucleotide molecules.",
            "Each nucleotide contains a sugar, phosphate, and base.",
            "There are four types of bases: adenine, thymine, guanine, and cytosine."
        ],
        'expected': "Base pairing rules determine how the two strands connect together."
    },
    {
        'context': [
            "The solar system formed about 4.6 billion years ago.",
            "It began from a rotating cloud of gas and dust.",
            "Gravity pulled material together at the center.",
            "This formed the Sun, our system's star.",
            "The remaining material formed planets and other bodies."
        ],
        'expected': "Rocky planets formed in the inner solar system near the Sun."
    },
    {
        'context': [
            "Machine translation automatically translates text between languages.",
            "Early systems used rule-based approaches.",
            "Statistical methods became popular in the 1990s.",
            "These analyzed bilingual text corpora.",
            "Neural machine translation emerged in the 2010s."
        ],
        'expected': "Transformer architectures now achieve near-human translation quality."
    },
    {
        'context': [
            "Photosynthesis converts light energy into chemical energy.",
            "Plants use this process to produce glucose from carbon dioxide and water.",
            "Chlorophyll absorbs light energy in plant cells.",
            "The light-dependent reactions occur in the thylakoid membranes.",
            "These reactions split water molecules and release oxygen."
        ],
        'expected': "The Calvin cycle then uses the energy to produce glucose molecules."
    },
    {
        'context': [
            "Quantum mechanics describes nature at the atomic scale.",
            "Particles exhibit both wave and particle properties.",
            "This is called wave-particle duality.",
            "The uncertainty principle limits what we can know simultaneously.",
            "We cannot precisely measure both position and momentum."
        ],
        'expected': "Quantum entanglement links particles across any distance instantaneously."
    }
]

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


print("=" * 100)
print("LVM FULL PIPELINE: TEXT OUTPUT EXAMPLES")
print("=" * 100)
print()

# Load model
print("Loading AMN model...")
model_path = 'artifacts/lvm/models/amn_20251016_133427/best_model.pt'
checkpoint = torch.load(model_path, map_location='cpu')
model = create_model('amn', input_dim=768, d_model=256, hidden_dim=512)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✓ AMN loaded (Val Cosine: {checkpoint['val_cosine']:.4f})")
print()

# Load orchestrator
print("Loading Vec2Text orchestrator...")
orch = IsolatedVecTextVectOrchestrator(steps=1, debug=False)
print("✓ Orchestrator ready")
print()

# Initialize scorers
rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
smoothing = SmoothingFunction().method1

print("=" * 100)
print("PROCESSING 10 SAMPLES")
print("=" * 100)
print()

all_results = []

for idx, sample in enumerate(SAMPLES, 1):
    print(f"\n{'='*100}")
    print(f"SAMPLE {idx}/10")
    print('='*100)

    # Show context
    print("\n📖 CONTEXT (5 chunks):")
    for i, chunk in enumerate(sample['context'], 1):
        print(f"   {i}. {chunk}")

    # Encode context
    t_start = time.perf_counter()
    ctx_vectors = orch.encode_texts(sample['context'])
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
    expected_vector = orch.encode_texts([sample['expected']])[0]
    if isinstance(expected_vector, torch.Tensor):
        expected_vector = expected_vector.cpu().numpy()

    pred_norm = pred_vector / (np.linalg.norm(pred_vector) + 1e-8)
    exp_norm = expected_vector / (np.linalg.norm(expected_vector) + 1e-8)
    vec_cos = float(np.dot(pred_norm, exp_norm))

    # Compute text scores
    rouge = rouge_scorer_obj.score(sample['expected'], predicted_text)
    bleu = sentence_bleu(
        [sample['expected'].lower().split()],
        predicted_text.lower().split(),
        smoothing_function=smoothing
    )

    print(f"\n🎯 EXPECTED:")
    print(f"   {sample['expected']}")
    print(f"\n🤖 PREDICTED:")
    print(f"   {predicted_text}")

    print(f"\n📊 SCORES:")
    print(f"   Vector Cosine:  {vec_cos:6.4f}")
    print(f"   ROUGE-1:        {rouge['rouge1'].fmeasure:6.4f}")
    print(f"   ROUGE-2:        {rouge['rouge2'].fmeasure:6.4f}")
    print(f"   ROUGE-L:        {rouge['rougeL'].fmeasure:6.4f}")
    print(f"   BLEU:           {bleu:6.4f}")

    print(f"\n⏱️  LATENCY:")
    total_ms = t_encode + t_lvm + t_decode
    print(f"   Encoding:    {t_encode:7.1f} ms  ({t_encode/total_ms*100:4.1f}%)")
    print(f"   LVM:         {t_lvm:7.2f} ms  ({t_lvm/total_ms*100:4.1f}%)")
    print(f"   Decoding:    {t_decode:7.1f} ms  ({t_decode/total_ms*100:4.1f}%)")
    print(f"   ──────────────────────")
    print(f"   TOTAL:       {total_ms:7.1f} ms")

    all_results.append({
        'vec_cos': vec_cos,
        'rouge1': rouge['rouge1'].fmeasure,
        'rouge2': rouge['rouge2'].fmeasure,
        'rougeL': rouge['rougeL'].fmeasure,
        'bleu': bleu,
        'total_ms': total_ms
    })

# Summary
print("\n")
print("=" * 100)
print("SUMMARY STATISTICS (10 samples)")
print("=" * 100)
print()
print(f"Average Vector Cosine:  {np.mean([r['vec_cos'] for r in all_results]):.4f}")
print(f"Average ROUGE-1:        {np.mean([r['rouge1'] for r in all_results]):.4f}")
print(f"Average ROUGE-2:        {np.mean([r['rouge2'] for r in all_results]):.4f}")
print(f"Average ROUGE-L:        {np.mean([r['rougeL'] for r in all_results]):.4f}")
print(f"Average BLEU:           {np.mean([r['bleu'] for r in all_results]):.4f}")
print()
print(f"Average Total Latency:  {np.mean([r['total_ms'] for r in all_results]):.1f} ms")
print()
print("=" * 100)
print("✅ FULL PIPELINE COMPLETE!")
print("=" * 100)

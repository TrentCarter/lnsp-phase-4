#!/usr/bin/env python3
"""
Test Vec2Text with Ground Truth Vectors
========================================

Tests vec2text decoder with the actual ground truth vectors from training data
to verify if the decoder itself is working correctly.

This bypasses LVM entirely and tests: Ground Truth Vector → Vec2Text → Text
"""

import numpy as np
import requests
import time

print("\n" + "="*80)
print("Vec2Text Ground Truth Vector Test")
print("="*80 + "\n")

# Load training data
print("Loading training data...")
train_data = np.load('artifacts/lvm/training_sequences_ctx5.npz', allow_pickle=True)
target_vectors = train_data['target_vectors']
target_texts = train_data['target_texts']

# Use validation split
split_idx = int(0.9 * len(target_vectors))
val_vectors = target_vectors[split_idx:split_idx + 10]
val_texts = target_texts[split_idx:split_idx + 10]

print(f"✓ Loaded 10 validation samples\n")

# Test each sample
results = []

for i in range(10):
    print(f"{'─'*80}")
    print(f"Sample {i+1}/10")
    print(f"{'─'*80}")

    ground_truth_text = str(val_texts[i])
    ground_truth_vec = val_vectors[i]

    print(f"Ground Truth Text:\n  {ground_truth_text[:100]}...\n")
    print(f"Ground Truth Vector norm: {np.linalg.norm(ground_truth_vec):.4f}\n")

    # Decode ground truth vector directly
    start_time = time.time()

    try:
        response = requests.post(
            'http://localhost:8766/decode',
            json={
                'vectors': [ground_truth_vec.tolist()],
                'subscribers': 'jxe',
                'steps': 1,  # ALWAYS USE STEPS=1
                'device': 'cpu'
            },
            timeout=120
        )

        decode_time = (time.time() - start_time) * 1000

        if response.status_code == 200:
            data = response.json()
            result = data['results'][0]['subscribers']['gtr → jxe']

            if result['status'] == 'success':
                reconstructed_text = result['output']
                cosine = result['cosine']

                print(f"Vec2Text Decoding:")
                print(f"  Time: {decode_time:.2f} ms")
                print(f"  Cosine: {cosine:.4f}")
                print(f"  Reconstructed Text:\n    {reconstructed_text[:100]}...\n")

                if cosine >= 0.65:
                    quality = "✓ EXCELLENT"
                elif cosine >= 0.50:
                    quality = "○ GOOD"
                elif cosine >= 0.30:
                    quality = "△ FAIR"
                else:
                    quality = "✗ POOR"

                print(f"Quality: {quality}\n")

                results.append({
                    'sample': i + 1,
                    'cosine': cosine,
                    'decode_time_ms': decode_time,
                    'quality': quality,
                    'success': True
                })
            else:
                print(f"✗ Decode Error: {result.get('error', 'Unknown')}\n")
                results.append({'sample': i + 1, 'success': False, 'error': result.get('error')})
        else:
            print(f"✗ HTTP Error: {response.status_code}\n")
            results.append({'sample': i + 1, 'success': False, 'error': f'HTTP {response.status_code}'})

    except Exception as e:
        print(f"✗ Exception: {e}\n")
        results.append({'sample': i + 1, 'success': False, 'error': str(e)})

# Summary
print("="*80)
print("Summary")
print("="*80 + "\n")

successful = [r for r in results if r.get('success')]

if successful:
    avg_cosine = np.mean([r['cosine'] for r in successful])
    avg_time = np.mean([r['decode_time_ms'] for r in successful])

    print(f"Successful decodings: {len(successful)}/10")
    print(f"Average cosine: {avg_cosine:.4f}")
    print(f"Average time: {avg_time:.2f} ms\n")

    excellent = sum(1 for r in successful if r['quality'] == "✓ EXCELLENT")
    good = sum(1 for r in successful if r['quality'] == "○ GOOD")
    fair = sum(1 for r in successful if r['quality'] == "△ FAIR")
    poor = sum(1 for r in successful if r['quality'] == "✗ POOR")

    print(f"Quality Distribution:")
    print(f"  Excellent: {excellent}/{len(successful)} ({100*excellent/len(successful):.1f}%)")
    print(f"  Good:      {good}/{len(successful)} ({100*good/len(successful):.1f}%)")
    print(f"  Fair:      {fair}/{len(successful)} ({100*fair/len(successful):.1f}%)")
    print(f"  Poor:      {poor}/{len(successful)} ({100*poor/len(successful):.1f}%)\n")

    if avg_cosine >= 0.65:
        print("✅ VEC2TEXT IS WORKING CORRECTLY")
        print("   Problem is with LVM vector predictions\n")
    elif avg_cosine >= 0.50:
        print("⚠️  VEC2TEXT HAS MODERATE QUALITY")
        print("   May need better training or model selection\n")
    else:
        print("❌ VEC2TEXT IS BROKEN")
        print("   Even ground truth vectors produce poor results\n")
else:
    print("❌ ALL TESTS FAILED\n")

print("="*80 + "\n")

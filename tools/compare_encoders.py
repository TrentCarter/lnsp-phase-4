#!/usr/bin/env python3
"""
Compare SentenceTransformer vs Vec2Text Orchestrator Encoders
Shows why sentence-transformers embeddings are INCOMPATIBLE with vec2text decoding.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg
from sentence_transformers import SentenceTransformer
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# Set environment for vec2text
os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
os.environ['VEC2TEXT_DEVICE'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("Encoder Comparison Test: SentenceTransformer vs Vec2Text Orchestrator")
print("=" * 80)
print()

# ============================================================================
# Step 1: Load Real Samples from Database
# ============================================================================

print("Step 1: Loading real samples from Wikipedia database...")
print()

conn = connect_pg()
cur = conn.cursor()

# Get 3 real samples from different articles
query = """
SELECT DISTINCT ON (e.batch_id)
    e.concept_text,
    e.batch_id
FROM cpe_entry e
WHERE e.dataset_source = 'wikipedia_500k'
  AND e.batch_id IS NOT NULL
  AND char_length(e.concept_text) > 50
  AND char_length(e.concept_text) < 200
ORDER BY e.batch_id, e.created_at
LIMIT 3;
"""

cur.execute(query)
rows = cur.fetchall()

if len(rows) == 0:
    print("✗ No samples found in database!")
    sys.exit(1)

samples = [row[0] for row in rows]
batch_ids = [row[1] for row in rows]

print(f"✓ Loaded {len(samples)} real samples from Wikipedia:")
for i, (text, batch) in enumerate(zip(samples, batch_ids)):
    print(f"  [{i+1}] ({batch}): {text[:70]}...")
print()

# ============================================================================
# Step 2: Initialize Both Encoders
# ============================================================================

print("Step 2: Initializing encoders...")
print()

print("  Loading WRONG encoder (sentence-transformers)...")
wrong_encoder = SentenceTransformer('sentence-transformers/gtr-t5-base')
print(f"  ✓ SentenceTransformer loaded (dim: {wrong_encoder.get_sentence_embedding_dimension()})")
print()

print("  Loading CORRECT encoder (vec2text orchestrator)...")
right_orchestrator = IsolatedVecTextVectOrchestrator()
print("  ✓ IsolatedVecTextVectOrchestrator loaded")
print()

# ============================================================================
# Step 3: Encode Samples with BOTH Encoders
# ============================================================================

print("Step 3: Encoding samples with both encoders...")
print()

# WRONG: SentenceTransformer
print("  Method 1 (WRONG): SentenceTransformer encoding...")
wrong_embeddings = wrong_encoder.encode(samples, convert_to_numpy=True, normalize_embeddings=True)
print(f"  ✓ Generated {len(wrong_embeddings)} embeddings, shape: {wrong_embeddings[0].shape}")
print()

# RIGHT: Vec2Text Orchestrator
print("  Method 2 (CORRECT): Vec2Text Orchestrator encoding...")
right_embeddings_tensor = right_orchestrator.encode_texts(samples)
right_embeddings = right_embeddings_tensor.cpu().detach().numpy()
print(f"  ✓ Generated {len(right_embeddings)} embeddings, shape: {right_embeddings[0].shape}")
print()

# ============================================================================
# Step 4: Decode with Vec2Text (Both Methods)
# ============================================================================

print("Step 4: Decoding with vec2text (IELab, 5 steps)...")
print()

results = []

for i, original_text in enumerate(samples):
    print(f"Sample {i+1}/{len(samples)}: {batch_ids[i]}")
    print("-" * 80)
    print(f"ORIGINAL TEXT:")
    print(f"  {original_text}")
    print()

    # Decode WRONG embedding
    print("  Method 1 (WRONG encoder - SentenceTransformer):")
    wrong_embedding_tensor = torch.from_numpy(wrong_embeddings[i]).unsqueeze(0)
    wrong_result = right_orchestrator._run_subscriber_subprocess(
        'ielab',
        wrong_embedding_tensor.cpu(),
        metadata={'original_texts': [original_text]},
        device_override='cpu'
    )

    if wrong_result['status'] == 'error':
        wrong_decoded = f"ERROR: {wrong_result['error']}"
        wrong_cosine = 0.0
    else:
        wrong_decoded = wrong_result['result'][0] if isinstance(wrong_result['result'], list) else wrong_result['result']
        # Calculate cosine similarity
        wrong_reencoded = right_orchestrator.encode_texts([wrong_decoded]).cpu()
        wrong_cosine = float(torch.nn.functional.cosine_similarity(
            wrong_embedding_tensor.cpu(),
            wrong_reencoded,
            dim=-1
        ).item())

    print(f"    Decoded: {wrong_decoded[:100]}...")
    print(f"    Cosine:  {wrong_cosine:.4f}")
    print()

    # Decode RIGHT embedding
    print("  Method 2 (CORRECT encoder - Vec2Text Orchestrator):")
    right_embedding_tensor = torch.from_numpy(right_embeddings[i]).unsqueeze(0)
    right_result = right_orchestrator._run_subscriber_subprocess(
        'ielab',
        right_embedding_tensor.cpu(),
        metadata={'original_texts': [original_text]},
        device_override='cpu'
    )

    if right_result['status'] == 'error':
        right_decoded = f"ERROR: {right_result['error']}"
        right_cosine = 0.0
    else:
        right_decoded = right_result['result'][0] if isinstance(right_result['result'], list) else right_result['result']
        # Calculate cosine similarity
        right_reencoded = right_orchestrator.encode_texts([right_decoded]).cpu()
        right_cosine = float(torch.nn.functional.cosine_similarity(
            right_embedding_tensor.cpu(),
            right_reencoded,
            dim=-1
        ).item())

    print(f"    Decoded: {right_decoded[:100]}...")
    print(f"    Cosine:  {right_cosine:.4f}")
    print()

    # Calculate vector difference between the two encodings
    vec_diff = np.linalg.norm(wrong_embeddings[i] - right_embeddings[i])
    vec_cosine = float(np.dot(wrong_embeddings[i], right_embeddings[i]) /
                       (np.linalg.norm(wrong_embeddings[i]) * np.linalg.norm(right_embeddings[i])))

    print(f"  Vector Comparison (between the two encodings):")
    print(f"    L2 Distance: {vec_diff:.6f}")
    print(f"    Cosine:      {vec_cosine:.4f}")
    print()

    results.append({
        'original': original_text,
        'batch_id': batch_ids[i],
        'wrong_decoded': wrong_decoded,
        'wrong_cosine': wrong_cosine,
        'right_decoded': right_decoded,
        'right_cosine': right_cosine,
        'encoding_cosine': vec_cosine,
        'encoding_l2': vec_diff
    })

    print()

# ============================================================================
# Step 5: Summary Statistics
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

avg_wrong_cosine = np.mean([r['wrong_cosine'] for r in results])
avg_right_cosine = np.mean([r['right_cosine'] for r in results])
avg_encoding_cosine = np.mean([r['encoding_cosine'] for r in results])
avg_encoding_l2 = np.mean([r['encoding_l2'] for r in results])

print(f"Average Decode Quality (text→vec→text cosine similarity):")
print(f"  WRONG Encoder (SentenceTransformer):     {avg_wrong_cosine:.4f}  {'❌ BROKEN' if avg_wrong_cosine < 0.5 else '✓'}")
print(f"  CORRECT Encoder (Vec2Text Orchestrator): {avg_right_cosine:.4f}  {'✓ WORKING' if avg_right_cosine > 0.6 else '❌'}")
print(f"  Improvement: {(avg_right_cosine / avg_wrong_cosine):.1f}x better")
print()

print(f"Encoding Vector Difference:")
print(f"  Cosine similarity between encodings: {avg_encoding_cosine:.4f}")
print(f"  L2 distance between encodings:       {avg_encoding_l2:.6f}")
print()

print("Conclusion:")
if avg_wrong_cosine < 0.5 and avg_right_cosine > 0.6:
    print("  ✅ CONFIRMED: SentenceTransformer produces INCOMPATIBLE embeddings")
    print("  ✅ CONFIRMED: Vec2Text Orchestrator produces COMPATIBLE embeddings")
    print()
    print("  ACTION REQUIRED:")
    print("  - Use IsolatedVecTextVectOrchestrator.encode_texts() for ALL encoding")
    print("  - NEVER use SentenceTransformer directly with vec2text decoding")
else:
    print("  ⚠️ Results inconclusive - see detailed output above")

print()
print("=" * 80)
print("Test Complete!")
print("=" * 80)

# Cleanup
cur.close()
conn.close()

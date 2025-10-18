#!/usr/bin/env python3
"""
Test batch encoding performance: Single vs Batch encoding
Verifies batch encoding is faster AND produces decodable vectors.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator

# Set environment for vec2text
os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
os.environ['VEC2TEXT_DEVICE'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("=" * 80)
print("Batch Encoding Performance Test")
print("=" * 80)
print()

# ============================================================================
# Step 1: Load 10 Real Chunks from Wikipedia Database
# ============================================================================

print("Step 1: Loading 10 real chunks from Wikipedia database...")
print()

conn = connect_pg()
cur = conn.cursor()

query = """
SELECT concept_text
FROM cpe_entry
WHERE dataset_source = 'wikipedia_500k'
  AND char_length(concept_text) > 50
  AND char_length(concept_text) < 300
ORDER BY created_at
LIMIT 10;
"""

cur.execute(query)
rows = cur.fetchall()

if len(rows) == 0:
    print("✗ No Wikipedia chunks found in database!")
    sys.exit(1)

test_chunks = [row[0] for row in rows]
print(f"✓ Loaded {len(test_chunks)} real chunks:")
for i, chunk in enumerate(test_chunks):
    print(f"  [{i+1}] {chunk[:70]}...")
print()

# ============================================================================
# Step 2: Initialize Vec2Text Orchestrator (Correct Encoder)
# ============================================================================

print("Step 2: Initializing vec2text orchestrator (CORRECT encoder)...")
print()

orchestrator = IsolatedVecTextVectOrchestrator()
print("✓ IsolatedVecTextVectOrchestrator loaded")
print()

# ============================================================================
# Step 3: Test Single Encoding (Loop Method)
# ============================================================================

print("Step 3: Testing SINGLE encoding (loop through 10 chunks)...")
print()

single_embeddings = []
start_time = time.time()

for i, chunk in enumerate(test_chunks):
    emb = orchestrator.encode_texts([chunk])  # Encode one at a time
    single_embeddings.append(emb.cpu().detach().numpy()[0])

single_time = time.time() - start_time

print(f"✓ Single encoding completed")
print(f"  Time: {single_time:.4f} seconds")
print(f"  Per-chunk: {single_time / len(test_chunks):.4f} seconds")
print()

# ============================================================================
# Step 4: Test Batch Encoding (All at Once)
# ============================================================================

print("Step 4: Testing BATCH encoding (all 10 chunks at once)...")
print()

start_time = time.time()

batch_embeddings_tensor = orchestrator.encode_texts(test_chunks)  # Encode all at once
batch_embeddings = batch_embeddings_tensor.cpu().detach().numpy()

batch_time = time.time() - start_time

print(f"✓ Batch encoding completed")
print(f"  Time: {batch_time:.4f} seconds")
print(f"  Per-chunk: {batch_time / len(test_chunks):.4f} seconds")
print()

# ============================================================================
# Step 5: Performance Comparison
# ============================================================================

print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)
print()

speedup = single_time / batch_time

print(f"Single encoding time:  {single_time:.4f} seconds")
print(f"Batch encoding time:   {batch_time:.4f} seconds")
print(f"Speedup:               {speedup:.2f}x faster")
print()

if speedup > 1.5:
    print(f"✅ Batch encoding is {speedup:.2f}x FASTER! Use batch method for full re-encoding.")
elif speedup > 1.0:
    print(f"✓ Batch encoding is {speedup:.2f}x faster (modest improvement)")
else:
    print(f"⚠️ Batch encoding is SLOWER ({speedup:.2f}x) - unexpected!")

print()

# ============================================================================
# Step 6: Verify Batch-Encoded Vectors Decode Correctly
# ============================================================================

print("=" * 80)
print("QUALITY VERIFICATION: Decode batch-encoded vectors back to text")
print("=" * 80)
print()

decode_results = []

for i, (original_text, embedding) in enumerate(zip(test_chunks, batch_embeddings)):
    print(f"Sample {i+1}/{len(test_chunks)}:")
    print(f"  Original: {original_text[:80]}...")

    # Decode with IELab (5 steps for good quality)
    embedding_tensor = torch.from_numpy(embedding).unsqueeze(0)
    result = orchestrator._run_subscriber_subprocess(
        'ielab',
        embedding_tensor.cpu(),
        metadata={'original_texts': [original_text]},
        device_override='cpu'
    )

    if result['status'] == 'error':
        decoded_text = f"ERROR: {result['error']}"
        cosine = 0.0
    else:
        decoded_text = result['result'][0] if isinstance(result['result'], list) else result['result']

        # Calculate cosine similarity (re-encode decoded text)
        reencoded = orchestrator.encode_texts([decoded_text]).cpu()
        cosine = float(torch.nn.functional.cosine_similarity(
            embedding_tensor.cpu(),
            reencoded,
            dim=-1
        ).item())

    print(f"  Decoded:  {decoded_text[:80]}...")
    print(f"  Cosine:   {cosine:.4f} {'✓' if cosine > 0.75 else '✗'}")
    print()

    decode_results.append({
        'original': original_text,
        'decoded': decoded_text,
        'cosine': cosine
    })

# ============================================================================
# Step 7: Summary and Recommendation
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

avg_cosine = np.mean([r['cosine'] for r in decode_results])

print(f"Performance:")
print(f"  Single encoding:  {single_time:.4f}s ({single_time / len(test_chunks):.4f}s per chunk)")
print(f"  Batch encoding:   {batch_time:.4f}s ({batch_time / len(test_chunks):.4f}s per chunk)")
print(f"  Speedup:          {speedup:.2f}x faster")
print()

print(f"Quality:")
print(f"  Average cosine similarity: {avg_cosine:.4f}")
print(f"  Status: {'✅ EXCELLENT (>0.85)' if avg_cosine > 0.85 else '✓ GOOD (>0.75)' if avg_cosine > 0.75 else '⚠️ POOR (<0.75)'}")
print()

print("Recommendation:")
if speedup > 1.2 and avg_cosine > 0.75:
    print("  ✅ PROCEED with batch encoding for full dataset re-encoding")
    print(f"  - Batch is {speedup:.2f}x faster")
    print(f"  - Quality is excellent (avg cosine {avg_cosine:.4f})")
    print()
    print("  Next step: Run full re-encoding of 80,634 chunks")
elif avg_cosine > 0.75:
    print("  ✓ Batch encoding quality is good, proceed with caution")
    print(f"  - Speedup is modest ({speedup:.2f}x)")
else:
    print("  ⚠️ Quality issue detected - investigate before full re-encoding")

print()
print("=" * 80)
print("Test Complete!")
print("=" * 80)

# Cleanup
cur.close()
conn.close()

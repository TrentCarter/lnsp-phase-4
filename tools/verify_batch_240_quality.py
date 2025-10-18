#!/usr/bin/env python3
"""
Verify that batches 1-240 were encoded with CORRECT encoder.
Samples 10 random chunks and checks vector quality.
"""

import os
import sys
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db_postgres import connect as connect_pg
from app.vect_text_vect.vec_text_vect_isolated import IsolatedVecTextVectOrchestrator
import numpy as np

# Set environment for vec2text
os.environ['VEC2TEXT_FORCE_PROJECT_VENV'] = '1'
os.environ['VEC2TEXT_DEVICE'] = 'cpu'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

SAMPLE_SIZE = 10
BATCH_240_LIMIT = 24000  # First 24,000 chunks (batches 1-240)

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("=" * 80)
    print("Verifying Batches 1-240 Quality")
    print("=" * 80)
    print()

    # Initialize orchestrator
    print("Loading CORRECT encoder...")
    orchestrator = IsolatedVecTextVectOrchestrator()
    print("✓ IsolatedVecTextVectOrchestrator loaded")
    print()

    # Connect to database
    conn = connect_pg()
    cur = conn.cursor()

    # Get random sample from first 24,000 chunks
    print(f"Sampling {SAMPLE_SIZE} random chunks from batches 1-240...")

    query = """
    SELECT cpe_id, concept_text, concept_vec
    FROM cpe_entry e
    JOIN cpe_vectors v USING (cpe_id)
    WHERE dataset_source = 'wikipedia_500k'
    ORDER BY created_at
    LIMIT %s
    """

    cur.execute(query, (BATCH_240_LIMIT,))
    first_24k = cur.fetchall()

    if len(first_24k) < SAMPLE_SIZE:
        print(f"✗ Only found {len(first_24k)} chunks, need at least {SAMPLE_SIZE}")
        sys.exit(1)

    sample = random.sample(first_24k, SAMPLE_SIZE)
    print(f"✓ Sampled {len(sample)} chunks")
    print()

    # Test each sample
    print("Testing vector quality...")
    print("-" * 80)

    similarities = []

    for idx, (cpe_id, text, stored_vec_str) in enumerate(sample, 1):
        # Parse stored vector
        stored_vec = np.array([float(x) for x in stored_vec_str.strip('[]').split(',')])

        # Re-encode with CORRECT encoder
        fresh_vec_tensor = orchestrator.encode_texts([text])
        fresh_vec = fresh_vec_tensor.cpu().detach().numpy()[0]

        # Compare
        sim = cosine_similarity(stored_vec, fresh_vec)
        similarities.append(sim)

        status = "✓" if sim > 0.98 else "✗"
        print(f"{status} Sample {idx}: cosine={sim:.6f} | {text[:60]}...")

    print("-" * 80)
    print()

    # Summary
    avg_sim = np.mean(similarities)
    min_sim = np.min(similarities)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Average cosine similarity: {avg_sim:.6f}")
    print(f"Minimum cosine similarity: {min_sim:.6f}")
    print()

    if avg_sim > 0.98:
        print("✅ EXCELLENT! Batches 1-240 are using CORRECT encoder")
        print("   → Safe to resume from batch 241")
        print()
        print("Run: ./.venv/bin/python tools/reencode_wikipedia_vectors_resume.py")
    elif avg_sim > 0.90:
        print("⚠️  MARGINAL. Vectors are mostly correct but show some drift")
        print("   → Recommend full restart for safety")
        print()
        print("Run: ./.venv/bin/python tools/reencode_wikipedia_vectors.py")
    else:
        print("✗ FAILED! Batches 1-240 are using WRONG encoder")
        print("   → Must restart from beginning")
        print()
        print("Run: ./.venv/bin/python tools/reencode_wikipedia_vectors.py")

    print()
    print("=" * 80)

    # Cleanup
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()

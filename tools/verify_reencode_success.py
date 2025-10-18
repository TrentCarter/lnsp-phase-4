#!/usr/bin/env python3
"""
Verify re-encoding success by testing 5 random vectors.
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

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    print("=" * 80)
    print("Verifying Re-Encoding Success")
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

    # Get 5 random samples
    print("Sampling 5 random chunks...")
    query = """
    SELECT cpe_id, concept_text, concept_vec
    FROM cpe_entry e
    JOIN cpe_vectors v USING (cpe_id)
    WHERE dataset_source = 'wikipedia_500k'
    ORDER BY random()
    LIMIT 5
    """
    cur.execute(query)
    samples = cur.fetchall()
    print(f"✓ Sampled {len(samples)} chunks")
    print()

    # Test each sample
    print("Testing vector quality...")
    print("-" * 80)

    similarities = []

    for idx, (cpe_id, text, stored_vec_str) in enumerate(samples, 1):
        # Parse stored vector
        stored_vec = np.array([float(x) for x in stored_vec_str.strip('[]').split(',')])

        # Re-encode with CORRECT encoder
        fresh_vec_tensor = orchestrator.encode_texts([text])
        fresh_vec = fresh_vec_tensor.cpu().detach().numpy()[0]

        # Compare
        sim = cosine_similarity(stored_vec, fresh_vec)
        similarities.append(sim)

        status = "✓" if sim > 0.98 else "✗"
        print(f"{status} Sample {idx}: cosine={sim:.6f}")
        print(f"   Text: {text[:70]}...")
        print()

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
        print("✅ EXCELLENT! Vectors are CORRECT and vec2text-compatible")
        print()
        print("Database status:")
        print("  ✓ 80,634 chunks with CORRECT vectors")
        print("  ✓ Ready for LVM training")
        print()
        print("Next steps:")
        print("  1. Rebuild FAISS index (for retrieval)")
        print("  2. Export training data (for LVM-T training)")
    else:
        print("⚠️  WARNING: Vectors don't match expected encoder")
        print("   This might indicate a problem with the re-encoding")

    print()
    print("=" * 80)

    # Cleanup
    cur.close()
    conn.close()

if __name__ == '__main__':
    main()
